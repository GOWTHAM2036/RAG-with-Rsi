"""Deterministic CPU-only query analysis and rule-based query classification.

Limitations of heuristic query analysis:
    1. Signal extraction relies on lexical patterns, punctuation, and closed-set
       vocabularies; it does not perform syntactic parsing or semantic role labeling.
    2. Ambiguous pronouns and conversation references are detected lexically and
       may trigger even when a pronoun refers to an entity within the same sentence.
    3. Multi-label classification uses deterministic rules and must not be
       interpreted as calibrated probabilities or semantic confidence scores.
    4. Non-whitespace-delimited languages (e.g., CJK, Thai) are not segmented into
       dictionary words by the underlying TextAnalyzer.
"""

from __future__ import annotations

import re
import unicodedata

from .analyzer import TextAnalyzer, is_technical_identifier
from .context_engine import estimate_tokens
from .query_types import QueryAnalysis, QueryClassification, QueryComplexity, QueryIntent

_MAX_QUERY_CHARS = 16_384

_QUESTION_WORDS: frozenset[str] = frozenset(
    {
        "what",
        "how",
        "why",
        "when",
        "where",
        "who",
        "whom",
        "whose",
        "which",
        "does",
        "do",
        "did",
        "is",
        "are",
        "was",
        "were",
        "can",
        "could",
        "should",
        "would",
        "will",
    }
)
_COMPARISON_TERMS: frozenset[str] = frozenset(
    {
        "compare",
        "comparing",
        "comparison",
        "versus",
        "vs",
        "difference",
        "differences",
        "differ",
        "between",
        "better",
        "worse",
        "tradeoff",
        "tradeoffs",
        "trade-off",
        "trade-offs",
        "contrast",
        "pros",
        "cons",
        "than",
    }
)
_CONJUNCTIONS: frozenset[str] = frozenset({"and", "or", "also", "plus", "while", "whereas", "both"})
_TEMPORAL_TERMS: frozenset[str] = frozenset(
    {
        "before",
        "after",
        "since",
        "until",
        "during",
        "latest",
        "previous",
        "current",
        "currently",
        "recent",
        "recently",
        "today",
        "yesterday",
        "phase",
        "version",
    }
)
_AMBIGUOUS_PRONOUNS: frozenset[str] = frozenset(
    {
        "it",
        "its",
        "they",
        "them",
        "their",
        "theirs",
        "this",
        "that",
        "these",
        "those",
        "he",
        "she",
        "him",
        "her",
        "former",
        "latter",
        "such",
    }
)
_CONVERSATION_PHRASES: tuple[str, ...] = (
    "last answer",
    "previous answer",
    "previous response",
    "above",
    "earlier",
    "as mentioned",
    "follow-up",
    "followup",
    "that error",
    "that mechanism",
    "that endpoint",
    "that model",
    "the same",
)
_NEGATION_TERMS: frozenset[str] = frozenset(
    {
        "not",
        "no",
        "never",
        "without",
        "neither",
        "nor",
        "none",
        "nothing",
        "nowhere",
        "cannot",
        "cant",
        "can't",
        "dont",
        "don't",
        "doesnt",
        "doesn't",
        "didnt",
        "didn't",
        "wont",
        "won't",
        "isnt",
        "isn't",
        "arent",
        "aren't",
        "wasnt",
        "wasn't",
        "werent",
        "weren't",
    }
)
_EXPLORATORY_TERMS: frozenset[str] = frozenset(
    {
        "explain",
        "overview",
        "describe",
        "summarize",
        "summary",
        "architecture",
        "limitations",
        "all",
        "guide",
        "walkthrough",
        "how",
        "why",
    }
)

_QUOTED_RE = re.compile(r'"([^"\n]{1,256})"|\'([^\'\n]{1,256})\'')
_NUMERIC_RE = re.compile(r"\b\d+(?:\.\d+)?(?:\s*(?:%|b|kb|mb|gb|tb|kib|mib|gib|tib|ms|s))?\b", re.I)
_ERROR_CODE_RE = re.compile(
    r"\b(?:HTTP\s*[45]\d{2}|ERR_[A-Z0-9_]+|E\d{2,4}|[a-z]+_[a-z0-9_]*(?:error|corrupted|failed|unavailable|timeout|overflow))\b",
    re.I,
)
_CAMEL_OR_SNAKE_RE = re.compile(
    r"\b(?:[a-z]+_[a-z0-9_]+|[a-z]+[A-Z][A-Za-z0-9]*)\b|(?<!\w)/[a-zA-Z0-9_./-]+"
)
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_WORD_RE = re.compile(r"[^\s,;?!]+")


def _dedupe(items: list[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(items))


def normalize_query_text(question: str) -> str:
    """Normalize whitespace and Unicode NFKC while preserving negation, numbers and identifiers."""
    nfkc = unicodedata.normalize("NFKC", question)
    return " ".join(nfkc.split())


def analyze_query(question: str, analyzer: TextAnalyzer | None = None) -> QueryAnalysis:
    """Extract deterministic lexical and structural signals from a user question."""
    active_analyzer = analyzer or TextAnalyzer()
    raw = question
    normalized = normalize_query_text(raw)
    validation_errors: list[str] = []
    if not normalized:
        validation_errors.append("empty_query")
    if len(raw) > _MAX_QUERY_CHARS:
        validation_errors.append("query_too_long")
    if any(ord(ch) < 32 and ch not in ("\n", "\r", "\t") for ch in raw):
        validation_errors.append("control_characters_present")

    lowered = normalized.casefold()
    raw_words = [w.strip(".,;:?!\"'()[]{}") for w in _WORD_RE.findall(lowered)]
    words = [w for w in raw_words if w]

    question_words = _dedupe([w for w in words if w in _QUESTION_WORDS])
    has_question_mark = "?" in normalized

    quoted_phrases = _dedupe(
        [
            m.group(1) or m.group(2)
            for m in _QUOTED_RE.finditer(normalized)
            if (m.group(1) or m.group(2))
        ]
    )
    numeric_references = _dedupe([m.group(0) for m in _NUMERIC_RE.finditer(normalized)])
    error_codes = _dedupe([m.group(0) for m in _ERROR_CODE_RE.finditer(normalized)])

    tokens = active_analyzer.tokenize(normalized, for_query=True)
    tech_ids: list[str] = []
    for tok in tokens:
        if is_technical_identifier(tok) and not tok.isdigit():
            tech_ids.append(tok)
    for m in _CAMEL_OR_SNAKE_RE.finditer(normalized):
        tech_ids.append(m.group(0))
    for code in error_codes:
        tech_ids.append(code)
    technical_identifiers = _dedupe(tech_ids)

    comparison_terms = _dedupe(
        [w for w in words if w in _COMPARISON_TERMS]
        + (["trade-off"] if "trade-off" in lowered else [])
    )
    conjunctions = _dedupe(
        [w for w in words if w in _CONJUNCTIONS]
        + (["as well as"] if "as well as" in lowered else [])
    )
    temporal_terms = [w for w in words if w in _TEMPORAL_TERMS]
    temporal_terms.extend(m.group(0) for m in _YEAR_RE.finditer(normalized))
    temporal_references = _dedupe(temporal_terms)

    ambiguous_pronouns = _dedupe([w for w in words if w in _AMBIGUOUS_PRONOUNS])
    conversation_references = _dedupe(
        [phrase for phrase in _CONVERSATION_PHRASES if phrase in lowered]
    )
    negation_terms = _dedupe([w for w in words if w in _NEGATION_TERMS])

    return QueryAnalysis(
        original_query=raw,
        normalized_query=normalized,
        char_length=len(normalized),
        estimated_tokens=estimate_tokens(normalized),
        question_words=question_words,
        has_question_mark=has_question_mark,
        technical_identifiers=technical_identifiers,
        error_codes=error_codes,
        quoted_phrases=quoted_phrases,
        numeric_references=numeric_references,
        comparison_terms=comparison_terms,
        conjunctions=conjunctions,
        temporal_references=temporal_references,
        ambiguous_pronouns=ambiguous_pronouns,
        conversation_references=conversation_references,
        negation_terms=negation_terms,
        has_negation=bool(negation_terms),
        is_valid=not validation_errors,
        validation_errors=tuple(validation_errors),
    )


def classify_query(
    analysis: QueryAnalysis, *, has_conversation_history: bool = False
) -> QueryClassification:
    """Deterministically classify query intent(s) and complexity from extracted signals."""
    if not analysis.is_valid:
        return QueryClassification(
            primary_intent="unsupported",
            labels=("unsupported",),
            complexity="invalid",
            triggered_signals=analysis.validation_errors,
            classifier_kind="heuristic",
            explanation="Query failed deterministic validation: "
            + ", ".join(analysis.validation_errors),
        )

    labels: list[QueryIntent] = []
    signals: list[str] = []
    lowered = analysis.normalized_query.casefold()
    words = lowered.split()

    if analysis.comparison_terms:
        labels.append("comparative")
        signals.append("comparison_terms:" + ",".join(analysis.comparison_terms))

    # Multi-part: multiple question marks, semicolon, or conjunction joining distinct clauses.
    multi_q = analysis.normalized_query.count("?") >= 2
    has_semicolon = ";" in analysis.normalized_query
    multi_clause_conj = bool(analysis.conjunctions) and (
        len(words) >= 8
        or any(
            kw in lowered
            for kw in (
                " and explain ",
                " and list ",
                " and compare ",
                " and describe ",
                " and how ",
                " and what ",
            )
        )
    )
    if multi_q or has_semicolon or multi_clause_conj:
        labels.append("multi_part")
        if multi_q:
            signals.append("multiple_question_marks")
        if has_semicolon:
            signals.append("semicolon_clause_separator")
        if multi_clause_conj:
            signals.append("multi_clause_conjunction:" + ",".join(analysis.conjunctions))

    if analysis.technical_identifiers or analysis.error_codes or analysis.quoted_phrases:
        labels.append("technical_lookup")
        if analysis.error_codes:
            signals.append("error_codes:" + ",".join(analysis.error_codes))
        if analysis.technical_identifiers:
            signals.append("technical_identifiers:" + str(len(analysis.technical_identifiers)))
        if analysis.quoted_phrases:
            signals.append("quoted_phrases:" + str(len(analysis.quoted_phrases)))

    if analysis.conversation_references or (
        analysis.ambiguous_pronouns and (has_conversation_history or len(words) <= 8)
    ):
        labels.append("conversational")
        if analysis.conversation_references:
            signals.append("conversation_references:" + ",".join(analysis.conversation_references))
        if analysis.ambiguous_pronouns:
            signals.append("ambiguous_pronouns:" + ",".join(analysis.ambiguous_pronouns))

    has_exploratory_term = any(w.strip(".,;:?!") in _EXPLORATORY_TERMS for w in words)
    if (
        has_exploratory_term
        and "comparative" not in labels
        and (
            len(words) >= 6
            or any(t in lowered for t in ("overview", "explain", "architecture", "limitations"))
        )
    ):
        labels.append("exploratory")
        signals.append("exploratory_structure")

    if not labels:
        labels.append("factual")
        signals.append("default_factual")

    # Precedence: comparative > multi_part > conversational > technical_lookup > exploratory
    precedence: tuple[QueryIntent, ...] = (
        "comparative",
        "multi_part",
        "conversational",
        "technical_lookup",
        "exploratory",
        "factual",
    )
    primary: QueryIntent = next((intent for intent in precedence if intent in labels), "factual")

    if primary in ("comparative", "multi_part") or len(labels) >= 3:
        complexity: QueryComplexity = "complex"
    elif primary in ("conversational", "exploratory") or len(labels) == 2:
        complexity = "moderate"
    else:
        complexity = "simple"

    explanation = f"Classified as {primary} ({complexity}) via signals: {', '.join(signals)}."
    return QueryClassification(
        primary_intent=primary,
        labels=tuple(labels),
        complexity=complexity,
        triggered_signals=tuple(signals),
        classifier_kind="heuristic",
        explanation=explanation,
    )
