import { Fragment, type ReactNode } from "react";

/**
 * Safe, zero-HTML-injection Markdown renderer for untrusted model responses.
 * Never uses dangerouslySetInnerHTML; validates URLs to http/https only.
 */
export function isSafeHttpUrl(raw: string): boolean {
  try {
    const u = new URL(raw);
    return u.protocol === "http:" || u.protocol === "https:";
  } catch {
    return false;
  }
}

export function renderInlineMarkdown(
  text: string,
  onCitationClick?: (citationNum: number) => void
): ReactNode[] {
  const parts: ReactNode[] = [];
  const tokenRegex = /(`[^`]+`|\*\*[^*]+\*\*|\[(\d+)\]|\[([^\]]+)\]\(([^)]+)\))/g;
  let lastIndex = 0;
  let match: RegExpExecArray | null;
  let key = 0;

  while ((match = tokenRegex.exec(text)) !== null) {
    if (match.index > lastIndex) {
      parts.push(<Fragment key={key++}>{text.slice(lastIndex, match.index)}</Fragment>);
    }
    const full = match[0];
    if (full.startsWith("`") && full.endsWith("`")) {
      parts.push(
        <code
          key={key++}
          className="rounded border border-slate-700/80 bg-[#05080F] px-1.5 py-0.5 font-mono text-[11px] text-cyan-300"
        >
          {full.slice(1, -1)}
        </code>
      );
    } else if (full.startsWith("**") && full.endsWith("**")) {
      parts.push(
        <strong key={key++} className="font-semibold text-slate-50">
          {full.slice(2, -2)}
        </strong>
      );
    } else if (match[2]) {
      const num = Number(match[2]);
      parts.push(
        <button
          key={key++}
          type="button"
          onClick={() => onCitationClick?.(num)}
          className="mx-0.5 inline-flex items-center rounded border border-cyan-500/45 bg-cyan-500/15 px-1.5 py-0 font-mono text-[10px] font-semibold text-cyan-200 hover:bg-cyan-500/30"
        >
          [{num}]
        </button>
      );
    } else if (match[3] && match[4]) {
      const label = match[3];
      const href = match[4];
      if (isSafeHttpUrl(href)) {
        parts.push(
          <a
            key={key++}
            href={href}
            target="_blank"
            rel="noopener noreferrer"
            className="text-cyan-300 underline decoration-cyan-500/50 hover:text-cyan-200"
          >
            {label}
          </a>
        );
      } else {
        parts.push(<span key={key++}>{label}</span>);
      }
    }
    lastIndex = tokenRegex.lastIndex;
  }

  if (lastIndex < text.length) {
    parts.push(<Fragment key={key++}>{text.slice(lastIndex)}</Fragment>);
  }
  return parts;
}

export interface MarkdownBlock {
  kind: "code" | "heading" | "list" | "paragraph";
  lang?: string;
  content: string;
  items?: string[];
}

export function parseMarkdownBlocks(raw: string): MarkdownBlock[] {
  const lines = raw.replace(/\r\n/g, "\n").split("\n");
  const blocks: MarkdownBlock[] = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];
    if (line.trim().startsWith("```")) {
      const lang = line.trim().slice(3).trim() || "text";
      const codeLines: string[] = [];
      i++;
      while (i < lines.length && !lines[i].trim().startsWith("```")) {
        codeLines.push(lines[i]);
        i++;
      }
      blocks.push({ kind: "code", lang, content: codeLines.join("\n") });
      i++;
      continue;
    }

    if (line.trim().startsWith("#")) {
      const clean = line.replace(/^#+\s*/, "").trim();
      if (clean) blocks.push({ kind: "heading", content: clean });
      i++;
      continue;
    }

    if (/^[-*]\s+/.test(line.trim())) {
      const items: string[] = [];
      while (i < lines.length && /^[-*]\s+/.test(lines[i].trim())) {
        items.push(lines[i].trim().replace(/^[-*]\s+/, ""));
        i++;
      }
      blocks.push({ kind: "list", content: "", items });
      continue;
    }

    if (!line.trim()) {
      i++;
      continue;
    }

    const paraLines: string[] = [];
    while (
      i < lines.length &&
      lines[i].trim() &&
      !lines[i].trim().startsWith("```") &&
      !lines[i].trim().startsWith("#") &&
      !/^[-*]\s+/.test(lines[i].trim())
    ) {
      paraLines.push(lines[i]);
      i++;
    }
    blocks.push({ kind: "paragraph", content: paraLines.join(" ") });
  }

  return blocks;
}
