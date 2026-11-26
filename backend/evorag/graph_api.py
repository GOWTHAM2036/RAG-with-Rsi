"""Bounded provenance graph API backed by SQLite document, embedding and RSI stores."""

from typing import Annotated, Literal

from fastapi import APIRouter, Query, Request

from .contracts import Contract
from .errors import IngestionError

router = APIRouter(prefix="/api/v1")

NodeKind = Literal["document", "revision", "chunk", "embedding", "config_version"]
EdgeKind = Literal[
    "document_contains_chunk",
    "document_has_revision",
    "chunk_has_embedding",
    "candidate_derives_from_parent",
]


class GraphNode(Contract):
    id: str
    kind: NodeKind
    label: str
    cluster: str
    metadata: dict[str, str | int | bool | None]


class GraphEdge(Contract):
    id: str
    source: str
    target: str
    kind: EdgeKind


class ProvenanceGraphResponse(Contract):
    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]
    total_nodes: int
    total_edges: int
    truncated: bool
    is_demo: bool = False


@router.get("/graph/provenance")
def provenance_graph(
    request: Request,
    max_nodes: Annotated[int, Query(ge=10, le=500)] = 250,
    include_rsi: bool = True,
) -> ProvenanceGraphResponse:
    """Return bounded real provenance nodes and edges from SQLite stores."""
    ingestion = getattr(request.app.state, "ingestion", None)
    inference = getattr(request.app.state, "inference", None)
    if ingestion is None:
        raise IngestionError("application_unavailable", 503)

    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    truncated = False

    docs = ingestion.repository.documents(limit=50, offset=0)
    for doc in docs:
        if len(nodes) >= max_nodes:
            truncated = True
            break
        doc_nid = f"doc:{doc.document_id}"
        nodes.append(
            GraphNode(
                id=doc_nid,
                kind="document",
                label=doc.source,
                cluster=doc.document_id,
                metadata={
                    "document_id": doc.document_id,
                    "source": doc.source,
                    "latest_version": doc.latest_version,
                },
            )
        )
        versions = ingestion.repository.versions(doc.document_id, limit=5, offset=0)
        for ver in versions:
            if len(nodes) >= max_nodes:
                truncated = True
                break
            rev_nid = f"rev:{doc.document_id}:v{ver.version}"
            nodes.append(
                GraphNode(
                    id=rev_nid,
                    kind="revision",
                    label=f"v{ver.version}",
                    cluster=doc.document_id,
                    metadata={
                        "document_id": doc.document_id,
                        "version": ver.version,
                        "parser": ver.parser,
                        "chunk_count": ver.chunk_count,
                    },
                )
            )
            edges.append(
                GraphEdge(
                    id=f"e:{doc_nid}->{rev_nid}",
                    source=doc_nid,
                    target=rev_nid,
                    kind="document_has_revision",
                )
            )
            chunks = ingestion.repository.chunks(doc.document_id, ver.version, limit=25, offset=0)
            for idx, ch in enumerate(chunks, start=1):
                if len(nodes) >= max_nodes:
                    truncated = True
                    break
                ch_nid = f"chunk:{ch.chunk_id}"
                nodes.append(
                    GraphNode(
                        id=ch_nid,
                        kind="chunk",
                        label=f"{doc.source}#c{idx}",
                        cluster=doc.document_id,
                        metadata={
                            "chunk_id": ch.chunk_id,
                            "document_id": ch.document_id,
                            "version": ver.version,
                            "token_count": ch.token_count,
                            "start": ch.start,
                            "end": ch.end,
                            "excerpt": ch.text[:160],
                        },
                    )
                )
                edges.append(
                    GraphEdge(
                        id=f"e:{rev_nid}->{ch_nid}",
                        source=rev_nid,
                        target=ch_nid,
                        kind="document_contains_chunk",
                    )
                )

    if include_rsi and inference is not None and len(nodes) < max_nodes:
        for ver_cfg in inference.rsi_engine.store.list_versions():
            if len(nodes) >= max_nodes:
                truncated = True
                break
            cfg_nid = f"cfg:{ver_cfg.version_id}"
            nodes.append(
                GraphNode(
                    id=cfg_nid,
                    kind="config_version",
                    label=ver_cfg.version_id,
                    cluster="rsi_evolution",
                    metadata={
                        "version_id": ver_cfg.version_id,
                        "fingerprint": ver_cfg.fingerprint,
                        "active": ver_cfg.active,
                        "approved": ver_cfg.approved,
                    },
                )
            )
            if ver_cfg.parent_version_id:
                parent_nid = f"cfg:{ver_cfg.parent_version_id}"
                edges.append(
                    GraphEdge(
                        id=f"e:{parent_nid}->{cfg_nid}",
                        source=parent_nid,
                        target=cfg_nid,
                        kind="candidate_derives_from_parent",
                    )
                )

    return ProvenanceGraphResponse(
        nodes=tuple(nodes),
        edges=tuple(edges),
        total_nodes=len(nodes),
        total_edges=len(edges),
        truncated=truncated,
        is_demo=False,
    )
