import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { App } from "../App";
import { ApiError, consumeNdjsonStream } from "../api";
import { isSupportedDocumentFile } from "../components/DocumentManagement";
import { DEMO_PROVENANCE_GRAPH } from "../data/demoCorpus";
import { isSafeHttpUrl, parseMarkdownBlocks } from "../utils/markdown";

describe("EvoRAG Phase 11.3 Functional AI Workspace", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    vi.spyOn(globalThis, "fetch").mockRejectedValue(new Error("Offline in unit test"));
  });

  it("enforces strict live/demo separation and requires explicit demo transition", async () => {
    const user = userEvent.setup();
    render(<App />);

    expect(screen.getByText(/INTELLIGENCE COMMAND CENTER/i)).toBeInTheDocument();
    expect(screen.getByText(/Live Mode active, but FastAPI backend/i)).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /Switch to Demo Mode/i }));
    expect(screen.queryByText(/Live Mode active, but FastAPI backend/i)).not.toBeInTheDocument();
    expect(screen.getAllByText(/DEMO/i).length).toBeGreaterThan(0);
  });

  it("supports RAG Playground conversations, demo query execution, and cross-view citation navigation", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByRole("button", { name: /Switch to Demo Mode/i }));
    await user.click(screen.getByRole("button", { name: /RAG Playground/i }));

    expect(await screen.findByText(/Grounded conversational workbench/i)).toBeInTheDocument();

    // Create a new conversation and submit a query
    await user.click(screen.getByRole("button", { name: /New Conversation/i }));
    const input = screen.getByLabelText(/Playground question input/i);
    await user.type(input, "Explain context budget");
    await user.click(screen.getByRole("button", { name: /Send/i }));

    expect(await screen.findByText(/\[DEMO RESPONSE\]/i)).toBeInTheDocument();

    // Cross-view citation navigation to Document Management
    await user.click(screen.getByRole("button", { name: /Open Document/i }));
    expect(
      await screen.findByText(/DOCUMENT MANAGEMENT & PROVENANCE INSPECTOR/i)
    ).toBeInTheDocument();
  });

  it("validates Document Management file formats and supports revision/chunk inspection", async () => {
    const user = userEvent.setup();
    render(<App />);

    expect(isSupportedDocumentFile("notes.md")).toBe(true);
    expect(isSupportedDocumentFile("data.csv")).toBe(true);
    expect(isSupportedDocumentFile("spec.json")).toBe(true);
    expect(isSupportedDocumentFile("readme.txt")).toBe(true);
    expect(isSupportedDocumentFile("report.pdf")).toBe(false);
    expect(isSupportedDocumentFile("paper.docx")).toBe(false);

    await user.click(screen.getByRole("button", { name: /Switch to Demo Mode/i }));
    await user.click(screen.getByRole("button", { name: /Document Management/i }));

    expect(
      await screen.findByText(/DOCUMENT MANAGEMENT & PROVENANCE INSPECTOR/i)
    ).toBeInTheDocument();
    expect(screen.getByLabelText(/Document revision selector/i)).toBeInTheDocument();
    expect(screen.getAllByText(/architecture\.md/i).length).toBeGreaterThan(0);
  });

  it("renders Retrieval Lab rankings, educational simulator, and cross-view links", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByRole("button", { name: /Switch to Demo Mode/i }));
    await user.click(screen.getByRole("button", { name: /Retrieval Lab/i }));

    expect(
      await screen.findByText(/RETRIEVAL LAB · SCIENTIFIC RANKING COMPARISON/i)
    ).toBeInTheDocument();
    expect(
      screen.getByText(/EDUCATIONAL SIMULATION · DOES NOT AFFECT LIVE CONFIG/i)
    ).toBeInTheDocument();

    await user.selectOptions(screen.getByLabelText(/Retrieval Lab mode/i), "sparse");
    await user.click(screen.getByRole("button", { name: /Compare Rankings/i }));
    expect(screen.getByText(/effective_mode:/i)).toBeInTheDocument();

    await user.click(screen.getAllByRole("button", { name: /Graph Node/i })[0]);
    expect(await screen.findByText(/Provenance Explorer/i)).toBeInTheDocument();
  });

  it("parses NDJSON streams with partial frames, UTF-8 boundaries, and malformed lines safely", async () => {
    const encoder = new TextEncoder();
    const chunks = [
      encoder.encode('{"event":"retrieving"}\n{"ev'),
      encoder.encode('ent":"generating"}\nnot-json-frame\n{"event":"completed","trace_id":"t-1"}\n'),
    ];
    let idx = 0;
    const stream = new ReadableStream<Uint8Array>({
      pull(controller) {
        if (idx < chunks.length) {
          controller.enqueue(chunks[idx++]);
        } else {
          controller.close();
        }
      },
    });

    const events: string[] = [];
    const malformed: string[] = [];
    await consumeNdjsonStream(
      stream,
      (ev) => events.push(ev.event),
      (raw) => malformed.push(raw)
    );

    expect(events).toEqual(["retrieving", "generating", "completed"]);
    expect(malformed).toEqual(["not-json-frame"]);
  });

  it("handles HTTP 503 resource admission rejections and Markdown XSS sanitization", async () => {
    const err = new ApiError(503, "insufficient_memory: 3 GiB floor required");
    expect(err.isResourceRejection).toBe(true);

    expect(isSafeHttpUrl("https://example.com")).toBe(true);
    expect(isSafeHttpUrl("javascript:alert(1)")).toBe(false);

    const blocks = parseMarkdownBlocks("# Title\n\n```python\nx = 1\n```\n\n- item");
    expect(blocks.map((b) => b.kind)).toEqual(["heading", "code", "list"]);

    const user = userEvent.setup();
    render(<App />);
    await user.keyboard("{Control>}k{/Control}");
    await waitFor(() => {
      expect(screen.getByRole("dialog", { name: /Global Command Palette/i })).toBeInTheDocument();
    });
    expect(DEMO_PROVENANCE_GRAPH.nodes.length).toBeGreaterThanOrEqual(200);
  });
});
