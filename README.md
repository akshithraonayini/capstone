# Enterprise Knowledge Assistant

A multi-agent Retrieval-Augmented Generation (RAG) chatbot that answers questions about internal company policies (HR, leave, security, work-from-home). It is built on **LangGraph** and orchestrates guardrail, retrieval, response-generation, and evaluation agents into a single stateful pipeline with self-correcting retries and quality scoring.

It runs both as a **command-line chatbot** and a **Streamlit web app**, uses an **MCP server** to expose the knowledge base as tools, and scores every answer with **RAGAS**.

---

## 1. Architecture Overview

```
                          ┌─────────────┐
   user query ──────────► │  GUARDRAIL  │ ── blocked ──► END (safe refusal)
                          └──────┬──────┘
                                 │ allowed
                                 ▼
                          ┌─────────────┐
                          │  RETRIEVER  │ ◄───────────────┐
                          │ (MCP + RAG) │                 │ retry
                          └──────┬──────┘                 │ (low RAGAS,
                                 ▼                        │  max 2)
                          ┌─────────────┐                 │
                          │  RESPONSE   │  (Groq LLM)     │
                          └──────┬──────┘                 │
                                 ▼                        │
                          ┌─────────────┐                 │
                          │  EVALUATOR  │ ── low score ───┘
                          │   (RAGAS)   │
                          └──────┬──────┘
                                 │ passed / max retries
                                 ▼
                                END
```

The pipeline is a stateful LangGraph. A guardrail screens the query, a retriever pulls relevant policy chunks (listing documents over MCP + semantic search in Chroma), a response agent answers strictly from that context with Groq, and an evaluator scores the answer with RAGAS. Low scores loop back to retrieval (capped at 2 attempts).

---

## 2. Setup Instructions

**Environment**

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) for dependency management and running
- A **Groq API key** (free tier at [console.groq.com](https://console.groq.com))

**Dependencies** — `uv` creates the virtual environment and installs everything from `pyproject.toml` / `uv.lock`:

```bash
uv sync
```

Or, using plain `pip` with the pinned `requirements.txt`:

```bash
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

**Configuration** — create a `.env` file in the `capstone/` directory:

```dotenv
# Required — LLM provider (response + RAGAS)
GROQ_API_KEY=your_groq_api_key_here

# Required — observability / tracing
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_PROJECT=enterprise-knowledge-assistant
HF_TOKEN=your_huggingface_token
```

> `.env` is git-ignored — never commit real secrets.

**Build the vector index** from the documents in `data/documents/`:

```bash
uv run rebuild_index.py
```

This embeds and persists the knowledge base into `chroma_db/`. (The retriever also rebuilds automatically if it finds the index empty.)

---

## 3. Execution Steps

**Command-line chatbot** — produces a full node-by-node trace in the console:

```bash
uv run chatbot.py
```

**Streamlit web app**:

```bash
uv run streamlit run streamlit_app.py
```

Opens a chat UI at http://localhost:8501. Type `exit` to quit the CLI.

**Sample input**

```
You: What does the HR policy say about professional conduct?
```

**Expected output** (abridged) — each agent prints its step, followed by the answer and RAGAS scores:

```
========== GUARDRAIL AGENT ==========
[GUARDRAIL] Input passed
[ROUTER] Guardrail PASSED

========== RETRIEVER AGENT ==========
Query: What does the HR policy say about professional conduct?
[MCP] Listing knowledge documents...
[MCP] Available documents: ['hr_policy.txt', 'leave_policy.txt', 'security_policy.txt', 'wfh_policy.txt']
Retrieved 3 chunks from Chroma
Relevant source documents: ['hr_policy.txt']
[MCP] Reading full document: hr_policy.txt

========== RESPONSE AGENT ==========
Generated Answer:
- Maintain the highest levels of professionalism, integrity, and ethical conduct.
- Treat colleagues, customers, vendors, and other stakeholders with respect.
- Avoid any harassment, discrimination, or bullying.
- Comply with all company policies, procedures, and applicable laws.

========== EVALUATOR AGENT ==========
RAGAS RESULTS
Faithfulness:     0.9000
Answer Relevancy: 0.7126

========== RAGAS ROUTER ==========
Faithfulness:     0.9000
Answer Relevancy: 0.7126
Retrieval Attempts: 1
[ROUTER] RAGAS PASSED → END
```

See `SCREENSHOTS/` for captured runs of application startup, node-by-node graph execution, the final generated answer, and the RAGAS evaluation result.

---

## 4. RAG Design

| Aspect | Choice |
|---|---|
| **Document source** | Plain-text company policies in `data/documents/` — `hr_policy.txt`, `leave_policy.txt`, `security_policy.txt`, `wfh_policy.txt`. Loaded with LangChain `TextLoader`. |
| **Chunking strategy** | `RecursiveCharacterTextSplitter` with `chunk_size=500` and `chunk_overlap=50` characters. The overlap preserves context across chunk boundaries so policy clauses are not split mid-thought. |
| **Embedding model** | **FastEmbed** `sentence-transformers/all-MiniLM-L6-v2` (384-dim), run locally as ONNX — no external embedding API required. |
| **Vector database** | **Chroma**, persisted to `chroma_db/`. Retrieval uses similarity search returning the **top-3** chunks (`k=3`). |

Flow: documents → load → split into chunks → embed → store in Chroma. At query time the question is embedded, the 3 nearest chunks are found, and their source documents are read back in full over MCP (see [MCP Integration](#6-mcp-integration)) to form the grounding context.

---

## 5. LangGraph Design

**State** (`src/state.py`) — a `GraphState` TypedDict carrying `query`, `retrieved_context`, `answer`, `evaluation`, `messages`, `guardrail_blocked`/`guardrail_reason`, and `retrieval_attempts`.

**Nodes and responsibilities**

| Node | Responsibility |
|---|---|
| `guardrail` | Screens the query for prompt-injection and sensitive-credential requests. Blocks with a safe refusal or passes it through. |
| `retriever` | Lists available documents over MCP, then retrieves the top-3 relevant chunks from Chroma. Increments `retrieval_attempts`. |
| `response` | Generates a concise answer with Groq (`openai/gpt-oss-120b`), grounded **only** in retrieved context and prior conversation. |
| `evaluator` | Scores the answer with RAGAS (Faithfulness + Answer Relevancy). |

**Graph flow**

- `START → guardrail`
- `guardrail` → conditional: **blocked** → `END`; **allowed** → `retriever`
- `retriever → response → evaluator`
- `evaluator` → conditional (`ragas_router`): both metrics ≥ **0.7** → `END`; else if `retrieval_attempts ≥ 2` → `END`; otherwise → `retriever` (retry)

Conversation state is persisted per thread via a LangGraph **SQLite checkpointer** (`chatbot.db`).

---

## 6. MCP Integration

- **Server:** `Enterprise Knowledge Server`, a **FastMCP** server (`src/mcp_server/server.py`) running over STDIO transport. A synchronous wrapper (`src/mcp_client.py`) lets the graph call it.
- **Tools exposed:**
  - `list_knowledge_documents()` → lists all policy files the server serves from `data/documents/`.
  - `read_knowledge_document(filename)` → returns the full text of a named document.
- **Use case implemented — MCP-driven "chunk → parent document" retrieval.** Both tools are functional in the live query path:
  1. The retriever calls `list_knowledge_documents` to discover the authoritative, tool-provided knowledge base (no hard-coded file list).
  2. Semantic search over Chroma identifies the most relevant *chunks*, and their `source` metadata is used to rank which *documents* those chunks came from.
  3. The retriever then calls `read_knowledge_document` for the top-ranked source file(s) to pull the **complete** policy text over MCP — so the response agent is grounded in whole authoritative documents rather than possibly-fragmented chunks. (If MCP returns nothing usable, it falls back to the raw chunks.)

  This makes MCP the actual grounding channel, with semantic search acting as the router that decides *which* documents to read.

---

## 7. RAGAS Evaluation

- **Metrics collected:** **Faithfulness** (is the answer supported by the retrieved context?) and **Answer Relevancy** (does the answer address the question?). Scored directly via `single_turn_ascore` using Groq as the judge LLM and the same FastEmbed embeddings.
- **Results** (captured run — query *"What does the HR policy say about professional conduct?"*, see `SCREENSHOTS/4. RAGAS evaluation result.png`):

  | Metric | Score | Interpretation |
  |---|---|---|
  | Faithfulness | **0.9000** | High — answer is well supported by the retrieved context |
  | Answer Relevancy | **0.7126** | Moderate/High — answer addresses the question |

- **Brief analysis:** Faithfulness of 0.90 confirms the answer stays almost entirely within the retrieved policy text with negligible unsupported content — the core goal of the guardrailed RAG design. Answer Relevancy of ~0.71 is solid; it sits below the faithfulness score mainly because the answer enumerates several well-grounded supporting points beyond the literal question, which slightly dilutes the measured question–answer similarity. Both metrics clear the **0.7** threshold, so `ragas_router` returns the answer on the first attempt without retrying. When either metric drops below 0.7, the router loops back to retrieval (up to 2 attempts) to self-correct before answering.

---

## Observability

Every run is traced to **LangSmith** (guardrail decision → retrieved chunks → Groq prompt/completion → RAGAS calls) as one connected trace tree. Enable it with the `LANGSMITH_*` variables above and view traces under the `enterprise-knowledge-assistant` project at [smith.langchain.com](https://smith.langchain.com). On networks that block LangSmith, uploads may emit non-fatal SSL warnings; answers still work (a `truststore` + `certifi` trust fix is already applied).

---

## Project Structure

```
capstone/
├── chatbot.py                 # CLI entry point
├── streamlit_app.py           # Streamlit web UI
├── rebuild_index.py           # (Re)build the Chroma vector index
├── requirements.txt           # Pinned dependencies (pip alternative to uv)
├── data/documents/            # Source knowledge base (.txt policies)
├── chroma_db/                 # Persisted Chroma vector store (generated)
├── chatbot.db                 # SQLite conversation checkpoints (generated)
├── SCREENSHOTS/               # Captured runs (startup, graph flow, output, RAGAS)
└── src/
    ├── graph.py               # LangGraph assembly (nodes, edges, routing)
    ├── state.py               # GraphState TypedDict
    ├── mcp_client.py          # Synchronous MCP client wrapper
    ├── agents/                # guardrail / retriever / response / evaluator
    ├── rag/                   # loader, vectorstore, retriever
    └── mcp_server/server.py   # FastMCP knowledge-base tools
```

---

## Tech Stack

| Concern | Technology |
|---|---|
| Orchestration | LangGraph |
| LLM | Groq (`openai/gpt-oss-120b`) via LangChain |
| Embeddings | FastEmbed (`all-MiniLM-L6-v2`, local ONNX) |
| Vector store | Chroma (persistent) |
| Tooling protocol | MCP (FastMCP) |
| Evaluation | RAGAS (Faithfulness, Answer Relevancy) |
| Observability | LangSmith tracing |
| Conversation state | LangGraph SQLite checkpointer |
| Web UI | Streamlit |
| Package management | uv |
