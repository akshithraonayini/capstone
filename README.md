# Enterprise Knowledge Assistant

A multi-agent Retrieval-Augmented Generation (RAG) chatbot that answers questions about internal company policies (HR, leave, security, work-from-home). It is built on **LangGraph** and orchestrates guardrail, retrieval, response-generation, and evaluation agents into a single stateful pipeline with self-correcting retries and quality scoring.

The assistant is available both as a **command-line chatbot** and a **Streamlit web app**.

---

## Features

- **Multi-agent LangGraph pipeline** — guardrail → retriever → response → evaluator, with conditional routing.
- **Guardrail agent** — blocks prompt-injection attempts and requests for sensitive credentials before they reach the model.
- **RAG retrieval** — documents chunked and embedded into a persistent **Chroma** vector store using local **FastEmbed** (`all-MiniLM-L6-v2`) embeddings, so no external embedding API is required.
- **MCP integration** — an **MCP server** (FastMCP) exposes the knowledge base as tools (`list_knowledge_documents`, `read_knowledge_document`).
- **Response agent** — answers strictly from retrieved context using **Groq** (`openai/gpt-oss-120b`), with conversation history awareness.
- **RAGAS evaluation** — scores each answer for **Faithfulness** and **Answer Relevancy**; low scores trigger automatic retrieval retries (up to 2).
- **Conversation memory** — LangGraph **SQLite checkpointer** persists chat history per thread.
- **Observability (mandatory)** — every LLM call, retrieval, and agent step is traced to **LangSmith** for full end-to-end visibility (see [Observability](#observability)).
- **Corporate-network friendly** — `truststore` + `certifi` SSL setup and a Google Cloud Storage mirror fallback for model downloads.

---

## Architecture

```
                ┌─────────────┐
   user query → │  GUARDRAIL  │ ── blocked ──→ END
                └──────┬──────┘
                       │ allowed
                       ▼
                ┌─────────────┐
                │  RETRIEVER  │ ←──────────────┐
                │ (MCP + RAG) │                │ retry (low RAGAS)
                └──────┬──────┘                │
                       ▼                       │
                ┌─────────────┐                │
                │  RESPONSE   │  (Groq LLM)    │
                └──────┬──────┘                │
                       ▼                       │
                ┌─────────────┐                │
                │  EVALUATOR  │ ── low score ──┘
                │   (RAGAS)   │
                └──────┬──────┘
                       │ passed / max retries
                       ▼
                      END
```

**Retry logic:** if Faithfulness or Answer Relevancy falls below `0.7`, the graph re-runs retrieval. It stops after 2 attempts to prevent loops.

---

## Project structure

```
capstone/
├── chatbot.py                 # CLI entry point
├── streamlit_app.py           # Streamlit web UI entry point
├── rebuild_index.py           # (Re)build the Chroma vector index
├── pyproject.toml             # Project metadata & dependencies (uv)
├── requirements.txt           # pip-style dependency list
├── data/
│   └── documents/             # Source knowledge base (.txt policies)
│       ├── hr_policy.txt
│       ├── leave_policy.txt
│       ├── security_policy.txt
│       └── wfh_policy.txt
├── chroma_db/                 # Persisted Chroma vector store (generated)
├── chatbot.db                 # SQLite conversation checkpoints (generated)
└── src/
    ├── graph.py               # LangGraph assembly (nodes, edges, routing)
    ├── state.py               # GraphState TypedDict
    ├── mcp_client.py          # Synchronous MCP client wrapper
    ├── agents/
    │   ├── guardrail_agent.py # Prompt-injection & sensitive-info filter
    │   ├── retriever_agent.py # MCP listing + Chroma retrieval
    │   ├── response_agent.py  # Groq answer generation
    │   └── evaluator_agent.py # RAGAS Faithfulness / Answer Relevancy
    ├── rag/
    │   ├── loader.py          # Load & chunk documents
    │   ├── vectorstore.py     # FastEmbed embeddings + Chroma helpers
    │   └── retriever.py
    └── mcp_server/
        └── server.py          # FastMCP knowledge-base tools
```

---

## Prerequisites

- **Python 3.11+**
- **[uv](https://docs.astral.sh/uv/)** for dependency management and running
- A **Groq API key** (free tier available at [console.groq.com](https://console.groq.com))

--

## Setup

1. **Install dependencies** (uv creates the virtual environment automatically):

   ```bash
   uv sync
   ```

2. **Configure environment variables.** Create a `.env` file in the `capstone/` directory:

   ```dotenv
   # Required — LLM provider
   GROQ_API_KEY=your_groq_api_key_here

   # Required — observability / tracing (see the Observability section)
   LANGSMITH_TRACING=true
   LANGSMITH_ENDPOINT=https://api.smith.langchain.com
   LANGSMITH_API_KEY=your_langsmith_key
   LANGSMITH_PROJECT=enterprise-knowledge-assistant

   # Optional — Hugging Face token (model downloads on restricted networks)
   HF_TOKEN=your_huggingface_token
   ```

   > **Note:** Never commit real secrets — `.env` is git-ignored. If your network blocks `api.smith.langchain.com`, see the [Observability](#observability) section for how to keep tracing working.

3. **Build the vector index** from the documents in `data/documents/`:

   ```bash
   uv run rebuild_index.py
   ```

   This embeds and persists the knowledge base into `chroma_db/`. The retriever also rebuilds the index automatically if it finds it empty.

---

## Usage

### Command-line chatbot

```bash
uv run chatbot.py
```

Type your question at the `You:` prompt. Type `exit` to quit. Each answer prints the RAGAS Faithfulness and Answer Relevancy scores.

### Streamlit web app

```bash
uv run streamlit run streamlit_app.py
```

Opens a chat interface in your browser (default: http://localhost:8501). Conversation history and RAGAS scores are shown inline.

---

## Adding knowledge

1. Drop new `.txt` files into `data/documents/`.
2. Re-run `uv run rebuild_index.py` to re-index.

The MCP server automatically exposes any file in that directory through its tools.

---

## How it works

1. **Guardrail** — the query is checked against prompt-injection and sensitive-info patterns. Blocked queries short-circuit to a safe refusal.
2. **Retriever** — lists available documents via MCP, then pulls the top-3 most relevant chunks from Chroma.
3. **Response** — Groq generates a concise answer grounded strictly in the retrieved context and prior conversation.
4. **Evaluator** — RAGAS scores the answer. Metrics are computed directly (`single_turn_ascore`) to keep scoring robust. If scores are below the `0.7` threshold, the pipeline retries retrieval; otherwise it returns the answer.

---

## Observability

Observability is a **mandatory** part of this project. Every run is fully traced with **LangSmith** so you can inspect exactly what each agent did — the guardrail decision, the documents retrieved, the prompt sent to Groq, the generated answer, and the RAGAS scoring calls — as a single connected trace tree.

### What is captured

- **End-to-end traces** of the LangGraph pipeline (guardrail → retriever → response → evaluator), including retrieval retries.
- **LLM calls** — prompts, completions, token usage, and latency for every Groq invocation.
- **Retrieval steps** — the query and the chunks returned from Chroma.
- **Evaluation** — the RAGAS Faithfulness / Answer Relevancy metric runs.

### Enabling it

Set the following in `.env` (already listed in [Setup](#setup)):

```dotenv
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_PROJECT=enterprise-knowledge-assistant
```

Get an API key from [smith.langchain.com](https://smith.langchain.com). Traces appear under the project named in `LANGSMITH_PROJECT`.

### Viewing traces

1. Open [smith.langchain.com](https://smith.langchain.com) and select the `enterprise-knowledge-assistant` project.
2. Each chatbot query produces one trace; click it to drill into every agent and LLM step, with inputs, outputs, latency, and token counts.

### Networks that block LangSmith

On restricted/corporate networks that block `api.smith.langchain.com`, trace uploads can emit SSL `CERTIFICATE_VERIFY_FAILED` warnings. These are non-fatal (answers still work). The app already applies a `truststore` + `certifi` SSL trust fix; if uploads still fail, either allowlist `api.smith.langchain.com` or temporarily set `LANGSMITH_TRACING=false` to silence the warnings while offline.

---

## Tech stack

| Concern            | Technology                                   |
|--------------------|----------------------------------------------|
| Orchestration      | LangGraph                                    |
| LLM                | Groq (`openai/gpt-oss-120b`) via LangChain   |
| Embeddings         | FastEmbed (`all-MiniLM-L6-v2`, local ONNX)   |
| Vector store       | Chroma (persistent)                          |
| Tooling protocol   | MCP (FastMCP)                                |
| Evaluation         | RAGAS (Faithfulness, Answer Relevancy)       |
| Observability      | LangSmith tracing                            |
| Conversation state | LangGraph SQLite checkpointer                |
| Web UI             | Streamlit                                    |
| Package management | uv                                           |
