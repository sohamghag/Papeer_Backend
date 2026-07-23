<img src="assets/logo.svg" width="48" height="48" alt="Papeer logo" align="left" style="margin-right: 12px" />

# Papeer — Research Paper Assistant

A conversational AI assistant for students and researchers to upload, explore, verify, and chat with research papers and web pages through natural language.

**v2** — rebuilt with a FastAPI backend and React frontend, replacing the original Streamlit prototype.

---

## Project Overview

Papeer is a Retrieval-Augmented Generation (RAG) application built using LangGraph, LangChain, Qdrant, OpenAI, Cohere, and a React + FastAPI stack.

Users can upload documents or load web pages into isolated chat sessions and ask questions about them. Each query is routed to the appropriate workflow:

- Retrieve information from uploaded content
- Verify claims using current web information
- Search the web for recent developments
- Answer general knowledge questions directly

The application supports multi-session conversations with persistent history, hybrid retrieval, and reranking.

---

## Target Users

- Students studying academic papers
- Researchers performing literature reviews
- Engineers exploring technical documentation
- Anyone who prefers conversational document exploration over manual reading

---

## Features

| Feature                            | Description                                                                                                                |
| ---------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| Document Q&A                       | Ask questions about uploaded documents and receive grounded, cited answers                                                 |
| Claim Verification                 | Verify whether a claim is still valid using current web information                                                        |
| Web Search                         | Search current information using Tavily                                                                                    |
| Direct Answers                     | General questions answered directly without retrieval                                                                      |
| Multi-Session Chat                 | Independent sessions with isolated document collections                                                                    |
| Persistent History                 | Conversations survive restarts and redeploys (Postgres-backed)                                                             |
| Auto Session Naming                | Session titles generated automatically from the first message                                                              |
| Session Management                 | Rename and delete conversations from the sidebar                                                                           |
| File Upload                        | Upload PDF, TXT, and Markdown files                                                                                        |
| Web Page Loading                   | Load and chat with website content by URL                                                                                  |
| Hybrid Search                      | Dense + Sparse (BM25) retrieval using Qdrant                                                                               |
| Cohere Reranking                   | Improves retrieval quality before generation                                                                               |
| Multi-Query Retrieval (RAG Fusion) | Generates query variations, retrieves for each, and RRF-fuses the results — improves recall on broad or compound questions |
| Table-Aware PDF Parsing            | Tables are extracted as structured markdown and kept as atomic chunks instead of being fragmented by chunking              |
| OCR Fallback                       | Scanned or image-only PDF pages are automatically OCR'd instead of producing empty content                                 |
| Guided Prompting                   | UI hints and autocomplete for phrasing that improves answer quality                                                        |
| Observability                      | Full request tracing via LangSmith                                                                                         |

---

## Technology Stack

### Backend

- FastAPI
- LangChain
- LangGraph
- Uvicorn

### Frontend

- React
- Vite
- Tailwind CSS

### Persistence

- Supabase (PostgreSQL) — conversation checkpoints and session metadata
- Qdrant Cloud — vector storage, scoped per session

### LLMs

- OpenAI GPT models
- Kimi (Moonshot AI)

### Retrieval

- Dense embeddings (OpenAI)
- Sparse embeddings (BM25, via fastembed)
- Hybrid search (Qdrant)
- Cohere Rerank

### Search

- Tavily Search API

### Observability

- LangSmith

### Deployment

- Render (backend)
- Vercel (frontend)

---

## Supported Data Sources

### File Upload

Supported formats:

- PDF (including scanned/image-only pages, via automatic OCR fallback)
- TXT
- Markdown

PDF tables are detected and extracted as structured markdown rather than being split apart by chunking. Files with no extractable text at all (empty, corrupted, or unrecoverable after OCR) are rejected with a clear error instead of being silently ingested.

### Web Pages

Load a single URL at a time. Pages that block automated access or rely heavily on client-side JavaScript rendering (e.g. some news sites) may fail to load meaningful content.

---

## How To Use

### 1. Start a Chat

Click **New Chat**, or simply start typing — a session is created automatically on your first message or upload.

### 2. Add Content

From the **+** menu in the chat input:

- **Add files** — upload a PDF, TXT, or Markdown file
- **Add web URL** — paste a link to load a web page

### 3. Ask Questions

Phrasing matters for routing accuracy. The input box surfaces suggestions and autocompletes trigger phrases as you type. Examples:

```text
as per the report, what methodology does the paper use?
```

```text
according to the document, summarize the conclusion
```

```text
verify this claim: transformers outperform RNNs on all sequence tasks
```

```text
what is the current exchange rate for USD to INR?
```

```text
explain gradient descent
```

### 4. Manage Sessions

Use the sidebar to switch between conversations, rename a chat, or delete it. Deleting a conversation removes its checkpoint history, its vector collection, and its metadata.

---

## Architecture

## <img width="660" height="680" alt="image" src="https://github.com/user-attachments/assets/c8d26d02-df57-4117-9cb4-228281290a96" />

## Performance Benchmarks

Measured via LangSmith tracing on real requests, one per route.

### Direct Answer

| Node            |   Latency |  Tokens |
| --------------- | --------: | ------: |
| Router          |     1.62s |     591 |
| Route decision  |     0.00s |       — |
| Generate answer |     1.65s |      29 |
| Other overhead  |    ~1.09s |       — |
| **Total**       | **4.36s** | **620** |

### Retrieve Answer

| Node                                         |    Latency |  Tokens |
| -------------------------------------------- | ---------: | ------: |
| Router                                       |      3.29s |     612 |
| Route decision                               |      0.00s |       — |
| First agent_node                             |      1.79s |     474 |
| Tool relevance decision                      |      0.00s |       — |
| Tool node / Retrieval (total)                |      2.38s |       — |
| &nbsp;&nbsp;— Dummy dense embedding          |      0.58s |       — |
| &nbsp;&nbsp;— ContextualCompressionRetriever |      1.18s |       — |
| &nbsp;&nbsp;— VectorStoreRetriever           |      0.84s |       — |
| &nbsp;&nbsp;— Actual dense query embedding   |      0.78s |       — |
| &nbsp;&nbsp;— Sparse BM25 query embedding    |     ~0.00s |       — |
| Second agent_node                            |      1.50s |     595 |
| Tool relevance decision                      |      0.00s |       — |
| Relevancy check                              |      1.63s |     597 |
| Check relevancy                              |      0.00s |       — |
| Generate answer                              |      2.54s |   ~1.7K |
| **Total**                                    | **14.03s** | **~4K** |

**Cost:** ~$0.0006 (input ~3.7K tokens / $0.0004, output 288 tokens / $0.0002) · **Answer quality:** 5/5

> Retrieval is the most expensive path — two full agent_node ↔ tool_node round trips (initial search + one rewrite_query retry) each cost a router-level LLM call's worth of tokens on top of the actual embedding/search work.

### Verify Claim

| Node                            |   Latency |    Tokens |
| ------------------------------- | --------: | --------: |
| Router                          |     1.37s |       603 |
| Route decision                  |     0.00s |         — |
| Verify claim (LLM time)         |     2.85s |       654 |
| Verify claim (other processing) |    ~2.01s |         — |
| Generate answer                 |     0.00s |         — |
| **Total**                       | **7.54s** | **~1.3K** |

**Token split:** input ~1.1K (87%) / output 159 (13%) · **Answer quality:** 5/5

### Document Ingestion (upload → searchable)

Measured uploading a 150-page PDF through `/api/upload`.

| Metric                      |     Value |
| --------------------------- | --------: |
| Pages                       |       150 |
| Chunks                      |       301 |
| Embedding tokens            |    83,845 |
| Embedding cost              | ~$0.00169 |
| Loading + splitting         |      ~25s |
| Embedding + DB storage      |      ~12s |
| **Total ingestion latency** |  **~40s** |

### Summary — all three routes

| Route           | Latency | Tokens | Quality |
| --------------- | ------: | -----: | ------- |
| Direct Answer   |   4.36s |    620 | Correct |
| Verify Claim    |   7.54s |  ~1.3K | 5/5     |
| Retrieve Answer |  14.03s |    ~4K | 5/5     |

---

## Optimization Techniques

### Hybrid Search

Combines dense semantic retrieval with sparse (BM25) keyword retrieval for better recall on both conceptual and exact-term queries.

### Cohere Reranker

Retrieved chunks are reranked before being passed to the LLM, improving relevance and reducing noise in the final answer.

### RAG Fusion (Multi-Query Retrieval)

Before retrieval, the query is expanded into 2 additional phrasings via a lightweight LLM call. All three queries are retrieved independently, then fused with Reciprocal Rank Fusion (RRF) — a second, separate fusion step from Qdrant's own dense+sparse RRF, this one combining ranks _across_ the different query phrasings. The fused candidate pool is then reranked using the original query. This targets a specific failure mode: a single embedding of a long or multi-part question can end up too "diluted" to match any chunk well, even when several chunks individually answer parts of it. Manually verified: a compound test question that returned zero retrieved documents on the single-query pipeline returned a fully cited answer once RAG Fusion was added.

### Table-Aware PDF Parsing + OCR Fallback

PDF pages are scanned for tables before chunking; detected tables are converted to markdown and kept as single, unsplit chunks so numeric data isn't fragmented across chunk boundaries. Pages with little to no extractable text (scanned/image-only pages) are rendered and passed through OCR (RapidOCR) instead of producing empty content.

### Session Isolation

Each session gets its own Qdrant collection (`papeer_<session_id>`), preventing document leakage between users or conversations.

### Retry Safety Valves

The retrieval loop is bounded — a maximum number of tool-calling attempts and a single query rewrite prevent runaway loops, with an LLM-swap fallback that removes tool access once the limit is reached.

### Guided Prompting

The frontend nudges users toward router-recognized phrasing (e.g. "as per the report...") via inline hints, quick-action suggestions, and live autocomplete — improving routing accuracy without requiring backend changes.

---

## Evaluation (Historical / Planned)

The original Streamlit prototype included an automated evaluation pipeline built with DeepEval, comparing dense-only retrieval against hybrid search + reranking. These results motivated the hybrid search architecture carried into v2:

| Metric               | Dense Retrieval + Reranker | Hybrid Search + Reranker | Improvement |
| -------------------- | -------------------------: | -----------------------: | ----------: |
| Contextual Precision |                       0.83 |                 **0.84** |       +0.01 |
| Contextual Recall    |                       0.77 |                 **0.86** |       +0.09 |
| Contextual Relevancy |                       0.82 |                 **0.84** |       +0.02 |
| Answer Relevancy     |                       0.83 |                 **0.88** |       +0.05 |
| Faithfulness         |                       0.84 |                 **0.87** |       +0.03 |

Hybrid Search improved Contextual Recall by +11.7% relative and increased "Excellent" scoring responses from 28% to 44%.

**Rebuilding this evaluation pipeline against the v2 backend is planned but not yet complete.** The RAG Fusion addition above has only been validated manually/qualitatively (see the RAG Fusion section) — no re-measured Contextual Recall score exists for v2 yet.

---

## Removed / Not Yet Rebuilt

The following existed in the original Streamlit prototype and were intentionally left out of this rebuild for now:

- YouTube transcript loading
- Password-gated access
- Docker / AWS EC2 deployment
- Automated evaluation pipeline (DeepEval)
- Token-by-token streaming (attempted, reverted — see Future Improvements)

---

## Installation

### Backend

```bash
cd backend
pip install -r requirements.txt
```

Create `.env`:

```env
OPENAI_API_KEY=your_openai_key
KIMI_API_KEY=your_kimi_key
TAVILY_API_KEY=your_tavily_key
CO_API_KEY=your_cohere_key
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
POSTGRES_URL=your_supabase_session_pooler_url
SUPABASE_URL=your_supabase_project_url
SUPABASE_KEY=your_supabase_anon_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=papeer-v2
```

> Use Supabase's **Session Pooler** connection string for `POSTGRES_URL`, not the direct connection — the direct connection is IPv6-only and unreachable from hosts like Render that don't support outbound IPv6.

Run:

```bash
uvicorn main:app --reload
```

### Frontend

```bash
cd frontend
npm install
```

Create `.env`:

```env
VITE_API_BASE=http://127.0.0.1:8000
```

Run:

```bash
npm run dev
```

---

## Deployment

- **Backend** — deployed on Render. Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- **Frontend** — deployed on Vercel. Set `VITE_API_BASE` to the deployed backend URL in Vercel's environment variables.

---

## Future Improvements

- User authentication
- Multi-user support
- Citation highlighting in the UI
- PDF annotations
- Rebuilt evaluation pipeline against the v2 backend
- Working token-by-token streaming
- Multi-modal document support

---

## Author

**Soham Ghag**

v2 rebuilt as a production-oriented RAG system using:

- LangGraph · LangChain · FastAPI · React
- OpenAI · Kimi · Qdrant · Cohere · Tavily
- Supabase (PostgreSQL) · Render · Vercel · LangSmith
