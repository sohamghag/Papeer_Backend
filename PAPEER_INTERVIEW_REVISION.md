# Papeer — Interview Revision Sheet

Everything covered, start to end. Read top-down the night before; skim §1, §14, §16 in the last 10 minutes.

---

## 1. THE OPENING PITCH

### WHAT

> "Papeer is a full-stack RAG application where users upload documents or paste URLs and have a conversational chat with that content. What makes it different from a basic RAG app is that it doesn't blindly retrieve for every query — it routes each message to the right workflow depending on intent."

### WHY

> "LLMs are powerful but they can't access your private documents or recent information. Existing solutions like ChatGPT file upload use basic RAG that retrieves for every query, which is slow and often inaccurate. I built Papeer to do it properly — route each query to the right workflow, use hybrid search with reranking for better retrieval quality, and support real use cases like claim verification and web search."

### HOW

> "The user uploads a document — it gets chunked, embedded, and stored in Qdrant. When they send a message, a LangGraph router classifies the intent and sends it to one of the workflows: retrieve from the vector store, search the web via Tavily, verify a claim, or answer directly. For retrieval, I use hybrid search — dense embeddings plus BM25 — fused with RRF, then Cohere reranks the results before the LLM generates a cited answer."

### The four problems RAG solves (framed for this project)

| Problem | How to say it |
|---|---|
| Knowledge cutoff | "Documents the user uploads are private and recent — the LLM has never seen them" |
| Hallucination | "Answers are grounded in retrieved chunks — it either found it or it didn't" |
| Source attribution | "Users see exactly which chunk the answer came from, which builds trust" |
| Private data | "Documents never leave their session — Qdrant collections are scoped per session" |

---

## 2. ⚠️ README vs CODE MISMATCHES — FIX BEFORE THE INTERVIEW

### (a) The router has THREE routes, not four

```python
class RouterDecision(BaseModel):
    route_decision: Literal["direct_answer", "retrieve", "verify_claim"]
```

`web_search` is **not** a router route — it's one of two **tools** the `agent_node` chooses between.

> "The router does a 3-way classification: direct_answer, retrieve, or verify_claim. Web search isn't a router decision — it's a tool available under the retrieve route. The router prompt folds 'current information' questions into `retrieve`, and the agent then decides whether that means the vector store or Tavily. So it's a two-level decision."

### (b) Retrievals are already parallelized

```python
ranked_lists = await asyncio.gather(
    *[asyncio.to_thread(base_retriever.invoke, q) for q in all_queries]
)
```

> "The README benchmark is from the sequential version. I've since switched to `asyncio.gather` with `to_thread`, so the three variation retrievals run concurrently. Wall-clock retrieval is now the slowest single query, not the sum."

### (c) The app is BYOK (not in the README at all)

```python
class MissingApiKeyError(Exception):
    """No fallback to the developer's own .env key — the request must fail
    instead of silently consuming the app owner's credits."""
```

> "Users supply their own OpenAI and Kimi keys. There's deliberately no fallback to my `.env` key. `/api/keys/verify` validates them by calling `models.list()` before chatting."

### (d) There's a dead edge in the graph diagram

`tool_relevancy_decision` declares `"generate_answer"` in its mapping but never returns it.

> "That edge is unreachable — a leftover mapping entry from an earlier version. LangGraph draws every declared branch, not just the reachable ones."

---

## 3. BACKEND FUNDAMENTALS

**Backend** — "The server-side part of an application that the user never interacts with directly. It handles business logic, data processing, database operations, and communication with external services."

**In Papeer** — "A FastAPI app that handles document ingestion — parsing, chunking, embedding, storing in Qdrant — orchestrates the LangGraph query routing, and manages session persistence in Postgres."

**Server** — Two senses: the program listening on a port (Uvicorn), and the host running it (Render).

**API** — "A defined interface that lets two programs communicate. It specifies what endpoints exist, what data format to send, and what comes back." Not only frontend↔backend — you also call OpenAI's, Cohere's, and Tavily's APIs.

### The layer stack

| Layer | What | In Papeer |
|---|---|---|
| Frontend | UI | React + Vite + Tailwind, on Vercel |
| Server | Listens for HTTP | Uvicorn |
| Framework | Defines routes/logic | FastAPI |
| Hosting | Where it runs | Render |

### Framework vs Library

> "A library is code you call — you stay in control of the flow. A framework calls your code — it owns the flow and you plug into it. That's inversion of control. FastAPI is a framework: I define endpoint functions, but FastAPI decides when to invoke them."

LangChain is used more like a library; LangGraph is the actual framework.

---

## 4. FASTAPI / ASGI / ASYNC

### Why FastAPI (the main answer)

> "The main reason is async. My requests spend most of their time waiting on external APIs — OpenAI, Qdrant, Cohere. FastAPI is async-native, so the server can handle other requests during that wait instead of sitting blocked."

Backups in your pocket: Pydantic auto-validation (422 before your code runs), auto-generated Swagger docs, Python-native AI ecosystem (no extra service boundary).

### FastAPI vs Uvicorn vs ASGI

- **FastAPI** — the framework. Defines endpoints and logic.
- **Uvicorn** — the ASGI server. Listens on a port, hands requests to FastAPI.
- **ASGI** — "The standard interface between a Python web server and a Python application. It defines the contract for how requests and responses are passed between them, **asynchronously**."

Request flow: `Uvicorn (ASGI server) → ASGI interface → FastAPI → your endpoint function`

### ASGI vs WSGI

| | WSGI | ASGI |
|---|---|---|
| Model | Synchronous | Asynchronous |
| Request handling | One request blocks a worker | Worker switches while awaiting I/O |
| Frameworks | Flask, classic Django | FastAPI, Starlette |
| Streaming / WebSockets | No | Yes |

> "My requests take 4 to 21 seconds, mostly waiting on OpenAI, Qdrant, and Cohere. Under WSGI each request holds a worker for that entire duration — a few concurrent users would exhaust the pool. With ASGI, while one request awaits a network call the event loop serves others."

### `asyncio.to_thread`

> "It offloads a blocking function to a threadpool so it doesn't block the event loop. The Qdrant Python client is synchronous, so calling it directly inside an async handler would stall every other request for the duration."

Your five usages: `_get_vectorstore_sync`, `add_documents`, `base_retriever.invoke` (inside `gather`), `_list_papers_sync`, `_delete_collection_sync`.

**"Why not the async Qdrant client?"** — "There is one, and I had that version; it's commented out at the bottom of `vector_store.py`. `QdrantVectorStore` expects the sync client for hybrid retrieval mode, so `to_thread` was the reliable path."

**"Does it scale?"** — "Not indefinitely. The threadpool is bounded, so under heavy load you queue on threads instead of blocking the loop. A native async client is the proper fix."

### Why not Node / Django

- **Node** — "Also async and would handle concurrency fine. But my whole pipeline is LangChain, LangGraph, and Qdrant's Python client. Node would mean a weaker JS port or a separate Python service — an extra network hop and deployment surface."
- **Django** — "Too heavy. I don't need an ORM, admin panel, or templates. My frontend is React and my data lives in Postgres checkpoints and Qdrant, not Django models."

---

## 5. HTTP METHODS & STATUS CODES

| Method | Purpose | Body | Idempotent |
|---|---|---|---|
| GET | Read | No | Yes |
| POST | Create / trigger | Yes | **No** |
| PUT | Replace entirely | Yes | Yes |
| PATCH | Partial update | Yes | Yes |
| DELETE | Remove | No | Yes |

**Idempotent** = calling it N times has the same effect as once. `DELETE /sessions/123` five times → still gone. `POST /sessions` five times → five sessions. That's why **POST is the one you shouldn't blindly retry**.

**PUT vs PATCH** — "PUT replaces the entire resource; omitted fields get wiped. PATCH sends only what changes. Renaming a session is a PATCH."

| Code | Meaning |
|---|---|
| 200 / 201 / 202 / 204 | OK / Created / Accepted (async) / No Content |
| 400 / 401 / 403 / 404 | Bad request / Unauthorized / Forbidden / Not found |
| 413 | Payload too large |
| **422** | **FastAPI's automatic Pydantic validation failure** |
| 500 / 504 | Server error / Upstream timeout |

Mentioning 422 signals you've actually used FastAPI.

---

## 6. LANGCHAIN vs LANGGRAPH vs LANGSMITH

**LangChain** — the toolkit. Standard interfaces for models, prompts, loaders, splitters, embeddings, vector stores, retrievers. Swap providers without rewriting logic.

**LangGraph** — the control flow. Stateful graph: State (TypedDict), Nodes (functions), Edges, Conditional edges.

> "LangChain is the components. LangGraph is the control flow. A chain is a straight line A → B → C. A graph can branch, loop, and revisit."

**LangSmith** — observability. Every node execution logged: inputs, outputs, latency, tokens, cost. It's how you found the double-embedding bug.

### The 4 things to memorize about LangGraph

1. **State** = TypedDict
2. **Node** = `function(state) -> dict` (returned dict merges into state)
3. **Edge** = `add_edge(a, b)` — always a→b
4. **Conditional edge** = `add_conditional_edges(a, fn, mapping)` — fn returns a string, mapping picks the next node

**Reducer gotcha:** by default a node's return *overwrites* that key. Use `Annotated[list, operator.add]` or `add_messages` to append.

### Minimal code you should be able to write

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    question: str
    answer: str

def answer_node(state: State):
    return {"answer": llm.invoke(state["question"]).content}

g = StateGraph(State)
g.add_node("answer", answer_node)
g.add_edge(START, "answer")
g.add_edge("answer", END)
app = g.compile()
```

Conditional edge:

```python
g.add_conditional_edges("router", decide, {
    "retrieve": "retrieve",
    "web_search": "web_search",
    "direct_answer": "direct_answer",
})
```

Memory:

```python
app = g.compile(checkpointer=MemorySaver())
config = {"configurable": {"thread_id": "session-1"}}
app.invoke({"question": "hi"}, config=config)
```

---

## 7. SUPABASE & PERSISTENCE

**Supabase** — "A Backend-as-a-Service built on PostgreSQL. Managed Postgres plus auth, storage, realtime, and auto-generated REST APIs."

### Two jobs in Papeer

1. **LangGraph checkpointing** — every graph execution writes state to Postgres via `AsyncPostgresSaver`, keyed by `thread_id` (= session_id). "The graph doesn't hold state in memory — it persists after each node. Conversations survive restarts and redeploys."
2. **Session metadata** — session IDs, titles, timestamps behind the sidebar.

### Why two databases?

> "They solve different problems. Qdrant is built for approximate nearest-neighbour search over embeddings — it's not designed for relational queries, transactions, or conversation history. Postgres handles the structured relational side. Using one for the other's job would be a bad fit in both directions."

### The IPv6 bug (tell this story)

> "Postgres connections worked locally but failed on Render. Supabase's direct connection string resolves to IPv6-only, and Render doesn't support outbound IPv6. Switching to Supabase's **Session Pooler** endpoint, which is IPv4-reachable, fixed it."

### Connection pooling

> "Opening a Postgres connection is expensive — TCP handshake, auth, session setup. A pooler keeps connections open and hands them out, and it protects the database, since Postgres has a hard connection limit."

| Mode | Behaviour | Use when |
|---|---|---|
| **Session** | Connection held for the whole client session | Prepared statements, cross-query transactions — **what the checkpointer needs** |
| Transaction | Returned to pool after each transaction | Serverless / high churn |

---

## 8. VECTORS, ANN, HNSW

### Embeddings

> "A fixed-length vector of floats produced by a model, positioned so semantically similar text lands close together. Retrieval is then a nearest-neighbour search in that space."

`text-embedding-3-small` → **1536 dimensions**.

**Cosine similarity** — "Measures the angle between vectors, ignoring magnitude. A short and a long passage on the same topic should score as similar, and magnitude scales with length — cosine normalises that away."

### ENN vs ANN

- **ENN (Exact NN)** — compare against every vector. 100% accurate, slow.
- **ANN (Approximate NN)** — use an index to skip most comparisons. Very fast, slight recall loss.

**Correction to remember:** ANN is the *goal*; HNSW and IVF are the *index structures* that achieve it. **PQ (Product Quantization) is compression, not an index** — it splits a vector into subvectors and replaces each with a centroid ID. Usually combined, e.g. IVF-PQ.

### HNSW

Layered graph — a skip list generalized to graphs.

- Upper layers: sparse, few nodes — "express lanes" covering big distances in few hops
- Layer 0: dense, contains all nodes
- Search: start at the top, greedy nearest-node traversal, descend layer by layer

> **Why hierarchical layers?** "Upper layers let a few hops cover huge distances. Each descent narrows the region, so by layer 0 you're already near the target and only need local refinement. Without layers, greedy search on a flat graph takes far more hops and can get stuck in a **local minimum**."

### HNSW params in Qdrant

| Param | Meaning | Tradeoff |
|---|---|---|
| `m` | edges per node (default 16) | higher = better recall, more memory |
| `ef_construct` | candidates during build (default 100) | higher = better graph, slower indexing |
| `ef` / `hnsw_ef` | candidates during search | higher = better recall, slower query |

"Recall vs latency is tunable at query time via `ef` — no rebuild needed."

### HNSW vs IVF

| | HNSW | IVF |
|---|---|---|
| Structure | Layered graph | Clusters + centroids |
| Build | Slow | Fast (needs a training pass) |
| Query | Very fast, high recall | Fast, recall depends on `nprobe` |
| Memory | High (graph edges) | Lower |
| Inserts | Handles well | Degrades, needs re-clustering |

> "HNSW is the default in modern vector DBs because it handles incremental inserts gracefully. For Papeer, where documents are uploaded continuously into fresh collections, insert behaviour matters more than IVF's lower memory."

### 🌟 The Qdrant detail that wins this question

> "Qdrant has an `indexing_threshold`, 20,000 vectors by default. Below that it just does a full scan, because building and traversing an HNSW graph costs more than brute-forcing a few thousand vectors. My collections are per-session — a 150-page PDF is about 300 chunks — so in practice most of my collections are doing exact search, not ANN."

### Why Qdrant

> "It supports dense and sparse vectors in the same collection with built-in RRF fusion, so hybrid search doesn't need two systems. Managed cloud tier, and collection-level scoping which I use for session isolation."

### What's stored per point

| Field | Content |
|---|---|
| `id` | unique point ID |
| `vector` | dense (1536-dim) + sparse BM25 |
| `payload` | chunk text + metadata (title, page, source_type) |

"The vector is what you search on, the payload is what you return — the chunk text lives in the payload and that's what gets injected into the prompt."

---

## 9. INGESTION PHASE

**The 4-step version:** Document Loader → Text Splitter → Embedding → Persist to vector DB

**Your version:**

> "The standard pipeline is load, split, embed, store. I added two things: table-aware parsing with OCR fallback before splitting, and dual embeddings — dense plus sparse — so I could do hybrid search at query time."

### Detail

1. **Upload & validate** — `POST /api/upload`, check extension (`.pdf .txt .md .markdown`), write to a temp file (loaders need a path), delete in `finally`
2. **Parse** (`loader.py`)
   - PyMuPDF `page.find_tables()` → `to_markdown()`, kept as **atomic unsplit chunks**
   - Prose blocks that intersect a table bbox are skipped so content isn't duplicated
   - Page under 20 chars and no tables → render at 200 DPI → RapidOCR
   - Total under 50 chars → reject with a clear error
3. **Chunk** — `RecursiveCharacterTextSplitter`, size **1500**, overlap **400**, `add_start_index=True`. Separate splitter for Markdown that also splits on `## ### ####`
4. **Embed** — dense (OpenAI) + sparse (BM25 via fastembed)
5. **Store** — upsert into `papeer_<session_id>`

**Chunk size answer:**
> "Tradeoff. Too small and the chunk lacks context to answer anything. Too large and the embedding gets diluted across topics so it matches nothing well, plus you waste context tokens. Overlap exists so a sentence on a boundary isn't lost from both neighbouring chunks."

**Table answer:**
> "A character-based splitter will happily cut a table in half, and the second half loses its headers — the numbers become meaningless. So tables are extracted first and kept whole."

### Ingestion numbers

| Metric | Value |
|---|---|
| Pages | 150 |
| Chunks | 301 |
| Embedding tokens | 83,845 |
| Embedding cost | ~$0.00085 |
| Loading + splitting | ~25s |
| Embedding + storage | ~12s |
| **Total** | **~40s** |

Parsing dominates, not embedding — that surprises people.

### 🐛 The cost bug

> "Embedding cost was double what I calculated. LangSmith traces showed every batch being embedded twice. It was `QdrantVectorStore`'s `validate_embeddings` pass silently re-embedding each batch before upload. Setting `validate_embeddings=False` halved it — $0.00169 to $0.00085."

---

## 10. RETRIEVAL PHASE

```
Query
 → Router (1.9s)
 → agent_node (2.6s)          decides to call a tool
 → tool_node (7.9s):
      generate 2 query variations        (RAG Fusion)
      embed 3 queries (dense + sparse)
      Qdrant hybrid search ×3            [RRF #1: dense + sparse]
      RRF across variations              [RRF #2]
      Cohere rerank → top 4
 → agent_node (1.0s)
 → relevancy_check (1.4s)
 → generate_answer (6.0s)

Total: 21.25s | 3.9K tokens | $0.0005
```

### Hybrid search

| | Catches | Misses |
|---|---|---|
| Dense | meaning, synonyms | exact terms, rare identifiers |
| Sparse (BM25) | exact keywords | synonyms |

> "Dense embeddings capture meaning but miss exact terms — a product code or a specific identifier gets smoothed away. BM25 catches exact keywords but has no notion of synonyms. Together they cover both failure modes."

### RRF

`score = Σ 1 / (k + rank)`, **k = 60**. Documents ranking well in any list rise to the top.

**The two-RRF answer:**
> "There are two separate RRF steps. Qdrant's internal RRF fuses dense and sparse results *per query*. My RAG Fusion RRF fuses results *across the three query phrasings*. Same algorithm, different problems."

### RAG Fusion

> "A single embedding of a long or compound question gets diluted — it's trying to match too many concepts at once and matches nothing well. Generating variations gives each sub-concept its own shot. I verified this manually: a compound question that returned zero documents on the single-query pipeline returned a fully cited answer once RAG Fusion was added."

The variation prompt is deliberately strict: same meaning, only change terminology, do NOT broaden or add concepts.

### Reranking

> "Embeddings are a **bi-encoder** — query and document are encoded separately, so the comparison is coarse. A reranker is a **cross-encoder**: it feeds the query and document through the model together and can attend across both. Far more accurate, but too slow to run over a whole corpus — so you retrieve broadly with fast methods and rerank only the top candidates."

Reranked against the **original** query, not the variations. Model: `rerank-v4.0-pro`, `top_n=4`.

### Retrieval funnel

`10 per query × 3 queries` → RRF → `15 candidates` → rerank → **4 chunks** to the LLM

---

## 11. CODEBASE MAP

**Open `main.py` first.** It's the entry point and shows you think in request flow.

```
main.py          → API surface, lifespan, CORS, sessions
  └── rag_graph.py    → the LangGraph state machine
        └── vector_store.py  → embeddings, Qdrant, RAG Fusion, rerank
  └── loader.py       → parsing and chunking
```

**Opening line:**
> "There are four files. `main.py` is the API layer, `rag_graph.py` has the LangGraph state machine where the routing lives, `vector_store.py` handles embeddings and retrieval, `loader.py` handles parsing and chunking. Want me to walk through a request end to end?"

### Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/health` | GET | health check |
| `/api/keys/verify` | POST | validate BYOK keys via `models.list()` |
| `/api/session` | POST | create session |
| `/api/conversations` | GET | list sessions |
| `/api/conversations/{id}` | PATCH / DELETE | rename / delete |
| `/api/chat` | POST | run the graph |
| `/api/history/{id}` | GET | replay from checkpointer |
| `/api/upload` | POST | multipart ingestion |
| `/api/load-url` | POST | web page ingestion |
| `/api/documents/{id}` | GET | list ingested titles |

### Key numbers table

| Thing | Value | Where |
|---|---|---|
| Chunk size / overlap | **1500 / 400** | `loader.py:9` |
| OCR trigger | page < **20 chars** | `loader.py:13` |
| Reject document | total < **50 chars** | `loader.py:15` |
| OCR DPI | **200** | `loader.py:81` |
| Embedding dim | **1536** | `vector_store.py:25` |
| RRF k | **60** | `vector_store.py:26` |
| Retriever k | **10** per query | `vector_store.py:246` |
| Fusion pool | **15** | `vector_store.py:27` |
| Rerank output | **top 4** | `vector_store.py:34` |
| Max tool attempts | **3** | `rag_graph.py:23` |
| Max rewrites | **1** | `rag_graph.py:117` |
| Agent model | `gpt-4.1-mini` | `rag_graph.py:30` |
| Everything else | `moonshot-v1-32k` | `rag_graph.py:41` |

**Model tiering:**
> "gpt-4.1-mini handles the agent node because tool-calling reliability matters most there. Everything else — routing, relevancy grading, query rewriting, claim verification, generation — runs on Kimi's moonshot-v1-32k, which is cheaper and adequate for classification and generation-shaped tasks."

### Lifespan

> "The checkpointer and the compiled graph are built once at startup, not per request. `AsyncPostgresSaver` needs an open connection pool for the app's lifetime, and `checkpointer.setup()` creates the checkpoint tables if they don't exist — it's a no-op after the first run."

Bonus: you can explain a context manager — "any object with `__enter__` and `__exit__`; it manages resource setup and teardown so cleanup happens even on failure."

### CORS

`allow_origins=["http://localhost:5173"]` + `allow_origin_regex=r"https://papeer-frontend.*\.vercel\.app"`

> "The regex is there because Vercel generates a new preview URL per deployment — pinning a single origin would break every preview build."

---

## 12. `rag_graph.py` DEEP DIVE

### State + reducers

```python
messages: Annotated[list[BaseMessage], add_messages]
retrieved_docs: Annotated[list[Document], merge_docs]
retrieval_attempts: int          # no reducer → plain overwrite
rewrite_query_count: int | None  # no reducer → plain overwrite
```

> "By default whatever a node returns for a key overwrites it. A reducer changes that to a merge function. `messages` uses `add_messages`, which appends and handles message IDs. `retrieved_docs` uses my own `merge_docs` — it appends and dedupes by content hash, because the same chunk can come back across a retry. The `new == []` case is an explicit reset: router and rewrite_query both clear the pool for a fresh attempt. Without that escape hatch an append-only reducer could never be cleared. Counters have no reducer because they must overwrite, not accumulate."

### The router prompt

- Keyword forcing: report / paper / document / pdf / article / study → `retrieve`
- Section B: current/live info also → `retrieve` (**this was the bug fix**)
- Tiebreaker: *"When in doubt between retrieve and direct_answer, ALWAYS choose retrieve."*

> "Retrieval is the safe default. A wrong `retrieve` costs latency; a wrong `direct_answer` gives the user a hallucinated answer about their own document. I biased toward the cheaper failure mode."

### `agent_node` — three things

**(a) Safety valve**
```python
llm_with_brain = (request_llm if current_attempts >= max_retrieval_attempt
                  else request_llm.bind_tools(tools, parallel_tool_calls=False))
```
> "After 3 attempts the model gets an LLM with no tools bound. It's structurally incapable of emitting a tool call, so the cycle must terminate. That's stronger than prompting — a prompt can be ignored, a missing tool schema can't."

**(b) Message trimming**
```python
current_turn_messages = all_messages[last_human_idx:]
```
> "Checkpointed history grows unboundedly. The agent only needs the current turn to pick a tool, so I slice from the last HumanMessage. Without this, agent-node token cost grows linearly with conversation length."

**(c) Attempt counting** — only increments when a tool is actually requested.

### 🌟 Tool injection — your strongest technical point

```python
@tool(args_schema=RetrieverInput)
async def retrieve_from_vectorstore(
    query: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    session_id: Annotated[str, InjectedState("session_id")],
    openai_api_key: Annotated[str | None, InjectedState("openai_api_key")] = None,
):
```

> "The LLM only sees `query` — that's the `args_schema`. `session_id` and the API keys are marked `InjectedState`, so LangGraph fills them from graph state at call time. The model never sees them and can't hallucinate a session ID or leak a key into a tool call."

```python
return [
    ToolMessage(content=f"Retrieved {len(docs)} chunk(s).", tool_call_id=tool_call_id),
    Command(update={"retrieved_docs": docs}),
]
```

> "Two different destinations. The ToolMessage goes into message history so the LLM knows retrieval happened — but it's a one-line summary, not the chunks. The actual documents go into state via `Command(update=...)`, bypassing the message list entirely. That keeps thousands of tokens of chunk text out of the agent's context. `generate_answer` reads them from state directly."

**Use this if asked "how did you control token cost?"**

### Relevancy grader is deliberately lenient

> "A strict grader triggers unnecessary rewrites, and each rewrite is a full extra retrieval cycle. False negatives are expensive; false positives just mean slightly noisier context that the reranker already filtered."

### `generate_answer` — three branches

| Route | Behaviour |
|---|---|
| `retrieve` | Not relevant → honest failure message. Else build cited context and generate. |
| `verify_claim` | Formats verdict + superseding papers — **no LLM call**, pure string building (why it traces 0.00s) |
| `direct_answer` | Plain LLM call, no context |

Citation format enforced in prompt: `[Source: <Title>, Page X]` for docs, `[Source: <Title>]` for web. Note `page + 1` — PyMuPDF is 0-indexed.

### The graph

```
START → router
          ├── verify_claim ─────→ generate_answer → END
          ├── generate_answer ──→ END              (direct_answer)
          └── agent_node
                ├── tool_node → agent_node                     (cycle 1)
                └── relevancy_check
                      ├── generate_answer → END
                      └── rewrite_query → agent_node           (cycle 2)
```

**Two cycles** = why LangGraph over a chain. **Single exit point** (`generate_answer`) = answer formatting lives in one place.

`tools_condition` — "A LangGraph prebuilt that checks whether the last message has `tool_calls`. I wrap it because I need the non-tool branch to go to `relevancy_check`, not END."

---

## 13. FALLBACKS ⭐

**The 15-second summary — memorize this:**

> "The system has three classes of fallback: **loop bounds** that guarantee termination, **graceful degradation** where an enhancement failing doesn't break the core path, and **honest failure** where the user gets a clear message rather than a hallucinated answer. The one place I deliberately don't fall back is API keys — that fails loudly by design."

| # | Fallback | Mechanism |
|---|---|---|
| 1 | **Tool unbinding** | After 3 attempts, no tools bound → loop must terminate |
| 2 | **Rewrite budget** | Max 1 rewrite, then answer anyway |
| 3 | **RAG Fusion degrades** | Variation LLM fails → `variations = []` → single-query retrieval, no user-visible error |
| 4 | **Honest failure** | Not relevant → "I couldn't find..." + reason + suggestion. Never falls through to general knowledge |
| 5 | **Short-circuit grader** | No docs → skip the grader LLM call entirely |
| 6 | **Collection race** | `create_collection` fails → re-check `collection_exists`, only re-raise if genuinely missing (**check-then-act race**) |
| 7 | **OCR fallback** | Page < 20 chars → OCR. Then fail-fast if total < 50 chars |
| 8 | **Metadata chains** | `title or video_title or source` |
| — | **Deliberate NON-fallback** | Missing API key → `MissingApiKeyError`, no fallback to the dev's `.env` key |

**The gap to volunteer:**
> "There's no LLM provider failover. If OpenAI is down, the agent node fails and the request errors. I have both OpenAI and Kimi wired up, so a try/except that swaps providers is the obvious next step — but it isn't built. There's also no retry with exponential backoff on transient errors."

---

## 14. THE WORKED EXAMPLE ⭐

**Setup:** paper says *"CO₂ sequestration rates"*. User asks about *"carbon capture efficiency"*.

| # | Node | What happens | State after |
|---|---|---|---|
| 1 | `router` | "as per the report" → keyword rule | `route="retrieve"`, `docs=[]`, `attempts=0`, `rewrites=0` |
| 2 | `agent_node` | Tools bound (0 < 3) → tool call | `attempts=1` |
| 3 | `tool_node` | 3 variations → hybrid ×3 → RRF → rerank → 4 chunks | `docs=[4]` + ToolMessage |
| 4 | `agent_node` | Prompt: don't call again if ToolMessage exists → plain text | — |
| 5 | `tool_relevancy_decision` | `tools_condition` ≠ "tools" | → `relevancy_check` |
| 6 | `relevancy_check` | Chunks are about soil carbon, not efficiency | `is_relevant=False` |
| 7 | `check_relevancy` | `rewrites=0 < 1` ✅ | → `rewrite_query` |

**🔁 FALLBACK 2 fires**

| # | Node | What happens | State after |
|---|---|---|---|
| 8 | `rewrite_query` | → `"CO2 sequestration rate report"` | `docs=[]` ← **reducer reset**, `attempts=0`, `rewrites=1` |
| 9 | `agent_node` | Tools bound again | `attempts=1` |
| 10 | `tool_node` | Retrieves with new wording | `docs=[4 new]` |
| 11–12 | `agent_node` → `relevancy_check` | Still not a match | `is_relevant=False` |
| 13 | `check_relevancy` | `rewrites=1`, not `< 1` ❌ | → `generate_answer` |

**🛑 FALLBACK 2 exhausted → FALLBACK 4 fires**

```
I couldn't find information in the retrieved documents that answers your question.
Reason: The chunks discuss soil carbon storage, not capture efficiency metrics.
You can try rephrasing the question or uploading additional documents.
```

**Why step 8 matters most:**
> "Without the `new == []` escape hatch in my reducer, the append-only merge would keep the stale chunks from attempt one, and the grader would judge a polluted mix. The rewrite has to start clean. And the two counters bound different things — `retrieval_attempts` bounds the tool loop, `rewrite_query_count` bounds the retry loop. Resetting attempts on rewrite is safe precisely because the rewrite counter is the outer bound."

**Variant A — model ignores the prompt and keeps calling tools:** attempts hits 3 → tools unbound → FALLBACK 1. "The prompt is the soft guard. Unbinding tools is the hard guard."

**Variant B — Kimi down during variation generation:** `variations = []` → single-query retrieval → FALLBACK 3. Slightly worse recall, zero user-visible error.

**The 20-second version:**
> "Say the user's wording doesn't match the document's. First retrieval comes back irrelevant, so the grader sends it to rewrite_query — that rephrases, wipes the stale doc pool via the reducer, and resets the tool budget. Second attempt still fails, but the rewrite counter is spent, so it goes to generate_answer, which returns an honest 'not found' with the reason. Three bounded fallbacks, and at no point does it loop forever or invent an answer."

---

## 15. PERFORMANCE, EVALUATION, DEPLOYMENT

### Benchmarks (LangSmith)

| Route | Latency | Tokens | Cost |
|---|---|---|---|
| Direct Answer | 4.36s | 620 | — |
| Verify Claim | 8.01s | 1.3K | — |
| Web Search | 10.87s | 3.6K | $0.0005 |
| **Retrieve** | **21.25s** | **3.9K** | **$0.0005** |

**Bottleneck:**
> "Two full agent_node ↔ tool_node round trips. Each costs a router-level LLM call just to decide whether to call a tool again, on top of the actual retrieval work."

**Optimizations left:** skip query expansion for short queries; cache embeddings for repeated queries; stream tokens to cut perceived latency; make ingestion a background job.

### Evaluation — v1, DeepEval (be clear it's v1)

| Metric | Dense + Rerank | Hybrid + Rerank |
|---|---|---|
| Contextual Precision | 0.83 | **0.84** |
| **Contextual Recall** | 0.77 | **0.86** |
| Contextual Relevancy | 0.82 | **0.84** |
| Answer Relevancy | 0.83 | **0.88** |
| Faithfulness | 0.84 | **0.87** |

Recall **+11.7% relative**. "Excellent" responses **28% → 44%**.

> "That's from the v1 Streamlit prototype — it's what motivated carrying hybrid search into v2. The v2 eval pipeline isn't rebuilt yet, so RAG Fusion has only been validated manually. Rebuilding it is on the roadmap."

**Metric definitions:** Contextual Recall = did retrieval find everything needed. Contextual Precision = are relevant chunks ranked high. Faithfulness = is the answer supported by context. Answer Relevancy = does it address the question.

### Deployment

- Backend on **Render** — `uvicorn main:app --host 0.0.0.0 --port $PORT` (`0.0.0.0` binds all interfaces; no `--reload` in prod)
- Frontend on **Vercel** — `VITE_API_BASE` env var
- Backend is **stateless** — all state in Postgres and Qdrant → scales horizontally behind a load balancer
- Real limits: external API rate limits, Qdrant collection count

**"How would you scale this?"**
> "The backend is stateless so horizontal scaling works. At real scale I'd move ingestion to a queue with workers, and reconsider collection-per-session in favour of metadata filtering — thousands of collections becomes a management problem."

### Session isolation tradeoff

> "I isolate at the collection level rather than filtering by metadata in a shared collection. Collection-per-session gives hard isolation — no query path can accidentally return another session's documents. The tradeoff is more collections to manage, and deletion has to be a coordinated cleanup: checkpoints, collection, and metadata all go together."

---

## 16. WEAKNESSES — VOLUNTEER THESE ⭐

Naming your own gaps beats being caught.

1. **No auth** — "Session IDs aren't protected. Anyone with an ID can access that session. Production needs auth on every endpoint plus ownership checks."
2. **Upload blocks ~40s** — "Risks proxy/gateway timeouts on large files. Should return a job ID immediately and process in a background worker with polling."
3. **Three sync nodes block the event loop** — "`agent_node`, `relevancy_check`, and `rewrite_query` are plain `def` calling `.invoke()`. They should be `async def` with `ainvoke`, or wrapped in `to_thread`. Same class of bug I already fixed on the Qdrant side."
4. **No v2 eval pipeline** — DeepEval numbers are v1; RAG Fusion validated only manually.
5. **Delete isn't transactional** — "Checkpoint, Qdrant collection, and Supabase row are deleted in sequence with no transaction. If one fails you get orphans."
6. **No provider failover or retry/backoff.**
7. **`hash()` is per-process salted** — fine for in-request dedup, not stable across restarts.
8. **Tavily key is app-owned** while OpenAI/Kimi are BYOK — inconsistent with the stated principle.

### The three bug stories (one per skill)

| Bug | Skill it shows |
|---|---|
| **Double embedding** — `validate_embeddings=False`, cost halved | Instrumentation → anomaly → root cause → measured fix |
| **IPv6** — Supabase direct is IPv6-only, Render has no outbound IPv6 → Session Pooler | Deployment/networking debugging |
| **Router misclassification** — current-event questions → `direct_answer`; fixed by broadening prompt examples | Prompt engineering, no code change needed |

---

## 17. RAPID-FIRE Q&A

**Why LangGraph not a chain?** Two cycles: agent↔tool, and relevancy→rewrite→agent. A chain is a DAG — it can't route backwards.

**Why Qdrant?** Dense + sparse in one collection with built-in RRF; collection-level scoping for session isolation; managed cloud tier.

**Why two databases?** Different jobs. Qdrant does ANN over embeddings; Postgres does relational history and transactions.

**Why hybrid search?** Dense misses exact terms, BM25 misses synonyms. Measured +11.7% relative recall.

**Why rerank after retrieving?** Bi-encoder vs cross-encoder. Cross-encoders are far more accurate but too slow for a whole corpus — retrieve broadly, rerank the top.

**Why two RRF steps?** Qdrant's fuses dense+sparse per query; mine fuses across query variations.

**Why temperature 0 for the router?** Classification should be deterministic — the same query should always take the same route.

**Why structured output instead of parsing text?** Free-text parsing is brittle. `with_structured_output` uses function-calling and validates against a Pydantic schema — valid enum or error, no regex.

**How do you control token cost?** Chunks bypass message history via `Command(update=...)`; agent messages trimmed to the current turn; cheap model for everything except tool calling.

**What if Qdrant is down?** Retrieval and ingestion fail. Direct answers and verify_claim still work — they don't touch Qdrant.

**Concurrent uploads to one session?** Qdrant upserts are idempotent by point ID. What isn't guarded is a delete mid-ingestion — that needs a lock or status flag.

**How do you know retrieval is good?** DeepEval on v1 (recall 0.77→0.86). v2 is manual verification only — rebuilding the pipeline is on the roadmap.

**What would you do next?** Auth, background-job ingestion, fix the three sync nodes, rebuild the eval pipeline, provider failover, token streaming.

---

## 18. MOCK ANSWERS — SPOKEN, WORD FOR WORD

### The opening (Q1) — 3 sentences, then hand off

> "Papeer is a full-stack RAG application where users upload documents or paste URLs and have a conversational chat with that content.
>
> I built it because LLMs can't access your private or recent documents, and the existing tools use basic RAG that retrieves on every query — which is slow and often inaccurate.
>
> So I built a **LangGraph router that classifies each query and sends it to the right workflow** — retrieve from the document, verify a claim, or answer directly. And for retrieval I use hybrid search with reranking instead of plain vector search, which measurably improved recall.
>
> Want me to go deeper on the architecture, or into the code?"

**Rule:** don't spend the opening explaining the *problem*. Spend it explaining your *solution*. Knowledge-cutoff material is a follow-up answer, not an opener. Always land on a full stop.

### The walkthrough — three layers, never an endpoint tour

1. **Verbal, 30s, no screen** — the pitch above, then ask where they want to go
2. **The diagram** — trace one path: START → router → 3 routes → agent loop → single exit at generate_answer
3. **Code, following ONE request** — 6–7 stops, not 10:

| Stop | File | Line to say |
|---|---|---|
| 1 | `main.py` lifespan | "Checkpointer and graph compiled once at startup, not per request" |
| 2 | `main.py` `/api/chat` | "Builds initial state, `thread_id = session_id`, calls `graph.ainvoke`" |
| 3 | `rag_graph.py` router | "Structured output, 3-way, temperature 0" |
| 4 | `rag_graph.py` agent_node | "Tools bound — unbound after 3 attempts so the loop must terminate" |
| 5 | `rag_graph.py` the tool | "`InjectedState` — model only sees `query`" |
| 6 | `vector_store.py` `search()` | "3 variations, parallel hybrid, RRF, rerank to top 4" |
| 7 | `rag_graph.py` generate_answer | "Cited output, or honest 'not found'" |

> An endpoint list says "I built CRUD." A traced request says "I understand my own control flow."

### Q — "What's the router routing between, and how does it decide?"

> "It's a 3-way classification: retrieve, verify_claim, or direct_answer. It's an LLM call with `with_structured_output` bound to a Pydantic model with a `Literal` of those three values — so I get a validated enum, not free text I have to parse. Temperature zero, because classification should be deterministic.
>
> The prompt has explicit keyword rules — 'report', 'paper', 'document', 'PDF', 'as per the report' always go to retrieve. And a tiebreaker: when in doubt between retrieve and direct_answer, always choose retrieve. That bias is deliberate — a wrong retrieve costs latency, a wrong direct_answer gives the user a hallucinated answer about their own document. I biased toward the cheaper failure mode.
>
> One thing worth noting — web search isn't a router route. It's one of two tools under the retrieve path. So it's a two-level decision: the router picks the workflow, the agent node picks the tool."

### Q — "21 seconds is a long time. Where does it go?"

> "Router about 2s, first agent node 2.6, tool node about 8, second agent node 1, relevancy check 1.4, generation 6.
>
> The bottleneck isn't the retrieval work — it's two full agent_node ↔ tool_node round trips. Each costs a router-level LLM call just to decide whether to call a tool again. Pure overhead on top of the actual search.
>
> Inside the tool node the three RAG Fusion queries already run concurrently via `asyncio.gather` with `to_thread`, since the Qdrant client is sync.
>
> Next steps: skip query expansion for short queries where dilution isn't a problem — that saves an LLM call plus two retrievals. Cache embeddings for repeated queries. And stream tokens so the user sees output at 2 seconds instead of 21 — that doesn't cut total latency but perceived latency is what matters in a chat interface."

### Q — "Why LangGraph instead of a plain chain?"

> "Two cycles: agent_node → tool_node → agent_node, and relevancy_check → rewrite_query → agent_node. A chain is a DAG — it can't route backwards. Plus the router needs conditional branching to three paths. LangGraph gives me typed state, conditional edges that read that state, cycles, and checkpointing for free — state persists to Postgres after each node, which is how conversations survive a redeploy."

### Q — "Why hybrid search? Wasn't vector search enough?"

> "Dense embeddings capture meaning but smooth away exact terms — a product code or rare identifier often won't match. BM25 catches exact keywords but has no notion of synonyms. They fail in opposite directions, so running both and fusing covers both.
>
> I measured it on v1 with DeepEval: contextual recall 0.77 → 0.86, about 11.7% relative, and 'excellent' responses went 28% → 44%. That's what justified carrying hybrid into v2. Qdrant supports dense and sparse in one collection with built-in RRF, so it doesn't need two systems."

### Q — "Explain reranking. Why not just take the top vector results?"

> "Embedding search is a bi-encoder — query and document are encoded separately, then compared. Fast but coarse, because neither encoding ever saw the other. A reranker is a cross-encoder — it feeds query and document through the model together so it can attend across both. Much more accurate, far too slow for a whole corpus.
>
> So it's a funnel: 10 per query across 3 variations, fuse to 15 with RRF, rerank those 15 and keep the top 4."

### Q — "How do you keep token costs down?"

> "Three things. First, retrieved chunks never enter message history — the tool returns a one-line ToolMessage and the documents go into state via `Command(update=...)`. Otherwise thousands of tokens of document text get resent on every turn. Second, the agent node trims messages to the current turn by slicing from the last HumanMessage. Third, model tiering — gpt-4.1-mini only for the agent node where tool-calling reliability matters, Kimi for everything else. A retrieve-path query lands around $0.0005."

### Q — "Tell me about a bug you found and fixed."

> "Ingestion embedding cost was about double what my own token math said. LangSmith traces showed every batch being embedded twice — it was `QdrantVectorStore`'s `validate_embeddings` pass silently re-embedding each batch before upload. Setting it to False halved the cost, $0.00169 to $0.00085 for a 150-page PDF.
>
> What I take from it is that I'd never have found it by reading code. It only surfaced because I had per-call tracing and was checking measured cost against expected cost."

### Q — "How would you scale this?"

> "The backend is stateless — all state is in Postgres checkpoints and Qdrant — so it scales horizontally behind a load balancer. The real constraints are elsewhere: ingestion runs inside the HTTP request and takes ~40s for a large PDF, which risks gateway timeouts, so that should be a queue with background workers returning a job ID. External API rate limits would bite before my own compute does. And collection-per-session is great for hard isolation but becomes a management problem at thousands of collections — at scale I'd move to a shared collection with metadata filtering."

### Q — "What would you do differently?"

> "Auth first — sessions are isolated but not authenticated, so anyone with a session ID can read it. Ingestion should be a background job, not a blocking request. And I'd finish the async migration — I fixed the Qdrant side with `to_thread`, but three graph nodes still call `.invoke()` synchronously and block the event loop. Same class of bug, just not cleaned up yet."

---

## 19. LAST 10 MINUTES — SKIM ONLY THIS

- **Pitch:** §1 — What / Why / How, three separate answers
- **Corrections:** router has **3 routes**; retrievals **already parallelized**; app is **BYOK**; one **dead edge** in the diagram
- **Numbers:** 1500/400 chunks · 1536 dims · k=60 · 10→15→4 funnel · 3 attempts · 1 rewrite · 21.25s · $0.0005 · 150pg→301 chunks→40s
- **Best three talking points:** `InjectedState` + `Command(update=...)` for token control · tool unbinding as a hard loop bound · the double-embedding cost bug found via LangSmith
- **Fallback summary sentence:** loop bounds, graceful degradation, honest failure — and one deliberate non-fallback
- **Volunteer:** no auth · 40s blocking upload · three sync nodes blocking the loop · no v2 eval
