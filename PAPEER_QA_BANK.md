# Papeer — Complete Q&A Bank

Every question they're likely to ask, with the answer to give. Read top to bottom.
Answers in `>` blocks are meant to be **spoken close to word-for-word**.

**Contents**
1. Project overview
2. Backend fundamentals
3. FastAPI, ASGI, async
4. HTTP & API design
5. LangChain, LangGraph, LangSmith
6. RAG concepts
7. Embeddings, vector DBs, ANN/HNSW
8. Ingestion pipeline
9. Retrieval pipeline
10. Your code specifically
11. Fallbacks & failure handling
12. Performance & cost
13. Persistence & sessions
14. Deployment & scaling
15. Security & weaknesses
16. Evaluation
17. Behavioral & design-judgment
18. Curveballs
19. Cheat card

---

# 1. PROJECT OVERVIEW

### Q: Tell me about this project.

> "Papeer is a full-stack RAG application where users upload documents or paste URLs and have a conversational chat with that content.
>
> I built it because LLMs can't access your private or recent documents, and the existing tools use basic RAG that retrieves on every query — which is slow and often inaccurate.
>
> So I built a LangGraph router that classifies each query and sends it to the right workflow — retrieve from the document, verify a claim, or answer directly. And for retrieval I use hybrid search with reranking instead of plain vector search, which measurably improved recall.
>
> Want me to go deeper on the architecture, or into the code?"

**Rules:** 3 sentences. Don't explain the problem for four sentences — explain your solution. Always end on a full stop, then hand off.

### Q: Why did you build this? What problem does it solve?

> "Three real users: a student reading a 60-page paper who'd otherwise skim for hours, a researcher doing a literature review who needs to verify claims against current sources, and an engineer exploring technical docs. The common thread is that reading documents is manual and slow, and LLMs can't help because they've never seen your document."

### Q: ChatGPT already does file upload. Why build this?

> "Four differences. ChatGPT always retrieves — Papeer routes, so a general question skips the vector store entirely and saves both latency and cost. I use hybrid search plus reranking plus RAG Fusion rather than naive vector search. I have multi-session isolation where each conversation has its own document collection. And there's a dedicated claim-verification workflow that checks a statement against live web sources."

### Q: What are the four problems RAG solves?

> "Knowledge cutoff — the model was trained on static data and doesn't know anything recent. Hallucination — grounding answers in retrieved text means the model either found it or it didn't. Source attribution — the user can see which chunk the answer came from. And private data — the LLM has never seen your documents and can't be trained on them.
>
> For Papeer specifically, RAG wasn't just a technical choice, it was the only architecture that solves the core problem: reliably answering questions about documents the LLM has never seen."

### Q: Why RAG instead of fine-tuning?

> "Fine-tuning teaches a model style and behaviour, not facts — and it's a poor fit for facts that change or are per-user. Every new document would need a retraining run, you'd have no source attribution, and you couldn't isolate one user's documents from another's. RAG gives you all three: instant updates, citations, and hard isolation."

### Q: What's your tech stack and why?

> "FastAPI and Uvicorn on the backend, React with Vite on the frontend. LangGraph for orchestration, LangChain for the components. Qdrant for vectors, Supabase Postgres for conversation checkpoints and session metadata. OpenAI for embeddings and the agent node, Kimi for everything else, Cohere for reranking, Tavily for web search, LangSmith for tracing. Render for the backend, Vercel for the frontend."

---

# 2. BACKEND FUNDAMENTALS

### Q: What is a backend?

> "The server-side part of an application that the user never interacts with directly. It handles business logic, data processing, database operations, and communication with external services."

**Then immediately make it yours:**

> "In Papeer, my backend is a FastAPI app that handles document ingestion — parsing, chunking, embedding, storing in Qdrant — orchestrates the LangGraph query routing, and manages session persistence in Postgres."

### Q: What's a server?

> "Two senses. In software terms it's the program listening on a port and handling requests — that's Uvicorn. In hosting terms it's the machine running it — that's Render."

### Q: What's an API?

> "A defined interface that lets two programs communicate. It specifies what endpoints exist, what data format to send, and what comes back. In Papeer the React frontend calls the FastAPI backend over HTTP — but it's not only frontend to backend; my backend also calls OpenAI's, Cohere's, and Tavily's APIs."

### Q: What's a database? Why do you have two?

> "A database persists data after a request ends. I have two because they solve different problems. Qdrant is a vector database built for approximate nearest-neighbour search over embeddings — it isn't designed for relational queries, transactions, or conversation history. Postgres handles the structured relational side: session metadata and LangGraph checkpoints. Using one for the other's job would be a bad fit in both directions."

### Q: Framework vs library?

> "A library is code you call — you stay in control of the flow. A framework calls your code — it owns the flow and you plug into it. That's inversion of control. FastAPI is a framework: I define endpoint functions, but FastAPI decides when to invoke them based on incoming requests. Pydantic is a library — I call it to validate."

**If pushed on LangChain:** "LangChain is more of a library in how I use it — I call its components directly. LangGraph is the actual framework, since it controls execution order across my nodes."

### Q: What's a context manager?

> "Any object with `__enter__` and `__exit__` defined. It manages resource setup and teardown so cleanup happens even if the block raises. I use one for the FastAPI lifespan — `AsyncPostgresSaver.from_conn_string` opens a connection pool on entry and closes it on shutdown."

---

# 3. FASTAPI, ASGI, ASYNC

### Q: Why FastAPI?

> "The main reason is async. My requests spend most of their time waiting on external APIs — OpenAI, Qdrant, Cohere. FastAPI is async-native, so the server can handle other requests during that wait instead of sitting blocked."

**Backups in your pocket:** Pydantic auto-validation (422 before your code runs), auto-generated Swagger docs at `/docs`, Python-native AI ecosystem so no extra service boundary.

**Do not say** "because it's fast" or "because it's modern." Those are marketing lines.

### Q: What's the difference between FastAPI and Uvicorn?

> "FastAPI is the framework — it defines what my endpoints are and what they do. Uvicorn is the ASGI server that actually listens on a port, accepts HTTP connections, and hands them to FastAPI. FastAPI doesn't run on its own. In production I run `uvicorn main:app --host 0.0.0.0 --port $PORT`."

### Q: What is ASGI?

> "The standard interface between a Python web server and a Python application. It defines the contract for how requests and responses are passed between them, asynchronously. Because it's a standard, any ASGI server can run any ASGI framework."

**Don't drop the word "asynchronously" — that's what distinguishes it from WSGI.**

### Q: ASGI vs WSGI?

| | WSGI | ASGI |
|---|---|---|
| Model | Synchronous | Asynchronous |
| Request handling | One request blocks a worker | Worker switches while awaiting I/O |
| Frameworks | Flask, classic Django | FastAPI, Starlette |
| Streaming / WebSockets | No | Yes |

> "My requests take 4 to 21 seconds, mostly waiting on OpenAI, Qdrant and Cohere. Under WSGI each request holds a worker for that entire duration — a few concurrent users would exhaust the pool. With ASGI, while one request awaits a network call the event loop serves others. That's why FastAPI was the right choice for an LLM-heavy backend."

### Q: What is `asyncio.to_thread` and why is it in your code?

> "It offloads a blocking function to a threadpool so it doesn't block the event loop. The Qdrant Python client is synchronous, so calling it directly inside an async handler would stall every other request for the duration of the call."

**Where you use it:** `_get_vectorstore_sync`, `add_documents`, `base_retriever.invoke` (inside `gather`), `_list_papers_sync`, `_delete_collection_sync`.

### Q: Why not use the async Qdrant client?

> "There is one — `AsyncQdrantClient`. I had that version and reverted it; it's commented out at the bottom of `vector_store.py`. `QdrantVectorStore` from `langchain_qdrant` expects the sync client for hybrid retrieval mode, so the `to_thread` wrapper was the reliable path."

### Q: Does `to_thread` scale?

> "Not indefinitely. The default threadpool is bounded, so under heavy load you queue on threads instead of blocking the loop — better, but not free. A native async client would be the proper fix."

### Q: What if a library isn't async?

> "Then calling it directly blocks the event loop, which defeats the purpose. You wrap it in `run_in_threadpool` or `asyncio.to_thread` so it runs on a separate thread."

### Q: What's the lifespan for?

> "It runs once on startup and once on shutdown. That's where I open the Postgres connection pool and compile the graph — once, not per request. `AsyncPostgresSaver` needs an open pool for the app's lifetime, and `checkpointer.setup()` creates the checkpoint tables if they don't exist. Opening a DB connection per request is a classic performance bug."

### Q: Why not Node? Why not Django?

> **Node:** "Also async and would handle the concurrency fine. But my whole pipeline is LangChain, LangGraph, and Qdrant's Python client. Node would mean either a weaker JS port or a separate Python service just for the AI layer — an unnecessary network hop and deployment surface."
>
> **Django:** "Too heavy. I don't need an ORM, admin panel, or template engine. My frontend is React and my data lives in Postgres checkpoints and Qdrant, not Django models."

---

# 4. HTTP & API DESIGN

### Q: Walk me through the HTTP methods.

| Method | Purpose | Body | Idempotent |
|---|---|---|---|
| GET | Read | No | Yes |
| POST | Create / trigger | Yes | **No** |
| PUT | Replace entirely | Yes | Yes |
| PATCH | Partial update | Yes | Yes |
| DELETE | Remove | No | Yes |

### Q: What does idempotent mean?

> "Calling it N times has the same effect as calling it once. `DELETE /sessions/123` five times — the session is gone, same result. `POST /sessions` five times creates five sessions. That's why POST is the one you shouldn't blindly retry on a network failure."

### Q: PUT vs PATCH?

> "PUT replaces the entire resource — you send the full object and any field you omit gets wiped. PATCH sends only the fields you want to change. In Papeer, renaming a session is a PATCH because I'm only changing the title, not replacing the whole record."

### Q: Which status codes do you use?

| Code | When |
|---|---|
| 200 / 201 / 204 | OK / Created / No Content (delete) |
| 202 | Accepted — request taken, processing async |
| 400 | Bad request (unsupported file type, missing key) |
| 404 | Session not found |
| 413 | File too large |
| **422** | **FastAPI's automatic Pydantic validation failure** |
| 500 | Unhandled server error |
| 504 | Upstream (OpenAI) timeout |

**Mentioning 422 signals you've actually used FastAPI.**

### Q: What endpoints do you have?

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/health` | GET | health check for Render |
| `/api/keys/verify` | POST | validate user-supplied keys via `models.list()` |
| `/api/session` | POST | create a session |
| `/api/conversations` | GET | list sessions |
| `/api/conversations/{id}` | PATCH / DELETE | rename / delete |
| `/api/chat` | POST | run the graph |
| `/api/history/{id}` | GET | replay conversation from checkpointer |
| `/api/upload` | POST | multipart file ingestion |
| `/api/load-url` | POST | web page ingestion |
| `/api/documents/{id}` | GET | list ingested document titles |

### Q: How does Pydantic help?

> "Request models give automatic validation — if the client sends a bad payload, FastAPI returns a 422 with a detailed error before my function ever runs, so I never write manual validation code. It also generates the OpenAPI schema, which is where the free Swagger docs come from."

### Q: Why do you use `with_structured_output` instead of parsing text?

> "Free-text parsing is brittle — the model might say 'Route: retrieve' or 'I think this should be retrieved'. `with_structured_output` uses the model's function-calling API and validates against a Pydantic schema, so I either get a valid enum value or an error. No regex parsing."

---

# 5. LANGCHAIN, LANGGRAPH, LANGSMITH

### Q: What's LangChain?

> "A framework of standard building blocks for LLM applications — models, prompts, document loaders, text splitters, vector stores, retrievers. It standardises the interfaces so I can swap OpenAI for Kimi, or Qdrant for another store, without rewriting logic."

### Q: What's LangGraph? How is it different?

> "LangGraph is built on LangChain and replaces linear chains with a stateful graph. You define State as a typed dict, Nodes as functions that read state and return updates, Edges for what runs next, and conditional edges that branch based on state.
>
> LangChain is the components; LangGraph is the control flow. A chain is a straight line A → B → C. A graph can branch, loop, and revisit."

### Q: Why LangGraph instead of a plain chain?

> "Two cycles. `agent_node → tool_node → agent_node`, and `relevancy_check → rewrite_query → agent_node`. A chain is a DAG — it can't route backwards. Plus the router needs conditional branching to three paths. LangGraph also gives me checkpointing for free — state persists to Postgres after each node, which is how conversations survive a redeploy."

### Q: What's a reducer?

> "By default, whatever a node returns for a state key overwrites that key. A reducer changes that to a merge function. My `messages` field uses LangGraph's `add_messages`, which appends and handles message IDs. My `retrieved_docs` field uses a custom `merge_docs` reducer that appends and dedupes by content hash. Counters like `retrieval_attempts` have no reducer, because they must overwrite, not accumulate."

### Q: Write a minimal LangGraph.

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
app.invoke({"question": "What is BM25?"})
```

**Conditional edge:**
```python
g.add_conditional_edges("router", decide, {
    "retrieve": "retrieve",
    "web_search": "web_search",
    "direct_answer": "direct_answer",
})
```

**Memory:**
```python
app = g.compile(checkpointer=MemorySaver())
config = {"configurable": {"thread_id": "session-1"}}
app.invoke({"question": "hi"}, config=config)
```

### Q: The 4 things to know about LangGraph

1. **State** = TypedDict
2. **Node** = `function(state) -> dict`, returned dict merges into state
3. **Edge** = `add_edge(a, b)` — always a → b
4. **Conditional edge** = `add_conditional_edges(a, fn, mapping)` — fn returns a string, mapping picks the next node

### Q: What's LangSmith for?

> "Observability. Every node execution is traced — inputs, outputs, latency, token usage, cost. Without it, debugging a multi-step retrieval pipeline is nearly impossible. It's also how I found my embedding cost bug: the traces showed every batch being embedded twice."

### Q: What's `Annotated` doing?

> "`Annotated[TYPE, metadata]` attaches metadata to a type hint without changing the type. Python ignores the metadata at runtime — it's there so libraries can read it. LangGraph reads it to know which reducer applies to a state key. ToolNode reads it to know which arguments to inject rather than expose to the model. It doesn't validate anything itself — Pydantic does that."

### Q: What's `Command`?

> "It lets a **tool** update graph state directly. Normally only nodes can write to state — a tool's return value just becomes a ToolMessage in the message history. `Command(update=...)` gives the tool that same power, which is how I route retrieved documents into `retrieved_docs` instead of into the message list. `Command` can also carry `goto` to control flow, though I use conditional edges for that."

---

# 6. RAG CONCEPTS

### Q: Explain RAG.

> "Retrieval-Augmented Generation. You retrieve relevant passages from a corpus and give them to the model as context, so the answer is grounded in real sources rather than model memory. Two phases: ingestion, which is load, split, embed, store; and retrieval, which is embed the query, search, rerank, generate."

### Q: Walk me through the ingestion phase.

> "Four steps: document loader, text splitter, embedding, persist to the vector database.
>
> I added two things on top. Table-aware parsing with OCR fallback before splitting, and dual embeddings — dense plus sparse — so I can do hybrid search at query time."

### Q: Walk me through the retrieval phase.

> "Query comes in. The router classifies intent. On the retrieve path, an agent node calls the retrieval tool. Inside that: two extra query phrasings are generated, all three queries are embedded dense and sparse, Qdrant runs hybrid search for each and fuses dense with sparse internally via RRF, then I fuse across the three variations with a second RRF, and Cohere reranks the pooled candidates against the original query. A relevancy check runs, and then the answer is generated with citations."

### Q: Why chunk at all? Why not embed the whole document?

> "Two reasons. Embedding models have a token limit. And more importantly, a single embedding of a whole document averages out to nothing useful — it can't match a specific question. Chunking gives you granular units that can each match precisely, and it means you only inject the relevant pieces into the prompt rather than the whole document."

### Q: How did you choose chunk size?

> "1500 characters with 400 overlap. It's a tradeoff. Too small and the chunk lacks enough context to answer anything. Too large and the embedding gets diluted across multiple topics so it matches nothing well, plus you waste context tokens on irrelevant text. The overlap exists so a sentence sitting on a chunk boundary isn't lost from both neighbouring chunks."

### Q: What's `RecursiveCharacterTextSplitter` doing?

> "It tries separators in priority order and only falls through to a more aggressive one if the chunk is still too big. Mine goes paragraph breaks, bullet markers, newlines, sentence boundaries, spaces, then raw characters. That way it splits at the most natural boundary available. I have a separate splitter for Markdown that also splits on heading levels."

---

# 7. EMBEDDINGS, VECTOR DBs, ANN/HNSW

### Q: What's an embedding?

> "A fixed-length vector of floats produced by a model, positioned so semantically similar text lands close together in vector space. Retrieval is then a nearest-neighbour search in that space. I use `text-embedding-3-small`, which is 1536 dimensions."

### Q: Why cosine similarity and not Euclidean distance?

> "Cosine measures the angle between vectors, ignoring magnitude. Two passages about the same topic should score as similar regardless of length, and magnitude scales with length. Cosine normalises that away."

### Q: What's a vector database and why can't you use Postgres?

> "A vector DB is built for approximate nearest-neighbour search. Comparing a query against 300 chunks is fine anywhere. Against ten million it isn't — you need an index that finds close vectors without scanning everything. That's what HNSW does, and it's the whole reason vector DBs exist."

### Q: ENN vs ANN?

> "ENN — exact nearest neighbour — compares against every vector. 100% accurate, slow, linear in corpus size. ANN — approximate — uses an index to skip most comparisons. Very fast, slight recall loss. The tradeoff is the entire point."

### Q: What's HNSW?

> "Hierarchical Navigable Small World — a graph-based index, essentially a skip list generalized to graphs. Upper layers are sparse with few nodes, layer zero contains everything. Search starts at the top layer and does greedy nearest-node traversal, descending layer by layer until it reaches layer zero, then does local refinement."

### Q: Why the hierarchical layers?

> "Upper layers are sparse, so a few hops cover huge distances in vector space — like express lanes. Each descent narrows the region, so by layer zero you're already near the target and only need local refinement. Without layers, greedy search on a single flat graph takes far more hops and can get stuck in a local minimum."

### Q: What's IVF? What's PQ?

> "IVF is clustering-based indexing — embeddings are grouped into clusters, each with a centroid. At query time you compare against centroids, pick the closest clusters, and only search inside those.
>
> PQ, Product Quantization, isn't an index — it's compression. It splits a vector into subvectors and replaces each with a centroid ID, shrinking 1536 floats to a few bytes. It's usually combined with an index, like IVF-PQ."

### Q: HNSW vs IVF — which and why?

| | HNSW | IVF |
|---|---|---|
| Structure | Layered graph | Clusters + centroids |
| Build | Slow | Fast, needs a training pass |
| Query | Very fast, high recall | Fast, recall depends on `nprobe` |
| Memory | High (graph edges) | Lower |
| Inserts | Handles well | Degrades, needs re-clustering |

> "HNSW is the default in most modern vector DBs because it handles incremental inserts gracefully. IVF needs a training pass over representative data and degrades as the distribution shifts. For Papeer, where documents are uploaded continuously into fresh collections, insert behaviour matters more than IVF's lower memory."

### Q: 🌟 Does your data even use ANN?

**This one wins the question. Volunteer it.**

> "Qdrant has an `indexing_threshold`, 20,000 vectors by default. Below that it just does a full scan, because building and traversing an HNSW graph costs more than brute-forcing a few thousand vectors. My collections are per-session — a 150-page PDF is about 300 chunks — so in practice most of my collections are doing exact search, not ANN."

### Q: How would you tune HNSW?

| Param | Meaning | Tradeoff |
|---|---|---|
| `m` | edges per node (default 16) | higher = better recall, more memory |
| `ef_construct` | candidates during build (default 100) | higher = better graph, slower indexing |
| `ef` / `hnsw_ef` | candidates during search | higher = better recall, slower query |

> "Recall vs latency is tunable at query time via `ef` — you don't have to rebuild the index to trade accuracy for speed."

### Q: Why Qdrant over Pinecone/Weaviate/Chroma?

> "It supports dense and sparse vectors in the same collection with built-in RRF fusion, so hybrid search doesn't need two systems. It has a managed cloud tier, and collection-level scoping which I use for session isolation — each session gets its own collection named `papeer_<session_id>`."

### Q: What's actually stored in Qdrant?

| Field | Content |
|---|---|
| `id` | unique point ID |
| `vector` | dense (1536-dim) + sparse BM25 |
| `payload` | chunk text + metadata (title, page, source_type) |

> "The vector is what you search on, the payload is what you return. The chunk text lives in the payload, and that's what gets injected into the prompt."

---

# 8. INGESTION PIPELINE

### Q: What happens when I upload a PDF?

> "It hits `/api/upload`, I validate the extension, and write it to a temp file because the loaders need a path — that gets deleted in a `finally` block.
>
> Then `loader.py` opens it with PyMuPDF. For each page it first finds tables and converts them to markdown, kept as atomic unsplit chunks. Prose blocks that overlap a table's bounding box are skipped so content isn't duplicated. If a page yields under 20 characters and has no tables, it's rendered at 200 DPI and passed through OCR.
>
> Then the prose pages go through the recursive splitter at 1500 characters with 400 overlap. Each chunk gets two embeddings — dense from OpenAI, sparse BM25 from fastembed — and both go on the same Qdrant point, upserted into that session's collection.
>
> A 150-page PDF gives about 301 chunks and takes roughly 40 seconds."

### Q: Why the special table handling?

> "A character-based splitter will happily cut a table in half, and the second half loses its column headers — the numbers become meaningless. So tables are extracted before chunking and kept whole as single chunks."

### Q: Why OCR?

> "Scanned or image-only PDF pages have no extractable text layer — they'd produce empty chunks. If a page yields under 20 characters and has no tables, I render it and run RapidOCR. And there's a guard behind it: if the whole document still has under 50 characters after OCR, I reject the upload with a clear error rather than silently ingesting an empty document."

### Q: What are your ingestion numbers?

| Metric | Value |
|---|---|
| Pages | 150 |
| Chunks | 301 |
| Embedding tokens | 83,845 |
| Embedding cost | ~$0.00085 |
| Loading + splitting | ~25s |
| Embedding + storage | ~12s |
| **Total** | **~40s** |

> "Parsing dominates, not embedding — that surprises people."

### Q: 🐛 Tell me about the embedding cost bug.

> "Ingestion embedding cost was about double what my own token math said it should be. I have LangSmith tracing on the whole pipeline, and the traces showed every batch being embedded twice.
>
> It was `QdrantVectorStore`'s `validate_embeddings` pass — it silently re-embeds every batch to check dimensions before upload. Setting `validate_embeddings=False` halved the cost, from about $0.00169 to $0.00085 for a 150-page PDF.
>
> What I take from it is that I'd never have found it by reading code. It only surfaced because I had per-call tracing and was checking measured cost against expected cost."

### Q: 40 seconds is a long HTTP request. Is that a problem?

> "It is. Right now ingestion runs synchronously inside the request, which risks proxy or gateway timeouts on large files. The right fix is to make it a background job — return a job ID immediately with a 202, process async, and let the frontend poll for status. That's on my list."

**Don't defend it. Naming the weakness and the fix scores better.**

---

# 9. RETRIEVAL PIPELINE

### Q: Walk me through a retrieval query.

```
Query
 → Router (1.9s)                     3-way classification
 → agent_node (2.6s)                 decides to call a tool
 → tool_node (7.9s):
      generate 2 query variations    (RAG Fusion)
      embed 3 queries (dense + sparse)
      Qdrant hybrid search ×3        [RRF #1: dense + sparse]
      RRF across variations          [RRF #2]
      Cohere rerank → top 4
 → agent_node (1.0s)
 → relevancy_check (1.4s)
 → generate_answer (6.0s)

Total: 21.25s | 3.9K tokens | $0.0005
```

### Q: Why hybrid search?

> "Dense embeddings capture meaning but smooth away exact terms — a product code, a rare identifier, a specific number often won't match. BM25 catches exact keywords but has no notion of synonyms, so 'car' won't match 'automobile'. They fail in opposite directions, so running both and fusing covers both.
>
> I measured it on v1 with DeepEval: contextual recall went from 0.77 to 0.86, about 11.7% relative, and the share of responses scored 'excellent' went from 28% to 44%."

### Q: What's BM25?

> "A sparse keyword-ranking function. It scores documents by term frequency and inverse document frequency — how often a term appears in a document, weighted down by how common that term is across the corpus. It matches exact keywords very reliably but has no semantic understanding. I use Qdrant's BM25 model via fastembed, and it's essentially free — the sparse embedding takes near-zero time because it's computed locally."

### Q: What's RRF?

> "Reciprocal Rank Fusion. It combines ranked lists by scoring each document as the sum of `1/(k + rank)` across every list it appears in, with k typically 60. Documents that rank well in any list rise to the top. The value is that it works on ranks rather than scores, so you don't have to normalise scores across systems that produce them on different scales."

### Q: Why do you have two RRF steps?

> "They solve different problems. Qdrant's internal RRF fuses dense and sparse results within a single query. My RAG Fusion RRF fuses results across the three different query phrasings. Same algorithm, different axes."

### Q: What's RAG Fusion and why did you add it?

> "Before retrieval I expand the query into two additional phrasings via a cheap LLM call, retrieve for all three independently, and fuse with RRF.
>
> It targets a specific failure mode: a single embedding of a long or compound question gets diluted — it's trying to match too many concepts at once and matches nothing well, even when several chunks each answer part of it. Generating variations gives each sub-concept its own shot at matching.
>
> I verified it manually — a compound test question that returned zero retrieved documents on the single-query pipeline returned a fully cited answer once RAG Fusion was added."

### Q: Explain reranking.

> "Embedding search is a **bi-encoder** — the query and the document are encoded separately into vectors, then compared. Fast, but coarse, because neither encoding ever saw the other.
>
> A reranker is a **cross-encoder** — it feeds the query and document through the model together, so it can attend across both. Much more accurate, but far too slow to run over a whole corpus.
>
> So the pipeline is a funnel: 10 per query across 3 variations, fuse to 15 candidates with RRF, then rerank those 15 with Cohere and keep the top 4. Fast methods retrieve broadly, the expensive accurate method only sees a small pool."

### Q: What's the retrieval funnel?

```
10 per query × 3 queries  →  RRF  →  15 candidates  →  rerank  →  4 chunks to the LLM
```

### Q: Why rerank against the original query and not the variations?

> "The variations exist to widen recall. The original query is what the user actually asked, so it's the right thing to judge relevance against. Reranking against a variation would optimise for a question the user didn't ask."

---

# 10. YOUR CODE SPECIFICALLY

### Q: Which file should I open first?

> "`main.py` — it's the entry point. There are four files: `main.py` is the API layer, `rag_graph.py` has the LangGraph state machine where the routing lives, `vector_store.py` handles embeddings and retrieval, `loader.py` handles parsing and chunking. Let me follow a single chat request through all of them."

**Never do an endpoint-by-endpoint tour. Trace one request, 6–7 stops.**

| Stop | File | Say |
|---|---|---|
| 1 | `main.py` lifespan | "Checkpointer and graph compiled once at startup" |
| 2 | `main.py` `/api/chat` | "Builds initial state, `thread_id = session_id`, `graph.ainvoke`" |
| 3 | `rag_graph.py` router | "Structured output, 3-way, temperature 0" |
| 4 | `rag_graph.py` agent_node | "Tools unbound after 3 attempts so the loop must terminate" |
| 5 | `rag_graph.py` the tool | "`InjectedState` — model only sees `query`" |
| 6 | `vector_store.py` `search()` | "3 variations, parallel hybrid, RRF, rerank to 4" |
| 7 | `rag_graph.py` generate_answer | "Cited output, or honest 'not found'" |

### Q: What's the router routing between and how does it decide?

> "It's a 3-way classification: retrieve, verify_claim, or direct_answer. It's an LLM call with `with_structured_output` bound to a Pydantic model with a `Literal` of those three values — so I get a validated enum, not free text I have to parse. Temperature zero, because classification should be deterministic.
>
> The prompt has explicit keyword rules — 'report', 'paper', 'document', 'PDF', 'as per the report' always go to retrieve. And a tiebreaker: when in doubt between retrieve and direct_answer, always choose retrieve. That bias is deliberate — a wrong retrieve costs latency, a wrong direct_answer gives the user a hallucinated answer about their own document. I biased toward the cheaper failure mode.
>
> One thing worth noting — web search isn't a router route. It's one of two tools under the retrieve path. So it's a two-level decision: the router picks the workflow, the agent node picks the tool."

### Q: How did you validate the router?

> "I found a bug through testing — 'current event' questions like sports results and award winners were being classified as `direct_answer`, so they'd get answered from stale training data. I fixed it by broadening the router prompt with explicit examples of live-information questions. Pure prompt engineering, no code change."

### Q: Who actually calls the tool — the LLM?

> "No. The LLM doesn't execute anything — it emits a structured request with a tool name and arguments as JSON. `ToolNode` is the executor: it reads `tool_calls` off the last AIMessage, looks the function up in its registry, injects the state-bound arguments, and actually runs the Python. 'Function calling' is a misleading name — the model is *asking* for a call, not making one.
>
> The separation is deliberate: the model decides *what* to call, my code controls *how* it's called and what it can see."

### Q: 🌟 Explain `InjectedState`.

```python
class RetrieverInput(BaseModel):
    query: str          # ← the ONLY thing the LLM sees
```

| Param | Supplied by | Source |
|---|---|---|
| `query` | **LLM** | generated |
| `session_id` | LangGraph | `state["session_id"]` |
| `openai_api_key` | LangGraph | `state["openai_api_key"]` |
| `kimi_api_key` | LangGraph | `state["kimi_api_key"]` |
| `tool_call_id` | LangGraph | the AIMessage's tool call id |

> "The LLM only sees `query` — that's what's in the `args_schema`. `session_id` and the API keys are marked `InjectedState`, so LangGraph fills them from graph state at call time. The model never sees them and can't hallucinate a session ID to read another user's documents, or leak a key into a tool call. Its influence is limited to the query string.
>
> That's the general answer to how you stop an LLM doing something dangerous with a tool — you don't trust it, you constrain the schema."

### Q: 🌟 How do you control token cost?

> "Three things.
>
> First, retrieved chunks never enter the message history. The tool returns a one-line `ToolMessage` saying how many chunks it found, and the actual documents go into graph state via `Command(update=...)`. `generate_answer` reads them from state. If they went into messages, thousands of tokens of document text would be resent to the LLM on every subsequent turn.
>
> Second, the agent node trims messages to the current turn — it slices from the last HumanMessage. Without that, agent-node token cost grows linearly with conversation length.
>
> Third, model tiering. `gpt-4.1-mini` handles the agent node because tool-calling reliability matters most there. Routing, relevancy grading, query rewriting, verification and generation all run on Kimi's `moonshot-v1-32k`, which is cheaper and fine for classification and generation-shaped work."

### Q: Why two different LLM providers?

> "Cost tiering. Tool-calling reliability is the one place where a weaker model actually breaks the system — a malformed tool call derails the whole graph. So gpt-4.1-mini handles the agent node. Everything else is classification or generation, where Kimi's moonshot-v1-32k is cheaper and good enough. It also gives me a natural failover path if I build one."

### Q: Explain your custom reducer.

```python
def merge_docs(existing, new):
    if new == []:
        return []
    seen = {hash(d.page_content[:100]) for d in existing}
    deduped = [d for d in new if hash(d.page_content[:100]) not in seen]
    return existing + deduped
```

> "It appends and dedupes by content hash, because across a retry the same chunk can come back twice. The `new == []` case is an explicit reset — the router and `rewrite_query` both set `retrieved_docs: []` to clear the pool for a fresh attempt. Without that escape hatch, an append-only reducer could never be cleared, and the relevancy grader would end up judging a polluted mix of stale and fresh chunks."

### Q: Why do some state fields have no reducer?

> "Counters — `retrieval_attempts`, `rewrite_query_count`, `is_relevant`. They must overwrite, not accumulate. A reducer on a counter would be a bug."

### Q: What's the graph shape?

```
START → router
          ├── verify_claim ─────→ generate_answer → END
          ├── generate_answer ──→ END              (direct_answer)
          └── agent_node
                ├── tool_node → agent_node                   (cycle 1)
                └── relevancy_check
                      ├── generate_answer → END
                      └── rewrite_query → agent_node         (cycle 2)
```

> "Seven nodes, four conditional edges, four fixed edges. Two cycles, which is the reason for LangGraph over a chain. And a single exit point — everything converges on `generate_answer`, so answer formatting lives in exactly one place."

### Q: What's `tools_condition`?

> "A LangGraph prebuilt that checks whether the last message has `tool_calls` and returns `'tools'` or `'__end__'`. I wrap it rather than using it directly, because I need the non-tool branch to go to `relevancy_check`, not to END."

### Q: Why is the relevancy grader so lenient?

> "A strict grader triggers unnecessary rewrites, and each rewrite is a full extra retrieval cycle — an LLM call plus three retrievals plus a rerank. False negatives are expensive. False positives just mean slightly noisier context that the reranker has already filtered down to four chunks. So the prompt says explicitly: when in doubt, return true."

### Q: Why does `verify_claim` show 0.00s for generate_answer?

> "Because that branch doesn't make an LLM call. The verification LLM call already happened in the `verify_claim` node, and `generate_answer` just formats the verdict and the superseding papers into markdown. It's pure string building."

---

# 11. FALLBACKS & FAILURE HANDLING

### Q: 🌟 What happens when things fail?

**The 15-second summary — memorize this:**

> "The system has three classes of fallback: **loop bounds** that guarantee termination, **graceful degradation** where an enhancement failing doesn't break the core path, and **honest failure** where the user gets a clear message rather than a hallucinated answer. The one place I deliberately don't fall back is API keys — that fails loudly by design."

| # | Fallback | Mechanism |
|---|---|---|
| 1 | **Tool unbinding** | After 3 attempts, no tools bound → loop must terminate |
| 2 | **Rewrite budget** | Max 1 rewrite, then answer anyway |
| 3 | **RAG Fusion degrades** | Variation LLM fails → `variations = []` → single-query retrieval, no user-visible error |
| 4 | **Honest failure** | Not relevant → "I couldn't find..." + reason + suggestion |
| 5 | **Short-circuit grader** | No docs → skip the grader LLM call entirely |
| 6 | **Collection race** | Create fails → re-check `collection_exists`, only re-raise if genuinely missing |
| 7 | **OCR fallback** | Page < 20 chars → OCR; then fail-fast if total < 50 chars |
| 8 | **Metadata chains** | `title or video_title or source` |
| — | **Deliberate NON-fallback** | Missing API key → `MissingApiKeyError`, never falls back to the dev's key |

### Q: How do you stop an agent loop running forever?

> "Two independent bounds. After three tool attempts I hand the model an LLM with **no tools bound** — it's structurally incapable of emitting a tool call, so the cycle must terminate. That's stronger than prompting it to stop; a prompt can be ignored, a missing tool schema can't. Separately, the relevancy retry loop allows exactly one query rewrite, after which it proceeds to generate an answer regardless."

### Q: What if retrieval finds nothing relevant?

> "It returns an honest message: 'I couldn't find information in the retrieved documents that answers your question', plus the grader's reason and a suggestion to rephrase or upload more. It never falls through to the LLM's general knowledge — that would defeat the entire purpose of grounding."

### Q: What's the concurrency issue in collection creation?

```python
except Exception:
    if not qdrant_client.collection_exists(collection_name):
        raise
```

> "It's a check-then-act race. Two concurrent uploads to a new session can both pass the `collection_exists` check and both try to create it — one fails. I swallow that failure only if the collection now exists, meaning the other request won. If it genuinely doesn't exist, I re-raise."

### Q: Why no fallback on API keys?

> "That's deliberate. Users supply their own OpenAI and Kimi keys. Falling back to my own `.env` key would work, which is exactly why it's dangerous — a public deployment would silently drain my credits. Failing loudly is the correct behaviour. There's a `/api/keys/verify` endpoint that validates keys with `models.list()` before the user starts chatting."

### Q: What if OpenAI goes down?

**The gap. Volunteer it.**

> "The agent node fails and the request errors out. There's no provider failover and no retry with exponential backoff. I have both OpenAI and Kimi clients wired up, so a try/except that swaps providers is the obvious next step — but it isn't built yet."

### Q: What if Qdrant goes down?

> "Retrieval and ingestion fail. Direct answers and claim verification still work, since neither touches Qdrant. The retry safety valves bound the agent loop but don't handle infrastructure outages — that's a gap."

### Q: 🌟 Give me a concrete example of the fallbacks firing.

**Setup:** the paper says *"CO₂ sequestration rates"*. The user asks about *"carbon capture efficiency"*.

| # | Node | What happens | State after |
|---|---|---|---|
| 1 | `router` | "as per the report" → keyword rule | `route="retrieve"`, `docs=[]`, `attempts=0`, `rewrites=0` |
| 2 | `agent_node` | Tools bound (0 < 3) → tool call | `attempts=1` |
| 3 | `tool_node` | 3 variations → hybrid ×3 → RRF → rerank | `docs=[4]` + ToolMessage |
| 4 | `agent_node` | Has a ToolMessage → plain text, no tool | — |
| 5 | decision | `tools_condition` ≠ "tools" | → `relevancy_check` |
| 6 | `relevancy_check` | Chunks are about soil carbon, not efficiency | `is_relevant=False` |
| 7 | `check_relevancy` | `rewrites=0 < 1` ✅ | → `rewrite_query` |

**🔁 FALLBACK 2 fires**

| # | Node | What happens | State after |
|---|---|---|---|
| 8 | `rewrite_query` | → "CO2 sequestration rate report" | `docs=[]` ← **reducer reset**, `attempts=0`, `rewrites=1` |
| 9–12 | agent → tool → agent → grader | Retrieves again, still not a match | `is_relevant=False` |
| 13 | `check_relevancy` | `rewrites=1`, not `< 1` ❌ | → `generate_answer` |

**🛑 FALLBACK 2 exhausted → FALLBACK 4 fires**

```
I couldn't find information in the retrieved documents that answers your question.
Reason: The chunks discuss soil carbon storage, not capture efficiency metrics.
You can try rephrasing the question or uploading additional documents.
```

**Why step 8 matters:**
> "Without the `new == []` escape hatch in the reducer, the append-only merge would keep the stale chunks from attempt one and the grader would judge a polluted mix. The rewrite has to start clean. And the two counters bound different things — `retrieval_attempts` bounds the tool loop, `rewrite_query_count` bounds the retry loop. Resetting attempts on rewrite is safe precisely because the rewrite counter is the outer bound."

**The 20-second version:**
> "Say the user's wording doesn't match the document's. First retrieval comes back irrelevant, so the grader sends it to rewrite_query — that rephrases, wipes the stale pool via the reducer, and resets the tool budget. Second attempt still fails, but the rewrite counter is spent, so it goes to generate_answer, which returns an honest 'not found' with the reason. Three bounded fallbacks, and at no point does it loop forever or invent an answer."

---

# 12. PERFORMANCE & COST

### Q: What are your latency numbers?

| Route | Latency | Tokens | Cost |
|---|---|---|---|
| Direct Answer | 4.36s | 620 | — |
| Verify Claim | 8.01s | 1.3K | — |
| Web Search | 10.87s | 3.6K | $0.0005 |
| **Retrieve** | **21.25s** | **3.9K** | **$0.0005** |

### Q: 21 seconds is a long time. Where does it go?

> "Router about 2 seconds, first agent node 2.6, tool node about 8, second agent node 1, relevancy check 1.4, generation 6.
>
> The bottleneck isn't the retrieval work — it's **two full agent_node ↔ tool_node round trips**. Each costs a router-level LLM call just for the model to decide whether to call a tool again. That's pure overhead on top of the actual embedding and search work.
>
> Inside the tool node the three RAG Fusion queries already run concurrently via `asyncio.gather` with `to_thread`, since the Qdrant client is synchronous.
>
> Next steps: skip query expansion for short queries where dilution isn't a problem — that saves an LLM call plus two retrievals. Cache embeddings for repeated queries. And stream tokens so the user sees output at 2 seconds instead of 21. That doesn't reduce total latency but perceived latency is what matters in a chat interface."

### Q: Why is `generate_answer` 6 seconds?

> "It's generating a long cited answer over four reranked chunks — about 1.7K output tokens. That's genuine generation time, not overhead. Streaming is the right fix there, not optimisation."

### Q: Why is retrieval 21s but web search only 11s?

> "Retrieval runs RAG Fusion — a query-expansion LLM call, three embedding calls, three Qdrant searches, and a rerank. Web search hits Tavily once and skips expansion entirely."

### Q: How do you measure any of this?

> "LangSmith. Every node execution is traced with inputs, outputs, latency and token usage, and I have custom `@traceable` decorators on the embedding functions with a token counter using tiktoken so I can see cost per batch. That's how I caught the double-embedding bug — measured cost didn't match expected cost."

### Q: How much does a query cost?

> "About $0.0005 for the retrieve path, $0.00085 to ingest a 150-page PDF."

---

# 13. PERSISTENCE & SESSIONS

### Q: What's Supabase and what do you use it for?

> "Supabase is a Backend-as-a-Service built on PostgreSQL — managed Postgres plus auth, storage, realtime and auto-generated REST APIs.
>
> I use it for two things. LangGraph checkpointing: every graph execution writes state to Postgres via `AsyncPostgresSaver`, keyed by `thread_id`, which is my session ID. And session metadata — IDs, titles, timestamps behind the sidebar."

### Q: How does conversation history survive a restart?

> "The graph doesn't hold state in memory. LangGraph's checkpointer persists state to Postgres after each node, keyed by `thread_id`. If the server restarts mid-conversation, the next message picks up from the last checkpoint. My `/api/history` endpoint reads it back via `graph.aget_state(config)`."

### Q: How do you isolate sessions?

> "Each session gets its own Qdrant collection, named `papeer_<session_id>`. I isolate at the collection level rather than filtering by metadata in a shared collection — that gives hard isolation, because there's no query path that can accidentally return another session's documents. The tradeoff is more collections to manage, and deletion has to be a coordinated cleanup."

### Q: What happens when a user deletes a conversation?

> "Three things in sequence: `checkpointer.adelete_thread` removes the LangGraph state, `delete_collection` drops the Qdrant collection, and the Supabase row is deleted. It isn't transactional — if one fails midway you get orphans. A proper fix would be a soft-delete flag plus a reconciliation job."

### Q: 🐛 Tell me about the IPv6 bug.

> "Postgres connections worked locally but failed on Render. Supabase's direct connection string resolves to IPv6-only, and Render doesn't support outbound IPv6. Switching to Supabase's **Session Pooler** endpoint, which is IPv4-reachable, fixed it. It's the kind of thing that works perfectly on your machine and only breaks in your deployment environment."

### Q: What's a connection pooler?

> "Opening a Postgres connection is expensive — TCP handshake, auth, session setup. A pooler keeps a set of connections open and hands them out to requests, so you're not paying that cost every time. It also protects the database, since Postgres has a hard connection limit that a multi-worker backend can blow through fast."

### Q: Session pooler vs transaction pooler?

| Mode | Behaviour | Use when |
|---|---|---|
| **Session** | Connection held for the whole client session | Prepared statements, cross-query transactions — **what the checkpointer needs** |
| Transaction | Returned to the pool after each transaction | Serverless / high connection churn |

> "I use Session mode because LangGraph's Postgres checkpointer needs persistent session semantics."

---

# 14. DEPLOYMENT & SCALING

### Q: How is it deployed?

> "Backend on Render — `uvicorn main:app --host 0.0.0.0 --port $PORT`. `0.0.0.0` binds all interfaces so it's reachable externally, and no `--reload` in production. Frontend on Vercel with `VITE_API_BASE` pointing at the backend URL."

### Q: Why no `--reload` in production?

> "It watches the filesystem and restarts on changes — wasteful and unstable in production, and it runs a single worker. In production you want fixed workers and no file watching."

### Q: How did you handle CORS?

> "`allow_origins` for localhost:5173 in development, plus `allow_origin_regex` matching `papeer-frontend*.vercel.app`. The regex is there because Vercel generates a new preview URL per deployment — pinning a single origin would break every preview build."

### Q: How would you scale this?

> "The backend is stateless — all state lives in Postgres checkpoints and Qdrant — so it scales horizontally behind a load balancer. That part's straightforward.
>
> The real constraints are elsewhere. Ingestion runs inside the HTTP request and takes ~40 seconds for a large PDF, which risks gateway timeouts; that should be a queue with background workers returning a job ID. External API rate limits on OpenAI and Cohere would bite before my own compute does. And collection-per-session is great for hard isolation but becomes a management problem at thousands of collections — at scale I'd move to a shared collection with metadata filtering and accept the softer isolation."

### Q: How do you handle concurrent uploads to the same session?

> "Qdrant upserts are idempotent by point ID, so concurrent writes to the same collection are safe. What isn't guarded is a user deleting a session mid-ingestion — that would need a lock or a status flag on the session row."

### Q: How would you add caching?

> "Two layers. Query embeddings are deterministic for a given text, so an embedding cache keyed on the query string would cut the dense embedding call on repeat queries. And a full response cache keyed on (session_id, normalised query) would short-circuit identical repeated questions entirely. Neither is built."

---

# 15. SECURITY & WEAKNESSES

### Q: How is it secured?

**Be honest.**

> "It isn't, properly. There's no auth yet — sessions are isolated but not authenticated, so anyone with a session ID can access that session. Production would need auth on every endpoint plus an ownership check on session IDs. That's the single biggest gap and it's top of my roadmap.
>
> What I did do: users supply their own API keys rather than using mine, and tool arguments like `session_id` are injected from server-side state rather than generated by the LLM, so the model can't reach into another session."

### Q: What's the risk of an LLM calling tools?

> "That the model supplies arguments it shouldn't control. I constrain it at the schema level — the `args_schema` only exposes `query`. Everything else is `InjectedState`, filled from server-side state. The model can't specify a different session ID or a different API key, because those parameters don't exist in the schema it sees."

### Q: 🌟 What are the weaknesses of this project?

**Volunteer these. Naming your own gaps beats being caught.**

1. **No auth** — session IDs aren't protected
2. **Upload blocks ~40s** — should be a background job with a job ID
3. **Three sync nodes block the event loop** — `agent_node`, `relevancy_check`, `rewrite_query` are plain `def` calling `.invoke()`. Same class of bug I already fixed on the Qdrant side
4. **No v2 eval pipeline** — DeepEval numbers are v1; RAG Fusion validated manually only
5. **Delete isn't transactional** — checkpoint, collection and row can orphan
6. **No provider failover or retry/backoff**
7. **`hash()` is per-process salted** — fine for in-request dedup, not stable across restarts
8. **Tavily key is app-owned** while OpenAI and Kimi are BYOK — inconsistent with my own stated principle

### Q: What would you do differently if you rebuilt it?

> "Auth from day one. Ingestion as a background job rather than a blocking request. And I'd have written the async layer properly the first time instead of migrating it piecemeal — I fixed the Qdrant side with `to_thread` but three graph nodes still block the event loop."

---

# 16. EVALUATION

### Q: How do you know retrieval is actually good?

> "I ran DeepEval on the v1 prototype comparing dense-only retrieval against hybrid search, both with the reranker."

| Metric | Dense + Rerank | Hybrid + Rerank |
|---|---|---|
| Contextual Precision | 0.83 | **0.84** |
| **Contextual Recall** | 0.77 | **0.86** |
| Contextual Relevancy | 0.82 | **0.84** |
| Answer Relevancy | 0.83 | **0.88** |
| Faithfulness | 0.84 | **0.87** |

> "Recall improved 11.7% relative, and 'excellent' scoring responses went from 28% to 44%. That's what justified carrying hybrid search into v2.
>
> To be straight with you — that's v1. The v2 eval pipeline isn't rebuilt yet, so RAG Fusion has only been validated manually. Rebuilding it is on the roadmap."

### Q: What do those metrics mean?

- **Contextual Recall** — did retrieval find everything needed to answer
- **Contextual Precision** — are the relevant chunks ranked highly
- **Contextual Relevancy** — what proportion of retrieved context is actually relevant
- **Answer Relevancy** — does the answer address the question
- **Faithfulness** — is the answer supported by the retrieved context (hallucination check)

### Q: How would you evaluate RAG Fusion properly?

> "Build a golden set of question-and-expected-source pairs from a known document, then measure contextual recall with and without fusion. The manual validation I did — a compound question that returned zero documents single-query and a fully cited answer with fusion — is a strong signal but it's one data point, not a measurement."

---

# 17. BEHAVIORAL & DESIGN JUDGMENT

### Q: What was the hardest part?

> "Retrieval quality. Getting a RAG demo working is easy; getting it to reliably find the right chunk is not. I went through three iterations — dense-only, then hybrid with BM25 after DeepEval showed dense was missing exact terms, then RAG Fusion after I found compound questions returning nothing at all because the single embedding was too diluted."

### Q: What are you most proud of?

> "Two things. The instrumentation — LangSmith tracing with per-batch token counting is what caught a bug that halved my ingestion cost, and I'd never have found it by reading code. And the way tool arguments are handled: the model only sees the query, everything else is injected from server-side state, so the LLM's blast radius is deliberately tiny."

### Q: Tell me about a bug you found and fixed.

**Use the embedding one — it's the best story.** See §8.

### Q: What did you learn?

> "That measurement changes decisions. I assumed embedding was the expensive part of ingestion — the traces showed parsing takes 25 seconds and embedding takes 12. I assumed retrieval latency was dominated by the vector search — it's actually the agent round trips. Almost every optimisation I'd have guessed at would have been aimed at the wrong thing."

### Q: Why rebuild from Streamlit to FastAPI + React?

> "Streamlit is fine for validating an idea but it isn't a production architecture — no real API layer, no proper session management, and very limited frontend control. FastAPI plus React gives me a proper client-server split, real routing, persistent multi-session state, and a UI I can actually design."

### Q: What's on the roadmap?

> "Auth and multi-user support. Background-job ingestion. Rebuilt evaluation pipeline. Working token streaming — I attempted it and reverted it. Citation highlighting in the UI. And provider failover."

---

# 18. CURVEBALLS

### Q: Why not just use a bigger context window and skip RAG?

> "Three reasons. Cost — you'd pay for the entire document on every single turn. Latency — long contexts are measurably slower. And accuracy — models degrade at retrieving specific facts from the middle of very long contexts, the 'lost in the middle' effect. RAG also gives you citations, which a stuffed context doesn't."

### Q: How do you prevent prompt injection from an uploaded document?

**Honest answer:**

> "I don't, currently. A document could contain text like 'ignore previous instructions'. My generation prompt has strict grounding rules — use only the context, cite everything — which helps somewhat, but it isn't a defence. Proper mitigations would be delimiting untrusted content clearly, instructing the model that context is data not instructions, and output validation. It's a real gap."

### Q: What if two chunks contradict each other?

> "The generation prompt handles this explicitly: when multiple sources give different values for the same fact, give one clear primary answer using the most authoritative source — I use the example of a government meteorological source over a generic weather site — then briefly note that other sources report slightly different values, without listing every number."

### Q: How would you support 10,000 users?

> "The API layer scales horizontally since it's stateless. The changes I'd need: ingestion moves to a queue with workers; collection-per-session becomes metadata filtering in shared collections; add caching on embeddings and repeated queries; add rate limiting per user; and I'd need to handle provider rate limits with backoff and failover. Auth becomes mandatory rather than a roadmap item."

### Q: How would you add streaming?

> "`generate_answer` already has the streaming version written and commented out — `astream` accumulating chunks. The endpoint would return a `StreamingResponse` with `text/event-stream` and the frontend would consume server-sent events. The reason I reverted it was that LangGraph's streaming interacts awkwardly with the checkpointer and the node-level structure, so I shipped the working non-streaming version rather than a broken streaming one."

### Q: If you had one week, what would you do?

> "Auth, background ingestion, and the eval pipeline — in that order. Auth because it's the blocker for anything real. Background ingestion because 40-second requests will eventually time out. The eval pipeline because right now I can't prove that changes to retrieval actually help — I'm making quality decisions on intuition."

### Q: Why should we hire you based on this?

> "Because I didn't just get it working — I measured it, found two real bugs from that data, and I can tell you exactly what's still broken. The cost bug only surfaced because I had per-call tracing and was checking measured against expected. And I can name my gaps precisely: no auth, blocking ingestion, three nodes still blocking the event loop. I'd rather be the person who knows where the bodies are buried than the one who thinks the system is perfect."

---

# 19. CHEAT CARD — LAST 10 MINUTES

### Fix these before you walk in

- Router has **3 routes**, not 4 — web_search is a **tool**, not a route
- Retrievals are **already parallelized** (`asyncio.gather` + `to_thread`)
- The app is **BYOK** — users supply their own keys, no fallback to yours
- There's a **dead edge** in the graph diagram (`agent_node → generate_answer` is unreachable)

### Numbers

| | |
|---|---|
| Chunks | 1500 size / 400 overlap |
| Embedding | 1536 dims, `text-embedding-3-small` |
| Funnel | 10 per query × 3 → RRF 15 → rerank → **4** |
| RRF k | 60 |
| Bounds | 3 tool attempts, 1 rewrite |
| Retrieve path | 21.25s, 3.9K tokens, $0.0005 |
| Ingestion | 150 pages → 301 chunks → ~40s → $0.00085 |
| Eval | recall 0.77 → 0.86 (+11.7% rel) |

### Your three best talking points

1. **`InjectedState` + `Command(update=...)`** — model only sees `query`; chunks bypass message history
2. **Tool unbinding after 3 attempts** — a hard structural loop bound, not a prompt
3. **The double-embedding bug** — found via LangSmith, halved ingestion cost

### Three bug stories, one skill each

| Bug | Shows |
|---|---|
| Double embedding | Instrumentation → anomaly → root cause → measured fix |
| IPv6 / Session Pooler | Deployment & networking debugging |
| Router misclassification | Prompt engineering, no code change |

### Volunteer these

No auth · 40s blocking upload · three sync nodes blocking the loop · no v2 eval pipeline

### The fallback sentence

> "Three classes of fallback: loop bounds that guarantee termination, graceful degradation where an enhancement failing doesn't break the core path, and honest failure where the user gets a clear message rather than a hallucinated answer. The one place I deliberately don't fall back is API keys — that fails loudly by design."

### The opening

> "Papeer is a full-stack RAG application where users upload documents or paste URLs and have a conversational chat with that content. I built it because LLMs can't access your private or recent documents, and existing tools use basic RAG that retrieves on every query — slow and often inaccurate. So I built a LangGraph router that classifies each query and sends it to the right workflow — retrieve, verify a claim, or answer directly. And for retrieval I use hybrid search with reranking instead of plain vector search, which measurably improved recall. Want me to go deeper on the architecture, or into the code?"
