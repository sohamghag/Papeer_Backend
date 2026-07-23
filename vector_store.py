import os
from dotenv import load_dotenv
from pathlib import Path


# Langchain
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langchain_openai import ChatOpenAI
from langchain_cohere import CohereRerank
from langsmith import traceable
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel
# Qdrant Vector Database
from qdrant_client import QdrantClient,models
from qdrant_client.http.models import Distance, VectorParams, SparseVectorParams, SparseIndexParams, Modifier
from fastembed import SparseTextEmbedding
from langchain_qdrant import QdrantVectorStore, RetrievalMode,SparseEmbeddings
from qdrant_client.models import SparseVector
import tiktoken

# Constants
EMBED_DIM=1536
RRF_K = 60  # standard RRF constant
FUSION_CANDIDATE_LIMIT = 15  # how many fused candidates go into the reranker
load_dotenv()
api_key=os.getenv("QDRANT_API_KEY")
api_key_co=os.getenv("CO_API_KEY")
url=os.getenv("QDRANT_URL")

 # cohere re-ranker
compressor = CohereRerank(model="rerank-v4.0-pro",top_n=4)

class MissingApiKeyError(Exception):
    """Raised when a request reaches an LLM/embedding call with no
    user-supplied key. No fallback to the developer's own .env key — the
    request must fail instead of silently consuming the app owner's credits."""

class QueryVariations(BaseModel):
    variations: list[str]

def get_query_variation_llm(kimi_api_key: str | None = None) -> ChatOpenAI:
    """Cheap model dedicated to generating RAG-Fusion query variations — same
    cost-tiering logic as rag_graph.py's kimi_llm (classification/generation-shaped
    task, doesn't need the stronger tool-calling model). Requires the user's own key."""
    if not kimi_api_key:
        raise MissingApiKeyError("A Kimi API key is required.")
    return ChatOpenAI(
        api_key=kimi_api_key,
        base_url="https://api.moonshot.ai/v1",
        model="moonshot-v1-32k",
    )


# Creating Qdrant Client
qdrant_client = QdrantClient(
    api_key=api_key,
    url=url,
    timeout=120   # wait max 120 seconds before failing
)
# With timeout=120 — after 120 seconds, give up and throw an error. Better to fail fast than hang forever.

# This for BM25Embedder
# text = ["India is beautiful","My name is chatbot"]
# list(self._model.embed(texts)) #Embed(texts) --> returns a generator and when we do list(genertor) --> we get a list []
# [
#     SparseEmbedding(indices=[4821, 5521, ...], values=[3.60, 2.80, ...]),  # for chunk 1  "India is beautiful"
#     SparseEmbedding(indices=[1234, 5678, ...], values=[2.10, 1.90, ...]),  # for chunk 2   "My name is chatbot"
# ]

# [
    #  SparseVector(indices=v.indices.tolist() [because indices is numpy array], values=v.values.tolist())

# ]

# Embedding Model --> Sparse Vector
class BM25Embedder(SparseEmbeddings):
    def __init__(self):
        self._model = SparseTextEmbedding(
            model_name="Qdrant/bm25"
        )

    @traceable(name="sparse_bm25_document_embeddings")
    def embed_documents(
        self,
        texts: list[str]
    ) -> list[SparseVector]:

        return [
            SparseVector(
                indices=v.indices.tolist(),
                values=v.values.tolist()
            )
            for v in self._model.embed(texts)
        ]

    @traceable(name="sparse_bm25_query_embedding")
    def embed_query(self, text: str) -> SparseVector:
        v = list(self._model.embed(text))[0]

        return SparseVector(
            indices=v.indices.tolist(),
            values=v.values.tolist()
        )

sparse_embedder = BM25Embedder()

EMBEDDING_PRICE_PER_1M_TOKENS = 0.02
# Embedding Model -> Dense Vector
class TracedOpenAIEmbeddings(Embeddings):
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.encoding = tiktoken.get_encoding("cl100k_base")

    @traceable(
        name="dense_openai_embeddings",
        run_type="embedding"
    )
    def embed_documents(self, texts):
        token_count = sum(
            len(self.encoding.encode(text))
            for text in texts
        )

        cost_usd = (
            token_count / 1_000_000
        ) * EMBEDDING_PRICE_PER_1M_TOKENS

        print(f"Batch size: {len(texts)}")
        print(f"Tokens: {token_count}")
        print(f"Estimated cost: ${cost_usd:.8f}")

        return self.embeddings.embed_documents(texts)

    @traceable(
        name="dense_openai_query_embedding",
        run_type="embedding"
    )
    def embed_query(self, text):
        return self.embeddings.embed_query(text)

def get_embeddings(openai_api_key: str | None = None) -> TracedOpenAIEmbeddings:
    """Builds the dense embedding client for this call — requires the user's own key."""
    if not openai_api_key:
        raise MissingApiKeyError("An OpenAI API key is required.")
    base_embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=openai_api_key,
    )
    return TracedOpenAIEmbeddings(base_embeddings)


# Collection Name which is stored in the Qdrant Cluster
def get_collection_name(session_id: str) -> str:
    return f"papeer_{session_id.replace('-', '_')}"


# creating the collection if not exist and returning the collection if already exist
def get_vectorstore(session_id: str, openai_api_key: str | None = None) -> QdrantVectorStore:
    collection_name = get_collection_name(session_id)
    print("Collection Name",collection_name)
    print("Document Exist in the Database",qdrant_client.collection_exists(collection_name))

    if not qdrant_client.collection_exists(collection_name):
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": VectorParams(size=EMBED_DIM, distance=Distance.COSINE)
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(modifier=Modifier.IDF)
            }
        )
    return QdrantVectorStore(
        client=qdrant_client,
        collection_name=collection_name,
        embedding=get_embeddings(openai_api_key),
        vector_name="dense",          # tell it to use the named dense vector
        sparse_embedding=sparse_embedder,  # attach sparse embedder
        sparse_vector_name="sparse",
        retrieval_mode=RetrievalMode.HYBRID,  # key change
    )


@traceable(name="embed_and_store_documents")
def add_paper(docs: list[Document], session_id: str, openai_api_key: str | None = None) -> None:
    try:
        vectorstore = get_vectorstore(session_id, openai_api_key)
        vectorstore.add_documents(docs)

    except Exception as e:
        print(f"[add_paper] ERROR: {type(e).__name__}: {e}")
        raise


@traceable(name="generate_query_variations")
def _generate_query_variations(query: str, kimi_api_key: str | None = None) -> list[str]:
    """RAG Fusion — generate 2 alternative phrasings of the query so retrieval
    isn't limited to whichever chunks happen to match the original wording."""
    prompt = (
        "You are a query rewriting assistant for a document retrieval system.\n"
        "Generate exactly 2 alternative phrasings of the given search query.\n\n"
        "STRICT RULES:\n"
        "- Keep the same meaning and intent as the original query.\n"
        "- Only change terminology/wording, not the topic.\n"
        "- Do NOT broaden, generalize, or add new concepts.\n"
        "- Return exactly 2 variations."
    )
    variation_llm = get_query_variation_llm(kimi_api_key).with_structured_output(QueryVariations)
    response = variation_llm.invoke([
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"Original query: {query}"},
    ])
    return response.variations[:2]


def _rrf_fuse(ranked_lists: list[list[Document]]) -> list[Document]:
    """Reciprocal Rank Fusion across multiple independently-retrieved ranked
    lists (one per query variation). Distinct from Qdrant's own dense+sparse
    RRF, which only fuses within a single query's search."""
    scores: dict[int, float] = {}
    doc_by_key: dict[int, Document] = {}
    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list):
            key = hash(doc.page_content[:100])
            scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
            doc_by_key.setdefault(key, doc)

    fused_keys = sorted(scores, key=lambda k: scores[k], reverse=True)
    return [doc_by_key[key] for key in fused_keys[:FUSION_CANDIDATE_LIMIT]]


# vector search + re-ranker — RAG Fusion: multi-query retrieval, RRF-fused, then reranked
def search(query:str,session_id:str, openai_api_key: str | None = None, kimi_api_key: str | None = None) -> list[Document]:
    try:
        vectorstore = get_vectorstore(session_id, openai_api_key)
        base_retriever=vectorstore.as_retriever(search_kwargs={"k":10})

        try:
            variations = _generate_query_variations(query, kimi_api_key)
        except Exception as e:
            print(f"[search] query variation generation failed, falling back to single query: {type(e).__name__}: {e}")
            variations = []

        all_queries = [query] + variations
        # each call: dense search + sparse search → Qdrant's internal RRF → top 10 chunks
        ranked_lists = [base_retriever.invoke(q) for q in all_queries]

        print("Ranked_List",len(ranked_lists))

        fused_candidates = _rrf_fuse(ranked_lists)

        print("Fused_candidates",len(fused_candidates))

        # rerank the fused pool using the ORIGINAL query, not the variations
        return list(compressor.compress_documents(fused_candidates, query=query))
    except Exception as e:
        print(f"[search] ERROR: {type(e).__name__}: {e}")
        raise  # re-raise so the caller also sees it


# used for return the documents name form the collection 
def list_papers(session_id: str) -> list[str]:
    collection_name = get_collection_name(session_id)
    if not qdrant_client.collection_exists(collection_name):
        return []

    seen = set()
    titles = []
    offset = None

    while True:

        points, offset = qdrant_client.scroll(
            collection_name=collection_name,
            with_payload=True,
            limit=100,
            offset=offset,
        )

        for point in points:
            metadata = (point.payload or {}).get("metadata", {})
            title = (
                metadata.get("title")
                or metadata.get("video_title")
                or metadata.get("source")
            )

            if title and title not in seen:
                seen.add(title)
                titles.append(title)

        if offset is None:
            break

    return titles


def delete_collection(session_id: str) -> None:
    """Deletes the entire Qdrant collection for a session, if it exists."""
    collection_name = get_collection_name(session_id)
    try:
        if qdrant_client.collection_exists(collection_name):
            qdrant_client.delete_collection(collection_name)
            print(f"[delete_collection] Deleted collection: {collection_name}")
    except Exception as e:
        print(f"[delete_collection] ERROR: {type(e).__name__}: {e}")
        raise