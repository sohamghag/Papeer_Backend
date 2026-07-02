import os
from dotenv import load_dotenv
from pathlib import Path


# Langchain
from langchain_openai import OpenAIEmbeddings, ChatOpenAI 
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_openai import ChatOpenAI
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank


# Qdrant Vector Database
from qdrant_client import QdrantClient,models
from qdrant_client.http.models import Distance, VectorParams, SparseVectorParams, SparseIndexParams, Modifier
from fastembed import SparseTextEmbedding
from langchain_qdrant import QdrantVectorStore, RetrievalMode,SparseEmbeddings
from qdrant_client.models import SparseVector


# Constants
EMBED_DIM=1536
load_dotenv()
api_key=os.getenv("QDRANT_API_KEY")
api_keys_kimi=os.getenv("KIMI_API_KEY")
api_key_co=os.getenv("CO_API_KEY")
url=os.getenv("QDRANT_URL")

# Embedding Model -> Dense Vector
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")



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
        self._model = SparseTextEmbedding(model_name="Qdrant/bm25")
    
    def embed_documents(self,texts:list[str]) ->  list[SparseVector]:
          return [
            SparseVector(indices=v.indices.tolist(), values=v.values.tolist())
            for v in list(self._model.embed(texts))
        ]
    
    def embed_query(self,text:str) -> SparseVector:
        v=list(self._model.embed(text))[0]
        return SparseVector(indices=v.indices.tolist(), values=v.values.tolist())

sparse_embedder = BM25Embedder()


# Collection Name which is stored in the Qdrant Cluster
def get_collection_name(session_id: str) -> str:
    return f"papeer_{session_id.replace('-', '_')}"


# creating the collection if not exist and returning the collection if already exist
def get_vectorstore(session_id: str) -> QdrantVectorStore:
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
        embedding=embeddings,
        vector_name="dense",          # tell it to use the named dense vector
        sparse_embedding=sparse_embedder,  # attach sparse embedder
        sparse_vector_name="sparse",
        retrieval_mode=RetrievalMode.HYBRID,  # key change
    )


def add_paper(docs: list[Document], session_id: str) -> None:
    try:
        vectorstore = get_vectorstore(session_id)
        vectorstore.add_documents(docs)
    except Exception as e:
        print(f"[add_paper] ERROR: {type(e).__name__}: {e}")
        raise  # re-raise so the caller also sees it

# vector search + re-ranker
def search(query:str,session_id:str) -> list[Document]:
    try:
        # query → dense search + sparse search → RRF merge → top 10 chunks (Quadrant does this automatically)
        vectorstore = get_vectorstore(session_id)
        base_retriever=vectorstore.as_retriever(search_kwargs={"k":10}) 
        # cohere re-ranker 
        compressor = CohereRerank(model="rerank-v4.0-pro",top_n=4)
        compression_retriever = ContextualCompressionRetriever(
            base_retriever=base_retriever,
            base_compressor=compressor
        )
        return compression_retriever.invoke(query)
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