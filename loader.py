from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 400



# Splitter for Text Documents
_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    add_start_index=True,
    separators=[
        "\n\n",   #(split at paragraph breaks — most natural)
        "\n•",
        "\n-",
        "\n",
        ". ",
        " ",
        "",
    ]
)
# Splitter for Markdown File
_md_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    add_start_index=True,
    separators=[
        "\n\n",
        "## ",
        "### ",
        "#### ",
        "\n•",
        "\n-",
        "\n",
        ". ",
        " ",
        "",
    ]
)


# Adding Title in the Documents Metadata
def _stamp_title(documents: list[Document], title: str) -> list[Document]:
    for doc in documents:
        doc.metadata["title"] = title
    return documents


# Loading and Splitting WebPage
def load_webpage(url: str) -> list[Document]:
    docs = WebBaseLoader(
        web_paths=[url],
        requests_kwargs={"timeout": 30}
    ).load()

    if not docs:
        raise ValueError(f"No content extracted from {url}")

    title = docs[0].metadata.get("title") or url
    documents = _splitter.split_documents(docs)
    return _stamp_title(documents, title)


# Loading and Splitting PDF File
def load_pdf(file_path: str) -> list[Document]:
    docs = PyMuPDFLoader(file_path).load()
    documents = _splitter.split_documents(docs)
    return _stamp_title(documents, Path(file_path).stem) #filename WITHOUT extension


# Loading and Splitting Text File
def load_text(file_path: str) -> list[Document]:
    docs = TextLoader(file_path, encoding="utf-8").load()
    documents = _splitter.split_documents(docs)
    return _stamp_title(documents, Path(file_path).stem)

# Loading and Splitting Markdown File
def load_markdown(file_path: str) -> list[Document]:
    docs = TextLoader(file_path, encoding="utf-8").load()
    documents = _md_splitter.split_documents(docs)
    return _stamp_title(documents, Path(file_path).stem)


def load_document(source: str) -> list[Document]:
    """Dispatch to the appropriate loader based on URL prefix or file extension."""
    try:
        if source.startswith(("http://", "https://")):
            return load_webpage(source)
        ext = Path(source).suffix.lower()
        if ext == ".pdf":
            return load_pdf(source)
        if ext == ".txt":
            return load_text(source)
        if ext in (".md", ".markdown"):
            return load_markdown(source)
        raise ValueError(f"Unsupported file type: {ext}")
    except Exception as e:
        print(f"[load_document] ERROR: {type(e).__name__}: {e}")
        raise

    
         