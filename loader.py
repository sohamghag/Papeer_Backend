from pathlib import Path
import fitz  # PyMuPDF
from langchain_community.document_loaders import TextLoader, WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable
from rapidocr_onnxruntime import RapidOCR

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 400

# Below this many extracted characters, a page is treated as image-only/scanned and sent to OCR.
MIN_PAGE_TEXT_CHARS = 20
# Below this many total characters, the whole loaded document is rejected as unusable.
MIN_TOTAL_CONTENT_CHARS = 50


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

# Lazily created — RapidOCR loads its ONNX models on first use.
_ocr_engine = None


def _get_ocr_engine() -> RapidOCR:
    global _ocr_engine
    if _ocr_engine is None:
        _ocr_engine = RapidOCR()
    return _ocr_engine


# Adding Title in the Documents Metadata
def _stamp_title(documents: list[Document], title: str) -> list[Document]:
    for doc in documents:
        doc.metadata["title"] = title
    return documents


# Rejects documents where nothing usable was actually extracted (scanned file OCR
# still failed, blocked webpage, empty file, etc.) instead of silently embedding junk.
def _check_min_content(documents: list[Document], source: str) -> None:
    total_chars = sum(len(d.page_content.strip()) for d in documents)
    if total_chars < MIN_TOTAL_CONTENT_CHARS:
        raise ValueError(
            f"No usable text could be extracted from '{source}'. It may be a "
            "scanned/image-only file, empty, or blocked from automated access."
        )


def _ocr_page(page: "fitz.Page") -> str:
    """Fallback for pages with no extractable text layer — render to an image and OCR it."""
    pix = page.get_pixmap(dpi=200)
    result, _ = _get_ocr_engine()(pix.tobytes("png"))
    if not result:
        return ""
    return "\n".join(line[1] for line in result)


def _extract_page(page: "fitz.Page") -> tuple[list[str], str]:
    """Splits one PDF page into (table markdown blocks, remaining prose text).

    Tables are pulled out and rendered as standalone markdown so a later
    character-based split never cuts a table row in half. Prose text
    excludes the table regions so content isn't duplicated.
    """
    table_finder = page.find_tables()
    table_bboxes = []
    tables_md = []
    for table in table_finder.tables:
        table_bboxes.append(fitz.Rect(table.bbox))
        tables_md.append(table.to_markdown())

    text_parts = []
    for block in page.get_text("blocks"):
        block_rect = fitz.Rect(block[:4])
        block_text = block[4]
        if any(block_rect.intersects(bbox) for bbox in table_bboxes):
            continue
        text_parts.append(block_text)
    remaining_text = "".join(text_parts).strip()

    if len(remaining_text) < MIN_PAGE_TEXT_CHARS and not tables_md:
        remaining_text = _ocr_page(page)

    return tables_md, remaining_text


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
    documents = _stamp_title(documents, title)
    _check_min_content(documents, url)
    return documents


# Loading and Splitting PDF File — table-aware, with OCR fallback for scanned pages
def load_pdf(file_path: str) -> list[Document]:
    title = Path(file_path).stem

    table_documents = []
    text_pages = []
    with fitz.open(file_path) as pdf:
        for page_num, page in enumerate(pdf):
            tables_md, page_text = _extract_page(page)

            for table_md in tables_md:
                table_documents.append(Document(
                    page_content=table_md,
                    metadata={"page": page_num, "source_type": "table"},
                ))

            if page_text:
                text_pages.append(Document(
                    page_content=page_text,
                    metadata={"page": page_num},
                ))

    text_documents = _splitter.split_documents(text_pages)
    documents = _stamp_title(text_documents + table_documents, title)
    _check_min_content(documents, file_path)
    return documents


# Loading and Splitting Text File
def load_text(file_path: str) -> list[Document]:
    docs = TextLoader(file_path, encoding="utf-8").load()
    documents = _splitter.split_documents(docs)
    documents = _stamp_title(documents, Path(file_path).stem)
    _check_min_content(documents, file_path)
    return documents

# Loading and Splitting Markdown File
def load_markdown(file_path: str) -> list[Document]:
    docs = TextLoader(file_path, encoding="utf-8").load()
    documents = _md_splitter.split_documents(docs)
    documents = _stamp_title(documents, Path(file_path).stem)
    _check_min_content(documents, file_path)
    return documents


@traceable(name="load_and_split_document")
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
