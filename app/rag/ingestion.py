"""
OmniRAG — Ingestion Pipeline
────────────────────────────
Processes PDFs/docs through Docling → chunk → embed (dense + sparse)
→ upsert into Qdrant hybrid collection.

Supports: PDF, DOCX, TXT, Markdown, Web URLs
"""

import hashlib
import logging
import os
from pathlib import Path
from typing import List, Optional

from docling.document_converter import DocumentConverter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, SparseVectorParams
from fastembed import SparseTextEmbedding

from app.config import settings

logger = logging.getLogger(__name__)


# ── Embeddings ────────────────────────────────────────────────────────────────

def get_dense_embeddings():
    """Gemini dense embeddings (768-dim)."""
    return GoogleGenerativeAIEmbeddings(
        model=settings.gemini_embedding_model,
        google_api_key=settings.google_api_key,
    )


def get_sparse_embeddings():
    """BM25 sparse embeddings via FastEmbed."""
    return SparseTextEmbedding(model_name="Qdrant/bm25")


# ── Qdrant Client & Collection ────────────────────────────────────────────────

def get_qdrant_client() -> QdrantClient:
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        timeout=60,
    )


def ensure_collection(client: QdrantClient, collection_name: str):
    """Create hybrid collection if it doesn't exist."""
    existing = [c.name for c in client.get_collections().collections]
    if collection_name not in existing:
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": VectorParams(size=768, distance=Distance.COSINE)
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams()
            },
        )
        logger.info("Created Qdrant collection: %s", collection_name)
    else:
        logger.info("Collection already exists: %s", collection_name)


def get_vector_store() -> QdrantVectorStore:
    client = get_qdrant_client()
    ensure_collection(client, settings.qdrant_collection)
    return QdrantVectorStore(
        client=client,
        collection_name=settings.qdrant_collection,
        embedding=get_dense_embeddings(),
        sparse_embedding=get_sparse_embeddings(),
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name="dense",
        sparse_vector_name="sparse",
    )


# ── Document Extraction ───────────────────────────────────────────────────────

def extract_with_docling(file_path: str) -> List[Document]:
    """
    Use Docling to extract rich structured content from PDF/DOCX/etc.
    Returns LangChain Documents with metadata.
    """
    converter = DocumentConverter()
    result = converter.convert(file_path)
    doc = result.document

    documents = []
    file_name = Path(file_path).stem
    file_hash = _file_hash(file_path)

    # ── 1. Text chunks ────────────────────────────────────────────
    full_md = doc.export_to_markdown()
    pages = full_md.split("<!-- page break -->")
    for i, page_text in enumerate(pages, start=1):
        text = page_text.strip()
        if len(text) < 50:
            continue
        documents.append(Document(
            page_content=text,
            metadata={
                "source": file_name,
                "file_hash": file_hash,
                "page": i,
                "content_type": "text",
            }
        ))

    # ── 2. Tables ────────────────────────────────────────────────
    for table in doc.tables:
        table_md = table.export_to_markdown()
        if table_md.strip():
            documents.append(Document(
                page_content=table_md,
                metadata={
                    "source": file_name,
                    "file_hash": file_hash,
                    "page": getattr(table, "page_no", 0),
                    "content_type": "table",
                }
            ))

    logger.info("Extracted %d documents from %s", len(documents), file_name)
    return documents


def extract_from_text(text: str, source: str = "manual") -> List[Document]:
    """Wrap raw text as a document for ingestion."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = splitter.create_documents([text], metadatas=[{
        "source": source,
        "content_type": "text",
    }])
    return chunks


# ── Ingestion ─────────────────────────────────────────────────────────────────

def ingest_documents(documents: List[Document]) -> int:
    """
    Deduplicate, chunk (if needed), and upsert into Qdrant.
    Returns count of vectors added.
    """
    if not documents:
        logger.warning("No documents to ingest.")
        return 0

    # Deduplicate by hash
    seen_hashes = set()
    unique_docs = []
    for doc in documents:
        h = hashlib.md5(doc.page_content.encode()).hexdigest()
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_docs.append(doc)

    logger.info("Ingesting %d unique documents (deduped from %d).",
                len(unique_docs), len(documents))

    # Chunk any large text blocks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunked = []
    for doc in unique_docs:
        if doc.metadata.get("content_type") == "text" and len(doc.page_content) > settings.chunk_size:
            sub_chunks = splitter.split_documents([doc])
            chunked.extend(sub_chunks)
        else:
            chunked.append(doc)

    vs = get_vector_store()
    ids = vs.add_documents(chunked)
    logger.info("Successfully ingested %d chunks.", len(ids))
    return len(ids)


def ingest_file(file_path: str) -> int:
    """Full pipeline: file path → Qdrant."""
    docs = extract_with_docling(file_path)
    return ingest_documents(docs)


def ingest_directory(dir_path: str, extensions: List[str] = None) -> int:
    """Ingest all supported files in a directory."""
    extensions = extensions or [".pdf", ".docx", ".txt", ".md"]
    total = 0
    for f in Path(dir_path).rglob("*"):
        if f.suffix.lower() in extensions:
            logger.info("Ingesting: %s", f)
            total += ingest_file(str(f))
    return total


# ── Helpers ───────────────────────────────────────────────────────────────────

def _file_hash(file_path: str) -> str:
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()
