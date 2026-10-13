"""Policy rulebook ingestion and vector store construction."""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from logistics_agent.config import Settings

logger = logging.getLogger(__name__)

COLLECTION_NAME = "logistics_policies"
_POLICY_ID = re.compile(r"POL-\d{3}")


def load_policy_documents(policy_file: Path) -> List[Document]:
    """Split the rulebook into overlapping chunks tagged with their policy ID."""
    if not policy_file.is_file():
        raise FileNotFoundError(f"Policy rulebook not found: {policy_file}")

    text = policy_file.read_text(encoding="utf-8")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\nPOL-", "\n\n", "\n", " ", ""],
    )

    docs: List[Document] = []
    current_id = None
    for chunk in splitter.split_text(text):
        match = _POLICY_ID.search(chunk)
        if match:
            current_id = match.group()
        docs.append(Document(page_content=chunk, metadata={"policy_id": current_id or "GENERAL"}))
    return docs


def build_vectorstore(settings: Settings) -> Chroma:
    docs = load_policy_documents(settings.policy_file)
    embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)

    kwargs = {}
    if settings.vector_store_dir:
        kwargs["persist_directory"] = str(settings.vector_store_dir)

    store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        ids=[f"chunk-{i}" for i in range(len(docs))],  # stable IDs -> idempotent re-ingest
        collection_name=COLLECTION_NAME,
        collection_metadata={"hnsw:space": "cosine"},
        **kwargs,
    )
    logger.info("Vector store ready: %d chunks from %s", len(docs), settings.policy_file)
    return store
