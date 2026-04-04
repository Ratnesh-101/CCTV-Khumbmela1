"""
LangChain + FAISS RAG over local SOP markdown files.
Falls back to extractive context if no chat LLM is configured.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def build_vectorstore(
    sop_dir: Optional[Path] = None,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> FAISS:
    root = _project_root()
    path = sop_dir or (root / "data" / "sops")
    if not path.exists():
        raise FileNotFoundError(f"SOP directory missing: {path}")

    loader = DirectoryLoader(
        str(path),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
    chunks = splitter.split_documents(docs)
    emb = HuggingFaceEmbeddings(model_name=embedding_model)
    return FAISS.from_documents(chunks, emb)


_VS: Optional[FAISS] = None


def get_vectorstore() -> FAISS:
    global _VS
    if _VS is None:
        _VS = build_vectorstore()
    return _VS


def retrieve_context(query: str, k: int = 4) -> str:
    vs = get_vectorstore()
    docs = vs.similarity_search(query, k=k)
    parts = []
    for i, d in enumerate(docs, 1):
        parts.append(f"[{i}] {d.page_content.strip()}")
    return "\n\n".join(parts)


def rag_operator_brief(alert_labels: str, risk_score: float) -> str:
    """RAG-backed brief for operators (extractive + optional LLM)."""
    q = f"Incident playbook steps for: {alert_labels}. Risk level around {risk_score}."
    context = retrieve_context(q, k=4)

    llm_text = _optional_llm_synthesis(q, context)
    if llm_text:
        return f"--- RAG context ---\n{context}\n\n--- LLM synthesis ---\n{llm_text}"
    return f"--- RAG context (extractive) ---\n{context}"


def _optional_llm_synthesis(query: str, context: str) -> Optional[str]:
    """Use Ollama or OpenAI if env vars are set; otherwise None."""
    # Ollama
    ollama_model = os.environ.get("OLLAMA_MODEL")
    if ollama_model:
        try:
            from langchain_community.llms import Ollama

            llm = Ollama(model=ollama_model, temperature=0.2)
            prompt = (
                "You are a security operations assistant. Use ONLY the context if possible.\n"
                f"Context:\n{context}\n\nQuestion: {query}\n"
                "Reply with 5 short bullet steps for operators."
            )
            return llm.invoke(prompt).strip()
        except Exception:
            pass

    if os.environ.get("OPENAI_API_KEY"):
        try:
            from langchain_core.messages import HumanMessage
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(
                model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
                temperature=0.2,
            )
            prompt = (
                "Use the context. If unsure, say verify on CCTV.\n"
                f"Context:\n{context}\n\nTask: {query}\n5 bullet operator steps."
            )
            return llm.invoke([HumanMessage(content=prompt)]).content.strip()
        except Exception:
            pass

    return None
