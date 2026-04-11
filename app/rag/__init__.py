from importlib import import_module

_EXPORTS = {
    "KUREEmbeddings": ".embedding.kure",
    "get_embeddings": ".embedding.kure",
    "load_jsonl": ".data_loader.jsonl_loader",
    "load_training_documents": ".data_loader.jsonl_loader",
    "parse_illegal_clauses": ".data_loader.special_clause_loader",
    "parse_normal_clauses": ".data_loader.special_clause_loader",
    "chunk_documents": ".chunking.legal_chunker",
    "MultiIndexStore": ".vector_store.multi_index",
    "search_collection": ".retriever.multi_retriever",
    "search_multi_index": ".retriever.multi_retriever",
    "build_context": ".chain.chain",
    "rag_query": ".chain.chain",
    "detect_risk": ".chain.chain",
    "RagBot": ".chain.chain",
    "CONTRACT_QA_PROMPT": ".chain.prompts",
    "RISK_PROMPT": ".chain.prompts",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name in _EXPORTS:
        module = import_module(_EXPORTS[name], __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
