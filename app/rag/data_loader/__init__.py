from importlib import import_module

_EXPORTS = {
    "load_jsonl": ".jsonl_loader",
    "load_training_documents": ".jsonl_loader",
    "parse_illegal_clauses": ".special_clause_loader",
    "parse_normal_clauses": ".special_clause_loader",
    "parse_qa_json": ".qa_json_loader",
    "load_qa_json_dir": ".qa_json_loader",
    "load_precedents": ".precedent_loader",
    "load_contract_json": ".contract_json_loader",
    "load_contract_json_dir": ".contract_json_loader",
    "load_doc_file": ".doc_loader",
    "load_doc_dir": ".doc_loader",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name in _EXPORTS:
        module = import_module(_EXPORTS[name], __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
