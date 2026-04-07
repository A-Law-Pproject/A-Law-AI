from .jsonl_loader import load_jsonl, load_training_documents
from .special_clause_loader import parse_illegal_clauses, parse_normal_clauses
from .qa_json_loader import parse_qa_json, load_qa_json_dir
from .precedent_loader import load_precedents
from .contract_json_loader import load_contract_json, load_contract_json_dir
from .doc_loader import load_doc_file, load_doc_dir

__all__ = [
    # JSONL (학습법률문서 임대차 계약)
    "load_jsonl",
    "load_training_documents",
    # Markdown (특약사항)
    "parse_illegal_clauses",
    "parse_normal_clauses",
    # JSON (민사법 QA)
    "parse_qa_json",
    "load_qa_json_dir",
    # XLS (판례)
    "load_precedents",
    # AI Hub 계약 JSON (TL_* 폴더)
    "load_contract_json",
    "load_contract_json_dir",
    # Word 법률 원문 (.doc / .docx)
    "load_doc_file",
    "load_doc_dir",
]
