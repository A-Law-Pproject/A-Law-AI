"""AI Hub unfair lease clause JSON loader.

The source files are FTC-style clause analysis JSON files. Each file is loaded
as one retrieval document so the disputed clause, unfairness basis, and related
law stay together during RAG retrieval.
"""

import json
from pathlib import Path

from langchain_core.documents import Document
from loguru import logger


def _as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _format_section(title: str, values: list[str]) -> str:
    if not values:
        return ""
    body = "\n".join(f"- {value}" for value in values)
    return f"[{title}]\n{body}"


def load_unfavorable_clause_json(file_path: str | Path, root_path: str | Path | None = None) -> Document | None:
    """Load one unfavorable lease clause JSON file as a LangChain Document."""
    file_path = Path(file_path)
    root = Path(root_path) if root_path is not None else file_path.parent

    try:
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        logger.warning(f"[unfavorable_clause_json] failed to parse {file_path}: {exc}")
        return None

    clause_articles = _as_list(data.get("clauseArticle"))
    unfair_bases = _as_list(data.get("illdcssBasiss"))
    related_laws = _as_list(data.get("relateLaword"))

    if not clause_articles:
        logger.warning(f"[unfavorable_clause_json] empty clauseArticle: {file_path}")
        return None

    sections = [
        _format_section("약관 조항", clause_articles),
        _format_section("불리 판단 근거", unfair_bases),
        _format_section("관련 법령", related_laws),
    ]
    page_content = "\n\n".join(section for section in sections if section)

    try:
        relative_path = file_path.relative_to(root).as_posix()
    except ValueError:
        relative_path = file_path.name

    metadata = {
        "title": file_path.stem,
        "source": "AIHub_legal_training_terms_TL_2",
        "source_type": "unfavorable_clause_json",
        "source_file": file_path.name,
        "source_path": relative_path,
        "domain": "lease_contract",
        "label": "unfavorable",
        "type": "unfavorable",
        "category": "임대차계약_불리약관",
        "clause_field": str(data.get("clauseField", "")),
        "ftc_conclusion": str(data.get("ftcCnclsns", "")),
        "advantageous_level": str(data.get("dvAntageous", "")),
        "unfavorable_provision": str(data.get("unfavorableProvision", "")),
        "related_law": "\n".join(related_laws),
        "basis": "\n".join(unfair_bases),
    }
    return Document(page_content=page_content, metadata=metadata)


def load_unfavorable_clause_json_dir(folder_path: str | Path) -> list[Document]:
    """Recursively load all remaining *.json files under the source folder."""
    folder = Path(folder_path)
    json_files = sorted(folder.rglob("*.json"))

    if not json_files:
        logger.warning(f"[unfavorable_clause_json] no JSON files: {folder}")
        return []

    documents: list[Document] = []
    for json_file in json_files:
        doc = load_unfavorable_clause_json(json_file, root_path=folder)
        if doc is not None:
            documents.append(doc)

    logger.info(
        f"[unfavorable_clause_json] loaded {len(documents)}/{len(json_files)} documents from {folder}"
    )
    return documents
