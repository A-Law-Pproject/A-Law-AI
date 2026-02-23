#------------------------------
# jsonl 파일 loader 🅾️
#------------------------------

import json
from pathlib import Path

from langchain_core.documents import Document
from loguru import logger


def load_jsonl(file_path: Path, max_docs: int | None = None) -> list[Document]:
    """JSONL 파일에서 Document 리스트 로드.

    Args:
        file_path: JSONL 파일 경로.
        max_docs: 최대 로드 문서 수 (None이면 전체 로드).

    Returns:
        Document 리스트.
    """
    documents: list[Document] = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)
            content = data.get("content", "").strip()
            if not content:
                continue

            metadata = {
                "doc_id": data.get("doc_id", ""),
                "document_type": data.get("document_type", ""),
                "chunk_id": data.get("chunk_id", ""),
                "title": data.get("title", ""),
                "article": data.get("article", ""),
                "source_file": str(file_path),
            }
            for field in [
                "clause_field", "ftc_conclusions", "unfavorable_provision",
                "case_no", "court_name", "judgment_date",
            ]:
                if data.get(field):
                    metadata[field] = data[field]
            if data.get("related_law_text"):
                metadata["related_law_text"] = data["related_law_text"]
            if data.get("analysis_basis"):
                metadata["analysis_basis"] = data["analysis_basis"]

            documents.append(Document(page_content=content, metadata=metadata))

            if max_docs and len(documents) >= max_docs:
                break

    return documents


def load_training_documents(
    data_path: str | Path,
    max_docs_per_file: int | None = None,
) -> tuple[list[Document], list[Document]]:
    """Training JSONL 파일만 로드 (계약_외 제외).

    Args:
        data_path: 학습법률문서 루트 경로.
        max_docs_per_file: 파일당 최대 문서 수 (None이면 전체).

    Returns:
        (law_docs, contract_docs) 튜플.
    """
    data_path = Path(data_path)
    all_jsonl = list(data_path.rglob("*.jsonl"))
    training_files = [
        f for f in all_jsonl
        if "Training" in str(f) and "계약_외" not in str(f)
    ]

    logger.info(f"Total JSONL: {len(all_jsonl)}, Training (excluding 계약_외): {len(training_files)}")

    law_docs: list[Document] = []
    contract_docs: list[Document] = []

    for jsonl_file in training_files:
        docs = load_jsonl(jsonl_file, max_docs=max_docs_per_file)
        file_str = str(jsonl_file)

        if "법률_규정" in file_str or "약관" in file_str or "판결문" in file_str:
            law_docs.extend(docs)
        elif "계약_법률_문서" in file_str or "라벨링_임대" in file_str:
            contract_docs.extend(docs)
        else:
            law_docs.extend(docs)

        logger.debug(f"  {jsonl_file.name}: {len(docs)} docs")

    logger.info(f"Law docs: {len(law_docs)}, Contract docs: {len(contract_docs)}")
    return law_docs, contract_docs
