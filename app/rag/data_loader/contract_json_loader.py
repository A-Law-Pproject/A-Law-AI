"""AI Hub 계약 JSON 로더.

data/raw/학습법률문서/계약_법률_문서_서식_데이터/TL_*/*.json 구조:
  {
    "document": {
      "type": "계약 법률 문서",
      "metadata": {
        "file_info": {
          "document_name": "상업용부동산임대계약서_0006",
          "document_category": {
            "main_category": "부동산매매_임대차",
            "sub_category": "부동산임대",
            "detail_category": "상업용부동산임대계약서"
          }
        }
      },
      "sections": [
        {
          "id": "1",
          "format": {"format_type": "제목", "article": null},
          "content": {
            "description": "부동산(상가)임대차계약서",
            "content_labels": ["기타"]
          }
        },
        ...
      ]
    }
  }

청킹 전략:
  - article 번호가 있는 sections → 같은 article끼리 묶어서 하나의 Document
  - article 없는 sections (제목/서문) → 문서 하나로 통합
  - 빈 description은 스킵
"""

import json
from pathlib import Path

from langchain_core.documents import Document
from loguru import logger


def _extract_metadata(doc_meta: dict) -> dict:
    """document.metadata에서 필요한 필드 추출."""
    fi = doc_meta.get("file_info", {})
    cat = fi.get("document_category", {})
    return {
        "document_name":   fi.get("document_name", ""),
        "main_category":   cat.get("main_category", ""),
        "sub_category":    cat.get("sub_category", ""),
        "detail_category": cat.get("detail_category", ""),
        "page_count":      fi.get("document_page_count", ""),
        "source_type":     "contract_json",
    }


def load_contract_json(file_path: Path) -> list[Document]:
    """단일 JSON 파일을 article 단위로 청킹하여 Document 리스트 반환."""
    try:
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.warning(f"[JSON로더] 파싱 실패 {file_path.name}: {e}")
        return []

    doc_data = data.get("document", {})
    sections = doc_data.get("sections", [])
    meta_base = _extract_metadata(doc_meta=doc_data.get("metadata", {}))

    # article 번호별로 sections 그룹화
    # article=None → 제목/서문 (문서당 하나의 chunk로 통합)
    article_groups: dict[str | None, list[str]] = {}

    for sec in sections:
        desc = sec.get("content", {}).get("description", "").strip()
        if not desc:
            continue
        art = sec.get("format", {}).get("article")  # None or int
        key = str(art) if art is not None else None
        article_groups.setdefault(key, []).append(desc)

    documents: list[Document] = []
    doc_name = meta_base["document_name"]

    for art_key, texts in article_groups.items():
        page_content = "\n".join(texts)
        if len(page_content.strip()) < 10:
            continue

        chunk_id = f"{doc_name}_art{art_key}" if art_key else f"{doc_name}_intro"
        metadata = {
            **meta_base,
            "title":    doc_name,
            "article":  f"제{art_key}조" if art_key else "",
            "chunk_id": chunk_id,
        }
        documents.append(Document(page_content=page_content, metadata=metadata))

    return documents


def load_contract_json_dir(folder_path: str | Path) -> list[Document]:
    """폴더 내 모든 *.json 파일 로드.

    Args:
        folder_path: TL_* 폴더 경로.

    Returns:
        Document 리스트 (article 단위 청킹).
    """
    folder_path = Path(folder_path)
    json_files = sorted(folder_path.glob("*.json"))

    if not json_files:
        logger.warning(f"[JSON로더] JSON 파일 없음: {folder_path}")
        return []

    logger.info(f"[JSON로더] {len(json_files)}개 파일 로드 시작: {folder_path.name}")

    documents: list[Document] = []
    for jf in json_files:
        docs = load_contract_json(jf)
        documents.extend(docs)

    logger.info(f"[JSON로더] {folder_path.name} → {len(documents)}개 Document (article 단위)")
    return documents
