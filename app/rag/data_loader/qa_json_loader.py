#------------------------------
# 민사법 QA JSON 로더
#------------------------------

import json
from pathlib import Path

from langchain_core.documents import Document
from loguru import logger


# 폴더명 패턴 → source_type 매핑
_FOLDER_SOURCE_MAP = {
    "판결문": "precedent_qa",
    "법령":   "law_qa",
    "유권해석": "interpretation_qa",
}


def _detect_source_type(file_path: Path) -> str:
    path_str = str(file_path)
    for keyword, stype in _FOLDER_SOURCE_MAP.items():
        if keyword in path_str:
            return stype
    return "law_qa"


def _extract_qa(item: dict) -> tuple[str, str]:
    """다양한 JSON 포맷에서 (question, answer) 추출.

    지원 포맷:
      1. AI Hub 민사법 포맷: {"taskinfo": {"input": "...", "output": "..."}}
      2. 표준 포맷:          {"question": "...", "answer": "..."}
      3. 한국어 키 포맷:     {"질문": "...", "답변": "..."}
    """
    # 1. AI Hub 민사법 포맷 (taskinfo.input / taskinfo.output)
    taskinfo = item.get("taskinfo", {})
    if taskinfo:
        question = taskinfo.get("input", "").strip()
        answer   = taskinfo.get("output", "").strip()
        if question and answer:
            return question, answer

    # 2. 표준 / 한국어 키 포맷
    question = item.get("question", item.get("질문", "")).strip()
    answer   = item.get("answer",   item.get("답변",   "")).strip()
    return question, answer


def parse_qa_json(file_path: str | Path) -> list[Document]:
    """AI Hub 민사법 QA JSON 파일 파싱.

    지원 포맷:
      1. AI Hub 민사법: {"info": {...}, "taskinfo": {"input": "...", "output": "..."}}
      2. 표준:          {"question": "...", "answer": "..."}
      3. 한국어 키:     {"질문": "...", "답변": "..."}
      4. 리스트 래퍼:   [item, ...]
      5. data 래퍼:     {"data": [...]}

    Args:
        file_path: JSON 파일 경로.

    Returns:
        Document 리스트 (page_content = "질문: ...\n답변: ...").
    """
    file_path = Path(file_path)
    source_type = _detect_source_type(file_path)
    documents: list[Document] = []

    with open(file_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            data = [json.loads(line) for line in f if line.strip()]

    # AI Hub 단일 오브젝트 포맷 (파일 1개 = 판결문 1건)
    if isinstance(data, dict) and "taskinfo" in data:
        items = [data]
    elif isinstance(data, list):
        items = data
    else:
        items = data.get("data", [data])

    for item in items:
        question, answer = _extract_qa(item)

        if not question or not answer:
            continue

        # AI Hub 포맷에서 문서 ID 및 사건명 추출
        info = item.get("info", {})
        doc_id    = info.get("doc_id", "")
        casenames = info.get("casenames", "")

        documents.append(Document(
            page_content=f"질문: {question}\n답변: {answer}",
            metadata={
                "source_type": source_type,
                "question":    question[:500],
                "reference":   item.get("reference", item.get("출처", doc_id))[:200],
                "casenames":   casenames[:100],
                "source":      str(file_path),
            },
        ))

    return documents


def load_qa_json_dir(data_dir: str | Path) -> list[Document]:
    """디렉토리 내 모든 민사법 QA JSON 파일 로드.

    Args:
        data_dir: 민사법_LLM_사전학습_및_Instruction_Tuning_데이터 루트 경로.

    Returns:
        전체 Document 리스트.
    """
    data_dir = Path(data_dir)
    all_docs: list[Document] = []

    json_files = list(data_dir.rglob("*.json"))
    logger.info(f"[QA JSON] {len(json_files)}개 파일 발견: {data_dir}")

    for json_file in json_files:
        docs = parse_qa_json(json_file)
        all_docs.extend(docs)
        logger.debug(f"  {json_file.name}: {len(docs)}개 청크")

    logger.info(f"[QA JSON] 총 {len(all_docs)}개 Document 로드 완료")
    return all_docs
