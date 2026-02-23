#------------------------------
# md 파일 loader 🅾️
#------------------------------

import re
from pathlib import Path

from langchain_core.documents import Document
from loguru import logger


def parse_illegal_clauses(file_path: str | Path) -> list[Document]:
    """illegal.md 파싱 - markdown 테이블에서 독소조항 추출.

    Args:
        file_path: illegal.md 파일 경로.

    Returns:
        독소 특약 Document 리스트.
    """
    file_path = Path(file_path)
    documents: list[Document] = []

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    for line in content.strip().split("\n"):
        line = line.strip()
        if not line.startswith("|") or line.startswith("| 번호") or line.startswith("| ---"):
            continue

        parts = [p.strip() for p in line.split("|")]
        parts = [p for p in parts if p]
        if len(parts) >= 3:
            documents.append(Document(
                page_content=parts[2],
                metadata={
                    "number": parts[0],
                    "category": parts[1],
                    "type": "illegal",
                    "source": "특약사항/illegal.md",
                },
            ))

    logger.info(f"Parsed {len(documents)} illegal clauses from {file_path.name}")
    return documents


def parse_normal_clauses(file_path: str | Path) -> list[Document]:
    """normal.md 파싱 - ### 헤더 기준으로 정상 특약 추출.

    Args:
        file_path: normal.md 파일 경로.

    Returns:
        정상 특약 Document 리스트.
    """
    file_path = Path(file_path)
    documents: list[Document] = []

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    current_category = ""
    sections = re.split(r"(?=^### )", content, flags=re.MULTILINE)

    for section in sections:
        section = section.strip()
        if not section.startswith("### "):
            cat_match = re.search(r"^## (.+?)(?:\s*\(\d+개\))?$", section, re.MULTILINE)
            if cat_match:
                current_category = cat_match.group(1).strip()
            continue

        lines = section.split("\n")
        header = lines[0].replace("### ", "").strip()
        number_match = re.match(r"(\d+)\.\s*(.+)", header)

        if number_match:
            number = number_match.group(1)
            clause_content = number_match.group(2)
        else:
            number = ""
            clause_content = header

        basis = ""
        for line in lines[1:]:
            if line.startswith("**관련 근거"):
                basis = line.replace("**관련 근거 및 기대 효과**:", "").strip()
            elif line.startswith("**피드백**"):
                feedback = line.replace("**피드백**:", "").strip()
                if feedback:
                    basis += f" (피드백: {feedback})"

        full_content = clause_content
        if basis:
            full_content += f"\n관련 근거: {basis}"

        documents.append(Document(
            page_content=full_content,
            metadata={
                "number": number,
                "category": current_category,
                "type": "normal",
                "source": "특약사항/normal.md",
            },
        ))

    logger.info(f"Parsed {len(documents)} normal clauses from {file_path.name}")
    return documents
