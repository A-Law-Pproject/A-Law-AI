"""판례 XLS 로더.

data/raw/판례/precedents.xls 구조:
  번호 | 제목 | 사건번호 | 선고일자 | 참조조문 | 조문번호 | 판시사항

특징:
  - 하나의 판례가 참조조문이 여러 개일 경우 여러 행에 걸쳐 있음
  - 번호가 비어있는 행은 직전 판례의 추가 참조조문 행
  → 번호가 있는 행 기준으로 판례를 묶어서 로드

page_content 구성:
  제목\n판시사항\n참조조문: ...
"""

from pathlib import Path

from langchain_core.documents import Document
from loguru import logger


def load_precedents(file_path: str | Path) -> list[Document]:
    """판례 XLS 파일을 Document 리스트로 변환.

    Args:
        file_path: precedents.xls 경로.

    Returns:
        Document 리스트. page_content = 제목 + 판시사항 + 참조조문.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        logger.warning(f"[판례] 파일 없음: {file_path}")
        return []

    try:
        import xlrd

        wb = xlrd.open_workbook(str(file_path), encoding_override="cp949")
    except ImportError:
        logger.error("[판례] xlrd 패키지가 설치되지 않았습니다. requirements-dev.txt를 설치하세요.")
        return []
    except Exception as e:
        logger.error(f"[판례] XLS 열기 실패: {e}")
        return []

    ws = wb.sheet_by_index(0)
    logger.info(f"[판례] {ws.nrows}행 로드 시작: {file_path.name}")

    # 헤더 파악 (0행)
    headers = [ws.cell_value(0, c) for c in range(ws.ncols)]
    col = {name: idx for idx, name in enumerate(headers)}

    documents: list[Document] = []
    current: dict | None = None  # 현재 처리 중인 판례

    def _flush(record: dict) -> None:
        """현재 판례를 Document로 변환해 documents에 추가."""
        title     = record.get("제목", "").strip()
        ruling    = record.get("판시사항", "").strip()
        law_refs  = " / ".join(record.get("참조조문_list", [])) or ""

        if not title and not ruling:
            return

        parts = []
        if title:
            parts.append(title)
        if ruling:
            parts.append(ruling)
        if law_refs:
            parts.append(f"참조조문: {law_refs}")

        documents.append(Document(
            page_content="\n".join(parts),
            metadata={
                "case_no":     record.get("사건번호", ""),
                "date":        record.get("선고일자", ""),
                "title":       title[:300],
                "law_refs":    law_refs[:300],
                "source_type": "precedent",
                "source":      str(file_path),
            },
        ))

    for row_idx in range(1, ws.nrows):
        번호_raw = str(ws.cell_value(row_idx, col["번호"])).strip()
        제목      = str(ws.cell_value(row_idx, col["제목"])).strip()
        사건번호  = str(ws.cell_value(row_idx, col["사건번호"])).strip()
        선고일자  = str(ws.cell_value(row_idx, col["선고일자"])).strip()
        참조조문  = str(ws.cell_value(row_idx, col["참조조문"])).strip()
        조문번호  = str(ws.cell_value(row_idx, col["조문번호"])).strip()
        판시사항  = str(ws.cell_value(row_idx, col["판시사항"])).strip()

        # 참조조문 + 조문번호 병합
        ref_text = " ".join(filter(None, [참조조문, 조문번호])).strip()

        has_번호 = 번호_raw and 번호_raw not in ("", "nan", "0.0")

        if has_번호:
            # 새 판례 시작 → 이전 판례 flush
            if current is not None:
                _flush(current)
            current = {
                "제목":        제목,
                "사건번호":    사건번호,
                "선고일자":    선고일자,
                "판시사항":    판시사항,
                "참조조문_list": [ref_text] if ref_text else [],
            }
        else:
            # 번호 없는 행 = 직전 판례의 추가 참조조문
            if current is not None and ref_text:
                current["참조조문_list"].append(ref_text)

    # 마지막 판례 flush
    if current is not None:
        _flush(current)

    logger.info(f"[판례] {len(documents)}개 Document 생성 완료")
    return documents
