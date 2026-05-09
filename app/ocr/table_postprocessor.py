"""
표 텍스트 후처리 모듈

OCR에서 줄바꿈 오류로 손상된 표 텍스트를 복원한다.
전략 우선순위:
  1. Upstage element의 markdown 기반 재구성 (elements 있을 때, 가장 신뢰도 높음)
  2. Word 바운딩 박스 y좌표 클러스터링 (단어 좌표만 있을 때)
  3. 텍스트 패턴 휴리스틱 (좌표 없을 때 fallback)
  4. LLM 후처리 (선택적, ENABLE_LLM_TABLE_FIX=true 시 최고 정확도)
"""
import re
import json
from typing import Any, Dict, List, Optional
from loguru import logger


# ============================================================
# 전략 1: Upstage element 마크다운 기반 재구성
# ============================================================

def extract_text_from_elements(elements: List[Dict[str, Any]]) -> str:
    """
    Upstage Document Parse의 elements에서 full_text 재구성.

    표(table) 요소는 content.markdown에서, 나머지는 content.text에서 추출.
    elements를 coordinates 기준 수직 순서대로 처리한다.
    """
    if not elements:
        return ""

    def _top_y(elem: Dict) -> float:
        coords = elem.get("coordinates", [])
        if coords and isinstance(coords[0], dict):
            return min(p.get("y", 0) for p in coords)
        return 0.0

    sorted_elems = sorted(elements, key=_top_y)
    parts: List[str] = []

    for elem in sorted_elems:
        category = elem.get("category", "paragraph")
        content = elem.get("content", {})

        if isinstance(content, str):
            text = content.strip()
        elif category == "table":
            md = content.get("markdown", "")
            text = _table_markdown_to_plain(md) if md else content.get("text", "").strip()
        else:
            text = content.get("text", "").strip()

        if text:
            parts.append(text)

    return "\n".join(parts)


def _table_markdown_to_plain(markdown: str) -> str:
    """
    마크다운 표(| col | col |)를 읽기 쉬운 평문으로 변환.
    구분선(|---|---|)은 제거, 셀은 ' | '로 구분.
    """
    result: List[str] = []

    for line in markdown.strip().split("\n"):
        stripped = line.strip()
        # 구분선 건너뜀: |---|---|, |:--|:--|
        if re.match(r"^[|\s\-:]+$", stripped) and "-" in stripped:
            continue
        if "|" in line:
            cells = [c.strip() for c in line.split("|") if c.strip()]
            if cells:
                result.append(" | ".join(cells))

    return "\n".join(result)


# ============================================================
# 전략 2: Word 바운딩 박스 y좌표 클러스터링
# ============================================================

def cluster_words_into_rows(
    words: List[Any],
    row_height_threshold: float = 0.5,
) -> List[List[Any]]:
    """
    Word 바운딩 박스를 y좌표 기준으로 행(row)으로 클러스터링.

    같은 행: y좌표 차이 ≤ avg_height × row_height_threshold (기본 50%)

    Args:
        words: WordBox 리스트 (y, x, height 속성 또는 키 보유)
        row_height_threshold: 같은 행 판별 임계값 (평균 높이의 배수)

    Returns:
        행 리스트 (각 행은 x 기준 정렬된 WordBox 리스트)
    """
    if not words:
        return []

    def _y(w) -> float:
        return w.y if hasattr(w, "y") else w.get("y", 0)

    def _x(w) -> float:
        return w.x if hasattr(w, "x") else w.get("x", 0)

    def _h(w) -> float:
        return w.height if hasattr(w, "height") else w.get("height", 0)

    heights = [_h(w) for w in words if _h(w) > 0]
    avg_height = sum(heights) / len(heights) if heights else 2.0
    threshold = avg_height * row_height_threshold

    sorted_words = sorted(words, key=lambda w: (_y(w), _x(w)))
    rows: List[List] = []
    current_row = [sorted_words[0]]
    current_y = _y(sorted_words[0])

    for word in sorted_words[1:]:
        if abs(_y(word) - current_y) <= threshold:
            current_row.append(word)
        else:
            rows.append(sorted(current_row, key=_x))
            current_row = [word]
            current_y = _y(word)

    if current_row:
        rows.append(sorted(current_row, key=_x))

    return rows


def reconstruct_text_from_word_clusters(words: List[Any]) -> str:
    """
    Word 바운딩 박스에서 y좌표 클러스터링으로 full_text 재구성.
    같은 행의 단어를 x 순서로 합치고, 행 간에 개행 삽입.
    """
    rows = cluster_words_into_rows(words)

    def _text(w) -> str:
        return w.text if hasattr(w, "text") else w.get("text", "")

    lines = [" ".join(_text(w) for w in row) for row in rows]
    return "\n".join(lines)


# ============================================================
# 전략 3: 텍스트 패턴 휴리스틱 줄 병합
# ============================================================

# 문장 종결 패턴 (끝나면 다음 줄 = 새 행)
_SENTENCE_END = re.compile(r"[.。!?!?]\s*$")

# 다음 줄이 이 패턴으로 시작하면 새 행 (번호 항목, 제N조, 원문자)
_NEW_ROW_STARTERS = re.compile(
    r"^(\d+[.)]\s|제\s*\d+\s*[조항]|[①②③④⑤⑥⑦⑧⑨⑩])"
)

# 다음 줄이 한국어 조사/접속어로 시작하면 앞 줄의 연속
_KO_CONTINUATION = re.compile(
    r"^(은|는|이|가|을|를|의|에서?|으?로|과|와|도|만|까지|부터|에게|보다|처럼|같이"
    r"|이며|이고|하고|하며|이나|이든|하는|하여|해서|으며|이어)"
)


def merge_continuation_lines(text: str) -> str:
    """
    OCR 줄바꿈 오류 텍스트에서 셀 내 연속 줄을 병합.

    보수적으로 동작 — 명확하게 연속임이 판별될 때만 병합:
      - 앞 줄이 종결부호 없이 끝나고 다음 줄이 조사/소문자/괄호로 시작
    명확히 새 행인 경우(번호 항목 등)는 병합 안 함.
    """
    lines = text.split("\n")
    if len(lines) <= 1:
        return text

    result = [lines[0]]

    for line in lines[1:]:
        prev = result[-1].rstrip()
        curr = line.strip()

        if not curr:
            result.append("")
            continue

        if _should_merge(prev, curr):
            result[-1] = prev + " " + curr
        else:
            result.append(line)

    return "\n".join(result)


def _should_merge(prev_line: str, next_line: str) -> bool:
    """두 줄을 같은 셀로 병합할지 판별"""
    if not prev_line or not next_line:
        return False

    # 앞 줄이 종결부호로 끝나면 새 행
    if _SENTENCE_END.search(prev_line):
        return False

    # 다음 줄이 새 행 시작 패턴이면 새 행 (번호 항목 등)
    if _NEW_ROW_STARTERS.match(next_line):
        return False

    # 다음 줄이 한국어 조사/접속어로 시작하면 병합
    if _KO_CONTINUATION.match(next_line):
        return True

    # 다음 줄이 소문자(영문)로 시작하면 병합
    if next_line[0].islower():
        return True

    # 다음 줄이 괄호나 쉼표로 시작하면 병합 (e.g. "(₩50,000,000)")
    if next_line[0] in ("(", "（", ",", "，"):
        return True

    return False


# ============================================================
# 전략 4: LLM 후처리 (선택적)
# ============================================================

class TableLLMReconstructor:
    """
    GPT를 이용한 표 구조 복원.
    비용이 들지만 휴리스틱이 실패하는 복잡한 표에 효과적.
    ENABLE_LLM_TABLE_FIX=true 일 때만 사용.
    """

    _SYSTEM = (
        "당신은 OCR 텍스트 후처리 전문가입니다. "
        "OCR로 추출된 임대차 계약서 텍스트에서 표 안의 셀 내용이 여러 줄로 "
        "잘못 분리된 경우를 찾아 하나의 셀 내용으로 합쳐 복원하세요. "
        "문서 전체 구조(조항 번호, 단락 구분)는 유지하고 "
        "반드시 JSON 형식으로만 응답하세요."
    )

    def __init__(self, openai_client, model: str = "gpt-4o-mini"):
        self._client = openai_client
        self._model = model

    def reconstruct(self, raw_text: str) -> str:
        """
        LLM으로 표 줄바꿈 오류 보정.

        Args:
            raw_text: 보정 전 full_text

        Returns:
            보정된 텍스트 (실패 시 raw_text 그대로 반환)
        """
        if not raw_text:
            return raw_text

        prompt = (
            "다음은 임대차 계약서 OCR 결과입니다. "
            "표 안의 셀 내용이 여러 줄로 잘못 나뉜 경우를 찾아 올바르게 복원하세요.\n\n"
            f"OCR 텍스트:\n{raw_text}\n\n"
            '응답 형식: {"corrected_text": "보정된 전체 텍스트"}'
        )

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": self._SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=4096,
            )
            result = json.loads(response.choices[0].message.content)
            corrected = result.get("corrected_text", "")
            if corrected:
                logger.info("LLM 표 후처리 완료")
                return corrected
        except Exception as e:
            logger.warning(f"LLM 표 후처리 실패 (원본 반환): {e}")

        return raw_text


# ============================================================
# 통합 진입점
# ============================================================

def postprocess_ocr_text(
    raw_text: str,
    elements: Optional[List[Dict[str, Any]]] = None,
    words: Optional[List[Any]] = None,
    llm_reconstructor: Optional[TableLLMReconstructor] = None,
) -> str:
    """
    전략 1→2→3 순으로 적용하고, LLM이 있으면 마지막에 추가 보정.

    Args:
        raw_text:           Upstage content.text (원본 flat 텍스트)
        elements:           Upstage parse_result["elements"] (있으면 전략 1)
        words:              OCR API WordBox 리스트 (있으면 전략 2)
        llm_reconstructor:  TableLLMReconstructor 인스턴스 (없으면 LLM 건너뜀)

    Returns:
        보정된 full_text
    """
    # 전략 1: element 마크다운 기반
    if elements:
        text = extract_text_from_elements(elements)
        if text:
            logger.debug("표 후처리: 전략 1(element 마크다운) 적용")
            return llm_reconstructor.reconstruct(text) if llm_reconstructor else text

    # 전략 2: word bbox 클러스터링
    if words:
        text = reconstruct_text_from_word_clusters(words)
        if text:
            logger.debug("표 후처리: 전략 2(word bbox 클러스터링) 적용")
            return llm_reconstructor.reconstruct(text) if llm_reconstructor else text

    # 전략 3: 텍스트 패턴 휴리스틱 (fallback)
    logger.debug("표 후처리: 전략 3(휴리스틱 줄 병합) 적용")
    text = merge_continuation_lines(raw_text)
    return llm_reconstructor.reconstruct(text) if llm_reconstructor else text
