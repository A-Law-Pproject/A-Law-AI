"""LLM 응답에서 원시 마크다운 마커를 제거해 가독성 좋은 텍스트로 정리한다.

챗봇 답변과 계약서 요약 모두 동일한 정리 규칙을 공유한다.
"""
from __future__ import annotations

import re


def ensure_readable_markdown_answer(answer: str) -> str:
    text = (answer or "").replace("\r\n", "\n").strip()
    if not text:
        return ""

    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 인라인으로 붙어 있는 헤딩/인용을 다음 줄로 떼어 렌더링 구조를 유지한다.
    text = re.sub(r"(?<!\n)(?=#{1,6}\s)", "\n\n", text)
    text = re.sub(r"(?<!\n)[ \t]+(?=>\s)", "\n\n", text)

    # 섹션 제목은 살리되 ##, ###, > 같은 마커는 숨긴다.
    text = re.sub(r"(?m)^\s*#{1,6}\s*", "", text)
    text = re.sub(r"(?m)^\s*>\s*", "", text)
    text = re.sub(r"(?<!\*)\*\*(.+?)\*\*(?!\*)", r"\1", text)
    text = re.sub(r"(?<!_)__(.+?)__(?!_)", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = text.replace("**", "")

    # 한 줄로 이어붙은 번호/굵은 글머리 목록을 줄바꿈한다.
    text = re.sub(r"(?<=:)[ \t]+(?=(?:\d+\.\s+|-\s+\*\*))", "\n\n", text)
    text = re.sub(r"(?<!\n)[ \t]+(?=(?:\d+\.\s+|-\s+\*\*))", "\n", text)

    # 목록 마지막 항목 뒤에 붙은 마무리 문장을 별도 문단으로 분리한다.
    text = re.sub(
        r"(?<=[.!?])[ \t]+(?=(?:따라서|정리하면|결론적으로|즉,|즉\s|다만|한편|추가로))",
        "\n\n",
        text,
    )

    text = re.sub(r"[ \t]{2,}", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()
