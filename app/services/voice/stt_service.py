"""
STT (음성→텍스트) 서비스
- OpenAI Whisper API를 사용해 음성 파일을 텍스트로 변환
- 언어: 한국어 (language="ko")
- 세그먼트 단위 전사 (start_time, end_time, text 포함)
- 발화에서 금액/날짜/합의 표현 패턴 자동 추출
"""
import re
import asyncio
import io
from typing import List, Optional

from loguru import logger
from openai import AsyncOpenAI

from app.core.config import settings
from app.schemas.voice_shared import SegmentResult, AgreementItem


# ========================
# 한국어 패턴 정규식
# ========================

# 금액 패턴: 숫자 + 단위 (천원~조 단위, 퍼센트 포함)
_AMOUNT_PATTERN = re.compile(
    r"(\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*(?:조|억|천만|백만|십만|만)?원|\d+(?:\.\d+)?\s*%)"
)

# 날짜/기간 패턴
_DATE_PATTERN = re.compile(
    r"(\d{4}년\s*\d{1,2}월(?:\s*\d{1,2}일)?|\d{1,2}월\s*\d{1,2}일|\d+\s*개월|\d+\s*년\s*\d*\s*개월?)"
)

# 합의 표현 패턴
_AGREEMENT_PATTERN = re.compile(
    r"(맞습니다|동의합니다?|그렇게\s*하죠|확인했습니다|알겠습니다|좋습니다|그렇습니다|"
    r"네\s*맞아요|맞아요|동의해요|수락합니다?|승낙합니다?|동의하겠습니다)"
)

# 조건 표현 패턴
_CONDITION_PATTERN = re.compile(
    r"(만약|만일|경우에는|경우\s*에는|단\s*,|단\s+|다만|단,|그러나|하지만|예외적으로)"
)


def _seconds_to_timestamp(seconds: float) -> str:
    """초 단위 시간을 HH:MM:SS 형식으로 변환한다."""
    total_secs = int(seconds)
    hours = total_secs // 3600
    minutes = (total_secs % 3600) // 60
    secs = total_secs % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def extract_agreements_from_segments(segments: List[SegmentResult]) -> List[AgreementItem]:
    """
    발화 세그먼트 목록에서 합의 관련 항목(금액, 날짜, 합의 표현, 조건)을 추출한다.

    Args:
        segments: STT 발화 세그먼트 목록

    Returns:
        추출된 AgreementItem 목록
    """
    agreements: List[AgreementItem] = []

    for seg in segments:
        text = seg.text

        # 금액 추출
        for match in _AMOUNT_PATTERN.finditer(text):
            agreements.append(AgreementItem(
                segment_id=seg.id,
                agreement_type="amount",
                value=match.group().strip(),
                context=text,
                timestamp_str=seg.timestamp_str,
            ))

        # 날짜/기간 추출
        for match in _DATE_PATTERN.finditer(text):
            agreements.append(AgreementItem(
                segment_id=seg.id,
                agreement_type="date",
                value=match.group().strip(),
                context=text,
                timestamp_str=seg.timestamp_str,
            ))

        # 합의 표현 추출
        for match in _AGREEMENT_PATTERN.finditer(text):
            agreements.append(AgreementItem(
                segment_id=seg.id,
                agreement_type="agreement",
                value=match.group().strip(),
                context=text,
                timestamp_str=seg.timestamp_str,
            ))

        # 조건 표현 추출 (컨텍스트만, 값은 조건 키워드)
        for match in _CONDITION_PATTERN.finditer(text):
            agreements.append(AgreementItem(
                segment_id=seg.id,
                agreement_type="condition",
                value=match.group().strip(),
                context=text,
                timestamp_str=seg.timestamp_str,
            ))

    logger.debug(f"합의 항목 추출 완료: {len(agreements)}개")
    return agreements


async def transcribe_audio(
    file_bytes: bytes,
    filename: str = "audio.mp3",
    language: str = "ko",
) -> List[SegmentResult]:
    """
    OpenAI Whisper API로 음성 파일을 세그먼트 단위로 전사한다.

    Args:
        file_bytes: 음성 파일 바이트
        filename: 파일명 (확장자 포함, MIME 타입 추론에 사용)
        language: 전사 언어 (기본값: 'ko' 한국어)

    Returns:
        SegmentResult 목록 (start_time, end_time, text 포함)

    Raises:
        RuntimeError: Whisper API 호출 실패
    """
    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    try:
        logger.info(f"Whisper STT 시작: filename={filename}, size={len(file_bytes)} bytes")

        # OpenAI Whisper API는 파일 객체를 요구하므로 BytesIO로 감쌈
        audio_file = io.BytesIO(file_bytes)
        audio_file.name = filename  # 파일명 속성 필요 (MIME 타입 추론용)

        response = await client.audio.transcriptions.create(
            model=settings.WHISPER_MODEL,
            file=audio_file,
            language=language,
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )

        logger.info(f"Whisper STT 완료: 세그먼트 수={len(response.segments or [])}")

        segments: List[SegmentResult] = []
        raw_segments = response.segments or []

        for idx, seg in enumerate(raw_segments):
            start = float(getattr(seg, "start", 0.0))
            end = float(getattr(seg, "end", 0.0))
            text = str(getattr(seg, "text", "")).strip()

            segments.append(SegmentResult(
                id=f"seg_{idx}",
                start_time=start,
                end_time=end,
                text=text,
                speaker=None,  # 화자 분리 미지원 시 None
                timestamp_str=_seconds_to_timestamp(start),
            ))

        # 세그먼트가 없는 경우 (짧은 파일 등) 전체 텍스트를 단일 세그먼트로 처리
        if not segments and hasattr(response, "text") and response.text:
            segments.append(SegmentResult(
                id="seg_0",
                start_time=0.0,
                end_time=0.0,
                text=response.text.strip(),
                speaker=None,
                timestamp_str="00:00:00",
            ))
            logger.warning("세그먼트 정보 없음 — 전체 텍스트를 단일 세그먼트로 처리")

        return segments

    except Exception as e:
        logger.error(f"Whisper STT 실패: {e}")
        raise RuntimeError(f"STT 처리 실패: {e}") from e


async def transcribe_and_extract(
    file_bytes: bytes,
    filename: str = "audio.mp3",
    language: str = "ko",
) -> tuple[List[SegmentResult], List[AgreementItem]]:
    """
    STT 전사 후 합의 항목까지 한번에 추출하는 편의 함수.

    Args:
        file_bytes: 음성 파일 바이트
        filename: 파일명
        language: 전사 언어

    Returns:
        (SegmentResult 목록, AgreementItem 목록) 튜플
    """
    segments = await transcribe_audio(file_bytes, filename, language)
    agreements = await asyncio.to_thread(extract_agreements_from_segments, segments)
    return segments, agreements
