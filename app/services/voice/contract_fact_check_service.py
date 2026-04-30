"""
Core logic for Spring-integrated async voice contract fact-check jobs.
"""
import asyncio
import json
from typing import List, Optional

from loguru import logger

from app.core.config import settings
from app.core.dependencies import fetch_contract_text, get_llm
from app.schemas.voice_contract_fact_check import (
    FactCheckItem,
    VoiceContractFactCheckRequest,
)
from app.services.voice.hash_service import download_from_s3
from app.services.voice.stt_service import transcribe_audio


_FACT_CHECK_PROMPT = """You are a legal contract analyst.
Compare the spoken transcript against the contract text and return only a JSON array.

[Contract Text]
{raw_text}

[Transcript]
{transcript}

Return this shape only:
[
  {{
    "claim": "claim from transcript",
    "contractContent": "matching contract clause or 'No matching clause'",
    "isMatch": true,
    "severity": null
  }},
  {{
    "claim": "another claim",
    "contractContent": "relevant contract clause",
    "isMatch": false,
    "severity": "HIGH"
  }}
]

Rules:
- Focus on key contractual terms such as deposit, rent, duration, and special clauses.
- Return at most 10 items.
- If `isMatch` is true, `severity` must be null.
- If the contract has no matching clause for a spoken claim, use `isMatch=false`,
  `severity=HIGH`, and `contractContent="No matching clause"`.
- If there are no contract-related claims in the transcript, return [].
"""

_VOICE_ONLY_PROMPT = """You are a legal contract analyst.
Extract key contractual claims from the spoken transcript below.
No contract document is available for verification, so every claim is unverifiable.

[Transcript]
{transcript}

Return only a JSON array (at most 10 items):
[
  {{
    "claim": "claim extracted from transcript",
    "contractContent": "계약서 없음",
    "isMatch": false,
    "severity": "HIGH"
  }}
]

Rules:
- Extract claims related to deposit, rent, duration, special clauses, or any agreed terms.
- Every item must have `isMatch=false` and `severity="HIGH"` because no contract exists to verify against.
- If no contractual claims are present in the transcript, return [].
"""


async def run_fact_check(transcript: str, contract_text: str) -> List[FactCheckItem]:
    """Run fact-checking against the resolved contract text."""
    llm = get_llm()

    prompt = _FACT_CHECK_PROMPT.format(
        raw_text=contract_text[:3000],
        transcript=transcript[:2000],
    )

    try:
        response = await asyncio.wait_for(
            llm.ainvoke(prompt),
            timeout=settings.VOICE_ANALYSIS_TIMEOUT,
        )
        content = response.content.strip()

        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]

        items_raw = json.loads(content)
        return [FactCheckItem(**item) for item in items_raw]

    except asyncio.TimeoutError:
        logger.warning(
            f"[VoiceContractFactCheck] Timeout ({settings.VOICE_ANALYSIS_TIMEOUT}s)"
        )
        return []
    except Exception as e:
        logger.error(f"[VoiceContractFactCheck] Parse error: {e}")
        return []


async def run_voice_only_analysis(transcript: str) -> List[FactCheckItem]:
    """계약서 없이 음성 녹음만으로 계약 관련 주장을 추출한다."""
    llm = get_llm()

    prompt = _VOICE_ONLY_PROMPT.format(transcript=transcript[:2000])

    try:
        response = await asyncio.wait_for(
            llm.ainvoke(prompt),
            timeout=settings.VOICE_ANALYSIS_TIMEOUT,
        )
        content = response.content.strip()

        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]

        items_raw = json.loads(content)
        return [FactCheckItem(**item) for item in items_raw]

    except asyncio.TimeoutError:
        logger.warning(
            f"[VoiceContractFactCheck] Voice-only timeout ({settings.VOICE_ANALYSIS_TIMEOUT}s)"
        )
        return []
    except Exception as e:
        logger.error(f"[VoiceContractFactCheck] Voice-only parse error: {e}")
        return []


async def resolve_contract_text(request: VoiceContractFactCheckRequest) -> Optional[str]:
    """OCR 저장소에서 계약서 텍스트를 조회한다. 없으면 None 반환."""
    try:
        text = await fetch_contract_text(request.contractId)
        text = text.strip()
        if not text:
            logger.warning(
                "[VoiceContractFactCheck] OCR text is empty, switching to voice-only: "
                f"contractId={request.contractId}"
            )
            return None
        logger.info(
            "[VoiceContractFactCheck] OCR lookup resolved contract text: "
            f"contractId={request.contractId}, length={len(text)}"
        )
        return text
    except ValueError:
        logger.warning(
            "[VoiceContractFactCheck] OCR result not found, switching to voice-only: "
            f"contractId={request.contractId}"
        )
        return None


async def transcribe_audio_from_request(
    request: VoiceContractFactCheckRequest,
) -> str:
    """Download the uploaded audio file and convert it to a combined transcript."""
    logger.info(f"[VoiceContractFactCheck] S3 download: s3Key={request.s3Key}")
    file_bytes = await download_from_s3(request.s3Key)
    filename = request.s3Key.split("/")[-1]

    logger.info(
        f"[VoiceContractFactCheck] STT start: voiceRecordId={request.voiceRecordId}"
    )
    segments = await transcribe_audio(file_bytes, filename)
    transcript = " ".join(seg.text for seg in segments).strip()
    logger.info(f"[VoiceContractFactCheck] STT completed: {len(transcript)} chars")
    return transcript
