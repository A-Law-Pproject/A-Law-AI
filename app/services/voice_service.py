"""
음성 STT + 팩트체크 서비스
- OpenAI Whisper API로 STT
- GPT-4o로 전사본 vs 계약서 팩트체크
"""
import json
import os
from typing import List

from loguru import logger
from openai import AsyncOpenAI

from app.core.config import settings
from app.schemas.voice_dto import FactCheckItem


class VoiceService:
    """음성 STT + 팩트체크 서비스"""

    def __init__(self):
        self._openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    def _infer_content_type(self, s3_key: str) -> str:
        ext = os.path.splitext(s3_key)[-1].lower()
        types = {
            ".mp3": "audio/mpeg",
            ".wav": "audio/wav",
            ".mp4": "audio/mp4",
            ".webm": "audio/webm",
            ".m4a": "audio/x-m4a",
        }
        return types.get(ext, "audio/mpeg")

    async def transcribe(self, audio_bytes: bytes, s3_key: str = "") -> str:
        """
        오디오 bytes → 텍스트 전사 (OpenAI Whisper API)

        Args:
            audio_bytes: S3에서 다운로드한 오디오 파일 bytes
            s3_key: S3 키 (파일명·MIME 타입 추출용)

        Returns:
            전사된 텍스트
        """
        filename = os.path.basename(s3_key) or "audio.mp4"
        content_type = self._infer_content_type(s3_key)

        response = await self._openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=(filename, audio_bytes, content_type),
            language="ko"
        )
        transcript = response.text.strip()
        logger.info(f"Transcription complete: {len(transcript)} chars")
        return transcript

    async def fact_check(self, transcript: str, raw_text: str) -> List[FactCheckItem]:
        """
        전사본 vs 계약서 팩트체크

        Args:
            transcript: Whisper 전사 텍스트
            raw_text: 계약서 원문 텍스트

        Returns:
            FactCheckItem 리스트
        """
        if not raw_text or not raw_text.strip():
            raise ValueError("계약서 원문이 비어 있습니다. OCR 처리 후 다시 시도해주세요.")

        prompt = f"""당신은 임대차 계약서 검증 전문가입니다.

아래 [음성 전사본]과 [계약서 원문]을 비교하여, 전사본에서 계약 내용을 언급한 주장(claim)들을 추출하고 계약서 원문과 일치 여부를 검증하세요.

[음성 전사본]
{transcript}

[계약서 원문]
{raw_text}

각 주장에 대해 다음 JSON 객체 형식으로 응답하세요:
{{"items": [
  {{
    "claim": "전사본에서 언급된 주장",
    "contractContent": "계약서에서 해당 내용의 실제 조항",
    "isMatch": true 또는 false,
    "severity": "HIGH 또는 MEDIUM 또는 LOW (불일치 시에만, 일치하면 null)"
  }}
]}}

- isMatch가 false인 경우: 전사본 내용이 계약서와 다른 경우
- severity 기준: HIGH(보증금/기간 등 핵심 조건 불일치), MEDIUM(부가 조건 불일치), LOW(사소한 표현 차이)
- 계약 내용과 관련 없는 발화는 포함하지 마세요
- 반드시 위 JSON 객체 형식으로만 응답하세요"""

        response = await self._openai_client.chat.completions.create(
            model=settings.MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0
        )

        raw = response.choices[0].message.content
        data = json.loads(raw)
        items_data = data.get("items", [])

        return [
            FactCheckItem(
                claim=item["claim"],
                contractContent=item["contractContent"],
                isMatch=item["isMatch"],
                severity=item.get("severity")
            )
            for item in items_data
        ]
