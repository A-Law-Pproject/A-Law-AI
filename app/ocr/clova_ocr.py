"""
Naver Clova OCR General API 기반 파이프라인

UpstageOCRPipeline과 동일한 process() 인터페이스를 제공하므로
ocr_service.py에서 주석 처리만으로 교체 가능.

필수 환경변수:
    CLOVA_OCR_API_URL   : https://{apigw}.ntruss.com/custom/v1/{domain_id}/{invoke_key}/general
    CLOVA_OCR_SECRET_KEY: X-OCR-SECRET 헤더 값 (NCloud Console에서 발급)
"""
import base64
import json
import uuid
import time as _time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx
from loguru import logger

try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False


# ──────────────────────────────────────────────────────────────
# 내부 결과 타입 (OCROverlayResponse 호환)
# ──────────────────────────────────────────────────────────────

@dataclass
class _WordBox:
    text: str
    x: float          # 좌상단 X (%, 0~100)
    y: float          # 좌상단 Y (%, 0~100)
    width: float      # 너비 (%)
    height: float     # 높이 (%)
    confidence: float = 1.0


@dataclass
class ClovaOCRResult:
    """ContractOCRResponse.from_result() 호환 결과 객체"""
    success: bool = True
    image_width: int = 0
    image_height: int = 0
    full_text: str = ""
    markdown: str = ""
    contract_data: Optional[Dict[str, Any]] = None
    validation: Optional[Dict[str, Any]] = None
    words: List[_WordBox] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None


# ──────────────────────────────────────────────────────────────
# 좌표 변환
# ──────────────────────────────────────────────────────────────

def _vertices_to_box(vertices: List[Dict], img_w: int, img_h: int) -> Dict[str, float]:
    """
    Clova boundingPoly.vertices (4점, 픽셀) → (x%, y%, width%, height%).
    vertices 순서가 일정하지 않을 수 있으므로 min/max로 계산.
    """
    xs = [v["x"] for v in vertices]
    ys = [v["y"] for v in vertices]
    x0, y0 = min(xs), min(ys)
    x1, y1 = max(xs), max(ys)

    safe_w = img_w or 1
    safe_h = img_h or 1
    return {
        "x": x0 / safe_w * 100,
        "y": y0 / safe_h * 100,
        "width": (x1 - x0) / safe_w * 100,
        "height": (y1 - y0) / safe_h * 100,
    }


# ──────────────────────────────────────────────────────────────
# 메인 파이프라인
# ──────────────────────────────────────────────────────────────

class ClovaOCRPipeline:
    """
    Naver Clova OCR General API 래퍼.

    process() 시그니처는 UpstageOCRPipeline과 동일하므로
    ocr_service.py에서 한 줄 교체로 사용 가능.
    """

    def __init__(self):
        from app.core.config import settings
        self.api_url: str = settings.CLOVA_OCR_API_URL
        self.secret_key: str = settings.CLOVA_OCR_SECRET_KEY
        self.structurize_model: str = settings.MODEL_NAME  # gpt-4o 또는 gpt-4o-mini

        if not self.api_url or not self.secret_key:
            raise ValueError(
                "CLOVA_OCR_API_URL 또는 CLOVA_OCR_SECRET_KEY가 설정되지 않았습니다. "
                ".env를 확인하세요."
            )

        self._openai: Optional[OpenAI] = None
        if _OPENAI_AVAILABLE:
            self._openai = OpenAI(api_key=settings.OPENAI_API_KEY)

        logger.info("ClovaOCRPipeline 초기화 완료")

    # ── Clova API 호출 ──────────────────────────────────────────

    def _call_clova(self, image_bytes: bytes, image_ext: str = "jpg") -> Dict:
        """Clova General OCR API를 호출하고 원시 응답을 반환한다."""
        b64 = base64.b64encode(image_bytes).decode("utf-8")

        payload = {
            "version": "V2",
            "requestId": str(uuid.uuid4()),
            "timestamp": int(_time.time() * 1000),
            "lang": "ko",
            "images": [
                {
                    "format": image_ext,
                    "name": "document",
                    "data": b64,
                }
            ],
            "enableTableDetection": False,
        }

        headers = {
            "Content-Type": "application/json",
            "X-OCR-SECRET": self.secret_key,
        }

        with httpx.Client(timeout=60.0) as client:
            resp = client.post(self.api_url, headers=headers, json=payload)
            resp.raise_for_status()
            return resp.json()

    # ── 응답 파싱 ───────────────────────────────────────────────

    def _parse_response(
        self, raw: Dict, image_width: int, image_height: int
    ) -> tuple[str, List[_WordBox]]:
        """
        Clova 응답에서 full_text와 WordBox 목록을 추출한다.
        lineBreak=True 인 field 뒤에 줄바꿈을 추가해 텍스트를 재현한다.
        """
        images = raw.get("images", [])
        if not images or images[0].get("inferResult") != "SUCCESS":
            msg = images[0].get("message", "unknown") if images else "응답 없음"
            raise RuntimeError(f"Clova OCR 실패: {msg}")

        fields = images[0].get("fields", [])
        text_parts: List[str] = []
        words: List[_WordBox] = []

        for f in fields:
            text = f.get("inferText", "").strip()
            confidence = float(f.get("inferConfidence", 1.0))
            vertices = f.get("boundingPoly", {}).get("vertices", [])
            line_break = f.get("lineBreak", False)

            text_parts.append(text)
            if line_break:
                text_parts.append("\n")

            if vertices and image_width and image_height:
                box = _vertices_to_box(vertices, image_width, image_height)
                words.append(
                    _WordBox(
                        text=text,
                        confidence=confidence,
                        **box,
                    )
                )

        full_text = " ".join(
            p if p == "\n" else p for p in text_parts
        ).strip()

        return full_text, words

    # ── GPT 구조화 ──────────────────────────────────────────────

    _STRUCTURIZE_SYSTEM = (
        "당신은 임대차 계약서 분석 전문가입니다. "
        "주어진 계약서 텍스트를 분석하여 핵심 정보를 JSON으로 추출하세요. "
        "필드가 없으면 빈 문자열 또는 null을 사용하세요."
    )

    _STRUCTURIZE_SCHEMA = {
        "임대인": {"성명": "", "주민번호": "", "주소": "", "연락처": ""},
        "임차인": {"성명": "", "주민번호": "", "주소": "", "연락처": ""},
        "부동산": {"소재지": "", "건물종류": "", "면적": ""},
        "계약기간": {"시작일": "", "종료일": ""},
        "보증금": {"총액": "", "계약금": "", "중도금": "", "잔금": ""},
        "월임대료": "",
        "특약사항": [],
    }

    def _structurize(self, full_text: str) -> Optional[Dict]:
        if not self._openai:
            logger.warning("OpenAI 미설치 — 구조화 생략")
            return None
        try:
            schema_hint = json.dumps(self._STRUCTURIZE_SCHEMA, ensure_ascii=False)
            resp = self._openai.chat.completions.create(
                model=self.structurize_model,
                messages=[
                    {"role": "system", "content": self._STRUCTURIZE_SYSTEM},
                    {
                        "role": "user",
                        "content": (
                            f"다음 스키마를 참고해 계약서를 JSON으로 추출하세요.\n\n"
                            f"스키마:\n{schema_hint}\n\n"
                            f"계약서:\n{full_text[:6000]}"
                        ),
                    },
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as e:
            logger.error(f"GPT 구조화 오류: {e}")
            return None

    # ── 공개 인터페이스 (UpstageOCRPipeline.process()와 동일 시그니처) ──

    def process(
        self,
        image_path: str = None,
        image_bytes: bytes = None,
        image_width: int = None,
        image_height: int = None,
        structurize: bool = True,
        enable_llm_table_fix: bool = False,
    ) -> ClovaOCRResult:
        """
        Clova OCR 파이프라인 실행.

        Args:
            image_bytes : 이미지 바이트 (image_path보다 우선)
            image_path  : 이미지 파일 경로
            image_width : 이미지 너비 px (없으면 bounding box 좌표 생략)
            image_height: 이미지 높이 px
            structurize : True면 GPT로 계약서 구조화 JSON 생성
            enable_llm_table_fix: 미사용 (Upstage 전용 파라미터, 호환성 유지용)

        Returns:
            ClovaOCRResult (ContractOCRResponse.from_result() 호환)
        """
        result = ClovaOCRResult(
            image_width=image_width or 0,
            image_height=image_height or 0,
        )

        try:
            # 이미지 바이트 준비
            if image_bytes is None:
                if image_path is None:
                    raise ValueError("image_bytes 또는 image_path 중 하나는 필수입니다.")
                with open(image_path, "rb") as f:
                    image_bytes = f.read()

            ext = "jpg"
            if image_path:
                ext = image_path.rsplit(".", 1)[-1].lower() if "." in image_path else "jpg"

            # 1. Clova OCR 호출
            logger.info("Clova OCR API 호출 시작")
            raw = self._call_clova(image_bytes, ext)

            # 2. 텍스트 + 단어 좌표 추출
            full_text, words = self._parse_response(
                raw, result.image_width, result.image_height
            )
            result.full_text = full_text
            result.markdown = full_text  # Clova는 Markdown 미지원 → full_text로 대체
            result.words = words
            logger.info(f"Clova OCR 완료: {len(words)}개 단어, {len(full_text)}자")

            # 3. GPT 구조화 (선택)
            if structurize and full_text:
                logger.info("GPT 구조화 시작")
                result.contract_data = self._structurize(full_text)

        except Exception as e:
            logger.error(f"ClovaOCRPipeline 오류: {e}")
            result.success = False
            result.error = str(e)

        return result
