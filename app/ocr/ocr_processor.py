# ============================================================
# [LEGACY] 이 파일은 더 이상 사용되지 않습니다.
# PaddleOCR/EasyOCR/TrOCR 기반 로컬 OCR로, Upstage OCR로 대체되었습니다.
# TODO: 안정화 후 삭제 예정
# ============================================================
"""
OCR 처리 모듈 (LEGACY - Upstage OCR로 대체됨)
- 3가지 OCR 방식 지원:
  1. UpstageOCR: Upstage Document Parse API
  2. CellBasedOCR: 셀 분할 + 전처리 + PaddleOCR
  3. EnsembleOCR: TrOCR + PaddleOCR + EasyOCR 앙상블 (99%+ 정확도 목표)
"""
import cv2
import numpy as np
import base64
import httpx
import os
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
from loguru import logger

# 내부 모듈 import
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing.image_cleaner import ImageCleaner
from preprocessing.table_detector import TableDetector
from preprocessing.cell_extractor import CellExtractor


# ===========================================
# 공통 데이터 클래스
# ===========================================

@dataclass
class OCRResult:
    """OCR 결과 데이터 클래스"""
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    points: List[Tuple[int, int]]  # 4점 좌표

    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "confidence": round(self.confidence, 4),
            "x": self.bbox[0],
            "y": self.bbox[1],
            "width": self.bbox[2],
            "height": self.bbox[3],
            "points": self.points
        }


@dataclass
class OCRResponse:
    """OCR 응답 전체 구조"""
    results: List[OCRResult] = field(default_factory=list)
    full_text: str = ""
    avg_confidence: float = 0.0
    method: str = ""  # 사용된 OCR 방식

    def to_dict(self) -> Dict:
        return {
            "results": [r.to_dict() for r in self.results],
            "full_text": self.full_text,
            "avg_confidence": round(self.avg_confidence, 4),
            "total_boxes": len(self.results),
            "method": self.method
        }


# ===========================================
# OCR 프로세서 기본 클래스
# ===========================================

class BaseOCRProcessor(ABC):
    """OCR 프로세서 기본 클래스"""

    def __init__(self, min_confidence: float = 0.6):
        self.min_confidence = min_confidence

    @abstractmethod
    def process(self, image: np.ndarray, **kwargs) -> OCRResponse:
        """OCR 수행"""
        pass

    def _combine_text(self, results: List[OCRResult]) -> str:
        """OCR 결과를 자연스러운 텍스트로 조합"""
        if not results:
            return ""

        lines = []
        current_line = []
        last_y = None
        line_threshold = 20

        for r in results:
            if last_y is None or abs(r.bbox[1] - last_y) > line_threshold:
                if current_line:
                    current_line.sort(key=lambda x: x.bbox[0])
                    lines.append(" ".join(x.text for x in current_line))
                current_line = [r]
            else:
                current_line.append(r)
            last_y = r.bbox[1]

        if current_line:
            current_line.sort(key=lambda x: x.bbox[0])
            lines.append(" ".join(x.text for x in current_line))

        return "\n".join(lines)

    def _points_to_bbox(self, points: List) -> Tuple[int, int, int, int]:
        """4점 좌표를 bbox로 변환"""
        pts = np.array(points)
        x_min = int(np.min(pts[:, 0]))
        y_min = int(np.min(pts[:, 1]))
        x_max = int(np.max(pts[:, 0]))
        y_max = int(np.max(pts[:, 1]))
        return (x_min, y_min, x_max - x_min, y_max - y_min)


# ===========================================
# 1. Upstage Document Parse OCR
# ===========================================

class UpstageOCR(BaseOCRProcessor):
    """Upstage Document Parse API 기반 OCR"""

    def __init__(self, min_confidence: float = 0.6):
        super().__init__(min_confidence)

        self.api_key = os.getenv("UPSTAGE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "UPSTAGE_API_KEY 환경 변수가 설정되지 않았습니다. "
                "https://console.upstage.ai 에서 API 키를 발급받아 설정하세요."
            )

        self.api_url = "https://api.upstage.ai/v1/document-ai/document-parse"
        logger.info("UpstageOCR 초기화 완료")

    def process(self, image: np.ndarray, **kwargs) -> OCRResponse:
        """Upstage API로 OCR 수행"""
        try:
            _, buffer = cv2.imencode('.png', image)
            image_bytes = buffer.tobytes()

            headers = {"Authorization": f"Bearer {self.api_key}"}
            files = {"document": ("image.png", image_bytes, "image/png")}
            data = {"ocr": "force"}

            with httpx.Client(timeout=60.0) as client:
                response = client.post(self.api_url, headers=headers, files=files, data=data)
                response.raise_for_status()
                result = response.json()

            return self._parse_response(result)

        except Exception as e:
            logger.error(f"Upstage OCR 오류: {e}")
            return OCRResponse(method="upstage")

    def _parse_response(self, response_data: Dict) -> OCRResponse:
        """Upstage API 응답 파싱"""
        ocr_results = []
        confidences = []

        for element in response_data.get("elements", []):
            text = element.get("text", "").strip()
            if not text:
                continue

            confidence = element.get("confidence", 0.95)
            if confidence < self.min_confidence:
                continue

            bounding_box = element.get("bounding_box", {})
            vertices = bounding_box.get("vertices", [])

            if len(vertices) >= 4:
                points = [(int(v.get("x", 0)), int(v.get("y", 0))) for v in vertices]
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]
                bbox = (min(x_coords), min(y_coords),
                        max(x_coords) - min(x_coords), max(y_coords) - min(y_coords))
            else:
                bbox = (0, 0, 0, 0)
                points = [(0, 0)] * 4

            ocr_results.append(OCRResult(text=text, confidence=confidence, bbox=bbox, points=points))
            confidences.append(confidence)

        ocr_results.sort(key=lambda r: (r.bbox[1], r.bbox[0]))

        full_text = response_data.get("content", {}).get("text", "")
        if not full_text:
            full_text = self._combine_text(ocr_results)

        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

        return OCRResponse(
            results=ocr_results,
            full_text=full_text,
            avg_confidence=avg_conf,
            method="upstage"
        )


# ===========================================
# 2. 셀 분할 + 전처리 OCR (PaddleOCR)
# ===========================================

class CellBasedOCR(BaseOCRProcessor):
    """
    셀 분할 + 전처리 + PaddleOCR

    기존 preprocessing 모듈을 활용:
    - ImageCleaner: 도장 제거, 기울기 보정, 대비 향상
    - TableDetector: 표 검출, 셀 좌표 추출
    - CellExtractor: 셀별 이미지 추출
    """

    def __init__(
        self,
        min_confidence: float = 0.6,
        use_gpu: bool = False,
        lang: str = "korean"
    ):
        super().__init__(min_confidence)

        try:
            from paddleocr import PaddleOCR
            self.ocr = PaddleOCR(
                use_angle_cls=True,
                lang=lang,
                use_gpu=use_gpu,
                show_log=False,
                det_db_thresh=0.2,
                det_db_box_thresh=0.3,
                det_db_unclip_ratio=1.8,
                rec_batch_num=16,
            )
            logger.info("CellBasedOCR 초기화 완료 (PaddleOCR)")
        except ImportError:
            raise ImportError("PaddleOCR 미설치. pip install paddleocr paddlepaddle")

        self.image_cleaner = ImageCleaner()
        self.table_detector = TableDetector()
        self.cell_extractor = CellExtractor(padding=5)

    def process(
        self,
        image: np.ndarray,
        preprocess: bool = True,
        use_cell_detection: bool = True,
        **kwargs
    ) -> OCRResponse:
        """
        셀 분할 OCR 수행

        Args:
            image: 입력 이미지
            preprocess: 전처리 수행 여부
            use_cell_detection: 셀 검출 사용 여부
        """
        # 1. 전처리
        if preprocess:
            processed = self.image_cleaner.process(
                image,
                remove_stamp=True,
                deskew=True,
                denoise=False,  # 속도 위해 비활성화
                enhance=True
            )
        else:
            processed = image

        all_results = []

        # 2. 셀 검출 및 셀별 OCR
        if use_cell_detection:
            table = self.table_detector.detect(processed)

            if table and table.cells:
                logger.info(f"표 검출: {len(table.cells)}개 셀")

                for cell in table.cells:
                    # 셀 이미지 추출
                    extracted = self.cell_extractor.extract_cell(processed, cell)
                    if not extracted:
                        continue

                    # 셀 OCR
                    cell_results = self._ocr_image(extracted.image)

                    # 좌표를 원본 이미지 기준으로 변환
                    for r in cell_results:
                        r.bbox = (
                            r.bbox[0] + cell.x,
                            r.bbox[1] + cell.y,
                            r.bbox[2],
                            r.bbox[3]
                        )
                        r.points = [(p[0] + cell.x, p[1] + cell.y) for p in r.points]
                        all_results.append(r)

        # 3. 전체 이미지 OCR (셀 검출 실패 시 또는 보완용)
        if not all_results:
            logger.info("셀 검출 실패, 전체 이미지 OCR 수행")
            all_results = self._ocr_image(processed)

        # 4. 저신뢰도 영역 재시도
        all_results = self._retry_low_confidence(processed, all_results)

        # 결과 정렬
        all_results.sort(key=lambda r: (r.bbox[1], r.bbox[0]))

        # 텍스트 조합
        full_text = self._combine_text(all_results)
        confidences = [r.confidence for r in all_results]
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

        return OCRResponse(
            results=all_results,
            full_text=full_text,
            avg_confidence=avg_conf,
            method="cell_based_paddle"
        )

    def _ocr_image(self, image: np.ndarray) -> List[OCRResult]:
        """단일 이미지 OCR"""
        results = []

        try:
            ocr_result = self.ocr.ocr(image, cls=True)

            if not ocr_result or not ocr_result[0]:
                return results

            for line in ocr_result[0]:
                if not line or len(line) < 2:
                    continue

                points = line[0]
                text = line[1][0]
                confidence = line[1][1]

                if confidence < self.min_confidence:
                    continue

                int_points = [(int(p[0]), int(p[1])) for p in points]
                bbox = self._points_to_bbox(points)

                results.append(OCRResult(
                    text=text,
                    confidence=confidence,
                    bbox=bbox,
                    points=int_points
                ))

        except Exception as e:
            logger.error(f"PaddleOCR 오류: {e}")

        return results

    def _retry_low_confidence(
        self,
        image: np.ndarray,
        results: List[OCRResult],
        threshold: float = 0.7
    ) -> List[OCRResult]:
        """저신뢰도 영역 고해상도 재시도"""
        updated = []

        for r in results:
            if r.confidence >= threshold:
                updated.append(r)
                continue

            # 영역 크롭 + 패딩
            x, y, w, h = r.bbox
            padding = 10
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(image.shape[1], x + w + padding)
            y2 = min(image.shape[0], y + h + padding)

            roi = image[y1:y2, x1:x2]

            # 2배 확대
            enlarged = cv2.resize(roi, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

            # 재시도
            try:
                retry_result = self.ocr.ocr(enlarged, cls=True)
                if retry_result and retry_result[0]:
                    best = max(retry_result[0], key=lambda x: x[1][1] if x else 0)
                    if best and len(best) >= 2:
                        new_text = best[1][0]
                        new_conf = best[1][1]

                        if new_conf > r.confidence:
                            r.text = new_text
                            r.confidence = new_conf
                            logger.debug(f"재시도 성공: {r.confidence:.2f}")
            except:
                pass

            updated.append(r)

        return updated


# ===========================================
# 3. 앙상블 OCR (TrOCR + PaddleOCR + EasyOCR)
# ===========================================

class EnsembleOCR(BaseOCRProcessor):
    """
    고정밀 앙상블 OCR (99%+ 정확도 목표)

    3개 OCR 엔진 결과를 투표/병합:
    1. PaddleOCR - 빠르고 정확한 기본 엔진
    2. EasyOCR - 다양한 폰트 지원
    3. TrOCR (Transformer OCR) - 최신 딥러닝 모델

    전략:
    - 각 엔진의 결과를 IoU 기반으로 매칭
    - 다수결 투표 또는 신뢰도 가중 평균
    - 불일치 시 TrOCR 결과 우선 (가장 정확)
    """

    def __init__(
        self,
        min_confidence: float = 0.5,
        use_gpu: bool = False,
        use_trocr: bool = True,
        use_easyocr: bool = True
    ):
        super().__init__(min_confidence)

        self.engines = {}
        self.use_gpu = use_gpu

        # 1. PaddleOCR (필수)
        try:
            from paddleocr import PaddleOCR
            self.engines["paddle"] = PaddleOCR(
                use_angle_cls=True,
                lang="korean",
                use_gpu=use_gpu,
                show_log=False,
                det_db_thresh=0.2,
                det_db_box_thresh=0.3,
            )
            logger.info("PaddleOCR 로드 완료")
        except ImportError:
            raise ImportError("PaddleOCR 필수. pip install paddleocr paddlepaddle")

        # 2. EasyOCR (선택)
        if use_easyocr:
            try:
                import easyocr
                self.engines["easy"] = easyocr.Reader(
                    ['ko', 'en'],
                    gpu=use_gpu,
                    verbose=False
                )
                logger.info("EasyOCR 로드 완료")
            except ImportError:
                logger.warning("EasyOCR 미설치. pip install easyocr")

        # 3. TrOCR (선택, 가장 정확)
        if use_trocr:
            try:
                from transformers import TrOCRProcessor, VisionEncoderDecoderModel
                import torch

                # 한국어 TrOCR 또는 다국어 모델
                model_name = "microsoft/trocr-base-handwritten"  # 또는 한국어 파인튜닝 모델

                self.trocr_processor = TrOCRProcessor.from_pretrained(model_name)
                self.trocr_model = VisionEncoderDecoderModel.from_pretrained(model_name)

                if use_gpu and torch.cuda.is_available():
                    self.trocr_model = self.trocr_model.cuda()

                self.engines["trocr"] = True
                logger.info("TrOCR 로드 완료")
            except ImportError:
                logger.warning("TrOCR 미설치. pip install transformers torch")
            except Exception as e:
                logger.warning(f"TrOCR 로드 실패: {e}")

        # 전처리기
        self.image_cleaner = ImageCleaner()

        logger.info(f"EnsembleOCR 초기화 완료. 엔진: {list(self.engines.keys())}")

    def process(
        self,
        image: np.ndarray,
        preprocess: bool = True,
        voting_strategy: str = "confidence_weighted",
        **kwargs
    ) -> OCRResponse:
        """
        앙상블 OCR 수행

        Args:
            image: 입력 이미지
            preprocess: 전처리 여부
            voting_strategy: "majority" | "confidence_weighted" | "trocr_priority"
        """
        # 전처리
        if preprocess:
            processed = self.image_cleaner.process(
                image, remove_stamp=True, deskew=True, denoise=False, enhance=True
            )
        else:
            processed = image

        # 각 엔진 결과 수집
        engine_results = {}

        # PaddleOCR
        if "paddle" in self.engines:
            engine_results["paddle"] = self._run_paddle(processed)

        # EasyOCR
        if "easy" in self.engines:
            engine_results["easy"] = self._run_easyocr(processed)

        # TrOCR (텍스트 영역별로 실행)
        if "trocr" in self.engines:
            # PaddleOCR에서 검출된 영역에 TrOCR 적용
            paddle_results = engine_results.get("paddle", [])
            engine_results["trocr"] = self._run_trocr(processed, paddle_results)

        # 결과 앙상블
        final_results = self._ensemble_results(engine_results, voting_strategy)

        # 정렬
        final_results.sort(key=lambda r: (r.bbox[1], r.bbox[0]))

        # 텍스트 조합
        full_text = self._combine_text(final_results)
        confidences = [r.confidence for r in final_results]
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

        return OCRResponse(
            results=final_results,
            full_text=full_text,
            avg_confidence=avg_conf,
            method="ensemble"
        )

    def _run_paddle(self, image: np.ndarray) -> List[OCRResult]:
        """PaddleOCR 실행"""
        results = []
        try:
            ocr_result = self.engines["paddle"].ocr(image, cls=True)
            if ocr_result and ocr_result[0]:
                for line in ocr_result[0]:
                    if line and len(line) >= 2:
                        points = line[0]
                        text = line[1][0]
                        confidence = line[1][1]

                        int_points = [(int(p[0]), int(p[1])) for p in points]
                        bbox = self._points_to_bbox(points)

                        results.append(OCRResult(
                            text=text, confidence=confidence,
                            bbox=bbox, points=int_points
                        ))
        except Exception as e:
            logger.error(f"PaddleOCR 오류: {e}")
        return results

    def _run_easyocr(self, image: np.ndarray) -> List[OCRResult]:
        """EasyOCR 실행"""
        results = []
        try:
            # EasyOCR는 RGB 이미지 기대
            if len(image.shape) == 3 and image.shape[2] == 3:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = image

            ocr_result = self.engines["easy"].readtext(rgb_image)

            for detection in ocr_result:
                points = detection[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                text = detection[1]
                confidence = detection[2]

                int_points = [(int(p[0]), int(p[1])) for p in points]
                bbox = self._points_to_bbox(points)

                results.append(OCRResult(
                    text=text, confidence=confidence,
                    bbox=bbox, points=int_points
                ))
        except Exception as e:
            logger.error(f"EasyOCR 오류: {e}")
        return results

    def _run_trocr(
        self,
        image: np.ndarray,
        detected_regions: List[OCRResult]
    ) -> List[OCRResult]:
        """TrOCR 실행 (검출된 영역별)"""
        results = []

        if not detected_regions:
            return results

        try:
            import torch
            from PIL import Image

            for region in detected_regions:
                x, y, w, h = region.bbox

                # 영역 크롭
                x1 = max(0, x - 5)
                y1 = max(0, y - 5)
                x2 = min(image.shape[1], x + w + 5)
                y2 = min(image.shape[0], y + h + 5)

                roi = image[y1:y2, x1:x2]

                if roi.size == 0:
                    continue

                # RGB 변환 및 PIL 이미지로
                if len(roi.shape) == 3:
                    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                else:
                    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)

                pil_image = Image.fromarray(roi_rgb)

                # TrOCR 추론
                pixel_values = self.trocr_processor(
                    pil_image, return_tensors="pt"
                ).pixel_values

                if self.use_gpu and torch.cuda.is_available():
                    pixel_values = pixel_values.cuda()

                generated_ids = self.trocr_model.generate(pixel_values)
                text = self.trocr_processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )[0]

                # TrOCR는 신뢰도를 직접 제공하지 않음 - 높은 기본값 사용
                confidence = 0.95

                results.append(OCRResult(
                    text=text.strip(),
                    confidence=confidence,
                    bbox=region.bbox,
                    points=region.points
                ))

        except Exception as e:
            logger.error(f"TrOCR 오류: {e}")

        return results

    def _ensemble_results(
        self,
        engine_results: Dict[str, List[OCRResult]],
        strategy: str
    ) -> List[OCRResult]:
        """
        앙상블 결과 병합

        전략:
        - majority: 다수결 투표
        - confidence_weighted: 신뢰도 가중 평균
        - trocr_priority: TrOCR 결과 우선
        """
        # 모든 결과 수집
        all_results = []
        for engine, results in engine_results.items():
            for r in results:
                all_results.append((engine, r))

        if not all_results:
            return []

        # IoU 기반 그룹화
        groups = self._group_by_iou(all_results)

        # 그룹별 최종 결과 결정
        final_results = []

        for group in groups:
            if len(group) == 1:
                final_results.append(group[0][1])
                continue

            if strategy == "trocr_priority":
                # TrOCR 결과 우선
                trocr_result = next((r for e, r in group if e == "trocr"), None)
                if trocr_result:
                    final_results.append(trocr_result)
                else:
                    # 가장 높은 신뢰도 선택
                    best = max(group, key=lambda x: x[1].confidence)
                    final_results.append(best[1])

            elif strategy == "confidence_weighted":
                # 신뢰도 가중 투표
                texts = {}
                for engine, r in group:
                    if r.text not in texts:
                        texts[r.text] = {"score": 0, "result": r}
                    texts[r.text]["score"] += r.confidence

                best_text = max(texts.items(), key=lambda x: x[1]["score"])
                final_results.append(best_text[1]["result"])

            else:  # majority
                # 다수결
                from collections import Counter
                text_counts = Counter(r.text for _, r in group)
                most_common = text_counts.most_common(1)[0][0]

                # 해당 텍스트 중 가장 높은 신뢰도 선택
                best = max(
                    (r for _, r in group if r.text == most_common),
                    key=lambda x: x.confidence
                )
                final_results.append(best)

        return final_results

    def _group_by_iou(
        self,
        results: List[Tuple[str, OCRResult]],
        threshold: float = 0.5
    ) -> List[List[Tuple[str, OCRResult]]]:
        """IoU 기반 결과 그룹화"""
        if not results:
            return []

        used = set()
        groups = []

        for i, (engine1, r1) in enumerate(results):
            if i in used:
                continue

            group = [(engine1, r1)]
            used.add(i)

            for j, (engine2, r2) in enumerate(results[i+1:], i+1):
                if j in used:
                    continue

                if self._calculate_iou(r1.bbox, r2.bbox) > threshold:
                    group.append((engine2, r2))
                    used.add(j)

            groups.append(group)

        return groups

    def _calculate_iou(self, box1: Tuple, box2: Tuple) -> float:
        """IoU 계산"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)

        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0

        inter = (xi2 - xi1) * (yi2 - yi1)
        union = w1 * h1 + w2 * h2 - inter

        return inter / union if union > 0 else 0.0


# ===========================================
# 통합 OCR 프로세서 (호환성 유지)
# ===========================================

class OCRProcessor(CellBasedOCR):
    """
    기본 OCR 프로세서 (호환성 유지)

    기존 코드와 호환되는 인터페이스 제공.
    내부적으로 CellBasedOCR 사용.
    """

    def __init__(
        self,
        lang: str = "korean",
        use_gpu: bool = False,
        min_confidence: float = 0.6,
        **kwargs
    ):
        super().__init__(
            min_confidence=min_confidence,
            use_gpu=use_gpu,
            lang=lang
        )


class HighPrecisionOCR(EnsembleOCR):
    """
    고정밀 OCR 프로세서 (호환성 유지)

    기존 코드와 호환되는 인터페이스 제공.
    내부적으로 EnsembleOCR 사용.
    """

    def __init__(
        self,
        lang: str = "korean",
        use_gpu: bool = False,
        min_confidence: float = 0.5,
        **kwargs
    ):
        # TrOCR, EasyOCR 사용 가능 여부 확인
        use_trocr = kwargs.get("use_trocr", True)
        use_easyocr = kwargs.get("use_easyocr", True)

        try:
            super().__init__(
                min_confidence=min_confidence,
                use_gpu=use_gpu,
                use_trocr=use_trocr,
                use_easyocr=use_easyocr
            )
        except Exception as e:
            logger.warning(f"EnsembleOCR 초기화 실패, CellBasedOCR로 대체: {e}")
            # Fallback to CellBasedOCR
            self.__class__ = CellBasedOCR
            CellBasedOCR.__init__(self, min_confidence=min_confidence, use_gpu=use_gpu, lang=lang)

    def process_multi_scale(
        self,
        image: np.ndarray,
        scales: List[float] = [1.0, 1.5]
    ) -> OCRResponse:
        """다중 스케일 OCR (호환성)"""
        # 앙상블이 이미 고정밀이므로 단일 스케일로 충분
        return self.process(image)


# ===========================================
# OCR 비교 유틸리티
# ===========================================

class OCRComparator:
    """OCR 방식 비교 유틸리티"""

    def __init__(self):
        self.processors = {}

    def add_processor(self, name: str, processor: BaseOCRProcessor):
        """OCR 프로세서 추가"""
        self.processors[name] = processor

    def compare(
        self,
        image: np.ndarray,
        ground_truth: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        여러 OCR 방식 비교

        Args:
            image: 입력 이미지
            ground_truth: 정답 텍스트 (정확도 계산용)

        Returns:
            각 방식별 결과 및 비교 분석
        """
        results = {}

        for name, processor in self.processors.items():
            import time
            start = time.time()

            try:
                response = processor.process(image)
                elapsed = time.time() - start

                results[name] = {
                    "full_text": response.full_text,
                    "confidence": response.avg_confidence,
                    "num_boxes": len(response.results),
                    "time_sec": round(elapsed, 3),
                    "success": True
                }

                # 정확도 계산 (ground truth 있는 경우)
                if ground_truth:
                    results[name]["accuracy"] = self._calculate_accuracy(
                        response.full_text, ground_truth
                    )

            except Exception as e:
                results[name] = {
                    "success": False,
                    "error": str(e)
                }

        return results

    def _calculate_accuracy(self, predicted: str, ground_truth: str) -> float:
        """문자 단위 정확도 계산"""
        from difflib import SequenceMatcher

        # 공백 제거 후 비교
        pred_clean = predicted.replace(" ", "").replace("\n", "")
        gt_clean = ground_truth.replace(" ", "").replace("\n", "")

        ratio = SequenceMatcher(None, pred_clean, gt_clean).ratio()
        return round(ratio * 100, 2)


# ===========================================
# 편의 함수
# ===========================================

def create_ocr_processor(
    method: str = "cell_based",
    **kwargs
) -> BaseOCRProcessor:
    """
    OCR 프로세서 생성

    Args:
        method: "upstage" | "cell_based" | "ensemble"
        **kwargs: 프로세서별 설정

    Returns:
        BaseOCRProcessor 인스턴스
    """
    if method == "upstage":
        return UpstageOCR(**kwargs)
    elif method == "ensemble":
        return EnsembleOCR(**kwargs)
    else:  # cell_based (기본)
        return CellBasedOCR(**kwargs)


if __name__ == "__main__":
    print("OCR 프로세서 모듈")
    print("\n사용 가능한 OCR 방식:")
    print("1. UpstageOCR - Upstage Document Parse API")
    print("2. CellBasedOCR - 셀 분할 + 전처리 + PaddleOCR")
    print("3. EnsembleOCR - TrOCR + PaddleOCR + EasyOCR 앙상블")
    print("\n사용 예시:")
    print("  from ocr_engine.ocr_processor import create_ocr_processor")
    print("  processor = create_ocr_processor('ensemble')")
    print("  result = processor.process(image)")
