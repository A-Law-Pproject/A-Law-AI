"""
데이터 합성 실행 스크립트

사용법:
    python -m data_augmentation.run --pdf data/계약서양식/법원/부동산_임대차_계약서.pdf --output data/augmented --count 100 --toxic-ratio 0.3
"""
import argparse
from pathlib import Path
from loguru import logger

from .contract_synthesizer import generate_batch


def main():
    parser = argparse.ArgumentParser(description="계약서 합성 데이터 생성")
    parser.add_argument(
        "--pdf",
        type=str,
        default="data/계약서양식/법원/부동산_임대차_계약서.pdf",
        help="양식 PDF 경로"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/augmented",
        help="출력 디렉토리"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="생성할 이미지 수"
    )
    parser.add_argument(
        "--toxic-ratio",
        type=float,
        default=0.3,
        help="독소조항 포함 계약서 비율 (0.0 ~ 1.0, 기본값: 0.3 = 30%%)"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="이미지 해상도 (DPI)"
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="미리보기 모드 (1개만 생성)"
    )

    args = parser.parse_args()

    # PDF 확인
    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        logger.error(f"PDF 파일이 없습니다: {pdf_path}")
        return 1

    # 미리보기 모드
    if args.preview:
        args.count = 1
        args.output = str(Path(args.output) / "preview")

    logger.info(f"=== 계약서 합성 데이터 생성 ===")
    logger.info(f"PDF: {args.pdf}")
    logger.info(f"출력: {args.output}")
    logger.info(f"개수: {args.count}")
    logger.info(f"독소조항 비율: {args.toxic_ratio * 100:.0f}%")
    logger.info(f"DPI: {args.dpi}")

    # 생성
    results = generate_batch(
        pdf_path=str(pdf_path),
        output_dir=args.output,
        count=args.count,
        toxic_ratio=args.toxic_ratio,
        dpi=args.dpi
    )

    logger.info(f"\n=== 완료 ===")
    logger.info(f"생성된 파일: {len(results)}개")
    logger.info(f"출력 디렉토리: {args.output}")

    # 샘플 출력
    if results:
        img_path, json_path = results[0]
        logger.info(f"\n샘플 파일:")
        logger.info(f"  이미지: {img_path}")
        logger.info(f"  라벨: {json_path}")

    return 0


if __name__ == "__main__":
    exit(main())
