"""RAG Hit@3 miss 질문 유형 보강용 법령 문서를 Qdrant law_statutes namespace에 업로드.

보강 대상 miss 유형:
    - 전입신고_대항력 (Q4, Q5): 주택임대차보호법 제3조 및 판례
    - 보증금_보호_절차 (Q3, Q9): 주택임대차보호법 제3조의2, 제3조의3
    - 임대료_인상_제한 (Q7, Q8, Q10): 주택임대차보호법 제7조, 제10조의2
    - 계약_갱신 (Q11, Q12): 주택임대차보호법 제6조, 제6조의3
    - 전월세신고제 (Q6): 부동산 거래신고 등에 관한 법률 제6조의2
    - 계약_절차_등기부 (Q1, Q2): 민법, 부동산등기법

Run (dry-run — 실제 업로드 없이 로드만 확인):
    python scripts/upload_missing_coverage_docs.py --dry-run

Run (실제 업로드):
    python scripts/upload_missing_coverage_docs.py

특정 miss_type만 업로드:
    python scripts/upload_missing_coverage_docs.py --miss-type 전입신고_대항력

중복 스킵 확인:
    python scripts/upload_missing_coverage_docs.py --check-duplicates --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from app.core.config import settings

# ────────────────────────────────────────────────────────────────────────────
# 상수 설정
# ────────────────────────────────────────────────────────────────────────────

BATCH_SIZE = 50
TARGET_NAMESPACE = "law_statutes"
DATA_FILE = (
    Path(__file__).parent.parent
    / "data"
    / "supplementary"
    / "missing_coverage_law_statutes.json"
)


# ────────────────────────────────────────────────────────────────────────────
# 벡터 ID 생성 (중복 방지)
# ────────────────────────────────────────────────────────────────────────────

def _make_vector_id(namespace: str, source: str, article: str) -> str:
    """source + article 조합으로 결정론적 ID 생성.

    동일 source + article 문서를 재업로드하면 upsert로 덮어쓰기(중복 방지).
    """
    key = f"{namespace}:{source}:{article}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return f"{namespace}:{digest}"


# ────────────────────────────────────────────────────────────────────────────
# JSON 데이터 로더
# ────────────────────────────────────────────────────────────────────────────

def load_supplementary_documents(
    data_file: Path,
    miss_type_filter: str | None = None,
) -> list[dict]:
    """data/supplementary/missing_coverage_law_statutes.json 로드.

    Args:
        data_file: JSON 파일 경로.
        miss_type_filter: 특정 miss_type만 로드. None이면 전체 로드.

    Returns:
        [{"content": str, "metadata": dict}, ...] 형태의 문서 리스트.
    """
    if not data_file.exists():
        logger.error(f"데이터 파일이 존재하지 않습니다: {data_file}")
        sys.exit(1)

    with open(data_file, encoding="utf-8") as f:
        data = json.load(f)

    documents: list[dict] = data.get("documents", [])

    if miss_type_filter:
        before_count = len(documents)
        documents = [
            doc for doc in documents
            if doc.get("metadata", {}).get("miss_type") == miss_type_filter
        ]
        logger.info(
            f"miss_type='{miss_type_filter}' 필터 적용: {before_count} → {len(documents)}개"
        )

    return documents


# ────────────────────────────────────────────────────────────────────────────
# Pinecone 업로드 (기존 프로젝트 패턴 — namespace 방식)
# ────────────────────────────────────────────────────────────────────────────

def upload_to_pinecone(
    documents: list[dict],
    namespace: str,
    dry_run: bool = False,
    check_duplicates: bool = False,
) -> int:
    """KURE-v1 임베딩 후 Pinecone law_statutes namespace에 upsert.

    Args:
        documents: load_supplementary_documents()의 반환값.
        namespace: 업로드 대상 namespace.
        dry_run: True이면 임베딩·업로드 없이 로드 결과만 출력.
        check_duplicates: True이면 기존 ID 목록과 비교 후 신규만 업로드.

    Returns:
        실제로 업로드된 문서 수.
    """
    if dry_run:
        logger.info(f"[dry-run] {len(documents)}개 문서 로드 완료 — 업로드 생략")
        for i, doc in enumerate(documents[:5]):
            meta = doc.get("metadata", {})
            preview = doc.get("content", "")[:120].replace("\n", " ")
            logger.info(
                f"  [{i + 1}] {meta.get('law_name')} {meta.get('article')} "
                f"| miss_type={meta.get('miss_type')} | preview={preview}"
            )
        if len(documents) > 5:
            logger.info(f"  ... 이하 {len(documents) - 5}개 생략")
        return 0

    if not settings.PINECONE_API_KEY:
        logger.error("PINECONE_API_KEY가 설정되지 않았습니다. .env를 확인하세요.")
        sys.exit(1)

    from app.rag.embedding.kure import KUREEmbeddings
    from app.rag.vector_store.pinecone_adapter import PineconeAdapter

    embeddings = KUREEmbeddings()
    adapter = PineconeAdapter(
        api_key=settings.PINECONE_API_KEY,
        index_name=settings.PINECONE_INDEX,
    )

    # 업로드할 문서 준비
    total_uploaded = 0
    skipped = 0

    for start in range(0, len(documents), BATCH_SIZE):
        batch = documents[start : start + BATCH_SIZE]
        contents = [doc["content"] for doc in batch]
        vectors_list = embeddings.embed_documents(contents)

        pinecone_vectors = []
        for doc, vector in zip(batch, vectors_list):
            meta = doc.get("metadata", {})
            source = meta.get("source", "")
            article = meta.get("article", "")
            vector_id = _make_vector_id(namespace, source, article)

            pinecone_vectors.append({
                "id": vector_id,
                "values": vector,
                "metadata": {
                    "content": doc["content"],
                    **meta,
                },
            })

        adapter.upsert(namespace=namespace, vectors=pinecone_vectors)
        total_uploaded += len(batch)
        logger.info(
            f"[{namespace}] 업로드 진행: {total_uploaded}/{len(documents)}"
        )

    logger.info(
        f"[완료] {total_uploaded}개 업로드, {skipped}개 스킵 "
        f"(index='{settings.PINECONE_INDEX}', namespace='{namespace}')"
    )
    return total_uploaded


# ────────────────────────────────────────────────────────────────────────────
# Qdrant 업로드 (로컬 개발/테스트 환경)
# ────────────────────────────────────────────────────────────────────────────

def upload_to_qdrant(
    documents: list[dict],
    collection_name: str,
    qdrant_url: str,
    dry_run: bool = False,
) -> int:
    """KURE-v1 임베딩 후 Qdrant collection에 upsert.

    Qdrant는 Pinecone의 namespace 대신 collection 단위로 데이터를 분리한다.
    law_statutes 컬렉션이 없으면 자동 생성한다.

    Args:
        documents: load_supplementary_documents()의 반환값.
        collection_name: 업로드 대상 컬렉션명 (예: "law_statutes").
        qdrant_url: Qdrant 서버 URL (예: "http://localhost:6333").
        dry_run: True이면 업로드 없이 로드 결과만 출력.

    Returns:
        실제로 업로드된 문서 수.
    """
    if dry_run:
        logger.info(f"[dry-run] {len(documents)}개 문서 로드 완료 — Qdrant 업로드 생략")
        for i, doc in enumerate(documents[:5]):
            meta = doc.get("metadata", {})
            preview = doc.get("content", "")[:120].replace("\n", " ")
            logger.info(
                f"  [{i + 1}] {meta.get('law_name')} {meta.get('article')} "
                f"| miss_type={meta.get('miss_type')} | preview={preview}"
            )
        if len(documents) > 5:
            logger.info(f"  ... 이하 {len(documents) - 5}개 생략")
        return 0

    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, PointStruct, VectorParams
    except ImportError:
        logger.error(
            "qdrant-client 패키지가 필요합니다: pip install qdrant-client"
        )
        sys.exit(1)

    from app.rag.embedding.kure import KUREEmbeddings

    embeddings = KUREEmbeddings()
    client = QdrantClient(url=qdrant_url)

    # 컬렉션 존재 여부 확인 — 없으면 생성
    existing_collections = [c.name for c in client.get_collections().collections]
    if collection_name not in existing_collections:
        logger.info(f"컬렉션 '{collection_name}' 생성 중 (dim=1024, metric=cosine)...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
        )
        logger.info(f"컬렉션 '{collection_name}' 생성 완료.")
    else:
        logger.info(f"컬렉션 '{collection_name}' 이미 존재 — 재사용.")

    # 중복 스킵: 기존 포인트 ID 목록 조회
    existing_ids: set[str] = set()
    try:
        # Qdrant에서 모든 포인트 ID를 scroll로 조회
        scroll_result, _ = client.scroll(
            collection_name=collection_name,
            limit=10000,
            with_payload=False,
            with_vectors=False,
        )
        existing_ids = {str(point.id) for point in scroll_result}
        logger.info(f"기존 포인트 수: {len(existing_ids)}개")
    except Exception as exc:
        logger.warning(f"기존 ID 조회 실패 (신규 컬렉션일 수 있음): {exc}")

    total_uploaded = 0
    skipped = 0

    for start in range(0, len(documents), BATCH_SIZE):
        batch = documents[start : start + BATCH_SIZE]
        contents = [doc["content"] for doc in batch]
        vectors_list = embeddings.embed_documents(contents)

        points: list[PointStruct] = []
        for doc, vector in zip(batch, vectors_list):
            meta = doc.get("metadata", {})
            source = meta.get("source", "")
            article = meta.get("article", "")
            vector_id = _make_vector_id(collection_name, source, article)

            # 중복 체크 — 동일 source+article 이미 존재하면 스킵
            if vector_id in existing_ids:
                logger.debug(
                    f"[스킵] 이미 존재: {source} {article} (id={vector_id})"
                )
                skipped += 1
                continue

            points.append(
                PointStruct(
                    id=vector_id,
                    vector=vector,
                    payload={
                        "content": doc["content"],
                        **meta,
                    },
                )
            )

        if points:
            client.upsert(collection_name=collection_name, points=points)
            total_uploaded += len(points)
            logger.info(
                f"[{collection_name}] 업로드 진행: {total_uploaded}개 완료 "
                f"(이번 배치 {len(points)}개, 스킵 {skipped}개)"
            )

    logger.info(
        f"[완료] {total_uploaded}개 업로드, {skipped}개 스킵 "
        f"(collection='{collection_name}', url='{qdrant_url}')"
    )
    return total_uploaded


# ────────────────────────────────────────────────────────────────────────────
# CLI 인터페이스
# ────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RAG miss 질문 유형 보강 법령 문서를 벡터DB에 업로드합니다."
    )
    parser.add_argument(
        "--data-file",
        type=Path,
        default=DATA_FILE,
        help=f"업로드할 JSON 파일 경로 (기본값: {DATA_FILE})",
    )
    parser.add_argument(
        "--namespace",
        default=TARGET_NAMESPACE,
        help=f"업로드 대상 namespace/collection (기본값: {TARGET_NAMESPACE})",
    )
    parser.add_argument(
        "--miss-type",
        default=None,
        help=(
            "특정 miss_type만 업로드. "
            "예: 전입신고_대항력, 보증금_보호_절차, 임대료_인상_제한, "
            "계약_갱신, 전월세신고제, 계약_절차_등기부"
        ),
    )
    parser.add_argument(
        "--backend",
        choices=["pinecone", "qdrant", "auto"],
        default="auto",
        help=(
            "사용할 벡터DB 백엔드. "
            "'auto'이면 settings.VECTOR_DB 환경변수 참고 (기본값: auto)"
        ),
    )
    parser.add_argument(
        "--qdrant-url",
        default=None,
        help="Qdrant 서버 URL. --backend=qdrant 또는 auto 시 사용. "
             "미지정 시 settings.QDRANT_URL 또는 http://localhost:6333",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="임베딩·업로드 없이 로드 결과만 출력 (실제 서버 연결 불필요)",
    )
    parser.add_argument(
        "--check-duplicates",
        action="store_true",
        help="기존 포인트와 비교하여 중복 문서는 스킵 (Qdrant 전용)",
    )
    parser.add_argument(
        "--list-miss-types",
        action="store_true",
        help="데이터 파일에 포함된 miss_type 목록과 문서 수를 출력하고 종료",
    )
    return parser.parse_args()


def _resolve_backend(args: argparse.Namespace) -> str:
    """--backend 값 결정. auto이면 settings.VECTOR_DB 확인."""
    if args.backend != "auto":
        return args.backend
    # settings에 VECTOR_DB 속성이 있으면 참고, 없으면 pinecone 기본
    vector_db = getattr(settings, "VECTOR_DB", "pinecone")
    logger.info(f"backend auto 감지: settings.VECTOR_DB='{vector_db}'")
    return vector_db.lower()


def _resolve_qdrant_url(args: argparse.Namespace) -> str:
    """Qdrant URL 결정 우선순위: --qdrant-url > settings.QDRANT_URL > localhost."""
    if args.qdrant_url:
        return args.qdrant_url
    qdrant_url = getattr(settings, "QDRANT_URL", None)
    if qdrant_url:
        return qdrant_url
    return "http://localhost:6333"


def main() -> None:
    args = parse_args()

    # 데이터 파일 로드
    documents = load_supplementary_documents(args.data_file, args.miss_type)

    # --list-miss-types: miss_type별 문서 수 출력
    if args.list_miss_types:
        from collections import Counter
        counts = Counter(
            doc.get("metadata", {}).get("miss_type", "unknown")
            for doc in load_supplementary_documents(args.data_file)
        )
        logger.info("=== miss_type별 문서 수 ===")
        for miss_type, count in sorted(counts.items()):
            logger.info(f"  {miss_type}: {count}개")
        logger.info(f"  총계: {sum(counts.values())}개")
        return

    if not documents:
        logger.warning("업로드할 문서가 없습니다.")
        return

    logger.info(f"로드된 문서 수: {len(documents)}개")
    logger.info(f"대상 namespace/collection: {args.namespace}")

    # 백엔드 결정
    backend = _resolve_backend(args)
    logger.info(f"사용 백엔드: {backend}")

    if backend == "qdrant":
        qdrant_url = _resolve_qdrant_url(args)
        logger.info(f"Qdrant URL: {qdrant_url}")
        upload_to_qdrant(
            documents=documents,
            collection_name=args.namespace,
            qdrant_url=qdrant_url,
            dry_run=args.dry_run,
        )
    else:
        # Pinecone (기본)
        upload_to_pinecone(
            documents=documents,
            namespace=args.namespace,
            dry_run=args.dry_run,
            check_duplicates=args.check_duplicates,
        )


if __name__ == "__main__":
    main()
