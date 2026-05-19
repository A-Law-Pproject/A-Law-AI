"""Pinecone 코퍼스 export 스크립트 (코퍼스 전체 BM25용, 오프라인 1회 실행).

Pinecone에 색인된 문서를 namespace별로 내려받아 로컬 JSONL 아티팩트로 저장한다.
이 아티팩트가 런타임 코퍼스 전체 BM25 인덱스의 입력이 된다.
Pinecone이 "실제 검색 가능한 코퍼스"의 단일 진실원천이므로, data/raw 원본을
재구성하는 방식(업로드 스크립트가 stale → drift 위험)보다 정합성이 보장된다.

실행:
    VECTOR_DB=pinecone python scripts/export_pinecone_corpus.py

출력:
    data/bm25_corpus/{namespace}.jsonl
    각 라인: {"content": str, "metadata": {...}}  (content 제외한 나머지 메타)
    → PineconeAdapter.search()의 Document 복원 방식과 동일하게 맞춰 _document_key 정합
"""
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from app.core.config import settings

# 코퍼스 전체 BM25 대상 namespace (multi_retriever._HYBRID_TARGET_COLLECTIONS와 일치)
TARGET_NAMESPACES = [
    "law_database",
    "law_statutes",
    "special_clauses_illegal",
    "special_clauses_normal",
]

FETCH_BATCH = 100


def export_namespace(index, namespace: str, out_dir: Path) -> int:
    """단일 namespace의 모든 문서를 JSONL로 내보낸다."""
    out_path = out_dir / f"{namespace}.jsonl"
    written = 0

    with out_path.open("w", encoding="utf-8") as fp:
        id_batch: list[str] = []

        def flush(ids: list[str]) -> int:
            if not ids:
                return 0
            resp = index.fetch(ids=ids, namespace=namespace)
            vectors = getattr(resp, "vectors", None) or {}
            count = 0
            for vid, record in vectors.items():
                meta = dict(getattr(record, "metadata", None) or {})
                content = meta.pop("content", "")
                if not content:
                    continue
                # Pinecone vector id를 메타에 보존 → PineconeAdapter.search()가
                # metadata["id"]=match.id로 세팅하므로, 코퍼스 BM25 결과와 dense
                # 결과의 _document_key가 정합되어 RRF 융합이 올바르게 동작한다.
                meta.setdefault("id", vid)
                fp.write(json.dumps({"content": content, "metadata": meta}, ensure_ascii=False) + "\n")
                count += 1
            return count

        try:
            for ids in index.list(namespace=namespace):
                # index.list는 ID 배치(list[str])를 yield
                id_batch.extend(ids if isinstance(ids, list) else [ids])
                while len(id_batch) >= FETCH_BATCH:
                    written += flush(id_batch[:FETCH_BATCH])
                    id_batch = id_batch[FETCH_BATCH:]
            written += flush(id_batch)
        except Exception as exc:
            logger.warning(f"[{namespace}] list/fetch 중 예외: {exc}")

    logger.info(f"[{namespace}] {written}개 저장 → {out_path}")
    return written


def main() -> None:
    if not settings.PINECONE_API_KEY:
        logger.error("PINECONE_API_KEY가 설정되지 않았습니다.")
        sys.exit(1)

    try:
        from pinecone import Pinecone
    except ImportError:
        logger.error("pinecone 패키지가 필요합니다: pip install pinecone")
        sys.exit(1)

    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    index = pc.Index(settings.PINECONE_INDEX)
    logger.info(f"Pinecone 인덱스: {settings.PINECONE_INDEX}")

    try:
        stats = index.describe_index_stats()
        ns_stats = getattr(stats, "namespaces", None) or {}
        logger.info(f"namespace 통계: { {k: getattr(v, 'vector_count', v) for k, v in ns_stats.items()} }")
    except Exception as exc:
        logger.warning(f"describe_index_stats 실패(무시): {exc}")

    out_dir = Path(settings.BM25_CORPUS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    totals: dict[str, int] = {}
    for ns in TARGET_NAMESPACES:
        totals[ns] = export_namespace(index, ns, out_dir)

    logger.info("=== export 완료 ===")
    for ns, count in totals.items():
        logger.info(f"  {ns}: {count}개")
    if sum(totals.values()) == 0:
        logger.warning("export된 문서가 0개입니다. namespace 이름/인덱스를 확인하세요.")


if __name__ == "__main__":
    main()
