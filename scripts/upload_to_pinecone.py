"""
Pinecone 데이터 업로드 스크립트 (배포 전 1회 실행)

실행 방법:
    VECTOR_DB=pinecone python scripts/upload_to_pinecone.py

업로드 대상 (namespace):
    - law_database          : 법률 문서 / 판결문
    - special_clauses_illegal: 독소 특약사항
    - special_clauses_normal : 정상 특약사항

Pinecone 인덱스 사전 생성 필요:
    - Index name: alaw-legal  (PINECONE_INDEX)
    - Dimension : 1024        (KURE-v1 임베딩 차원)
    - Metric    : cosine
    - Cloud     : aws / us-east-1 (Serverless 권장)
"""
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from app.core.config import settings
from app.rag.embedding.kure import KUREEmbeddings
from app.rag.vector_store.pinecone_adapter import PineconeAdapter
from app.rag.data_loader.special_clause_loader import parse_illegal_clauses, parse_normal_clauses
from app.rag.data_loader.jsonl_loader import load_jsonl_documents
from app.rag.data_loader.doc_loader import load_doc_dir

BATCH_SIZE = 100

DATA_DIR = Path(__file__).parent.parent / "data"

NAMESPACES = {
    "law_database": DATA_DIR / "law_database",
    "special_clauses_illegal": DATA_DIR / "특약사항" / "illegal.md",
    "special_clauses_normal": DATA_DIR / "특약사항" / "normal.md",
}


def upload_namespace(
    adapter: PineconeAdapter,
    embeddings: KUREEmbeddings,
    namespace: str,
    documents: list,
) -> int:
    """문서 리스트를 Pinecone namespace에 배치 업로드."""
    if not documents:
        logger.warning(f"[{namespace}] 문서 없음, 건너뜀")
        return 0

    total = 0
    for i in range(0, len(documents), BATCH_SIZE):
        batch = documents[i:i + BATCH_SIZE]
        texts = [doc.page_content for doc in batch]
        vectors_list = embeddings.embed_documents(texts)

        pinecone_vectors = [
            {
                "id": str(uuid.uuid4()),
                "values": vec,
                "metadata": {"content": doc.page_content, **doc.metadata},
            }
            for doc, vec in zip(batch, vectors_list)
        ]

        adapter.upsert(namespace=namespace, vectors=pinecone_vectors)
        total += len(batch)
        logger.info(f"[{namespace}] {total}/{len(documents)} 업로드 완료")

    return total


def main():
    if not settings.PINECONE_API_KEY:
        logger.error("PINECONE_API_KEY가 설정되지 않았습니다.")
        sys.exit(1)

    logger.info(f"Pinecone 인덱스: {settings.PINECONE_INDEX}")
    adapter = PineconeAdapter(
        api_key=settings.PINECONE_API_KEY,
        index_name=settings.PINECONE_INDEX,
    )
    embeddings = KUREEmbeddings()

    results = {}

    # 1. 독소 특약사항
    illegal_path = NAMESPACES["special_clauses_illegal"]
    if illegal_path.exists():
        docs = parse_illegal_clauses(illegal_path)
        results["special_clauses_illegal"] = upload_namespace(
            adapter, embeddings, "special_clauses_illegal", docs
        )
    else:
        logger.warning(f"파일 없음: {illegal_path}")

    # 2. 정상 특약사항
    normal_path = NAMESPACES["special_clauses_normal"]
    if normal_path.exists():
        docs = parse_normal_clauses(normal_path)
        results["special_clauses_normal"] = upload_namespace(
            adapter, embeddings, "special_clauses_normal", docs
        )
    else:
        logger.warning(f"파일 없음: {normal_path}")

    # 3. 법률 DB (jsonl 파일들)
    law_dir = NAMESPACES["law_database"]
    if law_dir.exists():
        all_law_docs = []
        for jsonl_file in law_dir.glob("*.jsonl"):
            docs = load_jsonl_documents(str(jsonl_file))
            all_law_docs.extend(docs)
            logger.info(f"  로드: {jsonl_file.name} ({len(docs)}개)")
        results["law_database"] = upload_namespace(
            adapter, embeddings, "law_database", all_law_docs
        )
    else:
        logger.warning(f"디렉토리 없음: {law_dir}")

    # 4. 법률 원문 .doc (주택임대차보호법, 공인중개사법 등)
    doc_law_dir = DATA_DIR / "raw" / "학습법률문서" / "법률"
    if doc_law_dir.exists():
        doc_law_docs = load_doc_dir(doc_law_dir)
        results["law_database(doc)"] = upload_namespace(
            adapter, embeddings, "law_database", doc_law_docs
        )
    else:
        logger.warning(f"디렉토리 없음: {doc_law_dir}")

    logger.info("=== 업로드 완료 ===")
    for ns, count in results.items():
        logger.info(f"  {ns}: {count}개")


if __name__ == "__main__":
    main()
