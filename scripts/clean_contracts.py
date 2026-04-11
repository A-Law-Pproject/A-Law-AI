# -*- coding: utf-8 -*-
"""contracts namespace 정리 스크립트
- 상업용부동산임대계약서 삭제
- 날짜/서명 등 껍데기 청크 삭제
- content 30자 미만 짧은 청크 삭제
"""
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import settings
from pinecone import Pinecone

pc = Pinecone(api_key=settings.PINECONE_API_KEY)
idx = pc.Index(settings.PINECONE_INDEX)
NS = "contracts"

# ── 시작 전 현황 ──────────────────────────────────────
stats_before = idx.describe_index_stats()
before_count = stats_before.namespaces.get(NS)
print(f"[시작] contracts 벡터 수: {before_count.vector_count if before_count else 0}개\n")

# ── STEP 1: 전체 ID 수집 ──────────────────────────────
print("[STEP 1] 전체 ID 수집 중...")
all_ids = []
for id_batch in idx.list(namespace=NS):
    all_ids.extend(id_batch)
print(f"  총 {len(all_ids)}개\n")

# ── STEP 2: 배치 fetch → 분류 ─────────────────────────
COMMERCIAL_TYPE = "상업용부동산임대계약서"
SHELL_TITLES = {"날짜", "서명", "주소", "성명", "인", "날인", "확인"}
MIN_CONTENT_LEN = 30

delete_ids = set()
keep_count = 0
BATCH = 100

print("[STEP 2] 메타데이터 분석 및 삭제 대상 분류 중...")
for i in range(0, len(all_ids), BATCH):
    batch = all_ids[i : i + BATCH]
    resp = idx.fetch(ids=batch, namespace=NS)
    for vid, vec in resp.vectors.items():
        meta = vec.metadata or {}
        doc_type = meta.get("document_type", "")
        title = meta.get("title", "")
        content = meta.get("content", "").strip()

        reason = None
        if doc_type == COMMERCIAL_TYPE:
            reason = "상업용계약서"
        elif title in SHELL_TITLES:
            reason = f"껍데기title({title})"
        elif len(content) < MIN_CONTENT_LEN:
            reason = f"짧은content({len(content)}자)"

        if reason:
            delete_ids.add(vid)
        else:
            keep_count += 1

    if (i // BATCH) % 10 == 0:
        print(f"  진행: {min(i+BATCH, len(all_ids))}/{len(all_ids)}")

print(f"\n  삭제 대상: {len(delete_ids)}개")
print(f"  유지 대상: {keep_count}개\n")

# ── STEP 3: ID 기반 삭제 ──────────────────────────────
print("[STEP 3] 삭제 실행 중...")
delete_list = list(delete_ids)
for i in range(0, len(delete_list), 1000):
    idx.delete(ids=delete_list[i : i + 1000], namespace=NS)
    print(f"  삭제 완료: {min(i+1000, len(delete_list))}/{len(delete_list)}")

# ── 최종 확인 ─────────────────────────────────────────
import time
time.sleep(2)
stats_after = idx.describe_index_stats()
after = stats_after.namespaces.get(NS)
after_count = after.vector_count if after else 0

print(f"\n=== 완료 ===")
print(f"  삭제 전: {before_count.vector_count if before_count else 0}개")
print(f"  삭제 수: {len(delete_ids)}개")
print(f"  삭제 후: {after_count}개")
