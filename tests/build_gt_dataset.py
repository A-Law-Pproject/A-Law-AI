"""평가 데이터셋 v3 빌드 스크립트

소스:
  챗봇_평가용_최종자료.xlsx  — 검증된 Q&A 302개 (주유형/세부유형/질문/상세 답변)
  lease_faq.jsonl           — 검증된 FAQ 23개 (question/related_laws[조문내용])

처리:
  1. xlsx: Pinecone 검색으로 chunk GT 매핑 (3단계 폴백)
  2. lease_faq: 조문내용 텍스트에서 chunk_id 직접 생성 (Pinecone 불필요)
  3. (선택) GPT-4o로 xlsx Q&A를 추가 증강

chunk_id = sha256(page_content[:50])[:12]

실행 예시:
    # Mock (구조 확인만, Pinecone/OpenAI 불필요)
    python tests/build_gt_dataset.py --mock --sample 10

    # xlsx GT 매핑만 (증강 없이)
    python tests/build_gt_dataset.py --min-score 0.45

    # 전체 (GPT-4o 증강 + GT 매핑, ~30분)
    python tests/build_gt_dataset.py --augment --target 150 --min-score 0.45
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

CHATBOT_XLSX    = ROOT / "tests" / "평가데이터셋" / "챗봇_평가용_최종자료.xlsx"
LEASE_FAQ_JSONL = ROOT / "tests" / "평가데이터셋" / "lease_faq.jsonl"
V3_DATASET      = ROOT / "tests" / "평가데이터셋" / "eval_dataset_v3_with_gt.json"
RESULTS_DIR     = ROOT / "results"

DEFAULT_COLLECTIONS = [
    "law_database",
    "law_statutes",
    "contracts",
    "special_clauses_illegal",
    "special_clauses_normal",
]


# ──────────────────────────────────────────────
# chunk_id 유틸
# ──────────────────────────────────────────────

def make_chunk_id(page_content: str) -> str:
    """page_content 앞 50자의 sha256 앞 12자를 chunk_id로 사용."""
    return hashlib.sha256(page_content[:50].encode("utf-8")).hexdigest()[:12]


# ──────────────────────────────────────────────
# 소스 로딩
# ──────────────────────────────────────────────

def load_chatbot_xlsx(path: Path, sample: int = 0) -> list[dict]:
    """챗봇_평가용_최종자료.xlsx 로드.

    반환 형식: [{"id", "question", "answer", "category", "sub_category"}, ...]
    """
    df = pd.read_excel(path, engine="openpyxl")
    df.columns = ["주유형", "세부유형", "question", "answer"]
    df = df.dropna(subset=["question"])
    df["주유형"]  = df["주유형"].fillna("기타")
    df["세부유형"] = df["세부유형"].fillna("-")
    df["answer"]  = df["answer"].fillna("").astype(str)

    cases = []
    for i, row in df.iterrows():
        cases.append({
            "id":           f"CHAT_{i+1:04d}",
            "question":     str(row["question"]).strip(),
            "answer":       str(row["answer"]).strip(),
            "category":     str(row["주유형"]).strip(),
            "sub_category": str(row["세부유형"]).strip(),
            "source":       "chatbot_xlsx",
            "gt_chunks":    [],
            "gt_coverage":  "none",
        })

    return cases[:sample] if sample > 0 else cases


def load_lease_faq(path: Path) -> list[dict]:
    """lease_faq.jsonl 로드.

    조문내용 텍스트에서 바로 chunk_id를 생성하므로 Pinecone 불필요.
    반환 형식: [{"id", "question", "answer", "gt_chunks", "gt_coverage"}, ...]
    """
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    cases = []
    for i, line in enumerate(lines):
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not obj.get("question"):
            continue

        gt_chunks = []
        for law in obj.get("related_laws", []):
            content = str(law.get("조문내용") or "").strip()
            if not content:
                continue
            gt_chunks.append({
                "chunk_id":             make_chunk_id(content),
                "page_content_preview": content[:100],
                "page_content_full":    content,
                "law_name":             str(law.get("법령명") or "").strip(),
                "article":              str(law.get("조문명") or "").strip(),
                "collection":           "law_statutes",
                "score":                1.0,
                "gt_source":            "faq_direct",
            })

        coverage = "faq_direct" if gt_chunks else "none"
        cases.append({
            "id":          f"FAQ_{i+1:03d}",
            "question":    str(obj["question"]).strip(),
            "answer":      str(obj.get("answer") or "").strip(),
            "category":    "FAQ",
            "sub_category": "-",
            "source":      "lease_faq",
            "gt_chunks":   gt_chunks,
            "gt_coverage": coverage,
        })

    return cases


# ──────────────────────────────────────────────
# GPT-4o 증강 (xlsx Q&A 기반)
# ──────────────────────────────────────────────

async def augment_with_gpt4o(
    existing: list[dict],
    target: int,
    llm: Any,
) -> list[dict]:
    """기존 xlsx Q&A를 참고하여 GPT-4o로 추가 질문을 생성한다.

    existing: chatbot_xlsx에서 로드된 케이스 목록
    target: 최종 목표 개수 (xlsx + faq 합산 기준)
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    needed = target - len(existing)
    if needed <= 0:
        print(f"  이미 {len(existing)}개 — 증강 불필요")
        return []

    categories = list({c["category"] for c in existing})
    examples = existing[:5]

    print(f"  {needed}개 생성 중 (카테고리: {categories[:5]})...")

    system = (
        "당신은 한국 임대차 법률 전문가이자 평가 데이터셋 설계자입니다.\n"
        "기존 Q&A를 참고하여 실용적인 임대차 법률 질문과 상세 답변을 생성하세요.\n"
        "반드시 아래 JSON 배열 형식만 출력하세요."
    )
    examples_json = json.dumps(
        [{"question": e["question"], "answer": e["answer"][:200],
          "category": e["category"]}
         for e in examples],
        ensure_ascii=False, indent=2
    )
    human = f"""기존 예시:
{examples_json}

위 예시와 같은 스타일로 {needed}개의 새 Q&A를 생성하세요.
각 항목: question (질문), answer (200자 내외 답변), category (임대차 카테고리)

JSON 배열만 출력:"""

    try:
        resp = await llm.ainvoke([
            SystemMessage(content=system),
            HumanMessage(content=human),
        ])
        content = resp.content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*\n?", "", content)
            content = re.sub(r"\n?```\s*$", "", content.strip())
        generated: list[dict] = json.loads(content)
    except Exception as e:
        print(f"  증강 실패: {e}")
        return []

    new_cases = []
    for i, item in enumerate(generated[:needed]):
        new_cases.append({
            "id":           f"AUG_{i+1:04d}",
            "question":     str(item.get("question", "")),
            "answer":       str(item.get("answer", "")),
            "category":     str(item.get("category", "증강")),
            "sub_category": "-",
            "source":       "augmented",
            "gt_chunks":    [],
            "gt_coverage":  "none",
            "_augmented":   True,
        })
    print(f"  {len(new_cases)}개 생성 완료")
    return new_cases


# ──────────────────────────────────────────────
# Chunk GT 매핑 (xlsx 케이스용, Pinecone 필요)
# ──────────────────────────────────────────────

async def map_chunk_ground_truth(
    case: dict,
    db: Any,
    embeddings: Any,
    collections: list[str],
    min_score: float = 0.45,
    top_k: int = 10,
) -> list[dict]:
    """질문+답변 기반으로 Pinecone에서 chunk GT를 검색한다.

    답변 텍스트에서 키워드를 추출해 폴백에 활용.
    3단계 폴백: (A) score≥min_score → (B) 답변 키워드 2개 이상 포함 → (C) 상위 1개
    """
    from app.rag.retriever.multi_retriever import search_multi_index

    try:
        docs = search_multi_index(
            db, embeddings, case["question"],
            collections=collections,
            k_per_collection=top_k // len(collections) + 1,
        )
    except Exception as e:
        print(f"    [GT매핑 오류] {case['id']}: {e}")
        return []

    # 답변에서 단어 추출 (3자 이상, 명사성 키워드)
    answer_words = [
        w for w in re.findall(r"[가-힣]{3,}", case.get("answer", ""))
    ][:10]

    def _build_chunk(doc: Any, gt_source: str) -> dict:
        meta    = dict(doc.metadata) if hasattr(doc, "metadata") else {}
        content = doc.page_content if hasattr(doc, "page_content") else str(doc)
        return {
            "chunk_id":             make_chunk_id(content),
            "page_content_preview": content[:100],
            "page_content_full":    content,
            "law_name":             str(meta.get("law_name") or ""),
            "article":              str(meta.get("조문명") or meta.get("article") or ""),
            "collection":           str(meta.get("collection") or ""),
            "score":                round(float(meta.get("score") or 0.0), 4),
            "gt_source":            gt_source,
        }

    # (A) score ≥ min_score
    tier_a = [
        _build_chunk(doc, "retrieval")
        for doc in docs
        if float((doc.metadata or {}).get("score") or 0.0) >= min_score
    ]
    if tier_a:
        return tier_a[:3]

    # (B) 답변 키워드 2개 이상 포함
    tier_b = []
    for doc in docs:
        content_lower = (doc.page_content if hasattr(doc, "page_content") else str(doc)).lower()
        hit = sum(1 for kw in answer_words if kw.lower() in content_lower)
        if hit >= 2:
            tier_b.append(_build_chunk(doc, "keyword_match"))
    if tier_b:
        return tier_b[:3]

    # (C) 최상위 1개
    if docs:
        return [_build_chunk(docs[0], "top1_fallback")]

    return []


# ──────────────────────────────────────────────
# Mock GT
# ──────────────────────────────────────────────

def _mock_gt_chunks(case: dict, idx: int) -> tuple[list[dict], str]:
    if case["source"] == "lease_faq":
        return case["gt_chunks"], case["gt_coverage"]

    answer_words = re.findall(r"[가-힣]{3,}", case.get("answer", ""))[:3]
    if not answer_words or idx % 5 == 0:
        return [], "none"

    fake_content = f"{case['category']} {' '.join(answer_words[:2])} 관련 조항"
    chunk = {
        "chunk_id":             make_chunk_id(fake_content),
        "page_content_preview": fake_content[:100],
        "page_content_full":    fake_content,
        "law_name":             "주택임대차보호법",
        "article":              "제1조",
        "collection":           "law_statutes",
        "score":                0.75,
        "gt_source":            "mock",
    }
    return [chunk], "mock"


# ──────────────────────────────────────────────
# 메인 빌드
# ──────────────────────────────────────────────

async def build(args: argparse.Namespace) -> None:
    # ── 소스 로딩 ─────────────────────────────────────────────────
    print(f"[로드] {CHATBOT_XLSX.name}")
    chatbot_cases = load_chatbot_xlsx(CHATBOT_XLSX, args.sample)
    print(f"  xlsx: {len(chatbot_cases)}개")

    print(f"[로드] {LEASE_FAQ_JSONL.name}")
    faq_cases = load_lease_faq(LEASE_FAQ_JSONL)
    print(f"  lease_faq: {len(faq_cases)}개 (chunk_id 직접 생성, Pinecone 불필요)")

    db = embeddings = llm = None

    if not args.mock:
        from app.core.dependencies import get_llm, get_vector_db
        from app.rag.embedding.kure import KUREEmbeddings

        db         = get_vector_db()
        embeddings = KUREEmbeddings(model_name="nlpai-lab/KURE-v1")
        llm        = get_llm()

    # ── GPT-4o 증강 (xlsx만 대상) ─────────────────────────────────
    if args.augment:
        xlsx_target = args.target - len(faq_cases)
        print(f"\n[GPT-4o 증강] xlsx {len(chatbot_cases)}개 → {xlsx_target}개 목표")
        if args.mock:
            print("  [Mock] 증강 스킵")
        elif xlsx_target > len(chatbot_cases):
            new_qs = await augment_with_gpt4o(chatbot_cases, xlsx_target, llm)
            chatbot_cases.extend(new_qs)

    # 전체 케이스 조합 (xlsx/augmented + faq)
    all_cases = chatbot_cases + faq_cases
    if args.target > 0:
        # faq는 항상 포함, xlsx는 target에서 faq 수를 뺀 만큼
        faq_count = len(faq_cases)
        xlsx_limit = max(0, args.target - faq_count)
        all_cases = chatbot_cases[:xlsx_limit] + faq_cases

    # ── Chunk GT 매핑 (xlsx/augmented 케이스만, faq는 이미 완료) ──
    needs_mapping = [c for c in all_cases if c["source"] != "lease_faq"]
    print(f"\n[Chunk GT 매핑] xlsx/augmented {len(needs_mapping)}개")
    t_start = time.perf_counter()

    coverage_stats: dict[str, int] = {
        "faq_direct": len(faq_cases), "retrieval": 0, "keyword_match": 0,
        "top1_fallback": 0, "mock": 0, "none": 0
    }

    for idx, case in enumerate(all_cases):
        if case["source"] == "lease_faq":
            coverage_stats["faq_direct"] = coverage_stats.get("faq_direct", 0)
            continue

        if args.mock:
            chunks, coverage = _mock_gt_chunks(case, idx)
        else:
            chunks = await map_chunk_ground_truth(
                case, db, embeddings,
                collections=DEFAULT_COLLECTIONS,
                min_score=args.min_score,
            )
            coverage = chunks[0]["gt_source"] if chunks else "none"

        case["gt_chunks"]   = chunks
        case["gt_coverage"] = coverage
        coverage_stats[coverage] = coverage_stats.get(coverage, 0) + 1

        if (idx + 1) % 20 == 0:
            elapsed = time.perf_counter() - t_start
            print(f"  [{idx+1}/{len(all_cases)}] {elapsed:.1f}s 경과")

    # ── 저장 ─────────────────────────────────────────────────────
    total = len(all_cases)
    gt_covered = sum(1 for c in all_cases if c["gt_chunks"])
    gt_coverage_rate = round(gt_covered / total, 4) if total > 0 else 0.0

    output = {
        "meta": {
            "version":           "3.0",
            "created":           __import__("datetime").datetime.now().strftime("%Y-%m-%d"),
            "description":       "임대차 법률 RAG 평가 데이터셋 — chunk 단위 GT 포함",
            "sources": {
                "chatbot_xlsx":  sum(1 for c in all_cases if c["source"] in ("chatbot_xlsx", "augmented")),
                "lease_faq":     sum(1 for c in all_cases if c["source"] == "lease_faq"),
            },
            "total":             total,
            "gt_coverage_rate":  gt_coverage_rate,
            "gt_coverage_by_source": coverage_stats,
            "min_score_used":    args.min_score,
        },
        "questions": all_cases,
    }

    out_path = args.out or V3_DATASET
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[완료] {out_path}")
    print(f"  총 {total}개 (xlsx/aug: {output['meta']['sources']['chatbot_xlsx']}, faq: {output['meta']['sources']['lease_faq']})")
    print(f"  GT 커버리지: {gt_coverage_rate:.1%}")
    print(f"  소스별: {coverage_stats}")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="평가 데이터셋 v3 빌드 (chunk GT 매핑)")
    p.add_argument("--augment",   action="store_true",
                   help="GPT-4o로 xlsx Q&A 증강 (없으면 xlsx 원본에만 GT 매핑)")
    p.add_argument("--target",    type=int, default=0,
                   help="목표 케이스 수 (0=전체 xlsx+faq)")
    p.add_argument("--sample",    type=int, default=0,
                   help="xlsx에서 처음 N개만 사용 (0=전체)")
    p.add_argument("--min-score", type=float, default=0.45, dest="min_score",
                   help="chunk GT 포함 최소 유사도 점수 (기본값: 0.45)")
    p.add_argument("--mock",      action="store_true",
                   help="인프라 없이 구조 확인 (Pinecone/OpenAI 불필요)")
    p.add_argument("--out",       type=Path, default=None,
                   help=f"출력 경로 (기본값: {V3_DATASET})")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(build(parse_args()))
