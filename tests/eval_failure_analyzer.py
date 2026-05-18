"""RAG 평가 실패 케이스 3-tier 분류 및 수정 제안

입력:
  - 평가 결과 JSON (ragas_qna_eval_*.json 또는 eval_suite_*.json)
  - v3 데이터셋 (eval_dataset_v3_with_gt.json, gt_chunks 포함)

3-tier 분류:
  retrieval_miss  GT chunk가 top-5 검색 결과에 없음
  rerank_miss     GT chunk가 top-5에 있지만 top-3에 없음 (reranker가 밀어냄)
  llm_miss        GT chunk가 top-3에 있지만 답변이 부정확

수정 제안 (action):
  rechunk          청킹 수정 대상 (score < 0.3)
  lower_threshold  컬렉션별 임계값 조정 (score 0.3~0.45)
  improve_reranker rerank_top_n 증가 또는 모델 교체
  improve_prompt   LLM 프롬프트 강화
  add_metadata     chunk law_name/article 메타데이터 보강

실행 예시:
    # Mock (구조 확인)
    python tests/eval_failure_analyzer.py --mock

    # 실제 분석
    python tests/eval_failure_analyzer.py \\
        --eval-result results/ragas_qna_eval_20260518_123456.json \\
        --dataset tests/평가데이터셋/eval_dataset_v3_with_gt.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

V3_DATASET  = ROOT / "tests" / "평가데이터셋" / "eval_dataset_v3_with_gt.json"
RESULTS_DIR = ROOT / "results"

FailureType = Literal["retrieval_miss", "rerank_miss", "llm_miss", "correct"]

# 컬렉션별 현재 임계값 (multi_retriever.py 기준)
CURRENT_THRESHOLDS: dict[str, float] = {
    "law_database":           0.15,
    "law_statutes":           0.20,
    "contracts":              0.30,
    "special_clauses_illegal": 0.45,
    "special_clauses_normal": 0.40,
}


# ──────────────────────────────────────────────
# chunk_id 유틸
# ──────────────────────────────────────────────

def make_chunk_id(page_content: str) -> str:
    return hashlib.sha256(page_content[:50].encode("utf-8")).hexdigest()[:12]


# ──────────────────────────────────────────────
# 데이터 로딩
# ──────────────────────────────────────────────

def load_eval_result(path: Path) -> dict:
    """평가 결과 JSON 로드.

    ragas_qna_eval_*.json 또는 eval_suite_*.json 모두 지원.
    반환 구조: {"rows": [...], "faq_rows": [...], "hr_k_results": {...}, ...}
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    # eval_suite가 래핑한 경우 풀기
    if "ragas_result" in data:
        data = data["ragas_result"]
    return data


def load_v3_dataset(path: Path) -> dict[str, dict]:
    """v3 데이터셋을 {question_id: question_obj} 매핑으로 로드."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {q["id"]: q for q in raw.get("questions", [])}


def _match_eval_to_gt(
    eval_rows: list[dict],
    gt_map: dict[str, dict],
) -> list[tuple[dict, dict | None]]:
    """eval rows와 GT 데이터셋을 question 텍스트로 매칭.

    반환: [(eval_row, gt_obj | None), ...]
    """
    gt_by_question = {v["question"].strip(): v for v in gt_map.values()}
    result = []
    for row in eval_rows:
        q = str(row.get("question") or "").strip()
        gt = gt_by_question.get(q)
        result.append((row, gt))
    return result


# ──────────────────────────────────────────────
# 검색 결과에서 chunk_id 추출
# ──────────────────────────────────────────────

def _get_retrieved_chunk_ids(row: dict) -> list[str]:
    """eval row에서 검색된 chunk_id 목록을 추출.

    retrieved_contexts (텍스트 목록) 또는 cited_laws (메타데이터)에서 추출.
    """
    chunk_ids = []

    # retrieved_contexts: 텍스트 목록
    for ctx in row.get("retrieved_contexts", []):
        if isinstance(ctx, str) and ctx.strip():
            chunk_ids.append(make_chunk_id(ctx))

    # cited_laws: 법령 메타데이터 (텍스트 재구성)
    if not chunk_ids:
        for law in row.get("cited_laws", []):
            if isinstance(law, dict):
                fake = f"{law.get('law_name', '')} {law.get('article', '')}"
                if fake.strip():
                    chunk_ids.append(make_chunk_id(fake))

    return chunk_ids


# ──────────────────────────────────────────────
# 3-tier 실패 분류
# ──────────────────────────────────────────────

def classify_failure(
    row: dict,
    gt_obj: dict | None,
    threshold_context_recall: float = 0.5,
    threshold_faithfulness:   float = 0.7,
    top_k: int = 5,
    top_n_rerank: int = 3,
) -> tuple[FailureType, dict]:
    """eval row + GT 객체를 받아 실패 유형과 진단 정보를 반환.

    GT가 없으면 키워드 기반 폴백으로 분류.
    """
    retrieved_ids = _get_retrieved_chunk_ids(row)
    context_recall  = float(row.get("context_recall")  or row.get("non_llm_context_recall") or 0.0)
    faithfulness    = float(row.get("faithfulness")     or row.get("answer_correctness") or 0.0)
    answer_correct  = float(row.get("answer_correctness") or 0.0)

    # GT chunk_id 집합
    gt_chunk_ids: set[str] = set()
    if gt_obj and gt_obj.get("gt_chunks"):
        gt_chunk_ids = {c["chunk_id"] for c in gt_obj["gt_chunks"]}

    # GT 없을 때: 키워드 기반 폴백
    if not gt_chunk_ids and gt_obj:
        keywords_lower = [k.lower() for k in gt_obj.get("expected_keywords", [])]
        for ctx in row.get("retrieved_contexts", [])[:top_k]:
            ctx_lower = ctx.lower() if isinstance(ctx, str) else ""
            hit = sum(1 for kw in keywords_lower if kw in ctx_lower)
            if hit >= 2:
                gt_chunk_ids.add(make_chunk_id(ctx))

    # GT 자체가 없으면 RAGAS 점수만으로 판단
    if not gt_chunk_ids:
        if context_recall >= threshold_context_recall and faithfulness >= threshold_faithfulness:
            return "correct", {"reason": "no_gt_but_scores_ok", "context_recall": context_recall, "faithfulness": faithfulness}
        if faithfulness < threshold_faithfulness:
            return "llm_miss", {"reason": "low_faithfulness_no_gt", "faithfulness": faithfulness}
        return "retrieval_miss", {"reason": "no_gt_low_recall", "context_recall": context_recall}

    top5_ids = set(retrieved_ids[:top_k])
    top3_ids = set(retrieved_ids[:top_n_rerank])
    gt_in_top5 = bool(gt_chunk_ids & top5_ids)
    gt_in_top3 = bool(gt_chunk_ids & top3_ids)

    # 1순위 GT의 score 추출 (재순위 판단용)
    first_gt_rank: int | None = None
    for rank, cid in enumerate(retrieved_ids, 1):
        if cid in gt_chunk_ids:
            first_gt_rank = rank
            break

    diag = {
        "gt_chunk_ids":    list(gt_chunk_ids),
        "top5_ids":        list(top5_ids),
        "first_gt_rank":   first_gt_rank,
        "context_recall":  context_recall,
        "faithfulness":    faithfulness,
        "answer_correctness": answer_correct,
    }

    if not gt_in_top5:
        diag["reason"] = "gt_not_in_top5"
        return "retrieval_miss", diag

    if gt_in_top5 and not gt_in_top3:
        diag["reason"] = "gt_in_top5_but_reranked_out"
        return "rerank_miss", diag

    # GT가 top-3에 있는 경우
    if context_recall < threshold_context_recall:
        diag["reason"] = "gt_in_top3_but_low_context_recall"
        return "retrieval_miss", diag

    if faithfulness < threshold_faithfulness:
        diag["reason"] = "gt_in_top3_context_ok_but_low_faithfulness"
        return "llm_miss", diag

    diag["reason"] = "all_ok"
    return "correct", diag


# ──────────────────────────────────────────────
# 수정 제안
# ──────────────────────────────────────────────

def suggest_fix(
    failure_type: FailureType,
    row: dict,
    gt_obj: dict | None,
    diag: dict,
) -> dict:
    """실패 유형에 따라 구체적인 수정 액션을 반환."""
    q_id  = str(gt_obj.get("id", "unknown")) if gt_obj else "unknown"
    q_text = str(row.get("question", ""))

    # GT chunk의 평균 score (retrieval 단계)
    gt_score = 0.0
    if gt_obj and gt_obj.get("gt_chunks"):
        scores = [c.get("score", 0.0) for c in gt_obj["gt_chunks"]]
        gt_score = sum(scores) / len(scores) if scores else 0.0

    # 주요 컬렉션 파악
    gt_collection = ""
    if gt_obj and gt_obj.get("gt_chunks"):
        gt_collection = gt_obj["gt_chunks"][0].get("collection", "")

    target_chunk_id = (gt_obj["gt_chunks"][0]["chunk_id"]
                       if gt_obj and gt_obj.get("gt_chunks") else None)

    if failure_type == "retrieval_miss":
        if gt_score < 0.3:
            return {
                "failure_type":    failure_type,
                "question_id":     q_id,
                "question":        q_text[:80],
                "action":          "rechunk",
                "target_chunk_id": target_chunk_id,
                "target_law":      gt_obj["gt_chunks"][0].get("law_name", "") if target_chunk_id else "",
                "target_collection": gt_collection,
                "gt_score":        gt_score,
                "suggestion":      (
                    f"GT chunk의 유사도 점수({gt_score:.3f})가 너무 낮습니다. "
                    f"'{gt_collection}' 컬렉션의 청킹 단위를 검토하세요. "
                    "조문 단위보다 더 세밀하게 분리하거나 overlap을 늘리는 것을 고려하세요."
                ),
                "priority":        "high",
            }
        else:
            current_thresh = CURRENT_THRESHOLDS.get(gt_collection, 0.3)
            suggested_thresh = round(max(0.1, gt_score - 0.05), 2)
            return {
                "failure_type":    failure_type,
                "question_id":     q_id,
                "question":        q_text[:80],
                "action":          "lower_threshold",
                "target_chunk_id": target_chunk_id,
                "target_law":      gt_obj["gt_chunks"][0].get("law_name", "") if target_chunk_id else "",
                "target_collection": gt_collection,
                "gt_score":        gt_score,
                "current_threshold": current_thresh,
                "suggested_threshold": suggested_thresh,
                "suggestion":      (
                    f"GT chunk(score={gt_score:.3f})이 현재 임계값({current_thresh}) 아래입니다. "
                    f"'{gt_collection}' 컬렉션의 임계값을 {current_thresh} → {suggested_thresh}로 낮추세요. "
                    "multi_retriever.py의 SCORE_THRESHOLDS 딕셔너리를 수정하세요."
                ),
                "priority":        "high" if gt_score < 0.4 else "medium",
            }

    if failure_type == "rerank_miss":
        first_rank = diag.get("first_gt_rank") or 4
        return {
            "failure_type":    failure_type,
            "question_id":     q_id,
            "question":        q_text[:80],
            "action":          "improve_reranker",
            "target_chunk_id": target_chunk_id,
            "target_law":      gt_obj["gt_chunks"][0].get("law_name", "") if target_chunk_id else "",
            "target_collection": gt_collection,
            "first_gt_rank":   first_rank,
            "suggestion":      (
                f"GT chunk가 벡터 검색 {first_rank}위에 있지만 reranker가 top-3 밖으로 밀어냈습니다. "
                "rerank_top_n을 5→8로 늘리거나, "
                "bge-reranker-v2-m3 → bge-reranker-large로 업그레이드를 고려하세요. "
                "chain.py의 rerank_top_n 파라미터를 조정하세요."
            ),
            "priority":        "medium",
        }

    if failure_type == "llm_miss":
        faithfulness = diag.get("faithfulness", 0.0)
        if faithfulness < 0.5:
            return {
                "failure_type":    failure_type,
                "question_id":     q_id,
                "question":        q_text[:80],
                "action":          "improve_prompt",
                "target_chunk_id": target_chunk_id,
                "target_law":      gt_obj["gt_chunks"][0].get("law_name", "") if target_chunk_id else "",
                "faithfulness":    faithfulness,
                "suggestion":      (
                    f"컨텍스트는 충분하지만 faithfulness={faithfulness:.3f}로 LLM 환각이 의심됩니다. "
                    "prompts.py의 CONTRACT_QA_PROMPT에 "
                    "'제공된 문서에 없는 내용은 절대 생성하지 마세요' 강화 지시를 추가하세요. "
                    "temperature를 0으로 고정하는 것도 검토하세요."
                ),
                "priority":        "high",
            }
        else:
            return {
                "failure_type":    failure_type,
                "question_id":     q_id,
                "question":        q_text[:80],
                "action":          "add_metadata",
                "target_chunk_id": target_chunk_id,
                "target_law":      gt_obj["gt_chunks"][0].get("law_name", "") if target_chunk_id else "",
                "target_collection": gt_collection,
                "faithfulness":    faithfulness,
                "suggestion":      (
                    f"컨텍스트와 faithfulness({faithfulness:.3f})는 양호하지만 answer_correctness가 낮습니다. "
                    f"'{gt_collection}' 컬렉션의 chunk에 law_name/article 메타데이터가 "
                    "정확히 붙어있는지 확인하세요. "
                    "Pinecone 재인덱싱 시 메타데이터 필드를 보강하세요."
                ),
                "priority":        "medium",
            }

    # correct
    return {
        "failure_type":    "correct",
        "question_id":     q_id,
        "question":        q_text[:80],
        "action":          "none",
        "suggestion":      "정상 작동",
        "priority":        "low",
    }


# ──────────────────────────────────────────────
# 집계 리포트
# ──────────────────────────────────────────────

def generate_fix_report(
    suggestions: list[dict],
    total: int,
    output_path: Path,
) -> dict:
    """실패 분류 결과를 집계하여 JSON 리포트를 생성한다."""
    type_counts: dict[str, int] = {
        "retrieval_miss": 0, "rerank_miss": 0, "llm_miss": 0, "correct": 0
    }
    action_counts: dict[str, int] = {}
    rechunk_targets: dict[str, dict] = {}
    threshold_adjustments: dict[str, dict] = {}
    by_difficulty: dict[str, dict] = {}

    for s in suggestions:
        ft = s["failure_type"]
        type_counts[ft] = type_counts.get(ft, 0) + 1
        action = s.get("action", "none")
        action_counts[action] = action_counts.get(action, 0) + 1

        if action == "rechunk" and s.get("target_chunk_id"):
            cid = s["target_chunk_id"]
            if cid not in rechunk_targets:
                rechunk_targets[cid] = {
                    "chunk_id":         cid,
                    "law_name":         s.get("target_law", ""),
                    "collection":       s.get("target_collection", ""),
                    "failed_question_count": 0,
                    "questions":        [],
                }
            rechunk_targets[cid]["failed_question_count"] += 1
            rechunk_targets[cid]["questions"].append(s["question_id"])

        if action == "lower_threshold" and s.get("target_collection"):
            coll = s["target_collection"]
            if coll not in threshold_adjustments:
                threshold_adjustments[coll] = {
                    "collection":    coll,
                    "current":       s.get("current_threshold", "?"),
                    "suggested_min": s.get("suggested_threshold", "?"),
                    "case_count":    0,
                }
            threshold_adjustments[coll]["case_count"] += 1
            # 제안값 중 최솟값을 사용
            prev = threshold_adjustments[coll]["suggested_min"]
            curr = s.get("suggested_threshold", prev)
            if isinstance(curr, float) and isinstance(prev, float):
                threshold_adjustments[coll]["suggested_min"] = min(prev, curr)

    high_priority = sorted(
        [s for s in suggestions if s.get("priority") == "high"],
        key=lambda x: x["failure_type"],
    )[:10]

    report = {
        "generated_at": datetime.now().isoformat(),
        "summary": {
            "total_evaluated": total,
            "total_failed":    total - type_counts.get("correct", 0),
            **type_counts,
        },
        "action_distribution":   action_counts,
        "high_priority_fixes":   high_priority,
        "rechunk_targets":        sorted(
            rechunk_targets.values(),
            key=lambda x: x["failed_question_count"],
            reverse=True,
        ),
        "threshold_adjustments":  list(threshold_adjustments.values()),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


# ──────────────────────────────────────────────
# Mock 실행
# ──────────────────────────────────────────────

def _run_mock() -> None:
    """Mock 데이터로 구조 확인."""
    mock_gt = {
        "Q001": {
            "id": "Q001", "question": "보증금을 돌려받을 수 없을 때 어떻게 하나요?",
            "difficulty": "easy", "expected_keywords": ["임차권등기", "보증금", "반환"],
            "gt_chunks": [{"chunk_id": "abc123def456", "law_name": "주택임대차보호법",
                           "collection": "law_statutes", "score": 0.82, "gt_source": "retrieval"}],
        },
        "Q002": {
            "id": "Q002", "question": "묵시적 갱신이란 무엇인가요?",
            "difficulty": "medium", "expected_keywords": ["묵시적", "갱신", "2년"],
            "gt_chunks": [{"chunk_id": "xyz789uvw012", "law_name": "주택임대차보호법",
                           "collection": "law_database", "score": 0.25, "gt_source": "keyword_match"}],
        },
        "Q003": {
            "id": "Q003", "question": "계약 갱신 거절 사유는 무엇인가요?",
            "difficulty": "hard", "expected_keywords": ["갱신거절", "정당한 사유"],
            "gt_chunks": [],
        },
    }

    mock_rows = [
        {"question": "보증금을 돌려받을 수 없을 때 어떻게 하나요?",
         "retrieved_contexts": ["임차권등기명령 신청 가능 abc123def456 dummy"],
         "context_recall": 0.8, "faithfulness": 0.85, "answer_correctness": 0.75},
        {"question": "묵시적 갱신이란 무엇인가요?",
         "retrieved_contexts": ["일반 계약 안내 사항"],
         "context_recall": 0.2, "faithfulness": 0.3, "answer_correctness": 0.2},
        {"question": "계약 갱신 거절 사유는 무엇인가요?",
         "retrieved_contexts": [],
         "context_recall": 0.0, "faithfulness": 0.0, "answer_correctness": 0.0},
    ]

    matched = _match_eval_to_gt(mock_rows, mock_gt)
    suggestions = []
    for row, gt in matched:
        ft, diag = classify_failure(row, gt)
        s = suggest_fix(ft, row, gt, diag)
        suggestions.append(s)
        print(f"  [{s['question_id']}] {ft} → {s['action']} | {s['suggestion'][:60]}...")

    out = RESULTS_DIR / f"failure_analysis_mock_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report = generate_fix_report(suggestions, len(mock_rows), out)
    print(f"\n[Mock 리포트] {out}")
    print(f"  요약: {report['summary']}")


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    if args.mock:
        print("[Mock 모드] 구조 확인 실행")
        _run_mock()
        return

    # 평가 결과 로드
    if not args.eval_result or not args.eval_result.exists():
        # 가장 최근 파일 자동 탐색
        candidates = sorted(RESULTS_DIR.glob("ragas_qna_eval_*.json"), reverse=True)
        if not candidates:
            candidates = sorted(RESULTS_DIR.glob("eval_suite_*.json"), reverse=True)
        if not candidates:
            print("[오류] 평가 결과 파일을 찾을 수 없습니다. --eval-result 로 경로를 지정하세요.")
            sys.exit(1)
        eval_path = candidates[0]
        print(f"[자동 탐색] 가장 최근 평가 결과: {eval_path}")
    else:
        eval_path = args.eval_result

    dataset_path = args.dataset or V3_DATASET
    if not dataset_path.exists():
        print(f"[오류] v3 데이터셋 없음: {dataset_path}")
        print("먼저 python tests/build_gt_dataset.py 를 실행하세요.")
        sys.exit(1)

    print(f"[로드] 평가 결과: {eval_path}")
    eval_data = load_eval_result(eval_path)
    rows = eval_data.get("rows") or eval_data.get("cases") or []

    print(f"[로드] v3 데이터셋: {dataset_path}")
    gt_map = load_v3_dataset(dataset_path)

    print(f"\n[분류] 총 {len(rows)}개 케이스 분석 중...")
    matched = _match_eval_to_gt(rows, gt_map)

    suggestions = []
    type_counts: dict[str, int] = {"retrieval_miss": 0, "rerank_miss": 0, "llm_miss": 0, "correct": 0}

    for i, (row, gt) in enumerate(matched):
        ft, diag = classify_failure(row, gt)
        s = suggest_fix(ft, row, gt, diag)
        suggestions.append(s)
        type_counts[ft] = type_counts.get(ft, 0) + 1

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(matched)}] 진행 중...")

    # 상위 N개만 출력
    top_n = args.top_n
    high = [s for s in suggestions if s["priority"] == "high"]
    print(f"\n[결과] 실패 분류:")
    for k, v in type_counts.items():
        pct = v / len(rows) * 100 if rows else 0
        print(f"  {k}: {v}개 ({pct:.1f}%)")

    print(f"\n[High Priority 상위 {min(top_n, len(high))}개]")
    for s in high[:top_n]:
        print(f"  [{s['question_id']}] {s['action']}: {s['suggestion'][:70]}...")

    # 리포트 저장
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"failure_analysis_{ts}.json"
    report = generate_fix_report(suggestions, len(rows), out)

    print(f"\n[저장] {out}")
    print(f"  rechunk_targets: {len(report['rechunk_targets'])}개")
    print(f"  threshold_adjustments: {len(report['threshold_adjustments'])}개")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RAG 평가 실패 케이스 분류 및 수정 제안")
    p.add_argument("--eval-result", type=Path, default=None, dest="eval_result",
                   help="평가 결과 JSON 경로 (미지정 시 results/에서 최신 파일 자동 탐색)")
    p.add_argument("--dataset",    type=Path, default=None,
                   help=f"v3 데이터셋 경로 (기본값: {V3_DATASET})")
    p.add_argument("--top-n",      type=int, default=10, dest="top_n",
                   help="출력할 High Priority 수정 제안 수 (기본값: 10)")
    p.add_argument("--mock",       action="store_true",
                   help="Mock 데이터로 구조 확인")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
