"""
RAG 평가 실행기 — eval_dataset.json 기반

사용 예시:
  # 데이터셋 미리보기만
  python scripts/run_evaluation.py

  # 전체 50개 평가 (RAG 직접 호출)
  python scripts/run_evaluation.py --run

  # 난이도 질문 30개만 평가 + LLM 심사 + 결과 저장
  python scripts/run_evaluation.py --run --only difficulty --judge --save

  # 어려운 질문 10개만 샘플 5개로 Mock 평가
  python scripts/run_evaluation.py --run --only difficulty --difficulty hard --sample 5 --mock

  # 함정 질문만 평가
  python scripts/run_evaluation.py --run --only trap --judge --save
"""

import asyncio
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")
sys.path.insert(0, str(Path(__file__).parent.parent))

DATASET_PATH = Path(__file__).parent.parent / "tests" / "eval_dataset.json"
RESULTS_DIR = Path(__file__).parent.parent / "results"

# 컬렉션: 법률 일반 QA에 사용할 네임스페이스
QA_COLLECTIONS = ["law_database", "contracts"]


# ──────────────────────────────────────────────
# 데이터셋 로드 및 필터
# ──────────────────────────────────────────────

def load_dataset(
    only: str | None = None,
    difficulty: str | None = None,
) -> list[dict]:
    """eval_dataset.json 로드 후 필터 적용."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"데이터셋 파일을 찾을 수 없습니다: {DATASET_PATH}")

    with open(DATASET_PATH, encoding="utf-8") as f:
        data = json.load(f)

    questions = data["questions"]

    if only == "difficulty":
        questions = [q for q in questions if q["category"] == "difficulty"]
        if difficulty:
            questions = [q for q in questions if q.get("difficulty") == difficulty]
    elif only == "colloquial":
        questions = [q for q in questions if q["category"] == "colloquial"]
    elif only == "trap":
        questions = [q for q in questions if q["category"] == "trap"]

    return questions


# ──────────────────────────────────────────────
# 스코어링
# ──────────────────────────────────────────────

def compute_keyword_score(answer: str, keywords: list[str]) -> tuple[float, int]:
    """expected_keywords 중 RAG 답변에 포함된 비율 반환."""
    if not keywords:
        return 0.0, 0
    answer_lower = answer.lower()
    found = sum(1 for kw in keywords if kw.lower() in answer_lower)
    return round(found / len(keywords), 3), found


async def llm_judge(
    question: str,
    expected: str,
    actual: str,
    openai_client,
) -> dict:
    """GPT로 답변 품질을 0~3점으로 평가."""
    prompt = (
        "당신은 임대차 법률 챗봇 답변을 평가하는 전문가입니다.\n\n"
        f"[질문]\n{question}\n\n"
        f"[기대 답변 (Ground Truth)]\n{expected}\n\n"
        f"[실제 챗봇 답변]\n{actual}\n\n"
        "실제 답변을 아래 기준으로 평가하고 JSON으로만 반환하세요.\n"
        "3점: 핵심 법률 내용이 정확하고 충분함\n"
        "2점: 핵심 내용은 맞지만 일부 누락이 있음\n"
        "1점: 일부 맞지만 중요한 오류나 누락이 있음\n"
        "0점: 완전히 틀렸거나 관련 없는 답변\n\n"
        '반환 형식: {"score": 0~3, "reason": "한 줄 이유"}'
    )
    try:
        resp = await asyncio.to_thread(
            openai_client.chat.completions.create,
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        return {"score": -1, "reason": f"심사 실패: {e}"}


# ──────────────────────────────────────────────
# RAG 호출
# ──────────────────────────────────────────────

def call_rag(question: str, rag_client, embeddings, llm) -> str:
    """rag_query를 직접 호출하여 답변 문자열 반환."""
    from app.rag.chain.chain import rag_query

    result = rag_query(
        question=question,
        client=rag_client,
        embeddings=embeddings,
        llm=llm,
        collections=QA_COLLECTIONS,
        k_per_collection=4,
    )
    return result["answer"]


# ──────────────────────────────────────────────
# 출력 헬퍼
# ──────────────────────────────────────────────

def _bar(score: float, width: int = 10) -> str:
    filled = round(score * width)
    return "█" * filled + "░" * (width - filled)


def print_overview(questions: list[dict]) -> None:
    total = len(questions)
    cats = {}
    diffs = {}
    for q in questions:
        cats[q["category"]] = cats.get(q["category"], 0) + 1
        if q.get("difficulty"):
            diffs[q["difficulty"]] = diffs.get(q["difficulty"], 0) + 1

    print(f"\n{'='*70}")
    print(f"  📋  평가 데이터셋 개요")
    print(f"{'='*70}")
    print(f"  총 질문 수    : {total}개")
    print(f"  카테고리 분포 :", " / ".join(f"{k} {v}개" for k, v in sorted(cats.items())))
    if diffs:
        print(f"  난이도 분포   :", " / ".join(f"{k} {v}개" for k, v in sorted(diffs.items())))
    print(f"{'='*70}\n")


def print_report(results: list[dict]) -> None:
    total = len(results)
    if total == 0:
        print("⚠️  평가 결과 없음")
        return

    kw_scores = [r["keyword_score"] for r in results]
    avg_kw = sum(kw_scores) / total
    avg_latency = sum(r["latency_sec"] for r in results) / total

    judge_scores = [r["llm_judge_score"] for r in results if r.get("llm_judge_score", -1) >= 0]
    avg_judge = sum(judge_scores) / len(judge_scores) if judge_scores else None

    print(f"\n{'='*70}")
    print(f"  📊  평가 결과 요약")
    print(f"{'='*70}")
    print(f"  평가 질문 수       : {total}개")
    print(f"  평균 키워드 점수   : {_bar(avg_kw)} {avg_kw:.1%}")
    if avg_judge is not None:
        print(f"  평균 LLM 심사 점수 : {avg_judge:.2f} / 3.00")
    print(f"  평균 응답 시간     : {avg_latency:.2f}초")

    # 카테고리별 집계
    cats = {}
    for r in results:
        cat = r["category"]
        if cat not in cats:
            cats[cat] = []
        cats[cat].append(r)

    print(f"\n  --- 카테고리별 ---")
    for cat, items in sorted(cats.items()):
        cat_kw = sum(i["keyword_score"] for i in items) / len(items)
        cat_judge_list = [i["llm_judge_score"] for i in items if i.get("llm_judge_score", -1) >= 0]
        cat_judge_str = f"  LLM {sum(cat_judge_list)/len(cat_judge_list):.2f}/3" if cat_judge_list else ""
        print(f"  [{cat:12s}]  키워드 {_bar(cat_kw)} {cat_kw:.1%}  ({len(items)}개){cat_judge_str}")

    # 난이도별 집계
    diffs = {}
    for r in results:
        d = r.get("difficulty")
        if d:
            diffs.setdefault(d, []).append(r)

    if diffs:
        print(f"\n  --- 난이도별 ---")
        for diff in ["easy", "medium", "hard"]:
            if diff not in diffs:
                continue
            items = diffs[diff]
            d_kw = sum(i["keyword_score"] for i in items) / len(items)
            print(f"  [{diff:6s}]  키워드 {_bar(d_kw)} {d_kw:.1%}  ({len(items)}개)")

    # 키워드 점수 하위 5개 (개선 필요)
    worst = sorted(results, key=lambda r: r["keyword_score"])[:5]
    print(f"\n  --- 키워드 점수 하위 5개 (개선 필요) ---")
    for r in worst:
        print(f"  [{r['id']}] {r['question'][:50]}...")
        print(f"    키워드 {r['keyword_score']:.1%}  |  {r.get('difficulty',r['category'])}")

    print(f"{'='*70}\n")


# ──────────────────────────────────────────────
# 결과 저장
# ──────────────────────────────────────────────

def save_results(results: list[dict], args_str: str = "") -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"eval_{ts}.json"

    total = len(results)
    kw_scores = [r["keyword_score"] for r in results]
    judge_scores = [r["llm_judge_score"] for r in results if r.get("llm_judge_score", -1) >= 0]

    output = {
        "meta": {
            "timestamp": datetime.now().isoformat(),
            "total": total,
            "args": args_str,
            "avg_keyword_score": round(sum(kw_scores) / total, 3) if total else 0,
            "avg_llm_judge_score": round(sum(judge_scores) / len(judge_scores), 3) if judge_scores else None,
            "avg_latency_sec": round(sum(r["latency_sec"] for r in results) / total, 2) if total else 0,
        },
        "results": results,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"💾 결과 저장 완료: {out_path}")
    return out_path


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────

async def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="eval_dataset.json 기반 RAG 평가")
    parser.add_argument(
        "--only",
        choices=["difficulty", "colloquial", "trap"],
        help="평가할 카테고리 (기본: 전체)",
    )
    parser.add_argument(
        "--difficulty",
        choices=["easy", "medium", "hard"],
        help="난이도 필터 (--only difficulty 와 함께 사용)",
    )
    parser.add_argument("--sample", type=int, default=0, help="랜덤 샘플 수 (0=전체)")
    parser.add_argument("--run", action="store_true", help="실제 평가 실행")
    parser.add_argument("--judge", action="store_true", help="GPT LLM 심사 점수 추가")
    parser.add_argument("--mock", action="store_true", help="RAG 호출 없이 Mock 실행")
    parser.add_argument("--save", action="store_true", help="결과를 results/ 에 JSON으로 저장")
    args = parser.parse_args()

    # ── 데이터 로드 ──
    questions = load_dataset(only=args.only, difficulty=args.difficulty)
    if not questions:
        print("❌ 조건에 맞는 질문이 없습니다.")
        return

    if args.sample > 0:
        questions = random.sample(questions, min(args.sample, len(questions)))

    print_overview(questions)

    if not args.run:
        print("ℹ️  미리보기 모드입니다. --run 플래그를 추가하면 실제 평가를 실행합니다.\n")
        print("  예시 질문 목록:")
        for q in questions[:5]:
            flag = "[함정]" if q.get("is_trap") else "[구어]" if q.get("is_colloquial") else f"[{q.get('difficulty','')}]"
            print(f"    {q['id']} {flag} {q['question'][:60]}")
        if len(questions) > 5:
            print(f"    ... 외 {len(questions)-5}개")
        return

    # ── 의존성 초기화 ──
    rag_client = embeddings_model = llm_model = None
    if not args.mock:
        print("🔌 RAG 의존성 초기화 중 (Pinecone + KURE + LLM)...")
        from app.core.dependencies import get_vector_db, get_embeddings, get_llm
        rag_client = get_vector_db()
        embeddings_model = get_embeddings()
        llm_model = get_llm()
        print("✅ 초기화 완료\n")

    openai_client = None
    if args.judge:
        from openai import OpenAI
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    print(f"{'='*70}")
    print(f"  🚀  평가 시작  |  총 {len(questions)}개  |  {datetime.now().strftime('%H:%M:%S')}")
    print(f"  Mock={args.mock}  Judge={args.judge}  Save={args.save}")
    print(f"{'='*70}\n")

    results = []

    for i, q in enumerate(questions, 1):
        flag = "[함정]" if q.get("is_trap") else "[구어]" if q.get("is_colloquial") else f"[{q.get('difficulty','')}]"
        print(f"[{i:02d}/{len(questions)}] {q['id']} {flag}")
        print(f"  Q: {q['question']}")

        # RAG 호출
        t0 = time.perf_counter()
        if args.mock:
            rag_answer = f"(Mock 답변) 질문 '{q['question'][:20]}...' 에 대한 임시 답변입니다."
        else:
            try:
                rag_answer = await asyncio.to_thread(
                    call_rag, q["question"], rag_client, embeddings_model, llm_model
                )
            except Exception as e:
                rag_answer = f"[ERROR] {e}"
        latency = round(time.perf_counter() - t0, 2)

        # 키워드 점수
        kw_score, kw_found = compute_keyword_score(rag_answer, q.get("expected_keywords", []))
        kw_total = len(q.get("expected_keywords", []))

        # LLM 심사 (선택)
        judge_result = None
        if args.judge and openai_client and q.get("expected_answer"):
            judge_result = await llm_judge(
                q["question"], q["expected_answer"], rag_answer, openai_client
            )

        # 결과 저장
        record = {
            "id": q["id"],
            "category": q["category"],
            "difficulty": q.get("difficulty"),
            "part": q.get("part"),
            "is_trap": q.get("is_trap", False),
            "is_colloquial": q.get("is_colloquial", False),
            "question": q["question"],
            "rag_answer": rag_answer,
            "expected_answer": q.get("expected_answer"),
            "expected_keywords": q.get("expected_keywords", []),
            "relevant_law": q.get("relevant_law", []),
            "keyword_score": kw_score,
            "latency_sec": latency,
        }
        if judge_result:
            record["llm_judge_score"] = judge_result.get("score", -1)
            record["llm_judge_reason"] = judge_result.get("reason", "")
        results.append(record)

        # 즉시 출력
        print(f"  A: {rag_answer[:120]}{'...' if len(rag_answer) > 120 else ''}")
        print(f"  키워드 [{_bar(kw_score)}] {kw_score:.0%}  ({kw_found}/{kw_total}개)  |  {latency:.2f}s")
        if judge_result:
            score = judge_result.get("score", -1)
            reason = judge_result.get("reason", "")
            star = "⭐" * max(0, score) if score >= 0 else "❌"
            print(f"  LLM 심사: {star} {score}/3  —  {reason}")
        print()

    # ── 최종 리포트 ──
    print_report(results)

    if args.save:
        args_str = (
            f"only={args.only} difficulty={args.difficulty} "
            f"sample={args.sample} mock={args.mock} judge={args.judge}"
        )
        save_results(results, args_str)


if __name__ == "__main__":
    asyncio.run(main())
