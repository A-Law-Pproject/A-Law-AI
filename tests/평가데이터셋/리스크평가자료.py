"""Risk RAG evaluation from verified source datasets.
>> risk_stress_eval

-  보증금 반환 질문은 “임차권등기명령/대항력/우선변제권을 주장하지 않는다” 같은 계약서 조항으로 변환하고, 예상 라벨도 위험/주의/안전으로 붙입니다. => 스트레스 테스트
- 법률 근거 정답은 lease_faq.jsonl에서 가져옴
- 평가 케이스 객체로 통일
-





The conversion from Q/A to clauses is deterministic and rule-based, not LLM-made.
Use --export-cases-only to inspect generated clauses before running real scoring.

Examples:
  .venv\\Scripts\\python.exe tests\\eval_risk_verified_sources.py --mock --sample 10
  .venv\\Scripts\\python.exe tests\\eval_risk_verified_sources.py --export-cases-only
  .venv\\Scripts\\python.exe tests\\eval_risk_verified_sources.py --sample 30
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd
from dotenv import load_dotenv

def _find_repo_root(start: Path) -> Path:
    """Locate repo root whether this script is under tests/ or tests/평가데이터셋/."""
    for path in (start, *start.parents):
        if (path / "app").exists() and (path / "tests").exists():
            return path
    return start.parents[1]


ROOT = _find_repo_root(Path(__file__).resolve())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

DATASET_DIR = ROOT / "tests" / "\ud3c9\uac00\ub370\uc774\ud130\uc14b"
CHATBOT_XLSX = DATASET_DIR / "\ucc57\ubd07_\ud3c9\uac00\uc6a9_\ucd5c\uc885\uc790\ub8cc.xlsx"
LEASE_FAQ_JSONL = DATASET_DIR / "lease_faq.jsonl"
RESULTS_DIR = ROOT / "results"

RISK_LEVELS = ("위험", "주의", "안전")
ARTICLE_RE = re.compile(r"제\s*\d+\s*조(?:의\s*\d+)?(?:\s*제\s*\d+\s*항)?")
LAW_REF_RE = re.compile(
    r"([가-힣A-Za-z0-9ㆍ·\s]+?(?:법|시행령|규칙|민법|국세기본법))\s*"
    r"(제\s*\d+\s*조(?:의\s*\d+)?(?:\s*제\s*\d+\s*항)?)?"
)

LAW_KEY = "\ubc95\ub839\uba85"
ARTICLE_TITLE_KEY = "\uc870\ubb38\uba85"
ARTICLE_CONTENT_KEY = "\uc870\ubb38\ub0b4\uc6a9"

RELEVANT_LAW_NAME_HINTS = (
    "임대차",
    "민법",
    "공인중개사",
    "민간임대",
    "전세",
    "부동산",
    "임차권",
    "국세기본법",
    "집합건물",
)

LEGAL_RELEVANCE_TERMS = (
    "권리금",
    "신규임차인",
    "차임",
    "보증금",
    "증액",
    "증감청구",
    "임차권등기명령",
    "대항력",
    "우선변제",
    "계약갱신",
    "갱신요구",
    "전세권",
    "수선",
    "원상복구",
    "중개보수",
    "공인중개사",
    "경매",
    "국세",
)

OWNER_CHANGE_TERMS = ("집주인", "주인", "임대인", "소유자", "새 주인")
OWNER_CHANGE_EVENTS = ("바뀌", "변경", "팔", "매매", "승계", "경락")
RESTORATION_TERMS = ("원상복구", "도배", "장판", "파손", "훼손", "마모")
REPAIR_TERMS = ("수리", "수선", "고장", "누수", "보일러", "설비")
DEPOSIT_RETURN_TERMS = ("보증금", "전세금")
DEPOSIT_PROBLEM_TERMS = ("못 받", "못받", "돌려", "반환", "제때", "새 세입자", "임차권등기", "이사")
EVICTION_TERMS = ("나가라고", "나가야", "퇴거", "비워", "강제 퇴거")
RENEWAL_TERMS = ("계약갱신", "갱신요구", "갱신 요구", "묵시적 갱신", "재계약")
EARLY_TERMINATION_TERMS = ("계약 만료 전", "만료 전", "중도 해지", "중도해지", "개인 사정으로 이사")
ARREARS_TERMS = ("연체", "미납", "월세")


@dataclass
class VerifiedRiskCase:
    id: str
    source: str
    source_row: str
    category: str
    question: str
    verified_answer: str
    contract_text: str
    target_clause: str
    expected_risk_level: str
    expected_score_min: int
    expected_score_max: int
    expected_law_refs: list[str]
    expected_law_names: list[str]
    requires_law_reference: bool
    transform_rule: str
    label_confidence: float
    source_url: str = ""


@dataclass
class CasePrediction:
    case_id: str
    predicted_risk_level: str
    predicted_score: float
    predicted_legal_reference: str
    predicted_analysis: str
    latency_ms: float
    raw: dict[str, Any]


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        missing = pd.isna(value)
        if isinstance(missing, bool):
            return missing
    except (TypeError, ValueError):
        pass
    return str(value).strip().lower() in {"", "nan", "none", "null"}


def _norm(text: Any) -> str:
    if _is_missing(text):
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def _article_from_text(text: str) -> str:
    match = ARTICLE_RE.search(text or "")
    return _norm(match.group(0)) if match else ""


def _unique(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        value = _norm(item)
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _has_any(text: str, terms: tuple[str, ...] | list[str]) -> bool:
    return any(term in text for term in terms)


def _case_scope(question: str, fallback: str) -> str:
    scope = _norm(question).strip("\"'“”‘’ ")
    replacements = (
        ("어떻게 해야하나요", "대응"),
        ("어떻게 해야 하나요", "대응"),
        ("어떻게 하나요", "대응"),
        ("해야 하나요", "여부"),
        ("해야하나요", "여부"),
        ("가능한가요", "가능 여부"),
        ("가능할까요", "가능 여부"),
        ("되나요", "여부"),
        ("하나요", "여부"),
        ("인가요", "여부"),
        ("무엇인가요", "내용"),
        ("뭔가요", "내용"),
        ("나가야 하나요", "퇴거 요구"),
        ("나가야하나요", "퇴거 요구"),
    )
    for old, new in replacements:
        scope = scope.replace(old, new)
    scope = re.sub(r"[?？!]+", "", scope)
    scope = re.sub(r"\s+", " ", scope).strip(" .,:;")
    if len(scope) < 4:
        scope = fallback
    if len(scope) > 48:
        scope = scope[:48].rstrip() + " 관련"
    return scope


def _scoped_clause(question: str, fallback_scope: str, body: str) -> str:
    return f"{_case_scope(question, fallback_scope)}와 관련하여, {body}"


def _score_band(level: str) -> tuple[int, int]:
    if level == "위험":
        return 70, 100
    if level == "주의":
        return 40, 69
    return 0, 39


def _contract_text(clause: str) -> str:
    return f"[특약사항]\n1. {clause}"


def _query_relevance_terms(text: str) -> list[str]:
    terms = [term for term in LEGAL_RELEVANCE_TERMS if term in text]
    if any(k in text for k in ("올려", "올리", "인상", "증액")):
        terms.extend(["차임", "보증금", "증액", "증감청구"])
    if any(k in text for k in ("돌려", "반환", "이사", "나가")) and "보증금" in text:
        terms.extend(["보증금", "임차권등기명령", "대항력", "우선변제"])
    if any(k in text for k in ("갱신", "계약갱신")):
        terms.extend(["계약갱신", "갱신요구"])
    return _unique(terms)


def _law_relevance_score(law: dict[str, Any], query_text: str, original_index: int) -> float:
    law_name = _norm(law.get(LAW_KEY))
    title = _norm(law.get(ARTICLE_TITLE_KEY))
    content = _norm(law.get(ARTICLE_CONTENT_KEY))
    haystack = f"{law_name} {title} {content}"
    score = 0.0

    for term in _query_relevance_terms(query_text):
        if term in title:
            score += 5.0
        if term in law_name:
            score += 2.0
        if term in content:
            score += 2.0

    if title in {"목적", "적용범위"}:
        score -= 4.0
    if "시행령" in law_name and "시행령" not in query_text:
        score -= 0.5
    if law_name and law_name in query_text:
        score += 1.0
    if ARTICLE_RE.search(content):
        score += 0.5

    # Preserve the source order only as a final tie breaker.
    score -= original_index * 0.01
    return score


def _law_refs_from_related_laws(
    related_laws: list[dict[str, Any]],
    *,
    max_refs: int,
    query_text: str,
) -> tuple[list[str], list[str]]:
    refs: list[str] = []
    names: list[str] = []

    ranked_laws = sorted(
        enumerate(related_laws),
        key=lambda item: _law_relevance_score(item[1], query_text, item[0]),
        reverse=True,
    )

    for _, law in ranked_laws:
        law_name = _norm(law.get(LAW_KEY))
        if not law_name:
            continue
        if not any(hint in law_name for hint in RELEVANT_LAW_NAME_HINTS):
            continue

        article = _article_from_text(_norm(law.get(ARTICLE_CONTENT_KEY)))
        if not article:
            article = _article_from_text(_norm(law.get(ARTICLE_TITLE_KEY)))

        names.append(law_name)
        refs.append(f"{law_name} {article}" if article else law_name)
        if len(refs) >= max_refs:
            break

    return _unique(refs), _unique(names)


def _derive_clause(question: str, answer: str) -> tuple[str, str, str, float]:
    """Return (clause, expected_level, rule_name, confidence).

    Rules intentionally produce contract-style clauses, not question prompts.
    The labels come from legal-domain deterministic transformations:
    "tenant waives statutory protection" => 위험, "ambiguous burden" => 주의,
    "preserves statutory rights" => 안전.
    """
    q = _norm(question)
    a = _norm(answer)
    text = f"{q} {a}"

    owner_change = (
        _has_any(q, OWNER_CHANGE_TERMS)
        and _has_any(q, OWNER_CHANGE_EVENTS)
        and _has_any(text, ("임대차", "계약", "거주", "세입자", "임차인", "보증금"))
    )
    if owner_change:
        return (
            _scoped_clause(
                q,
                "임대차 목적물 소유자 변경",
                "임대차 목적물의 소유자가 변경되어도 임차인은 계약기간까지 거주할 수 있고, 보증금 반환 의무는 새 소유자에게 승계된다.",
            ),
            "안전",
            "lawful_owner_succession",
            0.88,
        )

    if _has_any(text, ARREARS_TERMS) and _has_any(text, ("연체", "미납")):
        return (
            _scoped_clause(
                q,
                "차임 연체",
                "임차인이 차임을 1회만 연체해도 임대인은 즉시 계약을 해지하고 권리금 회수 기회와 퇴거 유예기간을 모두 박탈할 수 있다.",
            ),
            "위험",
            "arrears_overbroad_termination",
            0.86,
        )

    if _has_any(q, RESTORATION_TERMS):
        return (
            _scoped_clause(
                q,
                "원상복구",
                "임차인은 통상적인 사용으로 인한 자연 마모까지 포함하여 임차목적물을 새것과 같은 상태로 원상복구한다.",
            ),
            "주의",
            "overbroad_restoration",
            0.84,
        )

    if _has_any(q, REPAIR_TERMS) or "장기수선충당금" in text:
        return (
            _scoped_clause(
                q,
                "수선 및 유지비",
                "노후 또는 통상적인 사용으로 발생한 주요 설비의 고장, 장기수선충당금, 필수 수선 비용도 모두 임차인이 최종 부담한다.",
            ),
            "주의",
            "repair_burden_shift",
            0.84,
        )

    if "권리금" in text:
        return (
            _scoped_clause(
                q,
                "권리금 회수",
                "임대인은 신규임차인 주선이나 권리금 회수 기회를 정당한 사유 없이 거절하거나 방해할 수 있으며, 그 손해를 배상하지 않는다.",
            ),
            "위험",
            "premium_interference",
            0.92,
        )
    if _has_any(text, ("우선변제", "확정일자", "대항력")) and _has_any(text, ("증액", "올려", "올리")):
        return (
            _scoped_clause(
                q,
                "증액 보증금 보호",
                "임차인은 증액된 보증금에 관하여 별도 계약서 작성, 확정일자 취득, 우선변제권 확보를 요구하지 않는다.",
            ),
            "위험",
            "increased_deposit_priority_waiver",
            0.88,
        )
    if _has_any(text, ("전세금", "보증금", "차임", "월세")) and _has_any(
        text, ("올려", "인상", "증액", "올리")
    ):
        return (
            _scoped_clause(
                q,
                "차임 및 보증금 증액",
                "임대인은 계약기간 중 법정 한도, 증액 사유, 기간 제한과 무관하게 차임 또는 보증금을 임의로 증액할 수 있다.",
            ),
            "위험",
            "unlimited_rent_increase",
            0.92,
        )
    if _has_any(text, EARLY_TERMINATION_TERMS) and _has_any(text, ("이사", "월세", "차임", "계약")):
        return (
            _scoped_clause(
                q,
                "계약 만료 전 이사",
                "임차인은 계약 만료 전에 이사하는 경우 사유와 관계없이 남은 계약기간의 차임 전액과 신규 임차인 모집 비용을 모두 부담한다.",
            ),
            "주의",
            "early_termination_overburden",
            0.84,
        )
    if _has_any(text, DEPOSIT_RETURN_TERMS) and _has_any(text, DEPOSIT_PROBLEM_TERMS):
        return (
            _scoped_clause(
                q,
                "보증금 반환",
                "임차인은 보증금을 반환받지 못한 경우에도 즉시 이사하여야 하며, 임차권등기명령이나 대항력 및 우선변제권을 주장하지 않는다.",
            ),
            "위험",
            "deposit_right_waiver",
            0.9,
        )
    if _has_any(text, RENEWAL_TERMS):
        return (
            _scoped_clause(
                q,
                "계약갱신",
                "임차인은 계약갱신요구권과 묵시적 갱신으로 인정되는 권리를 행사하지 않으며, 임대인은 정당한 사유 없이 갱신 요구를 거절할 수 있다.",
            ),
            "위험",
            "renewal_right_waiver",
            0.9,
        )
    if _has_any(text, ("경매", "저당", "압류", "우선변제", "대항력", "확정일자")):
        return (
            _scoped_clause(
                q,
                "대항력 및 우선변제",
                "임차인은 대항요건과 확정일자를 갖추었더라도 경매 또는 공매 절차에서 보증금 우선변제를 청구하지 않는다.",
            ),
            "위험",
            "priority_right_waiver",
            0.84,
        )
    if _has_any(q, EVICTION_TERMS):
        return (
            _scoped_clause(
                q,
                "일방 퇴거 요구",
                "임대인은 필요하다고 판단하면 임차인의 동의 없이 언제든지 계약을 해지하고 즉시 퇴거를 요구할 수 있다.",
            ),
            "위험",
            "unilateral_eviction",
            0.86,
        )
    if _has_any(text, ("중개보수", "중개수수료", "복비")):
        return (
            _scoped_clause(
                q,
                "중개보수",
                "중개보수는 법정 상한과 부담 주체와 무관하게 임차인이 임대인 부담분까지 전액 부담한다.",
            ),
            "위험",
            "brokerage_fee_shift",
            0.82,
        )

    fallback_subject = question.rstrip("?.! ")
    return (
        f"임차인은 '{fallback_subject}'와 관련하여 법령상 인정되는 권리를 모두 포기한다.",
        "위험",
        "fallback_right_waiver",
        0.55,
    )


def _safe_control_case(case: VerifiedRiskCase, index: int) -> VerifiedRiskCase:
    law_hint = case.expected_law_names[0] if case.expected_law_names else "관계 법령"
    clause = _scoped_clause(
        case.question,
        "법령 준수",
        f"당사자는 {law_hint}에서 정한 임차인의 권리와 임대인의 의무를 제한하지 않으며, 법령에 반하는 특약은 적용하지 않는다.",
    )
    return VerifiedRiskCase(
        id=f"{case.id}_SAFE_{index}",
        source=f"{case.source}:safe_control",
        source_row=case.source_row,
        category=case.category,
        question=f"[안전 대조] {case.question}",
        verified_answer=case.verified_answer,
        contract_text=_contract_text(clause),
        target_clause=clause,
        expected_risk_level="안전",
        expected_score_min=0,
        expected_score_max=39,
        expected_law_refs=case.expected_law_refs,
        expected_law_names=case.expected_law_names,
        requires_law_reference=False,
        transform_rule="safe_statutory_rights_preserved",
        label_confidence=0.82,
        source_url=case.source_url,
    )


def load_lease_faq_cases(max_laws_per_case: int) -> list[VerifiedRiskCase]:
    cases: list[VerifiedRiskCase] = []
    with LEASE_FAQ_JSONL.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            question = _norm(row.get("question"))
            answer = _norm(row.get("answer"))
            clause, level, rule, confidence = _derive_clause(question, answer)
            law_refs, law_names = _law_refs_from_related_laws(
                row.get("related_laws") or [],
                max_refs=max_laws_per_case,
                query_text=f"{question} {answer} {clause}",
            )
            if not question or not law_refs:
                continue

            low, high = _score_band(level)
            cases.append(
                VerifiedRiskCase(
                    id=f"FAQ_{line_no:03d}",
                    source="lease_faq",
                    source_row=str(line_no),
                    category=row.get("source") or "법령AI FAQ",
                    question=question,
                    verified_answer=answer,
                    contract_text=_contract_text(clause),
                    target_clause=clause,
                    expected_risk_level=level,
                    expected_score_min=low,
                    expected_score_max=high,
                    expected_law_refs=law_refs,
                    expected_law_names=law_names,
                    requires_law_reference=level != "안전",
                    transform_rule=rule,
                    label_confidence=confidence,
                    source_url=row.get("url") or "",
                )
            )
    return cases


def _find_column(columns: list[Any], candidates: list[str], default_index: int) -> Any:
    normalized = {_norm(col): col for col in columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    return columns[default_index]


def load_chatbot_final_cases() -> list[VerifiedRiskCase]:
    df = pd.read_excel(CHATBOT_XLSX)
    columns = list(df.columns)
    main_col = _find_column(columns, ["주유형"], 0)
    sub_col = _find_column(columns, ["세부유형"], 1)
    question_col = _find_column(columns, ["질문"], 2)
    answer_col = _find_column(columns, ["상세 답변"], 3)

    cases: list[VerifiedRiskCase] = []
    for idx, row in df.iterrows():
        question = _norm(row.get(question_col))
        answer = _norm(row.get(answer_col))
        if not question:
            continue

        clause, level, rule, confidence = _derive_clause(question, answer)
        low, high = _score_band(level)
        category = " / ".join(
            part for part in [_norm(row.get(main_col)), _norm(row.get(sub_col))] if part
        )
        cases.append(
            VerifiedRiskCase(
                id=f"CHATBOT_{idx + 1:03d}",
                source="chatbot_final",
                source_row=str(idx + 1),
                category=category,
                question=question,
                verified_answer=answer,
                contract_text=_contract_text(clause),
                target_clause=clause,
                expected_risk_level=level,
                expected_score_min=low,
                expected_score_max=high,
                expected_law_refs=[],
                expected_law_names=[],
                requires_law_reference=False,
                transform_rule=rule,
                label_confidence=confidence,
            )
        )
    return cases


def _case_dedupe_key(case: VerifiedRiskCase) -> str:
    return _compact(case.question)


def _dedupe_cases(cases: list[VerifiedRiskCase]) -> list[VerifiedRiskCase]:
    """Keep the best row for duplicated verified questions."""
    ranked = sorted(
        cases,
        key=lambda case: (
            0 if case.source == "lease_faq" else 1,
            -len(case.expected_law_refs),
            -case.label_confidence,
            case.id,
        ),
    )
    by_key: dict[str, VerifiedRiskCase] = {}
    for case in ranked:
        key = _case_dedupe_key(case)
        if not key or key in by_key:
            continue
        by_key[key] = case
    return sorted(by_key.values(), key=lambda case: case.id)


def _balance_cases(cases: list[VerifiedRiskCase], max_per_label: int = 0) -> list[VerifiedRiskCase]:
    by_label: dict[str, list[VerifiedRiskCase]] = {label: [] for label in RISK_LEVELS}
    for case in cases:
        by_label.setdefault(case.expected_risk_level, []).append(case)

    non_empty_counts = [len(items) for items in by_label.values() if items]
    if not non_empty_counts:
        return cases

    target = min(non_empty_counts)
    if max_per_label > 0:
        target = min(target, max_per_label)

    balanced: list[VerifiedRiskCase] = []
    for label in RISK_LEVELS:
        label_cases = sorted(
            by_label.get(label, []),
            key=lambda case: (
                0 if case.source == "lease_faq" else 1,
                case.transform_rule,
                -case.label_confidence,
                case.id,
            ),
        )
        balanced.extend(label_cases[:target])
    return sorted(balanced, key=lambda case: case.id)


def build_cases(args: argparse.Namespace) -> list[VerifiedRiskCase]:
    cases: list[VerifiedRiskCase] = []
    if args.source in {"all", "lease_faq"}:
        cases.extend(load_lease_faq_cases(args.max_laws_per_case))
    if args.source in {"all", "chatbot_final"}:
        cases.extend(load_chatbot_final_cases())

    cases = [case for case in cases if case.label_confidence >= args.min_confidence]

    if not args.allow_duplicate_questions:
        cases = _dedupe_cases(cases)

    if args.include_safe_controls:
        controls = [_safe_control_case(case, i + 1) for i, case in enumerate(cases)]
        cases.extend(controls)

    if args.balance_labels:
        cases = _balance_cases(cases, args.max_per_label)

    if args.sample > 0:
        cases = cases[: args.sample]
    return cases


def _extract_predicted_clause(result: dict[str, Any]) -> dict[str, Any]:
    clauses = result.get("clauses") or []
    if not clauses:
        return {}
    return max(clauses, key=lambda item: float(item.get("score") or 0))


def _mock_prediction(case: VerifiedRiskCase) -> CasePrediction:
    midpoint = (case.expected_score_min + case.expected_score_max) / 2
    legal_ref = "; ".join(case.expected_law_refs[:2]) if case.requires_law_reference else ""
    return CasePrediction(
        case_id=case.id,
        predicted_risk_level=case.expected_risk_level,
        predicted_score=midpoint,
        predicted_legal_reference=legal_ref,
        predicted_analysis="mock prediction",
        latency_ms=0.0,
        raw={
            "overall_risk_score": midpoint,
            "clauses": [
                {
                    "risk_level": case.expected_risk_level,
                    "score": midpoint,
                    "legal_reference": legal_ref,
                    "analysis": "mock prediction",
                }
            ],
        },
    )


async def _real_prediction(case: VerifiedRiskCase, deps: dict[str, Any]) -> CasePrediction:
    from app.rag.chain.chain import detect_risk_contract

    start = time.perf_counter()
    result = await detect_risk_contract(
        case.contract_text,
        deps["client"],
        deps["embeddings"],
        deps["llm"],
    )
    latency_ms = (time.perf_counter() - start) * 1000
    clause = _extract_predicted_clause(result)
    return CasePrediction(
        case_id=case.id,
        predicted_risk_level=_norm(clause.get("risk_level")),
        predicted_score=float(clause.get("score") or result.get("overall_risk_score") or 0),
        predicted_legal_reference=_norm(clause.get("legal_reference")),
        predicted_analysis=_norm(clause.get("analysis")),
        latency_ms=latency_ms,
        raw=result,
    )


async def run_predictions(
    cases: list[VerifiedRiskCase],
    *,
    mock: bool,
    concurrency: int,
) -> list[CasePrediction]:
    if mock:
        return [_mock_prediction(case) for case in cases]

    from app.core.dependencies import get_embeddings, get_llm, get_vector_db

    deps = {
        "client": get_vector_db(),
        "embeddings": get_embeddings(),
        "llm": get_llm(),
    }
    sem = asyncio.Semaphore(concurrency)

    async def run_one(case: VerifiedRiskCase) -> CasePrediction:
        async with sem:
            try:
                return await _real_prediction(case, deps)
            except Exception as exc:  # keep the eval run alive and measurable
                return CasePrediction(
                    case_id=case.id,
                    predicted_risk_level="",
                    predicted_score=0.0,
                    predicted_legal_reference="",
                    predicted_analysis=f"ERROR: {exc}",
                    latency_ms=0.0,
                    raw={"error": str(exc)},
                )

    return await asyncio.gather(*(run_one(case) for case in cases))


def _split_predicted_refs(text: str) -> list[str]:
    if not text:
        return []
    chunks = re.split(r"[;\n,]+", text)
    refs: list[str] = []
    for chunk in chunks:
        match = LAW_REF_RE.search(chunk)
        refs.append(_norm(match.group(0) if match else chunk))
    return _unique(refs)


def _law_name_hit(predicted_ref: str, expected_law_names: list[str]) -> bool:
    compact_pred = _compact(predicted_ref)
    return any(_compact(name) in compact_pred for name in expected_law_names)


def _article_hit(predicted_ref: str, expected_ref: str) -> bool:
    expected_article = _article_from_text(expected_ref)
    if not expected_article:
        return False
    return _compact(expected_article) in _compact(predicted_ref)


def _reference_exact_hit(predicted_ref: str, expected_ref: str) -> bool:
    expected_law = expected_ref.replace(_article_from_text(expected_ref), "").strip()
    return _law_name_hit(predicted_ref, [expected_law]) and _article_hit(predicted_ref, expected_ref)


def _classification_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = list(RISK_LEVELS)
    total = len(rows)
    accuracy = mean(1.0 if r["level_correct"] else 0.0 for r in rows) if rows else 0.0

    by_label: dict[str, dict[str, float]] = {}
    f1_values: list[float] = []
    for label in labels:
        tp = sum(1 for r in rows if r["expected_risk_level"] == label and r["predicted_risk_level"] == label)
        fp = sum(1 for r in rows if r["expected_risk_level"] != label and r["predicted_risk_level"] == label)
        fn = sum(1 for r in rows if r["expected_risk_level"] == label and r["predicted_risk_level"] != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        support = sum(1 for r in rows if r["expected_risk_level"] == label)
        by_label[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }
        if support:
            f1_values.append(f1)

    def is_unsafe(level: str) -> bool:
        return level in {"위험", "주의"}

    tp = sum(1 for r in rows if is_unsafe(r["expected_risk_level"]) and is_unsafe(r["predicted_risk_level"]))
    fp = sum(1 for r in rows if not is_unsafe(r["expected_risk_level"]) and is_unsafe(r["predicted_risk_level"]))
    fn = sum(1 for r in rows if is_unsafe(r["expected_risk_level"]) and not is_unsafe(r["predicted_risk_level"]))
    unsafe_precision = tp / (tp + fp) if tp + fp else 0.0
    unsafe_recall = tp / (tp + fn) if tp + fn else 0.0
    unsafe_f1 = (
        2 * unsafe_precision * unsafe_recall / (unsafe_precision + unsafe_recall)
        if unsafe_precision + unsafe_recall
        else 0.0
    )

    return {
        "total": total,
        "accuracy": round(accuracy, 4),
        "macro_f1": round(mean(f1_values), 4) if f1_values else 0.0,
        "by_label": by_label,
        "unsafe_binary": {
            "precision": round(unsafe_precision, 4),
            "recall": round(unsafe_recall, 4),
            "f1": round(unsafe_f1, 4),
        },
    }


def evaluate(cases: list[VerifiedRiskCase], predictions: list[CasePrediction]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_id = {pred.case_id: pred for pred in predictions}
    rows: list[dict[str, Any]] = []

    for case in cases:
        pred = by_id[case.id]
        predicted_refs = _split_predicted_refs(pred.predicted_legal_reference)
        expected_refs = case.expected_law_refs
        expected_laws = case.expected_law_names

        law_name_hit = bool(
            expected_laws
            and any(_law_name_hit(pred_ref, expected_laws) for pred_ref in predicted_refs)
        )
        article_exact_hit = bool(
            expected_refs
            and any(
                _reference_exact_hit(pred_ref, expected_ref)
                for pred_ref in predicted_refs
                for expected_ref in expected_refs
            )
        )
        expected_ref_hits = sum(
            1
            for expected_ref in expected_refs
            if any(_reference_exact_hit(pred_ref, expected_ref) for pred_ref in predicted_refs)
        )
        predicted_ref_hits = sum(
            1
            for pred_ref in predicted_refs
            if any(_law_name_hit(pred_ref, expected_laws) for _ in [pred_ref])
        )
        citation_precision = predicted_ref_hits / len(predicted_refs) if predicted_refs else (1.0 if not case.requires_law_reference else 0.0)
        citation_recall = expected_ref_hits / len(expected_refs) if expected_refs else 0.0

        level_correct = pred.predicted_risk_level == case.expected_risk_level
        score_band_correct = case.expected_score_min <= pred.predicted_score <= case.expected_score_max
        expected_midpoint = (case.expected_score_min + case.expected_score_max) / 2
        score_abs_error = abs(pred.predicted_score - expected_midpoint)
        law_pass = (not case.requires_law_reference) or law_name_hit or article_exact_hit
        strict_pass = level_correct and score_band_correct and law_pass

        rows.append(
            {
                **asdict(case),
                "predicted_risk_level": pred.predicted_risk_level,
                "predicted_score": pred.predicted_score,
                "predicted_legal_reference": pred.predicted_legal_reference,
                "predicted_analysis": pred.predicted_analysis,
                "predicted_refs": predicted_refs,
                "level_correct": level_correct,
                "score_band_correct": score_band_correct,
                "score_abs_error": round(score_abs_error, 4),
                "law_name_hit": law_name_hit,
                "article_exact_hit": article_exact_hit,
                "citation_precision": round(citation_precision, 4),
                "citation_recall": round(citation_recall, 4),
                "strict_pass": strict_pass,
                "latency_ms": round(pred.latency_ms, 1),
                "raw_prediction": pred.raw,
            }
        )

    law_rows = [r for r in rows if r["requires_law_reference"] and r["expected_law_refs"]]
    summary = {
        "case_count": len(rows),
        "source_counts": _count_by(rows, "source"),
        "rule_counts": _count_by(rows, "transform_rule"),
        "classification": _classification_report(rows),
        "score": {
            "band_accuracy": round(mean(1.0 if r["score_band_correct"] else 0.0 for r in rows), 4) if rows else 0.0,
            "mae_to_band_midpoint": round(mean(r["score_abs_error"] for r in rows), 4) if rows else 0.0,
        },
        "legal_grounding": {
            "evaluated_cases": len(law_rows),
            "law_name_hit_rate": round(mean(1.0 if r["law_name_hit"] else 0.0 for r in law_rows), 4) if law_rows else 0.0,
            "article_exact_hit_rate": round(mean(1.0 if r["article_exact_hit"] else 0.0 for r in law_rows), 4) if law_rows else 0.0,
            "citation_precision": round(mean(r["citation_precision"] for r in law_rows), 4) if law_rows else 0.0,
            "citation_recall": round(mean(r["citation_recall"] for r in law_rows), 4) if law_rows else 0.0,
            "missing_required_reference_rate": round(
                mean(1.0 if not r["predicted_legal_reference"] else 0.0 for r in law_rows),
                4,
            ) if law_rows else 0.0,
        },
        "strict_pass_rate": round(mean(1.0 if r["strict_pass"] else 0.0 for r in rows), 4) if rows else 0.0,
        "latency_ms": {
            "avg": round(mean(r["latency_ms"] for r in rows), 1) if rows else 0.0,
            "max": round(max((r["latency_ms"] for r in rows), default=0.0), 1),
        },
    }
    return summary, rows


def _count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key) or "")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def save_outputs(summary: dict[str, Any], rows: list[dict[str, Any]], *, prefix: str) -> tuple[Path, Path]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = RESULTS_DIR / f"{prefix}_{ts}.json"
    csv_path = RESULTS_DIR / f"{prefix}_{ts}.csv"

    payload = {
        "meta": {
            "created_at": ts,
            "trusted_sources": [str(CHATBOT_XLSX), str(LEASE_FAQ_JSONL)],
            "excluded_sources": ["risk_dataset.json", "risk_eval_dataset.json"],
        },
        "summary": summary,
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    flat_fields = [
        "id",
        "source",
        "source_row",
        "category",
        "transform_rule",
        "label_confidence",
        "expected_risk_level",
        "predicted_risk_level",
        "level_correct",
        "expected_score_min",
        "expected_score_max",
        "predicted_score",
        "score_band_correct",
        "expected_law_refs",
        "predicted_legal_reference",
        "law_name_hit",
        "article_exact_hit",
        "citation_precision",
        "citation_recall",
        "strict_pass",
        "latency_ms",
        "target_clause",
        "question",
        "verified_answer",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=flat_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(row.get(key), ensure_ascii=False)
                    if isinstance(row.get(key), (list, dict))
                    else row.get(key)
                    for key in flat_fields
                }
            )

    return json_path, csv_path


def print_summary(summary: dict[str, Any], json_path: Path | None = None, csv_path: Path | None = None) -> None:
    print("\n=== Risk Verified Evaluation Summary ===")
    print(f"cases: {summary['case_count']}")
    if summary.get("mode") == "case_export":
        print("mode: case_export (no model predictions evaluated)")
        print(f"label_counts: {summary.get('label_counts', {})}")
        print(f"unique_target_clauses: {summary.get('unique_target_clauses', 0)}")
        print(f"legal_grounding_cases: {summary['legal_grounding']['evaluated_cases']}")
        if json_path and csv_path:
            print(f"saved_json: {json_path}")
            print(f"saved_csv: {csv_path}")
        return

    print(f"strict_pass_rate: {summary['strict_pass_rate']:.4f}")
    print(
        "classification: "
        f"accuracy={summary['classification']['accuracy']:.4f}, "
        f"macro_f1={summary['classification']['macro_f1']:.4f}, "
        f"unsafe_recall={summary['classification']['unsafe_binary']['recall']:.4f}"
    )
    print(
        "score: "
        f"band_accuracy={summary['score']['band_accuracy']:.4f}, "
        f"mae={summary['score']['mae_to_band_midpoint']:.4f}"
    )
    print(
        "legal_grounding: "
        f"cases={summary['legal_grounding']['evaluated_cases']}, "
        f"law_hit={summary['legal_grounding']['law_name_hit_rate']:.4f}, "
        f"article_exact={summary['legal_grounding']['article_exact_hit_rate']:.4f}, "
        f"citation_precision={summary['legal_grounding']['citation_precision']:.4f}, "
        f"missing_ref={summary['legal_grounding']['missing_required_reference_rate']:.4f}"
    )
    print(f"latency_ms: avg={summary['latency_ms']['avg']}, max={summary['latency_ms']['max']}")
    if json_path and csv_path:
        print(f"saved_json: {json_path}")
        print(f"saved_csv: {csv_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate risk RAG with verified source datasets.")
    parser.add_argument("--source", choices=["all", "lease_faq", "chatbot_final"], default="all")
    parser.add_argument("--sample", type=int, default=0, help="Limit number of generated cases.")
    parser.add_argument("--mock", action="store_true", help="Do not call Pinecone/OpenAI; validate loaders and metrics.")
    parser.add_argument("--concurrency", type=int, default=2, help="Concurrent real risk-analysis calls.")
    parser.add_argument("--min-confidence", type=float, default=0.7, help="Minimum rule confidence to include.")
    parser.add_argument("--max-laws-per-case", type=int, default=3)
    parser.add_argument("--include-safe-controls", action="store_true", help="Add safe control clauses derived from same trusted sources.")
    parser.add_argument("--balance-labels", action="store_true", help="Down-sample labels to the smallest label count.")
    parser.add_argument("--max-per-label", type=int, default=0, help="Optional cap used with --balance-labels.")
    parser.add_argument("--allow-duplicate-questions", action="store_true", help="Keep duplicated verified questions.")
    parser.add_argument("--export-cases-only", action="store_true", help="Write generated cases without running predictions.")
    parser.add_argument("--prefix", default="risk_verified_eval")
    return parser.parse_args()


async def main() -> int:
    args = parse_args()
    cases = build_cases(args)
    if not cases:
        print("No evaluation cases generated. Lower --min-confidence or inspect source files.")
        return 1

    if args.export_cases_only:
        rows = [{**asdict(case)} for case in cases]
        summary = {
            "mode": "case_export",
            "case_count": len(rows),
            "source_counts": _count_by(rows, "source"),
            "rule_counts": _count_by(rows, "transform_rule"),
            "label_counts": _count_by(rows, "expected_risk_level"),
            "unique_questions": len({row["question"] for row in rows}),
            "unique_target_clauses": len({row["target_clause"] for row in rows}),
            "classification": {"not_evaluated": True},
            "score": {"band_accuracy": 0.0, "mae_to_band_midpoint": 0.0},
            "legal_grounding": {
                "evaluated_cases": sum(1 for row in rows if row["requires_law_reference"]),
                "law_name_hit_rate": 0.0,
                "article_exact_hit_rate": 0.0,
                "citation_precision": 0.0,
                "citation_recall": 0.0,
                "missing_required_reference_rate": 0.0,
            },
            "strict_pass_rate": 0.0,
            "latency_ms": {"avg": 0.0, "max": 0.0},
        }
        json_path, csv_path = save_outputs(summary, rows, prefix=f"{args.prefix}_cases")
        print_summary(summary, json_path, csv_path)
        return 0

    predictions = await run_predictions(cases, mock=args.mock, concurrency=args.concurrency)
    summary, rows = evaluate(cases, predictions)
    json_path, csv_path = save_outputs(summary, rows, prefix=args.prefix)
    print_summary(summary, json_path, csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
