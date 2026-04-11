import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class QAItem:
    question_id: str
    question: str
    ground_truth: str
    relevant_doc_ids: List[str]
    category: str
    difficulty: Optional[str] = None
    expected_keywords: List[str] = field(default_factory=list)
    is_trap: bool = False
    is_colloquial: bool = False


class TestDataset:
    # 프로젝트 루트 기준 tests 폴더를 사용
    DATASET_PATH = Path(__file__).resolve().parents[2] / "tests" / "eval_dataset.json"

    @classmethod
    def _load_dataset(cls) -> List[dict]:
        if not cls.DATASET_PATH.exists():
            raise FileNotFoundError(f"Test dataset not found: {cls.DATASET_PATH}")

        with open(cls.DATASET_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        return data.get("questions", [])

    @classmethod
    def _to_qa_item(cls, raw: dict) -> QAItem:
        return QAItem(
            question_id=raw.get("id", ""),
            question=raw.get("question", ""),
            ground_truth=raw.get("expected_answer", ""),
            relevant_doc_ids=raw.get("relevant_law", []) or [],
            category=raw.get("category", ""),
            difficulty=raw.get("difficulty", None),
            expected_keywords=raw.get("expected_keywords", []) or [],
            is_trap=raw.get("is_trap", False),
            is_colloquial=raw.get("is_colloquial", False),
        )

    @classmethod
    def get_baseline_questions(cls) -> List[QAItem]:
        return [cls._to_qa_item(q) for q in cls._load_dataset()]

    @classmethod
    def get_sample_batch(cls, n: int) -> List[QAItem]:
        return cls.get_baseline_questions()[:n]

    @classmethod
    def print_statistics(cls) -> None:
        questions = cls._load_dataset()
        total = len(questions)
        categories = {}
        difficulties = {}

        for q in questions:
            categories[q.get("category", "unknown")] = categories.get(q.get("category", "unknown"), 0) + 1
            diff = q.get("difficulty")
            if diff:
                difficulties[diff] = difficulties.get(diff, 0) + 1

        print(f"총 질문 수: {total}")
        print("카테고리 분포:", ", ".join(f"{k}:{v}" for k, v in categories.items()))
        if difficulties:
            print("난이도 분포:", ", ".join(f"{k}:{v}" for k, v in difficulties.items()))
