"""간단한 평가 테스트 (Mock LLM 사용)"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.evaluation.test_dataset import TestDataset


async def main():
    """메인 실행"""
    print("\n🚀 Loading Test Dataset from Excel...\n")
    
    # 데이터셋 통계 출력
    TestDataset.print_statistics()
    
    # 첫 5개 질문 출력
    questions = TestDataset.get_baseline_questions()
    print(f"\n📌 First 5 Questions:\n")
    for i, q in enumerate(questions[:5], 1):
        print(f"{i}. [{q.question_id}] {q.question}")
        print(f"   Category: {q.category} | Difficulty: {q.difficulty}")
        print(f"   Ground Truth: {q.ground_truth[:100] if q.ground_truth else 'N/A'}...")
        print()


if __name__ == "__main__":
    asyncio.run(main())
