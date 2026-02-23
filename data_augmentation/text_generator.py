"""
가상 텍스트 데이터 생성
- Faker 기반 한국어 데이터
- 계약서 특화 데이터 (금액, 날짜, 주소 등)
"""
import random
from typing import Dict, List, Optional
from dataclasses import dataclass
from faker import Faker


@dataclass
class GeneratedText:
    """생성된 텍스트 정보"""
    text: str
    field_type: str  # name, amount, date, address, phone, etc.
    metadata: Dict


class TextGenerator:
    """계약서 텍스트 생성기"""

    def __init__(self, locale: str = "ko_KR"):
        self.fake = Faker(locale)
        Faker.seed(42)

        # 한글 금액 단위
        self.korean_units = [
            "", "일", "이", "삼", "사", "오", "육", "칠", "팔", "구"
        ]
        self.korean_positions = [
            "", "십", "백", "천", "만", "십만", "백만", "천만", "억"
        ]

    def generate_name(self) -> GeneratedText:
        """한국 이름 생성"""
        name = self.fake.name()
        return GeneratedText(
            text=name,
            field_type="name",
            metadata={"full_name": name}
        )

    def generate_phone(self) -> GeneratedText:
        """전화번호 생성"""
        phone = self.fake.phone_number()
        return GeneratedText(
            text=phone,
            field_type="phone",
            metadata={"raw": phone}
        )

    def generate_address(self) -> GeneratedText:
        """주소 생성"""
        address = self.fake.address()
        return GeneratedText(
            text=address,
            field_type="address",
            metadata={"raw": address}
        )

    def generate_date(self, format: str = "%Y년 %m월 %d일") -> GeneratedText:
        """날짜 생성"""
        date = self.fake.date_this_decade()
        formatted = date.strftime(format)
        return GeneratedText(
            text=formatted,
            field_type="date",
            metadata={"date": str(date), "format": format}
        )

    def generate_amount(
        self,
        min_amount: int = 100000,
        max_amount: int = 1000000000,
        style: str = "korean"  # "korean", "numeric", "mixed"
    ) -> GeneratedText:
        """
        금액 생성

        Args:
            min_amount: 최소 금액
            max_amount: 최대 금액
            style: "korean" (일천만원), "numeric" (10,000,000원), "mixed" (1천만원)
        """
        amount = random.randint(min_amount // 10000, max_amount // 10000) * 10000

        if style == "korean":
            text = self._to_korean_amount(amount)
        elif style == "numeric":
            text = f"{amount:,}원"
        else:  # mixed
            text = self._to_mixed_amount(amount)

        return GeneratedText(
            text=text,
            field_type="amount",
            metadata={"numeric": amount, "style": style}
        )

    def _to_korean_amount(self, amount: int) -> str:
        """숫자를 한글 금액으로 변환"""
        if amount == 0:
            return "영원"

        units = ["", "만", "억", "조"]
        result = []

        for i, unit in enumerate(units):
            part = (amount // (10000 ** i)) % 10000
            if part > 0:
                part_str = self._number_to_korean(part)
                result.append(part_str + unit)

        return "".join(reversed(result)) + "원정"

    def _number_to_korean(self, n: int) -> str:
        """4자리 이하 숫자를 한글로"""
        if n == 0:
            return ""

        korean_nums = ["", "일", "이", "삼", "사", "오", "육", "칠", "팔", "구"]
        positions = ["", "십", "백", "천"]

        result = []
        s = str(n).zfill(4)

        for i, digit in enumerate(s):
            d = int(digit)
            pos = 3 - i
            if d > 0:
                if d == 1 and pos > 0:  # 일십, 일백 -> 십, 백
                    result.append(positions[pos])
                else:
                    result.append(korean_nums[d] + positions[pos])

        return "".join(result)

    def _to_mixed_amount(self, amount: int) -> str:
        """혼합 형식 (예: 1천만원)"""
        if amount >= 100000000:  # 억 단위
            억 = amount // 100000000
            나머지 = amount % 100000000
            if 나머지 > 0:
                return f"{억}억 {나머지 // 10000}만원"
            return f"{억}억원"
        elif amount >= 10000:  # 만 단위
            만 = amount // 10000
            return f"{만}만원"
        else:
            return f"{amount:,}원"

    def generate_resident_id(self, masked: bool = True) -> GeneratedText:
        """
        주민등록번호 생성

        Args:
            masked: True면 뒷자리 마스킹 (123456-1******)
        """
        # 앞자리: 생년월일
        birth = self.fake.date_of_birth(minimum_age=20, maximum_age=70)
        front = birth.strftime("%y%m%d")

        # 뒷자리: 성별 + 랜덤
        gender = random.choice(["1", "2", "3", "4"])  # 1,2: 1900년대, 3,4: 2000년대
        back = gender + "".join([str(random.randint(0, 9)) for _ in range(6)])

        if masked:
            text = f"{front}-{gender}******"
        else:
            text = f"{front}-{back}"

        return GeneratedText(
            text=text,
            field_type="resident_id",
            metadata={"front": front, "masked": masked}
        )

    def generate_contract_period(self) -> GeneratedText:
        """계약 기간 생성"""
        years = random.choice([1, 2])
        start = self.fake.date_this_year()
        end_year = start.year + years

        text = f"{start.strftime('%Y년 %m월 %d일')}부터 {end_year}년 {start.strftime('%m월 %d일')}까지"
        return GeneratedText(
            text=text,
            field_type="period",
            metadata={"years": years, "start": str(start)}
        )

    def generate_batch(
        self,
        field_types: List[str],
        count: int = 100
    ) -> List[GeneratedText]:
        """
        배치 생성

        Args:
            field_types: 생성할 필드 타입 리스트
            count: 각 타입별 생성 개수
        """
        generators = {
            "name": self.generate_name,
            "phone": self.generate_phone,
            "address": self.generate_address,
            "date": self.generate_date,
            "amount": self.generate_amount,
            "resident_id": self.generate_resident_id,
            "period": self.generate_contract_period,
        }

        results = []
        for field_type in field_types:
            if field_type in generators:
                for _ in range(count):
                    results.append(generators[field_type]())

        return results


if __name__ == "__main__":
    gen = TextGenerator()

    print("=== 이름 ===")
    print(gen.generate_name().text)

    print("\n=== 금액 (한글) ===")
    print(gen.generate_amount(style="korean").text)

    print("\n=== 금액 (숫자) ===")
    print(gen.generate_amount(style="numeric").text)

    print("\n=== 주민등록번호 ===")
    print(gen.generate_resident_id().text)

    print("\n=== 계약 기간 ===")
    print(gen.generate_contract_period().text)

    print("\nTextGenerator 모듈 로드 완료")
