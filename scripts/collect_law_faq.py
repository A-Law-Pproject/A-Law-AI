"""
법령 AI FAQ 임대차 데이터 수집 스크립트
대상: https://www.law.go.kr/LSW/ais/faq.do?aiAstCd=140803&faqQuery=&pageIndex=1~12

HTML 구조 (개발자도구 확인):
  - 질문: div.question > a[onclick="getAnswer(n,this)"] > div.title
  - 답변: div.answer > div.text-cont  (JS 클릭 후 동적 로딩)
  - 관련법령: span.list-tit > strong (법령명)
             span.list-num (조문명)
             span.list-sum (조문내용)

실행 방법:
  pip install selenium webdriver-manager beautifulsoup4
  python collect_law_faq.py

결과: lease_faq.jsonl
"""

import json
import time
import re
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

BASE_URL = "https://www.law.go.kr/LSW/ais/faq.do?aiAstCd=140803&faqQuery=&pageIndex={page}"

LEASE_KEYWORDS = [
    "임대차", "임차", "임대인", "임차인", "전세", "월세", "보증금",
    "계약갱신", "갱신청구권", "묵시적갱신", "확정일자", "전입신고",
    "대항력", "우선변제", "최우선변제", "임차권등기", "퇴거",
    "명도", "원상복구", "수선의무", "차임", "연체", "해지",
    "계약해제", "계약해지", "임대료", "주택임대차", "상가임대차",
    "권리금", "재계약", "묵시", "갱신거절",
]


def is_lease_related(text: str) -> bool:
    return any(kw in text for kw in LEASE_KEYWORDS)


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip().strip('"').strip("'").strip()


def setup_driver() -> webdriver.Chrome:
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=options)


def parse_related_laws(answer_soup) -> list:
    """
    관련 법령 파싱
    <span class="list-tit"><strong>민법</strong><em>[시행...]</em></span>
    <span class="list-num">차임증감청구권</span>
    <span class="list-sum">제628조(차임증감청구권) ...</span>
    """
    laws = []
    for li in answer_soup.select("li"):
        law = {}
        law_name_el = li.select_one("span.list-tit strong")
        law_date_el  = li.select_one("span.list-tit em")
        article_name_el = li.select_one("span.list-num")
        article_content_el = li.select_one("span.list-sum")

        if law_name_el:
            law["법령명"] = clean_text(law_name_el.get_text())
        if law_date_el:
            law["시행일"] = clean_text(law_date_el.get_text())
        if article_name_el:
            law["조문명"] = clean_text(article_name_el.get_text())
        if article_content_el:
            law["조문내용"] = clean_text(article_content_el.get_text())

        if law:
            laws.append(law)
    return laws


def scrape_page(driver: webdriver.Chrome, page: int) -> list:
    url = BASE_URL.format(page=page)
    driver.get(url)
    time.sleep(2.5)

    soup = BeautifulSoup(driver.page_source, "html.parser")
    question_links = driver.find_elements(By.CSS_SELECTOR, "div.question a")
    total = len(question_links)
    print(f"  질문 수: {total}개")

    items = []

    for idx in range(total):
        try:
            # 매 클릭 후 DOM이 바뀌므로 재탐색
            links = driver.find_elements(By.CSS_SELECTOR, "div.question a")
            link = links[idx]

            title_el = link.find_element(By.CSS_SELECTOR, "div.title")
            question_text = clean_text(title_el.text)

            if not question_text:
                continue

            # 1차 필터 (질문 텍스트)
            if not is_lease_related(question_text):
                print(f"    - [{idx+1}] 스킵 (비임대차): {question_text[:30]}...")
                continue

            # 클릭 → 답변 동적 로딩
            driver.execute_script("arguments[0].scrollIntoView(true);", link)
            driver.execute_script("arguments[0].click();", link)
            time.sleep(1.8)

            page_soup = BeautifulSoup(driver.page_source, "html.parser")

            # 현재 idx번째 answer div
            answer_divs = page_soup.select("div.answer")
            answer_text = ""
            related_laws = []

            if idx < len(answer_divs):
                answer_div = answer_divs[idx]
                text_cont = answer_div.select_one("div.text-cont")

                if text_cont:
                    # 관련 법령 먼저 파싱 (decompose 전)
                    related_laws = parse_related_laws(text_cont)

                    # 법령 리스트 제거 후 순수 답변 텍스트 추출
                    for el in text_cont.select("ul, ol"):
                        el.decompose()

                    answer_text = clean_text(text_cont.get_text())

            # 2차 필터 (질문+답변 합산)
            if not is_lease_related(question_text + " " + answer_text):
                print(f"    - [{idx+1}] 스킵 (답변도 비임대차)")
                continue

            item = {
                "question": question_text,
                "answer": answer_text,
                "related_laws": related_laws,
                "source": "법령AI FAQ",
                "url": f"https://www.law.go.kr/LSW/ais/faq.do?aiAstCd=140803&faqQuery=&pageIndex={page}",
                "page": page,
                "idx": idx + 1,
            }
            items.append(item)
            print(f"    ✓ [{idx+1}] {question_text[:45]}...")

        except Exception as e:
            print(f"    ✗ [{idx+1}] 오류: {e}")

    return items


def main():
    print("=== 법령AI FAQ 임대차 데이터 수집 시작 ===\n")
    driver = setup_driver()
    all_items = []

    try:
        for page in range(1, 13):
            print(f"\n[페이지 {page}/12]")
            items = scrape_page(driver, page)
            all_items.extend(items)
            print(f"  → 임대차 관련 {len(items)}개")
            time.sleep(2)
    finally:
        driver.quit()

    # 저장
    output = "lease_faq.jsonl"
    with open(output, "w", encoding="utf-8") as f:
        for item in all_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n=== 완료 ===")
    print(f"총 수집: {len(all_items)}개 → {output}")

    # 샘플 출력
    print("\n--- 샘플 (첫 2개) ---")
    for item in all_items[:2]:
        print(json.dumps(item, ensure_ascii=False, indent=2))
        print()


if __name__ == "__main__":
    main()