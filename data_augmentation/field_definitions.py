"""
계약서 필드 정의 모듈
- 법원 부동산 임대차 계약서 필드 좌표
- 기타 계약서 템플릿 필드 정의
"""
from .models import FieldConfig, FieldDefinition


# ============================================================
# 법원 부동산 임대차 계약서 필드 정의 (1240x1753, DPI 150)
# 1페이지 기준 좌표
# x,y,width,height 단위: 픽셀
# ============================================================

COURT_LEASE_FIELDS = [
    # 1. 부동산의 표시 섹션 (소 재 지 행) - 녹색 표 영역
    FieldConfig("소재지", "address", 212, 245, 922, 54, font_size=20),

    # 토지 행 (지목, 면적)
    FieldConfig("토지_지목", "text", 118, 158, 120, 30, font_size=20),
    FieldConfig("토지_면적", "area", 465, 158, 80, 30, font_size=20),

    # 건물 행 (구조용도, 면적)
    FieldConfig("건물_구조용도", "structure", 118, 184, 200, 30, font_size=20),
    FieldConfig("건물_면적", "area", 465, 184, 80, 30, font_size=20),

    # 임대할부분 (면적)
    FieldConfig("임대부분_면적", "area", 465, 210, 80, 30, font_size=20),

    # 2. 계약내용 섹션 - 금액 표 (노란색 표 영역)
    # 보 증 금 행
    FieldConfig("보증금_한글", "amount", 118, 258, 300, 30, font_size=20),
    FieldConfig("보증금_숫자", "amount_numeric", 430, 258, 170, 30, font_size=20),

    # 계 약 금 행
    FieldConfig("계약금_한글", "amount", 118, 286, 280, 30, font_size=20),

    # 중 도 금 행
    FieldConfig("중도금_한글", "amount", 118, 314, 220, 30, font_size=20),
    FieldConfig("중도금_년", "year", 355, 314, 50, 30, font_size=20),
    FieldConfig("중도금_월", "month", 430, 314, 35, 30, font_size=20),
    FieldConfig("중도금_일", "day", 500, 314, 35, 30, font_size=20),

    # 잔 금 행
    FieldConfig("잔금_한글", "amount", 118, 342, 220, 30, font_size=20),
    FieldConfig("잔금_년", "year", 355, 342, 50, 30, font_size=20),
    FieldConfig("잔금_월", "month", 430, 342, 35, 30, font_size=20),
    FieldConfig("잔금_일", "day", 500, 342, 35, 30, font_size=20),

    # 차 임 행
    FieldConfig("차임_한글", "amount_small", 118, 370, 220, 30, font_size=20),
    FieldConfig("차임_지불일", "day", 480, 370, 35, 30, font_size=20),

    # 제2조 존속기간 (본문 내 빈칸)
    FieldConfig("인도_년", "year", 270, 398, 50, 25, font_size=20),
    FieldConfig("인도_월", "month", 345, 398, 35, 25, font_size=20),
    FieldConfig("인도_일", "day", 405, 398, 35, 25, font_size=20),
    FieldConfig("계약종료_년", "year", 540, 398, 50, 25, font_size=20),
    FieldConfig("계약종료_월", "month", 615, 398, 35, 25, font_size=20),
    FieldConfig("계약종료_일", "day", 675, 398, 35, 25, font_size=20),

    # 제9조 중개수수료율
    FieldConfig("수수료율", "percentage", 168, 535, 40, 25, font_size=20),

    # 중개대상물확인설명서 교부일
    FieldConfig("교부_년", "year", 315, 550, 50, 25, font_size=20),
    FieldConfig("교부_월", "month", 390, 550, 35, 25, font_size=20),
    FieldConfig("교부_일", "day", 450, 550, 35, 25, font_size=20),

    # 계약 체결일
    FieldConfig("계약_년", "year", 390, 630, 55, 35, font_size=20),
    FieldConfig("계약_월", "month", 490, 630, 40, 35, font_size=20),
    FieldConfig("계약_일", "day", 565, 630, 40, 35, font_size=20),

    # 임대인 정보
    FieldConfig("임대인_주소", "address", 175, 673, 380, 30, font_size=20),
    FieldConfig("임대인_주민등록번호", "resident_id", 175, 700, 160, 30, font_size=20),
    FieldConfig("임대인_전화", "phone", 430, 700, 130, 30, font_size=20),
    FieldConfig("임대인_성명", "name", 590, 700, 100, 30, font_size=20),

    # 임대인 대리인
    FieldConfig("임대인대리인_주소", "address", 310, 728, 250, 30, font_size=20),
    FieldConfig("임대인대리인_주민등록번호", "resident_id", 430, 728, 130, 30, font_size=20),
    FieldConfig("임대인대리인_성명", "name", 590, 728, 100, 30, font_size=20),

    # 임차인 정보
    FieldConfig("임차인_주소", "address", 175, 765, 380, 30, font_size=20),
    FieldConfig("임차인_주민등록번호", "resident_id", 175, 793, 160, 30, font_size=20),
    FieldConfig("임차인_전화", "phone", 430, 793, 130, 30, font_size=20),
    FieldConfig("임차인_성명", "name", 590, 793, 100, 30, font_size=20),

    # 임차인 대리인
    FieldConfig("임차인대리인_주소", "address", 310, 820, 250, 30, font_size=20),
    FieldConfig("임차인대리인_주민등록번호", "resident_id", 430, 820, 130, 30, font_size=20),
    FieldConfig("임차인대리인_성명", "name", 590, 820, 100, 30, font_size=20),

    # 중개업자 1 (좌측)
    FieldConfig("중개1_소재지", "address", 148, 863, 220, 30, font_size=20),
    FieldConfig("중개1_명칭", "company", 148, 890, 180, 30, font_size=20),
    FieldConfig("중개1_대표", "name", 148, 917, 100, 30, font_size=20),
    FieldConfig("중개1_등록번호", "business_number", 148, 944, 130, 30, font_size=20),
    FieldConfig("중개1_전화", "phone", 310, 944, 120, 30, font_size=20),
    FieldConfig("중개1_소속공인", "name", 148, 971, 100, 30, font_size=20),

    # 중개업자 2 (우측)
    FieldConfig("중개2_소재지", "address", 520, 863, 220, 30, font_size=20),
    FieldConfig("중개2_명칭", "company", 520, 890, 180, 30, font_size=20),
    FieldConfig("중개2_대표", "name", 580, 917, 100, 30, font_size=20),
    FieldConfig("중개2_등록번호", "business_number", 520, 944, 130, 30, font_size=20),
    FieldConfig("중개2_전화", "phone", 685, 944, 120, 30, font_size=20),
    FieldConfig("중개2_소속공인", "name", 580, 971, 100, 30, font_size=20),
]


# ============================================================
# pdf_template.py 호환용 필드 정의 (DPI 150 기준)
# ============================================================

COURT_LEASE_CONTRACT_FIELDS = [
    # 부동산 표시
    FieldDefinition("소재지", "address", 260, 235, 520, 34, font_size=14),
    FieldDefinition("토지_지목", "text", 210, 270, 120, 28, font_size=12),
    FieldDefinition("토지_면적", "area", 520, 270, 100, 28, font_size=12),
    FieldDefinition("건물_구조용도", "text", 210, 305, 240, 28, font_size=12),
    FieldDefinition("건물_면적", "area", 520, 305, 100, 28, font_size=12),
    FieldDefinition("임대할부분_면적", "area", 520, 340, 100, 28, font_size=12),

    # 계약 내용 - 금액
    FieldDefinition("보증금", "amount", 180, 430, 260, 30, font_size=14),
    FieldDefinition("보증금_숫자", "amount_numeric", 460, 430, 180, 30, font_size=14),
    FieldDefinition("계약금", "amount", 180, 470, 240, 28, font_size=13),
    FieldDefinition("중도금", "amount", 180, 510, 240, 28, font_size=13),
    FieldDefinition("중도금_일자", "date_short", 460, 510, 180, 28, font_size=13),
    FieldDefinition("잔금", "amount", 180, 550, 240, 28, font_size=13),
    FieldDefinition("잔금_일자", "date_short", 460, 550, 180, 28, font_size=13),
    FieldDefinition("차임", "amount", 180, 590, 240, 28, font_size=13),
    FieldDefinition("차임_지불일", "day", 560, 590, 40, 28, font_size=13),

    # 존속기간
    FieldDefinition("인도일", "date_short", 560, 630, 180, 28, font_size=13),
    FieldDefinition("계약종료일", "date_short", 380, 655, 180, 28, font_size=13),

    # 중개수수료
    FieldDefinition("중개수수료율", "percentage", 180, 920, 50, 26, font_size=12),
    FieldDefinition("확인설명서_교부일", "date_short", 270, 950, 180, 28, font_size=13),

    # 계약일
    FieldDefinition("계약년", "year", 400, 990, 60, 30, font_size=14),
    FieldDefinition("계약월", "month", 480, 990, 40, 30, font_size=14),
    FieldDefinition("계약일", "day", 540, 990, 40, 30, font_size=14),

    # 임대인 정보
    FieldDefinition("임대인_주소", "address", 190, 1040, 380, 28, font_size=12),
    FieldDefinition("임대인_주민번호", "resident_id", 190, 1080, 180, 28, font_size=12),
    FieldDefinition("임대인_전화", "phone", 400, 1080, 150, 28, font_size=12),
    FieldDefinition("임대인_성명", "name", 580, 1080, 100, 28, font_size=14),

    # 임차인 정보
    FieldDefinition("임차인_주소", "address", 190, 1140, 380, 28, font_size=12),
    FieldDefinition("임차인_주민번호", "resident_id", 190, 1180, 180, 28, font_size=12),
    FieldDefinition("임차인_전화", "phone", 400, 1180, 150, 28, font_size=12),
    FieldDefinition("임차인_성명", "name", 580, 1180, 100, 28, font_size=14),

    # 중개업자 정보
    FieldDefinition("중개업자1_소재지", "address", 160, 1245, 280, 24, font_size=10),
    FieldDefinition("중개업자1_명칭", "company", 160, 1280, 200, 24, font_size=11),
    FieldDefinition("중개업자1_대표", "name", 160, 1315, 100, 24, font_size=11),
    FieldDefinition("중개업자1_등록번호", "business_number", 160, 1350, 150, 24, font_size=10),
    FieldDefinition("중개업자1_전화", "phone", 330, 1350, 130, 24, font_size=10),
]


# ============================================================
# 특약사항 영역 설정
# ============================================================

SPECIAL_TERMS_AREA = {
    "x": 75,
    "y": 575,
    "width": 650,
    "font_size": 20,
    "line_spacing": 3,
    "clause_spacing": 5,
}
