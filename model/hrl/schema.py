"""표준 데이터 형식. 데이터 출처가 달라도 이 형식으로 바꾼 뒤 학습한다.

한 행 = 한 경주에 출전한 말 한 마리.
"""

# 필수 열
REQUIRED = [
    "race_id",  # 경주 고유 id (예: "서울-2024-01-06-3")
    "date",  # 경주일 (datetime64)
    "horse_id",  # 마번 또는 마명
    "rank",  # 착순. 기권·실격·취소는 NaN
]

# 있으면 특징으로 쓰는 열
OPTIONAL = [
    "meet",  # 경마장 (서울/부산경남/제주)
    "race_no",  # 경주 번호
    "distance",  # 경주 거리 (m)
    "track_condition",  # 주로 상태
    "horse_name",
    "gate",  # 출전 번호 (게이트)
    "jockey",  # 기수
    "trainer",  # 조교사
    "age",
    "sex",  # 수/암/거
    "burden_weight",  # 부담 중량 (kg)
    "body_weight",  # 마체중 (kg)
    "body_weight_change",  # 마체중 증감 (kg)
    "rating",  # 레이팅
    "odds",  # 단승식 배당률 (경주 직전 시장 평가)
    "finish_time",  # 경주 기록 (초)
]

COLUMNS = REQUIRED + OPTIONAL
