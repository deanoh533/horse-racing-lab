"""경주 전에 알 수 있는 정보만으로 특징을 만든다.

핵심 원칙: 어떤 경주의 특징에는 그 경주 '이전' 기록만 쓴다 (미래 정보 누설 금지).
  - 말 기록: 같은 말의 이전 출전들
  - 기수·조교사 승률: 경주일 '전날까지'의 누적 (같은 날 다른 경주 결과도 쓰지 않음)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# 베이지안 평활: 출전이 적은 말·기수의 승률이 0%나 100%로 튀지 않게 사전값 쪽으로 당긴다
HORSE_PRIOR_STRENGTH = 3.0
PEOPLE_PRIOR_STRENGTH = 20.0


def _smoothed(successes: pd.Series, trials: pd.Series, prior: float, strength: float) -> pd.Series:
    return (successes + prior * strength) / (trials + strength)


def build_features(df: pd.DataFrame, use_odds: bool = False) -> tuple[pd.DataFrame, list[str]]:
    """표준 형식 데이터 → (특징이 붙은 데이터, 특징 이름 목록)."""
    d = df.copy()
    sort_cols = ["date", "race_id"] + (["race_no"] if "race_no" in d else [])
    d = d.sort_values(sort_cols, kind="stable").reset_index(drop=True)

    d["field_size"] = d.groupby("race_id")["horse_id"].transform("size")
    d["won"] = (d["rank"] == 1).astype(float)
    d["top3"] = (d["rank"] <= 3).astype(float)
    # 착순 백분위: 0 = 1위, 1 = 꼴찌. 완주 못 하면 꼴찌로 본다
    d["rank_pct"] = ((d["rank"].fillna(d["field_size"]) - 1) / (d["field_size"] - 1).clip(lower=1)).clip(0, 1)

    base_win = 1.0 / d["field_size"].mean()
    feats: list[str] = []

    # ---- 말의 과거 기록 (shift(1)로 이번 경주 제외) ----
    g = d.groupby("horse_id", sort=False)
    starts = g.cumcount()
    prev_wins = g["won"].cumsum() - d["won"]
    prev_top3 = g["top3"].cumsum() - d["top3"]
    d["h_first_start"] = (starts == 0).astype(float)
    d["h_starts_log"] = np.log1p(starts)
    d["h_win_rate"] = _smoothed(prev_wins, starts, base_win, HORSE_PRIOR_STRENGTH)
    d["h_top3_rate"] = _smoothed(prev_top3, starts, min(1.0, 3 * base_win), HORSE_PRIOR_STRENGTH)
    d["h_rank_pct_last5"] = g["rank_pct"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean()).fillna(0.5)
    d["h_last_rank_pct"] = g["rank_pct"].shift(1).fillna(0.5)
    days = (d["date"] - g["date"].shift(1)).dt.days
    d["h_days_since_log"] = np.log1p(days.fillna(days.median() if days.notna().any() else 30))
    feats += ["h_first_start", "h_starts_log", "h_win_rate", "h_top3_rate", "h_rank_pct_last5", "h_last_rank_pct", "h_days_since_log"]

    # 속도 지수: 우승마 대비 1000m당 몇 초 늦었는지 (최근 3경주 평균, 클수록 느림)
    if "finish_time" in d and d["finish_time"].notna().any() and "distance" in d:
        winner_time = d.groupby("race_id")["finish_time"].transform("min")
        d["behind_per_km"] = ((d["finish_time"] - winner_time) / d["distance"] * 1000).clip(0, 10)
        d["h_behind_last3"] = (
            d.groupby("horse_id", sort=False)["behind_per_km"]
            .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
            .fillna(d["behind_per_km"].median())
        )
        feats.append("h_behind_last3")

    # ---- 기수·조교사: 전날까지 누적 승률 ----
    for col, name in (("jockey", "jk"), ("trainer", "tr")):
        if col not in d:
            continue
        daily = d.groupby([col, "date"])[["won"]].agg(wins=("won", "sum"), rides=("won", "size")).reset_index()
        daily = daily.sort_values([col, "date"])
        daily["prev_wins"] = daily.groupby(col)["wins"].cumsum() - daily["wins"]
        daily["prev_rides"] = daily.groupby(col)["rides"].cumsum() - daily["rides"]
        d = d.merge(daily[[col, "date", "prev_wins", "prev_rides"]], on=[col, "date"], how="left")
        d[f"{name}_win_rate"] = _smoothed(d["prev_wins"], d["prev_rides"], base_win, PEOPLE_PRIOR_STRENGTH)
        d = d.drop(columns=["prev_wins", "prev_rides"])
        feats.append(f"{name}_win_rate")

    # ---- 경주 내 상대값 (같은 경주 말들과 비교) ----
    def rel(col: str) -> pd.Series:
        v = d[col]
        return (v - d.groupby("race_id")[col].transform("mean")).fillna(0)

    for col, name in (("burden_weight", "burden_rel"), ("rating", "rating_rel"), ("age", "age_rel")):
        if col in d and d[col].notna().any():
            d[name] = rel(col)
            feats.append(name)
    if "body_weight_change" in d and d["body_weight_change"].notna().any():
        d["body_weight_change_abs"] = d["body_weight_change"].abs().fillna(0)
        feats.append("body_weight_change_abs")
    if "gate" in d and d["gate"].notna().any():
        d["gate_pct"] = ((d["gate"] - 1) / (d["field_size"] - 1).clip(lower=1)).fillna(0.5)
        feats.append("gate_pct")

    # ---- 시장 평가 (단승 배당) ----
    if use_odds:
        if "odds" not in d or d["odds"].isna().all():
            raise ValueError("use_odds=True인데 배당(odds) 열이 없습니다.")
        inv = 1.0 / d["odds"]
        inv = inv.fillna(inv.groupby(d["race_id"]).transform("min"))
        d["market_log_prob"] = np.log(inv / inv.groupby(d["race_id"]).transform("sum"))
        feats.append("market_log_prob")

    return d, feats
