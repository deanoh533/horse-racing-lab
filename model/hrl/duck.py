"""사용자 DuckDB(race_entries·races 테이블) → 표준 형식.

race_entries 한 행 = 한 경주에 나온 말 한 마리. 순위(ord)와 기록(rc_time)이 모두 없는 행은
출전 취소·제외로 보고 뺀다 (출전 두수·배당 정규화에 넣지 않기 위해).
결과가 하나도 없는 경주(예정 경주)는 include_upcoming=True일 때만 포함한다.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .schema import COLUMNS

MEET_NAMES = {1: "서울", 2: "제주", 3: "부산경남", 4: "영천"}

QUERY = """
with e as (
    select *,
           count(ord) over (partition by race_date, meet, rc_no) as n_results
    from race_entries
)
select
    e.race_date, e.meet, e.rc_no,
    coalesce(e.rc_dist, r.rc_dist) as distance,
    r.track as track_condition,
    e.hr_no as horse_id, e.hr_name as horse_name,
    e.pthr_no as gate,
    e.jcky_nm as jockey, e.trar_nm as trainer,
    e.ag as age, e.gndr as sex,
    e.burd_wgt as burden_weight,
    e.wg_hr as body_weight, cast(e.wg_hr_diff as double) as body_weight_change,
    e.ratg as rating,
    e.win_odds as odds,
    e.ord as rank,
    e.rc_time as finish_time,
    -- 서울은 se_/sj_, 부산경남·영천은 bu_ 열에 구간 기록이 있다
    e.rc_time - coalesce(e.se_g1f_acc_time, e.bu_g1f_acc_time) as late_200m,
    coalesce(e.sj_s1f_ord, e.bu_s1f_ord) as early_position,
    e.popularity,
    try_cast(r.chaksun1 as double) as purse,
    e.n_results
from e
left join races r using (race_date, meet, rc_no)
where (e.n_results > 0 and (e.ord is not null or e.rc_time is not null))
   or (e.n_results = 0 and $include_upcoming)
"""


def load_duckdb(path: str | Path, include_upcoming: bool = False) -> pd.DataFrame:
    import duckdb  # 선택 의존성: DuckDB 데이터를 쓸 때만 필요

    with duckdb.connect(str(path), read_only=True) as con:
        df = con.execute(QUERY, {"include_upcoming": include_upcoming}).df()

    df["meet"] = df["meet"].map(MEET_NAMES).fillna(df["meet"].astype(str))
    df["date"] = pd.to_datetime(df["race_date"].astype(str), format="%Y%m%d")
    df["race_id"] = df["meet"] + "-" + df["date"].dt.strftime("%Y-%m-%d") + "-" + df["rc_no"].astype(str)
    df = df.rename(columns={"rc_no": "race_no"})
    df.loc[df["rank"] <= 0, "rank"] = None
    df.loc[df["odds"] <= 0, "odds"] = None
    df.loc[df["finish_time"] <= 0, "finish_time"] = None
    # 계측 오류로 보이는 구간 기록은 버린다 (정상 범위 약 11~20초)
    df.loc[~df["late_200m"].between(10, 20), "late_200m"] = None
    df.loc[df["purse"] <= 0, "purse"] = None
    df["horse_id"] = df["horse_id"].astype(str)
    # 기수명 앞의 감량 표시 "(-1)김성현" → "김성현" (같은 기수를 하나로 집계)
    df["jockey"] = df["jockey"].str.replace(r"^\([-+]?\d+(?:\.\d+)?\)", "", regex=True).str.strip()
    return df[[c for c in COLUMNS if c in df]].reset_index(drop=True)
