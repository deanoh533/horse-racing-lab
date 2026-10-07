"""한국마사회(KRA) 경주성적 데이터 → 표준 형식 변환·수집.

공공데이터포털(data.go.kr)의 한국마사회 경주성적 데이터는 두 가지로 받을 수 있다.
  1) 파일 데이터(CSV) 내려받기 → load_kra_csv()
  2) 오픈 API(JSON) → fetch_kra_api()  (서비스키 필요)

열 이름은 제공 형태(API 영문 키 / CSV 한글 헤더)마다 다르므로 아래 COLUMN_ALIASES에
알려진 이름을 모두 등록해 두고, 처음 나오는 것을 쓴다.
※ 실제 데이터로 한 번 돌려 보고 매핑되지 않은 열이 있으면 여기에 추가한다.
"""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import COLUMNS, REQUIRED

# 표준 열 → 원본에서 쓰일 수 있는 이름들
COLUMN_ALIASES: dict[str, list[str]] = {
    "meet": ["meet", "경마장", "시행경마장명", "경마장명"],
    "date": ["rcDate", "rc_date", "경주일자", "경주일"],
    "race_no": ["rcNo", "rc_no", "경주번호"],
    "distance": ["rcDist", "rc_dist", "경주거리", "거리"],
    "track_condition": ["track", "주로상태", "주로"],
    "horse_id": ["hrNo", "hr_no", "마번", "마필번호"],
    "horse_name": ["hrName", "hr_name", "마명"],
    "gate": ["chulNo", "chul_no", "출전번호", "출주번호", "게이트"],
    "jockey": ["jkName", "jk_name", "기수명", "기수"],
    "trainer": ["trName", "tr_name", "조교사명", "조교사"],
    "age": ["age", "연령", "나이"],
    "sex": ["sex", "성별"],
    "burden_weight": ["wgBudam", "wg_budam", "부담중량"],
    "body_weight": ["wgHr", "wg_hr", "마체중"],
    "rating": ["rating", "레이팅"],
    "odds": ["winOdds", "win_odds", "단승식배당율", "단승배당", "단승식배당률"],
    "rank": ["ord", "rcOrd", "순위", "착순"],
    "finish_time": ["rcTime", "rc_time", "경주기록", "기록"],
}


def _pick(raw: pd.DataFrame, names: list[str]) -> pd.Series | None:
    for n in names:
        if n in raw.columns:
            return raw[n]
    return None


def parse_body_weight(value) -> tuple[float, float]:
    """'470(+2)' / '470(-5)' / '470' → (470.0, 2.0)"""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan, np.nan
    m = re.match(r"\s*(\d+(?:\.\d+)?)\s*(?:\(\s*([+-]?\d+(?:\.\d+)?)\s*\))?", str(value))
    if not m:
        return np.nan, np.nan
    return float(m.group(1)), float(m.group(2)) if m.group(2) is not None else np.nan


def parse_time(value) -> float:
    """'1:35.5' / '95.5' → 초"""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    s = str(value).strip()
    m = re.fullmatch(r"(\d+):(\d+(?:\.\d+)?)", s)
    if m:
        return int(m.group(1)) * 60 + float(m.group(2))
    try:
        t = float(s)
    except ValueError:
        return np.nan
    return t if t > 0 else np.nan


def parse_date(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace(r"[^0-9]", "", regex=True).str.slice(0, 8)
    return pd.to_datetime(s, format="%Y%m%d", errors="coerce")


def normalize_kra(raw: pd.DataFrame) -> pd.DataFrame:
    """마사회 원본 표 → 표준 형식."""
    out = pd.DataFrame(index=raw.index)
    for col, names in COLUMN_ALIASES.items():
        s = _pick(raw, names)
        if s is not None:
            out[col] = s

    missing = [c for c in ("date", "rank") if c not in out]
    if "horse_id" not in out and "horse_name" not in out:
        missing.append("horse_id/horse_name")
    if missing:
        raise ValueError(f"필수 열을 찾지 못했습니다: {missing}. 원본 열: {list(raw.columns)}")

    out["date"] = parse_date(out["date"])
    if "horse_id" not in out:
        out["horse_id"] = out["horse_name"]
    out["horse_id"] = out["horse_id"].astype(str).str.strip()

    if "body_weight" in out:
        parsed = out["body_weight"].map(parse_body_weight)
        out["body_weight"] = parsed.map(lambda t: t[0])
        out["body_weight_change"] = parsed.map(lambda t: t[1])
    if "finish_time" in out:
        out["finish_time"] = out["finish_time"].map(parse_time)

    for c in ("race_no", "distance", "gate", "age", "burden_weight", "rating", "odds", "rank"):
        if c in out:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    # 착순 0 이하는 기권·실격 등으로 보고 NaN
    out.loc[out["rank"] <= 0, "rank"] = np.nan
    if "odds" in out:
        out.loc[out["odds"] <= 0, "odds"] = np.nan

    meet = out["meet"].astype(str) if "meet" in out else "KRA"
    race_no = out["race_no"].astype("Int64").astype(str) if "race_no" in out else "0"
    out["race_id"] = meet + "-" + out["date"].dt.strftime("%Y-%m-%d") + "-" + race_no

    out = out.dropna(subset=["date"])
    return out[[c for c in COLUMNS if c in out]].reset_index(drop=True)


def load_kra_csv(paths: list[str | Path]) -> pd.DataFrame:
    """마사회 CSV 여러 개를 읽어 표준 형식으로 합친다. (공공데이터 CSV는 대개 cp949)"""
    frames = []
    for p in paths:
        for enc in ("utf-8-sig", "cp949"):
            try:
                frames.append(pd.read_csv(p, encoding=enc, dtype=str))
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError(f"인코딩을 알 수 없습니다: {p}")
    return normalize_kra(pd.concat(frames, ignore_index=True))


def fetch_kra_api(
    endpoint: str,
    service_key: str,
    params: dict[str, str] | None = None,
    rows_per_page: int = 1000,
    max_pages: int = 1000,
) -> pd.DataFrame:
    """공공데이터포털 오픈 API에서 모든 페이지를 받아 원본 표로 돌려준다.

    endpoint, 조회 조건 파라미터(예: 경주일·월, 경마장 코드)는 신청한 API 명세서대로 넘긴다.
    """
    items: list[dict] = []
    for page in range(1, max_pages + 1):
        q = {"serviceKey": service_key, "pageNo": page, "numOfRows": rows_per_page, "_type": "json", **(params or {})}
        url = f"{endpoint}?{urllib.parse.urlencode(q, safe='%')}"
        with urllib.request.urlopen(url, timeout=30) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        body = payload.get("response", {}).get("body", {})
        page_items = (body.get("items") or {}).get("item") or []
        if isinstance(page_items, dict):
            page_items = [page_items]
        items.extend(page_items)
        total = int(body.get("totalCount", 0) or 0)
        if not page_items or page * rows_per_page >= total:
            break
    return pd.DataFrame(items)


def validate(df: pd.DataFrame) -> list[str]:
    """표준 형식 데이터의 문제를 사람이 읽을 수 있는 경고 목록으로 돌려준다."""
    warnings = []
    for c in REQUIRED:
        if c not in df:
            warnings.append(f"필수 열 없음: {c}")
    if "rank" in df:
        winners = df[df["rank"] == 1].groupby("race_id").size()
        n_races = df["race_id"].nunique()
        no_winner = n_races - len(winners)
        if no_winner:
            warnings.append(f"우승마가 없는 경주 {no_winner}개 (학습에서 제외)")
    return warnings
