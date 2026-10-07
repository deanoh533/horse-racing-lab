"""⚠ 합성(가짜) 데이터 생성기 — 파이프라인 동작 확인용. 실제 경주 기록이 아니다.

마사회 CSV와 같은 한글 헤더로 만들어서 변환기(kra.py)까지 함께 시험한다.
숨은 능력치 + 기수·조교사 실력 + 중량·게이트 효과 + 운(검벨 잡음)으로 순위를 정하므로,
모델이 과거 기록에서 이 신호를 얼마나 찾아내는지 확인할 수 있다.

    python -m hrl.sample --out ../data/sample/synthetic_kra.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DISTANCES = [1000, 1200, 1300, 1400, 1600, 1800, 2000]
CONDITIONS = ["건조 (3%)", "건조 (5%)", "양호 (8%)", "다습 (12%)", "포화 (17%)"]


def generate(n_days: int = 200, races_per_day: int = 10, n_horses: int = 700, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ability = rng.normal(0, 1, n_horses)
    jockey_skill = rng.normal(0, 0.3, 45)
    trainer_skill = rng.normal(0, 0.2, 50)
    horse_trainer = rng.integers(0, 50, n_horses)
    horse_age = rng.integers(2, 7, n_horses)
    horse_sex = rng.choice(["수", "암", "거"], n_horses, p=[0.45, 0.35, 0.2])
    body_weight = rng.normal(470, 20, n_horses)
    last_raced = np.full(n_horses, -999)

    rows = []
    start = pd.Timestamp("2023-01-07")
    for day in range(n_days):
        # 토·일 경주
        date = start + pd.Timedelta(days=7 * (day // 2) + day % 2)
        dnum = (date - start).days
        for race_no in range(1, races_per_day + 1):
            rested = np.flatnonzero(dnum - last_raced >= 14)
            size = int(rng.integers(8, 15))
            if len(rested) < size:
                continue
            field = rng.choice(rested, size, replace=False)
            last_raced[field] = dnum
            ability[field] += rng.normal(0, 0.05, size)  # 능력치는 조금씩 변한다

            rating = np.clip(np.round(50 + 12 * ability[field] + rng.normal(0, 4, size)), 0, 120)
            burden = np.round(54 + (rating - rating.mean()) / 6 + rng.normal(0, 0.5, size), 1)
            bw_change = np.round(rng.normal(0, 5, size))
            body_weight[field] += bw_change
            jockeys = rng.choice(45, size, replace=False)
            gates = rng.permutation(size) + 1
            gate_pct = (gates - 1) / (size - 1)

            strength = (
                ability[field]
                + jockey_skill[jockeys]
                + trainer_skill[horse_trainer[field]]
                - 0.08 * (burden - 54)
                - 0.03 * np.abs(bw_change)
                - 0.25 * gate_pct
            )
            perf = strength + rng.gumbel(0, 1, size)
            order = np.argsort(-perf)
            rank = np.empty(size, dtype=int)
            rank[order] = np.arange(1, size + 1)
            rank[rng.random(size) < 0.01] = 0  # 가끔 기권·실격

            # 시장은 실제 실력을 잡음 섞어 본다 → 단승 배당 (공제율 20%)
            m = strength + rng.normal(0, 0.4, size)
            p_market = np.exp(m - m.max())
            p_market /= p_market.sum()
            odds = np.maximum(1.0, np.round(0.8 / p_market, 1))

            dist = int(rng.choice(DISTANCES))
            base = dist / 16.6 + rng.normal(0, 0.8)
            times = base + (rank - 1) * 0.22 + np.abs(rng.normal(0, 0.05, size))

            for k, h in enumerate(field):
                t = times[k]
                rows.append(
                    {
                        "경주일자": date.strftime("%Y%m%d"),
                        "경마장": "서울",
                        "경주번호": race_no,
                        "경주거리": dist,
                        "주로상태": CONDITIONS[int(rng.integers(len(CONDITIONS)))],
                        "마번": f"{h:07d}",
                        "마명": f"합성마{h:04d}",
                        "출전번호": gates[k],
                        "기수명": f"기수{jockeys[k]:02d}",
                        "조교사명": f"조교사{horse_trainer[h]:02d}",
                        "연령": horse_age[h],
                        "성별": horse_sex[h],
                        "부담중량": burden[k],
                        "마체중": f"{int(body_weight[h])}({int(bw_change[k]):+d})",
                        "레이팅": int(rating[k]),
                        "단승식배당율": odds[k],
                        "순위": rank[k],
                        "경주기록": "" if rank[k] == 0 else f"{int(t // 60)}:{t % 60:04.1f}",
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="합성 경주 데이터 생성 (파이프라인 시험용)")
    ap.add_argument("--out", default="../data/sample/synthetic_kra.csv")
    ap.add_argument("--days", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    df = generate(n_days=args.days, seed=args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"⚠ 합성 데이터 {len(df):,}행 저장: {args.out}")


if __name__ == "__main__":
    main()
