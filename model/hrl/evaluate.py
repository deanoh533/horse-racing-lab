"""승률 예측 평가 지표."""

from __future__ import annotations

import numpy as np
import pandas as pd


def race_metrics(p: np.ndarray, y: np.ndarray, race_codes: np.ndarray) -> dict:
    """p: 예측 승률, y: 실제 우승(공동 1위면 1/k). 같은 경주 행은 연속해 있어야 한다."""
    df = pd.DataFrame({"race": race_codes, "p": p, "y": y})
    g = df.groupby("race", sort=False)
    log_loss = -(df["y"] * np.log(np.clip(df["p"], 1e-15, None))).groupby(df["race"], sort=False).sum()
    brier = ((df["p"] - df["y"]) ** 2).groupby(df["race"], sort=False).sum()
    fav_idx = g["p"].idxmax()
    top1 = (df.loc[fav_idx, "y"] > 0).to_numpy()
    n = g.size()
    return {
        "races": int(len(n)),
        "log_loss": float(log_loss.mean()),
        "brier": float(brier.mean()),
        "top1_accuracy": float(top1.mean()),
        "uniform_log_loss": float(np.log(n).mean()),
        "uniform_top1": float((1 / n).mean()),
    }


def calibration_table(p: np.ndarray, y: np.ndarray, bins: tuple[float, ...] = (0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0)) -> pd.DataFrame:
    """예측 승률 구간별로 실제 우승 비율을 비교한다. 잘 보정된 모델이면 두 값이 비슷하다."""
    df = pd.DataFrame({"p": p, "y": (y > 0).astype(float)})
    df["구간"] = pd.cut(df["p"], bins=list(bins), include_lowest=True)
    t = df.groupby("구간", observed=True).agg(출전수=("y", "size"), 예측승률=("p", "mean"), 실제승률=("y", "mean"))
    return t.reset_index()


def odds_implied_prob(odds: pd.Series, race_codes: np.ndarray) -> np.ndarray:
    """단승 배당 → 시장 예측 승률 (1/배당을 경주별 합 1로 정규화)."""
    inv = 1.0 / odds.to_numpy(dtype=float)
    s = pd.Series(inv)
    s = s.fillna(s.groupby(race_codes).transform("min"))
    return (s / s.groupby(race_codes).transform("sum")).to_numpy()
