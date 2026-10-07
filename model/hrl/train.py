"""실제 경주 데이터로 승률 예측 모델을 학습·평가한다.

    python -m hrl.train --csv ../data/raw/*.csv --out out/model.json
    python -m hrl.train --csv ../data/raw/*.csv --use-odds        # 배당 정보도 특징으로 사용
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .clogit import ConditionalLogit
from .evaluate import calibration_table, odds_implied_prob, race_metrics
from .features import build_features
from .kra import load_kra_csv, validate

FEATURE_LABELS = {
    "h_first_start": "첫 출전",
    "h_starts_log": "출전 경험",
    "h_win_rate": "말 승률",
    "h_top3_rate": "말 입상률",
    "h_rank_pct_last5": "최근 5경주 착순 (낮을수록 좋음)",
    "h_last_rank_pct": "직전 경주 착순",
    "h_days_since_log": "휴양 기간",
    "h_behind_last3": "최근 3경주 기록 차 (우승마 대비)",
    "jk_win_rate": "기수 승률",
    "tr_win_rate": "조교사 승률",
    "burden_rel": "부담중량 (경주 평균 대비)",
    "rating_rel": "레이팅 (경주 평균 대비)",
    "age_rel": "나이 (경주 평균 대비)",
    "body_weight_change_abs": "마체중 변화폭",
    "gate_pct": "게이트 위치 (바깥쪽일수록 큼)",
    "market_log_prob": "시장 평가 (단승 배당)",
}


def prepare(df: pd.DataFrame, use_odds: bool) -> tuple[pd.DataFrame, list[str]]:
    d, feats = build_features(df, use_odds=use_odds)
    winners = d.groupby("race_id")["won"].transform("sum")
    d = d[winners > 0].copy()
    d["y"] = d["won"] / d.groupby("race_id")["won"].transform("sum")
    # 경주 코드: 날짜순으로 붙인 정수 (같은 경주 행은 이미 연속)
    d["race_code"] = pd.factorize(d["race_id"])[0]
    return d.reset_index(drop=True), feats


def split_by_date(d: pd.DataFrame, test_from: str | None, test_frac: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if test_from:
        cut = pd.Timestamp(test_from)
    else:
        dates = np.sort(d["date"].unique())
        cut = dates[int(len(dates) * (1 - test_frac))]
    return d[d["date"] < cut].copy(), d[d["date"] >= cut].copy()


def _fit(d: pd.DataFrame, feats: list[str], l2: float) -> ConditionalLogit:
    return ConditionalLogit(l2=l2).fit(d[feats].to_numpy(float), d["race_code"].to_numpy(), d["y"].to_numpy(), feats)


def choose_l2(train: pd.DataFrame, feats: list[str], grid=(0.1, 1.0, 10.0, 100.0, 1000.0)) -> float:
    """학습 기간의 마지막 20%로 정규화 강도를 고른다 (시험 기간은 건드리지 않음)."""
    inner_train, valid = split_by_date(train, None, 0.2)
    if valid["race_id"].nunique() < 20 or inner_train["race_id"].nunique() < 20:
        return 1.0
    scores = {}
    for l2 in grid:
        m = _fit(inner_train, feats, l2)
        p = m.predict_proba(valid[feats].to_numpy(float), valid["race_code"].to_numpy())
        scores[l2] = race_metrics(p, valid["y"].to_numpy(), valid["race_code"].to_numpy())["log_loss"]
    return min(scores, key=scores.get)


def run(df: pd.DataFrame, use_odds: bool = False, test_from: str | None = None, test_frac: float = 0.2) -> dict:
    d, feats = prepare(df, use_odds)
    train, test = split_by_date(d, test_from, test_frac)
    if train.empty or test.empty:
        raise ValueError("학습 또는 시험 기간에 경주가 없습니다. --test-from 날짜를 확인하세요.")
    l2 = choose_l2(train, feats)
    model = _fit(train, feats, l2)

    X_test = test[feats].to_numpy(float)
    codes = test["race_code"].to_numpy()
    y = test["y"].to_numpy()
    p = model.predict_proba(X_test, codes)
    report = {
        "features": feats,
        "l2": l2,
        "train": {"races": int(train["race_id"].nunique()), "from": str(train["date"].min().date()), "to": str(train["date"].max().date())},
        "test": {"from": str(test["date"].min().date()), "to": str(test["date"].max().date()), **race_metrics(p, y, codes)},
        "calibration": calibration_table(p, y).astype({"구간": str}).to_dict(orient="records"),
    }
    if "odds" in test and test["odds"].notna().any():
        report["market"] = race_metrics(odds_implied_prob(test["odds"], codes), y, codes)
    return {"model": model, "report": report, "test_frame": test.assign(pred=p)}


def print_report(model: ConditionalLogit, report: dict) -> None:
    t = report["test"]
    print(f"\n학습: {report['train']['from']} ~ {report['train']['to']} · {report['train']['races']:,}경주")
    print(f"시험: {t['from']} ~ {t['to']} · {t['races']:,}경주 (학습에 쓰지 않은 기간)")
    print(f"정규화 강도(L2): {report['l2']}\n")

    print("■ 특징별 영향 (표준화 계수, +면 승률을 올림)")
    order = np.argsort(-np.abs(model.coef))
    for i in order:
        f = model.features[i]
        print(f"  {model.coef[i]:+.3f}  {FEATURE_LABELS.get(f, f)}")

    print("\n■ 시험 기간 성능")
    rows = [("이 모델", t)]
    if "market" in report:
        rows.append(("시장 배당", report["market"]))
    print(f"  {'':10}{'로그손실↓':>10}{'브라이어↓':>10}{'1위 적중률↑':>12}")
    for name, m in rows:
        print(f"  {name:10}{m['log_loss']:>10.3f}{m['brier']:>10.3f}{m['top1_accuracy']:>12.1%}")
    print(f"  {'찍기(1/n)':10}{t['uniform_log_loss']:>10.3f}{'':>10}{t['uniform_top1']:>12.1%}")

    print("\n■ 보정 (예측 승률 구간별 실제 승률)")
    for r in report["calibration"]:
        print(f"  {r['구간']:>14}  출전 {r['출전수']:>6,}  예측 {r['예측승률']:6.1%}  실제 {r['실제승률']:6.1%}")


def main() -> None:
    ap = argparse.ArgumentParser(description="마사회 경주 데이터로 승률 예측 모델 학습")
    ap.add_argument("--csv", nargs="+", required=True, help="마사회 경주성적 CSV (glob 가능)")
    ap.add_argument("--use-odds", action="store_true", help="단승 배당을 특징으로 사용")
    ap.add_argument("--test-from", help="이 날짜부터를 시험 기간으로 (YYYY-MM-DD). 기본: 마지막 20%%")
    ap.add_argument("--out", default="out/model.json")
    args = ap.parse_args()

    paths = sorted({p for pat in args.csv for p in glob.glob(pat)})
    if not paths:
        raise SystemExit(f"CSV 파일을 찾지 못했습니다: {args.csv}")
    df = load_kra_csv(paths)
    print(f"데이터: 파일 {len(paths)}개 · {len(df):,}행 · {df['race_id'].nunique():,}경주 · 말 {df['horse_id'].nunique():,}두")
    for w in validate(df):
        print(f"  ⚠ {w}")

    result = run(df, use_odds=args.use_odds, test_from=args.test_from)
    print_report(result["model"], result["report"])
    result["model"].save(args.out, extra={"report": result["report"]})
    print(f"\n모델 저장: {args.out}")


if __name__ == "__main__":
    main()
