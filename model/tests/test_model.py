import numpy as np
import pandas as pd
import pytest

from hrl.clogit import ConditionalLogit, _group_softmax
from hrl.evaluate import race_metrics
from hrl.features import build_features
from hrl.kra import normalize_kra, parse_body_weight, parse_time
from hrl.sample import generate
from hrl.train import run


def test_마체중_기록_파싱():
    assert parse_body_weight("470(+2)") == (470.0, 2.0)
    assert parse_body_weight("455(-5)") == (455.0, -5.0)
    assert parse_body_weight("480")[0] == 480.0
    assert parse_time("1:35.5") == pytest.approx(95.5)
    assert parse_time("72.3") == pytest.approx(72.3)
    assert np.isnan(parse_time(""))


def test_마사회_영문_API_키도_변환():
    raw = pd.DataFrame(
        {
            "meet": ["서울", "서울"],
            "rcDate": [20240106, 20240106],
            "rcNo": [1, 1],
            "hrNo": ["001", "002"],
            "hrName": ["가", "나"],
            "chulNo": [1, 2],
            "ord": [1, 0],
            "wgHr": ["470(+2)", "460(-1)"],
            "winOdds": [2.5, 0],
        }
    )
    d = normalize_kra(raw)
    assert d["race_id"].iloc[0] == "서울-2024-01-06-1"
    assert d["rank"].iloc[0] == 1 and np.isnan(d["rank"].iloc[1])  # 착순 0 → 기권
    assert np.isnan(d["odds"].iloc[1])
    assert d["body_weight_change"].tolist() == [2.0, -1.0]


def test_필수_열이_없으면_알려준다():
    with pytest.raises(ValueError, match="필수 열"):
        normalize_kra(pd.DataFrame({"마명": ["가"]}))


def _two_races():
    return pd.DataFrame(
        {
            "race_id": ["r1", "r1", "r2", "r2"],
            "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-08", "2024-01-08"]),
            "horse_id": ["A", "B", "A", "B"],
            "jockey": ["J1", "J2", "J1", "J2"],
            "rank": [1, 2, 2, 1],
        }
    )


def test_특징에_이번_경주_결과가_새지_않는다():
    d, _ = build_features(_two_races())
    first = d[d["race_id"] == "r1"]
    # 첫 경주에서는 두 말 모두 과거 기록이 없으므로 특징이 같아야 한다
    assert first["h_win_rate"].nunique() == 1
    assert first["jk_win_rate"].nunique() == 1
    assert (first["h_first_start"] == 1).all()
    second = d[d["race_id"] == "r2"].set_index("horse_id")
    # 두 번째 경주에선 첫 경주 우승마 A의 기록이 더 좋다 (r2 결과는 반영 안 됨)
    assert second.loc["A", "h_win_rate"] > second.loc["B", "h_win_rate"]
    assert second.loc["A", "jk_win_rate"] > second.loc["B", "jk_win_rate"]


def test_경주별_소프트맥스_합은_1():
    p = _group_softmax(np.array([1.0, 2.0, 3.0, 0.0, 0.0]), np.array([0, 3]))
    assert p[:3].sum() == pytest.approx(1)
    assert p[3:].tolist() == pytest.approx([0.5, 0.5])


def test_조건부_로짓이_계수를_복원한다():
    rng = np.random.default_rng(1)
    true_beta = np.array([1.0, -0.5])
    X, codes, y = [], [], []
    for r in range(3000):
        n = 8
        x = rng.normal(size=(n, 2))
        winner = np.argmax(x @ true_beta + rng.gumbel(size=n))
        X.append(x)
        codes += [r] * n
        y += [1.0 if i == winner else 0.0 for i in range(n)]
    X = np.vstack(X)
    m = ConditionalLogit(l2=0.01).fit(X, np.array(codes), np.array(y), ["a", "b"])
    # 표준화 계수 → 원래 단위로 환산
    assert (m.coef / m.std) == pytest.approx(true_beta, abs=0.1)


def test_같은_경주_행이_흩어져_있으면_거부():
    m = ConditionalLogit()
    with pytest.raises(ValueError, match="연속"):
        m.fit(np.zeros((3, 1)), np.array([0, 1, 0]), np.array([1.0, 1.0, 0.0]), ["a"])


def test_합성_데이터_전체_파이프라인이_찍기보다_낫다():
    df = normalize_kra(generate(n_days=80, seed=3).astype(str))
    res = run(df)
    t = res["report"]["test"]
    assert t["races"] > 50
    assert t["log_loss"] < t["uniform_log_loss"]
    assert t["top1_accuracy"] > 2 * t["uniform_top1"]
    assert "market" in res["report"]


def test_배당_특징을_쓰면_시장_수준_이상():
    df = normalize_kra(generate(n_days=80, seed=4).astype(str))
    res = run(df, use_odds=True)
    assert "market_log_prob" in res["report"]["features"]
    assert res["report"]["test"]["log_loss"] <= res["report"]["market"]["log_loss"] + 0.02


def test_race_metrics_완벽한_예측():
    m = race_metrics(np.array([1.0, 0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0, 1.0]), np.array([0, 0, 1, 1]))
    assert m["top1_accuracy"] == 1.0
    assert m["brier"] == 0.0


def test_duckdb_로더(tmp_path):
    duckdb = pytest.importorskip("duckdb")
    from hrl.duck import load_duckdb

    path = tmp_path / "t.duckdb"
    with duckdb.connect(str(path)) as con:
        con.execute("create table races (race_date bigint, meet bigint, rc_no bigint, rc_dist bigint, track varchar, chaksun1 varchar)")
        con.execute("insert into races values (20240106, 1, 1, 1200, '건조 (3%)', '33000000'), (20240107, 1, 1, 1200, NULL, NULL)")
        cols = (
            "race_date bigint, meet bigint, rc_no bigint, pthr_no bigint, hr_no varchar, hr_name varchar, ag bigint, gndr varchar,"
            "burd_wgt double, ratg bigint, jcky_nm varchar, trar_nm varchar, rc_dist bigint, ord bigint, rc_time double,"
            "wg_hr bigint, wg_hr_diff hugeint, win_odds double, popularity bigint, se_g1f_acc_time double, bu_g1f_acc_time double,"
            "sj_s1f_ord bigint, bu_s1f_ord bigint"
        )
        con.execute(f"create table race_entries ({cols})")
        rows = [
            # 결과 있는 경주: 1·2위, 출전 취소(순위·기록 없음)
            (20240106, 1, 1, 1, "A", "가", 3, "수", 55, 40, "(-1)김기수", "박조교", 1200, 1, 72.5, 470, 2, 2.1, 1, 59.0, None, 1, None),
            (20240106, 1, 1, 2, "B", "나", 4, "암", 54, 35, "이기수", "최조교", 1200, 2, 73.0, 460, -1, 5.0, 2, 59.3, None, 2, None),
            (20240106, 1, 1, 3, "C", "다", 4, "거", 54, 30, "정기수", "최조교", 1200, None, None, 480, 0, None, None, None, None, None, None),
            # 예정 경주 (결과 없음)
            (20240107, 1, 1, 1, "A", "가", 3, "수", 55, 40, "김기수", "박조교", 1200, None, None, 471, 1, 3.0, None, None, None, None, None),
        ]
        con.executemany(f"insert into race_entries values ({','.join(['?'] * 23)})", rows)

    df = load_duckdb(path)
    assert sorted(df["horse_id"]) == ["A", "B"]  # 취소마·예정 경주 제외
    assert df["race_id"].iloc[0] == "서울-2024-01-06-1"
    assert set(df["jockey"]) == {"김기수", "이기수"}  # 감량 표시 제거
    assert df.loc[df["horse_id"] == "A", "late_200m"].iloc[0] == pytest.approx(13.5)
    assert df["purse"].iloc[0] == 33_000_000
    assert len(load_duckdb(path, include_upcoming=True)) == 3
