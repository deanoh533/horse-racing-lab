"""조건부 로짓 (Conditional Logit) 승률 모델.

경주 r에 나온 말 i의 점수 u_i = x_i · β 로 두고, 승률을 같은 경주 말들끼리의 소프트맥스로 계산한다.
    P(i 우승) = exp(u_i) / Σ_{j ∈ r} exp(u_j)
경주마다 출전 두수가 달라도 되고, 경주 전체에 같은 값(거리 등)은 자동으로 상쇄된다.
경마 예측의 표준 모델 (Bolton & Chapman 1986, Benter 1994).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import minimize


def _group_softmax(u: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """경주별(연속된 구간) 소프트맥스. starts는 각 경주 시작 인덱스."""
    m = np.maximum.reduceat(u, starts)
    lengths = np.diff(np.append(starts, len(u)))
    e = np.exp(u - np.repeat(m, lengths))
    s = np.add.reduceat(e, starts)
    return e / np.repeat(s, lengths)


def _race_starts(race_codes: np.ndarray) -> np.ndarray:
    if len(race_codes) == 0:
        return np.array([], dtype=int)
    starts = np.flatnonzero(np.r_[True, race_codes[1:] != race_codes[:-1]])
    if len(starts) != len(np.unique(race_codes)):
        raise ValueError("같은 경주의 행들이 연속해 있어야 합니다 (race_id로 정렬 필요).")
    return starts


@dataclass
class ConditionalLogit:
    l2: float = 1.0
    features: list[str] = field(default_factory=list)
    mean: np.ndarray | None = None
    std: np.ndarray | None = None
    coef: np.ndarray | None = None

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std

    def fit(self, X: np.ndarray, race_codes: np.ndarray, y: np.ndarray, features: list[str]) -> "ConditionalLogit":
        """y: 우승마 1 (공동 1위면 1/k), 나머지 0. 경주마다 합이 1이어야 한다."""
        self.features = list(features)
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std[self.std == 0] = 1.0
        Z = self._standardize(X)
        starts = _race_starts(race_codes)
        n_races = len(starts)

        def loss_grad(beta: np.ndarray):
            p = _group_softmax(Z @ beta, starts)
            loss = -np.sum(y * np.log(np.clip(p, 1e-300, None))) / n_races + 0.5 * self.l2 * beta @ beta / n_races
            grad = Z.T @ (p - y) / n_races + self.l2 * beta / n_races
            return loss, grad

        res = minimize(loss_grad, np.zeros(Z.shape[1]), jac=True, method="L-BFGS-B")
        if not res.success:
            raise RuntimeError(f"학습이 수렴하지 않았습니다: {res.message}")
        self.coef = res.x
        return self

    def predict_proba(self, X: np.ndarray, race_codes: np.ndarray) -> np.ndarray:
        return _group_softmax(self._standardize(X) @ self.coef, _race_starts(race_codes))

    def to_dict(self) -> dict:
        return {
            "model": "conditional_logit",
            "l2": self.l2,
            "features": self.features,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "coef": self.coef.tolist(),
        }

    def save(self, path: str | Path, extra: dict | None = None) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({**self.to_dict(), **(extra or {})}, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ConditionalLogit":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(l2=d["l2"], features=d["features"], mean=np.array(d["mean"]), std=np.array(d["std"]), coef=np.array(d["coef"]))
