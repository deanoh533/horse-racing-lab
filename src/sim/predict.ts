import type { MonteCarloResult } from './montecarlo';
import type { RaceResult } from './race';

/** 말 id → 예측 승률 */
export type WinPrediction = Map<number, number>;

export function toPrediction(mc: MonteCarloResult): WinPrediction {
  return new Map(mc.stats.map((s) => [s.horse.id, s.winRate]));
}

/** 예측 승률이 가장 높은 말 id (동률이면 먼저 나온 말) */
export function favoriteOf(pred: WinPrediction): number {
  let best = -1;
  let bestP = -Infinity;
  for (const [id, p] of pred) {
    if (p > bestP) {
      best = id;
      bestP = p;
    }
  }
  return best;
}

export interface RaceScore {
  favoriteId: number;
  favoriteProb: number;
  favoriteRank: number;
  /** 실제 우승마에게 매겼던 예측 승률 */
  winnerProb: number;
  /** 다중 분류 브라이어 점수: Σ(예측 − 실제)², 0이 완벽 */
  brier: number;
}

export function scoreRace(pred: WinPrediction, race: RaceResult): RaceScore {
  const favoriteId = favoriteOf(pred);
  const winnerId = race.results[0].horse.id;
  let brier = 0;
  for (const r of race.results) {
    const p = pred.get(r.horse.id) ?? 0;
    const o = r.horse.id === winnerId ? 1 : 0;
    brier += (p - o) ** 2;
  }
  return {
    favoriteId,
    favoriteProb: pred.get(favoriteId) ?? 0,
    favoriteRank: race.results.find((r) => r.horse.id === favoriteId)?.rank ?? Infinity,
    winnerProb: pred.get(winnerId) ?? 0,
    brier,
  };
}

/** 여러 경주에 걸친 예측 적중 기록 */
export interface PredictionRecord {
  races: number;
  favoriteWins: number;
  favoriteTop3: number;
  favoriteProbSum: number;
  winnerProbSum: number;
  brierSum: number;
}

export const emptyRecord = (): PredictionRecord => ({
  races: 0,
  favoriteWins: 0,
  favoriteTop3: 0,
  favoriteProbSum: 0,
  winnerProbSum: 0,
  brierSum: 0,
});

export function addToRecord(rec: PredictionRecord, s: RaceScore): PredictionRecord {
  return {
    races: rec.races + 1,
    favoriteWins: rec.favoriteWins + (s.favoriteRank === 1 ? 1 : 0),
    favoriteTop3: rec.favoriteTop3 + (s.favoriteRank <= 3 ? 1 : 0),
    favoriteProbSum: rec.favoriteProbSum + s.favoriteProb,
    winnerProbSum: rec.winnerProbSum + s.winnerProb,
    brierSum: rec.brierSum + s.brier,
  };
}

/** 모든 말에 같은 확률(1/n)을 주는 예측의 기대 브라이어 점수 — 비교 기준선 */
export function uniformBrier(fieldSize: number): number {
  return 1 - 1 / fieldSize;
}
