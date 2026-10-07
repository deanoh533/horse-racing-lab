import type { Horse } from './horse';
import { type TrackCondition, simulateRace } from './race';

export interface HorseStats {
  horse: Horse;
  wins: number;
  winRate: number;
  top3Rate: number;
  avgRank: number;
  avgTime: number;
  /** rankCounts[k] = (k+1)위 횟수 */
  rankCounts: number[];
}

export interface MonteCarloResult {
  runs: number;
  stats: HorseStats[]; // 승률 내림차순
}

export function runMonteCarlo(
  horses: Horse[],
  opts: { distance: number; condition: TrackCondition; runs: number; baseSeed: number },
  onProgress?: (done: number) => void,
): MonteCarloResult {
  const n = horses.length;
  const acc = horses.map((horse) => ({ horse, rankCounts: new Array<number>(n).fill(0), rankSum: 0, timeSum: 0 }));

  for (let i = 0; i < opts.runs; i++) {
    const race = simulateRace(horses, { distance: opts.distance, condition: opts.condition, seed: opts.baseSeed + i });
    for (const r of race.results) {
      const a = acc[horses.indexOf(r.horse)];
      a.rankCounts[r.rank - 1]++;
      a.rankSum += r.rank;
      a.timeSum += r.finishTime;
    }
    if (onProgress && (i + 1) % 100 === 0) onProgress(i + 1);
  }

  const stats = acc.map((a) => ({
    horse: a.horse,
    wins: a.rankCounts[0],
    winRate: a.rankCounts[0] / opts.runs,
    top3Rate: a.rankCounts.slice(0, 3).reduce((s, c) => s + c, 0) / opts.runs,
    avgRank: a.rankSum / opts.runs,
    avgTime: a.timeSum / opts.runs,
    rankCounts: a.rankCounts,
  }));
  stats.sort((x, y) => y.winRate - x.winRate || x.avgRank - y.avgRank);
  return { runs: opts.runs, stats };
}
