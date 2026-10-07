import { describe, expect, it } from 'vitest';
import { type Horse, generateHorses } from '../src/sim/horse';
import { runMonteCarlo } from '../src/sim/montecarlo';
import { conditionFactor, simulateRace } from '../src/sim/race';
import { createRng } from '../src/sim/rng';

function makeHorse(id: number, overrides: Partial<Horse> = {}): Horse {
  return {
    id,
    name: `말${id}`,
    color: '#000',
    topSpeed: 17.6,
    stamina: 60,
    acceleration: 60,
    consistency: 60,
    mudAffinity: 50,
    ...overrides,
  };
}

describe('rng', () => {
  it('같은 시드는 같은 수열을 만든다', () => {
    const a = createRng(7);
    const b = createRng(7);
    expect([a(), a(), a()]).toEqual([b(), b(), b()]);
  });
});

describe('simulateRace', () => {
  const horses = generateHorses(createRng(42), 8);

  it('같은 시드면 결과가 재현된다', () => {
    const cfg = { distance: 1600, condition: 'good' as const, seed: 3 };
    const r1 = simulateRace(horses, cfg).results.map((r) => [r.horse.id, r.finishTime]);
    const r2 = simulateRace(horses, cfg).results.map((r) => [r.horse.id, r.finishTime]);
    expect(r1).toEqual(r2);
  });

  it('모든 말이 완주하고 순위·기록이 일관된다', () => {
    const { results } = simulateRace(horses, { distance: 1200, condition: 'good', seed: 1 });
    expect(results).toHaveLength(8);
    expect(results.map((r) => r.rank)).toEqual([1, 2, 3, 4, 5, 6, 7, 8]);
    for (let i = 1; i < results.length; i++) {
      expect(results[i].finishTime).toBeGreaterThanOrEqual(results[i - 1].finishTime);
    }
    expect(results[0].gap).toBe(0);
  });

  it('기록이 실제 경마 수준 범위에 있다', () => {
    const { results } = simulateRace(horses, { distance: 1600, condition: 'good', seed: 1 });
    for (const r of results) {
      expect(r.finishTime).toBeGreaterThan(88);
      expect(r.finishTime).toBeLessThan(110);
    }
  });

  it('recordFrames일 때만 프레임을 기록한다', () => {
    const off = simulateRace(horses, { distance: 1200, condition: 'good', seed: 1 });
    const on = simulateRace(horses, { distance: 1200, condition: 'good', seed: 1, recordFrames: true });
    expect(off.frames).toHaveLength(0);
    expect(on.frames.length).toBeGreaterThan(500);
    expect(on.frames.at(-1)!.positions.every((p) => p === 1200)).toBe(true);
  });

  it('불량 주로에서는 기록이 느려진다', () => {
    const good = simulateRace(horses, { distance: 1600, condition: 'good', seed: 5 });
    const heavy = simulateRace(horses, { distance: 1600, condition: 'heavy', seed: 5 });
    expect(heavy.results[0].finishTime).toBeGreaterThan(good.results[0].finishTime);
  });
});

describe('conditionFactor', () => {
  it('주로 적응력이 높을수록 불량 주로 감속이 적다', () => {
    const mudder = makeHorse(0, { mudAffinity: 95 });
    const nonMudder = makeHorse(1, { mudAffinity: 10 });
    expect(conditionFactor(mudder, 'good')).toBe(1);
    expect(conditionFactor(mudder, 'heavy')).toBeGreaterThan(conditionFactor(nonMudder, 'heavy'));
  });
});

describe('runMonteCarlo', () => {
  it('승률 합이 1이고 순위 분포가 일관된다', () => {
    const horses = generateHorses(createRng(9), 6);
    const mc = runMonteCarlo(horses, { distance: 1400, condition: 'good', runs: 200, baseSeed: 1 });
    const totalWin = mc.stats.reduce((s, x) => s + x.winRate, 0);
    expect(totalWin).toBeCloseTo(1, 10);
    for (const s of mc.stats) {
      expect(s.rankCounts.reduce((a, b) => a + b, 0)).toBe(200);
    }
  });

  it('장거리에서는 지구력 높은 말이 유리하다', () => {
    const sprinter = makeHorse(0, { topSpeed: 17.9, stamina: 30 });
    const stayer = makeHorse(1, { topSpeed: 17.5, stamina: 95 });
    const field = [sprinter, stayer];
    const short = runMonteCarlo(field, { distance: 1000, condition: 'good', runs: 300, baseSeed: 1 });
    const long = runMonteCarlo(field, { distance: 2400, condition: 'good', runs: 300, baseSeed: 1 });
    const winRate = (mc: typeof short, id: number) => mc.stats.find((s) => s.horse.id === id)!.winRate;
    expect(winRate(short, 0)).toBeGreaterThan(0.5);
    expect(winRate(long, 1)).toBeGreaterThan(0.5);
  });
});
