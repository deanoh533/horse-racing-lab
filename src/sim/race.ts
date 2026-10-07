import type { Horse } from './horse';
import { type Rng, createRng, gaussian } from './rng';

export type TrackCondition = 'good' | 'soft' | 'heavy';

export const CONDITION_LABEL: Record<TrackCondition, string> = {
  good: '양호',
  soft: '다습',
  heavy: '불량',
};

export interface RaceConfig {
  distance: number; // m
  condition: TrackCondition;
  seed: number;
  /** 프레임 기록 여부 (몬테카를로에선 끔) */
  recordFrames?: boolean;
}

export interface HorseResult {
  horse: Horse;
  rank: number;
  finishTime: number; // s
  gap: number; // 1위와의 시간 차 (s)
  /** 거리 구간별 속도 [{distance, speed}] (약 50m 간격) */
  speedProfile: { distance: number; speed: number }[];
}

export interface RaceFrame {
  time: number;
  positions: number[]; // 출전마 순서와 동일
}

export interface RaceResult {
  config: RaceConfig;
  results: HorseResult[]; // 순위순
  frames: RaceFrame[];
}

const DT = 0.1;
const MAX_TIME = 600;
/** 막판 스퍼트를 시작하는 남은 거리 (m) */
const KICK_DISTANCE = 400;
/** 최고 속도로 1초 달릴 때 소모하는 체력 */
const DRAIN_PER_SECOND = 1.35;

const CONDITION_SPEED: Record<TrackCondition, number> = { good: 1, soft: 0.97, heavy: 0.93 };

interface Runner {
  horse: Horse;
  pos: number;
  v: number;
  energy: number;
  capacity: number;
  form: number; // 경주 중 순간 컨디션 (평균 1)
  dayForm: number; // 당일 컨디션
  finishTime: number | null;
  profile: { distance: number; speed: number }[];
  nextSample: number;
}

/** 주로 상태가 이 말의 속도에 주는 배율 */
export function conditionFactor(horse: Horse, condition: TrackCondition): number {
  const base = CONDITION_SPEED[condition];
  if (condition === 'good') return base;
  // 적응력이 높을수록 감속 폭이 줄어든다 (최대 70% 상쇄)
  const penalty = 1 - base;
  return 1 - penalty * (1 - 0.7 * (horse.mudAffinity / 100));
}

export function simulateRace(horses: Horse[], config: RaceConfig): RaceResult {
  const rng = createRng(config.seed);
  const runners: Runner[] = horses.map((horse) => {
    const volatility = 1 - horse.consistency / 100;
    const capacity = 50 + horse.stamina;
    return {
      horse,
      pos: 0,
      v: 0,
      energy: capacity,
      capacity,
      form: 1,
      dayForm: 1 + gaussian(rng) * (0.008 + 0.012 * volatility),
      finishTime: null,
      profile: [],
      nextSample: 0,
    };
  });

  const frames: RaceFrame[] = [];
  let t = 0;
  while (t < MAX_TIME && runners.some((r) => r.finishTime === null)) {
    for (const r of runners) {
      if (r.finishTime !== null) continue;
      step(r, config, rng, t);
    }
    t += DT;
    if (config.recordFrames) frames.push({ time: t, positions: runners.map((r) => Math.min(r.pos, config.distance)) });
  }

  const finished = [...runners].sort((a, b) => (a.finishTime ?? Infinity) - (b.finishTime ?? Infinity));
  const winnerTime = finished[0].finishTime ?? MAX_TIME;
  const results = finished.map((r, i) => ({
    horse: r.horse,
    rank: i + 1,
    finishTime: r.finishTime ?? MAX_TIME,
    gap: (r.finishTime ?? MAX_TIME) - winnerTime,
    speedProfile: r.profile,
  }));
  return { config, results, frames };
}

function step(r: Runner, config: RaceConfig, rng: Rng, t: number): void {
  const { horse } = r;
  const remaining = config.distance - r.pos;

  // 순간 컨디션: 평균 1로 회귀하는 랜덤 워크
  const volatility = 1 - horse.consistency / 100;
  r.form += (1 - r.form) * 0.05 + gaussian(rng) * (0.002 + 0.004 * volatility);

  // 페이스 전략: 순항 후 막판 스퍼트
  const pace = remaining > KICK_DISTANCE ? 0.92 : 1.0;

  // 체력이 25% 이하로 떨어지면 점점 느려진다 (최저 88%)
  const energyRatio = r.energy / r.capacity;
  const fatigue = energyRatio > 0.25 ? 1 : 0.88 + 0.48 * energyRatio;

  const target = horse.topSpeed * conditionFactor(horse, config.condition) * pace * fatigue * r.form * r.dayForm;
  const accelRate = 0.4 + (horse.acceleration / 100) * 0.8;
  r.v += (target - r.v) * Math.min(1, accelRate * DT);

  const effort = r.v / horse.topSpeed;
  r.energy = Math.max(0, r.energy - effort ** 3 * DRAIN_PER_SECOND * DT);

  const prev = r.pos;
  r.pos += r.v * DT;

  while (r.nextSample <= r.pos && r.nextSample <= config.distance) {
    r.profile.push({ distance: r.nextSample, speed: r.v });
    r.nextSample += 50;
  }

  if (r.pos >= config.distance) {
    // 결승선 통과 시각을 선형 보간
    const frac = (config.distance - prev) / (r.pos - prev);
    r.finishTime = t + frac * DT;
  }
}
