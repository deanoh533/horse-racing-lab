import { type Rng, randInt } from './rng';

export interface Horse {
  id: number;
  name: string;
  color: string;
  /** 최고 속도 (m/s) */
  topSpeed: number;
  /** 지구력 0–100: 체력 총량 */
  stamina: number;
  /** 가속력 0–100: 목표 속도에 도달하는 빠르기 */
  acceleration: number;
  /** 안정성 0–100: 높을수록 컨디션 기복이 적다 */
  consistency: number;
  /** 무거운 주로 적응력 0–100 */
  mudAffinity: number;
}

const NAMES = [
  '천둥번개', '바람의아들', '질풍', '은빛갈기', '새벽별', '흑룡', '황금마차', '청룡',
  '불꽃질주', '한라산', '백두대간', '푸른초원', '번개탄', '별빛기사', '돌풍', '태양마',
];

export const HORSE_COLORS = [
  '#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6', '#bfef45',
  '#469990', '#9a6324', '#800000', '#808000', '#000075', '#a9a9a9', '#fabed4', '#ffd8b1',
];

export function generateHorses(rng: Rng, count: number): Horse[] {
  const names = [...NAMES];
  const horses: Horse[] = [];
  for (let i = 0; i < count; i++) {
    const nameIdx = randInt(rng, 0, names.length - 1);
    const [name] = names.splice(nameIdx, 1);
    horses.push({
      id: i,
      name: name ?? `말 ${i + 1}`,
      color: HORSE_COLORS[i % HORSE_COLORS.length],
      topSpeed: round2(17.4 + rng() * 0.6),
      stamina: randInt(rng, 30, 95),
      acceleration: randInt(rng, 30, 95),
      consistency: randInt(rng, 30, 95),
      mudAffinity: randInt(rng, 10, 95),
    });
  }
  return horses;
}

function round2(x: number): number {
  return Math.round(x * 100) / 100;
}
