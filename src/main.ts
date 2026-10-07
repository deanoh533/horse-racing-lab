import './style.css';
import { type Horse, generateHorses } from './sim/horse';
import type { MonteCarloResult } from './sim/montecarlo';
import { CONDITION_LABEL, type RaceResult, type TrackCondition, simulateRace } from './sim/race';
import { createRng } from './sim/rng';
import type { WorkerRequest, WorkerResponse } from './sim/worker';
import { Dashboard } from './ui/charts';
import { TrackView } from './ui/track';

const $ = <T extends HTMLElement>(sel: string) => document.querySelector<T>(sel)!;

const els = {
  distance: $<HTMLSelectElement>('#distance'),
  condition: $<HTMLSelectElement>('#condition'),
  seed: $<HTMLInputElement>('#seed'),
  playback: $<HTMLSelectElement>('#playback'),
  runRace: $<HTMLButtonElement>('#run-race'),
  mcRuns: $<HTMLInputElement>('#mc-runs'),
  runMc: $<HTMLButtonElement>('#run-mc'),
  mcStatus: $<HTMLParagraphElement>('#mc-status'),
  fieldSize: $<HTMLSelectElement>('#field-size'),
  rosterSeed: $<HTMLInputElement>('#roster-seed'),
  regen: $<HTMLButtonElement>('#regen'),
  rosterTable: $<HTMLTableElement>('#roster-table'),
  resultTable: $<HTMLTableElement>('#result-table'),
  mcTable: $<HTMLTableElement>('#mc-table'),
  clock: $<HTMLSpanElement>('#race-clock'),
  leaderboard: $<HTMLOListElement>('#leaderboard'),
};

let horses: Horse[] = [];
const track = new TrackView($<HTMLCanvasElement>('#track'));
const dashboard = new Dashboard();
let animationId = 0;

const distance = () => Number(els.distance.value);
const condition = () => els.condition.value as TrackCondition;

// ---------- 출전마 ----------

type StatKey = 'topSpeed' | 'stamina' | 'acceleration' | 'consistency' | 'mudAffinity';
const STAT_COLUMNS: { key: StatKey; label: string; min: number; max: number; step: number }[] = [
  { key: 'topSpeed', label: '최고속도', min: 15, max: 19, step: 0.01 },
  { key: 'stamina', label: '지구력', min: 0, max: 100, step: 1 },
  { key: 'acceleration', label: '가속력', min: 0, max: 100, step: 1 },
  { key: 'consistency', label: '안정성', min: 0, max: 100, step: 1 },
  { key: 'mudAffinity', label: '불량주로', min: 0, max: 100, step: 1 },
];

function regenerate(): void {
  horses = generateHorses(createRng(Number(els.rosterSeed.value)), Number(els.fieldSize.value));
  renderRoster();
  resetRaceView();
}

function renderRoster(): void {
  const head = `<thead><tr><th>번호</th><th class="left">마명</th>${STAT_COLUMNS.map((c) => `<th>${c.label}</th>`).join('')}</tr></thead>`;
  const rows = horses
    .map(
      (h, i) => `<tr>
        <td>${i + 1}</td>
        <td class="left"><span class="swatch" style="background:${h.color}"></span>${h.name}</td>
        ${STAT_COLUMNS.map(
          (c) => `<td><input type="number" data-horse="${i}" data-key="${c.key}" min="${c.min}" max="${c.max}" step="${c.step}" value="${h[c.key]}" aria-label="${h.name} ${c.label}" /></td>`,
        ).join('')}
      </tr>`,
    )
    .join('');
  els.rosterTable.innerHTML = head + `<tbody>${rows}</tbody>`;
}

els.rosterTable.addEventListener('change', (e) => {
  const input = e.target as HTMLInputElement;
  const idx = input.dataset.horse;
  const key = input.dataset.key as StatKey | undefined;
  if (idx === undefined || !key) return;
  const col = STAT_COLUMNS.find((c) => c.key === key)!;
  const value = Math.min(col.max, Math.max(col.min, Number(input.value) || col.min));
  horses[Number(idx)][key] = value;
  input.value = String(value);
});

// ---------- 단일 경주 ----------

function resetRaceView(): void {
  cancelAnimationFrame(animationId);
  track.setField(horses, distance());
  track.draw(null);
  els.clock.textContent = '0.0s';
  els.leaderboard.innerHTML = '';
}

function runRace(): void {
  cancelAnimationFrame(animationId);
  const race = simulateRace(horses, {
    distance: distance(),
    condition: condition(),
    seed: Number(els.seed.value),
    recordFrames: true,
  });
  track.setField(horses, race.config.distance);
  const speed = Number(els.playback.value);
  if (speed === 0) {
    finishRace(race);
    return;
  }
  els.runRace.disabled = true;
  playRace(race, speed);
}

function playRace(race: RaceResult, speed: number): void {
  const { frames } = race;
  const dt = frames[0].time;
  const endTime = frames.at(-1)!.time;
  const finishTimes = new Map(race.results.map((r) => [r.horse.id, r.finishTime]));
  let simTime = 0;
  let last = performance.now();

  const tick = (now: number) => {
    simTime += ((now - last) / 1000) * speed;
    last = now;
    if (simTime >= endTime) {
      finishRace(race);
      return;
    }
    // 인접한 두 프레임 사이를 보간
    const f = simTime / dt - 1;
    const i0 = Math.max(0, Math.floor(f));
    const i1 = Math.min(frames.length - 1, i0 + 1);
    const w = Math.min(1, Math.max(0, f - i0));
    const positions = frames[i0].positions.map((p, k) => p + (frames[i1].positions[k] - p) * w);
    track.draw(positions);
    renderLeaderboard(positions, simTime, finishTimes);
    els.clock.textContent = `${simTime.toFixed(1)}s`;
    animationId = requestAnimationFrame(tick);
  };
  animationId = requestAnimationFrame(tick);
}

function renderLeaderboard(positions: number[], time: number, finishTimes: Map<number, number>): void {
  const entries = horses.map((h, i) => ({ h, i, pos: positions[i], ft: finishTimes.get(h.id)! }));
  // 결승선을 통과한 말은 도착 시간순, 나머지는 위치순
  entries.sort((a, b) => {
    const aDone = a.ft <= time;
    const bDone = b.ft <= time;
    if (aDone && bDone) return a.ft - b.ft;
    if (aDone !== bDone) return aDone ? -1 : 1;
    return b.pos - a.pos;
  });
  const leaderPos = entries[0].pos;
  els.leaderboard.innerHTML = entries
    .map((e, k) => {
      const gap = e.ft <= time ? `${e.ft.toFixed(2)}s` : k === 0 ? `${Math.round(distance() - e.pos)}m 남음` : `-${(leaderPos - e.pos).toFixed(1)}m`;
      return `<li><span class="pos">${k + 1}</span><span class="swatch" style="background:${e.h.color}"></span>${e.i + 1}. ${e.h.name}<span class="gap">${gap}</span></li>`;
    })
    .join('');
}

function finishRace(race: RaceResult): void {
  els.runRace.disabled = false;
  const finalPositions = horses.map(() => race.config.distance);
  track.draw(finalPositions);
  renderLeaderboard(finalPositions, Infinity, new Map(race.results.map((r) => [r.horse.id, r.finishTime])));
  els.clock.textContent = `${race.results[0].finishTime.toFixed(1)}s`;
  renderResults(race);
  dashboard.showRace(race);
}

function formatTime(sec: number): string {
  const m = Math.floor(sec / 60);
  const s = sec - m * 60;
  return `${m}:${s.toFixed(2).padStart(5, '0')}`;
}

function renderResults(race: RaceResult): void {
  const { distance: d, condition: c } = race.config;
  els.resultTable.innerHTML = `
    <caption class="muted small" style="caption-side:bottom;text-align:left">${d}m · 주로 ${CONDITION_LABEL[c]} · 시드 ${race.config.seed}</caption>
    <thead><tr><th>순위</th><th class="left">마명</th><th>기록</th><th>차이</th><th>평균 km/h</th></tr></thead>
    <tbody>${race.results
      .map(
        (r) => `<tr>
          <td class="${r.rank <= 3 ? `rank-${r.rank}` : ''}">${r.rank}</td>
          <td class="left"><span class="swatch" style="background:${r.horse.color}"></span>${horses.indexOf(r.horse) + 1}. ${r.horse.name}</td>
          <td>${formatTime(r.finishTime)}</td>
          <td>${r.rank === 1 ? '—' : `+${r.gap.toFixed(2)}s`}</td>
          <td>${((d / r.finishTime) * 3.6).toFixed(1)}</td>
        </tr>`,
      )
      .join('')}</tbody>`;
}

// ---------- 몬테카를로 ----------

let worker: Worker | null = null;

function runMonteCarlo(): void {
  worker?.terminate();
  const runs = Math.min(20000, Math.max(10, Math.round(Number(els.mcRuns.value) || 1000)));
  els.mcRuns.value = String(runs);
  els.runMc.disabled = true;
  els.mcStatus.textContent = `0 / ${runs}`;
  const startedAt = performance.now();
  const snapshot = horses.map((h) => ({ ...h }));

  worker = new Worker(new URL('./sim/worker.ts', import.meta.url), { type: 'module' });
  worker.onmessage = (e: MessageEvent<WorkerResponse>) => {
    const msg = e.data;
    if (msg.type === 'progress') {
      els.mcStatus.textContent = `${msg.done.toLocaleString()} / ${runs.toLocaleString()}`;
      return;
    }
    // 워커에서 복제된 말 객체를 현재 출전마 정보로 되돌린다 (색·이름 유지)
    const byId = new Map(snapshot.map((h) => [h.id, h]));
    const result: MonteCarloResult = {
      ...msg.result,
      stats: msg.result.stats.map((s) => ({ ...s, horse: byId.get(s.horse.id)! })),
    };
    const secs = ((performance.now() - startedAt) / 1000).toFixed(1);
    els.mcStatus.textContent = `${runs.toLocaleString()}회 완료 · ${distance()}m · 주로 ${CONDITION_LABEL[condition()]} · ${secs}s`;
    els.runMc.disabled = false;
    dashboard.showMonteCarlo(result);
    renderMcTable(result, snapshot);
    worker?.terminate();
    worker = null;
  };
  const req: WorkerRequest = {
    horses: snapshot,
    distance: distance(),
    condition: condition(),
    runs,
    baseSeed: Number(els.seed.value) * 100003,
  };
  worker.postMessage(req);
}

function renderMcTable(mc: MonteCarloResult, field: Horse[]): void {
  els.mcTable.innerHTML = `
    <thead><tr><th class="left">마명</th><th>승률</th><th>입상률(3위 이내)</th><th>평균 순위</th><th>평균 기록</th><th>적정 배당</th></tr></thead>
    <tbody>${mc.stats
      .map(
        (s) => `<tr>
          <td class="left"><span class="swatch" style="background:${s.horse.color}"></span>${field.findIndex((h) => h.id === s.horse.id) + 1}. ${s.horse.name}</td>
          <td>${(s.winRate * 100).toFixed(1)}%</td>
          <td>${(s.top3Rate * 100).toFixed(1)}%</td>
          <td>${s.avgRank.toFixed(2)}</td>
          <td>${formatTime(s.avgTime)}</td>
          <td>${s.winRate > 0 ? `${(1 / s.winRate).toFixed(1)}배` : '—'}</td>
        </tr>`,
      )
      .join('')}</tbody>`;
}

// ---------- 이벤트 ----------

els.runRace.addEventListener('click', runRace);
els.runMc.addEventListener('click', runMonteCarlo);
els.regen.addEventListener('click', regenerate);
els.fieldSize.addEventListener('change', regenerate);
els.distance.addEventListener('change', resetRaceView);

regenerate();
