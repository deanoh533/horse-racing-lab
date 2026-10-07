import './style.css';
import { type Horse, generateHorses } from './sim/horse';
import type { MonteCarloResult } from './sim/montecarlo';
import {
  type PredictionRecord,
  type RaceScore,
  type WinPrediction,
  addToRecord,
  emptyRecord,
  favoriteOf,
  scoreRace,
  toPrediction,
  uniformBrier,
} from './sim/predict';
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
  mcStatus: $<HTMLSpanElement>('#mc-status'),
  verify: $<HTMLButtonElement>('#verify'),
  record: $<HTMLDListElement>('#record'),
  verdict: $<HTMLDivElement>('#verdict'),
  fieldSize: $<HTMLSelectElement>('#field-size'),
  rosterSeed: $<HTMLInputElement>('#roster-seed'),
  regen: $<HTMLButtonElement>('#regen'),
  rosterTable: $<HTMLTableElement>('#roster-table'),
  resultTable: $<HTMLTableElement>('#result-table'),
  mcTable: $<HTMLTableElement>('#mc-table'),
  clock: $<HTMLSpanElement>('#race-clock'),
  leaderboard: $<HTMLOListElement>('#leaderboard'),
};

/** 예측용 시뮬레이션 시드 시작값. 사용자가 입력하는 경주 시드와 겹치지 않게 크게 잡는다. */
const PREDICTION_BASE_SEED = 1_000_000_000;

let horses: Horse[] = [];
const track = new TrackView($<HTMLCanvasElement>('#track'));
const dashboard = new Dashboard();
let animationId = 0;

/** 현재 출전마·조건에 대한 예측. 조건이 바뀌면 null로 비우고 다시 계산한다. */
let prediction: WinPrediction | null = null;
let record: PredictionRecord = emptyRecord();

const distance = () => Number(els.distance.value);
const condition = () => els.condition.value as TrackCondition;
const pct = (x: number) => `${(x * 100).toFixed(1)}%`;
const odds = (p: number) => (p > 0 ? `${(1 / p).toFixed(1)}배` : '—');

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
  invalidatePrediction();
}

function renderRoster(): void {
  const head = `<thead><tr><th>번호</th><th class="left">마명</th>${STAT_COLUMNS.map((c) => `<th>${c.label}</th>`).join('')}<th>예측 승률</th><th>배당</th></tr></thead>`;
  const rows = horses
    .map(
      (h, i) => `<tr>
        <td>${i + 1}</td>
        <td class="left"><span class="swatch" style="background:${h.color}"></span>${h.name}</td>
        ${STAT_COLUMNS.map(
          (c) => `<td><input type="number" data-horse="${i}" data-key="${c.key}" min="${c.min}" max="${c.max}" step="${c.step}" value="${h[c.key]}" aria-label="${h.name} ${c.label}" /></td>`,
        ).join('')}
        <td class="pred-cell" data-pred="${h.id}">…</td>
        <td class="pred-cell" data-odds="${h.id}">…</td>
      </tr>`,
    )
    .join('');
  els.rosterTable.innerHTML = head + `<tbody>${rows}</tbody>`;
}

/** 입력칸을 다시 그리지 않고 예측 열만 갱신 (편집 중 포커스 유지) */
function renderRosterPrediction(): void {
  const fav = prediction ? favoriteOf(prediction) : -1;
  for (const h of horses) {
    const p = prediction?.get(h.id);
    for (const [attr, text] of [
      ['data-pred', p === undefined ? '…' : pct(p)],
      ['data-odds', p === undefined ? '…' : odds(p)],
    ] as const) {
      const cell = els.rosterTable.querySelector<HTMLTableCellElement>(`[${attr}="${h.id}"]`);
      if (!cell) continue;
      cell.textContent = text;
      cell.classList.toggle('fav', h.id === fav);
    }
  }
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
  invalidatePrediction();
});

// ---------- 승률 예측 ----------

let worker: Worker | null = null;
let predictTimer = 0;

/** 조건이 바뀌었으므로 예측과 적중 기록을 비우고 잠시 뒤 다시 예측한다 (연속 입력 대비) */
function invalidatePrediction(): void {
  prediction = null;
  record = emptyRecord();
  renderRosterPrediction();
  renderRecord();
  els.verify.disabled = true;
  els.mcStatus.textContent = '예측 대기…';
  clearTimeout(predictTimer);
  predictTimer = window.setTimeout(runPrediction, 300);
}

function runPrediction(): void {
  worker?.terminate();
  clearTimeout(predictTimer);
  const runs = Math.min(20000, Math.max(100, Math.round(Number(els.mcRuns.value) || 2000)));
  els.mcRuns.value = String(runs);
  els.mcStatus.textContent = `계산 중 0 / ${runs.toLocaleString()}`;
  const snapshot = horses.map((h) => ({ ...h }));
  const dist = distance();
  const cond = condition();

  worker = new Worker(new URL('./sim/worker.ts', import.meta.url), { type: 'module' });
  worker.onmessage = (e: MessageEvent<WorkerResponse>) => {
    const msg = e.data;
    if (msg.type === 'progress') {
      els.mcStatus.textContent = `계산 중 ${msg.done.toLocaleString()} / ${runs.toLocaleString()}`;
      return;
    }
    // 워커에서 복제된 말 객체를 현재 출전마 객체로 되돌린다 (이름·색 유지)
    const byId = new Map(horses.map((h) => [h.id, h]));
    const result: MonteCarloResult = {
      ...msg.result,
      stats: msg.result.stats.map((s) => ({ ...s, horse: byId.get(s.horse.id)! })),
    };
    prediction = toPrediction(result);
    els.mcStatus.textContent = `${runs.toLocaleString()}회 시뮬레이션 · ${dist}m · ${CONDITION_LABEL[cond]}`;
    els.verify.disabled = false;
    renderRosterPrediction();
    dashboard.showMonteCarlo(result);
    renderMcTable(result);
    worker?.terminate();
    worker = null;
  };
  const req: WorkerRequest = { horses: snapshot, distance: dist, condition: cond, runs, baseSeed: PREDICTION_BASE_SEED };
  worker.postMessage(req);
}

function renderMcTable(mc: MonteCarloResult): void {
  els.mcTable.innerHTML = `
    <thead><tr><th class="left">마명</th><th>예측 승률</th><th>입상률(3위 이내)</th><th>평균 순위</th><th>평균 기록</th><th>적정 배당</th></tr></thead>
    <tbody>${mc.stats
      .map(
        (s) => `<tr>
          <td class="left"><span class="swatch" style="background:${s.horse.color}"></span>${horses.indexOf(s.horse) + 1}. ${s.horse.name}</td>
          <td>${pct(s.winRate)}</td>
          <td>${pct(s.top3Rate)}</td>
          <td>${s.avgRank.toFixed(2)}</td>
          <td>${formatTime(s.avgTime)}</td>
          <td>${odds(s.winRate)}</td>
        </tr>`,
      )
      .join('')}</tbody>`;
}

function renderRecord(): void {
  const r = record;
  if (r.races === 0) {
    els.record.innerHTML = `<dd class="empty">경주를 하거나 100경주 검증을 누르면 예측이 얼마나 맞는지 기록됩니다.</dd>`;
    return;
  }
  const n = r.races;
  els.record.innerHTML = `
    <dt>검증한 경주</dt><dd>${n.toLocaleString()}회</dd>
    <dt>예측 1위 우승</dt><dd>${pct(r.favoriteWins / n)} <small>(예측 ${pct(r.favoriteProbSum / n)})</small></dd>
    <dt>예측 1위 3위 이내</dt><dd>${pct(r.favoriteTop3 / n)}</dd>
    <dt>실제 우승마의 평균 예측 승률</dt><dd>${pct(r.winnerProbSum / n)}</dd>
    <dt>브라이어 점수 <small>(낮을수록 정확)</small></dt><dd>${(r.brierSum / n).toFixed(3)} <small>(찍기 ${uniformBrier(horses.length).toFixed(3)})</small></dd>`;
}

/** 100경주를 애니메이션 없이 돌려 예측을 검증한다 */
function verifyBatch(): void {
  if (!prediction) return;
  const start = Number(els.seed.value);
  for (let i = 0; i < 100; i++) {
    const race = simulateRace(horses, { distance: distance(), condition: condition(), seed: start + i });
    record = addToRecord(record, scoreRace(prediction, race));
  }
  els.seed.value = String(start + 100);
  renderRecord();
}

// ---------- 단일 경주 ----------

function resetRaceView(): void {
  cancelAnimationFrame(animationId);
  els.runRace.disabled = false;
  track.setField(horses, distance());
  track.draw(null);
  els.clock.textContent = '0.0s';
  els.leaderboard.innerHTML = '';
}

function runRace(): void {
  cancelAnimationFrame(animationId);
  const seed = Number(els.seed.value);
  const race = simulateRace(horses, { distance: distance(), condition: condition(), seed, recordFrames: true });
  // 경주 시작 시점의 예측으로 채점한다. 다음 경주를 위해 시드를 하나 올린다.
  const pred = prediction;
  els.seed.value = String(seed + 1);
  track.setField(horses, race.config.distance);
  els.verdict.innerHTML = '';
  const speed = Number(els.playback.value);
  if (speed === 0) {
    finishRace(race, pred);
    return;
  }
  els.runRace.disabled = true;
  playRace(race, speed, pred);
}

function playRace(race: RaceResult, speed: number, pred: WinPrediction | null): void {
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
      finishRace(race, pred);
      return;
    }
    // 인접한 두 프레임 사이를 보간
    const f = simTime / dt - 1;
    const i0 = Math.max(0, Math.floor(f));
    const i1 = Math.min(frames.length - 1, i0 + 1);
    const w = Math.min(1, Math.max(0, f - i0));
    const positions = frames[i0].positions.map((p, k) => p + (frames[i1].positions[k] - p) * w);
    track.draw(positions);
    renderLeaderboard(positions, simTime, finishTimes, race.config.distance);
    els.clock.textContent = `${simTime.toFixed(1)}s`;
    animationId = requestAnimationFrame(tick);
  };
  animationId = requestAnimationFrame(tick);
}

function renderLeaderboard(positions: number[], time: number, finishTimes: Map<number, number>, dist: number): void {
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
      const gap = e.ft <= time ? `${e.ft.toFixed(2)}s` : k === 0 ? `${Math.round(dist - e.pos)}m 남음` : `-${(leaderPos - e.pos).toFixed(1)}m`;
      return `<li><span class="pos">${k + 1}</span><span class="swatch" style="background:${e.h.color}"></span>${e.i + 1}. ${e.h.name}<span class="gap">${gap}</span></li>`;
    })
    .join('');
}

function finishRace(race: RaceResult, pred: WinPrediction | null): void {
  els.runRace.disabled = false;
  const finalPositions = horses.map(() => race.config.distance);
  track.draw(finalPositions);
  renderLeaderboard(finalPositions, Infinity, new Map(race.results.map((r) => [r.horse.id, r.finishTime])), race.config.distance);
  els.clock.textContent = `${race.results[0].finishTime.toFixed(1)}s`;

  // 경주 도중 조건이 바뀌었다면(예측이 비워졌다면) 기록에는 넣지 않는다
  const score = pred ? scoreRace(pred, race) : null;
  if (score && pred === prediction) {
    record = addToRecord(record, score);
    renderRecord();
  }
  renderVerdict(race, score);
  renderResults(race, pred);
  dashboard.showRace(race);
}

function renderVerdict(race: RaceResult, score: RaceScore | null): void {
  if (!score) {
    els.verdict.innerHTML = `<p class="verdict muted">예측이 준비되기 전에 시작한 경주라 채점하지 않았습니다.</p>`;
    return;
  }
  const fav = horses.find((h) => h.id === score.favoriteId)!;
  const winner = race.results[0].horse;
  const hit = score.favoriteRank === 1;
  els.verdict.innerHTML = `<p class="verdict ${hit ? 'hit' : 'miss'}">
    ${hit ? '✅ 적중' : '❌ 빗나감'} · 예측 1위 <b>${fav.name}</b>(${pct(score.favoriteProb)}) → 실제 ${score.favoriteRank}위
    ${hit ? '' : `<br />우승 <b>${winner.name}</b>의 예측 승률은 ${pct(score.winnerProb)}였습니다.`}
  </p>`;
}

function formatTime(sec: number): string {
  const m = Math.floor(sec / 60);
  const s = sec - m * 60;
  return `${m}:${s.toFixed(2).padStart(5, '0')}`;
}

function renderResults(race: RaceResult, pred: WinPrediction | null): void {
  const { distance: d, condition: c } = race.config;
  els.resultTable.innerHTML = `
    <caption class="muted small" style="caption-side:bottom;text-align:left">${d}m · 주로 ${CONDITION_LABEL[c]} · 시드 ${race.config.seed}</caption>
    <thead><tr><th>순위</th><th class="left">마명</th><th>기록</th><th>차이</th><th>예측 승률</th></tr></thead>
    <tbody>${race.results
      .map((r) => {
        const p = pred?.get(r.horse.id);
        return `<tr>
          <td class="${r.rank <= 3 ? `rank-${r.rank}` : ''}">${r.rank}</td>
          <td class="left"><span class="swatch" style="background:${r.horse.color}"></span>${horses.indexOf(r.horse) + 1}. ${r.horse.name}</td>
          <td>${formatTime(r.finishTime)}</td>
          <td>${r.rank === 1 ? '—' : `+${r.gap.toFixed(2)}s`}</td>
          <td>${p === undefined ? '—' : pct(p)}</td>
        </tr>`;
      })
      .join('')}</tbody>`;
}

// ---------- 이벤트 ----------

els.runRace.addEventListener('click', runRace);
els.verify.addEventListener('click', verifyBatch);
els.regen.addEventListener('click', regenerate);
els.fieldSize.addEventListener('change', regenerate);
els.distance.addEventListener('change', () => {
  resetRaceView();
  invalidatePrediction();
});
els.condition.addEventListener('change', invalidatePrediction);
els.mcRuns.addEventListener('change', invalidatePrediction);

regenerate();
