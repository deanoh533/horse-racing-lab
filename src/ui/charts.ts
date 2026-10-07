import { Chart, registerables } from 'chart.js';
import type { MonteCarloResult } from '../sim/montecarlo';
import type { RaceResult } from '../sim/race';

Chart.register(...registerables);

function themeColors() {
  const css = getComputedStyle(document.documentElement);
  return { text: css.getPropertyValue('--muted').trim(), grid: css.getPropertyValue('--border').trim() };
}

function baseScales(xTitle: string, yTitle: string) {
  const { text, grid } = themeColors();
  return {
    x: { title: { display: true, text: xTitle, color: text }, ticks: { color: text }, grid: { color: grid } },
    y: { title: { display: true, text: yTitle, color: text }, ticks: { color: text }, grid: { color: grid } },
  };
}

function legendColor() {
  return { labels: { color: themeColors().text, boxWidth: 12 } };
}

/** 순위 색: 1위 진하게 → 하위일수록 연하게 */
function rankColor(rank: number, total: number): string {
  const t = total <= 1 ? 0 : rank / (total - 1);
  const light = 30 + t * 55;
  return `hsl(215, 70%, ${light}%)`;
}

export class Dashboard {
  private speedChart: Chart | null = null;
  private winChart: Chart | null = null;
  private rankChart: Chart | null = null;

  showRace(race: RaceResult): void {
    this.speedChart?.destroy();
    const ctx = document.querySelector<HTMLCanvasElement>('#speed-chart')!;
    this.speedChart = new Chart(ctx, {
      type: 'line',
      data: {
        datasets: race.results.map((r) => ({
          label: `${r.rank}. ${r.horse.name}`,
          data: r.speedProfile.slice(1).map((p) => ({ x: p.distance, y: +(p.speed * 3.6).toFixed(1) })),
          borderColor: r.horse.color,
          backgroundColor: r.horse.color,
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.3,
        })),
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        interaction: { mode: 'index', intersect: false },
        scales: (() => {
          const s = baseScales('거리 (m)', '속도 (km/h)');
          return { x: { ...s.x, type: 'linear' as const, max: race.config.distance }, y: s.y };
        })(),
        plugins: { legend: { position: window.innerWidth < 700 ? 'bottom' : 'right', ...legendColor() } },
      },
    });
  }

  showMonteCarlo(mc: MonteCarloResult): void {
    const labels = mc.stats.map((s) => s.horse.name);
    const n = mc.stats.length;

    this.winChart?.destroy();
    this.winChart = new Chart(document.querySelector<HTMLCanvasElement>('#win-chart')!, {
      type: 'bar',
      data: {
        labels,
        datasets: [
          {
            label: '승률 (%)',
            data: mc.stats.map((s) => +(s.winRate * 100).toFixed(1)),
            backgroundColor: mc.stats.map((s) => s.horse.color),
          },
          {
            label: '입상률 · 3위 이내 (%)',
            data: mc.stats.map((s) => +(s.top3Rate * 100).toFixed(1)),
            backgroundColor: mc.stats.map((s) => s.horse.color + '55'),
            borderColor: mc.stats.map((s) => s.horse.color),
            borderWidth: 1,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 300 },
        scales: { ...baseScales('', '%'), y: { ...baseScales('', '%').y, beginAtZero: true, max: 100 } },
        plugins: { legend: legendColor() },
      },
    });

    this.rankChart?.destroy();
    this.rankChart = new Chart(document.querySelector<HTMLCanvasElement>('#rank-chart')!, {
      type: 'bar',
      data: {
        labels,
        datasets: Array.from({ length: n }, (_, k) => ({
          label: `${k + 1}위`,
          data: mc.stats.map((s) => +((s.rankCounts[k] / mc.runs) * 100).toFixed(1)),
          backgroundColor: rankColor(k, n),
          borderColor: getComputedStyle(document.documentElement).getPropertyValue('--card').trim(),
          borderWidth: 1,
        })),
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 300 },
        scales: {
          x: { ...baseScales('%', '').x, stacked: true, max: 100 },
          y: { ...baseScales('', '').y, stacked: true, title: { display: false } },
        },
        plugins: {
          legend: legendColor(),
          tooltip: { callbacks: { label: (c) => `${c.dataset.label}: ${c.parsed.x}%` } },
        },
      },
    });
  }
}
