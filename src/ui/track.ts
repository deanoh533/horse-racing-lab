import type { Horse } from '../sim/horse';

/** 트랙 한 바퀴를 몇 m로 볼지 (그림상의 비율만 결정) */
const LAP_METERS = 1600;
/** 결승선 위치: 아래쪽 직선주로의 80% 지점 */
const FINISH_ON_STRAIGHT = 0.8;

interface Geometry {
  cx: number;
  cy: number;
  straight: number; // 직선주로 길이 (px)
  radius: number; // 안쪽 펜스 반지름 (px)
  band: number; // 주로 폭 (px)
  perimeter: number; // 안쪽 펜스 둘레 (px)
}

/** 타원형(스타디움형) 트랙 위에 말들을 그린다. 반시계 방향으로 달린다. */
export class TrackView {
  private ctx: CanvasRenderingContext2D;
  private horses: Horse[] = [];
  private distance = 1600;
  private geo!: Geometry;

  constructor(private canvas: HTMLCanvasElement) {
    this.ctx = canvas.getContext('2d')!;
    new ResizeObserver(() => this.resize()).observe(canvas);
    this.resize();
  }

  setField(horses: Horse[], distance: number): void {
    this.horses = horses;
    this.distance = distance;
  }

  private lastPositions: number[] | null = null;

  private resize(): void {
    const dpr = window.devicePixelRatio || 1;
    const w = this.canvas.clientWidth;
    const h = this.canvas.clientHeight;
    if (w === 0 || h === 0) return;
    this.canvas.width = Math.round(w * dpr);
    this.canvas.height = Math.round(h * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const margin = 8;
    const band = Math.max(28, Math.min(70, h * 0.22));
    const radius = (h - 2 * margin) / 2 - band;
    const straight = Math.max(0, w - 2 * margin - 2 * (radius + band));
    this.geo = { cx: w / 2, cy: h / 2, straight, radius, band, perimeter: 2 * straight + 2 * Math.PI * radius };
    this.draw(this.lastPositions);
  }

  /** 안쪽 펜스를 따라 u(px)만큼 간 지점과 바깥 방향 법선 */
  private railPoint(u: number): { x: number; y: number; nx: number; ny: number } {
    const { cx, cy, straight: L, radius: R, perimeter: P } = this.geo;
    u = ((u % P) + P) % P;
    const arc = Math.PI * R;
    if (u < L) return { x: cx - L / 2 + u, y: cy + R, nx: 0, ny: 1 };
    u -= L;
    if (u < arc) {
      const th = Math.PI / 2 - u / R;
      return { x: cx + L / 2 + R * Math.cos(th), y: cy + R * Math.sin(th), nx: Math.cos(th), ny: Math.sin(th) };
    }
    u -= arc;
    if (u < L) return { x: cx + L / 2 - u, y: cy - R, nx: 0, ny: -1 };
    u -= L;
    const th = -Math.PI / 2 - u / R;
    return { x: cx - L / 2 + R * Math.cos(th), y: cy + R * Math.sin(th), nx: Math.cos(th), ny: Math.sin(th) };
  }

  /** 경주 거리상 위치(m) → 펜스 위 u(px) */
  private toRail(meters: number): number {
    const finishU = FINISH_ON_STRAIGHT * this.geo.straight;
    return finishU - ((this.distance - meters) / LAP_METERS) * this.geo.perimeter;
  }

  draw(positions: number[] | null): void {
    this.lastPositions = positions;
    if (!this.geo) return;
    const { ctx } = this;
    const css = getComputedStyle(document.documentElement);
    const color = (name: string) => css.getPropertyValue(name).trim();
    const { cx, cy, straight: L, radius: R, band } = this.geo;

    ctx.clearRect(0, 0, this.canvas.clientWidth, this.canvas.clientHeight);

    const stadium = (r: number) => {
      ctx.beginPath();
      ctx.arc(cx + L / 2, cy, r, -Math.PI / 2, Math.PI / 2);
      ctx.arc(cx - L / 2, cy, r, Math.PI / 2, (3 * Math.PI) / 2);
      ctx.closePath();
    };

    // 주로 + 내부 잔디
    stadium(R + band);
    ctx.fillStyle = color('--track');
    ctx.fill();
    stadium(R);
    ctx.fillStyle = color('--infield');
    ctx.fill();

    // 펜스
    ctx.strokeStyle = color('--track-line');
    ctx.lineWidth = 2;
    stadium(R);
    ctx.stroke();
    stadium(R + band);
    ctx.stroke();

    // 출발선
    const start = this.railPoint(this.toRail(0));
    this.drawMarker(start, '#ffffff', 3, [6, 4]);
    // 결승선 (체크무늬)
    const finish = this.railPoint(this.toRail(this.distance));
    this.drawCheckered(finish);

    ctx.fillStyle = color('--track-line');
    ctx.font = '600 12px system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('결승', finish.x, finish.y - finish.ny * 10);

    // 거리 표지 (남은 거리 200m 간격)
    ctx.fillStyle = color('--track-line');
    for (let rem = 200; rem < this.distance; rem += 200) {
      const p = this.railPoint(this.toRail(this.distance - rem));
      ctx.beginPath();
      ctx.arc(p.x - p.nx * 6, p.y - p.ny * 6, 2, 0, Math.PI * 2);
      ctx.fill();
    }

    if (!positions) positions = this.horses.map(() => 0);
    const n = this.horses.length;
    const laneW = band / (n + 1);
    const dotR = Math.max(5, Math.min(10, laneW * 0.45));
    // 뒤에 있는 말부터 그려야 선두가 위에 보인다
    const order = positions.map((p, i) => [p, i]).sort((a, b) => a[0] - b[0]);
    for (const [pos, i] of order) {
      const horse = this.horses[i];
      const p = this.railPoint(this.toRail(pos));
      const off = laneW * (i + 1);
      const x = p.x + p.nx * off;
      const y = p.y + p.ny * off;
      ctx.beginPath();
      ctx.arc(x, y, dotR, 0, Math.PI * 2);
      ctx.fillStyle = horse.color;
      ctx.fill();
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = '#ffffff';
      ctx.stroke();
      ctx.fillStyle = '#ffffff';
      ctx.font = `700 ${Math.round(dotR * 1.1)}px system-ui, sans-serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(String(i + 1), x, y + 0.5);
      ctx.textBaseline = 'alphabetic';
    }
  }

  private drawMarker(p: { x: number; y: number; nx: number; ny: number }, stroke: string, width: number, dash: number[]): void {
    const { ctx } = this;
    ctx.save();
    ctx.setLineDash(dash);
    ctx.strokeStyle = stroke;
    ctx.lineWidth = width;
    ctx.beginPath();
    ctx.moveTo(p.x, p.y);
    ctx.lineTo(p.x + p.nx * this.geo.band, p.y + p.ny * this.geo.band);
    ctx.stroke();
    ctx.restore();
  }

  private drawCheckered(p: { x: number; y: number; nx: number; ny: number }): void {
    const { ctx } = this;
    const cells = 8;
    const size = this.geo.band / cells;
    // 법선에 수직인 방향 (진행 방향)
    const tx = -p.ny;
    const ty = p.nx;
    for (let k = 0; k < cells; k++) {
      for (let j = 0; j < 2; j++) {
        ctx.fillStyle = (k + j) % 2 === 0 ? '#111111' : '#ffffff';
        const bx = p.x + p.nx * size * k + tx * size * (j - 1);
        const by = p.y + p.ny * size * k + ty * size * (j - 1);
        ctx.beginPath();
        ctx.moveTo(bx, by);
        ctx.lineTo(bx + p.nx * size, by + p.ny * size);
        ctx.lineTo(bx + p.nx * size + tx * size, by + p.ny * size + ty * size);
        ctx.lineTo(bx + tx * size, by + ty * size);
        ctx.closePath();
        ctx.fill();
      }
    }
  }
}
