import type { Horse } from './horse';
import { type MonteCarloResult, runMonteCarlo } from './montecarlo';
import type { TrackCondition } from './race';

export interface WorkerRequest {
  horses: Horse[];
  distance: number;
  condition: TrackCondition;
  runs: number;
  baseSeed: number;
}

export type WorkerResponse = { type: 'progress'; done: number } | { type: 'done'; result: MonteCarloResult };

self.onmessage = (e: MessageEvent<WorkerRequest>) => {
  const { horses, ...opts } = e.data;
  const result = runMonteCarlo(horses, opts, (done) => postMessage({ type: 'progress', done } satisfies WorkerResponse));
  postMessage({ type: 'done', result } satisfies WorkerResponse);
};
