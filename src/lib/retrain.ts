// The policy behind background retraining: when a dataset change is allowed to
// start a run, how a burst of changes collapses into one, and what happens when
// a run is already going.
//
// The clock is passed in, so the whole thing can be driven a step at a time in
// node without waiting four real seconds:
//
//   node scripts/check-store.mjs

import { boxesOf, DETECT_MIN_IMAGES, type ProjectManifest } from "./types";

/** How long the app waits after the last change before retraining itself. */
export const AUTO_RETRAIN_DELAY_MS = 4000;

/** The two timer calls the scheduler needs, so a check can hand it a fake one. */
export interface Clock {
  set(fn: () => void, ms: number): unknown;
  clear(handle: unknown): void;
}

const realClock: Clock = {
  set: (fn, ms) => setTimeout(fn, ms),
  clear: (handle) => clearTimeout(handle as ReturnType<typeof setTimeout>),
};

export interface SchedulerHooks {
  /** Whether a background run makes sense at all right now. */
  ready: () => boolean;
  /** Whether another run already owns the model. */
  busy: () => boolean;
  start: () => void;
  delayMs?: number;
  clock?: Clock;
}

export interface RetrainScheduler {
  /** Debounced request. A burst of edits produces exactly one run. */
  schedule(): void;
  /** Drops both the pending timer and anything waiting behind a run. */
  cancel(): void;
  /** Called when any run ends, so a request that waited gets its turn. */
  finished(): void;
  /** True while a run is waiting for the current one to finish. */
  queued(): boolean;
  /** True while the debounce timer is armed. */
  pending(): boolean;
}

export function createRetrainScheduler(hooks: SchedulerHooks): RetrainScheduler {
  const clock = hooks.clock ?? realClock;
  const delay = hooks.delayMs ?? AUTO_RETRAIN_DELAY_MS;
  let timer: unknown = null;
  let waiting = false;

  const fire = () => {
    timer = null;
    if (!hooks.ready()) return;
    if (hooks.busy()) {
      // A run owns the model right now, try again once it is done.
      waiting = true;
      return;
    }
    hooks.start();
  };

  const schedule = () => {
    if (!hooks.ready()) return;
    if (timer !== null) clock.clear(timer);
    timer = clock.set(fire, delay);
  };

  return {
    schedule,
    cancel() {
      if (timer !== null) clock.clear(timer);
      timer = null;
      waiting = false;
    },
    finished() {
      if (!waiting) return;
      waiting = false;
      schedule();
    },
    queued: () => waiting,
    pending: () => timer !== null,
  };
}

/**
 * Whether a dataset change in this project should start a background run.
 *
 * A classify project only needs a trained head to fold a correction into. A
 * detect project additionally needs enough drawn pictures to train on at all:
 * under the minimum the run would only refuse itself, loudly, four seconds
 * after every box the person draws on their way to ten.
 */
export function autoRetrainAllowed(
  manifest: ProjectManifest | null,
  hasModel: boolean,
): boolean {
  if (!manifest || !hasModel) return false;
  if (manifest.mode !== "detect") return true;
  const boxed = manifest.images.filter((image) => boxesOf(image).length > 0);
  return boxed.length >= DETECT_MIN_IMAGES;
}
