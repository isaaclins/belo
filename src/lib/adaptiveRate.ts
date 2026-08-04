/**
 * Rate control for a pushed screen feed.
 *
 * This is deliberately independent of ScreenCaptureKit and React. The native
 * side can therefore change, and the browser harness can drive fast and slow
 * pipelines without opening a real capture session.
 */

export const RATE_STEPS = [15, 30, 60, 90, 120, 144, 240] as const;
export const DEFAULT_INFERENCE_INTERVAL_MS = 220;

export interface PipelineTiming {
  nativeMs: number;
  transferMs: number;
  decodeMs: number;
  paintMs: number;
  inferenceMs: number;
  dropped: boolean;
}

export interface RateDecision {
  rate: number;
  changed: boolean;
  totalMs: number;
  budgetMs: number;
  reason: "hold" | "slow" | "fast" | "drop";
}

export function rateSteps(maxRate: number): number[] {
  const cap = Math.max(1, Math.round(maxRate));
  const steps: number[] = RATE_STEPS.filter((rate) => rate < cap);
  if (steps.length === 0 || steps[steps.length - 1] !== cap) steps.push(cap);
  return steps;
}

function finite(value: number): number {
  return Number.isFinite(value) && value >= 0 ? value : 0;
}

/**
 * Hysteretic controller. Three slow observations step down. Ten comfortably
 * fast observations step up. The asymmetry prevents a noisy decode from
 * oscillating while still recovering after a busy period.
 */
export class AdaptiveRateController {
  private readonly steps: number[];
  private readonly inferenceIntervalMs: number;
  private index: number;
  private slow = 0;
  private fast = 0;

  constructor({
    maxRate,
    initialRate = maxRate,
    inferenceIntervalMs = DEFAULT_INFERENCE_INTERVAL_MS,
  }: {
    maxRate: number;
    initialRate?: number;
    inferenceIntervalMs?: number;
  }) {
    this.steps = rateSteps(maxRate);
    const wanted = Math.max(1, Math.round(initialRate));
    let closest = 0;
    this.steps.forEach((rate, index) => {
      if (Math.abs(rate - wanted) < Math.abs(this.steps[closest] - wanted)) closest = index;
    });
    this.index = closest;
    this.inferenceIntervalMs = Math.max(1, inferenceIntervalMs);
  }

  get rate(): number {
    return this.steps[this.index];
  }

  get maxRate(): number {
    return this.steps[this.steps.length - 1];
  }

  observe(timing: PipelineTiming): RateDecision {
    const rate = this.rate;
    const frameMs = 1000 / rate;
    // Inference is intentionally cadence weighted. A 30 ms inference pass that
    // runs every 220 ms must not make a 120 fps capture pretend it needs 30 ms
    // for every frame. Capture and inference remain independent loops.
    const inferenceShare =
      finite(timing.inferenceMs) * Math.min(1, frameMs / this.inferenceIntervalMs);
    const totalMs =
      finite(timing.nativeMs) +
      finite(timing.transferMs) +
      finite(timing.decodeMs) +
      finite(timing.paintMs) +
      inferenceShare;
    const budgetMs = frameMs * 0.82;
    const slow = timing.dropped || totalMs > budgetMs;
    const fast = !timing.dropped && totalMs < frameMs * 0.42;

    if (slow) {
      this.slow += 1;
      this.fast = 0;
    } else if (fast) {
      this.fast += 1;
      this.slow = 0;
    } else {
      this.slow = 0;
      this.fast = 0;
    }

    let reason: RateDecision["reason"] = timing.dropped ? "drop" : "hold";
    let changed = false;
    if (this.slow >= 3 && this.index > 0) {
      this.index -= 1;
      this.slow = 0;
      this.fast = 0;
      reason = "slow";
      changed = true;
    } else if (this.fast >= 10 && this.index < this.steps.length - 1) {
      this.index += 1;
      this.slow = 0;
      this.fast = 0;
      reason = "fast";
      changed = true;
    }

    return { rate: this.rate, changed, totalMs, budgetMs, reason };
  }
}

export function makeFastTiming(): PipelineTiming {
  return {
    nativeMs: 1,
    transferMs: 1,
    decodeMs: 1,
    paintMs: 1,
    inferenceMs: 2,
    dropped: false,
  };
}

export function makeSlowTiming(): PipelineTiming {
  return {
    nativeMs: 8,
    transferMs: 8,
    decodeMs: 12,
    paintMs: 4,
    inferenceMs: 30,
    dropped: false,
  };
}
