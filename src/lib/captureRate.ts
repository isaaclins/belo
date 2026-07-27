// One capture rate for the whole app.
//
// The Label view burst and the Use view teach hold are the same idea, so they
// read the same setting rather than each carrying their own constant. It lives
// outside the project store because it is a preference of this machine, not a
// property of the project.

import { useCallback, useSyncExternalStore } from "react";

export const CAPTURE_RATES = [2, 4, 8, 12] as const;
export type CaptureRate = (typeof CAPTURE_RATES)[number];

const DEFAULT_RATE: CaptureRate = 8;
const KEY = "belo.capture.fps";

/** Never let more than this many writes be in flight, so a slow disk drops frames instead of queuing. */
export const MAX_IN_FLIGHT = 2;

function isRate(value: number): value is CaptureRate {
  return (CAPTURE_RATES as readonly number[]).includes(value);
}

function read(): CaptureRate {
  try {
    const raw = Number(window.localStorage.getItem(KEY));
    return isRate(raw) ? raw : DEFAULT_RATE;
  } catch {
    return DEFAULT_RATE;
  }
}

let current: CaptureRate = read();
const listeners = new Set<() => void>();

export function captureRate(): CaptureRate {
  return current;
}

export function setCaptureRate(rate: CaptureRate) {
  if (rate === current) return;
  current = rate;
  try {
    window.localStorage.setItem(KEY, String(rate));
  } catch {
    // A webview with storage disabled just forgets the choice.
  }
  listeners.forEach((fn) => fn());
}

/** Milliseconds between two captures at the current rate. */
export function captureIntervalMs(rate: CaptureRate): number {
  return Math.round(1000 / rate);
}

export function useCaptureRate(): [CaptureRate, (rate: CaptureRate) => void] {
  const subscribe = useCallback((fn: () => void) => {
    listeners.add(fn);
    return () => listeners.delete(fn);
  }, []);
  const rate = useSyncExternalStore(subscribe, captureRate, captureRate);
  return [rate, setCaptureRate];
}
