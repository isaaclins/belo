// One detection score cutoff for the whole app.
//
// The Use view owns the picker, but the box annotator's guesses have to be
// filtered by the same number: a person who decided that anything under 0.5 is
// noise means that everywhere, not only in front of the camera. Like the
// thumbnail size and the capture rate this is a property of the machine rather
// than of a project, so it lives in localStorage and not in the manifest.

import { useCallback, useSyncExternalStore } from "react";

/** The cutoffs the Use view offers, lowest first. */
export const THRESHOLDS = [0.2, 0.3, 0.5, 0.7] as const;
export type ScoreThreshold = (typeof THRESHOLDS)[number];

const DEFAULT_THRESHOLD: ScoreThreshold = 0.3;
const KEY = "belo.detect.scoreThreshold";

function isThreshold(value: number): value is ScoreThreshold {
  return (THRESHOLDS as readonly number[]).includes(value);
}

function read(): ScoreThreshold {
  try {
    const raw = Number(window.localStorage.getItem(KEY));
    return isThreshold(raw) ? raw : DEFAULT_THRESHOLD;
  } catch {
    return DEFAULT_THRESHOLD;
  }
}

let current: ScoreThreshold = read();
const listeners = new Set<() => void>();

export function scoreThreshold(): ScoreThreshold {
  return current;
}

export function setScoreThreshold(value: ScoreThreshold) {
  if (value === current) return;
  current = value;
  try {
    window.localStorage.setItem(KEY, String(value));
  } catch {
    // A webview with storage disabled still keeps the choice for this session.
  }
  listeners.forEach((fn) => fn());
}

export function useScoreThreshold(): ScoreThreshold {
  const subscribe = useCallback((fn: () => void) => {
    listeners.add(fn);
    return () => listeners.delete(fn);
  }, []);
  return useSyncExternalStore(subscribe, scoreThreshold, scoreThreshold);
}
