// One thumbnail size for the whole app.
//
// The Label grid is the only place that reads it today, but the choice is a
// property of this machine and not of a project, so it lives in localStorage
// next to the capture rate rather than in the manifest.

import { useCallback, useSyncExternalStore } from "react";

/** Grid cell edge in pixels. Four steps, small to large. */
export const THUMB_SIZES = [92, 126, 170, 224] as const;
export type ThumbSize = (typeof THUMB_SIZES)[number];

const DEFAULT_SIZE: ThumbSize = 126;
const KEY = "belo.grid.thumb";

function isSize(value: number): value is ThumbSize {
  return (THUMB_SIZES as readonly number[]).includes(value);
}

function read(): ThumbSize {
  try {
    const raw = Number(window.localStorage.getItem(KEY));
    return isSize(raw) ? raw : DEFAULT_SIZE;
  } catch {
    return DEFAULT_SIZE;
  }
}

let current: ThumbSize = read();
const listeners = new Set<() => void>();

export function thumbSize(): ThumbSize {
  return current;
}

export function setThumbSize(size: ThumbSize) {
  if (size === current) return;
  current = size;
  try {
    window.localStorage.setItem(KEY, String(size));
  } catch {
    // A webview with storage disabled just forgets the choice.
  }
  listeners.forEach((fn) => fn());
}

/** Moves one step along THUMB_SIZES and stops at the ends. */
export function stepThumbSize(direction: -1 | 1) {
  const next = THUMB_SIZES.indexOf(current) + direction;
  if (next < 0 || next >= THUMB_SIZES.length) return;
  setThumbSize(THUMB_SIZES[next]);
}

export function useThumbSize(): ThumbSize {
  const subscribe = useCallback((fn: () => void) => {
    listeners.add(fn);
    return () => listeners.delete(fn);
  }, []);
  return useSyncExternalStore(subscribe, thumbSize, thumbSize);
}
