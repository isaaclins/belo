import { useCallback, useSyncExternalStore } from "react";
import type { ScreenStartOptions } from "./screen";

export const SCREEN_QUALITY_STEPS = ["auto", "smooth", "balanced", "detail"] as const;
export type ScreenQuality = (typeof SCREEN_QUALITY_STEPS)[number];

export interface ScreenQualityPreset {
  id: ScreenQuality;
  label: string;
  requestedFps: number;
  maxEdge: number;
  jpegQuality: number;
}

/**
 * Auto starts at the display mode that ScreenCaptureKit reports. The named
 * steps are caps, not promises: the adaptive controller can step any of them
 * down when the measured pipeline cannot keep up.
 */
export const SCREEN_QUALITY_PRESETS: readonly ScreenQualityPreset[] = [
  { id: "auto", label: "Auto", requestedFps: 0, maxEdge: 1120, jpegQuality: 55 },
  { id: "smooth", label: "Smooth", requestedFps: 30, maxEdge: 960, jpegQuality: 55 },
  { id: "balanced", label: "Balanced", requestedFps: 60, maxEdge: 1440, jpegQuality: 68 },
  { id: "detail", label: "Detail", requestedFps: 120, maxEdge: 1920, jpegQuality: 76 },
];

const KEY = "belo.screen.quality";
const listeners = new Set<() => void>();

function isQuality(value: string): value is ScreenQuality {
  return (SCREEN_QUALITY_STEPS as readonly string[]).includes(value);
}

function read(): ScreenQuality {
  try {
    const value = window.localStorage.getItem(KEY) ?? "auto";
    return isQuality(value) ? value : "auto";
  } catch {
    return "auto";
  }
}

let current: ScreenQuality = read();

export function screenQuality(): ScreenQuality {
  return current;
}

export function setScreenQuality(value: ScreenQuality): void {
  if (value === current) return;
  current = value;
  try {
    window.localStorage.setItem(KEY, value);
  } catch {
    // A webview with storage disabled forgets the preference on restart.
  }
  listeners.forEach((listener) => listener());
}

export function screenQualityPreset(value: ScreenQuality): ScreenQualityPreset {
  return SCREEN_QUALITY_PRESETS.find((preset) => preset.id === value) ?? SCREEN_QUALITY_PRESETS[0];
}

export function screenStartOptions(
  value: ScreenQuality,
  onMetrics?: ScreenStartOptions["onMetrics"],
): ScreenStartOptions {
  const preset = screenQualityPreset(value);
  return { ...preset, onMetrics };
}

export function useScreenQuality(): [ScreenQuality, (value: ScreenQuality) => void] {
  const subscribe = useCallback((listener: () => void) => {
    listeners.add(listener);
    return () => listeners.delete(listener);
  }, []);
  return [useSyncExternalStore(subscribe, screenQuality, screenQuality), setScreenQuality];
}
