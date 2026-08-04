import { useEffect, useMemo, useRef, useState } from "react";
import type { BoxRect } from "./types";
import { screen } from "./screen";

export const HUD_WINDOW_LABEL = "belo-hud";
export const HUD_STATE_EVENT = "belo-hud-state";
export const HUD_COMMAND_EVENT = "belo-hud-command";
export const HUD_READY_EVENT = "belo-hud-ready";

export interface HudLabel {
  id: string;
  name: string;
  color: number;
}

export interface HudState {
  mode: "classify" | "detect";
  projectName: string;
  frozen: boolean;
  saving: boolean;
  frameDataUrl: string | null;
  labels: HudLabel[];
  activeLabelId: string | null;
  verdict: {
    labelId: string | null;
    labelIds: string[];
    confidence: number;
  } | null;
  boxCount: number;
  achievedFps?: number;
}

export type HudCommand =
  | { type: "freeze" }
  | { type: "resume" }
  | { type: "hide" }
  | { type: "teach"; labelIds: string[] }
  | { type: "set-label"; labelId: string }
  | { type: "box"; labelId: string; rect: BoxRect };

interface HudWindow {
  isFocused(): Promise<boolean>;
  onFocusChanged(handler: (event: { payload: boolean }) => void): Promise<() => void>;
  show(): Promise<void>;
  close(): Promise<void>;
  destroy?(): Promise<void>;
  setAlwaysOnTop(value: boolean): Promise<void>;
  setFocusable(value: boolean): Promise<void>;
  onMoved(handler: (event: { payload: { x: number; y: number } }) => void): Promise<() => void>;
}

interface HudRuntime {
  currentWindow(): HudWindow;
  getByLabel(label: string): Promise<HudWindow | null>;
  create(label: string, options: Record<string, unknown>): HudWindow;
  emitTo<T>(target: string, event: string, payload?: T): Promise<void>;
  listen<T>(event: string, handler: (payload: T) => void): Promise<() => void>;
}

let testRuntime: HudRuntime | null = null;

/** Headless checks inject this instead of creating a native window. */
export function setHudRuntimeForTests(runtime: HudRuntime | null): void {
  testRuntime = runtime;
}

function isMacDesktop(): boolean {
  return (
    typeof navigator !== "undefined" &&
    /Mac/i.test(navigator.platform || navigator.userAgent) &&
    !(navigator.maxTouchPoints > 1)
  );
}

export function isHudWindow(): boolean {
  if (typeof window === "undefined") return false;
  return new URLSearchParams(window.location.search).has("belo-hud");
}

export function shouldShowHud({
  eligible,
  focused,
  enabled,
  macDesktop = isMacDesktop(),
}: {
  eligible: boolean;
  focused: boolean;
  enabled: boolean;
  macDesktop?: boolean;
}): boolean {
  return macDesktop && eligible && enabled && !focused;
}

const ENABLED_KEY = "belo.screen.hud";
const POSITION_KEY = "belo.screen.hud.position";

function readEnabled(): boolean {
  try {
    const value = window.localStorage.getItem(ENABLED_KEY);
    return value === null ? true : value === "on";
  } catch {
    return true;
  }
}

function writeEnabled(value: boolean): void {
  try {
    window.localStorage.setItem(ENABLED_KEY, value ? "on" : "off");
  } catch {
    // A webview with storage disabled forgets the choice on restart.
  }
}

interface HudPosition {
  x: number;
  y: number;
}

function readPosition(): HudPosition | null {
  try {
    const value = JSON.parse(window.localStorage.getItem(POSITION_KEY) ?? "null") as Partial<HudPosition> | null;
    if (typeof value?.x !== "number" || typeof value.y !== "number") return null;
    return { x: value.x, y: value.y };
  } catch {
    return null;
  }
}

function writePosition(position: HudPosition): void {
  try {
    window.localStorage.setItem(POSITION_KEY, JSON.stringify(position));
  } catch {
    // Position is a convenience, never a reason to block live capture.
  }
}

async function defaultRuntime(): Promise<HudRuntime | null> {
  if (testRuntime) return testRuntime;
  if (!isMacDesktop() || typeof window === "undefined" || !("__TAURI_INTERNALS__" in window)) {
    return null;
  }
  const [{ WebviewWindow }, events, windowApi] = await Promise.all([
    import("@tauri-apps/api/webviewWindow"),
    import("@tauri-apps/api/event"),
    import("@tauri-apps/api/window"),
  ]);
  return {
    currentWindow: () => windowApi.getCurrentWindow(),
    getByLabel: (label) => WebviewWindow.getByLabel(label),
    create: (label, options) => new WebviewWindow(label, options as never),
    emitTo: (target, event, payload) => events.emitTo(target, event, payload),
    listen: <T>(event: string, handler: (payload: T) => void) =>
      events.listen<T>(event, (message) => handler(message.payload)),
  };
}

class HudManager {
  private runtime: HudRuntime | null = null;
  private window: HudWindow | null = null;
  private unlistenMoved: (() => void) | null = null;
  private latest: HudState | null = null;
  private generation = 0;

  setRuntime(runtime: HudRuntime): void {
    this.runtime = runtime;
  }

  async sync(visible: boolean, state: HudState): Promise<void> {
    this.latest = state;
    const generation = ++this.generation;
    if (!visible) {
      await this.close();
      return;
    }
    const runtime = this.runtime;
    if (!runtime) return;
    let created = false;
    if (!this.window) {
      const existing = await runtime.getByLabel(HUD_WINDOW_LABEL);
      if (generation !== this.generation) return;
      this.window = existing ?? this.create(runtime);
      created = true;
      await this.window.setAlwaysOnTop(true);
      // A HUD click must not activate the app. The HUD is still mouse
      // interactive, but it never becomes the focused application window.
      await this.window.setFocusable(false);
      this.unlistenMoved = await this.window.onMoved(({ payload }) => writePosition(payload));
    }
    if (generation !== this.generation || !this.window) return;
    if (created) {
      try {
        // Refresh before and after showing. The first pass catches the newly
        // created hidden webview; the second catches the visible HUD. If either
        // pass fails, the HUD is not shown, avoiding a self-capture feedback loop.
        await screen.refreshFilter();
        await this.window.show();
        await screen.refreshFilter();
      } catch (error) {
        await this.close();
        throw error;
      }
    }
    await runtime.emitTo(HUD_WINDOW_LABEL, HUD_STATE_EVENT, state);
  }

  async resend(): Promise<void> {
    if (!this.runtime || !this.window || !this.latest) return;
    await this.runtime.emitTo(HUD_WINDOW_LABEL, HUD_STATE_EVENT, this.latest);
  }

  async close(): Promise<void> {
    this.generation += 1;
    this.unlistenMoved?.();
    this.unlistenMoved = null;
    const window = this.window;
    this.window = null;
    if (!window) return;
    if (window.destroy) await window.destroy();
    else await window.close();
  }

  private create(runtime: HudRuntime): HudWindow {
    const position = readPosition();
    return runtime.create(HUD_WINDOW_LABEL, {
      url: "index.html?belo-hud=1",
      title: "Belo Capture HUD",
      width: 340,
      height: 280,
      ...(position ? { x: position.x, y: position.y } : { center: true }),
      visible: false,
      focus: false,
      focusable: false,
      decorations: false,
      resizable: false,
      alwaysOnTop: true,
      skipTaskbar: true,
      shadow: true,
    });
  }
}

export function useScreenHud({
  eligible,
  state,
  onCommand,
  onError,
}: {
  eligible: boolean;
  state: HudState;
  onCommand: (command: HudCommand) => void;
  onError?: (message: string) => void;
}) {
  const [enabled, setEnabledState] = useState(readEnabled);
  const [focused, setFocused] = useState(true);
  const commandRef = useRef(onCommand);
  const stateRef = useRef(state);
  commandRef.current = onCommand;
  stateRef.current = state;
  const manager = useMemo(() => new HudManager(), []);

  useEffect(() => {
    let cancelled = false;
    let unlistenFocus: (() => void) | null = null;
    let unlistenCommand: (() => void) | null = null;
    let unlistenReady: (() => void) | null = null;
    void defaultRuntime().then(async (runtime) => {
      if (cancelled || !runtime) return;
      manager.setRuntime(runtime);
      const main = runtime.currentWindow();
      unlistenFocus = await main.onFocusChanged(({ payload }) => setFocused(payload));
      const currentFocused = await main.isFocused();
      setFocused(currentFocused);
      unlistenCommand = await runtime.listen<HudCommand>(HUD_COMMAND_EVENT, (command) => {
        commandRef.current(command);
      });
      unlistenReady = await runtime.listen<null>(HUD_READY_EVENT, () => void manager.resend());
      void manager
        .sync(shouldShowHud({ eligible, focused: currentFocused, enabled }), stateRef.current)
        .catch((error: unknown) => onError?.(error instanceof Error ? error.message : String(error)));
    });
    return () => {
      cancelled = true;
      unlistenFocus?.();
      unlistenCommand?.();
      unlistenReady?.();
      void manager.close();
    };
  }, [manager]);

  useEffect(() => {
    void manager
      .sync(shouldShowHud({ eligible, focused, enabled }), state)
      .catch((error: unknown) => onError?.(error instanceof Error ? error.message : String(error)));
  }, [eligible, enabled, focused, manager, onError, state]);

  const setEnabled = (value: boolean) => {
    setEnabledState(value);
    writeEnabled(value);
  };

  return {
    enabled,
    setEnabled,
    focused,
    visible: shouldShowHud({ eligible, focused, enabled }),
  };
}

export async function emitHudReady(): Promise<void> {
  const runtime = await defaultRuntime();
  if (runtime) await runtime.emitTo("main", HUD_READY_EVENT, null);
}

export async function emitHudCommand(command: HudCommand): Promise<void> {
  const runtime = await defaultRuntime();
  if (runtime) await runtime.emitTo("main", HUD_COMMAND_EVENT, command);
}

export async function listenHudState(handler: (state: HudState) => void): Promise<() => void> {
  const runtime = await defaultRuntime();
  if (!runtime) return () => undefined;
  return runtime.listen<HudState>(HUD_STATE_EVENT, handler);
}
