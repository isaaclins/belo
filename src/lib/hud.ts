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

export interface HudDiagnostics {
  macDesktop: boolean;
  platform: string;
  maxTouchPoints: number;
  tauriInternals: boolean;
  eligible: boolean;
  enabled: boolean;
  focused: boolean;
  visible: boolean;
  runtimeNull: boolean;
}

interface HudWindow {
  isFocused(): Promise<boolean>;
  hide(): Promise<void>;
  onFocusChanged(handler: (event: { payload: boolean }) => void): Promise<() => void>;
  once<T>(event: string, handler: (event: { payload: T }) => void): Promise<() => void>;
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

function hudEnvironment(): {
  macDesktop: boolean;
  platform: string;
  userAgent: string;
  maxTouchPoints: number;
  tauriInternals: boolean;
} {
  const platform = typeof navigator === "undefined" ? "" : navigator.platform;
  const userAgent = typeof navigator === "undefined" ? "" : navigator.userAgent;
  const maxTouchPoints = typeof navigator === "undefined" ? 0 : navigator.maxTouchPoints;
  return {
    macDesktop: /Mac/i.test(platform || userAgent) && !(maxTouchPoints > 1),
    platform,
    userAgent,
    maxTouchPoints,
    tauriInternals: typeof window !== "undefined" && "__TAURI_INTERNALS__" in window,
  };
}

function isMacDesktop(): boolean {
  return hudEnvironment().macDesktop;
}

function logHudProbe(event: string, details: object): void {
  console.info(`[belo-hud] ${event} ${JSON.stringify(details)}`);
}


function makeHudDiagnostics(
  eligible: boolean,
  enabled: boolean,
  focused: boolean,
  runtimeNull: boolean,
): HudDiagnostics {
  const environment = hudEnvironment();
  return {
    ...environment,
    eligible,
    enabled,
    focused,
    visible: shouldShowHud({ eligible, focused, enabled, macDesktop: environment.macDesktop }),
    runtimeNull,
  };
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

/**
 * Creating the HUD window makes macOS report the main window as focused for a
 * moment, even though the HUD is created unfocusable and hidden. Trusting that
 * event tears the HUD down before it is ever shown, so a focus gain that lands
 * during creation is verified against the real window state instead.
 */
const FOCUS_SETTLE_MS = 700;

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
  if (testRuntime) {
    logHudProbe("runtime", { ...hudEnvironment(), testRuntime: true, runtimeNull: false });
    return testRuntime;
  }
  const environment = hudEnvironment();
  if (!environment.macDesktop || !environment.tauriInternals) {
    logHudProbe("runtime", { ...environment, testRuntime: false, runtimeNull: true });
    return null;
  }
  const [{ WebviewWindow }, events, windowApi] = await Promise.all([
    import("@tauri-apps/api/webviewWindow"),
    import("@tauri-apps/api/event"),
    import("@tauri-apps/api/window"),
  ]);
  logHudProbe("runtime", { ...environment, testRuntime: false, runtimeNull: false });
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
  private pendingCreation: Promise<void> | null = null;
  private pendingWindow: HudWindow | null = null;
  private latest: HudState | null = null;
  private generation = 0;
  private visible = false;

  setRuntime(runtime: HudRuntime): void {
    this.runtime = runtime;
  }

  hasRuntime(): boolean {
    return this.runtime !== null;
  }

  /**
   * State updates arrive on every inference tick. They must never touch the
   * window lifecycle: a create/show sequence spans several awaits, and letting
   * a newer frame invalidate it means the HUD is created hidden and never
   * shown. Streaming state goes through push() instead.
   */
  async push(state: HudState): Promise<void> {
    this.latest = state;
    const runtime = this.runtime;
    // While creation is in flight the newest state is emitted once it settles.
    if (!runtime || !this.window || this.pendingCreation) return;
    await runtime.emitTo(HUD_WINDOW_LABEL, HUD_STATE_EVENT, state);
  }

  /**
   * Creating the webview activates the application, which would immediately
   * make the main window focused again and hide the HUD that was just built.
   * The window is therefore created while capture starts, when the app is
   * expected to be focused anyway, and later focus changes only show or hide
   * this existing window.
   */
  async arm(armed: boolean, runtime: HudRuntime): Promise<void> {
    this.runtime = runtime;
    if (!armed) {
      await this.close();
      return;
    }
    if (this.window || this.pendingCreation) return;
    const generation = this.generation;
    // A window left over from a previous run is replaced rather than reused:
    // reusing a stale handle leaves the HUD registered but never visible.
    const existing = await runtime.getByLabel(HUD_WINDOW_LABEL);
    if (generation !== this.generation) return;
    if (existing) {
      try {
        if (existing.destroy) await existing.destroy();
        else await existing.close();
      } catch {
        // A window that cannot be destroyed is replaced below anyway.
      }
      if (generation !== this.generation) return;
    }
    const hudWindow = this.create(runtime);
    this.window = hudWindow;
    const creation = this.waitForCreation(hudWindow);
    this.pendingCreation = creation;
    this.pendingWindow = hudWindow;
    try {
      await creation;
    } catch (error) {
      if (this.window === hudWindow) this.window = null;
      throw error;
    } finally {
      if (this.pendingCreation === creation) {
        this.pendingCreation = null;
        this.pendingWindow = null;
      }
    }
    if (this.window !== hudWindow) return;
    try {
      await hudWindow.setAlwaysOnTop(true);
      // A HUD click must not activate the app. The HUD is still mouse
      // interactive, but it never becomes the focused application window.
      await hudWindow.setFocusable(false);
      this.unlistenMoved = await hudWindow.onMoved(({ payload }) => writePosition(payload));
      // The window is kept hidden until a focus loss asks for it.
      await hudWindow.hide();
    } catch (error) {
      if (this.window === hudWindow) await this.close();
      throw error;
    }
    // The app can already be unfocused by the time the window is ready, so the
    // standing decision is applied rather than waiting for the next change.
    await this.applyVisibility();
  }

  async sync(visible: boolean, state: HudState): Promise<void> {
    this.latest = state;
    if (visible === this.visible) {
      await this.push(state);
      return;
    }
    this.visible = visible;
    await this.applyVisibility();
  }

  private async applyVisibility(): Promise<void> {
    const runtime = this.runtime;
    const hudWindow = this.window;
    if (!runtime || !hudWindow) return;

    if (!this.visible) {
      await hudWindow.hide();
      await screen.refreshFilter();
      return;
    }

    try {
      // Refresh before and after showing. The first pass catches the hidden
      // webview; the second catches the visible HUD. If either pass fails the
      // HUD stays hidden, avoiding a self-capture feedback loop.
      await screen.refreshFilter();
      await hudWindow.show();
      await screen.refreshFilter();
    } catch (error) {
      await hudWindow.hide();
      throw error;
    }
    if (this.latest) await runtime.emitTo(HUD_WINDOW_LABEL, HUD_STATE_EVENT, this.latest);
  }

  async resend(): Promise<void> {
    if (!this.runtime || !this.window || !this.latest) return;
    await this.runtime.emitTo(HUD_WINDOW_LABEL, HUD_STATE_EVENT, this.latest);
  }

  async close(): Promise<void> {
    this.generation += 1;
    this.visible = false;
    this.unlistenMoved?.();
    this.unlistenMoved = null;
    const window = this.window;
    this.window = null;
    if (!window) return;
    if (this.pendingWindow === window && this.pendingCreation) {
      try {
        await this.pendingCreation;
      } catch {
        return;
      }
    }
    if (window.destroy) await window.destroy();
    else await window.close();
  }

  private waitForCreation(window: HudWindow): Promise<void> {
    return new Promise((resolve, reject) => {
      void window.once("tauri://created", () => resolve()).catch(reject);
      void window.once("tauri://error", ({ payload }) => {
        reject(payload instanceof Error ? payload : new Error(String(payload)));
      }).catch(reject);
    });
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
  onDiagnostics,
}: {
  eligible: boolean;
  state: HudState;
  onCommand: (command: HudCommand) => void;
  onError?: (message: string) => void;
  onDiagnostics?: (diagnostics: HudDiagnostics) => void;
}) {
  const [enabled, setEnabledState] = useState(readEnabled);
  const [focused, setFocused] = useState(true);
  const [runtimeReady, setRuntimeReady] = useState(false);
  const commandRef = useRef(onCommand);
  const stateRef = useRef(state);
  const diagnosticsRef = useRef(onDiagnostics);
  commandRef.current = onCommand;
  stateRef.current = state;
  diagnosticsRef.current = onDiagnostics;
  const manager = useMemo(() => new HudManager(), []);
  const ignoreFocusUntilRef = useRef(0);
  const previousVisibleRef = useRef(false);
  const reportDiagnostics = (currentFocused: boolean, runtimeNull: boolean) => {
    const diagnostics = makeHudDiagnostics(eligible, enabled, currentFocused, runtimeNull);
    logHudProbe("state", diagnostics);
    diagnosticsRef.current?.(diagnostics);
  };

  useEffect(() => {
    let cancelled = false;
    let unlistenFocus: (() => void) | null = null;
    let unlistenCommand: (() => void) | null = null;
    let unlistenReady: (() => void) | null = null;
    void defaultRuntime().then(async (runtime) => {
      if (cancelled) return;
      if (!runtime) {
        reportDiagnostics(true, true);
        return;
      }
      manager.setRuntime(runtime);
      const main = runtime.currentWindow();
      unlistenFocus = await main.onFocusChanged(({ payload }) => {
        if (payload && Date.now() < ignoreFocusUntilRef.current) {
          window.setTimeout(() => {
            void main.isFocused().then((real) => {
              setFocused(real);
              reportDiagnostics(real, false);
            });
          }, FOCUS_SETTLE_MS);
          return;
        }
        setFocused(payload);
        reportDiagnostics(payload, false);
      });
      const currentFocused = await main.isFocused();
      setFocused(currentFocused);
      reportDiagnostics(currentFocused, false);
      unlistenCommand = await runtime.listen<HudCommand>(HUD_COMMAND_EVENT, (command) => {
        commandRef.current(command);
      });
      unlistenReady = await runtime.listen<null>(HUD_READY_EVENT, () => void manager.resend());
      setRuntimeReady(true);
    });
    return () => {
      cancelled = true;
      unlistenFocus?.();
      unlistenCommand?.();
      unlistenReady?.();
      void manager.close();
    };
  }, [manager]);

  // Arming builds the hidden window while capture starts. Doing this on focus
  // loss instead would activate the app and hide the HUD immediately.
  const armed = eligible && enabled;
  useEffect(() => {
    if (!runtimeReady) return;
    void defaultRuntime().then((runtime) => {
      if (!runtime) return;
      return manager
        .arm(armed, runtime)
        .catch((error: unknown) => onError?.(error instanceof Error ? error.message : String(error)));
    });
  }, [armed, manager, onError, runtimeReady]);

  useEffect(() => {
    const visible = shouldShowHud({ eligible, focused, enabled });
    if (visible && !previousVisibleRef.current) {
      ignoreFocusUntilRef.current = Date.now() + FOCUS_SETTLE_MS;
    }
    previousVisibleRef.current = visible;
    reportDiagnostics(focused, manager.hasRuntime() === false);
    void manager
      .sync(visible, state)
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
