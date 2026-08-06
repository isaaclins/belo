// Headless coverage for the macOS capture HUD.
//
// The runtime below is a window and event double, not a real desktop window.
// It proves the lifecycle contract without taking focus or opening anything.

import { useState } from "react";
import { createRoot } from "react-dom/client";
import { HudView } from "../../src/views/HudView";
import {
  HUD_COMMAND_EVENT,
  HUD_READY_EVENT,
  HUD_STATE_EVENT,
  setHudRuntimeForTests,
  shouldShowHud,
  useScreenHud,
  type HudCommand,
  type HudState,
} from "../../src/lib/hud";
import { setScreenSource, type ScreenSource } from "../../src/lib/screen";
import { useStore } from "../../src/lib/store";
import { installTauriShim, reporter } from "./tauri-shim";

import "../../src/styles/tokens.css";
import "../../src/styles/base.css";
import "../../src/styles/ui.css";
import "../../src/styles/app.css";

installTauriShim();
Object.defineProperty(navigator, "platform", { configurable: true, value: "MacIntel" });
const { check, fail, finish } = reporter();

const wait = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

async function until(what: string, probe: () => boolean, timeoutMs = 10_000): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (!probe()) {
    if (Date.now() > deadline) throw new Error(`Timed out waiting for ${what}`);
    await wait(25);
  }
}

const frameCanvas = (): HTMLCanvasElement => {
  const canvas = document.createElement("canvas");
  canvas.width = 80;
  canvas.height = 80;
  const context = canvas.getContext("2d");
  if (!context) throw new Error("canvas unavailable");
  context.fillStyle = "#ff5c8a";
  context.fillRect(0, 0, 80, 80);
  return canvas;
};

const frameUrl = () => frameCanvas().toDataURL("image/jpeg", 0.72);

function makeRuntime() {
  const listeners = new Map<string, Set<(payload: unknown) => void>>();
  const focusListeners = new Set<(focused: boolean) => void>();
  const main = {
    focused: true,
    async isFocused() {
      return main.focused;
    },
    async onFocusChanged(handler: (event: { payload: boolean }) => void) {
      const callback = (focused: boolean) => handler({ payload: focused });
      focusListeners.add(callback);
      return () => focusListeners.delete(callback);
    },
  };
  const metrics = {
    created: 0,
    shown: 0,
    hidden: 0,
    destroyed: 0,
    refreshes: 0,
    focusThefts: 0,
    options: null as Record<string, unknown> | null,
    window: null as {
      destroyed: boolean;
      setAlwaysOnTop: boolean;
      focusable: boolean;
      visible: boolean;
    } | null,
  };

  const addListener = (event: string, handler: (payload: unknown) => void) => {
    const current = listeners.get(event) ?? new Set<(payload: unknown) => void>();
    current.add(handler);
    listeners.set(event, current);
    return () => current.delete(handler);
  };
  const dispatch = (event: string, payload: unknown) => {
    for (const handler of listeners.get(event) ?? []) handler(payload);
  };

  const runtime = {
    currentWindow: () => main,
    async getByLabel() {
      return metrics.window && !metrics.window.destroyed ? makeHudWindow(false) : null;
    },
    create(_label: string, options: Record<string, unknown>) {
      metrics.created += 1;
      metrics.options = options;
      return makeHudWindow(true);
    },
    async emitTo(target: string, event: string, payload: unknown) {
      if (target === "belo-hud" || target === "main") dispatch(event, payload);
    },
    async listen<T>(event: string, handler: (payload: T) => void) {
      return addListener(event, (payload) => handler(payload as T));
    },
    focus(value: boolean) {
      main.focused = value;
      for (const handler of focusListeners) handler(value);
    },
    focusListenerCount: () => focusListeners.size,
    dispatch,
    metrics,
  };

  function makeHudWindow(created: boolean) {
    const record = metrics.window ?? {
      destroyed: false,
      setAlwaysOnTop: false,
      focusable: true,
      visible: false,
    };
    metrics.window = record;
    return {
      async isFocused() {
        return false;
      },
      async onFocusChanged() {
        return () => undefined;
      },
      async once(event: string, handler: (event: { payload: unknown }) => void) {
        if (event === "tauri://created") handler({ payload: null });
        return () => undefined;
      },
      async show() {
        metrics.shown += 1;
        record.visible = true;
        if (record.focusable) metrics.focusThefts += 1;
      },
      async hide() {
        metrics.hidden += 1;
        record.visible = false;
      },
      async close() {
        record.destroyed = true;
      },
      async destroy() {
        record.destroyed = true;
        metrics.destroyed += 1;
      },
      async setAlwaysOnTop(value: boolean) {
        record.setAlwaysOnTop = value;
      },
      async setFocusable(value: boolean) {
        record.focusable = value;
      },
      async onMoved() {
        return () => undefined;
      },
    };
  }

  return runtime;
}

let screenRefreshes = 0;

const fakeScreen: ScreenSource = {
  isSupported: () => true,
  supported: async () => true,
  canList: () => true,
  list: async () => [],
  permission: async () => "granted",
  openSettings: async () => undefined,
  start: async () => {
    throw new Error("not needed in HUD check");
  },
  stop: () => undefined,
  activeDisplayId: () => null,
  reportInference: () => undefined,
  refreshFilter: async () => {
    screenRefreshes += 1;
  },
};

const lifecycleState: HudState = {
  mode: "classify",
  projectName: "Lifecycle",
  frozen: false,
  saving: false,
  frameDataUrl: null,
  labels: [],
  activeLabelId: null,
  verdict: null,
  boxCount: 0,
};

function LifecycleHarness({ runtime }: { runtime: ReturnType<typeof makeRuntime> }) {
  const [eligible, setEligible] = useState(true);
  const hud = useScreenHud({
    eligible,
    state: lifecycleState,
    onCommand: () => undefined,
  });
  (window as unknown as { hud: typeof hud }).hud = hud;
  return (
    <div>
      <button id="toggle" onClick={() => hud.setEnabled(!hud.enabled)}>
        Toggle HUD
      </button>
      <button id="eligible" onClick={() => setEligible((value) => !value)}>
        Toggle eligibility
      </button>
    </div>
  );
}

async function main(): Promise<void> {
  // The HUD is macOS-only, but this browser check runs on Linux CI too. Keep
  // the platform gate real in production and emulate only the browser signal.
  Object.defineProperty(navigator, "platform", { configurable: true, value: "MacIntel" });
  setScreenSource(fakeScreen);
  const runtime = makeRuntime();
  setHudRuntimeForTests(runtime as never);
  let root = createRoot(document.getElementById("mount")!);
  root.render(<LifecycleHarness runtime={runtime} />);
  await until("the lifecycle harness", () => Boolean((window as unknown as { hud?: unknown }).hud));
  await until("the main focus listener", () => runtime.focusListenerCount() > 0);
  await wait(80);

  // Creating the webview activates the app, which would immediately refocus the
  // main window and hide the HUD again. The window is therefore built while
  // capture arms and kept hidden until focus is actually lost.
  await until("the HUD window to be prepared", () => runtime.metrics.created === 1);
  check("the HUD window is prepared while capture is armed", runtime.metrics.created === 1);
  check("a prepared HUD stays hidden while the main window is focused", runtime.metrics.shown === 0 && runtime.metrics.window?.visible === false);
  check("the pure lifecycle gate requires an unfocused live screen", !shouldShowHud({ eligible: true, focused: true, enabled: true, macDesktop: true }));

  runtime.focus(false);
  await until("the HUD to appear", () => runtime.metrics.shown > 0);
  check("HUD appears only after the main window loses focus", runtime.metrics.window?.visible === true);
  check("losing focus reuses the prepared window instead of building another", runtime.metrics.created === 1);
  check("HUD is always on top", runtime.metrics.window?.setAlwaysOnTop === true);
  check("HUD is not focusable and never steals focus", runtime.metrics.window?.focusable === false && runtime.metrics.focusThefts === 0);
  check("native filter refresh is requested around HUD visibility", screenRefreshes >= 2, { screenRefreshes });

  runtime.focus(true);
  await until("the HUD to hide on main focus", () => runtime.metrics.window?.visible === false);
  check("main focus hides the HUD without destroying it", runtime.metrics.window?.destroyed === false);

  runtime.focus(false);
  await until("the HUD to return on a second focus loss", () => runtime.metrics.window?.visible === true);
  check("a second focus loss shows the HUD again", runtime.metrics.shown >= 2 && runtime.metrics.created === 1);

  document.getElementById("toggle")?.click();
  await until("the HUD to be torn down when switched off", () => runtime.metrics.destroyed === 1);
  check("turning the HUD off tears the window down", runtime.metrics.window?.destroyed === true);

  root.unmount();
  root = createRoot(document.getElementById("mount")!);

  // Classification teaching uses the same store intake as the main Use view.
  const classify = useStore.getState();
  await classify.createProject("HUD classify", "single");
  const red = useStore.getState().addLabel("Red");
  if (!red) throw new Error("classification label missing");
  const classifyFrame = frameCanvas();
  const commands: HudCommand[] = [];
  await runtime.listen<HudCommand>(HUD_COMMAND_EVENT, async (command) => {
    commands.push(command);
    if (command.type === "teach") {
      await useStore.getState().addFromDrawable(classifyFrame, command.labelIds, "feedback", "screen");
    }
  });
  root.render(<HudView />);
  await wait(100);
  const classifyState: HudState = {
    mode: "classify",
    projectName: "HUD classify",
    frozen: true,
    saving: false,
    frameDataUrl: classifyFrame.toDataURL("image/jpeg", 0.72),
    labels: [{ id: red, name: "Red", color: 2 }],
    activeLabelId: red,
    verdict: null,
    boxCount: 0,
  };
  runtime.emitTo("belo-hud", HUD_STATE_EVENT, classifyState);
  await until("the frozen classify controls", () => Boolean(document.querySelector(".hud__teach")));
  document.querySelector<HTMLElement>(".hud__label")?.click();
  await wait(20);
  document.querySelector<HTMLElement>(".hud__teachgo")?.click();
  await until("the taught frame", () => (useStore.getState().manifest?.images.length ?? 0) === 1);
  const taught = useStore.getState().manifest?.images[0];
  check("frozen classify teaching uses the normal feedback intake", taught?.source === "feedback" && taught.feed === "screen", taught);

  // Detection drawing writes the frozen capture through the normal capture and
  // box store paths, retaining feed metadata rather than making a HUD-only record.
  await useStore.getState().closeProject();
  await useStore.getState().createProject("HUD detect", "detect");
  const object = useStore.getState().addLabel("Object");
  if (!object) throw new Error("detection label missing");
  const detectFrame = frameCanvas();
  const saved = await useStore.getState().addFromDrawable(detectFrame, null, "capture", "screen");
  const detectImage = useStore.getState().manifest?.images[0];
  if (!saved || !detectImage) throw new Error("frozen detect frame did not save");
  commands.length = 0;
  root.unmount();
  root = createRoot(document.getElementById("mount")!);
  root.render(<HudView />);
  await wait(100);
  runtime.emitTo("belo-hud", HUD_STATE_EVENT, {
    mode: "detect",
    projectName: "HUD detect",
    frozen: true,
    saving: false,
    frameDataUrl: detectFrame.toDataURL("image/jpeg", 0.72),
    labels: [{ id: object, name: "Object", color: 3 }],
    activeLabelId: object,
    verdict: null,
    boxCount: 0,
  } satisfies HudState);
  await runtime.listen<HudCommand>(HUD_COMMAND_EVENT, async (command) => {
    commands.push(command);
    if (command.type === "box") useStore.getState().addBox(detectImage.id, command.labelId, command.rect);
  });
  await until("the frozen detect frame", () => Boolean(document.querySelector(".hud__frame")));
  const frame = document.querySelector<HTMLElement>(".hud__frame");
  if (!frame) throw new Error("HUD frame did not mount");
  const rect = frame.getBoundingClientRect();
  frame.dispatchEvent(new PointerEvent("pointerdown", { bubbles: true, button: 0, clientX: rect.left + rect.width * 0.15, clientY: rect.top + rect.height * 0.15, pointerId: 1 }));
  await wait(20);
  frame.dispatchEvent(new PointerEvent("pointerup", { bubbles: true, button: 0, clientX: rect.left + rect.width * 0.75, clientY: rect.top + rect.height * 0.8, pointerId: 1 }));
  await until("the HUD box to reach the store", () => (useStore.getState().manifest?.images[0]?.boxes?.length ?? 0) === 1);
  const boxed = useStore.getState().manifest?.images[0];
  check("HUD box drawing writes through addBox", boxed?.boxes?.length === 1);
  check("HUD box image keeps screen capture metadata", boxed?.source === "capture" && boxed.feed === "screen", boxed);
  check("HUD box command carries the selected label", commands.some((command) => command.type === "box" && command.labelId === object));

  runtime.dispatch(HUD_COMMAND_EVENT, { type: "freeze" } satisfies HudCommand);
  runtime.dispatch(HUD_COMMAND_EVENT, { type: "resume" } satisfies HudCommand);
  check("HUD exposes freeze and resume commands", commands.some((command) => command.type === "freeze") && commands.some((command) => command.type === "resume"));
  check("HUD ready handshake is addressable to the main window", HUD_READY_EVENT === "belo-hud-ready");

  root.unmount();
  setHudRuntimeForTests(null);
  finish();
}

void main().catch(fail);
