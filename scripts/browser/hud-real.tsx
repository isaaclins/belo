// Native smoke coverage for the macOS capture HUD.
// This page is loaded by a real Tauri webview. It does not use the browser
// runtime double from check:hud and keeps the main window hidden and unfocused.

import { useCallback, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { invoke } from "@tauri-apps/api/core";
import { WebviewWindow } from "@tauri-apps/api/webviewWindow";
import { setScreenSource, type ScreenSource } from "../../src/lib/screen";
import { useScreenHud, type HudCommand, type HudState } from "../../src/lib/hud";

const wait = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

async function until<T>(what: string, read: () => Promise<T | null>, timeoutMs = 8_000): Promise<T> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const value = await read();
    if (value) return value;
    await wait(50);
  }
  throw new Error(`Timed out waiting for ${what}`);
}

async function report(value: unknown): Promise<void> {
  await invoke("hud_real_check_report", { report: JSON.stringify(value) });
}

const fakeScreen: ScreenSource = {
  isSupported: () => true,
  supported: async () => true,
  canList: () => true,
  list: async () => [],
  permission: async () => "granted",
  openSettings: async () => undefined,
  start: async () => {
    throw new Error("not needed in native HUD check");
  },
  stop: () => undefined,
  activeDisplayId: () => null,
  reportInference: () => undefined,
  refreshFilter: async () => undefined,
};

const state: HudState = {
  mode: "classify",
  projectName: "Native HUD check",
  frozen: false,
  saving: false,
  frameDataUrl: null,
  labels: [],
  activeLabelId: null,
  verdict: null,
  boxCount: 0,
};

interface ProbeResult {
  stage: string;
  platform: string;
  maxTouchPoints: number;
  errors: string[];
  window?: Record<string, unknown>;
  closed?: boolean;
}

function Probe() {
  const [eligible, setEligible] = useState(true);
  const errors: string[] = [];
  const onError = useCallback((message: string) => {
    errors.push(message);
  }, []);
  useScreenHud({
    eligible,
    state,
    onCommand: (_command: HudCommand) => undefined,
    onError,
  });

  useEffect(() => {
    let cancelled = false;
    const run = async () => {
      const base = {
        platform: navigator.platform,
        maxTouchPoints: navigator.maxTouchPoints,
        errors,
      } satisfies Omit<ProbeResult, "stage">;
      try {
        const hud = await until("the visible native HUD window", async () => {
          const candidate = await WebviewWindow.getByLabel("belo-hud");
          return candidate && (await candidate.isVisible()) ? candidate : null;
        });
        const scaleFactor = await hud.scaleFactor();
        const size = await hud.innerSize();
        const position = await hud.outerPosition();
        const windowState = {
          exists: true,
          label: hud.label,
          visible: await hud.isVisible(),
          focused: await hud.isFocused(),
          alwaysOnTop: await hud.isAlwaysOnTop(),
          decorated: await hud.isDecorated(),
          resizable: await hud.isResizable(),
          logicalWidth: size.width / scaleFactor,
          logicalHeight: size.height / scaleFactor,
          positionX: position.x / scaleFactor,
          positionY: position.y / scaleFactor,
          scaleFactor,
        };
        if (!cancelled) {
          await report({ ...base, stage: "created", window: windowState });
          setEligible(false);
          const closed = await (async () => {
            const deadline = Date.now() + 8_000;
            while (Date.now() < deadline) {
              if (!(await WebviewWindow.getByLabel("belo-hud"))) return true;
              await wait(50);
            }
            return false;
          })();
          await report({ ...base, stage: "closed", window: windowState, closed });
        }
      } catch (error) {
        if (!cancelled) {
          await report({
            ...base,
            stage: "error",
            error: error instanceof Error ? error.message : String(error),
          });
        }
      }
    };
    void run();
    return () => {
      cancelled = true;
    };
  }, []);

  return null;
}

// Keep the intentionally visible native HUD off the user's monitors while the
// assertions read its real visibility and geometry.
localStorage.setItem("belo.screen.hud", "on");
localStorage.setItem("belo.screen.hud.position", JSON.stringify({ x: -10_000, y: -10_000 }));
setScreenSource(fakeScreen);
createRoot(document.getElementById("mount")!).render(<Probe />);
