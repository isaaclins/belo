// Native capture HUD trigger coverage.
//
// This page mounts the real Use view and starts its real macOS screen source.
// The HUD must therefore appear because the view becomes eligible and the main
// native window loses focus. The test never constructs belo-hud itself.

import { useEffect } from "react";
import { createRoot } from "react-dom/client";
import { invoke } from "@tauri-apps/api/core";
import { WebviewWindow } from "@tauri-apps/api/webviewWindow";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { UseView } from "../../src/views/UseView";
import { useStore } from "../../src/lib/store";
import { newManifest } from "../../src/lib/types";

const wait = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

async function until<T>(what: string, read: () => Promise<T | null>, timeoutMs = 15_000): Promise<T> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const value = await read();
    if (value !== null && value !== undefined) return value;
    await wait(50);
  }
  throw new Error(`Timed out waiting for ${what}`);
}

function button(label: string): HTMLButtonElement | null {
  return Array.from(document.querySelectorAll("button")).find(
    (node) => node.textContent?.trim() === label,
  ) as HTMLButtonElement | null;
}

async function hudWindow(): Promise<WebviewWindow | null> {
  return WebviewWindow.getByLabel("belo-hud");
}

async function visibleHud(): Promise<WebviewWindow | null> {
  const hud = await hudWindow();
  return hud && (await hud.isVisible()) ? hud : null;
}

async function hudState(hud: WebviewWindow): Promise<Record<string, unknown>> {
  const scaleFactor = await hud.scaleFactor();
  const size = await hud.innerSize();
  return {
    exists: true,
    label: hud.label,
    visible: await hud.isVisible(),
    focused: await hud.isFocused(),
    alwaysOnTop: await hud.isAlwaysOnTop(),
    decorated: await hud.isDecorated(),
    resizable: await hud.isResizable(),
    logicalWidth: size.width / scaleFactor,
    logicalHeight: size.height / scaleFactor,
    scaleFactor,
  };
}

async function blurMain(main: ReturnType<typeof getCurrentWindow>): Promise<boolean> {
  await main.setFocusable(false);
  if (await main.isFocused()) await main.hide();
  return main.isFocused();
}

/**
 * macOS only grants focus to a window of the active application. This check
 * runs its window offscreen and never activates the app, so a refused refocus
 * is reported as skipped instead of failing the run.
 */
async function refocusMain(main: ReturnType<typeof getCurrentWindow>): Promise<boolean> {
  await main.setFocusable(true);
  await main.show();
  try {
    await main.setFocus();
  } catch {
    return false;
  }
  const deadline = Date.now() + 4_000;
  while (Date.now() < deadline) {
    if (await main.isFocused()) return true;
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  return false;
}

async function run(): Promise<void> {
  const main = getCurrentWindow();
  const errors: string[] = [];
  const startedAt = Date.now();
  try {
    await until("the Screen source button", async () => button("Screen"));
    button("Screen")!.click();
    await until("the Screen source to be selected", async () => {
      const node = button("Screen");
      return node?.getAttribute("aria-pressed") === "true" ? node : null;
    });

    await until("the Start button", async () => button("Start"));
    button("Start")!.click();
    await until("the real screen feed", async () => {
      const video = document.querySelector("video");
      return video instanceof HTMLVideoElement && video.videoWidth > 0 && video.videoHeight > 0 ? video : null;
    }, 20_000);

    const stillFocused = await blurMain(main);
    const hud = await until("the HUD after the main window blur", visibleHud, 20_000);
    const created = await hudState(hud);
    if (stillFocused) throw new Error("the Tauri blur did not remove focus from the main window");

    // The window is kept alive but hidden between appearances: rebuilding it on
    // every focus loss would activate the app and hide the HUD again.
    const refocused = await refocusMain(main);
    let refocusClosed = false;
    if (refocused) {
      await until("the HUD to hide on refocus", async () => ((await visibleHud()) ? null : true));
      refocusClosed = true;
      await blurMain(main);
      await until("the HUD to reappear after a second blur", visibleHud, 20_000);
    }

    // Turning the HUD off is the user-driven hide path and needs no OS focus.
    const toggle = await until("the HUD toggle", async () => button("HUD on"));
    toggle.click();
    await until("the HUD to close when switched off", async () => ((await hudWindow()) ? null : true));
    toggle.click();
    await until("the HUD to return when switched back on", visibleHud, 20_000);

    const stop = await until("the Stop button", async () => button("Stop"));
    stop.click();
    await until("the HUD to disappear on stop", async () => ((await hudWindow()) ? null : true));

    await report({
      check: "check:hud-real",
      stage: "closed",
      elapsedMs: Date.now() - startedAt,
      created,
      refocusClosed,
      refocusSkipped: !refocused,
      toggleClosed: true,
      stopClosed: true,
      errors,
    });
  } catch (error) {
    errors.push(error instanceof Error ? error.message : String(error));
    let hudExists = false;
    let hudVisible = false;
    try {
      const hud = await hudWindow();
      hudExists = Boolean(hud);
      hudVisible = hud ? await hud.isVisible() : false;
    } catch (probeError) {
      errors.push(`hud probe failed: ${String(probeError)}`);
    }
    await report({
      check: "check:hud-real",
      stage: "error",
      hudExists,
      hudVisible,
      mainFocused: await main.isFocused().catch(() => null),
      buttons: Array.from(document.querySelectorAll("button")).map((node) => node.textContent?.trim()),
      video: (() => {
        const video = document.querySelector("video");
        return video instanceof HTMLVideoElement ? { w: video.videoWidth, h: video.videoHeight } : null;
      })(),
      errors,
    });
  }
}

async function report(value: unknown): Promise<void> {
  await invoke("hud_real_check_report", { report: JSON.stringify(value) });
}

const manifest = newManifest("Native HUD trigger check", "0.11.2", "single");
manifest.labels = [{ id: "thing", name: "Thing", color: 0 }];
localStorage.setItem("belo.screen.hud", "on");
localStorage.setItem("belo.screen.quality", "smooth");
localStorage.setItem("belo.screen.hud.position", JSON.stringify({ x: -10_000, y: -10_000 }));
useStore.setState({
  booted: true,
  path: "/tmp/belo-hud-real-trigger.belo",
  manifest,
  view: "use",
  hasModel: true,
  modelLabelIds: ["thing"],
});

function Harness() {
  useEffect(() => {
    void run();
  }, []);
  return <UseView />;
}

createRoot(document.getElementById("mount")!).render(<Harness />);
