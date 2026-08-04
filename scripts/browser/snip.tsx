// Headless shortcut-event coverage for the macOS screen-region path.
// The runtime seam supplies fixture PNG bytes, so this never launches
// screencapture or takes over a real desktop.

import { useStore } from "../../src/lib/store";
import {
  SCREEN_SNIP_SHORTCUT,
  ScreenSnipController,
  type ScreenShortcutEvent,
  type ScreenSnipPayload,
  type ScreenSnipRuntime,
  type ScreenSnipTarget,
} from "../../src/lib/screenSnip";
import { installTauriShim, reporter } from "./tauri-shim";

installTauriShim();
const { check, fail, finish } = reporter();

const FIXTURE_PNG: ScreenSnipPayload = {
  mime: "image/png",
  base64:
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=",
};

const wait = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

async function until(what: string, probe: () => boolean, timeoutMs = 10_000): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (!probe()) {
    if (Date.now() > deadline) throw new Error(`Timed out waiting for ${what}`);
    await wait(25);
  }
}

async function main(): Promise<void> {
  let shortcutHandler: ((event: ScreenShortcutEvent) => void) | null = null;
  let lastHandler: ((event: ScreenShortcutEvent) => void) | null = null;
  let registrationCount = 0;
  let unregisterCount = 0;
  let captureCount = 0;
  let backgrounded = false;
  let holdCapture = false;
  let releaseCapture: (() => void) | null = null;
  let nextCapture: ScreenSnipPayload | null = FIXTURE_PNG;
  const notifications: { title: string; body: string }[] = [];
  const foregroundImports: { projectName: string; count: number }[] = [];

  const runtime: ScreenSnipRuntime = {
    supported: async () => true,
    register: async (handler) => {
      shortcutHandler = handler;
      lastHandler = handler;
      registrationCount += 1;
    },
    unregister: async () => {
      shortcutHandler = null;
      unregisterCount += 1;
    },
    capture: async () => {
      captureCount += 1;
      if (holdCapture) {
        await new Promise<void>((resolve) => {
          releaseCapture = resolve;
        });
      }
      const capture = nextCapture;
      nextCapture = null;
      return capture;
    },
    backgrounded: async () => backgrounded,
    notify: async (title, body) => {
      notifications.push({ title, body });
    },
  };

  const controller = new ScreenSnipController(runtime);
  await useStore.getState().createProject("Snip fixture", "single");
  const label = useStore.getState().addLabel("Screen thing");
  if (!label) throw new Error("could not create fixture label");
  useStore.setState({ hasModel: true });
  const projectPath = useStore.getState().path;
  if (!projectPath) throw new Error("fixture project did not open");

  const target = (expectedPath: string): ScreenSnipTarget => ({
    isOpen: () => {
      const current = useStore.getState();
      return current.path === expectedPath && !current.closingProject && current.manifest !== null;
    },
    add: async (payload) => {
      const current = useStore.getState();
      if (!current.path || current.path !== expectedPath || current.closingProject || !current.manifest) {
        return null;
      }
      const added = await current.addFromBase64(
        payload.base64,
        payload.mime,
        current.activeLabelId,
        "capture",
        "screen",
        true,
      );
      const after = useStore.getState();
      if (!added || after.path !== expectedPath || !after.manifest) return null;
      return { projectName: after.manifest.name, count: after.manifest.images.length };
    },
  });

  const opened = await controller.open(target(projectPath), {
    onForegroundImport: (result) => foregroundImports.push(result),
  });
  check("the macOS-only shortcut registers for an open project", opened && registrationCount === 1);
  check("the registered key is Cmd+Shift+B", SCREEN_SNIP_SHORTCUT === "CommandOrControl+Shift+B");

  shortcutHandler?.({ state: "Released" });
  await wait(60);
  check("a released shortcut event does not launch the picker", captureCount === 0);
  shortcutHandler?.({ state: "Pressed" });
  await until("the foreground fixture import", () => useStore.getState().manifest?.images.length === 1);
  const first = useStore.getState().manifest?.images[0];
  check(
    "a foreground shortcut imports one fixture through the store",
    first?.source === "capture" && first.feed === "screen",
    first,
  );
  check("the screen capture requests the normal retrain schedule", useStore.getState().retrainPending);
  check("foreground capture uses the existing toast handoff", foregroundImports.length === 1);

  nextCapture = null;
  shortcutHandler?.({ state: "Pressed" });
  await wait(60);
  check("Escape-style cancellation is a clean no-op", useStore.getState().manifest?.images.length === 1);

  backgrounded = true;
  nextCapture = FIXTURE_PNG;
  shortcutHandler?.({ state: "Pressed" });
  await until("the background fixture import", () => useStore.getState().manifest?.images.length === 2);
  check("background capture uses the native notification handoff", notifications.length === 1);
  check(
    "the native notification names the project and count",
    notifications[0]?.title === "Screen capture added" &&
      notifications[0]?.body.includes("Snip fixture") &&
      notifications[0]?.body.includes("2 images"),
    notifications[0],
  );

  await controller.close();
  await useStore.getState().createProject("Detect snips", "detect");
  const detectLabel = useStore.getState().addLabel("Object");
  if (!detectLabel) throw new Error("could not create detect fixture label");
  const detectPath = useStore.getState().path;
  if (!detectPath) throw new Error("detect fixture did not open");
  nextCapture = FIXTURE_PNG;
  backgrounded = false;
  const detectOpened = await controller.open(target(detectPath));
  shortcutHandler?.({ state: "Pressed" });
  await until("the detect fixture import", () => useStore.getState().manifest?.images.length === 1);
  const detectImage = useStore.getState().manifest?.images[0];
  check("detect shortcut captures stay unboxed", detectOpened && detectImage?.boxes?.length === 0, detectImage);

  holdCapture = true;
  nextCapture = FIXTURE_PNG;
  shortcutHandler?.({ state: "Pressed" });
  await until("the picker to be in flight", () => releaseCapture !== null);
  const capturesBeforeClose = captureCount;
  const closeInFlight = controller.close();
  releaseCapture?.();
  holdCapture = false;
  await closeInFlight;
  await wait(60);
  check("closing an in-flight picker discards its returned image", useStore.getState().manifest?.images.length === 1);

  const lateHandler = lastHandler;
  check("closing unregisters the active shortcut", unregisterCount >= 2);
  await useStore.getState().closeProject();
  lateHandler?.({ state: "Pressed" });
  await wait(60);
  check(
    "a shortcut racing with close is a no-op",
    useStore.getState().manifest === null && useStore.getState().path === null,
    { capturesBeforeClose, captureCount },
  );
  check("the late shortcut did not invoke the capture runner", captureCount === capturesBeforeClose);
  finish();
}

void main().catch(fail);
