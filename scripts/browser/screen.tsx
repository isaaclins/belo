// End to end check of the screen source, in a real browser, without recording
// anything.
//
// The point of the seam in src/lib/screen.ts is that the platform only has to
// push encoded frames; everything above it, the canvas, the MediaStream, the
// hook that owns the session and the two Use views, is shared. So this harness
// implements ScreenPipe over a synthetic desktop instead of ScreenCaptureKit
// and drives the app's own code the way a person drives the toggle: start the
// camera, switch to the screen, switch displays, teach a frame, switch back,
// stop. The native pipe underneath it is the one thing that cannot be checked
// here, and the README says so.
//
// The fake desktop is 1443 by 902: sixteen to ten, odd on both axes and nothing
// like a webcam, which is exactly the aspect ratio the letterboxing math has to
// survive.
//
//   npm run check:screen

import { useState } from "react";
import { createRoot } from "react-dom/client";
import { cropSquare } from "../../src/lib/attention";
import { toSquareCanvas } from "../../src/lib/images";
import {
  createPipedScreenSource,
  screen,
  setScreenSource,
  type OpenFeed,
  type ScreenInfo,
  type ScreenPermission,
  type ScreenPipe,
} from "../../src/lib/screen";
import { useStore } from "../../src/lib/store";
import { useLiveSource, type FeedKind } from "../../src/lib/useWebcam";
import { installTauriShim, reporter } from "./tauri-shim";

installTauriShim();
const { check, fail, finish } = reporter();

const WIDE = 1443;
const TALL = 902;
const SECOND = { width: 1024, height: 768 };
const STAGE = 480;
const CAMERA_PREF = "belo.camera.deviceId";
const SCREEN_PREF = "belo.screen.displayId";

type Scene = "red" | "green";

/**
 * A desktop-ish picture: a menu bar, a dock and one big window whose content is
 * the thing the model is asked to learn. A moving clock keeps consecutive
 * frames from being identical, so a stuck feed cannot pass for a live one.
 */
function desktop(scene: Scene, width: number, height: number, tick: number): HTMLCanvasElement {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d")!;
  ctx.fillStyle = "#2b3140";
  ctx.fillRect(0, 0, width, height);
  ctx.fillStyle = "#11141c";
  ctx.fillRect(0, 0, width, Math.round(height * 0.035));
  ctx.fillStyle = "#e8e8ea";
  ctx.font = `${Math.round(height * 0.024)}px monospace`;
  ctx.fillText(`belo  ${String(tick % 60).padStart(2, "0")}`, 12, Math.round(height * 0.028));

  // The window, centered, and large enough to fill the center square that the
  // classifier actually sees.
  const w = Math.round(width * 0.52);
  const h = Math.round(height * 0.72);
  const x = Math.round((width - w) / 2);
  const y = Math.round((height - h) / 2);
  ctx.fillStyle = "#f3f1ea";
  ctx.fillRect(x, y, w, h);
  ctx.fillStyle = "#1b1b1f";
  ctx.fillRect(x, y, w, Math.round(h * 0.08));

  if (scene === "red") {
    ctx.fillStyle = "rgb(226,34,34)";
    ctx.beginPath();
    ctx.arc(x + w / 2, y + h / 2, Math.min(w, h) * 0.3, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "rgb(120,0,0)";
    ctx.beginPath();
    ctx.arc(x + w / 2, y + h / 2, Math.min(w, h) * 0.12, 0, Math.PI * 2);
    ctx.fill();
  } else {
    const cells = 6;
    const size = Math.min(w, h) * 0.6;
    const step = size / cells;
    for (let row = 0; row < cells; row += 1) {
      for (let col = 0; col < cells; col += 1) {
        ctx.fillStyle = (row + col) % 2 === 0 ? "rgb(18,140,60)" : "rgb(6,60,26)";
        ctx.fillRect(
          x + w / 2 - size / 2 + col * step,
          y + h / 2 - size / 2 + row * step,
          step,
          step,
        );
      }
    }
  }

  ctx.fillStyle = "#11141c";
  ctx.fillRect(0, height - Math.round(height * 0.07), width, Math.round(height * 0.07));
  return canvas;
}

function toJpeg(canvas: HTMLCanvasElement): Promise<ArrayBuffer> {
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => (blob ? resolve(blob.arrayBuffer()) : reject(new Error("no blob"))),
      "image/jpeg",
      0.72,
    );
  });
}

const displays: ScreenInfo[] = [
  { id: "display-1", label: "Display 1", width: WIDE, height: TALL },
  { id: "display-2", label: "Display 2", width: SECOND.width, height: SECOND.height },
];

/** Everything the fake platform did, so the checks can assert on it. */
const platform = {
  permission: "granted" as ScreenPermission,
  scene: "red" as Scene,
  opened: 0,
  closed: 0,
  settingsOpened: 0,
  lastRequest: null as string | null,
};

const fakePipe: ScreenPipe = {
  isSupported: () => true,
  list: async () => displays,
  permission: async () => platform.permission,
  openSettings: async () => {
    platform.settingsOpened += 1;
  },
  async open(displayId, onFrame): Promise<OpenFeed> {
    platform.lastRequest = displayId;
    const display = displays.find((entry) => entry.id === displayId) ?? displays[0];
    platform.opened += 1;
    let tick = 0;
    let sending = false;
    const timer = window.setInterval(() => {
      if (sending) return;
      sending = true;
      tick += 1;
      void toJpeg(desktop(platform.scene, display.width, display.height, tick))
        .then(onFrame)
        .finally(() => {
          sending = false;
        });
    }, 60);
    return {
      id: display.id,
      width: display.width,
      height: display.height,
      close() {
        window.clearInterval(timer);
        platform.closed += 1;
      },
    };
  },
};

setScreenSource(createPipedScreenSource(fakePipe));

type Feed = ReturnType<typeof useLiveSource>;

const live: {
  feed: Feed | null;
  setActive: ((on: boolean) => void) | null;
  setKind: ((kind: FeedKind) => void) | null;
} = { feed: null, setActive: null, setKind: null };

function Harness() {
  const [active, setActive] = useState(false);
  const [kind, setKind] = useState<FeedKind>("camera");
  const feed = useLiveSource(active, kind);
  live.feed = feed;
  live.setActive = setActive;
  live.setKind = setKind;
  return <video id="feed" ref={feed.videoRef} playsInline muted autoPlay />;
}

const wait = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

async function until(what: string, probe: () => boolean, timeoutMs = 15_000) {
  const deadline = Date.now() + timeoutMs;
  while (!probe()) {
    if (Date.now() > deadline) throw new Error(`Timed out waiting for ${what}`);
    await wait(50);
  }
}

const videoEl = () => document.getElementById("feed") as HTMLVideoElement;
const currentStream = () => videoEl().srcObject as MediaStream | null;

/** Mean color of the middle of the stage, read back through the video element. */
function centerColor(): { r: number; g: number; b: number } {
  const video = videoEl();
  const canvas = document.createElement("canvas");
  canvas.width = 32;
  canvas.height = 32;
  const ctx = canvas.getContext("2d")!;
  const side = Math.min(video.videoWidth, video.videoHeight) * 0.25;
  ctx.drawImage(
    video,
    (video.videoWidth - side) / 2,
    (video.videoHeight - side) / 2,
    side,
    side,
    0,
    0,
    32,
    32,
  );
  const { data } = ctx.getImageData(0, 0, 32, 32);
  let r = 0;
  let g = 0;
  let b = 0;
  for (let i = 0; i < data.length; i += 4) {
    r += data[i];
    g += data[i + 1];
    b += data[i + 2];
  }
  const pixels = data.length / 4;
  return { r: r / pixels, g: g / pixels, b: b / pixels };
}

async function main() {
  createRoot(document.getElementById("mount")!).render(<Harness />);
  await until("the harness to mount", () => Boolean(live.setActive));

  check("the screen seam reports itself supported through the swapped source", screen.isSupported(), {
    canList: screen.canList(),
  });

  // 1. The camera first, because that is what a Use view opens with.
  live.setActive!(true);
  await until("the camera to go live", () => live.feed!.status === "live");
  const cameras = await until("the camera list", () => live.feed!.options.length > 0)
    .then(() => live.feed!.options)
    .catch(() => []);
  const cameraStream = currentStream();
  check("the camera reaches live before anything else happens", live.feed!.status === "live", {
    tracks: cameraStream?.getVideoTracks().length,
    cameras: cameras.length,
  });

  // A second camera is picked on purpose, so the return trip has something to
  // restore that is not just the default.
  const second = cameras[1] ?? cameras[0];
  live.feed!.select(second.id);
  await until("the second camera", () => live.feed!.status === "live" && currentStream() !== cameraStream);
  const chosenCamera = currentStream();

  // 2. Switch to the screen. The camera has to let go, the screen has to arrive.
  const cameraTracks = chosenCamera!.getVideoTracks();
  live.setKind!("screen");
  await until(
    "the screen feed to go live",
    () => live.feed!.status === "live" && currentStream() !== chosenCamera,
  );
  const screenStream = currentStream();
  check("switching source stops every camera track", cameraTracks.every((t) => t.readyState === "ended"), {
    states: cameraTracks.map((t) => t.readyState),
  });
  check("the screen stream is live", screenStream?.getVideoTracks()[0]?.readyState === "live", {
    tracks: screenStream?.getVideoTracks().map((t) => t.readyState),
  });
  check("the platform was asked to open exactly one screen feed", platform.opened === 1, {
    opened: platform.opened,
    closed: platform.closed,
  });

  await until("the first painted frame", () => videoEl().videoWidth > 0);
  check(
    "the odd display resolution survives the whole frame path",
    videoEl().videoWidth === WIDE && videoEl().videoHeight === TALL,
    { size: `${videoEl().videoWidth}x${videoEl().videoHeight}`, wanted: `${WIDE}x${TALL}` },
  );

  // 3. The frames carry the picture, not just a size.
  await wait(400);
  const red = centerColor();
  check("the stage shows the screen's own content", red.r > 120 && red.r > red.g * 1.6, {
    color: `${Math.round(red.r)},${Math.round(red.g)},${Math.round(red.b)}`,
  });

  // 4. Letterboxing at an aspect ratio no camera produces. The overlay maps the
  //    square the model saw onto a stage that crops with object-fit: cover.
  const rect = cropSquare({ width: STAGE, height: STAGE }, { width: WIDE, height: TALL })!;
  const scale = Math.max(STAGE / WIDE, STAGE / TALL);
  const expected = {
    size: TALL * scale,
    x: (STAGE - WIDE * scale) / 2 + ((WIDE - TALL) / 2) * scale,
    y: 0,
  };
  check(
    "the model's square is placed correctly on a 16:10 display",
    Math.abs(rect.size - expected.size) < 0.01 &&
      Math.abs(rect.x - expected.x) < 0.01 &&
      Math.abs(rect.y - expected.y) < 0.01,
    { rect, expected },
  );
  check(
    "the square is as tall as the stage and centered horizontally",
    Math.abs(rect.size - STAGE) < 0.01 && Math.abs(rect.x + rect.size / 2 - STAGE / 2) < 0.01,
    { rect },
  );

  // 5. Display selection, the screen's answer to the camera picker.
  check("both displays are offered with their sizes", live.feed!.options.length === 2, {
    options: live.feed!.options.map((entry) => entry.label),
  });
  check(
    "the display list says how big each one is",
    live.feed!.options[0].label.includes(String(WIDE)),
    { label: live.feed!.options[0].label },
  );
  live.feed!.select("display-2");
  await until("the second display", () => currentStream() !== screenStream && live.feed!.status === "live");
  await until("the second display's frames", () => videoEl().videoWidth === SECOND.width);
  check("switching display releases the first capture", platform.closed === 1, {
    opened: platform.opened,
    closed: platform.closed,
  });
  check(
    "the second display's own resolution reaches the stage",
    videoEl().videoWidth === SECOND.width && videoEl().videoHeight === SECOND.height,
    { size: `${videoEl().videoWidth}x${videoEl().videoHeight}` },
  );
  check("the display pick is remembered for the next launch", localStorage.getItem(SCREEN_PREF) === "display-2", {
    stored: localStorage.getItem(SCREEN_PREF),
  });
  live.feed!.select("display-1");
  await until("the wide display again", () => videoEl().videoWidth === WIDE);

  // 6. A real project, a real model, and predictions against the fake desktop.
  const store = useStore.getState();
  await store.createProject("Screen check", "single");
  const redLabel = useStore.getState().addLabel("Redcircle")!;
  const greenLabel = useStore.getState().addLabel("Greenblock")!;
  for (let i = 0; i < 6; i += 1) {
    await store.addFromDrawable(desktop("red", WIDE, TALL, i), redLabel, "import");
    await store.addFromDrawable(desktop("green", WIDE, TALL, i), greenLabel, "import");
  }
  await store.startTraining();
  const train = useStore.getState().train;
  check("a model trained on what the screen shows", useStore.getState().hasModel, {
    phase: train.phase,
    accuracy: train.accuracy,
  });

  // The exact inference path of the Use view: the live element, squared.
  const readScreen = async () => {
    const prediction = await useStore.getState().predict(toSquareCanvas(videoEl()));
    return prediction;
  };
  const onRed = await readScreen();
  check("the live screen is classified as what it shows", onRed?.labelId === redLabel, {
    label: onRed?.labelId === redLabel ? "red" : "green",
    confidence: onRed?.confidence,
  });

  platform.scene = "green";
  await wait(600);
  const onGreen = await readScreen();
  check("changing what is on the screen changes the answer", onGreen?.labelId === greenLabel, {
    label: onGreen?.labelId === greenLabel ? "green" : "red",
    confidence: onGreen?.confidence,
  });

  // 7. Teaching from the screen, the way the Use view's teach button does.
  const before = useStore.getState().manifest!.images.length;
  const added = await useStore
    .getState()
    .addFromDrawable(toSquareCanvas(videoEl()), [greenLabel], "feedback", "screen");
  const images = useStore.getState().manifest!.images;
  const taught = images[images.length - 1];
  check("a taught frame is written", added && images.length === before + 1, {
    before,
    after: images.length,
  });
  check("the frame is flagged as a correction taken from the screen", taught.source === "feedback" && taught.feed === "screen", {
    source: taught.source,
    feed: taught.feed,
  });
  const fromCamera = images.find((image) => image.source === "import");
  check("imports carry no feed at all, so old projects stay unchanged", fromCamera?.feed === undefined, {
    feed: fromCamera?.feed,
  });

  // 8. Back to the camera. The screen capture has to be released at once, and
  //    the camera that was running has to come back.
  const runningScreen = currentStream();
  const closedBefore = platform.closed;
  live.setKind!("camera");
  await until("the camera again", () => live.feed!.status === "live" && currentStream() !== runningScreen);
  check("leaving the screen releases the capture immediately", platform.closed === closedBefore + 1, {
    closed: platform.closed,
    opened: platform.opened,
  });
  check("the screen tracks are stopped", runningScreen!.getVideoTracks().every((t) => t.readyState === "ended"), {
    states: runningScreen!.getVideoTracks().map((t) => t.readyState),
  });
  check("the camera that was running before comes back", localStorage.getItem(CAMERA_PREF) === second.id, {
    stored: localStorage.getItem(CAMERA_PREF)?.slice(0, 12),
    shown: live.feed!.optionId?.slice(0, 12),
  });
  check("the camera pick, not the display pick, is what the picker shows", live.feed!.optionId === second.id, {
    optionId: live.feed!.optionId?.slice(0, 12),
  });

  // 9. Stopping altogether leaves nothing running.
  const openedBeforeLast = platform.opened;
  live.setKind!("screen");
  await until(
    "the screen once more",
    () => platform.opened === openedBeforeLast + 1 && live.feed!.status === "live",
  );
  await until("its frames", () => videoEl().videoWidth === WIDE);
  const lastStream = currentStream();
  const closedNow = platform.closed;
  live.setActive!(false);
  await until("the feed to stop", () => live.feed!.status === "off");
  check("stopping the view releases the screen capture", platform.closed === closedNow + 1, {
    closed: platform.closed,
  });
  check("stopping the view ends every track", lastStream!.getVideoTracks().every((t) => t.readyState === "ended"), {
    states: lastStream!.getVideoTracks().map((t) => t.readyState),
  });

  // 10. The denial path: a refusal has to be a refusal, not a crash.
  platform.permission = "denied";
  const openedBefore = platform.opened;
  live.setActive!(true);
  await until("the refusal", () => live.feed!.status === "error");
  check("a refused screen never opens a capture", platform.opened === openedBefore, {
    opened: platform.opened,
  });
  check("the refusal is reported as a refusal, in plain words", live.feed!.refused, {
    error: live.feed!.error,
  });
  check(
    "the message says what to do rather than what failed",
    (live.feed!.error ?? "").toLowerCase().includes("not allowed"),
    { error: live.feed!.error },
  );
  await screen.openSettings();
  check("the guidance panel can open the platform's settings", platform.settingsOpened === 1, {
    settingsOpened: platform.settingsOpened,
  });

  // And the retry after the switch is turned on.
  platform.permission = "granted";
  live.feed!.retry();
  await until("the retry to succeed", () => live.feed!.status === "live");
  check("turning the permission on and retrying starts the feed", live.feed!.status === "live", {
    status: live.feed!.status,
  });
  live.setActive!(false);
  await until("the last teardown", () => live.feed!.status === "off");
}

main()
  .then(finish)
  .catch((err) => {
    console.error(err);
    fail(err);
  });
