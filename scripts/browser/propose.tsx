// The closed loop on assisted box labeling, in a real browser.
//
//   npm run check:propose
//
// Nothing here is mocked except the Tauri shim: a real detect project is built
// in the in-memory backend, a real detector is trained on it with the app's own
// training path, and the real BoxAnnotator is mounted on the result. Every
// assertion below is then made against what that component actually put in the
// DOM and against what the manifest actually holds afterwards.
//
// The point of the check is the safety story. A guess is only ever a drawing
// until somebody says yes to it, and the way to prove that is to look at the
// manifest after each gesture.

import { useState } from "react";
import { createRoot } from "react-dom/client";
import { BoxAnnotator } from "../../src/views/BoxAnnotator";
import { useStore } from "../../src/lib/store";
import { setScoreThreshold } from "../../src/lib/scoreThreshold";
import { boxesOf, type BoxRect, type ImageRecord } from "../../src/lib/types";
import { installTauriShim, reporter } from "./tauri-shim";

import "../../src/styles/tokens.css";
import "../../src/styles/base.css";
import "../../src/styles/ui.css";
import "../../src/styles/app.css";

installTauriShim();
const { check, fail, finish } = reporter();

const SIDE = 224;
/** Boxed pictures the detector learns from. Ten is the app's own minimum. */
const TRAIN_IMAGES = 12;
/** Unboxed pictures the annotator is asked to guess at, one per scenario. */
const TARGETS = 5;

type Kind = "red" | "green";

let seed = 90210;
function random(): number {
  seed = (seed * 1103515245 + 12345) % 2147483648;
  return seed / 2147483648;
}

/** A gray speckled backdrop, so the frozen extractor sees texture and not flat paint. */
function background(ctx: CanvasRenderingContext2D): void {
  ctx.fillStyle = "#8a8a8a";
  ctx.fillRect(0, 0, SIDE, SIDE);
  for (let i = 0; i < 700; i += 1) {
    const value = Math.round(110 + random() * 60);
    ctx.fillStyle = `rgb(${value},${value},${value})`;
    ctx.fillRect(random() * SIDE, random() * SIDE, 6, 6);
  }
}

function paint(ctx: CanvasRenderingContext2D, kind: Kind, cx: number, cy: number, half: number) {
  if (kind === "red") {
    ctx.fillStyle = "rgb(228,32,32)";
    ctx.beginPath();
    ctx.arc(cx, cy, half, 0, Math.PI * 2);
    ctx.fill();
    return;
  }
  const cells = 6;
  const step = (half * 2) / cells;
  for (let row = 0; row < cells; row += 1) {
    for (let col = 0; col < cells; col += 1) {
      ctx.fillStyle = (row + col) % 2 === 0 ? "rgb(18,140,60)" : "rgb(6,60,26)";
      ctx.fillRect(cx - half + col * step, cy - half + row * step, step, step);
    }
  }
}

interface Scene {
  canvas: HTMLCanvasElement;
  truth: { kind: Kind; rect: BoxRect }[];
}

/**
 * One picture with a red circle on the left and a green block on the right,
 * plus the rectangles a person would have drawn around them. jitter 0 gives the
 * exact same scene every time, which is what the position assertions compare
 * the detector's guesses against.
 */
function scene(jitter: number): Scene {
  const canvas = document.createElement("canvas");
  canvas.width = SIDE;
  canvas.height = SIDE;
  const ctx = canvas.getContext("2d")!;
  background(ctx);
  const truth: Scene["truth"] = [];
  const places: { kind: Kind; cx: number }[] = [
    { kind: "red", cx: 0.27 },
    { kind: "green", cx: 0.73 },
  ];
  for (const place of places) {
    const cx = (place.cx + (random() - 0.5) * jitter) * SIDE;
    const cy = (0.5 + (random() - 0.5) * jitter) * SIDE;
    const half = (0.14 + (random() - 0.5) * jitter * 0.4) * SIDE;
    paint(ctx, place.kind, cx, cy, half);
    truth.push({
      kind: place.kind,
      rect: {
        x: (cx - half) / SIDE,
        y: (cy - half) / SIDE,
        w: (half * 2) / SIDE,
        h: (half * 2) / SIDE,
      },
    });
  }
  return { canvas, truth };
}

// --- waiting -----------------------------------------------------------------

const tick = () => new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));

async function settle(frames = 4): Promise<void> {
  for (let i = 0; i < frames; i += 1) await tick();
}

async function until<T>(what: string, probe: () => T | null, timeoutMs = 20_000): Promise<T> {
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    const value = probe();
    if (value) return value;
    if (Date.now() > deadline) throw new Error(`Timed out waiting for ${what}`);
    await new Promise((resolve) => setTimeout(resolve, 50));
  }
}

// --- reading the annotator ----------------------------------------------------

const guessNodes = () => [...document.querySelectorAll<HTMLElement>(".aprop")];

function guessRect(node: HTMLElement): BoxRect {
  const [x, y, w, h] = (node.dataset.guessRect ?? "").split(",").map(Number);
  return { x, y, w, h };
}

function overlap(a: BoxRect, b: BoxRect): number {
  const w = Math.max(0, Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x));
  const h = Math.max(0, Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y));
  const inter = w * h;
  const union = a.w * a.h + b.w * b.h - inter;
  return union > 0 ? inter / union : 0;
}

function pointer(node: EventTarget, type: string, clientX: number, clientY: number): void {
  node.dispatchEvent(
    new PointerEvent(type, {
      bubbles: true,
      cancelable: true,
      composed: true,
      button: 0,
      buttons: type === "pointerup" ? 0 : 1,
      pointerId: 1,
      isPrimary: true,
      pointerType: "mouse",
      clientX,
      clientY,
    }),
  );
}

/** A press that does not travel: the gesture that keeps a guess. */
function tap(node: Element): void {
  const box = node.getBoundingClientRect();
  const x = box.left + box.width / 2;
  const y = box.top + box.height / 2;
  pointer(node, "pointerdown", x, y);
  pointer(node, "pointerup", x, y);
}

function press(key: string): void {
  window.dispatchEvent(new KeyboardEvent("keydown", { key, bubbles: true, cancelable: true }));
}

const recordOf = (id: string): ImageRecord =>
  useStore.getState().manifest!.images.find((image) => image.id === id)!;

const boxCount = (id: string) => boxesOf(recordOf(id)).length;

// --- the harness component ----------------------------------------------------

let goTo: (index: number) => void = () => {};
let shownIndex = 0;
let closes = 0;
/** One array for "no project", so the selector never hands React a new one. */
const NONE: ImageRecord[] = [];

function Harness() {
  const images = useStore((state) => state.manifest?.images ?? NONE);
  const [index, setIndex] = useState(0);
  goTo = setIndex;
  shownIndex = index;
  if (images.length === 0) return null;
  return (
    <BoxAnnotator
      images={images}
      index={index}
      onIndex={setIndex}
      onClose={() => {
        closes += 1;
      }}
    />
  );
}

// --- the run ------------------------------------------------------------------

async function main() {
  const store = useStore.getState();
  await store.createProject("Guess check", "detect");
  const red = useStore.getState().addLabel("Redcircle");
  const green = useStore.getState().addLabel("Greenblock");
  if (!red || !green) throw new Error("labels could not be created");
  const labelOf = (kind: Kind) => (kind === "red" ? red : green);

  // Fourteen hand-boxed pictures, the way a person would arrive at a first
  // model, then the training run the app itself would have scheduled.
  for (let i = 0; i < TRAIN_IMAGES; i += 1) {
    const shot = scene(0.18);
    await store.addFromDrawable(shot.canvas, null, "import");
    const record = useStore.getState().manifest!.images[i];
    for (const item of shot.truth) {
      useStore.getState().addBox(record.id, labelOf(item.kind), item.rect);
    }
  }
  // No mirrored copies and a short run: this check is about the annotator, and
  // two shapes on a flat backdrop do not need a hundred and fifty epochs.
  store.updateSettings({ epochs: 30, augment: false });
  const trainStarted = performance.now();
  await store.startTraining();
  const trainSeconds = (performance.now() - trainStarted) / 1000;
  check(
    "a real detector trained on the hand-boxed pictures",
    useStore.getState().hasModel && useStore.getState().train.phase === "done",
    {
      phase: useStore.getState().train.phase,
      boxes: TRAIN_IMAGES * 2,
      seconds: Number(trainSeconds.toFixed(1)),
    },
  );

  // The pictures the annotator has to guess at: identical scenes, no boxes.
  const truth = scene(0).truth;
  const targets: string[] = [];
  for (let i = 0; i < TARGETS; i += 1) {
    await store.addFromDrawable(scene(0).canvas, null, "import");
    targets.push(useStore.getState().manifest!.images[TRAIN_IMAGES + i].id);
  }
  check(
    "the pictures to be guessed at start with nothing drawn on them",
    targets.every((id) => boxCount(id) === 0),
    `${targets.length} pictures`,
  );

  setScoreThreshold(0.3);
  const root = createRoot(document.getElementById("root")!);
  root.render(<Harness />);
  await settle();

  // --- 1. an unboxed picture gets guesses --------------------------------------

  // The picture goes up first and the guessing follows it, so the thinking hint
  // exists only for as long as the head pass takes. How long that is depends
  // entirely on the machine, so it is watched for rather than sampled: a poll
  // that happens to land after the guesses is not evidence of anything.
  let thinking = false;
  const watcher = new MutationObserver((records) => {
    for (const record of records) {
      for (const node of record.addedNodes) {
        if (!(node instanceof HTMLElement)) continue;
        if (node.matches("[data-thinking]") || node.querySelector("[data-thinking]")) {
          thinking = true;
        }
      }
    }
  });
  watcher.observe(document.body, { childList: true, subtree: true });
  goTo(TRAIN_IMAGES);
  await tick();
  const shownFirst = document.querySelector(".anno__img") !== null;
  await until("the first guesses", () => guessNodes().length > 0);
  watcher.disconnect();
  const first = guessNodes();
  const dashed = first.every((node) => getComputedStyle(node).borderTopStyle === "dashed");
  check(
    "an unboxed picture is pre-drawn with the detector's guesses, dashed",
    first.length > 0 && dashed,
    `${first.length} guesses, dashed ${dashed}`,
  );
  check(
    "the picture is on screen before the guessing is, under a quiet thinking hint",
    shownFirst && thinking,
    `picture shown ${shownFirst}, thinking hint ${thinking}`,
  );

  const perLabel = new Map(first.map((node) => [node.dataset.guessLabel, node]));
  const best = truth.map((item) => {
    const wanted = labelOf(item.kind);
    const mine = first
      .filter((node) => node.dataset.guessLabel === wanted)
      .map((node) => overlap(guessRect(node), item.rect));
    return { kind: item.kind, iou: mine.length > 0 ? Math.max(...mine) : 0 };
  });
  check(
    "the guesses carry the right labels and sit on the right things",
    perLabel.size === 2 && best.every((item) => item.iou >= 0.4),
    best.map((item) => `${item.kind} overlap ${item.iou.toFixed(3)}`).join(", "),
  );
  check(
    "a guess is not a box: the manifest still holds nothing for this picture",
    boxCount(targets[0]) === 0,
    `${first.length} guesses on screen, 0 boxes stored`,
  );
  check(
    "the first sight of a guess explains itself once",
    document.querySelector(".anno__firsthint") !== null &&
      window.localStorage.getItem("belo.detect.guessHint") === null,
  );

  // --- 2. keeping exactly one --------------------------------------------------

  const chosen = first[0];
  const chosenRect = guessRect(chosen);
  const chosenLabel = chosen.dataset.guessLabel!;
  tap(chosen);
  await settle();
  const stored = boxesOf(recordOf(targets[0]));
  const sameRect =
    stored.length === 1 &&
    Math.abs(stored[0].x - chosenRect.x) < 1e-4 &&
    Math.abs(stored[0].y - chosenRect.y) < 1e-4 &&
    Math.abs(stored[0].w - chosenRect.w) < 1e-4 &&
    Math.abs(stored[0].h - chosenRect.h) < 1e-4;
  check(
    "clicking one guess commits exactly that one box, in exactly that place",
    stored.length === 1 && stored[0].labelId === chosenLabel && sameRect,
    `${stored.length} box stored, corner off by ${
      stored.length === 1 ? Math.abs(stored[0].x - chosenRect.x).toExponential(1) : "n/a"
    }, ${guessNodes().length} guesses left of ${first.length}`,
  );
  check(
    "a kept guess becomes a normal box, drawn solid and picked up for editing",
    document.querySelectorAll(".abox").length === 1 &&
      getComputedStyle(document.querySelector(".abox")!).borderTopStyle === "solid",
  );
  check(
    "keeping a guess counts as a change to the dataset",
    useStore.getState().stale && useStore.getState().manifest!.modelStale === true,
  );

  // Undo has to cover it, because it went in through the same door as a drawn box.
  window.dispatchEvent(
    new KeyboardEvent("keydown", { key: "z", metaKey: true, bubbles: true, cancelable: true }),
  );
  await settle();
  check(
    "undo takes a kept guess back out again",
    boxCount(targets[0]) === 0,
    `${boxCount(targets[0])} boxes after cmd z`,
  );

  // --- 3. throwing one away ----------------------------------------------------

  const beforeDrop = guessNodes().length;
  const dropTarget = guessNodes()[0];
  const dropButton = dropTarget.querySelector(".aprop__drop")!;
  const dropBox = dropButton.getBoundingClientRect();
  pointer(dropButton, "pointerdown", dropBox.left + 5, dropBox.top + 5);
  pointer(dropButton, "pointerup", dropBox.left + 5, dropBox.top + 5);
  await settle();
  check(
    "the x on a guess throws it away and writes nothing",
    guessNodes().length === beforeDrop - 1 && boxCount(targets[0]) === 0,
    `${beforeDrop} guesses became ${guessNodes().length}, still 0 boxes stored`,
  );

  // --- 4. drawing straight across a guess --------------------------------------

  goTo(TRAIN_IMAGES + 1);
  await settle();
  await until("guesses on the second picture", () => guessNodes().length > 0);
  const drawOver = guessNodes()[0];
  const drawCount = guessNodes().length;
  const frame = document.querySelector<HTMLElement>(".anno__frame")!;
  const frameBox = frame.getBoundingClientRect();
  const from = drawOver.getBoundingClientRect();
  const startX = from.left + from.width / 2;
  const startY = from.top + from.height / 2;
  const endX = startX + frameBox.width * 0.22;
  const endY = startY + frameBox.height * 0.22;
  pointer(drawOver, "pointerdown", startX, startY);
  pointer(window, "pointermove", endX, endY);
  // The release lands back on the guess, which is the case that would wrongly
  // read as a click if the gesture were not measured.
  pointer(drawOver, "pointerup", endX, endY);
  await settle();
  const drawn = boxesOf(recordOf(targets[1]));
  const wantedX = (startX - frameBox.left) / frameBox.width;
  const wantedW = (endX - startX) / frameBox.width;
  check(
    "a rectangle dragged across a guess is drawn, not mistaken for keeping it",
    drawn.length === 1 &&
      Math.abs(drawn[0].x - wantedX) < 0.01 &&
      Math.abs(drawn[0].w - wantedW) < 0.01 &&
      guessNodes().length === drawCount,
    `1 drawn box at x ${drawn[0]?.x.toFixed(3)} wanted ${wantedX.toFixed(3)}, ${guessNodes().length} guesses still on screen`,
  );

  // --- 5. leaving without deciding ---------------------------------------------

  goTo(TRAIN_IMAGES + 2);
  await settle();
  const passing = await until("guesses on the picture that gets walked away from", () => {
    const nodes = guessNodes();
    return nodes.length > 0 ? nodes.length : null;
  });
  goTo(TRAIN_IMAGES + 3);
  await settle();
  check(
    "walking away from guesses throws them out and leaves the manifest alone",
    boxCount(targets[2]) === 0,
    `${passing} guesses were on screen, ${boxCount(targets[2])} boxes stored afterwards`,
  );

  // --- 6. keep them all and go on ----------------------------------------------

  await until("guesses on the picture that is confirmed wholesale", () => guessNodes().length > 0);
  const sweep = guessNodes().length;
  const wasAt = shownIndex;
  press(" ");
  await settle();
  check(
    "space keeps every guess and turns the page",
    boxCount(targets[3]) === sweep && shownIndex === wasAt + 1,
    `${sweep} guesses became ${boxCount(targets[3])} boxes, picture ${wasAt} became ${shownIndex}`,
  );

  // --- 7. a picture that already has boxes -------------------------------------

  goTo(0);
  await settle(10);
  check(
    "a picture the person already boxed is left alone",
    guessNodes().length === 0 && document.querySelectorAll(".abox").length === 2,
    `${guessNodes().length} guesses, ${document.querySelectorAll(".abox").length} boxes drawn by hand`,
  );
  const firstPicture = useStore.getState().manifest!.images[0].id;
  const boxesBeforeAsking = boxCount(firstPicture);
  const suggestAgain = await until("the suggest-again control", () => {
    const button = document.querySelector<HTMLButtonElement>("[data-suggest-again]");
    return button && !button.disabled ? button : null;
  });
  suggestAgain.click();
  let lastSuggestionRequest = Date.now();
  const askedFor = await until("the guesses asked for explicitly", () => {
    const nodes = guessNodes();
    if (nodes.length > 0) return nodes.length;
    // A background detector run can briefly own the ML gate. The annotator
    // drops that inference rather than queueing it, so retry once the button
    // is enabled instead of turning a transient drop into a flaky timeout.
    const button = document.querySelector<HTMLButtonElement>("[data-suggest-again]");
    if (button && !button.disabled && Date.now() - lastSuggestionRequest >= 500) {
      lastSuggestionRequest = Date.now();
      button.click();
    }
    return null;
  });
  check(
    "Suggest again offers guesses on a boxed picture and replaces nothing",
    askedFor > 0 && boxCount(firstPicture) === boxesBeforeAsking,
    `${askedFor} guesses added on top of ${boxesBeforeAsking} untouched boxes`,
  );

  press("Enter");
  await settle();
  check(
    "enter keeps every guess without turning the page",
    boxCount(firstPicture) === boxesBeforeAsking + askedFor &&
      shownIndex === 0 &&
      guessNodes().length === 0,
    `${boxesBeforeAsking} boxes became ${boxCount(firstPicture)}, still on picture ${shownIndex}`,
  );
  window.dispatchEvent(
    new KeyboardEvent("keydown", { key: "z", metaKey: true, bubbles: true, cancelable: true }),
  );
  await settle();
  check(
    "one undo takes a whole batch of kept guesses back out",
    boxCount(firstPicture) === boxesBeforeAsking,
    `back to ${boxCount(firstPicture)} boxes in one step`,
  );

  // --- 8. a fresh model refreshes an untouched picture --------------------------

  goTo(TRAIN_IMAGES + 4);
  await settle();
  await until("guesses on the last picture", () => guessNodes().length > 0);
  for (const node of guessNodes()) node.dataset.mark = "before";
  const manifest = useStore.getState().manifest!;
  useStore.setState({
    manifest: {
      ...manifest,
      model: { ...manifest.model!, trainedAt: new Date(Date.now() + 1000).toISOString() },
    },
  });
  const refreshed = await until("the guesses to be made again", () => {
    const nodes = guessNodes();
    if (nodes.length === 0) return null;
    return nodes.every((node) => node.dataset.mark !== "before") ? nodes.length : null;
  });
  check(
    "a retrain that lands while the annotator is open refreshes an untouched picture",
    refreshed > 0 && boxCount(targets[4]) === 0,
    `${refreshed} guesses redrawn from the newer model, still 0 boxes stored`,
  );

  // The other half of that rule: a picture somebody is already deciding about
  // keeps the guesses it started with until the next picture.
  const held = guessNodes()[0].querySelector<HTMLElement>(".aprop__drop")!;
  const heldBox = held.getBoundingClientRect();
  pointer(held, "pointerdown", heldBox.left + 5, heldBox.top + 5);
  pointer(held, "pointerup", heldBox.left + 5, heldBox.top + 5);
  await settle();
  for (const node of guessNodes()) node.dataset.mark = "held";
  const nextManifest = useStore.getState().manifest!;
  useStore.setState({
    manifest: {
      ...nextManifest,
      model: { ...nextManifest.model!, trainedAt: new Date(Date.now() + 2000).toISOString() },
    },
  });
  await settle(20);
  check(
    "a picture already being decided about is not swapped out from under the hand",
    guessNodes().length > 0 && guessNodes().every((node) => node.dataset.mark === "held"),
    `${guessNodes().length} guesses kept their places through a new model`,
  );

  // --- 9. the shared score cutoff ----------------------------------------------

  goTo(TRAIN_IMAGES + 2);
  await settle();
  const atLow = await until("guesses at the low cutoff", () => {
    const nodes = guessNodes();
    return nodes.length > 0 ? nodes.length : null;
  });
  setScoreThreshold(0.7);
  await settle(20);
  const atHigh = guessNodes().length;
  const direct = await useStore.getState().proposeBoxes(targets[2], 0.7);
  check(
    "the Use view's score cutoff is the one the guesses obey",
    atHigh <= atLow && atHigh === (direct?.length ?? -1),
    `${atLow} guesses at 0.3, ${atHigh} at 0.7, the store itself answers ${direct?.length} at 0.7`,
  );
  setScoreThreshold(0.3);

  check("nothing closed the annotator on its own", closes === 0, `${closes} closes`);
  root.unmount();
  await store.closeProject();
}

main()
  .then(finish)
  .catch((err) => {
    console.error(err);
    fail(err);
  });
