// End to end check of the multi label attention overlay, in a real browser.
//
// Everything below the harness line is the app's own code: the store creates a
// multi label project, trains the real head on synthetic pictures, predicts with
// attention on, and the real AttentionOverlay component paints the result onto a
// canvas over a live camera feed. The assertions then read that canvas back,
// which is the part unit tests cannot do.
//
// The pictures are built so position is not a usable cue: each shape appears on
// both sides across the training set, and the two shapes share a frame in both
// arrangements. A model that learned "left means red" would pass a one sided
// test and fail the mirrored frame here.
//
//   npm run check:overlay

import { createRoot } from "react-dom/client";
import { AttentionOverlay } from "../../src/components/AttentionOverlay";
import { boxPixels, cropSquare, heat, HOT_RATIO, hotBox, mapRange } from "../../src/lib/attention";
import { camera } from "../../src/lib/capture";
import { useStore } from "../../src/lib/store";
import { MULTI_THRESHOLD } from "../../src/lib/types";
import { swatchHex } from "../../src/components/ui";
import { installTauriShim, reporter } from "./tauri-shim";

installTauriShim();
const { check, fail, finish } = reporter();

const SIDE = 256;
const FRAMES = 6;

/** Deterministic noise, so a failure is reproducible. */
let seed = 12345;
function random(): number {
  seed = (seed * 1103515245 + 12345) % 2147483648;
  return seed / 2147483648;
}

type Shape = "red" | "green";

function background(ctx: CanvasRenderingContext2D): void {
  ctx.fillStyle = "#8a8a8a";
  ctx.fillRect(0, 0, SIDE, SIDE);
  for (let i = 0; i < 900; i += 1) {
    const value = Math.round(110 + random() * 60);
    ctx.fillStyle = `rgb(${value},${value},${value})`;
    ctx.fillRect(random() * SIDE, random() * SIDE, 6, 6);
  }
}

/** Draws one shape centered in the left or the right half of the frame. */
function drawShape(ctx: CanvasRenderingContext2D, shape: Shape, side: "left" | "right"): void {
  const cx = side === "left" ? SIDE * 0.25 : SIDE * 0.75;
  const cy = SIDE * 0.5 + (random() - 0.5) * 20;
  const size = SIDE * 0.3 + random() * 10;
  if (shape === "red") {
    ctx.fillStyle = "rgb(228,32,32)";
    ctx.beginPath();
    ctx.arc(cx, cy, size / 2, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "rgb(120,0,0)";
    ctx.beginPath();
    ctx.arc(cx, cy, size / 5, 0, Math.PI * 2);
    ctx.fill();
    return;
  }
  // A checkered block, so the two classes differ in texture and not only in hue.
  const cells = 6;
  const step = size / cells;
  for (let row = 0; row < cells; row += 1) {
    for (let col = 0; col < cells; col += 1) {
      ctx.fillStyle = (row + col) % 2 === 0 ? "rgb(18,140,60)" : "rgb(6,60,26)";
      ctx.fillRect(cx - size / 2 + col * step, cy - size / 2 + row * step, step, step);
    }
  }
}

function frame(shapes: { shape: Shape; side: "left" | "right" }[]): HTMLCanvasElement {
  const canvas = document.createElement("canvas");
  canvas.width = SIDE;
  canvas.height = SIDE;
  const ctx = canvas.getContext("2d")!;
  background(ctx);
  for (const item of shapes) drawShape(ctx, item.shape, item.side);
  return canvas;
}

const wait = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));
const nextPaint = () =>
  new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));

interface Painted {
  count: number;
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
}

/**
 * Where one label was painted in its own opaque color. Only exact matches count:
 * the box stroke and its tag are drawn fully opaque in the label color, while
 * the heat map is alpha blended over the video and therefore never lands on the
 * exact value. So this reads back the box, not the tint.
 */
function paintedFor(canvas: HTMLCanvasElement, hex: string): Painted {
  const ctx = canvas.getContext("2d")!;
  const { data } = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  const found: Painted = { count: 0, minX: Infinity, maxX: -1, minY: Infinity, maxY: -1 };
  for (let i = 0; i < data.length; i += 4) {
    if (data[i + 3] < 250) continue;
    if (Math.abs(data[i] - r) > 4 || Math.abs(data[i + 1] - g) > 4 || Math.abs(data[i + 2] - b) > 4) {
      continue;
    }
    const pixel = i / 4;
    const x = pixel % canvas.width;
    const y = Math.floor(pixel / canvas.width);
    found.count += 1;
    found.minX = Math.min(found.minX, x);
    found.maxX = Math.max(found.maxX, x);
    found.minY = Math.min(found.minY, y);
    found.maxY = Math.max(found.maxY, y);
  }
  return found;
}

/**
 * Column of the center of mass of the hot cells, the ones the box is thresholded
 * on. The box bounds themselves are a poor side test: a hot region that reaches
 * into both halves gives a box centered on the frame no matter where its mass
 * sits, which is exactly the knife edge this avoids.
 */
function hotCenterCol(values: number[], rows: number, cols: number): number {
  const range = mapRange(values);
  let mass = 0;
  let sum = 0;
  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      const value = heat(values[row * cols + col] ?? 0, range);
      if (value < HOT_RATIO) continue;
      mass += value;
      sum += value * (col + 0.5);
    }
  }
  return mass > 0 ? sum / mass : -1;
}

/** Mean heat in the left and the right half of one label's score map. */
function halves(values: number[], rows: number, cols: number) {
  const range = mapRange(values);
  let left = 0;
  let right = 0;
  const mid = cols / 2;
  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      const value = heat(values[row * cols + col] ?? 0, range);
      if (col < Math.floor(mid)) left += value;
      else if (col >= Math.ceil(mid)) right += value;
    }
  }
  return { left, right, side: left > right ? "left" : "right" };
}

async function main() {
  const store = useStore.getState();

  // 1. A real multi label project, through the real store and storage seam.
  await store.createProject("Overlay check", "multi");
  const left = useStore.getState().addLabel("Redcircle");
  const right = useStore.getState().addLabel("Greenblock");
  check("a multi project with two labels was created", Boolean(left && right), {
    mode: useStore.getState().manifest?.mode,
  });
  if (!left || !right) throw new Error("labels could not be created");

  // 2. Training pictures. Each shape shows up on both sides, alone and paired,
  //    so the classifier cannot use position as a shortcut.
  for (let i = 0; i < FRAMES; i += 1) {
    const side = i % 2 === 0 ? "left" : "right";
    await store.addFromDrawable(frame([{ shape: "red", side }]), [left], "import");
    await store.addFromDrawable(
      frame([{ shape: "green", side: side === "left" ? "right" : "left" }]),
      [right],
      "import",
    );
    await store.addFromDrawable(
      i % 2 === 0
        ? frame([
            { shape: "red", side: "left" },
            { shape: "green", side: "right" },
          ])
        : frame([
            { shape: "red", side: "right" },
            { shape: "green", side: "left" },
          ]),
      [left, right],
      "import",
    );
  }
  check("the training set was written", useStore.getState().manifest?.images.length === FRAMES * 3, {
    images: useStore.getState().manifest?.images.length,
  });

  // 3. Real training run, real head.
  await store.startTraining();
  const train = useStore.getState().train;
  check("training finished with a model", useStore.getState().hasModel && train.phase === "done", {
    phase: train.phase,
    accuracy: train.accuracy,
    message: train.message,
  });
  check("the trained model separates the two labels", (train.accuracy ?? 0) >= 0.9, {
    accuracy: train.accuracy,
    valAccuracy: train.valAccuracy,
  });

  // 4. A live camera, only for its intrinsic size: the overlay maps the square
  //    the model saw onto the video's own aspect ratio.
  const video = document.getElementById("feed") as HTMLVideoElement;
  const stream = await camera.start(null);
  video.srcObject = stream;
  await video.play().catch(() => undefined);
  await wait(300);
  check("the fake camera is delivering a sized frame", video.videoWidth > 0 && video.videoHeight > 0, {
    size: `${video.videoWidth}x${video.videoHeight}`,
  });

  // 5. The real component, mounted over that video the way UseView mounts it.
  const stage = document.getElementById("stage")!;
  const host = document.createElement("div");
  stage.appendChild(host);
  const root = createRoot(host);
  const videoRef = { current: video };
  const labels = useStore.getState().manifest!.labels;

  const paint = async (attention: unknown, active = true) => {
    root.render(
      <AttentionOverlay
        videoRef={videoRef}
        attention={attention as never}
        labels={labels}
        active={active}
      />,
    );
    await nextPaint();
    await nextPaint();
  };

  const canvasOf = () => stage.querySelectorAll("canvas");
  const rect = cropSquare(
    { width: 480, height: 480 },
    { width: video.videoWidth, height: video.videoHeight },
  )!;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const splitX = (rect.x + rect.size / 2) * dpr;
  const hexOf = (labelId: string) =>
    swatchHex(labels.find((label) => label.id === labelId)!.color);
  const leftHex = hexOf(left);
  const rightHex = hexOf(right);

  // 6. Both labels present at once, in both arrangements.
  for (const arrangement of ["red-left", "red-right"] as const) {
    const shapes =
      arrangement === "red-left"
        ? ([
            { shape: "red", side: "left" },
            { shape: "green", side: "right" },
          ] as const)
        : ([
            { shape: "red", side: "right" },
            { shape: "green", side: "left" },
          ] as const);
    const test = frame([...shapes]);
    const prediction = await store.predict(test, true);
    if (!prediction) throw new Error("predict returned nothing");
    const above = prediction.scores
      .filter((row) => row.score >= MULTI_THRESHOLD)
      .map((row) => row.labelId);
    const mapped = prediction.attention?.maps.map((m) => m.labelId) ?? [];

    check(`${arrangement}: both labels answered yes`, above.length === 2, {
      scores: prediction.scores.map((row) => `${row.labelId}=${row.score.toFixed(3)}`),
    });
    check(
      `${arrangement}: one map per label above the threshold`,
      mapped.length === above.length && above.every((id) => mapped.includes(id)),
      { above, mapped },
    );
    check(
      `${arrangement}: the maps cover the whole feature grid`,
      (prediction.attention?.maps ?? []).every(
        (m) => m.values.length === prediction.attention!.rows * prediction.attention!.cols,
      ),
      { rows: prediction.attention?.rows, cols: prediction.attention?.cols },
    );

    // Where the evidence sits, straight off the real score maps.
    const wanted: Record<string, "left" | "right"> =
      arrangement === "red-left"
        ? { [left]: "left", [right]: "right" }
        : { [left]: "right", [right]: "left" };
    const grid = prediction.attention!;
    for (const map of grid.maps) {
      const spread = halves(map.values, grid.rows, grid.cols);
      check(
        `${arrangement}: the ${map.labelId === left ? "red" : "green"} map is hottest in the ${wanted[map.labelId]} half`,
        spread.side === wanted[map.labelId],
        { left: spread.left.toFixed(2), right: spread.right.toFixed(2) },
      );
    }

    // Both maps at once: one layer, one visible box per label.
    await paint(grid);
    check(`${arrangement}: exactly one canvas layer`, canvasOf().length === 1, {
      canvases: canvasOf().length,
    });
    const canvas = canvasOf()[0] as HTMLCanvasElement;
    check(
      `${arrangement}: both labels are painted on the shared layer`,
      paintedFor(canvas, leftHex).count > 50 && paintedFor(canvas, rightHex).count > 50,
      { red: paintedFor(canvas, leftHex).count, green: paintedFor(canvas, rightHex).count },
    );

    // Then one map at a time, so the later label's tint cannot sit on top of the
    // earlier label's box: this is where the painted box is measured against the
    // box the geometry says it should be.
    for (const map of grid.maps) {
      const name = map.labelId === left ? "red" : "green";
      await paint({ rows: grid.rows, cols: grid.cols, maps: [map] });
      const layer = canvasOf()[0] as HTMLCanvasElement;
      const drawn = paintedFor(layer, hexOf(map.labelId));
      const cells = hotBox(map.values, grid.rows, grid.cols)!;
      const expected = boxPixels(rect, grid.rows, grid.cols, cells);
      const off = (a: number, b: number) => Math.abs(a - b) <= 4;
      check(
        `${arrangement}: the painted ${name} box matches the computed box`,
        drawn.count > 50 &&
          off(drawn.minX, Math.max(0, expected.x) * dpr) &&
          off(drawn.maxX, Math.min(480, expected.x + expected.width) * dpr - 1) &&
          off(drawn.maxY, Math.min(480, expected.y + expected.height) * dpr - 1),
        {
          drawn: [drawn.minX, drawn.maxX, drawn.maxY],
          expected: [expected.x, expected.x + expected.width, expected.y + expected.height].map((v) =>
            Math.round(v * dpr),
          ),
          count: drawn.count,
        },
      );
      const hotCol = hotCenterCol(map.values, grid.rows, grid.cols);
      check(
        `${arrangement}: the ${name} box is built on hot cells in the ${wanted[map.labelId]} half`,
        (hotCol < grid.cols / 2 ? "left" : "right") === wanted[map.labelId],
        { hotCol: hotCol.toFixed(2), cols: grid.cols, cells },
      );
    }
  }

  // 7. One label only: the overlay must not invent a second box.
  const single = frame([{ shape: "red", side: "left" }]);
  const solo = await store.predict(single, true);
  const soloAbove = solo!.scores.filter((row) => row.score >= MULTI_THRESHOLD).map((r) => r.labelId);
  check("a single label frame yields one map", (solo!.attention?.maps.length ?? 0) === soloAbove.length, {
    maps: solo!.attention?.maps.map((m) => m.labelId),
    above: soloAbove,
  });
  await paint(solo!.attention);
  const soloCanvas = canvasOf()[0] as HTMLCanvasElement;
  check(
    "only the labels above the threshold are drawn",
    soloAbove.length === 1 &&
      soloAbove[0] === left &&
      paintedFor(soloCanvas, leftHex).count > 50 &&
      paintedFor(soloCanvas, rightHex).count === 0,
    {
      above: soloAbove,
      red: paintedFor(soloCanvas, leftHex).count,
      green: paintedFor(soloCanvas, rightHex).count,
    },
  );

  // 8. The toggle off clears the layer instead of leaving a stale frame behind.
  await paint(null, false);
  const cleared = canvasOf()[0] as HTMLCanvasElement;
  check(
    "turning the overlay off clears the canvas",
    paintedFor(cleared, leftHex).count === 0 && paintedFor(cleared, rightHex).count === 0,
    { red: paintedFor(cleared, leftHex).count, green: paintedFor(cleared, rightHex).count },
  );

  camera.stop(stream);
  await store.closeProject();
}

main()
  .then(finish)
  .catch((err) => {
    console.error(err);
    fail(err);
  });
