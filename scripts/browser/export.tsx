// First half of the closed loop on model export, in a real browser.
//
// This page trains a real classifier and builds a real detector, exports both
// through the app's own export path, and publishes three things for the node
// side: the two archives, a fixture picture, and the answers Belo itself gives
// for that picture. scripts/check-export.mjs then unzips an archive, serves it
// like any static folder, runs the example page inside it and compares.
//
// Nothing in src/ is mocked. The only stand-in is the Tauri shim, which is the
// same in-memory fake the overlay check uses, plus a save dialog that always
// answers with the same path.
//
//   npm run check:export

import { useStore } from "../../src/lib/store";
import { storage } from "../../src/lib/storage";
import { embedDetectFeatures, loadDetectFeatureModel } from "../../src/lib/ml/features";
import {
  buildDetector,
  detectSpatialMap,
  serializeDetector,
} from "../../src/lib/ml/detector";
import { SCORE_THRESHOLD } from "../../src/lib/detect";
import { fakeExportTarget, installTauriShim, reporter, writtenExport } from "./tauri-shim";

installTauriShim();
const { check, fail, finish } = reporter();

const SIDE = 224;
const FRAMES = 6;
/**
 * The detector under test is freshly built rather than trained, because what is
 * being compared is arithmetic and not accuracy: an independent implementation
 * of the head, the anchors and the decoder has to produce the same boxes from
 * the same weights. A fresh head answers a little above chance everywhere, so a
 * low cutoff yields a large set of boxes to disagree over, which is exactly what
 * makes the comparison worth running.
 */
const DETECT_CUTOFF = 0.05;

let seed = 24601;
function random(): number {
  seed = (seed * 1103515245 + 12345) % 2147483648;
  return seed / 2147483648;
}

type Shape = "red" | "green";

function background(ctx: CanvasRenderingContext2D): void {
  ctx.fillStyle = "#8a8a8a";
  ctx.fillRect(0, 0, SIDE, SIDE);
  for (let i = 0; i < 700; i += 1) {
    const value = Math.round(110 + random() * 60);
    ctx.fillStyle = `rgb(${value},${value},${value})`;
    ctx.fillRect(random() * SIDE, random() * SIDE, 6, 6);
  }
}

function drawShape(ctx: CanvasRenderingContext2D, shape: Shape, side: "left" | "right"): void {
  const cx = side === "left" ? SIDE * 0.25 : SIDE * 0.75;
  const cy = SIDE * 0.5 + (random() - 0.5) * 18;
  const size = SIDE * 0.3 + random() * 10;
  if (shape === "red") {
    ctx.fillStyle = "rgb(228,32,32)";
    ctx.beginPath();
    ctx.arc(cx, cy, size / 2, 0, Math.PI * 2);
    ctx.fill();
    return;
  }
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

/** Publishes one value in slices, because a 20 MB string is a poor CDP reply. */
function publish(name: string, value: string): void {
  (window as unknown as Record<string, unknown>)[name] = value;
}

async function main() {
  const store = useStore.getState();
  const out: Record<string, unknown> = {};
  (window as unknown as { __BELO_EXPORT__: Record<string, unknown> }).__BELO_EXPORT__ = out;

  // The picture both sides answer about. PNG, so the bytes the example loads are
  // the pixels Belo scored, without a JPEG round trip in between.
  const fixture = frame([
    { shape: "red", side: "left" },
    { shape: "green", side: "right" },
  ]);
  publish("__BELO_FIXTURE__", fixture.toDataURL("image/png").split(",")[1]);

  // --- a classify project, trained for real, then exported ------------------

  await store.createProject("Export check", "single");
  const red = useStore.getState().addLabel("Redcircle");
  const green = useStore.getState().addLabel("Greenblock");
  if (!red || !green) throw new Error("labels could not be created");
  for (let i = 0; i < FRAMES; i += 1) {
    await store.addFromDrawable(frame([{ shape: "red", side: i % 2 ? "left" : "right" }]), red, "import");
    await store.addFromDrawable(
      frame([{ shape: "green", side: i % 2 ? "right" : "left" }]),
      green,
      "import",
    );
  }
  await store.startTraining();
  check(
    "a real classifier trained inside the browser",
    useStore.getState().hasModel && useStore.getState().train.phase === "done",
    { phase: useStore.getState().train.phase, accuracy: useStore.getState().train.accuracy },
  );

  const prediction = await store.predict(fixture, false);
  if (!prediction) throw new Error("predict returned nothing");
  out.classifyLabelIds = useStore.getState().modelLabelIds;
  out.classifyScores = prediction.scores.map((row) => ({ labelId: row.labelId, score: row.score }));

  fakeExportTarget("/mem/export/classify.zip");
  const classifyExported = await store.exportModel();
  const classifyZip = writtenExport("/mem/export/classify.zip");
  check("exporting a trained classifier wrote an archive", classifyExported && Boolean(classifyZip), {
    bytes: classifyZip ? Math.round((classifyZip.length * 3) / 4) : 0,
  });
  publish("__BELO_ZIP_CLASSIFY__", classifyZip ?? "");

  // An export must refuse to ship weights that no longer describe the dataset.
  useStore.setState({ stale: true });
  check("a stale model refuses to export", (await store.exportModel()) === false);
  useStore.setState({ stale: false });

  // --- a detect project, exported with the detector it holds ----------------

  await store.createProject("Export detect", "detect");
  const one = useStore.getState().addLabel("Redcircle");
  const two = useStore.getState().addLabel("Greenblock");
  if (!one || !two) throw new Error("detect labels could not be created");
  const detectPath = useStore.getState().path!;
  await store.addFromDrawable(fixture, null, "import");
  const picture = useStore.getState().manifest!.images[0];
  useStore.getState().addBox(picture.id, one, { x: 0.14, y: 0.36, w: 0.28, h: 0.28 });
  useStore.getState().addBox(picture.id, two, { x: 0.62, y: 0.36, w: 0.28, h: 0.28 });

  const detector = buildDetector([one, two], 96);
  await storage.writeDetector(detectPath, JSON.stringify(serializeDetector(detector)));
  const manifest = useStore.getState().manifest!;
  useStore.setState({
    manifest: {
      ...manifest,
      modelStale: false,
      model: {
        trainedAt: new Date().toISOString(),
        mode: "detect",
        labelIds: [one, two],
        epochs: 1,
        accuracy: 0,
        valAccuracy: null,
        imageCount: 1,
        settings: manifest.settings,
        precision: 0,
        recall: 0,
        boxCount: 2,
      },
    },
    hasModel: true,
    stale: false,
    modelLabelIds: [one, two],
  });

  await loadDetectFeatureModel();
  const { pack } = await embedDetectFeatures(fixture);
  const boxes = detectSpatialMap(detector, pack, { scoreThreshold: DETECT_CUTOFF });
  check(
    `the detector answers with boxes at a ${DETECT_CUTOFF} cutoff`,
    boxes.length > 0 && detectSpatialMap(detector, pack, {}).length >= 0,
    { boxes: boxes.length, atDefault: detectSpatialMap(detector, pack, {}).length, SCORE_THRESHOLD },
  );
  out.detectLabelIds = [one, two];
  out.detectCutoff = DETECT_CUTOFF;
  out.detectBoxes = boxes;

  fakeExportTarget("/mem/export/detect.zip");
  const detectExported = await store.exportModel();
  const detectZip = writtenExport("/mem/export/detect.zip");
  check("exporting a detector wrote an archive", detectExported && Boolean(detectZip), {
    bytes: detectZip ? Math.round((detectZip.length * 3) / 4) : 0,
  });
  publish("__BELO_ZIP_DETECT__", detectZip ?? "");

  detector.model.dispose();
  await store.closeProject();
}

main()
  .then(finish)
  .catch((err) => {
    console.error(err);
    fail(err);
  });
