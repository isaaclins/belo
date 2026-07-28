// Proof that the detector actually learns, run against the real bundled
// MobileNet in node.
//
// The geometry checks in check-detect.mjs say the decision layer is right. They
// cannot say the thing trains. This builds a synthetic detect project, forty
// pictures with coloured shapes at known positions, extracts real features with
// the frozen extractor the app ships, trains the detection head exactly the way
// the app does, and then measures on pictures the head never saw:
//
//   - the share of drawn boxes that were found, per label, at IoU 0.5
//   - the share of detections that were real, per label
//   - the mean overlap of the boxes it did find
//
//   node scripts/check-learning.mjs
//
// It takes about a minute on a laptop CPU. Node 22.15 or newer, which runs the
// TypeScript source through type stripping and allows the resolve hook below.

import { createRequire, registerHooks } from "node:module";
import { createHash } from "node:crypto";
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import * as tf from "@tensorflow/tfjs";
import { setWasmPaths } from "@tensorflow/tfjs-backend-wasm";

// The app's own modules import each other without a file extension, the way
// every bundler expects. Node needs the extension, so it is filled in here
// rather than written into the source for the sake of this one script.
registerHooks({
  resolve(specifier, context, next) {
    if (!specifier.startsWith(".")) return next(specifier, context);
    try {
      return next(specifier, context);
    } catch (err) {
      if (err?.code !== "ERR_MODULE_NOT_FOUND") throw err;
      return next(`${specifier}.ts`, context);
    }
  },
});

const HERE = dirname(fileURLToPath(import.meta.url));
const MODEL_PATH = join(HERE, "..", "public", "models", "mobilenet-v1-1.0-224", "model.json");
const SIZE = 224;
const TRAIN_IMAGES = 40;
const HOLDOUT_IMAGES = 24;

// The bar this run has to clear. Set a little below what the head reaches, so a
// normal run passes, and far above what a head that stopped learning could
// reach. The run is seeded, so these numbers do not move on their own.
const MIN_RECALL = 0.7;
const MIN_PRECISION = 0.75;
const MIN_LABEL_RECALL = 0.65;
const MIN_LABEL_PRECISION = 0.7;
const MIN_MEAN_IOU = 0.6;
const MIN_USE_FRAME_RATE = 0.6;

/** Deterministic noise, so a failure here is a real one and not a bad draw. */
function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const rng = mulberry32(20260728);
// Weight initialization and batch shuffling both draw from here too.
Math.random = rng;

const { detect, detectSpatialMap, trainDetector } = await import(
  new URL("../src/lib/ml/detector.ts", import.meta.url),
);
const { matchDetections, iou } = await import(new URL("../src/lib/detect.ts", import.meta.url));
const { DETECT_SETTINGS } = await import(new URL("../src/lib/types.ts", import.meta.url));

// ---------- The bundled extractor, loaded straight off disk ----------

function fileHandler(modelPath) {
  return {
    async load() {
      const json = JSON.parse(readFileSync(modelPath, "utf8"));
      const dir = dirname(modelPath);
      const specs = [];
      const chunks = [];
      for (const group of json.weightsManifest) {
        for (const path of group.paths) chunks.push(readFileSync(join(dir, path)));
        specs.push(...group.weights);
      }
      const total = chunks.reduce((sum, chunk) => sum + chunk.byteLength, 0);
      const data = new Uint8Array(total);
      let at = 0;
      for (const chunk of chunks) {
        data.set(new Uint8Array(chunk.buffer, chunk.byteOffset, chunk.byteLength), at);
        at += chunk.byteLength;
      }
      return {
        modelTopology: json.modelTopology,
        weightSpecs: specs,
        weightData: data.buffer,
        format: json.format,
      };
    },
  };
}

// ---------- A synthetic detect project ----------

const LABELS = ["lbl_block", "lbl_dot"];
const INK = [0.11, 0.09, 0.14];
const SHAPE_COLOR = [
  [0.18, 0.42, 1.0], // block: strong blue
  [1.0, 0.55, 0.12], // dot: orange
];

function paintBackground(pixels, rng) {
  const base = [0.55 + rng() * 0.4, 0.5 + rng() * 0.45, 0.42 + rng() * 0.5];
  const stripe = 12 + Math.floor(rng() * 40);
  const tilt = rng() < 0.5;
  for (let y = 0; y < SIZE; y += 1) {
    for (let x = 0; x < SIZE; x += 1) {
      const band = ((tilt ? x + y : x - y) / stripe) % 2 < 1 ? 0.04 : -0.04;
      const at = (y * SIZE + x) * 3;
      for (let c = 0; c < 3; c += 1) {
        pixels[at + c] = Math.max(0, Math.min(1, base[c] + band + (rng() - 0.5) * 0.08));
      }
    }
  }
}

function paintShape(pixels, kind, rect) {
  const x0 = Math.round(rect.x * SIZE);
  const y0 = Math.round(rect.y * SIZE);
  const x1 = Math.round((rect.x + rect.w) * SIZE);
  const y1 = Math.round((rect.y + rect.h) * SIZE);
  const cx = (x0 + x1) / 2;
  const cy = (y0 + y1) / 2;
  const rx = (x1 - x0) / 2;
  const ry = (y1 - y0) / 2;
  const color = SHAPE_COLOR[kind];
  for (let y = y0; y < y1; y += 1) {
    for (let x = x0; x < x1; x += 1) {
      if (x < 0 || y < 0 || x >= SIZE || y >= SIZE) continue;
      const inside =
        kind === 0
          ? true
          : ((x + 0.5 - cx) / rx) ** 2 + ((y + 0.5 - cy) / ry) ** 2 <= 1;
      if (!inside) continue;
      const edge =
        kind === 0
          ? x - x0 < 3 || x1 - x < 3 || y - y0 < 3 || y1 - y < 3
          : ((x + 0.5 - cx) / rx) ** 2 + ((y + 0.5 - cy) / ry) ** 2 > 0.78;
      const at = (y * SIZE + x) * 3;
      const paint = edge ? INK : color;
      for (let c = 0; c < 3; c += 1) pixels[at + c] = paint[c];
    }
  }
}

function makePicture(rng) {
  const pixels = new Float32Array(SIZE * SIZE * 3);
  paintBackground(pixels, rng);
  const boxes = [];
  const wanted = rng() < 0.25 ? 1 : 2;
  let guard = 0;
  while (boxes.length < wanted && guard < 40) {
    guard += 1;
    const kind = boxes.length === 0 ? Math.floor(rng() * 2) : 1 - boxes[0].kind;
    const w = 0.18 + rng() * 0.24;
    const h = kind === 0 ? 0.16 + rng() * 0.26 : w * (0.85 + rng() * 0.3);
    const rect = {
      x: rng() * (1 - w - 0.04) + 0.02,
      y: rng() * (1 - h - 0.04) + 0.02,
      w,
      h,
    };
    if (boxes.some((box) => iou(box, rect) > 0)) continue;
    boxes.push({ ...rect, kind, labelId: LABELS[kind] });
  }
  for (const box of boxes) paintShape(pixels, box.kind, box);
  return { pixels, boxes: boxes.map(({ kind: _kind, ...box }) => box) };
}

// ---------- Run ----------

let failures = 0;
const results = [];
function check(name, ok, detail = "") {
  results.push(`${ok ? "ok  " : "FAIL"} ${name}${detail ? ` — ${detail}` : ""}`);
  if (!ok) failures += 1;
}

// WASM is one of the two backends the app itself runs on and the only one
// available here, so the proof is also a check that a detect project can train
// on a machine without WebGL.
const require = createRequire(import.meta.url);
setWasmPaths(`${dirname(require.resolve("@tensorflow/tfjs-backend-wasm/dist/tf-backend-wasm.js"))}/`);
await tf.setBackend("wasm");
await tf.ready();
console.log(`backend ${tf.getBackend()}`);

// Cancellation and thrown callbacks must not hand a partial detector back or
// leave its variables alive. These tiny runs exercise both exits before the
// longer learning proof starts.
const datasetRandom = Math.random;
Math.random = mulberry32(20260729);
const lifecycleSample = {
  imageId: "lifecycle",
  features: new Float32Array(49 * 1024),
  boxes: [],
};
const lifecycleSettings = {
  epochs: 1,
  batchSize: 1,
  learningRate: 0.001,
  hiddenUnits: 16,
  validationSplit: 0,
};
const lifecycleTensors = tf.memory().numTensors;
let cancelledCleanly = false;
try {
  await trainDetector({
    samples: [lifecycleSample],
    labelIds: ["lbl_test"],
    settings: lifecycleSettings,
    shouldStop: () => true,
  });
} catch (err) {
  cancelledCleanly = err instanceof Error && err.message === "Training cancelled";
}
check("cancellation rejects instead of returning a partial detector", cancelledCleanly);
check(
  "cancellation disposes detector variables",
  tf.memory().numTensors === lifecycleTensors,
  `tensor delta ${tf.memory().numTensors - lifecycleTensors}`,
);

const callbackFailure = new Error("progress callback failed");
let callbackPropagated = false;
try {
  await trainDetector({
    samples: [lifecycleSample],
    labelIds: ["lbl_test"],
    settings: lifecycleSettings,
    onProgress: () => {
      throw callbackFailure;
    },
  });
} catch (err) {
  callbackPropagated = err === callbackFailure;
}
check("training errors propagate to the store", callbackPropagated);
check(
  "training errors dispose detector variables",
  tf.memory().numTensors === lifecycleTensors,
  `tensor delta ${tf.memory().numTensors - lifecycleTensors}`,
);
Math.random = datasetRandom;

const full = await tf.loadLayersModel(fileHandler(MODEL_PATH));
const extractor = tf.model({ inputs: full.inputs, outputs: full.getLayer("conv_pw_13_relu").output });

/** Exactly what features.ts does: scale into -1..1, cut at conv_pw_13_relu. */
function featuresFor(pixels) {
  return tf.tidy(() => {
    const input = tf.tensor4d(pixels, [1, SIZE, SIZE, 3]).mul(2).sub(1);
    const activation = extractor.predict(input);
    return activation.reshape([49 * 1024]).dataSync().slice();
  });
}

// The dataset is fully determined by this file, so the extracted features are
// cached under a name that changes the moment the file does. A rerun while the
// detector itself is being worked on then costs seconds instead of half a
// minute, and can never read a cache that describes different pictures.
const stamp = createHash("sha1").update(readFileSync(fileURLToPath(import.meta.url))).digest("hex");
const cacheFile = join(tmpdir(), `belo-learning-${stamp.slice(0, 12)}.json`);

const started = Date.now();
const pictures = [];
if (existsSync(cacheFile)) {
  const cached = JSON.parse(readFileSync(cacheFile, "utf8"));
  for (const entry of cached) {
    pictures.push({
      imageId: entry.imageId,
      features: new Float32Array(Buffer.from(entry.features, "base64").buffer.slice(0)),
      boxes: entry.boxes,
    });
  }
} else {
  for (let i = 0; i < TRAIN_IMAGES + HOLDOUT_IMAGES; i += 1) {
    const picture = makePicture(rng);
    pictures.push({
      imageId: `img_${i}`,
      features: new Float32Array(featuresFor(picture.pixels)),
      boxes: picture.boxes,
    });
  }
  writeFileSync(
    cacheFile,
    JSON.stringify(
      pictures.map((p) => ({
        imageId: p.imageId,
        features: Buffer.from(p.features.buffer).toString("base64"),
        boxes: p.boxes,
      })),
    ),
  );
}
const readMs = Date.now() - started;
// Reading the cache draws no random numbers, so the stream is reset here and
// the training run that follows is the same one either way.
Math.random = mulberry32(913);

const samples = pictures.slice(0, TRAIN_IMAGES);
const holdout = pictures.slice(TRAIN_IMAGES);
const drawn = pictures.reduce((sum, p) => sum + p.boxes.length, 0);
console.log(
  `read ${pictures.length} pictures with ${drawn} boxes in ${(readMs / 1000).toFixed(1)}s ` +
    `(${(readMs / pictures.length).toFixed(0)}ms per picture)`,
);
console.log(
  `feature cache: ${((pictures[0].features.byteLength * pictures.length) / 1024 / 1024).toFixed(1)} MB ` +
    `for ${pictures.length} pictures, ${(pictures[0].features.byteLength / 1024).toFixed(0)} KB each`,
);

const trainStarted = Date.now();
const { detector, history } = await trainDetector({
  samples,
  validation: holdout,
  labelIds: LABELS,
  // Exactly the settings a new detect project is created with.
  settings: DETECT_SETTINGS,
  onProgress: (epoch) => {
    if (epoch.epoch % 25 === 0 || epoch.epoch === 1) {
      console.log(
        `epoch ${String(epoch.epoch).padStart(3)}  loss ${epoch.loss.toFixed(3)}  ` +
          `cls ${epoch.clsLoss.toFixed(3)}  loc ${epoch.locLoss.toFixed(3)}  ` +
          `val ${epoch.valLoss === null ? "--" : epoch.valLoss.toFixed(3)}`,
      );
    }
  },
});
console.log(`trained ${history.length} epochs in ${((Date.now() - trainStarted) / 1000).toFixed(1)}s`);

function evaluate(set, options = {}) {
  const stats = new Map(LABELS.map((id) => [id, { truth: 0, found: 0, predicted: 0, hit: 0 }]));
  const overlaps = [];
  for (const picture of set) {
    const detections = detect(detector, picture.features, options);
    const matched = matchDetections(picture.boxes, detections, 0.5);
    for (const box of picture.boxes) stats.get(box.labelId).truth += 1;
    detections.forEach((detection, i) => {
      const stat = stats.get(detection.labelId);
      stat.predicted += 1;
      if (matched.hits[i] >= 0) {
        stat.hit += 1;
        stat.found += 1;
        overlaps.push(iou(picture.boxes[matched.hits[i]], detection));
      }
    });
  }
  const meanIoU = overlaps.length === 0 ? 0 : overlaps.reduce((a, b) => a + b, 0) / overlaps.length;
  return { stats, meanIoU, matches: overlaps.length };
}

const trainScore = evaluate(samples);
const score = evaluate(holdout);

// This is the Use view seam, not a second decoding fixture: each cached map is
// rebuilt as the [49, 1024] tensor embedWithSpatial hands the store, then sent
// through detectSpatialMap before the same IoU matcher checks the rendered box
// list. The tensor count before and after also proves the caller-owned map can
// be disposed every tick without accumulating one per frame.
function evaluateUsePath(set) {
  const tensorsBefore = tf.memory().numTensors;
  const started = performance.now();
  let framesHit = 0;
  let boxesHit = 0;
  const overlaps = [];
  for (const picture of set) {
    const cells = tf.tensor2d(picture.features, [49, 1024]);
    let detections;
    try {
      detections = detectSpatialMap(detector, cells);
    } finally {
      cells.dispose();
    }
    const matched = matchDetections(picture.boxes, detections, 0.5);
    let frameHit = false;
    detections.forEach((detection, index) => {
      const truthIndex = matched.hits[index];
      if (truthIndex < 0) return;
      frameHit = true;
      boxesHit += 1;
      overlaps.push(iou(picture.boxes[truthIndex], detection));
    });
    if (frameHit) framesHit += 1;
  }
  return {
    framesHit,
    boxesHit,
    frameRate: framesHit / set.length,
    meanIoU: overlaps.reduce((sum, overlap) => sum + overlap, 0) / Math.max(1, overlaps.length),
    msPerFrame: (performance.now() - started) / set.length,
    tensorDelta: tf.memory().numTensors - tensorsBefore,
  };
}

const usePath = evaluateUsePath(holdout);
console.log(
  `\nUse data path at score 0.30: ${usePath.framesHit}/${holdout.length} frames have a box at IoU >= 0.5, ` +
    `${usePath.boxesHit} boxes hit, mean IoU ${usePath.meanIoU.toFixed(3)}, ` +
    `${usePath.msPerFrame.toFixed(1)}ms head + decode, tensor delta ${usePath.tensorDelta}`,
);
check(
  `Use boxes land on shapes in at least ${MIN_USE_FRAME_RATE * 100}% of held out frames`,
  usePath.frameRate >= MIN_USE_FRAME_RATE,
  `${(usePath.frameRate * 100).toFixed(0)}% (${usePath.framesHit}/${holdout.length})`,
);
check(
  "Use spatial maps are disposed after every frame",
  usePath.tensorDelta === 0,
  `tensor delta ${usePath.tensorDelta}`,
);

console.log("\nheld out set by score threshold");
for (const threshold of [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]) {
  const sweep = evaluate(holdout, { scoreThreshold: threshold });
  let sweptTruth = 0;
  let sweptFound = 0;
  let sweptPredicted = 0;
  for (const stat of sweep.stats.values()) {
    sweptTruth += stat.truth;
    sweptFound += stat.found;
    sweptPredicted += stat.predicted;
  }
  console.log(
    `  at ${threshold.toFixed(2)}  recall ${((sweptFound / sweptTruth) * 100).toFixed(0)}%  ` +
      `precision ${((sweptFound / Math.max(1, sweptPredicted)) * 100).toFixed(0)}%  ` +
      `mean IoU ${sweep.meanIoU.toFixed(3)}`,
  );
}

let truth = 0;
let found = 0;
let predicted = 0;
console.log("\nheld out set, IoU 0.5");
for (const [labelId, stat] of score.stats) {
  truth += stat.truth;
  found += stat.found;
  predicted += stat.predicted;
  const recall = stat.truth === 0 ? 0 : stat.found / stat.truth;
  const precision = stat.predicted === 0 ? 0 : stat.hit / stat.predicted;
  console.log(
    `  ${labelId.padEnd(10)} recall ${(recall * 100).toFixed(0)}% (${stat.found}/${stat.truth})  ` +
      `precision ${(precision * 100).toFixed(0)}% (${stat.hit}/${stat.predicted})`,
  );
  check(
    `${labelId} is found at least ${MIN_LABEL_RECALL * 100}% of the time`,
    recall >= MIN_LABEL_RECALL,
    `${(recall * 100).toFixed(0)}%`,
  );
  check(
    `${labelId} detections are right at least ${MIN_LABEL_PRECISION * 100}% of the time`,
    precision >= MIN_LABEL_PRECISION,
    `${(precision * 100).toFixed(0)}%`,
  );
}

const recall = truth === 0 ? 0 : found / truth;
const precision = predicted === 0 ? 0 : found / predicted;
console.log(
  `  overall     recall ${(recall * 100).toFixed(0)}%  precision ${(precision * 100).toFixed(0)}%  ` +
    `mean IoU ${score.meanIoU.toFixed(3)}`,
);
console.log(
  `  training set recall ${((trainScore.stats.get(LABELS[0]).found + trainScore.stats.get(LABELS[1]).found) / samples.reduce((s, p) => s + p.boxes.length, 0) * 100).toFixed(0)}%  mean IoU ${trainScore.meanIoU.toFixed(3)}\n`,
);

check(`held out recall is at least ${MIN_RECALL * 100}%`, recall >= MIN_RECALL, `${(recall * 100).toFixed(0)}%`);
check(`held out precision is at least ${MIN_PRECISION * 100}%`, precision >= MIN_PRECISION, `${(precision * 100).toFixed(0)}%`);
check(`found boxes overlap at least ${MIN_MEAN_IOU}`, score.meanIoU >= MIN_MEAN_IOU, score.meanIoU.toFixed(3));
check(
  "the loss fell over the run",
  history[history.length - 1].loss < history[0].loss * 0.5,
  `${history[0].loss.toFixed(3)} to ${history[history.length - 1].loss.toFixed(3)}`,
);
check(
  "a picture with nothing in it stays quiet",
  (() => {
    const empty = new Float32Array(49 * 1024);
    return detect(detector, empty).length === 0;
  })(),
);

console.log(results.join("\n"));
console.log(failures === 0 ? "\nthe detector learns" : `\n${failures} check(s) failed`);
process.exit(failures === 0 ? 0 : 1);
