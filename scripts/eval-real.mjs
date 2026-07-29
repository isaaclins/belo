// End to end evaluation of Belo's detector on real photographs.
//
// Every other check in this repository is either pure arithmetic or a synthetic
// proof: coloured rectangles on striped backgrounds, where the detector has an
// easy time and a passing score means very little about a kitchen table. This
// one takes photographs taken by other people, boxes drawn by other people, and
// asks the same question a user asks: does it find the thing.
//
//   npm run eval:real                 # train and score, using the cached data
//   npm run eval:real -- --no-augment # the same run without mirrored copies
//   npm run eval:real -- --rebuild    # rebuild the image list from COCO
//
// What it does:
//
//   1. Reads scripts/data/coco-subset.json, a committed list of COCO val2017
//      photographs whose objects are large enough for a 7x7 grid to place, with
//      the human drawn boxes that come with them. Only the list is committed;
//      no image data ever enters the repository.
//   2. Downloads those photographs from images.cocodataset.org into .cache,
//      skipping anything already there, so a second run needs no network.
//   3. Converts each one the way the app does: center crop to a square, resize
//      to 224, boxes transformed with it.
//   4. Trains the real detector through src/lib/ml/detector.ts with the real
//      settings a new detect project is created with.
//   5. Scores held out photographs at IoU 0.5 and writes a handful of rendered
//      side by side comparisons to /tmp for a human to look at.
//
// This is a manual tool, not a CI job: it needs the network the first time and
// it takes minutes rather than seconds. It exits nonzero only when something is
// actually broken, never because the score was low. A low score is a result.
//
// Node 22.15 or newer.

import { createRequire, registerHooks } from "node:module";
import { execFileSync } from "node:child_process";
import {
  existsSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  rmSync,
  statSync,
  writeFileSync,
} from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { deflateSync } from "node:zlib";
import * as tf from "@tensorflow/tfjs";
import { setWasmPaths } from "@tensorflow/tfjs-backend-wasm";
import jpeg from "jpeg-js";

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
const ROOT = join(HERE, "..");
const CACHE = join(ROOT, ".cache", "coco");
const IMAGE_DIR = join(CACHE, "images");
const FEATURE_DIR = join(CACHE, "features");
const SUBSET_FILE = join(HERE, "data", "coco-subset.json");
const MODEL_PATH = join(ROOT, "public", "models", "mobilenet-v1-1.0-224", "model.json");
const OUT_DIR = "/tmp/belo-eval-real";

const SIZE = 224;
const CELLS = 49;
const DEPTH = 1024;

const ANNOTATION_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip";
const IMAGE_BASE = "http://images.cocodataset.org/val2017";

// ---------- What counts as a usable photograph ----------
//
// The detector reads a 7x7 grid with anchors covering a fifth and a half of the
// frame. Objects much below that cannot be placed however well the head is
// trained, so a dataset full of them would measure the grid resolution rather
// than the detector. These bounds pick photographs where the question is fair.

/** Smallest share of the cropped frame a box may cover and still be counted. */
const MIN_AREA = 0.15;
/** Any instance of a chosen class at least this large must be labelled, or the photo is dropped. */
const MIN_LABELABLE_AREA = 0.05;
/** A box this large is the whole picture and teaches the head nothing about place. */
const MAX_EDGE = 0.92;
/** Share of a box that has to survive the square crop for it to stay honest. */
const MIN_KEPT = 0.6;
/** More objects than a 7x7 grid can separate. */
const MAX_BOXES = 4;

const args = process.argv.slice(2);
const flag = (name) => args.includes(`--${name}`);
const value = (name, fallback) => {
  const at = args.indexOf(`--${name}`);
  return at >= 0 && args[at + 1] ? args[at + 1] : fallback;
};

const REBUILD = flag("rebuild");
const AUGMENT = !flag("no-augment");
const EPOCHS = Number(value("epochs", 0)) || 0;
const PATIENCE = Number(value("patience", 0)) || 0;
const RENDERS = Number(value("renders", 8));

function die(message) {
  console.error(`\n${message}`);
  process.exit(1);
}

// ---------- Rebuilding the committed subset ----------

function unzipAnnotations() {
  mkdirSync(CACHE, { recursive: true });
  const zip = join(CACHE, "annotations_trainval2017.zip");
  const json = join(CACHE, "instances_val2017.json");
  if (existsSync(json)) return json;
  if (!existsSync(zip)) {
    console.log(`downloading ${ANNOTATION_URL} (about 240 MB, once)`);
    execFileSync("curl", ["-sSL", "-C", "-", "-o", zip, ANNOTATION_URL], { stdio: "inherit" });
  }
  console.log("extracting instances_val2017.json");
  const data = execFileSync("unzip", ["-p", zip, "annotations/instances_val2017.json"], {
    maxBuffer: 1024 * 1024 * 1024,
  });
  writeFileSync(json, data);
  return json;
}

/** Center crop rectangle of an image, in source pixels. */
function cropOf(width, height) {
  const side = Math.min(width, height);
  return { x: (width - side) / 2, y: (height - side) / 2, side };
}

/**
 * A COCO bbox in source pixels, moved into the cropped square and normalized.
 * Returns the share of the original box that survived the crop alongside it,
 * because a box cut in half by the crop is a wrong box, not a small one.
 */
function cropBox(bbox, crop) {
  const [bx, by, bw, bh] = bbox;
  const x0 = Math.max(bx, crop.x);
  const y0 = Math.max(by, crop.y);
  const x1 = Math.min(bx + bw, crop.x + crop.side);
  const y1 = Math.min(by + bh, crop.y + crop.side);
  const w = Math.max(0, x1 - x0);
  const h = Math.max(0, y1 - y0);
  const kept = bw * bh <= 0 ? 0 : (w * h) / (bw * bh);
  return {
    kept,
    rect: {
      x: (x0 - crop.x) / crop.side,
      y: (y0 - crop.y) / crop.side,
      w: w / crop.side,
      h: h / crop.side,
    },
  };
}

function rebuildSubset() {
  const json = unzipAnnotations();
  console.log("reading annotations");
  const coco = JSON.parse(readFileSync(json, "utf8"));
  const byId = new Map(coco.images.map((image) => [image.id, image]));
  const categories = new Map(coco.categories.map((c) => [c.id, c.name]));

  const annotationsByImage = new Map();
  for (const ann of coco.annotations) {
    const list = annotationsByImage.get(ann.image_id) ?? [];
    list.push(ann);
    annotationsByImage.set(ann.image_id, list);
  }

  /**
   * One photograph judged against a set of class names. Returns the boxes to
   * draw, or null when the photograph cannot be labelled honestly for that set.
   *
   * Honestly is the whole point of the function. Every instance of every chosen
   * class has to end up as a box: a cat left out because it is small is a cat
   * the detector gets punished for finding, and a score measured that way says
   * nothing. So a photograph containing an instance that cannot be labelled is
   * dropped rather than labelled partially.
   */
  function boxesFor(imageId, names) {
    const image = byId.get(imageId);
    if (!image) return null;
    const crop = cropOf(image.width, image.height);
    if (!(crop.side > 0)) return null;
    const mine = (annotationsByImage.get(imageId) ?? []).filter((ann) =>
      names.has(categories.get(ann.category_id)),
    );
    if (mine.length === 0 || mine.length > MAX_BOXES) return null;

    const boxes = [];
    let big = 0;
    for (const ann of mine) {
      if (ann.iscrowd) return null;
      const { kept, rect } = cropBox(ann.bbox, crop);
      const area = rect.w * rect.h;
      if (area < MIN_LABELABLE_AREA) return null;
      if (kept < MIN_KEPT || rect.w > MAX_EDGE || rect.h > MAX_EDGE) return null;
      if (area >= MIN_AREA) big += 1;
      boxes.push({
        labelId: categories.get(ann.category_id),
        x: Number(rect.x.toFixed(5)),
        y: Number(rect.y.toFixed(5)),
        w: Number(rect.w.toFixed(5)),
        h: Number(rect.h.toFixed(5)),
      });
    }
    if (big === 0) return null;
    return {
      id: imageId,
      file: image.file_name,
      width: image.width,
      height: image.height,
      boxes,
    };
  }

  // Classes are ranked by how many photographs they carry on their own, and the
  // top three are taken. Picking them by hand would be picking the answer.
  const ranked = [...categories.values()]
    .map((name) => {
      const names = new Set([name]);
      let count = 0;
      for (const imageId of annotationsByImage.keys()) {
        if (boxesFor(imageId, names)) count += 1;
      }
      return { name, count };
    })
    .sort((a, b) => b.count - a.count);
  console.log("\nclasses with the most usable large instances:");
  for (const entry of ranked.slice(0, 12)) {
    console.log(`  ${entry.name.padEnd(16)} ${entry.count}`);
  }

  const classes = ranked.slice(0, 3).map((entry) => entry.name);
  const names = new Set(classes);
  if (ranked[2].count < 40) die("Not enough classes with large instances. Loosen the bounds.");

  // Second pass over the photographs, this time judging each one against all
  // three classes at once, so a photograph with a person and a dog in it either
  // carries both boxes or is dropped.
  const perClass = new Map(classes.map((name) => [name, []]));
  for (const imageId of annotationsByImage.keys()) {
    const entry = boxesFor(imageId, names);
    if (!entry) continue;
    // Filed under whichever chosen class covers most of the frame, purely so
    // the per class cap below balances the set.
    const dominant = entry.boxes.reduce((best, box) =>
      box.w * box.h > best.w * best.h ? box : best,
    ).labelId;
    perClass.get(dominant).push(entry);
  }

  const PER_CLASS = 60;
  const images = classes.flatMap((name) =>
    perClass
      .get(name)
      .sort((a, b) => a.id - b.id)
      .slice(0, PER_CLASS),
  );
  images.sort((a, b) => a.id - b.id);
  const counts = new Map(classes.map((name) => [name, 0]));
  for (const image of images) {
    for (const box of image.boxes) counts.set(box.labelId, counts.get(box.labelId) + 1);
  }
  console.log("\nboxes in the subset:");
  for (const [name, count] of counts) console.log(`  ${name.padEnd(16)} ${count}`);

  const subset = {
    source: "COCO val2017, images.cocodataset.org, CC BY 4.0 annotations",
    builtBy: "npm run eval:real -- --rebuild",
    bounds: { MIN_AREA, MIN_LABELABLE_AREA, MAX_EDGE, MIN_KEPT, MAX_BOXES, PER_CLASS },
    classes,
    images,
  };
  mkdirSync(dirname(SUBSET_FILE), { recursive: true });
  writeFileSync(SUBSET_FILE, `${JSON.stringify(subset, null, 1)}\n`);
  console.log(
    `\nwrote ${SUBSET_FILE}: ${images.length} photographs across ${subset.classes.join(", ")}`,
  );
  return subset;
}

// ---------- Getting the photographs ----------

function downloadImages(subset) {
  mkdirSync(IMAGE_DIR, { recursive: true });
  const missing = subset.images.filter((image) => {
    const path = join(IMAGE_DIR, image.file);
    return !existsSync(path) || statSync(path).size < 1024;
  });
  if (missing.length === 0) return;
  console.log(`downloading ${missing.length} photographs from ${IMAGE_BASE}`);
  // curl in one process per batch, with the URL list on stdin, so a run that is
  // interrupted picks up exactly where it stopped.
  const batch = 25;
  for (let start = 0; start < missing.length; start += batch) {
    const slice = missing.slice(start, start + batch);
    const argv = ["-sS", "--fail", "--retry", "2", "--parallel"];
    for (const image of slice) {
      argv.push("-o", join(IMAGE_DIR, image.file), `${IMAGE_BASE}/${image.file}`);
    }
    execFileSync("curl", argv, { stdio: "inherit" });
    process.stdout.write(`  ${Math.min(start + batch, missing.length)}/${missing.length}\r`);
  }
  console.log("");
}

/** One photograph as RGB bytes, center cropped to a square at source resolution. */
function croppedPixels(image) {
  const raw = jpeg.decode(readFileSync(join(IMAGE_DIR, image.file)), { useTArray: true });
  const crop = cropOf(raw.width, raw.height);
  const side = Math.round(crop.side);
  const sx = Math.round(crop.x);
  const sy = Math.round(crop.y);
  const out = new Uint8Array(side * side * 3);
  for (let y = 0; y < side; y += 1) {
    for (let x = 0; x < side; x += 1) {
      const from = ((sy + y) * raw.width + (sx + x)) * 4;
      const to = (y * side + x) * 3;
      out[to] = raw.data[from];
      out[to + 1] = raw.data[from + 1];
      out[to + 2] = raw.data[from + 2];
    }
  }
  return { pixels: out, side };
}

// ---------- The app's own code ----------

const { detect, evaluateDetector, trainDetector } = await import(
  new URL("../src/lib/ml/detector.ts", import.meta.url)
);
const { iou, matchDetections } = await import(new URL("../src/lib/detect.ts", import.meta.url));
const { DETECT_SETTINGS } = await import(new URL("../src/lib/types.ts", import.meta.url));
const { augmentFor, flipBoxes, NO_AUGMENT } = await import(
  new URL("../src/lib/augment.ts", import.meta.url)
);
const { applyAugment } = await import(new URL("../src/lib/ml/features.ts", import.meta.url));

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

/** The 224 pixel view of one cropped photograph, exactly as ml/features.ts builds it. */
function resized(pixels, side, aug = NO_AUGMENT) {
  return tf.tidy(() => {
    const source = tf.tensor3d(pixels, [side, side, 3], "int32");
    const scaled =
      side === SIZE ? source.toFloat() : tf.image.resizeBilinear(source, [SIZE, SIZE], true);
    return applyAugment(scaled, aug);
  });
}

function featuresOf(extractor, pixels, side, aug) {
  return tf.tidy(() => {
    const view = resized(pixels, side, aug);
    const input = view.div(127.5).sub(1).expandDims(0);
    return new Float32Array(extractor.predict(input).reshape([CELLS * DEPTH]).dataSync());
  });
}

/** Cached on disk: one MobileNet pass per photograph per variant, ever. */
function cachedFeatures(extractor, image, variant) {
  mkdirSync(FEATURE_DIR, { recursive: true });
  const path = join(FEATURE_DIR, `${image.id}-v${variant}.bin`);
  if (existsSync(path)) {
    const buffer = readFileSync(path);
    if (buffer.byteLength === CELLS * DEPTH * 4) {
      return new Float32Array(buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength));
    }
  }
  const { pixels, side } = croppedPixels(image);
  const features = featuresOf(extractor, pixels, side, augmentFor(String(image.id), variant));
  writeFileSync(path, Buffer.from(features.buffer));
  return features;
}

// ---------- A tiny PNG writer, for the sample renders ----------

const crcTable = Array.from({ length: 256 }, (_, n) => {
  let c = n;
  for (let k = 0; k < 8; k += 1) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
  return c >>> 0;
});
const crc32 = (buf) => {
  let c = 0xffffffff;
  for (const b of buf) c = crcTable[(c ^ b) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
};
function chunk(type, body) {
  const len = Buffer.alloc(4);
  len.writeUInt32BE(body.length);
  const full = Buffer.concat([Buffer.from(type, "ascii"), body]);
  const crc = Buffer.alloc(4);
  crc.writeUInt32BE(crc32(full));
  return Buffer.concat([len, full, crc]);
}
function encodePng(rgb, width, height) {
  const raw = Buffer.alloc((width * 3 + 1) * height);
  for (let y = 0; y < height; y += 1) {
    raw[y * (width * 3 + 1)] = 0;
    Buffer.from(rgb.buffer, rgb.byteOffset + y * width * 3, width * 3).copy(
      raw,
      y * (width * 3 + 1) + 1,
    );
  }
  const head = Buffer.alloc(13);
  head.writeUInt32BE(width, 0);
  head.writeUInt32BE(height, 4);
  head[8] = 8;
  head[9] = 2;
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk("IHDR", head),
    chunk("IDAT", deflateSync(raw, { level: 9 })),
    chunk("IEND", Buffer.alloc(0)),
  ]);
}

function drawRect(rgb, width, height, rect, color, thickness = 3) {
  const x0 = Math.max(0, Math.round(rect.x * SIZE));
  const y0 = Math.max(0, Math.round(rect.y * SIZE));
  const x1 = Math.min(SIZE - 1, Math.round((rect.x + rect.w) * SIZE));
  const y1 = Math.min(SIZE - 1, Math.round((rect.y + rect.h) * SIZE));
  const put = (x, y, ox, oy) => {
    const px = x + ox;
    const py = y + oy;
    if (px < 0 || py < 0 || px >= width || py >= height) return;
    const at = (py * width + px) * 3;
    rgb[at] = color[0];
    rgb[at + 1] = color[1];
    rgb[at + 2] = color[2];
  };
  return (offsetX, offsetY) => {
    for (let t = 0; t < thickness; t += 1) {
      for (let x = x0; x <= x1; x += 1) {
        put(x, y0 + t, offsetX, offsetY);
        put(x, y1 - t, offsetX, offsetY);
      }
      for (let y = y0; y <= y1; y += 1) {
        put(x0 + t, y, offsetX, offsetY);
        put(x1 - t, y, offsetX, offsetY);
      }
    }
  };
}

/**
 * One photograph rendered twice: the human boxes on the left, whatever the
 * detector read on the right. No text, because a PNG writer that can do letters
 * is a bigger program than this whole script; the file name carries the verdict.
 */
function renderComparison(image, truth, predicted) {
  const { pixels, side } = croppedPixels(image);
  const view = tf.tidy(() => resized(pixels, side).toInt().dataSync());
  const gap = 8;
  const width = SIZE * 2 + gap * 3;
  const height = SIZE + gap * 2;
  const rgb = new Uint8Array(width * height * 3).fill(24);
  const blit = (offsetX) => {
    for (let y = 0; y < SIZE; y += 1) {
      for (let x = 0; x < SIZE; x += 1) {
        const from = (y * SIZE + x) * 3;
        const to = ((y + gap) * width + x + offsetX) * 3;
        rgb[to] = view[from];
        rgb[to + 1] = view[from + 1];
        rgb[to + 2] = view[from + 2];
      }
    }
  };
  blit(gap);
  blit(gap * 2 + SIZE);
  for (const box of truth) drawRect(rgb, width, height, box, [23, 194, 106])(gap, gap);
  for (const box of predicted) {
    drawRect(rgb, width, height, box, [255, 92, 138])(gap * 2 + SIZE, gap);
  }
  return encodePng(rgb, width, height);
}

// ---------- Run ----------

const subset = REBUILD || !existsSync(SUBSET_FILE)
  ? rebuildSubset()
  : JSON.parse(readFileSync(SUBSET_FILE, "utf8"));
if (REBUILD && args.length === 1) process.exit(0);

const LABELS = subset.classes;
console.log(
  `\nBelo detector on real photographs\n` +
    `  source     COCO val2017, ${subset.images.length} photographs, ${LABELS.join(", ")}\n` +
    `  bounds     every object covers at least ${(MIN_AREA * 100).toFixed(0)}% of the cropped frame\n` +
    `  augment    ${AUGMENT ? "on, one mirrored and relit copy per training photograph" : "off"}`,
);

downloadImages(subset);

const require = createRequire(import.meta.url);
setWasmPaths(
  `${dirname(require.resolve("@tensorflow/tfjs-backend-wasm/dist/tf-backend-wasm.js"))}/`,
);
await tf.setBackend("wasm");
await tf.ready();

const full = await tf.loadLayersModel(fileHandler(MODEL_PATH));
const extractor = tf.model({
  inputs: full.inputs,
  outputs: full.getLayer("conv_pw_13_relu").output,
});

// Every fifth photograph is held out, exactly the way the store splits a detect
// project, so the split is the same on every run and the two halves are drawn
// from the same shuffle of the whole set rather than from one class each.
function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const shuffle = mulberry32(20260729);
const order = subset.images
  .map((image) => ({ image, key: shuffle() }))
  .sort((a, b) => a.key - b.key)
  .map((entry) => entry.image);
const holdoutImages = order.filter((_, i) => i % 5 === 4);
const trainImages = order.filter((_, i) => i % 5 !== 4);

const readStarted = Date.now();
let read = 0;
const sampleOf = (image, variant) => {
  const aug = augmentFor(String(image.id), variant);
  read += 1;
  if (read % 25 === 0) process.stdout.write(`  read ${read} views\r`);
  return {
    imageId: variant === 0 ? String(image.id) : `${image.id}#${variant}`,
    features: cachedFeatures(extractor, image, variant),
    boxes: aug.flip ? flipBoxes(image.boxes) : image.boxes,
  };
};

const training = [];
for (const image of trainImages) {
  training.push(sampleOf(image, 0));
  if (AUGMENT) training.push(sampleOf(image, 1));
}
const holdout = holdoutImages.map((image) => sampleOf(image, 0));
console.log(
  `\nread ${read} views in ${((Date.now() - readStarted) / 1000).toFixed(1)}s ` +
    `(${trainImages.length} training photographs, ${holdoutImages.length} held out, ` +
    `${training.length} training examples after mirroring)`,
);

Math.random = mulberry32(913);
const settings = { ...DETECT_SETTINGS, augment: AUGMENT, ...(EPOCHS ? { epochs: EPOCHS } : {}) };
const trainStarted = Date.now();
const run = await trainDetector({
  samples: training,
  // --keep-last is how the old behaviour is measured against the new one: with
  // no holdout handed to the loop there is no early stop and no restore, so the
  // run keeps the weights it happens to end on.
  validation: flag("keep-last") ? undefined : holdout,
  labelIds: LABELS,
  settings,
  ...(PATIENCE ? { patience: PATIENCE } : {}),
  onProgress: (epoch) => {
    if (epoch.epoch % 10 === 0 || epoch.epoch === 1) {
      console.log(
        `epoch ${String(epoch.epoch).padStart(3)}  loss ${epoch.loss.toFixed(3)}  ` +
          `cls ${epoch.clsLoss.toFixed(3)}  loc ${epoch.locLoss.toFixed(3)}  ` +
          `val ${epoch.valLoss === null ? "--" : epoch.valLoss.toFixed(3)}`,
      );
    }
  },
});
const { detector } = run;
console.log(
  `trained ${run.history.length} of ${settings.epochs} epochs in ` +
    `${((Date.now() - trainStarted) / 1000).toFixed(1)}s, kept epoch ${run.bestEpoch ?? run.history.length}` +
    `${run.stoppedEarly ? " after patience ran out" : ""}`,
);

function score(set, options = {}) {
  const stats = new Map(LABELS.map((id) => [id, { truth: 0, found: 0, predicted: 0 }]));
  const overlaps = [];
  const perImage = [];
  for (const sample of set) {
    const detections = detect(detector, sample.features, options);
    const matched = matchDetections(sample.boxes, detections, 0.5);
    for (const box of sample.boxes) stats.get(box.labelId).truth += 1;
    detections.forEach((detection, i) => {
      const stat = stats.get(detection.labelId);
      if (!stat) return;
      stat.predicted += 1;
      if (matched.hits[i] < 0) return;
      stat.found += 1;
      overlaps.push(iou(sample.boxes[matched.hits[i]], detection));
    });
    perImage.push({ sample, detections, matched });
  }
  let truth = 0;
  let found = 0;
  let predicted = 0;
  for (const stat of stats.values()) {
    truth += stat.truth;
    found += stat.found;
    predicted += stat.predicted;
  }
  return {
    stats,
    perImage,
    recall: truth === 0 ? 0 : found / truth,
    precision: predicted === 0 ? 0 : found / predicted,
    meanIoU: overlaps.length === 0 ? 0 : overlaps.reduce((a, b) => a + b, 0) / overlaps.length,
    truth,
    found,
    predicted,
  };
}

const held = score(holdout);
const seen = score(training.filter((sample) => !sample.imageId.includes("#")));

console.log("\n================ held out real photographs, IoU 0.5 ================");
for (const [labelId, stat] of held.stats) {
  const recall = stat.truth === 0 ? 0 : stat.found / stat.truth;
  const precision = stat.predicted === 0 ? 0 : stat.found / stat.predicted;
  console.log(
    `  ${labelId.padEnd(14)} found ${(recall * 100).toFixed(0)}% (${stat.found}/${stat.truth})   ` +
      `right ${(precision * 100).toFixed(0)}% (${stat.found}/${stat.predicted})`,
  );
}
console.log(
  `  ${"overall".padEnd(14)} found ${(held.recall * 100).toFixed(0)}% (${held.found}/${held.truth})   ` +
    `right ${(held.precision * 100).toFixed(0)}% (${held.found}/${held.predicted})   ` +
    `mean overlap ${held.meanIoU.toFixed(3)}`,
);
console.log(
  `  ${"trained on".padEnd(14)} found ${(seen.recall * 100).toFixed(0)}%   ` +
    `right ${(seen.precision * 100).toFixed(0)}%   mean overlap ${seen.meanIoU.toFixed(3)}`,
);

console.log("\nheld out set by score threshold");
for (const threshold of [0.1, 0.2, 0.3, 0.5, 0.7]) {
  const sweep = score(holdout, { scoreThreshold: threshold });
  console.log(
    `  at ${threshold.toFixed(2)}  found ${(sweep.recall * 100).toFixed(0)}%  ` +
      `right ${(sweep.precision * 100).toFixed(0)}%  mean overlap ${sweep.meanIoU.toFixed(3)}`,
  );
}

const holdoutLoss = evaluateDetector(detector, holdout, settings.batchSize);
console.log(
  `\nheld out loss ${holdoutLoss?.toFixed(3)} against ${run.history[run.history.length - 1].loss.toFixed(3)} on the training set`,
);

// ---------- Sample renders ----------

// The renders are named after what went wrong in them, so a rerun that gets a
// different answer would otherwise leave the old verdict sitting next to the
// new one under a different name.
mkdirSync(OUT_DIR, { recursive: true });
for (const name of readdirSync(OUT_DIR)) {
  if (name.endsWith(".png")) rmSync(join(OUT_DIR, name));
}
const byId = new Map(subset.images.map((image) => [String(image.id), image]));
const interesting = [...held.perImage].sort((a, b) => {
  const badness = (entry) =>
    entry.matched.missed.length + entry.matched.hits.filter((hit) => hit < 0).length;
  return badness(b) - badness(a);
});
const picks = [
  ...interesting.slice(0, Math.ceil(RENDERS / 2)),
  ...interesting.slice(-Math.floor(RENDERS / 2)),
];
const written = [];
for (const entry of picks) {
  const image = byId.get(entry.sample.imageId);
  if (!image) continue;
  const missed = entry.matched.missed.length;
  const alarms = entry.matched.hits.filter((hit) => hit < 0).length;
  const tag = missed === 0 && alarms === 0 ? "clean" : `${missed}missed-${alarms}false`;
  const file = join(OUT_DIR, `${tag}-${image.id}.png`);
  writeFileSync(file, renderComparison(image, entry.sample.boxes, entry.detections));
  written.push(file);
}
console.log(`\nwrote ${written.length} side by side renders to ${OUT_DIR}`);
console.log("  left: the boxes a person drew. right: the boxes Belo read.");
for (const file of written) console.log(`  ${file}`);

console.log(
  `\nThis is a measurement, not a test. A low number here is a fact about a 7x7\n` +
    `grid trained on ${trainImages.length} photographs, not a broken build.`,
);
