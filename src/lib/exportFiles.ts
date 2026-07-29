// The three text files that make an exported model runnable somewhere else: a
// plain-language README, a page, and the inference code the page runs.
//
// The example is deliberately not an import of anything in src/. It is a second,
// independent implementation of the same arithmetic, written against the
// exported files alone, in plain JavaScript with the TensorFlow.js build from a
// pinned CDN URL. That is the whole point of the export: if the example can read
// the weights with nothing but the documented field meanings, so can a developer
// who never opens Belo's source. scripts/check-export.mjs closes that loop by
// running this file in a browser and comparing its numbers to Belo's own.

/** Pinned so an export made today still runs in a year. */
export const TFJS_CDN = "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js";

export const EXAMPLE_HTML = `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Belo exported model</title>
    <script src="${TFJS_CDN}"></script>
    <style>
      :root { color-scheme: light; }
      body {
        margin: 0;
        padding: 24px;
        font: 16px/1.5 ui-sans-serif, system-ui, sans-serif;
        color: #171717;
        background: #fdf8ee;
      }
      main { max-width: 760px; margin: 0 auto; }
      h1 { font-size: 28px; margin: 0 0 4px; }
      p { margin: 0 0 16px; }
      #drop {
        border: 3px solid #171717;
        border-radius: 10px;
        background: #fff;
        box-shadow: 6px 6px 0 #171717;
        padding: 28px;
        text-align: center;
        font-weight: 700;
      }
      #drop.over { background: #ffd83d; }
      #stage { position: relative; display: inline-block; margin-top: 20px; }
      #stage canvas { display: block; max-width: 100%; border: 3px solid #171717; }
      #out { margin-top: 20px; }
      .bar { height: 14px; background: #e7e0d2; border: 2px solid #171717; margin: 4px 0 10px; }
      .bar > span { display: block; height: 100%; background: #f473b9; }
      .row { display: flex; justify-content: space-between; font-weight: 700; }
      code { background: #efe8d9; padding: 1px 5px; border-radius: 4px; }
      #status { font-weight: 700; }
    </style>
  </head>
  <body>
    <main>
      <h1>Belo exported model</h1>
      <p id="status">Loading the feature extractor...</p>
      <div id="drop">Drop a picture here, or click to choose one</div>
      <input id="file" type="file" accept="image/*" hidden />
      <div id="stage"><canvas id="view" width="224" height="224"></canvas></div>
      <div id="out"></div>
      <p>
        Serve this folder over http, for example with <code>npx serve .</code>, and open the
        address it prints. Opening this file straight from disk does not work: the browser
        refuses to <code>fetch</code> the model files over <code>file://</code>.
      </p>
    </main>
    <script src="example.js"></script>
  </body>
</html>
`;

export const EXAMPLE_JS = String.raw`// Runs a model exported from Belo, using nothing but the files next to this one.
//
// Everything here is written from the field meanings in README.md: the weights
// are read as plain arrays, the arithmetic of the head is spelled out, and the
// boxes are decoded and suppressed by hand. There is no Belo code in this file.
//
// Serve the folder over http (npx serve .) and open the address. file:// cannot
// fetch the model, so it will not work.

const IMAGE_SIZE = 224;
const VAR_XY = 0.1;
const VAR_WH = 0.2;

const statusEl = document.getElementById("status");
const dropEl = document.getElementById("drop");
const fileEl = document.getElementById("file");
const viewEl = document.getElementById("view");
const outEl = document.getElementById("out");

function say(text) {
  if (statusEl) statusEl.textContent = text;
}

async function loadJson(path) {
  const response = await fetch(path, { cache: "no-store" });
  if (!response.ok) throw new Error(path + ": " + response.status);
  return response.json();
}

// --- the frozen feature extractor -------------------------------------------

// Belo cuts MobileNet v1 (1.0, 224) at conv_pw_13_relu, a 7x7x1024 grid, and
// averages it into one 1024 value embedding for classification. A detector also
// reads conv_pw_11_relu, 14x14x512, out of the same single pass.
async function loadExtractor(spec) {
  const full = await tf.loadLayersModel(spec.extractor.path);
  const cuts = spec.extractor.cuts.map((name) => full.getLayer(name).output);
  return tf.model({ inputs: full.inputs, outputs: cuts });
}

/**
 * The exact input Belo's own extractor sees: a center cropped square, resized to
 * 224, red green blue, then scaled from 0..255 into -1..1.
 */
function preprocess(source) {
  const width = source.naturalWidth || source.videoWidth || source.width;
  const height = source.naturalHeight || source.videoHeight || source.height;
  const side = Math.min(width, height);
  const canvas = document.createElement("canvas");
  canvas.width = IMAGE_SIZE;
  canvas.height = IMAGE_SIZE;
  const ctx = canvas.getContext("2d");
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(source, (width - side) / 2, (height - side) / 2, side, side, 0, 0, IMAGE_SIZE, IMAGE_SIZE);
  const input = tf.tidy(() =>
    tf.browser.fromPixels(canvas).toFloat().div(127.5).sub(1).expandDims(0),
  );
  return { input, canvas };
}

// --- plain arithmetic on the stored weights ---------------------------------

/** One stored weight tensor as a nested array, by its index in the weights list. */
function weightAt(stored, index) {
  const entry = stored.weights[index];
  if (!entry) throw new Error("the model file has no weight at index " + index);
  return { values: entry.values, shape: entry.shape };
}

/** rows x inner times inner x cols, plus an optional bias per column. */
function matmul(rows, inner, cols, xs, kernel, bias) {
  const out = new Float32Array(rows * cols);
  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < cols; c += 1) {
      let sum = bias ? bias[c] : 0;
      for (let k = 0; k < inner; k += 1) sum += xs[r * inner + k] * kernel[k * cols + c];
      out[r * cols + c] = sum;
    }
  }
  return out;
}

function relu(values) {
  const out = new Float32Array(values.length);
  for (let i = 0; i < values.length; i += 1) out[i] = values[i] > 0 ? values[i] : 0;
  return out;
}

function softmaxRow(values, at, count) {
  let max = -Infinity;
  for (let i = 0; i < count; i += 1) max = Math.max(max, values[at + i]);
  let sum = 0;
  const out = new Float32Array(count);
  for (let i = 0; i < count; i += 1) {
    out[i] = Math.exp(values[at + i] - max);
    sum += out[i];
  }
  for (let i = 0; i < count; i += 1) out[i] /= sum;
  return out;
}

/** Standardizes every grid cell on its own mean and spread, over its depth. */
function standardizeCells(values, cells, depth) {
  const out = new Float32Array(cells * depth);
  for (let cell = 0; cell < cells; cell += 1) {
    const at = cell * depth;
    let mean = 0;
    for (let i = 0; i < depth; i += 1) mean += values[at + i];
    mean /= depth;
    let variance = 0;
    for (let i = 0; i < depth; i += 1) {
      const centered = values[at + i] - mean;
      variance += centered * centered;
    }
    const spread = Math.sqrt(variance / depth) || 1;
    for (let i = 0; i < depth; i += 1) out[at + i] = (values[at + i] - mean) / spread;
  }
  return out;
}

// --- classification ---------------------------------------------------------

/**
 * The head is two dense layers. The first is 1024 to hiddenUnits with a relu,
 * the second hiddenUnits to one output per label. A single label model ends in a
 * softmax and carries no bias on the output layer; a multi label model ends in a
 * sigmoid and carries one.
 */
function classify(stored, pooled) {
  const labels = stored.labelIds.length;
  const hidden = stored.hiddenUnits;
  const w1 = weightAt(stored, 0);
  const b1 = weightAt(stored, 1);
  const w2 = weightAt(stored, 2);
  const multi = stored.mode === "multi";
  const b2 = multi ? weightAt(stored, 3) : null;

  const mixed = relu(matmul(1, w1.shape[0], hidden, pooled, w1.values, b1.values));
  const logits = matmul(1, hidden, labels, mixed, w2.values, b2 ? b2.values : null);
  if (multi) {
    const out = new Float32Array(labels);
    for (let i = 0; i < labels; i += 1) out[i] = 1 / (1 + Math.exp(-logits[i]));
    return out;
  }
  return softmaxRow(logits, 0, labels);
}

// --- detection --------------------------------------------------------------

function anchorShapes(scales, ratios) {
  const shapes = [];
  for (const scale of scales) {
    for (const ratio of ratios) {
      const root = Math.sqrt(ratio);
      shapes.push({ w: scale * root, h: scale / root });
    }
  }
  return shapes;
}

/** Cells row major, the anchors of one cell next to each other, coarse map first. */
function buildAnchors(featureScales) {
  const anchors = [];
  for (const scale of featureScales) {
    const shapes = anchorShapes(scale.scales, scale.ratios);
    for (let row = 0; row < scale.rows; row += 1) {
      for (let col = 0; col < scale.cols; col += 1) {
        for (const shape of shapes) {
          anchors.push({
            cx: (col + 0.5) / scale.cols,
            cy: (row + 0.5) / scale.rows,
            w: shape.w,
            h: shape.h,
          });
        }
      }
    }
  }
  return anchors;
}

function clipRect(rect) {
  const x0 = Math.max(0, Math.min(1, rect.x));
  const y0 = Math.max(0, Math.min(1, rect.y));
  const x1 = Math.max(0, Math.min(1, rect.x + rect.w));
  const y1 = Math.max(0, Math.min(1, rect.y + rect.h));
  return { x: x0, y: y0, w: Math.max(0, x1 - x0), h: Math.max(0, y1 - y0) };
}

function iou(a, b) {
  const x0 = Math.max(a.x, b.x);
  const y0 = Math.max(a.y, b.y);
  const x1 = Math.min(a.x + a.w, b.x + b.w);
  const y1 = Math.min(a.y + a.h, b.y + b.h);
  const overlap = Math.max(0, x1 - x0) * Math.max(0, y1 - y0);
  if (overlap <= 0) return 0;
  const union = a.w * a.h + b.w * b.h - overlap;
  return union <= 0 ? 0 : overlap / union;
}

/**
 * Runs the detection head over the packed feature maps and decodes boxes.
 *
 * The weight list is grouped by kind and then by map, coarse map first: every
 * mix layer, then every box layer, then every score layer, each one a kernel
 * followed by a bias.
 */
function detect(stored, maps, options) {
  const scoreThreshold = options.scoreThreshold;
  const nmsIoU = options.nmsIoU;
  const maxDetections = options.maxDetections;
  const featureScales = stored.featureScales || [
    { rows: stored.rows, cols: stored.cols, depth: stored.featureSize, scales: stored.scales, ratios: stored.ratios },
  ];
  const count = featureScales.length;
  const classes = stored.labelIds.length + 1;
  const anchors = buildAnchors(featureScales);

  const scores = [];
  const boxes = [];
  featureScales.forEach((scale, index) => {
    const cells = scale.rows * scale.cols;
    const perCell = anchorShapes(scale.scales, scale.ratios).length;
    const xs = standardizeCells(maps[index], cells, scale.depth);
    const mixKernel = weightAt(stored, index * 2);
    const mixBias = weightAt(stored, index * 2 + 1);
    const locKernel = weightAt(stored, count * 2 + index * 2);
    const locBias = weightAt(stored, count * 2 + index * 2 + 1);
    const clsKernel = weightAt(stored, count * 4 + index * 2);
    const clsBias = weightAt(stored, count * 4 + index * 2 + 1);
    const width = stored.headWidth;

    const mixed = relu(matmul(cells, scale.depth, width, xs, mixKernel.values, mixBias.values));
    const loc = matmul(cells, width, perCell * 4, mixed, locKernel.values, locBias.values);
    const cls = matmul(cells, width, perCell * classes, mixed, clsKernel.values, clsBias.values);
    for (let cell = 0; cell < cells; cell += 1) {
      for (let a = 0; a < perCell; a += 1) {
        boxes.push(loc.subarray(cell * perCell * 4 + a * 4, cell * perCell * 4 + a * 4 + 4));
        scores.push(softmaxRow(cls, cell * perCell * classes + a * classes, classes));
      }
    }
  });

  // Class 0 is background. A box is the anchor moved by the center offsets, in
  // anchor widths and heights, and resized by the exponent of the size offsets.
  const perLabel = new Map();
  for (let a = 0; a < anchors.length; a += 1) {
    let best = 1;
    for (let c = 2; c < classes; c += 1) if (scores[a][c] > scores[a][best]) best = c;
    const score = scores[a][best];
    if (!(score >= scoreThreshold)) continue;
    const anchor = anchors[a];
    const offsets = boxes[a];
    const cx = anchor.cx + offsets[0] * VAR_XY * anchor.w;
    const cy = anchor.cy + offsets[1] * VAR_XY * anchor.h;
    const w = anchor.w * Math.exp(offsets[2] * VAR_WH);
    const h = anchor.h * Math.exp(offsets[3] * VAR_WH);
    const rect = clipRect({ x: cx - w / 2, y: cy - h / 2, w: w, h: h });
    if (rect.w <= 0 || rect.h <= 0) continue;
    const labelId = stored.labelIds[best - 1];
    const list = perLabel.get(labelId) || [];
    list.push(Object.assign({ labelId: labelId, score: score }, rect));
    perLabel.set(labelId, list);
  }

  // Greedy suppression per label: keep the best, drop what overlaps it.
  const kept = [];
  for (const list of perLabel.values()) {
    const sorted = list.slice().sort((a, b) => b.score - a.score);
    const keep = [];
    for (const candidate of sorted) {
      if (keep.length >= maxDetections) break;
      if (keep.some((other) => iou(other, candidate) > nmsIoU)) continue;
      keep.push(candidate);
    }
    kept.push.apply(kept, keep);
  }
  return kept.sort((a, b) => b.score - a.score).slice(0, maxDetections);
}

// --- the page ---------------------------------------------------------------

const model = { spec: null, stored: null, extractor: null };

async function ready() {
  model.spec = await loadJson("belo-export.json");
  model.stored = await loadJson(model.spec.model.path);
  model.extractor = await loadExtractor(model.spec);
  say(
    'Ready: "' +
      model.spec.project +
      '", ' +
      model.spec.mode +
      " mode, " +
      model.spec.labels.length +
      " labels.",
  );
  return model.spec;
}

function labelName(labelId) {
  const found = model.spec.labels.find((label) => label.id === labelId);
  return found ? found.name : labelId;
}

/** One pass: the extractor, then the head. Options override the defaults. */
async function run(source, options) {
  const settings = Object.assign({}, model.spec.inference, options || {});
  const prepared = preprocess(source);
  try {
    const outputs = model.extractor.predict(prepared.input);
    const list = Array.isArray(outputs) ? outputs : [outputs];
    try {
      if (model.spec.mode === "detect") {
        const maps = list.map((tensor) => tensor.dataSync());
        const found = detect(model.stored, maps, settings);
        return { mode: "detect", canvas: prepared.canvas, detections: found };
      }
      // The classifier reads the mean of the 7x7 grid, one 1024 value vector.
      const pooledTensor = list[0].mean([1, 2]);
      const pooled = pooledTensor.dataSync();
      pooledTensor.dispose();
      const row = classify(model.stored, pooled);
      const scores = model.stored.labelIds.map((labelId, i) => ({
        labelId: labelId,
        name: labelName(labelId),
        score: row[i],
      }));
      return { mode: model.spec.mode, canvas: prepared.canvas, scores: scores };
    } finally {
      list.forEach((tensor) => tensor.dispose());
    }
  } finally {
    prepared.input.dispose();
  }
}

function draw(result) {
  const ctx = viewEl.getContext("2d");
  viewEl.width = result.canvas.width;
  viewEl.height = result.canvas.height;
  ctx.drawImage(result.canvas, 0, 0);
  if (result.mode !== "detect") return;
  ctx.lineWidth = 3;
  ctx.font = "bold 13px ui-sans-serif, system-ui, sans-serif";
  for (const found of result.detections) {
    const x = found.x * viewEl.width;
    const y = found.y * viewEl.height;
    const w = found.w * viewEl.width;
    const h = found.h * viewEl.height;
    ctx.strokeStyle = "#f473b9";
    ctx.strokeRect(x, y, w, h);
    const text = labelName(found.labelId) + " " + Math.round(found.score * 100) + "%";
    const width = ctx.measureText(text).width + 8;
    ctx.fillStyle = "#f473b9";
    ctx.fillRect(x, Math.max(0, y - 18), width, 18);
    ctx.fillStyle = "#171717";
    ctx.fillText(text, x + 4, Math.max(12, y - 5));
  }
}

function report(result) {
  if (result.mode === "detect") {
    outEl.innerHTML = result.detections.length
      ? result.detections
          .map(
            (found) =>
              '<div class="row"><span>' +
              labelName(found.labelId) +
              "</span><span>" +
              Math.round(found.score * 100) +
              "%</span></div>",
          )
          .join("")
      : "<p>Nothing above the score threshold.</p>";
    return;
  }
  outEl.innerHTML = result.scores
    .slice()
    .sort((a, b) => b.score - a.score)
    .map(
      (row) =>
        '<div class="row"><span>' +
        row.name +
        "</span><span>" +
        Math.round(row.score * 100) +
        '%</span></div><div class="bar"><span style="width:' +
        (row.score * 100).toFixed(1) +
        '%"></span></div>',
    )
    .join("");
}

async function handle(file) {
  if (!file) return;
  const url = URL.createObjectURL(file);
  try {
    const image = new Image();
    await new Promise((resolve, reject) => {
      image.onload = resolve;
      image.onerror = () => reject(new Error("could not decode that picture"));
      image.src = url;
    });
    const result = await run(image);
    draw(result);
    report(result);
  } catch (err) {
    say("Failed: " + err.message);
  } finally {
    URL.revokeObjectURL(url);
  }
}

if (dropEl) {
  dropEl.addEventListener("click", () => fileEl.click());
  fileEl.addEventListener("change", () => handle(fileEl.files[0]));
  ["dragenter", "dragover"].forEach((name) =>
    dropEl.addEventListener(name, (event) => {
      event.preventDefault();
      dropEl.classList.add("over");
    }),
  );
  ["dragleave", "drop"].forEach((name) =>
    dropEl.addEventListener(name, () => dropEl.classList.remove("over")),
  );
  dropEl.addEventListener("drop", (event) => {
    event.preventDefault();
    handle(event.dataTransfer.files[0]);
  });
}

// Also usable without a person: the check that proves this example agrees with
// Belo drives these three through the page.
window.beloExample = { ready: ready, run: run, model: model };

ready().catch((err) => {
  say("Failed to load the model: " + err.message);
  window.beloExample.error = err.message;
});
`;

export interface ReadmeInput {
  project: string;
  mode: string;
  appVersion: string;
  labels: { id: string; name: string }[];
  modelFile: string;
  extractorFiles: number;
  extractorBytes: number;
}

function labelTable(labels: { id: string; name: string }[]): string {
  return labels
    .map((label, index) => `| ${index} | \`${label.id}\` | ${label.name} |`)
    .join("\n");
}

const CLASSIFIER_FORMAT = `## \`classifier.json\`, the trained head

This is exactly the file Belo keeps inside a project at \`model/head.json\`. It is
the only part that was trained on your pictures.

\`\`\`jsonc
{
  "formatVersion": 1,
  "mode": "single",        // single: one winner, softmax. multi: one yes or no per label, sigmoid
  "labelIds": ["lbl_..."], // output order: score i belongs to labelIds[i]
  "hiddenUnits": 100,      // width of the one hidden layer
  "featureSize": 1024,     // length of the embedding it reads
  "weights": [{ "shape": [1024, 100], "values": [ ... ] }]
}
\`\`\`

\`weights\` is a list of plain row major arrays, in this order:

| index | shape | what it is |
| --- | --- | --- |
| 0 | \`[1024, hiddenUnits]\` | hidden layer weights |
| 1 | \`[hiddenUnits]\` | hidden layer bias |
| 2 | \`[hiddenUnits, labels]\` | output layer weights |
| 3 | \`[labels]\` | output layer bias, **multi mode only** |

A single label model has no output bias, which is why its list is three entries
and a multi label model's is four.

Inference is two matrix multiplies:

1. \`hidden = relu(embedding . W0 + b0)\`
2. \`logits = hidden . W2 (+ b3)\`
3. \`scores = softmax(logits)\` in single mode, \`sigmoid(logits)\` per label in multi mode.

In multi mode a label counts as yes at **0.5 or above**. In single mode the
answer is the highest score, and that score is the confidence.
`;

const DETECTOR_FORMAT = `## \`detector.json\`, the trained detection head

This is exactly the file Belo keeps inside a project at \`model/detect.json\`.

\`\`\`jsonc
{
  "formatVersion": 2,
  "labelIds": ["lbl_..."], // class c of the scores means labelIds[c - 1]
  "featureSize": 1024,
  "rows": 7, "cols": 7,               // the coarse map, kept for version 1 readers
  "scales": [0.58, 0.82], "ratios": [1, 2, 0.5],
  "featureScales": [                  // every map the head reads, coarse first
    { "rows": 7,  "cols": 7,  "depth": 1024, "scales": [0.58, 0.82], "ratios": [1, 2, 0.5] },
    { "rows": 14, "cols": 14, "depth": 512,  "scales": [0.3, 0.45],  "ratios": [1, 2, 0.5] }
  ],
  "headWidth": 96,
  "weights": [{ "shape": [1024, 96], "values": [ ... ] }]
}
\`\`\`

A version 1 file, written by Belo before 0.6.1, has no \`featureScales\` and
describes one 7x7 map with the four fields above it. Read it as a single map of
depth \`featureSize\` and everything below still applies.

\`weights\` is grouped by kind and then by map, coarse map first, each layer a
kernel followed by a bias. With \`M\` maps:

| index | shape | what it is |
| --- | --- | --- |
| \`2i\`, \`2i+1\` | \`[depth_i, headWidth]\`, \`[headWidth]\` | shared layer of map \`i\` |
| \`2M + 2i\`, \`2M + 2i + 1\` | \`[headWidth, anchors * 4]\`, \`[anchors * 4]\` | box offsets of map \`i\` |
| \`4M + 2i\`, \`4M + 2i + 1\` | \`[headWidth, anchors * classes]\`, \`[anchors * classes]\` | scores of map \`i\` |

\`anchors\` here is the anchors per cell, \`scales.length * ratios.length\`, and
\`classes\` is \`labelIds.length + 1\`.

### Anchors

Each map is a grid of cells. Every cell carries one anchor box per scale and
ratio pair, in this order: for each scale, for each ratio, a box of width
\`scale * sqrt(ratio)\` and height \`scale / sqrt(ratio)\`, as a fraction of the
frame, centered on the cell at \`((col + 0.5) / cols, (row + 0.5) / rows)\`.

The full anchor list is built map by map in the order \`featureScales\` gives, and
within a map cell by cell in row major order, anchors of one cell together.
Anchor index \`a\` is row \`a\` of the concatenated box and score outputs, which is
what makes the two lists line up.

### Reading one anchor

1. **Standardize the cell.** Before the head sees a cell, its values are
   standardized on that cell's own mean and standard deviation, over the map's
   depth. This has no parameters and is part of the contract: skip it and the
   scores are meaningless.
2. **Run the head.** \`mixed = relu(cells . Wmix + bmix)\`, then
   \`boxes = mixed . Wloc + bloc\` and \`logits = mixed . Wcls + bcls\`.
3. **Scores are a softmax** over \`classes\` values per anchor, with **background at
   index 0**. Class \`c\` above zero means \`labelIds[c - 1]\`.
4. **Boxes are four offsets** per anchor, in the standard SSD parameterization:

   \`\`\`
   cx = anchor.cx + o0 * 0.1 * anchor.w
   cy = anchor.cy + o1 * 0.1 * anchor.h
   w  = anchor.w * exp(o2 * 0.2)
   h  = anchor.h * exp(o3 * 0.2)
   \`\`\`

   0.1 and 0.2 are the variances the offsets were trained with. The result is a
   center and a size; the box Belo draws is \`x = cx - w / 2\`, \`y = cy - h / 2\`,
   clipped into 0..1.

### Coordinates and thresholds

Every coordinate is a fraction of the 224 pixel square the model saw, with
\`x\` and \`y\` at the **top left corner** and the origin at the top left of the
frame. Multiply by the width and height you drew the picture at.

An anchor counts as a detection when its best non-background score is **0.3 or
above**. Then greedy non maximum suppression runs **per label**: keep the highest
score, drop every remaining box of the same label overlapping it by more than
**0.15**, repeat, and keep at most **20** detections in all. Both numbers are in
\`belo-export.json\` under \`inference\`, and both are worth tuning for your case:
a lower score threshold finds more and invents more.
`;

export function exportReadme(input: ReadmeInput): string {
  const detect = input.mode === "detect";
  const megabytes = (input.extractorBytes / (1024 * 1024)).toFixed(1);
  return `# ${input.project}, a model exported from Belo

This folder is a **developer bundle**, not an app. It holds the model you trained
in Belo ${input.appVersion}, the frozen feature extractor it needs, and a small
example page that runs it in a browser. If you were looking for something to
install on a phone, this is not that: it is the model and the arithmetic around
it, in a form you can drop into your own project.

## Run the example

\`\`\`bash
npx serve .
\`\`\`

Then open the address it prints and drop a picture onto the page.

**Opening \`example.html\` straight from disk does not work.** The page has to
\`fetch\` the model files, and browsers refuse to do that over \`file://\`. Any
static server does the job; \`npx serve .\`, \`python3 -m http.server\` and the
"Live Server" editor extensions all work.

The example loads TensorFlow.js from a pinned CDN URL, so the first run needs
the internet. Everything else in this folder is local.

## What is in here

| File | What it is |
| --- | --- |
| \`belo-export.json\` | what this export is: mode, labels, input format, thresholds |
| \`${input.modelFile}\` | the part that was trained on your pictures |
| \`extractor/\` | the frozen MobileNet v1 (1.0, 224), ${input.extractorFiles} files, ${megabytes} MB |
| \`example.html\`, \`example.js\` | a working page that runs the model on a picture you drop |
| \`README.md\` | this file |

The extractor is included because the model is useless without it: the trained
part reads MobileNet's activations, not pixels. It is the same MobileNet that
ships inside Belo, unmodified, in the TensorFlow.js LayersModel format, so
\`tf.loadLayersModel('extractor/model.json')\` is all it takes to load.

## Labels

| output index | id | name |
| --- | --- | --- |
${labelTable(input.labels)}

The ids are what the model file carries; the names are yours. Nothing in the
model depends on the names, so you can rename them freely on your side.

## The input the model expects

Belo normalizes every picture the same way, on the way in and at inference time,
and a model fed anything else quietly gets worse:

1. **Center crop to a square.** The largest square in the middle of the picture;
   the edges of the long side are thrown away.
2. **Resize to 224 by 224.**
3. **Red, green, blue**, in that order.
4. **Scale into -1 to 1**: \`value / 127.5 - 1\` per channel, from the usual 0..255.

## The frozen extractor

MobileNet v1, width 1.0, 224 pixel input, truncated:

- **\`conv_pw_13_relu\`** is a 7 by 7 grid of 1024 values.${
    detect
      ? `\n- **\`conv_pw_11_relu\`** is a 14 by 14 grid of 512 values, one stage earlier.`
      : ""
  }
${
  detect
    ? `\nBoth come out of one forward pass, because the coarse cut is computed from
the fine one. The detection head reads them as one example, coarse map first.`
    : `\nThe classifier reads the **mean over the 49 cells**, which is one 1024 value
embedding per picture.`
}

${detect ? DETECTOR_FORMAT : CLASSIFIER_FORMAT}
## Reimplementing this without Belo

\`example.js\` is exactly that, and it is meant to be read: it loads the extractor
with TensorFlow.js, then does the head, the anchors, the decoding and the
suppression in plain JavaScript arithmetic on the arrays above, with no Belo code
in it at all. Everything the format needs is in this file and everything this
file describes is in that one, so porting to Python, Swift or Rust is a matter of
translating a page of arithmetic.

## What this model is and is not

It was trained on your pictures with a small head on a frozen extractor, which
is why it took seconds rather than hours. It is as good as the examples you gave
it and no better, it has never seen anything outside them, and it will answer
confidently about a picture of something it was never shown. Treat a score as a
hint and not as a fact.

Belo is MIT licensed. MobileNet v1's weights come from the TensorFlow.js model
zoo, Apache 2.0 licensed.
`;
}
