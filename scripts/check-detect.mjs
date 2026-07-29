// Deterministic check of the detection decision layer.
//
// Anchors, overlap, offset coding, matching, mining, suppression and scoring
// are the parts of a detector that fail quietly: a wrong anchor order or a
// flipped encode still trains, it just never learns. All of it is pure
// arithmetic, so all of it runs here without TensorFlow and without a DOM.
//
//   node scripts/check-detect.mjs
//
// The proof that the whole thing actually learns is scripts/check-learning.mjs.
//
// Node 22.6 or newer runs the TypeScript source directly through type stripping.

const {
  ANCHORS_PER_CELL,
  ANCHOR_RATIOS,
  ANCHOR_SCALES,
  anchorRect,
  anchorShapes,
  buildAnchors,
  buildTargets,
  clipRect,
  decodeBox,
  decodeDetections,
  encodeBox,
  iou,
  matchAnchors,
  matchDetections,
  mineNegatives,
  NEG_IOU,
  nms,
  POS_IOU,
  standardizeCells,
  GRID_COLS,
  GRID_ROWS,
} = await import(new URL("../src/lib/detect.ts", import.meta.url));

// Flipping is geometry too, and getting it wrong is the classic silent bug: the
// pixels mirror, the boxes do not, and the head learns that the object is on
// the other side of the picture from where it is.
const { augmentFor, flipBoxes, flipRect, NO_AUGMENT, VARIANTS } = await import(
  new URL("../src/lib/augment.ts", import.meta.url)
);

let failures = 0;
const results = [];

function check(name, ok, detail = "") {
  results.push(`${ok ? "ok  " : "FAIL"} ${name}${detail ? ` — ${detail}` : ""}`);
  if (!ok) failures += 1;
}

const near = (a, b, eps = 1e-9) => Math.abs(a - b) < eps;

// ---------- Anchors ----------

const shapes = anchorShapes();
check(
  `there are ${ANCHORS_PER_CELL} anchor shapes per cell`,
  shapes.length === ANCHOR_SCALES.length * ANCHOR_RATIOS.length && shapes.length === ANCHORS_PER_CELL,
  `${shapes.length}`,
);
check(
  "every shape keeps the area of its scale",
  shapes.every((shape, i) => {
    const scale = ANCHOR_SCALES[Math.floor(i / ANCHOR_RATIOS.length)];
    return near(Math.sqrt(shape.w * shape.h), scale, 1e-12);
  }),
);
check(
  "the ratio set covers square, wide and tall",
  shapes.some((s) => near(s.w, s.h)) &&
    shapes.some((s) => s.w > s.h * 1.5) &&
    shapes.some((s) => s.h > s.w * 1.5),
);

const anchors = buildAnchors();
check(
  `the grid holds ${GRID_ROWS * GRID_COLS * ANCHORS_PER_CELL} anchors`,
  anchors.length === GRID_ROWS * GRID_COLS * ANCHORS_PER_CELL,
  `${anchors.length}`,
);
check(
  "anchors are laid out row major with the shapes innermost",
  anchors.every((anchor, i) => {
    const cell = Math.floor(i / ANCHORS_PER_CELL);
    const shape = shapes[i % ANCHORS_PER_CELL];
    const row = Math.floor(cell / GRID_COLS);
    const col = cell % GRID_COLS;
    return (
      near(anchor.cx, (col + 0.5) / GRID_COLS) &&
      near(anchor.cy, (row + 0.5) / GRID_ROWS) &&
      near(anchor.w, shape.w) &&
      near(anchor.h, shape.h)
    );
  }),
);
check(
  "every cell center sits inside the picture",
  anchors.every((a) => a.cx > 0 && a.cx < 1 && a.cy > 0 && a.cy < 1),
);
check(
  "an anchor rect is its center form turned inside out",
  (() => {
    const rect = anchorRect({ cx: 0.5, cy: 0.25, w: 0.4, h: 0.2 });
    return near(rect.x, 0.3) && near(rect.y, 0.15) && near(rect.w, 0.4) && near(rect.h, 0.2);
  })(),
);

// ---------- Overlap ----------

const unit = { x: 0, y: 0, w: 1, h: 1 };
check("a box fully overlaps itself", near(iou(unit, unit), 1));
check("boxes that miss each other score zero", iou(unit, { x: 2, y: 2, w: 1, h: 1 }) === 0);
check("touching edges score zero", iou(unit, { x: 1, y: 0, w: 1, h: 1 }) === 0);
check(
  "a half covered box scores a third",
  near(iou({ x: 0, y: 0, w: 1, h: 1 }, { x: 0.5, y: 0, w: 1, h: 1 }), 1 / 3),
);
check(
  "a box inside another scores the area ratio",
  near(iou(unit, { x: 0.25, y: 0.25, w: 0.5, h: 0.5 }), 0.25),
);

// ---------- Offset coding ----------

const roundTrips = [
  [{ x: 0.1, y: 0.1, w: 0.3, h: 0.2 }, { cx: 0.25, cy: 0.25, w: 0.22, h: 0.22 }],
  [{ x: 0.6, y: 0.02, w: 0.35, h: 0.9 }, { cx: 0.5, cy: 0.5, w: 0.5, h: 0.5 }],
  [{ x: 0, y: 0, w: 1, h: 1 }, { cx: 0.5, cy: 0.5, w: 0.35, h: 0.71 }],
];
check(
  "encode then decode gives the box back",
  roundTrips.every(([box, anchor]) => {
    const back = decodeBox(encodeBox(box, anchor), anchor);
    return (
      near(back.x, box.x, 1e-9) &&
      near(back.y, box.y, 1e-9) &&
      near(back.w, box.w, 1e-9) &&
      near(back.h, box.h, 1e-9)
    );
  }),
);
check(
  "a box sitting exactly on its anchor encodes to zeros",
  encodeBox({ x: 0.3, y: 0.4, w: 0.2, h: 0.1 }, { cx: 0.4, cy: 0.45, w: 0.2, h: 0.1 }).every((v) =>
    near(v, 0),
  ),
);
check(
  "decoding reads four numbers from anywhere in a flat batch",
  (() => {
    const anchor = { cx: 0.5, cy: 0.5, w: 0.4, h: 0.4 };
    const box = { x: 0.2, y: 0.3, w: 0.5, h: 0.2 };
    const flat = new Float32Array([9, 9, 9, 9, ...encodeBox(box, anchor)]);
    const back = decodeBox(flat, anchor, 4);
    return near(back.x, box.x, 1e-6) && near(back.w, box.w, 1e-6);
  })(),
);
check(
  "clipping pulls a box back inside without flipping it",
  (() => {
    const clipped = clipRect({ x: -0.2, y: 0.8, w: 0.5, h: 0.5 });
    return (
      near(clipped.x, 0) && near(clipped.w, 0.3) && near(clipped.y, 0.8) && near(clipped.h, 0.2)
    );
  })(),
);

// ---------- Matching ----------

const big = { x: 0.3, y: 0.3, w: 0.4, h: 0.4 };
const match = matchAnchors(anchors, [big]);
check("a well sized box finds positive anchors", match.positives > 0, `${match.positives}`);
check(
  `positives overlap the box at least ${POS_IOU}`,
  anchors.every(
    (anchor, i) =>
      match.state[i] !== 1 ||
      iou(anchorRect(anchor), big) >= POS_IOU ||
      // the forced best match is allowed to be the one exception
      match.truth[i] === 0,
  ),
);
check(
  `negatives overlap the box less than ${NEG_IOU}`,
  anchors.every((anchor, i) => match.state[i] !== 0 || iou(anchorRect(anchor), big) < NEG_IOU),
);
check(
  "anchors between the two thresholds are ignored",
  anchors.some((anchor, i) => {
    const overlap = iou(anchorRect(anchor), big);
    return match.state[i] === -1 && overlap >= NEG_IOU && overlap < POS_IOU;
  }),
);
check(
  "every anchor is exactly one of positive, negative or ignored",
  anchors.every((_, i) => [1, 0, -1].includes(match.state[i])),
);
check("a picture with no boxes has no positives", matchAnchors(anchors, []).positives === 0);
check(
  "a picture with no boxes is all background",
  matchAnchors(anchors, []).state.every((s) => s === 0),
);

// A sliver no anchor can cover at half overlap still has to be learnable.
const sliver = { x: 0.02, y: 0.45, w: 0.06, h: 0.06 };
const forced = matchAnchors(anchors, [sliver]);
check("a box no anchor covers is force matched anyway", forced.positives >= 1, `${forced.positives}`);
check(
  "the forced anchor is the best overlapping one",
  (() => {
    const picked = [...forced.state].findIndex((s) => s === 1);
    if (picked < 0) return false;
    const best = anchors.reduce(
      (top, anchor, i) => {
        const overlap = iou(anchorRect(anchor), sliver);
        return overlap > top.overlap ? { i, overlap } : top;
      },
      { i: -1, overlap: -1 },
    );
    return picked === best.i;
  })(),
);
check(
  "every box in a crowded picture gets at least one anchor",
  (() => {
    const boxes = [
      { x: 0.02, y: 0.02, w: 0.2, h: 0.3 },
      { x: 0.4, y: 0.1, w: 0.5, h: 0.2 },
      { x: 0.3, y: 0.55, w: 0.12, h: 0.4 },
      { x: 0.75, y: 0.7, w: 0.22, h: 0.22 },
    ];
    const crowd = matchAnchors(anchors, boxes);
    const answered = new Set(
      [...crowd.state].map((s, i) => (s === 1 ? crowd.truth[i] : -1)).filter((t) => t >= 0),
    );
    return boxes.every((_, b) => answered.has(b));
  })(),
);
check(
  "nearby small boxes cannot overwrite each other's forced anchor",
  (() => {
    const boxes = [
      { x: 0.1886881884, y: 0.5995697857, w: 0.0284250014, h: 0.0997248055 },
      { x: 0.1325712843, y: 0.6965668716, w: 0.0552986691, h: 0.0471796872 },
    ];
    const nearby = matchAnchors(anchors, boxes);
    const answered = new Set(
      [...nearby.state].map((s, i) => (s === 1 ? nearby.truth[i] : -1)).filter((t) => t >= 0),
    );
    return answered.has(0) && answered.has(1);
  })(),
);

// ---------- Targets ----------

const labelIds = ["lbl_a", "lbl_b"];
const targets = buildTargets(anchors, [{ ...big, labelId: "lbl_b" }], labelIds);
check(
  "a positive anchor carries the label index plus one",
  [...targets.state].every((s, i) => s !== 1 || targets.classes[i] === 2),
);
check(
  "background anchors carry class zero",
  [...targets.state].every((s, i) => s === 1 || targets.classes[i] === 0),
);
check(
  "a positive anchor's offsets decode back to the drawn box",
  (() => {
    const i = [...targets.state].findIndex((s) => s === 1);
    const back = decodeBox(targets.offsets, anchors[i], i * 4);
    return near(back.x, big.x, 1e-5) && near(back.h, big.h, 1e-5);
  })(),
);
check(
  "a box drawn with a deleted label is ignored",
  buildTargets(anchors, [{ ...big, labelId: "lbl_gone" }], labelIds).positives === 0,
);

// ---------- Standardizing ----------

check(
  "standardizing centers every cell on zero with a spread of one",
  (() => {
    const raw = new Float32Array(3 * 4);
    for (let i = 0; i < raw.length; i += 1) raw[i] = (i % 4) * (i < 4 ? 1 : i < 8 ? 10 : 0.1) + 5;
    const out = standardizeCells(raw, 3, 4);
    for (let cell = 0; cell < 3; cell += 1) {
      let mean = 0;
      for (let i = 0; i < 4; i += 1) mean += out[cell * 4 + i];
      mean /= 4;
      let spread = 0;
      for (let i = 0; i < 4; i += 1) spread += (out[cell * 4 + i] - mean) ** 2;
      if (!near(mean, 0, 1e-6) || !near(Math.sqrt(spread / 4), 1, 1e-6)) return false;
    }
    return true;
  })(),
);
check(
  "standardizing leaves a flat cell alone instead of dividing by nothing",
  [...standardizeCells(new Float32Array([2, 2, 2, 2]), 1, 4)].every((v) => near(v, 0)),
);
check(
  "standardizing keeps the order of a cell's values",
  (() => {
    const out = standardizeCells(new Float32Array([1, 4, 2, 9]), 1, 4);
    return out[3] > out[1] && out[1] > out[2] && out[2] > out[0];
  })(),
);

// ---------- Mining ----------

const state = new Int8Array([1, 0, 0, 0, 0, -1, 0, 1]);
const loss = [0, 5, 1, 9, 3, 100, 7, 0];
const mask = mineNegatives(loss, state, 3, 0);
check("both positives are always kept", mask[0] === 1 && mask[7] === 1);
check(
  "every negative is kept when two positives ask for more than there are",
  [...mask].filter((v) => v === 1).length === 7,
  `${[...mask].filter((v) => v === 1).length}`,
);
check(
  "three negatives are kept for one positive",
  [...mineNegatives([9, 8, 7, 6, 5, 4, 3, 2], new Int8Array([1, 0, 0, 0, 0, 0, 0, 0]), 3, 0)].filter(
    (v) => v === 1,
  ).length === 4,
);
const mask1 = mineNegatives(loss, new Int8Array([1, 0, 0, 0, 0, -1, 0, 0]), 3, 0);
check(
  "the hardest negatives are the ones kept",
  mask1[3] === 1 && mask1[6] === 1 && mask1[1] === 1 && mask1[2] === 0 && mask1[4] === 0,
);
check("an ignored anchor is never mined", mask1[5] === 0);
check(
  "a picture with no positives still teaches from a floor of negatives",
  [...mineNegatives(loss, new Int8Array(8), 3, 4)].filter((v) => v === 1).length === 4,
);
check(
  "mining never asks for more negatives than there are",
  [...mineNegatives([1, 2], new Int8Array([1, 0]), 3, 8)].filter((v) => v === 1).length === 2,
);

// ---------- Suppression ----------

const overlapping = [
  { labelId: "a", score: 0.9, x: 0.1, y: 0.1, w: 0.4, h: 0.4 },
  { labelId: "a", score: 0.8, x: 0.12, y: 0.12, w: 0.4, h: 0.4 },
  { labelId: "a", score: 0.7, x: 0.6, y: 0.6, w: 0.3, h: 0.3 },
];
const suppressed = nms(overlapping, 0.45, 20);
check("suppression drops the duplicate", suppressed.length === 2, `${suppressed.length}`);
check("suppression keeps the highest score", near(suppressed[0].score, 0.9));
check("suppression keeps a box that stands on its own", near(suppressed[1].score, 0.7));
check("suppression respects the cap", nms(overlapping, 0.45, 1).length === 1);
check("suppression leaves the input alone", overlapping.length === 3 && overlapping[0].score === 0.9);

// ---------- Decoding ----------

function fakeHead(rows) {
  const classes = labelIds.length + 1;
  const scores = new Float32Array(anchors.length * classes);
  const offsets = new Float32Array(anchors.length * 4);
  for (let a = 0; a < anchors.length; a += 1) scores[a * classes] = 1;
  for (const [index, label, score, box] of rows) {
    scores[index * classes] = 1 - score;
    scores[index * classes + label] = score;
    offsets.set(encodeBox(box, anchors[index]), index * 4);
  }
  return { scores, offsets };
}

const wanted = { x: 0.25, y: 0.3, w: 0.3, h: 0.25 };
const twin = { x: 0.26, y: 0.31, w: 0.3, h: 0.25 };
const other = { x: 0.6, y: 0.6, w: 0.3, h: 0.3 };
const quiet = { x: 0.05, y: 0.75, w: 0.2, h: 0.2 };
const head = fakeHead([
  [40, 2, 0.92, wanted],
  [41, 2, 0.71, twin],
  [180, 1, 0.65, other],
  [250, 1, 0.05, quiet],
]);
const decoded = decodeDetections(head.scores, head.offsets, anchors, labelIds);
check("decoding drops anything under the threshold", decoded.length === 2, `${decoded.length}`);
check("decoding suppresses the near duplicate", decoded.filter((d) => d.labelId === "lbl_b").length === 1);
check(
  "decoding returns the box that was encoded",
  near(decoded[0].x, wanted.x, 1e-5) && near(decoded[0].w, wanted.w, 1e-5),
);
check("decoding names the winning label", decoded[0].labelId === "lbl_b");
check("decoding sorts by score", decoded[0].score > decoded[1].score);
check(
  "suppression is per label, so two labels can sit on the same spot",
  (() => {
    const stacked = fakeHead([
      [40, 2, 0.9, wanted],
      [41, 1, 0.8, wanted],
    ]);
    return decodeDetections(stacked.scores, stacked.offsets, anchors, labelIds).length === 2;
  })(),
);
check(
  "a lower threshold lets the quiet detection through",
  decodeDetections(head.scores, head.offsets, anchors, labelIds, { scoreThreshold: 0.04 }).length === 3,
);

// ---------- Scoring ----------

const truth = [
  { labelId: "lbl_a", x: 0.1, y: 0.1, w: 0.3, h: 0.3 },
  { labelId: "lbl_b", x: 0.6, y: 0.6, w: 0.2, h: 0.2 },
];
const scored = matchDetections(truth, [
  { labelId: "lbl_a", score: 0.9, x: 0.11, y: 0.11, w: 0.3, h: 0.3 },
  { labelId: "lbl_a", score: 0.5, x: 0.12, y: 0.12, w: 0.3, h: 0.3 },
  { labelId: "lbl_a", score: 0.8, x: 0.6, y: 0.6, w: 0.2, h: 0.2 },
]);
check("a good detection hits its box", scored.hits[0] === 0);
check("a second detection on the same box is a false alarm", scored.hits[1] === -1);
check("a detection with the wrong label hits nothing", scored.hits[2] === -1);
check("the box nothing found is reported missed", scored.missed.join(",") === "1");
check(
  "a loose detection under the overlap threshold does not count",
  matchDetections(truth, [{ labelId: "lbl_a", score: 1, x: 0.3, y: 0.3, w: 0.3, h: 0.3 }]).hits[0] ===
    -1,
);
check(
  "the highest scoring detection claims the box",
  (() => {
    const many = matchDetections(truth, [
      { labelId: "lbl_a", score: 0.4, x: 0.1, y: 0.1, w: 0.3, h: 0.3 },
      { labelId: "lbl_a", score: 0.95, x: 0.1, y: 0.1, w: 0.3, h: 0.3 },
    ]);
    return many.hits[1] === 0 && many.hits[0] === -1;
  })(),
);

// ---------- Flip geometry ----------

const leftBox = { x: 0.1, y: 0.3, w: 0.2, h: 0.4 };
const flipped = flipRect(leftBox);
check(
  "a box near the left edge lands the same distance from the right",
  near(flipped.x, 0.7) && near(flipped.w, 0.2),
  `x ${flipped.x}, w ${flipped.w}`,
);
check(
  "flipping changes nothing vertically",
  near(flipped.y, leftBox.y) && near(flipped.h, leftBox.h),
);
check(
  "flipping twice gives back the original box",
  (() => {
    const back = flipRect(flipped);
    return near(back.x, leftBox.x) && near(back.w, leftBox.w);
  })(),
);
check(
  "a centered box is its own mirror",
  (() => {
    const centered = flipRect({ x: 0.4, y: 0.1, w: 0.2, h: 0.2 });
    return near(centered.x, 0.4) && near(centered.w, 0.2);
  })(),
);
check(
  "a flipped box stays inside the picture",
  [
    { x: 0, y: 0, w: 1, h: 1 },
    { x: 0, y: 0.5, w: 0.05, h: 0.05 },
    { x: 0.95, y: 0.5, w: 0.05, h: 0.05 },
  ].every((rect) => {
    const out = flipRect(rect);
    return out.x >= -1e-12 && out.x + out.w <= 1 + 1e-12;
  }),
);
check(
  "flipping a list keeps every label on its own box",
  (() => {
    const boxes = flipBoxes([
      { labelId: "lbl_a", x: 0.1, y: 0.1, w: 0.2, h: 0.2 },
      { labelId: "lbl_b", x: 0.6, y: 0.1, w: 0.1, h: 0.3 },
    ]);
    return (
      boxes[0].labelId === "lbl_a" &&
      near(boxes[0].x, 0.7) &&
      boxes[1].labelId === "lbl_b" &&
      near(boxes[1].x, 0.3)
    );
  })(),
);
check(
  "overlap is preserved by a flip, so anchor matching sees the same picture",
  (() => {
    const a = { x: 0.1, y: 0.2, w: 0.3, h: 0.3 };
    const b = { x: 0.2, y: 0.25, w: 0.3, h: 0.3 };
    return near(iou(a, b), iou(flipRect(a), flipRect(b)), 1e-12);
  })(),
);

// ---------- Augmentation plan ----------

check("variant 0 is the picture as it was stored", augmentFor("img_1", 0) === NO_AUGMENT);
check(
  "an augmented variant is always mirrored",
  augmentFor("img_1", 1).flip && augmentFor("img_2", 1).flip,
);
check(
  "the same picture always gets the same variant",
  (() => {
    const a = augmentFor("img_seed", 1);
    const b = augmentFor("img_seed", 1);
    return a.brightness === b.brightness && a.saturation === b.saturation;
  })(),
);
check(
  "different pictures get different light",
  new Set(
    Array.from({ length: 20 }, (_, i) => augmentFor(`img_${i}`, 1).brightness.toFixed(6)),
  ).size > 15,
);
check(
  "the jitter stays inside bounds that keep a colour a colour",
  Array.from({ length: 200 }, (_, i) => augmentFor(`img_${i}`, 1)).every(
    (aug) =>
      aug.brightness >= 0.85 &&
      aug.brightness <= 1.18 &&
      aug.saturation >= 0.7 &&
      aug.saturation <= 1.3,
  ),
);
check(
  "one variant per picture, so a run holds twice the features and not five times",
  VARIANTS === 1,
  `${VARIANTS}`,
);

console.log(results.join("\n"));
console.log(failures === 0 ? "\nall detect checks passed" : `\n${failures} check(s) failed`);
process.exit(failures === 0 ? 0 : 1);
