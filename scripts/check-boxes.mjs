// Deterministic check of the detect mode box seam.
//
// The annotator's geometry only shows itself through a pointer on a picture,
// and the manifest rules only show themselves through a save and a reopen.
// This exercises both in node: drawing in every direction, moving against the
// edges, all eight grips, and a round trip of version 1, 2 and 3 manifests.
//
//   node scripts/check-boxes.mjs
//
// Node 22.6 or newer runs the TypeScript source directly through type stripping.

const { advance, moved, positive, resized, HANDLES } = await import(
  new URL("../src/lib/boxes.ts", import.meta.url)
);
const {
  boxesOf,
  canRestoreModel,
  labelsOf,
  migrateManifest,
  normalizeRect,
  FORMAT_VERSION,
  MIN_BOX,
} = await import(new URL("../src/lib/types.ts", import.meta.url));

let failures = 0;
const results = [];

function check(name, ok, detail = "") {
  results.push(`${ok ? "ok  " : "FAIL"} ${name}${detail ? ` — ${detail}` : ""}`);
  if (!ok) failures += 1;
}

const near = (a, b) => Math.abs(a - b) < 1e-9;
const sameRect = (a, b) =>
  a !== null && b !== null && near(a.x, b.x) && near(a.y, b.y) && near(a.w, b.w) && near(a.h, b.h);
const rectText = (r) => (r === null ? "null" : `${r.x},${r.y} ${r.w}x${r.h}`);

// ---------- Drawing ----------

function drawn(from, to) {
  const drag = {
    kind: "draw",
    boxId: null,
    handle: null,
    origin: { x: from.x, y: from.y, w: 0, h: 0 },
    grab: from,
    rect: { x: from.x, y: from.y, w: 0, h: 0 },
  };
  return positive(advance(drag, to).rect);
}

const target = { x: 0.2, y: 0.3, w: 0.4, h: 0.3 };
for (const [name, from, to] of [
  ["down right", { x: 0.2, y: 0.3 }, { x: 0.6, y: 0.6 }],
  ["up left", { x: 0.6, y: 0.6 }, { x: 0.2, y: 0.3 }],
  ["down left", { x: 0.6, y: 0.3 }, { x: 0.2, y: 0.6 }],
  ["up right", { x: 0.2, y: 0.6 }, { x: 0.6, y: 0.3 }],
]) {
  const rect = drawn(from, to);
  check(`drawing ${name} gives the same box`, sameRect(rect, target), rectText(rect));
}

const tap = normalizeRect(drawn({ x: 0.5, y: 0.5 }, { x: 0.505, y: 0.508 }));
check("a tap sized drag is not a box", tap === null, rectText(tap));
const justBig = normalizeRect(drawn({ x: 0.5, y: 0.5 }, { x: 0.5 + MIN_BOX, y: 0.5 + MIN_BOX }));
check("a box exactly at the minimum survives", justBig !== null, rectText(justBig));

// ---------- Moving ----------

const box = { x: 0.4, y: 0.4, w: 0.2, h: 0.2 };
const grab = { x: 0.5, y: 0.5 };
check(
  "moving follows the pointer",
  sameRect(moved(box, grab, { x: 0.7, y: 0.3 }), { x: 0.6, y: 0.2, w: 0.2, h: 0.2 }),
);
check(
  "moving stops at the top left",
  sameRect(moved(box, grab, { x: 0, y: 0 }), { x: 0, y: 0, w: 0.2, h: 0.2 }),
);
check(
  "moving stops at the bottom right",
  sameRect(moved(box, grab, { x: 1, y: 1 }), { x: 0.8, y: 0.8, w: 0.2, h: 0.2 }),
);
check(
  "moving never changes the size",
  [
    { x: -2, y: 0.5 },
    { x: 3, y: -1 },
    { x: 0.5, y: 0.5 },
  ].every((point) => {
    const next = moved(box, grab, point);
    return near(next.w, box.w) && near(next.h, box.h);
  }),
);

// ---------- Resizing ----------

const point = { x: 0.1, y: 0.15 };
const expected = {
  nw: { x: 0.1, y: 0.15, w: 0.5, h: 0.45 },
  n: { x: 0.4, y: 0.15, w: 0.2, h: 0.45 },
  ne: { x: 0.4, y: 0.15, w: -0.3, h: 0.45 },
  e: { x: 0.4, y: 0.4, w: -0.3, h: 0.2 },
  se: { x: 0.4, y: 0.4, w: -0.3, h: -0.25 },
  s: { x: 0.4, y: 0.4, w: 0.2, h: -0.25 },
  sw: { x: 0.1, y: 0.4, w: 0.5, h: -0.25 },
  w: { x: 0.1, y: 0.4, w: 0.5, h: 0.2 },
};
for (const handle of HANDLES) {
  const rect = resized(box, handle, point);
  check(`grip ${handle} moves only its own edges`, sameRect(rect, expected[handle]), rectText(rect));
}
check(
  "dragging a grip past the far edge flips the box",
  sameRect(positive(resized(box, "se", { x: 0.1, y: 0.15 })), {
    x: 0.1,
    y: 0.15,
    w: 0.3,
    h: 0.25,
  }),
);

// ---------- Manifest round trips ----------

const settings = {
  epochs: 40,
  batchSize: 16,
  learningRate: 0.001,
  hiddenUnits: 100,
  validationSplit: 0.15,
};
const base = {
  format: "belo-project",
  appVersion: "0.4.3",
  name: "Check",
  createdAt: "2026-01-01T00:00:00.000Z",
  updatedAt: "2026-01-01T00:00:00.000Z",
  labels: [{ id: "lbl_a", name: "A", color: 0, createdAt: "" }],
  model: null,
  settings,
};

const v1 = {
  ...base,
  formatVersion: 1,
  images: [{ id: "i1", file: "i1.jpg", labelId: "lbl_a", source: "import", addedAt: "" }],
};
const m1 = migrateManifest(structuredClone(v1));
check("a version 1 manifest reads as single label", m1.mode === "single");
check("a version 1 image keeps its label", m1.images[0].labelId === "lbl_a");
check("a version 1 image gains no labelIds", m1.images[0].labelIds === undefined);
check("a version 1 image gains no boxes", !("boxes" in m1.images[0]));
check(`a version 1 manifest is written back as ${FORMAT_VERSION}`, m1.formatVersion === FORMAT_VERSION);

const v2 = {
  ...base,
  formatVersion: 2,
  mode: "multi",
  checkpoint: null,
  images: [
    {
      id: "i1",
      file: "i1.jpg",
      labelId: "lbl_a",
      labelIds: ["lbl_a", "lbl_b"],
      source: "import",
      addedAt: "",
    },
  ],
};
const m2 = migrateManifest(structuredClone(v2));
check("a version 2 manifest stays multi label", m2.mode === "multi");
check(
  "a version 2 image keeps its whole label set",
  labelsOf(m2.images[0]).join(",") === "lbl_a,lbl_b",
);
check("a version 2 image gains no boxes", !("boxes" in m2.images[0]));

const boxes = [
  { id: "b1", labelId: "lbl_a", x: 0.1, y: 0.2, w: 0.3, h: 0.4 },
  { id: "b2", labelId: "lbl_a", x: 0.55, y: 0.05, w: 0.4, h: 0.9 },
];
const v3 = {
  ...base,
  formatVersion: 3,
  mode: "detect",
  checkpoint: null,
  images: [
    { id: "i1", file: "i1.jpg", labelId: null, boxes, source: "import", addedAt: "" },
    { id: "i2", file: "i2.jpg", labelId: null, boxes: [], source: "capture", addedAt: "" },
  ],
};
const m3 = migrateManifest(structuredClone(v3));
check("a detect manifest stays detect", m3.mode === "detect");
check(
  "every box survives a round trip unchanged",
  JSON.stringify(boxesOf(m3.images[0])) === JSON.stringify(boxes),
);
check(
  "a picture with no boxes stays a background example",
  Array.isArray(m3.images[1].boxes) && m3.images[1].boxes.length === 0,
);

const detectMeta = {
  trainedAt: "2026-01-02T00:00:00.000Z",
  mode: "detect",
  labelIds: ["lbl_a"],
  epochs: 1,
  accuracy: 1,
  valAccuracy: null,
  imageCount: 2,
  settings,
};
const stale = migrateManifest({ ...v3, model: detectMeta, modelStale: true });
check("a stale detector flag survives save and reopen", stale.modelStale === true);
check(
  "a stale detector with the same output order can still be restored with a warning",
  canRestoreModel(stale, "detect", ["lbl_a"]),
);
check(
  "a detector with a different label order cannot be restored",
  !canRestoreModel(stale, "detect", ["lbl_b"]),
);
check(
  "an orphan detector file cannot restore without manifest metadata",
  !canRestoreModel(m3, "detect", ["lbl_a"]),
);

const rough = migrateManifest({
  ...v3,
  images: [
    {
      id: "i1",
      file: "i1.jpg",
      labelId: null,
      source: "import",
      addedAt: "",
      boxes: [
        { id: "out", labelId: "lbl_a", x: -0.4, y: 0.5, w: 0.9, h: 0.2 },
        { id: "over", labelId: "lbl_a", x: 0.9, y: 0.9, w: 0.5, h: 0.5 },
        { id: "sliver", labelId: "lbl_a", x: 0.5, y: 0.5, w: 0.001, h: 0.4 },
        { id: "nameless", labelId: "", x: 0.1, y: 0.1, w: 0.4, h: 0.4 },
        { id: "backwards", labelId: "lbl_a", x: 0.6, y: 0.6, w: -0.3, h: -0.3 },
        { id: "broken", labelId: "lbl_a", x: Number.NaN, y: 0.1, w: 0.2, h: 0.2 },
      ],
    },
  ],
});
const kept = boxesOf(rough.images[0]);
check(
  "loading drops slivers, nameless and broken boxes",
  kept.map((b) => b.id).join(",") === "out,over,backwards",
  kept.map((b) => b.id).join(","),
);
check(
  "loading pulls every box back inside the picture",
  kept.every((b) => b.x >= 0 && b.y >= 0 && b.x + b.w <= 1 + 1e-9 && b.y + b.h <= 1 + 1e-9),
);
check(
  "loading straightens a backwards box",
  sameRect(kept.find((b) => b.id === "backwards"), { x: 0.3, y: 0.3, w: 0.3, h: 0.3 }),
);

console.log(results.join("\n"));
console.log(failures === 0 ? "\nall box checks passed" : `\n${failures} check(s) failed`);
process.exit(failures === 0 ? 0 : 1);
