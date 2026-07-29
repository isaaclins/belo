// Regression check for the project store: manifest staleness, serialized saves,
// the background retrain policy, the detector checkpoint and the annotator's
// undo stack. Uses an in-memory storage double and a /tmp handle. No project is
// created and nothing is trained.

import { registerHooks } from "node:module";

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

const { storage } = await import(new URL("../src/lib/storage.ts", import.meta.url));
const { APP_VERSION, useStore } = await import(new URL("../src/lib/store.ts", import.meta.url));
const { DETECT_MIN_IMAGES, newManifest } = await import(new URL("../src/lib/types.ts", import.meta.url));
const { autoRetrainAllowed, createRetrainScheduler, AUTO_RETRAIN_DELAY_MS } = await import(
  new URL("../src/lib/retrain.ts", import.meta.url)
);
const { emptyUndo, recordUndo, redo, undo, UNDO_LIMIT } = await import(
  new URL("../src/lib/undo.ts", import.meta.url)
);
const { buildDetector, serializeDetector, deserializeDetector } = await import(
  new URL("../src/lib/ml/detector.ts", import.meta.url)
);

// Importing the store registers the WASM backend, whose binaries are only in
// the app bundle and not on disk in node. Nothing here trains, so the plain CPU
// backend is enough, and picking it up front keeps the first tensor from
// reaching for the other one.
const tf = await import("@tensorflow/tfjs");
await tf.setBackend("cpu");

let failures = 0;
const results = [];
function check(name, ok, detail = "") {
  results.push(`${ok ? "ok  " : "FAIL"} ${name}${detail ? ` :: ${detail}` : ""}`);
  if (!ok) failures += 1;
}

const path = "/tmp/zz-belo-store-check.belo";
const manifest = newManifest("Store check", APP_VERSION, "detect");
manifest.labels = [
  { id: "lbl_a", name: "A", color: 0, createdAt: "2026-01-01T00:00:00.000Z" },
];
manifest.images = [
  {
    id: "img_a",
    file: "img_a.jpg",
    labelId: null,
    boxes: [],
    source: "import",
    addedAt: "2026-01-01T00:00:00.000Z",
  },
];
manifest.model = {
  trainedAt: "2026-01-02T00:00:00.000Z",
  mode: "detect",
  labelIds: ["lbl_a"],
  epochs: 1,
  accuracy: 1,
  valAccuracy: null,
  imageCount: 1,
  settings: manifest.settings,
};

storage.recentProjects = async () => [];
storage.scanProjects = async () => [];

const writes = [];
storage.writeManifest = async (_path, text) => {
  writes.push(JSON.parse(text));
};

useStore.setState({
  path,
  manifest,
  dirty: false,
  saving: false,
  hasModel: true,
  stale: false,
  modelLabelIds: ["lbl_a"],
});

const boxId = useStore.getState().addBox("img_a", "lbl_a", {
  x: 0.123456789,
  y: 0.234567891,
  w: 0.345678912,
  h: 0.456789123,
});
await useStore.getState().saveNow();
const savedBox = writes.at(-1)?.images[0]?.boxes[0];
check("adding a box marks the saved detector stale", writes.at(-1)?.modelStale === true);
check(
  "a store save preserves exact normalized box coordinates",
  boxId &&
    savedBox?.x === 0.123456789 &&
    savedBox?.y === 0.234567891 &&
    savedBox?.w === 0.345678912 &&
    savedBox?.h === 0.456789123,
);

let releaseFirst;
const firstBlocked = new Promise((resolve) => {
  releaseFirst = resolve;
});
writes.length = 0;
storage.writeManifest = async (_path, text) => {
  writes.push(JSON.parse(text));
  if (writes.length === 1) await firstBlocked;
};

useStore.setState({ dirty: true });
const firstSave = useStore.getState().saveNow();
useStore.getState().renameProject("Renamed while saving");
const secondSave = useStore.getState().saveNow();
releaseFirst();
await Promise.all([firstSave, secondSave]);
check(
  "overlapping saves serialize old then new manifests",
  writes.length === 2 &&
    writes[0].name === "Store check" &&
    writes[1].name === "Renamed while saving",
  writes.map((entry) => entry.name).join(", "),
);
check("the latest serialized save clears dirty state", useStore.getState().dirty === false);

writes.length = 0;
storage.writeManifest = async (_path, text) => {
  writes.push(JSON.parse(text));
};
useStore.getState().removeBox("img_a", boxId);
await useStore.getState().closeProject();
check("closing flushes a pending box deletion", writes.at(-1)?.images[0]?.boxes.length === 0);
check("a project closes only after its final save", useStore.getState().path === null);

// --- the background retrain policy, driven by a clock that never sleeps ---

function fakeClock() {
  let now = 0;
  let next = 1;
  const timers = new Map();
  return {
    clock: {
      set(fn, ms) {
        const id = next;
        next += 1;
        timers.set(id, { fn, at: now + ms });
        return id;
      },
      clear(id) {
        timers.delete(id);
      },
    },
    advance(ms) {
      now += ms;
      for (const [id, timer] of [...timers]) {
        if (timer.at <= now) {
          timers.delete(id);
          timer.fn();
        }
      }
    },
    armed: () => timers.size,
  };
}

let started = 0;
let busy = false;
const burst = fakeClock();
const scheduler = createRetrainScheduler({
  ready: () => true,
  busy: () => busy,
  start: () => {
    started += 1;
  },
  clock: burst.clock,
});

for (let i = 0; i < 6; i += 1) {
  scheduler.schedule();
  burst.advance(300);
}
check("a burst of edits leaves one timer armed", burst.armed() === 1 && started === 0);
burst.advance(AUTO_RETRAIN_DELAY_MS);
check("a burst of edits produces exactly one background run", started === 1, `started ${started}`);

busy = true;
started = 0;
scheduler.schedule();
burst.advance(AUTO_RETRAIN_DELAY_MS);
check(
  "a change during a manual run waits instead of starting a second one",
  started === 0 && scheduler.queued(),
);
busy = false;
scheduler.finished();
check("the queued run goes back on the clock rather than firing at once", started === 0);
burst.advance(AUTO_RETRAIN_DELAY_MS);
check("the queued run starts once the manual one is done", started === 1, `started ${started}`);
burst.advance(AUTO_RETRAIN_DELAY_MS * 4);
check("a queued run is not replayed forever", started === 1, `started ${started}`);

function boxedProject(name, count) {
  const project = newManifest(name, APP_VERSION, "detect");
  project.labels = [{ id: "lbl_a", name: "A", color: 0, createdAt: "2026-01-01T00:00:00.000Z" }];
  project.images = Array.from({ length: count }, (_, i) => ({
    id: `img_${i}`,
    file: `img_${i}.jpg`,
    labelId: null,
    boxes: [{ id: `box_${i}`, labelId: "lbl_a", x: 0.1, y: 0.1, w: 0.2, h: 0.2 }],
    source: "import",
    addedAt: "2026-01-01T00:00:00.000Z",
  }));
  return project;
}

const underMin = boxedProject("Too few", DETECT_MIN_IMAGES - 1);
const overMin = boxedProject("Enough", DETECT_MIN_IMAGES);
check(
  "a detect project under the training minimum never retrains in the background",
  autoRetrainAllowed(underMin, true) === false && autoRetrainAllowed(overMin, true) === true,
);
check(
  "a project with no model never retrains in the background",
  autoRetrainAllowed(overMin, false) === false,
);

// The same policy, this time reached the way the annotator reaches it: through
// the store, with the clock swapped out from under it.
const realSetTimeout = globalThis.setTimeout;
const realClearTimeout = globalThis.clearTimeout;
const armed = new Map();
let nextHandle = 0;
globalThis.setTimeout = (fn, ms) => {
  nextHandle += 1;
  armed.set(nextHandle, { fn, ms });
  return nextHandle;
};
globalThis.clearTimeout = (id) => void armed.delete(id);
const retrainTimers = () => [...armed.values()].filter((timer) => timer.ms === AUTO_RETRAIN_DELAY_MS);

const editBurst = (project) => {
  armed.clear();
  useStore.setState({
    path,
    manifest: project,
    dirty: false,
    hasModel: true,
    stale: false,
    modelLabelIds: ["lbl_a"],
  });
  for (let i = 0; i < 5; i += 1) {
    useStore.getState().updateBox(`img_${i}`, `box_${i}`, { x: 0.2, y: 0.2, w: 0.3, h: 0.3 });
  }
};

editBurst(boxedProject("Burst", DETECT_MIN_IMAGES + 2));
check(
  "five box edits in a row arm one background retrain, not five",
  retrainTimers().length === 1,
  `${retrainTimers().length} armed`,
);
check("the edits are queued for the manifest while the retrain waits", useStore.getState().dirty === true);

editBurst(boxedProject("Burst under the minimum", DETECT_MIN_IMAGES - 1));
check(
  "box edits below the training minimum arm nothing",
  retrainTimers().length === 0,
  `${retrainTimers().length} armed`,
);

useStore.getState().stopTraining();
armed.clear();
globalThis.setTimeout = realSetTimeout;
globalThis.clearTimeout = realClearTimeout;

// --- the detector checkpoint, written and put back byte for byte ---

const files = new Map();
storage.writeDetector = async (_path, text) => void files.set("detect.json", text);
storage.readDetector = async () => files.get("detect.json") ?? null;
storage.writeDetectorCheckpoint = async (_path, text) => void files.set("detect.prev.json", text);
storage.readDetectorCheckpoint = async () => files.get("detect.prev.json") ?? null;
storage.hasDetectorCheckpoint = async () => files.has("detect.prev.json");
storage.clearDetectorCheckpoint = async () => void files.delete("detect.prev.json");

const older = buildDetector(["lbl_a"], 16);
const newer = buildDetector(["lbl_a"], 16);
const olderText = JSON.stringify(serializeDetector(older));
const newerText = JSON.stringify(serializeDetector(newer));
older.model.dispose();
newer.model.dispose();

const reread = deserializeDetector(JSON.parse(olderText));
check(
  "a detector checkpoint keeps format version 2 and both feature maps",
  JSON.parse(olderText).formatVersion === 2 && JSON.parse(olderText).featureScales?.length === 2,
);
check(
  "a checkpointed detector survives serialization byte for byte",
  reread !== null && JSON.stringify(serializeDetector(reread)) === olderText,
);
reread?.model.dispose();
check("two runs really do produce different weights", olderText !== newerText);

const reverting = newManifest("Revert check", APP_VERSION, "detect");
reverting.labels = [{ id: "lbl_a", name: "A", color: 0, createdAt: "2026-01-01T00:00:00.000Z" }];
const olderMeta = {
  trainedAt: "2026-01-02T00:00:00.000Z",
  mode: "detect",
  labelIds: ["lbl_a"],
  epochs: 10,
  accuracy: 0.5,
  valAccuracy: null,
  imageCount: 12,
  settings: reverting.settings,
};
reverting.model = { ...olderMeta, trainedAt: "2026-01-03T00:00:00.000Z", accuracy: 0.9 };
reverting.checkpoint = olderMeta;

// Mirror one automatic replacement through the storage seam: copy the live
// detector first, then write the new one. The revert below must perform the
// other half of that round trip without changing one byte.
await storage.writeDetector(path, olderText);
await storage.writeDetectorCheckpoint(path, await storage.readDetector(path));
await storage.writeDetector(path, newerText);
check(
  "checkpoint writing copies the previous detector byte for byte",
  (await storage.readDetectorCheckpoint(path)) === olderText,
);
writes.length = 0;
useStore.setState({
  path,
  manifest: reverting,
  dirty: false,
  hasModel: true,
  hasCheckpoint: true,
  stale: false,
  modelLabelIds: ["lbl_a"],
});
await useStore.getState().revertModel();
check(
  "reverting puts the checkpointed detector back byte for byte",
  files.get("detect.json") === olderText,
);
check("reverting drops the checkpoint file", files.has("detect.prev.json") === false);
check(
  "reverting restores the metadata of the model it put back",
  useStore.getState().manifest.model?.trainedAt === olderMeta.trainedAt &&
    useStore.getState().manifest.checkpoint === null &&
    useStore.getState().manifest.modelStale === true,
);
check(
  "reverting leaves the project trained, marked stale and without a checkpoint",
  useStore.getState().hasModel === true &&
    useStore.getState().hasCheckpoint === false &&
    useStore.getState().stale === true,
);

await useStore.getState().revertModel();
check(
  "a bundle with no checkpoint reverts gracefully and changes nothing",
  files.get("detect.json") === olderText && useStore.getState().hasCheckpoint === false,
);

// --- the annotator's undo stack ---

const drawn = [
  { id: "box_1", labelId: "lbl_a", x: 0.123456789, y: 0.234567891, w: 0.345678912, h: 0.456789123 },
  { id: "box_2", labelId: "lbl_a", x: 0.5, y: 0.5, w: 0.25, h: 0.25 },
];
const moved = [{ ...drawn[0], x: 0.2, y: 0.3 }, drawn[1]];

let stack = recordUndo(emptyUndo(), drawn);
stack = recordUndo(stack, drawn);
check("recording the same boxes twice costs one undo step", stack.past.length === 1);

stack = recordUndo(stack, moved);
check("a real edit adds a step", stack.past.length === 2);

let deep = emptyUndo();
for (let i = 0; i < UNDO_LIMIT + 8; i += 1) {
  deep = recordUndo(deep, [{ ...drawn[0], x: i / 1000 }]);
}
check(
  `the stack keeps its last ${UNDO_LIMIT} steps and drops the oldest`,
  deep.past.length === UNDO_LIMIT && deep.past[0][0].x === 8 / 1000,
);

const back = undo(recordUndo(emptyUndo(), drawn), moved);
check(
  "undo restores the exact normalized coordinates that were stored",
  back !== null &&
    back.boxes[0].x === 0.123456789 &&
    back.boxes[0].y === 0.234567891 &&
    back.boxes[0].w === 0.345678912 &&
    back.boxes[0].h === 0.456789123 &&
    back.boxes.length === 2,
);
check("undo on an untouched picture does nothing", undo(emptyUndo(), drawn) === null);

const forward = redo(back.state, back.boxes);
check(
  "redo returns the edit undo took away",
  forward !== null && forward.boxes[0].x === 0.2 && forward.state.future.length === 0,
);
check("a new edit after an undo drops the redo trail", recordUndo(back.state, drawn).future.length === 0);

// --- an undone edit reaches the manifest with its numbers intact ---

const undoing = newManifest("Undo check", APP_VERSION, "detect");
undoing.labels = [{ id: "lbl_a", name: "A", color: 0, createdAt: "2026-01-01T00:00:00.000Z" }];
undoing.images = [
  {
    id: "img_a",
    file: "img_a.jpg",
    labelId: null,
    boxes: moved,
    source: "import",
    addedAt: "2026-01-01T00:00:00.000Z",
  },
];
writes.length = 0;
useStore.setState({
  path,
  manifest: undoing,
  dirty: false,
  hasModel: false,
  hasCheckpoint: false,
  stale: false,
  modelLabelIds: ["lbl_a"],
});
useStore.getState().setBoxes("img_a", back.boxes);
await useStore.getState().saveNow();
const restored = writes.at(-1)?.images[0]?.boxes ?? [];
check(
  "an undone move is saved with the coordinates it had before the drag",
  restored.length === 2 &&
    restored[0].id === "box_1" &&
    restored[0].x === 0.123456789 &&
    restored[0].y === 0.234567891 &&
    restored[0].w === 0.345678912 &&
    restored[0].h === 0.456789123,
);

console.log(results.join("\n"));
console.log(failures === 0 ? "\nall store checks passed" : `\n${failures} check(s) failed`);
process.exit(failures === 0 ? 0 : 1);
