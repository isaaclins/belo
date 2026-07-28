// Regression check for detector manifest staleness and serialized saves.
// Uses an in-memory storage double and a /tmp handle. No project is created.

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
const { newManifest } = await import(new URL("../src/lib/types.ts", import.meta.url));

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

console.log(results.join("\n"));
console.log(failures === 0 ? "\nall store checks passed" : `\n${failures} check(s) failed`);
process.exit(failures === 0 ? 0 : 1);
