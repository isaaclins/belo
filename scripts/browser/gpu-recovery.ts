// Headless integration check for the WebGL shader-link recovery path.
// It uses the real TFJS 4.22.0 WebGL and WASM backends, the real two-scale
// detector, and the real project store. The one failed link is injected at the
// WebGL API seam, never by touching a user-owned window.

import * as tf from "@tensorflow/tfjs";
import { initBackend } from "../../src/lib/ml/backend";
import { loadDetectFeatureModel } from "../../src/lib/ml/features";
import { buildDetector, serializeDetector } from "../../src/lib/ml/detector";
import { APP_VERSION, useStore } from "../../src/lib/store";
import { storage } from "../../src/lib/storage";
import { DETECT_MIN_IMAGES, newManifest } from "../../src/lib/types";

const results: { name: string; ok: boolean; detail?: string }[] = [];
let failures = 0;

function check(name: string, ok: boolean, detail = "") {
  results.push({ name, ok, detail });
  if (!ok) failures += 1;
}

const PIXEL_PNG =
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=";

const manifest = newManifest("GPU recovery", APP_VERSION, "detect");
manifest.labels = [
  { id: "lbl_object", name: "Object", color: 0, createdAt: "2026-01-01T00:00:00.000Z" },
];
manifest.images = Array.from({ length: DETECT_MIN_IMAGES }, (_, index) => ({
  id: `img_${index}`,
  file: `img_${index}.jpg`,
  labelId: null,
  boxes: [
    {
      id: `box_${index}`,
      labelId: "lbl_object",
      x: 0.2,
      y: 0.2,
      w: 0.4,
      h: 0.4,
    },
  ],
  source: "import" as const,
  addedAt: "2026-01-01T00:00:00.000Z",
}));
manifest.settings = {
  epochs: 1,
  batchSize: 2,
  learningRate: 0.001,
  hiddenUnits: 16,
  validationSplit: 0,
  augment: false,
};

storage.imageUrl = () => PIXEL_PNG;
let storedDetector: string | null = null;
let detectorReads = 0;
storage.writeDetector = async (_path, text) => {
  storedDetector = text;
};
storage.readDetector = async () => {
  detectorReads += 1;
  return storedDetector;
};
storage.writeManifest = async () => undefined;

const state = {
  path: "/tmp/GPU-recovery.belo",
  manifest,
  view: "train" as const,
  dirty: false,
  hasModel: false,
  stale: false,
  hasCheckpoint: false,
  modelLabelIds: [],
  train: useStore.getState().train,
  toasts: [],
};
useStore.setState(state);

await initBackend();
check("the harness starts on WebGL", tf.getBackend() === "webgl", tf.getBackend());
await loadDetectFeatureModel();
const previousDetector = buildDetector(["lbl_object"], 16);
storedDetector = JSON.stringify(serializeDetector(previousDetector));
previousDetector.model.dispose();
manifest.model = {
  trainedAt: "2026-01-02T00:00:00.000Z",
  mode: "detect",
  labelIds: ["lbl_object"],
  epochs: 1,
  accuracy: 0,
  valAccuracy: null,
  imageCount: DETECT_MIN_IMAGES,
  settings: manifest.settings,
};
useStore.setState({ hasModel: true, stale: true, modelLabelIds: ["lbl_object"] });

const gpu = tf.backend() as { getGPGPUContext?: () => { gl: WebGL2RenderingContext } };
const gl = gpu.getGPGPUContext?.().gl;
let failedLinks = 0;
const originalLink = gl.linkProgram.bind(gl);
let injectFailure = true;
gl.linkProgram = ((program: WebGLProgram) => {
  if (injectFailure) {
    injectFailure = false;
    failedLinks += 1;
    return;
  }
  return originalLink(program);
}) as typeof gl.linkProgram;

await useStore.getState().startTraining();
try {
  gl.linkProgram = originalLink;
} catch {
  // The browser owns this method on a few WebGL implementations. The one-shot
  // wrapper already restored itself after the injected call in that case.
}

const afterRecovery = useStore.getState();
check("the injected shader-link failure happened exactly once", failedLinks === 1, `${failedLinks}`);
check("the failed run recovered onto WASM", afterRecovery.backend === "wasm", afterRecovery.backend ?? "null");
check(
  "the detector training state recovered to done",
  afterRecovery.train.phase === "done" && afterRecovery.hasModel,
  `${afterRecovery.train.phase}, hasModel=${afterRecovery.hasModel}`,
);
check("the persisted detector was read back after the backend switch", detectorReads > 0, `${detectorReads} reads`);
check(
  "the only notice explains the CPU compatibility switch",
  afterRecovery.toasts.length === 1 && /CPU compatibility mode/i.test(afterRecovery.toasts[0]?.text ?? ""),
  afterRecovery.toasts.map((toast) => toast.text).join(" | "),
);
check(
  "the raw shader error is not surfaced as a toast",
  !afterRecovery.toasts.some((toast) => /Failed to link vertex and fragment shaders/i.test(toast.text)),
);

await new Promise((resolve) => setTimeout(resolve, 600));
check("the recovery path stays quiescent instead of retrying", useStore.getState().toasts.length === 1);

const canvas = document.createElement("canvas");
canvas.width = 224;
canvas.height = 224;
const beforePrediction = tf.memory().numTensors;
let predictionOk = false;
try {
  const detections = await useStore.getState().detectObjects(canvas, 0.2);
  predictionOk = Array.isArray(detections);
} catch {
  predictionOk = false;
}
check("a later detector prediction succeeds on the recovered backend", predictionOk);
check(
  "prediction disposes its temporary tensors",
  tf.memory().numTensors === beforePrediction,
  `before ${beforePrediction}, after ${tf.memory().numTensors}`,
);

const beforeSecondRun = tf.memory().numTensors;
await useStore.getState().startTraining();
const afterSecondRun = useStore.getState();
check("a later manual detector training succeeds", afterSecondRun.train.phase === "done");
check(
  "the later run does not accumulate tensors",
  tf.memory().numTensors === beforeSecondRun,
  `before ${beforeSecondRun}, after ${tf.memory().numTensors}`,
);

(globalThis as { __BELO_CHECK__?: unknown }).__BELO_CHECK__ = {
  done: true,
  results,
  error: failures === 0 ? null : `${failures} check(s) failed`,
};
