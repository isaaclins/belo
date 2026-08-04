// Backend selection and recovery. Both the WASM binaries and the MobileNet
// weights ship inside the app, so training works offline even when a WebGL
// context is not usable.

import * as tf from "@tensorflow/tfjs";
import { setWasmPaths } from "@tensorflow/tfjs-backend-wasm";

const WASM_PATH = "/tfjs-wasm/";
const IDLE_GATE = Promise.resolve();

let ready: Promise<string> | null = null;
let wasmReady: Promise<string> | null = null;
let wasmFailure: unknown = null;
let wasmForced = false;
let gateTail: Promise<void> = IDLE_GATE;

function errorMessage(error: unknown): string {
  if (error instanceof Error) return `${error.name}: ${error.message} ${error.stack ?? ""}`;
  return String(error);
}

/** Errors that mean the WebGL backend itself, rather than a project, failed. */
export function isWebglFailure(error: unknown): boolean {
  const text = errorMessage(error);
  return /failed to link vertex and fragment shaders|failed to compile (?:vertex|fragment) shader|webgl (?:error|context)|context lost|context_loss|contextlost|initialization of backend webgl failed|out of memory/i.test(
    text,
  );
}

async function activateWasm(): Promise<string> {
  setWasmPaths(WASM_PATH);
  const selected = await tf.setBackend("wasm");
  if (!selected) throw new Error("TensorFlow.js could not select the WASM backend");
  await tf.ready();
  if (tf.getBackend() !== "wasm") {
    throw new Error(`TensorFlow.js selected ${tf.getBackend() || "no backend"} instead of WASM`);
  }
  return "wasm";
}

function ensureWasm(): Promise<string> {
  if (wasmFailure) return Promise.reject(wasmFailure);
  if (!wasmReady) {
    wasmReady = activateWasm().catch((error) => {
      wasmFailure = error;
      wasmReady = null;
      throw error;
    });
  }
  return wasmReady;
}

async function pickBackend(): Promise<string> {
  if (wasmForced) return ensureWasm();

  try {
    const selected = await tf.setBackend("webgl");
    if (!selected) throw new Error("TensorFlow.js could not select the WebGL backend");
    await tf.ready();
    if (tf.getBackend() === "webgl") return "webgl";
  } catch {
    // WebGL is unavailable during startup. Fall through to the bundled WASM
    // backend, which is also the controlled recovery path below.
  }

  return ensureWasm();
}

export function initBackend(): Promise<string> {
  if (!ready) {
    ready = pickBackend().catch((error) => {
      ready = null;
      throw error;
    });
  }
  return ready;
}

/** Switches away from a failed WebGL context once for the lifetime of a session. */
export function switchToWasm(): Promise<string> {
  wasmForced = true;
  if (tf.getBackend() === "wasm") {
    ready = Promise.resolve("wasm");
    return ready;
  }
  ready = ensureWasm();
  return ready;
}

export function currentBackend(): string {
  return tf.getBackend();
}

/**
 * Serialize TensorFlow work. WebGL shader compilation and tensor reads share a
 * single context, and WKWebView is especially sensitive to overlapping model
 * work. Training takes the exclusive gate; live inference uses the non-queuing
 * form and drops a frame while training or recovery owns it.
 */
export function runMlExclusive<T>(operation: () => Promise<T> | T): Promise<T> {
  const previous = gateTail;
  let release!: () => void;
  const own = new Promise<void>((resolve) => {
    release = resolve;
  });
  gateTail = own;

  return previous.then(operation).finally(() => {
    release();
    if (gateTail === own) gateTail = IDLE_GATE;
  });
}

export function runMlIfIdle<T>(operation: () => Promise<T> | T): Promise<T | null> {
  if (gateTail !== IDLE_GATE) return Promise.resolve(null);

  let release!: () => void;
  const own = new Promise<void>((resolve) => {
    release = resolve;
  });
  gateTail = own;

  return Promise.resolve()
    .then(operation)
    .finally(() => {
      release();
      if (gateTail === own) gateTail = IDLE_GATE;
    });
}
