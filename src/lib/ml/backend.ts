// Backend selection: WebGL first, WASM as the fallback. Both the WASM binaries
// and the MobileNet weights ship inside the app, so training works offline.

import * as tf from "@tensorflow/tfjs";
import { setWasmPaths } from "@tensorflow/tfjs-backend-wasm";

let ready: Promise<string> | null = null;

async function pickBackend(): Promise<string> {
  try {
    await tf.setBackend("webgl");
    await tf.ready();
    if (tf.getBackend() === "webgl") return "webgl";
  } catch {
    // WebGL is unavailable in this webview, fall through to WASM.
  }

  // Importing the package already registers the backend, so only the binary
  // location needs pointing at the bundled copy.
  setWasmPaths("/tfjs-wasm/");
  await tf.setBackend("wasm");
  await tf.ready();
  return tf.getBackend();
}

export function initBackend(): Promise<string> {
  if (!ready) {
    ready = pickBackend().catch((err) => {
      ready = null;
      throw err;
    });
  }
  return ready;
}

export function currentBackend(): string {
  return tf.getBackend();
}
