// Drives the camera switching harness in headless Chrome with fake devices.
//
//   npm run check:camera-live
//
// The node level check (npm run check:camera) proves the seam's decisions against
// a mocked MediaDevices. This one proves the whole session against real
// MediaStreams: switching mid stream, stopped tracks, the remembered deviceId and
// a camera that disappears. It stays out of `npm test` because it needs a Chrome
// on the machine, and it reports a skip rather than failing where there is none.

import { findChrome, printReport, runHarness } from "./browser/cdp.mjs";

if (!findChrome()) {
  console.log("skipped: no Chrome or Chromium on this machine (set CHROME_PATH to force one)");
  process.exit(0);
}

const result = await runHarness({
  page: "/scripts/browser/camera.html",
  extraArgs: [
    "--use-fake-ui-for-media-stream",
    // Three cameras, so switching has somewhere to go and losing one still
    // leaves a fallback.
    "--use-fake-device-for-media-stream=device-count=3",
  ],
  timeoutMs: 90_000,
});

process.exit(printReport("check:camera-live", result));
