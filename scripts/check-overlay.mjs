// Drives the attention overlay harness in headless Chrome.
//
//   npm run check:overlay
//
// This one is deliberately not part of `npm test`: it trains a real model inside
// a browser and needs a Chrome on the machine, which CI does not install. Without
// a Chrome it reports the gap and exits 0, so it can be wired into a wrapper
// without breaking a machine that cannot run it.

import { findChrome, printReport, runHarness } from "./browser/cdp.mjs";

if (!findChrome()) {
  console.log("skipped: no Chrome or Chromium on this machine (set CHROME_PATH to force one)");
  process.exit(0);
}

const result = await runHarness({
  page: "/scripts/browser/overlay.html",
  // A fake camera, so the overlay gets a real MediaStream with real dimensions
  // on a machine whose webcam may be busy or absent.
  extraArgs: ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream"],
});

process.exit(printReport("check:overlay", result));
