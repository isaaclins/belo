// Reproduces a TFJS WebGL shader-link failure at the WebGL API seam and drives
// the real store through the one-time WASM recovery path in headless Chrome.
// This is deliberately part of npm test when Chrome exists, and skips cleanly
// on CI workers that have no browser.

import { findChrome, printReport, runHarness } from "./browser/cdp.mjs";

if (!findChrome()) {
  console.log("skipped: no Chrome or Chromium on this machine (set CHROME_PATH to force one)");
  process.exit(0);
}

const result = await runHarness({ page: "/scripts/browser/gpu-recovery.html" });
process.exit(printReport("check:gpu-recovery", result));
