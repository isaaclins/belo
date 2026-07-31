// Drives the screen source harness in headless Chrome.
//
//   npm run check:screen
//
// The synthetic desktop lives inside the page, so this never records a real
// screen and never asks for a permission. Chrome's fake camera devices give the
// harness the camera half of the toggle. It stays in `npm test` and reports a
// skip rather than failing on a machine without Chrome, like the other browser
// checks.

import { findChrome, printReport, runHarness } from "./browser/cdp.mjs";

if (!findChrome()) {
  console.log("skipped: no Chrome or Chromium on this machine (set CHROME_PATH to force one)");
  process.exit(0);
}

const result = await runHarness({
  page: "/scripts/browser/screen.html",
  extraArgs: [
    "--use-fake-ui-for-media-stream",
    // Two cameras, so leaving the camera for the screen and coming back has a
    // pick to restore that is not simply the default.
    "--use-fake-device-for-media-stream=device-count=2",
  ],
  timeoutMs: 180_000,
});

process.exit(printReport("check:screen", result));
