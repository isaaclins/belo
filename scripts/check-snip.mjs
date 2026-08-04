// Drives the global Cmd+Shift+B seam in headless Chrome.
// The browser page injects fixture bytes into a fake native runner, so this
// never executes screencapture or takes over a desktop.

import { findChrome, printReport, runHarness } from "./browser/cdp.mjs";

if (!findChrome()) {
  console.log("skipped: no Chrome or Chromium on this machine (set CHROME_PATH to force one)");
  process.exit(0);
}

const result = await runHarness({
  page: "/scripts/browser/snip.html",
  timeoutMs: 30_000,
});

process.exit(printReport("check:snip", result));
