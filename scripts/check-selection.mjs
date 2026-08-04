// Drives the contextual Cmd/Ctrl+A check in headless Chrome.
//
//   npm run check:selection
//
// A browser is needed because the behavior depends on real event targets and
// React's mounted views. Machines without Chrome report a skip like the other
// browser checks.

import { findChrome, printReport, runHarness } from "./browser/cdp.mjs";

if (!findChrome()) {
  console.log("skipped: no Chrome or Chromium on this machine (set CHROME_PATH to force one)");
  process.exit(0);
}

const result = await runHarness({
  page: "/scripts/browser/selection.html",
  timeoutMs: 30_000,
});

process.exit(printReport("check:selection", result));
