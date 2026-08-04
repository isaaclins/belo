// Headless capture HUD lifecycle and interaction check.
// The browser harness supplies a fake Tauri window runtime, so no native window
// is created and no desktop focus is touched.

import { findChrome, printReport, runHarness } from "./browser/cdp.mjs";

if (!findChrome()) {
  console.log("skipped: no Chrome or Chromium on this machine (set CHROME_PATH to force one)");
  process.exit(0);
}

const result = await runHarness({
  page: "/scripts/browser/hud.html",
  timeoutMs: 60_000,
});

process.exit(printReport("check:hud", result));
