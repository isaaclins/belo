// Drives the assisted labeling harness in headless Chrome.
//
//   npm run check:propose
//
// The page trains a real detector on a real detect project and then mounts the
// real box annotator on a picture with nothing drawn on it, so what is asserted
// is the component's own DOM and the project manifest behind it. Like the other
// two browser checks it needs a Chrome on the machine and reports a skip when
// there is none, so a machine without a browser still passes the suite.
//
// Nothing here reaches the network: the extractor is the copy bundled in
// public/models, so this one is safe to run offline and is part of `npm test`.

import { findChrome, printReport, runHarness } from "./browser/cdp.mjs";

if (!findChrome()) {
  console.log("skipped: no Chrome or Chromium on this machine (set CHROME_PATH to force one)");
  process.exit(0);
}

const result = await runHarness({ page: "/scripts/browser/propose.html" });

process.exit(printReport("check:propose", result));
