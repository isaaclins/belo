// Runs the whole check suite in one go, in the same order CI does.
//
//   npm test
//
// Each step is a normal npm script, run in its own process, one after the
// other. The two browser steps drive headless Chrome and report a skip on a
// machine that has none, so the suite still passes where no browser is
// installed. They are sequential on purpose: check:learning trains a real model and
// wants the whole CPU, and interleaved output from parallel runs would make a
// failure hard to read. The first failing step stops the run and its exit code
// becomes the exit code of the suite.
import { spawnSync } from "node:child_process";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const STEPS = [
  ["typecheck", "the TypeScript source still typechecks"],
  ["check:boxes", "boxes survive drawing, saving and loading"],
  ["check:detect", "the detection layer decides the way it should"],
  ["check:curve", "a loss curve is read into the right epoch and the right words"],
  ["check:camera", "the camera seam behaves without a camera"],
  ["check:camera-live", "a real camera session switches devices and survives a removal"],
  ["check:overlay", "the multi label attention overlay paints what the model saw"],
  ["check:store", "the project store saves and closes correctly"],
  ["check:learning", "the detector actually learns, on the real model"],
];

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const npm = process.platform === "win32" ? "npm.cmd" : "npm";
const started = Date.now();
const times = [];

for (const [script, what] of STEPS) {
  const stepStarted = Date.now();
  console.log(`\n=== ${script}: ${what}`);
  const run = spawnSync(npm, ["run", "-s", script], {
    cwd: root,
    stdio: "inherit",
    shell: process.platform === "win32",
  });
  const seconds = (Date.now() - stepStarted) / 1000;
  if (run.status !== 0) {
    console.error(`\nFAILED ${script} after ${seconds.toFixed(1)}s`);
    process.exit(run.status ?? 1);
  }
  times.push([script, seconds]);
  console.log(`--- ${script} passed in ${seconds.toFixed(1)}s`);
}

console.log("\n=== all checks passed");
for (const [script, seconds] of times) {
  console.log(`  ${script.padEnd(15)} ${seconds.toFixed(1)}s`);
}
console.log(`  ${"total".padEnd(15)} ${((Date.now() - started) / 1000).toFixed(1)}s`);
