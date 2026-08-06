// One version number, four places that have to agree.
//
//   node scripts/check-version.mjs
//
// package.json is the source of truth. The Tauri config and the Rust crate
// decide what a built app calls itself, Cargo.lock decides whether a
// `cargo --locked` build succeeds at all, and APP_VERSION is stamped into every
// project manifest and every export. A release that bumps some of them and not
// the others ships a build whose files claim the wrong version, which is how
// 0.11.2 came to write "0.11.1" into every manifest it touched.

import { readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("..", import.meta.url));
const read = (...parts) => readFileSync(join(root, ...parts), "utf8");

function fromJson(file, path) {
  const value = path.reduce((node, key) => node?.[key], JSON.parse(read(file)));
  return typeof value === "string" ? value : null;
}

/** The version of the crate's own [package] block, not of a dependency. */
function fromCargoToml(file) {
  const lines = read(file).split("\n");
  const start = lines.findIndex((line) => line.trim() === "[package]");
  if (start === -1) return null;
  for (const line of lines.slice(start + 1)) {
    if (line.trim().startsWith("[")) break;
    const match = /^version\s*=\s*"([^"]+)"/.exec(line.trim());
    if (match) return match[1];
  }
  return null;
}

/** The locked version of the belo package itself, which is what --locked checks. */
function fromCargoLock(file, name) {
  const blocks = read(file).split("[[package]]");
  for (const block of blocks) {
    if (!new RegExp(`^\\s*name\\s*=\\s*"${name}"\\s*$`, "m").test(block)) continue;
    const match = /^\s*version\s*=\s*"([^"]+)"\s*$/m.exec(block);
    if (match) return match[1];
  }
  return null;
}

function fromSource(file, pattern) {
  const match = pattern.exec(read(file));
  return match ? match[1] : null;
}

const expected = fromJson("package.json", ["version"]);
const found = [
  ["package.json", expected],
  ["src-tauri/tauri.conf.json", fromJson("src-tauri/tauri.conf.json", ["version"])],
  ["src-tauri/Cargo.toml", fromCargoToml("src-tauri/Cargo.toml")],
  ["src-tauri/Cargo.lock (belo)", fromCargoLock("src-tauri/Cargo.lock", "belo")],
  ["src-tauri/Cargo.lock (belo_lib)", fromCargoLock("src-tauri/Cargo.lock", "belo_lib")],
  ["src/lib/store.ts APP_VERSION", fromSource("src/lib/store.ts", /APP_VERSION\s*=\s*"([^"]+)"/)],
];

const results = [];
let failures = 0;
for (const [where, value] of found) {
  // A missing belo_lib entry is fine: the lock file only names it when the
  // library target is a package in its own right.
  const optional = where === "src-tauri/Cargo.lock (belo_lib)";
  if (value === null && optional) {
    results.push(`ok   ${where} :: not present, nothing to keep in step`);
    continue;
  }
  const ok = value !== null && value === expected;
  if (!ok) failures += 1;
  results.push(`${ok ? "ok  " : "FAIL"} ${where} :: ${value ?? "not found"}`);
}

console.log(`the release version is ${expected ?? "not readable"}`);
for (const line of results) console.log(line);

if (!expected) {
  console.error("\npackage.json has no version to compare against");
  process.exit(1);
}
if (failures > 0) {
  console.error(`\n${failures} FAILED :: every version must read ${expected}`);
  process.exit(1);
}
console.log(`\n${results.length} checks passed`);
