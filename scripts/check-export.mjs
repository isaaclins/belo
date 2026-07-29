// The closed loop on model export.
//
//   npm run check:export
//
// An export is only worth shipping if the thing that comes out of it runs
// somewhere else, so this check refuses to take the archive's word for it:
//
// 1. A headless Chrome page trains a real classifier, builds a real detector and
//    exports both through the app's own export path, then reports the two
//    archives, a fixture picture and Belo's own answers for it.
// 2. Node unzips an archive here, with its own reader, and checks that everything
//    the README promises is in it.
// 3. The unzipped folder is served by a plain static server, exactly the way the
//    zip's README tells a person to serve it, and the example page inside it is
//    loaded in the same browser and asked about the same picture.
// 4. The example's numbers have to match Belo's, to a tolerance printed with the
//    result.
//
// The example loads TensorFlow.js from a pinned CDN URL, which is the honest
// thing for an exported folder to do and means this check needs the network. It
// reports a skip rather than a failure when there is no Chrome or no CDN.

import { createServer } from "node:http";
import { createServer as createSocketServer } from "node:net";
import { inflateRawSync } from "node:zlib";
import { mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, extname, join, normalize } from "node:path";
import { findChrome, launchChrome, startDevServer } from "./browser/cdp.mjs";

// The pinned CDN URL is owned by the module that writes the example, so it is
// read from there rather than repeated here. It imports nothing, so plain node
// type stripping is enough to load it.
const { TFJS_CDN } = await import(new URL("../src/lib/exportFiles.ts", import.meta.url));

const results = [];
function check(name, ok, detail = "") {
  const text = typeof detail === "string" ? detail : JSON.stringify(detail);
  results.push({ name, ok: Boolean(ok), detail: text });
  console.log(`${ok ? "ok  " : "FAIL"} ${name}${text ? ` :: ${text}` : ""}`);
}

// --- a zip reader, so the check does not depend on an unzip binary ------------

function readUint32(bytes, at) {
  return (bytes[at] | (bytes[at + 1] << 8) | (bytes[at + 2] << 16) | (bytes[at + 3] << 24)) >>> 0;
}

function readUint16(bytes, at) {
  return bytes[at] | (bytes[at + 1] << 8);
}

/** Every entry of a zip, read through its central directory. */
function unzip(bytes) {
  let end = bytes.length - 22;
  while (end >= 0 && readUint32(bytes, end) !== 0x06054b50) end -= 1;
  if (end < 0) throw new Error("not a zip: no end of central directory");
  const count = readUint16(bytes, end + 10);
  let at = readUint32(bytes, end + 16);
  const entries = new Map();
  for (let i = 0; i < count; i += 1) {
    if (readUint32(bytes, at) !== 0x02014b50) throw new Error("bad central directory entry");
    const method = readUint16(bytes, at + 10);
    const compressedSize = readUint32(bytes, at + 20);
    const size = readUint32(bytes, at + 24);
    const nameLength = readUint16(bytes, at + 28);
    const extraLength = readUint16(bytes, at + 30);
    const commentLength = readUint16(bytes, at + 32);
    const offset = readUint32(bytes, at + 42);
    const name = Buffer.from(bytes.subarray(at + 46, at + 46 + nameLength)).toString("utf8");

    const localNameLength = readUint16(bytes, offset + 26);
    const localExtraLength = readUint16(bytes, offset + 28);
    const from = offset + 30 + localNameLength + localExtraLength;
    const raw = bytes.subarray(from, from + compressedSize);
    const data = method === 0 ? Buffer.from(raw) : inflateRawSync(Buffer.from(raw));
    if (data.length !== size) throw new Error(`${name}: inflated to ${data.length}, wanted ${size}`);
    entries.set(name, { data, method, compressedSize, size });
    at += 46 + nameLength + extraLength + commentLength;
  }
  return entries;
}

function writeTree(dir, entries) {
  for (const [name, entry] of entries) {
    const target = join(dir, name);
    mkdirSync(dirname(target), { recursive: true });
    writeFileSync(target, entry.data);
  }
}

// --- a static server, the one the zip's README tells people to use ------------

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".md": "text/markdown; charset=utf-8",
  ".png": "image/png",
  ".bin": "application/octet-stream",
};

function freePort() {
  return new Promise((resolve, reject) => {
    const probe = createSocketServer();
    probe.once("error", reject);
    probe.listen(0, "127.0.0.1", () => {
      const { port } = probe.address();
      probe.close(() => (port === 1420 ? reject(new Error("1420 belongs to tauri dev")) : resolve(port)));
    });
  });
}

async function serveFolder(dir) {
  const port = await freePort();
  const server = createServer((request, response) => {
    const path = decodeURIComponent(request.url.split("?")[0]);
    const relative = normalize(path === "/" ? "/example.html" : path).replace(/^([/\\])+/, "");
    const target = join(dir, relative);
    if (!target.startsWith(dir)) {
      response.writeHead(403).end();
      return;
    }
    try {
      const body = readFileSync(target);
      response.writeHead(200, {
        "Content-Type": MIME[extname(target)] ?? "application/octet-stream",
        "Content-Length": body.length,
      });
      response.end(body);
    } catch {
      response.writeHead(404).end("not found");
    }
  });
  await new Promise((resolve) => server.listen(port, "127.0.0.1", resolve));
  return { url: `http://127.0.0.1:${port}`, stop: () => server.close() };
}

// --- comparing two answers ---------------------------------------------------

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

/** Largest absolute difference between two score rows, by label. */
function scoreDelta(mine, theirs) {
  let worst = 0;
  for (const row of mine) {
    const other = theirs.find((entry) => entry.labelId === row.labelId);
    if (!other) return Infinity;
    worst = Math.max(worst, Math.abs(row.score - other.score));
  }
  return worst;
}

const boxKey = (box) => `${box.labelId}@${box.x.toFixed(3)},${box.y.toFixed(3)}`;

/** Worst coordinate and score difference over two box lists, matched by label. */
function boxDelta(mine, theirs) {
  if (mine.length !== theirs.length) return { count: Infinity, coord: Infinity, score: Infinity };
  const sortBy = (list) =>
    [...list].sort((a, b) => b.score - a.score || boxKey(a).localeCompare(boxKey(b)));
  const a = sortBy(mine);
  const b = sortBy(theirs);
  let coord = 0;
  let score = 0;
  let labelMismatch = 0;
  for (let i = 0; i < a.length; i += 1) {
    if (a[i].labelId !== b[i].labelId) labelMismatch += 1;
    coord = Math.max(
      coord,
      Math.abs(a[i].x - b[i].x),
      Math.abs(a[i].y - b[i].y),
      Math.abs(a[i].w - b[i].w),
      Math.abs(a[i].h - b[i].h),
    );
    score = Math.max(score, Math.abs(a[i].score - b[i].score));
  }
  return { count: 0, coord, score, labelMismatch };
}

// --- the run ------------------------------------------------------------------

if (!findChrome()) {
  console.log("skipped: no Chrome or Chromium on this machine (set CHROME_PATH to force one)");
  process.exit(0);
}

const cdnUp = await fetch(TFJS_CDN, { method: "HEAD" }).then(
  (response) => response.ok,
  () => false,
);
if (!cdnUp) {
  console.log(`skipped: ${TFJS_CDN} is unreachable, so the exported example cannot load`);
  process.exit(0);
}

const work = mkdtempSync(join(tmpdir(), "belo-export-"));
const server = await startDevServer();
const chrome = await launchChrome();
let failures = 0;
try {
  await chrome.navigate(`${server.url}/scripts/browser/export.html`);

  // The page trains a real model, so it gets a generous window.
  let report = null;
  for (let waited = 0; waited < 300_000; waited += 250) {
    report = await chrome.evaluate("window.__BELO_CHECK__ ?? null");
    if (report && report.done) break;
    await sleep(250);
  }
  if (!report?.done) throw new Error("the export page never finished");
  for (const line of report.results) check(line.name, line.ok, line.detail);
  if (report.error) throw new Error(`the export page threw: ${report.error}`);

  const truth = await chrome.evaluate("window.__BELO_EXPORT__");
  const fixture = await chrome.evaluate("window.__BELO_FIXTURE__");

  /** Reads one published archive back in slices, because 20 MB is a poor reply. */
  const readZip = async (name) => {
    const length = await chrome.evaluate(`window.${name}.length`);
    const step = 4_000_000;
    let base64 = "";
    for (let at = 0; at < length; at += step) {
      base64 += await chrome.evaluate(`window.${name}.slice(${at}, ${at + step})`);
    }
    return Buffer.from(base64, "base64");
  };

  // Both archives come off the export page before anything navigates away from
  // it: loading the example replaces the page and everything it published.
  const archives = {
    classify: await readZip("__BELO_ZIP_CLASSIFY__"),
    detect: await readZip("__BELO_ZIP_DETECT__"),
  };

  for (const kind of ["classify", "detect"]) {
    const bytes = archives[kind];
    const entries = unzip(bytes);
    const modelFile = kind === "classify" ? "classifier.json" : "detector.json";
    const dir = join(work, kind);
    mkdirSync(dir, { recursive: true });
    writeTree(dir, entries);
    writeFileSync(join(dir, "fixture.png"), Buffer.from(fixture, "base64"));

    const spec = JSON.parse(entries.get("belo-export.json").data.toString("utf8"));
    const shards = [...entries.keys()].filter((name) => name.startsWith("extractor/group"));
    const stored = JSON.parse(entries.get(modelFile).data.toString("utf8"));
    const deflated = [...entries.values()].filter((entry) => entry.method === 8).length;
    const raw = [...entries.values()].reduce((sum, entry) => sum + entry.size, 0);

    check(
      `${kind}: the archive holds the model, the extractor, the README and the example`,
      ["belo-export.json", modelFile, "README.md", "example.html", "example.js", "extractor/model.json"]
        .every((name) => entries.has(name)) && shards.length > 0,
      `${entries.size} files, ${shards.length} weight shards, ${(bytes.length / 1048576).toFixed(1)} MB packed from ${(raw / 1048576).toFixed(1)} MB, ${deflated} deflated`,
    );
    check(
      `${kind}: the exported weights are the project's own bytes, unconverted`,
      Array.isArray(stored.weights) && stored.weights.length > 0 && stored.featureSize === 1024,
      `formatVersion ${stored.formatVersion}, ${stored.weights.length} tensors`,
    );
    check(
      `${kind}: the descriptor names the mode, the labels and the extractor cuts`,
      spec.format === "belo-model-export" &&
        spec.mode === (kind === "classify" ? "single" : "detect") &&
        spec.labels.length === 2 &&
        spec.model.path === modelFile &&
        spec.extractor.cuts.length === (kind === "classify" ? 1 : 2),
      `${spec.mode}, cuts ${spec.extractor.cuts.join(" + ")}`,
    );
    const readme = entries.get("README.md").data.toString("utf8");
    check(
      `${kind}: the README says how to serve it and why file:// will not do`,
      readme.includes("npx serve .") && readme.includes("file://") && readme.includes("127.5"),
    );

    // Now the part that cannot be argued with: run the exported example.
    const site = await serveFolder(dir);
    try {
      await chrome.navigate(`${site.url}/example.html`);
      const answer = await chrome.evaluate(`(async () => {
        for (let i = 0; i < 600; i += 1) {
          const api = window.beloExample;
          if (api && api.error) return { error: api.error };
          if (api && api.model && api.model.stored && api.model.extractor) break;
          await new Promise((r) => setTimeout(r, 250));
        }
        if (!window.beloExample || !window.beloExample.model.extractor) {
          return { error: "the example never became ready" };
        }
        const image = new Image();
        image.src = "fixture.png";
        await image.decode();
        const result = await window.beloExample.run(image, ${
          kind === "detect" ? `{ scoreThreshold: ${truth.detectCutoff} }` : "{}"
        });
        return {
          mode: result.mode,
          scores: (result.scores || []).map((row) => ({ labelId: row.labelId, score: row.score })),
          detections: result.detections || [],
        };
      })()`);
      if (answer?.error) throw new Error(`the exported example failed: ${answer.error}`);

      if (kind === "classify") {
        const delta = scoreDelta(truth.classifyScores, answer.scores);
        check(
          "classify: the exported example reads the same picture the same way",
          answer.mode === "single" && answer.scores.length === 2 && delta < 0.01,
          `worst score difference ${delta.toExponential(2)}; belo ${truth.classifyScores
            .map((row) => row.score.toFixed(4))
            .join("/")} vs example ${answer.scores.map((row) => row.score.toFixed(4)).join("/")}`,
        );
      } else {
        const delta = boxDelta(truth.detectBoxes, answer.detections);
        check(
          "detect: the exported example finds the same boxes in the same places",
          answer.mode === "detect" &&
            truth.detectBoxes.length === answer.detections.length &&
            delta.labelMismatch === 0 &&
            delta.coord < 0.01 &&
            delta.score < 0.01,
          `${truth.detectBoxes.length} boxes on both sides at a ${truth.detectCutoff} cutoff, worst corner off by ${delta.coord.toExponential(2)}, worst score off by ${delta.score.toExponential(2)}`,
        );
      }
    } finally {
      site.stop();
    }
  }
} catch (err) {
  check("the export check ran to the end", false, err.message);
  if (chrome.logs.length) console.error(chrome.logs.slice(-40).join("\n"));
} finally {
  chrome.close();
  server.stop();
  rmSync(work, { recursive: true, force: true, maxRetries: 5 });
}

failures = results.filter((line) => !line.ok).length;
console.log(failures === 0 ? `\n${results.length} export checks passed` : `\n${failures} FAILED`);
process.exit(failures === 0 ? 0 : 1);
