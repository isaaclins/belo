// Shared plumbing for the two headless browser checks.
//
// Belo's frontend is a plain web app, so the parts that only exist in a real
// browser (a live MediaStream, a canvas the compositor actually painted) can be
// checked in headless Chrome instead of being reasoned about. This module keeps
// that machinery in one place: a Vite dev server on a throwaway port, a Chrome
// launched into a throwaway profile, and just enough of the DevTools protocol to
// navigate and read a result object back out of the page.
//
// Nothing here ever raises a window: Chrome runs with --headless=new, and the
// dev server never touches port 1420, so a running `npm run tauri dev` is safe.

import { spawn } from "node:child_process";
import { createServer } from "node:net";
import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

export const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..", "..");

const CHROME_CANDIDATES = [
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  "/Applications/Chromium.app/Contents/MacOS/Chromium",
  "/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary",
  "/usr/bin/google-chrome",
  "/usr/bin/google-chrome-stable",
  "/usr/bin/chromium",
  "/usr/bin/chromium-browser",
];

/** Path to a usable Chrome, or null when this machine has none. */
export function findChrome() {
  if (process.env.CHROME_PATH && existsSync(process.env.CHROME_PATH)) return process.env.CHROME_PATH;
  return CHROME_CANDIDATES.find((path) => existsSync(path)) ?? null;
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

/** A port the OS says is free right now. */
function freePort() {
  return new Promise((resolve, reject) => {
    const probe = createServer();
    probe.once("error", reject);
    probe.listen(0, "127.0.0.1", () => {
      const { port } = probe.address();
      probe.close(() => (port === 1420 ? reject(new Error("1420 belongs to tauri dev")) : resolve(port)));
    });
  });
}

async function waitFor(what, probe, timeoutMs = 30_000) {
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    const value = await probe();
    if (value) return value;
    if (Date.now() > deadline) throw new Error(`Timed out waiting for ${what}`);
    await sleep(150);
  }
}

/**
 * Serves the repository through Vite so the harness pages can import the real
 * TypeScript modules. The port is picked by the OS, never 1420.
 */
export async function startDevServer() {
  // The WASM backend binaries are copied into public/ by the assets step and
  // are gitignored, so a fresh clone needs them before the page can boot.
  if (!existsSync(join(ROOT, "public", "tfjs-wasm"))) {
    const copy = spawn(process.execPath, [join(ROOT, "scripts", "copy-wasm.mjs")], {
      cwd: ROOT,
      stdio: "inherit",
    });
    await new Promise((resolve, reject) => {
      copy.on("exit", (code) => (code === 0 ? resolve() : reject(new Error("copy-wasm failed"))));
    });
  }

  // Vite reads 0 as "unset" and falls back to its own default, which may well be
  // taken, so a free port is picked here instead. 1420 belongs to tauri dev and
  // is never used.
  const port = await freePort();
  const npx = process.platform === "win32" ? "npx.cmd" : "npx";
  const child = spawn(
    npx,
    [
      "vite",
      "--port",
      String(port),
      "--host",
      "127.0.0.1",
      "--strictPort",
      "false",
      "--clearScreen",
      "false",
    ],
    {
      cwd: ROOT,
      stdio: ["ignore", "pipe", "pipe"],
      shell: process.platform === "win32",
    },
  );

  let output = "";
  const collect = (chunk) => {
    output += chunk.toString();
  };
  child.stdout.on("data", collect);
  child.stderr.on("data", collect);

  // Waiting on the port rather than on a log line: what Vite prints differs
  // between platforms and versions, and its output is only useful for the error
  // message when the server never comes up at all.
  const url = `http://127.0.0.1:${port}`;
  try {
    await waitFor("the Vite dev server", () =>
      fetch(url, { method: "GET" }).then(
        () => true,
        () => false,
      ),
    );
  } catch (err) {
    child.kill("SIGTERM");
    throw new Error(`${err.message}. Vite said:\n${output.trim() || "(nothing)"}`);
  }

  return {
    url,
    stop() {
      child.kill("SIGTERM");
    },
  };
}

/** A minimal DevTools protocol client over the browser endpoint. */
class Session {
  constructor(socket) {
    this.socket = socket;
    this.nextId = 1;
    this.pending = new Map();
    this.listeners = new Set();
    socket.addEventListener("message", (event) => {
      const message = JSON.parse(event.data);
      if (message.id && this.pending.has(message.id)) {
        const { resolve, reject } = this.pending.get(message.id);
        this.pending.delete(message.id);
        if (message.error) reject(new Error(`${message.error.message} (${message.method ?? ""})`));
        else resolve(message.result);
        return;
      }
      for (const listener of this.listeners) listener(message);
    });
  }

  on(listener) {
    this.listeners.add(listener);
  }

  send(method, params = {}, sessionId) {
    const id = this.nextId++;
    const payload = { id, method, params };
    if (sessionId) payload.sessionId = sessionId;
    this.socket.send(JSON.stringify(payload));
    return new Promise((resolve, reject) => this.pending.set(id, { resolve, reject }));
  }
}

/**
 * Launches headless Chrome and attaches to one fresh tab.
 *
 * extraArgs is where the fake media device flags go, which is what makes camera
 * switching testable on a machine with a single real webcam.
 */
export async function launchChrome(extraArgs = []) {
  const binary = findChrome();
  if (!binary) throw new Error("No Chrome or Chromium found. Set CHROME_PATH.");
  const profile = mkdtempSync(join(tmpdir(), "belo-headless-"));
  const child = spawn(
    binary,
    [
      "--headless=new",
      "--remote-debugging-port=0",
      `--user-data-dir=${profile}`,
      "--no-first-run",
      "--no-default-browser-check",
      "--disable-extensions",
      "--disable-background-networking",
      "--disable-sync",
      "--mute-audio",
      "--autoplay-policy=no-user-gesture-required",
      "--window-size=1280,900",
      // A CI container has a tiny /dev/shm and often no usable sandbox. Neither
      // flag is wanted on a developer machine, so neither is set there.
      ...(process.env.CI ? ["--no-sandbox", "--disable-dev-shm-usage"] : []),
      ...extraArgs,
    ],
    { stdio: ["ignore", "pipe", "pipe"] },
  );

  const portFile = join(profile, "DevToolsActivePort");
  const port = await waitFor("the Chrome debugging port", () => {
    if (!existsSync(portFile)) return null;
    const first = readFileSync(portFile, "utf8").split("\n")[0]?.trim();
    return first ? Number(first) : null;
  });

  const version = await fetch(`http://127.0.0.1:${port}/json/version`).then((r) => r.json());
  const socket = new WebSocket(version.webSocketDebuggerUrl);
  await new Promise((resolve, reject) => {
    socket.addEventListener("open", resolve, { once: true });
    socket.addEventListener("error", () => reject(new Error("Could not open the CDP socket")), {
      once: true,
    });
  });

  const session = new Session(socket);
  const { targetId } = await session.send("Target.createTarget", { url: "about:blank" });
  const { sessionId } = await session.send("Target.attachToTarget", { targetId, flatten: true });

  const logs = [];
  session.on((message) => {
    if (message.sessionId !== sessionId) return;
    if (message.method === "Runtime.consoleAPICalled") {
      const text = (message.params.args ?? [])
        .map((arg) => arg.value ?? arg.description ?? arg.type)
        .join(" ");
      logs.push(`[${message.params.type}] ${text}`);
    }
    if (message.method === "Runtime.exceptionThrown") {
      const details = message.params.exceptionDetails;
      logs.push(`[error] ${details.exception?.description ?? details.text}`);
    }
  });
  await session.send("Runtime.enable", {}, sessionId);
  await session.send("Page.enable", {}, sessionId);

  return {
    logs,
    async navigate(url) {
      await session.send("Page.navigate", { url }, sessionId);
    },
    /** Evaluates an expression in the page and returns its resolved value. */
    async evaluate(expression) {
      const result = await session.send(
        "Runtime.evaluate",
        { expression, awaitPromise: true, returnByValue: true },
        sessionId,
      );
      if (result.exceptionDetails) {
        throw new Error(
          result.exceptionDetails.exception?.description ?? result.exceptionDetails.text,
        );
      }
      return result.result.value;
    },
    /** Grants camera access up front so getUserMedia never waits on a prompt. */
    async grantCamera(origin) {
      await session.send("Browser.grantPermissions", { origin, permissions: ["videoCapture"] });
    },
    close() {
      try {
        socket.close();
      } catch {
        // The socket may already be gone when Chrome exited on its own.
      }
      child.kill("SIGTERM");
      // Chrome takes a moment to let go of its profile, and a leftover temp
      // directory is not worth failing a check over.
      setTimeout(() => {
        try {
          rmSync(profile, { recursive: true, force: true, maxRetries: 5, retryDelay: 200 });
        } catch {
          // The OS will reap it from the temp directory.
        }
      }, 500).unref();
    },
  };
}

/**
 * Loads a harness page and waits for it to publish window.__BELO_CHECK__.
 * The page runs its own assertions, because the things worth asserting (canvas
 * pixels, live tracks) are cheapest to read from inside the page.
 */
export async function runHarness({ page, extraArgs = [], timeoutMs = 180_000 }) {
  const server = await startDevServer();
  const chrome = await launchChrome(extraArgs);
  try {
    await chrome.grantCamera(server.url);
    await chrome.navigate(`${server.url}${page}`);
    const report = await waitFor(
      `${page} to report`,
      async () => {
        const value = await chrome.evaluate("window.__BELO_CHECK__ ?? null");
        return value && value.done ? value : null;
      },
      timeoutMs,
    );
    return { report, logs: chrome.logs };
  } catch (err) {
    return { error: err, logs: chrome.logs };
  } finally {
    chrome.close();
    server.stop();
  }
}

/** Prints a page report the same way the node checks print theirs. */
export function printReport(name, result) {
  const { report, logs, error } = result;
  if (error) {
    console.error(`FAIL ${name}: ${error.message}`);
    if (logs?.length) console.error(logs.slice(-40).join("\n"));
    return 1;
  }
  for (const line of report.results) {
    console.log(`${line.ok ? "ok  " : "FAIL"} ${line.name}${line.detail ? ` :: ${line.detail}` : ""}`);
  }
  const failures = report.results.filter((line) => !line.ok).length;
  if (report.error) {
    console.error(`\nthe page threw: ${report.error}`);
    if (logs?.length) console.error(logs.slice(-40).join("\n"));
    return 1;
  }
  console.log(failures === 0 ? `\n${report.results.length} checks passed` : `\n${failures} FAILED`);
  if (failures > 0 && logs?.length) console.error(logs.slice(-40).join("\n"));
  return failures === 0 ? 0 : 1;
}
