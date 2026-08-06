// Native capture HUD trigger check.
//
// This launches a real Tauri webview on a dedicated Vite port, mounts the real
// Use view, starts the real macOS screen source, and blurs the real main window
// through Tauri APIs. The HUD is created only by useScreenHud. No test code
// constructs belo-hud.

import { execFileSync, spawn } from "node:child_process";
import {
  closeSync,
  existsSync,
  mkdtempSync,
  openSync,
  readFileSync,
  rmSync,
  unlinkSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { setTimeout as delay } from "node:timers/promises";
import { fileURLToPath } from "node:url";

if (process.platform !== "darwin") {
  console.log("skipped: the real HUD check requires macOS Tauri windows");
  process.exit(0);
}

const root = fileURLToPath(new URL("..", import.meta.url));
const port = 15000 + (process.pid % 1000);
const reportPath = join(tmpdir(), `belo-hud-real-${process.pid}.json`);
const configPath = join(tmpdir(), `belo-hud-real-${process.pid}.json5`);
const homePath = mkdtempSync(join(tmpdir(), `belo-hud-home-${process.pid}-`));
const vitePath = join(root, "node_modules", "vite", "bin", "vite.js");
const tauriPath = join(root, "node_modules", "@tauri-apps", "cli", "tauri.js");
const logPath = join(tmpdir(), `belo-hud-real-${process.pid}.log`);

const config = {
  build: {
    devUrl: `http://127.0.0.1:${port}/scripts/browser/hud-real.html`,
    beforeDevCommand: "true",
  },
  app: {
    windows: [
      {
        label: "main",
        title: "Belo HUD real trigger check",
        width: 600,
        height: 400,
        x: -10000,
        y: -10000,
        visible: true,
        focus: false,
        decorations: false,
        resizable: false,
        skipTaskbar: true,
      },
    ],
  },
};

writeFileSync(configPath, JSON.stringify(config));
try {
  unlinkSync(reportPath);
} catch {
  // A stale report cannot make a new run pass.
}

let vite;
let tauri;
let logFd;

function killGroup(child) {
  if (!child?.pid) return;
  try {
    process.kill(-child.pid, "SIGTERM");
  } catch {
    child.kill("SIGTERM");
  }
}

async function waitForServer() {
  const url = `http://127.0.0.1:${port}/scripts/browser/hud-real.html`;
  const deadline = Date.now() + 20_000;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(url);
      if (response.ok) return;
    } catch {
      // Vite is still starting.
    }
    await delay(100);
  }
  throw new Error(`Vite did not start on ${port}`);
}

async function waitForReport() {
  const deadline = Date.now() + 90_000;
  while (Date.now() < deadline) {
    if (existsSync(reportPath)) {
      try {
        const result = JSON.parse(readFileSync(reportPath, "utf8"));
        if (result.stage === "closed" || result.stage === "error") return result;
      } catch {
        // The Rust command may still be replacing the file.
      }
    }
    await delay(100);
  }
  throw new Error(`Tauri HUD check produced no report; see ${logPath}`);
}

try {
  execFileSync(process.execPath, [join(root, "scripts", "copy-wasm.mjs")], {
    cwd: root,
    stdio: "inherit",
  });

  logFd = openSync(logPath, "w");
  // HOME is redirected so the check never touches real project data, but the
  // Rust toolchain still has to be found in the real home.
  const env = {
    ...process.env,
    HOME: homePath,
    RUSTUP_HOME: process.env.RUSTUP_HOME ?? join(process.env.HOME ?? "", ".rustup"),
    CARGO_HOME: process.env.CARGO_HOME ?? join(process.env.HOME ?? "", ".cargo"),
    BELO_HUD_CHECK_REPORT: reportPath,
  };
  vite = spawn(process.execPath, [vitePath, "--host", "127.0.0.1", "--port", String(port), "--strictPort"], {
    cwd: root,
    env,
    detached: true,
    stdio: ["ignore", logFd, logFd],
  });
  await waitForServer();

  tauri = spawn(process.execPath, [tauriPath, "dev", "--config", configPath, "--no-watch", "--no-dev-server-wait"], {
    cwd: root,
    env,
    detached: true,
    stdio: ["ignore", logFd, logFd],
  });
  const result = await waitForReport();
  const window = result.created ?? {};
  const diagnostics = readFileSync(logPath, "utf8")
    .split("\n")
    .filter((line) => line.includes("[belo-hud]"));
  const ok =
    result.stage === "closed" &&
    // The OS refocus step is skipped when macOS refuses to focus the offscreen
    // window of an inactive app; the toggle path covers hiding in that case.
    (result.refocusClosed === true || result.refocusSkipped === true) &&
    result.toggleClosed === true &&
    result.stopClosed === true &&
    window.exists === true &&
    window.visible === true &&
    window.alwaysOnTop === true &&
    window.focused === false &&
    window.decorated === false &&
    window.resizable === false;

  console.log(JSON.stringify({
    check: "check:hud-real",
    port,
    ok,
    stage: result.stage,
    created: window,
    refocusClosed: result.refocusClosed ?? false,
    refocusSkipped: result.refocusSkipped ?? false,
    toggleClosed: result.toggleClosed ?? false,
    stopClosed: result.stopClosed ?? false,
    errors: result.errors ?? [],
    diagnostics,
    // Kept verbatim so a failure explains itself instead of only timing out.
    failureContext: ok
      ? undefined
      : {
          hudExists: result.hudExists,
          hudVisible: result.hudVisible,
          mainFocused: result.mainFocused,
          buttons: result.buttons,
          video: result.video,
        },
  }, null, 2));
  if (!ok) process.exitCode = 1;
} catch (error) {
  console.error(`check:hud-real failed: ${error instanceof Error ? error.message : String(error)}`);
  process.exitCode = 1;
} finally {
  killGroup(tauri);
  killGroup(vite);
  if (logFd !== undefined) {
    try {
      closeSync(logFd);
    } catch {
      // Best effort cleanup.
    }
  }
  try {
    unlinkSync(configPath);
  } catch {
    // Best effort cleanup.
  }
  try {
    unlinkSync(reportPath);
  } catch {
    // Best effort cleanup.
  }
  rmSync(homePath, { recursive: true, force: true });
}
