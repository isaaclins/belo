// Native capture HUD smoke check.
// Unlike check:hud, this launches the actual Tauri app and queries the native
// WebviewWindow from a real webview. The test instance is hidden, unfocused,
// and uses a dedicated Vite port so it never touches a user's app session.

import { execFileSync, spawn } from "node:child_process";
import { closeSync, existsSync, openSync, readFileSync, unlinkSync, writeFileSync } from "node:fs";
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
        title: "Belo HUD real check",
        width: 1,
        height: 1,
        x: -10000,
        y: -10000,
        visible: false,
        focus: false,
        decorations: false,
        resizable: false,
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
  const deadline = Date.now() + 45_000;
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
  vite = spawn(process.execPath, [vitePath, "--host", "127.0.0.1", "--port", String(port), "--strictPort"], {
    cwd: root,
    env: process.env,
    detached: true,
    stdio: ["ignore", logFd, logFd],
  });
  await waitForServer();

  tauri = spawn(process.execPath, [tauriPath, "dev", "--config", configPath, "--no-watch", "--no-dev-server-wait"], {
    cwd: root,
    env: { ...process.env, BELO_HUD_CHECK_REPORT: reportPath },
    detached: true,
    stdio: ["ignore", logFd, logFd],
  });
  const result = await waitForReport();
  const window = result.window ?? {};
  const sizeOk = Math.abs(Number(window.logicalWidth) - 340) <= 2 && Math.abs(Number(window.logicalHeight) - 280) <= 2;
  const ok =
    result.stage === "closed" &&
    result.closed === true &&
    window.exists === true &&
    window.alwaysOnTop === true &&
    window.visible === true &&
    window.focused === false &&
    window.decorated === false &&
    window.resizable === false &&
    Number(window.positionX) < -1000 &&
    Number(window.positionY) < -1000 &&
    sizeOk;

  console.log(JSON.stringify({
    check: "check:hud-real",
    port,
    ok,
    stage: result.stage,
    closed: result.closed ?? false,
    window,
    errors: result.errors ?? [],
  }, null, 2));
  process.exitCode = ok ? 0 : 1;
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
}
