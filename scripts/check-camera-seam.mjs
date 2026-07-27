// Deterministic check of the camera seam against a mocked MediaDevices.
//
// The picker depends on behavior that a single-camera dev machine cannot show:
// device switching, stale mobile deviceIds, empty labels before permission and
// hot plug events. This exercises all of it in node.
//
//   node scripts/check-camera-seam.mjs
//
// Node 22.6 or newer runs the TypeScript source directly through type stripping.

class FakeError extends Error {
  constructor(name) {
    super(name);
    this.name = name;
  }
}

function track(deviceId, facingMode) {
  return {
    kind: "video",
    stopped: false,
    getSettings: () => ({ deviceId, facingMode }),
    stop() {
      this.stopped = true;
    },
  };
}

function stream(deviceId, facingMode) {
  const video = track(deviceId, facingMode);
  return { getTracks: () => [video], getVideoTracks: () => [video], video };
}

/** devices: [{deviceId,label,facing}], open(constraints) decides what happens. */
function mockNavigator(devices, open) {
  const listeners = new Map();
  const calls = [];
  const nav = {
    mediaDevices: {
      async getUserMedia(constraints) {
        calls.push(constraints.video);
        return open(constraints.video, calls.length);
      },
      async enumerateDevices() {
        return devices;
      },
      addEventListener(type, fn) {
        listeners.set(type, (listeners.get(type) ?? new Set()).add(fn));
      },
      removeEventListener(type, fn) {
        listeners.get(type)?.delete(fn);
      },
    },
  };
  return { nav, calls, listeners };
}

let failures = 0;
const results = [];
function check(name, ok, detail = "") {
  results.push(`${ok ? "ok  " : "FAIL"} ${name}${detail ? ` :: ${detail}` : ""}`);
  if (!ok) failures += 1;
}

// Node ships a read-only navigator of its own, so it is swapped by descriptor.
const realNavigator = Object.getOwnPropertyDescriptor(globalThis, "navigator");

async function withNavigator(nav, run) {
  Object.defineProperty(globalThis, "navigator", { value: nav, configurable: true });
  try {
    return await run();
  } finally {
    if (realNavigator) Object.defineProperty(globalThis, "navigator", realNavigator);
    else delete globalThis.navigator;
  }
}

const cam = (await import(new URL("../src/lib/capture.ts", import.meta.url))).camera;

// 1. Labels are empty before a permission is granted: names must still be usable.
{
  const { nav } = mockNavigator(
    [
      { kind: "videoinput", deviceId: "a", label: "" },
      { kind: "videoinput", deviceId: "b", label: "" },
      { kind: "audioinput", deviceId: "mic", label: "" },
    ],
    () => stream("a"),
  );
  const list = await withNavigator(nav, () => cam.list());
  check(
    "unlabeled devices fall back to Camera 1/2 and drop non video inputs",
    list.length === 2 && list[0].label === "Camera 1" && list[1].label === "Camera 2",
    JSON.stringify(list),
  );
}

// 2. Real labels survive and front/back is inferred for the mobile retry path.
{
  const { nav } = mockNavigator(
    [
      { kind: "videoinput", deviceId: "f1", label: "Front Camera" },
      { kind: "videoinput", deviceId: "b1", label: "Back Dual Wide Camera" },
      { kind: "videoinput", deviceId: "u1", label: "Elgato Cam Link" },
    ],
    () => stream("f1"),
  );
  const list = await withNavigator(nav, () => cam.list());
  check(
    "labels are kept and facing is inferred",
    list[0].label === "Front Camera" &&
      list[0].facing === "front" &&
      list[1].facing === "back" &&
      list[2].facing === null,
    JSON.stringify(list),
  );
}

// 3. A specific pick opens exactly that device.
{
  const { nav, calls } = mockNavigator([], (video) => stream(video.deviceId.exact));
  const opened = await withNavigator(nav, () => cam.start("b1"));
  check(
    "start(deviceId) constrains on the exact device",
    calls.length === 1 && calls[0].deviceId.exact === "b1" && opened.getVideoTracks()[0].getSettings().deviceId === "b1",
    JSON.stringify(calls),
  );
}

// 4. Mobile: a stale deviceId retries on facingMode, which stays stable there.
{
  const { nav, calls } = mockNavigator(
    [
      { kind: "videoinput", deviceId: "old-back", label: "Back Camera" },
      { kind: "videoinput", deviceId: "old-front", label: "Front Camera" },
    ],
    (video, n) => {
      if (n === 1) throw new FakeError("OverconstrainedError");
      return stream("fresh-back", video.facingMode?.exact);
    },
  );
  const opened = await withNavigator(nav, async () => {
    await cam.list(); // learns that old-back faces backwards
    return cam.start("old-back");
  });
  check(
    "a stale deviceId retries on facingMode",
    calls.length === 2 &&
      calls[1].facingMode?.exact === "environment" &&
      opened.getVideoTracks()[0].getSettings().facingMode === "environment",
    JSON.stringify(calls),
  );
}

// 5. An unknown missing device still ends up on the default camera.
{
  const { nav, calls } = mockNavigator([], (video, n) => {
    if (n === 1) throw new FakeError("NotFoundError");
    if (video.deviceId) throw new Error("second attempt must not be constrained");
    return stream("default");
  });
  const opened = await withNavigator(nav, () => cam.start("unplugged"));
  check(
    "an unplugged device falls back to the default camera",
    calls.length === 2 && !calls[1].deviceId && opened.getVideoTracks()[0].getSettings().deviceId === "default",
    JSON.stringify(calls),
  );
}

// 6. A denial must surface instead of quietly opening some other camera.
{
  const { nav, calls } = mockNavigator([], () => {
    throw new FakeError("NotAllowedError");
  });
  let name = "none";
  await withNavigator(nav, async () => {
    try {
      await cam.start("a");
    } catch (err) {
      name = err.name;
    }
  });
  check("a denied permission is not retried away", name === "NotAllowedError" && calls.length === 1);
}

// 7. Simulator: no camera at all keeps reporting NotFoundError.
{
  const { nav } = mockNavigator([], () => {
    throw new FakeError("NotFoundError");
  });
  let name = "none";
  await withNavigator(nav, async () => {
    try {
      await cam.start(null);
    } catch (err) {
      name = err.name;
    }
  });
  check("no camera present still reports NotFoundError", name === "NotFoundError");
}

// 8. Stopping a stream stops every track, and the active id is readable.
{
  const live = stream("b1");
  const { nav } = mockNavigator([], () => live);
  await withNavigator(nav, async () => {
    check("activeDeviceId reads the running track", cam.activeDeviceId(live) === "b1");
    check("activeDeviceId of nothing is null", cam.activeDeviceId(null) === null);
    cam.stop(live);
  });
  check("stop() stops the tracks", live.video.stopped === true);
}

// 9. Hot plug: the subscription is registered and cleanly removed.
{
  const { nav, listeners } = mockNavigator([], () => stream("a"));
  await withNavigator(nav, () => {
    let fired = 0;
    const off = cam.onDevicesChanged(() => {
      fired += 1;
    });
    for (const fn of listeners.get("devicechange")) fn();
    off();
    const stillThere = listeners.get("devicechange").size;
    check("devicechange fires the handler and unsubscribes", fired === 1 && stillThere === 0);
  });
}

// 10. isSupported is false without a webview media API.
check("isSupported is false without mediaDevices", cam.isSupported() === false);

console.log(results.join("\n"));
console.log(failures === 0 ? `\n${results.length} checks passed` : `\n${failures} FAILED`);
process.exit(failures === 0 ? 0 : 1);
