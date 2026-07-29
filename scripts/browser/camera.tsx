// End to end check of camera switching, in a real browser with real streams.
//
// Chrome's fake capture devices give this machine the second and third camera it
// does not have, so the hook that owns a camera session (src/lib/useWebcam.ts)
// and the seam underneath it (src/lib/capture.ts) can be driven the way a user
// drives the picker: start, switch mid stream, unplug, come back.
//
// Everything under test is the app's own code. The harness only renders the hook
// into a video element and records what it reported.
//
//   npm run check:camera-live

import { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { camera } from "../../src/lib/capture";
import { useWebcam } from "../../src/lib/useWebcam";
import { reporter } from "./tauri-shim";

const { check, fail, finish } = reporter();
const PREF_KEY = "belo.camera.deviceId";

type Hook = ReturnType<typeof useWebcam>;

const live: {
  hook: Hook | null;
  setActive: ((on: boolean) => void) | null;
  switchingSeen: boolean;
  statuses: string[];
} = { hook: null, setActive: null, switchingSeen: false, statuses: [] };

function Harness() {
  const [active, setActive] = useState(false);
  const hook = useWebcam(active);
  live.hook = hook;
  live.setActive = setActive;
  if (hook.switching) live.switchingSeen = true;
  if (live.statuses[live.statuses.length - 1] !== hook.status) live.statuses.push(hook.status);
  useEffect(() => {
    // The picker only exists while the camera runs, so the video element has to
    // be mounted before the hook opens a stream, exactly like in UseView.
  }, []);
  return <video id="feed" ref={hook.videoRef} playsInline muted autoPlay />;
}

const wait = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

async function until(what: string, probe: () => boolean, timeoutMs = 15_000) {
  const deadline = Date.now() + timeoutMs;
  while (!probe()) {
    if (Date.now() > deadline) throw new Error(`Timed out waiting for ${what}`);
    await wait(50);
  }
}

const videoEl = () => document.getElementById("feed") as HTMLVideoElement;
const currentStream = () => videoEl().srcObject as MediaStream | null;

async function main() {
  createRoot(document.getElementById("mount")!).render(<Harness />);
  await until("the harness to mount", () => Boolean(live.setActive));

  // 1. Start the camera the way the Start button does.
  live.setActive!(true);
  await until("the camera to go live", () => live.hook!.status === "live");
  const first = currentStream();
  check("the camera reaches live with an active track", first?.getVideoTracks()[0]?.readyState === "live", {
    status: live.hook!.status,
    tracks: first?.getVideoTracks().length,
  });

  // 2. The picker's device list comes from the seam, after the first start.
  await until("the device list", () => live.hook!.cameras.length > 0);
  const cameras = live.hook!.cameras;
  check("more than one camera is enumerated through the seam", cameras.length >= 2, {
    cameras: cameras.map((c) => `${c.deviceId.slice(0, 8)}:${c.label}`),
  });
  const activeId = camera.activeDeviceId(first);
  check("the running stream reports which device it opened", Boolean(activeId), { activeId });

  // 3. Switch mid stream, which is the case a single camera machine cannot show.
  const other = cameras.find((c) => c.deviceId !== activeId) ?? cameras[1];
  const oldTracks = first!.getVideoTracks();
  live.switchingSeen = false;
  live.hook!.selectCamera(other.deviceId);
  await until("the switch to settle", () => live.hook!.status === "live" && currentStream() !== first);
  const second = currentStream();
  check("switching reports a switch in progress", live.switchingSeen, { statuses: live.statuses });
  check("the old tracks are stopped", oldTracks.every((track) => track.readyState === "ended"), {
    states: oldTracks.map((track) => track.readyState),
  });
  check("the new stream is live", second?.getVideoTracks()[0]?.readyState === "live", {
    tracks: second?.getVideoTracks().map((track) => track.readyState),
  });
  check(
    "the new stream opened the requested device",
    camera.activeDeviceId(second) === other.deviceId,
    { wanted: other.deviceId.slice(0, 12), got: camera.activeDeviceId(second)?.slice(0, 12) },
  );
  check("the video element shows the new stream", videoEl().srcObject === second, {
    paused: videoEl().paused,
  });
  check("the pick is persisted for the next launch", localStorage.getItem(PREF_KEY) === other.deviceId, {
    stored: localStorage.getItem(PREF_KEY)?.slice(0, 12),
  });

  // 4. Picking the camera that already runs must not restart the stream.
  live.hook!.selectCamera(other.deviceId);
  await wait(400);
  check("re-picking the running camera keeps the same stream", currentStream() === second, {
    same: currentStream() === second,
  });

  // 5. Hot plug: a devicechange refreshes the list through the seam.
  const seen = live.hook!.cameras.length;
  navigator.mediaDevices.dispatchEvent(new Event("devicechange"));
  await wait(500);
  check("a devicechange keeps the list populated", live.hook!.cameras.length === seen, {
    before: seen,
    after: live.hook!.cameras.length,
  });

  // 6. The failure path: the chosen camera disappears. enumerateDevices is
  //    shadowed so the device is really gone from the platform's point of view,
  //    then the real devicechange path has to drop the stale pick and fall back.
  const realEnumerate = navigator.mediaDevices.enumerateDevices.bind(navigator.mediaDevices);
  navigator.mediaDevices.enumerateDevices = async () =>
    (await realEnumerate()).filter((device) => device.deviceId !== other.deviceId);
  navigator.mediaDevices.dispatchEvent(new Event("devicechange"));
  await until(
    "the removed camera to be dropped",
    () => !live.hook!.cameras.some((c) => c.deviceId === other.deviceId),
  );
  await until("the fallback stream", () => live.hook!.status === "live" && currentStream() !== second);
  const third = currentStream();
  check("losing the chosen camera falls back to a live stream", third?.getVideoTracks()[0]?.readyState === "live", {
    status: live.hook!.status,
    error: live.hook!.error,
  });
  check("the stream of the removed camera is stopped", second!.getVideoTracks().every((t) => t.readyState === "ended"), {
    states: second!.getVideoTracks().map((t) => t.readyState),
  });
  check(
    "the picker no longer offers the removed camera",
    !live.hook!.cameras.some((c) => c.deviceId === other.deviceId),
    { cameras: live.hook!.cameras.length },
  );
  check(
    "the remembered pick survives the removal, so a returning camera is honored",
    localStorage.getItem(PREF_KEY) === other.deviceId,
    { stored: localStorage.getItem(PREF_KEY)?.slice(0, 12) },
  );
  navigator.mediaDevices.enumerateDevices = realEnumerate;

  // 7. A stale deviceId from a previous session, the mobile case: the seam has
  //    to fall back instead of failing to start.
  live.setActive!(false);
  await until("the camera to stop", () => live.hook!.status === "off");
  check("stopping the camera ends every track", third!.getVideoTracks().every((t) => t.readyState === "ended"), {
    states: third!.getVideoTracks().map((t) => t.readyState),
  });
  localStorage.setItem(PREF_KEY, "a-device-that-no-longer-exists");
  const stale = await camera
    .start("a-device-that-no-longer-exists")
    .then((stream) => ({ ok: true, stream }))
    .catch((err: Error) => ({ ok: false, name: err.name }));
  check("a stale deviceId still opens the default camera", "stream" in stale, stale);
  if ("stream" in stale) {
    check(
      "the fallback opened a camera the platform actually has",
      Boolean(camera.activeDeviceId(stale.stream as MediaStream)),
      { deviceId: camera.activeDeviceId(stale.stream as MediaStream)?.slice(0, 12) },
    );
    camera.stop(stale.stream as MediaStream);
  }
  localStorage.removeItem(PREF_KEY);
}

main()
  .then(finish)
  .catch((err) => {
    console.error(err);
    fail(err);
  });
