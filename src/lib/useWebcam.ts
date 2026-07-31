import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { camera, type CameraFacing, type CameraInfo } from "./capture";
import { screen } from "./screen";

export type CameraStatus = "off" | "starting" | "live" | "error";
export type { CameraInfo };

/** Which live feed a view is showing. Nothing below this line is camera only. */
export type FeedKind = "camera" | "screen";

/** One entry of whatever the platform offers for the current feed. */
export interface FeedOption {
  id: string;
  label: string;
  /** Cameras only, and a guess even there. */
  facing?: CameraFacing | null;
}

const PREF_KEY: Record<FeedKind, string> = {
  camera: "belo.camera.deviceId",
  screen: "belo.screen.displayId",
};

/** The picked device is app wide, not per project, so it lives in localStorage. */
function readPreferred(kind: FeedKind): string | null {
  try {
    return window.localStorage.getItem(PREF_KEY[kind]) || null;
  } catch {
    return null;
  }
}

function writePreferred(kind: FeedKind, id: string | null) {
  try {
    if (id) window.localStorage.setItem(PREF_KEY[kind], id);
    else window.localStorage.removeItem(PREF_KEY[kind]);
  } catch {
    // A webview with storage disabled just forgets the choice.
  }
}

/**
 * What the hook needs from a feed. The two seams, capture.ts and screen.ts,
 * differ in vocabulary and in what the platform can enumerate, and this is the
 * only place that knows about the difference.
 */
interface FeedAdapter {
  start(id: string | null): Promise<MediaStream>;
  stop(stream: MediaStream | null): void;
  list(): Promise<FeedOption[]>;
  activeId(stream: MediaStream | null): string | null;
  onChanged(handler: () => void): () => void;
  /** Plain language for a failure, in the words of this feed. */
  describe(err: unknown): string;
}

const cameraAdapter: FeedAdapter = {
  start: (id) => camera.start(id),
  stop: (stream) => camera.stop(stream),
  list: async () =>
    (await camera.list()).map((entry) => ({
      id: entry.deviceId,
      label: entry.label,
      facing: entry.facing,
    })),
  activeId: (stream) => camera.activeDeviceId(stream),
  onChanged: (handler) => camera.onDevicesChanged(handler),
  describe(err) {
    if (err instanceof Error && err.name === "NotAllowedError") {
      return "Camera access was denied. Allow it in your system privacy settings and try again.";
    }
    if (err instanceof Error && (err.name === "NotFoundError" || err.name === "DevicesNotFoundError")) {
      return "No camera is available. The iOS Simulator has no camera, so import photos instead.";
    }
    return err instanceof Error ? err.message : "The camera could not be started";
  },
};

const screenAdapter: FeedAdapter = {
  start: (id) => screen.start(id),
  stop: (stream) => screen.stop(stream),
  list: async () =>
    screen.canList()
      ? (await screen.list()).map((entry) => ({
          id: entry.id,
          label: `${entry.label} (${entry.width} by ${entry.height})`,
        }))
      : [],
  activeId: (stream) => screen.activeDisplayId(stream),
  // Displays do not come and go the way USB cameras do, and no platform here
  // offers a change event for them, so the list is read at every start.
  onChanged: () => () => undefined,
  describe(err) {
    if (err instanceof Error && err.name === "NotAllowedError") {
      return "Belo is not allowed to record the screen yet.";
    }
    return err instanceof Error ? err.message : "The screen feed could not be started";
  },
};

const ADAPTERS: Record<FeedKind, FeedAdapter> = { camera: cameraAdapter, screen: screenAdapter };

/**
 * Owns one live session on behalf of a view, camera or screen. The stream is
 * always torn down when the component unmounts, when the caller flips active to
 * false and when the feed changes, so neither the camera indicator nor the
 * screen recording indicator stays lit after leaving a view. All platform calls
 * go through the two seams.
 *
 * Selecting another camera or display restarts the stream in place: the old
 * tracks are stopped, the new device is opened and the view keeps every bit of
 * its own state. Camera labels are empty until a permission is granted, so the
 * list is only read after the first successful start and again on every
 * devicechange. Each feed remembers its own pick, which is why switching to the
 * screen and back returns to the camera that was running.
 */
export function useLiveSource(active: boolean, kind: FeedKind = "camera") {
  const adapter = ADAPTERS[kind];
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [status, setStatus] = useState<CameraStatus>("off");
  const [error, setError] = useState<string | null>(null);
  const [refused, setRefused] = useState(false);
  // Both feeds are remembered at once, and which one is read is decided during
  // the render that switches. Anything reset in an effect would open the screen
  // with a camera's deviceId first and then restart, which is a second capture
  // session nobody asked for.
  const [wantedByKind, setWantedByKind] = useState<Record<FeedKind, string | null>>(() => ({
    camera: readPreferred("camera"),
    screen: readPreferred("screen"),
  }));
  const wantedId = wantedByKind[kind];
  const [listed, setListed] = useState<{ kind: FeedKind; options: FeedOption[] }>({
    kind,
    options: [],
  });
  const options = listed.kind === kind ? listed.options : [];
  const [opened, setOpened] = useState<{ kind: FeedKind; id: string | null }>({ kind, id: null });
  const activeId = opened.kind === kind ? opened.id : null;
  const [switching, setSwitching] = useState(false);
  const [attempt, setAttempt] = useState(0);
  const requestedRef = useRef<string | null>(wantedId);
  const shownRef = useRef<string | null>(wantedId);
  const shown = wantedId ?? activeId;
  useEffect(() => {
    requestedRef.current = wantedId;
    shownRef.current = shown;
  }, [wantedId, shown]);

  const stop = useCallback(() => {
    adapter.stop(streamRef.current);
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
  }, [adapter]);

  useEffect(() => {
    let cancelled = false;

    if (!active) {
      stop();
      setStatus("off");
      setError(null);
      setRefused(false);
      setOpened({ kind, id: null });
      setSwitching(false);
      return;
    }

    setStatus("starting");
    setError(null);
    setRefused(false);

    (async () => {
      try {
        // Release the previous feed first: some platforms refuse to open a
        // second camera while the first one still holds the capture session,
        // and a screen capture left running keeps its indicator lit.
        stop();
        const stream = await adapter.start(wantedId);
        if (cancelled) {
          adapter.stop(stream);
          return;
        }
        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play().catch(() => undefined);
        }
        setStatus("live");
        setSwitching(false);
        // What actually opened can differ from what was asked for, because an
        // unplugged camera or a detached display falls back to the default.
        setOpened({ kind, id: adapter.activeId(stream) });

        const list = await adapter.list();
        if (!cancelled) setListed({ kind, options: list });
      } catch (err) {
        if (cancelled) return;
        setStatus("error");
        setSwitching(false);
        setRefused(err instanceof Error && err.name === "NotAllowedError");
        setError(adapter.describe(err));
      }
    })();

    return () => {
      cancelled = true;
      stop();
    };
  }, [active, adapter, kind, wantedId, attempt, stop]);

  // Hot plug: a USB camera arriving or leaving refreshes the list, and losing
  // the camera in use drops the request back to the platform default.
  useEffect(() => {
    if (!active) return;
    let cancelled = false;
    const unsubscribe = adapter.onChanged(() => {
      void adapter
        .list()
        .then((list) => {
          if (cancelled) return;
          setListed({ kind, options: list });
          const wanted = requestedRef.current;
          if (wanted && list.length > 0 && !list.some((entry) => entry.id === wanted)) {
            setWantedByKind((prev) => ({ ...prev, [kind]: null }));
          }
        })
        .catch(() => undefined);
    });
    return () => {
      cancelled = true;
      unsubscribe();
    };
  }, [active, adapter, kind]);

  const select = useCallback(
    (next: string | null) => {
      // Picking the device that is already running must not restart the stream.
      if (shownRef.current === next) {
        writePreferred(kind, next);
        return;
      }
      if (streamRef.current) setSwitching(true);
      writePreferred(kind, next);
      setWantedByKind((prev) => ({ ...prev, [kind]: next }));
    },
    [kind],
  );

  const retry = useCallback(() => setAttempt((n) => n + 1), []);

  return {
    videoRef,
    status,
    error,
    /** True when the platform refused, which is a settings problem, not a fault. */
    refused,
    options,
    /** The entry the picker should show as chosen. */
    optionId: shown,
    select,
    switching,
    retry,
  };
}

/**
 * The camera session, for the views that only ever show a camera. A thin name
 * over useLiveSource so the capture modal keeps reading the way it always did.
 */
export function useWebcam(active: boolean) {
  const feed = useLiveSource(active, "camera");
  const cameras = useMemo<CameraInfo[]>(
    () =>
      feed.options.map((entry) => ({
        deviceId: entry.id,
        label: entry.label,
        facing: entry.facing ?? null,
      })),
    [feed.options],
  );
  return {
    videoRef: feed.videoRef,
    status: feed.status,
    error: feed.error,
    cameras,
    deviceId: feed.optionId,
    selectCamera: feed.select,
    switching: feed.switching,
    retry: feed.retry,
  };
}
