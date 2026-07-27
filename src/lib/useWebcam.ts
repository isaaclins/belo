import { useCallback, useEffect, useRef, useState } from "react";
import { camera, type CameraInfo } from "./capture";

export type CameraStatus = "off" | "starting" | "live" | "error";
export type { CameraInfo };

const PREF_KEY = "belo.camera.deviceId";

/** The picked camera is app wide, not per project, so it lives in localStorage. */
function readPreferred(): string | null {
  try {
    return window.localStorage.getItem(PREF_KEY) || null;
  } catch {
    return null;
  }
}

function writePreferred(deviceId: string | null) {
  try {
    if (deviceId) window.localStorage.setItem(PREF_KEY, deviceId);
    else window.localStorage.removeItem(PREF_KEY);
  } catch {
    // A webview with storage disabled just forgets the choice.
  }
}

/**
 * Owns one camera session on behalf of a view. The stream is always torn down
 * when the component unmounts or the caller flips active to false, so the
 * camera indicator never stays lit after leaving a view. All platform calls go
 * through the capture module.
 *
 * Selecting another camera restarts the stream in place: the old tracks are
 * stopped, the new device is opened and the view keeps every bit of its own
 * state. Device labels are empty until a permission is granted, so the list is
 * only read after the first successful start and again on every devicechange.
 */
export function useWebcam(active: boolean) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [status, setStatus] = useState<CameraStatus>("off");
  const [error, setError] = useState<string | null>(null);
  const [cameras, setCameras] = useState<CameraInfo[]>([]);
  const [deviceId, setDeviceId] = useState<string | null>(() => readPreferred());
  const [activeDeviceId, setActiveDeviceId] = useState<string | null>(null);
  const [switching, setSwitching] = useState(false);
  const [attempt, setAttempt] = useState(0);
  const requestedRef = useRef<string | null>(deviceId);
  const shownRef = useRef<string | null>(deviceId);
  const shown = deviceId ?? activeDeviceId;
  useEffect(() => {
    requestedRef.current = deviceId;
    shownRef.current = shown;
  }, [deviceId, shown]);

  const stop = useCallback(() => {
    camera.stop(streamRef.current);
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
  }, []);

  useEffect(() => {
    let cancelled = false;

    if (!active) {
      stop();
      setStatus("off");
      setError(null);
      setActiveDeviceId(null);
      setSwitching(false);
      return;
    }

    setStatus("starting");
    setError(null);

    (async () => {
      try {
        // Release the previous device first: some platforms refuse to open a
        // second camera while the first one still holds the capture session.
        stop();
        const stream = await camera.start(deviceId);
        if (cancelled) {
          camera.stop(stream);
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
        // unplugged or stale deviceId falls back to the default camera.
        setActiveDeviceId(camera.activeDeviceId(stream));

        const devices = await camera.list();
        if (!cancelled) setCameras(devices);
      } catch (err) {
        if (cancelled) return;
        setStatus("error");
        setSwitching(false);
        const message =
          err instanceof Error && err.name === "NotAllowedError"
            ? "Camera access was denied. Allow it in your system privacy settings and try again."
            : err instanceof Error && (err.name === "NotFoundError" || err.name === "DevicesNotFoundError")
              ? "No camera is available. The iOS Simulator has no camera, so import photos instead."
              : err instanceof Error
                ? err.message
                : "The camera could not be started";
        setError(message);
      }
    })();

    return () => {
      cancelled = true;
      stop();
    };
  }, [active, deviceId, attempt, stop]);

  // Hot plug: a USB camera arriving or leaving refreshes the list, and losing
  // the camera in use drops the request back to the platform default.
  useEffect(() => {
    if (!active) return;
    let cancelled = false;
    const unsubscribe = camera.onDevicesChanged(() => {
      void camera
        .list()
        .then((devices) => {
          if (cancelled) return;
          setCameras(devices);
          const wanted = requestedRef.current;
          if (wanted && devices.length > 0 && !devices.some((d) => d.deviceId === wanted)) {
            setDeviceId(null);
          }
        })
        .catch(() => undefined);
    });
    return () => {
      cancelled = true;
      unsubscribe();
    };
  }, [active]);

  const selectCamera = useCallback((next: string | null) => {
    // Picking the camera that is already running must not restart the stream.
    if (shownRef.current === next) {
      writePreferred(next);
      return;
    }
    if (streamRef.current) setSwitching(true);
    writePreferred(next);
    setDeviceId(next);
  }, []);

  const retry = useCallback(() => setAttempt((n) => n + 1), []);

  return {
    videoRef,
    status,
    error,
    cameras,
    /** The camera the picker should show as chosen. */
    deviceId: shown,
    selectCamera,
    switching,
    retry,
  };
}
