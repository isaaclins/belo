import { useCallback, useEffect, useRef, useState } from "react";
import { camera, type CameraInfo } from "./capture";

export type CameraStatus = "off" | "starting" | "live" | "error";
export type { CameraInfo };

/**
 * Owns one camera session on behalf of a view. The stream is always torn down
 * when the component unmounts or the caller flips active to false, so the
 * camera indicator never stays lit after leaving a view. All platform calls go
 * through the capture module.
 */
export function useWebcam(active: boolean) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [status, setStatus] = useState<CameraStatus>("off");
  const [error, setError] = useState<string | null>(null);
  const [cameras, setCameras] = useState<CameraInfo[]>([]);
  const [deviceId, setDeviceId] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);

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
      return;
    }

    setStatus("starting");
    setError(null);

    (async () => {
      try {
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

        const devices = await camera.list();
        if (!cancelled) setCameras(devices);
      } catch (err) {
        if (cancelled) return;
        setStatus("error");
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

  const retry = useCallback(() => setAttempt((n) => n + 1), []);

  return { videoRef, status, error, cameras, deviceId, setDeviceId, retry };
}
