import { useCallback, useEffect, useRef, useState } from "react";

export type CameraStatus = "off" | "starting" | "live" | "error";

export interface CameraInfo {
  deviceId: string;
  label: string;
}

/**
 * Owns one getUserMedia session. The stream is always torn down when the
 * component unmounts or the caller flips active to false, so the camera
 * indicator never stays lit after leaving a view.
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
    streamRef.current?.getTracks().forEach((track) => track.stop());
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
        if (!navigator.mediaDevices?.getUserMedia) {
          throw new Error("This webview does not expose a camera API");
        }
        const stream = await navigator.mediaDevices.getUserMedia({
          video: deviceId
            ? { deviceId: { exact: deviceId }, width: { ideal: 1280 }, height: { ideal: 720 } }
            : { width: { ideal: 1280 }, height: { ideal: 720 } },
          audio: false,
        });
        if (cancelled) {
          stream.getTracks().forEach((track) => track.stop());
          return;
        }
        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play().catch(() => undefined);
        }
        setStatus("live");

        // Device labels only become readable once permission is granted.
        const devices = await navigator.mediaDevices.enumerateDevices();
        if (cancelled) return;
        setCameras(
          devices
            .filter((d) => d.kind === "videoinput")
            .map((d, i) => ({ deviceId: d.deviceId, label: d.label || `Camera ${i + 1}` })),
        );
      } catch (err) {
        if (cancelled) return;
        setStatus("error");
        setError(
          err instanceof Error && err.name === "NotAllowedError"
            ? "Camera access was denied. Allow it in your system privacy settings and try again."
            : err instanceof Error
              ? err.message
              : "The camera could not be started",
        );
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
