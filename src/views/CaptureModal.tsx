import { useCallback, useEffect, useRef, useState } from "react";
import { Camera, VideoCamera, Warning } from "@phosphor-icons/react";
import { useStore } from "../lib/store";
import { useWebcam } from "../lib/useWebcam";
import { toSquareCanvas } from "../lib/images";
import { Modal, swatch } from "../components/ui";
import type { Label } from "../lib/types";

const BURST_INTERVAL_MS = 130;
const STRIP_SIZE = 14;

export function CaptureModal({ label, onClose }: { label: Label; onClose: () => void }) {
  const addFromDrawable = useStore((s) => s.addFromDrawable);
  const { videoRef, status, error, cameras, deviceId, setDeviceId, retry } = useWebcam(true);
  const [count, setCount] = useState(0);
  const [strip, setStrip] = useState<string[]>([]);
  const [bursting, setBursting] = useState(false);
  const timer = useRef<ReturnType<typeof setInterval> | null>(null);

  const shoot = useCallback(() => {
    const video = videoRef.current;
    if (!video || video.videoWidth === 0) return;
    const canvas = toSquareCanvas(video);
    void addFromDrawable(canvas, label.id, "capture");
    setCount((n) => n + 1);
    setStrip((prev) => [canvas.toDataURL("image/jpeg", 0.6), ...prev].slice(0, STRIP_SIZE));
  }, [addFromDrawable, label.id, videoRef]);

  const startBurst = useCallback(() => {
    if (timer.current) return;
    setBursting(true);
    shoot();
    timer.current = setInterval(shoot, BURST_INTERVAL_MS);
  }, [shoot]);

  const stopBurst = useCallback(() => {
    if (timer.current) clearInterval(timer.current);
    timer.current = null;
    setBursting(false);
  }, []);

  useEffect(() => stopBurst, [stopBurst]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.code === "Space" && !event.repeat) {
        event.preventDefault();
        startBurst();
      }
    };
    const onKeyUp = (event: KeyboardEvent) => {
      if (event.code === "Space") stopBurst();
    };
    window.addEventListener("keydown", onKey);
    window.addEventListener("keyup", onKeyUp);
    return () => {
      window.removeEventListener("keydown", onKey);
      window.removeEventListener("keyup", onKeyUp);
    };
  }, [startBurst, stopBurst]);

  return (
    <Modal
      title={`Capture into ${label.name}`}
      wide
      onClose={() => {
        stopBurst();
        onClose();
      }}
      footer={
        <>
          <span className="capture__tally mono">{count} captured this session</span>
          <button
            className="btn"
            onClick={() => {
              stopBurst();
              onClose();
            }}
          >
            Done
          </button>
        </>
      }
    >
      <div className="capture">
        <div className="capture__stage">
          <video ref={videoRef} className="capture__video" playsInline muted autoPlay />
          {status !== "live" ? (
            <div className="capture__veil">
              {status === "error" ? (
                <>
                  <Warning size={26} weight="bold" />
                  <p>{error}</p>
                  <button className="btn btn--sm" onClick={retry}>
                    Try again
                  </button>
                </>
              ) : (
                <>
                  <VideoCamera size={26} weight="bold" />
                  <p>Waking the camera</p>
                </>
              )}
            </div>
          ) : null}
          <span className="capture__target" style={{ background: swatch(label.color) }}>
            {label.name}
          </span>
        </div>

        <div className="capture__controls">
          <button
            className={bursting ? "btn btn--lg btn--danger btn--block" : "btn btn--lg btn--primary btn--block"}
            disabled={status !== "live"}
            onPointerDown={startBurst}
            onPointerUp={stopBurst}
            onPointerLeave={stopBurst}
          >
            <Camera size={20} weight="bold" />
            {bursting ? "Recording" : "Hold to capture"}
          </button>
          <p className="field__hint">
            Hold the button or the space bar and move the object around. Roughly 8 frames per second.
          </p>

          {cameras.length > 1 ? (
            <div className="field">
              <label className="field__label" htmlFor="camera-pick">
                Camera
              </label>
              <select
                id="camera-pick"
                className="input"
                value={deviceId ?? cameras[0]?.deviceId ?? ""}
                onChange={(event) => setDeviceId(event.target.value)}
              >
                {cameras.map((camera) => (
                  <option key={camera.deviceId} value={camera.deviceId}>
                    {camera.label}
                  </option>
                ))}
              </select>
            </div>
          ) : null}
        </div>
      </div>

      {strip.length > 0 ? (
        <div className="capture__strip">
          {strip.map((src, i) => (
            <img key={`${src.length}-${i}`} src={src} alt="" className="capture__thumb" />
          ))}
        </div>
      ) : null}
    </Modal>
  );
}
