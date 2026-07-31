import { useCallback, useEffect, useRef, useState } from "react";
import {
  BoundingBox,
  Brain,
  CircleNotch,
  FloppyDisk,
  Image as ImageIcon,
  Play,
  Stop,
  VideoCamera,
  Warning,
} from "@phosphor-icons/react";
import { CameraPicker } from "../components/CameraPicker";
import { DetectionOverlay } from "../components/DetectionOverlay";
import { EmptyState, pct, swatch } from "../components/ui";
import type { Detection } from "../lib/detect";
import { isImageFile, loadImageElement, toSquareCanvas } from "../lib/images";
import { storage } from "../lib/storage";
import { useStore } from "../lib/store";
import {
  setScoreThreshold,
  THRESHOLDS,
  useScoreThreshold,
  type ScoreThreshold,
} from "../lib/scoreThreshold";
import { useWebcam } from "../lib/useWebcam";

const TICK_MS = 220;

function ThresholdPicker({
  value,
  onChange,
}: {
  value: number;
  onChange: (value: ScoreThreshold) => void;
}) {
  return (
    <fieldset className="threshold" aria-label="Detection score threshold">
      <legend>Score</legend>
      <div className="threshold__steps">
        {THRESHOLDS.map((threshold) => (
          <button
            key={threshold}
            type="button"
            className={threshold === value ? "threshold__step threshold__step--on" : "threshold__step"}
            aria-pressed={threshold === value}
            title={`Show detections scoring at least ${pct(threshold)}`}
            onClick={() => onChange(threshold)}
          >
            {threshold.toFixed(1)}
          </button>
        ))}
      </div>
    </fieldset>
  );
}

function DetectionList({ detections }: { detections: Detection[] }) {
  const manifest = useStore((state) => state.manifest);
  if (detections.length === 0) {
    return <p className="field__hint">No objects clear the current score threshold.</p>;
  }
  return (
    <ul className="detectionlist" data-detection-count={detections.length}>
      {detections.map((detection, index) => {
        const label = manifest?.labels.find((entry) => entry.id === detection.labelId);
        return (
          <li key={`${detection.labelId}-${index}`}>
            <span className="dot" style={{ background: swatch(label?.color ?? 0) }} />
            <span className="detectionlist__name">{label?.name ?? "removed label"}</span>
            <span className="mono detectionlist__score">{pct(detection.score)}</span>
          </li>
        );
      })}
    </ul>
  );
}

export function DetectUseView() {
  const hasModel = useStore((state) => state.hasModel);
  const stale = useStore((state) => state.stale);
  const autoRetraining = useStore((state) => state.autoRetraining);
  const manifest = useStore((state) => state.manifest);
  const setView = useStore((state) => state.setView);
  const toast = useStore((state) => state.toast);
  const detectObjects = useStore((state) => state.detectObjects);
  const addFromDrawable = useStore((state) => state.addFromDrawable);

  // Shared with the box annotator's guesses, so one cutoff governs the app.
  const threshold = useScoreThreshold();
  const [live, setLive] = useState(false);
  const { videoRef, status, error, cameras, deviceId, selectCamera, switching, retry } =
    useWebcam(live);
  const [liveDetections, setLiveDetections] = useState<Detection[]>([]);
  const [savingFrame, setSavingFrame] = useState(false);
  const liveInferenceRef = useRef(false);

  const [stillUrl, setStillUrl] = useState<string | null>(null);
  const [stillSource, setStillSource] = useState<HTMLImageElement | null>(null);
  const [stillDetections, setStillDetections] = useState<Detection[] | null>(null);
  const [dropping, setDropping] = useState(false);
  const [thinking, setThinking] = useState(false);
  const stillRef = useRef<HTMLImageElement | null>(null);



  useEffect(() => {
    if (!live || status !== "live" || !hasModel) {
      setLiveDetections([]);
      return;
    }
    let cancelled = false;
    const id = window.setInterval(() => {
      if (liveInferenceRef.current) return;
      const video = videoRef.current;
      if (!video || video.videoWidth === 0) return;
      liveInferenceRef.current = true;
      let frame: HTMLCanvasElement;
      try {
        frame = toSquareCanvas(video);
      } catch {
        liveInferenceRef.current = false;
        return;
      }
      void detectObjects(frame, threshold)
        .then((detections) => {
          if (!cancelled) setLiveDetections(detections ?? []);
        })
        .catch(() => undefined)
        .finally(() => {
          liveInferenceRef.current = false;
        });
    }, TICK_MS);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [detectObjects, hasModel, live, status, threshold, videoRef]);

  useEffect(() => {
    if (!stillSource || !hasModel) return;
    let cancelled = false;
    setThinking(true);
    void detectObjects(toSquareCanvas(stillSource), threshold)
      .then((detections) => {
        if (!cancelled) setStillDetections(detections ?? []);
      })
      .catch(() => {
        if (!cancelled) toast("Could not run the detector on that image", "bad");
      })
      .finally(() => {
        if (!cancelled) setThinking(false);
      });
    return () => {
      cancelled = true;
    };
  }, [detectObjects, hasModel, stillSource, threshold, toast]);

  useEffect(
    () => () => {
      if (stillUrl) URL.revokeObjectURL(stillUrl);
    },
    [stillUrl],
  );

  const runStill = useCallback(
    async (file: File) => {
      if (!isImageFile(file)) {
        toast("That file is not an image", "bad");
        return;
      }
      const url = URL.createObjectURL(file);
      setStillUrl((previous) => {
        if (previous) URL.revokeObjectURL(previous);
        return url;
      });
      setStillSource(null);
      setStillDetections(null);
      setThinking(true);
      try {
        setStillSource(await loadImageElement(url));
      } catch {
        setThinking(false);
        toast("Could not read that image", "bad");
      }
    },
    [toast],
  );

  const pickStill = async () => {
    try {
      const refs = await storage.pickImportImages(false);
      const ref = refs?.[0];
      if (!ref) return;
      const payload = await storage.readImport(ref);
      const binary = atob(payload.base64);
      const bytes = Uint8Array.from(binary, (char) => char.charCodeAt(0));
      await runStill(new File([bytes], ref.name, { type: payload.mime }));
    } catch {
      toast("Could not read that image", "bad");
    }
  };

  const clearStill = () => {
    setStillSource(null);
    setStillDetections(null);
    setThinking(false);
    setStillUrl((previous) => {
      if (previous) URL.revokeObjectURL(previous);
      return null;
    });
  };

  const saveFrame = async () => {
    const video = videoRef.current;
    if (!video || video.videoWidth === 0 || savingFrame) return;
    setSavingFrame(true);
    try {
      const saved = await addFromDrawable(toSquareCanvas(video), null, "capture");
      if (saved) toast("Frame saved. Open Label to draw its boxes.", "good");
    } finally {
      setSavingFrame(false);
    }
  };

  if (!manifest) return null;
  if (!hasModel) {
    return (
      <div className="use use--blank">
        <div className="card">
          <EmptyState
            icon={<Brain size={30} weight="bold" />}
            title="No trained detector yet"
            text="Draw boxes on your pictures and train once. Then live camera and single-image detection will appear here."
            action={
              <button className="btn btn--lg btn--go" onClick={() => setView("train")}>
                Go to training
              </button>
            }
          />
        </div>
      </div>
    );
  }

  return (
    <div className="use use--detect">
      <section className="card use__live">
        <div className="card__head">
          <h3>
            <VideoCamera size={16} weight="bold" /> Live detection
          </h3>
          <div className="card__tools">
            <ThresholdPicker value={threshold} onChange={setScoreThreshold} />
            {live ? (
              <CameraPicker cameras={cameras} deviceId={deviceId} onPick={selectCamera} compact />
            ) : null}
            <button
              className={live ? "btn btn--sm btn--danger" : "btn btn--sm btn--go"}
              onClick={() => setLive((value) => !value)}
            >
              {live ? <Stop size={15} weight="bold" /> : <Play size={15} weight="bold" />}
              {live ? "Stop" : "Start"}
            </button>
          </div>
        </div>

        <div className="use__stage">
          <video ref={videoRef} className="use__video" playsInline muted autoPlay />
          {live && status === "live" ? (
            <DetectionOverlay
              sourceRef={videoRef}
              fit="cover"
              detections={liveDetections}
              labels={manifest.labels}
            />
          ) : null}
          {!live ? (
            <div className="use__veil">
              <VideoCamera size={28} weight="bold" />
              <p>The camera is off</p>
            </div>
          ) : status === "error" ? (
            <div className="use__veil">
              <Warning size={26} weight="bold" />
              <p>{error}</p>
              <button className="btn btn--sm" onClick={retry}>
                Try again
              </button>
            </div>
          ) : status !== "live" ? (
            <div className="use__veil">
              <VideoCamera size={28} weight="bold" />
              <p>{switching ? "Switching camera" : "Waking the camera"}</p>
            </div>
          ) : null}
        </div>

        {live && status === "live" ? (
          <div className="card__body stack">
            <DetectionList detections={liveDetections} />
            <div className="detectsave">
              <button className="btn btn--sm btn--go" disabled={savingFrame} onClick={() => void saveFrame()}>
                <FloppyDisk size={15} weight="bold" />
                {savingFrame ? "Saving frame" : "Save frame to project"}
              </button>
              <p className="field__hint">Saves this frame without boxes, ready for you to box later in Label.</p>
            </div>
          </div>
        ) : null}
      </section>

      <section className="card use__still">
        <div className="card__head">
          <h3>
            <ImageIcon size={16} weight="bold" /> Single image
          </h3>
          <div className="card__tools">
            <span className="badge badge--idle">score ≥ {threshold.toFixed(1)}</span>
            {stillUrl ? (
              <button className="btn btn--sm" onClick={clearStill}>
                Clear
              </button>
            ) : null}
          </div>
        </div>

        <div
          className={`use__drop${dropping ? " use__drop--on" : ""}`}
          onDragOver={(event) => {
            event.preventDefault();
            setDropping(true);
          }}
          onDragLeave={() => setDropping(false)}
          onDrop={(event) => {
            event.preventDefault();
            setDropping(false);
            const file = event.dataTransfer.files[0];
            if (file) void runStill(file);
          }}
        >
          {stillUrl ? (
            <>
              <img ref={stillRef} src={stillUrl} alt="Image being detected" className="use__still-img" />
              <DetectionOverlay
                sourceRef={stillRef}
                fit="contain"
                detections={stillDetections ?? []}
                labels={manifest.labels}
              />
            </>
          ) : (
            <div className="use__hint">
              <BoundingBox size={26} weight="bold" />
              <p>{storage.mobile ? "Choose an image to detect objects" : "Drop one image here or choose one"}</p>
              <button className="btn btn--sm" onClick={() => void pickStill()}>
                Choose image
              </button>
            </div>
          )}
          {thinking ? <div className="use__veil">Finding objects</div> : null}
        </div>

        {stillDetections ? (
          <div className="card__body stack">
            <DetectionList detections={stillDetections} />
          </div>
        ) : null}
      </section>

      {autoRetraining ? (
        <p className="use__flag use__flag--calm">
          <CircleNotch size={14} weight="bold" className="spin" /> Retraining with your latest
          boxes. Keep going, the current detector stays live until the new one is ready.
        </p>
      ) : stale ? (
        <p className="use__flag">New pictures or boxes were added since training. This detector still runs the older model.</p>
      ) : null}
    </div>
  );
}
