import { useCallback, useEffect, useRef, useState } from "react";
import {
  Brain,
  CircleNotch,
  Image as ImageIcon,
  Play,
  PlusCircle,
  Stop,
  VideoCamera,
  Warning,
} from "@phosphor-icons/react";
import { useStore, type Prediction } from "../lib/store";
import { useWebcam } from "../lib/useWebcam";
import { EmptyState, Meter, pct, swatch } from "../components/ui";
import { isImageFile, loadImageElement, toSquareCanvas } from "../lib/images";
import { storage } from "../lib/storage";

const TICK_MS = 220;

function Scores({ prediction }: { prediction: Prediction | null }) {
  const manifest = useStore((s) => s.manifest);
  if (!prediction) return null;
  // Label creation order, never score order: a list that reshuffles every frame
  // is unreadable and, next to the teach buttons, unclickable.
  const rows = prediction.scores.map((row) => ({
    row,
    label: manifest?.labels.find((l) => l.id === row.labelId) ?? null,
  }));

  return (
    <ul className="scores">
      {rows.map(({ row, label }) => (
        <li key={row.labelId} className={row.labelId === prediction.labelId ? "score score--top" : "score"}>
          <span className="dot" style={{ background: swatch(label?.color ?? 0) }} />
          <span className="score__name">{label?.name ?? "removed label"}</span>
          <Meter value={row.score} />
          <span className="mono score__val">{pct(row.score)}</span>
        </li>
      ))}
    </ul>
  );
}

function FeedbackBar({
  prediction,
  onTeach,
}: {
  prediction: Prediction | null;
  onTeach: (labelId: string) => void;
}) {
  const manifest = useStore((s) => s.manifest);
  if (!prediction || !manifest) return null;

  // Fixed order and fixed captions. The predicted label is highlighted in place
  // so a button never moves out from under a finger mid tap.
  return (
    <div className="feedback">
      <span className="eyebrow">Teach it from this frame</span>
      <div className="feedback__chips">
        {manifest.labels.map((label) => {
          const top = label.id === prediction.labelId;
          return (
            <button
              key={label.id}
              className={`chip chip--teach${top ? " chip--on" : ""}`}
              style={top ? { background: swatch(label.color) } : undefined}
              aria-pressed={top}
              onClick={() => onTeach(label.id)}
            >
              {top ? (
                <PlusCircle size={14} weight="bold" />
              ) : (
                <span className="dot" style={{ background: swatch(label.color) }} />
              )}
              This is {label.name}
            </button>
          );
        })}
      </div>
    </div>
  );
}

export function UseView() {
  const hasModel = useStore((s) => s.hasModel);
  const stale = useStore((s) => s.stale);
  const autoRetraining = useStore((s) => s.autoRetraining);
  const manifest = useStore((s) => s.manifest);
  const predict = useStore((s) => s.predict);
  const addFromDrawable = useStore((s) => s.addFromDrawable);
  const setView = useStore((s) => s.setView);
  const toast = useStore((s) => s.toast);

  const [live, setLive] = useState(false);
  const { videoRef, status, error, retry } = useWebcam(live);
  const [livePrediction, setLivePrediction] = useState<Prediction | null>(null);

  const [stillUrl, setStillUrl] = useState<string | null>(null);
  const [stillPrediction, setStillPrediction] = useState<Prediction | null>(null);
  const [dropping, setDropping] = useState(false);
  const [thinking, setThinking] = useState(false);
  const stillRef = useRef<HTMLImageElement | null>(null);

  useEffect(() => {
    if (!live || status !== "live" || !hasModel) {
      setLivePrediction(null);
      return;
    }
    let cancelled = false;
    let inFlight = false;
    const id = setInterval(() => {
      if (inFlight) return;
      const video = videoRef.current;
      if (!video || video.videoWidth === 0) return;
      inFlight = true;
      const frame = toSquareCanvas(video);
      void predict(frame)
        .then((result) => {
          if (!cancelled) setLivePrediction(result);
        })
        .catch(() => undefined)
        .finally(() => {
          inFlight = false;
        });
    }, TICK_MS);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [live, status, hasModel, predict, videoRef]);

  useEffect(() => () => {
    if (stillUrl) URL.revokeObjectURL(stillUrl);
  }, [stillUrl]);

  const runStill = useCallback(
    async (file: File) => {
      if (!isImageFile(file)) {
        toast("That file is not an image", "bad");
        return;
      }
      const url = URL.createObjectURL(file);
      setStillUrl((prev) => {
        if (prev) URL.revokeObjectURL(prev);
        return url;
      });
      setThinking(true);
      try {
        const img = await loadImageElement(url);
        setStillPrediction(await predict(toSquareCanvas(img)));
      } catch {
        toast("Could not read that image", "bad");
      } finally {
        setThinking(false);
      }
    },
    [predict, toast],
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

  const teachFromVideo = (labelId: string) => {
    const video = videoRef.current;
    if (!video || video.videoWidth === 0) return;
    void addFromDrawable(toSquareCanvas(video), labelId, "feedback");
    const name = manifest?.labels.find((l) => l.id === labelId)?.name ?? "the label";
    toast(`Frame added to ${name}. It retrains itself in a moment.`, "good");
  };

  const teachFromStill = (labelId: string) => {
    const img = stillRef.current;
    if (!img) return;
    void addFromDrawable(toSquareCanvas(img), labelId, "feedback");
    const name = manifest?.labels.find((l) => l.id === labelId)?.name ?? "the label";
    toast(`Image added to ${name}. It retrains itself in a moment.`, "good");
  };

  if (!manifest) return null;

  if (!hasModel) {
    return (
      <div className="use use--blank">
        <div className="card">
          <EmptyState
            icon={<Brain size={30} weight="bold" />}
            title="No trained model yet"
            text="Once the classifier is trained you can point your camera at things or drop a single image in here and watch it decide."
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

  const topLabel = manifest.labels.find((l) => l.id === livePrediction?.labelId);

  return (
    <div className="use">
      <section className="card use__live">
        <div className="card__head">
          <h3>
            <VideoCamera size={16} weight="bold" /> Live camera
          </h3>
          <button
            className={live ? "btn btn--sm btn--danger" : "btn btn--sm btn--go"}
            onClick={() => setLive((v) => !v)}
          >
            {live ? <Stop size={15} weight="bold" /> : <Play size={15} weight="bold" />}
            {live ? "Stop" : "Start"}
          </button>
        </div>

        <div className="use__stage">
          <video ref={videoRef} className="use__video" playsInline muted autoPlay />
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
              <p>Waking the camera</p>
            </div>
          ) : null}

          {live && status === "live" && livePrediction ? (
            <div className="verdictbar" style={{ background: swatch(topLabel?.color ?? 0) }}>
              <span className="verdictbar__name">{topLabel?.name ?? "unknown"}</span>
              <span className="verdictbar__pct mono">{pct(livePrediction.confidence)}</span>
            </div>
          ) : null}
        </div>

        {livePrediction ? (
          <div className="card__body stack">
            <Scores prediction={livePrediction} />
            <FeedbackBar prediction={livePrediction} onTeach={teachFromVideo} />
          </div>
        ) : null}
      </section>

      <section className="card use__still">
        <div className="card__head">
          <h3>
            <ImageIcon size={16} weight="bold" /> Single image
          </h3>
          {stillPrediction ? (
            <button
              className="btn btn--sm"
              onClick={() => {
                setStillPrediction(null);
                setStillUrl((prev) => {
                  if (prev) URL.revokeObjectURL(prev);
                  return null;
                });
              }}
            >
              Clear
            </button>
          ) : null}
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
            <img ref={stillRef} src={stillUrl} alt="Image being classified" className="use__still-img" />
          ) : (
            <div className="use__hint">
              <ImageIcon size={26} weight="bold" />
              <p>{storage.mobile ? "Choose an image to classify it" : "Drop one image here to classify it"}</p>
              {storage.mobile ? (
                <button className="btn btn--sm" onClick={() => void pickStill()}>
                  Choose image
                </button>
              ) : null}
            </div>
          )}
          {thinking ? <div className="use__veil">Thinking</div> : null}
        </div>

        {stillPrediction ? (
          <div className="card__body stack">
            <Scores prediction={stillPrediction} />
            <FeedbackBar prediction={stillPrediction} onTeach={teachFromStill} />
          </div>
        ) : null}
      </section>

      {autoRetraining ? (
        <p className="use__flag use__flag--calm">
          <CircleNotch size={14} weight="bold" className="spin" /> Retraining with your corrections.
          Keep going, the current model stays live until the new one is ready.
        </p>
      ) : stale ? (
        <p className="use__flag">
          New images were added since training. The classifier below still runs the older model.
        </p>
      ) : null}
    </div>
  );
}
