import { useCallback, useEffect, useRef, useState } from "react";
import {
  Brain,
  Check,
  CircleNotch,
  Image as ImageIcon,
  Play,
  PlusCircle,
  Stop,
  VideoCamera,
  Warning,
} from "@phosphor-icons/react";
import { MULTI_THRESHOLD } from "../lib/types";
import { useStore, type Prediction } from "../lib/store";
import { useWebcam } from "../lib/useWebcam";
import { AttentionOverlay } from "../components/AttentionOverlay";
import { CameraPicker } from "../components/CameraPicker";
import { EmptyState, Meter, pct, swatch } from "../components/ui";
import { isImageFile, loadImageElement, toSquareCanvas } from "../lib/images";
import { storage } from "../lib/storage";

const TICK_MS = 220;

function Scores({ prediction, multi }: { prediction: Prediction | null; multi: boolean }) {
  const manifest = useStore((s) => s.manifest);
  if (!prediction) return null;
  // Label creation order, never score order: a list that reshuffles every frame
  // is unreadable and, next to the teach buttons, unclickable.
  const rows = prediction.scores.map((row) => ({
    row,
    label: manifest?.labels.find((l) => l.id === row.labelId) ?? null,
  }));

  return (
    <ul className={multi ? "scores scores--multi" : "scores"}>
      {rows.map(({ row, label }) => {
        // Multi mode: each label answers on its own, there is no winner.
        const on = multi ? row.score >= MULTI_THRESHOLD : row.labelId === prediction.labelId;
        return (
          <li key={row.labelId} className={on ? "score score--top" : "score"}>
            <span className="dot" style={{ background: swatch(label?.color ?? 0) }} />
            <span className="score__name">{label?.name ?? "removed label"}</span>
            <Meter value={row.score} />
            <span className="mono score__val">{pct(row.score)}</span>
            {multi ? (
              <span className={`badge ${on ? "badge--ok" : "badge--idle"} score__flag`}>
                {on ? "yes" : "no"}
              </span>
            ) : null}
          </li>
        );
      })}
    </ul>
  );
}

function FeedbackBar({
  prediction,
  multi,
  onTeach,
}: {
  prediction: Prediction | null;
  multi: boolean;
  onTeach: (labelIds: string[]) => void;
}) {
  const manifest = useStore((s) => s.manifest);
  const [picked, setPicked] = useState<string[]>([]);
  if (!prediction || !manifest) return null;

  // Fixed order and fixed captions in both modes. The button under a finger
  // never moves or changes meaning while the camera runs.
  if (multi) {
    const toggle = (id: string) =>
      setPicked((prev) => (prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]));
    return (
      <div className="feedback">
        <span className="eyebrow">Teach it from this frame</span>
        <div className="feedback__chips">
          {manifest.labels.map((label) => {
            const on = picked.includes(label.id);
            return (
              <button
                key={label.id}
                className={`chip chip--teach${on ? " chip--on" : ""}`}
                style={on ? { background: swatch(label.color) } : undefined}
                aria-pressed={on}
                onClick={() => toggle(label.id)}
              >
                {on ? (
                  <Check size={14} weight="bold" />
                ) : (
                  <span className="dot" style={{ background: swatch(label.color) }} />
                )}
                This is {label.name}
              </button>
            );
          })}
        </div>
        <button
          className="btn btn--sm btn--go"
          disabled={picked.length === 0}
          onClick={() => {
            onTeach(picked);
            setPicked([]);
          }}
        >
          <PlusCircle size={15} weight="bold" /> Add frame with {picked.length} label
          {picked.length === 1 ? "" : "s"}
        </button>
      </div>
    );
  }

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
              onClick={() => onTeach([label.id])}
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
  // Session state: the toggle survives a trip through the other views and is
  // never written to the manifest.
  const attentionOn = useStore((s) => s.attentionOn);
  const setAttentionOn = useStore((s) => s.setAttentionOn);

  const [live, setLive] = useState(false);
  const { videoRef, status, error, cameras, deviceId, selectCamera, switching, retry } =
    useWebcam(live);
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
      void predict(frame, attentionOn)
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
  }, [live, status, hasModel, predict, videoRef, attentionOn]);

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

  const namesFor = (labelIds: string[]) =>
    labelIds
      .map((id) => manifest?.labels.find((l) => l.id === id)?.name ?? "the label")
      .join(" + ") || "no label";

  const teachFromVideo = (labelIds: string[]) => {
    const video = videoRef.current;
    if (!video || video.videoWidth === 0) return;
    void addFromDrawable(toSquareCanvas(video), labelIds, "feedback");
    toast(`Frame added to ${namesFor(labelIds)}. It retrains itself in a moment.`, "good");
  };

  const teachFromStill = (labelIds: string[]) => {
    const img = stillRef.current;
    if (!img) return;
    void addFromDrawable(toSquareCanvas(img), labelIds, "feedback");
    toast(`Image added to ${namesFor(labelIds)}. It retrains itself in a moment.`, "good");
  };

  if (!manifest) return null;
  const multi = manifest.mode === "multi";

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
  // Multi mode has no winner, so the overlay lists everything above threshold.
  const liveOnLabels =
    livePrediction?.scores.filter((row) => row.score >= MULTI_THRESHOLD).map((row) => row.labelId) ??
    [];
  const liveOnNames = liveOnLabels
    .map((id) => manifest.labels.find((l) => l.id === id)?.name ?? "?")
    .join(" + ");

  return (
    <div className="use">
      <section className="card use__live">
        <div className="card__head">
          <h3>
            <VideoCamera size={16} weight="bold" /> Live camera
          </h3>
          <div className="card__tools">
            {live ? (
              <CameraPicker
                cameras={cameras}
                deviceId={deviceId}
                onPick={selectCamera}
                compact
              />
            ) : null}
            {live ? (
              <label className="switch">
                <input
                  type="checkbox"
                  className="switch__input"
                  checked={attentionOn}
                  onChange={(event) => setAttentionOn(event.target.checked)}
                />
                <span className="switch__box" aria-hidden="true">
                  <Check size={12} weight="bold" />
                </span>
                <span className="switch__text">Show attention</span>
              </label>
            ) : null}
            <button
              className={live ? "btn btn--sm btn--danger" : "btn btn--sm btn--go"}
              onClick={() => setLive((v) => !v)}
            >
              {live ? <Stop size={15} weight="bold" /> : <Play size={15} weight="bold" />}
              {live ? "Stop" : "Start"}
            </button>
          </div>
        </div>

        <div className="use__stage">
          <video ref={videoRef} className="use__video" playsInline muted autoPlay />
          {live && status === "live" ? (
            <AttentionOverlay
              videoRef={videoRef}
              attention={attentionOn ? (livePrediction?.attention ?? null) : null}
              labels={manifest.labels}
              active={attentionOn}
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

          {live && status === "live" && livePrediction ? (
            multi ? (
              <div
                className="verdictbar"
                style={{
                  background:
                    liveOnLabels.length === 0
                      ? "var(--paper-deep)"
                      : swatch(
                          manifest.labels.find((l) => l.id === liveOnLabels[0])?.color ?? 0,
                        ),
                }}
              >
                <span className="verdictbar__name">{liveOnNames || "none of them"}</span>
                <span className="verdictbar__pct mono">
                  {liveOnLabels.length}/{livePrediction.scores.length}
                </span>
              </div>
            ) : (
              <div className="verdictbar" style={{ background: swatch(topLabel?.color ?? 0) }}>
                <span className="verdictbar__name">{topLabel?.name ?? "unknown"}</span>
                <span className="verdictbar__pct mono">{pct(livePrediction.confidence)}</span>
              </div>
            )
          ) : null}
        </div>

        {livePrediction ? (
          <div className="card__body stack">
            <Scores prediction={livePrediction} multi={multi} />
            <FeedbackBar prediction={livePrediction} multi={multi} onTeach={teachFromVideo} />
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
            <Scores prediction={stillPrediction} multi={multi} />
            <FeedbackBar prediction={stillPrediction} multi={multi} onTeach={teachFromStill} />
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
