import { useMemo, useState } from "react";
import {
  ArrowCounterClockwise,
  Brain,
  CheckCircle,
  CircleNotch,
  Lightning,
  Sliders,
  Stop,
  WarningCircle,
} from "@phosphor-icons/react";
import { useStore, type ImageVerdict } from "../lib/store";
import { EmptyState, Meter, pct, swatch } from "../components/ui";
import { DEFAULT_SETTINGS, type TrainSettings } from "../lib/types";
import { storage } from "../lib/storage";

type Filter = "all" | "correct" | "wrong";

const NUMBER_FIELDS: {
  key: keyof TrainSettings;
  label: string;
  min: number;
  max: number;
  step: number;
  hint: string;
}[] = [
  { key: "epochs", label: "Epochs", min: 5, max: 200, step: 5, hint: "Passes over the dataset" },
  { key: "batchSize", label: "Batch size", min: 1, max: 128, step: 1, hint: "Images per step" },
  {
    key: "learningRate",
    label: "Learning rate",
    min: 0.0001,
    max: 0.05,
    step: 0.0001,
    hint: "Step size for Adam",
  },
  {
    key: "hiddenUnits",
    label: "Hidden units",
    min: 8,
    max: 512,
    step: 4,
    hint: "Width of the dense head",
  },
];

export function TrainView() {
  const manifest = useStore((s) => s.manifest);
  const path = useStore((s) => s.path);
  const train = useStore((s) => s.train);
  const backend = useStore((s) => s.backend);
  const startTraining = useStore((s) => s.startTraining);
  const stopTraining = useStore((s) => s.stopTraining);
  const updateSettings = useStore((s) => s.updateSettings);
  const moveImages = useStore((s) => s.moveImages);
  const setView = useStore((s) => s.setView);
  const hasModel = useStore((s) => s.hasModel);
  const stale = useStore((s) => s.stale);
  const autoRetraining = useStore((s) => s.autoRetraining);
  const hasCheckpoint = useStore((s) => s.hasCheckpoint);
  const revertModel = useStore((s) => s.revertModel);
  const [pickedFilter, setPickedFilter] = useState<Filter | null>(null);

  const labels = manifest?.labels ?? [];
  const images = manifest?.images ?? [];
  const labeledCount = images.filter((img) => img.labelId).length;
  const usableLabels = labels.filter((label) => images.some((img) => img.labelId === label.id));
  const ready = usableLabels.length >= 2;
  const running = train.phase === "reading" || train.phase === "training" || train.phase === "scoring";

  const perLabel = useMemo(() => {
    if (train.verdicts.length === 0) return [];
    return usableLabels.map((label) => {
      const mine = train.verdicts.filter((v) => v.actualLabelId === label.id);
      const hits = mine.filter((v) => v.correct).length;
      return {
        label,
        total: mine.length,
        hits,
        rate: mine.length === 0 ? 0 : hits / mine.length,
      };
    });
  }, [train.verdicts, usableLabels]);

  const wrongCount = train.verdicts.filter((v) => !v.correct).length;
  // Land on the mistakes when there are any, otherwise show the full set
  // instead of an empty bucket.
  const filter: Filter = pickedFilter ?? (wrongCount > 0 ? "wrong" : "all");
  const setFilter = setPickedFilter;

  const visible = useMemo(() => {
    if (filter === "correct") return train.verdicts.filter((v) => v.correct);
    if (filter === "wrong") return train.verdicts.filter((v) => !v.correct);
    return train.verdicts;
  }, [train.verdicts, filter]);
  const progress =
    train.phase === "training"
      ? train.total === 0
        ? 0
        : train.done / train.total
      : train.phase === "reading"
        ? train.total === 0
          ? 0
          : train.done / train.total
        : train.phase === "done"
          ? 1
          : 0;

  const fileFor = (verdict: ImageVerdict) =>
    images.find((img) => img.id === verdict.imageId)?.file ?? "";
  const nameFor = (id: string) => labels.find((l) => l.id === id)?.name ?? "gone";
  const colorFor = (id: string) => swatch(labels.find((l) => l.id === id)?.color ?? 0);

  if (!manifest || !path) return null;

  return (
    <div className="train scroll">
      <section className="train__stage card">
        <div className="train__status">
          <span className="eyebrow">
            {train.phase === "idle"
              ? hasModel
                ? "Trained in an earlier session"
                : "Not trained yet"
              : train.message || train.phase}
          </span>
          <h2 className="train__headline">
            {train.phase === "done"
              ? "Model is ready"
              : running
                ? train.phase === "training"
                  ? `Epoch ${train.done} of ${train.total}`
                  : train.message
                : ready
                  ? "Train the classifier"
                  : "Not enough data yet"}
          </h2>
          <p className="train__sub">
            {ready
              ? `${labeledCount} labeled images across ${usableLabels.length} labels. Features come from a frozen MobileNet, only the small head learns.`
              : "You need at least two labels that both hold images. Head back to the label view and collect a few more."}
          </p>

          {autoRetraining ? (
            <p className="train__flag train__flag--calm">
              <CircleNotch size={14} weight="bold" className="spin" /> Folding your latest
              corrections in. The camera keeps running while this happens.
            </p>
          ) : stale ? (
            <p className="train__flag">
              The dataset changed since the last run. Train again to fold the new images in.
            </p>
          ) : null}

          <Meter value={progress} large />

          <div className="row">
            {running ? (
              <button className="btn btn--lg btn--danger" onClick={stopTraining}>
                <Stop size={18} weight="bold" /> Stop
              </button>
            ) : (
              <button
                className="btn btn--lg btn--go"
                disabled={!ready}
                onClick={() => void startTraining()}
              >
                <Lightning size={18} weight="bold" />
                {train.phase === "done" ? "Train again" : "Train"}
              </button>
            )}
            {train.phase === "done" ? (
              <button className="btn btn--lg btn--primary" onClick={() => setView("use")}>
                Use it
              </button>
            ) : null}
            {hasCheckpoint ? (
              <button
                className="btn"
                disabled={running || autoRetraining}
                title="Restore the model from before the last background retrain"
                onClick={() => void revertModel()}
              >
                <ArrowCounterClockwise size={16} weight="bold" /> Revert to previous model
              </button>
            ) : null}
            <div className="spacer" />
            <span className="badge badge--idle" title="Active TensorFlow.js backend">
              {backend ?? "starting"}
            </span>
          </div>
        </div>

        <dl className="train__metrics">
          <div>
            <dt className="eyebrow">Accuracy</dt>
            <dd className="mono">{train.accuracy === null ? "--" : pct(train.accuracy)}</dd>
          </div>
          <div>
            <dt className="eyebrow">Holdout</dt>
            <dd className="mono">{train.valAccuracy === null ? "--" : pct(train.valAccuracy)}</dd>
          </div>
          <div>
            <dt className="eyebrow">Images</dt>
            <dd className="mono">{labeledCount}</dd>
          </div>
          <div>
            <dt className="eyebrow">Labels</dt>
            <dd className="mono">{usableLabels.length}</dd>
          </div>
        </dl>
      </section>

      <div className="train__cols">
        <section className="train__results">
          {train.verdicts.length === 0 ? (
            <div className="card">
              <EmptyState
                icon={<Brain size={30} weight="bold" />}
                title="No results yet"
                text="After a training run every image gets scored by the fresh model, so you can see exactly which ones it still gets wrong."
              />
            </div>
          ) : (
            <>
              <div className="train__filters">
                <button
                  className={`tab${filter === "wrong" ? " tab--on" : ""}`}
                  onClick={() => setFilter("wrong")}
                >
                  <WarningCircle size={16} weight="bold" /> Wrong {wrongCount}
                </button>
                <button
                  className={`tab${filter === "correct" ? " tab--on" : ""}`}
                  onClick={() => setFilter("correct")}
                >
                  <CheckCircle size={16} weight="bold" /> Correct {train.verdicts.length - wrongCount}
                </button>
                <button
                  className={`tab${filter === "all" ? " tab--on" : ""}`}
                  onClick={() => setFilter("all")}
                >
                  All {train.verdicts.length}
                </button>
              </div>

              {visible.length === 0 ? (
                <div className="card">
                  <EmptyState
                    icon={<CheckCircle size={30} weight="bold" />}
                    title="Nothing in this bucket"
                    text="Every scored image landed in the other pile."
                  />
                </div>
              ) : (
                <div className="verdicts">
                  {visible.map((verdict) => (
                    <article
                      key={verdict.imageId}
                      className={`verdict${verdict.correct ? "" : " verdict--bad"}`}
                    >
                      <img src={storage.imageUrl(path, fileFor(verdict))} alt="" loading="lazy" />
                      <div className="verdict__body">
                        <span className="verdict__row">
                          <span className="dot" style={{ background: colorFor(verdict.actualLabelId) }} />
                          <span className="verdict__name">{nameFor(verdict.actualLabelId)}</span>
                        </span>
                        {verdict.correct ? (
                          <span className="badge badge--ok">{pct(verdict.confidence)}</span>
                        ) : (
                          <>
                            <span className="verdict__guess mono">
                              read as {nameFor(verdict.predictedLabelId)} {pct(verdict.confidence)}
                            </span>
                            <button
                              className="btn btn--sm"
                              onClick={() => moveImages([verdict.imageId], verdict.predictedLabelId)}
                            >
                              Move to {nameFor(verdict.predictedLabelId)}
                            </button>
                          </>
                        )}
                      </div>
                    </article>
                  ))}
                </div>
              )}
            </>
          )}
        </section>

        <aside className="train__side">
          <section className="card">
            <div className="card__head">
              <h3>Per label</h3>
            </div>
            <div className="card__body stack">
              {perLabel.length === 0 ? (
                <p className="field__hint">Train once to see how each label performs.</p>
              ) : (
                perLabel.map((row) => (
                  <div key={row.label.id} className="labelrate">
                    <div className="row row--tight">
                      <span className="dot" style={{ background: swatch(row.label.color) }} />
                      <span className="labelrate__name">{row.label.name}</span>
                      <div className="spacer" />
                      <span className="mono labelrate__num">
                        {row.hits}/{row.total}
                      </span>
                    </div>
                    <Meter value={row.rate} />
                  </div>
                ))
              )}
            </div>
          </section>

          <section className="card">
            <div className="card__head">
              <h3>
                <Sliders size={16} weight="bold" /> Settings
              </h3>
              <button
                className="btn btn--sm"
                onClick={() => updateSettings({ ...DEFAULT_SETTINGS })}
                disabled={running}
              >
                Reset
              </button>
            </div>
            <div className="card__body stack">
              {NUMBER_FIELDS.map((field) => (
                <div className="field" key={field.key}>
                  <label className="field__label" htmlFor={`set-${field.key}`}>
                    {field.label}
                  </label>
                  <input
                    id={`set-${field.key}`}
                    className="input mono"
                    type="number"
                    disabled={running}
                    min={field.min}
                    max={field.max}
                    step={field.step}
                    value={manifest.settings[field.key]}
                    onChange={(event) => {
                      const value = Number(event.target.value);
                      if (Number.isFinite(value)) updateSettings({ [field.key]: value });
                    }}
                  />
                  <span className="field__hint">{field.hint}</span>
                </div>
              ))}
            </div>
          </section>
        </aside>
      </div>
    </div>
  );
}
