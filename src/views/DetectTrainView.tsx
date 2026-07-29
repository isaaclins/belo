// Train view for detect projects.
//
// The classifier Train view asks one question per picture, so its results are a
// grid of labels. A detector answers with rectangles, and the only honest way
// to show that is to draw them: the boxes that were drawn on the picture in
// their label colour, the boxes the model read on top of them in a dashed ink
// outline. A miss and a false alarm are then visible at a glance instead of
// hidden inside an average.

import { useMemo, useState } from "react";
import {
  BoundingBox,
  Check,
  CheckCircle,
  Lightning,
  Question,
  Sliders,
  Stop,
  WarningCircle,
} from "@phosphor-icons/react";
import { useStore, type DetectTrainState, type DetectVerdict } from "../lib/store";
import { EmptyState, Meter, pct, swatch } from "../components/ui";
import { storage } from "../lib/storage";
import { readCurves, type Verdict } from "../lib/curve";
import {
  boxesOf,
  DETECT_MIN_IMAGES,
  DETECT_SETTINGS,
  type BoxRect,
  type TrainSettings,
} from "../lib/types";
import type { DetectEpoch } from "../lib/ml/detector";

type Filter = "all" | "clean" | "wrong";

/** The settings that are numbers, so the number inputs cannot reach the toggle. */
type NumberSetting = {
  [K in keyof TrainSettings]: TrainSettings[K] extends number ? K : never;
}[keyof TrainSettings];

const NUMBER_FIELDS: {
  key: NumberSetting;
  label: string;
  min: number;
  max: number;
  step: number;
  hint: string;
}[] = [
  { key: "epochs", label: "Epochs", min: 10, max: 500, step: 10, hint: "Passes over the pictures" },
  { key: "batchSize", label: "Batch size", min: 1, max: 64, step: 1, hint: "Pictures per step" },
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
    label: "Head width",
    min: 16,
    max: 512,
    step: 8,
    hint: "Width of the layer above the features",
  },
];

function place(rect: BoxRect) {
  return {
    left: `${rect.x * 100}%`,
    top: `${rect.y * 100}%`,
    width: `${rect.w * 100}%`,
    height: `${rect.h * 100}%`,
  };
}

/** The loss per epoch, drawn flat and thick. No axes, the numbers are below. */
function LossCurve({ epochs }: { epochs: DetectEpoch[] }) {
  if (epochs.length < 2) return null;
  const values = epochs.map((e) => e.loss);
  const holdout = epochs.map((e) => e.valLoss).filter((v): v is number => v !== null);
  const top = Math.max(...values, ...holdout, 0.001);
  const path = (series: number[]) =>
    series
      .map((value, i) => {
        const x = (i / Math.max(1, series.length - 1)) * 100;
        const y = 100 - (value / top) * 96;
        return `${i === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`;
      })
      .join(" ");

  return (
    <div className="dcurve">
      <svg viewBox="0 0 100 100" preserveAspectRatio="none" role="img" aria-label="Loss per epoch">
        {holdout.length > 1 ? (
          <path d={path(holdout)} className="dcurve__line dcurve__line--held" />
        ) : null}
        <path d={path(values)} className="dcurve__line" />
      </svg>
      <div className="dcurve__keys">
        <span className="dcurve__key">
          <i className="dcurve__chip" /> training {values[values.length - 1].toFixed(3)}
        </span>
        {holdout.length > 0 ? (
          <span className="dcurve__key">
            <i className="dcurve__chip dcurve__chip--held" /> holdout{" "}
            {holdout[holdout.length - 1].toFixed(3)}
          </span>
        ) : null}
      </div>
    </div>
  );
}

/**
 * The one panel in this view a person can read without knowing what a loss is.
 *
 * Everything else here reports numbers. This says what they mean and what to do
 * about it, in the words someone who has never trained a model would use. The
 * reading itself is pure and lives in lib/curve.ts, checked by
 * scripts/check-curve.mjs, so the sentence shown here is a tested decision
 * rather than a guess made in a render.
 */
function VerdictPanel({ detect }: { detect: DetectTrainState }) {
  const verdict: Verdict = readCurves({
    training: detect.epochs.map((epoch) => epoch.loss),
    holdout: detect.epochs.map((epoch) => epoch.valLoss),
    heldOut: detect.heldOut,
  });
  const icon =
    verdict.tone === "good" ? (
      <CheckCircle size={20} weight="bold" />
    ) : verdict.tone === "unknown" ? (
      <Question size={20} weight="bold" />
    ) : (
      <WarningCircle size={20} weight="bold" />
    );
  return (
    <section className={`card verdict verdict--${verdict.tone}`}>
      <div className="card__body stack">
        <div className="verdict__head">
          <span className="verdict__icon">{icon}</span>
          <h3 className="verdict__headline">{verdict.headline}</h3>
        </div>
        <p className="verdict__advice">{verdict.advice}</p>
        <p className="verdict__note">
          {detect.bestEpoch === null
            ? `Belo kept the last of ${detect.epochs.length} rounds of practice.`
            : detect.stoppedEarly
              ? `Belo stopped after ${detect.epochs.length} rounds because it had stopped getting better, and kept round ${detect.bestEpoch}, its best on the pictures it had never seen.`
              : `Belo kept round ${detect.bestEpoch} of ${detect.epochs.length}, its best on the pictures it had never seen.`}
          {detect.augmented > 0
            ? ` It also studied ${detect.augmented} mirrored copies of your pictures, on top of the ${detect.trainedOn} originals.`
            : ""}
        </p>
      </div>
    </section>
  );
}

export function DetectTrainView() {
  const manifest = useStore((s) => s.manifest);
  const path = useStore((s) => s.path);
  const train = useStore((s) => s.train);
  const backend = useStore((s) => s.backend);
  const startTraining = useStore((s) => s.startTraining);
  const stopTraining = useStore((s) => s.stopTraining);
  const updateSettings = useStore((s) => s.updateSettings);
  const setView = useStore((s) => s.setView);
  const hasModel = useStore((s) => s.hasModel);
  const stale = useStore((s) => s.stale);
  const [pickedFilter, setPickedFilter] = useState<Filter | null>(null);

  const labels = manifest?.labels ?? [];
  const images = manifest?.images ?? [];
  const detect = train.detect;

  const boxed = images.filter((image) => boxesOf(image).length > 0);
  const boxCount = images.reduce((sum, image) => sum + boxesOf(image).length, 0);
  const ready = boxed.length >= DETECT_MIN_IMAGES;
  const running =
    train.phase === "reading" || train.phase === "training" || train.phase === "scoring";

  const nameFor = (id: string) => labels.find((l) => l.id === id)?.name ?? "gone";
  const colorFor = (id: string) => swatch(labels.find((l) => l.id === id)?.color ?? 0);
  const fileFor = (imageId: string) => images.find((img) => img.id === imageId)?.file ?? "";

  const wrong = useMemo(
    () =>
      (detect?.verdicts ?? []).filter(
        (verdict) => verdict.missed.length > 0 || verdict.hits.some((hit) => hit < 0),
      ),
    [detect],
  );
  const filter: Filter = pickedFilter ?? (wrong.length > 0 ? "wrong" : "all");
  const visible = useMemo(() => {
    const all = detect?.verdicts ?? [];
    if (filter === "wrong") return wrong;
    if (filter === "clean") return all.filter((verdict) => !wrong.includes(verdict));
    return all;
  }, [detect, filter, wrong]);

  const totals = useMemo(() => {
    const stats = detect?.stats ?? [];
    const truth = stats.reduce((sum, stat) => sum + stat.truth, 0);
    const found = stats.reduce((sum, stat) => sum + stat.found, 0);
    const predicted = stats.reduce((sum, stat) => sum + stat.predicted, 0);
    return {
      recall: truth === 0 ? null : found / truth,
      precision: predicted === 0 ? null : found / predicted,
      truth,
    };
  }, [detect]);

  const progress =
    train.phase === "done"
      ? 1
      : train.total === 0
        ? 0
        : train.done / train.total;

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
              ? "Detector is ready"
              : running
                ? train.phase === "training"
                  ? `Epoch ${train.done} of ${train.total}`
                  : train.message
                : ready
                  ? "Train the detector"
                  : "Not enough boxes yet"}
          </h2>
          <p className="train__sub">
            {ready
              ? `${boxCount} boxes across ${boxed.length} pictures. The frozen MobileNet grid stays as it is, only the small detection head learns where your labels sit.`
              : `Draw boxes on at least ${DETECT_MIN_IMAGES} pictures first. A detector has to learn where as well as what, and ${boxed.length} picture${boxed.length === 1 ? "" : "s"} is not enough to show it.`}
          </p>

          {stale ? (
            <p className="train__flag">
              The boxes changed since the last run. Train again to fold them in.
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
            <div className="spacer" />
            <span className="badge badge--idle" title="Active TensorFlow.js backend">
              {backend ?? "starting"}
            </span>
          </div>
        </div>

        <dl className="train__metrics">
          <div>
            <dt className="eyebrow">Found</dt>
            <dd className="mono">{totals.recall === null ? "--" : pct(totals.recall)}</dd>
          </div>
          <div>
            <dt className="eyebrow">Right</dt>
            <dd className="mono">{totals.precision === null ? "--" : pct(totals.precision)}</dd>
          </div>
          <div>
            <dt className="eyebrow">Overlap</dt>
            <dd className="mono">{detect ? detect.meanIoU.toFixed(2) : "--"}</dd>
          </div>
          <div>
            <dt className="eyebrow">Boxes</dt>
            <dd className="mono">{boxCount}</dd>
          </div>
        </dl>
      </section>

      <div className="train__cols">
        <section className="train__results">
          {detect && detect.epochs.length > 1 && !running ? (
            <VerdictPanel detect={detect} />
          ) : null}
          {!detect || detect.verdicts.length === 0 ? (
            <div className="card">
              <EmptyState
                icon={<BoundingBox size={30} weight="bold" />}
                title="No results yet"
                text="After a run every picture is drawn twice: your boxes in their label colour, the ones the detector read dashed on top. Mistakes show up as a dashed box with nothing under it, or a solid box with nothing on it."
              />
            </div>
          ) : (
            <>
              <p className="field__hint">
                {detect.scored === "holdout"
                  ? `Scored on ${detect.heldOut} pictures held back from training, at half overlap. The other ${detect.trainedOn} were used to train.`
                  : `Scored on the same ${detect.trainedOn} pictures it trained on, because there were not enough to hold any back. Expect these numbers to be kinder than real ones.`}
              </p>
              <div className="train__filters">
                <button
                  className={`tab${filter === "wrong" ? " tab--on" : ""}`}
                  onClick={() => setPickedFilter("wrong")}
                >
                  <WarningCircle size={16} weight="bold" /> Mistakes {wrong.length}
                </button>
                <button
                  className={`tab${filter === "clean" ? " tab--on" : ""}`}
                  onClick={() => setPickedFilter("clean")}
                >
                  <CheckCircle size={16} weight="bold" /> Clean{" "}
                  {detect.verdicts.length - wrong.length}
                </button>
                <button
                  className={`tab${filter === "all" ? " tab--on" : ""}`}
                  onClick={() => setPickedFilter("all")}
                >
                  All {detect.verdicts.length}
                </button>
              </div>

              <div className="dlegend">
                <span className="dlegend__item">
                  <i className="dlegend__chip dlegend__chip--gt" /> drawn by you
                </span>
                <span className="dlegend__item">
                  <i className="dlegend__chip dlegend__chip--pred" /> read by the detector
                </span>
              </div>

              {visible.length === 0 ? (
                <div className="card">
                  <EmptyState
                    icon={<CheckCircle size={30} weight="bold" />}
                    title="Nothing in this bucket"
                    text="Every scored picture landed in the other pile."
                  />
                </div>
              ) : (
                <div className="dgrid">
                  {visible.map((verdict) => (
                    <Shot
                      key={verdict.imageId}
                      verdict={verdict}
                      src={storage.imageUrl(path, fileFor(verdict.imageId))}
                      colorFor={colorFor}
                      nameFor={nameFor}
                    />
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
              {!detect || detect.stats.length === 0 ? (
                <p className="field__hint">
                  Train once to see how many of each label's boxes the detector finds.
                </p>
              ) : (
                <>
                  <p className="field__hint">
                    Found is how many drawn boxes it caught, right is how many of its own boxes
                    were real. Both at half overlap.
                  </p>
                  {detect.stats.map((stat) => (
                    <div key={stat.labelId} className="labelrate">
                      <div className="row row--tight">
                        <span className="dot" style={{ background: colorFor(stat.labelId) }} />
                        <span className="labelrate__name">{nameFor(stat.labelId)}</span>
                        <div className="spacer" />
                        <span className="mono labelrate__num">
                          {stat.found}/{stat.truth}
                        </span>
                      </div>
                      <Meter value={stat.recall} />
                      <span className="field__hint mono">
                        right {pct(stat.precision)} of {stat.predicted}
                      </span>
                    </div>
                  ))}
                </>
              )}
            </div>
          </section>

          {detect && detect.epochs.length > 1 ? (
            <section className="card">
              <div className="card__head">
                <h3>How the practice went</h3>
              </div>
              <div className="card__body">
                <LossCurve epochs={detect.epochs} />
                <p className="field__hint">
                  Lower is better. The first line is how well it read the pictures it studied,
                  the second how well it read the ones it had never seen.
                </p>
              </div>
            </section>
          ) : null}

          <section className="card">
            <div className="card__head">
              <h3>
                <Sliders size={16} weight="bold" /> Settings
              </h3>
              <button
                className="btn btn--sm"
                onClick={() => updateSettings({ ...DETECT_SETTINGS })}
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
              <div className="field">
                <label className="switch">
                  <input
                    id="set-augment"
                    type="checkbox"
                    className="switch__input"
                    disabled={running}
                    checked={manifest.settings.augment}
                    onChange={(event) => updateSettings({ augment: event.target.checked })}
                  />
                  <span className="switch__box" aria-hidden="true">
                    <Check size={12} weight="bold" />
                  </span>
                  <span className="switch__text">Practise on mirrored copies</span>
                </label>
                <span className="field__hint">
                  Shows the detector a flipped, slightly relit copy of every picture as well as
                  the original, so it learns the thing rather than where the thing happened to
                  sit. Doubles the memory a run needs and the time it spends reading pictures.
                </span>
              </div>
            </div>
          </section>
        </aside>
      </div>
    </div>
  );
}

function Shot({
  verdict,
  src,
  colorFor,
  nameFor,
}: {
  verdict: DetectVerdict;
  src: string;
  colorFor: (id: string) => string;
  nameFor: (id: string) => string;
}) {
  const falseAlarms = verdict.hits.filter((hit) => hit < 0).length;
  const bad = verdict.missed.length > 0 || falseAlarms > 0;
  return (
    <article className={`dshot${bad ? " dshot--bad" : ""}`}>
      <div className="dshot__frame">
        <img src={src} alt="" loading="lazy" />
        {verdict.truth.map((box, i) => (
          <span
            key={`t${i}`}
            className={`dbox dbox--gt${verdict.missed.includes(i) ? " dbox--missed" : ""}`}
            style={{ ...place(box), borderColor: colorFor(box.labelId) }}
          />
        ))}
        {verdict.predicted.map((box, i) => (
          <span
            key={`p${i}`}
            className={`dbox dbox--pred${verdict.hits[i] < 0 ? " dbox--alarm" : ""}`}
            style={place(box)}
          >
            <span className="dbox__tag mono">
              {nameFor(box.labelId)} {Math.round(box.score * 100)}
            </span>
          </span>
        ))}
      </div>
      <div className="dshot__body">
        <span className="mono dshot__count">
          {verdict.truth.length} drawn · {verdict.predicted.length} read
        </span>
        {bad ? (
          <span className="dshot__flag mono">
            {verdict.missed.length > 0 ? `${verdict.missed.length} missed` : ""}
            {verdict.missed.length > 0 && falseAlarms > 0 ? " · " : ""}
            {falseAlarms > 0 ? `${falseAlarms} false` : ""}
          </span>
        ) : (
          <span className="badge badge--ok">clean</span>
        )}
      </div>
    </article>
  );
}
