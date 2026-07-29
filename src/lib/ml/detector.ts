// The trainable detector: an SSD style head on the frozen MobileNet grid.
//
// Where the classifier head sees one pooled 1024 value embedding per picture,
// this head sees the 7x7x1024 activation before pooling and answers per cell.
// Six anchors sit on every cell, and for each of them the head predicts four
// box offsets and one score per label plus background.
//
// Choices, all of them documented next to the code that depends on them:
//
// - Anchors and offset coding live in lib/detect.ts, which is pure and checked
//   in node by scripts/check-detect.mjs.
// - The head is a per cell stack of 1x1 layers and never a 3x3 convolution.
//   Two reasons. The cheap one is speed: a convolution over 1024 channels is
//   two orders of magnitude slower to train in a webview than the equivalent
//   matrix multiply. The one that decides it is that the WASM backend, which
//   is what a machine without WebGL falls back to, has no gradient kernel for
//   convolution filters at all, so a convolutional head simply cannot be
//   trained there. A dense head trains on both backends. It gives up nothing
//   obvious either: one cell of conv_pw_13_relu already looks at almost the
//   whole picture, so the neighbouring cells carry little a cell does not have.
// - Classification is a softmax over labels plus a background class, not a set
//   of independent sigmoids. A drawn box carries exactly one label, so the
//   classes really are exclusive, and a background class is what makes hard
//   negative mining a straight ranking of one number rather than a rule about
//   several. Inference reads scores as softmax probabilities with background
//   at index 0.
// - The localization loss is smooth L1 over positives only, the classification
//   loss is cross entropy over positives plus three mined negatives each.
//
// Training runs a hand written loop rather than model.fit, because the loss
// needs the anchor bookkeeping above and mining has to look at the current
// predictions before every step.

import * as tf from "@tensorflow/tfjs";
import type { StoredDetector, TrainSettings } from "../types";
import { bestEpochIndex, outOfPatience, PATIENCE } from "../curve";
import { FEATURE_SIZE } from "./features";
import {
  ANCHOR_RATIOS,
  ANCHOR_SCALES,
  anchorShapes,
  buildAnchors,
  buildTargets,
  decodeDetections,
  GRID_COLS,
  GRID_ROWS,
  mineNegatives,
  standardizeCells,
  type Anchor,
  type DecodeOptions,
  type Detection,
  type LabeledRect,
} from "../detect";

/** Weight of the box loss against the classification loss. */
const LOC_WEIGHT = 1;

/**
 * Starting bias of the background class. A fresh head that answers "nothing"
 * everywhere is a sane starting point: the alternative is a first step where
 * hundreds of anchors all shout a random label at once.
 */
const BACKGROUND_BIAS = 2;

/**
 * Dropout on the shared layer. A detect project is a handful of pictures with a
 * few hundred anchors each, which is exactly the shape of dataset a head can
 * memorize, so it gets a little noise and an early stop rather than more width.
 */
const DROPOUT = 0.2;

export interface Detector {
  model: tf.LayersModel;
  labelIds: string[];
  anchors: Anchor[];
  rows: number;
  cols: number;
  headWidth: number;
  scales: number[];
  ratios: number[];
}

export interface DetectEpoch {
  epoch: number;
  totalEpochs: number;
  loss: number;
  locLoss: number;
  clsLoss: number;
  valLoss: number | null;
}

/** One picture as the detector trains on it: cached grid features plus boxes. */
export interface DetectSample {
  imageId: string;
  /** rows * cols * FEATURE_SIZE floats, row major, exactly as features.ts hands them back. */
  features: Float32Array;
  boxes: LabeledRect[];
}

export interface DetectTrainInput {
  samples: DetectSample[];
  /** Held out pictures, used only to report a validation loss per epoch. */
  validation?: DetectSample[];
  labelIds: string[];
  settings: TrainSettings;
  /** Epochs without a better holdout loss before the run gives up. */
  patience?: number;
  onProgress?: (epoch: DetectEpoch) => void;
  shouldStop?: () => boolean;
}

export interface DetectTrainResult {
  detector: Detector;
  history: DetectEpoch[];
  /**
   * Epoch whose weights the returned detector carries, 1 based. Null when there
   * was no holdout to pick one with, in which case the run keeps the last.
   */
  bestEpoch: number | null;
  bestValLoss: number | null;
  /** True when patience ran out before the configured epochs did. */
  stoppedEarly: boolean;
}

export function buildDetector(
  labelIds: string[],
  headWidth: number,
  rows = GRID_ROWS,
  cols = GRID_COLS,
  scales: number[] = ANCHOR_SCALES,
  ratios: number[] = ANCHOR_RATIOS,
): Detector {
  const shapes = anchorShapes(scales, ratios);
  const perCell = shapes.length;
  const classes = labelIds.length + 1;
  const total = rows * cols * perCell;

  // One row per grid cell, exactly the layout features.ts hands back.
  const input = tf.input({ shape: [rows * cols, FEATURE_SIZE] });
  const mixed = tf.layers
    .dense({
      units: headWidth,
      activation: "relu",
      kernelInitializer: "varianceScaling",
      name: "detect_mix",
    })
    .apply(input) as tf.SymbolicTensor;
  const noised = tf.layers
    .dropout({ rate: DROPOUT, name: "detect_drop" })
    .apply(mixed) as tf.SymbolicTensor;

  const locCell = tf.layers
    .dense({ units: perCell * 4, kernelInitializer: "varianceScaling", name: "detect_loc" })
    .apply(noised) as tf.SymbolicTensor;
  const loc = tf.layers
    .reshape({ targetShape: [total, 4], name: "detect_loc_out" })
    .apply(locCell) as tf.SymbolicTensor;

  const clsCell = tf.layers
    .dense({ units: perCell * classes, kernelInitializer: "varianceScaling", name: "detect_cls" })
    .apply(noised) as tf.SymbolicTensor;
  const cls = tf.layers
    .reshape({ targetShape: [total, classes], name: "detect_cls_out" })
    .apply(clsCell) as tf.SymbolicTensor;

  const model = tf.model({ inputs: input, outputs: [loc, cls] });

  // getWeights hands back the live variable tensors, so the prior is written
  // through setWeights and nothing here is disposed by hand.
  const clsLayer = model.getLayer("detect_cls");
  const [kernel, bias] = clsLayer.getWeights();
  const prior = (bias.dataSync() as Float32Array).slice();
  for (let a = 0; a < perCell; a += 1) prior[a * classes] = BACKGROUND_BIAS;
  const priorTensor = tf.tensor1d(prior);
  clsLayer.setWeights([kernel, priorTensor]);
  priorTensor.dispose();

  return {
    model,
    labelIds,
    anchors: buildAnchors(rows, cols, shapes),
    rows,
    cols,
    headWidth,
    scales,
    ratios,
  };
}

/** Everything about one picture the loss needs, computed once per run. */
interface PreparedSample extends DetectSample {
  classes: Int32Array;
  offsets: Float32Array;
  state: Int8Array;
  positives: number;
}

function prepare(
  samples: DetectSample[],
  detector: Detector,
  labelIds: string[],
): PreparedSample[] {
  return samples.map((sample) => ({
    ...sample,
    features: standardizeCells(sample.features, detector.rows * detector.cols, FEATURE_SIZE),
    ...buildTargets(detector.anchors, sample.boxes, labelIds),
  }));
}

function stackFeatures(batch: PreparedSample[], rows: number, cols: number): tf.Tensor3D {
  const stride = rows * cols * FEATURE_SIZE;
  const flat = new Float32Array(batch.length * stride);
  batch.forEach((sample, i) => flat.set(sample.features, i * stride));
  return tf.tensor3d(flat, [batch.length, rows * cols, FEATURE_SIZE]);
}

/** Cross entropy of every anchor against its target, without any gradient. */
function anchorLosses(model: tf.LayersModel, xs: tf.Tensor3D, anchorCount: number): number[][] {
  return tf.tidy(() => {
    const [, logits] = model.predict(xs) as tf.Tensor[];
    const logp = tf.logSoftmax(logits as tf.Tensor3D, -1);
    // Mining ranks background anchors by how badly the head currently rates the
    // background class, which is exactly minus the log of its probability.
    const background = logp
      .slice([0, 0, 0], [xs.shape[0], anchorCount, 1])
      .reshape([xs.shape[0], anchorCount])
      .mul(-1) as tf.Tensor2D;
    return background.arraySync() as number[][];
  });
}

interface LossParts {
  total: number;
  loc: number;
  cls: number;
}

/** Builds the per batch tensors the loss reads, mining included. */
function batchTensors(batch: PreparedSample[], anchorCount: number, classCount: number) {
  const size = batch.length;
  const classTargets = new Float32Array(size * anchorCount * classCount);
  const locTargets = new Float32Array(size * anchorCount * 4);
  const positiveMask = new Float32Array(size * anchorCount);
  let positives = 0;
  batch.forEach((sample, i) => {
    locTargets.set(sample.offsets, i * anchorCount * 4);
    for (let a = 0; a < anchorCount; a += 1) {
      classTargets[(i * anchorCount + a) * classCount + sample.classes[a]] = 1;
      if (sample.state[a] === 1) {
        positiveMask[i * anchorCount + a] = 1;
        positives += 1;
      }
    }
  });
  return { classTargets, locTargets, positiveMask, positives };
}

function lossFor(
  model: tf.LayersModel,
  xs: tf.Tensor3D,
  classTargets: tf.Tensor3D,
  locTargets: tf.Tensor3D,
  positiveMask: tf.Tensor2D,
  classMask: tf.Tensor2D,
  norm: number,
  training: boolean,
): { total: tf.Scalar; loc: tf.Scalar; cls: tf.Scalar } {
  const [locPred, logits] = model.apply(xs, { training }) as tf.Tensor[];
  const logp = tf.logSoftmax(logits as tf.Tensor3D, -1);
  const ce = tf.sum(tf.mul(classTargets, logp), -1).mul(-1) as tf.Tensor2D;
  const cls = tf.sum(ce.mul(classMask)).div(norm) as tf.Scalar;

  const diff = tf.abs(tf.sub(locPred as tf.Tensor3D, locTargets));
  // Smooth L1: quadratic near zero so small corrections keep a useful gradient,
  // linear further out so one badly placed box cannot dominate a step. Written
  // through minimum rather than a comparison, because a comparison has no
  // gradient and would break the tape.
  const clipped = tf.minimum(diff, 1);
  const smooth = tf.add(tf.mul(tf.square(clipped), 0.5), tf.sub(diff, clipped));
  const loc = tf.sum(tf.sum(smooth, -1).mul(positiveMask)).div(norm) as tf.Scalar;

  return { total: cls.add(loc.mul(LOC_WEIGHT)) as tf.Scalar, loc, cls };
}

function runBatch(
  detector: Detector,
  batch: PreparedSample[],
  optimizer: tf.Optimizer | null,
): LossParts {
  const anchorCount = detector.anchors.length;
  const classCount = detector.labelIds.length + 1;
  const { classTargets, locTargets, positiveMask, positives } = batchTensors(
    batch,
    anchorCount,
    classCount,
  );
  const xs = stackFeatures(batch, detector.rows, detector.cols);

  // Mining looks at what the head believes right now, so it runs outside the
  // gradient tape and hands the loss a plain 0/1 weight per anchor.
  const losses = anchorLosses(detector.model, xs, anchorCount);
  const classMaskFlat = new Float32Array(batch.length * anchorCount);
  batch.forEach((sample, i) => {
    classMaskFlat.set(mineNegatives(losses[i], sample.state), i * anchorCount);
  });

  const norm = Math.max(1, positives);
  const classT = tf.tensor3d(classTargets, [batch.length, anchorCount, classCount]);
  const locT = tf.tensor3d(locTargets, [batch.length, anchorCount, 4]);
  const posT = tf.tensor2d(positiveMask, [batch.length, anchorCount]);
  const maskT = tf.tensor2d(classMaskFlat, [batch.length, anchorCount]);

  let parts: LossParts = { total: 0, loc: 0, cls: 0 };
  try {
    if (optimizer) {
      const held = tf.tidy(() => {
        let snapshot: { loc: number; cls: number } = { loc: 0, cls: 0 };
        const total = optimizer.minimize(() => {
          const loss = lossFor(detector.model, xs, classT, locT, posT, maskT, norm, true);
          snapshot = { loc: loss.loc.dataSync()[0], cls: loss.cls.dataSync()[0] };
          return loss.total;
        }, true) as tf.Scalar;
        const value = total.dataSync()[0];
        total.dispose();
        return { total: value, ...snapshot };
      });
      parts = held;
    } else {
      parts = tf.tidy(() => {
        const loss = lossFor(detector.model, xs, classT, locT, posT, maskT, norm, false);
        return {
          total: loss.total.dataSync()[0],
          loc: loss.loc.dataSync()[0],
          cls: loss.cls.dataSync()[0],
        };
      });
    }
  } finally {
    xs.dispose();
    classT.dispose();
    locT.dispose();
    posT.dispose();
    maskT.dispose();
  }
  return parts;
}

/** The head's weights as plain arrays, cheap enough to keep one copy of. */
function snapshotWeights(model: tf.LayersModel): Float32Array[] {
  return model.getWeights().map((w) => (w.dataSync() as Float32Array).slice());
}

function restoreWeights(model: tf.LayersModel, snapshot: Float32Array[]): void {
  const shapes = model.getWeights().map((w) => w.shape.slice());
  const tensors = snapshot.map((values, i) => tf.tensor(values, shapes[i]));
  try {
    model.setWeights(tensors);
  } finally {
    tensors.forEach((t) => t.dispose());
  }
}

/**
 * Mean loss of a detector over a set of pictures, with dropout off and no
 * gradient. Same batching as the training loop's holdout pass, so the number it
 * returns is the same number that pass reports, which is what lets
 * scripts/check-learning.mjs prove the best epoch really was restored.
 */
export function evaluateDetector(
  detector: Detector,
  samples: DetectSample[],
  batchSize = 8,
): number | null {
  if (samples.length === 0) return null;
  const prepared = prepare(samples, detector, detector.labelIds);
  const size = Math.max(1, Math.min(batchSize, prepared.length));
  let sum = 0;
  let batches = 0;
  for (let start = 0; start < prepared.length; start += size) {
    sum += runBatch(detector, prepared.slice(start, start + size), null).total;
    batches += 1;
  }
  return batches === 0 ? null : sum / batches;
}

function shuffled(count: number): number[] {
  const order = Array.from({ length: count }, (_, i) => i);
  for (let i = order.length - 1; i > 0; i -= 1) {
    const j = Math.floor(Math.random() * (i + 1));
    [order[i], order[j]] = [order[j], order[i]];
  }
  return order;
}

export async function trainDetector(input: DetectTrainInput): Promise<DetectTrainResult> {
  const { samples, labelIds, settings } = input;
  if (samples.length === 0) throw new Error("There is nothing to train on yet");

  const detector = buildDetector(labelIds, Math.max(16, settings.hiddenUnits));
  const prepared = prepare(samples, detector, labelIds);
  const validation = input.validation?.length
    ? prepare(input.validation, detector, labelIds)
    : null;
  const optimizer = tf.train.adam(settings.learningRate);
  const batchSize = Math.max(1, Math.min(settings.batchSize, prepared.length));
  const history: DetectEpoch[] = [];

  // Early stopping bookkeeping. Only ever used when there is a holdout: without
  // one there is no honest signal to pick an epoch by, and picking by training
  // loss would reliably choose the most memorized head in the run.
  let bestWeights: Float32Array[] | null = null;
  let bestEpoch: number | null = null;
  let bestValLoss: number | null = null;
  let stoppedEarly = false;

  try {
    for (let epoch = 0; epoch < settings.epochs; epoch += 1) {
      if (input.shouldStop?.()) throw new Error("Training cancelled");
      const order = shuffled(prepared.length);
      let total = 0;
      let loc = 0;
      let cls = 0;
      let steps = 0;
      for (let start = 0; start < order.length; start += batchSize) {
        if (input.shouldStop?.()) throw new Error("Training cancelled");
        const batch = order.slice(start, start + batchSize).map((i) => prepared[i]);
        const parts = runBatch(detector, batch, optimizer);
        total += parts.total;
        loc += parts.loc;
        cls += parts.cls;
        steps += 1;
        // Yield so the progress bar paints and the window stays alive.
        await tf.nextFrame();
      }

      let valLoss: number | null = null;
      if (validation) {
        let sum = 0;
        let batches = 0;
        for (let start = 0; start < validation.length; start += batchSize) {
          if (input.shouldStop?.()) throw new Error("Training cancelled");
          sum += runBatch(detector, validation.slice(start, start + batchSize), null).total;
          batches += 1;
        }
        valLoss = batches === 0 ? null : sum / batches;
      }

      const record: DetectEpoch = {
        epoch: epoch + 1,
        totalEpochs: settings.epochs,
        loss: steps === 0 ? 0 : total / steps,
        locLoss: steps === 0 ? 0 : loc / steps,
        clsLoss: steps === 0 ? 0 : cls / steps,
        valLoss,
      };
      history.push(record);
      input.onProgress?.(record);
      await tf.nextFrame();

      if (!validation) continue;
      const holdout = history.map((entry) => entry.valLoss);
      // bestEpochIndex is the same rule the Train view reads the curve with, so
      // the epoch reported as kept is the epoch actually kept.
      const best = bestEpochIndex(holdout);
      if (best === history.length - 1) {
        bestWeights = snapshotWeights(detector.model);
        bestEpoch = record.epoch;
        bestValLoss = record.valLoss;
      }
      if (outOfPatience(holdout, input.patience ?? PATIENCE)) {
        stoppedEarly = true;
        break;
      }
    }

    // A run with a holdout keeps its best epoch rather than its last one. The
    // last epoch is the most memorized head in the run by construction, and on
    // a project of a few dozen pictures the difference is the whole game. A run
    // without a holdout keeps what it ends on, exactly as before, because there
    // is nothing to choose with.
    if (bestWeights) restoreWeights(detector.model, bestWeights);

    return { detector, history, bestEpoch, bestValLoss, stoppedEarly };
  } catch (err) {
    detector.model.dispose();
    throw err;
  } finally {
    optimizer.dispose();
  }
}

/**
 * The inference contract phase 3 consumes.
 *
 * Takes one picture's grid features, either the Float32Array the feature cache
 * holds or the [rows * cols, FEATURE_SIZE] tensor features.ts hands back, and
 * returns decoded detections in normalized image coordinates with the top left
 * corner at x/y. Thresholding and per label non maximum suppression happen
 * here, so a caller gets a list it can draw without any further filtering.
 */
export function detect(
  detector: Detector,
  cells: Float32Array | tf.Tensor2D,
  options: DecodeOptions = {},
): Detection[] {
  const { rows, cols } = detector;
  const raw = cells instanceof Float32Array ? cells : (cells.dataSync() as Float32Array);
  const standardized = standardizeCells(raw, rows * cols, FEATURE_SIZE);
  const [scores, offsets] = tf.tidy(() => {
    const xs = tf.tensor3d(standardized, [1, rows * cols, FEATURE_SIZE]);
    const [loc, logits] = detector.model.predict(xs) as tf.Tensor[];
    const probs = tf.softmax(logits as tf.Tensor3D, -1);
    return [
      probs.dataSync() as Float32Array,
      (loc as tf.Tensor3D).dataSync() as Float32Array,
    ] as const;
  });
  return decodeDetections(scores, offsets, detector.anchors, detector.labelIds, options);
}

/**
 * Use-view seam from one MobileNet spatial map to drawable boxes. Keeping this
 * named path outside React lets the node learning proof exercise the same
 * validation, head pass and decoding that live and single-image inference use.
 * The caller still owns a tensor and must dispose it after this returns.
 */
export function detectSpatialMap(
  detector: Detector,
  cells: Float32Array | tf.Tensor2D,
  options: DecodeOptions = {},
): Detection[] {
  const expected = detector.rows * detector.cols * FEATURE_SIZE;
  const actual = cells instanceof Float32Array ? cells.length : cells.size;
  if (actual !== expected) {
    throw new Error(`Detector expected ${expected} spatial values, got ${actual}`);
  }
  return detect(detector, cells, options);
}

export function serializeDetector(detector: Detector): StoredDetector {
  return {
    formatVersion: 1,
    labelIds: detector.labelIds,
    featureSize: FEATURE_SIZE,
    rows: detector.rows,
    cols: detector.cols,
    scales: detector.scales,
    ratios: detector.ratios,
    headWidth: detector.headWidth,
    weights: detector.model.getWeights().map((w) => ({
      shape: w.shape.slice(),
      values: Array.from(w.dataSync()),
    })),
  };
}

export function deserializeDetector(stored: StoredDetector): Detector | null {
  if (stored.featureSize !== FEATURE_SIZE) return null;
  const detector = buildDetector(
    stored.labelIds,
    stored.headWidth,
    stored.rows,
    stored.cols,
    stored.scales,
    stored.ratios,
  );
  const expected = detector.model.getWeights();
  if (expected.length !== stored.weights.length) {
    detector.model.dispose();
    return null;
  }
  const tensors = stored.weights.map((w) => tf.tensor(w.values, w.shape));
  try {
    detector.model.setWeights(tensors);
  } catch {
    detector.model.dispose();
    tensors.forEach((t) => t.dispose());
    return null;
  }
  tensors.forEach((t) => t.dispose());
  return detector;
}
