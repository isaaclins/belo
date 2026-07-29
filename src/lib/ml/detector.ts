// The trainable detector: an SSD style head on two frozen MobileNet grids.
//
// Where the classifier head sees one pooled 1024 value embedding per picture,
// this head sees the activations before pooling and answers per cell, on two
// maps at once: the 7x7x1024 cut at conv_pw_13_relu and the 14x14x512 cut one
// stage earlier. Six anchors sit on every cell of both, and for each of them
// the head predicts four box offsets and one score per label plus background.
//
// The second map is there for placement, not for recognition. One cell of
// conv_pw_13_relu already sees most of the picture and knows perfectly well
// that there is a dog in it; what it cannot do is say where the dog ends, since
// every box it proposes starts from a grid whose centers are 32 pixels apart.
// On real photographs that was the dominant failure: the right object, at the
// wrong size, just under the half overlap a detection is counted at.
//
// Choices, all of them documented next to the code that depends on them:
//
// - Anchors, feature scales and offset coding live in lib/detect.ts, which is
//   pure and checked in node by scripts/check-detect.mjs.
// - Each map gets its own small branch, and the branches are concatenated
//   coarse first. Sharing one branch is impossible: the maps have different
//   depths. Concatenating in a fixed order is what lets an old single grid
//   model be exactly the first branch of a new one, anchors included.
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
  anchorShapes,
  buildScaleAnchors,
  buildTargets,
  decodeDetections,
  DETECT_SCALES,
  featureLength,
  legacyScale,
  mineNegatives,
  standardizeScales,
  type Anchor,
  type DecodeOptions,
  type Detection,
  type FeatureScale,
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
  headWidth: number;
  /** Feature maps the head reads, coarse first. One entry on a restored v1 head. */
  featureScales: FeatureScale[];
  /** Floats one example has to carry for this head, sum over the maps. */
  featureLength: number;
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
  /** The packed maps, coarse then fine, exactly as features.ts hands them back. */
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

/** Layer names, suffixed per map so the first branch keeps the v1 names. */
function named(base: string, index: number): string {
  return index === 0 ? base : `${base}_${index}`;
}

export function buildDetector(
  labelIds: string[],
  headWidth: number,
  featureScales: FeatureScale[] = DETECT_SCALES,
): Detector {
  const classes = labelIds.length + 1;
  const inputs: tf.SymbolicTensor[] = [];
  const locParts: tf.SymbolicTensor[] = [];
  const clsParts: tf.SymbolicTensor[] = [];
  const clsLayerNames: string[] = [];

  featureScales.forEach((scale, index) => {
    const perCell = anchorShapes(scale.scales, scale.ratios).length;
    const cells = scale.rows * scale.cols;
    const total = cells * perCell;

    // One row per grid cell, exactly the layout features.ts hands back.
    const input = tf.input({ shape: [cells, scale.depth] });
    inputs.push(input);
    const mixed = tf.layers
      .dense({
        units: headWidth,
        activation: "relu",
        kernelInitializer: "varianceScaling",
        name: named("detect_mix", index),
      })
      .apply(input) as tf.SymbolicTensor;
    const noised = tf.layers
      .dropout({ rate: DROPOUT, name: named("detect_drop", index) })
      .apply(mixed) as tf.SymbolicTensor;

    const locName = named("detect_loc", index);
    const locCell = tf.layers
      .dense({ units: perCell * 4, kernelInitializer: "varianceScaling", name: locName })
      .apply(noised) as tf.SymbolicTensor;
    locParts.push(
      tf.layers
        .reshape({ targetShape: [total, 4], name: `${locName}_out` })
        .apply(locCell) as tf.SymbolicTensor,
    );

    const clsName = named("detect_cls", index);
    clsLayerNames.push(clsName);
    const clsCell = tf.layers
      .dense({ units: perCell * classes, kernelInitializer: "varianceScaling", name: clsName })
      .apply(noised) as tf.SymbolicTensor;
    clsParts.push(
      tf.layers
        .reshape({ targetShape: [total, classes], name: `${clsName}_out` })
        .apply(clsCell) as tf.SymbolicTensor,
    );
  });

  // Anchor index a means the same row of the same output on both sides, so the
  // maps are joined in exactly the order buildScaleAnchors walks them.
  const join = (parts: tf.SymbolicTensor[], name: string) =>
    parts.length === 1
      ? parts[0]
      : (tf.layers.concatenate({ axis: 1, name }).apply(parts) as tf.SymbolicTensor);

  const model = tf.model({
    inputs: inputs.length === 1 ? inputs[0] : inputs,
    outputs: [join(locParts, "detect_loc_all"), join(clsParts, "detect_cls_all")],
  });

  // getWeights hands back the live variable tensors, so the prior is written
  // through setWeights and nothing here is disposed by hand.
  clsLayerNames.forEach((name, index) => {
    const scale = featureScales[index];
    const perCell = anchorShapes(scale.scales, scale.ratios).length;
    const clsLayer = model.getLayer(name);
    const [kernel, bias] = clsLayer.getWeights();
    const prior = (bias.dataSync() as Float32Array).slice();
    for (let a = 0; a < perCell; a += 1) prior[a * classes] = BACKGROUND_BIAS;
    const priorTensor = tf.tensor1d(prior);
    clsLayer.setWeights([kernel, priorTensor]);
    priorTensor.dispose();
  });

  return {
    model,
    labelIds,
    anchors: buildScaleAnchors(featureScales),
    headWidth,
    featureScales,
    featureLength: featureLength(featureScales),
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
    features: standardizeScales(sample.features, detector.featureScales),
    ...buildTargets(detector.anchors, sample.boxes, labelIds),
  }));
}

/** One [batch, cells, depth] tensor per feature map, sliced out of the pack. */
function stackFeatures(batch: PreparedSample[], scales: FeatureScale[]): tf.Tensor3D[] {
  let at = 0;
  return scales.map((scale) => {
    const cells = scale.rows * scale.cols;
    const length = cells * scale.depth;
    const from = at;
    at += length;
    const flat = new Float32Array(batch.length * length);
    batch.forEach((sample, i) => {
      flat.set(sample.features.subarray(from, from + length), i * length);
    });
    return tf.tensor3d(flat, [batch.length, cells, scale.depth]);
  });
}

/** Cross entropy of every anchor against its target, without any gradient. */
function anchorLosses(model: tf.LayersModel, xs: tf.Tensor3D[], anchorCount: number): number[][] {
  return tf.tidy(() => {
    const size = xs[0].shape[0];
    const [, logits] = model.predict(xs) as tf.Tensor[];
    const logp = tf.logSoftmax(logits as tf.Tensor3D, -1);
    // Mining ranks background anchors by how badly the head currently rates the
    // background class, which is exactly minus the log of its probability.
    const background = logp
      .slice([0, 0, 0], [size, anchorCount, 1])
      .reshape([size, anchorCount])
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
  xs: tf.Tensor3D[],
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
  const xs = stackFeatures(batch, detector.featureScales);

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
    xs.forEach((x) => x.dispose());
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
 * Takes one picture's packed grid features, either the Float32Array the feature
 * cache holds or the tensor features.ts hands back, and returns decoded
 * detections in normalized image coordinates with the top left corner at x/y.
 * Thresholding and per label non maximum suppression happen here, so a caller
 * gets a list it can draw without any further filtering.
 *
 * A pack longer than the head asks for is read as a prefix, which is what lets
 * a detect model saved before the fine map existed run on the features the app
 * now caches.
 */
export function detect(
  detector: Detector,
  cells: Float32Array | tf.Tensor2D,
  options: DecodeOptions = {},
): Detection[] {
  const raw = cells instanceof Float32Array ? cells : (cells.dataSync() as Float32Array);
  const standardized = standardizeScales(raw, detector.featureScales);
  const [scores, offsets] = tf.tidy(() => {
    let at = 0;
    const xs = detector.featureScales.map((scale) => {
      const cellCount = scale.rows * scale.cols;
      const length = cellCount * scale.depth;
      const slice = standardized.subarray(at, at + length);
      at += length;
      return tf.tensor3d(slice, [1, cellCount, scale.depth]);
    });
    const [loc, logits] = detector.model.predict(xs.length === 1 ? xs[0] : xs) as tf.Tensor[];
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
  const expected = detector.featureLength;
  const actual = cells instanceof Float32Array ? cells.length : cells.size;
  if (actual < expected) {
    throw new Error(`Detector expected ${expected} spatial values, got ${actual}`);
  }
  return detect(detector, cells, options);
}

/**
 * Version 1 is a head on one 7x7 map. Version 2 carries a featureScales list
 * and may hold any number of maps. Both are written and read here: a saved
 * version 1 detector rebuilds its single branch and keeps working, and version
 * 2 is what every new run produces.
 */
export const DETECTOR_FORMAT_VERSION = 2;

export function serializeDetector(detector: Detector): StoredDetector {
  const first = detector.featureScales[0];
  return {
    formatVersion: DETECTOR_FORMAT_VERSION,
    labelIds: detector.labelIds,
    featureSize: FEATURE_SIZE,
    // The version 1 fields keep describing the coarse map, so an older reader
    // that only knows them sees something true rather than something absent.
    rows: first.rows,
    cols: first.cols,
    scales: first.scales,
    ratios: first.ratios,
    featureScales: detector.featureScales,
    headWidth: detector.headWidth,
    weights: detector.model.getWeights().map((w) => ({
      shape: w.shape.slice(),
      values: Array.from(w.dataSync()),
    })),
  };
}

/** The feature maps a stored head was trained on, whichever version wrote it. */
export function storedScales(stored: StoredDetector): FeatureScale[] {
  if (stored.featureScales?.length) return stored.featureScales;
  return [legacyScale(stored.rows, stored.cols, stored.featureSize, stored.scales, stored.ratios)];
}

export function deserializeDetector(stored: StoredDetector): Detector | null {
  if (stored.featureSize !== FEATURE_SIZE) return null;
  const detector = buildDetector(stored.labelIds, stored.headWidth, storedScales(stored));
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
