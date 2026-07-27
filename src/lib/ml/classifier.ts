// The trainable part: a small dense head on top of frozen MobileNet features.
// Training runs entirely in the webview and reports progress per epoch.

import * as tf from "@tensorflow/tfjs";
import type { ProjectMode, StoredHead, TrainSettings } from "../types";
import { FEATURE_SIZE } from "./features";

export interface TrainProgress {
  epoch: number;
  totalEpochs: number;
  loss: number;
  accuracy: number;
  valLoss: number | null;
  valAccuracy: number | null;
}

export interface TrainResult {
  model: tf.Sequential;
  accuracy: number;
  valAccuracy: number | null;
  history: TrainProgress[];
}

export function buildHead(
  numClasses: number,
  settings: TrainSettings,
  mode: ProjectMode = "single",
): tf.Sequential {
  const multi = mode === "multi";
  const model = tf.sequential();
  model.add(
    tf.layers.dense({
      inputShape: [FEATURE_SIZE],
      units: settings.hiddenUnits,
      activation: "relu",
      kernelInitializer: "varianceScaling",
      useBias: true,
    }),
  );
  model.add(
    tf.layers.dense({
      units: numClasses,
      // Multi mode scores every label on its own, so the outputs must not
      // compete the way a softmax makes them.
      activation: multi ? "sigmoid" : "softmax",
      kernelInitializer: "varianceScaling",
      useBias: multi,
    }),
  );
  model.compile({
    optimizer: tf.train.adam(settings.learningRate),
    loss:
      multi || numClasses === 2 ? "binaryCrossentropy" : "categoricalCrossentropy",
    metrics: ["accuracy"],
  });
  return model;
}

export interface TrainInput {
  features: Float32Array[];
  /** One row per sample: a one hot vector in single mode, a 0/1 set in multi. */
  targets: number[][];
  numClasses: number;
  mode: ProjectMode;
  settings: TrainSettings;
  onProgress?: (progress: TrainProgress) => void;
  shouldStop?: () => boolean;
}

export async function trainHead(input: TrainInput): Promise<TrainResult> {
  const { features, targets, numClasses, mode, settings } = input;
  const sampleCount = features.length;
  if (sampleCount === 0) throw new Error("There is nothing to train on yet");

  const flat = new Float32Array(sampleCount * FEATURE_SIZE);
  features.forEach((f, i) => flat.set(f, i * FEATURE_SIZE));

  const flatTargets = new Float32Array(sampleCount * numClasses);
  targets.forEach((row, i) => flatTargets.set(row, i * numClasses));

  const xs = tf.tensor2d(flat, [sampleCount, FEATURE_SIZE]);
  const ys = tf.tensor2d(flatTargets, [sampleCount, numClasses]);

  // Only hold out a validation split when every class can spare examples.
  const canValidate = sampleCount >= 20 && settings.validationSplit > 0;
  const model = buildHead(numClasses, settings, mode);
  const history: TrainProgress[] = [];

  try {
    await model.fit(xs, ys, {
      epochs: settings.epochs,
      batchSize: Math.max(1, Math.min(settings.batchSize, sampleCount)),
      shuffle: true,
      validationSplit: canValidate ? settings.validationSplit : 0,
      callbacks: {
        onEpochEnd: async (epoch, logs) => {
          const progress: TrainProgress = {
            epoch: epoch + 1,
            totalEpochs: settings.epochs,
            loss: Number(logs?.loss ?? 0),
            accuracy: Number(logs?.acc ?? logs?.accuracy ?? 0),
            valLoss: logs?.val_loss === undefined ? null : Number(logs.val_loss),
            valAccuracy:
              logs?.val_acc === undefined && logs?.val_accuracy === undefined
                ? null
                : Number(logs?.val_acc ?? logs?.val_accuracy),
          };
          history.push(progress);
          input.onProgress?.(progress);
          if (input.shouldStop?.()) model.stopTraining = true;
          // Yield to the compositor so the progress bar actually paints.
          await tf.nextFrame();
        },
      },
    });
  } finally {
    xs.dispose();
    ys.dispose();
  }

  const last = history[history.length - 1];
  return {
    model,
    accuracy: last?.accuracy ?? 0,
    valAccuracy: last?.valAccuracy ?? null,
    history,
  };
}

/** Runs the head over a batch of embeddings and returns per class scores. */
export function predictBatch(model: tf.Sequential, features: Float32Array[]): number[][] {
  if (features.length === 0) return [];
  return tf.tidy(() => {
    const flat = new Float32Array(features.length * FEATURE_SIZE);
    features.forEach((f, i) => flat.set(f, i * FEATURE_SIZE));
    const xs = tf.tensor2d(flat, [features.length, FEATURE_SIZE]);
    const out = model.predict(xs) as tf.Tensor2D;
    return out.arraySync() as number[][];
  });
}

export function predictOne(model: tf.Sequential, feature: Float32Array): number[] {
  return predictBatch(model, [feature])[0] ?? [];
}

export function serializeHead(
  model: tf.Sequential,
  labelIds: string[],
  hiddenUnits: number,
  mode: ProjectMode = "single",
): StoredHead {
  return {
    formatVersion: 1,
    mode,
    labelIds,
    hiddenUnits,
    featureSize: FEATURE_SIZE,
    weights: model.getWeights().map((w) => ({
      shape: w.shape.slice(),
      values: Array.from(w.dataSync()),
    })),
  };
}

export function deserializeHead(stored: StoredHead, settings: TrainSettings): tf.Sequential | null {
  if (stored.featureSize !== FEATURE_SIZE) return null;
  // Heads written before multi mode carry no mode and are always softmax.
  const model = buildHead(
    stored.labelIds.length,
    { ...settings, hiddenUnits: stored.hiddenUnits },
    stored.mode === "multi" ? "multi" : "single",
  );
  const expected = model.getWeights();
  if (expected.length !== stored.weights.length) {
    model.dispose();
    return null;
  }
  const tensors = stored.weights.map((w) => tf.tensor(w.values, w.shape));
  try {
    model.setWeights(tensors);
  } catch {
    model.dispose();
    tensors.forEach((t) => t.dispose());
    return null;
  }
  tensors.forEach((t) => t.dispose());
  return model;
}
