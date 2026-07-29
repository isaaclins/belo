// Frozen MobileNet v1 (1.0, 224) feature extractor. The graph is truncated at
// conv_pw_13_relu and spatially averaged, which yields a 1024 value embedding
// per image. Weights are bundled under public/models so nothing is fetched
// from the network at runtime.

import * as tf from "@tensorflow/tfjs";
import { IMAGE_SIZE } from "../types";
import { initBackend } from "./backend";
import { isIdentity, LUMA, NO_AUGMENT, type Augment } from "../augment";
import type { Drawable } from "../images";

const MODEL_URL = "/models/mobilenet-v1-1.0-224/model.json";
const CUT_LAYER = "conv_pw_13_relu";

export const FEATURE_SIZE = 1024;

/**
 * Applies one training time augmentation to a [height, width, 3] pixel tensor
 * whose values are in the 0 to 255 range. Returns a new tensor; the caller
 * still owns the one it passed in.
 *
 * Order is flip, then saturation, then brightness, then a clamp back into
 * range, so a brightened picture cannot wrap around into a dark one. It lives
 * here rather than with the rest of augment.ts because this is the file that
 * owns what a picture looks like by the time the frozen extractor sees it, and
 * because keeping the plan pure lets it be checked without TensorFlow.
 */
export function applyAugment(pixels: tf.Tensor3D, aug: Augment): tf.Tensor3D {
  if (isIdentity(aug)) return pixels.clone();
  return tf.tidy(() => {
    // Axis 1 is width in [height, width, channels], so a left to right mirror
    // is a reverse along it.
    let out = (aug.flip ? tf.reverse(pixels, 1) : pixels).toFloat() as tf.Tensor3D;
    if (aug.saturation !== 1) {
      const grey = tf.sum(tf.mul(out, tf.tensor1d(LUMA)), -1, true);
      out = tf.add(grey, tf.mul(tf.sub(out, grey), aug.saturation)) as tf.Tensor3D;
    }
    if (aug.brightness !== 1) out = tf.mul(out, aug.brightness) as tf.Tensor3D;
    return tf.clipByValue(out, 0, 255) as tf.Tensor3D;
  });
}

let loading: Promise<tf.LayersModel> | null = null;

export function loadFeatureModel(): Promise<tf.LayersModel> {
  if (!loading) {
    loading = (async () => {
      await initBackend();
      const full = await tf.loadLayersModel(MODEL_URL);
      const cut = full.getLayer(CUT_LAYER);
      const truncated = tf.model({ inputs: full.inputs, outputs: cut.output });
      // One warm-up pass so the first real capture is not the slow one.
      tf.tidy(() => truncated.predict(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3])) as tf.Tensor);
      return truncated;
    })().catch((err) => {
      loading = null;
      throw err;
    });
  }
  return loading;
}

/**
 * MobileNet v1 expects pixels scaled into the -1 to 1 range.
 *
 * Augmentation happens here, between the resize and the scaling, which is the
 * only place it can: it is a transform of pixels, and everything downstream of
 * this function is already a network activation.
 */
function preprocess(source: Drawable, aug: Augment = NO_AUGMENT): tf.Tensor4D {
  return tf.tidy(() => {
    const pixels = tf.browser.fromPixels(source);
    const resized =
      pixels.shape[0] === IMAGE_SIZE && pixels.shape[1] === IMAGE_SIZE
        ? pixels
        : tf.image.resizeBilinear(pixels, [IMAGE_SIZE, IMAGE_SIZE], true);
    const varied = applyAugment(resized.toFloat() as tf.Tensor3D, aug);
    return varied.div(127.5).sub(1).expandDims(0) as tf.Tensor4D;
  });
}

/**
 * The pre-pooling activation of the cut layer, one 1024 value vector per grid
 * cell, laid out row major as [rows * cols, FEATURE_SIZE]. The caller owns the
 * tensor and must dispose it.
 */
export interface SpatialFeatures {
  cells: tf.Tensor2D;
  rows: number;
  cols: number;
}

export interface EmbedResult {
  pooled: Float32Array;
  spatial: SpatialFeatures | null;
}

/** What one pass hands back out of tf.tidy. Nothing else survives the scope. */
interface HeldFeatures {
  // tf.tidy only accepts containers it can walk, which means an index
  // signature over the values it knows how to keep.
  [key: string]: tf.Tensor2D | number | undefined;
  pooled: tf.Tensor2D;
  cells?: tf.Tensor2D;
  rows: number;
  cols: number;
}

/** Returns a [1, FEATURE_SIZE] embedding. The caller owns the tensor. */
export async function embed(source: Drawable): Promise<tf.Tensor2D> {
  const model = await loadFeatureModel();
  return tf.tidy(() => {
    const input = preprocess(source);
    const activation = model.predict(input) as tf.Tensor4D;
    return activation.mean([1, 2]) as tf.Tensor2D;
  });
}

/**
 * One MobileNet pass that can yield both halves of the same activation: the
 * pooled embedding every caller needs, and optionally the spatial grid the
 * attention overlay scores cell by cell. Asking for both never runs the
 * extractor twice.
 */
export async function embedWithSpatial(
  source: Drawable,
  wantSpatial: boolean,
  aug: Augment = NO_AUGMENT,
): Promise<EmbedResult> {
  const model = await loadFeatureModel();
  // tf.tidy keeps whatever the scope returns and frees the rest, so both
  // outputs of the single pass survive and every intermediate is dropped.
  const held = tf.tidy<HeldFeatures>(() => {
    const input = preprocess(source, aug);
    const activation = model.predict(input) as tf.Tensor4D;
    const [, rows, cols, depth] = activation.shape;
    const kept: HeldFeatures = {
      pooled: activation.mean([1, 2]) as tf.Tensor2D,
      rows,
      cols,
    };
    if (wantSpatial) kept.cells = activation.reshape([rows * cols, depth]) as tf.Tensor2D;
    return kept;
  });
  try {
    const pooled = (await held.pooled.data()) as Float32Array;
    return {
      pooled,
      spatial: held.cells ? { cells: held.cells, rows: held.rows, cols: held.cols } : null,
    };
  } catch (err) {
    held.cells?.dispose();
    throw err;
  } finally {
    held.pooled.dispose();
  }
}

export async function embedToArray(source: Drawable): Promise<Float32Array> {
  const result = await embedWithSpatial(source, false);
  return result.pooled;
}
