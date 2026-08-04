// Frozen MobileNet v1 (1.0, 224) feature extractor. The graph is truncated at
// conv_pw_13_relu and spatially averaged, which yields a 1024 value embedding
// per image. Weights are bundled under public/models so nothing is fetched
// from the network at runtime.
//
// The detector needs a second, finer view of the same picture, so there is a
// second truncation at conv_pw_11_relu, the stage before MobileNet's last
// stride. Both come out of one pass: the two cuts share every layer below
// conv_pw_12, so a model with both as outputs costs one forward pass, not two.

import * as tf from "@tensorflow/tfjs";
import { IMAGE_SIZE } from "../types";
import { initBackend } from "./backend";
import { isIdentity, LUMA, NO_AUGMENT, type Augment } from "../augment";
import type { Drawable } from "../images";
import { COARSE_SCALE, DETECT_SCALES, featureLength } from "../detect";

const MODEL_URL = "/models/mobilenet-v1-1.0-224/model.json";
const CUT_LAYER = "conv_pw_13_relu";
const FINE_CUT_LAYER = "conv_pw_11_relu";

export const FEATURE_SIZE = 1024;
export const FINE_FEATURE_SIZE = 512;

/**
 * Floats one packed detect example holds: the 7x7x1024 map followed by the
 * 14x14x512 one, 150528 in all. That is 588 KB per view against the 196 KB a
 * single map cost, so the training cache of a hundred picture project grows
 * from about 20 MB to about 59 MB, and doubles again to 118 MB when practising
 * on mirrored copies is on. It is the price of the second grid and it is paid
 * once per picture per run rather than once per epoch.
 */
export const DETECT_PACK_SIZE = featureLength(DETECT_SCALES);

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

let loadingFull: Promise<tf.LayersModel> | null = null;
let loading: Promise<tf.LayersModel> | null = null;
let loadingDetect: Promise<tf.LayersModel> | null = null;
let fullModel: tf.LayersModel | null = null;
let featureModel: tf.LayersModel | null = null;
let detectFeatureModel: tf.LayersModel | null = null;
let loadGeneration = 0;

function loadFull(): Promise<tf.LayersModel> {
  if (!loadingFull) {
    const generation = loadGeneration;
    const pending = (async () => {
      await initBackend();
      const full = await tf.loadLayersModel(MODEL_URL);
      if (generation !== loadGeneration) {
        full.dispose();
        throw new Error("The feature extractor was reset while it was loading");
      }
      fullModel = full;
      return full;
    })();
    const tracked = pending.catch((err) => {
      if (loadingFull === tracked) loadingFull = null;
      throw err;
    });
    loadingFull = tracked;
  }
  return loadingFull;
}

export function loadFeatureModel(): Promise<tf.LayersModel> {
  if (!loading) {
    const generation = loadGeneration;
    const pending = (async () => {
      const full = await loadFull();
      const cut = full.getLayer(CUT_LAYER);
      const truncated = tf.model({ inputs: full.inputs, outputs: cut.output });
      // One warm-up pass so the first real capture is not the slow one.
      tf.tidy(() => truncated.predict(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3])) as tf.Tensor);
      if (generation !== loadGeneration) {
        throw new Error("The feature extractor was reset while it was loading");
      }
      featureModel = truncated;
      return truncated;
    })();
    const tracked = pending.catch((err) => {
      if (loading === tracked) loading = null;
      throw err;
    });
    loading = tracked;
  }
  return loading;
}

/**
 * The same frozen graph with both cuts as outputs, coarse first. It shares its
 * layers and its weights with the truncation above, so a project that trains a
 * detector and classifies at the same time still holds one copy of MobileNet.
 */
export function loadDetectFeatureModel(): Promise<tf.LayersModel> {
  if (!loadingDetect) {
    const generation = loadGeneration;
    const pending = (async () => {
      const full = await loadFull();
      const both = tf.model({
        inputs: full.inputs,
        outputs: [full.getLayer(CUT_LAYER).output, full.getLayer(FINE_CUT_LAYER).output] as
          tf.SymbolicTensor[],
      });
      // One warm-up pass so the first real capture is not the slow one. tidy
      // keeps whatever a callback returns, and this one has two outputs to
      // return, so they are disposed by hand instead.
      const warm = tf.tidy(() => {
        const zeros = tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3]);
        return (both.predict(zeros) as tf.Tensor[]).map((t) => t.reshape([t.size]));
      });
      warm.forEach((t) => t.dispose());
      if (generation !== loadGeneration) {
        throw new Error("The feature extractor was reset while it was loading");
      }
      detectFeatureModel = both;
      return both;
    })();
    const tracked = pending.catch((err) => {
      if (loadingDetect === tracked) loadingDetect = null;
      throw err;
    });
    loadingDetect = tracked;
  }
  return loadingDetect;
}

/**
 * Drops every graph that owns MobileNet variables. A WebGL context failure
 * invalidates those variables when the backend changes, so keeping the cached
 * graph would make the next WASM pass use dead tensors.
 */
export function resetFeatureModels(): void {
  loadGeneration += 1;
  // The truncated graphs share the full model's layer variables. Disposing all
  // three graphs independently disposes those variables three times and leaves
  // the next graph with layers that are already disposed. One owner is enough.
  const owner = fullModel ?? featureModel ?? detectFeatureModel;
  featureModel = null;
  detectFeatureModel = null;
  fullModel = null;
  loading = null;
  loadingDetect = null;
  loadingFull = null;
  owner?.dispose();
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

/** Both feature maps of one picture, packed, plus the pooled embedding. */
export interface DetectFeatures {
  /** DETECT_PACK_SIZE floats: the coarse map, then the fine one. */
  pack: Float32Array;
  pooled: Float32Array;
}

/**
 * One MobileNet pass that yields everything a detect project needs from a
 * picture: the coarse grid, the fine grid and the pooled embedding the
 * classifier cache gets for free.
 *
 * The two maps come back as one buffer rather than two, because every consumer
 * downstream treats them as one example: the cache stores one entry, the head
 * slices it by scale, and a detector saved before the fine map existed simply
 * reads the prefix.
 */
export async function embedDetectFeatures(
  source: Drawable,
  aug: Augment = NO_AUGMENT,
): Promise<DetectFeatures> {
  const model = await loadDetectFeatureModel();
  // The two maps are joined on the device and read back in one transfer. Two
  // reads would mean two round trips per frame in the Use view, and the pooled
  // embedding is the mean of the coarse map, so it is cheaper to average the
  // 49 rows here than to ask the device for a number it already handed over.
  const packed = tf.tidy(() => {
    const input = preprocess(source, aug);
    const [coarse, fine] = model.predict(input) as tf.Tensor4D[];
    return tf.concat([
      coarse.reshape([coarse.size]),
      fine.reshape([fine.size]),
    ]) as tf.Tensor1D;
  });
  try {
    const pack = (await packed.data()) as Float32Array;
    return { pack, pooled: poolCoarse(pack) };
  } finally {
    packed.dispose();
  }
}

/** Mean over the coarse map's cells, which is exactly what `embed` returns. */
function poolCoarse(pack: Float32Array): Float32Array {
  const cells = COARSE_SCALE.rows * COARSE_SCALE.cols;
  const out = new Float32Array(FEATURE_SIZE);
  for (let cell = 0; cell < cells; cell += 1) {
    const at = cell * FEATURE_SIZE;
    for (let i = 0; i < FEATURE_SIZE; i += 1) out[i] += pack[at + i];
  }
  for (let i = 0; i < FEATURE_SIZE; i += 1) out[i] /= cells;
  return out;
}
