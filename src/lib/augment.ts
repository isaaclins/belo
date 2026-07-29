// Training time augmentation for detect projects.
//
// A detect project is a few dozen pictures with a few hundred anchors each, and
// the head on top of the frozen features has enough capacity to remember all of
// them. Showing it the same pictures again cannot fix that; showing it the same
// objects under conditions it has not seen can.
//
// Two transforms, both chosen because they keep the picture honest:
//
// - A horizontal flip, with the drawn boxes flipped with it. This is the only
//   free doubling of a detect dataset there is, and it teaches the head that
//   the left half of the grid is not where things live.
// - A mild brightness and saturation jitter. Household lighting moves a lot
//   more than these bounds, and the frozen MobileNet features shift with it, so
//   a head trained under one light reads badly under another.
//
// What is deliberately not here: rotation, shear and crops. All three move the
// object relative to the 7x7 grid in ways that need the boxes recomputed
// against a cropped frame, and a crop can cut an object in half and turn a
// correct box into a wrong one. Flips and colour cannot lie about a box.
//
// The transform happens on pixels, before the frozen extractor runs, because
// that is the only place it can happen: the cached features are the output of a
// network and there is no flip of a 1024 value activation. Which means an
// augmented variant costs one more MobileNet pass and one more cached feature
// map per picture per run. See VARIANTS below for what that is worth in memory.
//
// Everything here is pure arithmetic and is checked in node by
// scripts/check-detect.mjs. The pixel transform itself needs tensors and lives
// with the rest of the preprocessing in ml/features.ts.

import type { BoxRect } from "./types";

/**
 * Augmented copies made of every training picture. One, so a run holds twice
 * the feature maps it used to: 196 KB per picture becomes 392 KB, a hundred
 * picture project moves from about 20 MB to about 40 MB, and three hundred
 * pictures from 60 MB to 120 MB. That is the ceiling a phone has to fit, which
 * is why this is one and not four. The variants are built once per run, not
 * once per epoch, so the extra cost in time is one MobileNet pass per picture.
 */
export const VARIANTS = 1;

export interface Augment {
  /** Mirror left to right. Boxes must be flipped with it. */
  flip: boolean;
  /** Multiplies every channel. 1 leaves the picture alone. */
  brightness: number;
  /** Pulls colour towards or away from grey. 1 leaves the picture alone. */
  saturation: number;
}

export const NO_AUGMENT: Augment = { flip: false, brightness: 1, saturation: 1 };

/**
 * Jitter bounds. Wide enough to cover the difference between a kitchen at noon
 * and the same kitchen in the evening, narrow enough that a red object never
 * becomes a grey one and a label defined by its colour survives.
 */
export const BRIGHTNESS_RANGE: [number, number] = [0.85, 1.18];
export const SATURATION_RANGE: [number, number] = [0.7, 1.3];

/** Mirrors a normalized rectangle. x is the left edge, so it becomes 1 - right. */
export function flipRect(rect: BoxRect): BoxRect {
  return { x: 1 - rect.x - rect.w, y: rect.y, w: rect.w, h: rect.h };
}

/** Mirrors a list of boxes, keeping whatever else each one carries. */
export function flipBoxes<T extends BoxRect>(boxes: readonly T[]): T[] {
  return boxes.map((box) => ({ ...box, ...flipRect(box) }));
}

/** Deterministic 32 bit hash of a string, so a picture always gets its own jitter. */
function hash(text: string): number {
  let h = 2166136261 >>> 0;
  for (let i = 0; i < text.length; i += 1) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 16777619) >>> 0;
  }
  return h >>> 0;
}

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * The augmentation for one picture's nth variant. Seeded from the image id, so
 * two runs over the same project build exactly the same variants and a run can
 * be compared with the one before it. Variant 0 is always the original.
 */
export function augmentFor(imageId: string, variant: number): Augment {
  if (variant <= 0) return NO_AUGMENT;
  const rng = mulberry32(hash(`${imageId}#${variant}`));
  const pick = ([low, high]: [number, number]) => low + rng() * (high - low);
  return {
    // Every variant flips. With one variant per picture that is the point of
    // the exercise; with more, alternating would only give back a duplicate.
    flip: true,
    brightness: pick(BRIGHTNESS_RANGE),
    saturation: pick(SATURATION_RANGE),
  };
}

export function isIdentity(aug: Augment): boolean {
  return !aug.flip && aug.brightness === 1 && aug.saturation === 1;
}

/** Rec. 601 luma, the same weights a photo editor desaturates with. */
export const LUMA = [0.299, 0.587, 0.114];
