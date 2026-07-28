// Detection geometry: anchors, overlap, offset coding, anchor matching, hard
// negative mining, non maximum suppression and scoring.
//
// Everything here is pure arithmetic on normalized 0..1 rectangles, with no
// TensorFlow and no DOM, so the whole decision layer of the detector can be
// checked in node:
//
//   node scripts/check-detect.mjs
//
// The tensor side lives in ml/detector.ts and calls into this module for every
// choice it makes, which keeps the parts that can be wrong in a subtle way in
// the part that is covered by tests.

import type { BoxRect } from "./types";

/** The grid the frozen MobileNet hands back for a 224 pixel input. */
export const GRID_ROWS = 7;
export const GRID_COLS = 7;

/**
 * Two scales and three aspect ratios, so six anchors per cell and 294 in total
 * on the 7x7 grid.
 *
 * The scales bracket the object sizes a 224 pixel picture can usefully hold: a
 * fifth of the frame and half of it. Anything much smaller than a fifth is
 * below the 32 pixel stride of the feature grid and cannot be localized well
 * whatever the anchor set says. The three ratios give a square, a wide and a
 * tall box at each scale, so an upright bottle and a lying pencil both have an
 * anchor they can match on overlap alone rather than only through the forced
 * best match. The box regression covers the rest of the shape space.
 */
export const ANCHOR_SCALES = [0.22, 0.5];
export const ANCHOR_RATIOS = [1, 2, 0.5];
export const ANCHORS_PER_CELL = ANCHOR_SCALES.length * ANCHOR_RATIOS.length;

/** Anchor and predicted boxes are kept in center form while they are decoded. */
export interface Anchor {
  cx: number;
  cy: number;
  w: number;
  h: number;
}

/** One decoded detection in normalized image coordinates, x/y at the top left. */
export interface Detection {
  labelId: string;
  score: number;
  x: number;
  y: number;
  w: number;
  h: number;
}

/** A ground truth rectangle carrying the label it was drawn with. */
export interface LabeledRect extends BoxRect {
  labelId: string;
}

export function anchorShapes(
  scales: number[] = ANCHOR_SCALES,
  ratios: number[] = ANCHOR_RATIOS,
): { w: number; h: number }[] {
  const shapes: { w: number; h: number }[] = [];
  for (const scale of scales) {
    for (const ratio of ratios) {
      const root = Math.sqrt(ratio);
      shapes.push({ w: scale * root, h: scale / root });
    }
  }
  return shapes;
}

/**
 * Every anchor on the grid, cells in row major order and the anchors of one
 * cell next to each other. That order is not a detail: the head reshapes a
 * [rows, cols, anchors * 4] convolution output straight into [anchors, 4], so
 * anchor index (row * cols + col) * perCell + shape has to mean the same thing
 * on both sides.
 */
export function buildAnchors(
  rows: number = GRID_ROWS,
  cols: number = GRID_COLS,
  shapes: { w: number; h: number }[] = anchorShapes(),
): Anchor[] {
  const anchors: Anchor[] = [];
  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      for (const shape of shapes) {
        anchors.push({
          cx: (col + 0.5) / cols,
          cy: (row + 0.5) / rows,
          w: shape.w,
          h: shape.h,
        });
      }
    }
  }
  return anchors;
}

export function anchorRect(anchor: Anchor): BoxRect {
  return {
    x: anchor.cx - anchor.w / 2,
    y: anchor.cy - anchor.h / 2,
    w: anchor.w,
    h: anchor.h,
  };
}

/** Intersection over union of two top left rectangles. */
export function iou(a: BoxRect, b: BoxRect): number {
  const x0 = Math.max(a.x, b.x);
  const y0 = Math.max(a.y, b.y);
  const x1 = Math.min(a.x + a.w, b.x + b.w);
  const y1 = Math.min(a.y + a.h, b.y + b.h);
  const overlap = Math.max(0, x1 - x0) * Math.max(0, y1 - y0);
  if (overlap <= 0) return 0;
  const union = a.w * a.h + b.w * b.h - overlap;
  return union <= 0 ? 0 : overlap / union;
}

/**
 * Offsets are the standard SSD parameterization: the center shift measured in
 * anchor widths, the size ratio in log space. The variances scale the four
 * numbers into a range Adam is comfortable with, which is the only reason they
 * exist.
 */
export const VAR_XY = 0.1;
export const VAR_WH = 0.2;

export type Offsets = [number, number, number, number];

export function encodeBox(box: BoxRect, anchor: Anchor): Offsets {
  const cx = box.x + box.w / 2;
  const cy = box.y + box.h / 2;
  return [
    (cx - anchor.cx) / anchor.w / VAR_XY,
    (cy - anchor.cy) / anchor.h / VAR_XY,
    Math.log(box.w / anchor.w) / VAR_WH,
    Math.log(box.h / anchor.h) / VAR_WH,
  ];
}

/** Reads four offsets starting at `at`, so a flat batch needs no slicing. */
export function decodeBox(offsets: ArrayLike<number>, anchor: Anchor, at = 0): BoxRect {
  const cx = anchor.cx + offsets[at] * VAR_XY * anchor.w;
  const cy = anchor.cy + offsets[at + 1] * VAR_XY * anchor.h;
  const w = anchor.w * Math.exp(offsets[at + 2] * VAR_WH);
  const h = anchor.h * Math.exp(offsets[at + 3] * VAR_WH);
  return { x: cx - w / 2, y: cy - h / 2, w, h };
}

/** Pulls a decoded box back inside the picture without letting it collapse. */
export function clipRect(rect: BoxRect): BoxRect {
  const x0 = Math.max(0, Math.min(1, rect.x));
  const y0 = Math.max(0, Math.min(1, rect.y));
  const x1 = Math.max(0, Math.min(1, rect.x + rect.w));
  const y1 = Math.max(0, Math.min(1, rect.y + rect.h));
  return { x: x0, y: y0, w: Math.max(0, x1 - x0), h: Math.max(0, y1 - y0) };
}

/**
 * Overlap at which an anchor answers for a box, and the one under which it is
 * background. The textbook pair is 0.5 and 0.4, which is written for a detector
 * with thousands of anchors across several feature maps. On one 7x7 grid with
 * six anchors per cell there are 294 of them, and half overlap leaves most
 * drawn boxes with a single positive anchor. Loosening the pair to 0.45 and
 * 0.35 gives every box a small cluster of anchors to learn from, which is worth
 * a lot: on the synthetic proof in scripts/check-learning.mjs it moves held out
 * recall from 67 percent to 81 percent at the same overlap quality.
 */
export const POS_IOU = 0.45;
export const NEG_IOU = 0.35;

/** 1 positive, 0 negative, -1 ignored because it sits between the two thresholds. */
export type MatchState = 1 | 0 | -1;

export interface Match {
  /** Index of the ground truth box each anchor answers for, -1 when none. */
  truth: Int32Array;
  state: Int8Array;
  positives: number;
}

/**
 * Anchor matching, IoU based, with the usual two safeguards:
 *
 * - An anchor over the positive threshold answers for its best box, an anchor
 *   under the negative threshold is background, and everything between is
 *   ignored so half overlapping anchors do not teach the head to say nothing.
 * - Every ground truth box then force matches a distinct anchor, whatever the
 *   overlap was. Without that, a box shaped unlike every anchor would never
 *   appear as a positive and could never be learned. The forced pairs are
 *   chosen greedily by overlap, so nearby boxes cannot overwrite each other's
 *   only positive anchor.
 */
export function matchAnchors(
  anchors: Anchor[],
  boxes: BoxRect[],
  posIoU: number = POS_IOU,
  negIoU: number = NEG_IOU,
): Match {
  const truth = new Int32Array(anchors.length).fill(-1);
  const state = new Int8Array(anchors.length);
  const rects = anchors.map(anchorRect);
  const overlaps = boxes.map(() => new Array<number>(anchors.length).fill(0));

  for (let a = 0; a < anchors.length; a += 1) {
    let bestBox = -1;
    let bestOverlap = 0;
    for (let b = 0; b < boxes.length; b += 1) {
      const overlap = iou(rects[a], boxes[b]);
      overlaps[b][a] = overlap;
      if (overlap > bestOverlap) {
        bestOverlap = overlap;
        bestBox = b;
      }
    }
    if (bestBox >= 0 && bestOverlap >= posIoU) {
      truth[a] = bestBox;
      state[a] = 1;
    } else if (bestOverlap < negIoU) {
      state[a] = 0;
    } else {
      state[a] = -1;
    }
  }

  const unmatchedBoxes = new Set(boxes.map((_, index) => index));
  const availableAnchors = new Set(anchors.map((_, index) => index));
  while (unmatchedBoxes.size > 0 && availableAnchors.size > 0) {
    let bestBox = -1;
    let bestAnchor = -1;
    let bestOverlap = -1;
    for (const b of unmatchedBoxes) {
      for (const a of availableAnchors) {
        const overlap = overlaps[b][a];
        if (overlap > bestOverlap) {
          bestBox = b;
          bestAnchor = a;
          bestOverlap = overlap;
        }
      }
    }
    truth[bestAnchor] = bestBox;
    state[bestAnchor] = 1;
    unmatchedBoxes.delete(bestBox);
    availableAnchors.delete(bestAnchor);
  }

  let positives = 0;
  for (let a = 0; a < state.length; a += 1) if (state[a] === 1) positives += 1;
  return { truth, state, positives };
}

/** Class targets and box targets for one picture, ready to be batched. */
export interface AnchorTargets {
  /** Label index plus one per anchor, 0 meaning background. */
  classes: Int32Array;
  /** Four encoded offsets per anchor, meaningful only where state is 1. */
  offsets: Float32Array;
  state: Int8Array;
  positives: number;
}

export function buildTargets(
  anchors: Anchor[],
  boxes: LabeledRect[],
  labelIds: string[],
): AnchorTargets {
  const usable = boxes.filter((box) => labelIds.includes(box.labelId));
  const match = matchAnchors(anchors, usable);
  const classes = new Int32Array(anchors.length);
  const offsets = new Float32Array(anchors.length * 4);
  for (let a = 0; a < anchors.length; a += 1) {
    if (match.state[a] !== 1) continue;
    const box = usable[match.truth[a]];
    classes[a] = labelIds.indexOf(box.labelId) + 1;
    offsets.set(encodeBox(box, anchors[a]), a * 4);
  }
  return { classes, offsets, state: match.state, positives: match.positives };
}

/** Negatives kept per positive when the classification loss is assembled. */
export const NEGATIVE_RATIO = 3;
/** Floor so a picture with nothing in it still teaches the head to stay quiet. */
export const MIN_NEGATIVES = 8;

/**
 * Hard negative mining. The head sees a few hundred anchors per picture and
 * almost all of them are background, so training on every one of them teaches
 * it to answer "nothing" and stop. Instead only the negatives it is currently
 * most wrong about are kept, three per positive, ranked by their own loss.
 *
 * Returns the weight of every anchor in the classification loss: 1 for the
 * positives and the mined negatives, 0 for everything else.
 */
export function mineNegatives(
  loss: ArrayLike<number>,
  state: Int8Array,
  ratio: number = NEGATIVE_RATIO,
  floor: number = MIN_NEGATIVES,
): Float32Array {
  const mask = new Float32Array(state.length);
  const negatives: number[] = [];
  let positives = 0;
  for (let a = 0; a < state.length; a += 1) {
    if (state[a] === 1) {
      mask[a] = 1;
      positives += 1;
    } else if (state[a] === 0) {
      negatives.push(a);
    }
  }
  const keep = Math.min(negatives.length, Math.max(floor, positives * ratio));
  negatives.sort((a, b) => loss[b] - loss[a]);
  for (let i = 0; i < keep; i += 1) mask[negatives[i]] = 1;
  return mask;
}

/**
 * Suppression overlap. 0.45 is the usual number and it is too generous here:
 * the looser anchor matching above makes several nearby anchors fire on one
 * object, and at 0.45 the second one survives as a false alarm. Measured on the
 * proof set, tightening this to 0.3 lifts precision from 71 to 92 percent
 * without costing a single true detection, and two real objects overlapping by
 * more than a third of their area are rare enough to be worth that trade.
 */
export const NMS_IOU = 0.3;
export const SCORE_THRESHOLD = 0.3;
export const MAX_DETECTIONS = 20;

/**
 * Greedy non maximum suppression over one label's detections: keep the highest
 * score, drop everything that overlaps it too much, repeat.
 */
export function nms(
  detections: Detection[],
  iouThreshold: number = NMS_IOU,
  max: number = MAX_DETECTIONS,
): Detection[] {
  const sorted = [...detections].sort((a, b) => b.score - a.score);
  const kept: Detection[] = [];
  for (const candidate of sorted) {
    if (kept.length >= max) break;
    if (kept.some((other) => iou(other, candidate) > iouThreshold)) continue;
    kept.push(candidate);
  }
  return kept;
}

export interface DecodeOptions {
  scoreThreshold?: number;
  iouThreshold?: number;
  maxDetections?: number;
}

/**
 * Turns one picture's raw head output into detections. Scores are softmax
 * probabilities with background at index 0, so a row is read by taking the best
 * label class and comparing it to the threshold. Suppression runs per label,
 * which lets a car sit inside a picture of a street without one erasing the
 * other.
 */
export function decodeDetections(
  scores: ArrayLike<number>,
  offsets: ArrayLike<number>,
  anchors: Anchor[],
  labelIds: string[],
  options: DecodeOptions = {},
): Detection[] {
  const scoreThreshold = options.scoreThreshold ?? SCORE_THRESHOLD;
  const iouThreshold = options.iouThreshold ?? NMS_IOU;
  const maxDetections = options.maxDetections ?? MAX_DETECTIONS;
  const classes = labelIds.length + 1;
  const perLabel = new Map<string, Detection[]>();

  for (let a = 0; a < anchors.length; a += 1) {
    let best = 1;
    for (let c = 2; c < classes; c += 1) {
      if (scores[a * classes + c] > scores[a * classes + best]) best = c;
    }
    const score = classes > 1 ? scores[a * classes + best] : 0;
    if (!(score >= scoreThreshold)) continue;
    const labelId = labelIds[best - 1];
    const rect = clipRect(decodeBox(offsets, anchors[a], a * 4));
    if (rect.w <= 0 || rect.h <= 0) continue;
    const list = perLabel.get(labelId) ?? [];
    list.push({ labelId, score, ...rect });
    perLabel.set(labelId, list);
  }

  const kept: Detection[] = [];
  for (const list of perLabel.values()) {
    kept.push(...nms(list, iouThreshold, maxDetections));
  }
  return kept.sort((a, b) => b.score - a.score).slice(0, maxDetections);
}

/**
 * Standardizes each grid cell's feature vector on its own mean and spread.
 *
 * The cut activation is a wall of positive numbers whose overall level moves
 * with the picture, mostly with how bright and busy it is, and a head reading
 * them raw spends its capacity on that rather than on what is in the cell.
 * Taking the level out is worth about eight points of held out recall on the
 * proof set. It has no parameters, so it is part of the contract rather than
 * part of the model, and inference has to apply exactly the same transform,
 * which is why every entry point in ml/detector.ts runs it.
 */
export function standardizeCells(
  features: Float32Array,
  cells: number,
  depth: number,
): Float32Array {
  const out = new Float32Array(features.length);
  for (let cell = 0; cell < cells; cell += 1) {
    const at = cell * depth;
    let mean = 0;
    for (let i = 0; i < depth; i += 1) mean += features[at + i];
    mean /= depth;
    let variance = 0;
    for (let i = 0; i < depth; i += 1) {
      const centered = features[at + i] - mean;
      variance += centered * centered;
    }
    const spread = Math.sqrt(variance / depth) || 1;
    for (let i = 0; i < depth; i += 1) out[at + i] = (features[at + i] - mean) / spread;
  }
  return out;
}

export interface DetectionMatch {
  /** Index of the truth box each detection hit, -1 when it hit nothing. */
  hits: number[];
  /** Indices of the truth boxes nothing found. */
  missed: number[];
}

/**
 * Matches detections against drawn boxes the way a precision and recall figure
 * needs it: highest scoring detection first, one truth box per detection, same
 * label only, IoU at or over the threshold.
 */
export function matchDetections(
  truth: LabeledRect[],
  detections: Detection[],
  iouThreshold = 0.5,
): DetectionMatch {
  const order = detections
    .map((detection, index) => ({ detection, index }))
    .sort((a, b) => b.detection.score - a.detection.score);
  const taken = new Set<number>();
  const hits = detections.map(() => -1);
  for (const { detection, index } of order) {
    let bestTruth = -1;
    let bestOverlap = iouThreshold;
    truth.forEach((box, t) => {
      if (taken.has(t) || box.labelId !== detection.labelId) return;
      const overlap = iou(box, detection);
      if (overlap >= bestOverlap) {
        bestOverlap = overlap;
        bestTruth = t;
      }
    });
    if (bestTruth >= 0) {
      taken.add(bestTruth);
      hits[index] = bestTruth;
    }
  }
  const missed = truth.map((_, t) => t).filter((t) => !taken.has(t));
  return { hits, missed };
}
