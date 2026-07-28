// Domain model for a .belo project. The manifest schema lives here and is the
// single source of truth for what gets written to project.json.

export const IMAGE_SIZE = 224;
/**
 * 1: single label projects, one labelId per image.
 * 2: adds project mode and, in multi mode, labelIds per image. Version 1
 *    manifests load unchanged and are written back as version 2 on the next
 *    save, with mode "single", which is exactly their old behavior.
 * 3: adds the detect mode and, in that mode only, per image boxes. Version 1
 *    and 2 manifests load exactly as before: no mode reads as "single", and
 *    boxes are absent everywhere outside a detect project.
 */
export const FORMAT_VERSION = 3;
export const SWATCH_COUNT = 8;

/** Score above which a label counts as present in multi mode. */
export const MULTI_THRESHOLD = 0.5;

/** Smallest edge a box may keep, as a share of the image. Below this it is a stray tap. */
export const MIN_BOX = 0.02;

/**
 * Pictures with at least one box needed before a detect project can train. A
 * detection head has to learn where as well as what, and under ten examples it
 * only learns the few places the boxes happened to be.
 */
export const DETECT_MIN_IMAGES = 10;

export type ImageSource = "import" | "capture" | "feedback";

/**
 * single: softmax over the labels, exactly one winner per image.
 * multi:  one sigmoid per label, so an image can be "X and Y but not Z".
 * detect: labels live on drawn boxes rather than on whole images.
 */
export type ProjectMode = "single" | "multi" | "detect";

export interface Label {
  id: string;
  name: string;
  /** Index into the swatch ramp, 0 to SWATCH_COUNT - 1. */
  color: number;
  createdAt: string;
}

/**
 * One drawn rectangle in a detect project. Coordinates are normalized 0..1
 * against the stored image, and x/y is the top left corner, so a box survives
 * any display size the annotator or a later training pass renders it at.
 */
export interface BoxRecord {
  id: string;
  labelId: string;
  x: number;
  y: number;
  w: number;
  h: number;
}

/** The part of a box the annotator drags around. */
export type BoxRect = Pick<BoxRecord, "x" | "y" | "w" | "h">;

export interface ImageRecord {
  id: string;
  /** File name inside the bundle images/ directory. */
  file: string;
  /** Single mode label, and the first label in multi mode. Null means unlabeled. */
  labelId: string | null;
  /** Multi mode label set. Absent in single mode projects. */
  labelIds?: string[];
  /**
   * Detect mode boxes. Absent in single and multi projects. An empty list is a
   * meaningful value: the image is a background example with nothing to find.
   */
  boxes?: BoxRecord[];
  source: ImageSource;
  addedAt: string;
}

export function boxesOf(image: ImageRecord): BoxRecord[] {
  return image.boxes ?? [];
}

/** Every label drawn on an image, each one once, in drawing order. */
export function boxLabelsOf(image: ImageRecord): string[] {
  const seen: string[] = [];
  for (const box of boxesOf(image)) {
    if (!seen.includes(box.labelId)) seen.push(box.labelId);
  }
  return seen;
}

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

/** Rounding room at the far edge, so 0.6 + 0.4 still counts as inside. */
const EDGE_SLACK = 1e-9;

/**
 * Pulls a rectangle back inside the image and drops anything too small to be a
 * deliberate box. Used by the annotator on every commit and by the loader on
 * every manifest read, so nothing out of range can reach the rest of the app.
 */
export function normalizeRect(rect: BoxRect): BoxRect | null {
  const numbers = [rect?.x, rect?.y, rect?.w, rect?.h];
  if (numbers.some((value) => typeof value !== "number" || !Number.isFinite(value))) return null;
  // A box that is already well formed is handed back with its own numbers, so a
  // save and a reopen give back exactly what was drawn instead of the same
  // rectangle plus a little floating point arithmetic.
  const inside =
    rect.x >= 0 &&
    rect.y >= 0 &&
    rect.w > 0 &&
    rect.h > 0 &&
    rect.x + rect.w <= 1 + EDGE_SLACK &&
    rect.y + rect.h <= 1 + EDGE_SLACK;
  if (inside) {
    if (rect.w < MIN_BOX || rect.h < MIN_BOX) return null;
    return { x: rect.x, y: rect.y, w: rect.w, h: rect.h };
  }
  // A negative width is a rectangle dragged leftwards, not an error.
  const x0 = clamp01(Math.min(rect.x, rect.x + rect.w));
  const y0 = clamp01(Math.min(rect.y, rect.y + rect.h));
  const x1 = clamp01(Math.max(rect.x, rect.x + rect.w));
  const y1 = clamp01(Math.max(rect.y, rect.y + rect.h));
  const w = x1 - x0;
  const h = y1 - y0;
  if (w < MIN_BOX || h < MIN_BOX) return null;
  return { x: x0, y: y0, w, h };
}

/** Every label on an image, whichever mode the project uses. */
export function labelsOf(image: ImageRecord): string[] {
  if (image.labelIds) return image.labelIds;
  return image.labelId ? [image.labelId] : [];
}

export function hasLabel(image: ImageRecord, labelId: string): boolean {
  return labelsOf(image).includes(labelId);
}

/** Rebuilds an image record around a label set, keeping both fields in step. */
export function withLabels(image: ImageRecord, labelIds: string[], mode: ProjectMode): ImageRecord {
  if (mode === "multi") return { ...image, labelId: labelIds[0] ?? null, labelIds };
  const { labelIds: _drop, ...rest } = image;
  return { ...rest, labelId: labelIds[0] ?? null };
}

/**
 * Keeps the boxes field where it belongs: present and sanitized in a detect
 * project, absent everywhere else, so a single or multi manifest is written
 * back byte for byte the way it was before detect mode existed.
 */
export function withBoxes(image: ImageRecord, mode: ProjectMode): ImageRecord {
  if (mode !== "detect") {
    if (!("boxes" in image)) return image;
    const { boxes: _drop, ...rest } = image;
    return rest;
  }
  const boxes: BoxRecord[] = [];
  for (const box of boxesOf(image)) {
    const rect = normalizeRect(box);
    if (!rect || typeof box.labelId !== "string" || !box.labelId) continue;
    boxes.push({ id: box.id, labelId: box.labelId, ...rect });
  }
  return { ...image, boxes };
}

export interface TrainSettings {
  epochs: number;
  batchSize: number;
  learningRate: number;
  hiddenUnits: number;
  validationSplit: number;
}

export interface ModelMeta {
  trainedAt: string;
  mode?: ProjectMode;
  labelIds: string[];
  epochs: number;
  accuracy: number;
  valAccuracy: number | null;
  imageCount: number;
  settings: TrainSettings;
  /** Detect projects only: how the last run scored at IoU 0.5. */
  precision?: number;
  recall?: number;
  boxCount?: number;
}

export interface ProjectManifest {
  format: "belo-project";
  formatVersion: number;
  appVersion: string;
  name: string;
  createdAt: string;
  updatedAt: string;
  /** Fixed when the project is created. Version 1 manifests are "single". */
  mode: ProjectMode;
  labels: Label[];
  images: ImageRecord[];
  model: ModelMeta | null;
  /** Metadata of the head kept at model/head.prev.json, null when there is none. */
  checkpoint?: ModelMeta | null;
  settings: TrainSettings;
}

export interface ProjectSummary {
  path: string;
  name: string;
  labelCount: number;
  imageCount: number;
  trained: boolean;
  updatedAt: string;
  missing: boolean;
}

/** Serialized dense head, stored at model/head.json inside the bundle. */
export interface StoredHead {
  formatVersion: number;
  /** Absent on heads written before multi mode existed, which are all softmax. */
  mode?: ProjectMode;
  labelIds: string[];
  hiddenUnits: number;
  featureSize: number;
  weights: { shape: number[]; values: number[] }[];
}

/** Serialized detection head, stored at model/detect.json inside the bundle. */
export interface StoredDetector {
  formatVersion: number;
  labelIds: string[];
  featureSize: number;
  /** Feature grid the head was built for, 7x7 for a 224 pixel input. */
  rows: number;
  cols: number;
  /** Anchor set, so a head keeps working if the defaults ever move. */
  scales: number[];
  ratios: number[];
  headWidth: number;
  weights: { shape: number[]; values: number[] }[];
}

export const DEFAULT_SETTINGS: TrainSettings = {
  epochs: 40,
  batchSize: 16,
  learningRate: 0.001,
  hiddenUnits: 100,
  validationSplit: 0.15,
};

/**
 * Detect projects start from different numbers. A detection head sees a few
 * hundred anchors per picture instead of one embedding, and almost all of them
 * are background, so it needs many more passes over a smaller batch before the
 * scores mean anything. These are the settings the learning proof in
 * scripts/check-learning.mjs is measured with.
 */
export const DETECT_SETTINGS: TrainSettings = {
  epochs: 150,
  batchSize: 8,
  learningRate: 0.002,
  hiddenUnits: 96,
  validationSplit: 0.2,
};

export function defaultSettings(mode: ProjectMode): TrainSettings {
  return mode === "detect" ? { ...DETECT_SETTINGS } : { ...DEFAULT_SETTINGS };
}

export function newManifest(
  name: string,
  appVersion: string,
  mode: ProjectMode = "single",
): ProjectManifest {
  const now = new Date().toISOString();
  return {
    format: "belo-project",
    formatVersion: FORMAT_VERSION,
    appVersion,
    name,
    createdAt: now,
    updatedAt: now,
    mode,
    labels: [],
    images: [],
    model: null,
    checkpoint: null,
    settings: defaultSettings(mode),
  };
}

/**
 * Brings a manifest read from disk up to the current schema. The only defaults
 * that matter are mode "single" and, in multi mode, a label set derived from the
 * old single labelId, so a version 1 project behaves exactly as it did before.
 * Detect projects additionally get their boxes clamped back into the image.
 */
export function migrateManifest(raw: ProjectManifest): ProjectManifest {
  const mode: ProjectMode =
    raw.mode === "multi" ? "multi" : raw.mode === "detect" ? "detect" : "single";
  return {
    ...raw,
    formatVersion: FORMAT_VERSION,
    mode,
    checkpoint: raw.checkpoint ?? null,
    settings: { ...defaultSettings(mode), ...raw.settings },
    labels: raw.labels ?? [],
    images: (raw.images ?? []).map((image) =>
      withBoxes(withLabels(image, labelsOf(image), mode), mode),
    ),
  };
}

let counter = 0;
export function makeId(prefix: string): string {
  counter += 1;
  const rand = Math.random().toString(36).slice(2, 8);
  return `${prefix}_${Date.now().toString(36)}${counter.toString(36)}${rand}`;
}
