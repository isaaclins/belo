// Domain model for a .belo project. The manifest schema lives here and is the
// single source of truth for what gets written to project.json.

export const IMAGE_SIZE = 224;
/**
 * 1: single label projects, one labelId per image.
 * 2: adds project mode and, in multi mode, labelIds per image. Version 1
 *    manifests load unchanged and are written back as version 2 on the next
 *    save, with mode "single", which is exactly their old behavior.
 */
export const FORMAT_VERSION = 2;
export const SWATCH_COUNT = 8;

/** Score above which a label counts as present in multi mode. */
export const MULTI_THRESHOLD = 0.5;

export type ImageSource = "import" | "capture" | "feedback";

/**
 * single: softmax over the labels, exactly one winner per image.
 * multi:  one sigmoid per label, so an image can be "X and Y but not Z".
 */
export type ProjectMode = "single" | "multi";

export interface Label {
  id: string;
  name: string;
  /** Index into the swatch ramp, 0 to SWATCH_COUNT - 1. */
  color: number;
  createdAt: string;
}

export interface ImageRecord {
  id: string;
  /** File name inside the bundle images/ directory. */
  file: string;
  /** Single mode label, and the first label in multi mode. Null means unlabeled. */
  labelId: string | null;
  /** Multi mode label set. Absent in single mode projects. */
  labelIds?: string[];
  source: ImageSource;
  addedAt: string;
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
  if (mode === "single") {
    const { labelIds: _drop, ...rest } = image;
    return { ...rest, labelId: labelIds[0] ?? null };
  }
  return { ...image, labelId: labelIds[0] ?? null, labelIds };
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

export const DEFAULT_SETTINGS: TrainSettings = {
  epochs: 40,
  batchSize: 16,
  learningRate: 0.001,
  hiddenUnits: 100,
  validationSplit: 0.15,
};

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
    settings: { ...DEFAULT_SETTINGS },
  };
}

/**
 * Brings a manifest read from disk up to the current schema. The only defaults
 * that matter are mode "single" and, in multi mode, a label set derived from the
 * old single labelId, so a version 1 project behaves exactly as it did before.
 */
export function migrateManifest(raw: ProjectManifest): ProjectManifest {
  const mode: ProjectMode = raw.mode === "multi" ? "multi" : "single";
  return {
    ...raw,
    formatVersion: FORMAT_VERSION,
    mode,
    checkpoint: raw.checkpoint ?? null,
    settings: { ...DEFAULT_SETTINGS, ...raw.settings },
    labels: raw.labels ?? [],
    images: (raw.images ?? []).map((image) => withLabels(image, labelsOf(image), mode)),
  };
}

let counter = 0;
export function makeId(prefix: string): string {
  counter += 1;
  const rand = Math.random().toString(36).slice(2, 8);
  return `${prefix}_${Date.now().toString(36)}${counter.toString(36)}${rand}`;
}
