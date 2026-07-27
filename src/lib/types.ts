// Domain model for a .belo project. The manifest schema lives here and is the
// single source of truth for what gets written to project.json.

export const IMAGE_SIZE = 224;
export const FORMAT_VERSION = 1;
export const SWATCH_COUNT = 8;

export type ImageSource = "import" | "capture" | "feedback";

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
  labelId: string | null;
  source: ImageSource;
  addedAt: string;
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
  labels: Label[];
  images: ImageRecord[];
  model: ModelMeta | null;
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

export function newManifest(name: string, appVersion: string): ProjectManifest {
  const now = new Date().toISOString();
  return {
    format: "belo-project",
    formatVersion: FORMAT_VERSION,
    appVersion,
    name,
    createdAt: now,
    updatedAt: now,
    labels: [],
    images: [],
    model: null,
    settings: { ...DEFAULT_SETTINGS },
  };
}

let counter = 0;
export function makeId(prefix: string): string {
  counter += 1;
  const rand = Math.random().toString(36).slice(2, 8);
  return `${prefix}_${Date.now().toString(36)}${counter.toString(36)}${rand}`;
}
