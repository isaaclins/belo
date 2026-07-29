// Packing a trained model into one zip somebody else can run.
//
// The rule that shapes this file: the exported model **is** Belo's own
// serialization, byte for byte, not a conversion of it. `model/head.json` and
// `model/detect.json` are copied into the archive unchanged and then documented
// precisely in the README that travels with them, so a developer can reimplement
// inference from the document alone. A converted format would have to be kept in
// step with the app forever and would be a second thing that can be wrong.
//
// Everything the archive needs is reachable from the frontend: the trained
// weights come back through the storage seam, the frozen extractor is fetched
// out of the app's own bundled assets over the same relative URL the extractor
// loads it from at runtime, and the README, the page and the example are
// generated text. So the archive is built here and the Rust side only writes the
// finished bytes where the save dialog pointed.

import { exportReadme, EXAMPLE_HTML, EXAMPLE_JS, TFJS_CDN } from "./exportFiles";
import { makeZip, type ZipEntry } from "./zip";
import {
  MULTI_THRESHOLD,
  IMAGE_SIZE,
  type Label,
  type ModelMeta,
  type ProjectMode,
} from "./types";
import { MAX_DETECTIONS, NMS_IOU, SCORE_THRESHOLD, VAR_WH, VAR_XY } from "./detect";

/** Where the bundled extractor sits, the same URL ml/features.ts loads. */
export const EXTRACTOR_URL = "/models/mobilenet-v1-1.0-224/model.json";
const EXTRACTOR_DIR = "extractor";

/** The cuts each kind of model reads, coarse first. */
export const CLASSIFIER_CUTS = ["conv_pw_13_relu"];
export const DETECTOR_CUTS = ["conv_pw_13_relu", "conv_pw_11_relu"];

export interface ExportInput {
  projectName: string;
  appVersion: string;
  mode: ProjectMode;
  labels: Label[];
  /** The bytes of model/head.json or model/detect.json, exactly as stored. */
  weights: string;
  meta: ModelMeta | null;
  /** Injected so the node side of the export check can run without a webview. */
  fetchAsset?: (url: string) => Promise<Response>;
  now?: () => string;
}

export interface ExportBundle {
  bytes: Uint8Array;
  /** Every path inside the archive, for the checks and for the toast. */
  names: string[];
}

/** File name a model of this mode is stored and exported under. */
export function exportModelFile(mode: ProjectMode): string {
  return mode === "detect" ? "detector.json" : "classifier.json";
}

/** Default file name offered by the save dialog. */
export function exportFileName(projectName: string): string {
  const clean = projectName.trim().replace(/[^\w .-]+/g, "-").replace(/\s+/g, "-");
  return `${clean || "belo"}-model.zip`;
}

interface WeightsManifestEntry {
  paths: string[];
}

/**
 * The extractor as it sits in the app: model.json plus every weight shard its
 * own manifest names. Nothing is rewritten, so the copy in the archive loads
 * with a plain `tf.loadLayersModel('extractor/model.json')`.
 */
async function extractorEntries(
  fetchAsset: (url: string) => Promise<Response>,
): Promise<{ entries: ZipEntry[]; bytes: number }> {
  const response = await fetchAsset(EXTRACTOR_URL);
  if (!response.ok) throw new Error(`The bundled MobileNet is missing (${response.status})`);
  const text = await response.text();
  const parsed = JSON.parse(text) as { weightsManifest?: WeightsManifestEntry[] };
  const shards = (parsed.weightsManifest ?? []).flatMap((group) => group.paths ?? []);
  if (shards.length === 0) throw new Error("The bundled MobileNet lists no weight files");

  const base = EXTRACTOR_URL.slice(0, EXTRACTOR_URL.lastIndexOf("/") + 1);
  const entries: ZipEntry[] = [{ name: `${EXTRACTOR_DIR}/model.json`, data: text }];
  let bytes = new TextEncoder().encode(text).length;
  for (const shard of shards) {
    const shardResponse = await fetchAsset(`${base}${shard}`);
    if (!shardResponse.ok) throw new Error(`MobileNet weight file ${shard} is missing`);
    const data = new Uint8Array(await shardResponse.arrayBuffer());
    bytes += data.length;
    entries.push({ name: `${EXTRACTOR_DIR}/${shard}`, data });
  }
  return { entries, bytes };
}

/**
 * The descriptor a reader opens first. It carries nothing the model files
 * already carry: it says which mode this is, what the labels are called, what
 * the input has to look like and which thresholds the app itself uses, so the
 * example needs no constants of its own.
 */
function descriptor(input: ExportInput, modelFile: string, extractorFiles: number) {
  const detect = input.mode === "detect";
  return {
    format: "belo-model-export",
    formatVersion: 1,
    appVersion: input.appVersion,
    exportedAt: (input.now ?? (() => new Date().toISOString()))(),
    project: input.projectName,
    mode: input.mode,
    labels: input.labels.map((label) => ({ id: label.id, name: label.name })),
    trained: input.meta
      ? {
          trainedAt: input.meta.trainedAt,
          epochs: input.meta.epochs,
          accuracy: input.meta.accuracy,
          valAccuracy: input.meta.valAccuracy,
          imageCount: input.meta.imageCount,
          precision: input.meta.precision ?? null,
          recall: input.meta.recall ?? null,
          boxCount: input.meta.boxCount ?? null,
        }
      : null,
    input: {
      size: IMAGE_SIZE,
      crop: "center square, then resize",
      channels: "rgb",
      scale: "value / 127.5 - 1",
    },
    extractor: {
      path: `${EXTRACTOR_DIR}/model.json`,
      architecture: "mobilenet-v1-1.0-224",
      format: "tfjs-layers-model",
      files: extractorFiles,
      cuts: detect ? DETECTOR_CUTS : CLASSIFIER_CUTS,
      pooling: detect ? null : "mean over the 7x7 grid",
    },
    model: { path: modelFile },
    inference: detect
      ? {
          scoreThreshold: SCORE_THRESHOLD,
          nmsIoU: NMS_IOU,
          maxDetections: MAX_DETECTIONS,
          varXY: VAR_XY,
          varWH: VAR_WH,
          boxFormat: "top left corner, normalized 0..1",
          backgroundClass: 0,
        }
      : { multiThreshold: MULTI_THRESHOLD, activation: input.mode === "multi" ? "sigmoid" : "softmax" },
    example: { page: "example.html", script: "example.js", tfjs: TFJS_CDN },
  };
}

/** Builds the whole archive. Pure apart from the asset fetches it is handed. */
export async function buildExportBundle(input: ExportInput): Promise<ExportBundle> {
  const fetchAsset = input.fetchAsset ?? ((url: string) => fetch(url));
  const modelFile = exportModelFile(input.mode);
  const extractor = await extractorEntries(fetchAsset);

  const spec = descriptor(input, modelFile, extractor.entries.length);
  const readme = exportReadme({
    project: input.projectName,
    mode: input.mode,
    appVersion: input.appVersion,
    labels: input.labels.map((label) => ({ id: label.id, name: label.name })),
    modelFile,
    extractorFiles: extractor.entries.length,
    extractorBytes: extractor.bytes,
  });

  const entries: ZipEntry[] = [
    { name: "belo-export.json", data: `${JSON.stringify(spec, null, 2)}\n` },
    { name: modelFile, data: input.weights },
    { name: "README.md", data: readme },
    { name: "example.html", data: EXAMPLE_HTML },
    { name: "example.js", data: EXAMPLE_JS },
    ...extractor.entries,
  ];

  return { bytes: await makeZip(entries), names: entries.map((entry) => entry.name) };
}
