// Single application store. Owns the open project, disk writes, the feature
// cache and the trained head. Views read from here and never touch disk.

import { create } from "zustand";
import type * as tfTypes from "@tensorflow/tfjs";
import { storage, type ImportRef } from "./storage";
import {
  hasLabel,
  labelsOf,
  makeId,
  migrateManifest,
  MULTI_THRESHOLD,
  newManifest,
  SWATCH_COUNT,
  withLabels,
  type ImageRecord,
  type ImageSource,
  type Label,
  type ProjectManifest,
  type ProjectMode,
  type ProjectSummary,
  type StoredHead,
  type TrainSettings,
} from "./types";
import {
  base64ToJpegBase64,
  fileToJpegBase64,
  loadImageElement,
  normalizeToJpegBase64,
  type Drawable,
} from "./images";
import { initBackend } from "./ml/backend";
import { embedToArray, embedWithSpatial, loadFeatureModel } from "./ml/features";
import type { Attention, AttentionMap } from "./attention";
import {
  deserializeHead,
  predictBatch,
  predictCells,
  predictOne,
  serializeHead,
  trainHead,
  type TrainProgress,
} from "./ml/classifier";

export const APP_VERSION = "0.4.1";

export type View = "label" | "train" | "use";

export interface Toast {
  id: string;
  text: string;
  tone: "info" | "good" | "bad";
}

export interface BusyState {
  title: string;
  done: number;
  total: number;
}

export type TrainPhase = "idle" | "reading" | "training" | "scoring" | "done" | "error";

export interface ImageVerdict {
  imageId: string;
  /** Single mode truth and guess. In multi mode these hold the first label. */
  actualLabelId: string;
  predictedLabelId: string;
  /** Multi mode truth and guess: the full label sets. */
  actualLabelIds: string[];
  predictedLabelIds: string[];
  confidence: number;
  correct: boolean;
}

/**
 * Per label score after a run. In single mode this is recall: how many images
 * of that label were read correctly. In multi mode it is the per label yes/no
 * accuracy over every scored image.
 */
export interface LabelStat {
  labelId: string;
  total: number;
  hits: number;
  rate: number;
}

export interface TrainState {
  phase: TrainPhase;
  message: string;
  done: number;
  total: number;
  history: TrainProgress[];
  accuracy: number | null;
  valAccuracy: number | null;
  verdicts: ImageVerdict[];
  labelStats: LabelStat[];
}

export interface Prediction {
  labelId: string;
  confidence: number;
  scores: { labelId: string; score: number }[];
  /** Per label score maps over the feature grid, only when they were asked for. */
  attention: Attention | null;
}

interface State {
  booted: boolean;
  backend: string | null;
  view: View;
  path: string | null;
  manifest: ProjectManifest | null;
  recents: ProjectSummary[];
  projectsDir: string | null;
  saving: boolean;
  dirty: boolean;
  busy: BusyState | null;
  toasts: Toast[];
  activeLabelId: string | null;
  train: TrainState;
  hasModel: boolean;
  /** True when the dataset changed after the last training run. */
  stale: boolean;
  modelLabelIds: string[];
  /** True while a debounced background retrain is running. */
  autoRetraining: boolean;
  /** True when model/head.prev.json holds a head the user can go back to. */
  hasCheckpoint: boolean;
  /** Live camera attention overlay. Session state, never written to the manifest. */
  attentionOn: boolean;

  boot: () => Promise<void>;
  refreshRecents: () => Promise<void>;
  setView: (view: View) => void;
  toast: (text: string, tone?: Toast["tone"]) => void;
  dismissToast: (id: string) => void;

  createProject: (name: string, mode?: ProjectMode) => Promise<void>;
  openProject: (path: string) => Promise<void>;
  openProjectDialog: () => Promise<void>;
  closeProject: () => void;
  forgetProject: (path: string) => Promise<void>;
  /** Erases a .belo bundle from disk and drops it from the list. */
  deleteProject: (path: string) => Promise<void>;
  renameProject: (name: string) => void;
  saveNow: () => Promise<void>;

  addLabel: (name: string) => string | null;
  renameLabel: (id: string, name: string) => void;
  deleteLabel: (id: string) => Promise<void>;
  setActiveLabel: (id: string | null) => void;

  addFiles: (files: File[], labelId: string | null) => Promise<number>;
  addFromImports: (refs: ImportRef[], labelId: string | null) => Promise<number>;
  /** Opens the platform image or photo picker and imports its selection. */
  importImages: (labelId: string | null) => Promise<number>;
  /** Opens the desktop folder picker and imports everything it returns. */
  importFolder: (labelId: string | null) => Promise<number>;
  /** Accepts one label, a label set for multi mode projects, or null. */
  addFromDrawable: (
    source: Drawable,
    labels: string | string[] | null,
    origin: ImageSource,
  ) => Promise<void>;
  moveImages: (imageIds: string[], labelId: string | null) => void;
  /** Multi mode: adds the label where it is missing, removes it where it is not. */
  toggleImageLabels: (imageIds: string[], labelId: string) => void;
  removeImages: (imageIds: string[]) => Promise<void>;

  updateSettings: (patch: Partial<TrainSettings>) => void;
  startTraining: () => Promise<void>;
  stopTraining: () => void;
  resetTraining: () => void;
  /** Restores model/head.prev.json as the live model and drops the checkpoint. */
  revertModel: () => Promise<void>;

  setAttentionOn: (on: boolean) => void;
  /** Asking for attention adds one batched head pass over the same features. */
  predict: (source: Drawable, wantAttention?: boolean) => Promise<Prediction | null>;
}

// Non reactive side tables. Keeping tensors and typed arrays out of the store
// avoids pointless React work on every frame of a webcam session.
const featureCache = new Map<string, Float32Array>();
let head: tfTypes.Sequential | null = null;
let saveTimer: ReturnType<typeof setTimeout> | null = null;

/** How long the app waits after the last correction before retraining itself. */
const AUTO_RETRAIN_DELAY_MS = 4000;

type RunMode = "manual" | "auto";

interface TrainRun {
  mode: RunMode;
  stop: boolean;
  done: Promise<void>;
}

let currentRun: TrainRun | null = null;
let autoTimer: ReturnType<typeof setTimeout> | null = null;
/** Set when feedback lands while a run is already going, so it retries after. */
let autoQueued = false;

const idleTrainState: TrainState = {
  phase: "idle",
  message: "",
  done: 0,
  total: 0,
  history: [],
  accuracy: null,
  valAccuracy: null,
  verdicts: [],
  labelStats: [],
};

/** Turns raw head outputs into per image verdicts for the Train view. */
function scoreSamples(
  samples: ImageRecord[],
  scores: number[][],
  labelIds: string[],
  mode: ProjectMode,
): ImageVerdict[] {
  return samples.map((record, i) => {
    const row = scores[i] ?? [];
    const actual = labelsOf(record);

    if (mode === "multi") {
      const predicted = labelIds.filter((_, index) => (row[index] ?? 0) >= MULTI_THRESHOLD);
      const correct =
        predicted.length === actual.length && predicted.every((id) => actual.includes(id));
      // Average distance from the threshold decision, so a row of confident
      // yes/no calls reads high and a row of coin flips reads low.
      const margin =
        labelIds.length === 0
          ? 0
          : labelIds.reduce((sum, id, index) => {
              const score = row[index] ?? 0;
              return sum + (actual.includes(id) ? score : 1 - score);
            }, 0) / labelIds.length;
      return {
        imageId: record.id,
        actualLabelId: actual[0] ?? "",
        predictedLabelId: predicted[0] ?? "",
        actualLabelIds: actual,
        predictedLabelIds: predicted,
        confidence: margin,
        correct,
      };
    }

    let best = 0;
    row.forEach((value, index) => {
      if (value > row[best]) best = index;
    });
    const predictedLabelId = labelIds[best] ?? "";
    return {
      imageId: record.id,
      actualLabelId: record.labelId ?? actual[0] ?? "",
      predictedLabelId,
      actualLabelIds: actual,
      predictedLabelIds: predictedLabelId ? [predictedLabelId] : [],
      confidence: row[best] ?? 0,
      correct: predictedLabelId === record.labelId,
    };
  });
}

function summarizeLabels(
  verdicts: ImageVerdict[],
  labelIds: string[],
  mode: ProjectMode,
): LabelStat[] {
  return labelIds.map((labelId) => {
    if (mode === "multi") {
      // Per label accuracy: every image votes on every label.
      const hits = verdicts.filter(
        (v) => v.actualLabelIds.includes(labelId) === v.predictedLabelIds.includes(labelId),
      ).length;
      const total = verdicts.length;
      return { labelId, total, hits, rate: total === 0 ? 0 : hits / total };
    }
    const mine = verdicts.filter((v) => v.actualLabelId === labelId);
    const hits = mine.filter((v) => v.correct).length;
    return { labelId, total: mine.length, hits, rate: mine.length === 0 ? 0 : hits / mine.length };
  });
}

function errorText(err: unknown): string {
  if (err instanceof Error) return err.message;
  if (typeof err === "string") return err;
  return "Something went wrong";
}

export const useStore = create<State>((set, get) => {
  const touch = (patch: Partial<ProjectManifest>) => {
    const manifest = get().manifest;
    if (!manifest) return;
    const next: ProjectManifest = {
      ...manifest,
      ...patch,
      updatedAt: new Date().toISOString(),
    };
    set({ manifest: next, dirty: true });
    scheduleSave();
  };

  const scheduleSave = () => {
    if (saveTimer) clearTimeout(saveTimer);
    saveTimer = setTimeout(() => {
      void get().saveNow();
    }, 500);
  };

  // A changed label set makes the trained head structurally invalid, so it has
  // to go. Everything else only makes the model out of date, which is worth a
  // warning but not worth throwing away a working classifier.
  const dropModel = () => {
    cancelAutoRetrain();
    head?.dispose();
    head = null;
    set({ hasModel: false, modelLabelIds: [], stale: false });
  };

  const markStale = () => {
    if (get().hasModel) set({ stale: true });
  };

  /** Loads and caches the embedding for one stored image. */
  const featureFor = async (record: ImageRecord): Promise<Float32Array> => {
    const cached = featureCache.get(record.id);
    if (cached) return cached;
    const path = get().path;
    if (!path) throw new Error("No project is open");
    const img = await loadImageElement(storage.imageUrl(path, record.file));
    const feature = await embedToArray(img);
    featureCache.set(record.id, feature);
    return feature;
  };

  /** Labels that actually hold images, plus the samples that can be trained on. */
  const trainingSet = (manifest: ProjectManifest) => {
    const usable = manifest.labels.filter((label) =>
      manifest.images.some((img) => hasLabel(img, label.id)),
    );
    const ids = new Set(usable.map((l) => l.id));
    const samples = manifest.images.filter((img) =>
      labelsOf(img).some((labelId) => ids.has(labelId)),
    );
    return { labelIds: usable.map((l) => l.id), samples };
  };

  /**
   * One training pass. Manual runs drive the whole Train view; automatic runs
   * stay quiet, keep a checkpoint of the head they replace and never block the
   * live camera beyond the frame yields the fit loop already does.
   */
  const executeRun = async (run: TrainRun): Promise<void> => {
    const manifest = get().manifest;
    const path = get().path;
    if (!manifest || !path) return;
    const auto = run.mode === "auto";
    const { labelIds, samples } = trainingSet(manifest);

    if (labelIds.length < 2) {
      if (!auto) get().toast("Add at least two labels with images before training", "bad");
      return;
    }

    if (auto) set({ autoRetraining: true });
    else {
      set({
        train: {
          ...idleTrainState,
          phase: "reading",
          message: "Loading the feature extractor",
          total: samples.length,
        },
      });
    }

    try {
      const backend = await initBackend();
      set({ backend });
      await loadFeatureModel();

      if (!auto) set((s) => ({ train: { ...s.train, message: "Reading images" } }));
      const mode = manifest.mode;
      const features: Float32Array[] = [];
      const targets: number[][] = [];

      for (let i = 0; i < samples.length; i += 1) {
        if (run.stop) throw new Error("Training cancelled");
        const record = samples[i];
        features.push(await featureFor(record));
        // Single mode gets a one hot row, multi mode a 0/1 membership row.
        const own = labelsOf(record);
        targets.push(labelIds.map((id) => (own.includes(id) ? 1 : 0)));
        if (!auto) set((s) => ({ train: { ...s.train, done: i + 1 } }));
      }

      if (!auto) {
        set((s) => ({
          train: { ...s.train, phase: "training", message: "Training the classifier", done: 0 },
        }));
      }

      const result = await trainHead({
        features,
        targets,
        numClasses: labelIds.length,
        mode,
        settings: manifest.settings,
        shouldStop: () => run.stop,
        onProgress: (progress) => {
          if (auto) return;
          set((s) => ({
            train: {
              ...s.train,
              done: progress.epoch,
              total: progress.totalEpochs,
              history: [...s.train.history, progress],
              accuracy: progress.accuracy,
              valAccuracy: progress.valAccuracy,
            },
          }));
        },
      });

      if (run.stop && auto) {
        result.model.dispose();
        return;
      }

      // Keep what is about to be replaced, so one tap can undo a silent run.
      if (auto) {
        try {
          const previous = await storage.readModel(path);
          if (previous) {
            await storage.writeCheckpoint(path, previous);
            touch({ checkpoint: get().manifest?.model ?? null });
            set({ hasCheckpoint: true });
          }
        } catch (err) {
          get().toast(`Could not save a checkpoint: ${errorText(err)}`, "bad");
        }
      }

      head?.dispose();
      head = result.model;

      if (!auto) {
        set((s) => ({ train: { ...s.train, phase: "scoring", message: "Checking each image" } }));
      }
      const scores = predictBatch(result.model, features);
      const verdicts = scoreSamples(samples, scores, labelIds, mode);
      const labelStats = summarizeLabels(verdicts, labelIds, mode);

      const stored = serializeHead(result.model, labelIds, manifest.settings.hiddenUnits, mode);
      await storage.writeModel(path, JSON.stringify(stored));

      touch({
        model: {
          trainedAt: new Date().toISOString(),
          mode,
          labelIds,
          epochs: result.history.length,
          accuracy: result.accuracy,
          valAccuracy: result.valAccuracy,
          imageCount: samples.length,
          settings: manifest.settings,
        },
      });

      set((s) => ({
        hasModel: true,
        stale: false,
        modelLabelIds: labelIds,
        train: {
          ...s.train,
          phase: "done",
          message: auto ? "Refreshed in the background" : "Ready to use",
          accuracy: result.accuracy,
          valAccuracy: result.valAccuracy,
          verdicts,
          labelStats,
        },
      }));
      await get().saveNow();
    } catch (err) {
      if (auto) {
        // A cancelled or failed background run is not worth a red banner: the
        // old head is still live and the next correction tries again.
        if (!run.stop) get().toast(`Background retraining stopped: ${errorText(err)}`, "bad");
        return;
      }
      set((s) => ({ train: { ...s.train, phase: "error", message: errorText(err) } }));
      get().toast(errorText(err), "bad");
    } finally {
      if (auto) set({ autoRetraining: false });
    }
  };

  const beginRun = (mode: RunMode): Promise<void> => {
    const run: TrainRun = { mode, stop: false, done: Promise.resolve() };
    currentRun = run;
    run.done = executeRun(run).finally(() => {
      if (currentRun === run) currentRun = null;
      if (autoQueued) {
        autoQueued = false;
        scheduleAutoRetrain();
      }
    });
    return run.done;
  };

  const cancelAutoRetrain = () => {
    if (autoTimer) clearTimeout(autoTimer);
    autoTimer = null;
    autoQueued = false;
  };

  /**
   * Debounced background retrain. Called whenever the user confirms or corrects
   * a prediction, so a burst of taps produces exactly one run.
   */
  const scheduleAutoRetrain = () => {
    if (!get().hasModel) return;
    if (autoTimer) clearTimeout(autoTimer);
    autoTimer = setTimeout(() => {
      autoTimer = null;
      if (!get().hasModel || !get().path) return;
      if (currentRun) {
        // A manual run owns the model right now, try again once it is done.
        autoQueued = true;
        return;
      }
      void beginRun("auto");
    }, AUTO_RETRAIN_DELAY_MS);
  };

  const writeImageRecord = async (
    jpegBase64: string,
    labels: string | string[] | null,
    origin: ImageSource,
  ): Promise<ImageRecord> => {
    const path = get().path;
    if (!path) throw new Error("No project is open");
    const mode = get().manifest?.mode ?? "single";
    const id = makeId("img");
    const file = `${id}.jpg`;
    await storage.writeImage(path, file, jpegBase64);
    const ids = labels === null ? [] : Array.isArray(labels) ? labels : [labels];
    const record: ImageRecord = {
      id,
      file,
      labelId: null,
      source: origin,
      addedAt: new Date().toISOString(),
    };
    return withLabels(record, ids, mode);
  };

  return {
    booted: false,
    backend: null,
    view: "label",
    path: null,
    manifest: null,
    recents: [],
    projectsDir: null,
    saving: false,
    dirty: false,
    busy: null,
    toasts: [],
    activeLabelId: null,
    train: idleTrainState,
    hasModel: false,
    stale: false,
    modelLabelIds: [],
    autoRetraining: false,
    hasCheckpoint: false,
    attentionOn: false,

    async boot() {
      try {
        const dir = await storage.defaultProjectsDir();
        set({ projectsDir: dir });
      } catch (err) {
        get().toast(errorText(err), "bad");
      }
      await get().refreshRecents();
      set({ booted: true });
      // Backend init is slow but not blocking, so let it settle in the back.
      initBackend()
        .then((backend) => set({ backend }))
        .catch((err) => get().toast(`Could not start TensorFlow: ${errorText(err)}`, "bad"));
    },

    async refreshRecents() {
      try {
        const recents = await storage.recentProjects();
        const dir = get().projectsDir;
        const known = new Set(recents.map((r) => r.path));
        const scanned = dir ? await storage.scanProjects(dir) : [];
        const merged = [...recents.filter((r) => !r.missing)];
        for (const project of scanned) {
          if (!known.has(project.path)) merged.push(project);
        }
        set({ recents: merged });
      } catch (err) {
        get().toast(errorText(err), "bad");
      }
    },

    setView(view) {
      set({ view });
    },

    toast(text, tone = "info") {
      const id = makeId("toast");
      set({ toasts: [...get().toasts, { id, text, tone }] });
      setTimeout(() => get().dismissToast(id), 4200);
    },

    dismissToast(id) {
      set({ toasts: get().toasts.filter((t) => t.id !== id) });
    },

    async createProject(name, mode = "single") {
      try {
        const dir = get().projectsDir ?? (await storage.defaultProjectsDir());
        const manifest = newManifest(name.trim() || "Untitled", APP_VERSION, mode);
        const path = await storage.createProject(dir, manifest.name, JSON.stringify(manifest, null, 2));
        featureCache.clear();
        dropModel();
        set({
          path,
          manifest,
          view: "label",
          dirty: false,
          activeLabelId: null,
          train: idleTrainState,
          hasCheckpoint: false,
        });
        await storage.rememberProject(path, manifest.name);
        await get().refreshRecents();
      } catch (err) {
        get().toast(errorText(err), "bad");
      }
    },

    async openProject(path) {
      try {
        const text = await storage.readManifest(path);
        const manifest = migrateManifest(JSON.parse(text) as ProjectManifest);
        featureCache.clear();
        dropModel();
        set({
          path,
          manifest,
          view: "label",
          dirty: false,
          activeLabelId: manifest.labels[0]?.id ?? null,
          train: idleTrainState,
          hasCheckpoint: false,
        });
        await storage.rememberProject(path, manifest.name);
        await get().refreshRecents();

        // Restore a previously trained head so the Use view works immediately.
        const stored = await storage.readModel(path);
        if (stored) {
          await loadFeatureModel();
          const parsed = JSON.parse(stored) as StoredHead;
          const restored = deserializeHead(parsed, manifest.settings);
          if (restored) {
            head = restored;
            set({ hasModel: true, stale: false, modelLabelIds: parsed.labelIds });
          }
        }
        set({ hasCheckpoint: await storage.hasCheckpoint(path) });
      } catch (err) {
        get().toast(errorText(err), "bad");
      }
    },

    async openProjectDialog() {
      try {
        const picked = await storage.pickProject(get().projectsDir);
        if (picked) await get().openProject(picked);
      } catch (err) {
        get().toast(errorText(err), "bad");
      }
    },

    closeProject() {
      if (saveTimer) clearTimeout(saveTimer);
      featureCache.clear();
      dropModel();
      set({
        path: null,
        manifest: null,
        dirty: false,
        activeLabelId: null,
        train: idleTrainState,
        view: "label",
        hasCheckpoint: false,
      });
      void get().refreshRecents();
    },

    async forgetProject(path) {
      await storage.forgetProject(path);
      await get().refreshRecents();
    },

    async deleteProject(path) {
      if (get().path === path) {
        get().toast("Close the project before deleting it", "bad");
        return;
      }
      try {
        await storage.deleteProject(path);
        await get().refreshRecents();
        get().toast("Project deleted", "good");
      } catch (err) {
        get().toast(errorText(err), "bad");
      }
    },

    renameProject(name) {
      touch({ name: name.trim() || "Untitled" });
    },

    async saveNow() {
      const { path, manifest } = get();
      if (!path || !manifest) return;
      if (saveTimer) {
        clearTimeout(saveTimer);
        saveTimer = null;
      }
      set({ saving: true });
      try {
        await storage.writeManifest(path, JSON.stringify(manifest, null, 2));
        set({ dirty: false });
      } catch (err) {
        get().toast(errorText(err), "bad");
      } finally {
        set({ saving: false });
      }
    },

    addLabel(name) {
      const manifest = get().manifest;
      if (!manifest) return null;
      const trimmed = name.trim();
      if (!trimmed) return null;
      if (manifest.labels.some((l) => l.name.toLowerCase() === trimmed.toLowerCase())) {
        get().toast(`Label "${trimmed}" already exists`, "bad");
        return null;
      }
      const label: Label = {
        id: makeId("lbl"),
        name: trimmed,
        color: manifest.labels.length % SWATCH_COUNT,
        createdAt: new Date().toISOString(),
      };
      touch({ labels: [...manifest.labels, label] });
      set({ activeLabelId: label.id });
      dropModel();
      return label.id;
    },

    renameLabel(id, name) {
      const manifest = get().manifest;
      if (!manifest) return;
      const trimmed = name.trim();
      if (!trimmed) return;
      touch({
        labels: manifest.labels.map((l) => (l.id === id ? { ...l, name: trimmed } : l)),
      });
    },

    async deleteLabel(id) {
      const { manifest, path } = get();
      if (!manifest || !path) return;
      // In multi mode an image can survive losing one of its labels, so only
      // the ones left with nothing are deleted from the bundle.
      const doomed = manifest.images.filter(
        (img) => hasLabel(img, id) && labelsOf(img).length === 1,
      );
      try {
        if (doomed.length > 0) await storage.deleteImages(path, doomed.map((img) => img.file));
      } catch (err) {
        get().toast(errorText(err), "bad");
      }
      doomed.forEach((img) => featureCache.delete(img.id));
      const gone = new Set(doomed.map((img) => img.id));
      const labels = manifest.labels.filter((l) => l.id !== id);
      touch({
        labels,
        images: manifest.images
          .filter((img) => !gone.has(img.id))
          .map((img) =>
            hasLabel(img, id)
              ? withLabels(img, labelsOf(img).filter((labelId) => labelId !== id), manifest.mode)
              : img,
          ),
        model: null,
      });
      dropModel();
      if (get().activeLabelId === id) set({ activeLabelId: labels[0]?.id ?? null });
    },

    setActiveLabel(id) {
      set({ activeLabelId: id });
    },

    async addFiles(files, labelId) {
      const manifest = get().manifest;
      if (!manifest || files.length === 0) return 0;
      const added: ImageRecord[] = [];
      set({ busy: { title: "Importing images", done: 0, total: files.length } });
      for (let i = 0; i < files.length; i += 1) {
        try {
          const jpeg = await fileToJpegBase64(files[i]);
          added.push(await writeImageRecord(jpeg, labelId, "import"));
        } catch {
          // Skip anything the webview cannot decode and keep the run going.
        }
        set({ busy: { title: "Importing images", done: i + 1, total: files.length } });
      }
      set({ busy: null });
      if (added.length > 0) {
        touch({ images: [...get().manifest!.images, ...added] });
        markStale();
      }
      const skipped = files.length - added.length;
      if (skipped > 0) get().toast(`${skipped} file(s) could not be read`, "bad");
      return added.length;
    },

    async addFromImports(refs, labelId) {
      const manifest = get().manifest;
      if (!manifest || refs.length === 0) return 0;
      const added: ImageRecord[] = [];
      set({ busy: { title: "Importing images", done: 0, total: refs.length } });
      for (let i = 0; i < refs.length; i += 1) {
        try {
          const payload = await storage.readImport(refs[i]);
          const jpeg = await base64ToJpegBase64(payload.base64, payload.mime);
          added.push(await writeImageRecord(jpeg, labelId, "import"));
        } catch {
          // Same as addFiles: unreadable files are skipped, not fatal.
        }
        set({ busy: { title: "Importing images", done: i + 1, total: refs.length } });
      }
      set({ busy: null });
      if (added.length > 0) {
        touch({ images: [...get().manifest!.images, ...added] });
        markStale();
      }
      return added.length;
    },

    async importImages(labelId) {
      try {
        const refs = await storage.pickImportImages(true);
        if (refs === null) return 0;
        return await get().addFromImports(refs, labelId);
      } catch (err) {
        get().toast(errorText(err), "bad");
        return 0;
      }
    },

    async importFolder(labelId) {
      try {
        const refs = await storage.pickImportFolder();
        if (refs === null) return 0;
        if (refs.length === 0) {
          get().toast("That folder has no images", "bad");
          return 0;
        }
        return await get().addFromImports(refs, labelId);
      } catch (err) {
        get().toast(errorText(err), "bad");
        return 0;
      }
    },

    async addFromDrawable(source, labels, origin) {
      const manifest = get().manifest;
      if (!manifest) return;
      try {
        const jpeg = normalizeToJpegBase64(source);
        const record = await writeImageRecord(jpeg, labels, origin);
        touch({ images: [...get().manifest!.images, record] });
        markStale();
        // Corrections from the Use view fold themselves in without a trip to
        // the Train view, once the taps stop coming.
        if (origin === "feedback") scheduleAutoRetrain();
      } catch (err) {
        get().toast(errorText(err), "bad");
      }
    },

    moveImages(imageIds, labelId) {
      const manifest = get().manifest;
      if (!manifest) return;
      const moved = new Set(imageIds);
      touch({
        images: manifest.images.map((img) =>
          moved.has(img.id) ? withLabels(img, labelId ? [labelId] : [], manifest.mode) : img,
        ),
      });
      markStale();
    },

    toggleImageLabels(imageIds, labelId) {
      const manifest = get().manifest;
      if (!manifest || imageIds.length === 0) return;
      const picked = new Set(imageIds);
      const targets = manifest.images.filter((img) => picked.has(img.id));
      // One tap, one outcome: add the label unless every image already has it.
      const removing = targets.every((img) => hasLabel(img, labelId));
      touch({
        images: manifest.images.map((img) => {
          if (!picked.has(img.id)) return img;
          const own = labelsOf(img);
          const next = removing
            ? own.filter((id) => id !== labelId)
            : own.includes(labelId)
              ? own
              : [...own, labelId];
          return withLabels(img, next, manifest.mode);
        }),
      });
      markStale();
    },

    async removeImages(imageIds) {
      const { manifest, path } = get();
      if (!manifest || !path) return;
      const doomed = manifest.images.filter((img) => imageIds.includes(img.id));
      if (doomed.length === 0) return;
      try {
        await storage.deleteImages(path, doomed.map((img) => img.file));
      } catch (err) {
        get().toast(errorText(err), "bad");
      }
      doomed.forEach((img) => featureCache.delete(img.id));
      touch({
        images: manifest.images.filter((img) => !imageIds.includes(img.id)),
      });
      markStale();
    },

    updateSettings(patch) {
      const manifest = get().manifest;
      if (!manifest) return;
      touch({ settings: { ...manifest.settings, ...patch } });
    },

    async startTraining() {
      // A background run is only ever a convenience, so a manual run wins.
      if (currentRun?.mode === "auto") {
        currentRun.stop = true;
        await currentRun.done;
      }
      if (currentRun) return;
      await beginRun("manual");
    },

    stopTraining() {
      if (currentRun) currentRun.stop = true;
    },

    resetTraining() {
      set({ train: idleTrainState });
    },

    async revertModel() {
      const { path, manifest } = get();
      if (!path || !manifest) return;
      cancelAutoRetrain();
      try {
        const text = await storage.readCheckpoint(path);
        if (!text) {
          set({ hasCheckpoint: false });
          get().toast("There is no earlier model to go back to", "bad");
          return;
        }
        await loadFeatureModel();
        const parsed = JSON.parse(text) as StoredHead;
        const restored = deserializeHead(parsed, manifest.settings);
        if (!restored) {
          get().toast("The earlier model no longer fits this project", "bad");
          return;
        }
        head?.dispose();
        head = restored;
        await storage.writeModel(path, text);
        await storage.clearCheckpoint(path);
        // A checkpoint written for a project that had no model metadata leaves
        // nothing to restore, and keeping the newer numbers would describe the
        // wrong weights, so the honest value is none.
        touch({ model: manifest.checkpoint ?? null, checkpoint: null });
        set({
          hasModel: true,
          hasCheckpoint: false,
          stale: true,
          modelLabelIds: parsed.labelIds,
          train: idleTrainState,
        });
        get().toast("Went back to the previous model", "good");
        await get().saveNow();
      } catch (err) {
        get().toast(errorText(err), "bad");
      }
    },

    setAttentionOn(on) {
      set({ attentionOn: on });
    },

    async predict(source, wantAttention = false) {
      const manifest = get().manifest;
      if (!manifest || !head) return null;
      const labelIds = get().modelLabelIds;
      const { pooled, spatial } = await embedWithSpatial(source, wantAttention);
      try {
        const row = predictOne(head, pooled);
        if (row.length === 0) return null;
        let best = 0;
        row.forEach((value, index) => {
          if (value > row[best]) best = index;
        });

        let attention: Attention | null = null;
        if (spatial && head) {
          // One batched pass: rows * cols tiny head inferences at once, on the
          // activation the pooled embedding was already made from.
          const cellScores = predictCells(head, spatial.cells);
          // Single mode has one winner worth showing. Multi mode shows every
          // label that answered yes on this frame.
          const shown =
            manifest.mode === "multi"
              ? labelIds.filter((_, i) => (row[i] ?? 0) >= MULTI_THRESHOLD)
              : labelIds[best]
                ? [labelIds[best]]
                : [];
          const maps: AttentionMap[] = shown.map((labelId) => {
            const index = labelIds.indexOf(labelId);
            return {
              labelId,
              values: cellScores.map((cell) => cell[index] ?? 0),
            };
          });
          attention = { rows: spatial.rows, cols: spatial.cols, maps };
        }

        return {
          labelId: labelIds[best] ?? "",
          confidence: row[best] ?? 0,
          scores: labelIds.map((labelId, i) => ({ labelId, score: row[i] ?? 0 })),
          attention,
        };
      } finally {
        spatial?.cells.dispose();
      }
    },
  };
});

/** Label lookup helper shared by the views. */
export function labelById(manifest: ProjectManifest | null, id: string | null): Label | null {
  if (!manifest || !id) return null;
  return manifest.labels.find((l) => l.id === id) ?? null;
}

export function imagesForLabel(manifest: ProjectManifest | null, id: string | null): ImageRecord[] {
  if (!manifest) return [];
  if (id === null) return manifest.images.filter((img) => labelsOf(img).length === 0);
  return manifest.images.filter((img) => hasLabel(img, id));
}
