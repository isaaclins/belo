// Single application store. Owns the open project, disk writes, the feature
// cache and the trained head. Views read from here and never touch disk.

import { create } from "zustand";
import type * as tfTypes from "@tensorflow/tfjs";
import { storage, type ImportRef } from "./storage";
import {
  DEFAULT_SETTINGS,
  makeId,
  newManifest,
  SWATCH_COUNT,
  type ImageRecord,
  type ImageSource,
  type Label,
  type ProjectManifest,
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
import { embedToArray, loadFeatureModel } from "./ml/features";
import {
  deserializeHead,
  predictBatch,
  predictOne,
  serializeHead,
  trainHead,
  type TrainProgress,
} from "./ml/classifier";

export const APP_VERSION = "0.2.0";

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
  actualLabelId: string;
  predictedLabelId: string;
  confidence: number;
  correct: boolean;
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
}

export interface Prediction {
  labelId: string;
  confidence: number;
  scores: { labelId: string; score: number }[];
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

  boot: () => Promise<void>;
  refreshRecents: () => Promise<void>;
  setView: (view: View) => void;
  toast: (text: string, tone?: Toast["tone"]) => void;
  dismissToast: (id: string) => void;

  createProject: (name: string) => Promise<void>;
  openProject: (path: string) => Promise<void>;
  openProjectDialog: () => Promise<void>;
  closeProject: () => void;
  forgetProject: (path: string) => Promise<void>;
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
  addFromDrawable: (source: Drawable, labelId: string | null, origin: ImageSource) => Promise<void>;
  moveImages: (imageIds: string[], labelId: string | null) => void;
  removeImages: (imageIds: string[]) => Promise<void>;

  updateSettings: (patch: Partial<TrainSettings>) => void;
  startTraining: () => Promise<void>;
  stopTraining: () => void;
  resetTraining: () => void;
  /** Restores model/head.prev.json as the live model and drops the checkpoint. */
  revertModel: () => Promise<void>;

  predict: (source: Drawable) => Promise<Prediction | null>;
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
};

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
      manifest.images.some((img) => img.labelId === label.id),
    );
    const ids = new Set(usable.map((l) => l.id));
    const samples = manifest.images.filter((img) => img.labelId && ids.has(img.labelId));
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
      const features: Float32Array[] = [];
      const labelIndices: number[] = [];

      for (let i = 0; i < samples.length; i += 1) {
        if (run.stop) throw new Error("Training cancelled");
        const record = samples[i];
        features.push(await featureFor(record));
        labelIndices.push(labelIds.indexOf(record.labelId!));
        if (!auto) set((s) => ({ train: { ...s.train, done: i + 1 } }));
      }

      if (!auto) {
        set((s) => ({
          train: { ...s.train, phase: "training", message: "Training the classifier", done: 0 },
        }));
      }

      const result = await trainHead({
        features,
        labelIndices,
        numClasses: labelIds.length,
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
      const verdicts: ImageVerdict[] = samples.map((record, i) => {
        const row = scores[i] ?? [];
        let best = 0;
        row.forEach((value, index) => {
          if (value > row[best]) best = index;
        });
        return {
          imageId: record.id,
          actualLabelId: record.labelId!,
          predictedLabelId: labelIds[best] ?? "",
          confidence: row[best] ?? 0,
          correct: labelIds[best] === record.labelId,
        };
      });

      const stored = serializeHead(result.model, labelIds, manifest.settings.hiddenUnits);
      await storage.writeModel(path, JSON.stringify(stored));

      touch({
        model: {
          trainedAt: new Date().toISOString(),
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
    labelId: string | null,
    origin: ImageSource,
  ): Promise<ImageRecord> => {
    const path = get().path;
    if (!path) throw new Error("No project is open");
    const id = makeId("img");
    const file = `${id}.jpg`;
    await storage.writeImage(path, file, jpegBase64);
    return { id, file, labelId, source: origin, addedAt: new Date().toISOString() };
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

    async createProject(name) {
      try {
        const dir = get().projectsDir ?? (await storage.defaultProjectsDir());
        const manifest = newManifest(name.trim() || "Untitled", APP_VERSION);
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
        const manifest = JSON.parse(text) as ProjectManifest;
        manifest.settings = { ...DEFAULT_SETTINGS, ...manifest.settings };
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
      const doomed = manifest.images.filter((img) => img.labelId === id);
      try {
        if (doomed.length > 0) await storage.deleteImages(path, doomed.map((img) => img.file));
      } catch (err) {
        get().toast(errorText(err), "bad");
      }
      doomed.forEach((img) => featureCache.delete(img.id));
      const labels = manifest.labels.filter((l) => l.id !== id);
      touch({
        labels,
        images: manifest.images.filter((img) => img.labelId !== id),
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

    async addFromDrawable(source, labelId, origin) {
      const manifest = get().manifest;
      if (!manifest) return;
      try {
        const jpeg = normalizeToJpegBase64(source);
        const record = await writeImageRecord(jpeg, labelId, origin);
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
        images: manifest.images.map((img) => (moved.has(img.id) ? { ...img, labelId } : img)),
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
        touch({ model: manifest.checkpoint ?? manifest.model, checkpoint: null });
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

    async predict(source) {
      const manifest = get().manifest;
      if (!manifest || !head) return null;
      const labelIds = get().modelLabelIds;
      const feature = await embedToArray(source);
      const row = predictOne(head, feature);
      if (row.length === 0) return null;
      let best = 0;
      row.forEach((value, index) => {
        if (value > row[best]) best = index;
      });
      return {
        labelId: labelIds[best] ?? "",
        confidence: row[best] ?? 0,
        scores: labelIds.map((labelId, i) => ({ labelId, score: row[i] ?? 0 })),
      };
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
  return manifest.images.filter((img) => img.labelId === id);
}
