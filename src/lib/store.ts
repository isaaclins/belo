// Single application store. Owns the open project, disk writes, the feature
// cache and the trained head. Views read from here and never touch disk.

import { create } from "zustand";
import type * as tfTypes from "@tensorflow/tfjs";
import * as api from "./api";
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
  mimeForPath,
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

export const APP_VERSION = "0.1.0";

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
  modelLabelIds: string[];

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
  addFromPaths: (paths: string[], labelId: string | null) => Promise<number>;
  addFromDrawable: (source: Drawable, labelId: string | null, origin: ImageSource) => Promise<void>;
  moveImages: (imageIds: string[], labelId: string | null) => void;
  removeImages: (imageIds: string[]) => Promise<void>;

  updateSettings: (patch: Partial<TrainSettings>) => void;
  startTraining: () => Promise<void>;
  stopTraining: () => void;
  resetTraining: () => void;

  predict: (source: Drawable) => Promise<Prediction | null>;
}

// Non reactive side tables. Keeping tensors and typed arrays out of the store
// avoids pointless React work on every frame of a webcam session.
const featureCache = new Map<string, Float32Array>();
let head: tfTypes.Sequential | null = null;
let stopRequested = false;
let saveTimer: ReturnType<typeof setTimeout> | null = null;

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

  const dropModel = () => {
    head?.dispose();
    head = null;
    set({ hasModel: false, modelLabelIds: [] });
  };

  /** Loads and caches the embedding for one stored image. */
  const featureFor = async (record: ImageRecord): Promise<Float32Array> => {
    const cached = featureCache.get(record.id);
    if (cached) return cached;
    const path = get().path;
    if (!path) throw new Error("No project is open");
    const img = await loadImageElement(api.imageUrl(path, record.file));
    const feature = await embedToArray(img);
    featureCache.set(record.id, feature);
    return feature;
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
    await api.writeImage(path, file, jpegBase64);
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
    modelLabelIds: [],

    async boot() {
      try {
        const dir = await api.defaultProjectsDir();
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
        const recents = await api.recentProjects();
        const dir = get().projectsDir;
        const known = new Set(recents.map((r) => r.path));
        const scanned = dir ? await api.scanProjects(dir) : [];
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
        const dir = get().projectsDir ?? (await api.defaultProjectsDir());
        const manifest = newManifest(name.trim() || "Untitled", APP_VERSION);
        const path = await api.createProject(dir, manifest.name, JSON.stringify(manifest, null, 2));
        featureCache.clear();
        dropModel();
        set({
          path,
          manifest,
          view: "label",
          dirty: false,
          activeLabelId: null,
          train: idleTrainState,
        });
        await api.rememberProject(path, manifest.name);
        await get().refreshRecents();
      } catch (err) {
        get().toast(errorText(err), "bad");
      }
    },

    async openProject(path) {
      try {
        const text = await api.readManifest(path);
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
        });
        await api.rememberProject(path, manifest.name);
        await get().refreshRecents();

        // Restore a previously trained head so the Use view works immediately.
        const stored = await api.readModel(path);
        if (stored) {
          await loadFeatureModel();
          const parsed = JSON.parse(stored) as StoredHead;
          const restored = deserializeHead(parsed, manifest.settings);
          if (restored) {
            head = restored;
            set({ hasModel: true, modelLabelIds: parsed.labelIds });
          }
        }
      } catch (err) {
        get().toast(errorText(err), "bad");
      }
    },

    async openProjectDialog() {
      try {
        const { open } = await import("@tauri-apps/plugin-dialog");
        const picked = await open({
          directory: true,
          multiple: false,
          title: "Open a .belo project",
          defaultPath: get().projectsDir ?? undefined,
        });
        if (typeof picked === "string") await get().openProject(picked);
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
      });
      void get().refreshRecents();
    },

    async forgetProject(path) {
      await api.forgetProject(path);
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
        await api.writeManifest(path, JSON.stringify(manifest, null, 2));
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
        if (doomed.length > 0) await api.deleteImages(path, doomed.map((img) => img.file));
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
        touch({ images: [...get().manifest!.images, ...added], model: null });
        dropModel();
      }
      const skipped = files.length - added.length;
      if (skipped > 0) get().toast(`${skipped} file(s) could not be read`, "bad");
      return added.length;
    },

    async addFromPaths(paths, labelId) {
      const manifest = get().manifest;
      if (!manifest || paths.length === 0) return 0;
      const added: ImageRecord[] = [];
      set({ busy: { title: "Importing images", done: 0, total: paths.length } });
      for (let i = 0; i < paths.length; i += 1) {
        try {
          const raw = await api.readFileBase64(paths[i]);
          const jpeg = await base64ToJpegBase64(raw, mimeForPath(paths[i]));
          added.push(await writeImageRecord(jpeg, labelId, "import"));
        } catch {
          // Same as addFiles: unreadable files are skipped, not fatal.
        }
        set({ busy: { title: "Importing images", done: i + 1, total: paths.length } });
      }
      set({ busy: null });
      if (added.length > 0) {
        touch({ images: [...get().manifest!.images, ...added], model: null });
        dropModel();
      }
      return added.length;
    },

    async addFromDrawable(source, labelId, origin) {
      const manifest = get().manifest;
      if (!manifest) return;
      try {
        const jpeg = normalizeToJpegBase64(source);
        const record = await writeImageRecord(jpeg, labelId, origin);
        touch({ images: [...get().manifest!.images, record], model: null });
        if (origin !== "feedback") dropModel();
      } catch (err) {
        get().toast(errorText(err), "bad");
      }
    },

    moveImages(imageIds, labelId) {
      const manifest = get().manifest;
      if (!manifest) return;
      const set0 = new Set(imageIds);
      touch({
        images: manifest.images.map((img) => (set0.has(img.id) ? { ...img, labelId } : img)),
        model: null,
      });
      dropModel();
    },

    async removeImages(imageIds) {
      const { manifest, path } = get();
      if (!manifest || !path) return;
      const doomed = manifest.images.filter((img) => imageIds.includes(img.id));
      if (doomed.length === 0) return;
      try {
        await api.deleteImages(path, doomed.map((img) => img.file));
      } catch (err) {
        get().toast(errorText(err), "bad");
      }
      doomed.forEach((img) => featureCache.delete(img.id));
      touch({
        images: manifest.images.filter((img) => !imageIds.includes(img.id)),
        model: null,
      });
      dropModel();
    },

    updateSettings(patch) {
      const manifest = get().manifest;
      if (!manifest) return;
      touch({ settings: { ...manifest.settings, ...patch } });
    },

    async startTraining() {
      const manifest = get().manifest;
      const path = get().path;
      if (!manifest || !path) return;

      const usable = manifest.labels.filter((label) =>
        manifest.images.some((img) => img.labelId === label.id),
      );
      const samples = manifest.images.filter(
        (img) => img.labelId && usable.some((l) => l.id === img.labelId),
      );
      if (usable.length < 2) {
        get().toast("Add at least two labels with images before training", "bad");
        return;
      }

      stopRequested = false;
      set({
        train: {
          ...idleTrainState,
          phase: "reading",
          message: "Loading the feature extractor",
          total: samples.length,
        },
      });

      try {
        const backend = await initBackend();
        set({ backend });
        await loadFeatureModel();

        set((s) => ({ train: { ...s.train, message: "Reading images" } }));
        const features: Float32Array[] = [];
        const labelIndices: number[] = [];
        const labelIds = usable.map((l) => l.id);

        for (let i = 0; i < samples.length; i += 1) {
          if (stopRequested) throw new Error("Training cancelled");
          const record = samples[i];
          features.push(await featureFor(record));
          labelIndices.push(labelIds.indexOf(record.labelId!));
          set((s) => ({ train: { ...s.train, done: i + 1 } }));
        }

        set((s) => ({
          train: { ...s.train, phase: "training", message: "Training the classifier", done: 0 },
        }));

        const result = await trainHead({
          features,
          labelIndices,
          numClasses: labelIds.length,
          settings: manifest.settings,
          shouldStop: () => stopRequested,
          onProgress: (progress) => {
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

        head?.dispose();
        head = result.model;

        set((s) => ({ train: { ...s.train, phase: "scoring", message: "Checking each image" } }));
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
        await api.writeModel(path, JSON.stringify(stored));

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
          modelLabelIds: labelIds,
          train: {
            ...s.train,
            phase: "done",
            message: "Ready to use",
            accuracy: result.accuracy,
            valAccuracy: result.valAccuracy,
            verdicts,
          },
        }));
        await get().saveNow();
      } catch (err) {
        set((s) => ({ train: { ...s.train, phase: "error", message: errorText(err) } }));
        get().toast(errorText(err), "bad");
      }
    },

    stopTraining() {
      stopRequested = true;
    },

    resetTraining() {
      set({ train: idleTrainState });
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
