// What belongs in a bundle's model/ folder, and what has to go.
//
// A trained model can stop being valid without anybody deleting it: adding or
// removing a label changes the head's output contract, so the weights on disk
// describe a project that no longer exists. Belo used to leave those files where
// they were and merely ignore them, which meant a bundle carried megabytes of
// dead JSON and a reader looking at the folder could not tell live weights from
// stale ones. They are deleted now, at the moment they become invalid and again
// when a project is opened.
//
// The decisions are pure and live here, so scripts/check-store.mjs can exercise
// every case without a filesystem. The rule that keeps this safe is the owned
// list below: nothing outside it is ever proposed for deletion, so an image, a
// manifest or a file somebody else put in model/ cannot be touched.

import type { ProjectMode } from "./types";

export const HEAD_FILE = "head.json";
export const HEAD_CHECKPOINT_FILE = "head.prev.json";
export const DETECTOR_FILE = "detect.json";
export const DETECTOR_CHECKPOINT_FILE = "detect.prev.json";

/** Every file inside model/ that Belo writes, and therefore may delete. */
export const OWNED_MODEL_FILES: string[] = [
  HEAD_FILE,
  HEAD_CHECKPOINT_FILE,
  DETECTOR_FILE,
  DETECTOR_CHECKPOINT_FILE,
];

export function ownsModelFile(name: string): boolean {
  return OWNED_MODEL_FILES.includes(name);
}

/** The live model file and its checkpoint for one project mode. */
export function modelFilesFor(mode: ProjectMode): { model: string; checkpoint: string } {
  return mode === "detect"
    ? { model: DETECTOR_FILE, checkpoint: DETECTOR_CHECKPOINT_FILE }
    : { model: HEAD_FILE, checkpoint: HEAD_CHECKPOINT_FILE };
}

/** Files that belong to the other kind of project and can only be leftovers. */
export function foreignModelFiles(mode: ProjectMode): string[] {
  const mine = modelFilesFor(mode);
  return OWNED_MODEL_FILES.filter((name) => name !== mine.model && name !== mine.checkpoint);
}

export interface ModelDirState {
  mode: ProjectMode;
  /** Everything currently inside model/, whoever wrote it. */
  files: string[];
  /** True when the manifest still describes a trained model. */
  hasModelMeta: boolean;
  /** True when the manifest still describes a checkpoint. */
  hasCheckpointMeta: boolean;
  /**
   * Whether the weights in a file still fit the project, for the owned files
   * that are present. A file missing from this map is treated as unreadable and
   * therefore incompatible.
   */
  compatible: Record<string, boolean>;
}

export interface ModelDirPlan {
  /** Owned files to delete, in a stable order. */
  remove: string[];
  /** True when the live model file went and the manifest must stop claiming one. */
  dropModelMeta: boolean;
  /** True when the checkpoint file went and its metadata must go with it. */
  dropCheckpointMeta: boolean;
}

/**
 * Decides what to delete from one model folder.
 *
 * Four things make an owned file dead, and nothing else does:
 *
 * - it belongs to the other kind of project, which cannot happen without a mode
 *   change that Belo does not allow, so it is a leftover from an older version
 *   or a hand edit;
 * - it holds weights whose label set no longer matches the manifest;
 * - it is a live model file the manifest no longer claims a model for;
 * - it is a checkpoint the manifest no longer describes, which would otherwise
 *   offer a revert to weights nothing knows the accuracy of.
 *
 * A missing file is never a problem: a bundle from before checkpoints existed
 * simply has none, and a project that was never trained has no model file.
 */
export function planModelDir(state: ModelDirState): ModelDirPlan {
  const present = new Set(state.files.filter(ownsModelFile));
  const mine = modelFilesFor(state.mode);
  const remove: string[] = [];

  for (const name of foreignModelFiles(state.mode)) {
    if (present.has(name)) remove.push(name);
  }

  const dead = (name: string, wanted: boolean): boolean => {
    if (!present.has(name)) return false;
    if (!wanted) return true;
    return state.compatible[name] !== true;
  };

  if (dead(mine.model, state.hasModelMeta)) remove.push(mine.model);
  if (dead(mine.checkpoint, state.hasCheckpointMeta)) remove.push(mine.checkpoint);

  return {
    remove: OWNED_MODEL_FILES.filter((name) => remove.includes(name)),
    dropModelMeta: state.hasModelMeta && remove.includes(mine.model),
    dropCheckpointMeta: state.hasCheckpointMeta && remove.includes(mine.checkpoint),
  };
}
