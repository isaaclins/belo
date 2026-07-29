// What the Home list says about a project, computed from the manifest.
//
// The Rust side summarizes a bundle by reading its project.json, and the Home
// list is a merge of that scan with a remembered recents index. Both are
// snapshots of the disk at the moment they were taken, and an index is by
// definition older than the project it points at.
//
// This module is the frontend's own reading of the same numbers, taken from the
// manifest the app just parsed, and the rule that the manifest wins. It is pure,
// so scripts/check-store.mjs can pin both: a manifest with content can never
// summarize to zero, and a recents entry that disagrees is replaced rather than
// merged. See "Known past anomalies" in the README for why that is written down
// as a check rather than trusted.

import { boxesOf, labelsOf, type ProjectManifest, type ProjectSummary } from "./types";

/**
 * The summary of one open project. `trained` means what it means everywhere
 * else: the manifest describes a model and the weights for it are on disk.
 */
export function summarizeManifest(
  path: string,
  manifest: ProjectManifest,
  hasModelFile: boolean,
): ProjectSummary {
  return {
    path,
    name: manifest.name,
    labelCount: manifest.labels.length,
    imageCount: manifest.images.length,
    trained: Boolean(manifest.model) && hasModelFile,
    updatedAt: manifest.updatedAt,
    missing: false,
  };
}

/** Labels, images and boxes a manifest actually holds, for a plain sanity read. */
export function manifestContent(manifest: ProjectManifest): {
  labels: number;
  images: number;
  labeled: number;
  boxes: number;
} {
  return {
    labels: manifest.labels.length,
    images: manifest.images.length,
    labeled: manifest.images.filter((image) => labelsOf(image).length > 0).length,
    boxes: manifest.images.reduce((sum, image) => sum + boxesOf(image).length, 0),
  };
}

/**
 * Replaces whatever the recents index and the folder scan believed about one
 * project with what its manifest says, keeping its place in the list. An entry
 * that was not in the list is put at the front, which is where opening a project
 * puts it anyway.
 */
export function reconcileRecents(
  recents: ProjectSummary[],
  truth: ProjectSummary,
): ProjectSummary[] {
  const at = recents.findIndex((entry) => entry.path === truth.path);
  if (at < 0) return [truth, ...recents];
  const next = [...recents];
  next[at] = truth;
  return next;
}
