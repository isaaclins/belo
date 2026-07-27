// The single seam between the app and whatever holds a project.
//
// Every read and write the frontend performs goes through the ProjectStorage
// interface below, including the native pickers. Nothing else in src/ imports
// Tauri APIs or assumes a filesystem path layout. Swapping the desktop
// implementation for a sandboxed mobile one means writing a second object that
// satisfies this interface and changing the final export.

import { invoke, convertFileSrc } from "@tauri-apps/api/core";
import type { ProjectSummary } from "./types";

/** Opaque handle for an image picked outside a project, resolved by the storage layer. */
export interface ImportRef {
  /** Stable identifier, a filesystem path on desktop. */
  id: string;
  /** Display name, used for logs and error messages. */
  name: string;
}

export interface ImportPayload {
  base64: string;
  mime: string;
}

export interface ProjectStorage {
  /** Where new projects are created when the user does not choose a location. */
  defaultProjectsDir(): Promise<string>;
  createProject(dir: string, name: string, manifest: string): Promise<string>;
  readManifest(handle: string): Promise<string>;
  writeManifest(handle: string, manifest: string): Promise<void>;

  writeImage(handle: string, name: string, base64: string): Promise<void>;
  deleteImages(handle: string, names: string[]): Promise<void>;
  /** URL the webview can render for an image stored inside a project. */
  imageUrl(handle: string, name: string): string;

  writeModel(handle: string, weights: string): Promise<void>;
  readModel(handle: string): Promise<string | null>;

  recentProjects(): Promise<ProjectSummary[]>;
  scanProjects(dir: string): Promise<ProjectSummary[]>;
  rememberProject(handle: string, name: string): Promise<void>;
  forgetProject(handle: string): Promise<void>;

  /** Native picker for an existing project. Resolves to null when cancelled. */
  pickProject(startIn?: string | null): Promise<string | null>;
  /** Native picker for a folder of images. Resolves to null when cancelled. */
  pickImportFolder(): Promise<ImportRef[] | null>;
  /** Loads one picked image so the app can normalize it. */
  readImport(ref: ImportRef): Promise<ImportPayload>;
}

const IS_WINDOWS = typeof navigator !== "undefined" && navigator.userAgent.includes("Windows");
const SEP = IS_WINDOWS ? "\\" : "/";

function join(...parts: string[]): string {
  return parts.join(SEP);
}

function baseName(path: string): string {
  const cut = Math.max(path.lastIndexOf("/"), path.lastIndexOf("\\"));
  return cut === -1 ? path : path.slice(cut + 1);
}

function mimeFor(path: string): string {
  const ext = path.split(".").pop()?.toLowerCase();
  if (ext === "png") return "image/png";
  if (ext === "webp") return "image/webp";
  if (ext === "gif") return "image/gif";
  if (ext === "bmp") return "image/bmp";
  return "image/jpeg";
}

/** Desktop backend: .belo directory bundles reached through the Rust commands. */
const desktopStorage: ProjectStorage = {
  defaultProjectsDir: () => invoke("default_projects_dir"),

  createProject: (dir, name, manifest) => invoke("create_project", { dir, name, manifest }),

  readManifest: (path) => invoke("read_manifest", { path }),

  writeManifest: (path, manifest) => invoke("write_manifest", { path, manifest }),

  writeImage: async (path, name, data) => {
    await invoke("write_image", { path, name, data });
  },

  deleteImages: (path, names) => invoke("delete_images", { path, names }),

  imageUrl: (path, name) => convertFileSrc(join(path, "images", name)),

  writeModel: (path, weights) => invoke("write_model", { path, weights }),

  readModel: (path) => invoke("read_model", { path }),

  recentProjects: () => invoke("recent_projects"),

  scanProjects: (dir) => invoke("scan_projects", { dir }),

  rememberProject: (path, name) =>
    invoke("remember_project", { path, name, openedAt: new Date().toISOString() }),

  forgetProject: (path) => invoke("forget_project", { path }),

  async pickProject(startIn) {
    const { open } = await import("@tauri-apps/plugin-dialog");
    const picked = await open({
      directory: true,
      multiple: false,
      title: "Open a .belo project",
      defaultPath: startIn ?? undefined,
    });
    return typeof picked === "string" ? picked : null;
  },

  async pickImportFolder() {
    const { open } = await import("@tauri-apps/plugin-dialog");
    const dir = await open({ directory: true, multiple: false, title: "Import a folder" });
    if (typeof dir !== "string") return null;
    const found: string[] = await invoke("list_images_in_dir", { dir, recursive: true });
    return found.map((path) => ({ id: path, name: baseName(path) }));
  },

  async readImport(ref) {
    const base64: string = await invoke("read_file_base64", { path: ref.id });
    return { base64, mime: mimeFor(ref.id) };
  },
};

export const storage: ProjectStorage = desktopStorage;
