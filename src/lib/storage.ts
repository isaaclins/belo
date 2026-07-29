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
  /** Platform capabilities used to avoid offering unsupported folder workflows. */
  readonly mobile: boolean;
  readonly canPickProjects: boolean;
  readonly canPickFolders: boolean;

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

  /** Detection head of a detect project, kept beside the classifier head. */
  writeDetector(handle: string, weights: string): Promise<void>;
  readDetector(handle: string): Promise<string | null>;

  /** Copy of the head kept before an automatic retrain, used by the revert action. */
  writeCheckpoint(handle: string, weights: string): Promise<void>;
  readCheckpoint(handle: string): Promise<string | null>;
  hasCheckpoint(handle: string): Promise<boolean>;
  clearCheckpoint(handle: string): Promise<void>;

  /** The same story for a detect project, kept in its own file. */
  writeDetectorCheckpoint(handle: string, weights: string): Promise<void>;
  readDetectorCheckpoint(handle: string): Promise<string | null>;
  hasDetectorCheckpoint(handle: string): Promise<boolean>;
  clearDetectorCheckpoint(handle: string): Promise<void>;

  recentProjects(): Promise<ProjectSummary[]>;
  scanProjects(dir: string): Promise<ProjectSummary[]>;
  rememberProject(handle: string, name: string): Promise<void>;
  forgetProject(handle: string): Promise<void>;
  /** Erases the whole bundle from disk. Only accepts a real .belo directory. */
  deleteProject(handle: string): Promise<void>;

  /** Native picker for an existing project. Resolves to null when cancelled. */
  pickProject(startIn?: string | null): Promise<string | null>;
  /** Native picker for image files or photos. Resolves to null when cancelled. */
  pickImportImages(multiple?: boolean): Promise<ImportRef[] | null>;
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

function mimeForBytes(bytes: Uint8Array, name: string): string {
  if (bytes[0] === 0x89 && bytes[1] === 0x50 && bytes[2] === 0x4e && bytes[3] === 0x47) {
    return "image/png";
  }
  if (bytes[0] === 0x47 && bytes[1] === 0x49 && bytes[2] === 0x46) return "image/gif";
  if (bytes[0] === 0x42 && bytes[1] === 0x4d) return "image/bmp";
  if (
    bytes[0] === 0x52 &&
    bytes[1] === 0x49 &&
    bytes[2] === 0x46 &&
    bytes[3] === 0x46 &&
    bytes[8] === 0x57 &&
    bytes[9] === 0x45 &&
    bytes[10] === 0x42 &&
    bytes[11] === 0x50
  ) {
    return "image/webp";
  }
  return mimeFor(name);
}

function bytesToBase64(bytes: Uint8Array): string {
  let binary = "";
  const chunkSize = 0x8000;
  for (let offset = 0; offset < bytes.length; offset += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(offset, offset + chunkSize));
  }
  return btoa(binary);
}

function pickedName(path: string, index: number): string {
  try {
    return decodeURIComponent(baseName(path)) || `Image ${index + 1}`;
  } catch {
    return baseName(path) || `Image ${index + 1}`;
  }
}

async function pickImages(mobile: boolean, multiple = true): Promise<ImportRef[] | null> {
  const { open } = await import("@tauri-apps/plugin-dialog");
  const picked = await open({
    multiple,
    directory: false,
    title: "Import images",
    pickerMode: mobile ? "image" : undefined,
    fileAccessMode: mobile ? "copy" : undefined,
    filters: [
      {
        name: "Images",
        extensions: mobile ? ["image/*"] : ["jpg", "jpeg", "png", "webp", "gif", "bmp"],
      },
    ],
  });
  if (!picked) return null;
  const paths = Array.isArray(picked) ? picked : [picked];
  return paths.map((path, index) => ({ id: path, name: pickedName(path, index) }));
}

/** Desktop backend: .belo directory bundles reached through the Rust commands. */
const desktopStorage: ProjectStorage = {
  mobile: false,
  canPickProjects: true,
  canPickFolders: true,

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

  writeDetector: (path, weights) => invoke("write_detector", { path, weights }),

  readDetector: (path) => invoke("read_detector", { path }),

  writeCheckpoint: (path, weights) => invoke("write_checkpoint", { path, weights }),

  readCheckpoint: (path) => invoke("read_checkpoint", { path }),

  hasCheckpoint: (path) => invoke("has_checkpoint", { path }),

  clearCheckpoint: (path) => invoke("clear_checkpoint", { path }),

  writeDetectorCheckpoint: (path, weights) => invoke("write_detector_checkpoint", { path, weights }),

  readDetectorCheckpoint: (path) => invoke("read_detector_checkpoint", { path }),

  hasDetectorCheckpoint: (path) => invoke("has_detector_checkpoint", { path }),

  clearDetectorCheckpoint: (path) => invoke("clear_detector_checkpoint", { path }),

  recentProjects: () => invoke("recent_projects"),

  scanProjects: (dir) => invoke("scan_projects", { dir }),

  rememberProject: (path, name) =>
    invoke("remember_project", { path, name, openedAt: new Date().toISOString() }),

  forgetProject: (path) => invoke("forget_project", { path }),

  deleteProject: (path) => invoke("delete_project", { path }),

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

  pickImportImages: (multiple) => pickImages(false, multiple),

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

/** Mobile backend: projects stay in the sandbox and media comes from the native picker. */
const mobileStorage: ProjectStorage = {
  ...desktopStorage,
  mobile: true,
  canPickProjects: false,
  canPickFolders: false,

  async pickProject() {
    throw new Error("Opening external .belo folders is not available on mobile yet");
  },

  pickImportImages: (multiple) => pickImages(true, multiple),

  async pickImportFolder() {
    throw new Error("Folder import is not available on mobile. Choose photos instead.");
  },

  async readImport(ref) {
    const { readFile } = await import("@tauri-apps/plugin-fs");
    const bytes = await readFile(ref.id);
    return {
      base64: bytesToBase64(bytes),
      mime: mimeForBytes(bytes, ref.name),
    };
  },
};

const IS_MOBILE =
  typeof navigator !== "undefined" &&
  (/Android|iPhone|iPad|iPod/i.test(navigator.userAgent) ||
    (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1));

export const storage: ProjectStorage = IS_MOBILE ? mobileStorage : desktopStorage;
