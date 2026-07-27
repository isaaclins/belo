// Thin wrapper over the Rust commands. Everything that touches disk goes
// through here so views never call invoke directly.

import { invoke, convertFileSrc } from "@tauri-apps/api/core";
import type { ProjectSummary } from "./types";

export function defaultProjectsDir(): Promise<string> {
  return invoke("default_projects_dir");
}

export function createProject(dir: string, name: string, manifest: string): Promise<string> {
  return invoke("create_project", { dir, name, manifest });
}

export function readManifest(path: string): Promise<string> {
  return invoke("read_manifest", { path });
}

export function writeManifest(path: string, manifest: string): Promise<void> {
  return invoke("write_manifest", { path, manifest });
}

export function writeImage(path: string, name: string, data: string): Promise<string> {
  return invoke("write_image", { path, name, data });
}

export function deleteImages(path: string, names: string[]): Promise<void> {
  return invoke("delete_images", { path, names });
}

export function writeModel(path: string, weights: string): Promise<void> {
  return invoke("write_model", { path, weights });
}

export function readModel(path: string): Promise<string | null> {
  return invoke("read_model", { path });
}

export function readFileBase64(path: string): Promise<string> {
  return invoke("read_file_base64", { path });
}

export function listImagesInDir(dir: string, recursive = true): Promise<string[]> {
  return invoke("list_images_in_dir", { dir, recursive });
}

export function projectSummary(path: string): Promise<ProjectSummary> {
  return invoke("project_summary", { path });
}

export function scanProjects(dir: string): Promise<ProjectSummary[]> {
  return invoke("scan_projects", { dir });
}

export function recentProjects(): Promise<ProjectSummary[]> {
  return invoke("recent_projects");
}

export function rememberProject(path: string, name: string): Promise<void> {
  return invoke("remember_project", { path, name, openedAt: new Date().toISOString() });
}

export function forgetProject(path: string): Promise<void> {
  return invoke("forget_project", { path });
}

/** Path separator that matches the host, good enough for join duties. */
export const SEP = navigator.userAgent.includes("Windows") ? "\\" : "/";

export function joinPath(...parts: string[]): string {
  return parts.join(SEP);
}

export function imageUrl(projectPath: string, file: string): string {
  return convertFileSrc(joinPath(projectPath, "images", file));
}

export function isTauri(): boolean {
  return typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
}
