// Image intake. Every image entering a project is center cropped and resized
// to IMAGE_SIZE and stored as JPEG, which keeps bundles small and makes the
// feature extractor input identical for imports, captures and feedback.

import { IMAGE_SIZE } from "./types";

const JPEG_QUALITY = 0.92;

export type Drawable = HTMLVideoElement | HTMLImageElement | ImageBitmap | HTMLCanvasElement;

function sourceSize(source: Drawable): { w: number; h: number } {
  if (source instanceof HTMLVideoElement) {
    return { w: source.videoWidth, h: source.videoHeight };
  }
  if (source instanceof HTMLImageElement) {
    return { w: source.naturalWidth, h: source.naturalHeight };
  }
  return { w: source.width, h: source.height };
}

/** Draws a center-cropped square of the source into a size x size canvas. */
export function toSquareCanvas(source: Drawable, size = IMAGE_SIZE): HTMLCanvasElement {
  const { w, h } = sourceSize(source);
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Canvas 2D is unavailable");
  if (!w || !h) throw new Error("Image has no pixels yet");

  const side = Math.min(w, h);
  const sx = (w - side) / 2;
  const sy = (h - side) / 2;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(source, sx, sy, side, side, 0, 0, size, size);
  return canvas;
}

export function canvasToJpegBase64(canvas: HTMLCanvasElement): string {
  const url = canvas.toDataURL("image/jpeg", JPEG_QUALITY);
  const comma = url.indexOf(",");
  return url.slice(comma + 1);
}

/** Normalizes any drawable into the base64 JPEG that gets written to disk. */
export function normalizeToJpegBase64(source: Drawable): string {
  return canvasToJpegBase64(toSquareCanvas(source));
}

export function loadImageElement(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.decoding = "async";
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("Could not decode this image"));
    img.src = src;
  });
}

export async function fileToJpegBase64(file: File): Promise<string> {
  const url = URL.createObjectURL(file);
  try {
    const img = await loadImageElement(url);
    return normalizeToJpegBase64(img);
  } finally {
    URL.revokeObjectURL(url);
  }
}

export async function base64ToJpegBase64(data: string, mime = "image/jpeg"): Promise<string> {
  const img = await loadImageElement(`data:${mime};base64,${data}`);
  return normalizeToJpegBase64(img);
}

const IMAGE_EXTENSIONS = ["jpg", "jpeg", "png", "webp", "bmp", "gif"];

export function isImageFile(file: File): boolean {
  if (file.type.startsWith("image/")) return true;
  const ext = file.name.split(".").pop()?.toLowerCase() ?? "";
  return IMAGE_EXTENSIONS.includes(ext);
}

/**
 * Pulls every image file out of a drop, walking directories when the webview
 * exposes the entry API. Falls back to the flat file list otherwise, which is
 * what platforms without folder drag and drop will hit. Drag and drop intake is
 * contained here so views never branch on platform behavior themselves.
 */
export async function filesFromDataTransfer(dt: DataTransfer): Promise<File[]> {
  const entries: FileSystemEntry[] = [];
  if (dt.items && dt.items.length > 0) {
    for (const item of Array.from(dt.items)) {
      const entry = item.webkitGetAsEntry?.();
      if (entry) entries.push(entry);
    }
  }
  if (entries.length === 0) {
    return Array.from(dt.files).filter(isImageFile);
  }

  const out: File[] = [];
  const walk = async (entry: FileSystemEntry, depth: number): Promise<void> => {
    if (depth > 6) return;
    if (entry.isFile) {
      const file = await new Promise<File | null>((resolve) => {
        (entry as FileSystemFileEntry).file(resolve, () => resolve(null));
      });
      if (file && isImageFile(file)) out.push(file);
      return;
    }
    if (entry.isDirectory) {
      const reader = (entry as FileSystemDirectoryEntry).createReader();
      // readEntries returns at most 100 entries per call, so drain it.
      for (;;) {
        const batch = await new Promise<FileSystemEntry[]>((resolve) => {
          reader.readEntries(resolve, () => resolve([]));
        });
        if (batch.length === 0) break;
        for (const child of batch) await walk(child, depth + 1);
      }
    }
  };

  for (const entry of entries) await walk(entry, 0);
  return out;
}
