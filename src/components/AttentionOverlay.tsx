// Canvas overlay for the live camera: a tinted heat map upsampled from the
// feature grid plus one coarse box per shown label. It only ever draws what
// the store already computed, so a frame with the toggle off costs nothing.

import { useCallback, useEffect, useRef, type RefObject } from "react";
import type { Attention } from "../lib/attention";
import { boxPixels, cropSquare, heat, hotBox, mapRange } from "../lib/attention";
import type { Label } from "../lib/types";
import { INK_HEX, swatchHex } from "./ui";

/** The hottest cell never fully hides the picture underneath it. */
const MAX_ALPHA = 0.6;
const TAG_HEIGHT = 18;

function rgb(hex: string): [number, number, number] {
  return [
    parseInt(hex.slice(1, 3), 16),
    parseInt(hex.slice(3, 5), 16),
    parseInt(hex.slice(5, 7), 16),
  ];
}

export function AttentionOverlay({
  videoRef,
  attention,
  labels,
  active,
}: {
  videoRef: RefObject<HTMLVideoElement | null>;
  attention: Attention | null;
  labels: Label[];
  active: boolean;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const gridRef = useRef<HTMLCanvasElement | null>(null);
  // Kept so a resize can repaint the last frame without waiting for a new one.
  const lastRef = useRef<Attention | null>(null);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    if (width === 0 || height === 0) return;

    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const pixelW = Math.round(width * dpr);
    const pixelH = Math.round(height * dpr);
    if (canvas.width !== pixelW || canvas.height !== pixelH) {
      canvas.width = pixelW;
      canvas.height = pixelH;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);

    const map = lastRef.current;
    const video = videoRef.current;
    if (!active || !map || map.maps.length === 0 || !video) return;
    const rect = cropSquare(
      { width, height },
      { width: video.videoWidth, height: video.videoHeight },
    );
    if (!rect) return;

    const grid = gridRef.current ?? document.createElement("canvas");
    gridRef.current = grid;
    grid.width = map.cols;
    grid.height = map.rows;
    const gridCtx = grid.getContext("2d");
    if (!gridCtx) return;

    for (const entry of map.maps) {
      const label = labels.find((l) => l.id === entry.labelId);
      const color = swatchHex(label?.color ?? 0);
      const [r, g, b] = rgb(color);
      const range = mapRange(entry.values);

      // One pixel per cell, then let the compositor interpolate it up to the
      // square the model actually saw. Same tint everywhere, only alpha moves,
      // so bilinear smoothing cannot smear colors into each other.
      const image = gridCtx.createImageData(map.cols, map.rows);
      for (let i = 0; i < entry.values.length; i += 1) {
        const alpha = heat(entry.values[i] ?? 0, range) * MAX_ALPHA;
        image.data[i * 4] = r;
        image.data[i * 4 + 1] = g;
        image.data[i * 4 + 2] = b;
        image.data[i * 4 + 3] = Math.round(alpha * 255);
      }
      gridCtx.putImageData(image, 0, 0);
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = "high";
      ctx.drawImage(grid, rect.x, rect.y, rect.size, rect.size);

      const cells = hotBox(entry.values, map.rows, map.cols);
      if (!cells) continue;
      const box = boxPixels(rect, map.rows, map.cols, cells);

      ctx.lineJoin = "miter";
      ctx.strokeStyle = INK_HEX;
      ctx.lineWidth = 6;
      ctx.strokeRect(box.x, box.y, box.width, box.height);
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.strokeRect(box.x, box.y, box.width, box.height);

      const name = (label?.name ?? "?").toUpperCase();
      ctx.font = '700 11px "Archivo Black", "Helvetica Neue", sans-serif';
      ctx.textBaseline = "middle";
      const tagW = Math.ceil(ctx.measureText(name).width) + 12;
      const tagX = Math.max(2, Math.min(box.x, width - tagW - 2));
      // Above the box when there is room, tucked inside its top edge otherwise.
      const tagY = box.y - TAG_HEIGHT - 3 >= 2 ? box.y - TAG_HEIGHT - 3 : box.y + 3;
      ctx.fillStyle = color;
      ctx.fillRect(tagX, tagY, tagW, TAG_HEIGHT);
      ctx.strokeStyle = INK_HEX;
      ctx.lineWidth = 2;
      ctx.strokeRect(tagX, tagY, tagW, TAG_HEIGHT);
      ctx.fillStyle = INK_HEX;
      ctx.fillText(name, tagX + 6, tagY + TAG_HEIGHT / 2 + 1);
    }
  }, [active, labels, videoRef]);

  useEffect(() => {
    lastRef.current = attention;
    draw();
  }, [attention, draw]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const observer = new ResizeObserver(() => draw());
    observer.observe(canvas);
    return () => observer.disconnect();
  }, [draw]);

  return <canvas ref={canvasRef} className="use__attn" aria-hidden="true" />;
}
