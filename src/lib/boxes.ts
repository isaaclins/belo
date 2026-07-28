// Box geometry for the detect mode annotator. Pure functions on normalized
// 0..1 rectangles, so they can be checked in node without a DOM:
//
//   node scripts/check-boxes.mjs

import type { BoxRect } from "./types";

/** The eight grips of a selected box, named by compass direction. */
export const HANDLES = ["nw", "n", "ne", "e", "se", "s", "sw", "w"] as const;
export type Handle = (typeof HANDLES)[number];

export interface Point {
  x: number;
  y: number;
}

export type DragKind = "draw" | "move" | "resize";

export interface Drag {
  kind: DragKind;
  /** Null while a brand new box is being dragged out. */
  boxId: string | null;
  handle: Handle | null;
  /** The rectangle as it was when the pointer went down. */
  origin: BoxRect;
  grab: Point;
  /** Live rectangle. Width and height may be negative mid drag. */
  rect: BoxRect;
}

export function clamp(value: number, low: number, high: number): number {
  return Math.max(low, Math.min(high, value));
}

/** Turns a possibly backwards rectangle into a positive one for rendering. */
export function positive(rect: BoxRect): BoxRect {
  return {
    x: Math.min(rect.x, rect.x + rect.w),
    y: Math.min(rect.y, rect.y + rect.h),
    w: Math.abs(rect.w),
    h: Math.abs(rect.h),
  };
}

/** Moving keeps the size and stops at the edges of the picture. */
export function moved(origin: BoxRect, grab: Point, point: Point): BoxRect {
  return {
    x: clamp(origin.x + (point.x - grab.x), 0, 1 - origin.w),
    y: clamp(origin.y + (point.y - grab.y), 0, 1 - origin.h),
    w: origin.w,
    h: origin.h,
  };
}

/**
 * Resizing moves exactly the edges the grip is named after. Dragging an edge
 * past its opposite gives a negative width or height, which positive() reads as
 * a flipped box rather than an error.
 */
export function resized(origin: BoxRect, handle: Handle, point: Point): BoxRect {
  let x0 = origin.x;
  let y0 = origin.y;
  let x1 = origin.x + origin.w;
  let y1 = origin.y + origin.h;
  if (handle.includes("w")) x0 = point.x;
  if (handle.includes("e")) x1 = point.x;
  if (handle.includes("n")) y0 = point.y;
  if (handle.includes("s")) y1 = point.y;
  return { x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
}

/** One pointer move applied to whichever gesture is in flight. */
export function advance(drag: Drag, point: Point): Drag {
  if (drag.kind === "draw") {
    return {
      ...drag,
      rect: {
        x: drag.grab.x,
        y: drag.grab.y,
        w: point.x - drag.grab.x,
        h: point.y - drag.grab.y,
      },
    };
  }
  if (drag.kind === "move") return { ...drag, rect: moved(drag.origin, drag.grab, point) };
  return { ...drag, rect: resized(drag.origin, drag.handle ?? "se", point) };
}
