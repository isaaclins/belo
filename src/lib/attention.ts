// Attention overlay math. Pure functions only: no tensors, no DOM, no React,
// so the geometry and the thresholding can be reasoned about and checked on
// their own. The drawing lives in components/AttentionOverlay.tsx.

/** One label's score map over the feature grid, row major, rows * cols long. */
export interface AttentionMap {
  labelId: string;
  values: number[];
}

export interface Attention {
  rows: number;
  cols: number;
  maps: AttentionMap[];
}

/** Cells at or above this share of the map's own range count as the hot region. */
export const HOT_RATIO = 0.6;

export interface Rect {
  x: number;
  y: number;
  size: number;
}

export interface Size {
  width: number;
  height: number;
}

/**
 * Where the classified pixels actually land on screen.
 *
 * Two crops stack up between the camera sensor and the overlay. The video
 * element is drawn with object-fit: cover, so the frame is scaled by
 * max(cw/vw, ch/vh) and centered, which pushes the overflowing axis out of
 * view. The classifier then sees toSquareCanvas(video), a centered square of
 * side min(vw, vh) taken from the same frame. Both crops are centered, so the
 * square the model saw maps to a centered square of side min(vw, vh) * scale.
 * Returning it in container pixels lets the canvas sit at inset 0 and still
 * line the 7x7 grid up with what the camera shows.
 */
function fittedSquare(container: Size, source: Size, fit: "cover" | "contain"): Rect | null {
  const { width: cw, height: ch } = container;
  const { width: sw, height: sh } = source;
  if (cw <= 0 || ch <= 0 || sw <= 0 || sh <= 0) return null;
  const scale = fit === "cover" ? Math.max(cw / sw, ch / sh) : Math.min(cw / sw, ch / sh);
  const offsetX = (cw - sw * scale) / 2;
  const offsetY = (ch - sh * scale) / 2;
  const side = Math.min(sw, sh);
  return {
    x: offsetX + ((sw - side) / 2) * scale,
    y: offsetY + ((sh - side) / 2) * scale,
    size: side * scale,
  };
}

export function cropSquare(container: Size, video: Size): Rect | null {
  return fittedSquare(container, video, "cover");
}

/** The model's center square inside an object-fit: contain single image. */
export function containSquare(container: Size, image: Size): Rect | null {
  return fittedSquare(container, image, "contain");
}

export interface Range {
  min: number;
  max: number;
}

export function mapRange(values: number[]): Range {
  if (values.length === 0) return { min: 0, max: 0 };
  let min = values[0];
  let max = values[0];
  for (const value of values) {
    if (value < min) min = value;
    if (value > max) max = value;
  }
  return { min, max };
}

/**
 * Tint strength per cell. Normalizing against the map's own range shows where
 * the evidence sits rather than how loud the whole frame is, which is the
 * honest reading of a relative map. A flat map yields nothing to show.
 */
export function heat(value: number, range: Range): number {
  const span = range.max - range.min;
  if (span <= 1e-6) return 0;
  return Math.max(0, Math.min(1, (value - range.min) / span));
}

/** Inclusive cell bounds of the hot region, or null when there is none. */
export interface CellBox {
  col0: number;
  row0: number;
  col1: number;
  row1: number;
}

/**
 * The box is thresholded on the same normalized heat the tint uses, not on a
 * share of the raw score. A confident softmax puts every cell near 1.0, so a
 * fraction of the maximum would box the whole frame and say nothing; the share
 * of the map's own range is what actually marks the evidence.
 */
export function hotBox(
  values: number[],
  rows: number,
  cols: number,
  ratio = HOT_RATIO,
): CellBox | null {
  const range = mapRange(values);
  if (range.max <= 0) return null;
  let col0 = cols;
  let row0 = rows;
  let col1 = -1;
  let row1 = -1;
  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      if (heat(values[row * cols + col] ?? 0, range) < ratio) continue;
      if (col < col0) col0 = col;
      if (col > col1) col1 = col;
      if (row < row0) row0 = row;
      if (row > row1) row1 = row;
    }
  }
  if (col1 < 0) return null;
  return { col0, row0, col1, row1 };
}

/** Turns inclusive cell bounds into pixels inside the square the model saw. */
export function boxPixels(rect: Rect, rows: number, cols: number, box: CellBox) {
  const cw = rect.size / cols;
  const chh = rect.size / rows;
  return {
    x: rect.x + box.col0 * cw,
    y: rect.y + box.row0 * chh,
    width: (box.col1 - box.col0 + 1) * cw,
    height: (box.row1 - box.row0 + 1) * chh,
  };
}
