// Draws the social preview card for the landing page.
//
//   node scripts/make-og.mjs        docs/og.png, 1200x630
//
// Link previews on every network want a raster. The site's own mark is an SVG,
// and the wordmark on the page is a webfont, neither of which a preview crawler
// will render for you. So the card is drawn here the same way the app icon is:
// flat shapes sampled on a design grid, supersampled, written as a PNG with no
// alpha channel, and no image or font dependency in the repo.
//
// The letters are hand authored as thick strokes and arcs rather than loaded
// from a font. The card only needs two short lines of uppercase text, the shape
// language is the same geometry the mark is built from, and a woff2 rasterizer
// would be a much bigger thing to carry than fifteen glyphs.
import { deflateSync } from "node:zlib";
import { writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const WIDTH = 1200;
const HEIGHT = 630;
const SAMPLES = 3; // supersampling per axis, so 9 samples per pixel

const INK = [17, 16, 20]; // --ink
const PAPER = [244, 239, 228]; // --paper
const PAPER_DEEP = [231, 223, 205]; // --paper-deep
const YELLOW = [255, 216, 61]; // --accent
const PINK = [255, 92, 138]; // --accent-3

// ---------------------------------------------------------------- primitives
//
// Every shape answers one question: is this point inside me. The canvas is then
// painted by asking the shapes in front to back order, which keeps the code as
// flat as the design.

const rect = (x, y, w, h) => (px, py) => px >= x && px <= x + w && py >= y && py <= y + h;

/** A rectangle outline of thickness `t`, drawn inside the given bounds. */
const frame = (x, y, w, h, t) => (px, py) =>
  rect(x, y, w, h)(px, py) && !rect(x + t, y + t, w - 2 * t, h - 2 * t)(px, py);

/** A thick straight stroke between two points, with square ends. */
const line = (x1, y1, x2, y2, t) => {
  const dx = x2 - x1;
  const dy = y2 - y1;
  const len2 = dx * dx + dy * dy || 1;
  const half = t / 2;
  return (px, py) => {
    const s = Math.max(0, Math.min(1, ((px - x1) * dx + (py - y1) * dy) / len2));
    const ox = px - (x1 + s * dx);
    const oy = py - (y1 + s * dy);
    return ox * ox + oy * oy <= half * half;
  };
};

/**
 * A slice of a ring: centre, radius to the middle of the stroke, thickness, and
 * the angular range in degrees measured clockwise from three o'clock, which is
 * how the glyph shapes below are easiest to read.
 */
const arc = (cx, cy, r, t, from = 0, to = 360) => {
  const inner = (r - t / 2) ** 2;
  const outer = (r + t / 2) ** 2;
  return (px, py) => {
    const dx = px - cx;
    const dy = py - cy;
    const d2 = dx * dx + dy * dy;
    if (d2 < inner || d2 > outer) return false;
    if (to - from >= 360) return true;
    let a = (Math.atan2(dy, dx) * 180) / Math.PI;
    if (a < 0) a += 360;
    if (a < from) a += 360;
    return a <= to;
  };
};

const any =
  (...shapes) =>
  (px, py) =>
    shapes.some((shape) => shape(px, py));

// ------------------------------------------------------------------- letters
//
// Each glyph is drawn in a box `w` wide and `CAP` tall with the stroke width
// `t`, on its own local coordinates with the origin at the top left. The widths
// are proportional to the stroke so one font serves both text sizes.

const CAP = 1; // cap height in units of the size passed to `text`

/** The glyphs the card needs, as functions from stroke width to shapes. */
const GLYPHS = {
  A: (h, t) => ({
    w: 0.72 * h,
    shapes: (x, y, W) =>
      any(
        line(x + t / 2, y + h, x + W / 2, y + t / 2, t),
        line(x + W - t / 2, y + h, x + W / 2, y + t / 2, t),
        rect(x + 0.19 * W, y + 0.62 * h, 0.62 * W, t),
      ),
  }),
  B: (h, t) => ({
    w: 0.66 * h,
    shapes: (x, y, W) => {
      const r = (h - t) / 4;
      const cx = x + W - r - t / 2;
      return any(
        rect(x, y, t, h),
        rect(x, y, cx - x, t),
        rect(x, y + h / 2 - t / 2, cx - x, t),
        rect(x, y + h - t, cx - x, t),
        arc(cx, y + r + t / 2, r, t, 270, 450),
        arc(cx, y + h - r - t / 2, r, t, 270, 450),
      );
    },
  }),
  C: (h, t) => ({
    w: 0.74 * h,
    shapes: (x, y, W) => {
      const r = h / 2 - t / 2;
      const cx = x + W - r - t / 2;
      return any(
        arc(cx, y + r + t / 2, r, t, 180, 340),
        arc(cx, y + h - r - t / 2, r, t, 20, 180),
        rect(x + t / 2 - t / 2, y + r + t / 2, t, h - 2 * r - t),
      );
    },
  }),
  D: (h, t) => ({
    w: 0.74 * h,
    shapes: (x, y, W) => {
      const r = h / 2 - t / 2;
      return any(
        rect(x, y, t, h),
        rect(x, y, W - r, t),
        rect(x, y + h - t, W - r, t),
        arc(x + W - r - t / 2, y + h / 2, r, t, 270, 450),
      );
    },
  }),
  E: (h, t) => ({
    w: 0.62 * h,
    shapes: (x, y, W) =>
      any(
        rect(x, y, t, h),
        rect(x, y, W, t),
        rect(x, y + h / 2 - t / 2, W * 0.86, t),
        rect(x, y + h - t, W, t),
      ),
  }),
  H: (h, t) => ({
    w: 0.72 * h,
    shapes: (x, y, W) =>
      any(rect(x, y, t, h), rect(x + W - t, y, t, h), rect(x, y + h / 2 - t / 2, W, t)),
  }),
  I: (h, t) => ({ w: t, shapes: (x, y, W) => rect(x, y, W, h) }),
  L: (h, t) => ({
    w: 0.58 * h,
    shapes: (x, y, W) => any(rect(x, y, t, h), rect(x, y + h - t, W, t)),
  }),
  M: (h, t) => ({
    w: 0.92 * h,
    shapes: (x, y, W) =>
      any(
        rect(x, y, t, h),
        rect(x + W - t, y, t, h),
        line(x + t / 2, y + t / 2, x + W / 2, y + 0.62 * h, t),
        line(x + W - t / 2, y + t / 2, x + W / 2, y + 0.62 * h, t),
      ),
  }),
  N: (h, t) => ({
    w: 0.76 * h,
    shapes: (x, y, W) =>
      any(
        rect(x, y, t, h),
        rect(x + W - t, y, t, h),
        line(x + t / 2, y + t / 2, x + W - t / 2, y + h - t / 2, t),
      ),
  }),
  // A full ring, the same shape as the lens in the mark.
  O: (h, t) => ({
    w: h,
    shapes: (x, y, W) => arc(x + W / 2, y + h / 2, h / 2 - t / 2, t),
  }),
  P: (h, t) => ({
    w: 0.64 * h,
    shapes: (x, y, W) => {
      const r = (h - t) / 4;
      const cx = x + W - r - t / 2;
      return any(
        rect(x, y, t, h),
        rect(x, y, cx - x, t),
        rect(x, y + h / 2 - t / 2, cx - x, t),
        arc(cx, y + r + t / 2, r, t, 270, 450),
      );
    },
  }),
  R: (h, t) => ({
    w: 0.7 * h,
    shapes: (x, y, W) => {
      const r = (h - t) / 4;
      const cx = x + W - r - t / 2 - t;
      return any(
        rect(x, y, t, h),
        rect(x, y, cx - x, t),
        rect(x, y + h / 2 - t / 2, cx - x, t),
        arc(cx, y + r + t / 2, r, t, 270, 450),
        line(cx, y + h / 2, x + W - t / 2, y + h - t / 2, t),
      );
    },
  }),
  // Two bowls that meet in the middle: the top one open at the lower left, the
  // bottom one open at the upper right.
  S: (h, t) => ({
    w: (h - t) / 2 + t,
    shapes: (x, y, W) => {
      const r = (h - t) / 4;
      const cx = x + W / 2;
      return any(
        arc(cx, y + r + t / 2, r, t, 110, 360),
        arc(cx, y + h - r - t / 2, r, t, 290, 540),
      );
    },
  }),
  T: (h, t) => ({
    w: 0.72 * h,
    shapes: (x, y, W) => any(rect(x, y, W, t), rect(x + W / 2 - t / 2, y, t, h)),
  }),
  U: (h, t) => ({
    w: 0.74 * h,
    shapes: (x, y, W) => {
      const r = W / 2 - t / 2;
      return any(
        rect(x, y, t, h - r - t / 2),
        rect(x + W - t, y, t, h - r - t / 2),
        arc(x + W / 2, y + h - r - t / 2, r, t, 0, 180),
      );
    },
  }),
  Y: (h, t) => ({
    w: 0.74 * h,
    shapes: (x, y, W) =>
      any(
        line(x + t / 2, y + t / 2, x + W / 2, y + 0.52 * h, t),
        line(x + W - t / 2, y + t / 2, x + W / 2, y + 0.52 * h, t),
        rect(x + W / 2 - t / 2, y + 0.5 * h, t, 0.5 * h),
      ),
  }),
  " ": (h) => ({ w: 0.3 * h, shapes: () => () => false }),
};

/**
 * Lays a line of uppercase text out left to right and returns its shape and its
 * total width, so the caller can centre or measure it before drawing.
 */
function text(string, x, y, height, thickness, tracking) {
  const shapes = [];
  let cursor = x;
  for (const char of string.toUpperCase()) {
    const glyph = GLYPHS[char];
    if (!glyph) throw new Error(`no glyph for ${JSON.stringify(char)}`);
    const { w, shapes: draw } = glyph(height, thickness);
    shapes.push(draw(cursor, y, w));
    cursor += w + tracking;
  }
  return { shape: any(...shapes), width: cursor - tracking - x, height };
}

/**
 * The same line, shrunk until it fits `maxWidth`. The card is laid out by hand,
 * so this only ever takes the edge off a line that would otherwise run into the
 * frame, and it keeps the stroke and the tracking in proportion when it does.
 */
function fitText(string, x, y, maxWidth, height, thickness, tracking) {
  const measured = text(string, x, y, height, thickness, tracking);
  if (measured.width <= maxWidth) return measured;
  const scale = maxWidth / measured.width;
  return text(string, x, y, height * scale, thickness * scale, tracking * scale);
}

// ---------------------------------------------------------------------- card
//
// The Belo mark, the same 512 unit geometry as src/assets/logo.svg and
// docs/index.html, placed as a square of side `size` at (x, y).
function mark(x, y, size) {
  const u = size / 512;
  const at = (n) => n * u;
  return (px, py) => {
    const lx = (px - x) / u;
    const ly = (py - y) / u;
    if (lx < 0 || ly < 0 || lx > 512 || ly > 512) return null;
    // frame, 40 wide centred on 20 and on 492
    if (lx < 40 || ly < 40 || lx > 472 || ly > 472) return INK;
    if (lx >= 302 && lx <= 434 && ly >= 302 && ly <= 434) {
      const inner = lx >= 322 && lx <= 414 && ly >= 322 && ly <= 414;
      return inner ? PINK : INK;
    }
    const d2 = (lx - 216) ** 2 + (ly - 216) ** 2;
    if (d2 >= 84 ** 2 && d2 <= 148 ** 2) return INK;
    return YELLOW;
    // `at` exists so the reader can see the unit scale; the branches above work
    // in mark units directly.
    void at;
  };
}

const MARK = { x: 92, y: 190, size: 288 };
const SHADOW = 14; // hard offset shadow, the site's own --shadow-lg, scaled up
const COL = 448; // left edge of the text column
const COL_W = WIDTH - COL - 76; // room left before the frame

const badgeText = fitText("OPEN SOURCE MIT", COL + 18, 148, COL_W - 36, 20, 6, 9);
const badge = rect(COL, 132, badgeText.width + 36, 52);
const wordmark = fitText("BELO", COL, 232, COL_W, 128, 28, 16);
const tagline = fitText("TEACH YOUR MACHINE TO SEE", COL + 4, 404, COL_W, 32, 8, 10);
const rule = rect(COL + 4, 458, 260, 8);
const subline = fitText("NO ACCOUNT NO UPLOAD NO INTERNET", COL + 4, 488, COL_W, 20, 5.5, 8);

const border = frame(24, 24, WIDTH - 48, HEIGHT - 48, 10);
const markShadow = rect(MARK.x + SHADOW, MARK.y + SHADOW, MARK.size, MARK.size);
const wordShadow = (px, py) => wordmark.shape(px - 11, py - 11);

/** Colour of the card at one point, in card pixels. */
function sample(x, y) {
  if (badge(x, y)) return badgeText.shape(x, y) ? INK : YELLOW;
  if (badgeText.shape(x, y)) return INK;
  if (wordShadow(x, y) && !wordmark.shape(x, y)) return PINK;
  if (wordmark.shape(x, y)) return INK;
  if (tagline.shape(x, y)) return INK;
  if (rule(x, y)) return PINK;
  if (subline.shape(x, y)) return INK;

  const inMark = mark(MARK.x, MARK.y, MARK.size)(x, y);
  if (inMark) return inMark;
  if (markShadow(x, y)) return INK;

  if (border(x, y)) return INK;
  return PAPER;
}

// ------------------------------------------------------------------ encoding
//
// Same PNG writer as scripts/make-icon.mjs: colour type 2, truecolor with no
// alpha, so no preview crawler has to guess what to composite the card against.

function render() {
  const px = Buffer.alloc(WIDTH * HEIGHT * 3);
  const step = 1 / SAMPLES;
  const half = step / 2;
  const n = SAMPLES * SAMPLES;
  for (let row = 0; row < HEIGHT; row++) {
    for (let col = 0; col < WIDTH; col++) {
      let r = 0;
      let g = 0;
      let b = 0;
      for (let sy = 0; sy < SAMPLES; sy++) {
        const y = row + sy * step + half;
        for (let sx = 0; sx < SAMPLES; sx++) {
          const c = sample(col + sx * step + half, y);
          r += c[0];
          g += c[1];
          b += c[2];
        }
      }
      const i = (row * WIDTH + col) * 3;
      px[i] = Math.round(r / n);
      px[i + 1] = Math.round(g / n);
      px[i + 2] = Math.round(b / n);
    }
  }
  return px;
}

const crcTable = Array.from({ length: 256 }, (_, n) => {
  let c = n;
  for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
  return c >>> 0;
});
const crc32 = (buf) => {
  let c = 0xffffffff;
  for (const b of buf) c = crcTable[(c ^ b) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
};
const chunk = (type, data) => {
  const len = Buffer.alloc(4);
  len.writeUInt32BE(data.length);
  const body = Buffer.concat([Buffer.from(type, "ascii"), data]);
  const crc = Buffer.alloc(4);
  crc.writeUInt32BE(crc32(body));
  return Buffer.concat([len, body, crc]);
};

function encodePng(px) {
  const stride = WIDTH * 3;
  const raw = Buffer.alloc(HEIGHT * (stride + 1));
  for (let y = 0; y < HEIGHT; y++) {
    raw[y * (stride + 1)] = 0; // filter type 0
    px.copy(raw, y * (stride + 1) + 1, y * stride, (y + 1) * stride);
  }
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(WIDTH, 0);
  ihdr.writeUInt32BE(HEIGHT, 4);
  ihdr[8] = 8; // bit depth
  ihdr[9] = 2; // colour type 2: truecolor, no alpha
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk("IHDR", ihdr),
    chunk("IDAT", deflateSync(raw, { level: 9 })),
    chunk("IEND", Buffer.alloc(0)),
  ]);
}

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const out = join(root, "docs", "og.png");
const png = encodePng(render());
writeFileSync(out, png);
console.log(`wrote ${out} (${WIDTH}x${HEIGHT}, ${(png.length / 1024).toFixed(0)} KB)`);
