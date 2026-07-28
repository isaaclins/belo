// Rasterizes the canonical Belo mark for the platform icon sets.
//
//   node scripts/make-icon.mjs master   src-tauri/app-icon.png, the 1024 master
//   node scripts/make-icon.mjs ios      the iOS AppIcon.appiconset images
//   node scripts/make-icon.mjs          both
//
// Platform icons drop the ink frame. The in-app mark keeps it, because there it
// sits on a page and needs an edge; a launcher icon is already framed by the
// system, and on iOS the rounded mask cut the black band into muddy dark
// corners. So the icons are drawn frameless: yellow edge to edge, with the ring
// and the chip scaled up slightly to fill the space the frame used to take.
//
// `npx tauri icon src-tauri/app-icon.png` turns the master into the desktop and
// Windows sets, but it writes iOS images with an alpha channel and Apple
// rejects those, so the iOS set is drawn here instead: exact sizes, no
// resampling, no alpha. Run the ios step after `tauri icon`, which is what
// `npm run icons` does.
//
// The numbers below are the same 512 unit grid as src/assets/logo.svg and
// src/components/BeloMark.tsx. There is no SVG rasterizer dependency on
// purpose: the mark is a handful of flat shapes, so drawing them directly keeps
// the repo dependency free and keeps the raster in step with the vector.
import { deflateSync } from "node:zlib";
import { writeFileSync, mkdirSync, readdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const GRID = 512; // design grid the geometry is authored on
const SAMPLES = 4; // supersampling per axis, so 16 samples per pixel

const INK = [17, 16, 20]; // --ink
const YELLOW = [255, 216, 61]; // --accent
const PINK = [255, 92, 138]; // --accent-3

// Lens: circle at 216,216 r 116 stroked 64 centered, so the annulus is 84..148.
const LENS = { cx: 216, cy: 216, inner: 84, outer: 148 };
// Chip: rect 302,302,132,132 stroked 20 centered, so outer 292..444, inner 312..424.
const CHIP = { outer: [292, 444], inner: [312, 424] };

// Frameless growth. The framed mark's contents span 68..444, so 68 of margin on
// every side. Scaling about the grid centre by this factor leaves about 53, a
// tenth of the grid, which reads full without pushing the chip's outer corner
// into the iOS rounded mask.
const ICON_SCALE = 1.08;

/**
 * Color of the icon at one point in design grid units: the same ring and chip
 * as the in-app mark, scaled about the centre, over a yellow field that runs
 * edge to edge with no ink frame.
 */
function sampleIcon(x, y) {
  const c = GRID / 2;
  return field(c + (x - c) / ICON_SCALE, c + (y - c) / ICON_SCALE);
}

/** Ring and chip over the yellow field, in design grid units. */
function field(x, y) {
  const [co, ci] = [CHIP.outer, CHIP.inner];
  if (x >= co[0] && x <= co[1] && y >= co[0] && y <= co[1]) {
    const insideChip = x >= ci[0] && x <= ci[1] && y >= ci[0] && y <= ci[1];
    return insideChip ? PINK : INK;
  }

  const d2 = (x - LENS.cx) ** 2 + (y - LENS.cy) ** 2;
  if (d2 >= LENS.inner ** 2 && d2 <= LENS.outer ** 2) return INK;

  return YELLOW;
}

/** Supersampled RGB buffer of the mark at `size` pixels square. */
function render(size) {
  const px = Buffer.alloc(size * size * 3);
  const step = GRID / size / SAMPLES;
  const half = step / 2;
  const n = SAMPLES * SAMPLES;

  for (let row = 0; row < size; row++) {
    for (let col = 0; col < size; col++) {
      let r = 0;
      let g = 0;
      let b = 0;
      for (let sy = 0; sy < SAMPLES; sy++) {
        const y = (row * GRID) / size + sy * step + half;
        for (let sx = 0; sx < SAMPLES; sx++) {
          const x = (col * GRID) / size + sx * step + half;
          const c = sampleIcon(x, y);
          r += c[0];
          g += c[1];
          b += c[2];
        }
      }
      const i = (row * size + col) * 3;
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

/**
 * Encodes an RGB buffer as PNG color type 2: truecolor with no alpha channel.
 * Every pixel of the mark is opaque, and iOS refuses app icons that carry
 * transparency, so there is nothing to lose and a rejection to avoid.
 */
function encodePng(px, size) {
  const stride = size * 3;
  const raw = Buffer.alloc(size * (stride + 1));
  for (let y = 0; y < size; y++) {
    raw[y * (stride + 1)] = 0; // filter type 0
    px.copy(raw, y * (stride + 1) + 1, y * stride, (y + 1) * stride);
  }

  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(size, 0);
  ihdr.writeUInt32BE(size, 4);
  ihdr[8] = 8; // bit depth
  ihdr[9] = 2; // color type 2: truecolor, no alpha

  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk("IHDR", ihdr),
    chunk("IDAT", deflateSync(raw, { level: 9 })),
    chunk("IEND", Buffer.alloc(0)),
  ]);
}

function write(path, size) {
  const png = encodePng(render(size), size);
  writeFileSync(path, png);
  console.log(`wrote ${path} (${size}px, ${png.length} bytes)`);
}

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const step = process.argv[2] ?? "both";

if (step === "master" || step === "both") {
  mkdirSync(join(root, "src-tauri"), { recursive: true });
  write(join(root, "src-tauri", "app-icon.png"), 1024);
}

if (step === "ios" || step === "both") {
  const set = join(root, "src-tauri", "gen", "apple", "Assets.xcassets", "AppIcon.appiconset");
  // Sizes come from the filenames Xcode's catalog already references, so the
  // set stays in step with Contents.json instead of a second hardcoded list.
  const named = /^AppIcon-([\d.]+)(?:x[\d.]+)?@(\d)x(?:-\d+)?\.png$/;
  let count = 0;
  for (const file of readdirSync(set).sort()) {
    const match = named.exec(file);
    if (!match) continue;
    write(join(set, file), Math.round(Number(match[1]) * Number(match[2])));
    count += 1;
  }
  if (count === 0) throw new Error(`no AppIcon images found in ${set}`);
}
