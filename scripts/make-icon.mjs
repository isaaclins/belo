// Draws the Belo app mark as a 1024px PNG without any image dependency.
// The mark is a flat neobrutalist lens: yellow field, thick ink frame,
// ink aperture with a yellow pupil, and one offset pink block.
import { deflateSync } from "node:zlib";
import { writeFileSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const S = 1024;
const INK = [17, 16, 20];
const YELLOW = [255, 216, 61];
const PINK = [255, 92, 138];

const px = Buffer.alloc(S * S * 4);
const set = (x, y, [r, g, b]) => {
  const i = (y * S + x) * 4;
  px[i] = r;
  px[i + 1] = g;
  px[i + 2] = b;
  px[i + 3] = 255;
};

const inCircle = (x, y, cx, cy, r) => (x - cx) ** 2 + (y - cy) ** 2 <= r * r;

for (let y = 0; y < S; y++) {
  for (let x = 0; x < S; x++) {
    let color = YELLOW;
    // Outer ink frame.
    if (x < 72 || y < 72 || x >= S - 72 || y >= S - 72) color = INK;
    // Lens.
    if (inCircle(x, y, 436, 436, 296)) color = INK;
    if (inCircle(x, y, 436, 436, 228)) color = YELLOW;
    if (inCircle(x, y, 436, 436, 114)) color = INK;
    // Offset block, drawn last so it sits on top of the lens.
    const inBlock = x >= 616 && x <= 888 && y >= 616 && y <= 888;
    if (inBlock) color = PINK;
    if (
      inBlock &&
      (x <= 632 || x >= 872 || y <= 632 || y >= 872)
    )
      color = INK;
    set(x, y, color);
  }
}

// PNG encode: filter byte 0 per scanline, then a single IDAT.
const raw = Buffer.alloc(S * (S * 4 + 1));
for (let y = 0; y < S; y++) {
  raw[y * (S * 4 + 1)] = 0;
  px.copy(raw, y * (S * 4 + 1) + 1, y * S * 4, (y + 1) * S * 4);
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

const ihdr = Buffer.alloc(13);
ihdr.writeUInt32BE(S, 0);
ihdr.writeUInt32BE(S, 4);
ihdr[8] = 8;
ihdr[9] = 6;

const png = Buffer.concat([
  Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
  chunk("IHDR", ihdr),
  chunk("IDAT", deflateSync(raw, { level: 9 })),
  chunk("IEND", Buffer.alloc(0)),
]);

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
mkdirSync(join(root, "src-tauri"), { recursive: true });
const out = join(root, "src-tauri", "app-icon.png");
writeFileSync(out, png);
console.log(`wrote ${out} (${png.length} bytes)`);
