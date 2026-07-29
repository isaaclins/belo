// A minimal zip writer.
//
// Belo's model export is one file the user can hand to somebody else, and a zip
// is the only archive every desktop opens with a double click. The bytes that go
// into it only exist on the frontend side: the model artifacts come back through
// the storage seam as strings, the bundled MobileNet is fetched from the app's
// own assets, and the README and the example are string templates. Building the
// archive here therefore needs no new Rust dependency and behaves identically on
// mobile; the Rust side only writes the finished bytes to the chosen path.
//
// Entries are deflated with the platform's own CompressionStream when there is
// one, which every WebView2, WKWebView and WebKitGTK Belo supports has, and
// stored uncompressed when there is not. Both are ordinary zip entries, so the
// reader never has to care which one it got.
//
// No zip64: the whole archive is the extractor plus a few hundred kilobytes of
// JSON, three orders of magnitude below the 4 GB where zip64 starts to matter.

export interface ZipEntry {
  name: string;
  /** Text is encoded as UTF-8, bytes are written as they are. */
  data: string | Uint8Array;
}

const CRC_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let n = 0; n < 256; n += 1) {
    let c = n;
    for (let k = 0; k < 8; k += 1) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    table[n] = c >>> 0;
  }
  return table;
})();

export function crc32(bytes: Uint8Array): number {
  let c = 0xffffffff;
  for (let i = 0; i < bytes.length; i += 1) c = CRC_TABLE[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

/** Raw deflate through the platform, or null when this runtime has none. */
async function deflateRaw(bytes: Uint8Array): Promise<Uint8Array | null> {
  if (typeof CompressionStream === "undefined") return null;
  try {
    const stream = new Blob([bytes as BlobPart]).stream().pipeThrough(
      new CompressionStream("deflate-raw"),
    );
    return new Uint8Array(await new Response(stream).arrayBuffer());
  } catch {
    // A runtime that knows CompressionStream but not deflate-raw stores instead.
    return null;
  }
}

interface Staged {
  name: Uint8Array;
  crc: number;
  size: number;
  compressed: Uint8Array;
  method: number;
}

function u16(value: number): number[] {
  return [value & 0xff, (value >>> 8) & 0xff];
}

function u32(value: number): number[] {
  return [value & 0xff, (value >>> 8) & 0xff, (value >>> 16) & 0xff, (value >>> 24) & 0xff];
}

/**
 * Packs entries into one zip archive.
 *
 * Timestamps are written as a fixed 1980-01-01, the zero of the DOS date field
 * this header format carries, so exporting the same model twice produces the
 * same bytes and a diff of two exports is about the model rather than the clock.
 */
export async function makeZip(entries: ZipEntry[]): Promise<Uint8Array> {
  const encoder = new TextEncoder();
  const staged: Staged[] = [];

  for (const entry of entries) {
    const bytes = typeof entry.data === "string" ? encoder.encode(entry.data) : entry.data;
    const deflated = bytes.length > 64 ? await deflateRaw(bytes) : null;
    const useDeflate = deflated !== null && deflated.length < bytes.length;
    staged.push({
      name: encoder.encode(entry.name),
      crc: crc32(bytes),
      size: bytes.length,
      compressed: useDeflate ? deflated : bytes,
      method: useDeflate ? 8 : 0,
    });
  }

  const chunks: Uint8Array[] = [];
  const offsets: number[] = [];
  let at = 0;
  const push = (bytes: Uint8Array) => {
    chunks.push(bytes);
    at += bytes.length;
  };

  for (const item of staged) {
    offsets.push(at);
    push(
      new Uint8Array([
        ...u32(0x04034b50),
        ...u16(20), // version needed: 2.0, deflate
        ...u16(0x0800), // UTF-8 names
        ...u16(item.method),
        ...u16(0), // time
        ...u16(0x0021), // date: 1980-01-01
        ...u32(item.crc),
        ...u32(item.compressed.length),
        ...u32(item.size),
        ...u16(item.name.length),
        ...u16(0),
      ]),
    );
    push(item.name);
    push(item.compressed);
  }

  const centralAt = at;
  staged.forEach((item, index) => {
    push(
      new Uint8Array([
        ...u32(0x02014b50),
        ...u16(20), // made by
        ...u16(20),
        ...u16(0x0800),
        ...u16(item.method),
        ...u16(0),
        ...u16(0x0021),
        ...u32(item.crc),
        ...u32(item.compressed.length),
        ...u32(item.size),
        ...u16(item.name.length),
        ...u16(0),
        ...u16(0), // comment
        ...u16(0), // disk
        ...u16(0), // internal attributes
        ...u32(0), // external attributes
        ...u32(offsets[index]),
      ]),
    );
    push(item.name);
  });

  push(
    new Uint8Array([
      ...u32(0x06054b50),
      ...u16(0),
      ...u16(0),
      ...u16(staged.length),
      ...u16(staged.length),
      ...u32(at - centralAt),
      ...u32(centralAt),
      ...u16(0),
    ]),
  );

  const out = new Uint8Array(at);
  let cursor = 0;
  for (const chunk of chunks) {
    out.set(chunk, cursor);
    cursor += chunk.length;
  }
  return out;
}

export function bytesToBase64(bytes: Uint8Array): string {
  let binary = "";
  const chunkSize = 0x8000;
  for (let offset = 0; offset < bytes.length; offset += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(offset, offset + chunkSize));
  }
  return btoa(binary);
}
