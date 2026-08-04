const FRAME_HEADER = 16;
const FRAME_MAGIC = [0x42, 0x4c, 0x46, 0x52]; // BLFR

function sameMagic(bytes: Uint8Array): boolean {
  return FRAME_MAGIC.every((value, index) => bytes[index] === value);
}

/** Reads the optional native header without copying the JPEG body. */
export function decodeScreenFrame(packet: ArrayBuffer): {
  jpeg: ArrayBuffer;
  nativeMs: number;
  sentAtMs: number | null;
} {
  const bytes = new Uint8Array(packet);
  if (bytes.byteLength < FRAME_HEADER || !sameMagic(bytes)) {
    return { jpeg: packet, nativeMs: 0, sentAtMs: null };
  }
  const view = new DataView(packet);
  const high = view.getUint32(4, true);
  const low = view.getUint32(8, true);
  const sentMicros = high * 2 ** 32 + low;
  return {
    jpeg: packet.slice(FRAME_HEADER),
    nativeMs: view.getUint32(12, true) / 1000,
    sentAtMs: sentMicros > 0 ? sentMicros / 1000 : null,
  };
}
