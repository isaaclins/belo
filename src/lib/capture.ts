// The single seam for live camera access.
//
// Nothing outside this module calls getUserMedia or enumerateDevices. A mobile
// port that needs a native camera plugin implements CameraSource and swaps the
// export at the bottom; useWebcam and the views stay untouched.

export type CameraFacing = "front" | "back";

export interface CameraInfo {
  deviceId: string;
  label: string;
  /** Best guess only. Null when the platform gives nothing to go on. */
  facing: CameraFacing | null;
}

export interface CameraSource {
  isSupported(): boolean;
  /** Opens a stream, optionally for a specific device. Falls back to the default camera. */
  start(deviceId?: string | null): Promise<MediaStream>;
  stop(stream: MediaStream | null): void;
  /** Cameras the platform exposes. Labels usually need a granted permission first. */
  list(): Promise<CameraInfo[]>;
  /** The device a running stream actually opened, which can differ from the request. */
  activeDeviceId(stream: MediaStream | null): string | null;
  /** Hot-plug notifications. Returns an unsubscribe. */
  onDevicesChanged(handler: () => void): () => void;
}

const IDEAL_SIZE = { width: { ideal: 1280 }, height: { ideal: 720 } };

/**
 * deviceId to facing, learned from device labels and from the settings of
 * streams we opened. Phones hand out deviceIds that go stale between sessions,
 * so when an exact id is refused we retry on facingMode instead, which stays
 * meaningful across restarts.
 */
const facingById = new Map<string, CameraFacing>();

function guessFacing(label: string): CameraFacing | null {
  const text = label.toLowerCase();
  if (/(^|\W)(front|user|selfie|face\s?time|facetime)/.test(text)) return "front";
  if (/(^|\W)(back|rear|environment|world)/.test(text)) return "back";
  return null;
}

function normalizeFacing(mode: string | undefined): CameraFacing | null {
  if (mode === "user") return "front";
  if (mode === "environment") return "back";
  return null;
}

function facingMode(facing: CameraFacing): "user" | "environment" {
  return facing === "front" ? "user" : "environment";
}

/** True for "that device is gone or cannot do that", false for denial and real faults. */
function isConstraintMiss(err: unknown): boolean {
  const name = err instanceof Error ? err.name : "";
  return (
    name === "OverconstrainedError" ||
    name === "ConstraintNotSatisfiedError" ||
    name === "NotFoundError" ||
    name === "DevicesNotFoundError"
  );
}

/** Records what a freshly opened stream turned out to be, for later retries. */
function noteStream(stream: MediaStream): MediaStream {
  const settings = stream.getVideoTracks()[0]?.getSettings?.();
  const facing = normalizeFacing(settings?.facingMode);
  if (settings?.deviceId && facing) facingById.set(settings.deviceId, facing);
  return stream;
}

const webCamera: CameraSource = {
  isSupported() {
    return typeof navigator !== "undefined" && Boolean(navigator.mediaDevices?.getUserMedia);
  },

  async start(deviceId) {
    if (!this.isSupported()) throw new Error("This webview does not expose a camera API");
    const open = (video: MediaTrackConstraints) =>
      navigator.mediaDevices.getUserMedia({ video, audio: false }).then(noteStream);

    if (deviceId) {
      try {
        return await open({ deviceId: { exact: deviceId }, ...IDEAL_SIZE });
      } catch (err) {
        // A denial or a broken device has to surface, only a missing device retries.
        if (!isConstraintMiss(err)) throw err;
      }
      const facing = facingById.get(deviceId);
      if (facing) {
        try {
          return await open({ facingMode: { exact: facingMode(facing) }, ...IDEAL_SIZE });
        } catch (err) {
          if (!isConstraintMiss(err)) throw err;
        }
      }
    }
    return open(IDEAL_SIZE);
  },

  stop(stream) {
    stream?.getTracks().forEach((track) => track.stop());
  },

  async list() {
    if (!navigator.mediaDevices?.enumerateDevices) return [];
    const devices = await navigator.mediaDevices.enumerateDevices();
    return devices
      .filter((device) => device.kind === "videoinput")
      .map((device, i) => {
        const label = device.label?.trim() ?? "";
        const facing = guessFacing(label) ?? facingById.get(device.deviceId) ?? null;
        if (facing) facingById.set(device.deviceId, facing);
        // Labels stay empty until a camera permission is granted, and some
        // platforms never fill them in, so there is always a plain name.
        return { deviceId: device.deviceId, label: label || `Camera ${i + 1}`, facing };
      });
  },

  activeDeviceId(stream) {
    return stream?.getVideoTracks()[0]?.getSettings?.().deviceId ?? null;
  },

  onDevicesChanged(handler) {
    const media = navigator.mediaDevices;
    if (!media?.addEventListener) return () => undefined;
    media.addEventListener("devicechange", handler);
    return () => media.removeEventListener("devicechange", handler);
  },
};

export const camera: CameraSource = webCamera;
