// The single seam for live camera access.
//
// Nothing outside this module calls getUserMedia or enumerateDevices. A mobile
// port that needs a native camera plugin implements CameraSource and swaps the
// export at the bottom; useWebcam and the views stay untouched.

export interface CameraInfo {
  deviceId: string;
  label: string;
}

export interface CameraSource {
  isSupported(): boolean;
  /** Opens a stream, optionally for a specific device. */
  start(deviceId?: string | null): Promise<MediaStream>;
  stop(stream: MediaStream | null): void;
  /** Cameras the platform exposes. Labels usually need a granted permission first. */
  list(): Promise<CameraInfo[]>;
}

const IDEAL_SIZE = { width: { ideal: 1280 }, height: { ideal: 720 } };

const webCamera: CameraSource = {
  isSupported() {
    return typeof navigator !== "undefined" && Boolean(navigator.mediaDevices?.getUserMedia);
  },

  async start(deviceId) {
    if (!this.isSupported()) throw new Error("This webview does not expose a camera API");
    return navigator.mediaDevices.getUserMedia({
      video: deviceId ? { deviceId: { exact: deviceId }, ...IDEAL_SIZE } : IDEAL_SIZE,
      audio: false,
    });
  },

  stop(stream) {
    stream?.getTracks().forEach((track) => track.stop());
  },

  async list() {
    if (!navigator.mediaDevices?.enumerateDevices) return [];
    const devices = await navigator.mediaDevices.enumerateDevices();
    return devices
      .filter((device) => device.kind === "videoinput")
      .map((device, i) => ({
        deviceId: device.deviceId,
        label: device.label || `Camera ${i + 1}`,
      }));
  },
};

export const camera: CameraSource = webCamera;
