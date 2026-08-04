// The single seam for live screen access, the sibling of capture.ts.
//
// Nothing outside this module calls getDisplayMedia or the screen commands on
// the Rust side. Everything above it, the hook and both Use views, sees a plain
// MediaStream and cannot tell a display from a webcam.
//
// There are two backings, because there is no one API that works everywhere:
//
//   - macOS goes through ScreenCaptureKit in Rust. WKWebView exposes
//     getDisplayMedia but always denies it: wry answers WebKit's camera and
//     microphone permission callback and never the display capture one, so
//     WebKit has no picker to show. See the research note in the README.
//     Frames arrive as JPEG over a Tauri channel, get painted into a canvas,
//     and that canvas's captureStream() is the MediaStream everything else
//     consumes.
//   - Windows (WebView2) and Linux (WebKitGTK, where its build allows it) use
//     the webview's own getDisplayMedia and hand back its stream directly.
//
// A platform with neither reports isSupported() false and the Use view hides
// the Screen button rather than offering something that cannot work.

import { Channel, invoke } from "@tauri-apps/api/core";
import {
  AdaptiveRateController,
  DEFAULT_INFERENCE_INTERVAL_MS,
  type PipelineTiming,
} from "./adaptiveRate";
import { decodeScreenFrame } from "./screenPacket";
export { decodeScreenFrame } from "./screenPacket";

export interface ScreenInfo {
  /** Opaque, and stringified so a display id and a deviceId look alike. */
  id: string;
  label: string;
  width: number;
  height: number;
  /** The current display mode, rounded to a usable frame rate. */
  refreshRate: number;
}

export type ScreenPermission = "granted" | "denied" | "unsupported";

export interface ScreenMetrics extends PipelineTiming {
  fps: number;
  totalMs: number;
  budgetMs: number;
  decision: "hold" | "slow" | "fast" | "drop";
}

export interface ScreenStartOptions {
  /** Zero means Auto: start at the display's actual refresh rate. */
  requestedFps?: number;
  /** The long edge of the native frame. */
  maxEdge?: number;
  /** JPEG quality, 1 to 100. */
  jpegQuality?: number;
  /** Receives timing only, never pixels. */
  onMetrics?: (metrics: ScreenMetrics) => void;
}

export interface ScreenSource {
  /** Cheap and synchronous: whether this build has any screen path at all. */
  isSupported(): boolean;
  /**
   * The platform's own answer, which the native path can only give from Rust:
   * ScreenCaptureKit arrived in macOS 12.3 and Belo runs on older ones.
   */
  supported(): Promise<boolean>;
  /** Whether the platform can say which displays exist before capturing one. */
  canList(): boolean;
  list(): Promise<ScreenInfo[]>;
  permission(): Promise<ScreenPermission>;
  /** Opens the platform's privacy settings. Silently does nothing where there are none. */
  openSettings(): Promise<void>;
  start(displayId?: string | null, options?: ScreenStartOptions): Promise<MediaStream>;
  stop(stream: MediaStream | null): void;
  activeDisplayId(stream: MediaStream | null): string | null;
  /** Adds the latest inference cost to the adaptive controller. */
  reportInference(stream: MediaStream | null, milliseconds: number): void;
  /** Rebuilds the native filter after Belo creates or shows another window. */
  refreshFilter(): Promise<void>;
}

/** Fallback for browser screen sharing when the display mode is not enumerable. */
export const SCREEN_FPS = 60;
const DEFAULT_MAX_EDGE = 1120;
const DEFAULT_JPEG_QUALITY = 55;

/** A frame that never arrives is a hang, so waiting for the first one has an end. */
const FIRST_FRAME_TIMEOUT_MS = 6000;

/** Thrown so the hook can tell a refusal apart from a fault. */
function denied(message: string): Error {
  const error = new Error(message);
  error.name = "NotAllowedError";
  return error;
}

/**
 * A platform that pushes encoded frames rather than handing over a MediaStream.
 * macOS implements this over ScreenCaptureKit; the headless check implements it
 * over a synthetic desktop, which is how the whole path above the pipe gets
 * tested without recording anybody's screen.
 */
export interface ScreenPipe {
  isSupported(): boolean;
  supported(): Promise<boolean>;
  list(): Promise<ScreenInfo[]>;
  permission(): Promise<ScreenPermission>;
  openSettings(): Promise<void>;
  /** Opens the feed. Every frame is a JPEG or a native timing packet. */
  open(
    displayId: string | null,
    onFrame: (packet: ArrayBuffer) => void,
    options?: ScreenStartOptions,
  ): Promise<OpenFeed>;
}

export interface OpenFeed {
  /** The display that actually opened, which can differ from the request. */
  id: string;
  width: number;
  height: number;
  /** Actual display mode rate, or the platform fallback. */
  refreshRate?: number;
  /** Rate the native stream is currently asking for. */
  fps?: number;
  close(): void | Promise<void>;
  setRate?(fps: number): void | Promise<void>;
}

interface PipedFeed {
  id: string;
  close: () => void;
  reportInference: (milliseconds: number) => void;
}

/** Teardown and feedback for a running piped stream, kept off the MediaStream itself. */
const pipedFeeds = new WeakMap<MediaStream, PipedFeed>();


export function createPipedScreenSource(pipe: ScreenPipe): ScreenSource {
  return {
    isSupported: () => pipe.isSupported(),
    supported: () => pipe.supported(),
    canList: () => true,
    list: () => pipe.list(),
    permission: () => pipe.permission(),
    openSettings: () => pipe.openSettings(),
    refreshFilter: async () => undefined,

    async start(displayId, options = {}) {
      if (await pipe.permission().then((state) => state === "denied")) {
        throw denied("Belo is not allowed to record the screen");
      }
      const canvas = document.createElement("canvas");
      const context = canvas.getContext("2d", { alpha: false });
      if (!context) throw new Error("This webview cannot draw the screen feed");

      let closed = false;
      let painting = false;
      let painted = 0;
      let dropped = 0;
      let announce: (() => void) | null = null;
      let inferenceMs = 0;
      let controller: AdaptiveRateController | null = null;
      let feed: OpenFeed | null = null;
      let lastRequestedRate = 0;

      const paint = async (packet: ArrayBuffer) => {
        if (closed || painting) {
          dropped += 1;
          return;
        }
        painting = true;
        const decoded = decodeScreenFrame(packet);
        const transferMs =
          decoded.sentAtMs === null ? 0 : Math.max(0, Date.now() - decoded.sentAtMs);
        const decodeStarted = performance.now();
        try {
          const bitmap = await createImageBitmap(
            new Blob([decoded.jpeg], { type: "image/jpeg" }),
          );
          const decodeMs = performance.now() - decodeStarted;
          if (closed) {
            bitmap.close();
            return;
          }
          if (canvas.width !== bitmap.width || canvas.height !== bitmap.height) {
            canvas.width = bitmap.width;
            canvas.height = bitmap.height;
          }
          const paintStarted = performance.now();
          context.drawImage(bitmap, 0, 0);
          const paintMs = performance.now() - paintStarted;
          bitmap.close();
          painted += 1;

          const timing: PipelineTiming = {
            nativeMs: decoded.nativeMs,
            transferMs,
            decodeMs,
            paintMs,
            inferenceMs,
            dropped: dropped > 0,
          };
          dropped = 0;
          const decision = controller?.observe(timing);
          if (decision && decision.changed && decision.rate !== lastRequestedRate) {
            lastRequestedRate = decision.rate;
            void Promise.resolve(feed?.setRate?.(decision.rate)).catch(() => undefined);
          }
          if (decision) {
            options.onMetrics?.({
              ...timing,
              fps: decision.rate,
              totalMs: decision.totalMs,
              budgetMs: decision.budgetMs,
              decision: decision.reason,
            });
          }
          announce?.();
        } finally {
          painting = false;
        }
      };

      feed = await pipe.open(displayId ?? null, (packet) => void paint(packet), options);
      const refreshRate = Math.max(1, Math.round(feed.refreshRate ?? feed.fps ?? SCREEN_FPS));
      const requested = options.requestedFps ?? 0;
      const maxRate = requested > 0 ? Math.min(refreshRate, requested) : refreshRate;
      const initialRate = Math.min(maxRate, Math.max(1, Math.round(feed.fps ?? maxRate)));
      controller = new AdaptiveRateController({
        maxRate,
        initialRate,
        inferenceIntervalMs: DEFAULT_INFERENCE_INTERVAL_MS,
      });
      lastRequestedRate = initialRate;
      canvas.width = feed.width;
      canvas.height = feed.height;
      const stream = canvas.captureStream(Math.max(60, refreshRate));
      const close = () => {
        if (closed) return;
        closed = true;
        void feed?.close();
      };
      pipedFeeds.set(stream, {
        id: feed.id,
        close,
        reportInference(milliseconds) {
          const next = Math.max(0, milliseconds);
          inferenceMs = inferenceMs === 0 ? next : inferenceMs * 0.75 + next * 0.25;
        },
      });

      try {
        // A stream whose first frame never lands would show an empty stage with
        // no explanation, so the wait ends in a message instead.
        await new Promise<void>((resolve, reject) => {
          if (painted > 0) return resolve();
          const timer = window.setTimeout(() => {
            announce = null;
            reject(new Error("The screen feed started but sent no picture"));
          }, FIRST_FRAME_TIMEOUT_MS);
          announce = () => {
            window.clearTimeout(timer);
            announce = null;
            resolve();
          };
        });
      } catch (err) {
        close();
        stream.getTracks().forEach((track) => track.stop());
        throw err;
      }
      return stream;
    },

    stop(stream) {
      if (!stream) return;
      pipedFeeds.get(stream)?.close();
      pipedFeeds.delete(stream);
      stream.getTracks().forEach((track) => track.stop());
    },

    activeDisplayId(stream) {
      return stream ? (pipedFeeds.get(stream)?.id ?? null) : null;
    },

    reportInference(stream, milliseconds) {
      if (!stream) return;
      pipedFeeds.get(stream)?.reportInference(milliseconds);
    },
  };
}

/** macOS: ScreenCaptureKit on the Rust side, one JPEG per frame over a channel. */
const nativePipe: ScreenPipe = {
  isSupported() {
    return typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
  },
  supported: () => invoke<boolean>("screen_supported"),
  list: () => invoke<ScreenInfo[]>("screen_displays"),
  permission: () => invoke<ScreenPermission>("screen_permission"),
  openSettings: async () => {
    await invoke("screen_open_settings");
  },
  async open(displayId, onFrame, options = {}) {
    const channel = new Channel<ArrayBuffer>();
    channel.onmessage = onFrame;
    const info = await invoke<{
      id: string;
      width: number;
      height: number;
      refreshRate: number;
      fps: number;
    }>("screen_start", {
      displayId,
      // Zero is the native Auto value. Explicit presets stay below the display
      // mode and the controller can still step them down when needed.
      fps: options.requestedFps ?? 0,
      maxEdge: options.maxEdge ?? DEFAULT_MAX_EDGE,
      jpegQuality: options.jpegQuality ?? DEFAULT_JPEG_QUALITY,
      onFrame: channel,
    });
    return {
      ...info,
      close: () => invoke("screen_stop"),
      setRate: (fps: number) => invoke("screen_set_rate", { fps }),
    };
  },
};

/** Windows and Linux: whatever the webview's own getDisplayMedia does. */
const webScreen: ScreenSource = {
  isSupported() {
    return typeof navigator !== "undefined" && Boolean(navigator.mediaDevices?.getDisplayMedia);
  },
  supported: async () => true,
  // The browser API deliberately never says what is on the machine; the picker
  // it shows is the enumeration. Full display selection is the platform's job.
  canList: () => false,
  list: async () => [],
  permission: async () => "granted",
  openSettings: async () => undefined,
  async start(_displayId, options = {}) {
    return navigator.mediaDevices.getDisplayMedia({
      video: { frameRate: { ideal: options.requestedFps || SCREEN_FPS } },
      audio: false,
    });
  },
  stop(stream) {
    stream?.getTracks().forEach((track) => track.stop());
  },
  activeDisplayId(stream) {
    return stream?.getVideoTracks()[0]?.getSettings?.().deviceId ?? null;
  },
  reportInference: () => undefined,
  refreshFilter: async () => undefined,
};

const unsupported: ScreenSource = {
  isSupported: () => false,
  supported: async () => false,
  canList: () => false,
  list: async () => [],
  permission: async () => "unsupported",
  openSettings: async () => undefined,
  async start() {
    throw new Error("This platform cannot share a screen with Belo");
  },
  stop: () => undefined,
  activeDisplayId: () => null,
  reportInference: () => undefined,
  refreshFilter: async () => undefined,
};

const IS_MAC =
  typeof navigator !== "undefined" &&
  /Mac/i.test(navigator.platform || navigator.userAgent) &&
  !(navigator.maxTouchPoints > 1);

function platformSource(): ScreenSource {
  const native = {
    ...createPipedScreenSource(nativePipe),
    refreshFilter: async () => {
      await invoke("screen_refresh_filter");
    },
  } satisfies ScreenSource;
  if (IS_MAC && native.isSupported()) return native;
  if (webScreen.isSupported()) return webScreen;
  return unsupported;
}

let active: ScreenSource | null = null;

/**
 * Swaps the backing, the way capture.ts expects a mobile port to swap its
 * camera. The headless screen check installs a synthetic desktop through this,
 * and a future platform port replaces the whole object rather than editing the
 * views.
 */
export function setScreenSource(next: ScreenSource): void {
  active = next;
}

function current(): ScreenSource {
  if (!active) active = platformSource();
  return active;
}

export const screen: ScreenSource = {
  isSupported: () => current().isSupported(),
  supported: () => current().supported(),
  canList: () => current().canList(),
  list: () => current().list(),
  permission: () => current().permission(),
  openSettings: () => current().openSettings(),
  start: (displayId, options) => current().start(displayId, options),
  stop: (stream) => current().stop(stream),
  activeDisplayId: (stream) => current().activeDisplayId(stream),
  reportInference: (stream, milliseconds) => current().reportInference(stream, milliseconds),
  refreshFilter: () => current().refreshFilter(),
};
