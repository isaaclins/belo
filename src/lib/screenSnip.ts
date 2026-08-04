import { invoke } from "@tauri-apps/api/core";
import { getCurrentWindow } from "@tauri-apps/api/window";
import {
  register as registerShortcut,
  unregister as unregisterShortcut,
} from "@tauri-apps/plugin-global-shortcut";
import {
  isPermissionGranted,
  requestPermission,
  sendNotification,
} from "@tauri-apps/plugin-notification";

/** The only global shortcut Belo registers, and only while a project is open. */
export const SCREEN_SNIP_SHORTCUT = "CommandOrControl+Shift+B";
export const SCREEN_SNIP_LABEL = "⌘⇧B";

export interface ScreenSnipPayload {
  base64: string;
  mime: "image/png";
}

export interface ScreenSnipImportResult {
  projectName: string;
  count: number;
}

export interface ScreenSnipTarget {
  /** The store can turn false while a close is saving or after it completes. */
  isOpen: () => boolean;
  /** Imports through the store's normal image intake path. Null means no-op. */
  add: (payload: ScreenSnipPayload) => Promise<ScreenSnipImportResult | null>;
}

export interface ScreenSnipStatus {
  supported: boolean;
  active: boolean;
}

export interface ScreenSnipCallbacks {
  onStatus?: (status: ScreenSnipStatus) => void;
  onForegroundImport?: (result: ScreenSnipImportResult) => void;
  onError?: (message: string) => void;
}

/** Runtime seam used by the controller and replaced by the headless check. */
export interface ScreenShortcutEvent {
  state: "Pressed" | "Released";
}

export interface ScreenSnipRuntime {
  supported: () => Promise<boolean>;
  register: (handler: (event: ScreenShortcutEvent) => void) => Promise<void>;
  unregister: () => Promise<void>;
  capture: () => Promise<ScreenSnipPayload | null>;
  backgrounded: () => Promise<boolean>;
  notify: (title: string, body: string) => Promise<void>;
}

function inTauri(): boolean {
  return typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
}

function errorText(error: unknown): string {
  if (error instanceof Error) return error.message;
  return typeof error === "string" ? error : "Something went wrong";
}

const browserBackgrounded = (): boolean =>
  typeof document !== "undefined" &&
  (document.visibilityState !== "visible" || !document.hasFocus());

const nativeRuntime: ScreenSnipRuntime = {
  async supported() {
    if (!inTauri()) return false;
    return await invoke<boolean>("snip_supported");
  },

  register: (handler) => registerShortcut(SCREEN_SNIP_SHORTCUT, handler),

  unregister: () => unregisterShortcut(SCREEN_SNIP_SHORTCUT),

  async capture() {
    const base64 = await invoke<string | null>("snip_region");
    return base64 ? { base64, mime: "image/png" } : null;
  },

  async backgrounded() {
    if (!inTauri()) return browserBackgrounded();
    try {
      return browserBackgrounded() || !(await getCurrentWindow().isFocused());
    } catch {
      // A webview that is going away should not make a native capture fail.
      return browserBackgrounded();
    }
  },

  async notify(title, body) {
    let granted = await isPermissionGranted();
    if (!granted) granted = (await requestPermission()) === "granted";
    if (!granted) throw new Error("macOS notifications are not permitted");
    sendNotification({ title, body });
  },
};

/**
 * Owns the global shortcut lifecycle. A generation token makes every callback
 * from an old registration harmless after a project closes, including a
 * capture that returns from Apple's picker after the close has begun.
 */
export class ScreenSnipController {
  private readonly runtime: ScreenSnipRuntime;
  private generation = 0;
  private target: ScreenSnipTarget | null = null;
  private callbacks: ScreenSnipCallbacks = {};
  private registering: Promise<void> | null = null;
  private registered = false;
  private busy = false;

  constructor(runtime: ScreenSnipRuntime = nativeRuntime) {
    this.runtime = runtime;
  }

  private status(supported: boolean, active: boolean): void {
    this.callbacks.onStatus?.({ supported, active });
  }

  private current(token: number, target: ScreenSnipTarget | null = this.target): boolean {
    return token === this.generation && target !== null && target === this.target && target.isOpen();
  }

  /** Register the shortcut for one project, replacing any older registration. */
  async open(target: ScreenSnipTarget, callbacks: ScreenSnipCallbacks = {}): Promise<boolean> {
    await this.close();
    const token = ++this.generation;
    this.target = target;
    this.callbacks = callbacks;
    this.status(false, false);

    let supported = false;
    try {
      supported = await this.runtime.supported();
    } catch (error) {
      if (token === this.generation) this.callbacks.onError?.(errorText(error));
      return false;
    }
    if (!supported || !this.current(token)) {
      if (token === this.generation) this.status(false, false);
      return false;
    }

    const handler = (event: ScreenShortcutEvent) => {
      if (event.state === "Pressed") void this.trigger(token);
    };
    const registration = this.runtime.register(handler);
    this.registering = registration;
    try {
      await registration;
    } catch (error) {
      if (this.registering === registration) this.registering = null;
      if (token === this.generation) {
        this.target = null;
        this.callbacks.onError?.(errorText(error));
        this.status(true, false);
      }
      return false;
    }
    const ownsRegistration = this.registering === registration;
    if (ownsRegistration) this.registering = null;

    if (!this.current(token)) {
      // close() invalidated this generation while the native registration was
      // in flight. Its close path owns cleanup, so only unregister here when
      // the target became invalid without a close call.
      if (ownsRegistration) {
        try {
          await this.runtime.unregister();
        } catch {
          // The late cleanup must not revive the project.
        }
      }
      return false;
    }

    this.registered = true;
    this.status(true, true);
    return true;
  }

  /** Unregister and invalidate the current project before awaiting any work. */
  async close(): Promise<void> {
    this.generation += 1;
    this.target = null;
    this.busy = false;
    const registration = this.registering;
    this.registering = null;
    const wasRegistered = this.registered;
    this.registered = false;
    this.status(false, false);

    if (registration) {
      try {
        await registration;
      } catch {
        // Registration failed, so there is nothing to unregister.
      }
    }
    if (wasRegistered || registration) {
      try {
        await this.runtime.unregister();
      } catch (error) {
        this.callbacks.onError?.(`Could not release ${SCREEN_SNIP_LABEL}: ${errorText(error)}`);
      }
    }
  }

  /** Test and native callback seam. A second press while the picker is open is ignored. */
  private async trigger(token: number): Promise<void> {
    if (this.busy || !this.current(token)) return;
    const target = this.target;
    if (!target) return;
    this.busy = true;

    try {
      const backgrounded = await this.runtime.backgrounded();
      if (!this.current(token, target)) return;
      const payload = await this.runtime.capture();
      if (!payload || !this.current(token, target)) return;
      const result = await target.add(payload);
      if (!result || !this.current(token, target)) return;

      const imageWord = result.count === 1 ? "image" : "images";
      if (backgrounded) {
        await this.runtime.notify(
          "Screen capture added",
          `${result.projectName}: ${result.count} ${imageWord} in the project`,
        );
      } else {
        this.callbacks.onForegroundImport?.(result);
      }
    } catch (error) {
      if (this.current(token, target)) this.callbacks.onError?.(errorText(error));
    } finally {
      if (token === this.generation) this.busy = false;
    }
  }
}

export const screenSnip = new ScreenSnipController();

export function openScreenSnip(
  target: ScreenSnipTarget,
  callbacks: ScreenSnipCallbacks = {},
): Promise<boolean> {
  return screenSnip.open(target, callbacks);
}

export function closeScreenSnip(): Promise<void> {
  return screenSnip.close();
}
