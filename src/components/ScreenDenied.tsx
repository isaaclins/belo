// What the stage shows when the system will not let Belo see the screen.
//
// macOS asks once, and after a no it never asks again: the app has to be turned
// on by hand in System Settings. So this panel says that in plain words, opens
// the exact settings page, and leaves a Try again for when the switch is on.

import { Monitor } from "@phosphor-icons/react";
import { screen } from "../lib/screen";
import { storage } from "../lib/storage";

/** Whether the Screen button belongs in this build on this machine. */
export function screenFeedOffered(): boolean {
  return !storage.mobile && screen.isSupported();
}

export function ScreenDenied({ onRetry }: { onRetry: () => void }) {
  return (
    <div className="use__veil">
      <Monitor size={26} weight="bold" />
      <p>
        Belo is not allowed to record the screen yet. Turn Belo on under Screen Recording in
        System Settings, then come back and try again.
      </p>
      <div className="use__veil-actions">
        <button className="btn btn--sm btn--go" onClick={() => void screen.openSettings()}>
          Open System Settings
        </button>
        <button className="btn btn--sm" onClick={onRetry}>
          Try again
        </button>
      </div>
    </div>
  );
}
