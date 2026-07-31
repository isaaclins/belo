// What the stage shows when the system will not let Belo see the screen.
//
// macOS asks once, and after a no it never asks again: the app has to be turned
// on by hand in System Settings. So this panel says that in plain words, opens
// the exact settings page, and leaves a Try again for when the switch is on.

import { useEffect, useState } from "react";
import { Monitor } from "@phosphor-icons/react";
import { screen } from "../lib/screen";
import { storage } from "../lib/storage";

/**
 * Whether the Screen button belongs in this build on this machine. The platform
 * has the last word and answers asynchronously, so the switch appears a tick
 * after the view does rather than appearing and then being taken away.
 */
export function useScreenOffered(): boolean {
  const [offered, setOffered] = useState(false);
  useEffect(() => {
    if (storage.mobile || !screen.isSupported()) return;
    let live = true;
    void screen
      .supported()
      .then((ok) => {
        if (live) setOffered(ok);
      })
      .catch(() => undefined);
    return () => {
      live = false;
    };
  }, []);
  return offered;
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
