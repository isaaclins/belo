// Camera or screen, for the two Use views. It renders nothing where a screen
// feed is impossible, which is every phone and any desktop webview without one,
// so the common case keeps the header it always had.

import type { ReactNode } from "react";
import { Monitor, VideoCamera } from "@phosphor-icons/react";
import type { FeedKind } from "../lib/useWebcam";

export function SourceToggle({
  value,
  onPick,
  disabled = false,
}: {
  value: FeedKind;
  onPick: (kind: FeedKind) => void;
  disabled?: boolean;
}) {
  const options: { kind: FeedKind; label: string; icon: ReactNode }[] = [
    { kind: "camera", label: "Camera", icon: <VideoCamera size={14} weight="bold" /> },
    { kind: "screen", label: "Screen", icon: <Monitor size={14} weight="bold" /> },
  ];

  return (
    <fieldset className="srcpick" aria-label="Live source">
      <legend>Source</legend>
      <div className="srcpick__steps">
        {options.map((option) => (
          <button
            key={option.kind}
            type="button"
            className={
              option.kind === value ? "srcpick__step srcpick__step--on" : "srcpick__step"
            }
            aria-pressed={option.kind === value}
            disabled={disabled}
            onClick={() => onPick(option.kind)}
          >
            {option.icon}
            {option.label}
          </button>
        ))}
      </div>
    </fieldset>
  );
}
