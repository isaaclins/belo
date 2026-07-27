// One picker for every place that shows live video. It renders nothing while
// the machine has a single camera, so the common case stays uncluttered.

import { useId } from "react";
import { VideoCamera } from "@phosphor-icons/react";
import type { CameraInfo } from "../lib/useWebcam";

export function CameraPicker({
  cameras,
  deviceId,
  onPick,
  disabled = false,
  compact = false,
}: {
  cameras: CameraInfo[];
  deviceId: string | null;
  onPick: (deviceId: string) => void;
  disabled?: boolean;
  /** Row layout for a card header, stacked field layout otherwise. */
  compact?: boolean;
}) {
  const id = useId();
  if (cameras.length < 2) return null;
  // An unknown id means the platform opened something it did not list, so the
  // select falls back to the first entry rather than showing a blank value.
  const value = cameras.some((c) => c.deviceId === deviceId) ? deviceId : cameras[0].deviceId;

  return (
    <div className={compact ? "campick campick--compact" : "campick"}>
      <label className="campick__label" htmlFor={id}>
        {compact ? <VideoCamera size={14} weight="bold" /> : null}
        Camera
      </label>
      <select
        id={id}
        className="input campick__select"
        value={value ?? ""}
        disabled={disabled}
        onChange={(event) => onPick(event.target.value)}
      >
        {cameras.map((entry) => (
          <option key={entry.deviceId} value={entry.deviceId}>
            {entry.label}
          </option>
        ))}
      </select>
    </div>
  );
}
