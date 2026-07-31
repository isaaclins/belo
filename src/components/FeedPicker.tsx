// One picker for every place that shows live video, camera or screen. It
// renders nothing while there is a single choice, so the common case stays
// uncluttered.

import { useId } from "react";
import { Monitor, VideoCamera } from "@phosphor-icons/react";
import type { FeedOption } from "../lib/useWebcam";

export function FeedPicker({
  what,
  options,
  optionId,
  onPick,
  disabled = false,
  compact = false,
}: {
  /** The word the label uses, and which icon it gets. */
  what: "Camera" | "Screen";
  options: FeedOption[];
  optionId: string | null;
  onPick: (id: string) => void;
  disabled?: boolean;
  /** Row layout for a card header, stacked field layout otherwise. */
  compact?: boolean;
}) {
  const id = useId();
  if (options.length < 2) return null;
  // An unknown id means the platform opened something it did not list, so the
  // select falls back to the first entry rather than showing a blank value.
  const value = options.some((entry) => entry.id === optionId) ? optionId : options[0].id;

  return (
    <div className={compact ? "campick campick--compact" : "campick"}>
      <label className="campick__label" htmlFor={id}>
        {compact ? (
          what === "Camera" ? (
            <VideoCamera size={14} weight="bold" />
          ) : (
            <Monitor size={14} weight="bold" />
          )
        ) : null}
        {what}
      </label>
      <select
        id={id}
        className="input campick__select"
        value={value ?? ""}
        disabled={disabled}
        onChange={(event) => onPick(event.target.value)}
      >
        {options.map((entry) => (
          <option key={entry.id} value={entry.id}>
            {entry.label}
          </option>
        ))}
      </select>
    </div>
  );
}
