import {
  SCREEN_QUALITY_PRESETS,
  type ScreenQuality,
} from "../lib/screenQuality";

export function ScreenQualityPicker({
  value,
  onChange,
  fps,
  compact = false,
}: {
  value: ScreenQuality;
  onChange: (value: ScreenQuality) => void;
  fps?: number;
  compact?: boolean;
}) {
  return (
    <fieldset className={compact ? "screenquality screenquality--compact" : "screenquality"}>
      <legend>Screen feed</legend>
      <div className="screenquality__steps" role="radiogroup" aria-label="Screen capture quality">
        {SCREEN_QUALITY_PRESETS.map((preset) => (
          <button
            key={preset.id}
            type="button"
            role="radio"
            aria-checked={preset.id === value}
            className={preset.id === value ? "screenquality__step screenquality__step--on" : "screenquality__step"}
            onClick={() => onChange(preset.id)}
            title={`${preset.label} screen capture${preset.requestedFps ? `, up to ${preset.requestedFps} fps` : ", follows the display"}`}
          >
            {preset.label}
          </button>
        ))}
      </div>
      {fps ? <span className="screenquality__fps mono">{Math.round(fps)} fps</span> : null}
    </fieldset>
  );
}
