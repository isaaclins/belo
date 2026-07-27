// Capture rate control, shared by the burst modal and the Use view hold.

import { CAPTURE_RATES, useCaptureRate, type CaptureRate } from "../lib/captureRate";

export function RatePicker({ compact = false }: { compact?: boolean }) {
  const [rate, setRate] = useCaptureRate();

  return (
    <div
      className={compact ? "rate rate--compact" : "rate"}
      role="radiogroup"
      aria-label="Capture rate in frames per second"
    >
      <span className="rate__label">Rate</span>
      <div className="rate__opts">
        {CAPTURE_RATES.map((option: CaptureRate) => (
          <button
            key={option}
            type="button"
            role="radio"
            aria-checked={rate === option}
            aria-label={`${option} frames per second`}
            className={`rate__opt${rate === option ? " rate__opt--on" : ""}`}
            onClick={() => setRate(option)}
          >
            {option}
          </button>
        ))}
      </div>
      <span className="rate__unit mono">fps</span>
    </div>
  );
}
