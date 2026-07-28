import { useCallback, useEffect, useRef, useState, type RefObject } from "react";
import type { Detection } from "../lib/detect";
import { containSquare, cropSquare, type Rect } from "../lib/attention";
import type { Label } from "../lib/types";
import { pct, swatch } from "./ui";

type DetectionSource = HTMLVideoElement | HTMLImageElement;

function sourceSize(source: DetectionSource) {
  return source instanceof HTMLVideoElement
    ? { width: source.videoWidth, height: source.videoHeight }
    : { width: source.naturalWidth, height: source.naturalHeight };
}

/**
 * DOM boxes over the exact center square handed to MobileNet. Live video uses
 * object-fit: cover while a picked image uses contain, so both go through the
 * matching half of the same fit math as the attention canvas.
 */
export function DetectionOverlay({
  sourceRef,
  fit,
  detections,
  labels,
}: {
  sourceRef: RefObject<DetectionSource | null>;
  fit: "cover" | "contain";
  detections: Detection[];
  labels: Label[];
}) {
  const overlayRef = useRef<HTMLDivElement | null>(null);
  const [square, setSquare] = useState<Rect | null>(null);

  const measure = useCallback(() => {
    const overlay = overlayRef.current;
    const source = sourceRef.current;
    if (!overlay || !source) return setSquare(null);
    const container = { width: overlay.clientWidth, height: overlay.clientHeight };
    const size = sourceSize(source);
    setSquare(fit === "cover" ? cropSquare(container, size) : containSquare(container, size));
  }, [fit, sourceRef]);

  useEffect(() => {
    measure();
    const overlay = overlayRef.current;
    const source = sourceRef.current;
    if (!overlay || !source) return;
    const observer = new ResizeObserver(measure);
    observer.observe(overlay);
    source.addEventListener("load", measure);
    source.addEventListener("loadedmetadata", measure);
    source.addEventListener("resize", measure);
    return () => {
      observer.disconnect();
      source.removeEventListener("load", measure);
      source.removeEventListener("loadedmetadata", measure);
      source.removeEventListener("resize", measure);
    };
  }, [measure]);

  return (
    <div ref={overlayRef} className="use__detections" aria-hidden="true">
      {square
        ? detections.map((detection, index) => {
            const label = labels.find((entry) => entry.id === detection.labelId);
            const color = swatch(label?.color ?? 0);
            const left = square.x + detection.x * square.size;
            const top = square.y + detection.y * square.size;
            return (
              <div
                key={`${detection.labelId}-${index}`}
                className="detectbox"
                data-label-id={detection.labelId}
                data-score={detection.score.toFixed(4)}
                style={{
                  left,
                  top,
                  width: detection.w * square.size,
                  height: detection.h * square.size,
                  borderColor: color,
                }}
              >
                <span
                  className={`detectbox__tag${top < 25 ? " detectbox__tag--inside" : ""}`}
                  style={{ background: color }}
                >
                  {label?.name ?? "removed label"} <b className="mono">{pct(detection.score)}</b>
                </span>
              </div>
            );
          })
        : null}
    </div>
  );
}
