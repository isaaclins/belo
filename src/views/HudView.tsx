import { useEffect, useMemo, useRef, useState } from "react";
import { ArrowCounterClockwise, Pause, Play, X } from "@phosphor-icons/react";
import { emitHudCommand, emitHudReady, listenHudState, type HudCommand, type HudState } from "../lib/hud";
import { normalizeRect, type BoxRect } from "../lib/types";
import { pct, swatch } from "../components/ui";

interface Point {
  x: number;
  y: number;
}

function rectBetween(start: Point, end: Point): BoxRect {
  return {
    x: Math.min(start.x, end.x),
    y: Math.min(start.y, end.y),
    w: Math.abs(end.x - start.x),
    h: Math.abs(end.y - start.y),
  };
}

function send(command: HudCommand): void {
  void emitHudCommand(command);
}

export function HudView() {
  const [state, setState] = useState<HudState | null>(null);
  const [picked, setPicked] = useState<string[]>([]);
  const [boxLabel, setBoxLabel] = useState<string | null>(null);
  const frameRef = useRef<HTMLDivElement | null>(null);
  const [drag, setDrag] = useState<{ start: Point; current: Point } | null>(null);

  useEffect(() => {
    let cancelled = false;
    let unlisten: (() => void) | null = null;
    void listenHudState((next) => {
      if (cancelled) return;
      setState(next);
      if (!boxLabel && next.activeLabelId) setBoxLabel(next.activeLabelId);
    }).then((stop) => {
      if (cancelled) stop();
      else unlisten = stop;
    });
    void emitHudReady();
    return () => {
      cancelled = true;
      unlisten?.();
    };
  }, []);

  useEffect(() => {
    if (!state) return;
    setPicked([]);
    setBoxLabel(state.activeLabelId ?? state.labels[0]?.id ?? null);
  }, [state?.frozen, state?.mode, state?.activeLabelId]);

  const labelById = useMemo(() => new Map(state?.labels.map((label) => [label.id, label]) ?? []), [state]);

  if (!state) {
    return (
      <main className="hud hud--waiting">
        <span className="eyebrow">Belo Capture HUD</span>
        <span className="mono">Waiting for the live screen</span>
      </main>
    );
  }

  const topLabel = state.verdict?.labelId ? labelById.get(state.verdict.labelId) : null;
  const multiLabels = state.verdict?.labelIds
    .map((id) => labelById.get(id)?.name)
    .filter(Boolean)
    .join(" + ");
  const title = state.mode === "detect" ? "Live detection" : "Live classify";
  const frame = drag ? rectBetween(drag.start, drag.current) : null;

  const pointAt = (event: React.PointerEvent): Point | null => {
    const rect = frameRef.current?.getBoundingClientRect();
    if (!rect || rect.width === 0 || rect.height === 0) return null;
    return {
      x: Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width)),
      y: Math.max(0, Math.min(1, (event.clientY - rect.top) / rect.height)),
    };
  };

  return (
    <main className="hud" aria-label="Belo capture HUD">
      <header className="hud__bar" data-tauri-drag-region="true">
        <span className="hud__brand">Belo</span>
        <span className="eyebrow">{title}</span>
        <span className="spacer" />
        {state.achievedFps ? <span className="hud__fps mono">{Math.round(state.achievedFps)} fps</span> : null}
        <button className="hud__icon" type="button" title="Hide HUD" aria-label="Hide HUD" onClick={() => send({ type: "hide" })}>
          <X size={14} weight="bold" />
        </button>
      </header>

      {state.frozen && state.frameDataUrl ? (
        <div
          className="hud__frame"
          ref={frameRef}
          onPointerDown={(event) => {
            if (state.mode !== "detect" || event.button !== 0) return;
            const point = pointAt(event);
            if (!point || !boxLabel) return;
            event.currentTarget.setPointerCapture(event.pointerId);
            setDrag({ start: point, current: point });
          }}
          onPointerMove={(event) => {
            if (!drag) return;
            const point = pointAt(event);
            if (point) setDrag({ ...drag, current: point });
          }}
          onPointerUp={(event) => {
            if (!drag) return;
            const point = pointAt(event) ?? drag.current;
            const clean = normalizeRect(rectBetween(drag.start, point));
            setDrag(null);
            if (clean && boxLabel) send({ type: "box", labelId: boxLabel, rect: clean });
          }}
        >
          <img src={state.frameDataUrl} alt="Frozen screen frame" draggable={false} />
          {frame && frame.w >= 0.02 && frame.h >= 0.02 ? <div className="hud__draft" style={{ left: `${frame.x * 100}%`, top: `${frame.y * 100}%`, width: `${frame.w * 100}%`, height: `${frame.h * 100}%` }} /> : null}
        </div>
      ) : null}

      <section className={`hud__verdict${state.frozen ? " hud__verdict--frozen" : ""}`}>
        <span className="eyebrow">{state.frozen ? "Frozen frame" : "Current verdict"}</span>
        {state.mode === "detect" ? (
          <strong>{state.boxCount} object{state.boxCount === 1 ? "" : "s"}</strong>
        ) : (
          <strong style={{ color: topLabel ? swatch(topLabel.color) : undefined }}>
            {multiLabels || topLabel?.name || "No verdict yet"}
          </strong>
        )}
        {state.mode === "classify" && state.verdict ? (
          <span className="mono hud__confidence">{pct(state.verdict.confidence)}</span>
        ) : null}
        {state.saving ? <span className="hud__saving mono">Saving frame...</span> : null}
      </section>

      {state.frozen && state.mode === "classify" ? (
        <section className="hud__teach" aria-label="Teach from frozen frame">
          <span className="eyebrow">Teach this frame</span>
          <div className="hud__labels">
            {state.labels.map((label) => {
              const selected = picked.includes(label.id);
              return (
                <button
                  key={label.id}
                  type="button"
                  className={`hud__label${selected ? " hud__label--on" : ""}`}
                  style={selected ? { background: swatch(label.color) } : undefined}
                  onClick={() => {
                    if (state.verdict?.labelIds.length && state.verdict.labelIds.includes(label.id)) {
                      setPicked([label.id]);
                      send({ type: "teach", labelIds: [label.id] });
                      return;
                    }
                    setPicked((current) => (current.includes(label.id) ? current.filter((id) => id !== label.id) : [...current, label.id]));
                  }}
                >
                  This is {label.name}
                </button>
              );
            })}
          </div>
          {picked.length > 0 ? (
            <button className="btn btn--sm btn--go hud__teachgo" type="button" onClick={() => { send({ type: "teach", labelIds: picked }); setPicked([]); }}>
              Teach selected
            </button>
          ) : null}
        </section>
      ) : null}

      {state.frozen && state.mode === "detect" ? (
        <section className="hud__teach" aria-label="Draw boxes on frozen frame">
          <span className="eyebrow">Draw boxes, then choose a label</span>
          <div className="hud__labels">
            {state.labels.map((label) => (
              <button
                key={label.id}
                type="button"
                className={`hud__label${boxLabel === label.id ? " hud__label--on" : ""}`}
                style={boxLabel === label.id ? { background: swatch(label.color) } : undefined}
                onClick={() => { setBoxLabel(label.id); send({ type: "set-label", labelId: label.id }); }}
              >
                {label.name}
              </button>
            ))}
          </div>
        </section>
      ) : null}

      <footer className="hud__foot">
        <button className="btn btn--sm btn--primary" type="button" onClick={() => send({ type: state.frozen ? "resume" : "freeze" })}>
          {state.frozen ? <Play size={14} weight="bold" /> : <Pause size={14} weight="bold" />}
          {state.frozen ? "Resume" : "Freeze"}
        </button>
        {state.frozen ? <span className="hud__hint"><ArrowCounterClockwise size={13} /> Resume returns to live</span> : <span className="hud__hint">Drag this bar to move</span>}
      </footer>
    </main>
  );
}
