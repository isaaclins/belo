// Detect mode annotator: one picture shown large, boxes drawn on top of it.
//
// Everything here works in normalized 0..1 coordinates, so what is stored never
// depends on the size the picture happens to be rendered at. The only place
// pixels appear is the pointer maths, which converts a client point into the
// frame's own space and back out through percentage positions.

import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import {
  ArrowLeft,
  ArrowRight,
  BoundingBox,
  Cursor,
  Plus,
  Trash,
  X,
} from "@phosphor-icons/react";
import { useStore } from "../lib/store";
import { swatch } from "../components/ui";
import { storage } from "../lib/storage";
import { boxesOf, MIN_BOX, type BoxRect, type ImageRecord, type Label } from "../lib/types";

/** The eight grips of the selected box, named by compass direction. */
const HANDLES = ["nw", "n", "ne", "e", "se", "s", "sw", "w"] as const;
type Handle = (typeof HANDLES)[number];

interface Point {
  x: number;
  y: number;
}

interface Drag {
  kind: "draw" | "move" | "resize";
  /** Null while a brand new box is being dragged out. */
  boxId: string | null;
  handle: Handle | null;
  /** The rectangle as it was when the pointer went down. */
  origin: BoxRect;
  grab: Point;
  /** Live rectangle. Width and height may be negative mid drag. */
  rect: BoxRect;
}

function clamp(value: number, low: number, high: number): number {
  return Math.max(low, Math.min(high, value));
}

/** Turns a possibly backwards rectangle into a positive one for rendering. */
function positive(rect: BoxRect): BoxRect {
  return {
    x: Math.min(rect.x, rect.x + rect.w),
    y: Math.min(rect.y, rect.y + rect.h),
    w: Math.abs(rect.w),
    h: Math.abs(rect.h),
  };
}

function moved(origin: BoxRect, grab: Point, point: Point): BoxRect {
  return {
    x: clamp(origin.x + (point.x - grab.x), 0, 1 - origin.w),
    y: clamp(origin.y + (point.y - grab.y), 0, 1 - origin.h),
    w: origin.w,
    h: origin.h,
  };
}

function resized(origin: BoxRect, handle: Handle, point: Point): BoxRect {
  let x0 = origin.x;
  let y0 = origin.y;
  let x1 = origin.x + origin.w;
  let y1 = origin.y + origin.h;
  if (handle.includes("w")) x0 = point.x;
  if (handle.includes("e")) x1 = point.x;
  if (handle.includes("n")) y0 = point.y;
  if (handle.includes("s")) y1 = point.y;
  return { x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
}

function advance(drag: Drag, point: Point): Drag {
  if (drag.kind === "draw") {
    return { ...drag, rect: { x: drag.grab.x, y: drag.grab.y, w: point.x - drag.grab.x, h: point.y - drag.grab.y } };
  }
  if (drag.kind === "move") return { ...drag, rect: moved(drag.origin, drag.grab, point) };
  return { ...drag, rect: resized(drag.origin, drag.handle ?? "se", point) };
}

function percent(rect: BoxRect) {
  return {
    left: `${rect.x * 100}%`,
    top: `${rect.y * 100}%`,
    width: `${rect.w * 100}%`,
    height: `${rect.h * 100}%`,
  };
}

export function BoxAnnotator({
  images,
  index,
  onIndex,
  onClose,
}: {
  images: ImageRecord[];
  index: number;
  onIndex: (next: number) => void;
  onClose: () => void;
}) {
  const path = useStore((s) => s.path);
  const labels = useStore((s) => s.manifest?.labels ?? []);
  const activeLabelId = useStore((s) => s.activeLabelId);
  const setActiveLabel = useStore((s) => s.setActiveLabel);
  const addLabel = useStore((s) => s.addLabel);
  const addBox = useStore((s) => s.addBox);
  const updateBox = useStore((s) => s.updateBox);
  const setBoxLabel = useStore((s) => s.setBoxLabel);
  const removeBox = useStore((s) => s.removeBox);
  const toast = useStore((s) => s.toast);

  const stageRef = useRef<HTMLDivElement>(null);
  const frameRef = useRef<HTMLDivElement>(null);
  const dragRef = useRef<Drag | null>(null);

  const [selected, setSelected] = useState<string | null>(null);
  const [preview, setPreview] = useState<{ boxId: string | null; rect: BoxRect } | null>(null);
  const [ratio, setRatio] = useState(1);
  const [frame, setFrame] = useState<{ w: number; h: number }>({ w: 0, h: 0 });
  /** Armed pencil: every press draws, even on top of an existing box. */
  const [drawTool, setDrawTool] = useState(false);

  const image = images[index] ?? null;
  const boxes = image ? boxesOf(image) : [];
  const activeLabel = labels.find((l) => l.id === activeLabelId) ?? null;
  const labelById = useMemo(() => new Map(labels.map((l) => [l.id, l] as const)), [labels]);

  // A new picture is a clean slate: nothing selected, nothing half drawn.
  useEffect(() => {
    setSelected(null);
    setPreview(null);
    dragRef.current = null;
  }, [image?.id]);

  // The frame is sized here rather than in CSS: an aspect ratio that has to fit
  // inside a box constrained on both axes is exactly the case where the pure
  // CSS answers stop agreeing across engines, and the box layer has to sit on
  // the picture to the pixel.
  useLayoutEffect(() => {
    const stage = stageRef.current;
    if (!stage) return;
    const measure = () => {
      // clientWidth and clientHeight still carry the padding, and a frame sized
      // from those runs off the bottom of the stage.
      const style = getComputedStyle(stage);
      const padX = parseFloat(style.paddingLeft) + parseFloat(style.paddingRight);
      const padY = parseFloat(style.paddingTop) + parseFloat(style.paddingBottom);
      const available = {
        w: stage.clientWidth - padX,
        h: stage.clientHeight - padY,
      };
      if (available.w <= 0 || available.h <= 0) return;
      const w = Math.min(available.w, available.h * ratio);
      setFrame({ w, h: w / ratio });
    };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(stage);
    return () => observer.disconnect();
  }, [ratio]);

  const pointAt = useCallback((event: { clientX: number; clientY: number }): Point => {
    const rect = frameRef.current?.getBoundingClientRect();
    if (!rect || rect.width === 0 || rect.height === 0) return { x: 0, y: 0 };
    return {
      x: clamp((event.clientX - rect.left) / rect.width, 0, 1),
      y: clamp((event.clientY - rect.top) / rect.height, 0, 1),
    };
  }, []);

  const commit = useCallback(
    (drag: Drag) => {
      if (!image) return;
      if (drag.kind === "draw") {
        if (!activeLabelId) return;
        const id = addBox(image.id, activeLabelId, drag.rect);
        // Too small to be deliberate: the press was a tap on empty space, which
        // already cleared the selection on the way down.
        if (id) setSelected(id);
        return;
      }
      if (drag.boxId) updateBox(image.id, drag.boxId, drag.rect);
    },
    [activeLabelId, addBox, image, updateBox],
  );

  // One pair of window listeners for the whole session. They no-op unless a
  // drag is in flight, and living on the window means a pointer that leaves the
  // picture, or is released outside it, still finishes the gesture.
  useEffect(() => {
    const onMove = (event: PointerEvent) => {
      const drag = dragRef.current;
      if (!drag) return;
      event.preventDefault();
      const next = advance(drag, pointAt(event));
      dragRef.current = next;
      setPreview({ boxId: next.boxId, rect: next.rect });
    };
    const onUp = (event: PointerEvent) => {
      const drag = dragRef.current;
      if (!drag) return;
      dragRef.current = null;
      setPreview(null);
      // The release point counts: a pointer can come up somewhere no move event
      // ever reported, and committing the stale rectangle loses that last bit
      // of the gesture.
      commit(event.type === "pointercancel" ? drag : advance(drag, pointAt(event)));
    };
    window.addEventListener("pointermove", onMove, { passive: false });
    window.addEventListener("pointerup", onUp);
    window.addEventListener("pointercancel", onUp);
    return () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      window.removeEventListener("pointercancel", onUp);
    };
  }, [commit, pointAt]);

  const step = useCallback(
    (delta: number) => {
      if (images.length === 0) return;
      const next = (index + delta + images.length) % images.length;
      onIndex(next);
    },
    [images.length, index, onIndex],
  );

  const dropSelected = useCallback(() => {
    if (!image || !selected) return;
    removeBox(image.id, selected);
    setSelected(null);
  }, [image, removeBox, selected]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (target && /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName)) return;
      if (event.key === "Escape") {
        event.preventDefault();
        onClose();
        return;
      }
      if (event.key === "ArrowRight" || event.key === "ArrowDown") {
        event.preventDefault();
        step(1);
        return;
      }
      if (event.key === "ArrowLeft" || event.key === "ArrowUp") {
        event.preventDefault();
        step(-1);
        return;
      }
      if (event.key === "Delete" || event.key === "Backspace") {
        if (!selected) return;
        event.preventDefault();
        dropSelected();
        return;
      }
      if (event.key >= "1" && event.key <= "9") {
        const label = labels[Number(event.key) - 1];
        if (label) {
          event.preventDefault();
          setActiveLabel(label.id);
        }
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [dropSelected, labels, onClose, selected, setActiveLabel, step]);

  const startDraw = (event: React.PointerEvent) => {
    if (event.button !== 0) return;
    if (!activeLabelId) {
      toast("Pick a label first, that is the color you draw with", "bad");
      return;
    }
    event.preventDefault();
    const grab = pointAt(event);
    const rect = { x: grab.x, y: grab.y, w: 0, h: 0 };
    dragRef.current = { kind: "draw", boxId: null, handle: null, origin: rect, grab, rect };
    setSelected(null);
    setPreview({ boxId: null, rect });
  };

  const startMove = (event: React.PointerEvent, boxId: string, rect: BoxRect) => {
    if (event.button !== 0) return;
    // The armed pencil hands the press straight through to the picture, which
    // is the only way to draw inside a box that already covers everything.
    if (drawTool) {
      startDraw(event);
      return;
    }
    event.preventDefault();
    event.stopPropagation();
    setSelected(boxId);
    const grab = pointAt(event);
    dragRef.current = { kind: "move", boxId, handle: null, origin: rect, grab, rect };
    setPreview({ boxId, rect });
  };

  const startResize = (event: React.PointerEvent, boxId: string, rect: BoxRect, handle: Handle) => {
    if (event.button !== 0) return;
    event.preventDefault();
    event.stopPropagation();
    setSelected(boxId);
    const grab = pointAt(event);
    dragRef.current = { kind: "resize", boxId, handle, origin: rect, grab, rect };
    setPreview({ boxId, rect });
  };

  const pickLabel = (label: Label) => {
    setActiveLabel(label.id);
    // Picking a label with a box selected is the natural way to fix a wrong tag.
    if (image && selected) setBoxLabel(image.id, selected, label.id);
  };

  if (!image || !path) return null;

  const drawing = preview && preview.boxId === null ? positive(preview.rect) : null;
  const tooSmall = drawing !== null && (drawing.w < MIN_BOX || drawing.h < MIN_BOX);

  return (
    <div className="anno" role="dialog" aria-modal="true" aria-label="Draw boxes on this picture">
      <header className="anno__bar">
        <span className="anno__mark">
          <BoundingBox size={18} weight="bold" />
        </span>
        <span className="eyebrow">
          Picture {index + 1} of {images.length}
        </span>
        <button
          className="iconbtn"
          onClick={() => step(-1)}
          disabled={images.length < 2}
          title="Previous picture (left arrow)"
          aria-label="Previous picture"
        >
          <ArrowLeft size={16} weight="bold" />
        </button>
        <button
          className="iconbtn"
          onClick={() => step(1)}
          disabled={images.length < 2}
          title="Next picture (right arrow)"
          aria-label="Next picture"
        >
          <ArrowRight size={16} weight="bold" />
        </button>
        <span className="badge badge--idle">
          {boxes.length} box{boxes.length === 1 ? "" : "es"}
        </span>
        <div className="spacer" />
        <button className="btn btn--sm" onClick={onClose} aria-label="Close the annotator">
          <X size={15} weight="bold" /> Done
        </button>
      </header>

      <div className="anno__body">
        <div className="anno__stage" ref={stageRef}>
          <div
            className={`anno__frame${drawTool ? " anno__frame--draw" : ""}`}
            ref={frameRef}
            style={frame.w > 0 ? { width: frame.w, height: frame.h } : { visibility: "hidden" }}
            onPointerDown={startDraw}
          >
            <img
              className="anno__img"
              src={storage.imageUrl(path, image.file)}
              alt=""
              draggable={false}
              onLoad={(event) => {
                const el = event.currentTarget;
                if (el.naturalWidth > 0 && el.naturalHeight > 0) {
                  setRatio(el.naturalWidth / el.naturalHeight);
                }
              }}
            />

            {boxes.map((box) => {
              const live = preview && preview.boxId === box.id ? positive(preview.rect) : box;
              const label = labelById.get(box.labelId) ?? null;
              const on = selected === box.id;
              return (
                <div
                  key={box.id}
                  className={`abox${on ? " abox--on" : ""}`}
                  style={{
                    ...percent(live),
                    borderColor: swatch(label?.color ?? 0),
                    zIndex: on ? 3 : 2,
                  }}
                  onPointerDown={(event) => startMove(event, box.id, box)}
                >
                  <span className="abox__tag" style={{ background: swatch(label?.color ?? 0) }}>
                    {label?.name ?? "removed label"}
                  </span>
                  {on ? (
                    <>
                      <button
                        className="abox__kill"
                        title="Delete this box"
                        aria-label="Delete this box"
                        onPointerDown={(event) => event.stopPropagation()}
                        onClick={dropSelected}
                      >
                        <X size={12} weight="bold" />
                      </button>
                      {HANDLES.map((handle) => (
                        <span
                          key={handle}
                          className={`abox__grip abox__grip--${handle}`}
                          onPointerDown={(event) => startResize(event, box.id, box, handle)}
                        />
                      ))}
                    </>
                  ) : null}
                </div>
              );
            })}

            {drawing ? (
              <div
                className={`abox abox--draft${tooSmall ? " abox--tiny" : ""}`}
                style={{
                  ...percent(drawing),
                  borderColor: swatch(activeLabel?.color ?? 0),
                }}
              />
            ) : null}
          </div>
        </div>

        <aside className="anno__side">
          <div className="anno__block">
            <span className="eyebrow">Drawing with</span>
            <ul className="anno__labels">
              {labels.map((label, i) => (
                <li key={label.id}>
                  <button
                    className={`railitem${activeLabelId === label.id ? " railitem--on" : ""}`}
                    title={
                      i < 9
                        ? `Draw with ${label.name}, or press ${i + 1}`
                        : `Draw with ${label.name}`
                    }
                    onClick={() => pickLabel(label)}
                  >
                    <span className="dot" style={{ background: swatch(label.color) }} />
                    <span className="railitem__name">{label.name}</span>
                    <span className="railitem__count mono">
                      {boxes.filter((box) => box.labelId === label.id).length}
                    </span>
                  </button>
                </li>
              ))}
            </ul>
            <button
              className="btn btn--sm btn--block"
              onClick={() => addLabel(`Label ${labels.length + 1}`)}
            >
              <Plus size={15} weight="bold" /> New label
            </button>
          </div>

          <div className="anno__block">
            <span className="eyebrow">Tool</span>
            <div className="row row--tight">
              <button
                className={`btn btn--sm${drawTool ? "" : " btn--primary"}`}
                aria-pressed={!drawTool}
                onClick={() => setDrawTool(false)}
              >
                <Cursor size={15} weight="bold" /> Pick
              </button>
              <button
                className={`btn btn--sm${drawTool ? " btn--primary" : ""}`}
                aria-pressed={drawTool}
                onClick={() => setDrawTool(true)}
              >
                <BoundingBox size={15} weight="bold" /> Draw
              </button>
            </div>
            <p className="field__hint">
              Drag on the picture to draw a box. Press a box to pick it up, drag a corner to resize
              it, and use Draw when a new box has to start on top of one that is already there.
            </p>
          </div>

          <div className="anno__block">
            <span className="eyebrow">Selected box</span>
            {selected ? (
              <button className="btn btn--sm btn--danger btn--block" onClick={dropSelected}>
                <Trash size={15} weight="bold" /> Delete box
              </button>
            ) : (
              <p className="field__hint">
                Nothing picked. A picture with no boxes is still useful: it teaches Belo what an
                empty scene looks like.
              </p>
            )}
            <p className="field__hint mono anno__keys">
              arrows: other pictures · 1 to 9: label · delete: remove box · esc: close
            </p>
          </div>
        </aside>
      </div>
    </div>
  );
}
