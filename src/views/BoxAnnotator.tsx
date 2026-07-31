// Detect mode annotator: one picture shown large, boxes drawn on top of it.
//
// Everything here works in normalized 0..1 coordinates, so what is stored never
// depends on the size the picture happens to be rendered at. The only place
// pixels appear is the pointer maths, which converts a client point into the
// frame's own space and back out through percentage positions.

import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useReducer,
  useRef,
  useState,
} from "react";
import {
  ArrowLeft,
  ArrowRight,
  ArrowUUpLeft,
  ArrowUUpRight,
  BoundingBox,
  Check,
  CircleNotch,
  Cursor,
  MagicWand,
  Plus,
  Trash,
  X,
} from "@phosphor-icons/react";
import { useStore } from "../lib/store";
import { pct, swatch } from "../components/ui";
import type { Detection } from "../lib/detect";
import { useScoreThreshold } from "../lib/scoreThreshold";
import { storage } from "../lib/storage";
import {
  advance,
  clamp,
  HANDLES,
  positive,
  type Drag,
  type Handle,
  type Point,
} from "../lib/boxes";
import {
  boxesOf,
  MIN_BOX,
  normalizeRect,
  type BoxRect,
  type BoxRecord,
  type ImageRecord,
  type Label,
} from "../lib/types";
import { emptyUndo, redo, undo, recordUndo, type UndoState } from "../lib/undo";

/** True when a committed drag ends exactly where the stored box already is. */
function unchanged(box: BoxRecord | undefined, rect: BoxRect | null): boolean {
  if (!box || !rect) return false;
  return box.x === rect.x && box.y === rect.y && box.w === rect.w && box.h === rect.h;
}

function percent(rect: BoxRect) {
  return {
    left: `${rect.x * 100}%`,
    top: `${rect.y * 100}%`,
    width: `${rect.w * 100}%`,
    height: `${rect.h * 100}%`,
  };
}

/**
 * One guess from the trained detector, waiting to be kept or thrown away.
 *
 * A proposal is session state and nothing else. It carries an id so the list
 * can be rendered and picked, but that id never reaches the manifest: accepting
 * one calls addBox exactly the way a drawn rectangle does, and the box that
 * comes back gets its own stored id.
 */
interface Proposal extends Detection {
  id: string;
}

/** How far a press may travel and still count as a tap on a guess, in pixels. */
const TAP_SLOP = 6;

/** Ids for guesses. Session only, never written anywhere. */
let guessSeq = 0;

const HINT_KEY = "belo.detect.guessHint";

function hintWasDismissed(): boolean {
  try {
    return window.localStorage.getItem(HINT_KEY) === "done";
  } catch {
    return false;
  }
}

function dismissHint(): void {
  try {
    window.localStorage.setItem(HINT_KEY, "done");
  } catch {
    // A webview with storage disabled shows the line again next time.
  }
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
  const setBoxes = useStore((s) => s.setBoxes);
  const toast = useStore((s) => s.toast);
  const proposeBoxes = useStore((s) => s.proposeBoxes);
  const hasModel = useStore((s) => s.hasModel);
  const modelStale = useStore((s) => s.stale);
  // The moment the live detector was trained. A background retrain finishing
  // moves it, which is what lets an open annotator pick up the better model.
  const trainedAt = useStore((s) => s.manifest?.model?.trainedAt ?? null);
  const threshold = useScoreThreshold();

  const stageRef = useRef<HTMLDivElement>(null);
  const frameRef = useRef<HTMLDivElement>(null);
  const dragRef = useRef<Drag | null>(null);
  /** Where a press on a guess started, so a tap can be told from a drag. */
  const tapRef = useRef<{ id: string; x: number; y: number } | null>(null);

  const [selected, setSelected] = useState<string | null>(null);
  const [preview, setPreview] = useState<{ boxId: string | null; rect: BoxRect } | null>(null);
  const [ratio, setRatio] = useState(1);
  const [frame, setFrame] = useState<{ w: number; h: number }>({ w: 0, h: 0 });
  /** Armed pencil: every press draws, even on top of an existing box. */
  const [drawTool, setDrawTool] = useState(false);

  // --- the detector's guesses -----------------------------------------------
  //
  // Proposals live exactly as long as one picture is on screen. Leaving the
  // picture drops them without a word, which is the whole safety story: what
  // was never accepted was never in the project.
  const [proposals, setProposals] = useState<Proposal[] | null>(null);
  const [thinking, setThinking] = useState(false);
  /** True once the user has kept or thrown away a guess on this picture. */
  const touchedRef = useRef(false);
  /** The picture the user explicitly asked to be guessed at again. */
  const askedRef = useRef<string | null>(null);
  const [asks, setAsks] = useState(0);
  const [hintOpen, setHintOpen] = useState(() => !hintWasDismissed());
  const guesses = proposals ?? [];
  // The key handler is registered once for the session, so it reads the current
  // guesses through a ref rather than closing over a list from one render.
  const proposalsRef = useRef<Proposal[]>(guesses);
  proposalsRef.current = guesses;

  // Undo history, one stack per picture, held in a ref so a press does not
  // depend on a render having landed. It lives exactly as long as the annotator
  // is open: closing it is what clears the history, on purpose, because a
  // reopened picture is a fresh session and the boxes on disk are the truth.
  const undoRef = useRef(new Map<string, UndoState>());
  const [, bump] = useReducer((count: number) => count + 1, 0);

  const image = images[index] ?? null;
  const boxes = image ? boxesOf(image) : [];
  const history = image ? (undoRef.current.get(image.id) ?? emptyUndo()) : emptyUndo();
  const activeLabel = labels.find((l) => l.id === activeLabelId) ?? null;
  const labelById = useMemo(() => new Map(labels.map((l) => [l.id, l] as const)), [labels]);

  // A new picture is a clean slate: nothing selected, nothing half drawn.
  useEffect(() => {
    setSelected(null);
    setPreview(null);
    dragRef.current = null;
    // Guesses belong to the picture they were made about. Leaving it throws
    // away every one that was not accepted, silently: what was never accepted
    // was never in the project.
    setProposals(null);
    setThinking(false);
    touchedRef.current = false;
    askedRef.current = null;
  }, [image?.id]);

  // The record the guessing effect works on, kept in a ref because every edit
  // to the project hands back a new object for the same picture and the effect
  // must not restart for that.
  const imageRef = useRef<ImageRecord | null>(image);
  imageRef.current = image;
  const imageId = image?.id ?? null;

  /**
   * Asks the trained detector what it thinks is in this picture.
   *
   * It runs after the picture is on screen rather than with it, and a press of
   * the arrow keys cancels it, so a project of a hundred pictures never waits
   * on inference to turn a page. The heavy half is the frozen extractor, and
   * that is read through the store's shared grid cache, so the second look at
   * a picture costs a head pass and nothing else.
   */
  useEffect(() => {
    const record = imageRef.current;
    if (!imageId || !record || !hasModel) return;
    // Guesses are for an empty picture. Somebody who already drew here has said
    // what is in it, and guessing over that would be noise; Suggest again is
    // how they ask for it anyway.
    const asked = askedRef.current === imageId;
    if (!asked && boxesOf(record).length > 0) return;
    // A fresh model may have landed mid review. Replacing the rectangles under
    // a hand that is already deciding about them is worse than letting this
    // picture finish with the guesses it started with.
    if (touchedRef.current) return;

    let cancelled = false;
    setThinking(true);
    void proposeBoxes(imageId, threshold)
      .then((found) => {
        if (cancelled) return;
        setProposals((found ?? []).map((box) => ({ ...box, id: `guess${(guessSeq += 1)}` })));
      })
      .catch(() => {
        // A detector that cannot answer is not an error worth a toast in the
        // middle of annotating. The picture simply gets no guesses.
        if (!cancelled) setProposals([]);
      })
      .finally(() => {
        if (!cancelled) setThinking(false);
      });
    return () => {
      cancelled = true;
    };
  }, [asks, hasModel, imageId, proposeBoxes, threshold, trainedAt]);

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

  /**
   * Remembers the boxes as they are, right before an edit changes them. Called
   * only where something really is about to change, so every press of undo is
   * one visible step back rather than a couple of no-ops first.
   */
  const remember = useCallback(() => {
    if (!image) return;
    const current = undoRef.current.get(image.id) ?? emptyUndo();
    undoRef.current.set(image.id, recordUndo(current, boxesOf(image)));
    bump();
  }, [image]);

  /** Puts one recorded state back, exactly as it was stored. */
  const walk = useCallback(
    (direction: "back" | "forward") => {
      if (!image) return;
      const current = undoRef.current.get(image.id) ?? emptyUndo();
      const step =
        direction === "back" ? undo(current, boxesOf(image)) : redo(current, boxesOf(image));
      if (!step) return;
      undoRef.current.set(image.id, step.state);
      setBoxes(image.id, step.boxes);
      setSelected((picked) => (step.boxes.some((box) => box.id === picked) ? picked : null));
      bump();
    },
    [image, setBoxes],
  );

  const commit = useCallback(
    (drag: Drag) => {
      if (!image) return;
      if (drag.kind === "draw") {
        if (!activeLabelId) return;
        // Too small to be deliberate: the press was a tap on empty space, which
        // already cleared the selection on the way down, and nothing to undo.
        if (!normalizeRect(drag.rect)) return;
        remember();
        const id = addBox(image.id, activeLabelId, drag.rect);
        if (id) setSelected(id);
        return;
      }
      if (!drag.boxId) return;
      const rect = normalizeRect(drag.rect);
      // A press that picked a box up and put it straight back down is not an
      // edit, so it neither marks the project stale nor costs an undo step.
      if (!rect || unchanged(boxesOf(image).find((box) => box.id === drag.boxId), rect)) return;
      remember();
      updateBox(image.id, drag.boxId, drag.rect);
    },
    [activeLabelId, addBox, image, remember, updateBox],
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
      // Whatever the press was about, it is over. A tap that came up somewhere
      // other than the guess it started on is simply not an acceptance.
      tapRef.current = null;
      const drag = dragRef.current;
      if (!drag) return;
      dragRef.current = null;
      setPreview(null);
      if (event.type === "pointercancel") return;
      // The release point counts: a pointer can come up somewhere no move event
      // ever reported, and committing the stale rectangle loses that last bit
      // of the gesture.
      commit(advance(drag, pointAt(event)));
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

  /**
   * Keeps some guesses. They go in through addBox, the same call a dragged
   * rectangle makes, so the manifest cannot tell the difference: undo covers
   * it, the project is marked changed, and the debounced background retrain
   * counts the new boxes like any others.
   *
   * One undo step covers the whole batch, so keeping five guesses is undone by
   * one press rather than five.
   */
  const acceptProposals = useCallback(
    (chosen: Proposal[]) => {
      if (!image || chosen.length === 0) return;
      remember();
      let last: string | null = null;
      for (const guess of chosen) {
        const id = addBox(image.id, guess.labelId, {
          x: guess.x,
          y: guess.y,
          w: guess.w,
          h: guess.h,
        });
        if (id) last = id;
      }
      touchedRef.current = true;
      const kept = new Set(chosen.map((guess) => guess.id));
      setProposals((current) => (current ?? []).filter((guess) => !kept.has(guess.id)));
      // One kept guess is now a normal box, ready to be nudged. A whole batch
      // has no single box worth picking.
      setSelected(chosen.length === 1 ? last : null);
    },
    [addBox, image, remember],
  );

  const dismissProposal = useCallback((id: string) => {
    touchedRef.current = true;
    setProposals((current) => (current ?? []).filter((guess) => guess.id !== id));
    setSelected((picked) => (picked === id ? null : picked));
  }, []);

  const acceptAll = useCallback(() => {
    acceptProposals(proposalsRef.current);
  }, [acceptProposals]);

  /** Asks for guesses on a picture that already has boxes, or had none to show. */
  const suggestAgain = useCallback(() => {
    if (!imageId) return;
    askedRef.current = imageId;
    touchedRef.current = false;
    setAsks((count) => count + 1);
  }, [imageId]);

  const dropSelected = useCallback(() => {
    if (!image || !selected) return;
    // Delete means "not that one" whether the rectangle is a stored box or a
    // guess. A guess is only forgotten; nothing is written either way.
    if ((proposals ?? []).some((guess) => guess.id === selected)) {
      dismissProposal(selected);
      return;
    }
    if (!boxesOf(image).some((box) => box.id === selected)) return;
    remember();
    removeBox(image.id, selected);
    setSelected(null);
  }, [dismissProposal, image, proposals, remember, removeBox, selected]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (target && /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName)) return;
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "z") {
        event.preventDefault();
        walk(event.shiftKey ? "forward" : "back");
        return;
      }
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
      // The confirm streak: enter keeps what Belo drew, space keeps it and
      // turns the page. Neither does anything when there is nothing to keep,
      // so a picture without guesses still behaves as it always did.
      if (event.key === "Enter" || event.key === " ") {
        // Both keys already mean "press this" when a button has the focus, and
        // stealing them there would fire two different things at once.
        if (target && target.tagName === "BUTTON") return;
        const waiting = proposalsRef.current;
        if (waiting.length === 0) return;
        event.preventDefault();
        acceptProposals(waiting);
        if (event.key === " ") step(1);
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
  }, [acceptProposals, dropSelected, labels, onClose, selected, setActiveLabel, step, walk]);

  const startDraw = (event: React.PointerEvent) => {
    if (event.button !== 0) return;
    if (!activeLabelId) {
      // A press that landed on a guess is a decision about that guess and not
      // an attempt to draw, so it does not deserve the nudge.
      if (!tapRef.current) toast("Pick a label first, that is the color you draw with", "bad");
      return;
    }
    event.preventDefault();
    const grab = pointAt(event);
    const rect = { x: grab.x, y: grab.y, w: 0, h: 0 };
    dragRef.current = { kind: "draw", boxId: null, handle: null, origin: rect, grab, rect };
    // A press on a guess has already picked it on the way up the tree, and
    // clearing the selection here would undo that.
    if (!tapRef.current) setSelected(null);
    setPreview({ boxId: null, rect });
  };

  /**
   * A guess is kept by a tap on it, and a tap is a press that did not travel.
   * Nothing here stops the press from reaching the picture underneath, so the
   * same gesture that would draw a rectangle across a guess still draws it: a
   * real drag ends as a drawn box and never as an accepted guess.
   */
  const startTap = (event: React.PointerEvent, id: string) => {
    if (event.button !== 0) return;
    tapRef.current = { id, x: event.clientX, y: event.clientY };
    setSelected(id);
  };

  const endTap = (event: React.PointerEvent, guess: Proposal) => {
    const tap = tapRef.current;
    tapRef.current = null;
    if (!tap || tap.id !== guess.id) return;
    if (Math.hypot(event.clientX - tap.x, event.clientY - tap.y) > TAP_SLOP) return;
    acceptProposals([guess]);
  };

  const startMove = (event: React.PointerEvent, boxId: string, rect: BoxRect) => {
    if (event.button !== 0) return;
    // The armed pencil hands the press straight through to the picture, which
    // is the only way to draw inside a box that already covers everything.
    if (drawTool) {
      event.stopPropagation();
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
    if (!image || !selected) return;
    const box = boxesOf(image).find((candidate) => candidate.id === selected);
    if (!box || box.labelId === label.id) return;
    remember();
    setBoxLabel(image.id, selected, label.id);
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
        <button
          className="iconbtn"
          onClick={() => walk("back")}
          disabled={history.past.length === 0}
          title="Undo the last change to this picture (cmd or ctrl + z)"
          aria-label="Undo"
        >
          <ArrowUUpLeft size={16} weight="bold" />
        </button>
        <button
          className="iconbtn"
          onClick={() => walk("forward")}
          disabled={history.future.length === 0}
          title="Redo the change you just undid (shift + cmd or ctrl + z)"
          aria-label="Redo"
        >
          <ArrowUUpRight size={16} weight="bold" />
        </button>
        <span className="badge badge--idle">
          {boxes.length} box{boxes.length === 1 ? "" : "es"}
        </span>
        {guesses.length > 0 ? (
          <span className="badge badge--guess" data-guess-count={guesses.length}>
            {guesses.length} guess{guesses.length === 1 ? "" : "es"}
          </span>
        ) : null}
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

            {guesses.map((guess) => {
              const label = labelById.get(guess.labelId) ?? null;
              const color = swatch(label?.color ?? 0);
              const on = selected === guess.id;
              return (
                <div
                  key={guess.id}
                  className={`aprop${on ? " aprop--on" : ""}`}
                  data-guess-label={guess.labelId}
                  data-guess-rect={`${guess.x.toFixed(4)},${guess.y.toFixed(4)},${guess.w.toFixed(4)},${guess.h.toFixed(4)}`}
                  style={{ ...percent(guess), borderColor: color }}
                  onPointerDown={(event) => startTap(event, guess.id)}
                  onPointerUp={(event) => endTap(event, guess)}
                >
                  <span className="aprop__tag" style={{ color }}>
                    {label?.name ?? "removed label"} <span className="mono">{pct(guess.score)}</span>
                  </span>
                  <button
                    className="aprop__drop"
                    title="Throw this guess away"
                    aria-label={`Throw away the guess ${label?.name ?? ""}`}
                    onPointerDown={(event) => {
                      event.stopPropagation();
                      event.preventDefault();
                    }}
                    onPointerUp={(event) => {
                      event.stopPropagation();
                      dismissProposal(guess.id);
                    }}
                    // Pointer and click both, so the keyboard reaches it too.
                    // Throwing the same guess away twice is already a no-op.
                    onClick={() => dismissProposal(guess.id)}
                  >
                    <X size={11} weight="bold" />
                  </button>
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

          {hintOpen && guesses.length > 0 ? (
            <p className="anno__firsthint">
              Belo has started guessing. Click a guess to keep it, or draw over it.
              <button
                className="anno__firsthint-x"
                aria-label="Hide this hint"
                onClick={() => {
                  setHintOpen(false);
                  dismissHint();
                }}
              >
                <X size={13} weight="bold" />
              </button>
            </p>
          ) : null}
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

          {hasModel ? (
            <div className="anno__block" data-guesses={guesses.length}>
              <span className="eyebrow">Belo's guesses</span>
              {thinking ? (
                <p className="field__hint anno__thinking" data-thinking="1">
                  <CircleNotch size={13} weight="bold" /> Looking at this picture
                </p>
              ) : null}
              {guesses.length > 0 ? (
                <>
                  <button
                    className="btn btn--sm btn--primary btn--block"
                    data-accept-all="1"
                    onClick={acceptAll}
                  >
                    <Check size={15} weight="bold" /> Keep all {guesses.length}
                  </button>
                  <p className="field__hint">
                    Click a guess to keep it, or press its x to throw it away. Nothing is saved
                    until you keep it.
                  </p>
                  {modelStale ? (
                    <p className="field__hint" data-guess-stale="1">
                      These are guesses from an older training run. Belo is still learning from
                      your newest boxes.
                    </p>
                  ) : null}
                </>
              ) : (
                <>
                  <p className="field__hint">
                    {thinking
                      ? "One moment."
                      : boxes.length > 0
                        ? "This picture already has your boxes on it, so Belo left it alone."
                        : "Belo had nothing to suggest here. Draw the boxes yourself."}
                  </p>
                  <button
                    className="btn btn--sm btn--block"
                    data-suggest-again="1"
                    disabled={thinking}
                    onClick={suggestAgain}
                  >
                    <MagicWand size={15} weight="bold" /> Suggest again
                  </button>
                </>
              )}
            </div>
          ) : null}

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
              arrows: other pictures · 1 to 9: label · delete: remove box · cmd z: undo · esc:
              close
              {hasModel ? " · enter: keep every guess · space: keep them and go on" : ""}
            </p>
          </div>
        </aside>
      </div>
    </div>
  );
}
