// Undo for the box annotator: a stack of whole box lists, one per picture.
//
// Snapshots rather than inverse operations. A box list is a handful of small
// records, so keeping the state before every edit costs nothing measurable, and
// it makes every kind of edit undoable through the same path: a draw, a move, a
// resize, a delete and a retag all become "the boxes were this before".
// Coordinates come back exactly as they were stored, because the numbers are
// the same numbers rather than the result of running the edit backwards.
//
// Pure, checked in node by scripts/check-store.mjs.

import type { BoxRecord } from "./types";

/** Steps kept per picture. Older ones fall off the bottom. */
export const UNDO_LIMIT = 24;

export interface UndoState {
  past: BoxRecord[][];
  future: BoxRecord[][];
}

export function emptyUndo(): UndoState {
  return { past: [], future: [] };
}

/** Same boxes, same order, same numbers. */
export function sameBoxes(a: BoxRecord[], b: BoxRecord[]): boolean {
  if (a.length !== b.length) return false;
  return a.every((box, i) => {
    const other = b[i];
    return (
      box.id === other.id &&
      box.labelId === other.labelId &&
      box.x === other.x &&
      box.y === other.y &&
      box.w === other.w &&
      box.h === other.h
    );
  });
}

function copy(boxes: BoxRecord[]): BoxRecord[] {
  return boxes.map((box) => ({ ...box }));
}

/**
 * Records the state a picture was in before an edit. A snapshot identical to
 * the last one recorded is dropped: a drag that ended where it started, or a
 * label picked twice, would otherwise cost two presses of undo to get one step
 * back. Recording a real edit drops the redo trail, which is now a branch of
 * history that no longer exists.
 */
export function recordUndo(state: UndoState, before: BoxRecord[]): UndoState {
  const top = state.past[state.past.length - 1];
  if (top && sameBoxes(top, before)) return state;
  const past = [...state.past, copy(before)];
  if (past.length > UNDO_LIMIT) past.splice(0, past.length - UNDO_LIMIT);
  return { past, future: [] };
}

export interface UndoStep {
  state: UndoState;
  /** The boxes the picture should be restored to. */
  boxes: BoxRecord[];
}

/** One step back. Null when there is nothing recorded for this picture. */
export function undo(state: UndoState, current: BoxRecord[]): UndoStep | null {
  const boxes = state.past[state.past.length - 1];
  if (!boxes) return null;
  return {
    state: { past: state.past.slice(0, -1), future: [...state.future, copy(current)] },
    boxes: copy(boxes),
  };
}

/** One step forward, available only until the next recorded edit. */
export function redo(state: UndoState, current: BoxRecord[]): UndoStep | null {
  const boxes = state.future[state.future.length - 1];
  if (!boxes) return null;
  return {
    state: { past: [...state.past, copy(current)], future: state.future.slice(0, -1) },
    boxes: copy(boxes),
  };
}
