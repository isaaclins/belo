import { useCallback, useEffect, useRef, useState } from "react";

/**
 * Press and hold repeat for a button.
 *
 * A short press is one plain action, exactly as a click always was. Holding
 * past the threshold turns the button into a repeater until it is released or
 * the cap is reached, which is what keeps a teaching burst from becoming spam
 * clicking. Pointer events carry mouse, pen and touch, so the same handlers
 * work in the desktop webview and in WKWebView.
 */
export interface HoldRepeatOptions {
  /** Runs once per capture. Returning false means the capture did not happen. */
  onFire: () => boolean | void;
  /** Runs once when the interaction ends, with the number of captures it made. */
  onEnd?: (count: number) => void;
  /** How long a press has to last before it starts repeating. */
  holdMs?: number;
  /** Gap between repeats once the hold started. */
  intervalMs?: number;
  /** Hard ceiling for one hold, so a stuck pointer cannot fill a project. */
  cap?: number;
  /** False keeps the plain click behavior, with no repeat at all. */
  repeat?: boolean;
}

const PULSE_MS = 140;

export function useHoldRepeat({
  onFire,
  onEnd,
  holdMs = 350,
  intervalMs = 400,
  cap = 60,
  repeat = true,
}: HoldRepeatOptions) {
  const [holding, setHolding] = useState(false);
  const [count, setCount] = useState(0);
  const [capped, setCapped] = useState(false);
  const [pulse, setPulse] = useState(false);

  const fireRef = useRef(onFire);
  const endRef = useRef(onEnd);
  fireRef.current = onFire;
  endRef.current = onEnd;

  const holdTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const repeatTimer = useRef<ReturnType<typeof setInterval> | null>(null);
  const pulseTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const madeRef = useRef(0);
  /** True while the press could still turn out to be a plain single click. */
  const singleRef = useRef(false);
  // A pointer sequence already decided what to do, so the click it synthesizes
  // afterwards must not fire a second time. Keyboard activation has no pointer
  // sequence, which is exactly how it stays reachable.
  const fromPointer = useRef(false);

  const clearTimers = useCallback(() => {
    if (holdTimer.current) clearTimeout(holdTimer.current);
    if (repeatTimer.current) clearInterval(repeatTimer.current);
    holdTimer.current = null;
    repeatTimer.current = null;
  }, []);

  const fire = useCallback(() => {
    if (fireRef.current() === false) return false;
    madeRef.current += 1;
    setCount(madeRef.current);
    setPulse(true);
    if (pulseTimer.current) clearTimeout(pulseTimer.current);
    pulseTimer.current = setTimeout(() => setPulse(false), PULSE_MS);
    return true;
  }, []);

  const finish = useCallback(() => {
    clearTimers();
    const made = madeRef.current;
    madeRef.current = 0;
    setHolding(false);
    setCount(0);
    setCapped(false);
    if (made > 0) endRef.current?.(made);
  }, [clearTimers]);

  const begin = useCallback(() => {
    clearTimers();
    madeRef.current = 0;
    singleRef.current = true;
    setCount(0);
    setCapped(false);
    if (!repeat) return;
    holdTimer.current = setTimeout(() => {
      holdTimer.current = null;
      singleRef.current = false;
      setHolding(true);
      fire();
      repeatTimer.current = setInterval(() => {
        if (madeRef.current >= cap) {
          if (repeatTimer.current) clearInterval(repeatTimer.current);
          repeatTimer.current = null;
          setCapped(true);
          return;
        }
        fire();
      }, intervalMs);
    }, holdMs);
  }, [cap, clearTimers, fire, holdMs, intervalMs, repeat]);

  /** Release: a press that never reached the threshold is a single action. */
  const release = useCallback(() => {
    if (!fromPointer.current) return;
    const short = singleRef.current;
    singleRef.current = false;
    clearTimers();
    if (short) fire();
    finish();
    // The click the webview synthesizes after a pointer sequence arrives later.
    setTimeout(() => {
      fromPointer.current = false;
    }, 400);
  }, [clearTimers, finish, fire]);

  useEffect(
    () => () => {
      clearTimers();
      if (pulseTimer.current) clearTimeout(pulseTimer.current);
    },
    [clearTimers],
  );

  const handlers = {
    onPointerDown: (event: React.PointerEvent) => {
      if (event.button !== 0 && event.pointerType === "mouse") return;
      fromPointer.current = true;
      begin();
    },
    onPointerUp: () => release(),
    onPointerCancel: () => release(),
    onPointerLeave: () => release(),
    // Long press on iOS otherwise raises the selection callout over the button.
    onContextMenu: (event: React.MouseEvent) => event.preventDefault(),
    // Keyboard activation synthesizes a click with no pointer detail, which is
    // the one case that still has to act on its own.
    onClick: (event: React.MouseEvent) => {
      if (fromPointer.current || event.detail !== 0) return;
      singleRef.current = false;
      fire();
      finish();
    },
  };

  return { holding, count, capped, pulse, handlers };
}
