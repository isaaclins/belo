// Page zoom guard.
//
// Belo is an app window, not a document, so user initiated zoom only breaks the
// layout. The viewport meta covers most mobile browsers, but WKWebView happily
// ignores it for pinch and double tap, and desktop webviews still honour the
// keyboard and ctrl plus wheel shortcuts. This module kills all of those.
// Responsive layout is untouched: nothing here reacts to window size.

/** Safari only gesture events, not in the DOM lib typings. */
const GESTURE_EVENTS = ["gesturestart", "gesturechange", "gestureend"] as const;

const DOUBLE_TAP_MS = 320;

function block(event: Event) {
  event.preventDefault();
}

export function installZoomGuard(): () => void {
  const cleanups: (() => void)[] = [];

  const on = <T extends Event>(
    target: EventTarget,
    type: string,
    handler: (event: T) => void,
    options?: AddEventListenerOptions,
  ) => {
    const listener = handler as EventListener;
    target.addEventListener(type, listener, options);
    cleanups.push(() => target.removeEventListener(type, listener, options));
  };

  // iOS pinch zoom inside WKWebView.
  for (const type of GESTURE_EVENTS) {
    on(document, type, block, { passive: false });
  }

  // Ctrl plus wheel and trackpad pinch on desktop webviews.
  on<WheelEvent>(
    window,
    "wheel",
    (event) => {
      if (event.ctrlKey || event.metaKey) event.preventDefault();
    },
    { passive: false, capture: true },
  );

  // Ctrl or Cmd with plus, minus and zero.
  on<KeyboardEvent>(
    window,
    "keydown",
    (event) => {
      if (!(event.ctrlKey || event.metaKey)) return;
      if (["+", "-", "=", "_", "0"].includes(event.key)) event.preventDefault();
    },
    { capture: true },
  );

  // Double tap zoom fallback for webviews that ignore user-scalable=no.
  let lastTouchEnd = 0;
  on<TouchEvent>(
    document,
    "touchend",
    (event) => {
      const now = Date.now();
      if (now - lastTouchEnd < DOUBLE_TAP_MS) event.preventDefault();
      lastTouchEnd = now;
    },
    { passive: false },
  );

  // Any multi finger touch is a zoom attempt, never app input.
  on<TouchEvent>(
    document,
    "touchmove",
    (event) => {
      if (event.touches.length > 1) event.preventDefault();
    },
    { passive: false },
  );

  return () => cleanups.forEach((fn) => fn());
}
