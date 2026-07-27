import { useEffect, useRef, type ReactNode } from "react";
import { X } from "@phosphor-icons/react";
import { useStore } from "../lib/store";

export function Meter({ value, large = false }: { value: number; large?: boolean }) {
  const pct = Math.max(0, Math.min(1, value)) * 100;
  return (
    <div
      className={large ? "meter meter--lg" : "meter"}
      role="progressbar"
      aria-valuenow={Math.round(pct)}
      aria-valuemin={0}
      aria-valuemax={100}
    >
      <div className="meter__fill" style={{ width: `${pct}%` }} />
    </div>
  );
}

export function Modal({
  title,
  onClose,
  children,
  footer,
  wide = false,
}: {
  title: string;
  onClose: () => void;
  children: ReactNode;
  footer?: ReactNode;
  wide?: boolean;
}) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    // Move focus into the dialog, preferring the first real control over the
    // close button that sits earlier in the DOM.
    const scope = ref.current;
    const target =
      scope?.querySelector<HTMLElement>("input, select, textarea") ??
      scope?.querySelector<HTMLElement>(".modal__body button") ??
      scope?.querySelector<HTMLElement>("button");
    target?.focus();
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div className="overlay" onMouseDown={onClose}>
      <div
        className="modal"
        style={wide ? { width: "min(880px, 100%)" } : undefined}
        role="dialog"
        aria-modal="true"
        aria-label={title}
        ref={ref}
        onMouseDown={(event) => event.stopPropagation()}
      >
        <div className="modal__head">
          <h2>{title}</h2>
          <button className="iconbtn" onClick={onClose} aria-label="Close dialog">
            <X size={18} weight="bold" />
          </button>
        </div>
        <div className="modal__body">{children}</div>
        {footer ? <div className="modal__foot">{footer}</div> : null}
      </div>
    </div>
  );
}

export function EmptyState({
  icon,
  title,
  text,
  action,
}: {
  icon: ReactNode;
  title: string;
  text: string;
  action?: ReactNode;
}) {
  return (
    <div className="empty">
      <div className="empty__mark">{icon}</div>
      <h2>{title}</h2>
      <p className="empty__text">{text}</p>
      {action}
    </div>
  );
}

export function Toasts() {
  const toasts = useStore((s) => s.toasts);
  if (toasts.length === 0) return null;
  return (
    <div className="toasts" aria-live="polite">
      {toasts.map((toast) => (
        <div
          key={toast.id}
          className={
            toast.tone === "bad" ? "toast toast--bad" : toast.tone === "good" ? "toast toast--good" : "toast"
          }
        >
          {toast.text}
        </div>
      ))}
    </div>
  );
}

export function BusyBar() {
  const busy = useStore((s) => s.busy);
  if (!busy) return null;
  return (
    <div className="busybar">
      <span className="eyebrow">{busy.title}</span>
      <Meter value={busy.total === 0 ? 0 : busy.done / busy.total} />
      <span className="mono busybar__count">
        {busy.done}/{busy.total}
      </span>
    </div>
  );
}

/** Mirrors --swatch-0 through --swatch-7 in styles/tokens.css. */
const SWATCH_HEX = [
  "#ffd83d",
  "#2f6bff",
  "#ff5c8a",
  "#17c26a",
  "#ff8a1f",
  "#8b5cf6",
  "#00c2c7",
  "#ff4a3d",
];

/** Mirrors --ink, for the canvas which cannot read CSS custom properties. */
export const INK_HEX = "#111014";

export function swatch(index: number): string {
  return `var(--swatch-${index % SWATCH_HEX.length})`;
}

/** The same ramp as a literal color, for canvas drawing. */
export function swatchHex(index: number): string {
  return SWATCH_HEX[index % SWATCH_HEX.length];
}

export function pct(value: number): string {
  return `${Math.round(value * 100)}%`;
}
