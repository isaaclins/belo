// The one Belo mark.
//
// Same geometry as src/assets/logo.svg and as the raster master that
// scripts/make-icon.mjs feeds to `tauri icon`, so the top bar, the home hero,
// the dock icon and the home screen icon are the same drawing at different
// sizes. Change the numbers here and you must change them in both those files.

export function BeloMark({
  size = 32,
  onDark = false,
  className,
}: {
  size?: number;
  /** Swaps the outer frame to paper so the mark still reads on the ink top bar. */
  onDark?: boolean;
  className?: string;
}) {
  const frame = onDark ? "#f4efe4" : "#111014";

  return (
    <svg
      className={className}
      width={size}
      height={size}
      viewBox="0 0 512 512"
      shapeRendering="geometricPrecision"
      aria-hidden="true"
      focusable="false"
    >
      <rect width="512" height="512" fill="#ffd83d" />
      <circle cx="216" cy="216" r="116" fill="none" stroke="#111014" strokeWidth="64" />
      <rect x="302" y="302" width="132" height="132" fill="#ff5c8a" stroke="#111014" strokeWidth="20" />
      <rect x="20" y="20" width="472" height="472" fill="none" stroke={frame} strokeWidth="40" />
    </svg>
  );
}
