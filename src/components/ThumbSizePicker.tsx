// Thumbnail size control for the image grid: minus packs more per row, plus
// makes each one bigger. Four steps, remembered across restarts.

import { Minus, Plus } from "@phosphor-icons/react";
import { stepThumbSize, THUMB_SIZES, useThumbSize } from "../lib/thumbSize";

export function ThumbSizePicker() {
  const size = useThumbSize();
  const index = THUMB_SIZES.indexOf(size);

  return (
    <div className="thumbsize" role="group" aria-label="Thumbnail size">
      <button
        className="iconbtn thumbsize__btn"
        disabled={index <= 0}
        title="Smaller thumbnails"
        aria-label="Smaller thumbnails"
        onClick={() => stepThumbSize(-1)}
      >
        <Minus size={16} weight="bold" />
      </button>
      <span className="thumbsize__step mono" aria-hidden="true">
        {index + 1}/{THUMB_SIZES.length}
      </span>
      <button
        className="iconbtn thumbsize__btn"
        disabled={index >= THUMB_SIZES.length - 1}
        title="Bigger thumbnails"
        aria-label="Bigger thumbnails"
        onClick={() => stepThumbSize(1)}
      >
        <Plus size={16} weight="bold" />
      </button>
    </div>
  );
}
