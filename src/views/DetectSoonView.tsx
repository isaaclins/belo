import { BoundingBox, PencilSimple } from "@phosphor-icons/react";
import { useStore } from "../lib/store";
import { boxesOf } from "../lib/types";

/**
 * Stand in for the Train and Use stages of a detect project. The detection head
 * lands in a later update, so this panel is the mode guard: App routes detect
 * projects here instead of the classifier views, and the real views slot in by
 * replacing these two branches.
 */
export function DetectSoonView({ stage }: { stage: "train" | "use" }) {
  const manifest = useStore((s) => s.manifest);
  const setView = useStore((s) => s.setView);

  const images = manifest?.images ?? [];
  const boxes = images.reduce((sum, image) => sum + boxesOf(image).length, 0);
  const boxed = images.filter((image) => boxesOf(image).length > 0).length;

  return (
    <div className="soon">
      <section className="soon__panel">
        <span className="eyebrow">{stage === "train" ? "02 Train" : "03 Use"}</span>
        <span className="soon__mark">
          <BoundingBox size={34} weight="bold" />
        </span>
        <h2 className="soon__title">Detection training arrives with the next update.</h2>
        <p className="soon__text">
          Draw boxes on your images meanwhile so training has something to learn from.
        </p>

        <dl className="soon__stats">
          <div>
            <dt className="eyebrow">Pictures</dt>
            <dd className="mono">{images.length}</dd>
          </div>
          <div>
            <dt className="eyebrow">With boxes</dt>
            <dd className="mono">{boxed}</dd>
          </div>
          <div>
            <dt className="eyebrow">Boxes drawn</dt>
            <dd className="mono">{boxes}</dd>
          </div>
        </dl>

        <button className="btn btn--primary" onClick={() => setView("label")}>
          <PencilSimple size={16} weight="bold" /> Back to drawing
        </button>
      </section>
    </div>
  );
}
