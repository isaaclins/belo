import { BoundingBox, Lightning } from "@phosphor-icons/react";
import { useStore } from "../lib/store";
import { boxesOf } from "../lib/types";

/**
 * Stand in for the Use stage of a detect project. Training works and the model
 * is saved with the project; running it on a live camera lands in the next
 * update, and this panel is the mode guard until it does.
 */
export function DetectSoonView() {
  const manifest = useStore((s) => s.manifest);
  const setView = useStore((s) => s.setView);
  const hasModel = useStore((s) => s.hasModel);

  const images = manifest?.images ?? [];
  const boxes = images.reduce((sum, image) => sum + boxesOf(image).length, 0);
  const boxed = images.filter((image) => boxesOf(image).length > 0).length;

  return (
    <div className="soon">
      <section className="soon__panel">
        <span className="eyebrow">03 Use</span>
        <span className="soon__mark">
          <BoundingBox size={34} weight="bold" />
        </span>
        <h2 className="soon__title">Live detection arrives with the next update.</h2>
        <p className="soon__text">
          {hasModel
            ? "Your detector is trained and saved with the project. The Train view shows what it finds on your own pictures until the camera can run it."
            : "Draw boxes on your pictures, then train the detector. The Train view shows what it finds on your own pictures."}
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

        <button className="btn btn--primary" onClick={() => setView("train")}>
          <Lightning size={16} weight="bold" /> Back to training
        </button>
      </section>
    </div>
  );
}
