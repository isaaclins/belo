import { useEffect, useMemo, useState } from "react";
import {
  BoundingBox,
  Camera,
  Check,
  FilePlus,
  FolderPlus,
  Images,
  PencilSimple,
  Plus,
  Question,
  Trash,
  UploadSimple,
} from "@phosphor-icons/react";
import { useStore } from "../lib/store";
import { EmptyState, Modal, swatch, swatchHex } from "../components/ui";
import { ThumbSizePicker } from "../components/ThumbSizePicker";
import { useThumbSize } from "../lib/thumbSize";
import { filesFromDataTransfer } from "../lib/images";
import { storage } from "../lib/storage";
import { CaptureModal } from "./CaptureModal";
import { BoxAnnotator } from "./BoxAnnotator";
import { boxesOf, hasLabel, labelsOf, type ImageRecord, type Label } from "../lib/types";
import { isSelectAllShortcut, isTextEditingTarget } from "../lib/keyboard";

const UNLABELED = "__unlabeled__";

/**
 * Box outlines drawn straight onto a grid thumbnail, so progress is readable
 * without opening anything. Stored pictures are always square and the cells are
 * square too, so the picture fills the tile exactly and a plain 0..100 viewBox
 * lands on the same place the annotator would.
 */
function BoxSketch({ image, labels }: { image: ImageRecord; labels: Label[] }) {
  const boxes = boxesOf(image);
  if (boxes.length === 0) return null;
  return (
    <svg className="tile__sketch" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
      {boxes.map((box) => (
        <rect
          key={box.id}
          x={box.x * 100}
          y={box.y * 100}
          width={box.w * 100}
          height={box.h * 100}
          fill="none"
          stroke={swatchHex(labels.find((l) => l.id === box.labelId)?.color ?? 0)}
          strokeWidth={2}
          vectorEffect="non-scaling-stroke"
        />
      ))}
    </svg>
  );
}

export function LabelView() {
  const manifest = useStore((s) => s.manifest);
  const path = useStore((s) => s.path);
  const activeLabelId = useStore((s) => s.activeLabelId);
  const setActiveLabel = useStore((s) => s.setActiveLabel);
  const addLabel = useStore((s) => s.addLabel);
  const renameLabel = useStore((s) => s.renameLabel);
  const deleteLabel = useStore((s) => s.deleteLabel);
  const addFiles = useStore((s) => s.addFiles);
  const importImages = useStore((s) => s.importImages);
  const importFolder = useStore((s) => s.importFolder);
  const moveImages = useStore((s) => s.moveImages);
  const toggleImageLabels = useStore((s) => s.toggleImageLabels);
  const removeImages = useStore((s) => s.removeImages);
  const toast = useStore((s) => s.toast);

  const [selection, setSelection] = useState<string[]>([]);
  const [dropping, setDropping] = useState(false);
  const [capturing, setCapturing] = useState(false);
  const [renaming, setRenaming] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [draft, setDraft] = useState("");
  /** Detect mode: index into the shown grid of the picture being annotated. */
  const [annotating, setAnnotating] = useState<number | null>(null);

  const thumb = useThumbSize();

  const labels = manifest?.labels ?? [];
  const images = manifest?.images ?? [];
  const multi = manifest?.mode === "multi";
  const detect = manifest?.mode === "detect";
  const bucket = activeLabelId ?? UNLABELED;
  const activeLabel = labels.find((l) => l.id === activeLabelId) ?? null;

  // A detect project has one bucket: every picture. The label rail there picks
  // the pencil rather than a folder, because the labels sit on the boxes.
  const shown = useMemo<ImageRecord[]>(() => {
    if (detect) return images.slice().reverse();
    const target = bucket === UNLABELED ? null : bucket;
    return images
      .filter((img) => (target === null ? labelsOf(img).length === 0 : hasLabel(img, target)))
      .slice()
      .reverse();
  }, [images, bucket, detect]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (
        !isSelectAllShortcut(event) ||
        isTextEditingTarget(event.target) ||
        annotating !== null ||
        capturing ||
        confirmDelete
      ) {
        return;
      }
      event.preventDefault();
      setSelection(shown.map((image) => image.id));
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [annotating, capturing, confirmDelete, shown]);

  // The annotator walks the same list the grid shows, so a deletion behind it
  // must not leave it pointing past the end.
  useEffect(() => {
    if (annotating !== null && annotating >= shown.length) {
      setAnnotating(shown.length === 0 ? null : shown.length - 1);
    }
  }, [annotating, shown.length]);

  const unlabeledCount = images.filter((img) => labelsOf(img).length === 0).length;
  const countFor = (id: string) =>
    detect
      ? images.reduce((sum, img) => sum + boxesOf(img).filter((box) => box.labelId === id).length, 0)
      : images.filter((img) => hasLabel(img, id)).length;
  const sharedWith = (id: string) =>
    selection.filter((imageId) => {
      const image = images.find((img) => img.id === imageId);
      return image ? hasLabel(image, id) : false;
    }).length;

  // Pictures arrive unlabeled in a detect project: what they hold is decided by
  // the boxes drawn on them afterwards.
  const targetLabelId = detect || bucket === UNLABELED ? null : bucket;
  const soleCount = activeLabel
    ? images.filter((img) => hasLabel(img, activeLabel.id) && labelsOf(img).length === 1).length
    : 0;

  const onNewLabel = () => {
    const name = `Label ${labels.length + 1}`;
    const id = addLabel(name);
    if (id) {
      setRenaming(true);
      setDraft(name);
    }
  };

  const commitRename = () => {
    if (activeLabel && draft.trim()) renameLabel(activeLabel.id, draft);
    setRenaming(false);
  };

  const handleDrop = async (event: React.DragEvent) => {
    event.preventDefault();
    setDropping(false);
    const files = await filesFromDataTransfer(event.dataTransfer);
    if (files.length === 0) {
      toast("No readable images in that drop", "bad");
      return;
    }
    const added = await addFiles(files, targetLabelId);
    if (added > 0) toast(`Added ${added} image${added === 1 ? "" : "s"}`, "good");
  };

  const pickImages = async () => {
    const added = await importImages(targetLabelId);
    if (added > 0) toast(`Added ${added} image${added === 1 ? "" : "s"}`, "good");
  };

  const pickFolder = async () => {
    const added = await importFolder(targetLabelId);
    if (added > 0) toast(`Added ${added} image${added === 1 ? "" : "s"}`, "good");
  };

  const toggle = (id: string, additive: boolean) => {
    setSelection((prev) => {
      if (!additive) return prev.includes(id) && prev.length === 1 ? [] : [id];
      return prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id];
    });
  };

  if (!manifest || !path) return null;

  return (
    <div className="label">
      <aside className="rail">
        <div className="rail__head">
          <span className="eyebrow">{detect ? "Pencils" : "Labels"}</span>
          <button className="iconbtn" onClick={onNewLabel} title="Add a label" aria-label="Add a label">
            <Plus size={16} weight="bold" />
          </button>
        </div>

        <ul className="rail__list scroll">
          {labels.map((label) => (
            <li key={label.id}>
              <button
                className={`railitem${activeLabelId === label.id ? " railitem--on" : ""}`}
                onClick={() => {
                  setActiveLabel(label.id);
                  setSelection([]);
                  setRenaming(false);
                }}
              >
                <span className="dot" style={{ background: swatch(label.color) }} />
                <span className="railitem__name">{label.name}</span>
                <span className="railitem__count mono">{countFor(label.id)}</span>
              </button>
            </li>
          ))}
          {detect ? null : (
            <li>
              <button
                className={`railitem${bucket === UNLABELED ? " railitem--on" : ""}`}
                onClick={() => {
                  setActiveLabel(null);
                  setSelection([]);
                  setRenaming(false);
                }}
              >
                <span className="dot dot--empty" />
                <span className="railitem__name">Unlabeled</span>
                <span className="railitem__count mono">{unlabeledCount}</span>
              </button>
            </li>
          )}
        </ul>

        <div className="rail__foot mono">
          {images.length} image{images.length === 1 ? "" : "s"} in this project
        </div>
      </aside>

      <section className="pane">
        <header className="pane__head">
          <div className="pane__title">
            {renaming && activeLabel ? (
              <input
                className="input pane__rename"
                value={draft}
                autoFocus
                onChange={(event) => setDraft(event.target.value)}
                onBlur={commitRename}
                onKeyDown={(event) => {
                  if (event.key === "Enter") commitRename();
                  if (event.key === "Escape") setRenaming(false);
                }}
              />
            ) : (
              <>
                {activeLabel ? (
                  <span className="pane__swatch" style={{ background: swatch(activeLabel.color) }} />
                ) : null}
                <h2>{detect ? "All pictures" : activeLabel ? activeLabel.name : "Unlabeled"}</h2>
                <span className="badge badge--idle">{shown.length}</span>
                {detect ? (
                  <span className="pane__pencil">
                    {activeLabel ? `drawing with ${activeLabel.name}` : "add a label to draw with"}
                  </span>
                ) : null}
                {activeLabel ? (
                  <button
                    className="iconbtn"
                    title="Rename label"
                    aria-label="Rename label"
                    onClick={() => {
                      setDraft(activeLabel.name);
                      setRenaming(true);
                    }}
                  >
                    <PencilSimple size={16} weight="bold" />
                  </button>
                ) : null}
              </>
            )}
          </div>

          <div className="row row--tight">
            <ThumbSizePicker />
            <button className="btn btn--sm" onClick={() => void pickImages()}>
              <FilePlus size={16} weight="bold" /> {storage.mobile ? "Photos" : "Files"}
            </button>
            {storage.canPickFolders ? (
              <button className="btn btn--sm" onClick={() => void pickFolder()}>
                <FolderPlus size={16} weight="bold" /> Folder
              </button>
            ) : null}
            <button
              className="btn btn--sm btn--primary"
              disabled={!activeLabel && !detect}
              title={activeLabel || detect ? "Capture from the webcam" : "Pick a label first"}
              onClick={() => setCapturing(true)}
            >
              <Camera size={16} weight="bold" /> Capture
            </button>
            {activeLabel ? (
              <button
                className="iconbtn iconbtn--danger"
                title={detect ? "Delete this label and its boxes" : "Delete this label and its images"}
                aria-label={
                  detect ? "Delete this label and its boxes" : "Delete this label and its images"
                }
                onClick={() => setConfirmDelete(true)}
              >
                <Trash size={16} weight="bold" />
              </button>
            ) : null}
          </div>
        </header>

        {selection.length > 0 && detect ? (
          <div className="seltools">
            <span className="eyebrow">{selection.length} selected</span>
            {/* No move and no label chips here: in a detect project a label
                belongs to a box, not to a whole picture. */}
            <span className="seltools__note">
              Labels live on the boxes you draw, so there is nothing to move a picture into. Open a
              picture to box what is inside it.
            </span>
            <div className="spacer" />
            <button
              className="btn btn--sm btn--danger"
              onClick={() => {
                void removeImages(selection);
                setSelection([]);
              }}
            >
              <Trash size={15} weight="bold" /> Delete
            </button>
            <button className="btn btn--sm" onClick={() => setSelection([])}>
              Clear
            </button>
          </div>
        ) : null}

        {selection.length > 0 && !detect ? (
          <div className="seltools">
            <span className="eyebrow">{selection.length} selected</span>
            <span className="seltools__label">{multi ? "Labels" : "Move to"}</span>
            <div className="seltools__chips">
              {multi
                ? labels.map((label) => {
                    const on = sharedWith(label.id) === selection.length;
                    const some = !on && sharedWith(label.id) > 0;
                    return (
                      <button
                        key={label.id}
                        className={`chip${on ? " chip--on" : some ? " chip--part" : ""}`}
                        style={on || some ? { background: swatch(label.color) } : undefined}
                        aria-pressed={on}
                        title={on ? `Remove ${label.name}` : `Add ${label.name}`}
                        onClick={() => toggleImageLabels(selection, label.id)}
                      >
                        {on ? <Check size={13} weight="bold" /> : <Plus size={13} weight="bold" />}
                        {label.name}
                      </button>
                    );
                  })
                : labels
                    .filter((label) => label.id !== activeLabelId)
                    .map((label) => (
                      <button
                        key={label.id}
                        className="chip"
                        style={{ background: swatch(label.color) }}
                        onClick={() => {
                          moveImages(selection, label.id);
                          setSelection([]);
                        }}
                      >
                        {label.name}
                      </button>
                    ))}
              {bucket !== UNLABELED ? (
                <button
                  className="chip chip--ghost"
                  onClick={() => {
                    moveImages(selection, null);
                    setSelection([]);
                  }}
                >
                  {multi ? "Clear all labels" : "Unlabeled"}
                </button>
              ) : null}
            </div>
            <div className="spacer" />
            <button
              className="btn btn--sm btn--danger"
              onClick={() => {
                void removeImages(selection);
                setSelection([]);
              }}
            >
              <Trash size={15} weight="bold" /> Delete
            </button>
            <button className="btn btn--sm" onClick={() => setSelection([])}>
              Clear
            </button>
          </div>
        ) : null}

        <div
          className={`drop${dropping ? " drop--on" : ""}`}
          onDragOver={(event) => {
            event.preventDefault();
            setDropping(true);
          }}
          onDragLeave={() => setDropping(false)}
          onDrop={(event) => void handleDrop(event)}
        >
          {shown.length === 0 ? (
            <EmptyState
              icon={
                detect ? (
                  <BoundingBox size={30} weight="bold" />
                ) : bucket === UNLABELED ? (
                  <Question size={30} weight="bold" />
                ) : (
                  <Images size={30} weight="bold" />
                )
              }
              title={labels.length === 0 ? "Add a label first" : "Drop images here"}
              text={
                labels.length === 0
                  ? detect
                    ? "Labels are the things Belo should find. Add one per kind of object, then draw a box around every one you see."
                    : "Labels are the classes your model learns. Two or more, with about 30 images each, is a good start."
                  : storage.mobile
                    ? "Choose photos from this device, or hold the capture button to grab a burst from the camera."
                    : "Drag in files or whole folders, import from disk, or hold the capture button to grab a burst from your webcam."
              }
              action={
                labels.length === 0 ? (
                  <button className="btn btn--primary" onClick={onNewLabel}>
                    <Plus size={16} weight="bold" /> Add a label
                  </button>
                ) : (
                  <button className="btn" onClick={() => void pickImages()}>
                    <UploadSimple size={16} weight="bold" />
                    {storage.mobile ? "Choose photos" : "Choose files"}
                  </button>
                )
              }
            />
          ) : (
            <div
              className="grid scroll"
              style={{ "--cell": `${thumb}px` } as React.CSSProperties}
            >
              {shown.map((image, position) => {
                const on = selection.includes(image.id);
                const boxes = boxesOf(image);
                const face = (
                  <>
                    <img
                      src={storage.imageUrl(path, image.file)}
                      alt=""
                      loading="lazy"
                      decoding="async"
                      draggable={false}
                    />
                    {detect ? <BoxSketch image={image} labels={labels} /> : null}
                    {image.source === "capture" ? (
                      <span className="tile__tag">{image.feed === "screen" ? "screen" : "cam"}</span>
                    ) : null}
                    {image.source === "feedback" ? (
                      <span className="tile__tag">{image.feed === "screen" ? "screen fix" : "fix"}</span>
                    ) : null}
                    {multi && labelsOf(image).length > 1 ? (
                      <span className="tile__labels">
                        {labelsOf(image).map((id) => (
                          <span
                            key={id}
                            className="dot dot--mini"
                            title={labels.find((l) => l.id === id)?.name ?? ""}
                            style={{
                              background: swatch(labels.find((l) => l.id === id)?.color ?? 0),
                            }}
                          />
                        ))}
                      </span>
                    ) : null}
                  </>
                );

                // Detect mode: the tile itself opens the annotator, so picking
                // pictures for deletion moves onto its own corner control.
                if (detect) {
                  return (
                    <div key={image.id} className={`tile tile--open${on ? " tile--on" : ""}`}>
                      <button
                        className="tile__face"
                        title="Draw boxes on this picture"
                        aria-label={`Draw boxes on picture ${position + 1}`}
                        onClick={() => setAnnotating(position)}
                      >
                        {face}
                      </button>
                      <button
                        className={`tile__pick${on ? " tile__pick--on" : ""}`}
                        aria-pressed={on}
                        title={on ? "Deselect this picture" : "Select this picture"}
                        aria-label={on ? "Deselect this picture" : "Select this picture"}
                        onClick={() => toggle(image.id, true)}
                      >
                        <Check size={14} weight="bold" />
                      </button>
                      <span
                        className={`tile__boxcount mono${boxes.length === 0 ? " tile__boxcount--none" : ""}`}
                      >
                        {boxes.length === 0 ? "none" : `${boxes.length} box${boxes.length === 1 ? "" : "es"}`}
                      </span>
                    </div>
                  );
                }

                return (
                  <button
                    key={image.id}
                    className={`tile${on ? " tile--on" : ""}`}
                    onClick={(event) => toggle(image.id, event.metaKey || event.ctrlKey || event.shiftKey)}
                    aria-pressed={on}
                  >
                    {face}
                    {on ? (
                      <span className="tile__check">
                        <Check size={14} weight="bold" />
                      </span>
                    ) : null}
                  </button>
                );
              })}
            </div>
          )}
          {dropping ? <div className="drop__veil">Release to import</div> : null}
        </div>
      </section>

      {annotating !== null && shown[annotating] ? (
        <BoxAnnotator
          images={shown}
          index={annotating}
          onIndex={setAnnotating}
          onClose={() => setAnnotating(null)}
        />
      ) : null}

      {capturing && (activeLabel || detect) ? (
        <CaptureModal
          label={detect ? null : activeLabel}
          onClose={() => setCapturing(false)}
        />
      ) : null}

      {confirmDelete && activeLabel ? (
        <Modal
          title={`Delete ${activeLabel.name}`}
          onClose={() => setConfirmDelete(false)}
          footer={
            <>
              <button className="btn" onClick={() => setConfirmDelete(false)}>
                Keep it
              </button>
              <button
                className="btn btn--danger"
                onClick={() => {
                  setConfirmDelete(false);
                  setSelection([]);
                  void deleteLabel(activeLabel.id);
                }}
              >
                <Trash size={16} weight="bold" /> Delete label
              </button>
            </>
          }
        >
          {detect ? (
            <p>
              This removes the label and the {countFor(activeLabel.id)} box
              {countFor(activeLabel.id) === 1 ? "" : "es"} drawn with it. Your pictures all stay
              where they are. It cannot be undone.
            </p>
          ) : multi ? (
            <p>
              This removes the label from {countFor(activeLabel.id)} image
              {countFor(activeLabel.id) === 1 ? "" : "s"} and deletes the {soleCount} that carry no
              other label. It cannot be undone.
            </p>
          ) : (
            <p>
              This removes the label and deletes its {countFor(activeLabel.id)} image
              {countFor(activeLabel.id) === 1 ? "" : "s"} from the project folder. It cannot be
              undone.
            </p>
          )}
        </Modal>
      ) : null}
    </div>
  );
}
