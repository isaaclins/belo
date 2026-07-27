import { useMemo, useRef, useState } from "react";
import {
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
import { EmptyState, Modal, swatch } from "../components/ui";
import { filesFromDataTransfer } from "../lib/images";
import { storage } from "../lib/storage";
import { CaptureModal } from "./CaptureModal";
import type { ImageRecord } from "../lib/types";

const UNLABELED = "__unlabeled__";

export function LabelView() {
  const manifest = useStore((s) => s.manifest);
  const path = useStore((s) => s.path);
  const activeLabelId = useStore((s) => s.activeLabelId);
  const setActiveLabel = useStore((s) => s.setActiveLabel);
  const addLabel = useStore((s) => s.addLabel);
  const renameLabel = useStore((s) => s.renameLabel);
  const deleteLabel = useStore((s) => s.deleteLabel);
  const addFiles = useStore((s) => s.addFiles);
  const importFolder = useStore((s) => s.importFolder);
  const moveImages = useStore((s) => s.moveImages);
  const removeImages = useStore((s) => s.removeImages);
  const toast = useStore((s) => s.toast);

  const [selection, setSelection] = useState<string[]>([]);
  const [dropping, setDropping] = useState(false);
  const [capturing, setCapturing] = useState(false);
  const [renaming, setRenaming] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [draft, setDraft] = useState("");
  const fileInput = useRef<HTMLInputElement>(null);

  const labels = manifest?.labels ?? [];
  const images = manifest?.images ?? [];
  const bucket = activeLabelId ?? UNLABELED;
  const activeLabel = labels.find((l) => l.id === activeLabelId) ?? null;

  const shown = useMemo<ImageRecord[]>(() => {
    const target = bucket === UNLABELED ? null : bucket;
    return images.filter((img) => img.labelId === target).slice().reverse();
  }, [images, bucket]);

  const unlabeledCount = images.filter((img) => img.labelId === null).length;
  const countFor = (id: string) => images.filter((img) => img.labelId === id).length;

  const targetLabelId = bucket === UNLABELED ? null : bucket;

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
          <span className="eyebrow">Labels</span>
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
                <h2>{activeLabel ? activeLabel.name : "Unlabeled"}</h2>
                <span className="badge badge--idle">{shown.length}</span>
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
            <button className="btn btn--sm" onClick={() => fileInput.current?.click()}>
              <FilePlus size={16} weight="bold" /> Files
            </button>
            <button className="btn btn--sm" onClick={() => void pickFolder()}>
              <FolderPlus size={16} weight="bold" /> Folder
            </button>
            <button
              className="btn btn--sm btn--primary"
              disabled={!activeLabel}
              title={activeLabel ? "Capture from the webcam" : "Pick a label first"}
              onClick={() => setCapturing(true)}
            >
              <Camera size={16} weight="bold" /> Capture
            </button>
            {activeLabel ? (
              <button
                className="iconbtn iconbtn--danger"
                title="Delete this label and its images"
                aria-label="Delete this label and its images"
                onClick={() => setConfirmDelete(true)}
              >
                <Trash size={16} weight="bold" />
              </button>
            ) : null}
          </div>
        </header>

        {selection.length > 0 ? (
          <div className="seltools">
            <span className="eyebrow">{selection.length} selected</span>
            <span className="seltools__label">Move to</span>
            <div className="seltools__chips">
              {labels
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
                  Unlabeled
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
              icon={bucket === UNLABELED ? <Question size={30} weight="bold" /> : <Images size={30} weight="bold" />}
              title={labels.length === 0 ? "Add a label first" : "Drop images here"}
              text={
                labels.length === 0
                  ? "Labels are the classes your model learns. Two or more, with about 30 images each, is a good start."
                  : "Drag in files or whole folders, import from disk, or hold the capture button to grab a burst from your webcam."
              }
              action={
                labels.length === 0 ? (
                  <button className="btn btn--primary" onClick={onNewLabel}>
                    <Plus size={16} weight="bold" /> Add a label
                  </button>
                ) : (
                  <button className="btn" onClick={() => fileInput.current?.click()}>
                    <UploadSimple size={16} weight="bold" /> Choose files
                  </button>
                )
              }
            />
          ) : (
            <div className="grid scroll">
              {shown.map((image) => {
                const on = selection.includes(image.id);
                return (
                  <button
                    key={image.id}
                    className={`tile${on ? " tile--on" : ""}`}
                    onClick={(event) => toggle(image.id, event.metaKey || event.ctrlKey || event.shiftKey)}
                    aria-pressed={on}
                  >
                    <img
                      src={storage.imageUrl(path, image.file)}
                      alt=""
                      loading="lazy"
                      draggable={false}
                    />
                    {on ? (
                      <span className="tile__check">
                        <Check size={14} weight="bold" />
                      </span>
                    ) : null}
                    {image.source === "capture" ? <span className="tile__tag">cam</span> : null}
                    {image.source === "feedback" ? <span className="tile__tag">fix</span> : null}
                  </button>
                );
              })}
            </div>
          )}
          {dropping ? <div className="drop__veil">Release to import</div> : null}
        </div>
      </section>

      <input
        ref={fileInput}
        className="sr-only"
        type="file"
        accept="image/*"
        multiple
        onChange={(event) => {
          const files = Array.from(event.target.files ?? []);
          event.target.value = "";
          if (files.length > 0) void addFiles(files, targetLabelId);
        }}
      />

      {capturing && activeLabel ? (
        <CaptureModal label={activeLabel} onClose={() => setCapturing(false)} />
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
          <p>
            This removes the label and deletes its {countFor(activeLabel.id)} image
            {countFor(activeLabel.id) === 1 ? "" : "s"} from the project folder. It cannot be
            undone.
          </p>
        </Modal>
      ) : null}
    </div>
  );
}
