import { useState } from "react";
import { ArrowRight, FolderOpen, Plus, Stack, Trash, Circuitry } from "@phosphor-icons/react";
import { useStore } from "../lib/store";
import { BeloMark } from "../components/BeloMark";
import { Modal } from "../components/ui";
import { storage } from "../lib/storage";
import type { ProjectMode, ProjectSummary } from "../lib/types";

const MODES: { id: ProjectMode; title: string; text: string }[] = [
  {
    id: "single",
    title: "One label per image",
    text: "Every image belongs to exactly one label and the model picks a single winner.",
  },
  {
    id: "multi",
    title: "Many labels per image",
    text: "An image can be X and Y but not Z. Each label gets its own yes or no.",
  },
  {
    id: "detect",
    title: "Find and box each object",
    text: "You draw a rectangle around every thing you care about, so one picture can hold several of them in different places.",
  },
];

function relativeTime(iso: string): string {
  if (!iso) return "never opened";
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return "unknown";
  const minutes = Math.round((Date.now() - then) / 60000);
  if (minutes < 1) return "just now";
  if (minutes < 60) return `${minutes} min ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours} h ago`;
  const days = Math.round(hours / 24);
  if (days < 30) return `${days} d ago`;
  return new Date(iso).toLocaleDateString();
}

function RecentCard({
  project,
  onDelete,
}: {
  project: ProjectSummary;
  onDelete: (project: ProjectSummary) => void;
}) {
  const openProject = useStore((s) => s.openProject);
  const openPath = useStore((s) => s.path);
  const isOpen = openPath === project.path;

  return (
    <li className="recent">
      <button className="recent__open" onClick={() => void openProject(project.path)}>
        <span className="recent__name">{project.name}</span>
        <span className="recent__stats mono">
          {project.labelCount} labels · {project.imageCount} images ·{" "}
          {project.trained ? "trained" : "untrained"}
        </span>
        <span className="recent__path mono">{project.path}</span>
      </button>
      <div className="recent__side">
        <span className="eyebrow">{relativeTime(project.updatedAt)}</span>
        <button
          className="iconbtn iconbtn--danger"
          title={isOpen ? "This project is open" : "Delete this project"}
          aria-label={`Delete the project ${project.name}`}
          disabled={isOpen}
          onClick={() => onDelete(project)}
        >
          <Trash size={16} weight="bold" />
        </button>
      </div>
    </li>
  );
}

export function HomeView() {
  const recents = useStore((s) => s.recents);
  const projectsDir = useStore((s) => s.projectsDir);
  const createProject = useStore((s) => s.createProject);
  const openProjectDialog = useStore((s) => s.openProjectDialog);
  const deleteProject = useStore((s) => s.deleteProject);
  const [naming, setNaming] = useState(false);
  const [name, setName] = useState("");
  const [mode, setMode] = useState<ProjectMode>("single");
  const [doomed, setDoomed] = useState<ProjectSummary | null>(null);
  const [deleting, setDeleting] = useState(false);

  const submit = () => {
    setNaming(false);
    void createProject(name, mode);
    setName("");
    setMode("single");
  };

  return (
    <div className="home scroll">
      <section className="home__intro">
        <BeloMark size={92} className="home__mark" />
        <h1 className="home__title">
          Teach your
          <br />
          machine to see.
        </h1>
        <p className="home__lede">
          Collect images, label them, train a classifier on this device. No account, no upload, no
          cloud.
        </p>
        <div className="home__actions">
          <button className="btn btn--lg btn--primary" onClick={() => setNaming(true)}>
            <Plus size={18} weight="bold" /> New project
          </button>
          {storage.canPickProjects ? (
            <button className="btn btn--lg" onClick={() => void openProjectDialog()}>
              <FolderOpen size={18} weight="bold" /> Open .belo
            </button>
          ) : null}
        </div>
        <p className="home__dir mono">
          {storage.mobile
            ? "Projects stay in Belo's on-device documents"
            : `Projects live in ${projectsDir ?? "your documents folder"}`}
        </p>
      </section>

      <section className="home__recents">
        <div className="home__recentshead">
          <h2>Your projects</h2>
          <span className="badge badge--idle">{recents.length}</span>
        </div>
        {recents.length === 0 ? (
          <div className="home__none">
            <Circuitry size={28} weight="bold" />
            <p>
              Nothing here yet. A project holds your labels, your images and the classifier trained
              from them.
            </p>
            <button className="btn btn--primary" onClick={() => setNaming(true)}>
              Start the first one <ArrowRight size={16} weight="bold" />
            </button>
          </div>
        ) : (
          <ul className="home__list">
            {recents.map((project) => (
              <RecentCard key={project.path} project={project} onDelete={setDoomed} />
            ))}
          </ul>
        )}
      </section>

      {naming ? (
        <Modal
          title="New project"
          onClose={() => setNaming(false)}
          footer={
            <>
              <button className="btn" onClick={() => setNaming(false)}>
                Cancel
              </button>
              <button className="btn btn--primary" onClick={submit} disabled={!name.trim()}>
                <Stack size={16} weight="bold" /> Create
              </button>
            </>
          }
        >
          <div className="field">
            <label className="field__label" htmlFor="project-name">
              Project name
            </label>
            <input
              id="project-name"
              className="input"
              value={name}
              placeholder="Bottle caps"
              autoComplete="off"
              onChange={(event) => setName(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && name.trim()) submit();
              }}
            />
            <span className="field__hint">
              {storage.mobile
                ? "Saved in Belo's on-device documents"
                : `Saved as a .belo folder in ${projectsDir ?? "your documents folder"}`}
            </span>
          </div>

          <div className="field">
            <span className="field__label">Classification mode</span>
            <div className="modepick">
              {MODES.map((option) => (
                <button
                  key={option.id}
                  type="button"
                  className={`modepick__opt${mode === option.id ? " modepick__opt--on" : ""}`}
                  aria-pressed={mode === option.id}
                  onClick={() => setMode(option.id)}
                >
                  <span className="modepick__title">{option.title}</span>
                  <span className="modepick__text">{option.text}</span>
                </button>
              ))}
            </div>
            <span className="field__hint">This is fixed once the project exists.</span>
          </div>
        </Modal>
      ) : null}

      {doomed ? (
        <Modal
          title={`Delete ${doomed.name}?`}
          onClose={() => (deleting ? undefined : setDoomed(null))}
          footer={
            <>
              <button className="btn" disabled={deleting} onClick={() => setDoomed(null)}>
                Keep it
              </button>
              <button
                className="btn btn--danger"
                disabled={deleting}
                onClick={async () => {
                  setDeleting(true);
                  await deleteProject(doomed.path);
                  setDeleting(false);
                  setDoomed(null);
                }}
              >
                <Trash size={16} weight="bold" /> {deleting ? "Deleting" : "Delete forever"}
              </button>
            </>
          }
        >
          <div className="stack">
            <p className="danger-note">
              <strong>{doomed.name}</strong> and everything inside it goes for good:{" "}
              {doomed.imageCount} image{doomed.imageCount === 1 ? "" : "s"}, {doomed.labelCount}{" "}
              label{doomed.labelCount === 1 ? "" : "s"}
              {doomed.trained ? " and the trained model" : ""}. This cannot be undone and there is
              no trash to fish it out of.
            </p>
            <p className="field__hint mono">{doomed.path}</p>
          </div>
        </Modal>
      ) : null}
    </div>
  );
}
