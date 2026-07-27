import { useState } from "react";
import { ArrowRight, FolderOpen, Plus, Stack, Trash, Circuitry } from "@phosphor-icons/react";
import { useStore } from "../lib/store";
import { Modal } from "../components/ui";
import { storage } from "../lib/storage";
import type { ProjectSummary } from "../lib/types";

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

function RecentCard({ project }: { project: ProjectSummary }) {
  const openProject = useStore((s) => s.openProject);
  const forgetProject = useStore((s) => s.forgetProject);

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
          title="Remove from this list"
          aria-label={`Remove ${project.name} from the recent list`}
          onClick={() => void forgetProject(project.path)}
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
  const [naming, setNaming] = useState(false);
  const [name, setName] = useState("");

  const submit = () => {
    setNaming(false);
    void createProject(name);
    setName("");
  };

  return (
    <div className="home scroll">
      <section className="home__intro">
        <div className="home__mark" aria-hidden="true">
          <span className="home__lens" />
          <span className="home__chip" />
        </div>
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
          Projects live in {projectsDir ?? "your documents folder"}
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
              <RecentCard key={project.path} project={project} />
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
              Saved as a .belo folder in {projectsDir ?? "your documents folder"}
            </span>
          </div>
        </Modal>
      ) : null}
    </div>
  );
}
