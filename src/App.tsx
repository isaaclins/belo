import { useEffect, useState } from "react";
import { CheckCircle, CircleNotch, FloppyDisk, SignOut } from "@phosphor-icons/react";
import { useStore, type View } from "./lib/store";
import { BusyBar, Toasts } from "./components/ui";
import { HomeView } from "./views/HomeView";
import { LabelView } from "./views/LabelView";
import { TrainView } from "./views/TrainView";
import { UseView } from "./views/UseView";

const TABS: { id: View; label: string; index: string }[] = [
  { id: "label", label: "Label", index: "01" },
  { id: "train", label: "Train", index: "02" },
  { id: "use", label: "Use", index: "03" },
];

function SaveState() {
  const saving = useStore((s) => s.saving);
  const dirty = useStore((s) => s.dirty);
  const saveNow = useStore((s) => s.saveNow);

  return (
    <button
      className="savestate"
      onClick={() => void saveNow()}
      title="Projects save themselves, this forces a write right now"
    >
      {saving ? (
        <>
          <CircleNotch size={14} weight="bold" className="spin" />
          <span className="savestate__label">Saving</span>
        </>
      ) : dirty ? (
        <>
          <FloppyDisk size={14} weight="bold" />
          <span className="savestate__label">Unsaved</span>
        </>
      ) : (
        <>
          <CheckCircle size={14} weight="bold" />
          <span className="savestate__label">Saved</span>
        </>
      )}
    </button>
  );
}

/** Quiet marker for the debounced background retrain. Never blocks anything. */
function RetrainFlag() {
  const autoRetraining = useStore((s) => s.autoRetraining);
  if (!autoRetraining) return null;
  return (
    <span className="retrainflag" role="status" aria-live="polite" title="Folding your corrections into the model">
      <CircleNotch size={13} weight="bold" className="spin" />
      <span className="retrainflag__label">Retraining</span>
    </span>
  );
}

function ProjectName() {
  const manifest = useStore((s) => s.manifest);
  const renameProject = useStore((s) => s.renameProject);
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState("");

  if (!manifest) return null;

  if (editing) {
    return (
      <input
        className="input topbar__rename"
        value={draft}
        autoFocus
        onChange={(event) => setDraft(event.target.value)}
        onBlur={() => {
          renameProject(draft);
          setEditing(false);
        }}
        onKeyDown={(event) => {
          if (event.key === "Enter") {
            renameProject(draft);
            setEditing(false);
          }
          if (event.key === "Escape") setEditing(false);
        }}
      />
    );
  }

  return (
    <button
      className="topbar__project"
      title="Rename this project"
      onClick={() => {
        setDraft(manifest.name);
        setEditing(true);
      }}
    >
      {manifest.name}
    </button>
  );
}

export default function App() {
  const booted = useStore((s) => s.booted);
  const boot = useStore((s) => s.boot);
  const manifest = useStore((s) => s.manifest);
  const view = useStore((s) => s.view);
  const setView = useStore((s) => s.setView);
  const closeProject = useStore((s) => s.closeProject);
  const hasModel = useStore((s) => s.hasModel);
  const images = manifest?.images.length ?? 0;

  useEffect(() => {
    void boot();
  }, [boot]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (!(event.metaKey || event.ctrlKey)) return;
      if (event.key === "s") {
        event.preventDefault();
        void useStore.getState().saveNow();
      }
      if (event.key >= "1" && event.key <= "3" && useStore.getState().manifest) {
        event.preventDefault();
        setView(TABS[Number(event.key) - 1].id);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [setView]);

  if (!booted) {
    return (
      <div className="boot">
        <span className="boot__mark" />
        <span className="eyebrow">Starting Belo</span>
      </div>
    );
  }

  return (
    <div className="shell">
      <header className="topbar">
        <div className="topbar__brand">
          <span className="topbar__mark" aria-hidden="true" />
          <span className="topbar__word">Belo</span>
        </div>

        {manifest ? (
          <>
            <span className="topbar__sep" aria-hidden="true" />
            <ProjectName />
            <SaveState />
            <RetrainFlag />
            <div className="spacer" />
            <nav className="tabs" aria-label="Project stages">
              {TABS.map((tab) => {
                const blocked = tab.id === "use" && !hasModel;
                return (
                  <button
                    key={tab.id}
                    className={`tab${view === tab.id ? " tab--on" : ""}`}
                    aria-current={view === tab.id ? "page" : undefined}
                    onClick={() => setView(tab.id)}
                  >
                    <span className="tab__index mono">{tab.index}</span>
                    {tab.label}
                    {tab.id === "label" && images > 0 ? (
                      <span className="tab__count mono">{images}</span>
                    ) : null}
                    {blocked ? <span className="tab__count mono">off</span> : null}
                  </button>
                );
              })}
            </nav>
            <button className="btn btn--sm topbar__close" onClick={closeProject} aria-label="Close project">
              <SignOut size={15} weight="bold" /> <span>Close</span>
            </button>
          </>
        ) : (
          <>
            <div className="spacer" />
            <span className="eyebrow">Local image classifier</span>
          </>
        )}
      </header>

      <BusyBar />

      <main className="stagearea">
        {!manifest ? (
          <HomeView />
        ) : view === "label" ? (
          <LabelView />
        ) : view === "train" ? (
          <TrainView />
        ) : (
          <UseView />
        )}
      </main>

      <Toasts />
    </div>
  );
}
