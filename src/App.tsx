import { useEffect, useState } from "react";
import { CheckCircle, CircleNotch, FloppyDisk, SignOut } from "@phosphor-icons/react";
import { useStore, type View } from "./lib/store";
import { BeloMark } from "./components/BeloMark";
import { BusyBar, Toasts } from "./components/ui";
import { HomeView } from "./views/HomeView";
import { LabelView } from "./views/LabelView";
import { TrainView } from "./views/TrainView";
import { UseView } from "./views/UseView";
import { DetectTrainView } from "./views/DetectTrainView";
import { DetectUseView } from "./views/DetectUseView";
import { isSelectAllShortcut, isTextEditingTarget } from "./lib/keyboard";
import { isHudWindow } from "./lib/hud";
import { HudView } from "./views/HudView";
import {
  closeScreenSnip,
  openScreenSnip,
  SCREEN_SNIP_LABEL,
  type ScreenSnipTarget,
} from "./lib/screenSnip";

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

/** Shows which classification mode the open project was created with. */
function ModeBadge() {
  const mode = useStore((s) => s.manifest?.mode ?? "single");
  const title =
    mode === "multi"
      ? "Multi label project: every label gets its own yes or no"
      : mode === "detect"
        ? "Detect project: labels live on the boxes you draw"
        : "Single label project: one winning label per image";
  return (
    <span className="badge badge--idle topbar__mode" title={title}>
      {mode}
      <span className="topbar__mode-word">&nbsp;{mode === "detect" ? "boxes" : "label"}</span>
    </span>
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

function MainApp() {
  const booted = useStore((s) => s.booted);
  const boot = useStore((s) => s.boot);
  const manifest = useStore((s) => s.manifest);
  const path = useStore((s) => s.path);
  const closingProject = useStore((s) => s.closingProject);
  const view = useStore((s) => s.view);
  const setView = useStore((s) => s.setView);
  const closeProject = useStore((s) => s.closeProject);
  const hasModel = useStore((s) => s.hasModel);
  const images = manifest?.images.length ?? 0;
  const detect = manifest?.mode === "detect";
  const [screenSnipActive, setScreenSnipActive] = useState(false);

  useEffect(() => {
    void boot();
  }, [boot]);

  useEffect(() => {
    let live = true;
    if (!path || closingProject) {
      setScreenSnipActive(false);
      void closeScreenSnip();
      return () => {
        live = false;
      };
    }

    const projectPath = path;
    const target: ScreenSnipTarget = {
      isOpen: () => {
        const current = useStore.getState();
        return current.path === projectPath && !current.closingProject && current.manifest !== null;
      },
      add: async (payload) => {
        const current = useStore.getState();
        if (!current.path || current.path !== projectPath || current.closingProject || !current.manifest) {
          return null;
        }
        const labels = current.manifest.mode === "detect" ? null : current.activeLabelId;
        const added = await current.addFromBase64(
          payload.base64,
          payload.mime,
          labels,
          "capture",
          "screen",
          true,
        );
        const after = useStore.getState();
        if (!added || after.path !== projectPath || !after.manifest) return null;
        return { projectName: after.manifest.name, count: after.manifest.images.length };
      },
    };

    void openScreenSnip(target, {
      onStatus: (status) => {
        if (live) setScreenSnipActive(status.active);
      },
      onForegroundImport: (result) => {
        if (live) {
          const imageWord = result.count === 1 ? "image" : "images";
          useStore
            .getState()
            .toast(`Screen capture added to ${result.projectName} (${result.count} ${imageWord})`, "good");
        }
      },
      onError: (message) => {
        if (live) useStore.getState().toast(`Screen capture failed: ${message}`, "bad");
      },
    });

    return () => {
      live = false;
      setScreenSnipActive(false);
      void closeScreenSnip();
    };
  }, [closingProject, path]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (!(event.metaKey || event.ctrlKey)) return;
      if (isSelectAllShortcut(event)) {
        if (!isTextEditingTarget(event.target)) event.preventDefault();
        return;
      }
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
        <span className="boot__mark">
          <BeloMark size={54} />
        </span>
        <span className="eyebrow">Starting Belo</span>
      </div>
    );
  }

  return (
    <div className={manifest ? "shell shell--project" : "shell"}>
      <header className="topbar">
        <div className="topbar__brand">
          <BeloMark size={26} onDark className="topbar__mark" />
          <span className="topbar__word">Belo</span>
        </div>

        {manifest ? (
          <>
            <span className="topbar__sep" aria-hidden="true" />
            <ProjectName />
            <ModeBadge />
            <SaveState />
            <RetrainFlag />
            {screenSnipActive ? (
              <span
                className="snipshortcut"
                title="Capture a screen region into this project"
                aria-label="Screen capture shortcut Command Shift B"
              >
                <span className="snipshortcut__key mono">{SCREEN_SNIP_LABEL}</span>
                <span className="snipshortcut__word">Snip</span>
              </span>
            ) : null}
            <div className="spacer" />
            <nav className="tabs" aria-label="Project stages">
              {TABS.map((tab) => {
                const blocked = tab.id === "use" && !hasModel && !detect;
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
            <button
              className="btn btn--sm topbar__close"
              onClick={() => void closeProject()}
              aria-label="Close project"
            >
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
          // Detect projects train and use their spatial head in dedicated
          // views. The classifier's single and multi paths stay shared.
          detect ? (
            <DetectTrainView />
          ) : (
            <TrainView />
          )
        ) : detect ? (
          <DetectUseView />
        ) : (
          <UseView />
        )}
      </main>

      <Toasts />
    </div>
  );
}

export default function App() {
  return isHudWindow() ? <HudView /> : <MainApp />;
}
