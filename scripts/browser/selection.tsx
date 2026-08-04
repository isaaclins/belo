// End to end check for contextual Cmd/Ctrl+A behavior.
//
// The real App and LabelView run in headless Chrome. The check proves that the
// active label's visible collection is selected, that switching collections
// changes the target, and that editable controls keep the browser shortcut.
//
//   npm run check:selection

import { createRoot } from "react-dom/client";
import App from "../../src/App";
import { useStore } from "../../src/lib/store";
import { installTauriShim, reporter } from "./tauri-shim";

import "../../src/styles/tokens.css";
import "../../src/styles/base.css";
import "../../src/styles/ui.css";
import "../../src/styles/app.css";

installTauriShim();
const { check, fail, finish } = reporter();

const wait = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));
const settle = () => wait(80);

async function until(what: string, probe: () => boolean, timeoutMs = 10_000): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (!probe()) {
    if (Date.now() > deadline) throw new Error(`Timed out waiting for ${what}`);
    await wait(50);
  }
}

function picture(color: string): HTMLCanvasElement {
  const canvas = document.createElement("canvas");
  canvas.width = 32;
  canvas.height = 32;
  const context = canvas.getContext("2d");
  if (!context) throw new Error("Canvas is unavailable");
  context.fillStyle = color;
  context.fillRect(0, 0, canvas.width, canvas.height);
  return canvas;
}

function press(target: EventTarget, modifiers: { metaKey?: boolean; ctrlKey?: boolean }): KeyboardEvent {
  const event = new KeyboardEvent("keydown", {
    key: "a",
    bubbles: true,
    cancelable: true,
    ...modifiers,
  });
  target.dispatchEvent(event);
  return event;
}

const selectedTiles = () => document.querySelectorAll(".grid .tile.tile--on").length;

async function main(): Promise<void> {
  const store = useStore.getState();
  await store.createProject("Selection check", "single");
  const first = useStore.getState().addLabel("First");
  const second = useStore.getState().addLabel("Second");
  if (!first || !second) throw new Error("labels could not be created");

  await store.addFromDrawable(picture("#e33"), first, "import");
  await store.addFromDrawable(picture("#3a6"), first, "import");
  await store.addFromDrawable(picture("#36e"), second, "import");
  useStore.getState().setActiveLabel(first);
  // App normally sets this during boot. The shim has already supplied the
  // project, so this keeps the harness focused on keyboard behavior.
  useStore.setState({ booted: true });

  const root = createRoot(document.getElementById("mount")!);
  root.render(<App />);
  await until("the Label view", () => Boolean(document.querySelector(".label")));
  await settle();

  const command = press(window, { metaKey: true });
  await settle();
  check("Cmd+A is consumed outside editable controls", command.defaultPrevented);
  check("Cmd+A selects every image in the active label", selectedTiles() === 2, {
    selected: selectedTiles(),
  });

  useStore.getState().setActiveLabel(second);
  await settle();
  const control = press(window, { ctrlKey: true });
  await settle();
  check("Ctrl+A selects the current label on non-macOS shortcuts", control.defaultPrevented);
  check("the selection follows the active label", selectedTiles() === 1, {
    selected: selectedTiles(),
  });

  const input = document.createElement("input");
  input.type = "text";
  input.value = "editable";
  document.body.appendChild(input);
  const inputCommand = press(input, { metaKey: true });
  check("Cmd+A remains available in a text input", !inputCommand.defaultPrevented);
  check("an input Cmd+A does not change image selection", selectedTiles() === 1);

  const editor = document.createElement("div");
  editor.contentEditable = "true";
  editor.textContent = "editable";
  document.body.appendChild(editor);
  const editorCommand = press(editor, { ctrlKey: true });
  check("Ctrl+A remains available in contenteditable", !editorCommand.defaultPrevented);
  check("contenteditable Ctrl+A does not change image selection", selectedTiles() === 1);

  useStore.getState().setView("use");
  useStore.setState({ hasModel: true });
  await until("the Use empty state", () =>
    document.body.textContent?.includes("Drop one image here to classify it") ?? false,
  );
  const useCommand = press(window, { metaKey: true });
  check("Cmd+A is consumed in the non-editable Use view", useCommand.defaultPrevented);

  input.remove();
  editor.remove();
  root.unmount();
  finish();
}

void main().catch(fail);
