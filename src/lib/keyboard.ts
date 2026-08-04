/** Keyboard helpers shared by app-level shortcuts and view-specific commands. */

const NON_TEXT_INPUT_TYPES = new Set([
  "button",
  "checkbox",
  "color",
  "file",
  "hidden",
  "image",
  "radio",
  "range",
  "reset",
  "submit",
]);

/** Whether a key event originated in a control that should keep text editing. */
export function isTextEditingTarget(target: EventTarget | null): boolean {
  if (!target || typeof target !== "object") return false;
  const element = target as Element;
  if (typeof element.closest !== "function") return false;

  const editable = element.closest("textarea, [contenteditable]");
  if (editable?.tagName === "TEXTAREA") return true;
  if (
    editable?.hasAttribute("contenteditable") &&
    editable.getAttribute("contenteditable") !== "false"
  ) {
    return true;
  }

  const input = element.closest("input");
  if (!input) return false;
  return !NON_TEXT_INPUT_TYPES.has(input.getAttribute("type")?.toLowerCase() ?? "text");
}

/** Cmd+A on macOS, Ctrl+A elsewhere, without consuming an Alt shortcut. */
export function isSelectAllShortcut(event: KeyboardEvent): boolean {
  return (event.metaKey || event.ctrlKey) && !event.altKey && event.key.toLowerCase() === "a";
}
