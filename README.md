# Belo

A desktop app for training image classifiers on your own pictures. Collect images,
label them, train, and use the result live from your webcam. Everything runs on the
machine it is installed on: no account, no upload, no network calls at runtime.

Version 0.1.0.

## What it does

1. **Label.** Create labels, then fill them by dragging in files or whole folders,
   importing from disk, or holding the capture button for a webcam burst. Images can be
   selected and moved between labels or deleted.
2. **Train.** One click trains a classifier and reports live progress, training accuracy
   and a holdout score. Every image is scored afterwards so you can see exactly which
   ones the model still gets wrong, per label.
3. **Use.** Point the webcam at something and watch the live prediction and confidence
   bars, or drop a single image in for a one-shot verdict. Either way you can confirm or
   correct the prediction, which adds that frame to the dataset for the next run.

Model export is out of scope for this version.

## How the model works

The classifier is a two-part stack:

- A frozen **MobileNet v1 (1.0, 224)** truncated at `conv_pw_13_relu` and spatially
  averaged, giving a 1024 value embedding per image. Its weights ship inside the app at
  `public/models/mobilenet-v1-1.0-224`, so it never fetches anything.
- A small trainable **dense head** (default 100 hidden units, softmax output) trained with
  Adam. Only this part learns, which is why training takes seconds rather than hours.

TensorFlow.js picks the **WebGL** backend and falls back to **WASM** if WebGL is
unavailable. The WASM binaries are copied into `public/tfjs-wasm` by `npm run assets`,
which runs automatically before `dev` and `build`.

Images are center cropped and resized to 224x224 JPEG when they enter a project. That
keeps bundles small and makes imports, captures and feedback frames identical in shape.
The originals are left untouched where they are.

## The .belo format

A project is a **directory bundle** named `<Project>.belo`. It is a plain folder, so it
works the same on macOS, Windows and Linux, and it can be inspected, diffed and backed up
with normal tools. Saves are incremental: only the manifest is rewritten on edits, and it
is written to a temp file and renamed so a crash cannot truncate it.

```
Shapes.belo/
  project.json        manifest: labels, image records, training settings, model metadata
  images/
    img_xxx.jpg       one normalized 224x224 JPEG per image record
  model/
    head.json         serialized dense head weights, written after a training run
```

`project.json`:

```jsonc
{
  "format": "belo-project",
  "formatVersion": 1,
  "appVersion": "0.1.0",
  "name": "Shapes",
  "createdAt": "2026-07-27T14:00:00.000Z",
  "updatedAt": "2026-07-27T14:05:00.000Z",
  "labels": [{ "id": "lbl_...", "name": "Circle", "color": 0, "createdAt": "..." }],
  "images": [
    {
      "id": "img_...",
      "file": "img_....jpg",     // relative to images/
      "labelId": "lbl_...",      // null means unlabeled
      "source": "import",        // import | capture | feedback
      "addedAt": "..."
    }
  ],
  "model": {
    "trainedAt": "...",
    "labelIds": ["lbl_..."],     // output order of the head
    "epochs": 40,
    "accuracy": 1,
    "valAccuracy": 1,
    "imageCount": 44,
    "settings": { "epochs": 40, "batchSize": 16, "learningRate": 0.001, "hiddenUnits": 100, "validationSplit": 0.15 }
  },
  "settings": { "epochs": 40, "batchSize": 16, "learningRate": 0.001, "hiddenUnits": 100, "validationSplit": 0.15 }
}
```

`model/head.json` stores the head weights as plain arrays plus the label order they were
trained against, so reopening a project restores a working classifier without retraining.

New projects are created in `~/Documents/Belo` by default. Any `.belo` folder elsewhere
can be opened with the file picker.

## Development

Requirements: Node 20 or newer, Rust 1.77 or newer, and the
[Tauri 2 system dependencies](https://v2.tauri.app/start/prerequisites/) for your platform.

```bash
npm install
npm run tauri dev     # run the app against the Vite dev server
npm run build         # typecheck plus production frontend build
npm run typecheck     # tsc --noEmit only
npm run tauri build   # produce a platform installer
cd src-tauri && cargo check
```

Useful extras:

- `node scripts/make-icon.mjs` regenerates `src-tauri/app-icon.png`, then
  `npx tauri icon src-tauri/app-icon.png` rebuilds the icon set.
- `npm run assets` refreshes the bundled TensorFlow.js WASM binaries.

### Layout

```
src/
  App.tsx                 shell, top bar, stage tabs
  components/ui.tsx       modal, meter, toasts, empty state
  lib/
    api.ts                wrappers for the Rust commands
    store.ts              application store: project, disk writes, feature cache, head
    images.ts             normalization, drag and drop extraction
    useWebcam.ts          getUserMedia session handling
    types.ts              manifest schema
    ml/backend.ts         WebGL then WASM backend selection
    ml/features.ts        truncated MobileNet feature extractor
    ml/classifier.ts      dense head: build, train, score, serialize
  views/                  HomeView, LabelView, CaptureModal, TrainView, UseView
  styles/                 tokens, base, ui primitives, app layout
src-tauri/src/lib.rs      bundle filesystem commands and the recent projects list
public/models/            bundled MobileNet weights
```

## Camera and permissions

The webcam is used through `getUserMedia` inside the webview.

- **macOS** ships `NSCameraUsageDescription` via `src-tauri/Info.plist`. The first capture
  triggers the normal system permission prompt.
- **Windows** works through WebView2 with the standard camera privacy setting.
- **Linux** uses WebKitGTK. Camera support there depends on the installed WebKitGTK build,
  so treat it as the least tested of the three.

Reading images back into the grid uses the Tauri asset protocol, scoped to `$HOME` in
`src-tauri/tauri.conf.json`, because projects can live anywhere under the home directory.

## License

MIT.
