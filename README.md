# Belo

An app for training image classifiers on your own pictures. Collect images, label
them, train, and use the result live from your camera. Everything runs on the device:
no account, no upload, no network calls at runtime.

Version 0.4.1. Desktop and iOS are supported. Android is code-complete but has not yet
been built on an Android toolchain.

## What it does

1. **Label.** Create labels, then fill them by dragging in files or whole folders,
   importing from disk, or holding the capture button for a webcam burst. Images can be
   selected and moved between labels or deleted. In a multi label project the same
   selection toggles several labels on and off instead of moving between them.
2. **Train.** One click trains a classifier and reports live progress, training accuracy,
   a holdout score and a per label score. Every image is scored afterwards so you can see
   exactly which ones the model still gets wrong.
3. **Use.** Point the webcam at something and watch the live prediction and confidence
   bars, or drop a single image in for a one-shot verdict. Either way you can confirm or
   correct the prediction, which adds that frame to the dataset and quietly retrains the
   head a few seconds later. While the camera runs, **Show attention** overlays where the
   evidence for the current call sits: a tinted heat map plus one box per label. It reads
   the classifier's own 7x7 feature grid, so it is a coarse hint at roughly a 32 pixel
   resolution and never an object detector. The toggle starts off and is remembered until
   the app closes, never in the project.

Model export is out of scope for this version.

## Classification modes

The mode is picked when the project is created and shown in the project header. It cannot
be switched afterwards.

- **Single label** (default): a softmax over the labels. Every image belongs to exactly
   one label and the model reports one winner with a confidence. This is what every
   project before 0.3.0 does, and those projects keep behaving identically.
- **Multi label**: one sigmoid output per label, binary cross entropy loss and a fixed
   0.5 threshold per label. An image can carry any number of labels, so "X and Y but not
   Z" is expressible. The Train view reports accuracy per label, meaning the share of
   images whose yes or no call for that label was right. The Use view shows an
   independent yes or no with a confidence for every label rather than a winner, and the
   teach chips become toggles that add one frame with the whole set you picked.

Images with no labels are ignored by training in both modes.

## Background retraining and the checkpoint

Confirming or correcting a prediction in the Use view adds that frame to the dataset and
schedules a retrain about four seconds after the last correction, so a burst of taps
produces one run and not one per tap. The run reuses the normal training path, keeps the
UI interactive, and shows a small "retraining" marker in the header while it works. The
live camera keeps classifying with the current model until the new one is ready. If a
manual training run is going, the automatic one waits for it and then retries.

Before each automatic run the current head is copied to `model/head.prev.json`. The Train
view then offers **Revert to previous model**, which puts that head back, writes it to
`model/head.json` and drops the checkpoint. Manual training runs do not write a
checkpoint, so the action always means "undo what the app did on its own".

## How the model works

The classifier is a two-part stack:

- A frozen **MobileNet v1 (1.0, 224)** truncated at `conv_pw_13_relu` and spatially
  averaged, giving a 1024 value embedding per image. Its weights ship inside the app at
  `public/models/mobilenet-v1-1.0-224`, so it never fetches anything. The same pass can
  hand back the 7x7x1024 activation before pooling, which is what the attention overlay
  scores cell by cell.
- A small trainable **dense head** (default 100 hidden units) trained with Adam. Only this
  part learns, which is why training takes seconds rather than hours. The output layer is
  a softmax in single label projects and one sigmoid per label in multi label projects.

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
    head.prev.json    checkpoint of the head replaced by the last automatic retrain
```

`project.json`:

```jsonc
{
  "format": "belo-project",
  "formatVersion": 2,
  "appVersion": "0.4.1",
  "name": "Shapes",
  "createdAt": "2026-07-27T14:00:00.000Z",
  "updatedAt": "2026-07-27T14:05:00.000Z",
  "mode": "single",              // single | multi, fixed at creation
  "labels": [{ "id": "lbl_...", "name": "Circle", "color": 0, "createdAt": "..." }],
  "images": [
    {
      "id": "img_...",
      "file": "img_....jpg",     // relative to images/
      "labelId": "lbl_...",      // null means unlabeled
      "labelIds": ["lbl_..."],   // multi mode only, the full label set
      "source": "import",        // import | capture | feedback
      "addedAt": "..."
    }
  ],
  "model": {
    "trainedAt": "...",
    "mode": "single",
    "labelIds": ["lbl_..."],     // output order of the head
    "epochs": 40,
    "accuracy": 1,
    "valAccuracy": 1,
    "imageCount": 44,
    "settings": { "epochs": 40, "batchSize": 16, "learningRate": 0.001, "hiddenUnits": 100, "validationSplit": 0.15 }
  },
  "checkpoint": null,            // same shape as model, describes head.prev.json
  "settings": { "epochs": 40, "batchSize": 16, "learningRate": 0.001, "hiddenUnits": 100, "validationSplit": 0.15 }
}
```

`model/head.json` stores the head weights as plain arrays, the label order they were
trained against and the mode they were trained in, so reopening a project restores a
working classifier without retraining.

### Schema versions

- **1**: single label projects. One `labelId` per image, no `mode`, no `checkpoint`.
- **2**: adds `mode`, per image `labelIds` in multi mode, `checkpoint`, and `mode` on the
  stored head. A version 1 manifest loads as `mode: "single"` with `labelIds` derived from
  `labelId`, which is exactly its old behavior, and is written back as version 2 on the
  next save. A head written without a mode is read as softmax.

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
- `npm run check:camera` runs the camera seam against a mocked `MediaDevices`, which covers
  device switching, stale mobile ids and hot plug on a machine with a single webcam.

### Layout

```
src/
  App.tsx                 shell, top bar, stage tabs
  components/ui.tsx       modal, meter, toasts, empty state
  components/AttentionOverlay.tsx  heat map and boxes drawn over the live video
  components/CameraPicker.tsx      camera select shown wherever live video runs
  lib/
    attention.ts          overlay geometry and thresholding, pure functions
    storage.ts            ProjectStorage seam: the only module that talks to Tauri
    capture.ts            CameraSource seam: the only module that calls getUserMedia
    store.ts              application store: project, disk writes, feature cache, head
    zoom.ts               kills user page zoom: pinch, double tap and zoom shortcuts
    images.ts             normalization, drag and drop extraction
    useWebcam.ts          camera session lifecycle on top of capture.ts
    types.ts              manifest schema
    ml/backend.ts         WebGL then WASM backend selection
    ml/features.ts        truncated MobileNet feature extractor
    ml/classifier.ts      dense head: build, train, score, serialize
  views/                  HomeView, LabelView, CaptureModal, TrainView, UseView
  styles/                 tokens, base, ui primitives, app layout
src-tauri/src/lib.rs      bundle filesystem commands and the recent projects list
public/models/            bundled MobileNet weights
```

## Mobile

The mobile backend keeps `.belo` directory bundles inside the app's sandboxed Documents
folder. Image import uses the native photo picker and copies selected media into the
sandbox before normalization. Mobile does not expose folder drag and drop or external
`.belo` folder opening, so projects currently move between devices only through a manual
container backup.

Phone viewports use bottom stage tabs, a horizontal label rail, safe-area insets and
44-point minimum touch targets. Desktop rules are unchanged above the phone breakpoint.
The camera still uses `getUserMedia`: wry grants the webview media request, while the OS
owns the user-facing runtime prompt.

### iOS

Requirements are Xcode, CocoaPods, XcodeGen, and the three Rust targets:

```bash
rustup target add aarch64-apple-ios aarch64-apple-ios-sim x86_64-apple-ios
npm install
```

The generated Xcode project is committed under `src-tauri/gen/apple`, so normal builds do
not need to run `tauri ios init`. Re-running init replaces generated customizations and
must be followed by restoring the 15.0 deployment target, `app.belo` bundle identifier,
camera usage description and `ASSETCATALOG_COMPILER_APPICON_NAME` from `project.yml`.
That last setting is what assigns `Assets.xcassets/AppIcon.appiconset` to the app: without
it the catalog still compiles but the installed app has no home screen icon. After editing
`project.yml`, run `~/.local/bin/xcodegen generate` inside `src-tauri/gen/apple` and commit
the regenerated `project.pbxproj` and `belo_iOS/Info.plist`.

Build an unsigned simulator app:

```bash
npx tauri ios build --debug --target aarch64-sim --no-sign --ci
xcrun simctl boot "iPhone 17 Pro"
xcrun simctl install "iPhone 17 Pro" src-tauri/gen/apple/build/arm64-sim/Belo.app
xcrun simctl launch "iPhone 17 Pro" app.belo
```

A physical-device or distribution build needs an Apple development team and signing:

```bash
export APPLE_DEVELOPMENT_TEAM="YOUR_TEAM_ID"
npx tauri ios build --target aarch64
```

The minimum is iOS 15.0 because Xcode 27 no longer builds for iOS 14, while WKWebView
camera capture itself requires iOS 14.3 or newer. The simulator has no camera. Belo shows
a clear no-camera message there and supports photo import as the test path instead.

Local verification used Xcode 27.0 and the iPhone 17 Pro simulators on iOS 26.4, 26.5
and 27.0. The app built, installed, launched, created its sandbox Documents/Belo folder
and rendered the responsive home flow. The generated plist opts into tao's scene-based
UIKit lifecycle, which is required by iOS 27. XCUITest interaction with WKWebView is
currently blocked by an iOS 26 simulator accessibility-bundle conflict in Xcode 27, so
picker, labeling and training interaction still need a physical-device pass.

### Android

Android was not built locally because this machine intentionally has no SDK or NDK. The
committed Android configuration sets API 24 as the minimum, version code 2000, sandboxed
Documents storage, and `android.permission.CAMERA`. wry requests the runtime camera grant.
The system photo picker supplies temporary URI access, and `tauri-plugin-fs` reads those
URIs without broad media-library permissions.

To finish verification on a machine with Android Studio:

1. In SDK Manager, install the current Android SDK Platform, Platform-Tools, Build-Tools,
   Command-line Tools, and NDK (Side by side).
2. Set the toolchain environment variables and Rust targets:

   ```bash
   export JAVA_HOME="/Applications/Android Studio.app/Contents/jbr/Contents/Home"
   export ANDROID_HOME="$HOME/Library/Android/sdk"
   export NDK_HOME="$ANDROID_HOME/ndk/$(ls -1 "$ANDROID_HOME/ndk" | tail -1)"
   rustup target add aarch64-linux-android armv7-linux-androideabi \
     i686-linux-android x86_64-linux-android
   ```

3. Generate the Android Studio project, then confirm its manifest contains the camera
   permission generated from `Cargo.toml`:

   ```bash
   npx tauri android init
   grep android.permission.CAMERA src-tauri/gen/android/app/src/main/AndroidManifest.xml
   ```

4. Build and run on an emulator or device:

   ```bash
   npx tauri android build --debug
   npx tauri android dev
   ```

5. Verify project creation and reopening, multi-photo import, two-label training, model
   restoration after relaunch, camera denial and grant behavior, live classification,
   and rotation or resize behavior.

## Camera and permissions

The webcam is used through `getUserMedia` inside the webview.

Wherever live video runs, meaning the capture modal and the Use view, a **Camera** picker
appears as soon as the machine exposes more than one video input, and choosing another one
swaps the stream in place without leaving the view or dropping what you already captured.
The list is read after the first successful start, because device labels stay empty until a
permission is granted, and again on every hot plug event; the choice is remembered across
restarts and falls back to the default camera when that device is gone. Phones hand out
deviceIds that go stale between sessions, so a refused id retries on `facingMode`, which is
what makes front and back switching reliable there.

- **macOS** ships `NSCameraUsageDescription` via `src-tauri/Info.plist`. The first capture
  triggers the normal system permission prompt.
- **iOS** ships the same description through `src-tauri/Info.ios.plist` and the generated
  Xcode project. wry grants the WKWebView media request before iOS shows its prompt.
- **Android** declares `android.permission.CAMERA` through Cargo mobile metadata. wry asks
  for the runtime permission when the webview requests video capture.
- **Windows** works through WebView2 with the standard camera privacy setting.
- **Linux** uses WebKitGTK. Camera support there depends on the installed WebKitGTK build,
  so treat it as the least tested of the three.

Reading images back into the grid uses the Tauri asset protocol, scoped to `$HOME` in
`src-tauri/tauri.conf.json`, because projects can live anywhere under the home directory.

## License

MIT.
