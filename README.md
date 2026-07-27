# Belo

An app for training image classifiers on your own pictures. Collect images, label
them, train, and use the result live from your camera. Everything runs on the device:
no account, no upload, no network calls at runtime.

Version 0.2.0. Desktop and iOS are supported. Android is code-complete but has not yet
been built on an Android toolchain.

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
  "appVersion": "0.2.0",
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
    storage.ts            ProjectStorage seam: the only module that talks to Tauri
    capture.ts            CameraSource seam: the only module that calls getUserMedia
    store.ts              application store: project, disk writes, feature cache, head
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
must be followed by restoring the 15.0 deployment target, `app.belo` bundle identifier
and camera usage description from `project.yml`.

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
