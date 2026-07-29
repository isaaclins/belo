# Belo

[![test](https://github.com/isaaclins/belo/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/isaaclins/belo/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-ffd83d?style=flat-square&labelColor=171717)](#license)
![Platforms: desktop and iPhone](https://img.shields.io/badge/platforms-Desktop%20%7C%20Mobile-f473b9?style=flat-square&labelColor=171717)

Teach your computer to recognize things using your own pictures. Belo keeps every
picture and project on your device. There is no account, no upload, and no internet
connection needed after installation.

## Pick what you want Belo to learn

Every new project offers three clear choices:

- **Classify one answer** gives the whole picture one best label.
- **Classify many answers** gives every label its own yes or no, so one picture can have
  several labels.
- **Detect objects** finds several things in one picture and draws a real box around each
  one.

## Classification in three simple steps

1. **Label.** Name the things you want Belo to recognize, then add pictures from your
   files or camera.
2. **Train.** Press one button. Belo learns from your examples and shows which pictures
   still confuse it.
3. **Use.** Point your camera at something, or choose one picture, and see Belo's answer.
   Correct it when it is wrong so it can learn from that example too.

## Detection in three simple steps

1. **Draw boxes.** Add pictures, then drag a box around every object you want Belo to
   find.
2. **Train.** Belo studies the box labels, positions and sizes.
3. **Use live.** Point the camera at your scene. Belo draws real boxes around the custom
   objects it finds, or does the same on one chosen picture.

Detection usually improves with about **50 to 200 boxes per label**, with varied
backgrounds, angles and sizes. Desktop is recommended for training because detection uses
more memory and takes longer. **Show attention is only for classification projects**;
detect projects show their detector's actual boxes instead.

## Useful things you can do

- **Teach classification from the live camera.** Hold a teaching button to add several
  examples, and choose how quickly Belo takes them.
- **See classification clues.** Turn on **Show attention** for a colored hint showing
  which parts of the picture influenced each classification answer.
- **Give one picture several labels.** For example, the same picture can be both
  "outdoors" and "dog".
- **Keep projects you can back up.** On desktop, each `.belo` project is a normal folder
  containing its pictures and saved learning. You can copy or back it up like any other
  folder.

## Get Belo

Desktop installers are built by GitHub Actions from the tagged source and attached to
every release. These links always point at the newest release:

| Platform | Download |
| --- | --- |
| macOS, Apple Silicon | [Belo-macos-aarch64.dmg](https://github.com/isaaclins/belo/releases/latest/download/Belo-macos-aarch64.dmg) |
| Windows, 64 bit | [Belo-windows-x86_64_setup.exe](https://github.com/isaaclins/belo/releases/latest/download/Belo-windows-x86_64_setup.exe) or [the .msi](https://github.com/isaaclins/belo/releases/latest/download/Belo-windows-x86_64.msi) |
| Linux, x86_64 | [Belo-linux-x86_64.AppImage](https://github.com/isaaclins/belo/releases/latest/download/Belo-linux-x86_64.AppImage) or [the .deb](https://github.com/isaaclins/belo/releases/latest/download/Belo-linux-x86_64.deb) |

The builds are not signed yet, because that needs paid developer accounts.

- **macOS.** Gatekeeper blocks a double click on an unsigned app. The first time, right
  click Belo in Applications, choose Open, and confirm. After that it opens normally.
- **Windows.** SmartScreen may warn about an unknown publisher. Choose More info, then
  Run anyway.
- **Linux.** Mark the AppImage executable with `chmod +x` before the first run. The
  AppImage and the deb both need WebKitGTK 4.1.

On an Intel Mac, or on any other architecture, [build Belo from source](#development).

**App Store: coming soon.** Apple developer license approval is currently in progress,
so there is no App Store listing yet.

## Honest limitations

Results depend on your examples and camera quality, so varied, clear pictures usually work
better. Detection reads a 7x7 grid over the picture, so an object much smaller than a
seventh of the frame is hard for it to place. Training detection on a phone can also use a
lot of memory, so desktop is the safer choice for a large project.

**Show attention** remains a rough classification hint, not a detector. It is available
only in Classify one answer and Classify many answers projects. Detect objects projects
use the separately trained detector and show its real boxes.

## FAQ

### Is my data uploaded anywhere?

No, never. Belo does not send your pictures or projects anywhere.

### Does Belo need the internet?

Only for installation. After Belo is installed, it works without an internet connection.

### Can I use it without a webcam?

Yes. You can import pictures and use a single imported picture for a prediction.

---

# Developer guide

Current app version: **0.5.0**. Desktop and iOS are supported. Android is code-complete
but has not yet been built on an Android toolchain.

## Detailed workflow

1. **Label.** Create labels, then fill them by dragging in files or whole folders,
   importing from disk, or holding the capture button for a webcam burst. Images can be
   selected and moved between labels or deleted. In a multi label project the same
   selection toggles several labels on and off instead of moving between them. In a detect
   project a click opens the picture large instead, and you drag rectangles onto it: the
   label rail picks the color you draw with, boxes can be moved, resized by their eight
   grips and deleted, and the thumbnails show the boxes with a count so progress is
   readable from the grid. The grid
   uses uniform cells and shows every thumbnail whole rather than cropping it to a square,
   so a portrait frame is letterboxed instead of having its top and bottom cut off. The
   **minus and plus** buttons above the grid pick one of four cell sizes, which changes how
   many fit per row; the choice is remembered across restarts and applies to every project.
2. **Train.** One click trains a classifier and reports live progress, training accuracy,
   a holdout score and a per label score. Every image is scored afterwards so you can see
   exactly which ones the model still gets wrong. **Move to X** on a wrong card accepts what
   the model read as the true label: the card flips to correct on the spot, the per label
   panel recounts, and the same debounced background retrain as a Use view correction folds
   it in.
3. **Use.** Point the webcam at something and watch the live prediction and confidence
   bars, or drop a single image in for a one-shot verdict. Either way you can confirm or
   correct the prediction, which adds that frame to the dataset and quietly retrains the
   head a few seconds later. Tapping a **This is X** button teaches one frame; holding it
   keeps teaching frames until you let go, counting up on the button and capping one hold
   at 60 frames. The **Rate** control next to those buttons picks 2, 4, 8 or 12 frames per
   second and drives the Label view burst as well, so the app has one capture speed;
   frames are dropped rather than queued if the disk cannot keep up. While the camera runs, **Show attention** overlays where the
   evidence for the current call sits: a tinted heat map plus one box per label. It reads
   the classifier's own 7x7 feature grid, so it is a coarse hint at roughly a 32 pixel
   resolution and never an object detector. The toggle starts off and is remembered until
   the app closes, never in the project. Detect projects use their real detector instead:
   every object above the chosen score threshold gets a coloured box, name and score over
   the live video or a dropped image. The threshold choices are 0.2, 0.3, 0.5 and 0.7 and
   the last choice is remembered. A live frame can be saved back to the project without
   boxes, ready to annotate later in Label.

Model export is out of scope for this version.

## Project modes

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
- **Detect**: labels sit on drawn rectangles rather than on whole pictures, so one picture
   can hold several objects in different places. The Label view becomes a box annotator
   and the label rail picks the pencil instead of a folder: every picture is always shown,
   and the count next to a label is how many boxes carry it. The Train view trains a
   detection head on those boxes and then shows every scored picture twice over: your
   boxes in their label colour, the boxes the detector read dashed on top, so a miss and a
   false alarm are visible instead of averaged away. It wants boxes on at least ten
   pictures before it will train. The Use view runs that head on the camera at the same
   paced cadence as classification, dropping a tick instead of queueing work when a frame
   is still running, and also accepts one image for a single pass.

Images with no labels are ignored by training in single and multi mode. In detect mode a
picture with no boxes is not ignored: it is a background example, and training uses it to
learn what nothing looks like.

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

### The detection head

A detect project trains a different head on the same frozen features, reading the 7x7x1024
grid rather than the pooled embedding. It follows the SSD recipe:

- **Anchors.** Six per cell, two scales (0.22 and 0.5 of the frame) crossed with three
  aspect ratios (square, wide, tall), so 294 anchors over the grid. The scales bracket the
  object sizes a 224 pixel picture can hold and the ratios give an upright and a lying
  object each an anchor to match on.
- **Head.** Each cell's 1024 values are standardized, then run through one shared 96 unit
  layer with dropout and two output layers: four box offsets per anchor, and one score per
  label plus a background class. Everything is 1x1, never a convolution across cells. One
  cell of `conv_pw_13_relu` already sees almost the whole picture, and the WASM backend has
  no gradient kernel for convolution filters at all, so a convolutional head could not be
  trained on a machine without WebGL.
- **Boxes.** Standard SSD parameterization: the center shift in anchor widths, the size in
  log scale, with variances 0.1 and 0.2.
- **Loss.** Anchors overlapping a drawn box by 0.45 or more answer for it, under 0.35 they
  are background, in between they are ignored, and every drawn box additionally force
  matches its single best anchor so nothing is unlearnable. Positives get a smooth L1 loss
  on the offsets; the classification loss is cross entropy over the positives plus the
  three hardest background anchors per positive.
- **Reading it back.** Scores are softmax probabilities with background at index 0.
  Anything at 0.3 or above survives by default, then per label suppression at 0.3 overlap
  removes the duplicates that neighbouring anchors produce. The Use view can move the
  score cutoff between 0.2, 0.3, 0.5 and 0.7 without retraining.

Training runs its own loop rather than `model.fit`, because of the anchor bookkeeping, and
it caches the grid features so MobileNet runs once per picture per run. Those cached maps
are the memory cost of a detect run: 196 KB per picture, so about 20 MB for a hundred
pictures and 60 MB for three hundred.

The runtime inference contract is `detectSpatialMap(detector, cells, options)` in
`src/lib/ml/detector.ts`. It accepts one 7x7 spatial feature map and returns normalized
`x`, `y`, `w`, `h`, label and score values after thresholding and per-label suppression.
The app-facing `store.detectObjects(source, scoreThreshold)` extracts that map, calls the
same contract used by the live and still-image views, and always disposes the tensor it
owns.

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
    detect.json       serialized detection head, detect projects only
```

`project.json`:

```jsonc
{
  "format": "belo-project",
  "formatVersion": 3,
  "appVersion": "0.5.0",
  "name": "Shapes",
  "createdAt": "2026-07-27T14:00:00.000Z",
  "updatedAt": "2026-07-27T14:05:00.000Z",
  "mode": "single",              // single | multi | detect, fixed at creation
  "labels": [{ "id": "lbl_...", "name": "Circle", "color": 0, "createdAt": "..." }],
  "images": [
    {
      "id": "img_...",
      "file": "img_....jpg",     // relative to images/
      "labelId": "lbl_...",      // null means unlabeled
      "labelIds": ["lbl_..."],   // multi mode only, the full label set
      "boxes": [                 // detect mode only, empty means a background picture
        {
          "id": "box_...",
          "labelId": "lbl_...",
          "x": 0.11, "y": 0.42,  // top left corner, 0..1 of the stored picture
          "w": 0.3, "h": 0.25
        }
      ],
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
    // detect projects only: the share of drawn boxes found and of detections
    // that were real, both at half overlap, plus how many boxes were drawn
    "precision": 0.88,
    "recall": 0.75,
    "boxCount": 49,
    "imageCount": 44,
    "settings": { "epochs": 40, "batchSize": 16, "learningRate": 0.001, "hiddenUnits": 100, "validationSplit": 0.15 }
  },
  "checkpoint": null,            // same shape as model, describes head.prev.json
  "modelStale": false,           // true when data changed after this model trained
  "settings": { "epochs": 40, "batchSize": 16, "learningRate": 0.001, "hiddenUnits": 100, "validationSplit": 0.15 }
}
```

`model/head.json` stores the head weights as plain arrays, the label order they were
trained against and the mode they were trained in, so reopening a project restores a
working classifier without retraining. `model/detect.json` does the same for a detect
project and additionally records the feature grid, the anchor scales and ratios and the
head width it was built with, so a head keeps loading if those defaults ever move.

Detect projects are created with different training settings, because a detection head
sees hundreds of anchors per picture instead of one embedding: 150 epochs, batches of 8, a
learning rate of 0.002, a head width of 96 and a fifth of the pictures held back. Single
and multi projects are created with the same settings they always were.

### Schema versions

- **1**: single label projects. One `labelId` per image, no `mode`, no `checkpoint`.
- **2**: adds `mode`, per image `labelIds` in multi mode, `checkpoint`, and `mode` on the
  stored head. A version 1 manifest loads as `mode: "single"` with `labelIds` derived from
  `labelId`, which is exactly its old behavior, and is written back at the current schema
  version on the next save. A head written without a mode is read as softmax.
- **3**: adds the `detect` mode and, in that mode only, per image `boxes` in normalized
  0..1 coordinates. Version 1 and 2 manifests load exactly as before and never grow a
  `boxes` field; boxes are clamped back inside the picture on load, and anything smaller
  than two percent of an edge is dropped as a stray tap.

New projects are created in `~/Documents/Belo` by default. Any `.belo` folder elsewhere
can be opened with the file picker. The trash button on a project card deletes the bundle
itself after a confirmation that names the project and what goes with it, images and
trained model included. The Rust side refuses to delete anything that is not a `.belo`
directory holding a `project.json`, and the currently open project cannot be deleted.

## Development

Requirements: Node 20 or newer, Rust 1.77 or newer, and the
[Tauri 2 system dependencies](https://v2.tauri.app/start/prerequisites/) for your platform.

```bash
npm install
npm run tauri dev     # run the app against the Vite dev server
npm test              # typecheck plus every check script below, in CI's order
npm run build         # typecheck plus production frontend build
npm run typecheck     # tsc --noEmit only
npm run tauri build   # produce a platform installer
cd src-tauri && cargo check
```

Detection release verification. `npm test` runs all five in this order and prints how long
each one took, which is exactly what CI does:

```bash
npm run check:boxes       # manifest v3 and one-finger box geometry
npm run check:detect      # anchors, loss inputs, decoding and scoring
npm run check:store       # stale-model persistence and serialized manifest saves
npm run check:camera      # live camera switching seam
npm run check:learning    # learning proof plus the exact Use inference path
```

### Continuous integration

Two workflows live in `.github/workflows`:

- **test.yml** runs on every push and pull request to `main`. One job runs `npm test` and
  the production frontend build on Ubuntu. A second job runs `cargo check --all-targets`
  and `cargo test` on Ubuntu, macOS and Windows, so a platform specific compile error
  shows up before a release tag, not during one.
- **release.yml** runs on a `v*` tag and on manual dispatch. It builds the macOS dmg, the
  Linux AppImage and deb and the Windows msi and nsis installers on their own runners,
  and on a tag it creates the GitHub Release and uploads them. Asset names carry no
  version number, which is what lets the download links above stay the same forever.

To cut a release: make sure `main` is green, then `git tag v0.6.0 && git push origin
v0.6.0`. The version in `package.json`, `src-tauri/tauri.conf.json` and
`src-tauri/Cargo.toml` should match the tag.

Useful extras:

- `npm run icons` redraws the app mark and rebuilds every platform icon set. See
  [The app mark](#the-app-mark).
- `npm run assets` refreshes the bundled TensorFlow.js WASM binaries.
- `npm run check:camera` runs the camera seam against a mocked `MediaDevices`, which covers
  device switching, stale mobile ids and hot plug on a machine with a single webcam.
- `npm run check:boxes` runs the detect mode box seam: drawing in every direction, moving
  against the edges of the picture, all eight resize grips, and a save and reopen round
  trip of version 1, 2 and 3 manifests.
- `npm run check:detect` runs the detection decision layer: anchor generation and layout,
  overlap, offset encode and decode round trips, anchor matching including the forced best
  match, cell standardizing, hard negative mining, suppression and the precision and recall
  matcher. No TensorFlow, no DOM.
- `npm run check:store` uses an in-memory storage double to prove box edits persist the
  stale-model flag, overlapping manifest writes serialize in order and Close flushes the latest
  box deletion. It uses only a fake `/tmp` handle and never opens a real project.
- `npm run check:learning` is the proof that the detector learns. It builds a synthetic
  detect project of 64 pictures with coloured shapes at known positions, extracts real
  features with the bundled MobileNet on the WASM backend, trains the head with the
  settings a new detect project is created with, and then measures on 24 pictures it never
  saw. The run is seeded. It currently finds 80 percent of the held out boxes with 91
  percent of its detections right and a mean overlap of 0.69, against 100 percent on the
  pictures it trained on. It also rebuilds each held out spatial map as the tensor handed
  to the Use view, runs the exact Use inference seam and checks both placement and tensor
  cleanup; 22 of 24 frames contain a correctly placed box at half overlap. It takes about
  half a minute, and feature extraction is cached in the temp folder for reruns. Needs
  Node 22.15 or newer.

### Layout

```
src/
  App.tsx                 shell, top bar, stage tabs
  assets/logo.svg         the canonical Belo mark
  components/ui.tsx       modal, meter, toasts, empty state
  components/BeloMark.tsx          the mark as a component, top bar and home hero
  components/AttentionOverlay.tsx  classifier heat map and coarse boxes over live video
  components/DetectionOverlay.tsx  detector boxes aligned to live and single-image pixels
  components/CameraPicker.tsx      camera select shown wherever live video runs
  components/ThumbSizePicker.tsx   the four step image grid cell size control
  lib/
    attention.ts          overlay geometry and thresholding, pure functions
    boxes.ts              detect mode box geometry, pure functions
    detect.ts             anchors, matching, offset coding, suppression, scoring
    thumbSize.ts          grid cell size, remembered per machine
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
    ml/detector.ts        detection head: build, train, detect, serialize
  views/                  HomeView, LabelView, CaptureModal, TrainView, UseView
                          BoxAnnotator (detect mode box drawing)
                          DetectTrainView (detect mode training and results)
                          DetectUseView (live and single-image detection)
  styles/                 tokens, base, ui primitives, app layout
src-tauri/src/lib.rs      bundle filesystem commands and the recent projects list
public/models/            bundled MobileNet weights
```

### The app mark

One drawing is used everywhere: the top bar, the home hero, the dock icon, the Windows
tiles and the iOS home screen. It is a yellow field inside a thick ink frame, a bold ink
ring in the upper left and a pink chip offset over its lower right, drawn on a 512 unit
grid with flat color and no gradients, blur or transparency.

Platform icons drop the frame: yellow edge to edge, with the ring and the chip scaled up
slightly to take the space the frame used to hold. A launcher icon is already framed by
the system, and on iOS the rounded mask cut the ink band into muddy dark corners. The
in-app mark keeps its frame, because there it sits on a page and needs an edge.

The same geometry is written three times and the three must be kept in step:

- `src/assets/logo.svg`, the vector master.
- `src/components/BeloMark.tsx`, the in-app component, which can swap its frame to paper
  so the mark still reads on the ink top bar.
- `scripts/make-icon.mjs`, which rasterizes the frameless variant for the icon pipeline.
- `docs/index.html`, which inlines the framed mark as an SVG symbol for the landing page.
- `scripts/make-og.mjs`, which draws the framed mark plus a hand authored wordmark into
  the social preview card.

`npm run icons` runs the whole pipeline: it draws `src-tauri/app-icon.png` at 1024, hands
that to `tauri icon` for the desktop and Windows sets, then redraws the iOS
`AppIcon.appiconset` images itself. That last step matters because `tauri icon` writes iOS
PNGs with an alpha channel and Apple rejects app icons that have one, so those images are
drawn at their exact final size as PNG color type 2, with no alpha and no resampling.

`npm run og` redraws `docs/og.png`, the 1200x630 card that link previews show. It is the
same flat shape sampler as the icon script, with the letters built from thick strokes and
arcs rather than a font, so the repo needs neither an image editor nor a woff2 rasterizer
to rebuild it.

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
