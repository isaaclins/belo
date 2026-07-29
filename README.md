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
2. **Train.** Belo studies the box labels, positions and sizes. It also practises on
   mirrored copies of your pictures, keeps the round that did best on pictures it had never
   seen, and then tells you in plain words whether it learned the idea or just memorized
   your pictures.
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
better. Detection reads a 14x14 and a 7x7 grid over the picture, so an object much smaller
than a seventh of the frame is still hard for it to place. Training detection on a phone
can also use a lot of memory, so desktop is the safer choice for a large project.

**Detection is much weaker on messy photographs than on tidy ones.** Measured on 169 real
photographs of people, cats and dogs from COCO, Belo finds a little over half of the
objects a person drew and about half of what it reports is real. Most of the misses are not
the wrong object; they are a box on the right object that is too small or off centre. It
does far better on the case it is built for: one object you photograph yourself, from a few
angles, filling a decent part of the frame. The numbers and the reasoning are in
[Detection on real photographs](#detection-on-real-photographs).

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

Current app version: **0.6.3**. Desktop, iOS and Android all build.

## Detailed workflow

1. **Label.** Create labels, then fill them by dragging in files or whole folders,
   importing from disk, or holding the capture button for a webcam burst. Images can be
   selected and moved between labels or deleted. In a multi label project the same
   selection toggles several labels on and off instead of moving between them. In a detect
   project a click opens the picture large instead, and you drag rectangles onto it: the
   label rail picks the color you draw with, boxes can be moved, resized by their eight
   grips, retagged and deleted, every one of those can be undone with cmd or ctrl + z, and
   the thumbnails show the boxes with a count so progress is readable from the grid. The grid
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
   evidence for the current call sits: a tinted heat map plus one box per label. In a multi
   label project every label that answered yes gets its own tint and its own box on one
   shared layer, with all the tints painted first and the boxes on top, so a second label's
   heat map cannot wash out the first label's box. It reads the classifier's own 7x7 feature
   grid, so it is a coarse hint at roughly a 32 pixel
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
   false alarm are visible instead of averaged away, plus one plain-language verdict on
   whether the run learned the idea or memorized the pictures. It wants boxes on at least
   ten pictures before it will train. The Use view runs that head on the camera at the same
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

A detect project has no live corrections to fold in, so the trigger there is the dataset
itself: a box drawn, moved, resized, retagged, removed or undone, and pictures added or
deleted. The same four second debounce applies, so a session of annotating produces one
run after the last change rather than one per box, and a run is never started in the
middle of a drag because only a committed change schedules one. Early stopping and the
mirrored copies work exactly as they do in a manual run. Nothing is scheduled at all until
the project has boxes on the ten pictures a detect run needs, which keeps the first ten
from each arming a run that would only refuse itself.

Before each automatic run the current model is copied aside: a classifier head to
`model/head.prev.json`, a detector to `model/detect.prev.json`. The Train view then offers
**Revert to previous model**, which puts those weights back byte for byte, writes them to
`model/head.json` or `model/detect.json`, restores the metadata that described them and
drops the checkpoint. Manual training runs do not write a checkpoint, so the action always
means "undo what the app did on its own". A bundle from before checkpoints existed simply
has no such file and never offers the action.

## Undo in the annotator

The box annotator keeps an undo history per picture: **cmd or ctrl + z** steps back,
adding **shift** steps forward again, and the two arrows in the annotator bar do the same
thing with the mouse and grey out when there is nothing to go back to. Drawing, moving,
resizing, deleting and retagging a box are all covered, up to 24 steps per picture.

Each step is a snapshot of the whole box list rather than an inverse edit, so what comes
back is the numbers that were stored and not the result of running a drag backwards. A
drag that ends where it started, or a label picked twice, changes nothing and costs no
step. The history is session state: it lives while the annotator is open, one stack per
picture, and closing it clears everything, because the boxes on disk are the truth. An
undo counts as a change to the dataset, so it schedules the same background retrain any
other edit would.

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

A detect project trains a different head on the same frozen features, reading grids rather
than the pooled embedding. It follows the SSD recipe:

- **Two feature maps.** `conv_pw_13_relu` at 7x7x1024, and `conv_pw_11_relu` one stage
  earlier at 14x14x512. Both come out of a single MobileNet pass, because the fine map is
  an activation the coarse one is computed from. The second map is there for placement, not
  for recognition: one cell of `conv_pw_13_relu` already sees most of the picture and knows
  there is a dog in it, but every box it can propose starts from a grid whose centres are
  32 pixels apart, and on real photographs that was the difference between a box on the
  right animal and a box on the right animal at IoU 0.5.
- **Anchors.** Six per cell on each map, two scales crossed with three aspect ratios
  (square, wide, tall): 0.3 and 0.45 of the frame on the fine map, 0.58 and 0.82 on the
  coarse one, for 1470 anchors in all against 294 on one map. The finer map carries the
  smaller anchors, the standard SSD assignment. The ladder itself was measured rather than
  copied: the textbook ladder puts the fine anchors around a sixth of the frame, which is
  right when the fine map is an early layer with a small receptive field, and
  `conv_pw_11_relu` is eleven blocks deep and nothing like that.
- **Head.** Each map gets its own branch: the cell values are standardized on that map's
  own depth, then run through one shared 96 unit layer with dropout and two output layers,
  four box offsets per anchor and one score per label plus a background class. The branches
  are concatenated coarse first, which is what makes a one map head saved by an older
  version exactly the first branch of a two map one. Everything is 1x1, never a convolution
  across cells, because the WASM backend has no gradient kernel for convolution filters at
  all, so a convolutional head could not be trained on a machine without WebGL. The second
  branch costs about half again as many parameters: 102,474 to 155,796 at 96 units and two
  labels.
- **Boxes.** Standard SSD parameterization: the center shift in anchor widths, the size in
  log scale, with variances 0.1 and 0.2.
- **Loss.** Anchors overlapping a drawn box by 0.5 or more answer for it, under 0.4 they
  are background, in between they are ignored, and every drawn box additionally force
  matches its single best anchor so nothing is unlearnable. Positives get a smooth L1 loss
  on the offsets; the classification loss is cross entropy over the positives plus the
  three hardest background anchors per positive. On a single grid this pair had to be
  loosened to 0.45 and 0.35 to give each box more than one positive anchor; with five times
  the anchors the textbook pair is right again, and looser matching only teaches the head
  that a roughly placed box is correct.
- **Reading it back.** Scores are softmax probabilities with background at index 0.
  Anything at 0.3 or above survives by default, then per label suppression at 0.15 overlap
  removes the duplicates that neighbouring anchors produce. Two maps make the same object
  answer twice, once per grid, and the two answers rarely overlap enough for the older 0.3
  cutoff to merge them. The Use view can move the score cutoff between 0.2, 0.3, 0.5 and
  0.7 without retraining.
- **Mirrored copies.** Detect training shows the head a horizontally flipped copy of every
  training picture, with the boxes flipped with it, under a mild brightness and saturation
  jitter. It is on by default and can be switched off in the Train view settings. The
  transform happens on pixels before the frozen extractor runs, because there is no flip of
  a 1024 value activation, and the variants are built once per run rather than once per
  epoch. Rotation, shear and random crops are deliberately absent: a crop can cut an object
  in half and turn a correct box into a wrong one, while a flip and a colour shift cannot
  lie about a box. The plan is pure and lives in `src/lib/augment.ts`; the pixel transform
  is `applyAugment` in `src/lib/ml/features.ts`.
- **Early stopping.** When a run has a holdout, it watches the holdout loss, gives up after
  ten epochs without an improvement and hands back the weights from the best epoch rather
  than the last. The last epoch is the most memorized head in the run by construction. A
  run without a holdout trains its configured epochs and keeps what it ends on, exactly as
  before. The rule is a pure function in `src/lib/curve.ts` and the Train view reports which
  round was kept. On real photographs this usually ends a run inside twenty epochs instead
  of a hundred and fifty, which is a large part of why detect training now takes seconds.

Training runs its own loop rather than `model.fit`, because of the anchor bookkeeping, and
it caches the grid features so MobileNet runs once per picture per run. Those cached maps
are the memory cost of a detect run, and the second map is the one real price of the finer
grid: a packed example is 588 KB per picture against 196 KB before, so about 59 MB for a
hundred pictures and 177 MB for three hundred. Mirrored copies double that, because an
augmented variant is a second pair of maps over the same file: about 118 MB for a hundred
pictures. Switching the toggle off halves it again. Inference is barely affected, because
the extra map is an activation MobileNet computes either way: measured on WASM, one
extractor pass costs the same to the tenth of a millisecond with one output or two, and the
head and decoder go from 0.6 ms to 2.0 ms per frame.

The runtime inference contract is `detectSpatialMap(detector, cells, options)` in
`src/lib/ml/detector.ts`. It accepts one packed pair of feature maps, coarse first, and
returns normalized `x`, `y`, `w`, `h`, label and score values after thresholding and per
label suppression. A head restored from an older project reads only the coarse prefix of
the same buffer, which is what lets a detect model trained before 0.6.1 keep working
unchanged. The app-facing `store.detectObjects(source, scoreThreshold)` extracts the maps
and calls the same contract used by the live and still-image views.

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
    detect.prev.json  checkpoint of the detector replaced by the last automatic retrain
```

`project.json`:

```jsonc
{
  "format": "belo-project",
  "formatVersion": 3,
  "appVersion": "0.6.3",
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
    // epochs is the round the run kept, which in a detect project with a
    // holdout is the best one rather than the last one it ran
    "settings": { "epochs": 40, "batchSize": 16, "learningRate": 0.001, "hiddenUnits": 100, "validationSplit": 0.15, "augment": false }
  },
  "checkpoint": null,            // same shape as model, describes head.prev.json
                                 // or, in a detect project, detect.prev.json
  "modelStale": false,           // true when data changed after this model trained
  // augment is detect only and inert elsewhere: mirrored, slightly relit copies
  "settings": { "epochs": 40, "batchSize": 16, "learningRate": 0.001, "hiddenUnits": 100, "validationSplit": 0.15, "augment": false }
}
```

`model/head.json` stores the head weights as plain arrays, the label order they were
trained against and the mode they were trained in, so reopening a project restores a
working classifier without retraining. `model/detect.json` does the same for a detect
project and additionally records every feature map the head reads, with the anchor scales
and ratios on each, and the head width it was built with, so a head keeps loading if those
defaults ever move. That file carries its own `formatVersion`: version 1, written before
0.6.1, describes one 7x7 map in `rows`, `cols`, `scales` and `ratios`, and still loads and
still answers exactly what it used to. Version 2 adds a `featureScales` list and keeps the
version 1 fields filled in with the coarse map. Nothing retrains itself on upgrade; a
version 1 head keeps reading the coarse map until the project is trained again.

Detect projects are created with different training settings, because a detection head
sees hundreds of anchors per picture instead of one embedding: 150 epochs, batches of 8, a
learning rate of 0.002, a head width of 96, a fifth of the pictures held back and mirrored
copies on. Those 150 epochs are a ceiling rather than a plan: with a holdout, a run stops
as soon as the holdout loss has not improved for ten of them. Single and multi projects are
created with the same settings they always were, with `augment` off, where it does nothing.

A manifest written before `augment` existed gains it from the defaults for its mode on the
next load, which means an old detect project starts using mirrored copies and an old
classifier project does not change at all.

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

Release verification. `npm test` runs all eight in this order and prints how long each one
took, which is exactly what CI does:

```bash
npm run check:boxes       # manifest v3 and one-finger box geometry
npm run check:detect      # anchors, loss inputs, decoding, scoring and flip geometry
npm run check:curve       # which epoch a run keeps and what it tells you afterwards
npm run check:store       # stale-model persistence, saves, retrain policy, checkpoint, undo
npm run check:camera      # camera seam against a mocked MediaDevices
npm run check:camera-live # a real camera session in headless Chrome, with fake devices
npm run check:overlay     # the multi label attention overlay, trained and painted for real
npm run check:learning    # learning proof plus the exact Use inference path
```

The two browser checks drive headless Chrome over the DevTools protocol. They never take
over the screen, they serve the app on a free port that is never 1420, and they report
`skipped` instead of failing on a machine with no Chrome, which is why CI stays green
without one. `CHROME_PATH` picks a specific browser.

One more measurement is not part of `npm test`, because it downloads photographs:

```bash
npm run eval:real         # train and score on real COCO photographs
```

### Detection on real photographs

```bash
npm run eval:real                  # train and score, using the cached data
npm run eval:real -- --repeat 5    # average over five seeds instead of three
npm run eval:real -- --no-augment  # the same run without mirrored copies
npm run eval:real -- --keep-last   # keep the last epoch instead of the best one
npm run eval:real -- --rebuild     # rebuild the image list from the COCO annotations
```

`scripts/eval-real.mjs` is a manual tool, not a CI job: it needs the network the first time
and it downloads about 25 MB of photographs. It exits nonzero only when something is
broken, never because the score was low.

**Method.** `scripts/data/coco-subset.json` is a committed list of 169 COCO val2017
photographs across `person`, `cat` and `dog`, with the human drawn boxes that come with
them. No image data is committed. The classes were not chosen by hand: the rebuild ranks
every COCO class by how many photographs it carries whose objects are large enough for the
grid to place, and takes the top three. A photograph qualifies only when every instance
of every chosen class in it can be labelled honestly: at least one box covers 15 percent of
the cropped frame, no box is smaller than 5 percent, none was cut by more than 40 percent by
the square crop, none fills more than 92 percent of an edge, none is a crowd region, and
there are at most four. A photograph containing an object that cannot be labelled is
dropped rather than labelled partially, because a cat left out of the boxes is a cat the
detector would be punished for finding. Each photograph is then converted exactly the way
the app converts an import: center cropped to a square, resized to 224, boxes moved with it.
Every fifth photograph is held out. Training goes through `trainDetector` with the settings
a new detect project is created with.

**One run is not a measurement.** On 136 photographs the head reaches its best holdout loss
within a handful of epochs and then overfits, so which epoch early stopping keeps, and
therefore what the run scores, moves by up to eight points with nothing changed but the
weight initialization. The tool trains at three fixed seeds by default and `--repeat 5`
uses five; the numbers below are means over the same five seeds on both sides.

**What the second feature map is worth**, 136 training photographs, 33 held out, 215 boxes,
score threshold 0.3, IoU 0.5, mean of five seeded runs:

| | found | right | mean overlap | seconds per run |
| --- | --- | --- | --- | --- |
| 0.6.0, one 7x7 map, 294 anchors | 44.5% | 41.5% | 0.664 | 13.7 |
| **0.6.1, 7x7 and 14x14, 1470 anchors** | **56.5%** | **48.7%** | **0.676** | 36.9 |

The second map is worth twelve points of recall and seven of precision, and it costs 2.7
times the training time and three times the feature cache. The gain in mean overlap looks
small at 0.012 and is understated, because mean overlap is measured only over the boxes
that already count as found: raising recall by twelve points adds a tail of marginal boxes
just over the 0.5 bar and drags the average down. On the single seed the tables below are
printed from it moves from 0.624 to 0.684.

**The detailed run, seed 913**, which is what `npm run eval:real` prints first:

| | found | right | mean overlap |
| --- | --- | --- | --- |
| person | 70% (7/10) | 70% (7/10) | |
| cat | 50% (8/16) | 50% (8/16) | |
| dog | 43% (6/14) | 35% (6/17) | |
| **held out overall** | **53%** (21/40) | **49%** (21/43) | **0.684** |
| the pictures it trained on | 86% | 87% | 0.763 |

That is still weak, and it is worth saying plainly rather than burying: on real photographs
this detector finds a little over half of what a person drew, and about half of what it
reports is real. The synthetic proof's 98 percent and 100 percent do not transfer. Three
reasons, in the order they matter:

- **136 photographs.** The holdout loss turns upward after the third epoch and never comes
  back down. There is not enough data here for the head to learn the idea rather than the
  pictures, which is exactly what the Train view now says out loud. This is now clearly the
  binding constraint: it is why the seed matters as much as it does, and it is why the
  second feature map bought twelve points rather than thirty.
- **Still a coarse frame.** 1470 anchors on a 16 pixel stride is a great deal better than
  294 on a 32 pixel one, and the sample renders show it: a box that used to sit a cell off
  the cat now tracks the drawn box closely. What is left over is mostly label confusion
  rather than placement, a cat box and a dog box on the same animal, which per label
  suppression cannot merge by construction.
- **COCO is harder than a kitchen table.** Low light, black cats on dark sofas, animals
  half behind furniture, several people at different depths. A user photographing one
  object on their own desk from a few angles is a much easier problem than this, which is
  the case Belo is actually for.

**What the other training choices are worth here**, five seeds each unless noted:

| | found | right | mean overlap | seconds |
| --- | --- | --- | --- | --- |
| originals only | 54.5% | 49.8% | 0.665 | 21.7 |
| plus mirrored copies | 56.5% | 48.7% | 0.676 | 36.9 |
| keeping the last epoch instead of the best, one seed | 60% | 33% | 0.628 | 402.7 |

Mirrored copies were worth about five points of recall and nine of precision on the one map
head. On the two map head they are close to a wash, two points of recall for one of
precision, at double the reading time: the finer grid appears to have supplied the same
invariance the mirror was buying. They stay on by default because they still help on the
small, one-object projects Belo is actually for, but this is now a defensible toggle to
switch off on a large project. On the synthetic set in `check:learning` they were always
close to a wash, which makes sense: those shapes are symmetric and their backgrounds are
random noise, so a mirror tells the head almost nothing it did not already have.

Early stopping is a trade rather than a free win, and a starker one on the two map head:
keeping the best epoch gives up seven points of recall and gains sixteen of precision, and
turns a 403 second run into a 37 second one. There is nothing after the first few epochs
but memorization, and the holdout loss says so at the time.

**Sample renders.** Every run writes side by side PNGs to `/tmp/belo-eval-real`, the human
boxes on the left in green and what Belo read on the right in pink, named after what went
wrong in each. They are worth looking at, because they say something the table cannot. On
the same photographs at 0.6.0 and 0.6.1: a cat curled in a bathroom sink used to come back
with a box shifted right and down, clipping the tail, and now comes back with one whose
edges sit on the drawn box; a child under a blue umbrella used to lose a shoulder to a box
too narrow, and now keeps it. What is left is a different failure. A cat standing in a sink
now gets a well placed box and a second, near identical one next to it, because the two
carry different labels and suppression runs per label; and a black cat on a dark striped
sofa is still missed completely. The failures have moved from placement to naming and to
low contrast.

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

To cut a release: make sure `main` is green, then `git tag v0.6.3 && git push origin
v0.6.3`. The version in `package.json`, `src-tauri/tauri.conf.json` and
`src-tauri/Cargo.toml` should match the tag.

Useful extras:

- `npm run icons` redraws the app mark and rebuilds every platform icon set. See
  [The app mark](#the-app-mark).
- `npm run assets` refreshes the bundled TensorFlow.js WASM binaries.
- `npm run check:camera` runs the camera seam against a mocked `MediaDevices`, which covers
  device switching, stale mobile ids and hot plug on a machine with a single webcam.
- `npm run check:camera-live` runs the same seam plus `useWebcam` for real, in headless
  Chrome started with three fake capture devices. It starts a session, enumerates the
  cameras through the seam, switches device mid stream and asserts the old tracks ended,
  the new stream opened the device that was asked for, the video element shows it and the
  pick landed in `localStorage`. It then hides one device behind `enumerateDevices` and
  fires a real `devicechange`: the picker has to drop it, the stream has to come back on
  another camera, and the remembered pick is deliberately kept so a camera that returns is
  honored again. Finally it proves a stale deviceId opens the default camera instead of
  failing. About three seconds.
- `npm run check:overlay` is the end to end proof of the attention overlay. In headless
  Chrome it creates a real multi label project through the store, writes eighteen synthetic
  pictures, trains the real head, predicts with attention on and mounts the real
  `AttentionOverlay` over a live camera feed. Each shape appears on both sides of the frame
  across the training set, so a model that learned position instead of content fails the
  mirrored frame. It asserts one canvas layer, one map per label above the multi label
  threshold, the hot half of each score map, and then reads the canvas back: the painted box
  of each label has to match the box the geometry says it should be, to within a pixel, and
  turning the overlay off has to clear the layer. The storage backend is an in-memory fake
  of the two Tauri entry points in `scripts/browser/tauri-shim.ts`; nothing in `src/` is
  changed or mocked. About five seconds.
- `npm run check:boxes` runs the detect mode box seam: drawing in every direction, moving
  against the edges of the picture, all eight resize grips, and a save and reopen round
  trip of version 1, 2 and 3 manifests.
- `npm run check:detect` runs the detection decision layer: anchor generation and layout
  across both feature maps, overlap, offset encode and decode round trips, anchor matching
  including the forced best match, cell standardizing per map depth, hard negative mining,
  suppression and the precision and recall matcher. It pins the claims the two map layout
  rests on: the maps are joined coarse first, a one map head's anchors are exactly the
  prefix of a two map head's, and a head that wants one map reads the prefix of a packed
  example and ignores the rest. No TensorFlow, no DOM.
- `npm run check:store` uses an in-memory storage double to prove box edits persist the
  stale-model flag, overlapping manifest writes serialize in order and Close flushes the latest
  box deletion. It uses only a fake `/tmp` handle and never opens a real project.
- `npm run check:curve` runs the two decisions taken from a loss curve: which epoch a run
  keeps, including the patience window and the tie and noise rules, and the plain-language
  verdict shown after a run. The verdict checks assert the wording as behaviour, not as
  prose: a memorizing run has to ask for varied angles, backgrounds and lighting, and no
  verdict may contain the words loss, epoch, overfitting or validation, or any number.
- `npm run check:learning` is the proof that the detector learns. It builds a synthetic
  detect project of 64 pictures with coloured shapes at known positions, extracts real
  features with the bundled MobileNet on the WASM backend, trains the head with the
  settings a new detect project is created with, mirrored copies included, and then
  measures on 24 pictures it never saw. The run is seeded. It currently finds 98 percent of
  the held out boxes with 100 percent of its detections right and a mean overlap of 0.85,
  against 99 percent on the pictures it trained on; at 0.6.0, on one feature map, those
  numbers were 80 percent, 89 percent and 0.68, which is why the pass bars were raised with
  the second map. It also rebuilds each held out example as the packed tensor handed to the
  Use view, runs the exact Use inference seam and checks both placement and tensor cleanup;
  all 24 frames contain a correctly placed box at half overlap. A further part saves and
  reloads the head in both stored formats, and proves a format 1 file, one 7x7 map and six
  weight tensors, still loads and still reads exactly the boxes it used to off the packed
  features. Two more check the pixel transform against a hand written probe tensor, and
  prove early stopping restores the best epoch: the holdout loss recomputed from the
  returned weights has to equal the best loss the run recorded, and a deliberately
  overfitting run has to stop early and hand back weights that beat the ones it ended with.
  It takes about a minute, and feature extraction is cached in the temp folder for reruns.
  `--compare` additionally trains the same data without mirrored copies and prints both,
  which doubles the runtime. Needs Node 22.15 or newer.
- `npm run eval:real` is the honest one. Everything above is either pure arithmetic or
  coloured rectangles on striped backgrounds; this trains the real detector on real
  photographs with boxes drawn by other people, and reports what it finds. See
  [Detection on real photographs](#detection-on-real-photographs).

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
    augment.ts            training time flip and colour jitter plan, pure functions
    boxes.ts              detect mode box geometry, pure functions
    curve.ts              which epoch a run keeps, and what to tell the person
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
scripts/data/            the committed COCO photograph list eval:real trains on
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

The Android app builds. `src-tauri/gen/android` is committed the same way
`src-tauri/gen/apple` is: the project sources are in git, and the generated sources, the
native libraries, `local.properties` and every build output are covered by the `.gitignore`
files Tauri writes inside that folder. Configuration is API 24 as the minimum, version code
2000, sandboxed Documents storage and `android.permission.CAMERA`. wry asks for the runtime
camera grant. The system photo picker supplies temporary URI access, and `tauri-plugin-fs`
reads those URIs without broad media-library permissions.

The camera permission is declared in `gen/android/app/src/main/AndroidManifest.xml` and
nowhere else. Tauri does not generate Android permissions from `Cargo.toml`, so the
`[package.metadata.android] app-permissions` key an earlier version of this file suggested
was silently ignored and the first real build shipped without the permission. Editing the
committed manifest is the supported route.

What the toolchain needs, user-local and without sudo:

```bash
# One time: SDK platform, platform-tools, build-tools, cmake and the NDK, under
# ~/Library/Android/sdk, with the licences accepted non-interactively.
sdkmanager --install "platforms;android-36" "platform-tools" "build-tools;36.0.0" \
  "ndk;28.2.13676358" "cmake;4.1.2"
yes | sdkmanager --licenses

rustup target add aarch64-linux-android armv7-linux-androideabi \
  i686-linux-android x86_64-linux-android
```

Then, per shell, a JDK 21 and the SDK paths. Any JDK 21 works; this was built with the
Homebrew `openjdk@21`:

```bash
export JAVA_HOME="$(/usr/libexec/java_home -v 21 2>/dev/null || echo /opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home)"
export ANDROID_HOME="$HOME/Library/Android/sdk"
export NDK_HOME="$ANDROID_HOME/ndk/$(ls -1 "$ANDROID_HOME/ndk" | tail -1)"
export PATH="$JAVA_HOME/bin:$ANDROID_HOME/platform-tools:$PATH"

npx tauri android init            # only needed once, gen/android is committed
npx tauri android build --debug --apk --split-per-abi
npx tauri android dev             # on an emulator or a connected device
```

One APK per ABI keeps each artifact to one native library instead of four. The first build
downloads Gradle 8.14.3 and takes a while; afterwards the Rust compile dominates.

Verified on 2026-07-30 with the SDK above, NDK 28.2.13676358, Gradle 8.14.3 and JDK 21.0.11
on an Apple Silicon Mac. All four debug APKs built:

| ABI | APK | Size |
| --- | --- | --- |
| arm64-v8a | `app-arm64-debug.apk` | 141 MB |
| armeabi-v7a | `app-arm-debug.apk` | 126 MB |
| x86 | `app-x86-debug.apk` | 133 MB |
| x86_64 | `app-x86_64-debug.apk` | 142 MB |

A debug APK is that large because the unstripped debug `libbelo_lib.so` carries the whole
frontend bundle, the MobileNet weights and the TensorFlow.js WASM binaries. `aapt2 dump
badging` on the arm64 APK reads:

```
package: name='app.belo' versionCode='2000' versionName='0.6.3'
minSdkVersion:'24'  targetSdkVersion:'36'
uses-permission: name='android.permission.INTERNET'
uses-permission: name='android.permission.CAMERA'
uses-feature-not-required: name='android.hardware.camera'
native-code: 'arm64-v8a'
```

Not done yet: no emulator run. No arm64 system image is installed and booting one is a
foreground activity, so installing, launching and exercising the app on Android is still
outstanding, as is the on-device pass for project creation and reopening, multi-photo
import, two-label training, model restoration after relaunch, camera denial and grant
behavior, live classification, and rotation or resize behavior.

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
