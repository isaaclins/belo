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
   find. Once Belo has learned from your first pictures, it opens each new one with its
   own guesses already drawn on it, dashed, and you mostly just say yes.
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

- **Let Belo draw the boxes and just confirm them.** After you have boxed enough pictures
  by hand, Belo trains itself in the background. From then on, a picture you have not drawn
  on yet opens with Belo's guesses already on it, as dashed rectangles with a name and how
  sure it is. Click one to keep it, press its small x to throw it away, or draw your own box
  straight over it. **Enter** keeps every guess on the picture and **space** keeps them and
  opens the next one, which is the quick way through a folder of photographs. Nothing is
  saved into your project until you keep it: leave the picture and the guesses are simply
  gone. Guesses only appear on pictures you have not drawn on; on one you have, the
  **Suggest again** button asks for them anyway and still changes nothing on its own.

- **Teach classification from the live camera.** Hold a teaching button to add several
  examples, and choose how quickly Belo takes them.
- **Watch a screen instead of a camera.** On a computer, the Use view has a **Camera |
  Screen** switch. Choose Screen and Belo watches one of your displays exactly the way it
  watches the camera: it says what it sees, draws its boxes in a detect project, and the
  same teaching buttons take frames from the screen, so you can teach it from a video, a
  game, a map or anything else you can put on a display. The first time you try it, macOS
  asks for permission to record the screen, and if you have said no before, Belo shows you
  the settings page where you can turn it back on. Belo starts on the camera every time,
  stopping the screen feed releases it at once, and no screen frame ever leaves your
  device. Phones have no screen source, so the switch does not appear there.
- **Tune a live screen.** The small **Auto, Smooth, Balanced, Detail** control is persisted
  on this device. Auto follows the display mode and Belo measures the native conversion,
  transfer, decode and inference work, dropping frames and stepping the capture rate down
  when needed, then recovering after the pipeline settles. Inference still runs on its own
  cadence, so a fast screen feed does not make the model queue old frames.
- **Keep a screen verdict beside your work.** When Screen is live and the Belo window is
  behind another app, the optional always-on-top capture HUD appears. It never captures
  Belo's own windows, never takes focus, and can be dragged to a remembered position. Use
  **Freeze** to teach a classification frame or draw and label detection boxes; saving uses
  the same project intake and retrain path as the main view. The HUD closes when the main
  window comes forward or the live screen stops, and its **HUD on/off** switch is persisted.
- **Grab one screen region from anywhere.** On macOS, open a project, choose a label if it
  is a classification project, and press **Cmd+Shift+B**. Drag Apple's crosshair over a
  region and it lands in the project like a normal capture, marked as a screen frame and
  queued for the usual background retrain. Press **Esc** to cancel. Detect projects keep
  the new picture unboxed so Belo's existing assisted proposals can suggest boxes later.
  The shortcut is shown in the project header, is removed when the project closes, and a
  background capture gets a native notification with the project name and count. Windows
  and Linux do not offer this shortcut yet.
- **See classification clues.** Turn on **Show attention** for a colored hint showing
  which parts of the picture influenced each classification answer.
- **Give one picture several labels.** For example, the same picture can be both
  "outdoors" and "dog".
- **Gather practice pictures quickly.** [imgscout](https://github.com/isaaclins/imgscout)
  is a separate command line tool that downloads pictures for a search term and lets you
  cull them in your browser, which gives you a folder of test images to add to a project.
- **Keep projects you can back up.** On desktop, each `.belo` project is a normal folder
  containing its pictures and saved learning. You can copy or back it up like any other
  folder.
- **Take a trained model with you.** **Export model** in the Train view saves one zip
  holding what your project learned, the feature extractor it needs, a plain-language
  description of every file, and a small page that runs the model on a picture you drop
  onto it. It works for classification and for detection. It is a bundle for a developer
  to build on, not an app you can install on a phone.

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

Current app version: **0.11.0**. Desktop, iOS and Android all build. The native screen feed and capture HUD are macOS-only; the existing web screen seam remains available where the webview provides it.

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
   Once the project has a trained detector, a picture with nothing drawn on it opens with
   the detector's own guesses on it, dashed, to be kept or thrown away rather than drawn.
   See [Guesses in the annotator](#guesses-in-the-annotator).
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
   frames are dropped rather than queued if the disk cannot keep up. A live Screen adds
   the persisted Auto, Smooth, Balanced and Detail control, plus the optional capture HUD.
   While the camera runs, **Show attention** overlays where the
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
   boxes, ready to annotate later in Label. On macOS, when the main window is not focused,
   the always-on-top capture HUD shows the verdict. Freeze a frame to expose classification
   teach buttons or the small detection box surface. HUD saves use `source: "feedback"` for
   classification and `source: "capture"` for detection, with `feed: "screen"` in both cases,
   so the normal manifest and retrain paths remain the only write path.

4. **Export.** **Export model** packs the current model into one zip, through the native
   save dialog. See [Model export](#model-export).

## Model export

The Train view of a trained project offers **Export model**, in both classification and
detect projects. It is disabled, with the reason written next to it, when there is no
trained model or when the dataset changed after the last run: an export of weights that no
longer describe the pictures would be a file nobody could interpret later.

What comes out is one zip that runs anywhere a static file server does:

```
Shapes-model.zip
  belo-export.json    what this is: mode, labels, input format, thresholds
  classifier.json     the trained head, byte for byte the project's model/head.json
                      (a detect project exports detector.json, its model/detect.json)
  extractor/          the bundled MobileNet v1 (1.0, 224), 56 files, about 16 MB
    model.json
    group1-shard1of1 ...
  example.html        a page that runs the model on a picture you drop on it
  example.js          the inference path, written from the format alone
  README.md           every field above, explained
```

Four decisions are worth stating:

- **The exported format is Belo's own serialization, unconverted.** `model/head.json` and
  `model/detect.json` are copied into the archive as they are, and the README documents
  them field by field: the weight order, the anchor layout, the offset variances, the
  coordinate convention and both thresholds. A converted format would be a second thing to
  keep in step with the app and a second thing that can be wrong.
- **The extractor travels with the model**, because the trained part reads MobileNet
  activations rather than pixels and is useless without it. It is the same copy that ships
  inside Belo, in the TensorFlow.js LayersModel format, so
  `tf.loadLayersModel('extractor/model.json')` loads it.
- **The archive is built in the webview**, in `src/lib/zip.ts`, and the Rust side only
  writes the finished bytes to the path the save dialog returned. Everything that goes in
  is reachable from the frontend: the weights come back through the storage seam, the
  extractor is fetched over the same relative URL the extractor loader uses, and the text
  files are generated. Doing it in Rust would mean a new crate, a second path to the
  bundled model files and a different story on mobile. Entries are deflated with the
  platform's own `CompressionStream` and stored uncompressed where there is none; both are
  ordinary zip entries. The 16 MB extractor packs to about 16 MB, because float32 weights
  are close to incompressible, so the whole export is about 16 to 17 MB.
- **The example is a second implementation, not a call into Belo.** `example.js` loads the
  extractor with TensorFlow.js from a pinned CDN URL and then does the head, the anchors,
  the decoding and the suppression itself, in plain arithmetic over the stored arrays. That
  is the only honest proof that the documentation is sufficient, and
  `npm run check:export` runs exactly that code and compares it to Belo's own answers.

The zip's README says to serve the folder, for example with `npx serve .`, and says plainly
that opening `example.html` from disk cannot work, because a browser refuses to `fetch`
the model files over `file://`.

On a phone there is no save dialog to point at a folder, so an export lands next to the
projects inside the app's own Documents folder and the toast says where it went.

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

## Guesses in the annotator

Once a detect project has a trained detector, the box annotator stops being a blank sheet.
A picture with no boxes on it is run through the detector as soon as it is on screen, and
what comes back is drawn as **guesses**: dashed rectangles in the label's colour, each with
its name and its score. The Use view's score cutoff is the one they obey, so a person who
decided that anything under 0.5 is noise gets that everywhere; it lives in
`src/lib/scoreThreshold.ts` and both views read the same value.

The whole design is one invariant: **a guess is a drawing and nothing else until somebody
says yes to it.**

- Guesses never reach `project.json`. They are React state in `BoxAnnotator`, they carry a
  session id that is thrown away, and the only way one becomes a box is `addBox`, the exact
  call a dragged rectangle makes. Undo covers an acceptance, the project is marked changed,
  and the debounced background retrain counts the new boxes like any others.
- Leaving the picture, or closing the annotator, discards every guess that was not accepted,
  without a prompt and without a toast.
- A picture that already carries a box gets no guesses. Somebody who drew there has said
  what is in the picture. **Suggest again** in the side panel asks anyway, and it adds
  rectangles on top of the drawn ones rather than replacing anything.
- Keeping one guess does not throw the rest away. The trigger for guessing is an empty
  picture; the set that came back stays on screen while it is worked through.

The gestures: a press that does not travel more than six pixels keeps that guess, its small
x throws it away, and **delete** does the same to whichever one is picked. **Enter** keeps
every guess on the picture and **space** keeps them and turns the page, which is the fast
path through a confirm streak. A whole batch is one undo step, not one per box. A press on
a guess is deliberately not stopped from reaching the picture underneath, so the same drag
that would draw a rectangle across a guess still draws it; only the measured distance
decides which of the two happened, and the guess layer sits under the committed boxes so a
real box always wins a press.

Computing the guesses never blocks the picture. The request is started after the frame is
on screen, it is cancelled if the person turns the page before it lands, and a quiet
"Looking at this picture" line stands in for it meanwhile. It reads the same packed pair of
feature maps a training run does, through the store's shared cache in
`store.proposeBoxes`, so a picture that has already been through a run costs a head pass
and nothing else, and one that has not leaves its maps behind for the next run to reuse.
That cache is the memory cost: 588 KB per picture looked at, the same number detect
training already pays, and it is dropped with the project.

**A retrain that finishes while the annotator is open refreshes the current picture only if
nobody has decided anything on it yet.** Untouched means no guess kept and none thrown away.
That is the less glitchy of the two options: an untouched picture has nothing to lose by
being redrawn from the better model, while swapping rectangles out from under a hand that
is halfway through deciding about them is exactly the kind of thing that makes a tool feel
haunted. A picture that has been touched keeps what it started with and picks up the new
model on the next picture, which is at most one page away.

The first time guesses ever appear, one line explains them and can be dismissed; the
dismissal is remembered in `localStorage` under `belo.detect.guessHint`, not in the project.

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

Those four names are the only files Belo writes into `model/`, and they are the only ones
it will ever delete. **A model that stops being valid is deleted rather than ignored.**
Adding or removing a label changes the head's output contract, so the stored weights
describe a project that no longer exists and there is no honest way to map the old outputs
onto the new label set; the weights and the checkpoint go at that moment, and the manifest
stops claiming a model in the same edit. Opening a project cleans the same folder again: a
file belonging to the other kind of project, weights whose label set no longer matches the
manifest, a model file the manifest describes no model for, or a checkpoint the manifest no
longer describes. When a live model file is removed the manifest is corrected in place,
with `updatedAt` untouched so opening a project does not reorder the Home list. Anything
else in `model/`, including a file somebody put there by hand, is left alone: the Rust
command refuses any name outside those four rather than deleting it. The decisions are pure
and live in `src/lib/hygiene.ts`, and every case is in `npm run check:store`.

`project.json`:

```jsonc
{
  "format": "belo-project",
  "formatVersion": 3,
  "appVersion": "0.11.0",
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
      "feed": "screen",          // only on frames taken from a screen; absent means camera
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

`feed` arrived in 0.9.0 and did not move the format version: it is written only on frames
taken from a screen, so every project written before it reads exactly as it always did and
an older Belo ignores a field it does not know.

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

## Known past anomalies

Two things were seen once during this project's life, are not reproducible now, and are
written down here rather than quietly forgotten.

- **A project read "0 labels, 0 images, trained" in the Home list** while it actually had
  labels and pictures in it. It was deleted before anybody looked at its manifest, so there
  is no evidence left and no root cause. It is unknown whether the manifest was damaged,
  whether the summary was read from a bundle that was mid-write, or whether the list simply
  showed a stale entry.
- **A recents entry disagreed with the manifest**, reporting 45 images where the project
  held 44. Also seen once, also never explained.

Neither has a fix, because neither has a cause. What exists instead are two checks in
`npm run check:store` that pin the claims those symptoms violated: a summary taken from a
manifest with content can never report zero labels or zero images, and a recents entry that
disagrees with the manifest resolves to the manifest when the project is opened. The second
one is behaviour as well as a check: `openProject` recomputes the summary from the manifest
it just parsed and replaces the list entry with it, in `src/lib/summary.ts`, because the
recents index and the folder scan are by construction older than the manifest the app is
holding. `trained` is also only ever true when the manifest describes a model **and** the
weights are on disk, on both the Rust and the frontend side, which is why the model folder
cleanup above corrects the manifest when it deletes a live model file.

If either symptom ever comes back, the thing to save is the bundle's `project.json` and the
recents index at `<app config dir>/recents.json`, before the project is touched again.

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

Release verification. `npm test` runs all seventeen in this order and prints how long each one
took, which is exactly what CI does:

```bash
npm run check:boxes       # manifest v3 and one-finger box geometry
npm run check:detect      # anchors, loss inputs, decoding, scoring and flip geometry
npm run check:curve       # which epoch a run keeps and what it tells you afterwards
npm run check:store       # model files, saves, retrain policy, checkpoint, undo, summaries
npm run check:selection   # Cmd/Ctrl+A follows the visible collection without editing surprises
npm run check:snip        # shortcut capture, cancel, notification and close race with fixtures
npm run check:adaptive    # slow screen pipelines step down and recover
npm run check:camera      # camera seam against a mocked MediaDevices
npm run check:camera-live # a real camera session in headless Chrome, with fake devices
npm run check:screen      # the screen source, driven against a synthetic desktop
npm run check:hud         # HUD lifecycle, focus safety, freeze, teach and box flows
npm run check:overlay     # the multi label attention overlay, trained and painted for real
npm run check:propose     # the annotator's guesses stay guesses until they are kept
npm run check:export      # an exported model, unzipped and run outside Belo
npm run check:learning    # learning proof plus the exact Use inference path
```

The headless browser checks drive Chrome over the DevTools protocol. They never take over
the screen, they serve the app on a free port that is never 1420, and they report `skipped`
instead of failing on a machine with no Chrome, which is why CI stays green without one.
The screen-region check uses fixture bytes and a fake native runner; it never executes
`screencapture`. `check:adaptive` drives the rate controller with pure fast, slow and dropped
frame timings. `check:hud` uses a fake Tauri window and event bridge, so it proves lifecycle,
focus safety, filter refresh, frozen classification teaching and detection box writes without
creating a native window. `CHROME_PATH` picks a specific browser. `check:export` additionally needs the
network, because the page it exports loads TensorFlow.js from a pinned CDN URL the way any
exported folder does, and it skips with a message when that URL is unreachable.

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

To cut a release: make sure `main` is green, then `git tag v0.11.0 && git push origin
v0.11.0`. The version in `package.json`, `src-tauri/tauri.conf.json` and
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
- `npm run check:export` is the closed loop on model export. In headless Chrome it trains a
  real classifier and builds a real detector, exports both through the app's own export
  path, and reports the two archives plus the answers Belo itself gives for one fixture
  picture. Node then unzips an archive with its own reader, asserts the model, the extractor,
  the README and the example are all in it, serves the unzipped folder on a free port the
  way the zip's README tells a person to, loads `example.html` in the same browser and asks
  it about the same picture. The example's answer has to match Belo's: the classifier's
  scores within 0.01, and every detected box's corners and score within 0.01, with the same
  number of boxes and the same labels. Measured on this machine the agreement is far tighter
  than the bar, around 1e-7, which is what you would expect from two implementations of the
  same arithmetic rather than two implementations of the same idea. The detector under test
  is freshly built rather than trained, because what is being compared is arithmetic and not
  accuracy, and a fresh head at a 0.05 cutoff produces twenty boxes to disagree over. About
  six seconds, and it needs the network for the pinned TensorFlow.js CDN URL.
- `npm run check:propose` is the closed loop on assisted labeling. In headless Chrome it
  builds a real detect project, hand boxes twelve pictures, trains the real detector, adds
  five pictures with nothing drawn on them and mounts the real `BoxAnnotator` on one of
  them. Every assertion is then made against the component's own DOM and against the
  manifest behind it: the guesses are drawn dashed and land on the shapes at an overlap
  above 0.8, clicking one commits exactly that box to within 1e-4 of the rectangle that was
  drawn, undo takes it back out, the x throws one away and writes nothing, a rectangle
  dragged across a guess is drawn instead of mistaken for keeping it, walking away leaves
  the manifest untouched, space keeps them all and turns the page, enter keeps them all and
  stays, a picture that already has boxes gets no guesses until **Suggest again** is
  pressed, and a newer model refreshes an untouched picture while leaving one that is being
  decided about alone. It needs no network, because the extractor is the copy bundled in
  `public/models`. About six seconds.
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
  stale-model flag, overlapping manifest writes serialize in order and Close flushes the
  latest box deletion. It also covers the model folder: every case of the cleanup plan as
  pure arithmetic, then the same rules through the store, so adding a label really deletes
  the head and its checkpoint and opening a project really removes an incompatible detector
  and the foreign head files while leaving a file Belo does not own untouched. Two more pin
  the claims under [Known past anomalies](#known-past-anomalies), and four pin the stored
  weight layout the export documents, tensor shape by tensor shape. It uses only a fake
  `/tmp` handle and never opens a real project.
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
  components/ScreenQualityPicker.tsx persisted Auto, Smooth, Balanced, Detail control
  lib/
    attention.ts          overlay geometry and thresholding, pure functions
    augment.ts            training time flip and colour jitter plan, pure functions
    boxes.ts              detect mode box geometry, pure functions
    hygiene.ts            which model files have gone dead, pure functions
    summary.ts            what the Home list says about a project, pure functions
    zip.ts                the zip writer the model export packs into
    exportModel.ts        assembling one export: weights, extractor, docs, example
    exportFiles.ts        the README, page and inference code that travel in the zip
    curve.ts              which epoch a run keeps, and what to tell the person
    detect.ts             anchors, matching, offset coding, suppression, scoring
    thumbSize.ts          grid cell size, remembered per machine
    storage.ts            ProjectStorage seam: the only module that talks to Tauri
    capture.ts            CameraSource seam: the only module that calls getUserMedia
    screen.ts             ScreenSource seam: display capture, native or getDisplayMedia
    screenPacket.ts        native timing header decoding, pure and allocation-light
    screenQuality.ts       persisted quality presets for live screen capture
    adaptiveRate.ts        pure slow, fast and dropped-frame rate controller
    hud.ts                 macOS HUD lifecycle, events, position and focus policy
    store.ts              application store: project, disk writes, feature cache, head
    zoom.ts               kills user page zoom: pinch, double tap and zoom shortcuts
    images.ts             normalization, drag and drop extraction
    useWebcam.ts          one live session, camera or screen, over both seams
    types.ts              manifest schema
    ml/backend.ts         WebGL then WASM backend selection
    ml/features.ts        truncated MobileNet feature extractor
    ml/classifier.ts      dense head: build, train, score, serialize
    ml/detector.ts        detection head: build, train, detect, serialize
  views/                  HomeView, LabelView, CaptureModal, TrainView, UseView
                          BoxAnnotator (detect mode box drawing)
                          DetectTrainView (detect mode training and results)
                          DetectUseView (live and single-image detection)
                          HudView (the separate always-on-top capture window)
  styles/                 tokens, base, ui primitives, app layout
src-tauri/src/lib.rs      bundle filesystem commands and the recent projects list
src-tauri/src/screen.rs   ScreenCaptureKit capture and the JPEG frame path, macOS only
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

## The screen as a live source

Both Use views carry a **Camera | Screen** switch on the desktop. Selecting Screen puts a
display into the same stage the camera uses, and everything above the seam, live
predictions, confidence bars, the teach buttons and their hold to repeat, the detection
boxes, the attention overlay and Save frame, runs unchanged, because all of it consumes a
plain `MediaStream`. The switch is hidden on mobile and on any desktop webview that has no
screen source at all. Belo always opens on the camera; each feed remembers its own pick, so
going to the screen and back returns to the camera that was running.

### The research verdict

`getDisplayMedia` cannot be used inside Belo's macOS webview, and the reason is worth
writing down because the API looks available.

A hidden wry webview built against the same versions Belo ships (wry 0.55, tao 0.35) was
loaded from `http://localhost` and asked what it had. It reported
`typeof navigator.mediaDevices.getDisplayMedia === "function"`, with the method on the
prototype, and a call without a user gesture rejected with
`InvalidStateError: getDisplayMedia must be called from a user gesture handler`, which is
the standard's own refusal and not a missing feature.

What happens after a gesture is decided in wry, not in the page. WebKit asks its UI
delegate for permission, and wry implements exactly one of the two callbacks:
`webView:requestMediaCapturePermissionForOrigin:...` is present in
`wry/src/wkwebview/class/wry_web_view_ui_delegate.rs` and grants camera and microphone,
while the display capture callback is absent, so WebKit has no picker to present and
denies the request. That is the state tauri-apps/wry#1195 describes and it is still open;
tauri-apps/tauri#2600 records the same conclusion and names ScreenCaptureKit as the way
out. So on macOS the frames come from Rust.

Windows and Linux keep the browser path: WebView2 implements `getDisplayMedia` with its
own picker, and WebKitGTK's support depends on the build and the portal, which is why the
seam checks for the method and hides the switch when it is missing rather than promising
something the machine cannot do. Neither platform has been smoke tested on real hardware
by this repository.

### The frame path

`src-tauri/src/screen.rs` starts a ScreenCaptureKit stream over one display. Auto queries
that display's current CoreGraphics mode and starts at its usable refresh rate; the named
quality steps cap the rate and choose a long edge and JPEG quality. Every frame is converted
to RGB, JPEG encoded and pushed down a raw Tauri channel with a 16 byte timing header. The
frontend decodes each JPEG with `createImageBitmap`, paints it into a canvas and hands
`canvas.captureStream()` to the views, which is why nothing above the seam knows the
difference between a display and a webcam.

The frame controller measures native conversion plus encoding, wall-clock transfer,
`createImageBitmap` decode, canvas paint and the most recent inference pass. Three slow
observations or a dropped-frame streak steps down through the available rates; ten fast
observations step back up. There is no frame queue. The live classifier and detector still
sample every 220 ms, independently of the capture rate.

The release-build synthetic pixel measurement on this M-series Mac is a useful path check,
not a promise about ScreenCaptureKit or somebody else's display: the old 1280 by 830 q72
path measured 10.5 ms and 288 KB, Auto's 1120 by 700 q55 path measured 5.4 ms and 279 KB,
Smooth's 960 by 600 q55 path 3.7 ms and 138 KB, and Balanced's 1440 by 900 q68 path 9.1 ms
and 556 KB. Those numbers cover BGRA stride conversion plus JPEG encoding only. The native
packet and the browser harness expose the remaining transfer and decode timings; a real
ScreenCaptureKit capture still needs the manual smoke pass below. At 60 frames per second,
5.4 ms of native work would be about 32 percent of one core before the adaptive controller
accounts for the webview and inference, while the old path at ten frames per second was
about 11 percent. The alternatives, an `IOSurface` handed straight to the compositor or an
H.264 stream, would both need a video element fed by something WKWebView cannot be given
from Rust, and neither is worth its complexity for a stage that is squared to 224 pixels
before anything looks at it.

ScreenCaptureKit arrived in macOS 12.3 and Belo's declared minimum is 10.15, so the
classes are looked up before anything is offered: an Objective C message to a class that
is not there is a crash, not an error. The Screen button therefore waits for the platform's
own answer and simply never appears on an older Mac.

Display enumeration is `SCShareableContent`, listed after the permission is granted, and
the picker shows each display with its resolution. Window and application capture are not
offered: ScreenCaptureKit can do both, but picking a window means either a system picker
that Belo cannot style or a long list of window titles, and a whole display is enough for
teaching a model. Windows and Linux enumerate nothing, because `getDisplayMedia` deliberately
answers with its own picker instead.

### Capture HUD and window exclusion

The HUD is a second Tauri `WebviewWindow` loaded with `index.html?belo-hud=1`. The main
window owns the live stream and the project store; it sends compact verdict state to the
HUD over Tauri events and receives freeze, teach, resume and box commands back. This keeps
all writes in `addFromDrawable` and `addBox`, so a frozen classification frame is a
`feedback` screen image and a frozen detection frame is a `capture` screen image with the
same retrain debounce as the main view. The HUD is created only on macOS when Screen is
live, the main window is unfocused and the persisted HUD switch is on. It is frameless,
always on top, non-focusable, draggable and position-persisted. Main focus or stopping the
feed destroys it.

ScreenCaptureKit's display filter is built with `initWithDisplay:excludingWindows:`. Before
start, and again after the HUD is shown, Belo enumerates `SCWindow` objects and excludes
every window whose owning application bundle id is `app.belo`, covering both the main
window and the HUD. The Rust test `the_capture_filter_excludes_belo_windows_by_bundle_id`
pins that rule, and the HUD harness refuses to show a HUD if the native filter refresh
fails. That is the honest boundary: if macOS ever cannot return the app's own window
objects, the HUD stays closed instead of shipping a feedback loop.

### Screen-region shortcut

Belo 0.11.0 keeps one-shot screen captures on macOS. With a project open, **Cmd+Shift+B**
registers only for that project, resolves Apple's `screencapture` executable at runtime,
and invokes it with `-i -x` and a private temporary PNG path. **Esc** leaves no file and no
manifest change. A completed region goes
through the same 224 pixel JPEG intake as every other image, with `source: "capture"` and
`feed: "screen"`; the active label is used for classification projects, while detect
projects receive an empty `boxes` list so the existing assisted proposal path can suggest
boxes later. A trained project requests the ordinary debounced background retrain. If the
window was backgrounded, the native notification names the project and its new count;
otherwise the normal Belo toast is used.

The shortcut is unregistered before a project close can finish, and a callback that returns
from the picker after close is discarded. Windows and Linux do not register it in 0.11.0:
there is no clean background region-capture path there yet. Phones compile neither the
shortcut nor notification plugin.

The Rust seam accepts a picker runner, so unit tests cover bytes, cancellation and cleanup
without executing `screencapture`. The headless frontend check supplies fixture PNG bytes
and covers the manifest metadata, retrain request, native notification handoff and close
race.

### Permission

Screen recording on macOS is TCC gated, with no entitlement that can grant it.
`src-tauri/Info.plist` carries `NSScreenCaptureUsageDescription`, without which the system
would terminate the app the moment it triggered the prompt. `CGPreflightScreenCaptureAccess`
is read before every start, so a refusal is known without provoking a dialog, and a refused
feed shows a panel that says what is wrong in plain words and opens
`x-apple.systempreferences:...?Privacy_ScreenCapture` directly. macOS only asks once, so
after a no the settings page is the only way back, and the panel keeps a Try again for the
moment the switch is flipped.

One build detail, learned the hard way. The convenience crate for this framework,
`screencapturekit`, reaches Apple through a Swift bridge and drags in `apple-metal`, whose
Swift sources need the macOS 15 SDK; CI builds the Rust side on `macos-14` and it failed
there while compiling happily on a newer Xcode locally. The bindings used instead are the
plain `objc2-screen-capture-kit` ones, which are generated Rust with no Swift toolchain, no
Metal and no runpath to add, and the frame handler is an `objc2` class implementing
`SCStreamOutput` written out in `src-tauri/src/screen.rs`.

### What is checked and what is not

`npm run check:screen` drives the whole live path in headless Chrome against a synthetic
1443 by 902 desktop, which is a sixteen to ten aspect ratio with odd numbers on both axes
that no webcam produces. It implements the `ScreenPipe` interface the way macOS does, with
JPEG frames, and then drives the app's own hook and store: the source toggle stops every
camera track before the screen arrives, the odd resolution survives the whole frame path,
the stage shows the screen's own pixels, the letterbox math places the model's square
correctly at that ratio, switching display releases the first capture, a real model trained
on the synthetic desktop classifies the live feed and follows it when the content changes, a
taught frame is written with `feed: "screen"`, leaving the screen releases the capture
immediately and a refusal is reported as a refusal. The Rust side's own tests cover the
stride handling, the JPEG encoding and the display fitting, and list the real displays
through the actual bindings when the permission happens to be granted already, which is
what proves the objc2 side links and answers.

What none of that covers is ScreenCaptureKit itself. The native pipe has to be smoke tested
by hand, and so does whatever ScreenCaptureKit costs while it runs:

1. `npm run tauri dev`, open any trained project and go to Use.
2. Press **Screen**. Grant the permission when macOS asks, then press Try again.
3. Check that the stage shows the display, that the purple recording indicator is lit, and
   that the predictions follow what you put on screen. Try **Auto**, **Smooth**, **Balanced**
   and **Detail** while live; the header should show the achieved rate and the stage should
   restart without changing the inference cadence.
4. With two displays connected, switch between them in the picker and confirm the stage
   follows.
5. Press Stop, and confirm the purple indicator goes out at once.
6. Deny the permission in System Settings and press Screen again: the guidance panel has to
   appear instead of an error, and its button has to open the right settings page.
7. Open a classification project, choose a label, press **Cmd+Shift+B**, drag a region and
   confirm the new tile is tagged `screen`, the header still shows **Cmd+Shift+B**, and the
   normal retrain marker appears. Press **Esc** in the picker and confirm no tile is added.
8. Background Belo before the next capture and confirm the native notification names the
   project and the image count. Bring it forward and confirm the next capture uses the Belo
   toast instead.
9. Open a detect project and capture a region. Confirm its tile starts with zero boxes so
   assisted proposals remain available. Close the project, press the shortcut, and confirm
   no picker opens and no image is added.
10. With Screen still live, leave the Belo window unfocused with another app in front. The
    compact always-on-top HUD should appear without activating Belo. Drag its dark header,
    bring Belo forward and confirm the HUD closes, then unfocus Belo again and confirm it
    returns at the remembered position. Turn **HUD off**, unfocus and confirm it stays hidden.
11. In a classification project, press **Freeze** in the HUD. Confirm the image stops, the
    **This is X** controls appear, and teaching one label adds a feedback image with
    `feed: "screen"` and starts the normal retrain marker. Resume and confirm the live frame
    moves again.
12. In a detect project, freeze the HUD, choose a label, drag a box over the frozen image and
    confirm the saved capture has `source: "capture"`, `feed: "screen"` and one stored box.
13. Put the HUD over a visible part of the display and confirm the main Screen stage never
    shows the Belo window or HUD. If it does, stop and report that the ScreenCaptureKit
    exclusion is not working rather than using the HUD.

## License

MIT.
