// What a loss curve means, in two answers nobody needs a course to read.
//
// Two questions get asked of the same numbers after a training run:
//
//   1. Which epoch should be kept? That is the early stop rule, and it belongs
//      next to the training loop in ml/detector.ts, which calls into here.
//   2. What should the person be told? That is the verdict, and it belongs in
//      the Train views, which call into here too.
//
// Both are pure arithmetic on plain arrays, so both are checked in node by
// scripts/check-curve.mjs without TensorFlow and without a DOM. Nothing in this
// file may import tf, React or the store.

/** Epochs without a better holdout loss before a run gives up. */
export const PATIENCE = 10;

/**
 * How much better a holdout loss has to be to count as better at all. Without a
 * margin, floating point noise in the fourth decimal keeps resetting patience
 * and a run that stopped improving twenty epochs ago never stops.
 */
export const MIN_DELTA = 1e-4;

/**
 * Index of the best epoch in a list of holdout losses, or -1 when none of them
 * is a number. Ties keep the earlier epoch, which is the cheaper model and the
 * one that has memorized less.
 */
export function bestEpochIndex(
  holdout: readonly (number | null)[],
  minDelta: number = MIN_DELTA,
): number {
  let best = -1;
  let bestValue = Infinity;
  holdout.forEach((value, index) => {
    if (value === null || !Number.isFinite(value)) return;
    if (value < bestValue - minDelta) {
      best = index;
      bestValue = value;
    }
  });
  return best;
}

/**
 * True when the holdout loss has not improved for `patience` epochs, counted
 * from the best epoch. A run with no holdout numbers never stops early: there
 * is nothing to judge it by.
 */
export function outOfPatience(
  holdout: readonly (number | null)[],
  patience: number = PATIENCE,
): boolean {
  const best = bestEpochIndex(holdout);
  if (best < 0) return false;
  const scored = holdout.filter((value) => value !== null && Number.isFinite(value)).length;
  if (scored <= best + 1) return false;
  return holdout.length - 1 - best >= patience;
}

// ---------- The verdict ----------

/**
 * good:    both curves fell, the thing learned the idea.
 * watch:   it learned something, but not much of it survives on unseen pictures.
 * bad:     it memorized the pictures, or it did not learn at all.
 * unknown: the run cannot be judged, usually for want of held out pictures.
 */
export type VerdictTone = "good" | "watch" | "bad" | "unknown";

export interface Verdict {
  tone: VerdictTone;
  /** One sentence, no jargon, no numbers. */
  headline: string;
  /** What to do next, in the same voice. */
  advice: string;
}

export interface CurveInput {
  /** Training loss per epoch, oldest first. */
  training: readonly number[];
  /** Holdout loss per epoch, same length, null where there was none. */
  holdout: readonly (number | null)[];
  /** Pictures held back from training. Zero means the run judged itself. */
  heldOut: number;
}

/** A curve has to fall by this share before it counts as falling at all. */
const FALLING = 0.15;
/** Under this, a curve is flat rather than falling. */
const FLAT = 0.05;
/**
 * How much higher the holdout loss has to sit than the training loss before the
 * run is called memorizing. A held out picture is always a little harder than a
 * studied one, so a small gap is normal and only a wide one is a verdict.
 */
const WIDE_GAP = 1.4;

/** Mean of the first or last quarter of a series, so one noisy epoch cannot decide. */
function edge(series: readonly number[], side: "start" | "end"): number {
  const window = Math.max(1, Math.floor(series.length / 4));
  const slice = side === "start" ? series.slice(0, window) : series.slice(-window);
  return slice.reduce((sum, value) => sum + value, 0) / slice.length;
}

/** Share of its starting height a curve gave up. Negative means it rose. */
function fell(series: readonly number[]): number {
  if (series.length < 2) return 0;
  const start = edge(series, "start");
  const end = edge(series, "end");
  if (!(start > 0)) return 0;
  return (start - end) / start;
}

/**
 * Turns the two loss curves into one thing to say. The order of the branches is
 * the order a person would check them in: can this be judged at all, did it
 * learn anything, did what it learned survive on pictures it had not seen.
 */
export function readCurves(input: CurveInput): Verdict {
  const training = input.training.filter((value) => Number.isFinite(value));
  const holdout = input.holdout.filter(
    (value): value is number => value !== null && Number.isFinite(value),
  );

  if (training.length < 3) {
    return {
      tone: "unknown",
      headline: "This run was too short to judge.",
      advice: "Train for more rounds so there is a shape to read.",
    };
  }

  if (input.heldOut === 0 || holdout.length < 3) {
    return {
      tone: "unknown",
      headline: "There were too few pictures to keep any aside.",
      advice:
        "Belo could only mark its own homework here, so these numbers will look kinder than real life. Add more pictures and it will keep some back and test itself honestly.",
    };
  }

  const trainingFall = fell(training);
  const holdoutFall = fell(holdout);
  const gap = edge(holdout, "end") / Math.max(1e-6, edge(training, "end"));

  if (trainingFall < FLAT) {
    return {
      tone: "bad",
      headline: "Belo is not picking anything up yet.",
      advice:
        "Nothing improved while it studied. Check that the boxes sit tightly around the thing you named, that each label means one kind of thing, and give it more rounds.",
    };
  }

  if (trainingFall >= FALLING && holdoutFall < FLAT && gap >= WIDE_GAP) {
    return {
      tone: "bad",
      headline: "Belo is memorizing your pictures, not learning the idea.",
      advice:
        "It got better and better on the pictures it studied while the ones it had never seen got no easier. Add more varied pictures: different angles, backgrounds and lighting.",
    };
  }

  if (trainingFall >= FALLING && holdoutFall >= FALLING) {
    return {
      tone: "good",
      headline: "Learning well.",
      advice:
        "Pictures it had never seen got easier at the same time as the ones it studied, which is what learning the idea looks like. More varied pictures will still help.",
    };
  }

  if (holdoutFall < FLAT) {
    return {
      tone: "watch",
      headline: "Some of this stuck, but not much.",
      advice:
        "Pictures it had never seen barely got easier. More pictures of the same thing in different places is the fix that helps most.",
    };
  }

  return {
    tone: "watch",
    headline: "Learning, slowly.",
    advice:
      "Both kinds of picture are getting easier, but not by much yet. Train for more rounds, and add pictures where it still gets things wrong.",
  };
}

export interface ScoreInput {
  /** Share right on the pictures it studied, 0 to 1. */
  training: number | null;
  /** Share right on the pictures it kept aside, or null when it kept none. */
  holdout: number | null;
  /** Pictures the run had to work with. */
  images: number;
}

/**
 * The classifier's version of the same favour. It reports accuracy rather than
 * a loss curve, so the reading is a gap between two shares rather than two
 * slopes, but the sentences a person needs are the same ones.
 */
export function readScores(input: ScoreInput): Verdict {
  const { training, holdout } = input;
  if (training === null) {
    return {
      tone: "unknown",
      headline: "Nothing has been trained yet.",
      advice: "Label a few pictures and press Train.",
    };
  }
  if (holdout === null) {
    return {
      tone: "unknown",
      headline: "There were too few pictures to keep any aside.",
      advice:
        "Belo could only mark its own homework here, so this score will look kinder than real life. Add more pictures per label and it will keep some back and test itself honestly.",
    };
  }
  if (training - holdout >= 0.2 && training >= 0.85) {
    return {
      tone: "bad",
      headline: "Belo is memorizing your pictures, not learning the idea.",
      advice:
        "It is almost perfect on the pictures it studied and much worse on the ones it had never seen. Add more varied pictures: different angles, backgrounds and lighting.",
    };
  }
  if (holdout >= 0.85) {
    return {
      tone: "good",
      headline: "Learning well.",
      advice:
        "It gets pictures it had never seen right about as often as the ones it studied. Add pictures of anything it still misses.",
    };
  }
  if (holdout >= 0.6) {
    return {
      tone: "watch",
      headline: "It has the rough idea, not the detail.",
      advice:
        "Look at the pictures it got wrong below. Usually two labels that look alike need more examples of each before they come apart.",
    };
  }
  return {
    tone: "bad",
    headline: "Belo cannot tell your labels apart yet.",
    advice:
      "Most pictures it had never seen came out wrong. Make sure every label means one clear kind of thing, and add many more pictures of each.",
  };
}
