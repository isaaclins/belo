// Deterministic check of everything Belo reads out of a loss curve.
//
// Two decisions come out of the same numbers and both are easy to get subtly
// wrong: which epoch a run keeps, and what the person is told afterwards. Both
// are pure functions in src/lib/curve.ts, so both are checked here without
// TensorFlow, without a DOM and in a few milliseconds.
//
//   node scripts/check-curve.mjs
//
// Node 22.6 or newer runs the TypeScript source directly through type stripping.

const { bestEpochIndex, outOfPatience, readCurves, readScores, PATIENCE } = await import(
  new URL("../src/lib/curve.ts", import.meta.url)
);

let failures = 0;
const results = [];

function check(name, ok, detail = "") {
  results.push(`${ok ? "ok  " : "FAIL"} ${name}${detail ? ` — ${detail}` : ""}`);
  if (!ok) failures += 1;
}

// ---------- Which epoch to keep ----------

check(
  "the lowest holdout loss wins",
  bestEpochIndex([1.0, 0.8, 0.4, 0.6, 0.9]) === 2,
  `${bestEpochIndex([1.0, 0.8, 0.4, 0.6, 0.9])}`,
);
check("a run that only improves keeps its last epoch", bestEpochIndex([1.0, 0.8, 0.6]) === 2);
check("a run with no holdout numbers has no best epoch", bestEpochIndex([null, null]) === -1);
check(
  "a tie keeps the earlier epoch",
  bestEpochIndex([1.0, 0.5, 0.5, 0.5]) === 1,
  `${bestEpochIndex([1.0, 0.5, 0.5, 0.5])}`,
);
check(
  "noise below the minimum delta is not an improvement",
  bestEpochIndex([0.5, 0.5 - 1e-7, 0.5 - 2e-7]) === 0,
);
check("nulls among the numbers are skipped", bestEpochIndex([1.0, null, 0.3, null]) === 2);
check(
  "an infinite loss is never the best epoch",
  bestEpochIndex([Infinity, 0.9, NaN]) === 1,
  `${bestEpochIndex([Infinity, 0.9, NaN])}`,
);

// ---------- When to stop ----------

const flatAfter = (best, tail) => [1.0, best, ...Array.from({ length: tail }, () => best + 0.2)];

check(
  "patience is not out while the loss is still falling",
  !outOfPatience([1.0, 0.9, 0.8, 0.7, 0.6, 0.5], 3),
);
check("patience runs out after the configured window", outOfPatience(flatAfter(0.5, 4), 3));
check("one epoch short of the window keeps going", !outOfPatience(flatAfter(0.5, 2), 3));
check(
  "a rising curve stops as soon as the window is full",
  outOfPatience([0.4, 0.5, 0.6, 0.7, 0.8], 3),
  `best at 0, ${5 - 1 - 0} epochs since`,
);
check("a run with no holdout never stops early", !outOfPatience([null, null, null, null], 1));
check(
  "a run shorter than the window never stops early",
  !outOfPatience([1.0, 0.9], PATIENCE),
);
check(
  "the default patience is the one the detector documents",
  PATIENCE >= 8 && PATIENCE <= 12,
  `${PATIENCE}`,
);

// ---------- What to say about it ----------

const falling = (from, to, n) =>
  Array.from({ length: n }, (_, i) => from + ((to - from) * i) / (n - 1));
const flat = (value, n) => Array.from({ length: n }, () => value);

const noHoldout = readCurves({ training: falling(2, 0.2, 40), holdout: flat(null, 40), heldOut: 0 });
check(
  "a run with no holdout says so instead of guessing",
  noHoldout.tone === "unknown" && /too few pictures/i.test(noHoldout.headline),
  noHoldout.headline,
);
check(
  "the no holdout advice says the numbers will look kind",
  /kinder than real life/i.test(noHoldout.advice),
);

const tooShort = readCurves({ training: [2, 1], holdout: [2, 1], heldOut: 8 });
check("two epochs is not a curve", tooShort.tone === "unknown", tooShort.headline);

// The shape a memorizing run really has: the training loss collapses while the
// holdout loss dips a little, turns around and climbs back past where it began.
const memorizing = readCurves({
  training: falling(2, 0.1, 40),
  holdout: [...falling(2, 1.6, 10), ...falling(1.6, 2.3, 30)],
  heldOut: 10,
});
check(
  "training down and holdout flat and high reads as memorizing",
  memorizing.tone === "bad" && /memorizing/i.test(memorizing.headline),
  memorizing.headline,
);
check(
  "the memorizing advice asks for varied pictures",
  /angles/i.test(memorizing.advice) &&
    /background/i.test(memorizing.advice) &&
    /lighting/i.test(memorizing.advice),
  memorizing.advice,
);
check(
  "the memorizing verdict never mentions loss, epochs or overfitting",
  !/loss|epoch|overfit|validation|variance/i.test(
    `${memorizing.headline} ${memorizing.advice}`,
  ),
);

const learning = readCurves({
  training: falling(2, 0.2, 40),
  holdout: falling(2.1, 0.4, 40),
  heldOut: 12,
});
check(
  "both falling reads as learning well",
  learning.tone === "good" && learning.headline === "Learning well.",
  learning.headline,
);

const stuck = readCurves({ training: flat(1.4, 40), holdout: flat(1.5, 40), heldOut: 12 });
check(
  "nothing moving reads as not learning yet",
  stuck.tone === "bad" && /not picking anything up/i.test(stuck.headline),
  stuck.headline,
);

// Training loss down hard, holdout loss stuck but not far above it. Nothing was
// memorized badly enough to call it out, and nothing much was learned either.
const stalled = readCurves({
  training: falling(2, 1.6, 40),
  holdout: flat(1.7, 40),
  heldOut: 12,
});
check(
  "a holdout that stopped moving without a wide gap is a warning, not a pass",
  stalled.tone === "watch" && /not much/i.test(stalled.headline),
  `${stalled.tone}: ${stalled.headline}`,
);

const partial = readCurves({
  training: falling(2, 1.5, 40),
  holdout: falling(2.1, 1.85, 40),
  heldOut: 12,
});
check(
  "both curves easing down slowly is a warning, not a pass",
  partial.tone === "watch",
  `${partial.tone}: ${partial.headline}`,
);

check(
  "every verdict is a whole sentence with advice attached",
  [noHoldout, tooShort, memorizing, learning, stuck, stalled, partial].every(
    (verdict) =>
      verdict.headline.trim().length > 0 &&
      verdict.headline.trim().endsWith(".") &&
      verdict.advice.trim().length > 20,
  ),
);
check(
  "no verdict contains a number the person would have to interpret",
  [noHoldout, memorizing, learning, stuck, stalled, partial].every(
    (verdict) => !/\d/.test(`${verdict.headline}${verdict.advice}`),
  ),
);

// ---------- The classifier's version ----------

check(
  "a classifier with no holdout says so",
  readScores({ training: 0.99, holdout: null, images: 12 }).tone === "unknown",
);
check(
  "a wide gap between studied and unseen reads as memorizing",
  /memorizing/i.test(readScores({ training: 0.99, holdout: 0.62, images: 60 }).headline),
);
check(
  "a high holdout score reads as learning well",
  readScores({ training: 0.95, holdout: 0.92, images: 60 }).tone === "good",
);
check(
  "a middling holdout score is a warning",
  readScores({ training: 0.8, holdout: 0.7, images: 60 }).tone === "watch",
);
check(
  "a holdout score near chance reads as a failure",
  readScores({ training: 0.7, holdout: 0.4, images: 60 }).tone === "bad",
);
check(
  "nothing trained yet is not a verdict about the data",
  readScores({ training: null, holdout: null, images: 0 }).tone === "unknown",
);

console.log(results.join("\n"));
console.log(failures === 0 ? "\nall curve checks passed" : `\n${failures} check(s) failed`);
process.exit(failures === 0 ? 0 : 1);
