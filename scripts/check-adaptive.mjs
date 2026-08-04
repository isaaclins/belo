// Pure checks for the screen feed's adaptive controller.
//
// No browser, capture session or window is involved. The controller receives
// synthetic timing observations and must shed work on a slow pipeline, then
// recover after a sustained fast period without oscillating.

const { AdaptiveRateController, makeFastTiming, makeSlowTiming } = await import(
  new URL("../src/lib/adaptiveRate.ts", import.meta.url)
);
const { screenQualityPreset } = await import(new URL("../src/lib/screenQuality.ts", import.meta.url));
const { decodeScreenFrame } = await import(new URL("../src/lib/screenPacket.ts", import.meta.url));

let failures = 0;
const results = [];

function check(name, ok, detail = "") {
  results.push(`${ok ? "ok  " : "FAIL"} ${name}${detail ? ` :: ${detail}` : ""}`);
  if (!ok) failures += 1;
}

const slow = new AdaptiveRateController({ maxRate: 120, initialRate: 120 });
const slowStart = slow.rate;
for (let i = 0; i < 12; i += 1) slow.observe(makeSlowTiming());
check("slow pipeline steps the capture rate down", slow.rate < slowStart, `${slowStart} -> ${slow.rate}`);

const recovered = slow.rate;
for (let i = 0; i < 120; i += 1) slow.observe(makeFastTiming());
check("fast pipeline recovers upward", slow.rate > recovered, `${recovered} -> ${slow.rate}`);
check("recovery never exceeds the display ceiling", slow.rate <= 120, `${slow.rate}`);

const dropping = new AdaptiveRateController({ maxRate: 60, initialRate: 60 });
for (let i = 0; i < 3; i += 1) {
  dropping.observe({ ...makeFastTiming(), dropped: true });
}
check("a dropped-frame streak also steps down", dropping.rate < 60, `${dropping.rate}`);

const cadence = new AdaptiveRateController({ maxRate: 120, initialRate: 120 });
const decision = cadence.observe({
  nativeMs: 1,
  transferMs: 1,
  decodeMs: 1,
  paintMs: 1,
  inferenceMs: 40,
  dropped: false,
});
check("inference cost is cadence weighted instead of blocking every capture frame", decision.totalMs < 10, `${decision.totalMs.toFixed(2)} ms`);

check("Auto follows the native display rate", screenQualityPreset("auto").requestedFps === 0);
check("quality steps reduce resolution before they reduce capture cadence", screenQualityPreset("smooth").maxEdge < screenQualityPreset("detail").maxEdge);

const packet = new ArrayBuffer(19);
const packetBytes = new Uint8Array(packet);
packetBytes.set([0x42, 0x4c, 0x46, 0x52]);
new DataView(packet).setUint32(12, 2500, true);
packetBytes.set([0xff, 0xd8, 0xff], 16);
const decoded = decodeScreenFrame(packet);
check("native timing packets keep JPEG bytes as a raw body", decoded.jpeg.byteLength === 3 && decoded.nativeMs === 2.5);

console.log(results.join("\n"));
console.log(failures === 0 ? "\nall adaptive checks passed" : `\n${failures} check(s) failed`);
process.exit(failures === 0 ? 0 : 1);
