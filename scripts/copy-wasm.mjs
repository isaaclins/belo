// Copies the TensorFlow.js WASM backend binaries into public/ so the app can
// load its fallback backend offline, without touching a CDN.
import { copyFileSync, mkdirSync, readdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const from = join(root, "node_modules", "@tensorflow", "tfjs-backend-wasm", "dist");
const to = join(root, "public", "tfjs-wasm");

mkdirSync(to, { recursive: true });
const files = readdirSync(from).filter((f) => f.endsWith(".wasm"));
for (const file of files) copyFileSync(join(from, file), join(to, file));
console.log(`copied ${files.length} wasm binaries to public/tfjs-wasm`);
