// An in-memory stand-in for the Rust side, for the headless browser checks.
//
// The frontend reaches the filesystem only through the ProjectStorage seam in
// src/lib/storage.ts, and that seam reaches Tauri only through invoke() and
// convertFileSrc(). Faking those two functions therefore swaps the whole
// backend without touching a single line of src/: the real storage module, the
// real store and the real components run unchanged, they just talk to a Map.
//
// This file lives in scripts/ on purpose. It is test scaffolding, never shipped.

interface Report {
  done: boolean;
  results: { name: string; ok: boolean; detail?: string }[];
  error?: string;
}

const files = new Map<string, string>();
const projects = new Map<string, { name: string; openedAt: string }>();

function must(map: Map<string, string>, key: string, what: string): string {
  const value = map.get(key);
  if (value === undefined) throw new Error(`${what} is missing: ${key}`);
  return value;
}

type Args = Record<string, unknown>;

const commands: Record<string, (args: Args) => unknown> = {
  default_projects_dir: () => "/mem/Belo",

  create_project: ({ dir, name, manifest }: Args) => {
    let root = `${dir}/${String(name).replace(/[^\w .-]+/g, "_")}.belo`;
    let n = 2;
    while (files.has(`${root}/project.json`)) {
      root = `${dir}/${name} ${n}.belo`;
      n += 1;
    }
    files.set(`${root}/project.json`, String(manifest));
    return root;
  },

  read_manifest: ({ path }: Args) => must(files, `${path}/project.json`, "project.json"),
  write_manifest: ({ path, manifest }: Args) => {
    files.set(`${path}/project.json`, String(manifest));
  },

  write_image: ({ path, name, data }: Args) => {
    const target = `${path}/images/${name}`;
    files.set(target, String(data));
    return target;
  },
  delete_images: ({ path, names }: Args) => {
    for (const name of names as string[]) files.delete(`${path}/images/${name}`);
  },

  write_model: ({ path, weights }: Args) => void files.set(`${path}/model/head.json`, String(weights)),
  read_model: ({ path }: Args) => files.get(`${path}/model/head.json`) ?? null,
  write_detector: ({ path, weights }: Args) =>
    void files.set(`${path}/model/detect.json`, String(weights)),
  read_detector: ({ path }: Args) => files.get(`${path}/model/detect.json`) ?? null,

  list_model_files: ({ path }: Args) =>
    [...files.keys()]
      .filter((key) => key.startsWith(`${path}/model/`))
      .map((key) => key.slice(`${path}/model/`.length))
      .filter((name) => !name.includes("/"))
      .sort(),
  delete_model_files: ({ path, names }: Args) => {
    for (const name of names as string[]) files.delete(`${path}/model/${name}`);
  },

  // The export lands in the same map, so a check can read the archive back out.
  write_export: ({ path, data }: Args) => void files.set(String(path), String(data)),
  "plugin:dialog|save": () => exportTarget,

  write_checkpoint: ({ path, weights }: Args) =>
    void files.set(`${path}/model/head.prev.json`, String(weights)),
  read_checkpoint: ({ path }: Args) => files.get(`${path}/model/head.prev.json`) ?? null,
  has_checkpoint: ({ path }: Args) => files.has(`${path}/model/head.prev.json`),
  clear_checkpoint: ({ path }: Args) => void files.delete(`${path}/model/head.prev.json`),

  write_detector_checkpoint: ({ path, weights }: Args) =>
    void files.set(`${path}/model/detector.prev.json`, String(weights)),
  read_detector_checkpoint: ({ path }: Args) => files.get(`${path}/model/detector.prev.json`) ?? null,
  has_detector_checkpoint: ({ path }: Args) => files.has(`${path}/model/detector.prev.json`),
  clear_detector_checkpoint: ({ path }: Args) => void files.delete(`${path}/model/detector.prev.json`),

  recent_projects: () =>
    [...projects.entries()].map(([path, meta]) => ({
      path,
      name: meta.name,
      openedAt: meta.openedAt,
      labels: 0,
      images: 0,
      mode: "single",
    })),
  scan_projects: () => [],
  remember_project: ({ path, name, openedAt }: Args) => {
    projects.set(String(path), { name: String(name), openedAt: String(openedAt) });
  },
  forget_project: ({ path }: Args) => void projects.delete(String(path)),
  delete_project: ({ path }: Args) => {
    projects.delete(String(path));
    for (const key of [...files.keys()]) if (key.startsWith(`${path}/`)) files.delete(key);
  },

  read_file_base64: ({ path }: Args) => must(files, String(path), "file"),
  list_images_in_dir: () => [],
};

/** Where the fake save dialog says the next export should go. */
let exportTarget = "/mem/export/model.zip";

export function fakeExportTarget(path: string): void {
  exportTarget = path;
}

/** The base64 archive a finished export wrote, or null when there is none. */
export function writtenExport(path: string): string | null {
  return files.get(path) ?? null;
}

export function installTauriShim(): void {
  const internals = {
    invoke(cmd: string, args: Args = {}) {
      const run = commands[cmd];
      if (!run) return Promise.reject(new Error(`No fake for the ${cmd} command`));
      try {
        return Promise.resolve(run(args ?? {}));
      } catch (err) {
        return Promise.reject(err);
      }
    },
    // Images are written as JPEG base64, so the webview can read them straight
    // back as a data URL instead of going through an asset protocol.
    convertFileSrc(path: string) {
      const data = files.get(path);
      return data ? `data:image/jpeg;base64,${data}` : "data:,";
    },
    transformCallback(callback: unknown) {
      return callback;
    },
  };
  (window as unknown as { __TAURI_INTERNALS__: typeof internals }).__TAURI_INTERNALS__ = internals;
}

/** Collects assertions and publishes them where the node side can read them. */
export function reporter() {
  const report: Report = { done: false, results: [] };
  (window as unknown as { __BELO_CHECK__: Report }).__BELO_CHECK__ = report;
  return {
    check(name: string, ok: boolean, detail: unknown = "") {
      const text = typeof detail === "string" ? detail : JSON.stringify(detail);
      report.results.push({ name, ok: Boolean(ok), detail: text });
      console.log(`${ok ? "ok  " : "FAIL"} ${name}${text ? ` :: ${text}` : ""}`);
    },
    fail(err: unknown) {
      report.error = err instanceof Error ? `${err.message}\n${err.stack ?? ""}` : String(err);
      report.done = true;
    },
    finish() {
      report.done = true;
    },
  };
}
