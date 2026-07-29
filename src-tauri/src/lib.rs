// Belo shell: filesystem access for .belo project bundles.
//
// A .belo project is a directory bundle:
//   MyProject.belo/
//     project.json      manifest: labels, image records, model metadata
//     images/*.jpg      normalized 224x224 training images
//     model/head.json       trained classifier head weights
//     model/head.prev.json  checkpoint of the head before the last auto retrain
//     model/detect.json     trained detection head weights, detect projects only
//
// The manifest schema is owned by the frontend, so this layer moves strings
// and bytes only. It never parses project.json beyond a couple of display
// fields used by the recent-projects screen.

use std::fs;
use std::path::{Path, PathBuf};

use base64::engine::general_purpose::STANDARD as B64;
use base64::Engine;
use serde::{Deserialize, Serialize};
use tauri::Manager;

const BUNDLE_EXT: &str = "belo";
const MANIFEST: &str = "project.json";
const IMAGES_DIR: &str = "images";
const MODEL_DIR: &str = "model";
const CHECKPOINT_FILE: &str = "head.prev.json";
const DETECTOR_FILE: &str = "detect.json";
const RECENTS_FILE: &str = "recents.json";
const MAX_RECENTS: usize = 12;

type Res<T> = Result<T, String>;

fn fail(context: &str, err: impl std::fmt::Display) -> String {
    format!("{context}: {err}")
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct RecentProject {
    pub path: String,
    pub name: String,
    pub opened_at: String,
}

#[derive(Serialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct ProjectSummary {
    pub path: String,
    pub name: String,
    pub label_count: usize,
    pub image_count: usize,
    pub trained: bool,
    pub updated_at: String,
    pub missing: bool,
}

/// Rejects path segments that could escape the bundle directory.
fn safe_name(name: &str) -> Res<&str> {
    let bad = name.is_empty()
        || name.contains('/')
        || name.contains('\\')
        || name.contains("..")
        || name.starts_with('.');
    if bad {
        return Err(format!("unsafe file name: {name}"));
    }
    Ok(name)
}

fn bundle_root(path: &str) -> Res<PathBuf> {
    let p = PathBuf::from(path);
    if p.extension().and_then(|e| e.to_str()) != Some(BUNDLE_EXT) {
        return Err(format!("not a .belo project: {path}"));
    }
    Ok(p)
}

fn sanitize_project_name(name: &str) -> String {
    let cleaned: String = name
        .chars()
        .map(|c| match c {
            '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|' => '-',
            c if c.is_control() => '-',
            c => c,
        })
        .collect();
    let trimmed = cleaned.trim().trim_matches('.').trim();
    if trimmed.is_empty() {
        "Untitled".to_string()
    } else {
        trimmed.chars().take(80).collect()
    }
}

fn is_image_file(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_ascii_lowercase())
            .as_deref(),
        Some("jpg" | "jpeg" | "png" | "webp" | "bmp" | "gif")
    )
}

fn projects_base(app: &tauri::AppHandle) -> Res<PathBuf> {
    #[cfg(mobile)]
    {
        return app
            .path()
            .document_dir()
            .map_err(|e| fail("no sandbox documents directory", e));
    }

    #[cfg(not(mobile))]
    app.path()
        .document_dir()
        .or_else(|_| app.path().home_dir())
        .map_err(|e| fail("no documents directory", e))
}

#[tauri::command]
fn default_projects_dir(app: tauri::AppHandle) -> Res<String> {
    let dir = projects_base(&app)?.join("Belo");
    fs::create_dir_all(&dir).map_err(|e| fail("could not create the Belo folder", e))?;
    Ok(dir.to_string_lossy().to_string())
}

#[tauri::command]
fn create_project(dir: String, name: String, manifest: String) -> Res<String> {
    let clean = sanitize_project_name(&name);
    let parent = PathBuf::from(&dir);
    fs::create_dir_all(&parent).map_err(|e| fail("could not create the target folder", e))?;

    // Never clobber an existing bundle: walk up a numeric suffix instead.
    let mut root = parent.join(format!("{clean}.{BUNDLE_EXT}"));
    let mut n = 2;
    while root.exists() {
        root = parent.join(format!("{clean} {n}.{BUNDLE_EXT}"));
        n += 1;
    }

    fs::create_dir_all(root.join(IMAGES_DIR))
        .map_err(|e| fail("could not create the bundle", e))?;
    fs::create_dir_all(root.join(MODEL_DIR)).map_err(|e| fail("could not create the bundle", e))?;
    fs::write(root.join(MANIFEST), manifest)
        .map_err(|e| fail("could not write project.json", e))?;
    Ok(root.to_string_lossy().to_string())
}

#[tauri::command]
fn read_manifest(path: String) -> Res<String> {
    let root = bundle_root(&path)?;
    fs::read_to_string(root.join(MANIFEST)).map_err(|e| fail("could not read project.json", e))
}

#[tauri::command]
fn write_manifest(path: String, manifest: String) -> Res<()> {
    let root = bundle_root(&path)?;
    fs::create_dir_all(root.join(IMAGES_DIR)).map_err(|e| fail("could not open the bundle", e))?;
    fs::create_dir_all(root.join(MODEL_DIR)).map_err(|e| fail("could not open the bundle", e))?;
    // Write to a sibling temp file first so a crash cannot truncate the manifest.
    let tmp = root.join("project.json.tmp");
    fs::write(&tmp, manifest).map_err(|e| fail("could not write project.json", e))?;
    fs::rename(&tmp, root.join(MANIFEST)).map_err(|e| fail("could not save project.json", e))
}

#[tauri::command]
fn write_image(path: String, name: String, data: String) -> Res<String> {
    let root = bundle_root(&path)?;
    let name = safe_name(&name)?;
    let bytes = B64
        .decode(data.as_bytes())
        .map_err(|e| fail("bad image payload", e))?;
    let dir = root.join(IMAGES_DIR);
    fs::create_dir_all(&dir).map_err(|e| fail("could not open the images folder", e))?;
    let target = dir.join(name);
    fs::write(&target, bytes).map_err(|e| fail("could not write the image", e))?;
    Ok(target.to_string_lossy().to_string())
}

#[tauri::command]
fn delete_images(path: String, names: Vec<String>) -> Res<()> {
    let root = bundle_root(&path)?;
    for name in names {
        let name = safe_name(&name)?;
        let target = root.join(IMAGES_DIR).join(name);
        if target.exists() {
            fs::remove_file(&target).map_err(|e| fail("could not delete the image", e))?;
        }
    }
    Ok(())
}

#[tauri::command]
fn write_model(path: String, weights: String) -> Res<()> {
    let root = bundle_root(&path)?;
    let dir = root.join(MODEL_DIR);
    fs::create_dir_all(&dir).map_err(|e| fail("could not open the model folder", e))?;
    fs::write(dir.join("head.json"), weights).map_err(|e| fail("could not write the model", e))
}

#[tauri::command]
fn read_model(path: String) -> Res<Option<String>> {
    let root = bundle_root(&path)?;
    let file = root.join(MODEL_DIR).join("head.json");
    if !file.exists() {
        return Ok(None);
    }
    fs::read_to_string(file)
        .map(Some)
        .map_err(|e| fail("could not read the model", e))
}

// A detect project trains a detection head instead of a classifier, so it gets
// its own file next to head.json rather than sharing one.
#[tauri::command]
fn write_detector(path: String, weights: String) -> Res<()> {
    let root = bundle_root(&path)?;
    let dir = root.join(MODEL_DIR);
    fs::create_dir_all(&dir).map_err(|e| fail("could not open the model folder", e))?;
    fs::write(dir.join(DETECTOR_FILE), weights).map_err(|e| fail("could not write the detector", e))
}

#[tauri::command]
fn read_detector(path: String) -> Res<Option<String>> {
    let root = bundle_root(&path)?;
    let file = root.join(MODEL_DIR).join(DETECTOR_FILE);
    if !file.exists() {
        return Ok(None);
    }
    fs::read_to_string(file)
        .map(Some)
        .map_err(|e| fail("could not read the detector", e))
}

// The checkpoint is a plain copy of a previous head.json, kept so an automatic
// background retrain can always be undone from the Train view.
#[tauri::command]
fn write_checkpoint(path: String, weights: String) -> Res<()> {
    let root = bundle_root(&path)?;
    let dir = root.join(MODEL_DIR);
    fs::create_dir_all(&dir).map_err(|e| fail("could not open the model folder", e))?;
    fs::write(dir.join(CHECKPOINT_FILE), weights)
        .map_err(|e| fail("could not write the checkpoint", e))
}

#[tauri::command]
fn read_checkpoint(path: String) -> Res<Option<String>> {
    let root = bundle_root(&path)?;
    let file = root.join(MODEL_DIR).join(CHECKPOINT_FILE);
    if !file.exists() {
        return Ok(None);
    }
    fs::read_to_string(file)
        .map(Some)
        .map_err(|e| fail("could not read the checkpoint", e))
}

#[tauri::command]
fn has_checkpoint(path: String) -> Res<bool> {
    let root = bundle_root(&path)?;
    Ok(root.join(MODEL_DIR).join(CHECKPOINT_FILE).exists())
}

#[tauri::command]
fn clear_checkpoint(path: String) -> Res<()> {
    let root = bundle_root(&path)?;
    let file = root.join(MODEL_DIR).join(CHECKPOINT_FILE);
    if file.exists() {
        fs::remove_file(&file).map_err(|e| fail("could not clear the checkpoint", e))?;
    }
    Ok(())
}

#[tauri::command]
fn read_file_base64(path: String) -> Res<String> {
    let bytes = fs::read(&path).map_err(|e| fail("could not read the file", e))?;
    Ok(B64.encode(bytes))
}

/// Lists image files inside a directory, one level deep by default.
#[tauri::command]
fn list_images_in_dir(dir: String, recursive: bool) -> Res<Vec<String>> {
    let mut found = Vec::new();
    let mut queue = vec![PathBuf::from(dir)];
    let mut depth_guard = 0usize;

    while let Some(current) = queue.pop() {
        depth_guard += 1;
        if depth_guard > 5000 {
            break;
        }
        let entries = match fs::read_dir(&current) {
            Ok(entries) => entries,
            Err(_) => continue,
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                if recursive && path.extension().and_then(|e| e.to_str()) != Some(BUNDLE_EXT) {
                    queue.push(path);
                }
            } else if is_image_file(&path) {
                found.push(path.to_string_lossy().to_string());
            }
        }
    }
    found.sort();
    Ok(found)
}

fn summarize(path: &Path) -> ProjectSummary {
    let name = path
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "Untitled".to_string());
    let mut summary = ProjectSummary {
        path: path.to_string_lossy().to_string(),
        name,
        label_count: 0,
        image_count: 0,
        trained: false,
        updated_at: String::new(),
        missing: !path.join(MANIFEST).exists(),
    };
    if summary.missing {
        return summary;
    }
    if let Ok(text) = fs::read_to_string(path.join(MANIFEST)) {
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(&text) {
            if let Some(n) = value.get("name").and_then(|v| v.as_str()) {
                summary.name = n.to_string();
            }
            summary.label_count = value
                .get("labels")
                .and_then(|v| v.as_array())
                .map(|a| a.len())
                .unwrap_or(0);
            summary.image_count = value
                .get("images")
                .and_then(|v| v.as_array())
                .map(|a| a.len())
                .unwrap_or(0);
            summary.updated_at = value
                .get("updatedAt")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();
            let has_model_meta = value.get("model").is_some_and(|model| !model.is_null());
            let model_file = if value.get("mode").and_then(|v| v.as_str()) == Some("detect") {
                DETECTOR_FILE
            } else {
                "head.json"
            };
            summary.trained = has_model_meta && path.join(MODEL_DIR).join(model_file).exists();
        }
    }
    summary
}

#[tauri::command]
fn project_summary(path: String) -> Res<ProjectSummary> {
    Ok(summarize(&PathBuf::from(path)))
}

#[tauri::command]
fn scan_projects(dir: String) -> Res<Vec<ProjectSummary>> {
    let entries = match fs::read_dir(&dir) {
        Ok(entries) => entries,
        Err(_) => return Ok(Vec::new()),
    };
    let mut out: Vec<ProjectSummary> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir() && p.extension().and_then(|e| e.to_str()) == Some(BUNDLE_EXT))
        .map(|p| summarize(&p))
        .filter(|s| !s.missing)
        .collect();
    out.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
    Ok(out)
}

fn recents_path(app: &tauri::AppHandle) -> Res<PathBuf> {
    let dir = app
        .path()
        .app_config_dir()
        .map_err(|e| fail("no config directory", e))?;
    fs::create_dir_all(&dir).map_err(|e| fail("could not create the config directory", e))?;
    Ok(dir.join(RECENTS_FILE))
}

fn load_recents(app: &tauri::AppHandle) -> Vec<RecentProject> {
    recents_path(app)
        .ok()
        .and_then(|p| fs::read_to_string(p).ok())
        .and_then(|t| serde_json::from_str::<Vec<RecentProject>>(&t).ok())
        .unwrap_or_default()
}

#[tauri::command]
fn recent_projects(app: tauri::AppHandle) -> Res<Vec<ProjectSummary>> {
    Ok(load_recents(&app)
        .into_iter()
        .map(|r| summarize(&PathBuf::from(r.path)))
        .collect())
}

#[tauri::command]
fn remember_project(
    app: tauri::AppHandle,
    path: String,
    name: String,
    opened_at: String,
) -> Res<()> {
    let mut recents = load_recents(&app);
    recents.retain(|r| r.path != path);
    recents.insert(
        0,
        RecentProject {
            path,
            name,
            opened_at,
        },
    );
    recents.truncate(MAX_RECENTS);
    let file = recents_path(&app)?;
    let text = serde_json::to_string_pretty(&recents).map_err(|e| fail("bad recents data", e))?;
    fs::write(file, text).map_err(|e| fail("could not save recent projects", e))
}

#[tauri::command]
fn forget_project(app: tauri::AppHandle, path: String) -> Res<()> {
    let mut recents = load_recents(&app);
    recents.retain(|r| r.path != path);
    let file = recents_path(&app)?;
    let text = serde_json::to_string_pretty(&recents).map_err(|e| fail("bad recents data", e))?;
    fs::write(file, text).map_err(|e| fail("could not save recent projects", e))
}

/// Erases a project bundle. This is the only recursive delete in the app, so it
/// refuses anything that is not a real .belo directory holding a manifest: no
/// files, no symlinks, no plain folders that merely carry the extension.
#[tauri::command]
fn delete_project(app: tauri::AppHandle, path: String) -> Res<()> {
    let root = bundle_root(&path)?;
    let meta = fs::symlink_metadata(&root).map_err(|e| fail("no such project", e))?;
    if meta.file_type().is_symlink() || !meta.is_dir() {
        return Err(format!("not a project directory: {path}"));
    }
    if !root.join(MANIFEST).is_file() {
        return Err(format!("not a .belo project, no {MANIFEST}: {path}"));
    }
    fs::remove_dir_all(&root).map_err(|e| fail("could not delete the project", e))?;
    forget_project(app, path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summary_ignores_an_orphan_detector_file() {
        // The suffix has to stay filename safe on every platform, so it is the
        // process id plus the clock rather than the thread name, which on this
        // test is the module path and carries colons Windows will not accept.
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let root = std::env::temp_dir().join(format!(
            "zz-belo-summary-{}-{unique}.belo",
            std::process::id()
        ));
        let model_dir = root.join(MODEL_DIR);
        fs::create_dir_all(&model_dir).unwrap();
        fs::write(model_dir.join(DETECTOR_FILE), "{}").unwrap();
        fs::write(
            root.join(MANIFEST),
            r#"{"name":"Check","mode":"detect","labels":[],"images":[],"model":null}"#,
        )
        .unwrap();

        assert!(!summarize(&root).trained);

        fs::write(
            root.join(MANIFEST),
            r#"{"name":"Check","mode":"detect","labels":[],"images":[],"model":{"labelIds":[]}}"#,
        )
        .unwrap();
        assert!(summarize(&root).trained);

        fs::remove_dir_all(root).unwrap();
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![
            default_projects_dir,
            create_project,
            read_manifest,
            write_manifest,
            write_image,
            delete_images,
            write_model,
            read_model,
            write_detector,
            read_detector,
            write_checkpoint,
            read_checkpoint,
            has_checkpoint,
            clear_checkpoint,
            read_file_base64,
            list_images_in_dir,
            project_summary,
            scan_projects,
            recent_projects,
            remember_project,
            forget_project,
            delete_project
        ])
        .run(tauri::generate_context!())
        .expect("error while running Belo");
}
