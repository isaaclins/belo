// Snipping a region of the screen straight into the open project.
//
// The hotkey is registered by the frontend through the global shortcut plugin;
// when it fires, the frontend calls `snip_region` and this module runs the
// system's own region picker. On macOS that is `/usr/bin/screencapture -i -x`:
// Apple's crosshair drag select, silent, writing a PNG to a path we hand it.
// The user started it with a key press, so the picker taking over the screen is
// the point rather than a surprise.
//
// Escape is not an error. The picker exits without writing anything, and this
// module reports that as "nothing came back" so the caller can do nothing.
//
// Everything except the process spawn itself is a plain function taking the
// picker as an argument, so the tests drive the whole path with a fake picker
// that writes a file and never touch the real tool.

use std::fs;
use std::path::{Path, PathBuf};

use base64::engine::general_purpose::STANDARD as B64;
use base64::Engine;

/// Apple's own screenshot tool. Present on every macOS Belo runs on.
#[cfg(target_os = "macos")]
pub const PICKER_BIN: &str = "/usr/bin/screencapture";

type Res<T> = Result<T, String>;

fn now_nanos() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0)
}

/// Where one snip is written before it is read back and deleted. Unique per
/// call, so two quick presses cannot read each other's picture.
pub fn snip_target(dir: &Path, pid: u32, unique: u128) -> PathBuf {
    dir.join(format!("belo-snip-{pid}-{unique}.png"))
}

/// Runs `picker` against a fresh temp path and hands back what it wrote, base64
/// encoded for the trip to the webview.
///
/// `Ok(None)` is the cancellation: the picker ran and left no picture. The file
/// is the answer rather than the exit code, because `screencapture` is not
/// consistent about what it returns when the person presses Escape. The temp
/// file is removed on every path, including the failing one.
pub fn snip_with<F>(dir: &Path, picker: F) -> Res<Option<String>>
where
    F: FnOnce(&Path) -> Res<()>,
{
    let target = snip_target(dir, std::process::id(), now_nanos());
    let _ = fs::remove_file(&target);
    let outcome = picker(&target);
    let written = fs::read(&target).ok();
    let _ = fs::remove_file(&target);
    outcome?;
    match written {
        Some(bytes) if !bytes.is_empty() => Ok(Some(B64.encode(bytes))),
        _ => Ok(None),
    }
}

#[cfg(target_os = "macos")]
fn run_system_picker(target: &Path) -> Res<()> {
    // -i is the interactive region select, -x keeps it silent. Only a picker
    // that could not be started at all is an error here; whether it wrote
    // anything is decided by the file.
    std::process::Command::new(PICKER_BIN)
        .arg("-i")
        .arg("-x")
        .arg(target)
        .status()
        .map(|_| ())
        .map_err(|e| format!("could not start the region picker: {e}"))
}

/// Whether this build can snip at all. macOS only for now: see the README note
/// on why Windows and Linux have no equivalent one shot region picker.
#[tauri::command]
pub fn snip_supported() -> bool {
    cfg!(target_os = "macos")
}

/// Opens the system region picker and returns the snip as base64 PNG. Null when
/// the person cancelled.
#[tauri::command]
pub fn snip_region() -> Res<Option<String>> {
    #[cfg(target_os = "macos")]
    {
        snip_with(&std::env::temp_dir(), run_system_picker)
    }
    #[cfg(not(target_os = "macos"))]
    {
        Err("Snipping the screen is a macOS feature for now".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One empty scratch directory per test, filename safe on every platform.
    fn scratch(what: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "zz-belo-snip-{what}-{}-{}",
            std::process::id(),
            now_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// A four pixel PNG, standing in for whatever the picker would have written.
    const FAKE_PNG: &[u8] = &[
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x01, 0x02, 0xFE, 0xFF, 0x7F,
    ];

    #[test]
    fn a_snip_comes_back_as_the_exact_bytes_the_picker_wrote() {
        let dir = scratch("bytes");
        let seen = std::cell::RefCell::new(None);
        let encoded = snip_with(&dir, |target| {
            *seen.borrow_mut() = Some(target.to_path_buf());
            fs::write(target, FAKE_PNG).unwrap();
            Ok(())
        })
        .unwrap()
        .expect("a written snip is handed back");

        assert_eq!(
            B64.decode(encoded).unwrap(),
            FAKE_PNG,
            "the webview has to receive the picture byte for byte"
        );
        let target = seen.into_inner().unwrap();
        assert_eq!(target.extension().and_then(|e| e.to_str()), Some("png"));
        assert!(!target.exists(), "the temp file is not left behind");
        assert_eq!(
            fs::read_dir(&dir).unwrap().count(),
            0,
            "and neither is anything else"
        );
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_cancelled_snip_is_a_clean_nothing() {
        let dir = scratch("cancel");
        // Escape: the picker runs, writes no file, and exits.
        assert_eq!(snip_with(&dir, |_| Ok(())).unwrap(), None);
        // And a picker that made an empty file is the same nothing, not a
        // zero byte image handed to the decoder.
        let empty = snip_with(&dir, |target| {
            fs::write(target, []).unwrap();
            Ok(())
        })
        .unwrap();
        assert_eq!(empty, None);
        assert_eq!(fs::read_dir(&dir).unwrap().count(), 0);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_picker_that_cannot_run_is_an_error_and_leaves_nothing_behind() {
        let dir = scratch("error");
        let failed = snip_with(&dir, |target| {
            fs::write(target, FAKE_PNG).unwrap();
            Err("could not start the region picker: nope".to_string())
        });
        assert!(
            failed.is_err(),
            "a picker that failed is reported, not swallowed"
        );
        assert_eq!(
            fs::read_dir(&dir).unwrap().count(),
            0,
            "a half written snip is cleaned up too"
        );
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn two_snips_never_share_a_path() {
        let dir = std::env::temp_dir();
        let first = snip_target(&dir, 42, 1);
        let second = snip_target(&dir, 42, 2);
        assert_ne!(first, second);
        assert_ne!(snip_target(&dir, 43, 1), first);
        assert!(first.starts_with(&dir));
    }

    #[test]
    fn the_feature_is_offered_only_where_a_picker_exists() {
        assert_eq!(snip_supported(), cfg!(target_os = "macos"));
        #[cfg(target_os = "macos")]
        {
            // The test checks the command shape only. It never probes or runs
            // the real picker, because verification must not take over a desktop.
            assert_eq!(PICKER_BIN, "/usr/bin/screencapture");
            assert!(Path::new(PICKER_BIN).is_absolute());
        }
        #[cfg(not(target_os = "macos"))]
        assert!(snip_region().is_err(), "and elsewhere the command says so");
    }
}
