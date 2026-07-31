// Screen capture for the Use view's second live source.
//
// Why this exists at all: the webview Belo runs in on macOS is WKWebView, and
// WKWebView refuses `getDisplayMedia`. The API is present and callable, but wry
// only answers WebKit's camera and microphone permission callback
// (`requestMediaCapturePermissionForOrigin`), never the display capture one, so
// WebKit has no picker to show and denies the request. Windows and Linux use
// their webview's own `getDisplayMedia`; the frontend picks. On macOS the
// frames come from here instead, through ScreenCaptureKit.
//
// The frame path, honestly described: ScreenCaptureKit hands us BGRA frames at
// the size we ask for, we convert them to RGB, JPEG encode them and push the
// bytes down a Tauri channel. The frontend decodes each JPEG and paints it into
// a canvas whose `captureStream()` is a normal MediaStream, so every view keeps
// consuming exactly one seam and cannot tell a screen from a camera. The cost
// of that path is one JPEG encode plus one decode per frame, and one IPC
// message per frame; frames above 1 KB travel through Tauri's fetch based
// channel route rather than an eval, which is why the capture is asked for at
// ten frames a second and at most 1280 pixels on the long edge. Live inference
// samples the stage every 220 ms, so a faster feed would only be thrown away.
//
// Everything that can be tested without a display is a plain function at the
// top of this file. The parts that need a real screen are in `mac` below and
// can only be smoke tested by hand, which the README says out loud.

use serde::Serialize;

/// The frontend never asks for more than this on the long edge.
#[cfg(target_os = "macos")]
pub const MAX_EDGE: u32 = 1280;
#[cfg(target_os = "macos")]
const JPEG_QUALITY: u8 = 72;

#[derive(Serialize, Clone, Debug, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct DisplayInfo {
    /// Stable for the session. Stringified so the frontend can treat screen ids
    /// and camera deviceIds the same way.
    pub id: String,
    pub label: String,
    pub width: u32,
    pub height: u32,
}

#[derive(Serialize, Clone, Debug, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct FeedInfo {
    pub id: String,
    pub width: u32,
    pub height: u32,
}

/// Fits a display into `max` on its long edge, keeping the aspect ratio and
/// never scaling up. Both sides stay even, because odd sized planes make some
/// encoders unhappy and cost nothing to avoid.
#[cfg(target_os = "macos")]
pub fn fit_within(width: u32, height: u32, max: u32) -> (u32, u32) {
    let even = |value: u32| value.max(2) & !1;
    if width == 0 || height == 0 {
        return (2, 2);
    }
    if width <= max && height <= max {
        return (even(width), even(height));
    }
    let scale = if width >= height {
        max as f64 / width as f64
    } else {
        max as f64 / height as f64
    };
    (
        even((width as f64 * scale).round() as u32),
        even((height as f64 * scale).round() as u32),
    )
}

/// BGRA with a row stride to tightly packed RGB.
///
/// The stride is the interesting part: CoreVideo pads every row out to its own
/// alignment, so a 1442 pixel wide frame usually arrives with rows longer than
/// 1442 * 4 bytes and reading it as a flat buffer shears the picture.
#[cfg(target_os = "macos")]
pub fn bgra_to_rgb(src: &[u8], width: u32, height: u32, bytes_per_row: usize) -> Option<Vec<u8>> {
    let (width, height) = (width as usize, height as usize);
    if width == 0 || height == 0 || bytes_per_row < width * 4 {
        return None;
    }
    if src.len() < (height - 1) * bytes_per_row + width * 4 {
        return None;
    }
    let mut out = vec![0u8; width * height * 3];
    for y in 0..height {
        let row = &src[y * bytes_per_row..y * bytes_per_row + width * 4];
        let target = &mut out[y * width * 3..(y + 1) * width * 3];
        for x in 0..width {
            target[x * 3] = row[x * 4 + 2];
            target[x * 3 + 1] = row[x * 4 + 1];
            target[x * 3 + 2] = row[x * 4];
        }
    }
    Some(out)
}

/// Encodes tightly packed RGB as a baseline JPEG.
#[cfg(target_os = "macos")]
pub fn encode_jpeg(rgb: &[u8], width: u32, height: u32) -> Result<Vec<u8>, String> {
    if rgb.len() != width as usize * height as usize * 3 {
        return Err("the pixel buffer does not match its dimensions".into());
    }
    let mut out = Vec::with_capacity(rgb.len() / 8);
    let encoder = jpeg_encoder::Encoder::new(&mut out, JPEG_QUALITY);
    encoder
        .encode(rgb, width as u16, height as u16, jpeg_encoder::ColorType::Rgb)
        .map_err(|err| format!("could not encode the frame: {err}"))?;
    Ok(out)
}

#[cfg(target_os = "macos")]
mod mac {
    use super::{bgra_to_rgb, encode_jpeg, fit_within, DisplayInfo, FeedInfo, MAX_EDGE};
    use core_graphics::access::ScreenCaptureAccess;
    use screencapturekit::cm::SCFrameStatus;
    use screencapturekit::cv::CVPixelBufferLockFlags;
    use screencapturekit::prelude::*;
    use std::sync::Mutex;
    use tauri::ipc::{Channel, InvokeResponseBody};

    /// One capture at a time. Holding the stream here is what makes stopping
    /// immediate: dropping or stopping it releases ScreenCaptureKit, which is
    /// what turns the purple recording indicator off.
    static RUNNING: Mutex<Option<SCStream>> = Mutex::new(None);

    pub fn permission() -> &'static str {
        // preflight never prompts, so a view can render the right panel without
        // provoking a system dialog nobody asked for.
        if ScreenCaptureAccess.preflight() {
            "granted"
        } else {
            "denied"
        }
    }

    /// Asks macOS for the permission. Returns whether it is granted afterwards.
    /// The system only ever shows its prompt once per app; after a denial it
    /// silently says no and the user has to go to System Settings.
    pub fn request_permission() -> bool {
        ScreenCaptureAccess.request()
    }

    pub fn displays() -> Result<Vec<DisplayInfo>, String> {
        if !ScreenCaptureAccess.preflight() {
            return Err("denied".into());
        }
        let content = SCShareableContent::get().map_err(|err| format!("{err}"))?;
        Ok(content
            .displays()
            .iter()
            .enumerate()
            .map(|(index, display)| DisplayInfo {
                id: display.display_id().to_string(),
                label: format!("Display {}", index + 1),
                width: display.width(),
                height: display.height(),
            })
            .collect())
    }

    pub fn start(
        display_id: Option<String>,
        fps: u32,
        on_frame: Channel<InvokeResponseBody>,
    ) -> Result<FeedInfo, String> {
        if !ScreenCaptureAccess.preflight() {
            return Err("denied".into());
        }
        stop()?;

        let content = SCShareableContent::get().map_err(|err| format!("{err}"))?;
        let all = content.displays();
        let wanted = display_id.as_deref().and_then(|id| id.parse::<u32>().ok());
        // A remembered display that is no longer attached falls back to the
        // first one rather than failing to start, the way a missing camera does.
        let display = wanted
            .and_then(|id| all.iter().find(|d| d.display_id() == id))
            .or_else(|| all.first())
            .ok_or_else(|| "This Mac reports no display to capture".to_string())?;

        let (width, height) = fit_within(display.width(), display.height(), MAX_EDGE);
        let interval = CMTime::new(1, fps.clamp(1, 30) as i32);
        let filter = SCContentFilter::create()
            .with_display(display)
            .with_excluding_windows(&[])
            .build();
        let config = SCStreamConfiguration::new()
            .with_width(width)
            .with_height(height)
            .with_pixel_format(PixelFormat::BGRA)
            .with_shows_cursor(true)
            .with_queue_depth(3)
            .with_minimum_frame_interval(&interval);

        let mut stream = SCStream::new(&filter, &config);
        stream.add_output_handler(
            move |sample: CMSampleBuffer, _: SCStreamOutputType| {
                // Idle frames carry no new pixels and blank ones carry none at
                // all. Forwarding either would just cost an encode.
                if !matches!(
                    sample.frame_status(),
                    Some(SCFrameStatus::Complete) | Some(SCFrameStatus::Started)
                ) {
                    return;
                }
                let Some(buffer) = sample.image_buffer() else {
                    return;
                };
                let Ok(guard) = buffer.lock(CVPixelBufferLockFlags::READ_ONLY) else {
                    return;
                };
                let width = guard.width() as u32;
                let height = guard.height() as u32;
                let Some(rgb) = bgra_to_rgb(guard.as_slice(), width, height, guard.bytes_per_row())
                else {
                    return;
                };
                let Ok(jpeg) = encode_jpeg(&rgb, width, height) else {
                    return;
                };
                // A closed channel means the view is gone; the stop command is
                // what actually releases the capture, so nothing to do here.
                let _ = on_frame.send(InvokeResponseBody::Raw(jpeg));
            },
            SCStreamOutputType::Screen,
        );
        stream
            .start_capture()
            .map_err(|err| format!("The screen feed could not start: {err}"))?;

        let info = FeedInfo {
            id: display.display_id().to_string(),
            width,
            height,
        };
        *RUNNING.lock().map_err(|_| "capture state is poisoned")? = Some(stream);
        Ok(info)
    }

    pub fn stop() -> Result<(), String> {
        let taken = RUNNING
            .lock()
            .map_err(|_| "capture state is poisoned")?
            .take();
        if let Some(stream) = taken {
            stream
                .stop_capture()
                .map_err(|err| format!("The screen feed could not stop: {err}"))?;
        }
        Ok(())
    }

    /// Opens the Screen Recording page of System Settings, because telling
    /// somebody to find it themselves is not guidance.
    pub fn open_settings() -> Result<(), String> {
        std::process::Command::new("open")
            .arg("x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture")
            .status()
            .map_err(|err| format!("could not open System Settings: {err}"))?;
        Ok(())
    }
}

#[tauri::command]
pub fn screen_supported() -> bool {
    cfg!(target_os = "macos")
}

#[tauri::command]
pub fn screen_permission() -> String {
    #[cfg(target_os = "macos")]
    {
        mac::permission().to_string()
    }
    #[cfg(not(target_os = "macos"))]
    {
        "unsupported".to_string()
    }
}

#[tauri::command]
pub fn screen_request_permission() -> bool {
    #[cfg(target_os = "macos")]
    {
        mac::request_permission()
    }
    #[cfg(not(target_os = "macos"))]
    {
        false
    }
}

#[tauri::command]
pub fn screen_displays() -> Result<Vec<DisplayInfo>, String> {
    #[cfg(target_os = "macos")]
    {
        mac::displays()
    }
    #[cfg(not(target_os = "macos"))]
    {
        Err("unsupported".to_string())
    }
}

#[tauri::command]
pub fn screen_start(
    display_id: Option<String>,
    fps: u32,
    on_frame: tauri::ipc::Channel<tauri::ipc::InvokeResponseBody>,
) -> Result<FeedInfo, String> {
    #[cfg(target_os = "macos")]
    {
        mac::start(display_id, fps, on_frame)
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (display_id, fps, on_frame);
        Err("unsupported".to_string())
    }
}

#[tauri::command]
pub fn screen_stop() -> Result<(), String> {
    #[cfg(target_os = "macos")]
    {
        mac::stop()
    }
    #[cfg(not(target_os = "macos"))]
    {
        Ok(())
    }
}

#[tauri::command]
pub fn screen_open_settings() -> Result<(), String> {
    #[cfg(target_os = "macos")]
    {
        mac::open_settings()
    }
    #[cfg(not(target_os = "macos"))]
    {
        Err("unsupported".to_string())
    }
}

// The pure halves of the frame path are only compiled where the native feed
// exists, so the platforms that use the webview's own getDisplayMedia carry
// none of it.
#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;

    #[test]
    fn a_display_is_fitted_without_growing_and_without_odd_sides() {
        // A 16:10 retina display, the case this feature was written for.
        assert_eq!(fit_within(3024, 1964, 1280), (1280, 830));
        // Already small enough: left alone apart from the even rounding.
        assert_eq!(fit_within(1024, 640, 1280), (1024, 640));
        assert_eq!(fit_within(1443, 903, 1280), (1280, 800));
        // Taller than wide, which is what a rotated display gives.
        assert_eq!(fit_within(1080, 1920, 1280), (720, 1280));
        // Odd input never leaks an odd side out.
        let (w, h) = fit_within(1001, 667, 1280);
        assert_eq!((w % 2, h % 2), (0, 0));
        // Nonsense in, something drawable out, never a panic.
        assert_eq!(fit_within(0, 0, 1280), (2, 2));
    }

    #[test]
    fn padded_bgra_rows_convert_without_shearing_the_picture() {
        // Three pixels a row, but CoreVideo hands out sixteen bytes of row.
        let width = 3u32;
        let height = 2u32;
        let stride = 16usize;
        let mut src = vec![0u8; stride * height as usize];
        for y in 0..height as usize {
            for x in 0..width as usize {
                let at = y * stride + x * 4;
                src[at] = 10 + x as u8; // B
                src[at + 1] = 20 + x as u8; // G
                src[at + 2] = 30 + x as u8; // R
                src[at + 3] = 255; // A
            }
            // Padding bytes that must never reach the output.
            for pad in width as usize * 4..stride {
                src[y * stride + pad] = 0xEE;
            }
        }
        let rgb = bgra_to_rgb(&src, width, height, stride).expect("a valid buffer converts");
        assert_eq!(rgb.len(), 18);
        assert_eq!(&rgb[0..3], &[30, 20, 10]);
        assert_eq!(&rgb[3..6], &[31, 21, 11]);
        // Second row starts where the first one ended, with no padding between.
        assert_eq!(&rgb[9..12], &[30, 20, 10]);
        assert!(!rgb.contains(&0xEE));
    }

    #[test]
    fn a_buffer_that_cannot_be_trusted_is_refused_rather_than_read_past() {
        assert!(bgra_to_rgb(&[0u8; 64], 4, 4, 8).is_none(), "stride under the row width");
        assert!(bgra_to_rgb(&[0u8; 16], 4, 4, 16).is_none(), "buffer shorter than the frame");
        assert!(bgra_to_rgb(&[], 0, 0, 0).is_none(), "an empty frame");
    }

    #[test]
    fn a_frame_encodes_to_a_jpeg_the_webview_can_decode() {
        let (width, height) = (32u32, 20u32);
        let mut rgb = vec![0u8; (width * height * 3) as usize];
        for (i, chunk) in rgb.chunks_mut(3).enumerate() {
            chunk[0] = (i % 255) as u8;
            chunk[1] = 128;
            chunk[2] = 200;
        }
        let jpeg = encode_jpeg(&rgb, width, height).expect("a full buffer encodes");
        assert_eq!(&jpeg[0..2], &[0xFF, 0xD8], "starts with the JPEG marker");
        assert_eq!(&jpeg[jpeg.len() - 2..], &[0xFF, 0xD9], "and ends with one");
        assert!(jpeg.len() < rgb.len(), "an encode that grows the frame is not an encode");
        // Dimensions live in the SOF0 segment, which is what the decoder reads.
        let sof = jpeg
            .windows(2)
            .position(|pair| pair == [0xFF, 0xC0])
            .expect("a baseline JPEG has a SOF0 segment");
        let stated_height = u16::from_be_bytes([jpeg[sof + 5], jpeg[sof + 6]]);
        let stated_width = u16::from_be_bytes([jpeg[sof + 7], jpeg[sof + 8]]);
        assert_eq!((stated_width, stated_height), (32, 20));
    }

    #[test]
    fn a_mismatched_buffer_is_an_error_and_not_a_panic() {
        assert!(encode_jpeg(&[0u8; 10], 32, 20).is_err());
    }

    /// The one thing about the native path that can be checked without
    /// recording anything: the display list. It reads sizes and identifiers, no
    /// pixels, and it skips itself where the permission is not already granted,
    /// so it never provokes a system prompt on a developer machine or in CI.
    #[cfg(target_os = "macos")]
    #[test]
    fn displays_are_listed_where_the_permission_already_exists() {
        match super::screen_displays() {
            Err(reason) => assert_eq!(reason, "denied", "the only expected failure"),
            Ok(displays) => {
                assert!(!displays.is_empty(), "a Mac with permission has a display");
                for display in &displays {
                    assert!(!display.id.is_empty());
                    assert!(display.width > 0 && display.height > 0);
                    assert_eq!(fit_within(display.width, display.height, MAX_EDGE).0 % 2, 0);
                }
            }
        }
    }
}
