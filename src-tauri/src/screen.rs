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
// consuming exactly one seam and cannot tell a screen from a camera. The native
// packet carries only a tiny timing header, while the JPEG remains a raw channel
// payload. The frontend drops frames while decoding and the adaptive controller
// changes the native interval when measured work cannot keep up. Live inference
// still samples the stage every 220 ms, independently of capture rate.
//
// Everything that can be tested without a display is a plain function at the
// top of this file. The parts that need a real screen are in `mac` below and
// can only be smoke tested by hand, which the README says out loud.

use serde::Serialize;

/// Defaults for Auto. Named quality presets can pass a different edge and JPEG
/// quality, while the display mode supplies the rate.
#[cfg(target_os = "macos")]
pub const DEFAULT_MAX_EDGE: u32 = 1120;
#[cfg(target_os = "macos")]
pub const DEFAULT_JPEG_QUALITY: u8 = 55;
#[cfg(target_os = "macos")]
const FALLBACK_REFRESH_RATE: u32 = 60;
#[cfg(target_os = "macos")]
const MAX_FPS: u32 = 240;

#[derive(Serialize, Clone, Debug, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct DisplayInfo {
    /// Stable for the session. Stringified so the frontend can treat screen ids
    /// and camera deviceIds the same way.
    pub id: String,
    pub label: String,
    pub width: u32,
    pub height: u32,
    pub refresh_rate: u32,
}

#[derive(Serialize, Clone, Debug, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct FeedInfo {
    pub id: String,
    pub width: u32,
    pub height: u32,
    pub refresh_rate: u32,
    pub fps: u32,
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
#[cfg_attr(not(test), allow(dead_code))]
pub fn encode_jpeg(rgb: &[u8], width: u32, height: u32) -> Result<Vec<u8>, String> {
    encode_jpeg_with_quality(rgb, width, height, DEFAULT_JPEG_QUALITY)
}

#[cfg(target_os = "macos")]
fn encode_jpeg_with_quality(
    rgb: &[u8],
    width: u32,
    height: u32,
    quality: u8,
) -> Result<Vec<u8>, String> {
    if rgb.len() != width as usize * height as usize * 3 {
        return Err("the pixel buffer does not match its dimensions".into());
    }
    let mut out = Vec::with_capacity(rgb.len() / 8);
    let encoder = jpeg_encoder::Encoder::new(&mut out, quality.clamp(1, 100));
    encoder
        .encode(
            rgb,
            width as u16,
            height as u16,
            jpeg_encoder::ColorType::Rgb,
        )
        .map_err(|err| format!("could not encode the frame: {err}"))?;
    Ok(out)
}

#[cfg(target_os = "macos")]
fn frame_packet(jpeg: Vec<u8>, native_micros: u32) -> Vec<u8> {
    let sent_micros = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_micros().min(u64::MAX as u128) as u64)
        .unwrap_or(0);
    let mut packet = Vec::with_capacity(16 + jpeg.len());
    packet.extend_from_slice(b"BLFR");
    packet.extend_from_slice(&sent_micros.to_le_bytes());
    packet.extend_from_slice(&native_micros.to_le_bytes());
    packet.extend_from_slice(&jpeg);
    packet
}

#[cfg(target_os = "macos")]
mod mac {
    // ScreenCaptureKit through the objc2 bindings, which are plain Rust: no
    // Swift toolchain, no Metal, nothing that needs an SDK newer than the one
    // that builds the rest of the app. The convenience crate that wraps this
    // framework pulls a Swift bridge that fails to compile on the macOS 14 SDK
    // CI builds on, which is why the messy part is written out here.
    use super::{
        bgra_to_rgb, encode_jpeg_with_quality, fit_within, frame_packet, DisplayInfo, FeedInfo,
        DEFAULT_JPEG_QUALITY, DEFAULT_MAX_EDGE, FALLBACK_REFRESH_RATE, MAX_FPS,
    };
    use block2::RcBlock;
    use dispatch2::DispatchQueue;
    use objc2::rc::Retained;
    use objc2::runtime::AnyClass;
    use objc2::runtime::{NSObject, ProtocolObject};
    use objc2::{define_class, msg_send, AllocAnyThread, DefinedClass};
    use objc2_core_graphics::{
        CGDisplayCopyDisplayMode, CGDisplayMode, CGPreflightScreenCaptureAccess,
        CGRequestScreenCaptureAccess,
    };
    use objc2_core_media::{CMSampleBuffer, CMTime};
    use objc2_core_video::{
        CVPixelBuffer, CVPixelBufferGetBaseAddress, CVPixelBufferGetBytesPerRow,
        CVPixelBufferGetHeight, CVPixelBufferGetWidth, CVPixelBufferLockBaseAddress,
        CVPixelBufferLockFlags, CVPixelBufferUnlockBaseAddress,
    };
    use objc2_foundation::NSObjectProtocol;
    use objc2_screen_capture_kit::{
        SCContentFilter, SCDisplay, SCShareableContent, SCStream, SCStreamConfiguration,
        SCStreamOutput, SCStreamOutputType, SCWindow,
    };
    use std::sync::mpsc;
    use std::sync::Mutex;
    use std::time::{Duration, Instant};
    use tauri::ipc::{Channel, InvokeResponseBody};

    /// BGRA, the four character code CoreVideo wants for 32 bit little endian.
    const PIXEL_FORMAT_BGRA: u32 = u32::from_be_bytes(*b"BGRA");
    /// ScreenCaptureKit answers its content query on another queue.
    const CONTENT_TIMEOUT: Duration = Duration::from_secs(5);
    pub(super) const BELO_BUNDLE_ID: &str = "app.belo";

    /// The running capture. Holding it here is what makes stopping immediate:
    /// stopping and dropping the stream releases ScreenCaptureKit, which is
    /// what turns the purple recording indicator off.
    static RUNNING: Mutex<Option<Capture>> = Mutex::new(None);

    struct Capture {
        stream: Retained<SCStream>,
        output: Retained<FrameOutput>,
        display_id: u32,
    }

    // Apple calls the output handler on its own queue, and the two Retained
    // pointers are only ever touched under the mutex above.
    unsafe impl Send for Capture {}

    struct FrameSink {
        channel: Channel<InvokeResponseBody>,
        jpeg_quality: u8,
    }

    define_class!(
        #[unsafe(super(NSObject))]
        #[name = "BeloScreenFrameOutput"]
        #[ivars = FrameSink]
        struct FrameOutput;

        unsafe impl NSObjectProtocol for FrameOutput {}

        unsafe impl SCStreamOutput for FrameOutput {
            #[unsafe(method(stream:didOutputSampleBuffer:ofType:))]
            fn did_output(
                &self,
                _stream: &SCStream,
                sample: &CMSampleBuffer,
                kind: SCStreamOutputType,
            ) {
                if kind != SCStreamOutputType::Screen {
                    return;
                }
                let started = Instant::now();
                let Some(jpeg) = encode_sample(sample, self.ivars().jpeg_quality) else {
                    return;
                };
                let native_micros = started.elapsed().as_micros().min(u32::MAX as u128) as u32;
                // A closed channel means the view is gone. Releasing the capture
                // is the stop command's job, so there is nothing to do here.
                let _ = self
                    .ivars()
                    .channel
                    .send(InvokeResponseBody::Raw(frame_packet(jpeg, native_micros)));
            }
        }
    );

    impl FrameOutput {
        fn new(channel: Channel<InvokeResponseBody>, jpeg_quality: u8) -> Retained<Self> {
            let this = Self::alloc().set_ivars(FrameSink {
                channel,
                jpeg_quality,
            });
            unsafe { msg_send![super(this), init] }
        }
    }

    /// One frame, from CoreVideo's padded BGRA to the JPEG the webview decodes.
    fn encode_sample(sample: &CMSampleBuffer, jpeg_quality: u8) -> Option<Vec<u8>> {
        let image = unsafe { sample.image_buffer() }?;
        let buffer: &CVPixelBuffer = &image;
        unsafe { CVPixelBufferLockBaseAddress(buffer, CVPixelBufferLockFlags::ReadOnly) };
        let width = CVPixelBufferGetWidth(buffer);
        let height = CVPixelBufferGetHeight(buffer);
        let stride = CVPixelBufferGetBytesPerRow(buffer);
        let base = CVPixelBufferGetBaseAddress(buffer);
        let jpeg = if base.is_null() || width == 0 || height == 0 {
            None
        } else {
            let pixels = unsafe { std::slice::from_raw_parts(base as *const u8, stride * height) };
            bgra_to_rgb(pixels, width as u32, height as u32, stride).and_then(|rgb| {
                encode_jpeg_with_quality(&rgb, width as u32, height as u32, jpeg_quality).ok()
            })
        };
        unsafe { CVPixelBufferUnlockBaseAddress(buffer, CVPixelBufferLockFlags::ReadOnly) };
        jpeg
    }

    /// Belo runs on macOS 10.15 and up, ScreenCaptureKit arrived in 12.3, and a
    /// message to a class that is not there is a crash rather than an error. So
    /// the class is looked up before anything else is offered or attempted.
    pub fn available() -> bool {
        AnyClass::get(c"SCStream").is_some() && AnyClass::get(c"SCShareableContent").is_some()
    }

    pub fn permission() -> &'static str {
        if !available() {
            return "unsupported";
        }
        // preflight never prompts, so a view can render the right panel without
        // provoking a system dialog nobody asked for.
        if CGPreflightScreenCaptureAccess() {
            "granted"
        } else {
            "denied"
        }
    }

    /// Asks macOS for the permission. Returns whether it is granted afterwards.
    /// The system only ever shows its prompt once per app; after a denial it
    /// silently says no and the user has to go to System Settings.
    pub fn request_permission() -> bool {
        CGRequestScreenCaptureAccess()
    }

    fn display_refresh_rate(display_id: u32) -> u32 {
        let mode = CGDisplayCopyDisplayMode(display_id);
        let rate = mode
            .as_deref()
            .map(|mode| CGDisplayMode::refresh_rate(Some(mode)))
            .unwrap_or(0.0);
        if rate.is_finite() && rate > 0.0 {
            (rate.round() as u32).clamp(1, MAX_FPS)
        } else {
            FALLBACK_REFRESH_RATE
        }
    }

    /// The app owns both the main and HUD windows. ScreenCaptureKit accepts
    /// window objects in the display filter, so every Belo window is excluded
    /// by bundle id before the first frame is delivered.
    pub(super) fn is_belo_bundle(bundle_id: &str) -> bool {
        bundle_id == BELO_BUNDLE_ID
    }

    fn excluded_belo_windows(
        content: &SCShareableContent,
    ) -> Retained<objc2_foundation::NSArray<SCWindow>> {
        let windows = unsafe { content.windows() };
        let own: Vec<Retained<SCWindow>> = windows
            .iter()
            .filter(|window| {
                unsafe { window.owningApplication() }.is_some_and(|app| {
                    is_belo_bundle(&unsafe { app.bundleIdentifier() }.to_string())
                })
            })
            .collect();
        objc2_foundation::NSArray::from_retained_slice(&own)
    }

    fn display_filter(
        content: &SCShareableContent,
        display: &SCDisplay,
    ) -> Retained<SCContentFilter> {
        let excluded = excluded_belo_windows(content);
        unsafe {
            SCContentFilter::initWithDisplay_excludingWindows(
                SCContentFilter::alloc(),
                display,
                &excluded,
            )
        }
    }

    /// The displays ScreenCaptureKit is willing to share, waited for on this
    /// thread because every caller here is a command that has to answer.
    fn shareable_content() -> Result<Retained<SCShareableContent>, String> {
        let (sender, receiver) = mpsc::channel::<Result<Retained<SCShareableContent>, String>>();
        let handler = RcBlock::new(
            move |content: *mut SCShareableContent, error: *mut objc2_foundation::NSError| {
                let answer = if content.is_null() {
                    Err(if error.is_null() {
                        "ScreenCaptureKit returned no content".to_string()
                    } else {
                        unsafe { (*error).localizedDescription() }.to_string()
                    })
                } else {
                    Ok(unsafe { Retained::retain(content) }.expect("content is not null"))
                };
                let _ = sender.send(answer);
            },
        );
        unsafe { SCShareableContent::getShareableContentWithCompletionHandler(&handler) };
        receiver
            .recv_timeout(CONTENT_TIMEOUT)
            .map_err(|_| "ScreenCaptureKit did not answer in time".to_string())?
    }

    pub fn displays() -> Result<Vec<DisplayInfo>, String> {
        if !available() {
            return Err("This macOS is older than the screen capture it would need".into());
        }
        if !CGPreflightScreenCaptureAccess() {
            return Err("denied".into());
        }
        let content = shareable_content()?;
        Ok(unsafe { content.displays() }
            .iter()
            .enumerate()
            .map(|(index, display)| {
                let id = unsafe { display.displayID() };
                DisplayInfo {
                    id: id.to_string(),
                    label: format!("Display {}", index + 1),
                    width: unsafe { display.width() } as u32,
                    height: unsafe { display.height() } as u32,
                    refresh_rate: display_refresh_rate(id),
                }
            })
            .collect())
    }

    pub fn start(
        display_id: Option<String>,
        fps: u32,
        max_edge: u32,
        jpeg_quality: u8,
        on_frame: Channel<InvokeResponseBody>,
    ) -> Result<FeedInfo, String> {
        if !available() {
            return Err("This macOS is older than the screen capture it would need".into());
        }
        if !CGPreflightScreenCaptureAccess() {
            return Err("denied".into());
        }
        stop()?;

        let content = shareable_content()?;
        let all = unsafe { content.displays() };
        let wanted = display_id.as_deref().and_then(|id| id.parse::<u32>().ok());
        // A remembered display that is no longer attached falls back to the
        // first one rather than failing to start, the way a missing camera does.
        let display = wanted
            .and_then(|id| all.iter().find(|d| unsafe { d.displayID() } == id))
            .or_else(|| all.iter().next())
            .ok_or_else(|| "This Mac reports no display to capture".to_string())?;

        let display_id = unsafe { display.displayID() };
        let refresh_rate = display_refresh_rate(display_id);
        let effective_fps = if fps == 0 {
            refresh_rate
        } else {
            fps.min(refresh_rate)
        }
        .clamp(1, MAX_FPS);
        let edge = if max_edge == 0 {
            DEFAULT_MAX_EDGE
        } else {
            max_edge.clamp(224, 4096)
        };
        let (width, height) = fit_within(
            unsafe { display.width() } as u32,
            unsafe { display.height() } as u32,
            edge,
        );

        let filter = display_filter(&content, &display);
        let jpeg_quality = if jpeg_quality == 0 {
            DEFAULT_JPEG_QUALITY
        } else {
            jpeg_quality.clamp(1, 100)
        };
        let config = unsafe { SCStreamConfiguration::new() };
        unsafe {
            config.setWidth(width as usize);
            config.setHeight(height as usize);
            config.setPixelFormat(PIXEL_FORMAT_BGRA);
            config.setShowsCursor(true);
            config.setQueueDepth(3);
            config.setMinimumFrameInterval(CMTime::with_seconds(1.0 / effective_fps as f64, 600));
        }

        let stream = unsafe {
            SCStream::initWithFilter_configuration_delegate(
                SCStream::alloc(),
                &filter,
                &config,
                None,
            )
        };
        let output = FrameOutput::new(on_frame, jpeg_quality);
        let queue = DispatchQueue::new("com.belo.screen", None);
        unsafe {
            stream.addStreamOutput_type_sampleHandlerQueue_error(
                ProtocolObject::from_ref(&*output),
                SCStreamOutputType::Screen,
                Some(&queue),
            )
        }
        .map_err(|err| {
            format!(
                "The screen feed could not start: {}",
                err.localizedDescription()
            )
        })?;

        // startCapture answers asynchronously, and a failure here is the whole
        // difference between a black stage and an honest message.
        let (sender, receiver) = mpsc::channel::<Option<String>>();
        let handler = RcBlock::new(move |error: *mut objc2_foundation::NSError| {
            let _ = sender.send(if error.is_null() {
                None
            } else {
                Some(unsafe { (*error).localizedDescription() }.to_string())
            });
        });
        unsafe { stream.startCaptureWithCompletionHandler(Some(&handler)) };
        match receiver.recv_timeout(CONTENT_TIMEOUT) {
            Ok(None) => {}
            Ok(Some(message)) => return Err(format!("The screen feed could not start: {message}")),
            Err(_) => return Err("The screen feed did not start in time".into()),
        }

        let info = FeedInfo {
            id: display_id.to_string(),
            width,
            height,
            refresh_rate,
            fps: effective_fps,
        };
        *RUNNING.lock().map_err(|_| "capture state is poisoned")? = Some(Capture {
            stream,
            output,
            display_id,
        });
        Ok(info)
    }

    pub fn set_rate(fps: u32) -> Result<u32, String> {
        let rate = fps.clamp(1, MAX_FPS);
        let running = RUNNING.lock().map_err(|_| "capture state is poisoned")?;
        let Some(capture) = running.as_ref() else {
            return Ok(rate);
        };
        let config = unsafe { SCStreamConfiguration::new() };
        unsafe {
            config.setMinimumFrameInterval(CMTime::with_seconds(1.0 / rate as f64, 600));
        }
        let (sender, receiver) = mpsc::channel::<Option<String>>();
        let handler = RcBlock::new(move |error: *mut objc2_foundation::NSError| {
            let _ = sender.send(if error.is_null() {
                None
            } else {
                Some(unsafe { (*error).localizedDescription() }.to_string())
            });
        });
        unsafe {
            capture
                .stream
                .updateConfiguration_completionHandler(&config, Some(&handler));
        }
        match receiver.recv_timeout(CONTENT_TIMEOUT) {
            Ok(None) => Ok(rate),
            Ok(Some(message)) => Err(format!("The screen rate could not change: {message}")),
            Err(_) => Err("The screen rate change did not finish in time".into()),
        }
    }

    pub fn refresh_filter() -> Result<(), String> {
        let running = RUNNING.lock().map_err(|_| "capture state is poisoned")?;
        let Some(capture) = running.as_ref() else {
            return Ok(());
        };
        let content = shareable_content()?;
        let display = unsafe { content.displays() }
            .iter()
            .find(|display| unsafe { display.displayID() } == capture.display_id)
            .ok_or_else(|| "The captured display is no longer available".to_string())?;
        let filter = display_filter(&content, &display);
        let (sender, receiver) = mpsc::channel::<Option<String>>();
        let handler = RcBlock::new(move |error: *mut objc2_foundation::NSError| {
            let _ = sender.send(if error.is_null() {
                None
            } else {
                Some(unsafe { (*error).localizedDescription() }.to_string())
            });
        });
        unsafe {
            capture
                .stream
                .updateContentFilter_completionHandler(&filter, Some(&handler));
        }
        match receiver.recv_timeout(CONTENT_TIMEOUT) {
            Ok(None) => Ok(()),
            Ok(Some(message)) => Err(format!("The screen filter could not refresh: {message}")),
            Err(_) => Err("The screen filter refresh did not finish in time".into()),
        }
    }

    pub fn stop() -> Result<(), String> {
        let taken = RUNNING
            .lock()
            .map_err(|_| "capture state is poisoned")?
            .take();
        let Some(capture) = taken else {
            return Ok(());
        };
        let (sender, receiver) = mpsc::channel::<()>();
        let handler = RcBlock::new(move |_error: *mut objc2_foundation::NSError| {
            let _ = sender.send(());
        });
        unsafe {
            let _ = capture.stream.removeStreamOutput_type_error(
                ProtocolObject::from_ref(&*capture.output),
                SCStreamOutputType::Screen,
            );
            capture
                .stream
                .stopCaptureWithCompletionHandler(Some(&handler));
        }
        // The stream is dropped either way: waiting is only so the caller knows
        // the indicator is out by the time the command returns.
        let _ = receiver.recv_timeout(CONTENT_TIMEOUT);
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

/// Whether this build on this machine can capture a screen at all. The webview
/// path on Windows and Linux answers for itself in the frontend; this is the
/// native one.
#[tauri::command]
pub fn screen_supported() -> bool {
    #[cfg(target_os = "macos")]
    {
        mac::available()
    }
    #[cfg(not(target_os = "macos"))]
    {
        false
    }
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
    max_edge: u32,
    jpeg_quality: u8,
    on_frame: tauri::ipc::Channel<tauri::ipc::InvokeResponseBody>,
) -> Result<FeedInfo, String> {
    #[cfg(target_os = "macos")]
    {
        mac::start(display_id, fps, max_edge, jpeg_quality, on_frame)
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (display_id, fps, max_edge, jpeg_quality, on_frame);
        Err("unsupported".to_string())
    }
}

#[tauri::command]
pub fn screen_set_rate(fps: u32) -> Result<u32, String> {
    #[cfg(target_os = "macos")]
    {
        mac::set_rate(fps)
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = fps;
        Err("unsupported".to_string())
    }
}

#[tauri::command]
pub fn screen_refresh_filter() -> Result<(), String> {
    #[cfg(target_os = "macos")]
    {
        mac::refresh_filter()
    }
    #[cfg(not(target_os = "macos"))]
    {
        Ok(())
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
        assert!(
            bgra_to_rgb(&[0u8; 64], 4, 4, 8).is_none(),
            "stride under the row width"
        );
        assert!(
            bgra_to_rgb(&[0u8; 16], 4, 4, 16).is_none(),
            "buffer shorter than the frame"
        );
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
        assert!(
            jpeg.len() < rgb.len(),
            "an encode that grows the frame is not an encode"
        );
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

    #[test]
    fn frame_path_measurement_reports_the_native_work() {
        fn measure(width: u32, height: u32, quality: u8) -> (f64, usize) {
            let stride = width as usize * 4;
            let mut bgra = vec![0u8; stride * height as usize];
            for (index, byte) in bgra.iter_mut().enumerate() {
                *byte = (index.wrapping_mul(37) & 0xff) as u8;
            }
            let mut total = 0u128;
            let mut bytes = 0usize;
            for _ in 0..3 {
                let started = std::time::Instant::now();
                let rgb = bgra_to_rgb(&bgra, width, height, stride).expect("synthetic pixels");
                let jpeg = encode_jpeg_with_quality(&rgb, width, height, quality)
                    .expect("synthetic frame");
                total += started.elapsed().as_micros();
                bytes = jpeg.len();
            }
            (total as f64 / 3_000.0, bytes)
        }

        let before = measure(1280, 830, 72);
        let auto = measure(DEFAULT_MAX_EDGE, 700, DEFAULT_JPEG_QUALITY);
        let smooth = measure(960, 600, 55);
        let balanced = measure(1440, 900, 68);
        println!(
            "synthetic native frame path: before 1280x830 q72 = {:.2} ms, {} KB; Auto 1120x700 q55 = {:.2} ms, {} KB; Smooth 960x600 q55 = {:.2} ms, {} KB; Balanced 1440x900 q68 = {:.2} ms, {} KB",
            before.0,
            before.1 / 1024,
            auto.0,
            auto.1 / 1024,
            smooth.0,
            smooth.1 / 1024,
            balanced.0,
            balanced.1 / 1024
        );
        assert!(before.0 > 0.0 && auto.0 > 0.0);
        assert!(before.1 > 0 && auto.1 > 0);
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
                    assert!(display.refresh_rate > 0);
                    assert_eq!(
                        fit_within(display.width, display.height, DEFAULT_MAX_EDGE).0 % 2,
                        0
                    );
                }
            }
        }
    }

    #[test]
    fn the_capture_filter_excludes_belo_windows_by_bundle_id() {
        assert!(super::mac::is_belo_bundle("app.belo"));
        assert!(!super::mac::is_belo_bundle("com.apple.finder"));
        assert_eq!(super::mac::BELO_BUNDLE_ID, "app.belo");
    }
}
