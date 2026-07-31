fn main() {
    // ScreenCaptureKit reaches Apple's framework through a Swift shim, and a
    // plain cargo built binary has no runpath to the Swift runtime, so it dies
    // in dyld before main. Every macOS since 10.14.4 ships that runtime at this
    // path. Only the desktop binary needs it: iOS builds a static library and
    // compiles no screen capture at all.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rustc-link-arg=-Wl,-rpath,/usr/lib/swift");
    }
    tauri_build::build()
}
