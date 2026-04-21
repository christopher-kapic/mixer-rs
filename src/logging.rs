//! Tracing subscriber setup for the CLI.
//!
//! Behavior:
//!   - `--log-format text` (default) uses a human-readable formatter to stderr
//!     with span CLOSE events so each instrumented request flushes one final
//!     line carrying the accumulated fields.
//!   - `--log-format json` uses the JSON formatter — same span CLOSE behavior.
//!   - `RUST_LOG` wins whenever it is set; otherwise the filter defaults to
//!     `mixer=info` so library crates stay quiet.

use tracing_subscriber::EnvFilter;
use tracing_subscriber::fmt::{self, format::FmtSpan};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

use crate::cli::LogFormat;

/// Install the global tracing subscriber.
pub fn init(format: LogFormat) {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("mixer=info"));

    let registry = tracing_subscriber::registry().with(filter);

    match format {
        LogFormat::Text => {
            registry
                .with(
                    fmt::layer()
                        .with_writer(std::io::stderr)
                        .with_target(false)
                        .with_span_events(FmtSpan::CLOSE),
                )
                .init();
        }
        LogFormat::Json => {
            registry
                .with(
                    fmt::layer()
                        .json()
                        .flatten_event(true)
                        .with_current_span(true)
                        .with_writer(std::io::stderr)
                        .with_span_events(FmtSpan::CLOSE),
                )
                .init();
        }
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    //! In-memory subscriber helper for unit tests. Each call installs a fresh
    //! subscriber via `with_default` (scoped to a closure) so tests can run in
    //! parallel without stomping on each other's global state.

    use std::io::{self, Write};
    use std::sync::{Arc, Mutex};

    use tracing_subscriber::fmt::{self, MakeWriter, format::FmtSpan};

    #[derive(Clone, Default)]
    pub struct CapturedWriter {
        pub buf: Arc<Mutex<Vec<u8>>>,
    }

    impl CapturedWriter {
        pub fn new() -> Self {
            Self::default()
        }
        pub fn contents(&self) -> String {
            String::from_utf8(self.buf.lock().unwrap().clone()).unwrap()
        }
    }

    impl Write for CapturedWriter {
        fn write(&mut self, data: &[u8]) -> io::Result<usize> {
            self.buf.lock().unwrap().extend_from_slice(data);
            Ok(data.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    impl<'a> MakeWriter<'a> for CapturedWriter {
        type Writer = CapturedWriter;
        fn make_writer(&'a self) -> Self::Writer {
            self.clone()
        }
    }

    pub fn json_subscriber(writer: CapturedWriter) -> impl tracing::Subscriber + Send + Sync {
        fmt::Subscriber::builder()
            .json()
            .flatten_event(true)
            .with_current_span(true)
            .with_span_list(false)
            .with_writer(writer)
            .with_span_events(FmtSpan::CLOSE)
            .finish()
    }

    pub fn text_subscriber(writer: CapturedWriter) -> impl tracing::Subscriber + Send + Sync {
        fmt::Subscriber::builder()
            .with_writer(writer)
            .with_span_events(FmtSpan::CLOSE)
            .with_ansi(false)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::*;

    #[test]
    fn json_format_emits_parseable_lines() {
        let writer = CapturedWriter::new();
        let sub = json_subscriber(writer.clone());
        tracing::subscriber::with_default(sub, || {
            tracing::info!(provider = "codex", mixer_model = "claude", "test event");
        });
        let text = writer.contents();
        assert!(!text.is_empty(), "expected at least one JSON log line");
        for line in text.lines().filter(|l| !l.is_empty()) {
            let parsed: serde_json::Value =
                serde_json::from_str(line).unwrap_or_else(|e| panic!("invalid JSON `{line}`: {e}"));
            assert!(parsed.is_object(), "top-level JSON must be an object");
            assert!(parsed.get("level").is_some(), "level field present");
            assert_eq!(
                parsed.get("message").and_then(|v| v.as_str()),
                Some("test event")
            );
            assert_eq!(
                parsed.get("provider").and_then(|v| v.as_str()),
                Some("codex")
            );
            assert_eq!(
                parsed.get("mixer_model").and_then(|v| v.as_str()),
                Some("claude")
            );
        }
    }

    #[test]
    fn text_format_does_not_panic_on_startup() {
        // Smoke test: text subscriber accepts a basic info event without errors.
        let writer = CapturedWriter::new();
        let sub = text_subscriber(writer.clone());
        tracing::subscriber::with_default(sub, || {
            tracing::info!("text format smoke test");
        });
        assert!(writer.contents().contains("text format smoke test"));
    }
}
