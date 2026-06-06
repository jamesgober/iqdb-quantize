//! Integration test for `iqdb-quantize` tracing emission.
//!
//! Exercises a representative failure path
//! (`ScalarQuantizer::train` on an empty training set → `InvalidConfig`)
//! and a representative success path. The local recording subscriber is
//! inlined in the `recorder` module below; library code NEVER installs a
//! subscriber.

#![allow(clippy::unwrap_used)]

use iqdb_quantize::{Quantizer, ScalarQuantizer};
use tracing::Level;

use crate::recorder::with_recorder;

mod recorder {
    //! Inlined recording subscriber for this test crate. Captures both
    //! span creations (`on_new_span`) and events (`on_event`) into a
    //! shared `Vec` so the test can treat "instrumentation fired" as a
    //! uniform check.

    use std::sync::{Arc, Mutex};

    use tracing::field::{Field, Visit};
    use tracing::span::Attributes;
    use tracing::{Event, Id, Level, Subscriber};
    use tracing_subscriber::layer::{Context, Layer};
    use tracing_subscriber::registry::LookupSpan;

    /// A single recorded span creation or event.
    #[derive(Debug, Clone)]
    pub struct RecordedEvent {
        pub level: Level,
        pub target: String,
        pub fields: Vec<(String, String)>,
    }

    /// A `tracing::Layer` that records every span creation and event into
    /// a shared `Vec`.
    #[derive(Debug, Clone, Default)]
    pub struct RecordingLayer {
        events: Arc<Mutex<Vec<RecordedEvent>>>,
    }

    impl RecordingLayer {
        pub fn new() -> Self {
            Self::default()
        }

        pub fn snapshot(&self) -> Vec<RecordedEvent> {
            match self.events.lock() {
                Ok(events) => events.clone(),
                Err(_) => Vec::new(),
            }
        }
    }

    struct FieldRecorder<'a>(&'a mut Vec<(String, String)>);

    impl Visit for FieldRecorder<'_> {
        fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
            self.0
                .push((field.name().to_string(), format!("{value:?}")));
        }
        fn record_str(&mut self, field: &Field, value: &str) {
            self.0.push((field.name().to_string(), value.to_string()));
        }
        fn record_i64(&mut self, field: &Field, value: i64) {
            self.0.push((field.name().to_string(), value.to_string()));
        }
        fn record_u64(&mut self, field: &Field, value: u64) {
            self.0.push((field.name().to_string(), value.to_string()));
        }
        fn record_bool(&mut self, field: &Field, value: bool) {
            self.0.push((field.name().to_string(), value.to_string()));
        }
        fn record_f64(&mut self, field: &Field, value: f64) {
            self.0.push((field.name().to_string(), value.to_string()));
        }
    }

    impl<S> Layer<S> for RecordingLayer
    where
        S: Subscriber + for<'a> LookupSpan<'a>,
    {
        fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
            let mut fields = Vec::new();
            event.record(&mut FieldRecorder(&mut fields));
            let recorded = RecordedEvent {
                level: *event.metadata().level(),
                target: event.metadata().target().to_string(),
                fields,
            };
            if let Ok(mut events) = self.events.lock() {
                events.push(recorded);
            }
        }

        fn on_new_span(&self, attrs: &Attributes<'_>, _id: &Id, _ctx: Context<'_, S>) {
            let mut fields = Vec::new();
            attrs.record(&mut FieldRecorder(&mut fields));
            let recorded = RecordedEvent {
                level: *attrs.metadata().level(),
                target: attrs.metadata().target().to_string(),
                fields,
            };
            if let Ok(mut events) = self.events.lock() {
                events.push(recorded);
            }
        }
    }

    /// Install a [`RecordingLayer`] for the duration of `f` and return both
    /// the captured recordings and `f`'s return value. The subscriber is
    /// installed via `tracing::subscriber::with_default`, so it is local
    /// to the current thread for the duration of the call and is removed
    /// automatically on return — exactly what an integration test wants.
    pub fn with_recorder<F, R>(f: F) -> (Vec<RecordedEvent>, R)
    where
        F: FnOnce() -> R,
    {
        let layer = RecordingLayer::new();
        use tracing_subscriber::layer::SubscriberExt;
        let subscriber = tracing_subscriber::registry().with(layer.clone());
        let value = tracing::subscriber::with_default(subscriber, f);
        (layer.snapshot(), value)
    }
}

#[test]
fn train_empty_set_emits_error_event() {
    let (events, result) = with_recorder(|| {
        let mut sq = ScalarQuantizer::new();
        let empty: [&[f32]; 0] = [];
        sq.train(&empty)
    });

    assert!(result.is_err(), "empty training set must error");

    let error_events: Vec<_> = events.iter().filter(|e| e.level == Level::ERROR).collect();
    assert!(
        !error_events.is_empty(),
        "expected at least one error event, got: {events:?}",
    );

    let has_kind_field = error_events
        .iter()
        .flat_map(|e| e.fields.iter())
        .any(|(name, _)| name == "error.kind");
    assert!(
        has_kind_field,
        "error event must carry structured `error.kind` field, got: {error_events:?}",
    );

    let from_quantize_crate = error_events
        .iter()
        .any(|e| e.target.starts_with("iqdb_quantize"));
    assert!(
        from_quantize_crate,
        "error event must originate from iqdb_quantize, got targets: {:?}",
        error_events.iter().map(|e| &e.target).collect::<Vec<_>>(),
    );
}

#[test]
fn train_success_creates_info_lifecycle_span() {
    let (events, result) = with_recorder(|| {
        let mut sq = ScalarQuantizer::new();
        sq.train(&[&[0.0_f32, 1.0, 2.0][..], &[1.0_f32, 0.0, 1.0][..]])
    });
    assert!(result.is_ok(), "valid train must succeed");

    let info_spans_from_quantize: Vec<_> = events
        .iter()
        .filter(|e| e.level == Level::INFO && e.target.starts_with("iqdb_quantize"))
        .collect();
    assert!(
        !info_spans_from_quantize.is_empty(),
        "expected info-level lifecycle span from iqdb_quantize, got: {events:?}",
    );

    let has_quantizer_field = info_spans_from_quantize
        .iter()
        .flat_map(|e| e.fields.iter())
        .any(|(name, value)| name == "quantizer" && value == "sq8");
    assert!(
        has_quantizer_field,
        "lifecycle span must carry `quantizer = \"sq8\"`, got: {info_spans_from_quantize:?}",
    );
}
