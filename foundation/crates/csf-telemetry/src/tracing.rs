//! Distributed tracing implementation

use super::*;
use opentelemetry::{global, KeyValue};
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::{propagation::TraceContextPropagator, trace as sdktrace, Resource};
use std::sync::atomic::{AtomicUsize, Ordering};
use tracing_subscriber::{layer::SubscriberExt, Registry};

/// Tracing configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TracingConfig {
    /// Service name
    pub service_name: String,

    /// OTLP endpoint
    pub otlp_endpoint: String,

    /// Sampling rate (0.0 - 1.0)
    pub sampling_rate: f64,

    /// Max attributes per span
    pub max_attributes_per_span: u32,

    /// Max events per span
    pub max_events_per_span: u32,

    /// Enable console exporter
    pub enable_console: bool,

    /// Enable jaeger exporter
    pub enable_jaeger: bool,
}

impl Default for TracingConfig {
    fn default() -> Self {
        Self {
            service_name: "ares-csf".to_string(),
            otlp_endpoint: "http://localhost:4317".to_string(),
            sampling_rate: 1.0,
            max_attributes_per_span: 128,
            max_events_per_span: 128,
            enable_console: false,
            enable_jaeger: false,
        }
    }
}

/// Tracer for distributed tracing
pub struct Tracer {
    #[allow(dead_code)]
    config: TracingConfig,
    tracer: opentelemetry::global::BoxedTracer,
    active_spans: AtomicUsize,
}

impl Tracer {
    /// Create new tracer
    pub async fn new(config: &TracingConfig) -> Result<Self> {
        // Set global propagator
        global::set_text_map_propagator(TraceContextPropagator::new());

        // Create OTLP exporter
        let otlp_exporter = opentelemetry_otlp::new_exporter()
            .tonic()
            .with_endpoint(&config.otlp_endpoint);

        // Create trace config
        let trace_config = sdktrace::config()
            .with_sampler(sdktrace::Sampler::TraceIdRatioBased(config.sampling_rate))
            .with_max_attributes_per_span(config.max_attributes_per_span)
            .with_max_events_per_span(config.max_events_per_span)
            .with_resource(Resource::new(vec![
                KeyValue::new("service.name", config.service_name.clone()),
                KeyValue::new("service.version", env!("CARGO_PKG_VERSION")),
            ]));

        // Install tracer pipeline (returns a concrete Tracer and sets global provider)
        let sdk_tracer = opentelemetry_otlp::new_pipeline()
            .tracing()
            .with_exporter(otlp_exporter)
            .with_trace_config(trace_config)
            .install_batch(opentelemetry_sdk::runtime::Tokio)
            .map_err(|e| TelemetryError::Tracing(e.to_string()))?;

        // Obtain a boxed tracer from the global provider so we can use a stable object-safe API
        let tracer = opentelemetry::global::tracer("csf-tracer");

        // Set up tracing subscriber
        let telemetry = tracing_opentelemetry::layer().with_tracer(sdk_tracer.clone());
        let subscriber = Registry::default().with(telemetry);

        if config.enable_console {
            let console = tracing_subscriber::fmt::layer();
            let subscriber = subscriber.with(console);
            ::tracing::subscriber::set_global_default(subscriber)
                .map_err(|e| TelemetryError::Tracing(e.to_string()))?;
        } else {
            ::tracing::subscriber::set_global_default(subscriber)
                .map_err(|e| TelemetryError::Tracing(e.to_string()))?;
        }

        Ok(Self {
            config: config.clone(),
            tracer,
            active_spans: AtomicUsize::new(0),
        })
    }

    /// Start a new span
    pub fn start_span(&self, name: &str) -> Span {
        use opentelemetry::trace::Tracer as _;
        let span = self.tracer.start(name.to_string());
        self.active_spans.fetch_add(1, Ordering::Relaxed);

        Span {
            inner: span,
            start_time: super::now_nanos(),
        }
    }

    /// Start a span with context
    pub fn start_span_with_context(&self, name: &str, parent: &SpanContext) -> Span {
        use opentelemetry::trace::Tracer as _;
        let span = self
            .tracer
            .start_with_context(name.to_string(), &parent.context);
        self.active_spans.fetch_add(1, Ordering::Relaxed);

        Span {
            inner: span,
            start_time: super::now_nanos(),
        }
    }

    /// Get active span count
    pub fn active_spans(&self) -> usize {
        self.active_spans.load(Ordering::Relaxed)
    }

    /// Flush all pending spans
    pub async fn flush(&self) -> Result<()> {
        global::shutdown_tracer_provider();
        Ok(())
    }
}

impl Drop for Span {
    fn drop(&mut self) {
        // Decrement active span count
        // This would need access to the tracer instance in a real implementation
    }
}

/// Span context for propagation
#[derive(Clone)]
pub struct SpanContext {
    context: opentelemetry::Context,
}

impl SpanContext {
    /// Extract from headers
    pub fn extract<T>(extractor: &T) -> Self
    where
        T: opentelemetry::propagation::Extractor,
    {
        use opentelemetry::propagation::TextMapPropagator;

        let propagator = TraceContextPropagator::new();
        let context = propagator.extract(extractor);

        Self { context }
    }

    /// Inject into headers
    pub fn inject<T>(&self, injector: &mut T)
    where
        T: opentelemetry::propagation::Injector,
    {
        use opentelemetry::propagation::TextMapPropagator;

        let propagator = TraceContextPropagator::new();
        propagator.inject_context(&self.context, injector);
    }
}
