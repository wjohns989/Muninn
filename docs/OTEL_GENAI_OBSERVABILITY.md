# OTel GenAI Observability Runbook

This runbook documents how to enable OpenTelemetry tracing for Muninn's retrieval and memory operations, with privacy-safe defaults.

## Scope

- Feature flag integration: `MUNINN_OTEL_GENAI=1`
- Privacy toggle: `MUNINN_OTEL_CAPTURE_CONTENT=0` by default
- Optional content bound: `MUNINN_OTEL_CAPTURE_CONTENT_MAX_CHARS` (default `1000`)
- GenAI semantic convention alignment for events/attributes where feasible

## Why This Exists

Muninn already emits explainable recall traces for retrieval quality. OTel spans add operational visibility:
- where latency is spent (retrieval fusion, rerank, add/search paths),
- how many results are returned,
- whether regressions are caused by infrastructure or ranking behavior.

## Prerequisites

Install OTel runtime dependencies in your environment:

```bash
python -m pip install opentelemetry-sdk opentelemetry-exporter-otlp
```

## Minimal Local Collector Setup

Use the checked-in collector config:

```bash
docker run --rm -p 4317:4317 -p 4318:4318 \
  -v ${PWD}/examples/otel/collector-config.yaml:/etc/otelcol/config.yaml \
  otel/opentelemetry-collector:latest \
  --config /etc/otelcol/config.yaml
```

## Recommended Environment Configuration

```bash
# Enable Muninn OTel feature flag
export MUNINN_OTEL_GENAI=1

# Privacy-safe defaults
export MUNINN_OTEL_CAPTURE_CONTENT=0
export MUNINN_OTEL_CAPTURE_CONTENT_MAX_CHARS=1000

# OTel exporter config
export OTEL_SERVICE_NAME=muninn
export OTEL_TRACES_EXPORTER=otlp
export OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318

# Optional semconv stability mode (recommended where supported by your SDK)
export OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental
```

## Privacy Policy (Operational)

Default policy:
1. Raw content capture is disabled (`MUNINN_OTEL_CAPTURE_CONTENT=0`).
2. Traces include operational metadata (counts/latency), not conversation payloads.
3. If content capture is enabled for diagnostics, enforce short retention and restricted access in your backend.

When enabling content capture:
1. Keep `MUNINN_OTEL_CAPTURE_CONTENT_MAX_CHARS` low and task-specific.
2. Use environment-specific enablement (staging only unless incident response requires production).
3. Rotate and delete trace data under your normal retention controls.

## Smoke Test

1. Start collector (above).
2. Run Muninn with OTel flags enabled.
3. Trigger `add` and `search` requests.
4. Confirm spans/events are visible in collector output.

Expected span/event patterns:
- span: `muninn.retrieval.search`
- event: `muninn.retrieval.result`
- attributes such as:
  - `gen_ai.operation.name`
  - `muninn.limit`
  - `muninn.result_count`
  - `muninn.elapsed_ms`

## Troubleshooting

No spans observed:
1. Verify `MUNINN_OTEL_GENAI=1`.
2. Verify OTel SDK/exporter packages are installed.
3. Verify OTLP endpoint/protocol env vars match collector listener.
4. Check collector logs for dropped spans or protocol mismatches.

## References

- OpenTelemetry traces concepts: https://opentelemetry.io/docs/concepts/signals/traces/
- GenAI semantic events: https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-events/
- OTel Collector config reference: https://opentelemetry.io/docs/collector/configuration/
- OTLP exporter env vars (Python): https://opentelemetry-python.readthedocs.io/en/latest/exporter/otlp/otlp.html
- Handling sensitive data guidance: https://opentelemetry.io/docs/security/handling-sensitive-data/
