# Realtime Contract and Evidence

Read this when adding or changing a native realtime session, frame delivery,
controls, post-processing, or realtime performance claims. The
[Studio guide](../../../docs/fumadocs/content/docs/guides/studio.mdx) owns the
user-facing controls; [Studio internals](../../../docs/fumadocs/content/docs/maintainers/architecture/studio-internals.mdx)
owns the serving boundaries.

## Cadence belongs to the model

`prepare_realtime()` returns `{"realtime_spec": spec.to_payload()}` built from
`RealtimeSpec` in `worldfoundry/core/execution/realtime/contracts.py`:

| Field | Meaning |
| --- | --- |
| `fps` | Playback rate of generated frames |
| `first_chunk_frames`, `steady_chunk_frames` | Frames emitted by the first and later native chunks |
| `controls` | Control vocabulary the session accepts; never empty |
| `input_fps`, `first_input_frames`, `steady_input_frames` | Control sampling clock when it differs from output |
| `transport`, `stateful` | Frame transport kind and whether the session keeps rollout state |
| `sensors` | Extra RGB, depth, or lidar channels with unique keys and indices |

`configure_realtime(...)` seeds the session (for example an image) and
allocates caches without generating. Each `stream_realtime(...)` call advances
exactly one native chunk; `realtime_next_output_frames()` reports the next
chunk's size; `reset_realtime()` drops rollout caches and keeps weights. The
Matrix-Game 2 pipeline is the reference implementation.

## Delivery and controls

- Frame queues use `latest-interactive` or `ordered-quality`
  (`worldfoundry/studio/serving/realtime/media.py`). `latest-interactive`
  backpressures briefly, then evicts stale frames to protect control latency.
  `ordered-quality` never drops and is required for prompt-scheduled or queued
  segments, whose `chunk_done` frame count is a playback contract.
- Keyboard and joystick edges are resampled into each model chunk without a
  new request. A short tap stays pending until it spans one input-frame
  sampling interval; retention guarantees a control sample, not visible motion
  in the same chunk.
- Event, step, and output-resolution requests are acknowledged with
  `event_catalog_ack`, `event_ack`, `step_ack`, and `output_config_ack`
  carrying the request's `request_id`; `chunk_done` signals generated output.
- Output resizing changes only the transmitted stream (even dimensions, at
  least 160x90, at most the native size and the 1920x1080 budget), never
  inference resolution or model state.
- Post-processors run after decoded RGB extraction with per-session state and
  flush buffered tails in chain order. The identity processor uses the same
  boundary; the RTX adapter requires the separately licensed
  `rtx_postprocess` extra.

## Performance gate

Record a representative interaction with raw timing, then check it against
explicit limits:

```bash
export WORLDFOUNDRY_REALTIME_PERF_JSONL=tmp/realtime/timing.jsonl
export WORLDFOUNDRY_REALTIME_PERF_WARMUP_CHUNKS=1
```

```json
{
  "schema_version": "worldfoundry-realtime-regression-v1",
  "cases": [{
    "name": "<model>-webrtc",
    "selectors": {"model_id": "<model-id>", "transport": "webrtc"},
    "thresholds": {
      "min_chunks": 10,
      "min_throughput_fps": 16,
      "max_dropped_frames": 0,
      "max_queue_depth": 16,
      "max_stage_p90_ms": {"generation_ms": 650}
    }
  }]
}
```

Load the manifest with `RealtimeRegressionManifest.read_json()`, the trace with
`read_realtime_trace()`, and evaluate both with `evaluate_realtime_manifest()`
from `worldfoundry/runtime/realtime_regression.py`. Set limits from an
accepted, matched-stack baseline; warmup chunks stay in the trace but are
excluded by their marker.

## Checkpoint-backed browser replay

`tests/manual/realtime_replay.py` launches the public Studio CLI with local
weights and drives Chromium through clean, lossy, and interrupted links on a
loopback UDP relay:

```bash
cp tests/manual/realtime_replay.example.json /private/realtime-profile.json
PYTHONPATH=. python tests/manual/realtime_replay.py \
  --profile /private/realtime-profile.json --output tmp/realtime-replay/<run>
```

Edit the model reference, image, and GPU index in the private copy; the output
directory must be new. Reports keep per-repeat end-to-end p50/p90,
presentation gaps, WebRTC statistics, process CPU/RSS/GPU samples, and the
source state. This is transport and lifecycle evidence, not the numerical
replay required for inference changes. Shared-GPU measurements must state
their co-tenancy, and negotiated H.264 alone does not prove NVENC execution.
Weight-free contracts for these scripts run in `make test-infer-tensors` and
`make test-realtime-client`.
