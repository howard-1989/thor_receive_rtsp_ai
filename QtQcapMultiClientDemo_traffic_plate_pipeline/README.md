# Layer 1 model_0 / model_1 / model_2 pipeline

This demo displays up to nine RTSP channels and runs three independent QDEEP
Layer 1 workers: `model_0`, `model_1`, and `model_2`. They currently load the
same Taiwan-traffic model, but have separate handles, queues, workers, result
buffers, drawing state, and timing statistics. `model_1` and `model_2` can be
changed later to chained models without changing buffer ownership.

## Buffer ownership

The QCAP decoded callback buffer is owned by QCAP. The callback only locks it,
copies it into one application-owned NV12 `SharedFrame`, and unlocks it. It
never calls `qcap2_rcbuffer_release` on that callback buffer.

After the copy completes, the display queue and each model queue hold their own
`std::shared_ptr<SharedFrame>` reference. The per-channel display/input queues
are depth-two and model queues are depth-one, all drop-oldest; replacing a
queued frame cannot invalidate a frame currently in
OpenCV or a synchronous QDEEP call. The QDEEP API receives the `SharedFrame`
pixels directly and the worker keeps its reference until the API returns.

All three default to `/home/nvidia/Documents/new_model/taiwantraffic_batch8/`.
Override each Layer 1 model without rebuilding with `QDEEP_LAYER1_MODEL_0`,
`QDEEP_LAYER1_MODEL_1`, and `QDEEP_LAYER1_MODEL_2`.
