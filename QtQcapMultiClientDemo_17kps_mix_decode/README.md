# 17KPS mixed client decoding

Channels 1–24 use QCAP_DECODER_TYPE_ZZNVCODEC. Channels 25–64 use
QCAP_DECODER_TYPE_SOFTWARE. Selection is based on the zero-based channel ID.

Pipeline: RTSP broadcast client → decoded SYSBUF callback → default scaler →
640×384 NV12 SYSBUF → latest-frame AI queue → contiguous NV12 → QDeep 17KPS →
NV12-to-BGR conversion and skeleton overlay → Qt display.

Both decoder types share the same scaler. Display uses the downscaled inference
frame; it does not display the original-resolution stream. Scaling uses fixed
640×384 dimensions, retaining the original 17kps aspect-ratio behavior.
The callback input belongs to QCAP and is not released by the application.
No external qcap2 decoder, NVBUF scaler, or NVBUF-to-SYSBUF copy stage is used.

QDeep API latency is measured with steady_clock in milliseconds. Every five
seconds the application logs calls/second, active-channel batch size, and
avg/min/max latency. Every channel keeps its most recent downscaled frame. Once
the initial cache is ready, a new frame from any channel triggers the next
QDeep call with the latest cached frame from every active channel.

Build with Qt 5 qmake and make from a separate build directory. The project uses
the sibling include/, lib/, and qdeep/ directories, plus the existing 17KPS model
path from the original project. The UI supports up to 64 channels; achievable
throughput depends on decoder, CPU, and model capacity.
