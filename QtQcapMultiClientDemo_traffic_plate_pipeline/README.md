# Layer 1 + Layer 2 face pipeline

The application keeps the three existing Layer 1 workers: `model_0`,
`model_1`, and `model_2`. They retain independent QDEEP batch handles, queues,
result buffers, timing stats, and overlays.

After `model_0` successfully completes a frame, it pushes the same immutable
NV12 `SharedFrame` to Layer 2. Layer 2 creates one
`QDEEP_CREATE_OBJECT_DETECT` face-landmark handle and one `std::thread` for
each RTSP channel. Each of those workers calls
`QDEEP_SET_VIDEO_OBJECT_DETECT_UNCOMPRESSION_BUFFER` and draws the returned
face box plus five landmarks on that channel's video label.

`layer1/model_1` has an independent Layer 2 child named `layer2/model_1`.
It creates one non-batch QDEEP license-plate-recognition handle/thread per
RTSP channel, receives the same `SharedFrame` only after `layer1/model_1`
finishes, and draws the plate bounding box plus the recognized text extracted
from `fFeatureVectors` with OpenCV.

The Layer 2 queue is one frame deep and drop-oldest. Therefore a slow face
model cannot block decoding, model_0, model_1, model_2, or another channel.
All frame data is application-owned `std::shared_ptr<SharedFrame>` memory;
there are no shared QDEEP handles or output buffers between face workers.
On stop, threads are notified and joined before their QDEEP handles are
stopped/destroyed.

Default models:

- Layer 1: `/home/nvidia/Documents/thor_receive_rtsp_ai/new_model/taiwantraffic_batch8/QDEEP.OD.TAIWAN.TRAFFIC.C4.TINY.CFG`
- Layer 2 face: `/home/nvidia/Documents/thor_receive_rtsp_ai/new_model/face/QDEEP.OD.FACE.LANDMARK.5KPS.CFG`
- Layer 2 plate: `/home/nvidia/Documents/thor_receive_rtsp_ai/new_model/plate/QDEEP.OD.LICENSE.PLATE.RECOGNITION.LAW.TINY.TK.CFG`

Override with `QDEEP_LAYER1_MODEL_0`, `QDEEP_LAYER1_MODEL_1`,
`QDEEP_LAYER1_MODEL_2`, and `QDEEP_LAYER2_FACE_MODEL`.
Set `QDEEP_LAYER2_PLATE_MODEL` to override the plate model.
