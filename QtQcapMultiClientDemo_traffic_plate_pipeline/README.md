# Traffic + plate pipeline

This project displays up to nine RTSP channels in a fixed 3 x 3 grid. A bounded per-channel `qcap2_rcbuffer_queue` holds at most three scaled NV12 frames. When full, the oldest frame is released; the AI thread drains the queue and uses only its latest frame. The same compact batch is then submitted sequentially to the traffic and plate QDEEP handles.

Traffic uses `../model/traffic/QDEEP.OD.TAIWAN.TRAFFIC.C4.TINY.CFG`.

Until a genuine QDEEP licence-plate model is available, the second handle defaults to `../model/people/QDEEP.OD.TINY.PERSON.V10N.CFG`, matching the current plate demo. Set `QDEEP_PLATE_MODEL` to replace it with a plate `.CFG` file; the matching weights must be in the location required by that config, for example:

```bash
export QDEEP_PLATE_MODEL=/path/to/QDEEP.OD.LICENSE.PLATE.RECOGNITION.CFG
./QtQcapMultiClientDemo_traffic_plate_pipeline
```

The repository currently has no plate model asset, so the people model is an intentional temporary default.
