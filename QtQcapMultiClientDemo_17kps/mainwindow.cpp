#include "mainwindow.h"
#include <QHBoxLayout>
#include <QElapsedTimer>
#include <QVBoxLayout>
#include <QGroupBox>
#include <QHeaderView>
#include <QApplication>
#include <QScreen>
#include <QDebug>
#include <QCloseEvent>
#include <QResizeEvent>
#include <QMoveEvent>
#include <QHideEvent>
#include <QShowEvent>
#include <QPainter>
#include <QDateTime>
#include <cmath>

MainWindow *g_pMainwindow = nullptr;

extern "C" {
    QDEEP_EXT_API QRESULT QDEEP_EXPORT QDEEP_GET_OBJECT_DETECT_RESERVED_STATUS(PVOID pDetector, ULONG* pCheckNum);
}

// ── Overlay Widget (Skeleton 17 Keypoints) ──────────────────────────────────
class OverlayWidget : public QWidget {
public:
    OverlayWidget(QWidget *parent = nullptr) : QWidget(parent) {
        setAttribute(Qt::WA_TransparentForMouseEvents);
        setAttribute(Qt::WA_TranslucentBackground);
        setAttribute(Qt::WA_ShowWithoutActivating);
        setWindowFlags(Qt::Window |
                       Qt::FramelessWindowHint |
                       Qt::Tool |
                       Qt::WindowStaysOnTopHint |
                       Qt::BypassWindowManagerHint);
    }

protected:
    void paintEvent(QPaintEvent *event) override {
        if (!g_pMainwindow) return;
        if (!g_pMainwindow->m_bShowOverlay) return;

        QPainter painter(this);
        painter.setRenderHint(QPainter::Antialiasing);

        // COCO 17 Keypoints connections
        const std::vector<std::pair<int, int>> connections = {
            {0, 14}, {0, 13},       // Nose to Eyes
            {14, 16}, {13, 15},     // Eyes to Ears
            {4, 1},                 // Shoulder line
            {4, 5}, {5, 6},         // Left arm
            {1, 2}, {2, 3},         // Right arm
            {10, 7},                // Hip line
            {4, 10}, {1, 7},        // Torso sides
            {10, 11}, {11, 12},     // Left leg
            {7, 8}, {8, 9}          // Right leg
        };

        std::lock_guard<std::mutex> lock(g_pMainwindow->draw_mtx);

        for (int i = 0; i < MainWindow::MAX_CHANNELS; ++i) {
            // Find the channel context for this index
            ChannelContext* ctx = nullptr;
            for (auto* c : g_pMainwindow->channels) {
                if (c->channelId == i) {
                    ctx = c;
                    break;
                }
            }
            if (!ctx || !ctx->m_bSendBuffer || ctx->m_nVideoWidth == 0) continue;

            // Find the corresponding video frame widget
            QWidget* w = nullptr;
            for (int fi = 0; fi < g_pMainwindow->videoFrames.size(); ++fi) {
                if (fi == i) {
                    w = g_pMainwindow->videoFrames[fi];
                    break;
                }
            }
            if (!w || !w->isVisible()) continue;

            QPoint globalPos = w->mapToGlobal(QPoint(0, 0));
            QPoint localPos = this->mapFromGlobal(globalPos);

            double scale_x = (double)w->width() / (double)ctx->m_nAIWidth;
            double scale_y = (double)w->height() / (double)ctx->m_nAIHeight;

            // Draw camera channel and people count overlay at top-left
            int people_count = (int)g_pMainwindow->draw_persons[i].size();
            QString headerText = QString("CH %1 | People: %2").arg(i + 1).arg(people_count);

            painter.setPen(Qt::NoPen);
            painter.setBrush(QColor(0, 0, 0, 180));

            QFont font = painter.font();
            font.setPointSize(10);
            font.setBold(true);
            painter.setFont(font);

            QRect headerRect = painter.fontMetrics().boundingRect(headerText);
            headerRect.adjust(-6, -2, 6, 2);
            headerRect.moveTo(localPos.x() + 10, localPos.y() + 10);
            painter.drawRoundedRect(headerRect, 4, 4);

            painter.setPen(QColor(0, 255, 200));
            painter.drawText(headerRect, Qt::AlignCenter, headerText);

            // Draw skeleton for each person
            for (const auto& person : g_pMainwindow->draw_persons[i]) {
                const float kpt_thresh = 0.3f;
                QPoint mapped_kpts[17];
                bool kpt_valid[17] = {false};

                for (int k = 0; k < 17; ++k) {
                    if (person.keypoints[k].probability >= kpt_thresh) {
                        int px = localPos.x() + (int)(person.keypoints[k].x * scale_x);
                        int py = localPos.y() + (int)(person.keypoints[k].y * scale_y);
                        mapped_kpts[k] = QPoint(px, py);
                        kpt_valid[k] = true;
                    }
                }

                // Draw skeleton lines
                QPen linePen;
                linePen.setWidth(3);

                for (const auto& conn : connections) {
                    int p1 = conn.first;
                    int p2 = conn.second;
                    if (kpt_valid[p1] && kpt_valid[p2]) {
                        QColor connColor;
                        if (p1 == 0 || p2 == 0 || p1 >= 13 || p2 >= 13) {
                            connColor = QColor(255, 235, 59); // Yellow for face
                        } else if (p1 == 4 || p1 == 5 || p1 == 6 || p1 == 10 || p1 == 11 || p1 == 12 ||
                                   p2 == 4 || p2 == 5 || p2 == 6 || p2 == 10 || p2 == 11 || p2 == 12) {
                            connColor = QColor(0, 229, 255); // Cyan for left side
                        } else {
                            connColor = QColor(255, 20, 147); // Neon pink for right side
                        }

                        linePen.setColor(connColor);
                        painter.setPen(linePen);
                        painter.drawLine(mapped_kpts[p1], mapped_kpts[p2]);
                    }
                }

                // Draw joint points
                for (int k = 0; k < 17; ++k) {
                    if (kpt_valid[k]) {
                        QColor jointColor;
                        if (k == 0 || k >= 13) {
                            jointColor = QColor(255, 255, 0); // Yellow (Face)
                        } else if (k == 4 || k == 5 || k == 6 || k == 10 || k == 11 || k == 12) {
                            jointColor = QColor(0, 255, 255); // Cyan (Left side)
                        } else {
                            jointColor = QColor(255, 0, 128); // Pink/Red (Right side)
                        }

                        painter.setPen(QPen(Qt::white, 1));
                        painter.setBrush(jointColor);
                        painter.drawEllipse(mapped_kpts[k], 4, 4);
                    }
                }
            }
        }
    }
};

// ── Static callback functions delegating to ChannelContext ──────────────────
static QRETURN on_connected_callback(
    PVOID  pClient,
    UINT   iSessionNum,
    ULONG  nVideoEncoderFormat,
    ULONG  nVideoWidth,
    ULONG  nVideoHeight,
    BOOL   bVideoIsInterleaved,
    double dVideoFrameRate,
    ULONG  nAudioEncoderFormat,
    ULONG  nAudioChannels,
    ULONG  nAudioBitsPerSample,
    ULONG  nAudioSampleFrequency,
    PVOID  pUserData)
{
    ChannelContext* ctx = static_cast<ChannelContext*>(pUserData);
    return ctx->onConnected(pClient, iSessionNum, nVideoEncoderFormat, nVideoWidth, nVideoHeight, bVideoIsInterleaved, dVideoFrameRate);
}

static QRETURN on_video_callback(
    PVOID pClient,
    UINT iSessionNum,
    double dSampleTime,
    BYTE * pStreamBuffer,
    ULONG nStreamBufferLen,
    BOOL bIsKeyFrame,
    PVOID pUserData)
{
    ChannelContext* ctx = static_cast<ChannelContext*>(pUserData);
    return ctx->onVideoCallback(dSampleTime, pStreamBuffer, nStreamBufferLen, bIsKeyFrame);
}

static QRETURN on_event_vdec_callback(PVOID pUserData) {
    ChannelContext* ctx = static_cast<ChannelContext*>(pUserData);
    return ctx->onEventVdec();
}

// ── ChannelContext Implementation ───────────────────────────────────────────
ChannelContext::ChannelContext(int id, const QString& streamUrl, uintptr_t winId)
    : channelId(id), url(streamUrl), m_winId(winId),
      pClient(nullptr), pVdec(nullptr), pEventHandlers(nullptr),
      pEvent_vdec(nullptr), pVideoSink(nullptr),
      pScaler2(nullptr), pScaler3(nullptr),
      m_pCurrentAIRCBuffer(nullptr),
      m_nVideoWidth(0), m_nVideoHeight(0), m_dVideoFrameRate(0.0), m_nVideoEncoderFormat(0),
      m_frameCount(0), m_bDisplayEnabled(true),
      m_pushFrameCount(0), m_decFrameCount(0),
      // AI init
      m_bSendBuffer(false), m_lastProcessTime(0.0), m_bFrameReady(false),
      m_pAIBuffer(nullptr), m_nAIBufferLen(0), m_nAIWidth(0), m_nAIHeight(0)
{
}

ChannelContext::~ChannelContext() {
    if (m_pAIBuffer) {
        delete[] m_pAIBuffer;
        m_pAIBuffer = nullptr;
    }
    if (m_pCurrentAIRCBuffer) {
        qcap2_rcbuffer_release(m_pCurrentAIRCBuffer);
        m_pCurrentAIRCBuffer = nullptr;
    }
    stop();
}

bool ChannelContext::start() {
    QMutexLocker locker(&m_mutex);

    qDebug() << "Starting channel" << channelId << "URL:" << url;

    // Create broadcast client
    QRESULT qres = QCAP_CREATE_BROADCAST_CLIENT(channelId, url.toLatin1().data(), &pClient, QCAP_DECODER_TYPE_ZZNVCODEC, nullptr);
    if (qres != QCAP_RS_SUCCESSFUL) {
        qCritical() << "QCAP_CREATE_BROADCAST_CLIENT failed for CH" << channelId << "qres =" << qres;
        return false;
    }

    // Disable client-side video decoding
    BOOL bVideoDecode = FALSE;
    QCAP_SET_BROADCAST_CLIENT_CUSTOM_PROPERTY_EX(pClient, QCAP_BCPROP_VIDEO_DECODE, reinterpret_cast<BYTE*>(&bVideoDecode), sizeof(bVideoDecode));

    // Disable client-side audio decoding
    BOOL bAudioDecode = FALSE;
    QCAP_SET_BROADCAST_CLIENT_CUSTOM_PROPERTY_EX(pClient, QCAP_BCPROP_AUDIO_DECODE, reinterpret_cast<BYTE*>(&bAudioDecode), sizeof(bAudioDecode));

    // Register connected and raw stream callbacks
    QCAP_REGISTER_BROADCAST_CLIENT_CONNECTED_CALLBACK(pClient, on_connected_callback, this);
    QCAP_REGISTER_VIDEO_BROADCAST_CLIENT_CALLBACK(pClient, on_video_callback, this);

    // Start stream receiver (TCP mode for RTSP)
    qres = QCAP_START_BROADCAST_CLIENT(pClient, QCAP_BROADCAST_PROTOCOL_TCP, 10000, -1);
    if (qres != QCAP_RS_SUCCESSFUL) {
        qCritical() << "QCAP_START_BROADCAST_CLIENT failed for CH" << channelId << "qres =" << qres;
        QCAP_DESTROY_BROADCAST_CLIENT(pClient);
        pClient = nullptr;
        return false;
    }

    return true;
}

void ChannelContext::stop() {
    PVOID pLocalClient = nullptr;
    qcap2_event_handlers_t* pLocalEventHandlers = nullptr;
    qcap2_video_scaler_t* pLocalScaler2 = nullptr;
    qcap2_video_scaler_t* pLocalScaler3 = nullptr;
    qcap2_video_decoder_t* pLocalVdec = nullptr;
    qcap2_video_sink_t* pLocalVideoSink = nullptr;

    {
        QMutexLocker locker(&m_mutex);
        pLocalClient = pClient;
        pLocalEventHandlers = pEventHandlers;
        pLocalScaler2 = pScaler2;
        pLocalScaler3 = pScaler3;
        pLocalVdec = pVdec;
        pLocalVideoSink = pVideoSink;

        pClient = nullptr;
        pEventHandlers = nullptr;
        pScaler2 = nullptr;
        pScaler3 = nullptr;
        pVdec = nullptr;
        pVideoSink = nullptr;
    }

    qDebug() << "========== CH" << channelId << "Stop Sequence Started ==========";

    if (pLocalClient) {
        QCAP_STOP_BROADCAST_CLIENT(pLocalClient);
    }
    if (pLocalVdec) {
        qcap2_video_decoder_stop(pLocalVdec);
    }
    uintptr_t nHandle_vdec = 0;
    if (pEvent_vdec) {
        qcap2_event_get_native_handle(pEvent_vdec, &nHandle_vdec);
        qcap2_event_stop(pEvent_vdec);
    }
    if (pLocalEventHandlers) {
        if (nHandle_vdec != 0) {
            qcap2_event_handlers_remove_handler(pLocalEventHandlers, nHandle_vdec);
        }
        qcap2_event_handlers_stop(pLocalEventHandlers);
    }
    if (pLocalVideoSink) {
        qcap2_video_sink_stop(pLocalVideoSink);
    }
    if (pLocalScaler2) {
        qcap2_video_scaler_stop(pLocalScaler2);
    }
    if (pLocalScaler3) {
        qcap2_video_scaler_stop(pLocalScaler3);
    }

    if (pLocalVideoSink) {
        qcap2_video_sink_delete(pLocalVideoSink);
    }
    if (pLocalScaler2) {
        qcap2_video_scaler_delete(pLocalScaler2);
    }
    if (pLocalScaler3) {
        qcap2_video_scaler_delete(pLocalScaler3);
    }
    if (pLocalVdec) {
        qcap2_video_decoder_delete(pLocalVdec);
    }
    if (pLocalEventHandlers) {
        qcap2_event_handlers_delete(pLocalEventHandlers);
    }
    if (pEvent_vdec) {
        qcap2_event_stop(pEvent_vdec);
        qcap2_event_delete(pEvent_vdec);
        pEvent_vdec = nullptr;
    }
    if (pLocalClient) {
        QCAP_DESTROY_BROADCAST_CLIENT(pLocalClient);
    }

    if (m_pCurrentAIRCBuffer) {
        qcap2_rcbuffer_release(m_pCurrentAIRCBuffer);
        m_pCurrentAIRCBuffer = nullptr;
    }

    qDebug() << "========== CH" << channelId << "Stop Sequence Finished ==========";
}

void ChannelContext::setDisplayEnabled(bool enabled) {
    QMutexLocker locker(&m_mutex);
    m_bDisplayEnabled = enabled;
}

QRETURN ChannelContext::onConnected(
    PVOID pClient,
    UINT iSessionNum,
    ULONG nVideoEncoderFormat,
    ULONG nVideoWidth,
    ULONG nVideoHeight,
    BOOL bVideoIsInterleaved,
    double dVideoFrameRate)
{
    Q_UNUSED(pClient);
    Q_UNUSED(iSessionNum);

    QMutexLocker locker(&m_mutex);

    if (nVideoWidth == 0 || nVideoHeight == 0 || nVideoWidth > 8192 || nVideoHeight > 8192) {
        qCritical() << "CH" << channelId << "Connected with unreasonable dimensions:" << nVideoWidth << "x" << nVideoHeight;
        m_statusInfo = QString("Aborted (unreasonable dimensions: %1x%2)").arg(nVideoWidth).arg(nVideoHeight);
        return QCAP_RT_OK;
    }

    m_nVideoWidth = nVideoWidth;
    m_nVideoHeight = nVideoHeight;
    m_dVideoFrameRate = dVideoFrameRate;
    m_nVideoEncoderFormat = nVideoEncoderFormat;

    QString formatStr;
    switch (nVideoEncoderFormat) {
        case QCAP_ENCODER_FORMAT_H264: formatStr = "H.264"; break;
        case QCAP_ENCODER_FORMAT_H265: formatStr = "H.265"; break;
        case QCAP_ENCODER_FORMAT_AV1:  formatStr = "AV1"; break;
        case QCAP_ENCODER_FORMAT_MPEG2: formatStr = "MPEG2"; break;
        case QCAP_ENCODER_FORMAT_RAW:  formatStr = "RAW"; break;
        default: formatStr = QString("Unknown (%1)").arg(nVideoEncoderFormat); break;
    }

    m_statusInfo = QString("%1x%2 @%3fps (%4)")
                   .arg(nVideoWidth).arg(nVideoHeight).arg(dVideoFrameRate).arg(formatStr);

    qDebug() << "CH" << channelId << "Connected info:" << m_statusInfo;

    // Allocate AI buffer
    m_nAIWidth = 640;
    m_nAIHeight = 384;
    m_nAIBufferLen = 640 * 384 * 3 / 2;
    if (m_pAIBuffer) delete[] m_pAIBuffer;
    m_pAIBuffer = new BYTE[m_nAIBufferLen]();

    // Initialize Event Handlers
    pEventHandlers = qcap2_event_handlers_new();
    if (!pEventHandlers) {
        qCritical() << "CH" << channelId << "Failed to create event handlers.";
        m_statusInfo = "Error: Failed to create event handlers";
        return QCAP_RT_OK;
    }
    qcap2_event_handlers_start(pEventHandlers);

    pEvent_vdec = qcap2_event_new();
    if (!pEvent_vdec) {
        qCritical() << "CH" << channelId << "Failed to create decoder event.";
        m_statusInfo = "Error: Failed to create decoder event";
        qcap2_event_handlers_stop(pEventHandlers);
        qcap2_event_handlers_delete(pEventHandlers);
        pEventHandlers = nullptr;
        return QCAP_RT_OK;
    }
    qcap2_event_start(pEvent_vdec);

    uintptr_t nHandle_vdec = 0;
    qcap2_event_get_native_handle(pEvent_vdec, &nHandle_vdec);
    qcap2_event_handlers_add_handler(pEventHandlers, nHandle_vdec, on_event_vdec_callback, this);

    // Initialize nvv4l2 Hardware Decoder
    pVdec = qcap2_video_decoder_new();
    if (!pVdec) {
        qCritical() << "CH" << channelId << "Failed to create video decoder.";
        m_statusInfo = "Error: Failed to create video decoder";
        qcap2_event_handlers_remove_handler(pEventHandlers, nHandle_vdec);
        qcap2_event_handlers_stop(pEventHandlers);
        qcap2_event_handlers_delete(pEventHandlers);
        pEventHandlers = nullptr;
        qcap2_event_stop(pEvent_vdec);
        qcap2_event_delete(pEvent_vdec);
        pEvent_vdec = nullptr;
        return QCAP_RT_OK;
    }

    qDebug() << "Trace: Creating video decoder property...";
    qcap2_video_encoder_property_t* pProp = qcap2_video_encoder_property_new();
    if (!pProp) {
        qCritical() << "CH" << channelId << "Failed to create video encoder property.";
        m_statusInfo = "Error: Failed to create video encoder property";
        qcap2_video_decoder_delete(pVdec);
        pVdec = nullptr;
        qcap2_event_handlers_remove_handler(pEventHandlers, nHandle_vdec);
        qcap2_event_handlers_stop(pEventHandlers);
        qcap2_event_handlers_delete(pEventHandlers);
        pEventHandlers = nullptr;
        qcap2_event_stop(pEvent_vdec);
        qcap2_event_delete(pEvent_vdec);
        pEvent_vdec = nullptr;
        return QCAP_RT_OK;
    }

    qDebug() << "Trace: Setting property1 on property object...";
    qcap2_video_encoder_property_set_property1(pProp,
        0,
        QCAP_ENCODER_TYPE_NVIDIA_NVENC,
        nVideoEncoderFormat,
        QCAP_COLORSPACE_TYPE_NV12,
        640, 384, dVideoFrameRate,
        QCAP_RECORD_PROFILE_MAIN, QCAP_RECORD_LEVEL_51, QCAP_RECORD_ENTROPY_CABAC, QCAP_RECORD_COMPLEXITY_0, QCAP_RECORD_MODE_CBR,
        8000, 40000000, 60, 0, FALSE, 0, 0, 0, FALSE, FALSE, FALSE, 0, 0, 0, 0, 0, 0);
    qcap2_video_encoder_property_set_high_perf(pProp, TRUE);
    qDebug() << "Trace: Setting video property on decoder...";
    qcap2_video_decoder_set_video_property(pVdec, pProp);
    qcap2_video_encoder_property_delete(pProp);

    qDebug() << "Trace: Configuring decoder events/queues...";
    qcap2_video_decoder_set_event(pVdec, pEvent_vdec);
    qcap2_video_decoder_set_multithread(pVdec, false);
    qcap2_video_decoder_set_packet_count(pVdec, 16);
    qcap2_video_decoder_set_frame_count(pVdec, 10);

    qDebug() << "Trace: Starting video decoder...";
    QRESULT qres = qcap2_video_decoder_start(pVdec);
    if (qres != QCAP_RS_SUCCESSFUL) {
        qCritical() << "qcap2_video_decoder_start failed for CH" << channelId << "qres =" << qres;
        m_statusInfo = QString("Decoder start failed (%1)").arg(qres);
        qcap2_video_decoder_delete(pVdec);
        pVdec = nullptr;
        qcap2_event_handlers_remove_handler(pEventHandlers, nHandle_vdec);
        qcap2_event_handlers_stop(pEventHandlers);
        qcap2_event_handlers_delete(pEventHandlers);
        pEventHandlers = nullptr;
        qcap2_event_stop(pEvent_vdec);
        qcap2_event_delete(pEvent_vdec);
        pEvent_vdec = nullptr;
        return QCAP_RT_OK;
    }

    qDebug() << "Trace: Creating video sink...";
    // Initialize Video Sink
    pVideoSink = qcap2_video_sink_new();
    if (!pVideoSink) {
        qCritical() << "CH" << channelId << "Failed to create video sink.";
        m_statusInfo = "Error: Failed to create video sink";
        qcap2_video_decoder_delete(pVdec);
        pVdec = nullptr;
        qcap2_event_handlers_remove_handler(pEventHandlers, nHandle_vdec);
        qcap2_event_handlers_stop(pEventHandlers);
        qcap2_event_handlers_delete(pEventHandlers);
        pEventHandlers = nullptr;
        qcap2_event_stop(pEvent_vdec);
        qcap2_event_delete(pEvent_vdec);
        pEvent_vdec = nullptr;
        return QCAP_RT_OK;
    }

    qDebug() << "Trace: Setting video sink properties...";
    int backendType = QCAP2_VIDEO_SINK_BACKEND_TYPE_DAVMF;
    if (getenv("QCAP_VO_USE_GSTREAMER") && QString(getenv("QCAP_VO_USE_GSTREAMER")) == "1") {
        backendType = QCAP2_VIDEO_SINK_BACKEND_TYPE_GSTREAMER;
    }
    qcap2_video_sink_set_backend_type(pVideoSink, backendType);
    qcap2_video_sink_set_native_handle(pVideoSink, m_winId);

    if (m_bDisplayEnabled) {
        qcap2_video_sink_set_nvbuf(pVideoSink, false);
    }

    qcap2_video_format_t* pFormat = qcap2_video_format_new();
    if (pFormat) {
        qcap2_video_format_set_property(pFormat, QCAP_COLORSPACE_TYPE_NV12, 640, 384, bVideoIsInterleaved, dVideoFrameRate);
        qcap2_video_sink_set_video_format(pVideoSink, pFormat);
        qcap2_video_format_delete(pFormat);
    }

    if (m_bDisplayEnabled) {
        qDebug() << "Trace: Starting video sink...";
        qres = qcap2_video_sink_start(pVideoSink);
        if (qres != QCAP_RS_SUCCESSFUL) {
            qCritical() << "qcap2_video_sink_start failed for CH" << channelId << "qres =" << qres;
        }
    }

    qDebug() << "Trace: Creating video scaler 3...";
    // Initialize Video Scalers (Scaler 3 only, Scaler 2 is bypassed because the HW decoder outputs 640x384 directly)
    pScaler2 = nullptr;
    pScaler3 = qcap2_video_scaler_new();
    if (!pScaler3) {
        qCritical() << "CH" << channelId << "Failed to create video scaler 3.";
        m_statusInfo = "Error: Failed to create video scaler 3";
        qcap2_video_sink_stop(pVideoSink);
        qcap2_video_sink_delete(pVideoSink);
        pVideoSink = nullptr;
        qcap2_video_decoder_stop(pVdec);
        qcap2_video_decoder_delete(pVdec);
        pVdec = nullptr;
        qcap2_event_handlers_remove_handler(pEventHandlers, nHandle_vdec);
        qcap2_event_handlers_stop(pEventHandlers);
        qcap2_event_handlers_delete(pEventHandlers);
        pEventHandlers = nullptr;
        qcap2_event_stop(pEvent_vdec);
        qcap2_event_delete(pEvent_vdec);
        pEvent_vdec = nullptr;
        return QCAP_RT_OK;
    }

    qDebug() << "Trace: Setting video scaler 3 properties...";
    // 3. Scaler 3 (640x384 nvbuf transform to sysbuf via NPP)
    qcap2_video_scaler_set_backend_type(pScaler3, QCAP2_VIDEO_SCALER_BACKEND_TYPE_NPP);
    qcap2_video_format_t* pScalerFormat3 = qcap2_video_format_new();
    if (pScalerFormat3) {
        qcap2_video_format_set_property(pScalerFormat3, QCAP_COLORSPACE_TYPE_NV12, 640, 384, bVideoIsInterleaved, dVideoFrameRate);
        qcap2_video_scaler_set_video_format(pScaler3, pScalerFormat3);
        qcap2_video_format_delete(pScalerFormat3);
    }
    qcap2_video_scaler_set_frame_count(pScaler3, 8);
    qcap2_video_scaler_set_src_buffer_hint(pScaler3, QCAP2_BUFFER_HINT_CUDA);
    qcap2_video_scaler_set_dst_buffer_hint(pScaler3, QCAP2_BUFFER_HINT_DEFAULT);
    qcap2_video_scaler_set_auto_run(pScaler3, true);
    
    qDebug() << "Trace: Starting video scaler 3...";
    qres = qcap2_video_scaler_start(pScaler3);
    if (qres != QCAP_RS_SUCCESSFUL) {
        qCritical() << "qcap2_video_scaler_start for Scaler 3 failed for CH" << channelId << "qres =" << qres;
    }

    qDebug() << "Trace: onConnected completed successfully!";
    return QCAP_RT_OK;
}

QRETURN ChannelContext::onVideoCallback(double dSampleTime, BYTE * pStreamBuffer, ULONG nStreamBufferLen, BOOL bIsKeyFrame) {
    Q_UNUSED(dSampleTime);
    Q_UNUSED(bIsKeyFrame);

    qDebug() << "Trace: onVideoCallback entry";
    qcap2_video_decoder_t* pLocalVdec = nullptr;

    {
        QMutexLocker locker(&m_mutex);
        if (!pClient || !pVdec) {
            return QCAP_RT_OK;
        }
        pLocalVdec = pVdec;
    }

    qcap2_rcbuffer_t* pRCBuffer = qcap2_rcbuffer_cast(pStreamBuffer, nStreamBufferLen);
    if (!pRCBuffer) {
        qDebug() << "Trace: onVideoCallback - pRCBuffer is null";
        return QCAP_RT_OK;
    }

    qDebug() << "Trace: onVideoCallback - pushing H.264 packet to decoder...";
    // Push H.264 packet directly to the hardware decoder
    QRESULT qres = qcap2_video_decoder_push(pLocalVdec, pRCBuffer);
    if (qres != QCAP_RS_SUCCESSFUL) {
        qCritical() << "qcap2_video_decoder_push failed for CH" << channelId << "qres =" << qres;
    }
    qDebug() << "Trace: onVideoCallback finished pushing";

    if (!m_pushTimer.isValid()) {
        m_pushTimer.start();
    }
    m_pushFrameCount++;
    if (m_pushTimer.elapsed() >= 2000) {
        m_pushFrameCount = 0;
        m_pushTimer.restart();
    }

    return QCAP_RT_OK;
}

QRETURN ChannelContext::onEventVdec() {
    qDebug() << "Trace: onEventVdec entry";
    qcap2_video_decoder_t* pLocalVdec = nullptr;
    qcap2_video_scaler_t* pLocalScaler2 = nullptr;
    qcap2_video_scaler_t* pLocalScaler3 = nullptr;
    qcap2_video_sink_t* pLocalVideoSink = nullptr;
    bool bDisplayEnabled = false;
    bool bSendBuffer = false;

    {
        QMutexLocker locker(&m_mutex);
        if (!pVdec) return QCAP_RT_OK;
        pLocalVdec = pVdec;
        pLocalScaler2 = pScaler2;
        pLocalScaler3 = pScaler3;
        pLocalVideoSink = pVideoSink;
        bDisplayEnabled = m_bDisplayEnabled;
        bSendBuffer = m_bSendBuffer;
    }

    qDebug() << "Trace: onEventVdec - popping frame from decoder...";
    qcap2_rcbuffer_t* pRCBuffer_vdec = nullptr;
    QRESULT qres = qcap2_video_decoder_pop(pLocalVdec, &pRCBuffer_vdec);
    if (qres != QCAP_RS_SUCCESSFUL) {
        qDebug() << "Trace: onEventVdec - pop failed, qres=" << qres;
        return QCAP_RT_OK;
    }
    qDebug() << "Trace: onEventVdec - pop succeeded, buffer=" << pRCBuffer_vdec;

    // Print Decoder Output Info
    static int dec_log_cnt = 0;
    if (++dec_log_cnt % 30 == 0) {
        NvBufSurface* pSurface = nullptr;
        qcap2_rcbuffer_get_nvbuf(pRCBuffer_vdec, &pSurface);
        PVOID pLockedData = qcap2_rcbuffer_lock_data(pRCBuffer_vdec);
        qcap2_rcbuffer_unlock_data(pRCBuffer_vdec);
        qDebug() << "[Decoder Output Info] CH" << channelId 
                 << "Buffer:" << pRCBuffer_vdec 
                 << "Is NVBUF:" << (pSurface != nullptr) 
                 << "NvBufSurface Addr:" << pSurface 
                 << "Cpu/Locked Addr:" << pLockedData;
    }

    // Decoder FPS tracking
    if (!m_fpsTimer.isValid()) {
        m_fpsTimer.start();
    }
    m_decFrameCount++;
    if (m_fpsTimer.elapsed() >= 2000) {
        m_decFrameCount = 0;
        m_fpsTimer.restart();
    }

    // ── AI Frame Capture ────────────────────────────────────────────
    if (bSendBuffer && g_pMainwindow && g_pMainwindow->ai_running) {
        double current_time = QCAP_GET_TIME();
        if ((current_time - m_lastProcessTime) >= FRAME_INTERVAL && !m_bFrameReady) {
            m_lastProcessTime = current_time;

            NvBufSurface* pSurface = nullptr;
            qcap2_rcbuffer_get_nvbuf(pRCBuffer_vdec, &pSurface);

            static int qdeep_log_cnt = 0;
            if (++qdeep_log_cnt % 30 == 0) {
                qDebug() << "[QDEEP Copy Info] CH" << channelId 
                         << "Buffer:" << pRCBuffer_vdec 
                         << "Is NVBUF:" << (pSurface != nullptr) 
                         << "NvBufSurface Addr:" << pSurface;
            }

            if (pSurface) {
                // Release the old QDEEP nvbuf buffer reference if there was one
                if (m_pCurrentAIRCBuffer) {
                    qcap2_rcbuffer_release(m_pCurrentAIRCBuffer);
                }
                // Hold a reference to the new buffer so it is not freed until next frame or shutdown
                m_pCurrentAIRCBuffer = pRCBuffer_vdec;
                qcap2_rcbuffer_add_ref(m_pCurrentAIRCBuffer);

                g_pMainwindow->buffer_vec[channelId] = reinterpret_cast<BYTE*>(pSurface);
                g_pMainwindow->buffer_len_vec[channelId] = sizeof(NvBufSurface);

                // 標記有新 frame，通知 AI thread
                {
                    std::lock_guard<std::mutex> lock(g_pMainwindow->mtx);
                    if (!m_bFrameReady) {
                        m_bFrameReady = true;
                        g_pMainwindow->ready_count++;
                        if (g_pMainwindow->ready_count >= g_pMainwindow->active_camera_count)
                            g_pMainwindow->cv.notify_one();
                    }
                }
            }
        }
    }
    // 2. Scaler 3 (NPP: NVBUF to 640x384 SYSBUF)
    if (pLocalScaler3) {
        qDebug() << "Trace: onEventVdec - pushing frame to scaler 3...";
        qres = qcap2_video_scaler_push(pLocalScaler3, pRCBuffer_vdec);
        if (qres == QCAP_RS_SUCCESSFUL) {
            qDebug() << "Trace: onEventVdec - scaler 3 push succeeded, popping output...";
            qcap2_rcbuffer_t* pSysBuffer = nullptr;
            qres = qcap2_video_scaler_pop(pLocalScaler3, &pSysBuffer);
            if (qres == QCAP_RS_SUCCESSFUL && pSysBuffer) {
                // Push to Video Sink
                bool bSinkActive = false;
                {
                    QMutexLocker locker(&m_mutex);
                    if (pVideoSink) bSinkActive = true;
                }
                if (bDisplayEnabled && bSinkActive && pLocalVideoSink) {
                    qDebug() << "Trace: onEventVdec - pushing to video sink...";
                    qcap2_video_sink_push(pLocalVideoSink, pSysBuffer);
                }
                qcap2_rcbuffer_release(pSysBuffer);
            } else {
                qDebug() << "Trace: onEventVdec - scaler 3 pop failed or empty, qres=" << qres;
            }
        } else {
            qDebug() << "Trace: onEventVdec - scaler 3 push failed, qres=" << qres;
        }
    }

    qcap2_rcbuffer_release(pRCBuffer_vdec);

    {
        QMutexLocker locker(&m_mutex);
        m_frameCount++;
    }

    return QCAP_RT_OK;
}


// ── MainWindow Implementation ───────────────────────────────────────────────
MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), m_bFullscreen(false),
      // AI init
      overlayWidget(nullptr), m_bShowOverlay(true),
      handle(nullptr), flag(1),
      ai_running(false), pAiThread(nullptr),
      ready_count(0), active_camera_count(0)
{
    setWindowTitle("QCAP Multichannel RTSP + QDEEP 17KPS Skeleton");
    resize(1280, 720);

    g_pMainwindow = this;

    // ── Initialize AI members ────────────────────────────────────────────
    color_space.resize(MAX_BATCH);
    width_vec.resize(MAX_BATCH);
    height_vec.resize(MAX_BATCH);
    buffer_vec.assign(MAX_BATCH, nullptr);
    buffer_len_vec.resize(MAX_BATCH);
    box_size_vec.assign(MAX_BATCH, BOX_SIZE);
    box_list_vec.assign(MAX_BATCH, nullptr);

    centralWidget = new QWidget(this);
    setCentralWidget(centralWidget);

    QHBoxLayout *mainLayout = new QHBoxLayout(centralWidget);
    mainLayout->setContentsMargins(5, 5, 5, 5);
    mainLayout->setSpacing(5);

    // Create Control Panel (Left Side)
    controlPanel = new QWidget(centralWidget);
    controlPanel->setFixedWidth(380);
    QVBoxLayout *controlLayout = new QVBoxLayout(controlPanel);
    controlLayout->setContentsMargins(0, 0, 0, 0);

    QGroupBox *grpConfig = new QGroupBox("RTSP Configuration", controlPanel);
    QVBoxLayout *grpLayout = new QVBoxLayout(grpConfig);

    grpLayout->addWidget(new QLabel("Channel Count (1-64):"));
    spinChannelCount = new QSpinBox(grpConfig);
    spinChannelCount->setRange(1, MAX_CHANNELS);
    spinChannelCount->setValue(4);
    grpLayout->addWidget(spinChannelCount);

    grpLayout->addWidget(new QLabel("RTSP URLs:"));
    tableUrls = new QTableWidget(4, 2, grpConfig);
    tableUrls->setHorizontalHeaderLabels({"CH", "RTSP URL"});
    tableUrls->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    tableUrls->horizontalHeader()->setSectionResizeMode(1, QHeaderView::Stretch);
    tableUrls->verticalHeader()->setVisible(false);
    grpLayout->addWidget(tableUrls);

    btnStart = new QPushButton("Start Broadcast Clients", grpConfig);
    btnStop = new QPushButton("Stop All", grpConfig);
    btnStop->setEnabled(false);
    grpLayout->addWidget(btnStart);
    grpLayout->addWidget(btnStop);

    chkEnableDisplay = new QCheckBox("Enable Display Rendering", grpConfig);
    chkEnableDisplay->setChecked(true);
    grpLayout->addWidget(chkEnableDisplay);

    chkShowOverlay = new QCheckBox("Show AI Skeleton Overlay", grpConfig);
    chkShowOverlay->setChecked(true);
    grpLayout->addWidget(chkShowOverlay);

    controlLayout->addWidget(grpConfig);

    lblStatus = new QLabel("Status: Idle", controlPanel);
    lblStatus->setWordWrap(true);
    controlLayout->addWidget(lblStatus);

    mainLayout->addWidget(controlPanel);

    // Create Video Grid Container (Right Side)
    videoContainer = new QWidget(centralWidget);
    videoGridLayout = new QGridLayout(videoContainer);
    videoGridLayout->setContentsMargins(0, 0, 0, 0);
    videoGridLayout->setSpacing(2);
    mainLayout->addWidget(videoContainer, 1);

    // Set default URLs
    onChannelCountChanged(4);
    m_bEnableDisplay = true;

    // Signal Slot connections
    connect(spinChannelCount, QOverload<int>::of(&QSpinBox::valueChanged), this, &MainWindow::onChannelCountChanged);
    connect(btnStart, &QPushButton::clicked, this, &MainWindow::onBtnStartClicked);
    connect(btnStop, &QPushButton::clicked, this, &MainWindow::onBtnStopClicked);
    connect(chkEnableDisplay, &QCheckBox::toggled, this, &MainWindow::onDisplayToggled);
    connect(chkShowOverlay, &QCheckBox::toggled, this, &MainWindow::onOverlayToggled);

    videoContainer->installEventFilter(this);

    m_timerId = startTimer(1000);

    // ── Initialize QDEEP models ─────────────────────────────────────────
    init_models();

    // ── Create Overlay Widget (top-level transparent window) ───────────
    overlayWidget = new OverlayWidget(nullptr);
    overlayWidget->setGeometry(this->geometry());
    overlayWidget->show();
    overlayWidget->raise();
}

MainWindow::~MainWindow()
{
    if (overlayWidget) {
        delete overlayWidget;
        overlayWidget = nullptr;
    }
    uninit_models();
    stopAllChannels();
}

// ── UI Event Handlers ───────────────────────────────────────────────────────
void MainWindow::onChannelCountChanged(int count)
{
    tableUrls->setRowCount(count);
    for (int i = 0; i < count; ++i) {
        QTableWidgetItem *itemCh = tableUrls->item(i, 0);
        if (!itemCh) {
            itemCh = new QTableWidgetItem(QString("CH%1").arg(i + 1));
            itemCh->setFlags(itemCh->flags() & ~Qt::ItemIsEditable);
            itemCh->setTextAlignment(Qt::AlignCenter);
            tableUrls->setItem(i, 0, itemCh);
        }

        QTableWidgetItem *itemUrl = tableUrls->item(i, 1);
        if (!itemUrl || itemUrl->text().isEmpty()) {
            QString defaultUrl = QString("rtsp://root:root@192.168.191.6:1554/session0.mpg");
            tableUrls->setItem(i, 1, new QTableWidgetItem(defaultUrl));
        }
    }
}

void MainWindow::onBtnStartClicked()
{
    stopAllChannels();
    clearGrid();

    int count = spinChannelCount->value();

    int cols = 2;
    if (count <= 2) cols = count;
    else if (count <= 4) cols = 2;
    else if (count <= 9) cols = 3;
    else cols = 4;

    for (int i = 0; i < count; ++i) {
        QFrame *frame = new QFrame(videoContainer);
        frame->setFrameShape(QFrame::Box);
        frame->setLineWidth(1);
        frame->setStyleSheet("background-color: black; border: 1px solid #333333;");
        frame->installEventFilter(this);
        videoFrames.append(frame);

        int row = i / cols;
        int col = i % cols;
        videoGridLayout->addWidget(frame, row, col);
        frame->show();

        QString url = tableUrls->item(i, 1)->text().trimmed();
        ChannelContext *ctx = new ChannelContext(i, url, frame->winId());
        ctx->setDisplayEnabled(m_bEnableDisplay);
        channels.append(ctx);

        ctx->start();
    }

    spinChannelCount->setEnabled(false);
    tableUrls->setEnabled(false);
    btnStart->setEnabled(false);
    btnStop->setEnabled(true);
    lblStatus->setText("Status: Running");

    // Start AI inference
    yolo_start();
}

void MainWindow::onBtnStopClicked()
{
    yolo_stop();

    stopAllChannels();
    clearGrid();

    spinChannelCount->setEnabled(true);
    tableUrls->setEnabled(true);
    btnStart->setEnabled(true);
    btnStop->setEnabled(false);
    lblStatus->setText("Status: Stopped");
}

void MainWindow::stopAllChannels()
{
    for (ChannelContext *ctx : channels) {
        ctx->m_bSendBuffer = false;
        delete ctx;
    }
    channels.clear();
}

void MainWindow::clearGrid()
{
    QLayoutItem *item;
    while ((item = videoGridLayout->takeAt(0)) != nullptr) {
        if (item->widget()) {
            item->widget()->deleteLater();
        }
        delete item;
    }
    videoFrames.clear();
}

void MainWindow::timerEvent(QTimerEvent *event)
{
    if (event->timerId() == m_timerId) {
        QString statusText = "Channel Stats:\n";
        for (int i = 0; i < channels.size(); ++i) {
            ChannelContext *ctx = channels[i];
            QMutexLocker locker(&ctx->m_mutex);
            statusText += QString("CH%1: %2 | %3 fps\n")
                          .arg(i + 1)
                          .arg(ctx->m_statusInfo.isEmpty() ? "Connecting..." : ctx->m_statusInfo)
                          .arg(ctx->m_frameCount);
            ctx->m_frameCount = 0;
        }
        lblStatus->setText(statusText);
    }
}

void MainWindow::closeEvent(QCloseEvent *event)
{
    yolo_stop();
    stopAllChannels();
    event->accept();
}

bool MainWindow::eventFilter(QObject *watched, QEvent *event)
{
    if (event->type() == QEvent::MouseButtonDblClick) {
        m_bFullscreen = !m_bFullscreen;
        if (m_bFullscreen) {
            controlPanel->hide();
            showFullScreen();
        } else {
            controlPanel->show();
            showNormal();
        }

        int count = videoFrames.size();
        if (count > 0) {
            int cols = 2;
            if (m_bFullscreen) {
                cols = 8;
            } else {
                if (count <= 2) cols = count;
                else if (count <= 4) cols = 2;
                else if (count <= 9) cols = 3;
                else cols = 4;
            }

            for (int i = 0; i < count; ++i) {
                videoGridLayout->removeWidget(videoFrames[i]);
                int row = i / cols;
                int col = i % cols;
                videoGridLayout->addWidget(videoFrames[i], row, col);
            }
        }
        return true;
    }
    return QMainWindow::eventFilter(watched, event);
}

void MainWindow::onDisplayToggled(bool checked)
{
    m_bEnableDisplay = checked;
    for (ChannelContext *ctx : channels) {
        ctx->setDisplayEnabled(checked);
    }
}

void MainWindow::onOverlayToggled(bool checked)
{
    m_bShowOverlay = checked;
    if (overlayWidget) {
        overlayWidget->update();
    }
}

// ── Top-level OverlayWidget 事件追蹤 ──────────────────────────────────────
void MainWindow::resizeEvent(QResizeEvent *event) {
    QMainWindow::resizeEvent(event);
    if (overlayWidget) {
        overlayWidget->setGeometry(this->geometry());
        overlayWidget->raise();
    }
}

void MainWindow::moveEvent(QMoveEvent *event) {
    QMainWindow::moveEvent(event);
    if (overlayWidget) {
        overlayWidget->setGeometry(this->geometry());
        overlayWidget->raise();
    }
}

void MainWindow::hideEvent(QHideEvent *event) {
    QMainWindow::hideEvent(event);
    if (overlayWidget) {
        overlayWidget->hide();
    }
}

void MainWindow::showEvent(QShowEvent *event) {
    QMainWindow::showEvent(event);
    if (overlayWidget) {
        overlayWidget->setGeometry(this->geometry());
        overlayWidget->show();
        overlayWidget->raise();
    }
}

// ── QDEEP / YOLO AI Functions ──────────────────────────────────────────────
void MainWindow::init_models()
{
    for (int i = 0; i < MAX_BATCH; ++i) {
        box_list_vec[i] = new QDEEP_API::QDEEP_OBJECT_DETECT_BOUNDING_BOX[BOX_SIZE];
        buffer_vec[i] = nullptr;
        color_space[i] = QDEEP_API::QDEEP_COLORSPACE_TYPE_NV12;
        width_vec[i] = 640;
        height_vec[i] = 384;
        buffer_len_vec[i] = 0;
    }

    QRESULT res = QDEEP_API::QDEEP_CREATE_BATCH_OBJECT_DETECT(
        QDEEP_API::QDEEP_GPU_TYPE_NVIDIA, 0,
        QDEEP_API::QDEEP_OBJECT_DETECT_CONFIG_MODEL_HUMAN_SKELETON_17_KEYPOINTS_EX,
        (char*)"/home/nvidia/Documents/QtQcapMultiClientDemo_onlydecode_npptosys/model/skeleton_ex/QDEEP.OD.HUMAN.SKELETON.17KPS.EX.CFG",
        &handle, flag, MAX_BATCH);

    qDebug() << "[AI Log] QDEEP_CREATE_BATCH_OBJECT_DETECT res:" << QString("0x%1").arg(res, 8, 16, QChar('0')) << "handle:" << handle;

    if (res == 0 && handle != nullptr) {
        QDEEP_API::QDEEP_START_OBJECT_DETECT(handle);
        QDEEP_API::QDEEP_SET_OBJECT_DETECT_PROPERTY(handle, 0.1);
    }

    res = QDEEP_GET_OBJECT_DETECT_RESERVED_STATUS(reinterpret_cast<PVOID>(0xD7CBB416), reinterpret_cast<ULONG*>(0x3B98119E));
    qDebug() << "[AI Log] QDEEP_GET_OBJECT_DETECT_RESERVED_STATUS res:" << QString("0x%1").arg(res, 8, 16, QChar('0'));
}

void MainWindow::uninit_models()
{
    yolo_stop();

    if (handle != nullptr) {
        QDEEP_API::QDEEP_STOP_OBJECT_DETECT(handle);
        QDEEP_API::QDEEP_DESTROY_OBJECT_DETECT(handle);
        handle = nullptr;
    }

    for (size_t i = 0; i < MAX_BATCH; ++i) {
        if (box_list_vec[i]) { delete[] box_list_vec[i]; box_list_vec[i] = nullptr; }
        buffer_vec[i] = nullptr;
    }
}

void MainWindow::yolo_start()
{
    if (ai_running) return;

    active_camera_count = 0;
    for (ChannelContext *ctx : channels) {
        if (ctx->pClient != nullptr) {
            ctx->m_bSendBuffer = true;
            ctx->m_lastProcessTime = 0.0;
            ctx->m_bFrameReady = false;
            active_camera_count++;
        }
    }

    if (active_camera_count == 0) {
        qDebug() << "[Warning] No active cameras found! AI will not start.";
        return;
    }

    ai_running = true;
    pAiThread = new std::thread(&MainWindow::ai_inference_thread, this);
    qDebug() << "[Info] AI inference started.";
}

void MainWindow::yolo_stop()
{
    if (!ai_running) return;

    for (ChannelContext *ctx : channels) {
        ctx->m_bSendBuffer = false;
    }

    ai_running = false;
    cv.notify_one();

    if (pAiThread && pAiThread->joinable()) {
        pAiThread->join();
        delete pAiThread;
        pAiThread = nullptr;
    }
    qDebug() << "[Info] AI inference stopped.";
}

void MainWindow::ai_inference_thread()
{
    auto last_log_time = std::chrono::steady_clock::now();
    int inference_count = 0;

    while (ai_running) {
        std::unique_lock<std::mutex> lock(mtx);

        // 重新計算 active camera 數量
        active_camera_count = 0;
        for (ChannelContext *ctx : channels) {
            if (ctx->m_bSendBuffer && ctx->m_nVideoWidth > 0 && ctx->m_nVideoHeight > 0) {
                ctx->m_nAIWidth = 640;
                ctx->m_nAIHeight = 384;
                active_camera_count++;
            }
        }

        if (active_camera_count == 0) {
            lock.unlock();
            std::this_thread::sleep_for(std::chrono::milliseconds(67));
            continue;
        }

        // 等待全部 active channel 都有新 frame，或 timeout 後用舊資料送出
        cv.wait_for(lock, std::chrono::milliseconds(67), [this] {
            return (ready_count >= active_camera_count) || !ai_running;
        });

        if (!ai_running) break;

        // 重置 box sizes
        for (size_t i = 0; i < MAX_BATCH; ++i) {
            box_size_vec[i] = BOX_SIZE;
        }

        // 資料已在 onEventVdec 中直接寫入 buffer_vec[i]，不需再 memcpy
        // 若 m_bFrameReady == false，保留 buffer_vec[i] 上次的資料
        for (int i = 0; i < MAX_BATCH; ++i) {
            ChannelContext* ctx = nullptr;
            for (auto* c : channels) {
                if (c->channelId == i) {
                    ctx = c;
                    break;
                }
            }
            if (ctx && ctx->m_bSendBuffer && ctx->m_nVideoWidth > 0 && ctx->m_nVideoHeight > 0) {
                width_vec[i] = 640;
                height_vec[i] = 384;
            } else {
                width_vec[i] = 0;
                height_vec[i] = 0;
                buffer_len_vec[i] = 0;
            }
        }

        // 執行 QDEEP batch 推論
        QRESULT api_res = QDEEP_API::QDEEP_SET_VIDEO_OBJECT_DETECT_BATCH_UNCOMPRESSION_BUFFER(
            handle, color_space.data(), width_vec.data(), height_vec.data(),
            buffer_vec.data(), buffer_len_vec.data(), box_list_vec.data(), box_size_vec.data(), MAX_BATCH);

        inference_count++;

        // 記錄 AI FPS（每 5 秒輸出一次）
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_log_time).count();
        if (elapsed >= 5) {
            double ai_fps = (double)inference_count / elapsed;
            qDebug() << QString("[AI FPS 17kps] %1 Hz (%2 inferences in %3s, active=%4)")
                        .arg(ai_fps, 0, 'f', 1)
                        .arg(inference_count)
                        .arg(elapsed)
                        .arg(active_camera_count);
            inference_count = 0;
            last_log_time = now;
        }

        // 處理結果
        {
            std::lock_guard<std::mutex> draw_lock(draw_mtx);
            for (int i = 0; i < MAX_BATCH; ++i) {
                draw_persons[i].clear();

                if (width_vec[i] > 0 && box_size_vec[i] > 0) {
                    for (ULONG j = 0; j < box_size_vec[i]; ++j) {
                        auto& deep_box = box_list_vec[i][j];

                        DrawPerson dp;
                        dp.classId = deep_box.nClassID;
                        dp.probability = deep_box.fProbability;

                        for (int k = 0; k < 17; ++k) {
                            dp.keypoints[k].x = deep_box.sKeypoints[k].nX;
                            dp.keypoints[k].y = deep_box.sKeypoints[k].nY;
                            dp.keypoints[k].probability = deep_box.sKeypoints[k].fProbability;
                        }

                        draw_persons[i].push_back(dp);
                    }
                }
            }
        }

        // 觸發 overlay 更新（UI thread）
        QMetaObject::invokeMethod(this, [this]() {
            if (overlayWidget) {
                overlayWidget->raise();
                overlayWidget->update();
            }
        }, Qt::QueuedConnection);

        // 批次重置所有 frame ready flags
        for (ChannelContext *ctx : channels) {
            ctx->m_bFrameReady = false;
        }
        ready_count = 0;
    }
}
