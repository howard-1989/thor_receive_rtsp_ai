#include "mainwindow.h"
#include <QDebug>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QGroupBox>
#include <QHeaderView>
#include <QEvent>
#include <QMouseEvent>
#include <QPointer>
#include <QMetaObject>
#include <QPixmap>

#include <cstring>

#include <opencv2/opencv.hpp>

MainWindow *g_pMainwindow = nullptr;

extern "C" {
QDEEP_EXT_API QRESULT QDEEP_EXPORT QDEEP_GET_OBJECT_DETECT_RESERVED_STATUS(PVOID pDetector, ULONG* pCheckNum);
}

QImage cvMatToQImage(const cv::Mat& mat) {
    if (mat.type() == CV_8UC3) {
        return QImage(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_RGB888).rgbSwapped().copy();
    }
    return QImage();
}

static QString captionFromResult(const QDEEP_API::QDEEP_OBJECT_DETECT_BOUNDING_BOX& result)
{
    const char* text = reinterpret_cast<const char*>(result.fFeatureVectors);
    const size_t textLength = ::strnlen(text, sizeof(result.fFeatureVectors));
    return QString::fromUtf8(text, static_cast<int>(textLength)).trimmed();
}

static void drawCaption(cv::Mat& image, const QString& caption)
{
    if (caption.isEmpty()) {
        return;
    }

    const int maxWidth = image.cols - 20;
    const QStringList words = caption.split(' ', Qt::SkipEmptyParts);
    QString line;
    int baseline = 0;
    int y = 52;
    for (const QString& word : words) {
        const QString candidate = line.isEmpty() ? word : line + ' ' + word;
        const cv::Size candidateSize = cv::getTextSize(candidate.toStdString(), cv::FONT_HERSHEY_SIMPLEX, 0.45, 1, &baseline);
        if (!line.isEmpty() && candidateSize.width > maxWidth) {
            const cv::Size lineSize = cv::getTextSize(line.toStdString(), cv::FONT_HERSHEY_SIMPLEX, 0.45, 1, &baseline);
            cv::rectangle(image, cv::Point(6, y - lineSize.height - 5), cv::Point(14 + lineSize.width, y + baseline + 4), cv::Scalar(0, 0, 0), cv::FILLED);
            cv::putText(image, line.toStdString(), cv::Point(10, y), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(255, 255, 255), 1);
            y += lineSize.height + baseline + 10;
            line = word;
        } else {
            line = candidate;
        }
    }
    if (!line.isEmpty()) {
        const cv::Size lineSize = cv::getTextSize(line.toStdString(), cv::FONT_HERSHEY_SIMPLEX, 0.45, 1, &baseline);
        cv::rectangle(image, cv::Point(6, y - lineSize.height - 5), cv::Point(14 + lineSize.width, y + baseline + 4), cv::Scalar(0, 0, 0), cv::FILLED);
        cv::putText(image, line.toStdString(), cv::Point(10, y), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(255, 255, 255), 1);
    }
}


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

static QRETURN on_fail_callback(
        PVOID pClient,
        UINT iSessionNum,
        QRESULT nErrorStatus,
        DWORD nErrorCode,
        PVOID pUserData)
{
    ChannelContext* ctx = static_cast<ChannelContext*>(pUserData);
    return ctx->onFail(iSessionNum, nErrorStatus, nErrorCode);
}

// ── ChannelContext Implementation ───────────────────────────────────────────
ChannelContext::ChannelContext(int id, const QString& streamUrl, QLabel* pLabel)
    : channelId(id), url(streamUrl), m_pLabel(pLabel),
      pClient(nullptr), pVdec(nullptr), pEventHandlers(nullptr),
      pEvent_vdec(nullptr),
      pScaler2(nullptr), m_pCurrentAIRCBuffer(nullptr),
      m_pAIQueue(nullptr),
      m_nVideoWidth(0), m_nVideoHeight(0), m_dVideoFrameRate(0.0), m_nVideoEncoderFormat(0),
      m_frameCount(0), m_bDisplayEnabled(true),
      m_pushFrameCount(0), m_decFrameCount(0),
      // AI init
      m_bSendBuffer(false), m_lastProcessTime(0.0), m_bFrameReady(false),
      m_pAIBuffer(nullptr), m_nAIBufferLen(0), m_nAIWidth(0), m_nAIHeight(0)
{
    m_pPendingUpdate = std::make_shared<std::atomic<bool>>(false);
    m_displayFrameCount = 0;
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
    QCAP_REGISTER_BROADCAST_CLIENT_FAIL_CALLBACK(pClient, on_fail_callback, this);

    // Start stream receiver (TCP mode for RTSP)
    qres = QCAP_START_BROADCAST_CLIENT(pClient, QCAP_BROADCAST_PROTOCOL_TCP, 10000, 0);
    if (qres != QCAP_RS_SUCCESSFUL) {
        qCritical() << "QCAP_START_BROADCAST_CLIENT failed for CH" << channelId << "qres =" << qres;
        QCAP_DESTROY_BROADCAST_CLIENT(pClient);
        pClient = nullptr;
        return false;
    }

    return true;
}

void ChannelContext::cleanupPipeline() {
    qcap2_event_handlers_t* pLocalEventHandlers = nullptr;
    qcap2_video_scaler_t* pLocalScaler2 = nullptr;
    qcap2_video_decoder_t* pLocalVdec = nullptr;
    qcap2_rcbuffer_queue_t* pLocalAIQueue = nullptr;
    qcap2_event_t* pLocalEvent_vdec = nullptr;
    qcap2_rcbuffer_t* pLocalCurrentAIRCBuffer = nullptr;

    {
        QMutexLocker locker(&m_mutex);
        pLocalEventHandlers = pEventHandlers;
        pLocalScaler2 = pScaler2;
        pLocalVdec = pVdec;
        pLocalAIQueue = m_pAIQueue;
        pLocalEvent_vdec = pEvent_vdec;
        pLocalCurrentAIRCBuffer = m_pCurrentAIRCBuffer;

        pEventHandlers = nullptr;
        pScaler2 = nullptr;
        pVdec = nullptr;
        m_pAIQueue = nullptr;
        pEvent_vdec = nullptr;
        m_pCurrentAIRCBuffer = nullptr;
    }

    qDebug() << "========== CH" << channelId << "Pipeline Cleanup Started ==========";

    uintptr_t nHandle_vdec = 0;
    if (pLocalEvent_vdec) {
        qcap2_event_get_native_handle(pLocalEvent_vdec, &nHandle_vdec);
    }

    // 1. Stop the decoder first to unblock any active pop/push operations in other threads
    if (pLocalVdec) {
        qDebug() << "CH" << channelId << "cleanup: Calling qcap2_video_decoder_stop...";
        qcap2_video_decoder_stop(pLocalVdec);
    }

    // 2. Stop the event object
    if (pLocalEvent_vdec) {
        qDebug() << "CH" << channelId << "cleanup: Stopping pEvent_vdec...";
        qcap2_event_stop(pLocalEvent_vdec);
    }

    // 3. Stop the event handlers (the loop thread will exit immediately since pop is unblocked)
    if (pLocalEventHandlers) {
        qDebug() << "CH" << channelId << "cleanup: Stopping event handlers...";
        qcap2_event_handlers_stop(pLocalEventHandlers);
    }

    // 4. Stop the scalers
    if (pLocalScaler2) {
        qDebug() << "CH" << channelId << "cleanup: Stopping Scaler 2...";
        qcap2_video_scaler_stop(pLocalScaler2);
    }

    // 5. Stop the AI Queue
    if (pLocalAIQueue) {
        qDebug() << "CH" << channelId << "cleanup: Draining and stopping AI Queue...";
        qcap2_rcbuffer_t* pBuf = nullptr;
        while (qcap2_rcbuffer_queue_pop(pLocalAIQueue, &pBuf) == QCAP_RS_SUCCESSFUL && pBuf) {
            qcap2_rcbuffer_release(pBuf);
        }
        qcap2_rcbuffer_queue_stop(pLocalAIQueue);
    }

    if (pLocalScaler2) {
        qDebug() << "CH" << channelId << "cleanup: Deleting Scaler 2...";
        qcap2_video_scaler_delete(pLocalScaler2);
    }
    if (pLocalAIQueue) {
        qDebug() << "CH" << channelId << "cleanup: Deleting AI Queue...";
        qcap2_rcbuffer_queue_delete(pLocalAIQueue);
    }
    if (pLocalVdec) {
        qDebug() << "CH" << channelId << "cleanup: Deleting video decoder...";
        qcap2_video_decoder_delete(pLocalVdec);
    }
    if (pLocalEventHandlers) {
        qDebug() << "CH" << channelId << "cleanup: Deleting event handlers...";
        qcap2_event_handlers_delete(pLocalEventHandlers);
    }
    if (pLocalEvent_vdec) {
        qDebug() << "CH" << channelId << "cleanup: Deleting event vdec...";
        qcap2_event_delete(pLocalEvent_vdec);
    }
    if (pLocalCurrentAIRCBuffer) {
        qDebug() << "CH" << channelId << "cleanup: Releasing m_pCurrentAIRCBuffer...";
        qcap2_rcbuffer_release(pLocalCurrentAIRCBuffer);
    }

    qDebug() << "========== CH" << channelId << "Pipeline Cleanup Finished ==========";
}

void ChannelContext::stop() {
    PVOID pLocalClient = nullptr;
    {
        QMutexLocker locker(&m_mutex);
        pLocalClient = pClient;
        pClient = nullptr;
    }

    qDebug() << "========== CH" << channelId << "Stop Sequence Started ==========";

    // 1. Stop the broadcast client first to stop receiving raw stream packets
    if (pLocalClient) {
        qDebug() << "CH" << channelId << "stop: Stopping broadcast client...";
        QCAP_STOP_BROADCAST_CLIENT(pLocalClient);
        qDebug() << "CH" << channelId << "stop: Broadcast client stopped.";
    }

    // 2. Clean up the decoding pipeline safely
    cleanupPipeline();

    // 3. Destroy the broadcast client
    if (pLocalClient) {
        qDebug() << "CH" << channelId << "stop: Destroying broadcast client...";
        QCAP_DESTROY_BROADCAST_CLIENT(pLocalClient);
        qDebug() << "CH" << channelId << "stop: Broadcast client destroyed.";
    }

    qDebug() << "========== CH" << channelId << "Stop Sequence Finished ==========";
}

QRETURN ChannelContext::onFail(UINT iSessionNum, QRESULT nErrorStatus, DWORD nErrorCode) {
    Q_UNUSED(iSessionNum);
    QMutexLocker locker(&m_mutex);
    qCritical() << "CH" << channelId << "Broadcast client failure callback! Status:" << nErrorStatus << "Code:" << nErrorCode;
    m_statusInfo = QString("Disconnected (Error 0x%1)").arg(nErrorStatus, 8, 16, QChar('0'));
    return QCAP_RT_OK;
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

    bool need_cleanup = false;
    {
        QMutexLocker locker(&m_mutex);
        if (pVdec || pEventHandlers) {
            need_cleanup = true;
        }
    }
    if (need_cleanup) {
        qDebug() << "CH" << channelId << "Reconnecting: Cleaning up previous pipeline...";
        cleanupPipeline();
    }

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

    // Initialize software Decoder
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
                                               QCAP_ENCODER_TYPE_SOFTWARE,
                                               nVideoEncoderFormat,
                                               QCAP_COLORSPACE_TYPE_NV12,
                                               nVideoWidth, nVideoHeight, dVideoFrameRate,
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

    qDebug() << "Trace: Creating video scaler 2 (Scaler 2)...";
    pScaler2 = qcap2_video_scaler_new();
    if (!pScaler2) {
        qCritical() << "CH" << channelId << "Failed to create video scaler 2.";
        m_statusInfo = "Error: Failed to create video scaler 2";
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

    qDebug() << "Trace: Setting video scaler 2 properties (Scaler 2)...";
    qcap2_video_scaler_set_backend_type(pScaler2, QCAP2_VIDEO_SCALER_BACKEND_TYPE_DEFAULT);
    qcap2_video_format_t* pScalerFormat2 = qcap2_video_format_new();
    if (pScalerFormat2) {
        qcap2_video_format_set_property(pScalerFormat2, QCAP_COLORSPACE_TYPE_NV12, 640, 384, bVideoIsInterleaved, dVideoFrameRate);
        qcap2_video_scaler_set_video_format(pScaler2, pScalerFormat2);
        qcap2_video_format_delete(pScalerFormat2);
    }
    qcap2_video_scaler_set_frame_count(pScaler2, 8);

    qcap2_video_scaler_set_src_buffer_hint(pScaler2, QCAP2_BUFFER_HINT_DEFAULT);
    qcap2_video_scaler_set_dst_buffer_hint(pScaler2, QCAP2_BUFFER_HINT_DEFAULT); // sysbuf output
    qcap2_video_scaler_set_auto_run(pScaler2, true);

    qDebug() << "Trace: Starting video scaler 2 (Scaler 2)...";
    qres = qcap2_video_scaler_start(pScaler2);
    if (qres != QCAP_RS_SUCCESSFUL) {
        qCritical() << "qcap2_video_scaler_start for Scaler 2 failed for CH" << channelId << "qres =" << qres;
    }

    // ── Create AI Queue ────────────────────────────────────────────────
    m_pAIQueue = qcap2_rcbuffer_queue_new();
    if (m_pAIQueue) {
        qcap2_rcbuffer_queue_set_max_buffers(m_pAIQueue, 3);
        qcap2_rcbuffer_queue_start(m_pAIQueue);
        qDebug() << "CH" << channelId << "AI queue created and started (max=3).";
    } else {
        qCritical() << "CH" << channelId << "Failed to create AI queue!";
    }

    qDebug() << "Trace: onConnected completed successfully!";
    return QCAP_RT_OK;
}

QRETURN ChannelContext::onVideoCallback(double dSampleTime, BYTE * pStreamBuffer, ULONG nStreamBufferLen, BOOL bIsKeyFrame) {
    Q_UNUSED(dSampleTime);
    Q_UNUSED(bIsKeyFrame);

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
        return QCAP_RT_OK;
    }

    // A compressed H.264 access unit can span more than one callback. The
    // decoder reports NEED_MORE_DATA until it has enough bytes to produce a
    // frame; this is normal and the packet has still been consumed.
    QRESULT qres = qcap2_video_decoder_push(pLocalVdec, pRCBuffer);
    if (qres != QCAP_RS_SUCCESSFUL && qres != QCAP_RS_ERROR_NEED_MORE_DATA) {
        // qCritical() << "qcap2_video_decoder_push failed for CH" << channelId << "qres =" << qres;
    }

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
    qcap2_video_decoder_t* pLocalVdec = nullptr;
    qcap2_video_scaler_t* pLocalScaler2 = nullptr;
    bool bDisplayEnabled = false;
    bool bSendBuffer = false;
    double dSourceFrameRate = DEFAULT_AI_TARGET_FPS;
    qcap2_rcbuffer_queue_t* pLocalAIQueue = nullptr;

    {
        QMutexLocker locker(&m_mutex);
        if (!pVdec) return QCAP_RT_OK;
        pLocalVdec = pVdec;
        pLocalScaler2 = pScaler2;
        bDisplayEnabled = m_bDisplayEnabled;
        bSendBuffer = m_bSendBuffer;
        dSourceFrameRate = m_dVideoFrameRate;
        pLocalAIQueue = m_pAIQueue;
    }

    // A decoder event can cover several completed frames. Drain them all:
    // leaving completed frames in the decoder eventually stalls its pipeline.
    while (true) {
        qcap2_rcbuffer_t* pRCBuffer_vdec = nullptr;
        QRESULT qres = qcap2_video_decoder_pop(pLocalVdec, &pRCBuffer_vdec);
        if (qres != QCAP_RS_SUCCESSFUL) {
            // The event is level-triggered; an empty queue is the normal end
            // of this drain cycle.
            if (qres != QCAP_RS_ERROR_TIMEOUT && qres != QCAP_RS_ERROR_NEED_MORE_DATA) {
                qDebug() << "Trace: onEventVdec - pop failed, qres=" << qres;
            }
            break;
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

    qcap2_rcbuffer_t* pScaledBuffer = nullptr;
    if (pLocalScaler2) {
        qres = qcap2_video_scaler_push(pLocalScaler2, pRCBuffer_vdec);
        if (qres == QCAP_RS_SUCCESSFUL) {
            qres = qcap2_video_scaler_pop(pLocalScaler2, &pScaledBuffer);
            if (qres != QCAP_RS_SUCCESSFUL || !pScaledBuffer) {
                qDebug() << "Trace: onEventVdec - scaler pop failed, qres=" << qres;
            }
        } else {
            qDebug() << "Trace: onEventVdec - scaler push failed, qres=" << qres;
        }
    }

    if (pScaledBuffer) {
        // ── Destination 1: Push to AI queue (non-blocking) ──────────────────
        if (bSendBuffer && g_pMainwindow && g_pMainwindow->ai_running && pLocalAIQueue) {
            // Follow the frame rate reported by the RTSP connected callback.
            // Some devices report 0 fps, so keep a safe fallback in that case.
            const double targetFps = dSourceFrameRate > 0.0 ? dSourceFrameRate : DEFAULT_AI_TARGET_FPS;
            const double frameInterval = 1.0 / targetFps;
            const double current_time = QCAP_GET_TIME();
            if ((current_time - m_lastProcessTime) >= frameInterval) {
                m_lastProcessTime = current_time;

                // If queue full, drop oldest frame to make room
                if (qcap2_rcbuffer_queue_is_full(pLocalAIQueue)) {
                    qcap2_rcbuffer_t* pOld = nullptr;
                    if (qcap2_rcbuffer_queue_pop(pLocalAIQueue, &pOld) == QCAP_RS_SUCCESSFUL && pOld) {
                        qcap2_rcbuffer_release(pOld);
                    }
                }

                // Push current frame to queue (queue addref's the buffer)
                QRESULT qr = qcap2_rcbuffer_queue_push(pLocalAIQueue, pScaledBuffer);
                if (qr != QCAP_RS_SUCCESSFUL) {
                    qDebug() << "[AI Queue] CH" << channelId << "push failed, qres=" << qr;
                }

            }
        }

        // ── Destination 2: Continuously refresh display screen ──────────────
        if (bDisplayEnabled && m_pLabel) {
            bool skip_this_frame = false;
            if (g_pMainwindow && g_pMainwindow->m_bHalfRefreshRate) {
                int f_cnt = m_displayFrameCount.fetch_add(1);
                if (f_cnt % 2 != 0) {
                    skip_this_frame = true;
                }
            }

            if (!skip_this_frame) {
                // Check backpressure: skip frame if the GUI thread is busy rendering the previous one
                if (m_pPendingUpdate && !m_pPendingUpdate->exchange(true)) {
                    PVOID pLockedData = qcap2_rcbuffer_lock_data(pScaledBuffer);
                    if (pLockedData) {
                        qcap2_av_frame_t* pAVFrame = reinterpret_cast<qcap2_av_frame_t*>(pLockedData);
                        uint8_t* pBuffer[4] = {nullptr};
                        int pStride[4] = {0};
                        qcap2_av_frame_get_buffer1(pAVFrame, pBuffer, pStride);

                        if (pBuffer[0] && pStride[0] > 0) {
                            int copyWidth = 640;
                            int copyHeight = 384;
                            
                            // Convert NV12 to BGR Mat using OpenCV
                            std::vector<BYTE> contiguousNV12(copyWidth * copyHeight * 3 / 2);
                            BYTE* pDstY = contiguousNV12.data();
                            BYTE* pDstUV = pDstY + copyWidth * copyHeight;

                            for (int row = 0; row < copyHeight; ++row) {
                                memcpy(pDstY + row * copyWidth, pBuffer[0] + row * pStride[0], copyWidth);
                            }
                            if (pBuffer[1] && pStride[1] > 0) {
                                for (int row = 0; row < copyHeight / 2; ++row) {
                                    memcpy(pDstUV + row * copyWidth, pBuffer[1] + row * pStride[1], copyWidth);
                                }
                            }

                            cv::Mat nv12_mat(copyHeight * 3 / 2, copyWidth, CV_8UC1, contiguousNV12.data());
                            cv::Mat bgr_mat;
                            cv::cvtColor(nv12_mat, bgr_mat, cv::COLOR_YUV2BGR_NV12);

                            std::string headerText = "CH " + std::to_string(channelId + 1);
                            cv::putText(bgr_mat, headerText, cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 200), 2);
                            if (g_pMainwindow && g_pMainwindow->m_bShowOverlay) {
                                QString caption;
                                {
                                    std::lock_guard<std::mutex> caption_lock(g_pMainwindow->draw_mtx);
                                    caption = g_pMainwindow->captions.value(channelId);
                                }
                                drawCaption(bgr_mat, caption);
                            }

                            // Convert cv::Mat to QImage
                            QImage qimg = cvMatToQImage(bgr_mat);
                            QPointer<QLabel> safeLabel = m_pLabel;
                            std::shared_ptr<std::atomic<bool>> pending = m_pPendingUpdate;

                            // Update GUI QLabel asynchronously
                            QMetaObject::invokeMethod(m_pLabel, [safeLabel, qimg, pending]() {
                                if (safeLabel) {
                                    safeLabel->setPixmap(QPixmap::fromImage(qimg));
                                }
                                if (pending) {
                                    pending->store(false);
                                }
                            }, Qt::QueuedConnection);
                        }
                        qcap2_rcbuffer_unlock_data(pScaledBuffer);
                    }
                }
            }
        }
        qcap2_rcbuffer_release(pScaledBuffer);
    }

        qcap2_rcbuffer_release(pRCBuffer_vdec);

        {
            QMutexLocker locker(&m_mutex);
            m_frameCount++;
        }
    }

    return QCAP_RT_OK;
}


// ── MainWindow Implementation ───────────────────────────────────────────────
MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), m_bFullscreen(false),
      // AI init
      m_bShowOverlay(true), m_bHalfRefreshRate(false),
      ai_running(false), active_camera_count(0)
{
    setWindowTitle("QCAP Multichannel RTSP + QDEEP VLM Caption");
    resize(1280, 720);

    g_pMainwindow = this;

    // ── Initialize AI members ────────────────────────────────────────────
    captions.resize(MAX_CHANNELS);
    for (int channelId = 0; channelId < MAX_CHANNELS; ++channelId) {
        channelHandle[channelId] = nullptr;
        channelModelVersion[channelId] = 0.0f;
        channelAiThread[channelId] = nullptr;
    }
    color_space.resize(MAX_CHANNELS);
    width_vec.resize(MAX_CHANNELS);
    height_vec.resize(MAX_CHANNELS);
    buffer_vec.assign(MAX_CHANNELS, nullptr);
    buffer_len_vec.resize(MAX_CHANNELS);
    box_size_vec.assign(MAX_CHANNELS, BOX_SIZE);
    box_list_vec.assign(MAX_CHANNELS, nullptr);

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

    chkEnableQdeepInference = new QCheckBox("Enable QDEEP Inference", grpConfig);
    chkEnableQdeepInference->setChecked(true);
    grpLayout->addWidget(chkEnableQdeepInference);

    chkShowOverlay = new QCheckBox("Show AI Captions", grpConfig);
    chkShowOverlay->setChecked(true);
    grpLayout->addWidget(chkShowOverlay);

    chkHalfRefreshRate = new QCheckBox("Half Display Refresh Rate", grpConfig);
    chkHalfRefreshRate->setChecked(false);
    grpLayout->addWidget(chkHalfRefreshRate);

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
    connect(chkEnableQdeepInference, &QCheckBox::toggled, this, &MainWindow::onQdeepInferenceToggled);
    connect(chkShowOverlay, &QCheckBox::toggled, this, &MainWindow::onOverlayToggled);
    connect(chkHalfRefreshRate, &QCheckBox::toggled, this, &MainWindow::onHalfRefreshRateToggled);

    videoContainer->installEventFilter(this);

    m_timerId = startTimer(1000);

    // ── Initialize QDEEP models ─────────────────────────────────────────
    init_models();
}

MainWindow::~MainWindow()
{
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
            tableUrls->setItem(i, 1, new QTableWidgetItem("rtsp://root:root@192.168.180.67:554/session0.mpg"));
        }
    }
}

void MainWindow::onBtnStartClicked()
{
    stopAllChannels();
    clearGrid();

    int count = spinChannelCount->value();
    int cols = 2;
    if (m_bFullscreen) {
        if (count >= 17) {
            cols = 8;
        } else {
            cols = 4;
        }
    } else {
        if (count <= 2) cols = count;
        else if (count <= 4) cols = 2;
        else if (count <= 9) cols = 3;
        else cols = 4;
    }

    for (int i = 0; i < count; ++i) {
        QFrame *frame = new QFrame(videoContainer);
        frame->setFrameShape(QFrame::Box);
        frame->setLineWidth(1);
        frame->setStyleSheet("background-color: black; border: 1px solid #333333;");
        frame->installEventFilter(this);

        QVBoxLayout *layout = new QVBoxLayout(frame);
        layout->setContentsMargins(0, 0, 0, 0);
        QLabel *label = new QLabel(frame);
        label->setAlignment(Qt::AlignCenter);
        label->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
        label->setScaledContents(true);
        layout->addWidget(label);

        videoFrames.append(frame);

        int row = i / cols;
        int col = i % cols;
        videoGridLayout->addWidget(frame, row, col);
        frame->show();

        QString url = tableUrls->item(i, 1)->text().trimmed();
        ChannelContext *ctx = new ChannelContext(i, url, label);
        ctx->setDisplayEnabled(m_bEnableDisplay);
        channels.append(ctx);

        ctx->start();
    }

    spinChannelCount->setEnabled(false);
    tableUrls->setEnabled(false);
    btnStart->setEnabled(false);
    btnStop->setEnabled(true);
    lblStatus->setText("Status: Running");

    // Start AI inference only when the UI option is enabled.
    if (chkEnableQdeepInference->isChecked()) {
        yolo_start();
    }
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

        // Adjust Grid layout
        int count = videoFrames.size();
        if (count > 0) {
            int cols = 2;
            if (m_bFullscreen) {
                if (count >= 17) {
                    cols = 8;
                } else {
                    cols = 4;
                }
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
    for (auto* c : channels) {
        c->setDisplayEnabled(checked);
    }
}

void MainWindow::onQdeepInferenceToggled(bool checked)
{
    if (checked) {
        // Streams may not be running yet; yolo_start() safely does nothing in that case.
        yolo_start();
    } else {
        // Join the inference thread before dropping queued buffers, so no queued or stale
        // frame can be submitted after the option is turned off.
        yolo_stop();
    }
}

void MainWindow::onOverlayToggled(bool checked)
{
    m_bShowOverlay = checked;
}

void MainWindow::onHalfRefreshRateToggled(bool checked)
{
    m_bHalfRefreshRate = checked;
}

// ── QDEEP / YOLO AI Functions ──────────────────────────────────────────────
void MainWindow::init_models()
{
    for (int i = 0; i < MAX_CHANNELS; ++i) {
        box_list_vec[i] = new QDEEP_API::QDEEP_OBJECT_DETECT_BOUNDING_BOX[BOX_SIZE]();
        buffer_vec[i] = new BYTE[MAX_BUFFER_SIZE]();
        color_space[i] = QDEEP_API::QDEEP_COLORSPACE_TYPE_NV12;
        width_vec[i] = 640;
        height_vec[i] = 384;
        buffer_len_vec[i] = 0;
    }

    // QRESULT res = QDEEP_GET_OBJECT_DETECT_RESERVED_STATUS(reinterpret_cast<PVOID>(0xD7CBB416), reinterpret_cast<ULONG*>(0x3B98119E));
     QRESULT res = QDEEP_GET_OBJECT_DETECT_RESERVED_STATUS(reinterpret_cast<PVOID>(23921233), reinterpret_cast<ULONG*>(23921233));
    qDebug() << "[AI Log] QDEEP_GET_OBJECT_DETECT_RESERVED_STATUS res:" << QString("0x%1").arg(res, 8, 16, QChar('0'));
}

void MainWindow::uninit_models()
{
    yolo_stop();
    for (int i = 0; i < MAX_CHANNELS; ++i) {
        delete[] box_list_vec[i];
        box_list_vec[i] = nullptr;
        delete[] buffer_vec[i];
        buffer_vec[i] = nullptr;
    }
}

bool MainWindow::createCaptionHandle(int channelId)
{
    if (channelId < 0 || channelId >= MAX_CHANNELS) {
        return false;
    }
    if (channelHandle[channelId] != nullptr) {
        return true;
    }

    char modelPath[] = "/home/nvidia/qdeep/vlm/genai_caption_thor/genai_caption/model/QDEEP.OD.TRAFFIC.CFG";
    channelModelVersion[channelId] = 0.0f;
    QRESULT res = QDEEP_API::QDEEP_CREATE_OBJECT_DETECT_EX(
        QDEEP_API::QDEEP_GPU_TYPE_NVIDIA, 0,
        QDEEP_API::QDEEP_OBJECT_DETECT_CONFIG_MODEL_PHOTO_CAPTION,
        modelPath, &channelHandle[channelId], &channelModelVersion[channelId], 0);
    if (res == QCAP_RS_SUCCESSFUL && channelHandle[channelId] != nullptr) {
        res = QDEEP_API::QDEEP_START_OBJECT_DETECT(channelHandle[channelId]);
    }
    if (res == QCAP_RS_SUCCESSFUL) {
        char prompt[] = "Concisely describe the image content in fifteen words.";
        res = QDEEP_API::QDEEP_SET_OBJECT_DETECT_CUSTOM_SEARCH_TEXT(channelHandle[channelId], prompt);
    }
    if (res != QCAP_RS_SUCCESSFUL || channelHandle[channelId] == nullptr) {
        qCritical() << "[AI Caption] Failed to initialize CH" << channelId + 1 << "qres=" << res;
        if (channelHandle[channelId]) {
            QDEEP_API::QDEEP_STOP_OBJECT_DETECT(channelHandle[channelId]);
            QDEEP_API::QDEEP_DESTROY_OBJECT_DETECT(channelHandle[channelId]);
            channelHandle[channelId] = nullptr;
        }
        return false;
    }

    qDebug() << "[AI Caption] Initialized handle for CH" << channelId + 1;
    return true;
}

void MainWindow::destroyCaptionHandles()
{
    for (int channelId = 0; channelId < MAX_CHANNELS; ++channelId) {
        if (channelHandle[channelId]) {
            QDEEP_API::QDEEP_STOP_OBJECT_DETECT(channelHandle[channelId]);
            QDEEP_API::QDEEP_DESTROY_OBJECT_DETECT(channelHandle[channelId]);
            channelHandle[channelId] = nullptr;
            channelModelVersion[channelId] = 0.0f;
        }
    }
}

void MainWindow::yolo_start()
{
    if (ai_running) return;

    active_camera_count = 0;
    for (ChannelContext *ctx : channels) {
        if (ctx->pClient != nullptr && createCaptionHandle(ctx->channelId)) {
            ctx->m_bSendBuffer = true;
            ctx->m_lastProcessTime = 0.0;
            ctx->m_bFrameReady = false;
            active_camera_count++;
        }
    }

    if (active_camera_count == 0) {
        qDebug() << "[Warning] No active cameras found!";
        return;
    }

    ai_running = true;
    for (ChannelContext *ctx : channels) {
        const int channelId = ctx->channelId;
        if (ctx->m_bSendBuffer && channelId >= 0 && channelId < MAX_CHANNELS) {
            channelAiThread[channelId] = new std::thread(&MainWindow::ai_inference_thread, this, channelId);
        }
    }
}

void MainWindow::yolo_stop()
{
    ai_running = false;

    for (ChannelContext *ctx : channels) {
        ctx->m_bSendBuffer = false;
    }

    for (int channelId = 0; channelId < MAX_CHANNELS; ++channelId) {
        if (channelAiThread[channelId]) {
            if (channelAiThread[channelId]->joinable()) {
                channelAiThread[channelId]->join();
            }
            delete channelAiThread[channelId];
            channelAiThread[channelId] = nullptr;
        }
    }

    destroyCaptionHandles();

    discardQueuedAIFrames();

    {
        std::lock_guard<std::mutex> draw_lock(draw_mtx);
        for (QString& caption : captions) {
            caption.clear();
        }
    }
}

void MainWindow::discardQueuedAIFrames()
{
    for (ChannelContext *ctx : channels) {
        if (!ctx->m_pAIQueue) {
            continue;
        }

        qcap2_rcbuffer_t* pBuffer = nullptr;
        while (qcap2_rcbuffer_queue_pop(ctx->m_pAIQueue, &pBuffer) == QCAP_RS_SUCCESSFUL && pBuffer) {
            qcap2_rcbuffer_release(pBuffer);
            pBuffer = nullptr;
        }
    }
}

void MainWindow::ai_inference_thread(int channelId)
{
    ChannelContext* ctx = nullptr;
    for (ChannelContext* candidate : channels) {
        if (candidate->channelId == channelId) {
            ctx = candidate;
            break;
        }
    }
    if (!ctx || channelId < 0 || channelId >= MAX_CHANNELS) {
        qWarning() << "[AI Caption] Invalid channel thread" << channelId + 1;
        return;
    }

    int inferenceCount = 0;
    double apiTotalMs = 0.0;
    double apiMinMs = 0.0;
    double apiMaxMs = 0.0;

    while (ai_running) {
        if (!ctx->m_bSendBuffer || !ctx->m_pAIQueue || !channelHandle[channelId]) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        // This thread only drains and processes its own channel queue.
        qcap2_rcbuffer_t* pLatest = nullptr;
        qcap2_rcbuffer_t* pBuf = nullptr;
        while (qcap2_rcbuffer_queue_pop(ctx->m_pAIQueue, &pBuf) == QCAP_RS_SUCCESSFUL && pBuf) {
            if (pLatest) qcap2_rcbuffer_release(pLatest);
            pLatest = pBuf;
        }
        if (!pLatest) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        bool frameCopied = false;
        PVOID pLockedData = qcap2_rcbuffer_lock_data(pLatest);
        if (pLockedData) {
            qcap2_av_frame_t* pAVFrame = reinterpret_cast<qcap2_av_frame_t*>(pLockedData);
            uint8_t* pBuffer[4] = {nullptr};
            int pStride[4] = {0};
            qcap2_av_frame_get_buffer1(pAVFrame, pBuffer, pStride);

            const ULONG copyWidth = 640;
            const ULONG copyHeight = 384;
            BYTE* pDstBuf = buffer_vec[channelId];
            if (pDstBuf && pBuffer[0] && pStride[0] >= static_cast<int>(copyWidth) &&
                pBuffer[1] && pStride[1] >= static_cast<int>(copyWidth)) {
                for (ULONG row = 0; row < copyHeight; ++row) {
                    memcpy(pDstBuf + row * copyWidth, pBuffer[0] + row * pStride[0], copyWidth);
                }
                BYTE* pDstUV = pDstBuf + (copyWidth * copyHeight);
                for (ULONG row = 0; row < copyHeight / 2; ++row) {
                    memcpy(pDstUV + row * copyWidth, pBuffer[1] + row * pStride[1], copyWidth);
                }
                width_vec[channelId] = copyWidth;
                height_vec[channelId] = copyHeight;
                buffer_len_vec[channelId] = (copyWidth * copyHeight * 3) / 2;
                frameCopied = true;
            }
            qcap2_rcbuffer_unlock_data(pLatest);
        }
        qcap2_rcbuffer_release(pLatest);

        if (!frameCopied) {
            continue;
        }

        box_size_vec[channelId] = BOX_SIZE;
        const auto inferenceStart = std::chrono::steady_clock::now();
        const QRESULT apiRes = QDEEP_API::QDEEP_SET_VIDEO_OBJECT_DETECT_UNCOMPRESSION_BUFFER(
            channelHandle[channelId], color_space[channelId], width_vec[channelId], height_vec[channelId],
            buffer_vec[channelId], buffer_len_vec[channelId], box_list_vec[channelId], &box_size_vec[channelId]);
        const auto inferenceEnd = std::chrono::steady_clock::now();
        const double apiMs = std::chrono::duration<double, std::milli>(inferenceEnd - inferenceStart).count();

        inferenceCount++;
        apiTotalMs += apiMs;
        if (inferenceCount == 1 || apiMs < apiMinMs) apiMinMs = apiMs;
        if (apiMs > apiMaxMs) apiMaxMs = apiMs;

        QString caption;
        if (apiRes == QCAP_RS_SUCCESSFUL && box_size_vec[channelId] > 0) {
            caption = captionFromResult(box_list_vec[channelId][0]);
        } else {
            caption = QString("Caption error (0x%1)").arg(apiRes, 8, 16, QChar('0'));
        }
        {
            std::lock_guard<std::mutex> captionLock(draw_mtx);
            captions[channelId] = caption;
        }

        if (inferenceCount >= 100) {
            qDebug() << QString("[AI QDEEP caption] CH%1 average API time for 100 frames: %2 ms (min/max=%3/%4)")
                        .arg(channelId + 1)
                        .arg(apiTotalMs / inferenceCount, 0, 'f', 2)
                        .arg(apiMinMs, 0, 'f', 2)
                        .arg(apiMaxMs, 0, 'f', 2);
            inferenceCount = 0;
            apiTotalMs = 0.0;
            apiMinMs = 0.0;
            apiMaxMs = 0.0;
        }
    }
}
