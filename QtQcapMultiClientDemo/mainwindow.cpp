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

// Static callback functions delegating to ChannelContext
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

// ---------------------------------------------------------------------
// ChannelContext Implementation
// ---------------------------------------------------------------------
ChannelContext::ChannelContext(int id, const QString& streamUrl, uintptr_t winId)
    : channelId(id), url(streamUrl), m_winId(winId),
      pClient(nullptr), pVdec(nullptr), pEventHandlers(nullptr),
      pEvent_vdec(nullptr), pVideoSink(nullptr), pScaler(nullptr),
      m_nVideoWidth(0), m_nVideoHeight(0), m_dVideoFrameRate(0.0), m_nVideoEncoderFormat(0),
      m_frameCount(0), m_bDisplayEnabled(true),
      m_pushFrameCount(0), m_decFrameCount(0)
{
}

ChannelContext::~ChannelContext() {
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
    qcap2_video_scaler_t* pLocalScaler = nullptr;
    qcap2_video_decoder_t* pLocalVdec = nullptr;
    qcap2_video_sink_t* pLocalVideoSink = nullptr;

    {
        QMutexLocker locker(&m_mutex);
        pLocalClient = pClient; 
        pLocalEventHandlers = pEventHandlers;
        pLocalScaler = pScaler;
        pLocalVdec = pVdec;
        pLocalVideoSink = pVideoSink;

        // Clear all handles to prevent any new callbacks from processing
        pClient = nullptr; 
        pEventHandlers = nullptr;
        pScaler = nullptr;
        pVdec = nullptr;
        pVideoSink = nullptr;
    }
    
    qDebug() << "========== CH" << channelId << "Stop Sequence Started ==========";
    
    // 1. 停止 Broadcast Client (自前端切斷資料源，阻止新資料進入，並讓剩餘資料自然向下游流動排空)
    qDebug() << "[Stop] 1. Stopping Broadcast Client... (CH" << channelId << ")";
    if (pLocalClient) {
        QCAP_STOP_BROADCAST_CLIENT(pLocalClient);
    }

    // 2. 停止解碼器 (此時已無新資料輸入，且背景 Event Handlers 仍處於活動狀態，將解碼器內剩餘畫面 Pop 完畢)
    qDebug() << "[Stop] 2. Stopping Decoder... (CH" << channelId << ")";
    if (pLocalVdec) {
        qcap2_video_decoder_stop(pLocalVdec);
    }

    // 3. 停止事件與 Event Handlers (此時解碼器已關閉，可以安全地加入並停止背景線程)
    qDebug() << "[Stop] 3. Stopping Event & Event Handlers... (CH" << channelId << ")";
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

    // 4. 停止 Video Sink (不呼叫 set_native_handle(0)，直接 stop，讓 Sink 乾淨釋放佇列中的所有 Buffer)
    qDebug() << "[Stop] 4. Stopping Video Sink... (CH" << channelId << ")";
    if (pLocalVideoSink) {
        qcap2_video_sink_stop(pLocalVideoSink);
    }

    // 5. 停止 Video Scaler
    qDebug() << "[Stop] 5. Stopping Video Scaler... (CH" << channelId << ")";
    if (pLocalScaler) {
        qcap2_video_scaler_stop(pLocalScaler);
    }

    // 6. 銷毀各組件資源
    qDebug() << "[Stop] 6. Deleting resources... (CH" << channelId << ")";
    if (pLocalVideoSink) {
        qcap2_video_sink_delete(pLocalVideoSink);
        pLocalVideoSink = nullptr;
    }
    if (pLocalScaler) {
        qcap2_video_scaler_delete(pLocalScaler);
        pLocalScaler = nullptr;
    }
    if (pLocalVdec) {
        qcap2_video_decoder_delete(pLocalVdec);
        pLocalVdec = nullptr;
    }
    if (pLocalEventHandlers) {
        qcap2_event_handlers_delete(pLocalEventHandlers);
        pLocalEventHandlers = nullptr;
    }
    if (pEvent_vdec) {
        qcap2_event_stop(pEvent_vdec);
        qcap2_event_delete(pEvent_vdec);
        pEvent_vdec = nullptr;
    }
    if (pLocalClient) {
        QCAP_DESTROY_BROADCAST_CLIENT(pLocalClient);
        pLocalClient = nullptr;
    }
    
    qDebug() << "========== CH" << channelId << "Stop Sequence Finished ==========";
}

void ChannelContext::setDisplayEnabled(bool enabled) {
    // 加上 Mutex 保護，因為這個函式會由 Qt UI 主執行緒呼叫，
    // 而 onEventVdec (背景執行緒) 同時會讀取這個變數來決定要不要畫圖
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
    
    // Avoid unreasonable extreme values of width and height
    if (nVideoWidth == 0 || nVideoHeight == 0 || nVideoWidth > 8192 || nVideoHeight > 8192) {
        qCritical() << "CH" << channelId << "Connected with unreasonable dimensions:" << nVideoWidth << "x" << nVideoHeight << ". Aborting decoder and sink startup.";
        m_statusInfo = QString("Aborted (unreasonable dimensions: %1x%2)").arg(nVideoWidth).arg(nVideoHeight);
        return QCAP_RT_OK;
    }
    
    m_nVideoWidth = nVideoWidth;
    m_nVideoHeight = nVideoHeight;
    m_dVideoFrameRate = dVideoFrameRate;
    m_nVideoEncoderFormat = nVideoEncoderFormat;
    
    QString formatStr;
    switch (nVideoEncoderFormat) {
        case QCAP_ENCODER_FORMAT_H264:
            formatStr = "H.264";
            break;
        case QCAP_ENCODER_FORMAT_H265:
            formatStr = "H.265";
            break;
        case QCAP_ENCODER_FORMAT_AV1:
            formatStr = "AV1";
            break;
        case QCAP_ENCODER_FORMAT_MPEG2:
            formatStr = "MPEG2";
            break;
        case QCAP_ENCODER_FORMAT_RAW:
            formatStr = "RAW";
            break;
        default:
            formatStr = QString("Unknown (%1)").arg(nVideoEncoderFormat);
            break;
    }
    
    m_statusInfo = QString("%1x%2 @%3fps (%4)")
                   .arg(nVideoWidth).arg(nVideoHeight).arg(dVideoFrameRate).arg(formatStr);
    
    qDebug() << "CH" << channelId << "Connected info:" << m_statusInfo;
    
    // Initialize Event Handlers for decoder popping
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
    
    // Setup decoder using NVIDIA hardware acceleration with NV12 format
    qcap2_video_encoder_property_set_property1(pProp, 
        0,                                 // GPU num
        QCAP_ENCODER_TYPE_SOFTWARE,        // Encoder type (Software decoder workaround)
        nVideoEncoderFormat,               // Format (H264 / H265)
        QCAP_COLORSPACE_TYPE_NV12,         // Colorspace (NV12)
        nVideoWidth, nVideoHeight, dVideoFrameRate,
        QCAP_RECORD_PROFILE_MAIN, QCAP_RECORD_LEVEL_51, QCAP_RECORD_ENTROPY_CABAC, QCAP_RECORD_COMPLEXITY_0, QCAP_RECORD_MODE_CBR, 
        8000, 40000000, 60, 0, FALSE, 0, 0, 0, FALSE, FALSE, FALSE, 0, 0, 0, 0, 0, 0);
    qcap2_video_encoder_property_set_high_perf(pProp, TRUE);
    qcap2_video_decoder_set_video_property(pVdec, pProp);
    qcap2_video_encoder_property_delete(pProp);
    
    qcap2_video_decoder_set_event(pVdec, pEvent_vdec);
    qcap2_video_decoder_set_multithread(pVdec, false);
    qcap2_video_decoder_set_packet_count(pVdec, 16);
    qcap2_video_decoder_set_frame_count(pVdec, 10);
    
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
    
    // Initialize Video Sink (GSTREAMER or DAVMF backend rendering on the Qt frame Window ID)
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
    
    int backendType = QCAP2_VIDEO_SINK_BACKEND_TYPE_DAVMF;
    if (getenv("QCAP_VO_USE_GSTREAMER") && QString(getenv("QCAP_VO_USE_GSTREAMER")) == "1") {
        backendType = QCAP2_VIDEO_SINK_BACKEND_TYPE_GSTREAMER;
    }
    qcap2_video_sink_set_backend_type(pVideoSink, backendType);
    qcap2_video_sink_set_native_handle(pVideoSink, m_winId);
    
    // Set display enabled/disabled state correctly on initialization
    if (m_bDisplayEnabled) {
        qcap2_video_sink_set_nvbuf(pVideoSink, false); // Sink consumes system memory frames
    }
    
    qcap2_video_format_t* pFormat = qcap2_video_format_new();
    if (pFormat) {
        qcap2_video_format_set_property(pFormat, QCAP_COLORSPACE_TYPE_NV12, nVideoWidth, nVideoHeight, bVideoIsInterleaved, dVideoFrameRate);
        qcap2_video_sink_set_video_format(pVideoSink, pFormat);
        qcap2_video_format_delete(pFormat);
    }
    
    // Start video sink only if display is enabled
    if (m_bDisplayEnabled) {
        qres = qcap2_video_sink_start(pVideoSink);
        if (qres != QCAP_RS_SUCCESSFUL) {
            qCritical() << "qcap2_video_sink_start failed for CH" << channelId << "qres =" << qres;
        }
    }
    
    // Initialize Video Scaler to copy nvbuf to system memory
    pScaler = qcap2_video_scaler_new();
    if (!pScaler) {
        qCritical() << "CH" << channelId << "Failed to create video scaler.";
        m_statusInfo = "Error: Failed to create video scaler";
        qcap2_video_sink_stop(pVideoSink);
        qcap2_video_sink_delete(pVideoSink);
        pVideoSink = nullptr;
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
    
    // Set default backend for scaler
    qcap2_video_scaler_set_backend_type(pScaler, QCAP2_VIDEO_SCALER_BACKEND_TYPE_DEFAULT);
    
    // Set scaler output format (NV12 format matching the input width and height)
    qcap2_video_format_t* pScalerFormat = qcap2_video_format_new();
    if (pScalerFormat) {
        qcap2_video_format_set_property(pScalerFormat, QCAP_COLORSPACE_TYPE_NV12, nVideoWidth, nVideoHeight, bVideoIsInterleaved, dVideoFrameRate);
        qcap2_video_scaler_set_video_format(pScaler, pScalerFormat);
        qcap2_video_format_delete(pScalerFormat);
    }
    
    qcap2_video_scaler_set_frame_count(pScaler, 8);
    
    // Hint that source is NVBuf (hardware memory) and dest is system memory
    qcap2_video_scaler_set_src_buffer_hint(pScaler, QCAP2_BUFFER_HINT_DEFAULT);
    qcap2_video_scaler_set_dst_buffer_hint(pScaler, QCAP2_BUFFER_HINT_DEFAULT);
    qcap2_video_scaler_set_auto_run(pScaler, true);
    
    qres = qcap2_video_scaler_start(pScaler);
    if (qres != QCAP_RS_SUCCESSFUL) {
        qCritical() << "qcap2_video_scaler_start failed for CH" << channelId << "qres =" << qres;
    }
    
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
    if (!pRCBuffer) return QCAP_RT_OK;
    
    QRESULT qres = qcap2_video_decoder_push(pLocalVdec, pRCBuffer);
    if (qres != QCAP_RS_SUCCESSFUL) {
        qCritical() << "qcap2_video_decoder_push failed for CH" << channelId << "qres =" << qres;
    }
    
    // Calculate INPUT FPS per channel
    if (!m_pushTimer.isValid()) {
        m_pushTimer.start();
    }
    
    m_pushFrameCount++;
    
    if (m_pushTimer.elapsed() >= 2000) {
        double inputFps = double(m_pushFrameCount) * 1000.0 / double(m_pushTimer.elapsed());
        // qDebug() << "CH" << channelId << "[INPUT FPS]" << inputFps << "fps (Received:" << m_pushFrameCount << "packets)";
        
        m_pushFrameCount = 0;
        m_pushTimer.restart();
    }
    
    return QCAP_RT_OK;
}

QRETURN ChannelContext::onEventVdec() {
    qcap2_video_decoder_t* pLocalVdec = nullptr;
    qcap2_video_scaler_t* pLocalScaler = nullptr;
    qcap2_video_sink_t* pLocalVideoSink = nullptr;
    bool bDisplayEnabled = false;
    
    {
        QMutexLocker locker(&m_mutex);
        if (!pVdec) {
            return QCAP_RT_OK;
        }
        pLocalVdec = pVdec;
        pLocalScaler = pScaler;
        pLocalVideoSink = pVideoSink;
        bDisplayEnabled = m_bDisplayEnabled;
    }
    
    qcap2_rcbuffer_t* pRCBuffer_vdec = nullptr;
    QRESULT qres = qcap2_video_decoder_pop(pLocalVdec, &pRCBuffer_vdec);
    if (qres != QCAP_RS_SUCCESSFUL) {
        return QCAP_RT_OK;
    }
    
    // Calculate PURE DEC FPS per channel (instance-specific, non-static)
    if (!m_fpsTimer.isValid()) {
        m_fpsTimer.start();
    }
    
    m_decFrameCount++;
    
    if (m_fpsTimer.elapsed() >= 2000) {
        double fps = double(m_decFrameCount) * 1000.0 / double(m_fpsTimer.elapsed());
        qDebug() << "CH" << channelId << "[PURE DEC FPS]" << fps << "fps (Decoded:" << m_decFrameCount << "frames)";
        
        m_decFrameCount = 0;
        m_fpsTimer.restart();
    }
    
    // Check if we have an active video scaler
    bool bScalerActive = false;
    {
        QMutexLocker locker(&m_mutex);
        if (pScaler) {
            bScalerActive = true;
        }
    }
    
    if (bScalerActive && pLocalScaler) {
        // Push the decoded nvbuf to the scaler
        qres = qcap2_video_scaler_push(pLocalScaler, pRCBuffer_vdec);
        // Release the decoded buffer reference since scaler now owns it (or has its own reference)
        qcap2_rcbuffer_release(pRCBuffer_vdec);
        
        if (qres != QCAP_RS_SUCCESSFUL) {
            qWarning() << "CH" << channelId << "qcap2_video_scaler_push failed, qres =" << qres;
            return QCAP_RT_OK;
        }
        
        // Pop the system memory buffer from the scaler
        qcap2_rcbuffer_t* pSysBuffer = nullptr;
        qres = qcap2_video_scaler_pop(pLocalScaler, &pSysBuffer);
        if (qres != QCAP_RS_SUCCESSFUL || !pSysBuffer) {
            qWarning() << "CH" << channelId << "qcap2_video_scaler_pop failed, qres =" << qres;
            return QCAP_RT_OK;
        }
        
        // Push to Video Sink only if display rendering is enabled
        bool bSinkActive = false;
        {
            QMutexLocker locker(&m_mutex);
            if (pVideoSink) {
                bSinkActive = true;
            }
        }
        if (bDisplayEnabled && bSinkActive && pLocalVideoSink) {
            qcap2_video_sink_push(pLocalVideoSink, pSysBuffer);
        }
        
        // Release the popped scaler buffer reference
        qcap2_rcbuffer_release(pSysBuffer);
    } else {
        // Fallback: If scaler is not ready, just release the decoder frame
        qcap2_rcbuffer_release(pRCBuffer_vdec);
    }
    
    {
        QMutexLocker locker(&m_mutex);
        m_frameCount++;
    }
    
    return QCAP_RT_OK;
}



// ---------------------------------------------------------------------
// MainWindow Implementation
// ---------------------------------------------------------------------
MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), m_bFullscreen(false)
{
    setWindowTitle("QCAP Multichannel RTSP Client");
    resize(1280, 720);
    
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
    
    grpLayout->addWidget(new QLabel("Channel Count (1-16):"));
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
    
    // Set default URLs in table
    onChannelCountChanged(4);
    m_bEnableDisplay = true;
    
    // Signal Slot connections
    connect(spinChannelCount, QOverload<int>::of(&QSpinBox::valueChanged), this, &MainWindow::onChannelCountChanged);
    connect(btnStart, &QPushButton::clicked, this, &MainWindow::onBtnStartClicked);
    connect(btnStop, &QPushButton::clicked, this, &MainWindow::onBtnStopClicked);
    connect(chkEnableDisplay, &QCheckBox::toggled, this, &MainWindow::onDisplayToggled);
    
    // Event filter to handle double click on video frames
    videoContainer->installEventFilter(this);
    
    // Timer to update stats every 1 second
    m_timerId = startTimer(1000);
}

MainWindow::~MainWindow()
{
    stopAllChannels();
}

void MainWindow::onChannelCountChanged(int count)
{
    tableUrls->setRowCount(count);
    for (int i = 0; i < count; ++i) {
        // Column 0: Channel label
        QTableWidgetItem *itemCh = tableUrls->item(i, 0);
        if (!itemCh) {
            itemCh = new QTableWidgetItem(QString("CH%1").arg(i + 1));
            itemCh->setFlags(itemCh->flags() & ~Qt::ItemIsEditable);
            itemCh->setTextAlignment(Qt::AlignCenter);
            tableUrls->setItem(i, 0, itemCh);
        }
        
        // Column 1: URL value
        QTableWidgetItem *itemUrl = tableUrls->item(i, 1);
        if (!itemUrl || itemUrl->text().isEmpty()) {
            // Set default local/test RTSP URL
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
    
    // Calculate grid rows and cols dynamically
    int cols = 2;
    if (count <= 2) cols = count;
    else if (count <= 4) cols = 2;
    else if (count <= 9) cols = 3;
    else cols = 4;
    
    // Instantiate display frames
    for (int i = 0; i < count; ++i) {
        QFrame *frame = new QFrame(videoContainer);
        frame->setFrameShape(QFrame::Box);
        frame->setLineWidth(1);
        frame->setStyleSheet("background-color: black; border: 1px solid #333333;");
        frame->installEventFilter(this); // Catch double clicks
        videoFrames.append(frame);
        
        int row = i / cols;
        int col = i % cols;
        videoGridLayout->addWidget(frame, row, col);
        
        // Wait, Qt widgets must be fully initialized to have valid winId()
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
}

void MainWindow::onBtnStopClicked()
{
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
        delete ctx;
    }
    channels.clear();
}

void MainWindow::clearGrid()
{
    // Remove widgets from grid
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
            ctx->m_frameCount = 0; // Reset frame count for next second
        }
        lblStatus->setText(statusText);
    }
}

void MainWindow::closeEvent(QCloseEvent *event)
{
    stopAllChannels();
    event->accept();
}

bool MainWindow::eventFilter(QObject *watched, QEvent *event)
{
    // Check if user double-clicked one of the video frames
    if (event->type() == QEvent::MouseButtonDblClick) {
        m_bFullscreen = !m_bFullscreen;
        if (m_bFullscreen) {
            controlPanel->hide();
            showFullScreen();
        } else {
            controlPanel->show();
            showNormal();
        }
        
        // Re-arrange the frames based on fullscreen or normal layout
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
