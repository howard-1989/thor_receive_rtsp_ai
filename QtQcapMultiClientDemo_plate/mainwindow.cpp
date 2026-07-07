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

// ── Overlay Widget (Bounding Boxes + Plate Text) ───────────────────────────
class OverlayWidget : public QWidget {
public:
    OverlayWidget(QWidget *parent = nullptr) : QWidget(parent) {
        setAttribute(Qt::WA_TransparentForMouseEvents);
        setAttribute(Qt::WA_TranslucentBackground);
        // 設為 child widget，隨 parent 自動移動/縮放
    }

protected:
    void paintEvent(QPaintEvent *event) override {
        if (!g_pMainwindow) return;
        if (!g_pMainwindow->m_bShowOverlay) return;

        QPainter painter(this);
        painter.setRenderHint(QPainter::Antialiasing);

        QPen boxPen(Qt::green, 2);
        QPen textPen(Qt::white, 1);
        QFont font = painter.font();
        font.setPointSize(10);
        font.setBold(true);
        painter.setFont(font);

        std::lock_guard<std::mutex> lock(g_pMainwindow->draw_mtx);

        for (int i = 0; i < MainWindow::MAX_CHANNELS; ++i) {
            ChannelContext* ctx = nullptr;
            for (auto* c : g_pMainwindow->channels) {
                if (c->channelId == i) { ctx = c; break; }
            }
            if (!ctx || !ctx->m_bSendBuffer || ctx->m_nVideoWidth == 0) continue;

            QWidget* w = nullptr;
            for (int fi = 0; fi < g_pMainwindow->videoFrames.size(); ++fi) {
                if (fi == i) { w = g_pMainwindow->videoFrames[fi]; break; }
            }
            if (!w || !w->isVisible()) continue;

            // w 和 overlay 都是 videoContainer 的子 widget，座標系統相同
            QPoint localPos = w->pos();

            double scale_x = (double)w->width() / (double)ctx->m_nVideoWidth;
            double scale_y = (double)w->height() / (double)ctx->m_nVideoHeight;

            for (const auto& box : g_pMainwindow->draw_boxes[i]) {
                int draw_x = localPos.x() + (int)(box.original_x * scale_x);
                int draw_y = localPos.y() + (int)(box.original_y * scale_y);
                int draw_w = (int)(box.original_w * scale_x);
                int draw_h = (int)(box.original_h * scale_y);

                painter.setPen(boxPen);
                painter.setBrush(Qt::NoBrush);
                painter.drawRect(draw_x, draw_y, draw_w, draw_h);

                painter.setPen(textPen);
                painter.setBrush(QColor(0, 0, 0, 150));
                QString label = QString("Ch%1 C:%2 [%3%]")
                    .arg(i + 1).arg(box.classId).arg((int)(box.probability * 100));

                QRect textRect = painter.fontMetrics().boundingRect(label);
                textRect.moveTo(draw_x, draw_y - textRect.height());
                painter.drawRect(textRect);
                painter.setPen(Qt::green);
                painter.drawText(textRect, Qt::AlignCenter, label);

                // Draw license plate text below the box (if present)
                if (!box.plateText.isEmpty()) {
                    QPen platePen(QColor(255, 220, 0), 1);
                    painter.setPen(platePen);
                    painter.setBrush(QColor(0, 0, 0, 180));
                    QString plateLabel = QString("[LP] %1").arg(box.plateText);
                    QRect plateRect = painter.fontMetrics().boundingRect(plateLabel);
                    plateRect.moveTo(draw_x, draw_y + draw_h + 2);
                    painter.drawRect(plateRect);
                    painter.setPen(QColor(255, 220, 0));
                    painter.drawText(plateRect, Qt::AlignCenter, plateLabel);
                }
            }
        }
    }
};

// ── Static callback functions ──────────────────────────────────────────────
static QRETURN on_connected_callback(
    PVOID  pClient, UINT iSessionNum, ULONG nVideoEncoderFormat,
    ULONG nVideoWidth, ULONG nVideoHeight, BOOL bVideoIsInterleaved,
    double dVideoFrameRate, ULONG nAudioEncoderFormat, ULONG nAudioChannels,
    ULONG nAudioBitsPerSample, ULONG nAudioSampleFrequency, PVOID pUserData)
{
    ChannelContext* ctx = static_cast<ChannelContext*>(pUserData);
    return ctx->onConnected(pClient, iSessionNum, nVideoEncoderFormat, nVideoWidth, nVideoHeight, bVideoIsInterleaved, dVideoFrameRate);
}

static QRETURN on_video_callback(PVOID pClient, UINT iSessionNum, double dSampleTime,
    BYTE * pStreamBuffer, ULONG nStreamBufferLen, BOOL bIsKeyFrame, PVOID pUserData)
{
    ChannelContext* ctx = static_cast<ChannelContext*>(pUserData);
    return ctx->onVideoCallback(dSampleTime, pStreamBuffer, nStreamBufferLen, bIsKeyFrame);
}

static QRETURN on_event_vdec_callback(PVOID pUserData) {
    ChannelContext* ctx = static_cast<ChannelContext*>(pUserData);
    return ctx->onEventVdec();
}

// ── ChannelContext Implementation ──────────────────────────────────────────
ChannelContext::ChannelContext(int id, const QString& streamUrl, uintptr_t winId)
    : channelId(id), url(streamUrl), m_winId(winId),
      pClient(nullptr), pVdec(nullptr), pEventHandlers(nullptr),
      pEvent_vdec(nullptr), pVideoSink(nullptr), pScaler(nullptr),
      m_nVideoWidth(0), m_nVideoHeight(0), m_dVideoFrameRate(0.0), m_nVideoEncoderFormat(0),
      m_frameCount(0), m_bDisplayEnabled(true),
      m_pushFrameCount(0), m_decFrameCount(0),
      m_bSendBuffer(false), m_lastProcessTime(0.0), m_bFrameReady(false),
      m_pAIBuffer(nullptr), m_nAIBufferLen(0), m_nAIWidth(0), m_nAIHeight(0)
{
}

ChannelContext::~ChannelContext() {
    if (m_pAIBuffer) { delete[] m_pAIBuffer; m_pAIBuffer = nullptr; }
    stop();
}

bool ChannelContext::start() {
    QMutexLocker locker(&m_mutex);
    qDebug() << "Starting channel" << channelId << "URL:" << url;

    QRESULT qres = QCAP_CREATE_BROADCAST_CLIENT(channelId, url.toLatin1().data(), &pClient, QCAP_DECODER_TYPE_ZZNVCODEC, nullptr);
    if (qres != QCAP_RS_SUCCESSFUL) {
        qCritical() << "QCAP_CREATE_BROADCAST_CLIENT failed for CH" << channelId << "qres =" << qres;
        return false;
    }

    BOOL bVideoDecode = FALSE;
    QCAP_SET_BROADCAST_CLIENT_CUSTOM_PROPERTY_EX(pClient, QCAP_BCPROP_VIDEO_DECODE, reinterpret_cast<BYTE*>(&bVideoDecode), sizeof(bVideoDecode));
    BOOL bAudioDecode = FALSE;
    QCAP_SET_BROADCAST_CLIENT_CUSTOM_PROPERTY_EX(pClient, QCAP_BCPROP_AUDIO_DECODE, reinterpret_cast<BYTE*>(&bAudioDecode), sizeof(bAudioDecode));

    QCAP_REGISTER_BROADCAST_CLIENT_CONNECTED_CALLBACK(pClient, on_connected_callback, this);
    QCAP_REGISTER_VIDEO_BROADCAST_CLIENT_CALLBACK(pClient, on_video_callback, this);

    qres = QCAP_START_BROADCAST_CLIENT(pClient, QCAP_BROADCAST_PROTOCOL_TCP, 10000, -1);
    if (qres != QCAP_RS_SUCCESSFUL) {
        qCritical() << "QCAP_START_BROADCAST_CLIENT failed for CH" << channelId;
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
        pClient = nullptr; pEventHandlers = nullptr; pScaler = nullptr; pVdec = nullptr; pVideoSink = nullptr;
    }

    if (pLocalClient) QCAP_STOP_BROADCAST_CLIENT(pLocalClient);
    if (pLocalVdec) qcap2_video_decoder_stop(pLocalVdec);

    uintptr_t nHandle_vdec = 0;
    if (pEvent_vdec) { qcap2_event_get_native_handle(pEvent_vdec, &nHandle_vdec); qcap2_event_stop(pEvent_vdec); }
    if (pLocalEventHandlers) {
        if (nHandle_vdec != 0) qcap2_event_handlers_remove_handler(pLocalEventHandlers, nHandle_vdec);
        qcap2_event_handlers_stop(pLocalEventHandlers);
    }
    if (pLocalVideoSink) qcap2_video_sink_stop(pLocalVideoSink);
    if (pLocalScaler) qcap2_video_scaler_stop(pLocalScaler);

    if (pLocalVideoSink) qcap2_video_sink_delete(pLocalVideoSink);
    if (pLocalScaler) qcap2_video_scaler_delete(pLocalScaler);
    if (pLocalVdec) qcap2_video_decoder_delete(pLocalVdec);
    if (pLocalEventHandlers) qcap2_event_handlers_delete(pLocalEventHandlers);
    if (pEvent_vdec) { qcap2_event_stop(pEvent_vdec); qcap2_event_delete(pEvent_vdec); pEvent_vdec = nullptr; }
    if (pLocalClient) QCAP_DESTROY_BROADCAST_CLIENT(pLocalClient);
}

void ChannelContext::setDisplayEnabled(bool enabled) {
    QMutexLocker locker(&m_mutex);
    m_bDisplayEnabled = enabled;
}

QRETURN ChannelContext::onConnected(
    PVOID pClient, UINT iSessionNum, ULONG nVideoEncoderFormat,
    ULONG nVideoWidth, ULONG nVideoHeight, BOOL bVideoIsInterleaved, double dVideoFrameRate)
{
    Q_UNUSED(pClient); Q_UNUSED(iSessionNum);
    QMutexLocker locker(&m_mutex);

    if (nVideoWidth == 0 || nVideoHeight == 0 || nVideoWidth > 8192 || nVideoHeight > 8192) {
        qCritical() << "CH" << channelId << "Unreasonable dimensions:" << nVideoWidth << "x" << nVideoHeight;
        return QCAP_RT_OK;
    }

    m_nVideoWidth = nVideoWidth; m_nVideoHeight = nVideoHeight;
    m_dVideoFrameRate = dVideoFrameRate; m_nVideoEncoderFormat = nVideoEncoderFormat;
    m_statusInfo = QString("%1x%2 @%3fps").arg(nVideoWidth).arg(nVideoHeight).arg(dVideoFrameRate);

    m_nAIWidth = nVideoWidth; m_nAIHeight = nVideoHeight;
    m_nAIBufferLen = (nVideoWidth * nVideoHeight * 3) / 2;
    if (m_pAIBuffer) delete[] m_pAIBuffer;
    m_pAIBuffer = new BYTE[m_nAIBufferLen]();

    pEventHandlers = qcap2_event_handlers_new();
    qcap2_event_handlers_start(pEventHandlers);
    pEvent_vdec = qcap2_event_new();
    qcap2_event_start(pEvent_vdec);
    uintptr_t nHandle_vdec = 0;
    qcap2_event_get_native_handle(pEvent_vdec, &nHandle_vdec);
    qcap2_event_handlers_add_handler(pEventHandlers, nHandle_vdec, on_event_vdec_callback, this);

    pVdec = qcap2_video_decoder_new();
    qcap2_video_encoder_property_t* pProp = qcap2_video_encoder_property_new();
    qcap2_video_encoder_property_set_property1(pProp, 0, QCAP_ENCODER_TYPE_SOFTWARE,
        nVideoEncoderFormat, QCAP_COLORSPACE_TYPE_NV12, nVideoWidth, nVideoHeight, dVideoFrameRate,
        QCAP_RECORD_PROFILE_MAIN, QCAP_RECORD_LEVEL_51, QCAP_RECORD_ENTROPY_CABAC,
        QCAP_RECORD_COMPLEXITY_0, QCAP_RECORD_MODE_CBR, 8000, 40000000, 60, 0, FALSE,
        0, 0, 0, FALSE, FALSE, FALSE, 0, 0, 0, 0, 0, 0);
    qcap2_video_encoder_property_set_high_perf(pProp, TRUE);
    qcap2_video_decoder_set_video_property(pVdec, pProp);
    qcap2_video_encoder_property_delete(pProp);
    qcap2_video_decoder_set_event(pVdec, pEvent_vdec);
    qcap2_video_decoder_set_multithread(pVdec, false);
    qcap2_video_decoder_set_packet_count(pVdec, 16);
    qcap2_video_decoder_set_frame_count(pVdec, 10);
    qcap2_video_decoder_start(pVdec);

    pVideoSink = qcap2_video_sink_new();
    int backendType = QCAP2_VIDEO_SINK_BACKEND_TYPE_DAVMF;
    if (getenv("QCAP_VO_USE_GSTREAMER") && QString(getenv("QCAP_VO_USE_GSTREAMER")) == "1")
        backendType = QCAP2_VIDEO_SINK_BACKEND_TYPE_GSTREAMER;
    qcap2_video_sink_set_backend_type(pVideoSink, backendType);
    qcap2_video_sink_set_native_handle(pVideoSink, m_winId);
    if (m_bDisplayEnabled) qcap2_video_sink_set_nvbuf(pVideoSink, false);
    qcap2_video_format_t* pFormat = qcap2_video_format_new();
    if (pFormat) {
        qcap2_video_format_set_property(pFormat, QCAP_COLORSPACE_TYPE_NV12, nVideoWidth, nVideoHeight, bVideoIsInterleaved, dVideoFrameRate);
        qcap2_video_sink_set_video_format(pVideoSink, pFormat);
        qcap2_video_format_delete(pFormat);
    }
    if (m_bDisplayEnabled) qcap2_video_sink_start(pVideoSink);

    pScaler = qcap2_video_scaler_new();
    qcap2_video_scaler_set_backend_type(pScaler, QCAP2_VIDEO_SCALER_BACKEND_TYPE_DEFAULT);
    qcap2_video_format_t* pScalerFormat = qcap2_video_format_new();
    if (pScalerFormat) {
        qcap2_video_format_set_property(pScalerFormat, QCAP_COLORSPACE_TYPE_NV12, nVideoWidth, nVideoHeight, bVideoIsInterleaved, dVideoFrameRate);
        qcap2_video_scaler_set_video_format(pScaler, pScalerFormat);
        qcap2_video_format_delete(pScalerFormat);
    }
    qcap2_video_scaler_set_frame_count(pScaler, 8);
    qcap2_video_scaler_set_src_buffer_hint(pScaler, QCAP2_BUFFER_HINT_DEFAULT);
    qcap2_video_scaler_set_dst_buffer_hint(pScaler, QCAP2_BUFFER_HINT_DEFAULT);
    qcap2_video_scaler_set_auto_run(pScaler, true);
    qcap2_video_scaler_start(pScaler);

    return QCAP_RT_OK;
}

QRETURN ChannelContext::onVideoCallback(double dSampleTime, BYTE * pStreamBuffer, ULONG nStreamBufferLen, BOOL bIsKeyFrame) {
    Q_UNUSED(dSampleTime); Q_UNUSED(bIsKeyFrame);
    qcap2_video_decoder_t* pLocalVdec = nullptr;
    { QMutexLocker locker(&m_mutex); if (!pClient || !pVdec) return QCAP_RT_OK; pLocalVdec = pVdec; }
    qcap2_rcbuffer_t* pRCBuffer = qcap2_rcbuffer_cast(pStreamBuffer, nStreamBufferLen);
    if (!pRCBuffer) return QCAP_RT_OK;
    qcap2_video_decoder_push(pLocalVdec, pRCBuffer);
    return QCAP_RT_OK;
}

QRETURN ChannelContext::onEventVdec() {
    qcap2_video_decoder_t* pLocalVdec = nullptr;
    qcap2_video_scaler_t* pLocalScaler = nullptr;
    qcap2_video_sink_t* pLocalVideoSink = nullptr;
    bool bDisplayEnabled = false, bSendBuffer = false;
    {
        QMutexLocker locker(&m_mutex);
        if (!pVdec) return QCAP_RT_OK;
        pLocalVdec = pVdec; pLocalScaler = pScaler; pLocalVideoSink = pVideoSink;
        bDisplayEnabled = m_bDisplayEnabled; bSendBuffer = m_bSendBuffer;
    }

    qcap2_rcbuffer_t* pRCBuffer_vdec = nullptr;
    if (qcap2_video_decoder_pop(pLocalVdec, &pRCBuffer_vdec) != QCAP_RS_SUCCESSFUL) return QCAP_RT_OK;

    if (pLocalScaler) {
        qcap2_video_scaler_push(pLocalScaler, pRCBuffer_vdec);
        qcap2_rcbuffer_release(pRCBuffer_vdec);
        qcap2_rcbuffer_t* pSysBuffer = nullptr;
        if (qcap2_video_scaler_pop(pLocalScaler, &pSysBuffer) != QCAP_RS_SUCCESSFUL || !pSysBuffer) return QCAP_RT_OK;

        if (bSendBuffer && g_pMainwindow && g_pMainwindow->ai_running) {
            double current_time = QCAP_GET_TIME();
            if ((current_time - m_lastProcessTime) >= FRAME_INTERVAL && !m_bFrameReady) {
                m_lastProcessTime = current_time;
                PVOID pLockedData = qcap2_rcbuffer_lock_data(pSysBuffer);
                if (pLockedData) {
                    qcap2_av_frame_t* pAVFrame = reinterpret_cast<qcap2_av_frame_t*>(pLockedData);
                    uint8_t* pBuffer[4] = {nullptr};
                    int pStride[4] = {0};
                    qcap2_av_frame_get_buffer1(pAVFrame, pBuffer, pStride);
                    // Copy NV12 data directly to the batch buffer (skip m_pAIBuffer intermediate)
                    if (pBuffer[0] && pStride[0] > 0) {
                        BYTE* pDstBuf = g_pMainwindow->buffer_vec[channelId];
                        ULONG totalBytes = (m_nVideoWidth * m_nVideoHeight * 3) / 2;
                        if (pDstBuf) {
                            int src_pitch_Y = pStride[0];
                            for (ULONG row = 0; row < m_nVideoHeight; ++row)
                                memcpy(pDstBuf + row * m_nVideoWidth, pBuffer[0] + row * src_pitch_Y, m_nVideoWidth);
                            if (pBuffer[1] && pStride[1] > 0) {
                                int src_pitch_UV = pStride[1];
                                BYTE* pDstUV = pDstBuf + (m_nVideoWidth * m_nVideoHeight);
                                for (ULONG row = 0; row < m_nVideoHeight / 2; ++row)
                                    memcpy(pDstUV + row * m_nVideoWidth, pBuffer[1] + row * src_pitch_UV, m_nVideoWidth);
                            }
                            g_pMainwindow->buffer_len_vec[channelId] = totalBytes;
                        }
                    }
                    qcap2_rcbuffer_unlock_data(pSysBuffer);
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

        bool bSinkActive = false;
        { QMutexLocker locker(&m_mutex); if (pVideoSink) bSinkActive = true; }
        if (bDisplayEnabled && bSinkActive && pLocalVideoSink)
            qcap2_video_sink_push(pLocalVideoSink, pSysBuffer);
        qcap2_rcbuffer_release(pSysBuffer);
    } else {
        qcap2_rcbuffer_release(pRCBuffer_vdec);
    }
    { QMutexLocker locker(&m_mutex); m_frameCount++; }
    return QCAP_RT_OK;
}


// ── MainWindow Implementation ──────────────────────────────────────────────
MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), m_bFullscreen(false),
      overlayWidget(nullptr), m_bShowOverlay(true),
      handle(nullptr), flag(1),
      ai_running(false), pAiThread(nullptr),
      ready_count(0), active_camera_count(0)
{
    setWindowTitle("QCAP Multichannel RTSP + QDEEP Plate Detection");
    resize(1280, 720);
    g_pMainwindow = this;

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
    chkShowOverlay = new QCheckBox("Show AI Detection Overlay", grpConfig);
    chkShowOverlay->setChecked(true);
    grpLayout->addWidget(chkShowOverlay);
    controlLayout->addWidget(grpConfig);
    lblStatus = new QLabel("Status: Idle", controlPanel);
    lblStatus->setWordWrap(true);
    controlLayout->addWidget(lblStatus);
    mainLayout->addWidget(controlPanel);

    videoContainer = new QWidget(centralWidget);
    videoGridLayout = new QGridLayout(videoContainer);
    videoGridLayout->setContentsMargins(0, 0, 0, 0);
    videoGridLayout->setSpacing(2);
    mainLayout->addWidget(videoContainer, 1);

    onChannelCountChanged(4);
    m_bEnableDisplay = true;

    connect(spinChannelCount, QOverload<int>::of(&QSpinBox::valueChanged), this, &MainWindow::onChannelCountChanged);
    connect(btnStart, &QPushButton::clicked, this, &MainWindow::onBtnStartClicked);
    connect(btnStop, &QPushButton::clicked, this, &MainWindow::onBtnStopClicked);
    connect(chkEnableDisplay, &QCheckBox::toggled, this, &MainWindow::onDisplayToggled);
    connect(chkShowOverlay, &QCheckBox::toggled, this, &MainWindow::onOverlayToggled);
    videoContainer->installEventFilter(this);
    m_timerId = startTimer(1000);

    init_models();
    overlayWidget = new OverlayWidget(nullptr);
    overlayWidget->setGeometry(this->geometry());
    overlayWidget->show();
    overlayWidget->raise();
}

MainWindow::~MainWindow() {
    if (overlayWidget) { delete overlayWidget; overlayWidget = nullptr; }
    uninit_models();
    stopAllChannels();
}

void MainWindow::onChannelCountChanged(int count) {
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
        if (!itemUrl || itemUrl->text().isEmpty())
            tableUrls->setItem(i, 1, new QTableWidgetItem("rtsp://root:root@192.168.191.6:1554/session0.mpg"));
    }
}

void MainWindow::onBtnStartClicked() {
    stopAllChannels(); clearGrid();
    int count = spinChannelCount->value();
    int cols = (count <= 2) ? count : (count <= 4) ? 2 : (count <= 9) ? 3 : 4;
    for (int i = 0; i < count; ++i) {
        QFrame *frame = new QFrame(videoContainer);
        frame->setFrameShape(QFrame::Box); frame->setLineWidth(1);
        frame->setStyleSheet("background-color: black; border: 1px solid #333333;");
        frame->installEventFilter(this);
        videoFrames.append(frame);
        videoGridLayout->addWidget(frame, i / cols, i % cols);
        frame->show();
        ChannelContext *ctx = new ChannelContext(i, tableUrls->item(i, 1)->text().trimmed(), frame->winId());
        ctx->setDisplayEnabled(m_bEnableDisplay);
        channels.append(ctx);
        ctx->start();
    }
    spinChannelCount->setEnabled(false); tableUrls->setEnabled(false);
    btnStart->setEnabled(false); btnStop->setEnabled(true);
    lblStatus->setText("Status: Running");
    yolo_start();
}

void MainWindow::onBtnStopClicked() {
    yolo_stop();
    stopAllChannels(); clearGrid();
    spinChannelCount->setEnabled(true); tableUrls->setEnabled(true);
    btnStart->setEnabled(true); btnStop->setEnabled(false);
    lblStatus->setText("Status: Stopped");
}

void MainWindow::stopAllChannels() {
    for (ChannelContext *ctx : channels) { ctx->m_bSendBuffer = false; delete ctx; }
    channels.clear();
}

void MainWindow::clearGrid() {
    QLayoutItem *item;
    while ((item = videoGridLayout->takeAt(0)) != nullptr) {
        if (item->widget()) item->widget()->deleteLater();
        delete item;
    }
    videoFrames.clear();
}

void MainWindow::timerEvent(QTimerEvent *event) {
    if (event->timerId() == m_timerId) {
        QString statusText = "Channel Stats:\n";
        for (int i = 0; i < channels.size(); ++i) {
            ChannelContext *ctx = channels[i];
            QMutexLocker locker(&ctx->m_mutex);
            statusText += QString("CH%1: %2 | %3 fps\n").arg(i + 1)
                .arg(ctx->m_statusInfo.isEmpty() ? "Connecting..." : ctx->m_statusInfo).arg(ctx->m_frameCount);
            ctx->m_frameCount = 0;
        }
        lblStatus->setText(statusText);
    }
}

void MainWindow::closeEvent(QCloseEvent *event) { yolo_stop(); stopAllChannels(); event->accept(); }

bool MainWindow::eventFilter(QObject *watched, QEvent *event) {
    if (event->type() == QEvent::MouseButtonDblClick) {
        m_bFullscreen = !m_bFullscreen;
        if (m_bFullscreen) { controlPanel->hide(); showFullScreen(); }
        else { controlPanel->show(); showNormal(); }
        int count = videoFrames.size();
        if (count > 0) {
            int cols = m_bFullscreen ? 8 : (count <= 2) ? count : (count <= 4) ? 2 : (count <= 9) ? 3 : 4;
            for (int i = 0; i < count; ++i) {
                videoGridLayout->removeWidget(videoFrames[i]);
                videoGridLayout->addWidget(videoFrames[i], i / cols, i % cols);
            }
        }
        return true;
    }
    return QMainWindow::eventFilter(watched, event);
}

void MainWindow::onDisplayToggled(bool checked) { m_bEnableDisplay = checked; for (auto* c : channels) c->setDisplayEnabled(checked); }
void MainWindow::onOverlayToggled(bool checked) { m_bShowOverlay = checked; if (overlayWidget) overlayWidget->update(); }

// ── Top-level OverlayWidget 事件追蹤 ──────────────────────────────────────
void MainWindow::resizeEvent(QResizeEvent *event) {
    QMainWindow::resizeEvent(event);
    if (overlayWidget) { overlayWidget->setGeometry(this->geometry()); overlayWidget->raise(); }
}
void MainWindow::moveEvent(QMoveEvent *event) {
    QMainWindow::moveEvent(event);
    if (overlayWidget) { overlayWidget->setGeometry(this->geometry()); overlayWidget->raise(); }
}
void MainWindow::hideEvent(QHideEvent *event) {
    QMainWindow::hideEvent(event);
    if (overlayWidget) overlayWidget->hide();
}
void MainWindow::showEvent(QShowEvent *event) {
    QMainWindow::showEvent(event);
    if (overlayWidget) { overlayWidget->setGeometry(this->geometry()); overlayWidget->show(); overlayWidget->raise(); }
}

// ── QDEEP / YOLO AI Functions ──────────────────────────────────────────────
void MainWindow::init_models() {
    for (int i = 0; i < MAX_BATCH; ++i) {
        box_list_vec[i] = new QDEEP_API::QDEEP_OBJECT_DETECT_BOUNDING_BOX[BOX_SIZE];
        buffer_vec[i] = new BYTE[MAX_BUFFER_SIZE]();
        color_space[i] = QDEEP_API::QDEEP_COLORSPACE_TYPE_NV12;
        width_vec[i] = 1920; height_vec[i] = 1080; buffer_len_vec[i] = MAX_BUFFER_SIZE;
    }

    QDEEP_API::QDEEP_CREATE_BATCH_OBJECT_DETECT(
        QDEEP_API::QDEEP_GPU_TYPE_NVIDIA, 0,
        QDEEP_API::QDEEP_OBJECT_DETECT_CONFIG_MODEL_CUSTOMIZED_LITE_NEW,
        (char*)"/home/nvidia/Documents/QtQcapMultiClientDemo_onlydecode_npptosys/model/people_/QDEEP.OD.TINY.PERSON.V10N.CFG",
        &handle, flag, MAX_BATCH);

    QDEEP_API::QDEEP_START_OBJECT_DETECT(handle);
    QDEEP_API::QDEEP_SET_OBJECT_DETECT_PROPERTY(handle, 0.1);
    QDEEP_GET_OBJECT_DETECT_RESERVED_STATUS(reinterpret_cast<PVOID>(0xD7CBB416), reinterpret_cast<ULONG*>(0x3B98119E));
}

void MainWindow::uninit_models() {
    yolo_stop();
    if (handle != nullptr) { QDEEP_API::QDEEP_STOP_OBJECT_DETECT(handle); QDEEP_API::QDEEP_DESTROY_OBJECT_DETECT(handle); handle = nullptr; }
    for (size_t i = 0; i < MAX_BATCH; ++i) {
        if (box_list_vec[i]) { delete[] box_list_vec[i]; box_list_vec[i] = nullptr; }
        if (buffer_vec[i]) { delete[] buffer_vec[i]; buffer_vec[i] = nullptr; }
    }
}

void MainWindow::yolo_start() {
    if (ai_running) return;
    active_camera_count = 0;
    for (ChannelContext *ctx : channels) {
        if (ctx->pClient != nullptr) { ctx->m_bSendBuffer = true; ctx->m_lastProcessTime = 0.0; ctx->m_bFrameReady = false; active_camera_count++; }
    }
    if (active_camera_count == 0) { qDebug() << "[Warning] No active cameras found!"; return; }
    ai_running = true;
    pAiThread = new std::thread(&MainWindow::ai_inference_thread, this);
}

void MainWindow::yolo_stop() {
    if (!ai_running) return;
    for (ChannelContext *ctx : channels) ctx->m_bSendBuffer = false;
    ai_running = false; cv.notify_one();
    if (pAiThread && pAiThread->joinable()) { pAiThread->join(); delete pAiThread; pAiThread = nullptr; }
}

void MainWindow::ai_inference_thread() {
    auto last_log_time = std::chrono::steady_clock::now();
    int inference_count = 0;

    while (ai_running) {
        std::unique_lock<std::mutex> lock(mtx);
        active_camera_count = 0;
        for (ChannelContext *ctx : channels) {
            if (ctx->m_bSendBuffer && ctx->m_nVideoWidth > 0 && ctx->m_nVideoHeight > 0) active_camera_count++;
        }
        if (active_camera_count == 0) { lock.unlock(); std::this_thread::sleep_for(std::chrono::milliseconds(67)); continue; }

        // 等待全部 active channel 都有新 frame，或 timeout 後用舊資料送出
        cv.wait_for(lock, std::chrono::milliseconds(67), [this]{ return (ready_count >= active_camera_count) || !ai_running; });
        if (!ai_running) break;

        // 準備 batch 資料：有新 frame 就複製，沒新 frame 保留舊 buffer
        for (size_t i = 0; i < MAX_BATCH; ++i) {
            box_size_vec[i] = BOX_SIZE;
            ChannelContext* ctx = nullptr;
            for (auto* c : channels) { if (c->channelId == (int)i) { ctx = c; break; } }
            // 資料已在 onEventVdec 中直接寫入 buffer_vec[i]，不需再 memcpy
            if (ctx && ctx->m_bSendBuffer && ctx->m_nVideoWidth > 0 && ctx->m_nVideoHeight > 0) {
                width_vec[i] = ctx->m_nVideoWidth; height_vec[i] = ctx->m_nVideoHeight;
            } else { width_vec[i] = 0; height_vec[i] = 0; buffer_len_vec[i] = 0; }
        }

        // 執行 batch 推論
        QDEEP_API::QDEEP_SET_VIDEO_OBJECT_DETECT_BATCH_UNCOMPRESSION_BUFFER(
            handle, color_space.data(), width_vec.data(), height_vec.data(),
            buffer_vec.data(), buffer_len_vec.data(), box_list_vec.data(), box_size_vec.data(), MAX_BATCH);

        inference_count++;

        // 每 5 秒輸出 AI FPS
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_log_time).count();
        if (elapsed >= 5) {
            double ai_fps = (double)inference_count / elapsed;
            qDebug() << QString("[AI FPS plate] %1 Hz (%2 inferences in %3s, active=%4)")
                        .arg(ai_fps, 0, 'f', 1).arg(inference_count).arg(elapsed).arg(active_camera_count);
            inference_count = 0;
            last_log_time = now;
        }

        // 處理結果
        {
            std::lock_guard<std::mutex> draw_lock(draw_mtx);
            for (int i = 0; i < MAX_BATCH; ++i) {
                draw_boxes[i].clear();
                if (width_vec[i] > 0 && box_size_vec[i] > 0) {
                    for (ULONG j = 0; j < box_size_vec[i]; ++j) {
                        auto& deep_box = box_list_vec[i][j];
                        DrawBox db;
                        db.classId = deep_box.nClassID;
                        db.probability = deep_box.fProbability;
                        db.original_x = deep_box.nX;
                        db.original_y = deep_box.nY;
                        db.original_w = deep_box.nWidth;
                        db.original_h = deep_box.nHeight;

                        // Extract plate text from feature vectors
                        const char* pFeatureStr = reinterpret_cast<const char*>(deep_box.fFeatureVectors);
                        int max_len = QDEEP_MAX_FEATURE_VECTOR_SIZE * sizeof(float);
                        int len = 0;
                        while (len < max_len && pFeatureStr[len] != '\0') len++;
                        db.plateText = QString::fromUtf8(pFeatureStr, len);

                        draw_boxes[i].push_back(db);
                    }
                }
            }
        }

        QMetaObject::invokeMethod(this, [this](){ if (overlayWidget) { overlayWidget->raise(); overlayWidget->update(); } }, Qt::QueuedConnection);

        // 批次重置所有 frame ready flags
        for (ChannelContext *ctx : channels) ctx->m_bFrameReady = false;
        ready_count = 0;
    }
}
