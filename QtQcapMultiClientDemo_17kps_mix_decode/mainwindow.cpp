#include "mainwindow.h"
#include <fstream>
#include "qcap.linux.h"
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
//arya
uint32_t count = 0;
double elapsed = 0.0;

extern "C" {
QDEEP_EXT_API QRESULT QDEEP_EXPORT QDEEP_GET_OBJECT_DETECT_RESERVED_STATUS(PVOID pDetector, ULONG* pCheckNum);
}

#include <opencv2/opencv.hpp>

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

QImage cvMatToQImage(const cv::Mat& mat) {
    if (mat.type() == CV_8UC3) {
        return QImage(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_RGB888).rgbSwapped().copy();
    }
    return QImage();
}


namespace {
// Channel IDs are zero-based, so IDs 0 through 23 are RTSP channels 1 through
// 24.  zznvcodec produces system-memory buffers, allowing both decoder types
// to use the same scaler and display pipeline below.
constexpr int kZznvcodecChannelLimit = 24;

ULONG decoderTypeForChannel(int channelId)
{
    return channelId < kZznvcodecChannelLimit
            ? static_cast<ULONG>(QCAP_DECODER_TYPE_ZZNVCODEC)
            : static_cast<ULONG>(QCAP_DECODER_TYPE_SOFTWARE);
}

const char* decoderName(ULONG decoderType)
{
    return decoderType == QCAP_DECODER_TYPE_ZZNVCODEC
            ? "zznvcodec"
            : "software";
}

} // namespace

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

static QRETURN on_decoder_video_callback(
        PVOID pClient,
        UINT iSessionNum,
        double dSampleTime,
        BYTE * pFrameBuffer,
        ULONG nFrameBufferLen,
        PVOID pUserData)
{
    Q_UNUSED(pClient);
    Q_UNUSED(iSessionNum);
    ChannelContext* ctx = static_cast<ChannelContext*>(pUserData);
    return ctx->onDecodedVideoFrame(dSampleTime, pFrameBuffer, nFrameBufferLen);
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
      pClient(nullptr), pScaler2(nullptr),
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
    stop();
}

bool ChannelContext::start() {
    QMutexLocker locker(&m_mutex);

    qDebug() << "Starting channel" << channelId << "URL:" << url;

    // QCAP owns decoding and delivers its decoded SYSBUF to the decoder
    // callback. Use zznvcodec for channels 1-24 and software from channel 25.
    const ULONG decoderType = decoderTypeForChannel(channelId);
    qDebug() << "CH" << channelId << "selected" << decoderName(decoderType)
             << "decoder for the broadcast client";
    QRESULT qres = QCAP_CREATE_BROADCAST_CLIENT(channelId, url.toLatin1().data(),
                                                 &pClient, decoderType, nullptr,
                                                 FALSE, FALSE);
    if (qres != QCAP_RS_SUCCESSFUL) {
        qCritical() << "QCAP_CREATE_BROADCAST_CLIENT failed for CH" << channelId << "qres =" << qres;
        return false;
    }

    // Receive QCAP-decoded SYSBUF directly; no raw stream callback and no
    // application-created qcap2 decoder are used in this pipeline.
    QCAP_REGISTER_BROADCAST_CLIENT_CONNECTED_CALLBACK(pClient, on_connected_callback, this);
    QCAP_REGISTER_VIDEO_DECODER_BROADCAST_CLIENT_CALLBACK(pClient, on_decoder_video_callback, this);
    QCAP_REGISTER_BROADCAST_CLIENT_FAIL_CALLBACK(pClient, on_fail_callback, this);

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

void ChannelContext::cleanupPipeline()
{
    qcap2_video_scaler_t* pLocalScaler = nullptr;
    qcap2_rcbuffer_queue_t* pLocalAIQueue = nullptr;
    {
        QMutexLocker locker(&m_mutex);
        pLocalScaler = pScaler2;
        pLocalAIQueue = m_pAIQueue;
        pScaler2 = nullptr;
        m_pAIQueue = nullptr;
    }

    if (pLocalAIQueue) {
        qcap2_rcbuffer_t* pBuffer = nullptr;
        while (qcap2_rcbuffer_queue_pop(pLocalAIQueue, &pBuffer) == QCAP_RS_SUCCESSFUL && pBuffer) {
            qcap2_rcbuffer_release(pBuffer);
        }
        qcap2_rcbuffer_queue_stop(pLocalAIQueue);
        qcap2_rcbuffer_queue_delete(pLocalAIQueue);
    }
    if (pLocalScaler) {
        qcap2_video_scaler_stop(pLocalScaler);
        qcap2_video_scaler_delete(pLocalScaler);
    }
}

void ChannelContext::stop()
{
    PVOID pLocalClient = nullptr;
    {
        QMutexLocker locker(&m_mutex);
        pLocalClient = pClient;
        pClient = nullptr;
    }
    if (pLocalClient) QCAP_STOP_BROADCAST_CLIENT(pLocalClient);
    cleanupPipeline();
    if (pLocalClient) QCAP_DESTROY_BROADCAST_CLIENT(pLocalClient);
}

QRETURN ChannelContext::onFail(UINT iSessionNum, QRESULT nErrorStatus, DWORD nErrorCode)
{
    Q_UNUSED(iSessionNum);
    QMutexLocker locker(&m_mutex);
    qCritical() << "CH" << channelId << "Broadcast client failure callback! Status:"
                << nErrorStatus << "Code:" << nErrorCode;
    m_statusInfo = QString("Disconnected (Error 0x%1)").arg(nErrorStatus, 8, 16, QChar('0'));
    return QCAP_RT_OK;
}

void ChannelContext::setDisplayEnabled(bool enabled)
{
    QMutexLocker locker(&m_mutex);
    m_bDisplayEnabled = enabled;
}

QRETURN ChannelContext::onConnected(
        PVOID pClient, UINT iSessionNum, ULONG nVideoEncoderFormat,
        ULONG nVideoWidth, ULONG nVideoHeight, BOOL bVideoIsInterleaved,
        double dVideoFrameRate)
{
    Q_UNUSED(pClient);
    Q_UNUSED(iSessionNum);

    bool needCleanup = false;
    {
        QMutexLocker locker(&m_mutex);
        needCleanup = pScaler2 || m_pAIQueue;
    }
    if (needCleanup) cleanupPipeline();

    QMutexLocker locker(&m_mutex);
    if (nVideoWidth == 0 || nVideoHeight == 0 || nVideoWidth > 8192 || nVideoHeight > 8192) {
        qCritical() << "CH" << channelId << "Connected with unreasonable dimensions:"
                    << nVideoWidth << "x" << nVideoHeight;
        m_statusInfo = QString("Aborted (unreasonable dimensions: %1x%2)")
                           .arg(nVideoWidth).arg(nVideoHeight);
        return QCAP_RT_OK;
    }

    m_nVideoWidth = nVideoWidth;
    m_nVideoHeight = nVideoHeight;
    m_dVideoFrameRate = dVideoFrameRate;
    m_nVideoEncoderFormat = nVideoEncoderFormat;
    m_nAIWidth = 640;
    m_nAIHeight = 384;
    m_nAIBufferLen = m_nAIWidth * m_nAIHeight * 3 / 2;

    qcap2_video_scaler_t* scaler = qcap2_video_scaler_new();
    qcap2_video_format_t* scalerFormat = scaler ? qcap2_video_format_new() : nullptr;
    if (!scaler || !scalerFormat) {
        if (scalerFormat) qcap2_video_format_delete(scalerFormat);
        if (scaler) qcap2_video_scaler_delete(scaler);
        qCritical() << "CH" << channelId << "cannot create SYSBUF downscaler";
        m_statusInfo = QStringLiteral("Downscaler creation failed");
        return QCAP_RT_OK;
    }
    qcap2_video_scaler_set_backend_type(scaler, QCAP2_VIDEO_SCALER_BACKEND_TYPE_DEFAULT);
    qcap2_video_format_set_property(scalerFormat, QCAP_COLORSPACE_TYPE_NV12,
                                    m_nAIWidth, m_nAIHeight,
                                    bVideoIsInterleaved, dVideoFrameRate);
    qcap2_video_scaler_set_video_format(scaler, scalerFormat);
    qcap2_video_format_delete(scalerFormat);
    qcap2_video_scaler_set_frame_count(scaler, 8);
    qcap2_video_scaler_set_src_buffer_hint(scaler, QCAP2_BUFFER_HINT_DEFAULT);
    qcap2_video_scaler_set_dst_buffer_hint(scaler, QCAP2_BUFFER_HINT_DEFAULT);
    qcap2_video_scaler_set_auto_run(scaler, true);
    const QRESULT scalerResult = qcap2_video_scaler_start(scaler);
    if (scalerResult != QCAP_RS_SUCCESSFUL) {
        qcap2_video_scaler_delete(scaler);
        qCritical() << "CH" << channelId << "cannot start SYSBUF downscaler:" << scalerResult;
        m_statusInfo = QString("Downscaler start failed (%1)").arg(scalerResult);
        return QCAP_RT_OK;
    }
    pScaler2 = scaler;

    m_pAIQueue = qcap2_rcbuffer_queue_new();
    if (m_pAIQueue) {
        qcap2_rcbuffer_queue_set_max_buffers(m_pAIQueue, 3);
        qcap2_rcbuffer_queue_start(m_pAIQueue);
    } else {
        qWarning() << "CH" << channelId << "cannot create AI queue";
    }

    QString formatStr;
    switch (nVideoEncoderFormat) {
    case QCAP_ENCODER_FORMAT_H264: formatStr = "H.264"; break;
    case QCAP_ENCODER_FORMAT_H265: formatStr = "H.265"; break;
    case QCAP_ENCODER_FORMAT_AV1: formatStr = "AV1"; break;
    default: formatStr = QString("Unknown (%1)").arg(nVideoEncoderFormat); break;
    }
    m_statusInfo = QString("%1x%2 @%3fps (%4), downscale %5x%6")
            .arg(nVideoWidth).arg(nVideoHeight).arg(dVideoFrameRate).arg(formatStr)
            .arg(m_nAIWidth).arg(m_nAIHeight);
    return QCAP_RT_OK;
}

QRETURN ChannelContext::onDecodedVideoFrame(
        double dSampleTime, BYTE* pFrameBuffer, ULONG nFrameBufferLen)
{
    Q_UNUSED(dSampleTime);
    qcap2_rcbuffer_queue_t* pLocalAIQueue = nullptr;
    bool bSendBuffer = false;
    double dSourceFrameRate = DEFAULT_AI_TARGET_FPS;
    QMutexLocker frameLocker(&m_mutex);
    {
        if (!pClient || !pScaler2) return QCAP_RT_OK;
        pLocalAIQueue = m_pAIQueue;
        bSendBuffer = m_bSendBuffer;
        dSourceFrameRate = m_dVideoFrameRate;
    }

    // The decoder callback SYSBUF is owned by QCAP. It is only used as input
    // to the downscaler during this callback and is never released here.
    qcap2_rcbuffer_t* pQcapFrame = qcap2_rcbuffer_cast(pFrameBuffer, nFrameBufferLen);
    if (!pQcapFrame) return QCAP_RT_OK;

    qcap2_rcbuffer_t* pScaledBuffer = nullptr;
    QRESULT qres = QCAP_RS_ERROR_GENERAL;
    // frameLocker keeps the scaler and AI queue alive throughout this callback.
    {
        if (pClient && pScaler2) {
            qres = qcap2_video_scaler_push(pScaler2, pQcapFrame);
            if (qres == QCAP_RS_SUCCESSFUL)
                qres = qcap2_video_scaler_pop(pScaler2, &pScaledBuffer);
        }
    }
    if (qres != QCAP_RS_SUCCESSFUL || !pScaledBuffer) {
        qWarning() << "[SYSBUF downscaler] CH" << channelId << "push/pop failed:" << qres;
        return QCAP_RT_OK;
    }

    if (bSendBuffer && g_pMainwindow && g_pMainwindow->ai_running && pLocalAIQueue) {
        const double targetFps = dSourceFrameRate > 0.0 ? dSourceFrameRate : DEFAULT_AI_TARGET_FPS;
        const double currentTime = QCAP_GET_TIME();
        if ((currentTime - m_lastProcessTime) >= 1.0 / targetFps) {
            m_lastProcessTime = currentTime;
            if (qcap2_rcbuffer_queue_is_full(pLocalAIQueue)) {
                qcap2_rcbuffer_t* pOld = nullptr;
                if (qcap2_rcbuffer_queue_pop(pLocalAIQueue, &pOld) == QCAP_RS_SUCCESSFUL && pOld)
                    qcap2_rcbuffer_release(pOld);
            }
            if (qcap2_rcbuffer_queue_push(pLocalAIQueue, pScaledBuffer) == QCAP_RS_SUCCESSFUL)
                g_pMainwindow->cv.notify_one();
        }
    }

    qcap2_rcbuffer_release(pScaledBuffer);
    if (!m_fpsTimer.isValid()) m_fpsTimer.start();
    ++m_decFrameCount;
    if (m_fpsTimer.elapsed() >= 2000) { m_decFrameCount = 0; m_fpsTimer.restart(); }
    ++m_frameCount;
    return QCAP_RT_OK;
}

// ── MainWindow Implementation ───────────────────────────────────────────────
MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), m_bFullscreen(false),
      // AI init
      m_bShowOverlay(true), m_bHalfRefreshRate(false),
      handle(nullptr), flag(1),
      ai_running(false), pAiThread(nullptr),
      ready_count(0), active_camera_count(0)
{
    setWindowTitle("QCAP RTSP + QDEEP 17KPS (24 zznvcodec + software)");
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
            QString defaultUrl = QString("rtsp://root:root@192.168.190.228:554/session0.mpg");
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
    std::vector<std::thread> stop_threads;
    for (ChannelContext *ctx : channels) {
        ctx->m_bSendBuffer = false;
        stop_threads.push_back(std::thread([ctx]() {
            delete ctx;
        }));
    }
    for (auto& t : stop_threads) {
        if (t.joinable()) {
            t.join();
        }
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
    for (ChannelContext *ctx : channels) {
        ctx->setDisplayEnabled(checked);
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
    for (int i = 0; i < MAX_BATCH; ++i) {
        box_list_vec[i] = new QDEEP_API::QDEEP_OBJECT_DETECT_BOUNDING_BOX[BOX_SIZE];
        buffer_vec[i] = new BYTE[MAX_BUFFER_SIZE]();
        color_space[i] = QDEEP_API::QDEEP_COLORSPACE_TYPE_NV12;
        width_vec[i] = 640;
        height_vec[i] = 384;
        buffer_len_vec[i] = MAX_BUFFER_SIZE;
    }

        QRESULT res = QDEEP_API::QDEEP_CREATE_BATCH_OBJECT_DETECT(
            QDEEP_API::QDEEP_GPU_TYPE_NVIDIA, 0,
            QDEEP_API::QDEEP_OBJECT_DETECT_CONFIG_MODEL_HUMAN_SKELETON_17_KEYPOINTS_EX,
            (char*)"/home/nvidia/Music/ai_model/RELEASES.QDEEP.MODEL.HUMAN.SKELETON.17KPS.EX 1.1.0.203.9.5.1/QDEEP.OD.HUMAN.SKELETON.17KPS.EX.CFG",
            &handle, flag, MAX_BATCH);

    //arya
//    QRESULT res = QDEEP_API::QDEEP_CREATE_BATCH_OBJECT_DETECT(
//                QDEEP_API::QDEEP_GPU_TYPE_NVIDIA, 0,
//                QDEEP_API::QDEEP_OBJECT_DETECT_CONFIG_MODEL_HUMAN_SKELETON_17_KEYPOINTS_EX,
//                (char*)"/home/nvidia/Downloads/arya/sdvoe_bacth/demo/model/skeleton_ex/QDEEP.OD.HUMAN.SKELETON.17KPS.EX.CFG",
//                &handle, flag, MAX_BATCH);

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
        if (buffer_vec[i]) { delete[] buffer_vec[i]; buffer_vec[i] = nullptr; }
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
    // Retain a downscaled frame for every active channel.  After the initial
    // cache is complete, one refreshed channel immediately triggers the next
    // QDEEP call with the latest cached frame from every active channel.
    std::vector<bool> hasCachedFrame(MAX_BATCH, false);
    auto last_log_time = std::chrono::steady_clock::now();
    int inference_count = 0;
    double api_total_ms = 0.0;
    double api_min_ms = 0.0;
    double api_max_ms = 0.0;

    while (ai_running) {
        std::vector<int> active_channels;
        bool any_frame_refreshed = false;

        for (ChannelContext *ctx : channels) {
            if (!ctx || ctx->channelId < 0 || ctx->channelId >= MAX_BATCH)
                continue;

            QMutexLocker channelLocker(&ctx->m_mutex);
            if (!ctx->m_bSendBuffer || ctx->m_nVideoWidth == 0 ||
                ctx->m_nVideoHeight == 0 || !ctx->m_pAIQueue)
                continue;

            const int channel_id = ctx->channelId;
            active_channels.push_back(channel_id);
            qcap2_rcbuffer_t* latest = nullptr;
            qcap2_rcbuffer_t* buffer = nullptr;
            while (qcap2_rcbuffer_queue_pop(ctx->m_pAIQueue, &buffer) == QCAP_RS_SUCCESSFUL && buffer) {
                if (latest) qcap2_rcbuffer_release(latest);
                latest = buffer;
            }
            if (!latest)
                continue;

            PVOID locked = qcap2_rcbuffer_lock_data(latest);
            if (locked) {
                qcap2_av_frame_t* frame = reinterpret_cast<qcap2_av_frame_t*>(locked);
                uint8_t* planes[4] = {nullptr};
                int strides[4] = {0};
                qcap2_av_frame_get_buffer1(frame, planes, strides);
                constexpr ULONG width = 640;
                constexpr ULONG height = 384;
                if (planes[0] && planes[1] && strides[0] >= static_cast<int>(width) &&
                    strides[1] >= static_cast<int>(width) && buffer_vec[channel_id]) {
                    BYTE* destination = buffer_vec[channel_id];
                    for (ULONG row = 0; row < height; ++row)
                        memcpy(destination + row * width, planes[0] + row * strides[0], width);
                    BYTE* destination_uv = destination + width * height;
                    for (ULONG row = 0; row < height / 2; ++row)
                        memcpy(destination_uv + row * width, planes[1] + row * strides[1], width);
                    width_vec[channel_id] = width;
                    height_vec[channel_id] = height;
                    buffer_len_vec[channel_id] = width * height * 3 / 2;
                    hasCachedFrame[channel_id] = true;
                    any_frame_refreshed = true;
                }
                qcap2_rcbuffer_unlock_data(latest);
            }
            qcap2_rcbuffer_release(latest);
        }

        active_camera_count = static_cast<int>(active_channels.size());
        bool all_channels_cached = !active_channels.empty();
        for (int channel_id : active_channels)
            all_channels_cached = all_channels_cached && hasCachedFrame[channel_id];

        if (!all_channels_cached || !any_frame_refreshed) {
            std::unique_lock<std::mutex> lock(mtx);
            cv.wait_for(lock, std::chrono::milliseconds(10), [this] { return !ai_running; });
            continue;
        }

        {
            const size_t count = active_channels.size();
            std::vector<ULONG> batch_color_space;
            std::vector<ULONG> batch_width;
            std::vector<ULONG> batch_height;
            std::vector<BYTE*> batch_buffer;
            std::vector<ULONG> batch_buffer_len;
            std::vector<QDEEP_API::QDEEP_OBJECT_DETECT_BOUNDING_BOX*> batch_box_list;
            std::vector<ULONG> batch_box_size;
            batch_color_space.reserve(count);
            batch_width.reserve(count);
            batch_height.reserve(count);
            batch_buffer.reserve(count);
            batch_buffer_len.reserve(count);
            batch_box_list.reserve(count);
            batch_box_size.reserve(count);
            for (size_t offset = 0; offset < count; ++offset) {
                const int channel_id = active_channels[offset];
                box_size_vec[channel_id] = BOX_SIZE;
                batch_color_space.push_back(color_space[channel_id]);
                batch_width.push_back(width_vec[channel_id]);
                batch_height.push_back(height_vec[channel_id]);
                batch_buffer.push_back(buffer_vec[channel_id]);
                batch_buffer_len.push_back(buffer_len_vec[channel_id]);
                batch_box_list.push_back(box_list_vec[channel_id]);
                batch_box_size.push_back(box_size_vec[channel_id]);
            }

            const auto inference_start = std::chrono::steady_clock::now();
            const QRESULT api_res = QDEEP_API::QDEEP_SET_VIDEO_OBJECT_DETECT_BATCH_UNCOMPRESSION_BUFFER(
                handle, batch_color_space.data(), batch_width.data(), batch_height.data(),
                batch_buffer.data(), batch_buffer_len.data(), batch_box_list.data(), batch_box_size.data(),
                static_cast<ULONG>(count));
            const auto inference_end = std::chrono::steady_clock::now();
            const double api_ms = std::chrono::duration<double, std::milli>(inference_end - inference_start).count();

            if (api_res != QCAP_RS_SUCCESSFUL) {
                qWarning() << "[AI QDEEP 17kps] batch call failed:" << api_res
                           << "channels" << static_cast<int>(count);
            } else {
                for (size_t offset = 0; offset < count; ++offset)
                    box_size_vec[active_channels[offset]] = batch_box_size[offset];
            }
            ++inference_count;
            api_total_ms += api_ms;
            if (inference_count == 1 || api_ms < api_min_ms) api_min_ms = api_ms;
            if (api_ms > api_max_ms) api_max_ms = api_ms;
        }

        auto now = std::chrono::steady_clock::now();
        const auto elapsed_seconds = std::chrono::duration_cast<std::chrono::seconds>(now - last_log_time).count();
        if (elapsed_seconds >= 5 && inference_count > 0) {
            qDebug() << QString("[AI QDEEP 17kps] %1 Hz (%2 calls in %3s, channels=%4, api_ms avg/min/max=%5/%6/%7)")
                        .arg(static_cast<double>(inference_count) / elapsed_seconds, 0, 'f', 1)
                        .arg(inference_count)
                        .arg(elapsed_seconds)
                        .arg(active_camera_count)
                        .arg(api_total_ms / inference_count, 0, 'f', 2)
                        .arg(api_min_ms, 0, 'f', 2)
                        .arg(api_max_ms, 0, 'f', 2);
            inference_count = 0;
            api_total_ms = 0.0;
            api_min_ms = 0.0;
            api_max_ms = 0.0;
            last_log_time = now;
        }

        // Render every cached, downscaled input after all QDEEP chunks finish.
        std::vector<int> batch_channels = active_channels;
        // Render only the downscaled frames submitted in this inference batch.
        for (int i : batch_channels) {
            if (width_vec[i] > 0 && buffer_vec[i] != nullptr) {
                ChannelContext* ctx = nullptr;
                for (auto* c : channels) {
                    if (c->channelId == i) {
                        ctx = c;
                        break;
                    }
                }

                if (ctx && ctx->m_bDisplayEnabled && ctx->m_pLabel) {
                    // Convert contiguous NV12 to BGR Mat
                    cv::Mat nv12_mat(384 * 3 / 2, 640, CV_8UC1, buffer_vec[i]);
                    cv::Mat bgr_mat;
                    cv::cvtColor(nv12_mat, bgr_mat, cv::COLOR_YUV2BGR_NV12);

                    // Draw AI results if enabled
                    if (m_bShowOverlay) {
                        // Draw channel header text
                        std::string headerText = "CH " + std::to_string(i + 1) + " | People: " + std::to_string(box_size_vec[i]);
                        cv::putText(bgr_mat, headerText, cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 200), 2);

                        // Draw bounding boxes and skeleton lines
                        for (ULONG j = 0; j < box_size_vec[i]; ++j) {
                            auto& deep_box = box_list_vec[i][j];

                            // Collect keypoints
                            const float kpt_thresh = 0.3f;
                            cv::Point kpts[17];
                            bool kpt_valid[17] = {false};
                            for (int k = 0; k < 17; ++k) {
                                if (deep_box.sKeypoints[k].fProbability >= kpt_thresh) {
                                    kpts[k] = cv::Point(deep_box.sKeypoints[k].nX, deep_box.sKeypoints[k].nY);
                                    kpt_valid[k] = true;
                                }
                            }

                            // Draw skeleton connections
                            for (const auto& conn : connections) {
                                int p1 = conn.first;
                                int p2 = conn.second;
                                if (kpt_valid[p1] && kpt_valid[p2]) {
                                    cv::Scalar connColor;
                                    if (p1 == 0 || p2 == 0 || p1 >= 13 || p2 >= 13) {
                                        connColor = cv::Scalar(0, 255, 255); // Yellow (Face)
                                    } else if (p1 == 4 || p1 == 5 || p1 == 6 || p1 == 10 || p1 == 11 || p1 == 12 ||
                                               p2 == 4 || p2 == 5 || p2 == 6 || p2 == 10 || p2 == 11 || p2 == 12) {
                                        connColor = cv::Scalar(255, 255, 0); // Cyan (Left side)
                                    } else {
                                        connColor = cv::Scalar(147, 20, 255); // Neon pink (Right side)
                                    }
                                    cv::line(bgr_mat, kpts[p1], kpts[p2], connColor, 2);
                                }
                            }

                            // Draw joints
                            for (int k = 0; k < 17; ++k) {
                                if (kpt_valid[k]) {
                                    cv::Scalar jointColor;
                                    if (k == 0 || k >= 13) {
                                        jointColor = cv::Scalar(0, 255, 255); // Yellow (Face)
                                    } else if (k == 4 || k == 5 || k == 6 || k == 10 || k == 11 || k == 12) {
                                        jointColor = cv::Scalar(255, 255, 0); // Cyan (Left side)
                                    } else {
                                        jointColor = cv::Scalar(128, 0, 255); // Pink/Red (Right side)
                                    }
                                    cv::circle(bgr_mat, kpts[k], 4, jointColor, -1);
                                    cv::circle(bgr_mat, kpts[k], 4, cv::Scalar(255, 255, 255), 1);
                                }
                            }
                        }
                    }

                    // Check display rate halving
                    bool skip_this_frame = false;
                    if (m_bHalfRefreshRate) {
                        int f_cnt = ctx->m_displayFrameCount.fetch_add(1);
                        if (f_cnt % 2 != 0) {
                            skip_this_frame = true;
                        }
                    }

                    if (!skip_this_frame) {
                        // Check backpressure: skip frame if the GUI thread is busy rendering the previous one
                        if (ctx->m_pPendingUpdate && !ctx->m_pPendingUpdate->exchange(true)) {
                            // Convert cv::Mat to QImage
                            QImage qimg = cvMatToQImage(bgr_mat);
                            QPointer<QLabel> safeLabel = ctx->m_pLabel;
                            std::shared_ptr<std::atomic<bool>> pending = ctx->m_pPendingUpdate;

                            // Update GUI QLabel asynchronously
                            QMetaObject::invokeMethod(ctx->m_pLabel, [safeLabel, qimg, pending]() {
                                if (safeLabel) {
                                    safeLabel->setPixmap(QPixmap::fromImage(qimg));
                                }
                                if (pending) {
                                    pending->store(false);
                                }
                            }, Qt::QueuedConnection);
                        }
                    }
                }
            }
        }
    }
}
