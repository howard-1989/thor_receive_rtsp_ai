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
#include <QFileInfo>

#include <opencv2/opencv.hpp>
#include <cstring>
#include <algorithm>

MainWindow *g_pMainwindow = nullptr;

// Layer 1 currently runs the same detector in all three independent workers.
// Model_1 and model_2 can later be changed to their chained-model inputs
// without changing queue ownership or worker lifecycle.
static const char* kLayer1Model0 =
    "/home/nvidia/Documents/new_model/taiwantraffic_batch8/QDEEP.OD.TAIWAN.TRAFFIC.C4.TINY.CFG";
static const char* kLayer1Model1 = kLayer1Model0;
static const char* kLayer1Model2 = kLayer1Model0;

extern "C" {
QDEEP_EXT_API QRESULT QDEEP_EXPORT QDEEP_GET_OBJECT_DETECT_RESERVED_STATUS(PVOID pDetector, ULONG* pCheckNum);
}

QImage cvMatToQImage(const cv::Mat& mat) {
    if (mat.type() == CV_8UC3) {
        return QImage(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_RGB888).rgbSwapped().copy();
    }
    return QImage();
}

// Display is fed from an immutable CPU frame, never from a QCAP rcbuffer.
// Each QDEEP worker holds a separate shared_ptr reference until its synchronous
// API call completes, so no consumer can free pixels used by another consumer.
static void postDisplayFrame(ChannelContext* ctx, const std::shared_ptr<SharedFrame>& frame)
{
    if (!ctx || !frame || !ctx->m_pLabel || !ctx->m_pPendingUpdate ||
        ctx->m_pPendingUpdate->exchange(true)) {
        return;
    }

    const ULONG width = frame->width;
    const ULONG height = frame->height;
    if (width == 0 || height == 0 || frame->nv12.size() < width * height * 3 / 2) {
        ctx->m_pPendingUpdate->store(false);
        return;
    }

    cv::Mat nv12Mat(height * 3 / 2, width, CV_8UC1, const_cast<BYTE*>(frame->nv12.data()));
    cv::Mat bgrMat;
    cv::cvtColor(nv12Mat, bgrMat, cv::COLOR_YUV2BGR_NV12);

    if (g_pMainwindow && g_pMainwindow->m_bShowOverlay) {
        std::vector<DrawBox> model0Boxes;
        std::vector<DrawBox> model1Boxes;
        std::vector<DrawBox> model2Boxes;
        {
            std::lock_guard<std::mutex> lock(g_pMainwindow->draw_mtx);
            model0Boxes = g_pMainwindow->layer1Model0DrawBoxes[ctx->channelId];
            model1Boxes = g_pMainwindow->layer1Model1DrawBoxes[ctx->channelId];
            model2Boxes = g_pMainwindow->layer1Model2DrawBoxes[ctx->channelId];
        }
        cv::putText(bgrMat, "CH " + std::to_string(ctx->channelId + 1), cv::Point(10, 25),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 200), 2);
        const auto drawBoxes = [&bgrMat](const std::vector<DrawBox>& boxes, const cv::Scalar& color) {
            for (const DrawBox& box : boxes) {
            const int x = std::max(0, std::min(box.x, bgrMat.cols - 1));
            const int y = std::max(0, std::min(box.y, bgrMat.rows - 1));
            const int w = std::max(1, std::min(box.width, bgrMat.cols - x));
            const int h = std::max(1, std::min(box.height, bgrMat.rows - y));
            cv::rectangle(bgrMat, cv::Rect(x, y, w, h), color, 2);
            cv::putText(bgrMat, box.label.toStdString(), cv::Point(x, std::max(14, y - 4)),
                        cv::FONT_HERSHEY_SIMPLEX, 0.45, color, 2);
            }
        };
        drawBoxes(model0Boxes, cv::Scalar(0, 255, 0));
        drawBoxes(model1Boxes, cv::Scalar(255, 128, 0));
        drawBoxes(model2Boxes, cv::Scalar(0, 255, 255));
    }

    const QImage image = cvMatToQImage(bgrMat);
    QPointer<QLabel> safeLabel = ctx->m_pLabel;
    std::shared_ptr<std::atomic<bool>> pending = ctx->m_pPendingUpdate;
    QMetaObject::invokeMethod(ctx->m_pLabel, [safeLabel, image, pending]() {
        if (safeLabel) safeLabel->setPixmap(QPixmap::fromImage(image));
        if (pending) pending->store(false);
    }, Qt::QueuedConnection);
}

static std::shared_ptr<SharedFrame> copyRawNV12Frame(
    int channelId, const BYTE* source, ULONG sourceLength, ULONG width, ULONG height)
{
    const size_t expectedBytes = static_cast<size_t>(width) * height * 3 / 2;
    if (!source || width == 0 || height == 0 || sourceLength < expectedBytes) return nullptr;
    std::shared_ptr<SharedFrame> frame = std::make_shared<SharedFrame>();
    frame->channelId = channelId;
    frame->width = width;
    frame->height = height;
    frame->nv12.assign(source, source + expectedBytes);
    return frame;
}

// QCAP owns the callback rcbuffer. We lock it once, copy it directly into an
// application-owned SharedFrame, and only unlock it. This code never calls
// release on the callback buffer, so it cannot steal or double-release QCAP's
// reference.
static std::shared_ptr<SharedFrame> copyQcapFrame(int channelId, qcap2_rcbuffer_t* sourceBuffer)
{
    if (!sourceBuffer) return nullptr;
    PVOID locked = qcap2_rcbuffer_lock_data(sourceBuffer);
    if (!locked) return nullptr;
    qcap2_av_frame_t* avFrame = reinterpret_cast<qcap2_av_frame_t*>(locked);
    uint8_t* source[4] = {nullptr};
    int stride[4] = {0};
    ULONG colorSpace = 0, width = 0, height = 0;
    qcap2_av_frame_get_buffer1(avFrame, source, stride);
    qcap2_av_frame_get_video_property(avFrame, &colorSpace, &width, &height);

    const bool nv12 = colorSpace == QCAP_COLORSPACE_TYPE_NV12;
    const bool i420 = colorSpace == QCAP_COLORSPACE_TYPE_I420;
    const bool yv12 = colorSpace == QCAP_COLORSPACE_TYPE_YV12;
    const bool valid = width > 0 && height > 0 && !(width & 1) && !(height & 1) &&
        source[0] && stride[0] >= static_cast<int>(width) &&
        ((nv12 && source[1] && stride[1] >= static_cast<int>(width)) ||
         ((i420 || yv12) && source[1] && source[2] &&
          stride[1] >= static_cast<int>(width / 2) && stride[2] >= static_cast<int>(width / 2)));
    std::shared_ptr<SharedFrame> frame;
    if (valid) {
        frame = std::make_shared<SharedFrame>();
        frame->channelId = channelId;
        frame->width = width;
        frame->height = height;
        frame->nv12.resize(static_cast<size_t>(width) * height * 3 / 2);
        for (ULONG row = 0; row < height; ++row)
            memcpy(frame->nv12.data() + row * width, source[0] + row * stride[0], width);
        BYTE* dstUV = frame->nv12.data() + static_cast<size_t>(width) * height;
        if (nv12) {
            for (ULONG row = 0; row < height / 2; ++row)
                memcpy(dstUV + row * width, source[1] + row * stride[1], width);
        } else {
            const BYTE* sourceU = i420 ? source[1] : source[2];
            const BYTE* sourceV = i420 ? source[2] : source[1];
            const int strideU = i420 ? stride[1] : stride[2];
            const int strideV = i420 ? stride[2] : stride[1];
            for (ULONG row = 0; row < height / 2; ++row) {
                for (ULONG col = 0; col < width / 2; ++col) {
                    dstUV[row * width + col * 2] = sourceU[row * strideU + col];
                    dstUV[row * width + col * 2 + 1] = sourceV[row * strideV + col];
                }
            }
        }
    }
    qcap2_rcbuffer_unlock_data(sourceBuffer);
    return frame;
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

static QRETURN on_decoder_video_cb(
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
      pClient(nullptr),
      m_nVideoWidth(0), m_nVideoHeight(0), m_dVideoFrameRate(0.0), m_nVideoEncoderFormat(0),
      m_frameCount(0), m_bDisplayEnabled(true),
      m_pushFrameCount(0), m_decFrameCount(0),
      // AI init
      m_bSendBuffer(false), m_lastProcessTime(0.0), m_bFrameReady(false)
{
    m_pPendingUpdate = std::make_shared<std::atomic<bool>>(false);
    m_displayFrameCount = 0;
}

ChannelContext::~ChannelContext() {
    stop();
}

bool ChannelContext::enqueueAIFrame(const std::shared_ptr<SharedFrame>& frame)
{
    if (!frame) return false;
    std::lock_guard<std::mutex> queueLocker(m_aiQueueMutex);
    if (m_aiFrames.size() == 2) m_aiFrames.pop_front();
    m_aiFrames.push_back(frame);
    return true;
}

std::shared_ptr<SharedFrame> ChannelContext::takeLatestAIFrame()
{
    std::lock_guard<std::mutex> queueLocker(m_aiQueueMutex);
    if (m_aiFrames.empty()) return nullptr;
    std::shared_ptr<SharedFrame> pLatest = m_aiFrames.back();
    m_aiFrames.clear();
    return pLatest;
}

bool ChannelContext::enqueueDisplayFrame(const std::shared_ptr<SharedFrame>& frame)
{
    if (!frame) return false;
    std::lock_guard<std::mutex> queueLocker(m_aiQueueMutex);
    if (m_displayFrames.size() == 2) m_displayFrames.pop_front();
    m_displayFrames.push_back(frame);
    return true;
}

std::shared_ptr<SharedFrame> ChannelContext::takeLatestDisplayFrame()
{
    std::lock_guard<std::mutex> queueLocker(m_aiQueueMutex);
    if (m_displayFrames.empty()) return nullptr;
    std::shared_ptr<SharedFrame> pLatest = m_displayFrames.back();
    m_displayFrames.clear();
    return pLatest;
}

bool ChannelContext::start() {
    QMutexLocker locker(&m_mutex);

    qDebug() << "Starting channel" << channelId << "URL:" << url;

    // QCAP owns the RTSP decoder.  The decoded SYSBUF is delivered through
    // on_decoder_video_cb below; there is no raw-stream callback or external
    // qcap2 decoder/scaler in this pipeline.
    QRESULT qres = QCAP_CREATE_BROADCAST_CLIENT(
        channelId, url.toLatin1().data(), &pClient,
        QCAP_DECODER_TYPE_ZZNVCODEC, nullptr, FALSE, FALSE);
    if (qres != QCAP_RS_SUCCESSFUL) {
        qCritical() << "QCAP_CREATE_BROADCAST_CLIENT failed for CH" << channelId << "qres =" << qres;
        return false;
    }

    // Match NetReceiver_8ch: receive the decoded video callback directly.
    QCAP_REGISTER_BROADCAST_CLIENT_CONNECTED_CALLBACK(pClient, on_connected_callback, this);
    QCAP_REGISTER_VIDEO_DECODER_BROADCAST_CLIENT_CALLBACK(pClient, on_decoder_video_cb, this);
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

void ChannelContext::cleanupPipeline() {
    std::lock_guard<std::mutex> queueLocker(m_aiQueueMutex);
    m_aiFrames.clear();
    m_displayFrames.clear();
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
    Q_UNUSED(bVideoIsInterleaved);

    bool need_cleanup = false;
    {
        std::lock_guard<std::mutex> queueLocker(m_aiQueueMutex);
        if (!m_aiFrames.empty() || !m_displayFrames.empty()) {
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

    return QCAP_RT_OK;
}

QRETURN ChannelContext::onDecodedVideoFrame(
    double dSampleTime, BYTE* pFrameBuffer, ULONG nFrameBufferLen)
{
    Q_UNUSED(dSampleTime);

    ULONG width = 0;
    ULONG height = 0;
    bool bSendBuffer = false;
    bool bDisplayEnabled = false;
    {
        QMutexLocker locker(&m_mutex);
        if (!pClient || m_nVideoWidth == 0 || m_nVideoHeight == 0) {
            return QCAP_RT_OK;
        }
        width = m_nVideoWidth;
        height = m_nVideoHeight;
        bSendBuffer = m_bSendBuffer && g_pMainwindow && g_pMainwindow->ai_running.load();
        bDisplayEnabled = m_bDisplayEnabled;
    }

    if (!bSendBuffer && !bDisplayEnabled) return QCAP_RT_OK;

    // Complete the application-owned copy before this callback returns. The
    // QCAP callback buffer is only locked/unlocked here; it is never released
    // by this application.
    qcap2_rcbuffer_t* pQcapFrame = qcap2_rcbuffer_cast(pFrameBuffer, nFrameBufferLen);
    std::shared_ptr<SharedFrame> frame = pQcapFrame
        ? copyQcapFrame(channelId, pQcapFrame)
        : copyRawNV12Frame(channelId, pFrameBuffer, nFrameBufferLen, width, height);
    if (!frame) {
        qWarning() << "[QCAP decoder] CH" << channelId
                   << "cannot copy decoded SYSBUF:" << nFrameBufferLen
                   << "bytes for" << width << "x" << height;
        return QCAP_RT_OK;
    }

    // Both consumers receive a shared ownership reference only after the copy
    // above is complete. Their queues do not contain any QCAP rcbuffer.
    if (bDisplayEnabled) {
        enqueueDisplayFrame(frame);
    }
    if (bSendBuffer) {
        if (enqueueAIFrame(frame)) {
            g_pMainwindow->cv.notify_one();
        }
    }

    if (!m_fpsTimer.isValid()) m_fpsTimer.start();
    ++m_decFrameCount;
    if (m_fpsTimer.elapsed() >= 2000) {
        m_decFrameCount = 0;
        m_fpsTimer.restart();
    }
    {
        QMutexLocker locker(&m_mutex);
        ++m_frameCount;
    }
    return QCAP_RT_OK;
}



// ── MainWindow Implementation ───────────────────────────────────────────────
MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), m_bFullscreen(false),
      // AI init
      m_bShowOverlay(true), m_bHalfRefreshRate(false),
      layer1Model0Handle(nullptr), layer1Model1Handle(nullptr), layer1Model2Handle(nullptr), flag(1),
      ai_running(false), pAiThread(nullptr), pLayer1Model0Thread(nullptr), pLayer1Model1Thread(nullptr), pLayer1Model2Thread(nullptr), pDisplayThread(nullptr),
      ready_count(0), active_camera_count(0),
      layer1Model0Path(qEnvironmentVariable("QDEEP_LAYER1_MODEL_0", QString::fromLatin1(kLayer1Model0))),
      layer1Model1Path(qEnvironmentVariable("QDEEP_LAYER1_MODEL_1", QString::fromLatin1(kLayer1Model1))),
      layer1Model2Path(qEnvironmentVariable("QDEEP_LAYER1_MODEL_2", QString::fromLatin1(kLayer1Model2))),
      layer1Model0Ready(false), layer1Model1Ready(false), layer1Model2Ready(false),
      layer1Model0NextChannel(0), layer1Model1NextChannel(0), layer1Model2NextChannel(0)
{
    setWindowTitle("QCAP Multichannel RTSP + Layer 1 model_0 / model_1 / model_2");
    resize(1280, 720);

    g_pMainwindow = this;

    // ── Initialize AI members ────────────────────────────────────────────
    layer1Model0BoxLists.assign(MAX_BATCH, nullptr);
    layer1Model1BoxLists.assign(MAX_BATCH, nullptr);
    layer1Model2BoxLists.assign(MAX_BATCH, nullptr);
    layer1Model0Frames.assign(MAX_BATCH, nullptr);
    layer1Model1Frames.assign(MAX_BATCH, nullptr);
    layer1Model2Frames.assign(MAX_BATCH, nullptr);

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

    grpLayout->addWidget(new QLabel("Channel Count (1-8):"));
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

    chkShowOverlay = new QCheckBox("Show AI Detection Boxes", grpConfig);
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
    onChannelCountChanged(9);
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
            tableUrls->setItem(i, 1, new QTableWidgetItem("rtsp://root:root@192.168.190.96:554/session0.mpg"));
        }
    }
}

void MainWindow::onBtnStartClicked()
{
    stopAllChannels();
    clearGrid();

    int count = spinChannelCount->value();
    const int cols = 3; // fixed 3 x 3 display grid

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
            const int cols = 3; // retain the 3 x 3 grid in fullscreen too

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
        layer1Model0BoxLists[i] = new QDEEP_API::QDEEP_OBJECT_DETECT_BOUNDING_BOX[BOX_SIZE];
        layer1Model1BoxLists[i] = new QDEEP_API::QDEEP_OBJECT_DETECT_BOUNDING_BOX[BOX_SIZE];
        layer1Model2BoxLists[i] = new QDEEP_API::QDEEP_OBJECT_DETECT_BOUNDING_BOX[BOX_SIZE];
    }

    const auto startModel = [this](const char* name, const QString& path, ULONG config, void** handle) {
        if (!QFileInfo::exists(path)) {
            qCritical() << "[" << name << "model] not found:" << path;
            return false;
        }
        const QRESULT result = QDEEP_API::QDEEP_CREATE_BATCH_OBJECT_DETECT(
            QDEEP_API::QDEEP_GPU_TYPE_NVIDIA, 0, config, path.toLocal8Bit().data(),
            handle, flag, QDEEP_MODEL_BATCH_SIZE);
        if (result != QCAP_RS_SUCCESSFUL || !*handle) {
            qCritical() << "[" << name << "model] create failed:" << path << "result=" << result;
            return false;
        }
        const QRESULT startResult = QDEEP_API::QDEEP_START_OBJECT_DETECT(*handle);
        if (startResult != QCAP_RS_SUCCESSFUL) {
            qCritical() << "[" << name << "model] start failed:" << startResult;
            QDEEP_API::QDEEP_DESTROY_OBJECT_DETECT(*handle);
            *handle = nullptr;
            return false;
        }
        return true;
    };

    layer1Model0Ready = startModel("layer1/model_0", layer1Model0Path,
               QDEEP_API::QDEEP_OBJECT_DETECT_CONFIG_MODEL_CUSTOMIZED_LITE_NEW, &layer1Model0Handle);
    layer1Model1Ready = startModel("layer1/model_1", layer1Model1Path,
               QDEEP_API::QDEEP_OBJECT_DETECT_CONFIG_MODEL_CUSTOMIZED_LITE_NEW, &layer1Model1Handle);
    layer1Model2Ready = startModel("layer1/model_2", layer1Model2Path,
               QDEEP_API::QDEEP_OBJECT_DETECT_CONFIG_MODEL_CUSTOMIZED_LITE_NEW, &layer1Model2Handle);
}

void MainWindow::uninit_models()
{
    yolo_stop();
    const auto stopModel = [](void*& handle) {
        if (!handle) return;
        QDEEP_API::QDEEP_STOP_OBJECT_DETECT(handle);
        QDEEP_API::QDEEP_DESTROY_OBJECT_DETECT(handle);
        handle = nullptr;
    };
    stopModel(layer1Model0Handle);
    stopModel(layer1Model1Handle);
    stopModel(layer1Model2Handle);
    layer1Model0Ready = false;
    layer1Model1Ready = false;
    layer1Model2Ready = false;
    for (size_t i = 0; i < MAX_BATCH; ++i) {
        delete[] layer1Model0BoxLists[i];
        delete[] layer1Model1BoxLists[i];
        delete[] layer1Model2BoxLists[i];
        layer1Model0BoxLists[i] = nullptr;
        layer1Model1BoxLists[i] = nullptr;
        layer1Model2BoxLists[i] = nullptr;
    }
}

void MainWindow::yolo_start()
{
    if (ai_running.load() || !layer1Model0Handle) return;

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
        qDebug() << "[Warning] No active cameras found!";
        return;
    }

    ai_running.store(true);
    pAiThread = new std::thread(&MainWindow::ai_dispatch_thread, this);
    pLayer1Model0Thread = new std::thread(&MainWindow::layer1_model_0_inference_thread, this);
    if (layer1Model1Ready && layer1Model1Handle)
        pLayer1Model1Thread = new std::thread(&MainWindow::layer1_model_1_inference_thread, this);
    if (layer1Model2Ready && layer1Model2Handle)
        pLayer1Model2Thread = new std::thread(&MainWindow::layer1_model_2_inference_thread, this);
    pDisplayThread = new std::thread(&MainWindow::display_thread, this);
}

void MainWindow::yolo_stop()
{
    ai_running.store(false);
    cv.notify_all();
    frame_cv.notify_all();

    if (pAiThread) {
        if (pAiThread->joinable()) {
            pAiThread->join();
        }
        delete pAiThread;
        pAiThread = nullptr;
    }
    if (pLayer1Model1Thread) {
        if (pLayer1Model1Thread->joinable()) pLayer1Model1Thread->join();
        delete pLayer1Model1Thread;
        pLayer1Model1Thread = nullptr;
    }
    if (pLayer1Model2Thread) {
        if (pLayer1Model2Thread->joinable()) pLayer1Model2Thread->join();
        delete pLayer1Model2Thread;
        pLayer1Model2Thread = nullptr;
    }
    if (pLayer1Model0Thread) {
        if (pLayer1Model0Thread->joinable()) pLayer1Model0Thread->join();
        delete pLayer1Model0Thread;
        pLayer1Model0Thread = nullptr;
    }
    if (pDisplayThread) {
        if (pDisplayThread->joinable()) pDisplayThread->join();
        delete pDisplayThread;
        pDisplayThread = nullptr;
    }

    for (ChannelContext *ctx : channels) {
        ctx->m_bSendBuffer = false;
    }
    {
        std::lock_guard<std::mutex> lock(frame_mtx);
        std::fill(layer1Model0Frames.begin(), layer1Model0Frames.end(), nullptr);
        std::fill(layer1Model1Frames.begin(), layer1Model1Frames.end(), nullptr);
        std::fill(layer1Model2Frames.begin(), layer1Model2Frames.end(), nullptr);
    }
}

void MainWindow::submitFrame(const std::shared_ptr<SharedFrame>& frame)
{
    if (!frame || frame->channelId < 0 || frame->channelId >= MAX_BATCH) return;
    {
        std::lock_guard<std::mutex> lock(frame_mtx);
        const int channelId = frame->channelId;
        // Each model entry is a depth-one, drop-oldest queue.  Replacing it
        // releases only this queue's shared_ptr; a detector that already took
        // the old frame can finish safely.
        if (ai_running.load()) {
            if (layer1Model0Ready && layer1Model0Handle) layer1Model0Frames[channelId] = frame;
            if (layer1Model1Ready && layer1Model1Handle) layer1Model1Frames[channelId] = frame;
            if (layer1Model2Ready && layer1Model2Handle) layer1Model2Frames[channelId] = frame;
        }
    }
    frame_cv.notify_all();
}

std::shared_ptr<SharedFrame> MainWindow::takeLatestFrame(
    std::vector<std::shared_ptr<SharedFrame>>& frames, int& nextChannel)
{
    for (int offset = 0; offset < MAX_BATCH; ++offset) {
        const int channelId = (nextChannel + offset) % MAX_BATCH;
        if (frames[channelId]) {
            std::shared_ptr<SharedFrame> frame = frames[channelId];
            frames[channelId].reset();
            nextChannel = (channelId + 1) % MAX_BATCH;
            return frame;
        }
    }
    return nullptr;
}

void MainWindow::recordInferenceTiming(InferenceTimingStats& stats, double elapsedMs)
{
    std::lock_guard<std::mutex> lock(stats.mutex);
    ++stats.sampleCount;
    stats.totalMs += elapsedMs;
    stats.minMs = std::min(stats.minMs, elapsedMs);
    stats.maxMs = std::max(stats.maxMs, elapsedMs);
}

void MainWindow::printInferenceTimingStats()
{
    const auto formatAndReset = [](const char* name, InferenceTimingStats& stats) {
        std::lock_guard<std::mutex> lock(stats.mutex);
        QString output;
        if (stats.sampleCount == 0) {
            output = QStringLiteral("%1: samples=0").arg(QString::fromLatin1(name));
        } else {
            output = QStringLiteral("%1: samples=%2 min_ms=%3 max_ms=%4 avg_ms=%5")
                         .arg(QString::fromLatin1(name))
                         .arg(stats.sampleCount)
                         .arg(stats.minMs, 0, 'f', 3)
                         .arg(stats.maxMs, 0, 'f', 3)
                         .arg(stats.totalMs / stats.sampleCount, 0, 'f', 3);
        }
        stats.sampleCount = 0;
        stats.totalMs = 0.0;
        stats.minMs = std::numeric_limits<double>::infinity();
        stats.maxMs = 0.0;
        return output;
    };

    qInfo().noquote() << QStringLiteral("[QDEEP timing: last 3s]\n%1\n%2\n%3")
                            .arg(formatAndReset("layer1/model_0", layer1Model0TimingStats))
                            .arg(formatAndReset("layer1/model_1", layer1Model1TimingStats))
                            .arg(formatAndReset("layer1/model_2", layer1Model2TimingStats));
}

void MainWindow::layer1_model_0_inference_thread()
{
    while (ai_running.load()) {
        std::shared_ptr<SharedFrame> frame;
        {
            std::unique_lock<std::mutex> lock(frame_mtx);
            frame_cv.wait(lock, [this] {
                if (!ai_running.load()) return true;
                return std::any_of(layer1Model0Frames.begin(), layer1Model0Frames.end(),
                                   [](const std::shared_ptr<SharedFrame>& frame) { return frame != nullptr; });
            });
            if (!ai_running.load()) break;
            frame = takeLatestFrame(layer1Model0Frames, layer1Model0NextChannel);
        }
        if (!frame) continue;

        ULONG colorSpace = QDEEP_API::QDEEP_COLORSPACE_TYPE_NV12;
        ULONG width = frame->width;
        ULONG height = frame->height;
        BYTE* buffer = frame->nv12.data();
        ULONG bufferLength = static_cast<ULONG>(frame->nv12.size());
        QDEEP_API::QDEEP_OBJECT_DETECT_BOUNDING_BOX* boxList = layer1Model0BoxLists[frame->channelId];
        ULONG boxSize = BOX_SIZE;
        QElapsedTimer inferenceTimer;
        inferenceTimer.start();
        QRESULT result = QDEEP_API::QDEEP_SET_VIDEO_OBJECT_DETECT_BATCH_UNCOMPRESSION_BUFFER(
            layer1Model0Handle, &colorSpace, &width, &height, &buffer, &bufferLength,
            &boxList, &boxSize, 1);
        recordInferenceTiming(layer1Model0TimingStats, inferenceTimer.nsecsElapsed() / 1000000.0);
        if (result != QCAP_RS_SUCCESSFUL) {
            qWarning() << "[layer1/model_0] inference failed:" << result
                       << "for native frame" << width << "x" << height;
            continue;
        }

        std::lock_guard<std::mutex> lock(draw_mtx);
        std::vector<DrawBox>& drawList = layer1Model0DrawBoxes[frame->channelId];
        drawList.clear();
        for (ULONG i = 0; i < boxSize; ++i) {
            const auto& box = layer1Model0BoxLists[frame->channelId][i];
            if (box.fProbability < LAYER1_MODEL_CONFIDENCE) continue;
            drawList.push_back({static_cast<int>(box.nX), static_cast<int>(box.nY),
                                static_cast<int>(box.nWidth), static_cast<int>(box.nHeight),
                                static_cast<int>(box.nClassID), box.fProbability,
                                QString("model_0 class %1").arg(box.nClassID)});
        }
    }
}

void MainWindow::layer1_model_1_inference_thread()
{
    while (ai_running.load()) {
        std::shared_ptr<SharedFrame> frame;
        {
            std::unique_lock<std::mutex> lock(frame_mtx);
            frame_cv.wait(lock, [this] {
                if (!ai_running.load()) return true;
                return std::any_of(layer1Model1Frames.begin(), layer1Model1Frames.end(),
                                   [](const std::shared_ptr<SharedFrame>& frame) { return frame != nullptr; });
            });
            if (!ai_running.load()) break;
            frame = takeLatestFrame(layer1Model1Frames, layer1Model1NextChannel);
        }
        if (!frame) continue;

        ULONG colorSpace = QDEEP_API::QDEEP_COLORSPACE_TYPE_NV12;
        ULONG width = frame->width;
        ULONG height = frame->height;
        BYTE* buffer = frame->nv12.data();
        ULONG bufferLength = static_cast<ULONG>(frame->nv12.size());
        QDEEP_API::QDEEP_OBJECT_DETECT_BOUNDING_BOX* boxList = layer1Model1BoxLists[frame->channelId];
        ULONG boxSize = BOX_SIZE;
        QElapsedTimer inferenceTimer;
        inferenceTimer.start();
        QRESULT result = QDEEP_API::QDEEP_SET_VIDEO_OBJECT_DETECT_BATCH_UNCOMPRESSION_BUFFER(
            layer1Model1Handle, &colorSpace, &width, &height, &buffer, &bufferLength,
            &boxList, &boxSize, 1);
        recordInferenceTiming(layer1Model1TimingStats, inferenceTimer.nsecsElapsed() / 1000000.0);
        if (result != QCAP_RS_SUCCESSFUL) {
            qWarning() << "[layer1/model_1] inference failed:" << result
                       << "for native frame" << width << "x" << height;
            continue;
        }

        std::lock_guard<std::mutex> lock(draw_mtx);
        std::vector<DrawBox>& drawList = layer1Model1DrawBoxes[frame->channelId];
        drawList.clear();
        for (ULONG i = 0; i < boxSize; ++i) {
            const auto& box = layer1Model1BoxLists[frame->channelId][i];
            if (box.fProbability < LAYER1_MODEL_CONFIDENCE) continue;
            drawList.push_back({static_cast<int>(box.nX), static_cast<int>(box.nY),
                                static_cast<int>(box.nWidth), static_cast<int>(box.nHeight),
                                static_cast<int>(box.nClassID), box.fProbability,
                                QString("model_1 class %1").arg(box.nClassID)});
        }
    }
}

void MainWindow::layer1_model_2_inference_thread()
{
    while (ai_running.load()) {
        std::shared_ptr<SharedFrame> frame;
        {
            std::unique_lock<std::mutex> lock(frame_mtx);
            frame_cv.wait(lock, [this] {
                return !ai_running.load() || std::any_of(
                    layer1Model2Frames.begin(), layer1Model2Frames.end(),
                    [](const std::shared_ptr<SharedFrame>& value) { return value != nullptr; });
            });
            if (!ai_running.load()) break;
            frame = takeLatestFrame(layer1Model2Frames, layer1Model2NextChannel);
        }
        if (!frame) continue;

        ULONG colorSpace = QDEEP_API::QDEEP_COLORSPACE_TYPE_NV12;
        ULONG width = frame->width;
        ULONG height = frame->height;
        BYTE* buffer = frame->nv12.data();
        ULONG bufferLength = static_cast<ULONG>(frame->nv12.size());
        QDEEP_API::QDEEP_OBJECT_DETECT_BOUNDING_BOX* boxList = layer1Model2BoxLists[frame->channelId];
        ULONG boxSize = BOX_SIZE;
        QElapsedTimer inferenceTimer;
        inferenceTimer.start();
        const QRESULT result = QDEEP_API::QDEEP_SET_VIDEO_OBJECT_DETECT_BATCH_UNCOMPRESSION_BUFFER(
            layer1Model2Handle, &colorSpace, &width, &height, &buffer, &bufferLength,
            &boxList, &boxSize, 1);
        recordInferenceTiming(layer1Model2TimingStats, inferenceTimer.nsecsElapsed() / 1000000.0);
        if (result != QCAP_RS_SUCCESSFUL) {
            qWarning() << "[layer1/model_2] inference failed:" << result
                       << "for native frame" << width << "x" << height;
            continue;
        }

        std::lock_guard<std::mutex> lock(draw_mtx);
        std::vector<DrawBox>& drawList = layer1Model2DrawBoxes[frame->channelId];
        drawList.clear();
        for (ULONG i = 0; i < boxSize; ++i) {
            const auto& box = layer1Model2BoxLists[frame->channelId][i];
            if (box.fProbability < LAYER1_MODEL_CONFIDENCE) continue;
            drawList.push_back({static_cast<int>(box.nX), static_cast<int>(box.nY),
                                static_cast<int>(box.nWidth), static_cast<int>(box.nHeight),
                                static_cast<int>(box.nClassID), box.fProbability,
                                QString("model_2 class %1").arg(box.nClassID)});
        }
    }
}

void MainWindow::display_thread()
{
    while (ai_running.load()) {
        bool displayedFrame = false;
        for (ChannelContext* ctx : channels) {
            std::shared_ptr<SharedFrame> frame = ctx->takeLatestDisplayFrame();
            if (!frame) continue;
            postDisplayFrame(ctx, frame);
            displayedFrame = true;
        }
        if (!displayedFrame) std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
}

void MainWindow::ai_dispatch_thread()
{
    int nextChannel = 0;
    auto nextTimingReport = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (ai_running.load()) {
        const auto now = std::chrono::steady_clock::now();
        if (now >= nextTimingReport) {
            printInferenceTimingStats();
            nextTimingReport = now + std::chrono::seconds(3);
        }

        std::shared_ptr<SharedFrame> frame;
        for (int offset = 0; offset < MAX_BATCH; ++offset) {
            const int channelId = (nextChannel + offset) % MAX_BATCH;
            ChannelContext* ctx = nullptr;
            for (ChannelContext* channel : channels) {
                if (channel->channelId == channelId) {
                    ctx = channel;
                    break;
                }
            }
            if (!ctx) continue;
            frame = ctx->takeLatestAIFrame();
            if (!frame) continue;
            nextChannel = (channelId + 1) % MAX_BATCH;
            break;
        }
        if (frame) {
            submitFrame(frame);
        } else {
            std::unique_lock<std::mutex> lock(mtx);
            cv.wait_for(lock, std::chrono::milliseconds(2), [this] { return !ai_running.load(); });
        }
    }
}
