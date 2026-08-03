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

// Layer 1 runs model_0 once for all channels. Layer 2 uses a face-landmark
// detector per channel after model_0 has completed for that frame.
static const char* kLayer1Model0 =
     "/home/nvidia/Documents/thor_receive_rtsp_ai/new_model/taiwantraffic_batch8/QDEEP.OD.TAIWAN.TRAFFIC.C4.TINY.CFG";
    // "/home/nvidia/Downloads/MY/model/people/QDEEP.OD.TINY.PERSON.V10N.CFG";
static const char* kLayer1Model1 = kLayer1Model0;
static const char* kLayer1Model2 = kLayer1Model0;
static const char* kLayer2FaceModel =
    // "/home/nvidia/Documents/thor_receive_rtsp_ai/new_model/face/QDEEP.OD.FACE.LANDMARK.5KPS.CFG";
    "/home/nvidia/Downloads/MY/model/face/QDEEP.OD.FACE.LANDMARK.5KPS.CFG";
static const char* kLayer2PlateModel =
    // "/home/nvidia/Documents/thor_receive_rtsp_ai/new_model/lpr_batch8/QDEEP.OD.LICENSE.PLATE.RECOGNITION.LAW.TINY.CFG";
    "/home/nvidia/Downloads/MY/model/lpr/QDEEP.OD.LICENSE.PLATE.RECOGNITION.LAW.TINY.CFG";
static const char* kLayer3PersonModel =
    // "/home/nvidia/Documents/thor_receive_rtsp_ai/new_model/person_batch8/QDEEP.OD.TINY.PERSON.V10N.CFG";
    "/home/nvidia/Downloads/MY/model/people/QDEEP.OD.TINY.PERSON.V10N.CFG";

extern "C" {
QDEEP_EXT_API QRESULT QDEEP_EXPORT QDEEP_GET_OBJECT_DETECT_RESERVED_STATUS(PVOID pDetector, ULONG* pCheckNum);
}

// 將 OpenCV 的 BGR 三通道影像複製並轉成可交由 Qt 顯示的 QImage。
QImage cvMatToQImage(const cv::Mat& mat) {
    if (mat.type() == CV_8UC3) {
        return QImage(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_RGB888).rgbSwapped().copy();
    }
    return QImage();
}

// 從 QDEEP 車牌框的 feature vector 取出以 NUL 結尾的車牌文字。
static QString plateTextFromFeatureVector(const QDEEP_API::QDEEP_OBJECT_DETECT_BOUNDING_BOX& box)
{
    const char* text = reinterpret_cast<const char*>(box.fFeatureVectors);
    const size_t capacity = QDEEP_MAX_FEATURE_VECTOR_SIZE * sizeof(float);
    size_t length = 0;
    while (length < capacity && text[length] != '\0') ++length;
    return QString::fromUtf8(text, static_cast<int>(length)).trimmed();
}

// Display is fed from an immutable CPU frame, never from a QCAP rcbuffer.
// Each QDEEP worker holds a separate shared_ptr reference until its synchronous
// API call completes, so no consumer can free pixels used by another consumer.
// 將最新解碼畫面與各層推論框以 OpenCV 疊圖，並排入 Qt UI 執行緒更新。
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
        std::vector<DrawBox> faceBoxes;
        std::vector<DrawBox> plateBoxes;
        std::vector<DrawBox> layer3Model0Boxes;
        std::vector<DrawBox> layer3Model1Boxes;
        {
            std::lock_guard<std::mutex> lock(g_pMainwindow->draw_mtx);
            model0Boxes = g_pMainwindow->layer1Model0DrawBoxes[ctx->channelId];
            model1Boxes = g_pMainwindow->layer1Model1DrawBoxes[ctx->channelId];
            model2Boxes = g_pMainwindow->layer1Model2DrawBoxes[ctx->channelId];
            faceBoxes = g_pMainwindow->layer2FaceDrawBoxes[ctx->channelId];
            plateBoxes = g_pMainwindow->layer2PlateDrawBoxes[ctx->channelId];
            layer3Model0Boxes = g_pMainwindow->layer3Model0DrawBoxes[ctx->channelId];
            layer3Model1Boxes = g_pMainwindow->layer3Model1DrawBoxes[ctx->channelId];
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
            for (const DrawKeypoint& point : box.keypoints) {
                if (point.x < 0 || point.y < 0 || point.x >= bgrMat.cols || point.y >= bgrMat.rows) continue;
                cv::circle(bgrMat, cv::Point(point.x, point.y), 3, color, cv::FILLED, cv::LINE_AA);
            }
            }
        };
        drawBoxes(model0Boxes, cv::Scalar(0, 255, 0));
        drawBoxes(model1Boxes, cv::Scalar(0, 255, 255));
        drawBoxes(model2Boxes, cv::Scalar(255, 0, 255));
        drawBoxes(faceBoxes, cv::Scalar(255, 128, 0));
        drawBoxes(plateBoxes, cv::Scalar(0, 215, 255));
        drawBoxes(layer3Model0Boxes, cv::Scalar(255, 255, 0));
        drawBoxes(layer3Model1Boxes, cv::Scalar(128, 255, 128));
    }

    const QImage image = cvMatToQImage(bgrMat);
    QPointer<QLabel> safeLabel = ctx->m_pLabel;
    std::shared_ptr<std::atomic<bool>> pending = ctx->m_pPendingUpdate;
    QMetaObject::invokeMethod(ctx->m_pLabel, [safeLabel, image, pending]() {
        if (safeLabel) safeLabel->setPixmap(QPixmap::fromImage(image));
        if (pending) pending->store(false);
    }, Qt::QueuedConnection);
}

// 複製 raw NV12 解碼資料成應用程式自行持有、可跨 thread 共用的 frame。
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
// 鎖定 QCAP rcbuffer 後轉拷貝為標準 NV12，並在離開前解鎖 QCAP buffer。
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
// 將 QCAP 連線成功 callback 轉交給對應的 ChannelContext。
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

// 將 QCAP 解碼完成的影像 callback 轉交給對應的 ChannelContext。
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

// 將 QCAP 連線/接收失敗 callback 轉交給對應的 ChannelContext。
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
// 建立單一路 RTSP 的狀態、佇列與畫面更新旗標。
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

// 解構前確保停止此路 RTSP client 並釋放其 pipeline。
ChannelContext::~ChannelContext() {
    stop();
}

// 將最新畫面放入 AI 佇列；佇列滿時丟棄最舊畫面避免解碼端阻塞。
bool ChannelContext::enqueueAIFrame(const std::shared_ptr<SharedFrame>& frame)
{
    if (!frame) return false;
    std::lock_guard<std::mutex> queueLocker(m_aiQueueMutex);
    if (m_aiFrames.size() == 2) m_aiFrames.pop_front();
    m_aiFrames.push_back(frame);
    return true;
}

// 取出 AI 佇列中最新畫面並清空較舊畫面。
std::shared_ptr<SharedFrame> ChannelContext::takeLatestAIFrame()
{
    std::lock_guard<std::mutex> queueLocker(m_aiQueueMutex);
    if (m_aiFrames.empty()) return nullptr;
    std::shared_ptr<SharedFrame> pLatest = m_aiFrames.back();
    m_aiFrames.clear();
    return pLatest;
}

// 將最新畫面放入顯示佇列；佇列滿時丟棄最舊畫面以維持低延遲。
bool ChannelContext::enqueueDisplayFrame(const std::shared_ptr<SharedFrame>& frame)
{
    if (!frame) return false;
    std::lock_guard<std::mutex> queueLocker(m_aiQueueMutex);
    if (m_displayFrames.size() == 2) m_displayFrames.pop_front();
    m_displayFrames.push_back(frame);
    return true;
}

// 取出顯示佇列中最新畫面並清空較舊畫面。
std::shared_ptr<SharedFrame> ChannelContext::takeLatestDisplayFrame()
{
    std::lock_guard<std::mutex> queueLocker(m_aiQueueMutex);
    if (m_displayFrames.empty()) return nullptr;
    std::shared_ptr<SharedFrame> pLatest = m_displayFrames.back();
    m_displayFrames.clear();
    return pLatest;
}

// 建立、註冊 callback 並啟動單一路 QCAP RTSP broadcast client。
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

// 清空此路尚未消費的 AI 與顯示畫面，不操作 QCAP 所有權。
void ChannelContext::cleanupPipeline() {
    std::lock_guard<std::mutex> queueLocker(m_aiQueueMutex);
    m_aiFrames.clear();
    m_displayFrames.clear();
}

// 依序停止並銷毀此路 QCAP client，再清除本地 frame 佇列。
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

// 記錄 QCAP 失敗狀態，供 UI 定時顯示該 RTSP 路的錯誤資訊。
QRETURN ChannelContext::onFail(UINT iSessionNum, QRESULT nErrorStatus, DWORD nErrorCode) {
    Q_UNUSED(iSessionNum);
    QMutexLocker locker(&m_mutex);
    qCritical() << "CH" << channelId << "Broadcast client failure callback! Status:" << nErrorStatus << "Code:" << nErrorCode;
    m_statusInfo = QString("Disconnected (Error 0x%1)").arg(nErrorStatus, 8, 16, QChar('0'));
    return QCAP_RT_OK;
}

// 安全地切換此路是否把解碼畫面送入 display 佇列。
void ChannelContext::setDisplayEnabled(bool enabled) {
    QMutexLocker locker(&m_mutex);
    m_bDisplayEnabled = enabled;
}

// 保存新連線的影像格式與尺寸；重連時清掉前一次遺留的 frame。
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

// 複製 decoder callback 的畫面，分送到顯示與 AI 佇列，絕不保留 QCAP buffer。
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
// 初始化 UI、預設模型路徑、控制項與 QDEEP 啟動期檢查。
MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), m_bShowOverlay(true), m_bEnableFaceModel(true), m_bEnablePlateModel(true), m_bFullscreen(false),
      // AI init
      m_bHalfRefreshRate(false),
      flag(0), roundExpected(0), roundCompleted(0), ai_running(false), pAiThread(nullptr), pDisplayThread(nullptr),
      ready_count(0), active_camera_count(0),
      layer1Model0Path(qEnvironmentVariable("QDEEP_LAYER1_MODEL_0", QString::fromLatin1(kLayer1Model0))),
      layer1Model1Path(qEnvironmentVariable("QDEEP_LAYER1_MODEL_1", QString::fromLatin1(kLayer1Model1))),
      layer1Model2Path(qEnvironmentVariable("QDEEP_LAYER1_MODEL_2", QString::fromLatin1(kLayer1Model2))),
      layer2FaceModelPath(qEnvironmentVariable("QDEEP_LAYER2_FACE_MODEL", QString::fromLatin1(kLayer2FaceModel))),
      layer2PlateModelPath(qEnvironmentVariable("QDEEP_LAYER2_PLATE_MODEL", QString::fromLatin1(kLayer2PlateModel))),
      layer3PersonModelPath(qEnvironmentVariable("QDEEP_LAYER3_PERSON_MODEL", QString::fromLatin1(kLayer3PersonModel)))
{
    setWindowTitle("QCAP Multichannel RTSP + Layer 1 / Layer 2 face + plate");
    resize(1280, 720);

    g_pMainwindow = this;

    // ── Initialize AI members ────────────────────────────────────────────
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

    grpLayout->addWidget(new QLabel(QString("Channel Count (1-%1):").arg(MAX_BATCH)));
    spinChannelCount = new QSpinBox(grpConfig);
    spinChannelCount->setRange(1, MAX_BATCH);
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

    chkEnableFaceModel = new QCheckBox("Enable FR (Layer 2 Face)", grpConfig);
    chkEnableFaceModel->setChecked(true);
    grpLayout->addWidget(chkEnableFaceModel);

    chkEnablePlateModel = new QCheckBox("Enable Plate Recognition (Layer 2 model_1)", grpConfig);
    chkEnablePlateModel->setChecked(true);
    grpLayout->addWidget(chkEnablePlateModel);

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

    // Keep the initial URL rows aligned with the channel-count default.
    onChannelCountChanged(spinChannelCount->value());
    m_bEnableDisplay = true;

    // Signal Slot connections
    connect(spinChannelCount, QOverload<int>::of(&QSpinBox::valueChanged), this, &MainWindow::onChannelCountChanged);
    connect(btnStart, &QPushButton::clicked, this, &MainWindow::onBtnStartClicked);
    connect(btnStop, &QPushButton::clicked, this, &MainWindow::onBtnStopClicked);
    connect(chkEnableDisplay, &QCheckBox::toggled, this, &MainWindow::onDisplayToggled);
    connect(chkShowOverlay, &QCheckBox::toggled, this, &MainWindow::onOverlayToggled);
    connect(chkEnableFaceModel, &QCheckBox::toggled, this, &MainWindow::onFaceModelToggled);
    connect(chkEnablePlateModel, &QCheckBox::toggled, this, &MainWindow::onPlateModelToggled);
    connect(chkHalfRefreshRate, &QCheckBox::toggled, this, &MainWindow::onHalfRefreshRateToggled);

    videoContainer->installEventFilter(this);

    m_timerId = startTimer(1000);

    // ── Initialize QDEEP models ─────────────────────────────────────────
    init_models();
}

// 關閉 AI 與 RTSP 資源，確保所有 worker 都在視窗銷毀前結束。
MainWindow::~MainWindow()
{
    uninit_models();
    stopAllChannels();
}

// ── UI Event Handlers ───────────────────────────────────────────────────────
// 依 UI 路數調整 URL 表格並補上缺少的預設 RTSP URL。
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
            tableUrls->setItem(i, 1, new QTableWidgetItem("rtsp://root:root@192.168.190.128:554/session0.mpg"));
        }
    }
}

// 建立所有 UI channel 與 RTSP client，鎖定設定後啟動 AI pipeline。
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
    chkEnableFaceModel->setEnabled(false);
    chkEnablePlateModel->setEnabled(false);
    btnStart->setEnabled(false);
    btnStop->setEnabled(true);
    lblStatus->setText("Status: Running");

    // Start AI inference
    yolo_start();
}

// 停止 AI、RTSP 與顯示格，並恢復可編輯的啟動設定。
void MainWindow::onBtnStopClicked()
{
    yolo_stop();

    stopAllChannels();
    clearGrid();

    spinChannelCount->setEnabled(true);
    tableUrls->setEnabled(true);
    chkEnableFaceModel->setEnabled(true);
    chkEnablePlateModel->setEnabled(true);
    btnStart->setEnabled(true);
    btnStop->setEnabled(false);
    lblStatus->setText("Status: Stopped");
}

// 停止並刪除目前所有 ChannelContext 與其 QCAP client。
void MainWindow::stopAllChannels()
{
    for (ChannelContext *ctx : channels) {
        ctx->m_bSendBuffer = false;
        delete ctx;
    }
    channels.clear();
}

// 移除並延後刪除所有影片顯示格。
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

// 每秒彙整每一路連線狀態與解碼 FPS 後更新狀態列。
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

// 視窗關閉前停止 AI 與所有 RTSP client。
void MainWindow::closeEvent(QCloseEvent *event)
{
    yolo_stop();
    stopAllChannels();
    event->accept();
}

// 處理影片格雙擊；全螢幕時改為每列兩格的四宮格排列。
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

        // 全螢幕採 2x2 四宮格；回到一般視窗則還原每列三格。
        int count = videoFrames.size();
        if (count > 0) {
            const int cols = m_bFullscreen ? 2 : 3;

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

// 將顯示開關同步套用到目前所有 RTSP channel。
void MainWindow::onDisplayToggled(bool checked)
{
    m_bEnableDisplay = checked;
    for (auto* c : channels) {
        c->setDisplayEnabled(checked);
    }
}

// 切換 OpenCV 是否在顯示畫面上繪製 AI 推論結果。
void MainWindow::onOverlayToggled(bool checked)
{
    m_bShowOverlay = checked;
}

// 儲存下次啟動時是否建立 Layer2 FR/Face model 的設定。
void MainWindow::onFaceModelToggled(bool checked)
{
    m_bEnableFaceModel.store(checked);
}

// 儲存下次啟動時是否建立 Layer2 車牌 model 的設定。
void MainWindow::onPlateModelToggled(bool checked)
{
    m_bEnablePlateModel.store(checked);
}

// 切換顯示更新頻率減半設定。
void MainWindow::onHalfRefreshRateToggled(bool checked)
{
    m_bHalfRefreshRate = checked;
}

// ── QDEEP / YOLO AI Functions ──────────────────────────────────────────────
// 在 app 啟動期執行原有 QDEEP reserved-status 檢查。
void MainWindow::init_models()
{
    // Keep this startup call at application initialization. Layer 1 itself is
    // created at yolo_start(), after the active RTSP channel count is known.
    const QRESULT reservedResult = QDEEP_GET_OBJECT_DETECT_RESERVED_STATUS(
        reinterpret_cast<PVOID>(0xD7CBB416), reinterpret_cast<ULONG*>(0x3B98119E));
    qDebug() << "[AI Log] QDEEP_GET_OBJECT_DETECT_RESERVED_STATUS res:"
             << QString("0x%1").arg(reservedResult, 8, 16, QChar('0'));
}

// 在 app 結束期停止所有 AI worker 與其 QDEEP handle。
void MainWindow::uninit_models()
{
    yolo_stop();
}

// 為每個有效 RTSP channel 建立一個非 batch 的 Layer2 Face/FR worker。
void MainWindow::create_layer2_face_workers()
{
    destroy_layer2_face_workers();
    {
        std::lock_guard<std::mutex> lock(draw_mtx);
        for (int channelId = 0; channelId < MAX_BATCH; ++channelId)
            layer2FaceDrawBoxes[channelId].clear();
    }
    if (!QFileInfo::exists(layer2FaceModelPath)) {
        qCritical() << "[layer2/face] model not found:" << layer2FaceModelPath;
        return;
    }

    for (ChannelContext* ctx : channels) {
        if (!ctx || !ctx->pClient) continue;
        std::unique_ptr<Layer2FaceWorker> worker(new Layer2FaceWorker(ctx->channelId));
        const DWORD faceFlags = QDEEP_API::QDEEP_OBJECT_DETECT_FLAG_FEATURE_VECTOR;
        const QByteArray path = layer2FaceModelPath.toLocal8Bit();
        const QRESULT createResult = QDEEP_API::QDEEP_CREATE_OBJECT_DETECT(
            QDEEP_API::QDEEP_GPU_TYPE_NVIDIA, 0,
            QDEEP_API::QDEEP_OBJECT_DETECT_CONFIG_MODEL_FACE_LANDMARK_5_KEYPOINTS,
            const_cast<CHAR*>(path.constData()), &worker->handle, faceFlags);
        if (createResult != QCAP_RS_SUCCESSFUL || !worker->handle) {
            qCritical() << "[layer2/face] create failed for CH" << (ctx->channelId + 1)
                        << "result=" << createResult;
            continue;
        }
        const QRESULT startResult = QDEEP_API::QDEEP_START_OBJECT_DETECT(worker->handle);
        if (startResult != QCAP_RS_SUCCESSFUL) {
            qCritical() << "[layer2/face] start failed for CH" << (ctx->channelId + 1)
                        << "result=" << startResult;
            QDEEP_API::QDEEP_DESTROY_OBJECT_DETECT(worker->handle);
            worker->handle = nullptr;
            continue;
        }
        worker->ready = true;
        layer2FaceWorkers.push_back(std::move(worker));
    }
}

// 等待所有 Face worker 結束，再停止與銷毀各自的 QDEEP handle。
void MainWindow::destroy_layer2_face_workers()
{
    // This function is called only after ai_running is false and every face
    // thread has been joined, so QDEEP never sees STOP/DESTROY concurrently
    // with SET_VIDEO on the same handle.
    for (const std::unique_ptr<Layer2FaceWorker>& worker : layer2FaceWorkers) {
        if (worker->thread.joinable()) worker->thread.join();
        if (worker->handle) {
            QDEEP_API::QDEEP_STOP_OBJECT_DETECT(worker->handle);
            QDEEP_API::QDEEP_DESTROY_OBJECT_DETECT(worker->handle);
            worker->handle = nullptr;
        }
    }
    layer2FaceWorkers.clear();
}

// 為每個有效 RTSP channel 建立一個非 batch 的 Layer2 車牌 worker。
void MainWindow::create_layer2_plate_workers()
{
    destroy_layer2_plate_workers();
    {
        std::lock_guard<std::mutex> lock(draw_mtx);
        for (int channelId = 0; channelId < MAX_BATCH; ++channelId)
            layer2PlateDrawBoxes[channelId].clear();
    }
    if (!QFileInfo::exists(layer2PlateModelPath)) {
        qCritical() << "[layer2/model_1 plate] model not found:" << layer2PlateModelPath;
        return;
    }

    for (ChannelContext* ctx : channels) {
        if (!ctx || !ctx->pClient) continue;
        std::unique_ptr<Layer2PlateWorker> worker(new Layer2PlateWorker(ctx->channelId));
        const QByteArray path = layer2PlateModelPath.toLocal8Bit();
        const DWORD plateFlags = QDEEP_API::QDEEP_OBJECT_DETECT_FLAG_FEATURE_VECTOR;
        const QRESULT createResult = QDEEP_API::QDEEP_CREATE_OBJECT_DETECT(
            QDEEP_API::QDEEP_GPU_TYPE_NVIDIA, 0,
            QDEEP_API::QDEEP_OBJECT_DETECT_CONFIG_MODEL_LICENSE_PLATE_RECOGNITION_LAW_ENFORCEMENT,
            const_cast<CHAR*>(path.constData()), &worker->handle, plateFlags);
        if (createResult != QCAP_RS_SUCCESSFUL || !worker->handle) {
            qCritical() << "[layer2/model_1 plate] create failed for CH" << (ctx->channelId + 1)
                        << "result=" << createResult;
            continue;
        }
        const QRESULT startResult = QDEEP_API::QDEEP_START_OBJECT_DETECT(worker->handle);
        if (startResult != QCAP_RS_SUCCESSFUL) {
            qCritical() << "[layer2/model_1 plate] start failed for CH" << (ctx->channelId + 1)
                        << "result=" << startResult;
            QDEEP_API::QDEEP_DESTROY_OBJECT_DETECT(worker->handle);
            worker->handle = nullptr;
            continue;
        }
        worker->ready = true;
        layer2PlateWorkers.push_back(std::move(worker));
    }
}

// 等待所有車牌 worker 結束，再停止與銷毀各自的 QDEEP handle。
void MainWindow::destroy_layer2_plate_workers()
{
    for (const std::unique_ptr<Layer2PlateWorker>& worker : layer2PlateWorkers) {
        if (worker->thread.joinable()) worker->thread.join();
        if (worker->handle) {
            QDEEP_API::QDEEP_STOP_OBJECT_DETECT(worker->handle);
            QDEEP_API::QDEEP_DESTROY_OBJECT_DETECT(worker->handle);
            worker->handle = nullptr;
        }
    }
    layer2PlateWorkers.clear();
}

// 依成功連線路數建立三個 Layer1 動態 batch model worker。
void MainWindow::create_layer1_batch_workers()
{
    destroy_layer1_batch_workers();

    std::vector<int> channelIds;
    for (ChannelContext* ctx : channels) {
        if (ctx && ctx->pClient) channelIds.push_back(ctx->channelId);
    }
    if (channelIds.empty()) {
        qWarning() << "[layer1] no active RTSP channels; Layer 1 is disabled for this run.";
        return;
    }

    const QString modelPaths[] = {layer1Model0Path, layer1Model1Path, layer1Model2Path};
    for (int modelId = 0; modelId < 3; ++modelId) {
        if (!QFileInfo::exists(modelPaths[modelId])) {
            qCritical() << "[layer1/model_" << modelId << "] model not found:" << modelPaths[modelId];
            continue;
        }

        std::unique_ptr<Layer1BatchWorker> worker(new Layer1BatchWorker(modelId));
        worker->channelIds = channelIds;
        worker->pendingFrames.assign(channelIds.size(), nullptr);
        const QByteArray path = modelPaths[modelId].toLocal8Bit();
        const QRESULT createResult = QDEEP_API::QDEEP_CREATE_BATCH_OBJECT_DETECT(
            QDEEP_API::QDEEP_GPU_TYPE_NVIDIA, 0,
            QDEEP_API::QDEEP_OBJECT_DETECT_CONFIG_MODEL_CUSTOMIZED_LITE_NEW,
            const_cast<CHAR*>(path.constData()), &worker->handle, flag,
            static_cast<ULONG>(channelIds.size()));
        if (createResult != QCAP_RS_SUCCESSFUL || !worker->handle) {
            qCritical() << "[layer1/model_" << modelId << "] create failed, result=" << createResult;
            continue;
        }
        const QRESULT startResult = QDEEP_API::QDEEP_START_OBJECT_DETECT(worker->handle);
        if (startResult != QCAP_RS_SUCCESSFUL) {
            qCritical() << "[layer1/model_" << modelId << "] start failed, result=" << startResult;
            QDEEP_API::QDEEP_DESTROY_OBJECT_DETECT(worker->handle);
            worker->handle = nullptr;
            continue;
        }
        worker->ready = true;
        qInfo() << "[layer1/model_" << modelId << "] started with batch size" << channelIds.size();
        layer1BatchWorkers.push_back(std::move(worker));
    }
}

// 等待 Layer1 batch worker 結束，再安全地停止與銷毀 QDEEP handle。
void MainWindow::destroy_layer1_batch_workers()
{
    for (const std::unique_ptr<Layer1BatchWorker>& worker : layer1BatchWorkers) {
        if (worker->thread.joinable()) worker->thread.join();
        if (worker->handle) {
            QDEEP_API::QDEEP_STOP_OBJECT_DETECT(worker->handle);
            QDEEP_API::QDEEP_DESTROY_OBJECT_DETECT(worker->handle);
            worker->handle = nullptr;
        }
    }
    layer1BatchWorkers.clear();
}

// 依成功連線路數建立兩個 Layer3 person 動態 batch model worker。
void MainWindow::create_layer3_batch_workers()
{
    destroy_layer3_batch_workers();
    std::vector<int> channelIds;
    for (ChannelContext* ctx : channels) {
        if (!ctx || !ctx->pClient) continue;
        channelIds.push_back(ctx->channelId);
    }
    if (channelIds.empty()) {
        qWarning() << "[layer3] no active RTSP channels; Layer 3 is disabled for this run.";
        return;
    }
    if (!QFileInfo::exists(layer3PersonModelPath)) {
        qCritical() << "[layer3] person model not found:" << layer3PersonModelPath;
        return;
    }
    {
        std::lock_guard<std::mutex> lock(draw_mtx);
        for (int channelId = 0; channelId < MAX_BATCH; ++channelId) {
            layer3Model0DrawBoxes[channelId].clear();
            layer3Model1DrawBoxes[channelId].clear();
        }
    }

    for (int modelId = 0; modelId < 2; ++modelId) {
        std::unique_ptr<Layer3BatchWorker> worker(new Layer3BatchWorker(modelId));
        worker->channelIds = channelIds;
        worker->pendingFrames.assign(channelIds.size(), nullptr);
        const QByteArray path = layer3PersonModelPath.toLocal8Bit();
        const QRESULT createResult = QDEEP_API::QDEEP_CREATE_BATCH_OBJECT_DETECT(
            QDEEP_API::QDEEP_GPU_TYPE_NVIDIA, 0,
            QDEEP_API::QDEEP_OBJECT_DETECT_CONFIG_MODEL_CUSTOMIZED_LITE_NEW,
            const_cast<CHAR*>(path.constData()), &worker->handle, flag,
            static_cast<ULONG>(channelIds.size()));
        if (createResult != QCAP_RS_SUCCESSFUL || !worker->handle) {
            qCritical() << "[layer3/model_" << modelId << "] create failed, result=" << createResult;
            continue;
        }
        const QRESULT startResult = QDEEP_API::QDEEP_START_OBJECT_DETECT(worker->handle);
        if (startResult != QCAP_RS_SUCCESSFUL) {
            qCritical() << "[layer3/model_" << modelId << "] start failed, result=" << startResult;
            QDEEP_API::QDEEP_DESTROY_OBJECT_DETECT(worker->handle);
            worker->handle = nullptr;
            continue;
        }
        worker->ready = true;
        qInfo() << "[layer3/model_" << modelId << "] started with batch size" << channelIds.size();
        layer3BatchWorkers.push_back(std::move(worker));
    }
}

// 等待 Layer3 batch worker 結束，再停止與銷毀 QDEEP handle。
void MainWindow::destroy_layer3_batch_workers()
{
    // Layer 2 threads are joined before this function, so no more frames can
    // be queued while a Layer 3 QDEEP handle is being stopped/destroyed.
    for (const std::unique_ptr<Layer3BatchWorker>& worker : layer3BatchWorkers) {
        if (worker->thread.joinable()) worker->thread.join();
        if (worker->handle) {
            QDEEP_API::QDEEP_STOP_OBJECT_DETECT(worker->handle);
            QDEEP_API::QDEEP_DESTROY_OBJECT_DETECT(worker->handle);
            worker->handle = nullptr;
        }
    }
    layer3BatchWorkers.clear();
}

// 建立各層 QDEEP worker，啟動 dispatcher、推論與顯示 thread。
void MainWindow::yolo_start()
{
    if (ai_running.load()) return;

    active_camera_count = 0;
    for (ChannelContext *ctx : channels) {
        if (ctx->pClient != nullptr) {
            QMutexLocker lock(&ctx->m_mutex);
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

    create_layer1_batch_workers();
    if (layer1BatchWorkers.empty()) {
        qCritical() << "[layer1] no model was started; AI pipeline will not run.";
        return;
    }
    if (m_bEnableFaceModel.load()) {
        create_layer2_face_workers();
    } else {
        std::lock_guard<std::mutex> lock(draw_mtx);
        for (int channelId = 0; channelId < MAX_BATCH; ++channelId)
            layer2FaceDrawBoxes[channelId].clear();
    }
    if (m_bEnablePlateModel.load()) create_layer2_plate_workers();
    create_layer3_batch_workers();
    ai_running.store(true);
    pAiThread = new std::thread(&MainWindow::ai_dispatch_thread, this);
    for (const std::unique_ptr<Layer1BatchWorker>& worker : layer1BatchWorkers) {
        worker->thread = std::thread(&MainWindow::layer1_batch_inference_thread, this, worker.get());
    }
    for (const std::unique_ptr<Layer2FaceWorker>& worker : layer2FaceWorkers) {
        worker->thread = std::thread(&MainWindow::layer2_face_inference_thread, this, worker.get());
    }
    for (const std::unique_ptr<Layer2PlateWorker>& worker : layer2PlateWorkers) {
        worker->thread = std::thread(&MainWindow::layer2_plate_inference_thread, this, worker.get());
    }
    for (const std::unique_ptr<Layer3BatchWorker>& worker : layer3BatchWorkers) {
        worker->thread = std::thread(&MainWindow::layer3_batch_inference_thread, this, worker.get());
    }
    pDisplayThread = new std::thread(&MainWindow::display_thread, this);
}

// 通知、等待並銷毀全部 AI thread 與 QDEEP worker，避免 handle 併發釋放。
void MainWindow::yolo_stop()
{
    ai_running.store(false);
    cv.notify_all();
    round_cv.notify_all();
    for (const std::unique_ptr<Layer1BatchWorker>& worker : layer1BatchWorkers) {
        worker->queueCv.notify_all();
    }
    for (const std::unique_ptr<Layer2FaceWorker>& worker : layer2FaceWorkers) {
        worker->queueCv.notify_all();
    }
    for (const std::unique_ptr<Layer2PlateWorker>& worker : layer2PlateWorkers) {
        worker->queueCv.notify_all();
    }
    for (const std::unique_ptr<Layer3BatchWorker>& worker : layer3BatchWorkers) {
        worker->queueCv.notify_all();
    }

    if (pAiThread) {
        if (pAiThread->joinable()) {
            pAiThread->join();
        }
        delete pAiThread;
        pAiThread = nullptr;
    }
    if (pDisplayThread) {
        if (pDisplayThread->joinable()) pDisplayThread->join();
        delete pDisplayThread;
        pDisplayThread = nullptr;
    }

    destroy_layer1_batch_workers();
    destroy_layer2_face_workers();
    destroy_layer2_plate_workers();
    destroy_layer3_batch_workers();

    for (ChannelContext *ctx : channels) {
        QMutexLocker lock(&ctx->m_mutex);
        ctx->m_bSendBuffer = false;
    }
}

// 將同一份解碼 frame 放入三個 Layer1 model 的對應動態 batch slot。
void MainWindow::submitFrame(const std::shared_ptr<SharedFrame>& frame)
{
    if (!frame || frame->channelId < 0 || frame->channelId >= MAX_BATCH) return;
    if (!ai_running.load()) return;
    for (const std::unique_ptr<Layer1BatchWorker>& worker : layer1BatchWorkers) {
        if (!worker->ready) continue;
        for (size_t slot = 0; slot < worker->channelIds.size(); ++slot) {
            if (worker->channelIds[slot] != frame->channelId) continue;
            {
                std::lock_guard<std::mutex> lock(worker->queueMutex);
                // One newest frame per channel and model.  The decoder and
                // downstream Layer 2/3 workers never wait for Layer 1 batch completion.
                worker->pendingFrames[slot] = frame;
            }
            worker->queueCv.notify_one();
            break;
        }
    }
}

// 將 Layer1 model0 完成的 frame 送給同一路 Layer2 Face worker。
void MainWindow::submitLayer2FaceFrame(const std::shared_ptr<SharedFrame>& frame)
{
    if (!frame || !ai_running.load()) return;
    for (const std::unique_ptr<Layer2FaceWorker>& worker : layer2FaceWorkers) {
        if (!worker->ready || worker->channelId != frame->channelId) continue;
        {
            std::lock_guard<std::mutex> lock(worker->queueMutex);
            // Depth-one/drop-oldest: model_0 and face inference cannot block
            // each other, while a face call already in progress retains its
            // own shared_ptr until it returns.
            worker->pendingFrame = frame;
        }
        worker->queueCv.notify_one();
        return;
    }
}

// 將 Layer1 model1 完成的 frame 送給同一路 Layer2 車牌 worker。
void MainWindow::submitLayer2PlateFrame(const std::shared_ptr<SharedFrame>& frame)
{
    if (!frame || !ai_running.load()) return;
    for (const std::unique_ptr<Layer2PlateWorker>& worker : layer2PlateWorkers) {
        if (!worker->ready || worker->channelId != frame->channelId) continue;
        {
            std::lock_guard<std::mutex> lock(worker->queueMutex);
            // Depth-one/drop-oldest prevents a plate inference on this stream
            // from blocking layer1/model_1 or any other RTSP stream.
            worker->pendingFrame = frame;
        }
        worker->queueCv.notify_one();
        return;
    }
}

// 將 frame 放入指定 Layer3 model 的對應 channel slot，僅保留最新一份。
static void submitLayer3Frame(
    const std::vector<std::unique_ptr<Layer3BatchWorker>>& workers,
    int modelId, const std::shared_ptr<SharedFrame>& frame, const std::atomic<bool>& running)
{
    if (!frame || !running.load()) return;
    for (const std::unique_ptr<Layer3BatchWorker>& worker : workers) {
        if (!worker->ready || worker->modelId != modelId) continue;
        for (size_t slot = 0; slot < worker->channelIds.size(); ++slot) {
            if (worker->channelIds[slot] != frame->channelId) continue;
            {
                std::lock_guard<std::mutex> lock(worker->queueMutex);
                // Per-channel depth-one/drop-oldest input. The face/plate
                // worker never waits for the current-channel batch to complete.
                worker->pendingFrames[slot] = frame;
            }
            worker->queueCv.notify_one();
            return;
        }
    }
}

// 將 Face 鏈路（或 FR 關閉時的 Layer1 model0）結果送往 Layer3 model0。
void MainWindow::submitLayer3Model0Frame(const std::shared_ptr<SharedFrame>& frame)
{
    submitLayer3Frame(layer3BatchWorkers, 0, frame, ai_running);
}

// 將車牌鏈路（或 Plate 關閉時的 Layer1 model1）結果送往 Layer3 model1。
void MainWindow::submitLayer3Model1Frame(const std::shared_ptr<SharedFrame>& frame)
{
    submitLayer3Frame(layer3BatchWorkers, 1, frame, ai_running);
}

// 設定目前 round 階段要等待的 worker 數；每個被派工的 worker 必須回報一次完成。
void MainWindow::beginRoundStage(int expected)
{
    std::lock_guard<std::mutex> lock(round_mtx);
    roundExpected = expected;
    roundCompleted = 0;
}

// 由已完成 QDEEP 呼叫的 worker 回報，喚醒等待此 round 階段的 coordinator。
void MainWindow::completeRoundStage()
{
    {
        std::lock_guard<std::mutex> lock(round_mtx);
        ++roundCompleted;
    }
    round_cv.notify_one();
}

// 等待目前 round 階段所有已派工 worker 完成；停止 AI 時立即離開。
bool MainWindow::waitForRoundStage()
{
    std::unique_lock<std::mutex> lock(round_mtx);
    round_cv.wait(lock, [this] {
        return !ai_running.load() || roundCompleted >= roundExpected;
    });
    return ai_running.load() && roundCompleted >= roundExpected;
}

// 以 mutex 累計單次 QDEEP API 呼叫的耗時統計資料。
void MainWindow::recordInferenceTiming(InferenceTimingStats& stats, double elapsedMs)
{
    std::lock_guard<std::mutex> lock(stats.mutex);
    ++stats.sampleCount;
    stats.totalMs += elapsedMs;
    stats.minMs = std::min(stats.minMs, elapsedMs);
    stats.maxMs = std::max(stats.maxMs, elapsedMs);
}

// 每三秒輸出並重設所有 Layer1/2/3 model 的推論時間統計與完整 AI round FPS。
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

    const auto formatRoundAndReset = [](InferenceTimingStats& stats) {
        std::lock_guard<std::mutex> lock(stats.mutex);
        const std::uint64_t roundCount = stats.sampleCount;
        QString output;
        if (roundCount == 0) {
            output = QStringLiteral("round Layer1->Layer2->Layer3: rounds=0 ai_fps=0.000");
        } else {
            output = QStringLiteral("round Layer1->Layer2->Layer3: rounds=%1 ai_fps=%2 min_ms=%3 max_ms=%4 avg_ms=%5")
                         .arg(roundCount)
                         .arg(static_cast<double>(roundCount) / 3.0, 0, 'f', 3)
                         .arg(stats.minMs, 0, 'f', 3)
                         .arg(stats.maxMs, 0, 'f', 3)
                         .arg(stats.totalMs / roundCount, 0, 'f', 3);
        }
        stats.sampleCount = 0;
        stats.totalMs = 0.0;
        stats.minMs = std::numeric_limits<double>::infinity();
        stats.maxMs = 0.0;
        return output;
    };

    QString output = QStringLiteral("[QDEEP timing: last 3s]\n%1")
        .arg(formatRoundAndReset(roundTimingStats));
    for (const std::unique_ptr<Layer1BatchWorker>& worker : layer1BatchWorkers) {
        output += QStringLiteral("\n%1").arg(formatAndReset(
            QStringLiteral("layer1/model_%1 batch%2").arg(worker->modelId)
                .arg(worker->channelIds.size()).toLocal8Bit().constData(),
            worker->timing));
    }
    for (const std::unique_ptr<Layer2FaceWorker>& worker : layer2FaceWorkers) {
        output += QStringLiteral("\n%1").arg(formatAndReset(
            QStringLiteral("layer2/face CH%1").arg(worker->channelId + 1).toLocal8Bit().constData(),
            worker->timing));
    }
    for (const std::unique_ptr<Layer3BatchWorker>& worker : layer3BatchWorkers) {
        output += QStringLiteral("\n%1").arg(formatAndReset(
            QStringLiteral("layer3/model_%1 batch%2").arg(worker->modelId)
                .arg(worker->channelIds.size()).toLocal8Bit().constData(),
            worker->timing));
    }
    for (const std::unique_ptr<Layer2PlateWorker>& worker : layer2PlateWorkers) {
        output += QStringLiteral("\n%1").arg(formatAndReset(
            QStringLiteral("layer2/model_1 plate CH%1").arg(worker->channelId + 1).toLocal8Bit().constData(),
            worker->timing));
    }
    qInfo().noquote() << output;
}

// 執行單一 Layer1 model 的完整動態 batch 推論、畫框資料更新與後續鏈路轉送。
void MainWindow::layer1_batch_inference_thread(Layer1BatchWorker* worker)
{
    if (!worker || !worker->handle) return;
    while (ai_running.load()) {
        std::vector<std::shared_ptr<SharedFrame>> batchFrames;
        {
            std::unique_lock<std::mutex> lock(worker->queueMutex);
            worker->queueCv.wait(lock, [this, worker] {
                if (!ai_running.load()) return true;
                return std::all_of(worker->pendingFrames.begin(), worker->pendingFrames.end(),
                                   [](const std::shared_ptr<SharedFrame>& frame) { return frame != nullptr; });
            });
            if (!ai_running.load()) break;

            batchFrames.resize(worker->channelIds.size());
            for (size_t slot = 0; slot < worker->channelIds.size(); ++slot) {
                batchFrames[slot] = worker->pendingFrames[slot];
                worker->pendingFrames[slot].reset();
            }
        }

        const size_t batchSize = worker->channelIds.size();
        std::vector<ULONG> colorSpaces(batchSize, QDEEP_API::QDEEP_COLORSPACE_TYPE_NV12);
        std::vector<ULONG> widths(batchSize);
        std::vector<ULONG> heights(batchSize);
        std::vector<BYTE*> buffers(batchSize);
        std::vector<ULONG> bufferLengths(batchSize);
        std::vector<ULONG> boxSizes(batchSize, BOX_SIZE);
        std::vector<std::vector<QDEEP_API::QDEEP_OBJECT_DETECT_BOUNDING_BOX>> boxStorage(batchSize);
        std::vector<QDEEP_API::QDEEP_OBJECT_DETECT_BOUNDING_BOX*> boxLists(batchSize);
        for (size_t slot = 0; slot < batchSize; ++slot) {
            widths[slot] = batchFrames[slot]->width;
            heights[slot] = batchFrames[slot]->height;
            buffers[slot] = batchFrames[slot]->nv12.data();
            bufferLengths[slot] = static_cast<ULONG>(batchFrames[slot]->nv12.size());
            boxStorage[slot].resize(BOX_SIZE);
            boxLists[slot] = boxStorage[slot].data();
        }

        QElapsedTimer inferenceTimer;
        inferenceTimer.start();
        const QRESULT result = QDEEP_API::QDEEP_SET_VIDEO_OBJECT_DETECT_BATCH_UNCOMPRESSION_BUFFER(
            worker->handle, colorSpaces.data(), widths.data(), heights.data(), buffers.data(), bufferLengths.data(),
            boxLists.data(), boxSizes.data(), static_cast<ULONG>(batchSize));
        recordInferenceTiming(worker->timing, inferenceTimer.nsecsElapsed() / 1000000.0);
        if (result != QCAP_RS_SUCCESSFUL) {
            qWarning() << "[layer1/model_" << worker->modelId << "] batch inference failed:" << result;
            completeRoundStage();
            continue;
        }

        {
            std::lock_guard<std::mutex> lock(draw_mtx);
            for (size_t slot = 0; slot < batchSize; ++slot) {
                const int channelId = worker->channelIds[slot];
                std::vector<DrawBox>* drawList = worker->modelId == 0 ? &layer1Model0DrawBoxes[channelId]
                    : worker->modelId == 1 ? &layer1Model1DrawBoxes[channelId]
                                           : &layer1Model2DrawBoxes[channelId];
                drawList->clear();
                for (ULONG index = 0; index < boxSizes[slot]; ++index) {
                    const auto& box = boxStorage[slot][index];
                    if (box.fProbability < LAYER1_MODEL_CONFIDENCE) continue;
                    drawList->push_back({static_cast<int>(box.nX), static_cast<int>(box.nY),
                                        static_cast<int>(box.nWidth), static_cast<int>(box.nHeight),
                                        static_cast<int>(box.nClassID), box.fProbability,
                                        QString("model_%1 class %2").arg(worker->modelId).arg(box.nClassID), {}});
                }
            }
        }

        // The round coordinator waits for all Layer1 workers before it
        // dispatches this same frame generation to Layer2.
        completeRoundStage();
    }
}

// 執行單一路 Layer2 Face/FR 推論，保存框與五點結果後轉送 Layer3 model0。
void MainWindow::layer2_face_inference_thread(Layer2FaceWorker* worker)
{
    if (!worker || !worker->handle) return;
    while (ai_running.load()) {
        std::shared_ptr<SharedFrame> frame;
        {
            std::unique_lock<std::mutex> lock(worker->queueMutex);
            worker->queueCv.wait(lock, [this, worker] {
                return !ai_running.load() || worker->pendingFrame != nullptr;
            });
            if (!ai_running.load()) break;
            frame = worker->pendingFrame;
            worker->pendingFrame.reset();
        }
        if (!frame) continue;

        std::vector<QDEEP_API::QDEEP_OBJECT_DETECT_BOUNDING_BOX> boxes(BOX_SIZE);
        ULONG boxSize = BOX_SIZE;
        QElapsedTimer inferenceTimer;
        inferenceTimer.start();
        const QRESULT result = QDEEP_API::QDEEP_SET_VIDEO_OBJECT_DETECT_UNCOMPRESSION_BUFFER(
            worker->handle, QDEEP_API::QDEEP_COLORSPACE_TYPE_NV12, frame->width, frame->height,
            frame->nv12.data(), static_cast<ULONG>(frame->nv12.size()), boxes.data(), &boxSize, flag);
        recordInferenceTiming(worker->timing, inferenceTimer.nsecsElapsed() / 1000000.0);
        if (result != QCAP_RS_SUCCESSFUL) {
            qWarning() << "[layer2/face] inference failed for CH" << (worker->channelId + 1)
                       << "result=" << result;
            completeRoundStage();
            continue;
        }

        std::vector<DrawBox> drawList;
        for (ULONG i = 0; i < boxSize; ++i) {
            const auto& box = boxes[i];
            if (box.fProbability < LAYER1_MODEL_CONFIDENCE) continue;
            DrawBox drawBox = {static_cast<int>(box.nX), static_cast<int>(box.nY),
                               static_cast<int>(box.nWidth), static_cast<int>(box.nHeight),
                               static_cast<int>(box.nClassID), box.fProbability,
                               QStringLiteral("face %1%").arg(box.fProbability * 100.0f, 0, 'f', 0), {}};
            for (int keypoint = 0; keypoint < 5; ++keypoint) {
                const auto& point = box.sKeypoints[keypoint];
                if (point.fProbability >= 0.05f)
                    drawBox.keypoints.push_back({static_cast<int>(point.nX),
                                                 static_cast<int>(point.nY), point.fProbability});
            }
            drawList.push_back(std::move(drawBox));
        }
        {
            std::lock_guard<std::mutex> lock(draw_mtx);
            layer2FaceDrawBoxes[worker->channelId] = std::move(drawList);
        }
        // The round coordinator waits for every Layer2 worker before
        // dispatching this frame generation to Layer3.
        completeRoundStage();
    }
}

// 執行單一路 Layer2 車牌推論，保存車牌文字後轉送 Layer3 model1。
void MainWindow::layer2_plate_inference_thread(Layer2PlateWorker* worker)
{
    if (!worker || !worker->handle) return;
    const DWORD plateFlags = QDEEP_API::QDEEP_OBJECT_DETECT_FLAG_FEATURE_VECTOR;
    while (ai_running.load()) {
        std::shared_ptr<SharedFrame> frame;
        {
            std::unique_lock<std::mutex> lock(worker->queueMutex);
            worker->queueCv.wait(lock, [this, worker] {
                return !ai_running.load() || worker->pendingFrame != nullptr;
            });
            if (!ai_running.load()) break;
            frame = worker->pendingFrame;
            worker->pendingFrame.reset();
        }
        if (!frame) continue;

        // This buffer is local to this worker invocation; no other channel or
        // thread can read/write the returned plate-recognition data.
        std::vector<QDEEP_API::QDEEP_OBJECT_DETECT_BOUNDING_BOX> boxes(BOX_SIZE);
        ULONG boxSize = BOX_SIZE;
        QElapsedTimer inferenceTimer;
        inferenceTimer.start();
        const QRESULT result = QDEEP_API::QDEEP_SET_VIDEO_OBJECT_DETECT_UNCOMPRESSION_BUFFER(
            worker->handle, QDEEP_API::QDEEP_COLORSPACE_TYPE_NV12, frame->width, frame->height,
            frame->nv12.data(), static_cast<ULONG>(frame->nv12.size()), boxes.data(), &boxSize, plateFlags);
        recordInferenceTiming(worker->timing, inferenceTimer.nsecsElapsed() / 1000000.0);
        if (result != QCAP_RS_SUCCESSFUL) {
            qWarning() << "[layer2/model_1 plate] inference failed for CH" << (worker->channelId + 1)
                       << "result=" << result;
            completeRoundStage();
            continue;
        }

        std::vector<DrawBox> drawList;
        for (ULONG i = 0; i < boxSize; ++i) {
            const auto& box = boxes[i];
            if (box.fProbability < LAYER1_MODEL_CONFIDENCE) continue;
            const QString plateText = plateTextFromFeatureVector(box);
            const QString label = plateText.isEmpty()
                ? QStringLiteral("plate %1%").arg(box.fProbability * 100.0f, 0, 'f', 0)
                : QStringLiteral("plate: %1").arg(plateText);
            drawList.push_back({static_cast<int>(box.nX), static_cast<int>(box.nY),
                                static_cast<int>(box.nWidth), static_cast<int>(box.nHeight),
                                static_cast<int>(box.nClassID), box.fProbability, label, {}});
        }
        {
            std::lock_guard<std::mutex> lock(draw_mtx);
            layer2PlateDrawBoxes[worker->channelId] = std::move(drawList);
        }
        // The round coordinator waits for every Layer2 worker before
        // dispatching this frame generation to Layer3.
        completeRoundStage();
    }
}

// 執行單一 Layer3 person model 的完整動態 batch 推論並更新各路繪圖資料。
void MainWindow::layer3_batch_inference_thread(Layer3BatchWorker* worker)
{
    if (!worker || !worker->handle) return;
    while (ai_running.load()) {
        std::vector<std::shared_ptr<SharedFrame>> batchFrames;
        {
            std::unique_lock<std::mutex> lock(worker->queueMutex);
            worker->queueCv.wait(lock, [this, worker] {
                if (!ai_running.load()) return true;
                return std::all_of(worker->pendingFrames.begin(), worker->pendingFrames.end(),
                                   [](const std::shared_ptr<SharedFrame>& frame) { return frame != nullptr; });
            });
            if (!ai_running.load()) break;

            batchFrames.resize(worker->channelIds.size());
            for (size_t slot = 0; slot < worker->channelIds.size(); ++slot) {
                batchFrames[slot] = worker->pendingFrames[slot];
                worker->pendingFrames[slot].reset();
            }
        }

        const size_t batchSize = worker->channelIds.size();
        std::vector<ULONG> colorSpaces(batchSize);
        std::vector<ULONG> widths(batchSize);
        std::vector<ULONG> heights(batchSize);
        std::vector<BYTE*> buffers(batchSize);
        std::vector<ULONG> bufferLengths(batchSize);
        std::vector<ULONG> boxSizes(batchSize, BOX_SIZE);
        std::vector<std::vector<QDEEP_API::QDEEP_OBJECT_DETECT_BOUNDING_BOX>> boxStorage(batchSize);
        std::vector<QDEEP_API::QDEEP_OBJECT_DETECT_BOUNDING_BOX*> boxLists(batchSize);
        for (size_t slot = 0; slot < batchSize; ++slot) {
            colorSpaces[slot] = QDEEP_API::QDEEP_COLORSPACE_TYPE_NV12;
            widths[slot] = batchFrames[slot]->width;
            heights[slot] = batchFrames[slot]->height;
            buffers[slot] = batchFrames[slot]->nv12.data();
            bufferLengths[slot] = static_cast<ULONG>(batchFrames[slot]->nv12.size());
            boxStorage[slot].resize(BOX_SIZE);
            boxLists[slot] = boxStorage[slot].data();
        }

        QElapsedTimer inferenceTimer;
        inferenceTimer.start();
        const QRESULT result = QDEEP_API::QDEEP_SET_VIDEO_OBJECT_DETECT_BATCH_UNCOMPRESSION_BUFFER(
            worker->handle, colorSpaces.data(), widths.data(), heights.data(), buffers.data(), bufferLengths.data(),
            boxLists.data(), boxSizes.data(), static_cast<ULONG>(batchSize));
        recordInferenceTiming(worker->timing, inferenceTimer.nsecsElapsed() / 1000000.0);
        if (result != QCAP_RS_SUCCESSFUL) {
            qWarning() << "[layer3/model_" << worker->modelId << "] batch inference failed:" << result;
            completeRoundStage();
            continue;
        }

        {
            std::lock_guard<std::mutex> lock(draw_mtx);
            for (size_t slot = 0; slot < batchSize; ++slot) {
                const int channelId = worker->channelIds[slot];
                std::vector<DrawBox>& drawList = worker->modelId == 0
                    ? layer3Model0DrawBoxes[channelId] : layer3Model1DrawBoxes[channelId];
                drawList.clear();
                for (ULONG index = 0; index < boxSizes[slot]; ++index) {
                    const auto& box = boxStorage[slot][index];
                    if (box.fProbability < LAYER1_MODEL_CONFIDENCE) continue;
                    drawList.push_back({static_cast<int>(box.nX), static_cast<int>(box.nY),
                                        static_cast<int>(box.nWidth), static_cast<int>(box.nHeight),
                                        static_cast<int>(box.nClassID), box.fProbability,
                                        QString("layer3/model_%1 person %2%").arg(worker->modelId)
                                            .arg(box.fProbability * 100.0f, 0, 'f', 0), {}});
                }
            }
        }
        completeRoundStage();
    }
}

// 持續取每一路最新顯示 frame，交由 OpenCV 疊圖並排入 Qt UI 更新。
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

// 以完整多路 frame 組成一個推論 round，依序等待 Layer1、Layer2、Layer3
// 全部完成後才啟動下一 round；RTSP callback 在等待期間仍持續覆蓋 latest frame。
void MainWindow::ai_dispatch_thread()
{
    if (layer1BatchWorkers.empty()) return;

    const std::vector<int> channelIds = layer1BatchWorkers.front()->channelIds;
    std::vector<std::shared_ptr<SharedFrame>> latestInputFrames(channelIds.size());
    auto nextTimingReport = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (ai_running.load()) {
        const auto now = std::chrono::steady_clock::now();
        if (now >= nextTimingReport) {
            printInferenceTimingStats();
            nextTimingReport = now + std::chrono::seconds(3);
        }

        // Continuously take only the newest decoded frame from every channel.
        // If a previous round is busy, the RTSP queues have already collapsed
        // older frames, so the next round starts from current input.
        for (size_t slot = 0; slot < channelIds.size(); ++slot) {
            const int channelId = channelIds[slot];
            ChannelContext* ctx = nullptr;
            for (ChannelContext* channel : channels) {
                if (channel->channelId == channelId) {
                    ctx = channel;
                    break;
                }
            }
            if (!ctx) continue;
            std::shared_ptr<SharedFrame> frame = ctx->takeLatestAIFrame();
            if (frame) latestInputFrames[slot] = frame;
        }

        const bool roundReady = std::all_of(
            latestInputFrames.begin(), latestInputFrames.end(),
            [](const std::shared_ptr<SharedFrame>& frame) { return frame != nullptr; });
        if (!roundReady) {
            std::unique_lock<std::mutex> lock(mtx);
            cv.wait_for(lock, std::chrono::milliseconds(2), [this] { return !ai_running.load(); });
            continue;
        }

        // Take one immutable frame per channel for this whole round. New RTSP
        // frames arriving from now on remain queued for the next round.
        const std::vector<std::shared_ptr<SharedFrame>> roundFrames = latestInputFrames;
        std::fill(latestInputFrames.begin(), latestInputFrames.end(), nullptr);
        QElapsedTimer roundTimer;
        roundTimer.start();

        // Stage 1: all retained Layer1 models run their dynamic batch in parallel.
        beginRoundStage(static_cast<int>(layer1BatchWorkers.size()));
        for (const std::shared_ptr<SharedFrame>& frame : roundFrames) submitFrame(frame);
        if (!waitForRoundStage()) break;

        // Stage 2: one non-batch Face/Plate job per available channel worker.
        beginRoundStage(static_cast<int>(layer2FaceWorkers.size() + layer2PlateWorkers.size()));
        for (const std::shared_ptr<SharedFrame>& frame : roundFrames) {
            submitLayer2FaceFrame(frame);
            submitLayer2PlateFrame(frame);
        }
        if (!waitForRoundStage()) break;

        // Stage 3: both person batch models consume the same completed round.
        beginRoundStage(static_cast<int>(layer3BatchWorkers.size()));
        for (const std::shared_ptr<SharedFrame>& frame : roundFrames) {
            submitLayer3Model0Frame(frame);
            submitLayer3Model1Frame(frame);
        }
        if (!waitForRoundStage()) break;
        recordInferenceTiming(roundTimingStats, roundTimer.nsecsElapsed() / 1000000.0);
    }
}
