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

static const char* kTrafficModel =
    "/home/nvidia/Music/thor_receive_rtsp_ai/model/traffic/QDEEP.OD.TAIWAN.TRAFFIC.C4.TINY.CFG";
// Keep compatibility with the existing plate demo: until a real licence-plate
// model is supplied, the second handle uses the local people model.  An
// environment variable can replace it without rebuilding.
static const char* kDefaultPlateModel =
    "/home/nvidia/Music/thor_receive_rtsp_ai/model/people/QDEEP.OD.TINY.PERSON.V10N.CFG";

extern "C" {
QDEEP_EXT_API QRESULT QDEEP_EXPORT QDEEP_GET_OBJECT_DETECT_RESERVED_STATUS(PVOID pDetector, ULONG* pCheckNum);
}

QImage cvMatToQImage(const cv::Mat& mat) {
    if (mat.type() == CV_8UC3) {
        return QImage(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_RGB888).rgbSwapped().copy();
    }
    return QImage();
}

static std::string trafficClassName(int classId)
{
    switch (classId) {
    case 0: return "Pedestrian";
    case 1: return "Motorcycle";
    case 2: return "Car";
    case 3: return "Large Vehicle";
    default: return "Class " + std::to_string(classId);
    }
}

// Display is fed from an immutable CPU frame, never from a QCAP rcbuffer.
// QDEEP receives the same SharedFrame through its own depth-two queue.
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
        std::vector<DrawBox> boxes;
        {
            std::lock_guard<std::mutex> lock(g_pMainwindow->draw_mtx);
            boxes = g_pMainwindow->draw_boxes[ctx->channelId];
        }
        cv::putText(bgrMat, "CH " + std::to_string(ctx->channelId + 1), cv::Point(10, 25),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 200), 2);
        for (const DrawBox& box : boxes) {
            const int x = std::max(0, std::min(box.x, bgrMat.cols - 1));
            const int y = std::max(0, std::min(box.y, bgrMat.rows - 1));
            const int w = std::max(1, std::min(box.width, bgrMat.cols - x));
            const int h = std::max(1, std::min(box.height, bgrMat.rows - y));
            const cv::Scalar color = box.isPlate ? cv::Scalar(0, 165, 255) : cv::Scalar(0, 255, 0);
            cv::rectangle(bgrMat, cv::Rect(x, y, w, h), color, 2);
            cv::putText(bgrMat, box.label.toStdString(), cv::Point(x, std::max(14, y - 4)),
                        cv::FONT_HERSHEY_SIMPLEX, 0.45, color, 2);
        }
    }

    const QImage image = cvMatToQImage(bgrMat);
    QPointer<QLabel> safeLabel = ctx->m_pLabel;
    std::shared_ptr<std::atomic<bool>> pending = ctx->m_pPendingUpdate;
    QMetaObject::invokeMethod(ctx->m_pLabel, [safeLabel, image, pending]() {
        if (safeLabel) safeLabel->setPixmap(QPixmap::fromImage(image));
        if (pending) pending->store(false);
    }, Qt::QueuedConnection);
}

static std::shared_ptr<SharedFrame> copyScalerFrame(int channelId, qcap2_rcbuffer_t* pSysBuffer)
{
    if (!pSysBuffer) return nullptr;
    PVOID pLockedData = qcap2_rcbuffer_lock_data(pSysBuffer);
    if (!pLockedData) return nullptr;
    qcap2_av_frame_t* pAVFrame = reinterpret_cast<qcap2_av_frame_t*>(pLockedData);
    uint8_t* pBuffer[4] = {nullptr};
    int pStride[4] = {0};
    qcap2_av_frame_get_buffer1(pAVFrame, pBuffer, pStride);
    ULONG colorSpace = 0;
    ULONG width = 0;
    ULONG height = 0;
    qcap2_av_frame_get_video_property(pAVFrame, &colorSpace, &width, &height);
    const bool validNV12 = colorSpace == QCAP_COLORSPACE_TYPE_NV12 && width > 0 && height > 0 &&
                           pBuffer[0] && pBuffer[1] && pStride[0] >= static_cast<int>(width) &&
                           pStride[1] >= static_cast<int>(width);
    std::shared_ptr<SharedFrame> frame;
    if (validNV12) {
        frame = std::make_shared<SharedFrame>();
        frame->channelId = channelId;
        frame->width = width;
        frame->height = height;
        frame->nv12.resize(width * height * 3 / 2);
        for (ULONG row = 0; row < height; ++row) {
            memcpy(frame->nv12.data() + row * width, pBuffer[0] + row * pStride[0], width);
        }
        BYTE* pDstUV = frame->nv12.data() + width * height;
        for (ULONG row = 0; row < height / 2; ++row) {
            memcpy(pDstUV + row * width, pBuffer[1] + row * pStride[1], width);
        }
    }
    qcap2_rcbuffer_unlock_data(pSysBuffer);
    return validNV12 ? frame : nullptr;
}

// The decoded-frame callback owns pFrameBuffer only for its duration.  Copy
// it once into an application-owned SYSBUF before returning from the callback.
// The owned buffer can then be referenced by both independent queues without
// ever retaining or locking a QCAP decoder buffer outside the callback.
static qcap2_rcbuffer_t* copyDecodedNV12ToRCBuffer(
    const BYTE* pFrameBuffer, ULONG nFrameBufferLen, ULONG width, ULONG height)
{
    const ULONG expectedBytes = width * height * 3 / 2;
    if (!pFrameBuffer || width == 0 || height == 0 || nFrameBufferLen < expectedBytes) {
        return nullptr;
    }

    qcap2_rcbuffer_t* pClone = qcap2_rcbuffer_new_av_frame();
    if (!pClone) return nullptr;

    PVOID pCloneData = qcap2_rcbuffer_lock_data(pClone);
    if (!pCloneData) {
        qcap2_rcbuffer_release(pClone);
        return nullptr;
    }
    qcap2_av_frame_t* pCloneFrame = reinterpret_cast<qcap2_av_frame_t*>(pCloneData);
    uint8_t* pDest[4] = {nullptr};
    int destStride[4] = {0};
    qcap2_av_frame_set_video_property(pCloneFrame, QCAP_COLORSPACE_TYPE_NV12, width, height);
    const bool allocated = qcap2_av_frame_alloc_buffer(pCloneFrame, 32, 1);
    if (allocated) qcap2_av_frame_get_buffer1(pCloneFrame, pDest, destStride);
    const bool validNV12 = allocated && pDest[0] && pDest[1] &&
                           destStride[0] >= static_cast<int>(width) &&
                           destStride[1] >= static_cast<int>(width);
    if (validNV12) {
        const BYTE* pSourceY = pFrameBuffer;
        const BYTE* pSourceUV = pFrameBuffer + width * height;
        for (ULONG row = 0; row < height; ++row) {
            memcpy(pDest[0] + row * destStride[0], pSourceY + row * width, width);
        }
        for (ULONG row = 0; row < height / 2; ++row) {
            memcpy(pDest[1] + row * destStride[1], pSourceUV + row * width, width);
        }
    }
    qcap2_rcbuffer_unlock_data(pClone);
    if (!validNV12) {
        qcap2_rcbuffer_release(pClone);
        return nullptr;
    }
    return pClone;
}

// QCAP decoded callbacks normally provide a ref-counted AVFrame handle, not
// a direct pixel pointer.  Lock it only in the callback, copy its planes into
// an application-owned NV12 frame, then unlock it before returning.
static qcap2_rcbuffer_t* copyQcapDecodedFrameToNV12RCBuffer(qcap2_rcbuffer_t* pQcapFrame)
{
    if (!pQcapFrame) return nullptr;

    PVOID pSourceData = qcap2_rcbuffer_lock_data(pQcapFrame);
    if (!pSourceData) return nullptr;

    qcap2_av_frame_t* pSourceFrame = reinterpret_cast<qcap2_av_frame_t*>(pSourceData);
    uint8_t* pSource[4] = {nullptr};
    int sourceStride[4] = {0};
    ULONG colorSpace = 0;
    ULONG width = 0;
    ULONG height = 0;
    qcap2_av_frame_get_buffer1(pSourceFrame, pSource, sourceStride);
    qcap2_av_frame_get_video_property(pSourceFrame, &colorSpace, &width, &height);

    const bool isNV12 = colorSpace == QCAP_COLORSPACE_TYPE_NV12;
    const bool isI420 = colorSpace == QCAP_COLORSPACE_TYPE_I420;
    const bool isYV12 = colorSpace == QCAP_COLORSPACE_TYPE_YV12;
    const bool validSource = width > 0 && height > 0 && pSource[0] &&
                             sourceStride[0] >= static_cast<int>(width) &&
                             ((isNV12 && pSource[1] && sourceStride[1] >= static_cast<int>(width)) ||
                              ((isI420 || isYV12) && pSource[1] && pSource[2] &&
                               sourceStride[1] >= static_cast<int>(width / 2) &&
                               sourceStride[2] >= static_cast<int>(width / 2)));
    if (!validSource) {
        qcap2_rcbuffer_unlock_data(pQcapFrame);
        return nullptr;
    }

    qcap2_rcbuffer_t* pClone = qcap2_rcbuffer_new_av_frame();
    if (!pClone) {
        qcap2_rcbuffer_unlock_data(pQcapFrame);
        return nullptr;
    }
    PVOID pCloneData = qcap2_rcbuffer_lock_data(pClone);
    if (!pCloneData) {
        qcap2_rcbuffer_unlock_data(pQcapFrame);
        qcap2_rcbuffer_release(pClone);
        return nullptr;
    }

    qcap2_av_frame_t* pCloneFrame = reinterpret_cast<qcap2_av_frame_t*>(pCloneData);
    uint8_t* pDest[4] = {nullptr};
    int destStride[4] = {0};
    qcap2_av_frame_set_video_property(pCloneFrame, QCAP_COLORSPACE_TYPE_NV12, width, height);
    const bool allocated = qcap2_av_frame_alloc_buffer(pCloneFrame, 32, 1);
    if (allocated) qcap2_av_frame_get_buffer1(pCloneFrame, pDest, destStride);
    const bool validDest = allocated && pDest[0] && pDest[1] &&
                           destStride[0] >= static_cast<int>(width) &&
                           destStride[1] >= static_cast<int>(width);
    if (validDest) {
        for (ULONG row = 0; row < height; ++row) {
            memcpy(pDest[0] + row * destStride[0], pSource[0] + row * sourceStride[0], width);
        }
        if (isNV12) {
            for (ULONG row = 0; row < height / 2; ++row) {
                memcpy(pDest[1] + row * destStride[1], pSource[1] + row * sourceStride[1], width);
            }
        } else {
            // I420 is Y/U/V; YV12 is Y/V/U.  QDEEP and the renderer both
            // consume NV12, so interleave the source chroma into U/V pairs.
            uint8_t* pSourceU = isI420 ? pSource[1] : pSource[2];
            uint8_t* pSourceV = isI420 ? pSource[2] : pSource[1];
            const int sourceStrideU = isI420 ? sourceStride[1] : sourceStride[2];
            const int sourceStrideV = isI420 ? sourceStride[2] : sourceStride[1];
            for (ULONG row = 0; row < height / 2; ++row) {
                BYTE* pUV = pDest[1] + row * destStride[1];
                const BYTE* pU = pSourceU + row * sourceStrideU;
                const BYTE* pV = pSourceV + row * sourceStrideV;
                for (ULONG col = 0; col < width / 2; ++col) {
                    pUV[col * 2] = pU[col];
                    pUV[col * 2 + 1] = pV[col];
                }
            }
        }
    }

    qcap2_rcbuffer_unlock_data(pClone);
    qcap2_rcbuffer_unlock_data(pQcapFrame);
    if (!validDest) {
        qcap2_rcbuffer_release(pClone);
        return nullptr;
    }
    return pClone;
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

    // QCAP's decoder callback normally supplies a tagged rcbuffer handle.
    // Never memcpy pFrameBuffer until the handle has been locked and its
    // AVFrame planes/strides have been read.
    qcap2_rcbuffer_t* pQcapFrame = qcap2_rcbuffer_cast(pFrameBuffer, nFrameBufferLen);
    qcap2_rcbuffer_t* pOwnedFrame = pQcapFrame
        ? copyQcapDecodedFrameToNV12RCBuffer(pQcapFrame)
        : copyDecodedNV12ToRCBuffer(pFrameBuffer, nFrameBufferLen, width, height);
    if (!pOwnedFrame) {
        qWarning() << "[QCAP decoder] CH" << channelId
                   << "cannot copy decoded SYSBUF:" << nFrameBufferLen
                   << "bytes for" << width << "x" << height;
        return QCAP_RT_OK;
    }

    // Convert the temporary owned rcbuffer into one immutable SharedFrame.
    // From this point both queues use only std::shared_ptr ownership; no
    // queue can retain/release a QCAP rcbuffer.
    std::shared_ptr<SharedFrame> frame = copyScalerFrame(channelId, pOwnedFrame);
    qcap2_rcbuffer_release(pOwnedFrame);
    if (!frame) {
        qWarning() << "[QCAP decoder] CH" << channelId << "cannot build SharedFrame";
        return QCAP_RT_OK;
    }
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
      trafficHandle(nullptr), plateHandle(nullptr), flag(1),
      ai_running(false), pAiThread(nullptr), pTrafficAiThread(nullptr), pPlateAiThread(nullptr), pDisplayThread(nullptr),
      ready_count(0), active_camera_count(0),
      trafficModelPath(QString::fromLatin1(kTrafficModel)),
      plateModelPath(qEnvironmentVariable("QDEEP_PLATE_MODEL", QString::fromLatin1(kDefaultPlateModel))),
      plateModelReady(false), traffic_next_channel(0), plate_next_channel(0), display_next_channel(0)
{
    setWindowTitle("QCAP Multichannel RTSP + QDEEP Traffic + Plate Pipeline");
    resize(1280, 720);

    g_pMainwindow = this;

    // ── Initialize AI members ────────────────────────────────────────────
    box_list_vec.assign(MAX_BATCH, nullptr);
    plate_box_list_vec.assign(MAX_BATCH, nullptr);
    traffic_frames.assign(MAX_BATCH, nullptr);
    plate_frames.assign(MAX_BATCH, nullptr);
    display_frames.assign(MAX_BATCH, nullptr);

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
    spinChannelCount->setValue(8);
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
        box_list_vec[i] = new QDEEP_API::QDEEP_OBJECT_DETECT_BOUNDING_BOX[BOX_SIZE];
        plate_box_list_vec[i] = new QDEEP_API::QDEEP_OBJECT_DETECT_BOUNDING_BOX[BOX_SIZE];
    }

//    QRESULT res = QDEEP_API::QDEEP_CREATE_BATCH_OBJECT_DETECT(
//        QDEEP_API::QDEEP_GPU_TYPE_NVIDIA, 0,
//        QDEEP_API::QDEEP_OBJECT_DETECT_CONFIG_MODEL_CUSTOMIZED_LITE_NEW,
//        trafficModelPath.toLocal8Bit().data(), &trafficHandle, flag, QDEEP_MODEL_BATCH_SIZE);

    QRESULT res = QDEEP_API::QDEEP_CREATE_BATCH_OBJECT_DETECT(
        QDEEP_API::QDEEP_GPU_TYPE_NVIDIA, 0,
        QDEEP_API::QDEEP_OBJECT_DETECT_CONFIG_MODEL_CUSTOMIZED_LITE_NEW,
        "/home/nvidia/Downloads/arya/sdvoe_bacth/demo/model/taiwan_traffic_bath8/QDEEP.OD.TAIWAN.TRAFFIC.C4.TINY.CFG", &trafficHandle, flag, QDEEP_MODEL_BATCH_SIZE);

    if (res == QCAP_RS_SUCCESSFUL && trafficHandle) {
        QDEEP_API::QDEEP_START_OBJECT_DETECT(trafficHandle);
    } else {
        qCritical() << "[Traffic model] create failed:" << trafficModelPath << "result=" << res;
    }

    if (!QFileInfo::exists(plateModelPath)) {
        qWarning() << "[Plate model] not found:" << plateModelPath
                   << "-- set QDEEP_PLATE_MODEL to the plate .CFG file.";
        res = QDEEP_GET_OBJECT_DETECT_RESERVED_STATUS(
            reinterpret_cast<PVOID>(0xD7CBB416), reinterpret_cast<ULONG*>(0x3B98119E));
        qDebug() << "[AI Log] QDEEP_GET_OBJECT_DETECT_RESERVED_STATUS res:"
                 << QString("0x%1").arg(res, 8, 16, QChar('0'));
        return;
    }

//    res = QDEEP_API::QDEEP_CREATE_BATCH_OBJECT_DETECT(
//        QDEEP_API::QDEEP_GPU_TYPE_NVIDIA, 0,
//        QDEEP_API::QDEEP_OBJECT_DETECT_CONFIG_MODEL_CUSTOMIZED_LITE_NEW,
//        plateModelPath.toLocal8Bit().data(), &plateHandle, flag, QDEEP_MODEL_BATCH_SIZE);
    res = QDEEP_API::QDEEP_CREATE_BATCH_OBJECT_DETECT(
        QDEEP_API::QDEEP_GPU_TYPE_NVIDIA, 0,
        QDEEP_API::QDEEP_OBJECT_DETECT_CONFIG_MODEL_CUSTOMIZED_LITE_NEW,
        "/home/nvidia/Downloads/arya/sdvoe_bacth/demo/model/people_bath8/QDEEP.OD.TINY.PERSON.V10N.CFG", &plateHandle, flag, QDEEP_MODEL_BATCH_SIZE);
    if (res == QCAP_RS_SUCCESSFUL && plateHandle) {
        QDEEP_API::QDEEP_START_OBJECT_DETECT(plateHandle);
        plateModelReady = true;
    } else {
        qCritical() << "[Plate model] create failed:" << plateModelPath << "result=" << res;
    }

    res = QDEEP_GET_OBJECT_DETECT_RESERVED_STATUS(
        reinterpret_cast<PVOID>(0xD7CBB416), reinterpret_cast<ULONG*>(0x3B98119E));
    qDebug() << "[AI Log] QDEEP_GET_OBJECT_DETECT_RESERVED_STATUS res:"
             << QString("0x%1").arg(res, 8, 16, QChar('0'));
}

void MainWindow::uninit_models()
{
    yolo_stop();
    if (trafficHandle) {
        QDEEP_API::QDEEP_STOP_OBJECT_DETECT(trafficHandle);
        QDEEP_API::QDEEP_DESTROY_OBJECT_DETECT(trafficHandle);
        trafficHandle = nullptr;
    }
    if (plateHandle) {
        QDEEP_API::QDEEP_STOP_OBJECT_DETECT(plateHandle);
        QDEEP_API::QDEEP_DESTROY_OBJECT_DETECT(plateHandle);
        plateHandle = nullptr;
    }
    for (size_t i = 0; i < MAX_BATCH; ++i) {
        if (box_list_vec[i]) {
            delete[] box_list_vec[i];
            box_list_vec[i] = nullptr;
        }
        if (plate_box_list_vec[i]) {
            delete[] plate_box_list_vec[i];
            plate_box_list_vec[i] = nullptr;
        }
    }
}

void MainWindow::yolo_start()
{
    if (ai_running.load() || !trafficHandle) return;

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
    pTrafficAiThread = new std::thread(&MainWindow::traffic_inference_thread, this);
    if (plateModelReady && plateHandle) {
        pPlateAiThread = new std::thread(&MainWindow::plate_inference_thread, this);
    }
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
    if (pPlateAiThread) {
        if (pPlateAiThread->joinable()) pPlateAiThread->join();
        delete pPlateAiThread;
        pPlateAiThread = nullptr;
    }
    if (pTrafficAiThread) {
        if (pTrafficAiThread->joinable()) pTrafficAiThread->join();
        delete pTrafficAiThread;
        pTrafficAiThread = nullptr;
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
        std::fill(traffic_frames.begin(), traffic_frames.end(), nullptr);
        std::fill(plate_frames.begin(), plate_frames.end(), nullptr);
        std::fill(display_frames.begin(), display_frames.end(), nullptr);
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
            traffic_frames[channelId] = frame;
            if (plateModelReady && plateHandle) plate_frames[channelId] = frame;
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

void MainWindow::traffic_inference_thread()
{
    while (ai_running.load()) {
        std::shared_ptr<SharedFrame> frame;
        {
            std::unique_lock<std::mutex> lock(frame_mtx);
            frame_cv.wait(lock, [this] {
                if (!ai_running.load()) return true;
                return std::any_of(traffic_frames.begin(), traffic_frames.end(),
                                   [](const std::shared_ptr<SharedFrame>& frame) { return frame != nullptr; });
            });
            if (!ai_running.load()) break;
            frame = takeLatestFrame(traffic_frames, traffic_next_channel);
        }
        if (!frame) continue;

        ULONG colorSpace = QDEEP_API::QDEEP_COLORSPACE_TYPE_NV12;
        ULONG width = frame->width;
        ULONG height = frame->height;
        BYTE* buffer = frame->nv12.data();
        ULONG bufferLength = static_cast<ULONG>(frame->nv12.size());
        QDEEP_API::QDEEP_OBJECT_DETECT_BOUNDING_BOX* boxList = box_list_vec[frame->channelId];
        ULONG boxSize = BOX_SIZE;
        QRESULT result = QDEEP_API::QDEEP_SET_VIDEO_OBJECT_DETECT_BATCH_UNCOMPRESSION_BUFFER(
            trafficHandle, &colorSpace, &width, &height, &buffer, &bufferLength,
            &boxList, &boxSize, 1);
        if (result != QCAP_RS_SUCCESSFUL) {
            qWarning() << "[Traffic model] inference failed:" << result
                       << "for native frame" << width << "x" << height;
            continue;
        }

        std::lock_guard<std::mutex> lock(draw_mtx);
        std::vector<DrawBox>& drawList = draw_boxes[frame->channelId];
        drawList.erase(std::remove_if(drawList.begin(), drawList.end(),
                                      [](const DrawBox& box) { return !box.isPlate; }), drawList.end());
        for (ULONG i = 0; i < boxSize; ++i) {
            const auto& box = box_list_vec[frame->channelId][i];
            if (box.fProbability < TRAFFIC_CONFIDENCE) continue;
            drawList.push_back({static_cast<int>(box.nX), static_cast<int>(box.nY),
                                static_cast<int>(box.nWidth), static_cast<int>(box.nHeight),
                                static_cast<int>(box.nClassID), box.fProbability, false,
                                QString::fromStdString(trafficClassName(box.nClassID))});
        }
    }
}

void MainWindow::plate_inference_thread()
{
    while (ai_running.load()) {
        std::shared_ptr<SharedFrame> frame;
        {
            std::unique_lock<std::mutex> lock(frame_mtx);
            frame_cv.wait(lock, [this] {
                if (!ai_running.load()) return true;
                return std::any_of(plate_frames.begin(), plate_frames.end(),
                                   [](const std::shared_ptr<SharedFrame>& frame) { return frame != nullptr; });
            });
            if (!ai_running.load()) break;
            frame = takeLatestFrame(plate_frames, plate_next_channel);
        }
        if (!frame) continue;

        ULONG colorSpace = QDEEP_API::QDEEP_COLORSPACE_TYPE_NV12;
        ULONG width = frame->width;
        ULONG height = frame->height;
        BYTE* buffer = frame->nv12.data();
        ULONG bufferLength = static_cast<ULONG>(frame->nv12.size());
        QDEEP_API::QDEEP_OBJECT_DETECT_BOUNDING_BOX* boxList = plate_box_list_vec[frame->channelId];
        ULONG boxSize = BOX_SIZE;
        QRESULT result = QDEEP_API::QDEEP_SET_VIDEO_OBJECT_DETECT_BATCH_UNCOMPRESSION_BUFFER(
            plateHandle, &colorSpace, &width, &height, &buffer, &bufferLength,
            &boxList, &boxSize, 1);
        if (result != QCAP_RS_SUCCESSFUL) {
            qWarning() << "[People model] inference failed:" << result
                       << "for native frame" << width << "x" << height;
            continue;
        }

        std::lock_guard<std::mutex> lock(draw_mtx);
        std::vector<DrawBox>& drawList = draw_boxes[frame->channelId];
        drawList.erase(std::remove_if(drawList.begin(), drawList.end(),
                                      [](const DrawBox& box) { return box.isPlate; }), drawList.end());
        for (ULONG i = 0; i < boxSize; ++i) {
            const auto& box = plate_box_list_vec[frame->channelId][i];
            if (box.fProbability < PLATE_CONFIDENCE) continue;
            const char* feature = reinterpret_cast<const char*>(box.fFeatureVectors);
            const size_t featureLen = strnlen(feature, QDEEP_MAX_FEATURE_VECTOR_SIZE * sizeof(float));
            const QString label = featureLen ? QString("LP %1").arg(QString::fromUtf8(feature, static_cast<int>(featureLen)))
                                             : QStringLiteral("LP");
            drawList.push_back({static_cast<int>(box.nX), static_cast<int>(box.nY),
                                static_cast<int>(box.nWidth), static_cast<int>(box.nHeight),
                                static_cast<int>(box.nClassID), box.fProbability, true, label});
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
    while (ai_running.load()) {
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
