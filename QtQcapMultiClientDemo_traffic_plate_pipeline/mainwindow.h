#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QFrame>
#include <QGridLayout>
#include <QTableWidget>
#include <QSpinBox>
#include <QPushButton>
#include <QTimer>
#include <QLabel>
#include <QVector>
#include <QMutex>
#include <QCheckBox>
#include <QPointer>

// Include QCAP headers
#include "qcap.h"
#include "qcap2.h"
#include "qcap2.user.h"
#include "qcap2.nvbuf.h"
#include "qcap2.gst.h"

#include <QElapsedTimer>

#include <vector>
#include <atomic>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <chrono>
#include <memory>
#include <deque>
#include <QString>

namespace QDEEP_API {
    #include "QDEEP.H"
}

// ── AI Constants ────────────────────────────────────────────────────────────
#define BOX_SIZE 100
// UI/source-channel storage.  This is deliberately independent from the
// QDEEP batch allocation below.
#define MAX_BATCH 9
// The installed QDEEP licence permits five internal channels.  We create two
// detectors, so 2 + 2 = 4 slots stays within that limit.
#define QDEEP_MODEL_BATCH_SIZE 2
#define DEFAULT_AI_TARGET_FPS 30.0
#define AI_QUEUE_MAX_BUFFERS 3
#define TRAFFIC_CONFIDENCE 0.75f
#define PLATE_CONFIDENCE 0.50f

struct DrawBox {
    int x;
    int y;
    int width;
    int height;
    int classId;
    float probability;
    bool isPlate;
    QString label;
};

// Owned copy of a decoded client frame.  It is immutable after construction
// and can therefore be safely shared by display, traffic and people workers.
struct SharedFrame {
    int channelId;
    ULONG width;
    ULONG height;
    std::vector<BYTE> nv12;
};


struct ChannelContext {
    int channelId;
    QString url;
    QLabel* m_pLabel;

    PVOID pClient;
    // Two independent depth-two/drop-oldest queues.  They own shared CPU
    // frames, never QCAP rc-buffers, so queue ownership cannot stall or
    // corrupt the client decoder callback.
    std::deque<std::shared_ptr<SharedFrame>> m_aiFrames;
    std::deque<std::shared_ptr<SharedFrame>> m_displayFrames;
    std::mutex m_aiQueueMutex;

    // Connected format properties
    ULONG m_nVideoWidth;
    ULONG m_nVideoHeight;
    double m_dVideoFrameRate;
    ULONG m_nVideoEncoderFormat;

    // Stats
    int m_frameCount;
    QString m_statusInfo;
    QMutex m_mutex;

    // Display toggle
    bool m_bDisplayEnabled;
    std::shared_ptr<std::atomic<bool>> m_pPendingUpdate;
    std::atomic<int> m_displayFrameCount;

    // Profiling
    QElapsedTimer m_pushTimer;
    int m_pushFrameCount;
    QElapsedTimer m_fpsTimer;
    int m_decFrameCount;

    // ── AI fields ────────────────────────────────────────────────────────
    bool m_bSendBuffer;         // Whether to send frames to AI
    double m_lastProcessTime;   // Last AI frame submission time
    bool m_bFrameReady;         // Whether a frame is ready for AI

    ChannelContext(int id, const QString& streamUrl, QLabel* pLabel);
    ~ChannelContext();

    bool start();
    void stop();
    void cleanupPipeline();
    void setDisplayEnabled(bool enabled);
    bool enqueueAIFrame(const std::shared_ptr<SharedFrame>& frame);
    std::shared_ptr<SharedFrame> takeLatestAIFrame();
    bool enqueueDisplayFrame(const std::shared_ptr<SharedFrame>& frame);
    std::shared_ptr<SharedFrame> takeLatestDisplayFrame();

    QRETURN onConnected(PVOID pClient, UINT iSessionNum, ULONG nVideoEncoderFormat, ULONG nVideoWidth, ULONG nVideoHeight, BOOL bVideoIsInterleaved, double dVideoFrameRate);
    QRETURN onDecodedVideoFrame(double dSampleTime, BYTE * pFrameBuffer, ULONG nFrameBufferLen);
    QRETURN onFail(UINT iSessionNum, QRESULT nErrorStatus, DWORD nErrorCode);
};

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

protected:
    void timerEvent(QTimerEvent *event) override;
    void closeEvent(QCloseEvent *event) override;
    bool eventFilter(QObject *watched, QEvent *event) override;

private slots:
    void onBtnStartClicked();
    void onBtnStopClicked();
    void onChannelCountChanged(int count);
    void onDisplayToggled(bool checked);
    void onOverlayToggled(bool checked);
    void onHalfRefreshRateToggled(bool checked);

public:
    bool m_bShowOverlay;
    QVector<QFrame*> videoFrames;
    QVector<ChannelContext*> channels;
    int m_timerId;
    bool m_bFullscreen;
    bool m_bEnableDisplay;
    bool m_bHalfRefreshRate;
    static const int MAX_CHANNELS = 9;

public:
    void* trafficHandle;
    void* plateHandle;
    std::vector<QDEEP_API::QDEEP_OBJECT_DETECT_BOUNDING_BOX*> box_list_vec;
    std::vector<QDEEP_API::QDEEP_OBJECT_DETECT_BOUNDING_BOX*> plate_box_list_vec;
    std::vector<DrawBox> draw_boxes[MAX_BATCH];
    std::mutex draw_mtx;
    DWORD flag;

    std::mutex mtx;
    std::condition_variable cv;
    std::atomic<bool> ai_running;
    std::thread* pAiThread;
    std::thread* pTrafficAiThread;
    std::thread* pPlateAiThread;
    std::thread* pDisplayThread;
    int ready_count;
    int active_camera_count;

    void submitFrame(const std::shared_ptr<SharedFrame>& frame);

    QString trafficModelPath;
    QString plateModelPath;
    bool plateModelReady;
    std::mutex frame_mtx;
    std::condition_variable frame_cv;
    std::vector<std::shared_ptr<SharedFrame>> traffic_frames;
    std::vector<std::shared_ptr<SharedFrame>> plate_frames;
    std::vector<std::shared_ptr<SharedFrame>> display_frames;
    int traffic_next_channel;
    int plate_next_channel;
    int display_next_channel;


private:
    void clearGrid();
    void stopAllChannels();

    // ── AI Functions ─────────────────────────────────────────────────────
    void init_models();
    void uninit_models();
    void yolo_start();
    void yolo_stop();
    void traffic_inference_thread();
    void plate_inference_thread();
    void display_thread();
    void ai_dispatch_thread();
    std::shared_ptr<SharedFrame> takeLatestFrame(std::vector<std::shared_ptr<SharedFrame>>& frames, int& nextChannel);

    // UI elements
    QWidget *centralWidget;
    QWidget *controlPanel;
    QWidget *videoContainer;
    QGridLayout *videoGridLayout;

    QSpinBox *spinChannelCount;
    QTableWidget *tableUrls;
    QPushButton *btnStart;
    QPushButton *btnStop;
    QCheckBox *chkEnableDisplay;
    QCheckBox *chkShowOverlay;
    QCheckBox *chkHalfRefreshRate;
    QLabel *lblStatus;
};

#endif // MAINWINDOW_H
