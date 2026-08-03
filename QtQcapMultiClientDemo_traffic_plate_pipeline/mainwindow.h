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
#include <cstdint>
#include <limits>
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
#define MAX_BATCH 4
#define DEFAULT_AI_TARGET_FPS 30.0
#define AI_QUEUE_MAX_BUFFERS 3
#define LAYER1_MODEL_CONFIDENCE 0.50f

struct DrawKeypoint {
    int x;
    int y;
    float probability;
};

struct DrawBox {
    int x;
    int y;
    int width;
    int height;
    int classId;
    float probability;
    QString label;
    std::vector<DrawKeypoint> keypoints;
};

// Owned copy of a decoded client frame.  It is immutable after construction
// and can therefore be safely shared by display, model_0, and its Layer 2 child.
struct SharedFrame {
    int channelId;
    ULONG width;
    ULONG height;
    std::vector<BYTE> nv12;
};

struct InferenceTimingStats {
    std::mutex mutex;
    std::uint64_t sampleCount;
    double totalMs;
    double minMs;
    double maxMs;

    InferenceTimingStats()
        : sampleCount(0), totalMs(0.0),
          minMs(std::numeric_limits<double>::infinity()), maxMs(0.0) {}
};

// This object is owned by MainWindow. Each active RTSP channel gets one of
// these, so QDEEP handles, input queues, threads, and output buffers never
// cross channel boundaries.
struct Layer2FaceWorker {
    explicit Layer2FaceWorker(int id) : channelId(id), handle(nullptr), ready(false) {}

    int channelId;
    PVOID handle;
    bool ready;
    std::thread thread;
    std::mutex queueMutex;
    std::condition_variable queueCv;
    std::shared_ptr<SharedFrame> pendingFrame;
    InferenceTimingStats timing;
};

// Layer 2 model_1 is chained from Layer 1 model_1. Like the face layer, every
// RTSP channel owns its own handle, queue, output buffer, and thread.
struct Layer2PlateWorker {
    explicit Layer2PlateWorker(int id) : channelId(id), handle(nullptr), ready(false) {}

    int channelId;
    PVOID handle;
    bool ready;
    std::thread thread;
    std::mutex queueMutex;
    std::condition_variable queueCv;
    std::shared_ptr<SharedFrame> pendingFrame;
    InferenceTimingStats timing;
};

// Layer 3 uses one batch slot for every currently active RTSP channel. A
// coordinator submits a complete round, so every slot must be filled before
// this worker invokes QDEEP.
struct Layer3BatchWorker {
    explicit Layer3BatchWorker(int id) : modelId(id), handle(nullptr), ready(false) {}

    int modelId;
    PVOID handle;
    bool ready;
    std::thread thread;
    std::mutex queueMutex;
    std::condition_variable queueCv;
    std::vector<int> channelIds;
    std::vector<std::shared_ptr<SharedFrame>> pendingFrames;
    InferenceTimingStats timing;
};

// Layer 1 mirrors Layer 3: every model owns one dynamic-size QDEEP batch,
// with one slot for each RTSP channel that successfully connected for this
// run. The round coordinator fills every slot with the same generation of
// RTSP frames before this worker invokes QDEEP.
struct Layer1BatchWorker {
    explicit Layer1BatchWorker(int id) : modelId(id), handle(nullptr), ready(false) {}

    int modelId;
    PVOID handle;
    bool ready;
    std::thread thread;
    std::mutex queueMutex;
    std::condition_variable queueCv;
    std::vector<int> channelIds;
    std::vector<std::shared_ptr<SharedFrame>> pendingFrames;
    InferenceTimingStats timing;
};

struct ChannelContext {
    int channelId;
    QString url;
    QLabel* m_pLabel;

    PVOID pClient;
    // Two independent depth-two/drop-oldest queues. They own shared CPU
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
    void onFaceModelToggled(bool checked);
    void onPlateModelToggled(bool checked);
    void onHalfRefreshRateToggled(bool checked);

public:
    std::atomic<bool> m_bShowOverlay;
    std::atomic<bool> m_bEnableFaceModel;
    std::atomic<bool> m_bEnablePlateModel;
    QVector<QFrame*> videoFrames;
    QVector<ChannelContext*> channels;
    int m_timerId;
    bool m_bFullscreen;
    bool m_bEnableDisplay;
    bool m_bHalfRefreshRate;
    static const int MAX_CHANNELS = 9;

public:
    std::vector<DrawBox> layer1Model0DrawBoxes[MAX_BATCH];
    std::vector<DrawBox> layer1Model1DrawBoxes[MAX_BATCH];
    std::vector<DrawBox> layer1Model2DrawBoxes[MAX_BATCH];
    std::vector<DrawBox> layer2FaceDrawBoxes[MAX_BATCH];
    std::vector<DrawBox> layer2PlateDrawBoxes[MAX_BATCH];
    std::vector<DrawBox> layer3Model0DrawBoxes[MAX_BATCH];
    std::vector<DrawBox> layer3Model1DrawBoxes[MAX_BATCH];
    std::mutex draw_mtx;
    DWORD flag;

    std::mutex mtx;
    std::condition_variable cv;
    std::mutex round_mtx;
    std::condition_variable round_cv;
    int roundExpected;
    int roundCompleted;
    std::atomic<bool> ai_running;
    std::thread* pAiThread;
    std::thread* pDisplayThread;
    int ready_count;
    int active_camera_count;

    void submitFrame(const std::shared_ptr<SharedFrame>& frame);

    QString layer1Model0Path;
    QString layer1Model1Path;
    QString layer1Model2Path;
    QString layer2FaceModelPath;
    QString layer2PlateModelPath;
    QString layer3PersonModelPath;
    std::vector<std::unique_ptr<Layer1BatchWorker>> layer1BatchWorkers;
    std::vector<std::unique_ptr<Layer2FaceWorker>> layer2FaceWorkers;
    std::vector<std::unique_ptr<Layer2PlateWorker>> layer2PlateWorkers;
    std::vector<std::unique_ptr<Layer3BatchWorker>> layer3BatchWorkers;
    InferenceTimingStats roundTimingStats;


private:
    void clearGrid();
    void stopAllChannels();

    // ── AI Functions ─────────────────────────────────────────────────────
    void init_models();
    void uninit_models();
    void yolo_start();
    void yolo_stop();
    void layer1_batch_inference_thread(Layer1BatchWorker* worker);
    void layer2_face_inference_thread(Layer2FaceWorker* worker);
    void layer2_plate_inference_thread(Layer2PlateWorker* worker);
    void layer3_batch_inference_thread(Layer3BatchWorker* worker);
    void display_thread();
    void ai_dispatch_thread();
    void create_layer2_face_workers();
    void destroy_layer2_face_workers();
    void submitLayer2FaceFrame(const std::shared_ptr<SharedFrame>& frame);
    void create_layer2_plate_workers();
    void destroy_layer2_plate_workers();
    void submitLayer2PlateFrame(const std::shared_ptr<SharedFrame>& frame);
    void create_layer3_batch_workers();
    void destroy_layer3_batch_workers();
    void submitLayer3Model0Frame(const std::shared_ptr<SharedFrame>& frame);
    void submitLayer3Model1Frame(const std::shared_ptr<SharedFrame>& frame);
    void recordInferenceTiming(InferenceTimingStats& stats, double elapsedMs);
    void printInferenceTimingStats();
    void create_layer1_batch_workers();
    void destroy_layer1_batch_workers();
    void beginRoundStage(int expected);
    void completeRoundStage();
    bool waitForRoundStage();

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
    QCheckBox *chkEnableFaceModel;
    QCheckBox *chkEnablePlateModel;
    QCheckBox *chkHalfRefreshRate;
    QLabel *lblStatus;
};

#endif // MAINWINDOW_H
