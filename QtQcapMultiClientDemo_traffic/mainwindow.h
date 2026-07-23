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

namespace QDEEP_API {
    #include "QDEEP.H"
}

// ── AI Constants ────────────────────────────────────────────────────────────
#define BOX_SIZE 100
#define MAX_BATCH 64
#define MAX_BUFFER_SIZE (1920 * 1080 * 3 / 2)
#define DEFAULT_AI_TARGET_FPS 30.0


struct ChannelContext {
    int channelId;
    QString url;
    QLabel* m_pLabel;

    PVOID pClient;
    qcap2_video_decoder_t* pVdec;
    qcap2_event_handlers_t* pEventHandlers;
    qcap2_event_t* pEvent_vdec;
    qcap2_video_scaler_t* pScaler2;
    qcap2_video_scaler_t* pScaler3;
    qcap2_rcbuffer_t* m_pScalerBuffers3[8];
    qcap2_rcbuffer_t* m_pCurrentAIRCBuffer;
    qcap2_rcbuffer_queue_t* m_pAIQueue;      // AI frame queue for pipeline optimization

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
    BYTE* m_pAIBuffer;          // NV12 data buffer for AI
    ULONG m_nAIBufferLen;       // Length of AI buffer
    int m_nAIWidth;             // Width for AI processing
    int m_nAIHeight;            // Height for AI processing

    ChannelContext(int id, const QString& streamUrl, QLabel* pLabel);
    ~ChannelContext();

    bool start();
    void stop();
    void cleanupPipeline();
    void setDisplayEnabled(bool enabled);

    QRETURN onConnected(PVOID pClient, UINT iSessionNum, ULONG nVideoEncoderFormat, ULONG nVideoWidth, ULONG nVideoHeight, BOOL bVideoIsInterleaved, double dVideoFrameRate);
    QRETURN onVideoCallback(double dSampleTime, BYTE * pStreamBuffer, ULONG nStreamBufferLen, BOOL bIsKeyFrame);
    QRETURN onFail(UINT iSessionNum, QRESULT nErrorStatus, DWORD nErrorCode);
    QRETURN onEventVdec();
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
    static const int MAX_CHANNELS = 64;

public:
    void* handle;
    std::vector<ULONG> color_space;
    std::vector<ULONG> width_vec;
    std::vector<ULONG> height_vec;
    std::vector<BYTE*> buffer_vec;
    std::vector<ULONG> buffer_len_vec;
    std::vector<ULONG> box_size_vec;
    std::vector<QDEEP_API::QDEEP_OBJECT_DETECT_BOUNDING_BOX*> box_list_vec;
    DWORD flag;

    std::mutex mtx;
    std::condition_variable cv;
    bool ai_running;
    std::thread* pAiThread;
    int ready_count;
    int active_camera_count;


private:
    void clearGrid();
    void stopAllChannels();

    // ── AI Functions ─────────────────────────────────────────────────────
    void init_models();
    void uninit_models();
    void yolo_start();
    void yolo_stop();
    void ai_inference_thread();

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
