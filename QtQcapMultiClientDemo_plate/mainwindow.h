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

// Include QCAP headers
#include "qcap.h"
#include "qcap2.h"
#include "qcap2.user.h"
#include "qcap2.nvbuf.h"
#include "qcap2.gst.h"

#include <QElapsedTimer>

#include <vector>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <chrono>

namespace QDEEP_API {
    #include "QDEEP.H"
}

// ── AI Constants ────────────────────────────────────────────────────────────
#define BOX_SIZE 100
#define MAX_BATCH 64
#define MAX_BUFFER_SIZE (1920 * 1080 * 3 / 2)
#define TARGET_FPS 30.0
#define FRAME_INTERVAL (1.0 / TARGET_FPS)

// ── Detection Box Structure (with plate text) ──────────────────────────────
struct DrawBox {
    int original_x, original_y, original_w, original_h;
    int classId;
    float probability;
    QString plateText;  // License plate text extracted from feature vectors
};

class OverlayWidget;

struct ChannelContext {
    int channelId;
    QString url;
    uintptr_t m_winId;

    PVOID pClient;
    qcap2_video_decoder_t* pVdec;
    qcap2_event_handlers_t* pEventHandlers;
    qcap2_event_t* pEvent_vdec;
    qcap2_video_sink_t* pVideoSink;
    qcap2_video_scaler_t* pScaler;

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

    // Profiling
    QElapsedTimer m_pushTimer;
    int m_pushFrameCount;
    QElapsedTimer m_fpsTimer;
    int m_decFrameCount;

    // ── AI fields ────────────────────────────────────────────────────────
    bool m_bSendBuffer;
    double m_lastProcessTime;
    bool m_bFrameReady;
    BYTE* m_pAIBuffer;
    ULONG m_nAIBufferLen;
    int m_nAIWidth;
    int m_nAIHeight;

    ChannelContext(int id, const QString& streamUrl, uintptr_t winId);
    ~ChannelContext();

    bool start();
    void stop();
    void setDisplayEnabled(bool enabled);

    QRETURN onConnected(PVOID pClient, UINT iSessionNum, ULONG nVideoEncoderFormat, ULONG nVideoWidth, ULONG nVideoHeight, BOOL bVideoIsInterleaved, double dVideoFrameRate);
    QRETURN onVideoCallback(double dSampleTime, BYTE * pStreamBuffer, ULONG nStreamBufferLen, BOOL bIsKeyFrame);
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
    void resizeEvent(QResizeEvent *event) override;
    void moveEvent(QMoveEvent *event) override;
    void hideEvent(QHideEvent *event) override;
    void showEvent(QShowEvent *event) override;

private slots:
    void onBtnStartClicked();
    void onBtnStopClicked();
    void onChannelCountChanged(int count);
    void onDisplayToggled(bool checked);
    void onOverlayToggled(bool checked);

public:
    OverlayWidget *overlayWidget;
    bool m_bShowOverlay;
    QVector<QFrame*> videoFrames;
    QVector<ChannelContext*> channels;
    int m_timerId;
    bool m_bFullscreen;
    bool m_bEnableDisplay;
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

    std::vector<DrawBox> draw_boxes[MAX_BATCH];
    std::mutex draw_mtx;

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
    QLabel *lblStatus;
};

#endif // MAINWINDOW_H
