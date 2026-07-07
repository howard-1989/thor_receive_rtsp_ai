#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QFile>
#include <QTextStream>
#include <qcap.h>
#include <qcap.windef.h>
#include <qcap.linux.h>

#include <vector>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <chrono>

namespace QDEEP_API {
    #include "QDEEP.H"
}

#define BOX_SIZE 100
#define MAX_BATCH 28
#define MAX_BUFFER_SIZE (1920 *1080  * 3 / 2)
#define TARGET_FPS 30.0
#define FRAME_INTERVAL (1.0 / TARGET_FPS)

struct DrawKeypoint {
    int x;
    int y;
    float probability;
};

struct DrawPerson {
    DrawKeypoint keypoints[17];
    int classId;
    float probability;
};

class OverlayWidget;
class QFrame;

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

    void HwUninitialize();

    void timerEvent( QTimerEvent *event );

    bool eventFilter(QObject *obj, QEvent *event) override;

protected:
    void resizeEvent(QResizeEvent *event) override;
    void moveEvent(QMoveEvent *event) override;
    void hideEvent(QHideEvent *event) override;
    void showEvent(QShowEvent *event) override;

private:
    Ui::MainWindow *ui;

public:

    static const int MAX_DEVICE = 28;  // change this value to support more channels

    QFrame              *m_pFrame[MAX_DEVICE];
    PVOID               m_pClient[MAX_DEVICE];

    ULONG               m_nVideoWidth[MAX_DEVICE];

    ULONG				m_nVideoHeight[MAX_DEVICE];

    double				m_dVideoFrameRate[MAX_DEVICE];

    QString             m_str_info[MAX_DEVICE];

    int                 m_nTimerId;

    ULONG               m_nCount[MAX_DEVICE];

    bool m_bExpanded;   // 雙擊全螢幕狀態

    void toggleExpandedMode();  // 切換展開 / 恢復

    QFile        m_logFile;    // CSV 記錄檔
    QTextStream  m_logStream;  // 寫入流
    bool         m_bLogging;   // 是否正在記錄

    // YOLO / QDEEP Integration
    void init_models();
    void uninit_models();
    void yolo_start();
    void yolo_stop();
    void ai_inference_thread();

    double last_process_time[MAX_DEVICE];
    bool m_bSendBuffer[MAX_DEVICE];

    void* handle = nullptr;

    std::vector<ULONG> color_space;
    std::vector<ULONG> width;
    std::vector<ULONG> height;
    std::vector<BYTE*> buffer;
    std::vector<ULONG> buffer_len;
    std::vector<ULONG> box_size;
    std::vector<QDEEP_API::QDEEP_OBJECT_DETECT_BOUNDING_BOX*> box_list;
    DWORD flag = 1;

    std::mutex mtx;
    std::condition_variable cv;
    bool ai_running = false;
    std::thread* pAiThread = nullptr;
    bool frame_ready[MAX_BATCH];
    int ready_count = 0;
    int active_camera_count = 0;

    std::vector<DrawPerson> draw_persons[MAX_DEVICE];
    std::mutex draw_mtx;

    OverlayWidget *overlayWidget = nullptr;
    bool m_bShowOverlay = true;

private slots:
    void on_btn_start_clicked();
    void on_btn_stop_clicked();
};

#endif // MAINWINDOW_H
