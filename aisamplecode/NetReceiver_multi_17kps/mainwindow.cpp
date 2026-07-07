#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QFrame>
#include <QCheckBox>
#include <QTableWidget>
#include <QTableWidgetItem>
#include <QHeaderView>
#include <QShortcut>
#include <QKeySequence>
#include <QApplication>
#include <QScreen>
#include <QMouseEvent>
#include <QDateTime>
#include <cmath>
#include <QPainter>
#include <QResizeEvent>
#include <QMoveEvent>
#include <QHideEvent>
#include <QShowEvent>
#include <QDebug>

MainWindow *g_pMainwindow = nullptr;

extern "C" {
    QDEEP_EXT_API QRESULT QDEEP_EXPORT QDEEP_GET_OBJECT_DETECT_RESERVED_STATUS(PVOID pDetector, ULONG* pCheckNum);
}

class OverlayWidget : public QWidget {
public:
    OverlayWidget(QWidget *parent = nullptr) : QWidget(parent) {
        setAttribute(Qt::WA_TransparentForMouseEvents);
        setAttribute(Qt::WA_TranslucentBackground);

        setAttribute(Qt::WA_ShowWithoutActivating);

        setWindowFlags(Qt::Window |
                       Qt::FramelessWindowHint |
                       Qt::Tool |
                       Qt::WindowStaysOnTopHint |
                       Qt::BypassWindowManagerHint);
    }

protected:
    void paintEvent(QPaintEvent *event) override {
        if (!g_pMainwindow) return;
        if (!g_pMainwindow->m_bShowOverlay) return;

        QPainter painter(this);
        painter.setRenderHint(QPainter::Antialiasing);

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

        std::lock_guard<std::mutex> lock(g_pMainwindow->draw_mtx);

        for (int i = 0; i < MainWindow::MAX_DEVICE; ++i) {
            if (!g_pMainwindow->m_bSendBuffer[i] || g_pMainwindow->m_nVideoWidth[i] == 0) continue;

            QWidget* w = g_pMainwindow->m_pFrame[i];
            if (!w || !w->isVisible()) continue;

            QPoint globalPos = w->mapToGlobal(QPoint(0, 0));
            QPoint localPos = this->mapFromGlobal(globalPos);

            double scale_x = (double)w->width() / (double)g_pMainwindow->m_nVideoWidth[i];
            double scale_y = (double)w->height() / (double)g_pMainwindow->m_nVideoHeight[i];

            // 1. Draw camera channel and people count overlay at top-left of the viewport
            int people_count = (int)g_pMainwindow->draw_persons[i].size();
            QString headerText = QString("CH %1 | People: %2").arg(i + 1).arg(people_count);
            
            painter.setPen(Qt::NoPen);
            painter.setBrush(QColor(0, 0, 0, 180));
            
            QFont font = painter.font();
            font.setPointSize(10);
            font.setBold(true);
            painter.setFont(font);
            
            QRect headerRect = painter.fontMetrics().boundingRect(headerText);
            headerRect.adjust(-6, -2, 6, 2);
            headerRect.moveTo(localPos.x() + 10, localPos.y() + 10);
            painter.drawRoundedRect(headerRect, 4, 4);
            
            painter.setPen(QColor(0, 255, 200)); // Neon cyan text
            painter.drawText(headerRect, Qt::AlignCenter, headerText);

            // 2. Draw skeleton for each person
            for (const auto& person : g_pMainwindow->draw_persons[i]) {
                const float kpt_thresh = 0.3f;
                QPoint mapped_kpts[17];
                bool kpt_valid[17] = {false};

                for (int k = 0; k < 17; ++k) {
                    if (person.keypoints[k].probability >= kpt_thresh) {
                        int px = localPos.x() + (int)(person.keypoints[k].x * scale_x);
                        int py = localPos.y() + (int)(person.keypoints[k].y * scale_y);
                        mapped_kpts[k] = QPoint(px, py);
                        kpt_valid[k] = true;
                    }
                }

                // Draw skeleton lines
                QPen linePen;
                linePen.setWidth(3);
                
                for (const auto& conn : connections) {
                    int p1 = conn.first;
                    int p2 = conn.second;
                    if (kpt_valid[p1] && kpt_valid[p2]) {
                        QColor connColor;
                        if (p1 == 0 || p2 == 0 || p1 >= 13 || p2 >= 13) {
                            connColor = QColor(255, 235, 59); // Yellow for face
                        } else if (p1 == 4 || p1 == 5 || p1 == 6 || p1 == 10 || p1 == 11 || p1 == 12 ||
                                   p2 == 4 || p2 == 5 || p2 == 6 || p2 == 10 || p2 == 11 || p2 == 12) {
                            connColor = QColor(0, 229, 255); // Cyan for left side
                        } else {
                            connColor = QColor(255, 20, 147); // Neon pink for right side
                        }
                        
                        linePen.setColor(connColor);
                        painter.setPen(linePen);
                        painter.drawLine(mapped_kpts[p1], mapped_kpts[p2]);
                    }
                }

                // Draw joint points
                for (int k = 0; k < 17; ++k) {
                    if (kpt_valid[k]) {
                        QColor jointColor;
                        if (k == 0 || k >= 13) {
                            jointColor = QColor(255, 255, 0); // Yellow (Face)
                        } else if (k == 4 || k == 5 || k == 6 || k == 10 || k == 11 || k == 12) {
                            jointColor = QColor(0, 255, 255); // Cyan (Left side)
                        } else {
                            jointColor = QColor(255, 0, 128); // Pink/Red (Right side)
                        }

                        painter.setPen(QPen(Qt::white, 1));
                        painter.setBrush(jointColor);
                        painter.drawEllipse(mapped_kpts[k], 4, 4);
                    }
                }
            }
        }
    }
};

static const QString DEFAULT_URL = "rtsp://root:root@192.168.191.6:1554/session0.mpg";
// static const QString DEFAULT_URL = "rtsp://192.168.190.139:554/session0";

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    this->setFixedSize(1400, 920);
    this->setWindowTitle("Client");

    g_pMainwindow = this;

    // YOLO / QDEEP variables initialization
    color_space.resize(MAX_BATCH);
    width.resize(MAX_BATCH);
    height.resize(MAX_BATCH);
    buffer.assign(MAX_BATCH, nullptr);
    buffer_len.resize(MAX_BATCH);
    box_size.assign(MAX_BATCH, BOX_SIZE);
    box_list.assign(MAX_BATCH, nullptr);

    for (int i = 0; i < MAX_DEVICE; ++i) {
        last_process_time[i] = 0.0;
        m_bSendBuffer[i] = false;
    }
    for (int i = 0; i < MAX_BATCH; ++i) {
        frame_ready[i] = false;
    }

    init_models();

    overlayWidget = new OverlayWidget(this);
    overlayWidget->show();
    if (overlayWidget) {
        overlayWidget->raise();
        overlayWidget->setGeometry(this->geometry());
    }

    // ── 建立 28 個影像預覽 Frame（7 欄 × 4 列）──────────────────────────
    const int frameWidth    = 188;
    const int frameHeight   = 150;
    const int frameSpacingX = 10;
    const int frameSpacingY = 10;
    const int columns       = 7;

    for(int i = 0; i < MAX_DEVICE; i++)
    {
        m_pFrame[i] = new QFrame(ui->centralWidget);
        m_pFrame[i]->setFrameShape(QFrame::StyledPanel);
        m_pFrame[i]->setFrameShadow(QFrame::Raised);

        int row = i / columns;
        int col = i % columns;
        m_pFrame[i]->setGeometry(
            10 + col * (frameWidth  + frameSpacingX),
            10 + row * (frameHeight + frameSpacingY),
            frameWidth,
            frameHeight);
        m_pFrame[i]->show();
        m_pFrame[i]->installEventFilter(this);  // 監聽雙擊事件

        m_pClient[i]      = NULL;
        m_nVideoWidth[i]  = 0;
        m_nVideoHeight[i] = 0;
        m_nCount[i]       = 0;
    }

    // ── 設定每路 RTSP URL 表格 ────────────────────────────────────────────
    ui->tableWidget_URLs->setRowCount(MAX_DEVICE);
    ui->tableWidget_URLs->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    ui->tableWidget_URLs->horizontalHeader()->setSectionResizeMode(1, QHeaderView::Stretch);
    ui->tableWidget_URLs->verticalHeader()->setDefaultSectionSize(22);
    ui->tableWidget_URLs->setSelectionBehavior(QAbstractItemView::SelectRows);

    for(int i = 0; i < MAX_DEVICE; i++)
    {
        // 第 0 欄：頻道編號（唯讀）
        QTableWidgetItem *chItem = new QTableWidgetItem(QString("CH %1").arg(i + 1));
        chItem->setFlags(chItem->flags() & ~Qt::ItemIsEditable);
        chItem->setTextAlignment(Qt::AlignCenter);
        ui->tableWidget_URLs->setItem(i, 0, chItem);

        // 第 1 欄：RTSP URL（可編輯，預設值與原本相同）
        QTableWidgetItem *urlItem = new QTableWidgetItem(DEFAULT_URL);
        ui->tableWidget_URLs->setItem(i, 1, urlItem);
    }

    ui->btn_start->setEnabled(true);
    ui->btn_stop->setEnabled(false);

    m_bExpanded = false;  // 初始狀態：未展開
    m_bLogging  = false;  // 初始狀態：未記錄
    m_bShowOverlay = true;

    connect(ui->chk_show_overlay, &QCheckBox::toggled, this, [this](bool checked) {
        m_bShowOverlay = checked;
        if (overlayWidget) {
            overlayWidget->update();
        }
    });

    // ── 鍵盤快捷鍵：Ctrl+S = Start，Ctrl+T = Stop ─────────────────────────
    QShortcut *shortcutStart = new QShortcut(QKeySequence(Qt::CTRL + Qt::Key_S), this);
    connect(shortcutStart, &QShortcut::activated, this, &MainWindow::on_btn_start_clicked);

    QShortcut *shortcutStop  = new QShortcut(QKeySequence(Qt::CTRL + Qt::Key_T), this);
    connect(shortcutStop,  &QShortcut::activated, this, &MainWindow::on_btn_stop_clicked);

    m_nTimerId = startTimer(1000);
}

// ── 雙擊事件過濾器：所有 Frame 雙擊都觸發切換 ────────────────────────
bool MainWindow::eventFilter(QObject *obj, QEvent *event)
{
    if (event->type() == QEvent::MouseButtonDblClick) {
        toggleExpandedMode();
        return true;   // 吸收事件，不再传遞
    }
    return QMainWindow::eventFilter(obj, event);
}

// ── 切換展開 / 恢復 ───────────────────────────────────────────────
void MainWindow::toggleExpandedMode()
{
    if (!m_bExpanded) {
        // ── 進入全螢幕展開模式 ────────────────────────────────────
        QRect screen = QApplication::primaryScreen()->geometry();
        int sw = screen.width();
        int sh = screen.height();

        // 依螢幕寬高比和 MAX_DEVICE 自動計算最佳欄數
        // 公式：cols = round( sqrt(N * sw / sh) )，讓每格接近16:9
        int cols = qMax(1, (int)round(sqrt((double)MAX_DEVICE * sw / sh)));
        int rows = (MAX_DEVICE + cols - 1) / cols;
        int fw   = sw / cols;
        int fh   = sh / rows;

        // 解除固定大小限制，切換全螢幕
        this->setMinimumSize(0, 0);
        this->setMaximumSize(QWIDGETSIZE_MAX, QWIDGETSIZE_MAX);
        this->showFullScreen();

        // 重新配置所有影像 Frame
        for (int i = 0; i < MAX_DEVICE; i++) {
            int row = i / cols;
            int col = i % cols;
            m_pFrame[i]->setGeometry(col * fw, row * fh, fw, fh);
        }

        // 隱藏按鈕 / 表格 / 資訊列
        ui->tableWidget_URLs->hide();
        ui->m_info_1->hide();
        ui->btn_start->hide();
        ui->btn_stop->hide();
        ui->chk_show_overlay->hide();

        m_bExpanded = true;

    } else {
        // ── 恢復原本 UI ───────────────────────────────────────────
        this->showNormal();
        this->setFixedSize(1400, 920);

        const int frameWidth    = 188;
        const int frameHeight   = 150;
        const int frameSpacingX = 10;
        const int frameSpacingY = 10;
        const int columns       = 7;

        for (int i = 0; i < MAX_DEVICE; i++) {
            int row = i / columns;
            int col = i % columns;
            m_pFrame[i]->setGeometry(
                10 + col * (frameWidth  + frameSpacingX),
                10 + row * (frameHeight + frameSpacingY),
                frameWidth,
                frameHeight);
        }

        // 顯示負負控制 UI
        ui->tableWidget_URLs->show();
        ui->m_info_1->show();
        ui->btn_start->show();
        ui->btn_stop->show();
        ui->chk_show_overlay->show();

        m_bExpanded = false;
    }
}

// ── 連線成功回呼 ──────────────────────────────────────────────────────────
QRETURN on_broadcast_client_connected_cb(
    PVOID  pClient          /*IN*/,
    UINT   iSessionNum      /*IN*/,
    ULONG  nVideoEncoderFormat /*IN*/,
    ULONG  nVideoWidth      /*IN*/,
    ULONG  nVideoHeight     /*IN*/,
    BOOL   bVideoIsInterleaved /*IN*/,
    double dVideoFrameRate  /*IN*/,
    ULONG  nAudioEncoderFormat /*IN*/,
    ULONG  nAudioChannels   /*IN*/,
    ULONG  nAudioBitsPerSample /*IN*/,
    ULONG  nAudioSampleFrequency /*IN*/,
    PVOID  pUserData        /*IN*/ )
{
    CHAR m_info[128];
    sprintf( m_info, "Resolution: %lu x %lu %.2f FPS %lu ch %lu HZ",
             nVideoWidth, nVideoHeight, dVideoFrameRate,
             nAudioChannels, nAudioSampleFrequency );

    int i = (int)(uintptr_t)pUserData;
    if (i >= 0 && i < MainWindow::MAX_DEVICE)
    {
        g_pMainwindow->m_str_info[i] = m_info;
        g_pMainwindow->m_nVideoWidth[i] = nVideoWidth;
        g_pMainwindow->m_nVideoHeight[i] = nVideoHeight;
        if(g_pMainwindow->ai_running) {
            g_pMainwindow->m_bSendBuffer[i] = true;
        }
    }

    return QCAP_RT_OK;
}

QRETURN on_video_client_cb(
    PVOID pClient /*IN*/,
    UINT iSessionNum /*IN*/,
    double dSampleTime /*IN*/,
    BYTE * pStreamBuffer /*IN*/,
    ULONG nStreamBufferLen /*IN*/,
    BOOL bIsKeyFrame /*IN*/,
    PVOID pUserData /*IN*/ )
{
    ULONG i = (uintptr_t)pUserData;
    // g_pMainwindow->m_nCount[i]++;
    return QCAP_RT_OK;
}

// ── 影像解碼回呼（計算每秒幀數 & YOLO 辨識）─────────────────────────
QRETURN on_decoder_video_cb(
    PVOID  pClient       /*IN*/,
    UINT   iSessionNum   /*IN*/,
    double dSampleTime   /*IN*/,
    BYTE  *pFrameBuffer  /*IN*/,
    ULONG  nFrameBufferLen /*IN*/,
    PVOID  pUserData     /*IN*/ )
{
    ULONG i = (uintptr_t)pUserData;
    if (!g_pMainwindow) return QCAP_RT_OK;

    g_pMainwindow->m_nCount[i]++;

    if (!g_pMainwindow->m_bSendBuffer[i]) return QCAP_RT_OK;

    double current_time = QCAP_GET_TIME();
    if ((current_time - g_pMainwindow->last_process_time[i]) < FRAME_INTERVAL) {
        return QCAP_RT_OK;
    }

    if (g_pMainwindow->frame_ready[i]) {
        return QCAP_RT_OK;
    }

    g_pMainwindow->last_process_time[i] = current_time;

    PVOID pRCBuffer = QCAP_BUFFER_GET_RCBUFFER(pFrameBuffer, nFrameBufferLen);
    qcap_av_frame_t* pAVFrame = (qcap_av_frame_t*)QCAP_RCBUFFER_LOCK_DATA(pRCBuffer);

    int target_width = g_pMainwindow->m_nVideoWidth[i];
    int target_height = g_pMainwindow->m_nVideoHeight[i];
    if (target_width > 0 && target_height > 0) {
        int src_pitch_Y = pAVFrame->nPitch[0];
        int src_pitch_UV = pAVFrame->nPitch[1];

        g_pMainwindow->buffer_len[i] = (target_width * target_height * 3) / 2;

        if (g_pMainwindow->buffer[i] != nullptr && g_pMainwindow->buffer_len[i] <= MAX_BUFFER_SIZE) {
            BYTE* pDstY = g_pMainwindow->buffer[i];
            BYTE* pSrcY = pAVFrame->pData[0];
            for (int h = 0; h < target_height; ++h) {
                memcpy(pDstY + h * target_width, pSrcY + h * src_pitch_Y, target_width);
            }

            BYTE* pDstUV = g_pMainwindow->buffer[i] + (target_width * target_height);
            BYTE* pSrcUV = pAVFrame->pData[1];
            for (int h = 0; h < target_height / 2; ++h) {
                memcpy(pDstUV + h * target_width, pSrcUV + h * src_pitch_UV, target_width);
            }

            {
                std::lock_guard<std::mutex> lock(g_pMainwindow->mtx);
                if (!g_pMainwindow->frame_ready[i]) {
                    g_pMainwindow->frame_ready[i] = true;
                    g_pMainwindow->ready_count++;

                    if (g_pMainwindow->ready_count >= g_pMainwindow->active_camera_count) {
                        g_pMainwindow->cv.notify_one();
                    }
                }
            }
        }
    }

    QCAP_RCBUFFER_UNLOCK_DATA(pRCBuffer);
    return QCAP_RT_OK;
}

// ─────────────────────────────────────────────────────────────────────────
MainWindow::~MainWindow()
{
    if (overlayWidget) {
        delete overlayWidget;
        overlayWidget = nullptr;
    }
    uninit_models();
    HwUninitialize();
    delete ui;
}

void MainWindow::HwUninitialize()
{
    for(int i = 0; i < MAX_DEVICE; i++)
    {
        if(m_pClient[i] != NULL)
        {
            QCAP_STOP_BROADCAST_CLIENT(m_pClient[i]);
            QCAP_DESTROY_BROADCAST_CLIENT(m_pClient[i]);
            m_pClient[i] = NULL;
        }
    }
}

// ── Start 按鈕：每路讀取各自的 URL ───────────────────────────────────────
void MainWindow::on_btn_start_clicked()
{
    for(int i = 0; i < MAX_DEVICE; i++)
    {
        QTableWidgetItem *urlItem = ui->tableWidget_URLs->item(i, 1);
        QString URL = (urlItem && !urlItem->text().trimmed().isEmpty())
                      ? urlItem->text().trimmed()
                      : DEFAULT_URL;

        QCAP_CREATE_BROADCAST_CLIENT(i, URL.toLatin1().data(), &m_pClient[i],
                                     QCAP_DECODER_TYPE_ZZNVCODEC,
                                     (HWND)m_pFrame[i]->winId(), 1, 0);
    }

    for(int i = 0; i < MAX_DEVICE; i++)
    {
        if (m_pClient[i] != NULL)
        {
            QCAP_REGISTER_BROADCAST_CLIENT_CONNECTED_CALLBACK(
                m_pClient[i], on_broadcast_client_connected_cb, (PVOID)(uintptr_t)i);
            QCAP_REGISTER_VIDEO_BROADCAST_CLIENT_CALLBACK( m_pClient[i], on_video_client_cb, (PVOID)(uintptr_t)i);
            QCAP_REGISTER_VIDEO_DECODER_BROADCAST_CLIENT_CALLBACK(
                m_pClient[i], on_decoder_video_cb, (PVOID)(uintptr_t)i);

            ULONG nRen = 1  ;
            QCAP_SET_BROADCAST_CLIENT_CUSTOM_PROPERTY_EX(m_pClient[i], QCAP_BCPROP_VO_BACKEND, (BYTE*)&nRen, sizeof(nRen));
            QCAP_START_BROADCAST_CLIENT(m_pClient[i], QCAP_BROADCAST_PROTOCOL_TCP, 10000, 0);
        }
    }

    ui->btn_start->setEnabled(false);
    ui->btn_stop->setEnabled(true);

    // ── 開啟 CSV 記錄檔 ─────────────────────────────────────────────
    {
        QString ts = QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss");
        m_logFile.setFileName(QString("fps_log_%1.csv").arg(ts));
        if (m_logFile.open(QIODevice::WriteOnly | QIODevice::Text)) {
            m_logStream.setDevice(&m_logFile);
            // 標頭列：Time,CH1,CH2,...
            QString header = "Time";
            for (int i = 0; i < MAX_DEVICE; i++)
                header += QString(",CH%1").arg(i + 1);
            m_logStream << header << "\n";
            m_logStream.flush();
            m_bLogging = true;
            // 在視窗標題顯示檔案路徑
            setWindowTitle(QString("Client  [記錄中: %1]").arg(m_logFile.fileName()));
        }
    }

    // Start YOLO
    yolo_start();
}

// ── Stop 按鈕 ────────────────────────────────────────────────────────
void MainWindow::on_btn_stop_clicked()
{
    yolo_stop();

    for(int i = 0; i < MAX_DEVICE; i++)
    {
        if(m_pClient[i] != NULL)
        {
            QCAP_STOP_BROADCAST_CLIENT(m_pClient[i]);
            QCAP_DESTROY_BROADCAST_CLIENT(m_pClient[i]);
            m_pClient[i] = NULL;
        }
    }

    // ── 關閉 CSV 檔 ───────────────────────────────────────────────
    if (m_bLogging && m_logFile.isOpen()) {
        m_logStream.flush();
        m_logFile.close();
        m_bLogging = false;
        setWindowTitle("Client");  // 恢復視窗標題
    }

    ui->btn_start->setEnabled(true);
    ui->btn_stop->setEnabled(false);
}

// ── 定時更新狀態列（顯示所有已連線頻道的資訊）────────────────
void MainWindow::timerEvent( QTimerEvent *event )
{
    if( event->timerId() == m_nTimerId )
    {
        // 1. 建立顯示列
        QString displayInfo;
        for (int i = 0; i < MAX_DEVICE; ++i)
        {
            if (!g_pMainwindow->m_str_info[i].isEmpty())
            {
                if (!displayInfo.isEmpty())
                    displayInfo += " | ";
                displayInfo += QString("CH%1: %2 [%3fps]")
                    .arg(i + 1)
                    .arg(g_pMainwindow->m_str_info[i])
                    .arg(g_pMainwindow->m_nCount[i]);
            }
        }
        ui->m_info_1->setText(displayInfo);

        // 2. 寫入 CSV 記錄（讀完 count 後再重置）
        if (m_bLogging && m_logFile.isOpen()) {
            QString line = QDateTime::currentDateTime().toString("HH:mm:ss");
            for (int i = 0; i < MAX_DEVICE; i++)
                line += QString(",%1").arg(g_pMainwindow->m_nCount[i]);
            m_logStream << line << "\n";
            m_logStream.flush();
        }

        // 3. 重置所有計數器
        for (int i = 0; i < MAX_DEVICE; ++i)
            g_pMainwindow->m_nCount[i] = 0;
    }
}

// ── QDEEP / YOLO functions ───────────────────────────────────────────────
void MainWindow::init_models()
{
    for(int i=0; i<MAX_DEVICE; i++){
        m_bSendBuffer[i] = false;
    }

    QRESULT res = QDEEP_API::QDEEP_CREATE_BATCH_OBJECT_DETECT(
        QDEEP_API::QDEEP_GPU_TYPE_NVIDIA, 0,
        QDEEP_API::QDEEP_OBJECT_DETECT_CONFIG_MODEL_HUMAN_SKELETON_17_KEYPOINTS_EX,
        (char*)"/home/nvidia/Documents/NetReceiver_multi_6/model/skeleton_ex/QDEEP.OD.HUMAN.SKELETON.17KPS.EX.CFG",
        &handle, flag, MAX_BATCH);

    qDebug() << "[AI Log] QDEEP_CREATE_BATCH_OBJECT_DETECT res:" << QString("0x%1").arg(res, 8, 16, QChar('0')) << "handle:" << handle;

    if (res == 0 && handle != nullptr) {
        QDEEP_API::QDEEP_START_OBJECT_DETECT(handle);
        QDEEP_API::QDEEP_SET_OBJECT_DETECT_PROPERTY(handle, 0.1);
    }

    res = QDEEP_GET_OBJECT_DETECT_RESERVED_STATUS(reinterpret_cast<PVOID>(0xD7CBB416), reinterpret_cast<ULONG*>(0x3B98119E));
    qDebug() << "[AI Log] QDEEP_GET_OBJECT_DETECT_RESERVED_STATUS res:" << QString("0x%1").arg(res, 8, 16, QChar('0'));

    for(size_t i = 0; i < MAX_BATCH; i++) {
        box_list[i] = new QDEEP_API::QDEEP_OBJECT_DETECT_BOUNDING_BOX[BOX_SIZE];
        buffer[i] = new BYTE[MAX_BUFFER_SIZE]();
        color_space[i] = QDEEP_API::QDEEP_COLORSPACE_TYPE_NV12;
        width[i] = 1920;
        height[i] = 1080;
        buffer_len[i] = MAX_BUFFER_SIZE;
    }
}

void MainWindow::uninit_models()
{
    yolo_stop();

    if(handle != nullptr) {
        QDEEP_API::QDEEP_STOP_OBJECT_DETECT(handle);
        QDEEP_API::QDEEP_DESTROY_OBJECT_DETECT(handle);
        handle = nullptr;
    }

    for(size_t i = 0; i < MAX_BATCH; i++) {
        if (box_list[i]) { delete[] box_list[i]; box_list[i] = nullptr; }
        if (buffer[i]) { delete[] buffer[i]; buffer[i] = nullptr; }
    }
}

void MainWindow::yolo_start()
{
    if (ai_running) return;

    active_camera_count = 0;
    for(int i=0; i<MAX_DEVICE; i++){
        if(m_pClient[i] != nullptr) {
            m_bSendBuffer[i] = true;
            last_process_time[i] = 0.0;
            active_camera_count++;
        }
    }

    if(active_camera_count == 0) {
        qDebug() << "[Warning] No active cameras found! YOLO will not start.";
        return;
    }

    ai_running = true;
    pAiThread = new std::thread(&MainWindow::ai_inference_thread, this);
    qDebug() << "[Info] YOLO started.";
}

void MainWindow::yolo_stop()
{
    if (!ai_running) return;

    for(int i=0; i<MAX_DEVICE; i++){
        m_bSendBuffer[i] = false;
    }

    ai_running = false;
    cv.notify_one();

    if (pAiThread && pAiThread->joinable()) {
        pAiThread->join();
        delete pAiThread;
        pAiThread = nullptr;
    }
    qDebug() << "[Info] YOLO stopped.";
}

void MainWindow::ai_inference_thread()
{
    while(ai_running) {
        std::unique_lock<std::mutex> lock(mtx);

        active_camera_count = 0;
        for(int i=0; i<MAX_DEVICE; i++){
            if(m_bSendBuffer[i] && m_nVideoWidth[i] > 0 && m_nVideoHeight[i] > 0) {
                active_camera_count++;
            }
        }

        if (active_camera_count == 0) {
            lock.unlock();
            std::this_thread::sleep_for(std::chrono::milliseconds(67));
            continue;
        }

        cv.wait_for(lock, std::chrono::milliseconds(67), [this] {
            return (ready_count >= active_camera_count) || !ai_running;
        });

        if(!ai_running) break;

        for(size_t i = 0; i < MAX_BATCH; i++) {
            box_size[i] = BOX_SIZE;
        }

        for(int i = 0; i < MAX_DEVICE; i++) {
            if (m_bSendBuffer[i] && m_nVideoWidth[i] > 0 && m_nVideoHeight[i] > 0) {
                width[i] = m_nVideoWidth[i];
                height[i] = m_nVideoHeight[i];
            } else {
                width[i] = 0;
                height[i] = 0;
            }
        }

        QRESULT api_res = QDEEP_API::QDEEP_SET_VIDEO_OBJECT_DETECT_BATCH_UNCOMPRESSION_BUFFER(
            handle, color_space.data(), width.data(), height.data(),
            buffer.data(), buffer_len.data(), box_list.data(), box_size.data(), MAX_BATCH);

        static int log_counter = 0;
        log_counter++;
        if (log_counter % 30 == 0) {
            QString active_channel_sizes = "";
            QString active_channel_dims = "";
            for(int i = 0; i < MAX_DEVICE; i++) {
                if (m_bSendBuffer[i]) {
                    active_channel_sizes += QString("CH%1_size=%2 ")
                                            .arg(i + 1)
                                            .arg(box_size[i]);
                    active_channel_dims += QString("CH%1:%2x%3 ")
                                           .arg(i + 1)
                                           .arg(width[i])
                                           .arg(height[i]);
                }
            }
            // qDebug() << QString("[AI Inference Log] res:0x%1 | Active camera count:%2 | Ready:%3 | %4")
            //             .arg(api_res, 8, 16, QChar('0'))
            //             .arg(active_camera_count)
            //             .arg(ready_count)
            //             .arg(active_channel_sizes);
            // qDebug() << "  Resolutions:" << active_channel_dims;
        }

        {
            std::lock_guard<std::mutex> draw_lock(draw_mtx);
            for(int i = 0; i < MAX_BATCH; i++) {
                draw_persons[i].clear();

                if (m_bSendBuffer[i] && box_size[i] > 0 && width[i] > 0) {
                    for(int j = 0; j < (int)box_size[i]; j++) {
                        auto& deep_box = box_list[i][j];

                        DrawPerson dp;
                        dp.classId = deep_box.nClassID;
                        dp.probability = deep_box.fProbability;

                        for(int k = 0; k < 17; ++k) {
                            dp.keypoints[k].x = deep_box.sKeypoints[k].nX;
                            dp.keypoints[k].y = deep_box.sKeypoints[k].nY;
                            dp.keypoints[k].probability = deep_box.sKeypoints[k].fProbability;
                        }

                        draw_persons[i].push_back(dp);

                        static int result_log_counter = 0;
                        result_log_counter++;
                        if (result_log_counter % 30 == 0) {
                            QString kpts_info = "";
                            for(int k = 0; k < 17; ++k) {
                                kpts_info += QString("kpt%1:(%2,%3,prob=%4) ")
                                             .arg(k)
                                             .arg(deep_box.sKeypoints[k].nX)
                                             .arg(deep_box.sKeypoints[k].nY)
                                             .arg(deep_box.sKeypoints[k].fProbability, 0, 'f', 2);
                            }
                            // qDebug() << QString("[AI BoundingBox Log] CH%1 Person%2: Box=(x:%3,y:%4,w:%5,h:%6) prob:%7")
                            //             .arg(i + 1)
                            //             .arg(j + 1)
                            //             .arg(deep_box.nX)
                            //             .arg(deep_box.nY)
                            //             .arg(deep_box.nWidth)
                            //             .arg(deep_box.nHeight)
                            //             .arg(deep_box.fProbability);
                            // qDebug() << "  " << kpts_info;
                        }
                    }
                }
            }
        }

        QMetaObject::invokeMethod(this, [this]() {
            if (overlayWidget) {
                overlayWidget->raise();
                overlayWidget->update();
            }
        }, Qt::QueuedConnection);

        for(int i = 0; i < MAX_BATCH; i++) {
            frame_ready[i] = false;
        }
        ready_count = 0;
    }
}

// ── UI Events ─────────────────────────────────────────────────────────────
void MainWindow::resizeEvent(QResizeEvent *event) {
    QMainWindow::resizeEvent(event);
    if (overlayWidget) {
        overlayWidget->setGeometry(this->geometry());
        overlayWidget->raise();
    }
}

void MainWindow::moveEvent(QMoveEvent *event) {
    QMainWindow::moveEvent(event);
    if (overlayWidget) {
        overlayWidget->setGeometry(this->geometry());
        overlayWidget->raise();
    }
}

void MainWindow::hideEvent(QHideEvent *event) {
    QMainWindow::hideEvent(event);
    if (overlayWidget) {
        overlayWidget->hide();
    }
}

void MainWindow::showEvent(QShowEvent *event) {
    QMainWindow::showEvent(event);
    if (overlayWidget) {
        overlayWidget->setGeometry(this->geometry());
        overlayWidget->show();
        overlayWidget->raise();
    }
}
