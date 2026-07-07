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
        if (!g_pMainwindow->m_bDrawBoxes) return;

        QPainter painter(this);
        painter.setRenderHint(QPainter::Antialiasing);

        QPen boxPen(Qt::green, 2);
        QPen textPen(Qt::white, 1);
        QFont font = painter.font();
        font.setPointSize(10);
        font.setBold(true);
        painter.setFont(font);

        std::lock_guard<std::mutex> lock(g_pMainwindow->draw_mtx);

        for (int i = 0; i < MainWindow::MAX_DEVICE; ++i) {
            if (!g_pMainwindow->m_bSendBuffer[i] || g_pMainwindow->m_nVideoWidth[i] == 0) continue;

            QWidget* w = g_pMainwindow->m_pFrame[i];
            if (!w || !w->isVisible()) continue;

            QPoint globalPos = w->mapToGlobal(QPoint(0, 0));
            QPoint localPos = this->mapFromGlobal(globalPos);

            double scale_x = (double)w->width() / (double)g_pMainwindow->m_nVideoWidth[i];
            double scale_y = (double)w->height() / (double)g_pMainwindow->m_nVideoHeight[i];

            for (const auto& box : g_pMainwindow->draw_boxes[i]) {

                int draw_x = localPos.x() + (int)(box.original_x * scale_x);
                int draw_y = localPos.y() + (int)(box.original_y * scale_y);
                int draw_w = (int)(box.original_w * scale_x);
                int draw_h = (int)(box.original_h * scale_y);

                painter.setPen(boxPen);
                painter.setBrush(Qt::NoBrush);
                painter.drawRect(draw_x, draw_y, draw_w, draw_h);

                painter.setPen(textPen);
                painter.setBrush(QColor(0, 0, 0, 150));
                QString label = QString("Ch%1 C:%2 [%3%]").arg(i + 1).arg(box.classId).arg((int)(box.probability * 100));

                QRect textRect = painter.fontMetrics().boundingRect(label);
                textRect.moveTo(draw_x, draw_y - textRect.height());
                painter.drawRect(textRect);
                painter.setPen(Qt::green);
                painter.drawText(textRect, Qt::AlignCenter, label);

                // 如果有車牌文字，在框下方顯示
                if (!box.plateText.isEmpty()) {
                    QPen platePen(QColor(255, 220, 0), 1);  // 金黃色
                    painter.setPen(platePen);
                    painter.setBrush(QColor(0, 0, 0, 180));
                    QString plateLabel = QString("[LP] %1").arg(box.plateText);
                    QRect plateRect = painter.fontMetrics().boundingRect(plateLabel);
                    plateRect.moveTo(draw_x, draw_y + draw_h + 2);
                    painter.drawRect(plateRect);
                    painter.setPen(QColor(255, 220, 0));
                    painter.drawText(plateRect, Qt::AlignCenter, plateLabel);
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

    // ── 建立 24 個影像預覽 Frame（6 欄 × 4 列）──────────────────────────
    const int frameWidth    = 210;
    const int frameHeight   = 155;
    const int frameSpacingX = 10;
    const int frameSpacingY = 10;
    const int columns       = 6;

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

    // ── 建立畫框開關 CheckBox ───────────────────────────────────────────
    m_bDrawBoxes = true;
    m_pChkDrawBoxes = new QCheckBox(ui->centralWidget);
    m_pChkDrawBoxes->setObjectName("chkDrawBoxes");
    m_pChkDrawBoxes->setText("顯示畫框");
    m_pChkDrawBoxes->setGeometry(240, 872, 120, 32);
    m_pChkDrawBoxes->setChecked(true);
    connect(m_pChkDrawBoxes, &QCheckBox::stateChanged, this, [this](int state) {
        m_bDrawBoxes = (state == Qt::Checked);
        if (overlayWidget) {
            overlayWidget->update();
        }
    });

    m_bExpanded = false;  // 初始狀態：未展開
    m_bLogging  = false;  // 初始狀態：未記錄

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
        if (m_pChkDrawBoxes) m_pChkDrawBoxes->hide();

        m_bExpanded = true;

    } else {
        // ── 恢復原本 UI ───────────────────────────────────────────
        this->showNormal();
        this->setFixedSize(1400, 920);

        const int frameWidth    = 210;
        const int frameHeight   = 155;
        const int frameSpacingX = 10;
        const int frameSpacingY = 10;
        const int columns       = 6;

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
        if (m_pChkDrawBoxes) m_pChkDrawBoxes->show();

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

    //Yolo
    QDEEP_API::QDEEP_CREATE_BATCH_OBJECT_DETECT(
        QDEEP_API::QDEEP_GPU_TYPE_NVIDIA, 0,
        QDEEP_API::QDEEP_OBJECT_DETECT_CONFIG_MODEL_CUSTOMIZED_LITE_NEW,
        (char*)"/home/nvidia/Documents/NetReceiver_multi_6/model/people_/QDEEP.OD.TINY.PERSON.V10N.CFG",
        &handle, flag, MAX_BATCH);

    //Pose
    // QDEEP_API::QDEEP_CREATE_BATCH_OBJECT_DETECT(
    //     QDEEP_API::QDEEP_GPU_TYPE_NVIDIA, 0,
    //     QDEEP_API::QDEEP_OBJECT_DETECT_CONFIG_MODEL_HUMAN_SKELETON_17_KEYPOINTS_EX,
    //     (char*)"/home/nvidia/Documents/NetReceiver_multi_6/model/RELEASES.QDEEP.MODEL.HUMAN.SKELETON.17KPS.EX 1.1.0.203.7/QDEEP.OD.HUMAN.SKELETON.17KPS.EX.CFG",
    //     &handle, flag, MAX_BATCH);

    QDEEP_API::QDEEP_START_OBJECT_DETECT(handle);
    QDEEP_API::QDEEP_SET_OBJECT_DETECT_PROPERTY(handle, 0.1);
    QDEEP_GET_OBJECT_DETECT_RESERVED_STATUS(reinterpret_cast<PVOID>(0xD7CBB416), reinterpret_cast<ULONG*>(0x3B98119E));

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
            }
        }

        QDEEP_API::QDEEP_SET_VIDEO_OBJECT_DETECT_BATCH_UNCOMPRESSION_BUFFER(
            handle, color_space.data(), width.data(), height.data(),
            buffer.data(), buffer_len.data(), box_list.data(), box_size.data(), MAX_BATCH);

        {
            std::lock_guard<std::mutex> draw_lock(draw_mtx);
            for(int i = 0; i < MAX_BATCH; i++) {
                draw_boxes[i].clear();

                if (m_bSendBuffer[i] && box_size[i] > 0 && width[i] > 0) {
                    for(int j = 0; j < (int)box_size[i]; j++) {
                        auto& deep_box = box_list[i][j];

                        DrawBox db;
                        db.classId = deep_box.nClassID;
                        db.probability = deep_box.fProbability;

                        db.original_x = deep_box.nX;
                        db.original_y = deep_box.nY;
                        db.original_w = deep_box.nWidth;
                        db.original_h = deep_box.nHeight;

                        // 解析車牌文字：從 fFeatureVectors 讀取 char* 內容
                        const char* pFeatureStr = reinterpret_cast<const char*>(deep_box.fFeatureVectors);
                        int max_len = QDEEP_MAX_FEATURE_VECTOR_SIZE * sizeof(float);
                        int len = 0;
                        while (len < max_len && pFeatureStr[len] != '\0') {
                            len++;
                        }
                        QString plateStr = QString::fromUtf8(pFeatureStr, len);
                        db.plateText = plateStr;

                        // 印出 bounding box 資訊（含車牌文字）
                        // if (!plateStr.isEmpty()) {
                        //     qDebug() << QString("[Plate] Ch%1 Box[%2]: X=%3 Y=%4 W=%5 H=%6 Prob=%7 Plate=\"%8\"")
                        //         .arg(i + 1)
                        //         .arg(j)
                        //         .arg(deep_box.nX)
                        //         .arg(deep_box.nY)
                        //         .arg(deep_box.nWidth)
                        //         .arg(deep_box.nHeight)
                        //         .arg(deep_box.fProbability, 0, 'f', 2)
                        //         .arg(plateStr);
                        // } else {
                        //     qDebug() << QString("[Detect] Ch%1 Box[%2]: X=%3 Y=%4 W=%5 H=%6 Prob=%7 (no plate text)")
                        //         .arg(i + 1)
                        //         .arg(j)
                        //         .arg(deep_box.nX)
                        //         .arg(deep_box.nY)
                        //         .arg(deep_box.nWidth)
                        //         .arg(deep_box.nHeight)
                        //         .arg(deep_box.fProbability, 0, 'f', 2);
                        // }

                        draw_boxes[i].push_back(db);
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
        overlayWidget->show();
        overlayWidget->raise();
    }
}
