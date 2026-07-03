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

private slots:
    void onBtnStartClicked();
    void onBtnStopClicked();
    void onChannelCountChanged(int count);
    void onDisplayToggled(bool checked);

private:
    void relayoutGrid();
    void clearGrid();
    void stopAllChannels();

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
    QLabel *lblStatus;
    
    QVector<QFrame*> videoFrames;
    QVector<ChannelContext*> channels;
    
    int m_timerId;
    bool m_bFullscreen;
    bool m_bEnableDisplay;
    
    static const int MAX_CHANNELS = 64;
};

#endif // MAINWINDOW_H
