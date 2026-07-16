#include "mainwindow.h"
#include <QApplication>
#include <sys/resource.h>
#include <QDebug>

void increaseFileDescriptorLimit()
{
    struct rlimit limit;
    if (getrlimit(RLIMIT_NOFILE, &limit) == 0) {
        qDebug() << "Current FD limits: soft =" << limit.rlim_cur << ", hard =" << limit.rlim_max;
        rlim_t target = 65535;
        if (limit.rlim_max < target) {
            target = limit.rlim_max;
        }
        if (limit.rlim_cur < target) {
            limit.rlim_cur = target;
            if (setrlimit(RLIMIT_NOFILE, &limit) == 0) {
                qDebug() << "Successfully increased FD soft limit to" << target;
            } else {
                qWarning() << "Failed to set FD limit to" << target;
            }
        }
    } else {
        qWarning() << "Failed to get FD limit";
    }
}

int main(int argc, char *argv[])
{
    increaseFileDescriptorLimit();

    // Enable support for High DPI displays
    QCoreApplication::setAttribute(Qt::AA_EnableHighDpiScaling);

    QApplication a(argc, argv);
    MainWindow w;
    w.show();

    return a.exec();
}

