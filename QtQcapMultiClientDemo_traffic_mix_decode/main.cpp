#include "mainwindow.h"
#include <QApplication>

int main(int argc, char *argv[])
{
    // Enable support for High DPI displays
    QCoreApplication::setAttribute(Qt::AA_EnableHighDpiScaling);

    QApplication a(argc, argv);
    MainWindow w;
    w.show();

    return a.exec();
}
