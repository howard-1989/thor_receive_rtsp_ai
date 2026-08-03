QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = QtQcapMultiClientDemo_traffic_plate_pipeline
TEMPLATE = app

CONFIG += c++11

DEFINES += QT_DEPRECATED_WARNINGS

SOURCES += \
        main.cpp \
        mainwindow.cpp

HEADERS += \
        mainwindow.h

INCLUDEPATH += \
    ../include \
    /usr/src/jetson_multimedia_api/include \
    ../qdeep/include \
    /usr/include/opencv4

# 原本 QDEEP runtime library 路徑：-Wl,-rpath,$$PWD/../qdeep/lib
# 原本 QDEEP link library 路徑：-L$$PWD/../qdeep/lib -lQDEEP
# 將 MY 的路徑放在 qcap 所在路徑之前，確保 runtime 先解析到 MY 的 libQDEEP。
QMAKE_LFLAGS += -Wl,-rpath,/home/nvidia/Downloads/MY/lib \
                -Wl,-rpath,$$PWD/../lib

LIBS += -L$$PWD/../lib -lqcap -lqcap2_rcbuffer \
        -L/home/nvidia/Downloads/MY/lib -lQDEEP

CONFIG += link_pkgconfig
PKGCONFIG += opencv4
