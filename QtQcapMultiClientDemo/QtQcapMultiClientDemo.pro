QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = QtQcapMultiClientDemo
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
    /usr/src/jetson_multimedia_api/include

QMAKE_LFLAGS += -Wl,-rpath,../lib

LIBS += -L ../lib -lqcap -lqcap2_rcbuffer
