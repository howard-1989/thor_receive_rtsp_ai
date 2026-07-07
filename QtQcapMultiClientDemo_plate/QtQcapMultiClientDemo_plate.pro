QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = QtQcapMultiClientDemo_plate
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
    ../qdeep/include

QMAKE_LFLAGS += -Wl,-rpath,$$PWD/../lib \
                -Wl,-rpath,$$PWD/../qdeep/lib

LIBS += -L$$PWD/../lib -lqcap -lqcap2_rcbuffer
LIBS += -L$$PWD/../qdeep/lib -lQDEEP
