#include "MainWindow.h"
#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    // 禁用 llama.cpp 无关日志
    // llama_log_set(nullptr, nullptr);

    MainWindow w;
    w.show();
    return a.exec();
}
