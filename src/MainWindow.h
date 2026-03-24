#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QThread>
#include "LlamaBackend.h"

namespace Ui { class MainWindow; }

class MainWindow : public QMainWindow
{
    Q_OBJECT
public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void on_loadModelPushButton_clicked();
    void on_sendPushButton_2_clicked();
    void onToken(const QString &t);
    void onTaskDone();

private:
    Ui::MainWindow *ui;
    LlamaBackend *m_backend;
    QThread *m_workerThread;
    bool m_generating = false;
};

#endif // MAINWINDOW_H