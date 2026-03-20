#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QStringListModel>
#include <LlamaBackend.h>
#include "ChatWorker.h"

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private:
    void appendMessge(const QString &sender, const QString &text);
    void enableUI(bool enable);

private slots:

    void onNewTokenGenerated(const QString &token);
    void onGenerateFinished();

    void on_loadModelPushButton_clicked();
    void on_sendPushButton_2_clicked();


private:
    Ui::MainWindow *ui;
    LlamaBackend *m_llama;
    ChatWorker *m_worker;
    QThread *m_workerThread;

    QStringList m_chatHistory;

};
#endif // MAINWINDOW_H
