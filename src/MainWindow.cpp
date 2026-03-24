#include "MainWindow.h"
#include "ui_MainWindow.h"
#include <QFileDialog>
#include <QMessageBox>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    m_backend = new LlamaBackend;
    m_workerThread = new QThread;
    m_backend->moveToThread(m_workerThread);
    m_workerThread->start();
}

MainWindow::~MainWindow()
{
    m_workerThread->quit();
    m_workerThread->wait();
    delete ui;
}

void MainWindow::on_loadModelPushButton_clicked()
{
    QString path = QFileDialog::getOpenFileName(this, "选择模型", ".", "*.gguf");
    if (path.isEmpty()) return;

    bool ok = m_backend->loadModel(path);
    if (ok) QMessageBox::information(this, "ok", "模型加载成功");
    else QMessageBox::warning(this, "no", "失败");
}

void MainWindow::on_sendPushButton_2_clicked()
{
    if (m_generating || !m_backend->isModelLoaded()) return;
    QString txt = ui->sentTextLineEdit_2->text().trimmed();
    if (txt.isEmpty()) return;

    m_generating = true;
    ui->chatTextEdit->append("[用户]: " + txt);
    ui->chatTextEdit->append("[AI]: ");
    ui->sentTextLineEdit_2->clear();

    auto func = [this](const QString& t) {
        QMetaObject::invokeMethod(this, [this,t](){ onToken(t); });
    };

    QMetaObject::invokeMethod(m_backend, [=]() {
        m_backend->generateStreaming(txt, func);
        QMetaObject::invokeMethod(this, [this](){ onTaskDone(); });
    });
}

void MainWindow::onToken(const QString &t)
{
    ui->chatTextEdit->moveCursor(QTextCursor::End);
    ui->chatTextEdit->insertPlainText(t);
}

void MainWindow::onTaskDone()
{
    m_generating = false;
    ui->chatTextEdit->append("\n------------------\n");
}