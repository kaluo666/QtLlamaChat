#include "MainWindow.h"
#include "ui_MainWindow.h"
#include <QFileDialog>
#include <QMessageBox>
#include <QTextCursor>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    setWindowTitle("QtLlamaChat - 本地聊天机器人");

    // 初始化后端线程
    m_backend = new LlamaBackend;
    m_workerThread = new QThread;
    m_backend->moveToThread(m_workerThread);
    m_workerThread->start();

    setUIEnabled(true);
}

MainWindow::~MainWindow()
{
    m_workerThread->quit();
    m_workerThread->wait();
    delete m_backend;
    delete ui;
}

void MainWindow::setUIEnabled(bool enable)
{
    ui->sendPushButton_2->setEnabled(enable && !m_isGenerating);
    ui->loadModelPushButton->setEnabled(enable && !m_isGenerating);
    ui->stopPushButton_3->setEnabled(!enable && m_isGenerating); // 停止按钮反向控制
    ui->sentTextLineEdit_2->setEnabled(enable && !m_isGenerating);
}

// 加载模型（7B加速：GPU加载20层，可自行修改）
void MainWindow::on_loadModelPushButton_clicked()
{
    QString path = QFileDialog::getOpenFileName(this, "选择GGUF模型", ".", "*.gguf");
    if (path.isEmpty()) return;

    setUIEnabled(false);
    // 7B模型加速：n_gpu_layers=20，纯CPU设为0
    bool ok = m_backend->loadModel(path, 20);

    if (ok) {
        QMessageBox::information(this, "成功", "模型加载成功！支持7B加速&上下文记忆");
    } else {
        QMessageBox::warning(this, "失败", "模型加载失败");
    }
    setUIEnabled(true);
}

// 发送消息
void MainWindow::on_sendPushButton_2_clicked()
{
    if (m_isGenerating || !m_backend->isModelLoaded()) return;
    QString txt = ui->sentTextLineEdit_2->text().trimmed();
    if (txt.isEmpty()) return;

    m_isGenerating = true;
    setUIEnabled(false);

    ui->chatTextEdit->append("\n[用户]: " + txt);
    ui->chatTextEdit->append("[AI]: ");
    ui->sentTextLineEdit_2->clear();

    // 跨线程安全回调
    auto tokenCallback = [this](const QString& t) {
        QMetaObject::invokeMethod(this, [this, t]() { onToken(t); });
    };

    // 后端执行生成
    QMetaObject::invokeMethod(m_backend, [=]() {
        m_backend->generateStreaming(txt, tokenCallback);
        QMetaObject::invokeMethod(this, [this]() { onTaskDone(); });
    });
}


// 流式输出Token
void MainWindow::onToken(const QString &t)
{
    ui->chatTextEdit->moveCursor(QTextCursor::End);
    ui->chatTextEdit->insertPlainText(t);
}

// 生成完成
void MainWindow::onTaskDone()
{
    m_isGenerating = false;
    setUIEnabled(true);
    ui->chatTextEdit->append("\n------------------------\n");
}


void MainWindow::on_stopPushButton_3_clicked()
{
    if (!m_isGenerating) return;
    qDebug() << "点击停止按钮";
    // 🔴 直接调用 stopGeneration，确保原子变量立刻生效
    m_backend->stopGeneration();
    // 🔴 额外：手动重置 UI 状态，避免卡住
    m_isGenerating = false;
    setUIEnabled(true);
}
