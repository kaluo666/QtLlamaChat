#include "MainWindow.h"
#include "./ui_MainWindow.h"
#include <QFileDialog>
#include <QMessageBox>
#include <QThread>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
    , m_llama(new LlamaBackend(this))
{
    ui->setupUi(this);
    setWindowTitle("QtLlamaChat - 本地离线聊天机器人");


}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::appendMessge(const QString &sender, const QString &text)
{
    ui->chatTextEdit->append(QString("[%1]:[%2]").arg(sender,text));
    m_chatHistory << text;
}

void MainWindow::enableUI(bool enable)
{
    ui->sendPushButton_2->setEnabled(enable);
    ui->loadModelPushButton->setEnabled(enable);
}

void MainWindow::onNewTokenGenerated(const QString &token)
{
    ui->chatTextEdit->moveCursor(QTextCursor::End);
    ui->chatTextEdit->insertPlainText(token);
}

void MainWindow::onGenerateFinished()
{
    enableUI(true);
    ui->chatTextEdit->append("\n");
}

//加载模型
void MainWindow::on_loadModelPushButton_clicked()
{
    qDebug() <<"加载模型";
    QString modelPath = QFileDialog::getOpenFileName(this,"选择GGUF模型","./","Model Files(*.gguf)");
    if(modelPath.isEmpty()) return;

    ui->modelPathLineEdit->setText(modelPath);
    enableUI(false);

    //加载
    bool ok = m_llama->loadModel(modelPath);
    if(ok){
        QMessageBox::information(this,"成功","模型加载成功");
        ui->chatTextEdit->append("模型加载成功！\n");
        enableUI(true);
    }else{
        QMessageBox::warning(this,"失败","模型加载失败");
        ui->chatTextEdit->append("模型加载失败\n");
    }
}

//发送
void MainWindow::on_sendPushButton_2_clicked()
{
    QString input = ui->sentTextLineEdit_2->text();
    if (input.isEmpty() || !m_llama->isModelLoaded()) return;

    // 🔴 关键：只输出用户提问，不输出prompt
    ui->chatTextEdit->append(QString("[User]:%1").arg(input));
    ui->chatTextEdit->append("[AI]:"); // 预留AI回答位置

    // 创建ChatWorker，启动子线程生成
    ChatWorker *worker = new ChatWorker(m_llama, input);
    QThread *thread = new QThread(this);
    worker->moveToThread(thread);

    // 连接信号（队列连接，线程安全）
    connect(thread, &QThread::started, worker, &ChatWorker::doWork);
    connect(worker, &ChatWorker::newToken, this, [this](const QString &token) {
        // 🔴 关键：只追加AI回答，不追加用户输入
        QTextCursor cursor = ui->chatTextEdit->textCursor();
        cursor.movePosition(QTextCursor::End);
        cursor.insertText(token);
        ui->chatTextEdit->setTextCursor(cursor);
    });
    connect(worker, &ChatWorker::finished, thread, &QThread::quit);
    connect(worker, &ChatWorker::finished, worker, &ChatWorker::deleteLater);
    connect(thread, &QThread::finished, thread, &QThread::deleteLater);

    thread->start();
    ui->sentTextLineEdit_2->clear();
}
