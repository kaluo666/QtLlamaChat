#ifndef LLAMABACKEND_H
#define LLAMABACKEND_H

#include <QObject>
#include <QString>
#include <functional>
#include "llama.h"

// 回调函数类型：用于流式输出 token
using TokenCallback = std::function<void(const QString &)>;

class LlamaBackend : public QObject
{
    Q_OBJECT
public:
    explicit LlamaBackend(QObject *parent = nullptr);
    ~LlamaBackend();

    // 加载模型
    bool loadModel(const QString &modelPath);
    // 检查模型是否加载
    bool isModelLoaded() const;
    // 流式生成文本
    void generateStreaming(const QString &prompt, TokenCallback callback);
    // 停止生成
    void stopGeneration();

private:
    // 构建 Qwen2 提示模板（必须，否则回答错乱）
    QString buildPrompt(const QString &userInput);

    llama_model *m_model = nullptr;
    llama_context *m_ctx = nullptr;
    bool m_stopGeneration = false;
};

#endif // LLAMABACKEND_H