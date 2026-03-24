#ifndef LLAMABACKEND_H
#define LLAMABACKEND_H

#include <QObject>
#include <QString>
#include <QStringList>
#include <functional>
#include <atomic>
#include "llama.h"

using TokenCallback = std::function<void(const QString&)>;

class LlamaBackend : public QObject
{
    Q_OBJECT
public:
    explicit LlamaBackend(QObject *parent = nullptr);
    ~LlamaBackend();

    bool loadModel(const QString &modelPath, int n_gpu_layers = 20);
    bool isModelLoaded() const;
    void stopGeneration();
    void clearHistory();

public slots:
    void generateStreaming(const QString &userInput, TokenCallback callback);

private:
    QString buildPromptWithHistory();
    void clearCache();

private:
    llama_model *m_model = nullptr;
    llama_context *m_ctx = nullptr;
    std::atomic<bool> m_stopGeneration{false};

    QStringList m_chatHistory;
    const int MAX_HISTORY = 5; // 🔴 减少历史轮数，避免跑题
};

#endif // LLAMABACKEND_H