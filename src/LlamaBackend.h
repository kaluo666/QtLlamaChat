#ifndef LLAMABACKEND_H
#define LLAMABACKEND_H

#include <QObject>
#include <QString>
#include <functional>
#include "llama.h"

using TokenCallback = std::function<void(const QString&)>;

class LlamaBackend : public QObject
{
    Q_OBJECT
public:
    explicit LlamaBackend(QObject *parent = nullptr);
    ~LlamaBackend();

    bool loadModel(const QString &modelPath);
    bool isModelLoaded() const;
    void stopGeneration();

public slots:
    void generateStreaming(const QString &prompt, TokenCallback callback);

private:
    QString buildPrompt(const QString &userInput);
    void clearCache();

private:
    llama_model *m_model = nullptr;
    llama_context *m_ctx = nullptr;
    bool m_stopGeneration = false;
};

#endif // LLAMABACKEND_H