#include "ChatWorker.h"

ChatWorker::ChatWorker(LlamaBackend *backend, const QString &prompt)
    : m_backend(backend), m_prompt(prompt)
{}

void ChatWorker::doWork()
{
    m_backend->generateStreaming(m_prompt, [this](const QString &token) {
        emit newToken(token);
    });
    emit finished();
}