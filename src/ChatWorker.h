#ifndef CHATWORKER_H
#define CHATWORKER_H

#include <QObject>
#include "LlamaBackend.h"

class ChatWorker : public QObject
{
    Q_OBJECT
public:
    explicit ChatWorker(LlamaBackend *backend, const QString &prompt);

public slots:
    void doWork();

signals:
    void newToken(const QString &token);
    void finished();

private:
    LlamaBackend *m_backend;
    QString m_prompt;
};

#endif // CHATWORKER_H