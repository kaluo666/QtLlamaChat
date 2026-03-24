#include "LlamaBackend.h"
#include <QDebug>
#include <QFile>
#include <thread>
#include <algorithm>

LlamaBackend::LlamaBackend(QObject *parent)
    : QObject{parent}
{
    llama_backend_init();
}

LlamaBackend::~LlamaBackend()
{
    stopGeneration();
    if (m_ctx) { llama_free(m_ctx); m_ctx = nullptr; }
    if (m_model) { llama_model_free(m_model); m_model = nullptr; }
    llama_backend_free();
}

bool LlamaBackend::loadModel(const QString &modelPath)
{
    stopGeneration();
    clearCache();

    if (m_ctx) { llama_free(m_ctx); m_ctx = nullptr; }
    if (m_model) { llama_model_free(m_model); m_model = nullptr; }

    QFile f(modelPath);
    if (!f.exists()) return false;

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    m_model = llama_model_load_from_file(modelPath.toUtf8().constData(), mp);
    if (!m_model) return false;

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 2048;
    cp.n_threads = 1;
    cp.n_threads_batch = 1;

    m_ctx = llama_init_from_model(m_model, cp);
    return m_ctx != nullptr;
}

bool LlamaBackend::isModelLoaded() const
{
    return m_ctx && m_model;
}

void LlamaBackend::clearCache()
{
    if (m_ctx) llama_kv_self_clear(m_ctx);
}

void LlamaBackend::stopGeneration()
{
    m_stopGeneration = true;
    clearCache();
}

QString LlamaBackend::buildPrompt(const QString &userInput)
{
    return QStringLiteral(
               "<|im_start|>system\n你是一个有用的助手<|im_end|>\n"
               "<|im_start|>user\n%1<|im_end|>\n"
               "<|im_start|>assistant\n"
               ).arg(userInput);
}

void LlamaBackend::generateStreaming(const QString &prompt, TokenCallback callback)
{
    if (!isModelLoaded()) return;

    m_stopGeneration = false;
    clearCache();

    const llama_vocab *vocab = llama_model_get_vocab(m_model);
    QString full = buildPrompt(prompt);
    QByteArray buf = full.toUtf8();

    std::vector<llama_token> tokens(buf.size() * 2);
    int nt = llama_tokenize(vocab, buf.constData(), buf.size(), tokens.data(), tokens.size(), true, false);
    if (nt <= 0) return;

    const int n_ctx = llama_n_ctx(m_ctx);
    llama_batch batch = llama_batch_init(n_ctx, 0, 1);
    batch.n_tokens = nt;

    for (int i=0; i<nt; ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = (i == nt-1);
    }

    if (llama_decode(m_ctx, batch) != 0) {
        llama_batch_free(batch);
        return;
    }

    const int max_gen = 512;
    int cur = nt;

    llama_sampler *smpl = llama_sampler_init_greedy();

    while (cur < nt + max_gen && !m_stopGeneration) {
        llama_token id = llama_sampler_sample(smpl, m_ctx, -1);
        if (id == LLAMA_TOKEN_NULL || id == llama_vocab_eos(vocab)) break;

        char tbuf[256];
        int len = llama_token_to_piece(vocab, id, tbuf, sizeof(tbuf), 0, true);
        if (len > 0) {
            QString token = QString::fromUtf8(tbuf, len);
            if (!token.contains("<|") && !token.contains("|>") && token != "<" && token != "|" && token != ">") {
                if (callback) callback(token);
            }
        }

        batch.n_tokens = 1;
        batch.token[0] = id;
        batch.pos[0] = cur;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = 1;

        if (llama_decode(m_ctx, batch) != 0) break;
        cur++;
    }

    llama_sampler_free(smpl);
    llama_batch_free(batch);
}