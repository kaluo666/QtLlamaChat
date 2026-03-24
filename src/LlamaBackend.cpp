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
    clearHistory();
    if (m_ctx) { llama_free(m_ctx); m_ctx = nullptr; }
    if (m_model) { llama_model_free(m_model); m_model = nullptr; }
    llama_backend_free();
}

bool LlamaBackend::loadModel(const QString &modelPath, int n_gpu_layers)
{
    stopGeneration();
    clearCache();
    clearHistory();

    if (m_ctx) { llama_free(m_ctx); m_ctx = nullptr; }
    if (m_model) { llama_model_free(m_model); m_model = nullptr; }

    QFile f(modelPath);
    if (!f.exists()) {
        qDebug() << "模型文件不存在";
        return false;
    }

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = n_gpu_layers;

    m_model = llama_model_load_from_file(modelPath.toUtf8().constData(), mp);
    if (!m_model) {
        qDebug() << "模型加载失败";
        return false;
    }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 4096;
    cp.n_threads = std::min(4, (int)std::thread::hardware_concurrency()); // 🔴 减少线程数，避免阻塞
    cp.n_threads_batch = cp.n_threads;

    m_ctx = llama_init_from_model(m_model, cp);
    if (!m_ctx) {
        llama_model_free(m_model);
        m_model = nullptr;
        qDebug() << "上下文创建失败";
        return false;
    }

    qDebug() << "模型加载成功，GPU分层:" << n_gpu_layers;
    return true;
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
    qDebug() << "stopGeneration() called";
    // 🔴 不再在这里 clearCache，避免干扰正在执行的 llama_decode
}

void LlamaBackend::clearHistory()
{
    m_chatHistory.clear();
}

QString LlamaBackend::buildPromptWithHistory()
{
    QString prompt = "<|im_start|>system\n你是一个有用的AI助手，回答简洁明了，不重复内容<|im_end|>\n";

    for (int i = 0; i < m_chatHistory.size(); i += 2) {
        if (i + 1 >= m_chatHistory.size()) break;
        prompt += QString("<|im_start|>user\n%1<|im_end|>\n").arg(m_chatHistory[i]);
        prompt += QString("<|im_start|>assistant\n%1<|im_end|>\n").arg(m_chatHistory[i+1]);
    }

    prompt += QString("<|im_start|>user\n%1<|im_end|>\n").arg(m_chatHistory.last());
    prompt += "<|im_start|>assistant\n";

    return prompt;
}

void LlamaBackend::generateStreaming(const QString &userInput, TokenCallback callback)
{
    if (!isModelLoaded()) return;

    m_chatHistory << userInput;
    if (m_chatHistory.size() > MAX_HISTORY * 2) {
        m_chatHistory = m_chatHistory.mid(m_chatHistory.size() - MAX_HISTORY * 2);
    }

    m_stopGeneration = false;
    clearCache();

    const llama_vocab *vocab = llama_model_get_vocab(m_model);
    QString fullPrompt = buildPromptWithHistory();
    QByteArray buf = fullPrompt.toUtf8();

    std::vector<llama_token> tokens(buf.size() * 2);
    int nt = llama_tokenize(vocab, buf.constData(), buf.size(),
                            tokens.data(), tokens.size(), true, false);
    if (nt <= 0) {
        qDebug() << "Tokenize失败";
        return;
    }

    const int n_ctx = llama_n_ctx(m_ctx);
    llama_batch batch = llama_batch_init(n_ctx, 0, 1);
    batch.n_tokens = nt;

    for (int i = 0; i < nt; ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = (i == nt - 1);
    }

    // 🔴 首次 decode 前检查停止信号
    if (m_stopGeneration.load()) {
        llama_batch_free(batch);
        return;
    }

    if (llama_decode(m_ctx, batch) != 0) {
        llama_batch_free(batch);
        qDebug() << "首次Decode失败";
        return;
    }

    const int max_gen = 256; // 🔴 大幅缩短生成长度，避免重复
    int cur = nt;
    QString aiReply;

    llama_sampler *smpl = llama_sampler_init_greedy();

    while (cur < nt + max_gen && !m_stopGeneration.load()) {
        // 🔴 每次循环开始先检查停止信号
        if (m_stopGeneration.load()) break;

        llama_token id = llama_sampler_sample(smpl, m_ctx, -1);
        if (id == LLAMA_TOKEN_NULL) break;

        // 🔴 强化 EOS 检测
        const llama_token eos_token = llama_vocab_eos(vocab);
        if (id == eos_token) break;

        char eos_buf[32];
        int eos_len = llama_token_to_piece(vocab, id, eos_buf, sizeof(eos_buf), 0, true);
        if (eos_len > 0) {
            QString eos_str = QString::fromUtf8(eos_buf, eos_len);
            if (eos_str.contains("<|im_end|>")) break;
        }

        char tbuf[256];
        int len = llama_token_to_piece(vocab, id, tbuf, sizeof(tbuf), 0, true);
        if (len > 0) {
            QString token = QString::fromUtf8(tbuf, len);
            if (!token.contains("<|") && !token.contains("|>") && token != "<" && token != "|" && token != ">") {
                aiReply += token;
                if (callback) callback(token);
            }
        }

        batch.n_tokens = 1;
        batch.token[0] = id;
        batch.pos[0] = cur;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = 1;

        // 🔴 调用 llama_decode 前再次检查停止信号
        if (m_stopGeneration.load()) break;

        if (llama_decode(m_ctx, batch) != 0) break;
        cur++;
    }

    if (!aiReply.isEmpty() && !m_stopGeneration.load()) {
        m_chatHistory << aiReply;
    } else if (m_stopGeneration.load()) {
        if (!m_chatHistory.isEmpty()) {
            m_chatHistory.pop_back();
        }
    }

    clearCache(); // 🔴 生成结束后再清空缓存，避免干扰
    llama_sampler_free(smpl);
    llama_batch_free(batch);
    qDebug() << "生成完成，是否被停止：" << m_stopGeneration.load();
}