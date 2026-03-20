#include "LlamaBackend.h"
#include <QDebug>
#include <QFile>
#include <cstring>
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
    if (m_ctx) {
        llama_free(m_ctx);
        m_ctx = nullptr;
    }
    if (m_model) {
        llama_model_free(m_model);
        m_model = nullptr;
    }
    llama_backend_free();
}

bool LlamaBackend::loadModel(const QString &modelPath)
{
    if (m_ctx) {
        llama_free(m_ctx);
        m_ctx = nullptr;
    }
    if (m_model) {
        llama_model_free(m_model);
        m_model = nullptr;
    }

    qDebug() << "开始加载模型，路径：" << modelPath;

    QFile file(modelPath);
    if (!file.exists()) {
        qDebug() << "❌ 模型文件不存在！";
        return false;
    }
    qDebug() << "✅ 模型文件存在，大小：" << file.size() / 1024 / 1024 << "MB";

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;

    m_model = llama_model_load_from_file(modelPath.toUtf8().constData(), model_params);
    if (!m_model) {
        qDebug() << "❌ 模型加载失败！";
        return false;
    }
    qDebug() << "✅ 模型加载成功";

    // 🔴 核心修复1：固定上下文大小 2048，线程数 1（彻底避免多线程竞争崩溃）
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 2048;
    ctx_params.n_threads = 1;
    ctx_params.n_threads_batch = 1;

    m_ctx = llama_init_from_model(m_model, ctx_params);
    if (!m_ctx) {
        qDebug() << "❌ 上下文创建失败！";
        llama_model_free(m_model);
        m_model = nullptr;
        return false;
    }
    qDebug() << "✅ 上下文创建成功，n_ctx:" << llama_n_ctx(m_ctx);

    return true;
}

bool LlamaBackend::isModelLoaded() const
{
    return m_ctx != nullptr && m_model != nullptr;
}

QString LlamaBackend::buildPrompt(const QString &userInput)
{
    return QStringLiteral(
               "<|im_start|>system\n"
               "你是一个有用的助手，会用简洁清晰的语言回答用户问题<|im_end|>\n"
               "<|im_start|>user\n%1<|im_end|>\n"
               "<|im_start|>assistant\n"
               ).arg(userInput);
}

void LlamaBackend::generateStreaming(const QString &prompt, TokenCallback callback)
{
    if (!isModelLoaded()) {
        qDebug() << "❌ 模型未加载，无法生成";
        return;
    }

    // 🔴 核心修复2：每次生成前彻底重置 KV 缓存，确保全新状态
    llama_kv_self_clear(m_ctx);
    qDebug() << "✅ KV 缓存已重置";

    m_stopGeneration = false;
    const QString full_prompt = buildPrompt(prompt);

    const llama_vocab *vocab = llama_model_get_vocab(m_model);
    if (!vocab) {
        qDebug() << "❌ llama_model_get_vocab 返回空指针";
        return;
    }

    const QByteArray prompt_bytes = full_prompt.toUtf8();
    std::vector<llama_token> tokens(prompt_bytes.size() + 4);
    const int32_t n_tokens = llama_tokenize(
        vocab,
        prompt_bytes.constData(),
        static_cast<int32_t>(prompt_bytes.size()),
        tokens.data(),
        static_cast<int32_t>(tokens.size()),
        true,
        false
        );

    if (n_tokens <= 0) {
        qDebug() << "❌ llama_tokenize 失败，n_tokens:" << n_tokens;
        return;
    }
    tokens.resize(n_tokens);

    const int32_t n_ctx = llama_n_ctx(m_ctx);
    // 🔴 核心修复3：用固定大小 2048 初始化 batch，避免动态分配越界
    llama_batch batch = llama_batch_init(n_ctx, 0, 1);
    batch.n_tokens = static_cast<int32_t>(tokens.size());

    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        batch.token[i]     = tokens[i];
        batch.pos[i]       = i;
        batch.n_seq_id[i]   = 1;
        batch.seq_id[i][0]  = 0;
        batch.logits[i]     = (i == batch.n_tokens - 1);
    }

    if (llama_decode(m_ctx, batch) != 0) {
        qDebug() << "❌ 首次 llama_decode 失败";
        llama_batch_free(batch);
        return;
    }

    // 🔴 核心修复4：限制生成长度 512，避免上下文溢出
    const int32_t n_max = 512;
    int32_t n_cur = batch.n_tokens;

    // 🔴 核心修复5：用贪婪采样器，每次生成重新初始化，避免状态残留
    llama_sampler *smpl = llama_sampler_init_greedy();
    if (!smpl) {
        qDebug() << "❌ 采样器初始化失败";
        llama_batch_free(batch);
        return;
    }

    while (n_cur <= n_max && !m_stopGeneration) {
        const llama_token id = llama_sampler_sample(smpl, m_ctx, -1);
        if (id == LLAMA_TOKEN_NULL) {
            qDebug() << "❌ 采样到空 token，终止生成";
            break;
        }

        const llama_token eos_token = llama_vocab_eos(vocab);
        if (id == eos_token) {
            qDebug() << "✅ 生成结束，遇到 EOS token";
            break;
        }

        char buf[128];
        int32_t n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
        if (n <= 0) continue;

        QString token = QString::fromUtf8(buf, n);

        // 🔴 终极过滤：彻底过滤特殊 token
        if (token == "<" || token == "|" || token == ">" ||
            token.contains("im_end") || token.contains("<|im_end|>") ||
            token.contains("<|im_start|>") || token == "_end") {
            continue;
        }

        if (callback) {
            callback(token);
        }

        // 🔴 核心修复6：b5001 原生重置 batch（零越界、零残留）
        // 直接赋值新 token，无需 memset，避免越界
        batch.n_tokens = 1;
        batch.token[0]     = id;
        batch.pos[0]       = n_cur;
        batch.n_seq_id[0]   = 1;
        batch.seq_id[0][0]  = 0;
        batch.logits[0]     = 1;

        if (llama_decode(m_ctx, batch) != 0) {
            qDebug() << "❌ 生成阶段 llama_decode 失败，n_cur:" << n_cur;
            break;
        }

        ++n_cur;
    }

    llama_sampler_free(smpl);
    llama_batch_free(batch);
    qDebug() << "✅ 生成循环结束，资源释放完成";
}

void LlamaBackend::stopGeneration()
{
    m_stopGeneration = true;
    if (m_ctx) {
        llama_kv_self_clear(m_ctx);
    }
}
