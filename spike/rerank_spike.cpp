// M0 spike: run the teacher reranker (Qwen3-Reranker-0.6B) through llama.cpp and check its scores.
//   1. Reference check: reproduce the model card's scores for "What is the capital of China?"
//      (+7.625 vs Beijing, -11.375 vs gravity; PASS if within 0.5).
//   2. Relevant beats irrelevant for both reference queries (the M0 checkbox).
//   3. Code example: an upload-retry question against two code snippets.
//   4. Rough time per pair, after a warm-up.
// Score = logit difference ln(P(yes) / P(no)), the same scale the model card reports.
// Throwaway: the real batched reranker lands in src/model/ in M4.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "llama.h"

using namespace std;
namespace fs = std::filesystem;

struct BackendGuard {
    BackendGuard() { llama_backend_init(); }
    ~BackendGuard() { llama_backend_free(); }
};

// Replace the first occurrence of `key` in `text` with `value`.
// string::npos is the "not found" value that find() returns.
string replace_once(string text, const string& key, const string& value) {
    size_t pos = text.find(key);
    if (pos != string::npos) {
        text.replace(pos, key.length(), value);
    }
    return text;
}

// Fill the model's built-in rerank template with one (query, document) pair.
string make_prompt(const string& tmpl, const string& query, const string& document) {
    return replace_once(replace_once(tmpl, "{query}", query), "{document}", document);
}

vector<llama_token> tokenize_text(const llama_vocab* vocab, const string& text, bool add_special, bool parse_special) {
    vector<llama_token> tokens(text.length() + 3);
    int32_t n_tokens = llama_tokenize(vocab, text.c_str(), (int32_t)text.length(),
                                      tokens.data(), (int32_t)tokens.size(), add_special, parse_special);
    if (n_tokens < 0) {  // buffer too small: -n_tokens is the size needed
        tokens.resize(-n_tokens);
        n_tokens = llama_tokenize(vocab, text.c_str(), (int32_t)text.length(),
                                  tokens.data(), (int32_t)tokens.size(), add_special, parse_special);
    }
    if (n_tokens < 0) {
        cerr << "Error: Failed to tokenize text.\n";
        return {};
    }
    tokens.resize(n_tokens);
    return tokens;
}

// Run one prompt through the reranker and return ln(P(yes) / P(no)), or NAN on failure.
float rerank_score(llama_context* ctx, const vector<llama_token>& tokens) {
    if (tokens.empty()) return NAN;
    int n_tokens = (int)tokens.size();

    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    batch.n_tokens = n_tokens;
    for (int i = 0; i < n_tokens; ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = true;  // with embeddings on, every token must be an output
    }

    llama_memory_t mem = llama_get_memory(ctx);
    float score = NAN;
    if (llama_decode(ctx, batch) != 0) {
        cerr << "Error: llama_decode failed.\n";
    } else {
        // With RANK pooling this is float[n_cls_out] = {P(yes), P(no)}: llama.cpp already applied softmax.
        const float* out = llama_get_embeddings_seq(ctx, 0);
        if (out == nullptr) {
            cerr << "Error: no rerank output.\n";
        } else {
            // Undo the softmax: ln(p_yes / p_no) = logit_yes - logit_no.
            // fmax guards against log(0) if a probability underflows.
            score = logf(fmaxf(out[0], 1e-30f)) - logf(fmaxf(out[1], 1e-30f));
        }
    }

    // Same rule as the embedder: wipe the memory so the next pair starts fresh.
    llama_batch_free(batch);
    llama_memory_clear(mem, true);
    return score;
}

int main(int argc, char* argv[]) {
    string default_path = "models/qwen3-reranker-0.6b-q8_0.gguf";
    string model_path = (argc > 1) ? argv[1] : default_path;
    if (!fs::exists(model_path)) {
        cerr << "Error: Model file does not exist at '" << model_path << "'\n";
        return 1;
    }

    // Keep warnings and errors only.
    llama_log_set([](enum ggml_log_level level, const char* text, void* /*user_data*/) {
        if (level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR) {
            fputs(text, stderr);
        }
    }, nullptr);

    BackendGuard guard;

    cout << "Loading model from: " << model_path << "\n";
    unique_ptr<llama_model, decltype(&llama_model_free)> model(
        llama_model_load_from_file(model_path.c_str(), llama_model_default_params()), &llama_model_free);
    if (!model) {
        fprintf(stderr, "error: unable to load model\n");
        return 1;
    }
    const llama_vocab* vocab = llama_model_get_vocab(model.get());

    // Confirm the classifier head: expect 2 outputs labelled yes, no.
    uint32_t n_cls = llama_model_n_cls_out(model.get());
    printf("Classifier outputs: %u", n_cls);
    for (uint32_t i = 0; i < n_cls; ++i) {
        const char* label = llama_model_cls_label(model.get(), i);
        printf("%s%s", i == 0 ? " (" : ", ", label ? label : "?");
    }
    printf("%s (expect 2: yes, no)\n", n_cls ? ")" : "");

    // The prompt template is stored in the GGUF file.
    const char* tmpl_c = llama_model_chat_template(model.get(), "rerank");
    if (tmpl_c == nullptr) {
        fprintf(stderr, "error: model has no 'rerank' template\n");
        return 1;
    }
    const string tmpl = tmpl_c;

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.embeddings = true;  // pooling (RANK) is read from the GGUF file
    ctx_params.n_ctx = 1024;       // template is ~70 tokens; leaves room for the document
    ctx_params.n_batch = 1024;
    ctx_params.n_ubatch = 1024;

    unique_ptr<llama_context, decltype(&llama_free)> ctx(
        llama_init_from_model(model.get(), ctx_params), &llama_free);
    if (!ctx) {
        fprintf(stderr, "error: unable to create context\n");
        return 1;
    }

    // Template uses <|im_start|> etc. itself: add no specials, but parse the ones in the text as control tokens.
    vector<double> pair_ms;  // time per pair, for the rough cost estimate
    auto score = [&](const string& query, const string& document) {
        vector<llama_token> tokens = tokenize_text(vocab, make_prompt(tmpl, query, document), false, true);
        auto t0 = chrono::steady_clock::now();
        float s = rerank_score(ctx.get(), tokens);
        auto t1 = chrono::steady_clock::now();
        // duration<double, milli> = a time span counted in milliseconds, as a double.
        pair_ms.push_back(chrono::duration<double, milli>(t1 - t0).count());
        return s;
    };

    const string china = "What is the capital of China?";
    const string gravity_q = "Explain gravity";
    const string beijing_doc = "The capital of China is Beijing.";
    const string gravity_doc =
        "Gravity is a force that attracts two bodies towards each other. It gives weight to physical "
        "objects and is responsible for the movement of planets around the sun.";

    // Warm-up: the first run pays one-off GPU setup costs; drop it from the timings.
    score(china, beijing_doc);
    pair_ms.clear();

    // ---- 1. Reference check (model card CrossEncoder example) ----
    const float tolerance = 0.5f;
    float s_cb = score(china, beijing_doc);
    float s_cg = score(china, gravity_doc);
    float d1 = fabsf(s_cb - 7.625f), d2 = fabsf(s_cg - (-11.375f));
    float max_diff = (isnan(d1) || isnan(d2)) ? INFINITY : fmaxf(d1, d2);
    bool ref_pass = max_diff <= tolerance;

    printf("\nReference check: \"%s\"   ours (model card)\n", china.c_str());
    printf("  vs Beijing doc   %+8.3f (%+8.3f)\n", s_cb, 7.625f);
    printf("  vs gravity doc   %+8.3f (%+8.3f)\n", s_cg, -11.375f);
    printf("Largest difference: %.3f -> %s (tolerance %.1f)\n", max_diff, ref_pass ? "PASS" : "FAIL", tolerance);

    // ---- 2. Relevant beats irrelevant, both queries ----
    float s_gg = score(gravity_q, gravity_doc);
    float s_gb = score(gravity_q, beijing_doc);
    bool order_pass = (s_cb > s_cg) && (s_gg > s_gb);
    printf("\nOrdering check: \"%s\"  gravity doc %+.3f  vs  Beijing doc %+.3f\n",
           gravity_q.c_str(), s_gg, s_gb);
    printf("Relevant > irrelevant for both queries -> %s\n", order_pass ? "PASS" : "FAIL");

    // ---- 3. Code example (uses the template's web-search instruction as-is) ----
    const string question = "where do we retry failed uploads?";
    const string snippet_a = R"(def upload_with_retry(client, path, max_attempts=3):
    for attempt in range(max_attempts):
        try:
            return client.put(path)
        except ConnectionError:
            time.sleep(2 ** attempt)
    raise UploadFailed(path))";
    const string snippet_b = R"(def load_config(path):
    with open(path) as f:
        cfg = json.load(f)
    cfg.setdefault("timeout", 30)
    return cfg)";
    float s_a = score(question, snippet_a);
    float s_b = score(question, snippet_b);
    printf("\nCode example: \"%s\"\n", question.c_str());
    printf("  A (retry)  %+8.3f\n  B (config) %+8.3f\n  gap A-B    %+8.3f\n", s_a, s_b, s_a - s_b);

    // ---- 4. Rough cost ----
    double total = 0.0;
    for (double ms : pair_ms) total += ms;
    double mean_ms = total / pair_ms.size();
    printf("\nTime per pair (one at a time, %zu pairs after warm-up): %.1f ms mean\n", pair_ms.size(), mean_ms);
    printf("-> rough cost of reranking N=50 candidates unbatched: %.2f s\n", mean_ms * 50 / 1000.0);

    return (ref_pass && order_pass) ? 0 : 1;
}
