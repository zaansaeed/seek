// M0 spike: can a small BERT cross-encoder (the future student's shape) run from C++ through llama.cpp?
// Off-the-shelf model: cross-encoder/ms-marco-MiniLM-L6-v2, converted to an f32 GGUF.
//
// llama.cpp runs BERT differently from how this model was trained: the converter drops the pooler
// layer, and token types are hardcoded to 0. So we don't expect the model card's scores. Instead we
// check that llama.cpp computes exactly what we think it does:
//   1. Token IDs identical to the Hugging Face tokenizer.
//   2. Scores match PyTorch "B" (no pooler, all token types 0) within 0.01.
// Reference values come from build/student_ref.txt, written by spike/ref_student.py.
// Throwaway: the real student wrapper lands in src/model/ in M5.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "llama.h"

using namespace std;
namespace fs = std::filesystem;

struct BackendGuard {
    BackendGuard() { llama_backend_init(); }
    ~BackendGuard() { llama_backend_free(); }
};

// One line of build/student_ref.txt.
struct Reference {
    float score_a = NAN;  // PyTorch, model as trained
    float score_b = NAN;  // PyTorch, llama.cpp-equivalent
    vector<llama_token> ids;
};

// Format per line: <score_a> <score_b> <n> <id_1> ... <id_n>
vector<Reference> load_reference(const string& path) {
    vector<Reference> refs;
    ifstream file(path);
    string line;
    while (getline(file, line)) {
        // istringstream reads whitespace-separated values out of a string, like cin does from the terminal.
        istringstream in(line);
        Reference r;
        size_t n = 0;
        in >> r.score_a >> r.score_b >> n;
        r.ids.resize(n);
        for (size_t i = 0; i < n; ++i) in >> r.ids[i];
        if (in) refs.push_back(r);  // `in` is false if any read failed
    }
    return refs;
}

vector<llama_token> tokenize_text(const llama_vocab* vocab, const string& text, bool add_special) {
    vector<llama_token> tokens(text.length() + 3);
    int32_t n = llama_tokenize(vocab, text.c_str(), (int32_t)text.length(),
                               tokens.data(), (int32_t)tokens.size(), add_special, false);
    if (n < 0) {
        tokens.resize(-n);
        n = llama_tokenize(vocab, text.c_str(), (int32_t)text.length(),
                           tokens.data(), (int32_t)tokens.size(), add_special, false);
    }
    if (n < 0) return {};
    tokens.resize(n);
    return tokens;
}

// BERT pair format: [CLS] query [SEP] document [SEP]
vector<llama_token> make_pair(const llama_vocab* vocab, const string& query, const string& doc) {
    vector<llama_token> tokens = tokenize_text(vocab, query, true);  // adds [CLS] ... [SEP]
    vector<llama_token> doc_tokens = tokenize_text(vocab, doc, false);
    tokens.insert(tokens.end(), doc_tokens.begin(), doc_tokens.end());
    tokens.push_back(llama_vocab_sep(vocab));
    return tokens;
}

// Run one pair; returns the classifier's single output logit, or NAN on failure.
float student_score(llama_context* ctx, const vector<llama_token>& tokens) {
    if (tokens.empty()) return NAN;
    int n_tokens = (int)tokens.size();

    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    batch.n_tokens = n_tokens;
    for (int i = 0; i < n_tokens; ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = true;
    }

    // BERT has no KV cache, so llama_decode hands the batch to llama_encode internally,
    // and llama_memory_clear does nothing (it accepts a null memory). Kept for the same shape as the other spikes.
    float score = NAN;
    if (llama_decode(ctx, batch) == 0) {
        // RANK pooling on a non-causal model takes the first token ([CLS]) and applies cls.output.
        const float* out = llama_get_embeddings_seq(ctx, 0);
        if (out != nullptr) score = out[0];
    } else {
        cerr << "Error: llama_decode failed.\n";
    }
    llama_batch_free(batch);
    llama_memory_clear(llama_get_memory(ctx), true);
    return score;
}

int main(int argc, char* argv[]) {
    string model_path = (argc > 1) ? argv[1] : "models/ms-marco-MiniLM-L6-v2-f32.gguf";
    string ref_path = (argc > 2) ? argv[2] : "build/student_ref.txt";
    if (!fs::exists(model_path)) {
        cerr << "Error: Model file does not exist at '" << model_path << "'\n";
        return 1;
    }
    vector<Reference> refs = load_reference(ref_path);
    if (refs.empty()) {
        cerr << "Error: no reference data at '" << ref_path << "'. Run: python spike/ref_student.py\n";
        return 1;
    }

    llama_log_set([](enum ggml_log_level level, const char* text, void* /*user_data*/) {
        if (level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR) fputs(text, stderr);
    }, nullptr);

    BackendGuard guard;

    unique_ptr<llama_model, decltype(&llama_model_free)> model(
        llama_model_load_from_file(model_path.c_str(), llama_model_default_params()), &llama_model_free);
    if (!model) {
        fprintf(stderr, "error: unable to load model\n");
        return 1;
    }
    const llama_vocab* vocab = llama_model_get_vocab(model.get());
    printf("Loaded %s: %u classifier output(s)\n", model_path.c_str(), llama_model_n_cls_out(model.get()));

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.embeddings = true;
    ctx_params.pooling_type = LLAMA_POOLING_TYPE_RANK;  // this GGUF records no pooling type, so set it here
    ctx_params.n_ctx = 512;                             // BERT's maximum sequence length
    ctx_params.n_batch = 512;                           // non-causal: a whole pair must fit in one batch
    ctx_params.n_ubatch = 512;

    unique_ptr<llama_context, decltype(&llama_free)> ctx(
        llama_init_from_model(model.get(), ctx_params), &llama_free);
    if (!ctx) {
        fprintf(stderr, "error: unable to create context\n");
        return 1;
    }

    // Same pairs, same order, as spike/ref_student.py. The token-ID check catches any text mismatch.
    const string gravity_doc =
        "Gravity is a force that attracts two bodies towards each other. It gives weight to physical "
        "objects and is responsible for the movement of planets around the sun.";
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

    struct Pair { string label, query, doc; };
    const vector<Pair> pairs = {
        {"Berlin / population", "How many people live in Berlin?",
         "Berlin had a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers."},
        {"Berlin / museums", "How many people live in Berlin?", "Berlin is well known for its museums."},
        {"China / Beijing", "What is the capital of China?", "The capital of China is Beijing."},
        {"China / gravity", "What is the capital of China?", gravity_doc},
        {"retry / snippet A", "where do we retry failed uploads?", snippet_a},
        {"retry / snippet B", "where do we retry failed uploads?", snippet_b},
    };
    if (refs.size() != pairs.size()) {
        fprintf(stderr, "error: reference has %zu pairs, spike has %zu\n", refs.size(), pairs.size());
        return 1;
    }

    // Warm-up run, excluded from timing.
    student_score(ctx.get(), make_pair(vocab, pairs[0].query, pairs[0].doc));

    const float tolerance = 0.01f;
    bool tokens_ok = true;
    float max_diff = 0.0f;
    double total_ms = 0.0;

    printf("\n%-20s %10s %12s %12s %8s  %s\n", "pair", "C++", "PyTorch B", "PyTorch A", "|C++-B|", "tokens");
    for (size_t i = 0; i < pairs.size(); ++i) {
        vector<llama_token> tokens = make_pair(vocab, pairs[i].query, pairs[i].doc);
        bool same_tokens = (tokens == refs[i].ids);  // vector == compares length and every element
        tokens_ok = tokens_ok && same_tokens;

        auto t0 = chrono::steady_clock::now();
        float s = student_score(ctx.get(), tokens);
        auto t1 = chrono::steady_clock::now();
        total_ms += chrono::duration<double, milli>(t1 - t0).count();

        float diff = fabsf(s - refs[i].score_b);
        if (isnan(diff)) diff = INFINITY;
        if (diff > max_diff) max_diff = diff;

        printf("%-20s %+10.4f %+12.4f %+12.4f %8.4f  %s (%zu)\n", pairs[i].label.c_str(), s, refs[i].score_b,
               refs[i].score_a, diff, same_tokens ? "same" : "DIFFERENT", tokens.size());
        if (!same_tokens) {
            printf("    C++ ids:");
            for (llama_token t : tokens) printf(" %d", t);
            printf("\n    ref ids:");
            for (llama_token t : refs[i].ids) printf(" %d", t);
            printf("\n");
        }
    }

    bool scores_ok = max_diff <= tolerance;
    printf("\nToken IDs identical to Hugging Face: %s\n", tokens_ok ? "PASS" : "FAIL");
    printf("Largest |C++ - PyTorch B|: %.5f -> %s (tolerance %.2f)\n", max_diff, scores_ok ? "PASS" : "FAIL",
           tolerance);
    printf("Time per pair (one at a time, %zu pairs after warm-up): %.2f ms mean\n", pairs.size(),
           total_ms / pairs.size());

    return (tokens_ok && scores_ok) ? 0 : 1;
}
