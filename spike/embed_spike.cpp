// M0 spike: load Qwen3-Embedding through llama.cpp and check its vectors.
//   1. Reference check: reproduce the model card's 2x2 query/document scores (PASS if within 0.02).
//   2. Code-search check: an upload-retry question against two code snippets, with and without the prefix.
// Throwaway: the real batched embedder lands in src/model/ in M2.

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "llama.h"

using namespace std;
namespace fs = std::filesystem;

const string reference_task = "Given a web search query, retrieve relevant passages that answer the query";
const string code_task = "Given a question about a codebase, retrieve the code that answers it";

struct BackendGuard {
    BackendGuard() {  // constructor
        llama_backend_init();
    }
    ~BackendGuard() {  // destructor
        llama_backend_free();
    }
};

// Queries get the instruction prefix; documents are embedded as plain text.
string make_query(const string& task, const string& question) {
    return "Instruct: " + task + "\nQuery:" + question;
}

vector<llama_token> tokenize_text(const llama_vocab* vocab, const string& text, bool add_special = true) {
    // 1. First guess at the buffer size; one token per byte plus a few specials is almost always enough.
    vector<llama_token> tokens(text.length() + 3);

    // 2. Tokenize. add_special adds the model's special tokens; for this model that is the
    //    end-of-text token (add_eos_token = true, add_bos_token = false in the GGUF).
    int32_t n_tokens = llama_tokenize(vocab, text.c_str(), (int32_t)text.length(),
                                      tokens.data(), (int32_t)tokens.size(), add_special, true);

    // 3. A negative result means "buffer too small": -n_tokens is the size needed. Resize and retry.
    if (n_tokens < 0) {
        tokens.resize(-n_tokens);
        n_tokens = llama_tokenize(vocab, text.c_str(), (int32_t)text.length(),
                                  tokens.data(), (int32_t)tokens.size(), add_special, true);
    }
    if (n_tokens < 0) {
        cerr << "Error: Failed to tokenize text.\n";
        return {};
    }

    // 4. Shrink to the exact number of tokens produced.
    tokens.resize(n_tokens);
    return tokens;
}

vector<float> get_embeddings(llama_context* ctx, const vector<llama_token>& tokens) {
    if (tokens.empty()) return {};

    int n_tokens = (int)tokens.size();

    // 1. Allocate a batch: (n_tokens, embd = 0 because we pass token IDs, n_seq_max = 1)
    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    batch.n_tokens = n_tokens;

    // 2. Fill it: every token is in sequence 0 at position i.
    for (int i = 0; i < n_tokens; ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0; //all have the same seq_id
        // With embeddings on, llama.cpp requires every token to be an output (it overrides with a
        // warning otherwise). Last-token pooling then happens inside llama.cpp.
        batch.logits[i] = true; //output is embeddings
    }

    // 3. Run the model.
    llama_memory_t mem = llama_get_memory(ctx);
    if (llama_decode(ctx, batch) != 0) {
        cerr << "Error: llama_decode failed during embedding generation.\n";
        llama_batch_free(batch);
        llama_memory_clear(mem, true);
        return {};
    }

    // 4. Read the pooled vector for sequence 0.
    int n_embd = llama_model_n_embd(llama_get_model(ctx));
    const float* embd_data = llama_get_embeddings_seq(ctx, 0);

    vector<float> embedding_vector;
    if (embd_data != nullptr) {
        embedding_vector.assign(embd_data, embd_data + n_embd);
    } else {
        cerr << "Error: Failed to retrieve embedding tensor from context.\n";
    }

    // 5. Clean up. Clearing the memory is essential: without it the next text is read as a
    //    continuation of this one, and every later vector is silently wrong.
    llama_batch_free(batch);
    llama_memory_clear(mem, true);

    return embedding_vector;
}

vector<float> normalize(const vector<float>& vec) {
    if (vec.empty()) return {};

    float sum_squares = inner_product(vec.begin(), vec.end(), vec.begin(), 0.0f);
    float mag = sqrt(sum_squares);
    if (mag == 0.0f) return vec;

    vector<float> newVec;
    newVec.reserve(vec.size());  // pre-allocate memory for efficiency
    for (float entry : vec) {
        newVec.push_back(entry / mag);
    }
    return newVec;
}

// Cosine similarity, given vectors already normalized to length 1.
float dot(const vector<float>& a, const vector<float>& b) {
    if (a.size() != b.size() || a.empty()) {
        fprintf(stderr, "error: vectors empty or not equal length\n");
        return NAN;  // can't be mistaken for a real score
    }
    float sum = 0.0f;
    for (size_t i = 0; i < a.size(); i++) {
        sum += a[i] * b[i];
    }
    return sum;
}


int main(int argc, char* argv[]) {
    string default_path = "models/Qwen3-Embedding-0.6B-Q8_0.gguf";
    string model_path = (argc > 1) ? argv[1] : default_path;

    if (!fs::exists(model_path)) {
        cerr << "Error: Model file does not exist at '" << model_path << "'\n";
        return 1;
    }

    // Keep llama.cpp's warnings and errors, drop its info/debug chatter.
    // A lambda with no captures ([]) converts to a plain C function pointer, which is what this C API expects.
    llama_log_set([](enum ggml_log_level level, const char* text, void* /*user_data*/) {
        if (level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR) {
            fputs(text, stderr);
        }
    }, nullptr);

    BackendGuard guard;

    // Model and context are freed automatically, in reverse order of creation (context, then model,
    // then the backend via the guard), including on every early return.
    // decltype(&llama_model_free) = "type of a pointer to llama_model_free": the deleter's type.
    cout << "Loading model from: " << model_path << "\n";
    unique_ptr<llama_model, decltype(&llama_model_free)> model(
        llama_model_load_from_file(model_path.c_str(), llama_model_default_params()), &llama_model_free);
    if (!model) {
        fprintf(stderr, "error: unable to load model\n");
        return 1;
    }
    const llama_vocab* vocab = llama_model_get_vocab(model.get());  // .get() = the raw pointer, still owned

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.embeddings = true;  // return embeddings instead of next-token logits
    ctx_params.n_ctx = 512;        // our texts are short; each must fit in one batch
    ctx_params.n_batch = 512;
    ctx_params.n_ubatch = 512;
    // pooling_type left as UNSPECIFIED, so llama.cpp reads it from the GGUF file.

    unique_ptr<llama_context, decltype(&llama_free)> ctx(
        llama_init_from_model(model.get(), ctx_params), &llama_free);
    if (!ctx) {
        fprintf(stderr, "error: unable to create context\n");
        return 1;
    }

    printf("Embedding dimension: %d (expect 1024)\n", llama_model_n_embd(model.get()));

    // [&] captures local variables (ctx, vocab) by reference, so the lambda can use them.
    auto embed = [&](const string& text) {
        return normalize(get_embeddings(ctx.get(), tokenize_text(vocab, text)));
    };

    // ---- 1. Reference check (model card example) ----
    vector<vector<float>> queries = {
        embed(make_query(reference_task, "What is the capital of China?")),
        embed(make_query(reference_task, "Explain gravity")),
    };
    vector<vector<float>> documents = {
        embed("The capital of China is Beijing."),
        embed("Gravity is a force that attracts two bodies towards each other. It gives weight to physical "
              "objects and is responsible for the movement of planets around the sun."),
    };
    const float reference[2][2] = {{0.7646f, 0.1414f}, {0.1355f, 0.6000f}};
    const float tolerance = 0.02f;

    printf("Reference check: ours (model card)\n");
    printf("      d1                 d2\n");
    float max_diff = 0.0f;
    for (int q = 0; q < 2; ++q) {
        printf("q%d", q + 1);
        for (int d = 0; d < 2; ++d) {
            float score = dot(queries[q], documents[d]);
            float diff = fabs(score - reference[q][d]);
            if (isnan(diff)) diff = INFINITY;  // a failed embedding must fail the check
            if (diff > max_diff) max_diff = diff;
            printf("    %.4f (%.4f)", score, reference[q][d]);
        }
        printf("\n");
    }
    bool pass = max_diff <= tolerance;
    printf("Largest difference: %.4f -> %s (tolerance %.2f)\n\n", max_diff, pass ? "PASS" : "FAIL", tolerance);

    // ---- 2. Code-search check (no reference; snippet A should win) ----
    // R"( ... )" is a raw string: newlines and quotes are kept exactly as written.
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

    vector<float> a = embed(snippet_a);
    vector<float> b = embed(snippet_b);
    vector<float> q_prefixed = embed(make_query(code_task, question));
    vector<float> q_plain = embed(question);

    float pa = dot(q_prefixed, a), pb = dot(q_prefixed, b);
    float ra = dot(q_plain, a), rb = dot(q_plain, b);
    printf("Code-search check: \"%s\"\n", question.c_str());
    printf("                 A (retry)   B (config)   gap A-B\n");
    printf("with prefix      %.4f      %.4f       %+.4f\n", pa, pb, pa - pb);
    printf("without prefix   %.4f      %.4f       %+.4f\n", ra, rb, ra - rb);

    return pass ? 0 : 1;  // exit code 0 only if the reference check passed
}
