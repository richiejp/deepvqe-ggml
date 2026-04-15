/**
 * Re-quantize a DeepVQE F32 GGUF to Q8_0 with correct 2D tensor shapes.
 *
 * Stores Q8_0 tensors as 2D (KW*KH*IC, OC) for conv or (ne0, ne1) for matmul,
 * enabling native quantized inference through conv_2d_q and ggml_mul_mat.
 *
 * Layers with row_size not divisible by 32 (e.g., IC=2) are kept F32.
 *
 * Usage:
 *   quantize deepvqe.gguf deepvqe-q8.gguf
 */

#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static bool should_quantize(const char* name) {
    std::string s(name);
    if (s.find(".bias") != std::string::npos) return false;
    if (s.find(".bn.") != std::string::npos) return false;
    if (s.rfind("align.", 0) == 0) return false;
    if (s.rfind("dec1.", 0) == 0) return false;
    if (s.find(".weight") != std::string::npos) return true;
    if (s.find("weight_ih") != std::string::npos) return true;
    if (s.find("weight_hh") != std::string::npos) return true;
    return false;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: quantize input.gguf output.gguf\n");
        return 1;
    }
    const char* inp_path = argv[1];
    const char* out_path = argv[2];

    const int64_t blk = ggml_blck_size(GGML_TYPE_Q8_0);
    const auto* traits = ggml_get_type_traits(GGML_TYPE_Q8_0);

    printf("Requantizing %s -> %s (Q8_0, block=%lld)\n", inp_path, out_path, (long long)blk);

    struct ggml_context* inp_ctx = nullptr;
    struct gguf_init_params uparams = { /*.no_alloc=*/ false, /*.ctx=*/ &inp_ctx };
    struct gguf_context* inp = gguf_init_from_file(inp_path, uparams);
    if (!inp) { fprintf(stderr, "Failed to load: %s\n", inp_path); return 1; }

    int n_tensors = gguf_get_n_tensors(inp);
    printf("Loaded %d tensors\n", n_tensors);

    struct ggml_init_params qp = { (size_t)32 * 1024 * 1024, nullptr, false };
    struct ggml_context* q_ctx = ggml_init(qp);

    struct gguf_context* out = gguf_init_empty();
    gguf_set_kv(out, inp);

    int n_quantized = 0, n_f32 = 0;
    size_t total_inp = 0, total_out = 0;
    std::vector<std::vector<uint8_t>> qdata_storage;

    for (int i = 0; i < n_tensors; i++) {
        const char* name = gguf_get_tensor_name(inp, i);
        struct ggml_tensor* t = ggml_get_tensor(inp_ctx, name);
        if (!t) continue;

        size_t inp_bytes = ggml_nbytes(t);
        total_inp += inp_bytes;

        if (t->type != GGML_TYPE_F32 || !should_quantize(name)) {
            gguf_add_tensor(out, t);
            total_out += inp_bytes;
            n_f32++;
            continue;
        }

        int ndim = ggml_n_dims(t);
        int64_t row_size = (ndim == 4) ? t->ne[0] * t->ne[1] * t->ne[2] : t->ne[0];
        int64_t n_rows = (ndim == 4) ? t->ne[3] : t->ne[1];

        if (row_size % blk != 0) {
            printf("  %-42s F32 (row=%lld, not aligned)\n", name, (long long)row_size);
            gguf_add_tensor(out, t);
            total_out += inp_bytes;
            n_f32++;
            continue;
        }

        int64_t n_elem = ggml_nelements(t);
        size_t out_bytes = ggml_row_size(GGML_TYPE_Q8_0, n_elem);
        qdata_storage.emplace_back(out_bytes);
        auto& qdata = qdata_storage.back();
        traits->from_float_ref((float*)t->data, qdata.data(), n_elem);

        struct ggml_tensor* qt = ggml_new_tensor_2d(q_ctx, GGML_TYPE_Q8_0, row_size, n_rows);
        ggml_set_name(qt, name);
        memcpy(qt->data, qdata.data(), out_bytes);

        gguf_add_tensor(out, qt);
        total_out += out_bytes;
        n_quantized++;

        char key[256];
        for (int d = 0; d < ndim; d++) {
            snprintf(key, sizeof(key), "deepvqe.shape.%s.%d", name, d);
            gguf_set_val_u32(out, key, (uint32_t)t->ne[d]);
        }
        snprintf(key, sizeof(key), "deepvqe.shape.%s.ndim", name);
        gguf_set_val_u32(out, key, (uint32_t)ndim);

        printf("  %-42s Q8_0 (%lldx%lld)\n", name, (long long)row_size, (long long)n_rows);
    }

    gguf_write_to_file(out, out_path, false);

    printf("\nDone: %d Q8_0 + %d F32\n", n_quantized, n_f32);
    printf("Size: %.1f MB -> %.1f MB (%.0f%%)\n",
           total_inp / 1e6, total_out / 1e6, total_out * 100.0 / total_inp);

    gguf_free(out);
    gguf_free(inp);
    ggml_free(q_ctx);
    ggml_free(inp_ctx);
    return 0;
}
