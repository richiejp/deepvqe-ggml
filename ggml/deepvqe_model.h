#pragma once

/**
 * DeepVQE model — internal C++ structs and inference.
 * Used by both the CLI tool and the C API shared library.
 */

#include "common.h"

#include <map>
#include <string>
#include <vector>

struct deepvqe_hparams {
    int n_fft        = 512;
    int hop_length   = 256;
    int n_freq_bins  = 257;
    int sample_rate  = 16000;
    int dmax         = 32;
    int align_hidden = 32;
    // AlignBlock softmax temperature. Default 1.0 preserves behaviour for
    // GGUFs exported before the metadata key existed; new exports embed the
    // trained value (typically the align_temp_end of the curriculum).
    float align_temperature = 1.0f;
    float power_law_c = 0.3f;
    bool bn_folded   = true;
    std::vector<int> mic_channels;
    std::vector<int> far_channels;
};

struct deepvqe_model {
    deepvqe_hparams hparams;
    std::map<std::string, NpyArray> tensors;
};

struct Buf {
    std::vector<float> data;
    int C, T, F;

    Buf() : C(0), T(0), F(0) {}
    Buf(int c, int t, int f) : data(c * t * f, 0.0f), C(c), T(t), F(f) {}

    int64_t numel() const { return (int64_t)C * T * F; }
    float& operator()(int c, int t, int f) { return data[c * T * F + t * F + f]; }
    const float& operator()(int c, int t, int f) const { return data[c * T * F + t * F + f]; }
    float* ptr() { return data.data(); }
    const float* ptr() const { return data.data(); }
};

// ── Streaming state ───────────────────────────────────────────────────────

/// Shift-buffer history for a single causal conv layer.
struct conv_history {
    std::vector<float> buf;  // (C, history_T, F_padded) — zero-initialized
    int C = 0;
    int F_padded = 0;        // F + 2 (freq padding applied)
    int history_T = 0;       // 3 for standard causal_conv, 4 for align smooth

    void init(int c, int f_padded, int ht) {
        C = c; F_padded = f_padded; history_T = ht;
        buf.assign(c * ht * f_padded, 0.0f);
    }
    void reset() { std::fill(buf.begin(), buf.end(), 0.0f); }
    float& at(int c, int t, int f) { return buf[c * history_T * F_padded + t * F_padded + f]; }
    const float& at(int c, int t, int f) const { return buf[c * history_T * F_padded + t * F_padded + f]; }
};

struct deepvqe_stream_state {
    int frame_count = 0;

    // STFT analysis: previous hop for mic and ref (256 samples each)
    std::vector<float> stft_prev_mic;
    std::vector<float> stft_prev_ref;

    // iSTFT overlap-add buffer (n_fft samples, second half carries over)
    std::vector<float> istft_prev;

    // Encoder causal conv histories (main conv + residual)
    conv_history mic_enc_conv[5];
    conv_history mic_enc_res[5];
    conv_history far_enc_conv[2];
    conv_history far_enc_res[2];

    // AlignBlock
    conv_history align_smooth;                // (H, 4, dmax+2)
    std::vector<float> align_K_hist;          // (H, dmax, F_enc2)
    std::vector<float> align_ref_hist;        // (C_enc2, dmax, F_enc2)

    // Bottleneck GRU hidden state
    std::vector<float> gru_hidden;

    // Decoder causal conv histories (residual + deconv)
    conv_history dec_res[5];
    conv_history dec_deconv[5];

    // CCM: last 2 STFT frames (2, 2, F+2) freq-padded
    std::vector<float> ccm_stft_hist;
};

// ── Public API ────────────────────────────────────────────────────────────

/// Load model from GGUF file. Returns true on success.
bool load_model(const char* path, deepvqe_model& model, bool verbose = true);

/// Full forward pass (batch). Returns enhanced STFT (1, F, T, 2).
/// dump_dir: if non-empty, save intermediates as .npy files.
NpyArray forward(const NpyArray& mic_stft, const NpyArray& ref_stft,
                 const deepvqe_model& m,
                 const std::string& dump_dir = "", bool verbose = true);

/// Initialize streaming state from loaded model (sizes buffers from weights).
void init_stream_state(deepvqe_stream_state& state, const deepvqe_model& m);

/// Reset streaming state to zeros (call between utterances).
void reset_stream_state(deepvqe_stream_state& state);

/// Single-frame forward pass for streaming.
/// mic_stft_frame, ref_stft_frame: (F, 1, 2) = 257 complex values each.
/// enhanced_stft_frame: (F, 1, 2) output.
void forward_frame(const float* mic_stft_frame, const float* ref_stft_frame,
                   const deepvqe_model& m, deepvqe_stream_state& state,
                   float* enhanced_stft_frame);
