/**
 * DeepVQE model — all block ops, forward pass, and GGUF loading.
 *
 * Extracted from deepvqe.cpp for use by both the CLI tool and C API.
 * All block ops verified against PyTorch (max error < 6e-6).
 */

#include "deepvqe_model.h"

#include "ggml.h"
#include "gguf.h"

#include <cmath>
#include <cstdio>
#include <cstring>

// ── Tensor helpers ─────────────────────────────────────────────────────────

static const NpyArray& W(const deepvqe_model& m, const std::string& name) {
    auto it = m.tensors.find(name);
    if (it == m.tensors.end()) {
        fprintf(stderr, "Missing tensor: %s\n", name.c_str());
        static NpyArray empty;
        return empty;
    }
    return it->second;
}

// ── Constants ─────────────────────────────────────────────────────────────

// DFT basis vectors for 3-tap complex convolving mask (CCM).
static const float VR[3] = {1.0f, -0.5f, -0.5f};
static const float VI[3] = {0.0f, 0.86602540378f, -0.86602540378f};

// ── Primitive ops ──────────────────────────────────────────────────────────

static inline float elu_f(float x) {
    return x > 0.0f ? x : std::exp(x) - 1.0f;
}

static inline float sigmoid_f(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

static void conv2d(
    const float* input, int C_in, int T_in, int F_in,
    const float* weight, const float* bias,
    int C_out, int kT, int kF, int sT, int sF,
    float* output, int T_out, int F_out
) {
    for (int co = 0; co < C_out; co++) {
        for (int t = 0; t < T_out; t++) {
            for (int f = 0; f < F_out; f++) {
                float sum = bias ? bias[co] : 0.0f;
                for (int ci = 0; ci < C_in; ci++) {
                    for (int kt = 0; kt < kT; kt++) {
                        for (int kf = 0; kf < kF; kf++) {
                            sum += weight[co * C_in * kT * kF + ci * kT * kF + kt * kF + kf] *
                                   input[ci * T_in * F_in + (t * sT + kt) * F_in + (f * sF + kf)];
                        }
                    }
                }
                output[co * T_out * F_out + t * F_out + f] = sum;
            }
        }
    }
}

static Buf zero_pad(const Buf& x, int pl, int pr, int pt, int pb) {
    Buf out(x.C, x.T + pt + pb, x.F + pl + pr);
    for (int c = 0; c < x.C; c++)
        for (int t = 0; t < x.T; t++)
            for (int f = 0; f < x.F; f++)
                out(c, t + pt, f + pl) = x(c, t, f);
    return out;
}

// Causal Conv2d with ZeroPad2d([1,1,3,0]) + kernel (4,3)
static Buf causal_conv(const Buf& x, const float* w, const float* b, int C_out, int sT, int sF) {
    Buf padded = zero_pad(x, 1, 1, 3, 0);
    int T_out = (padded.T - 4) / sT + 1;
    int F_out = (padded.F - 3) / sF + 1;
    Buf out(C_out, T_out, F_out);
    conv2d(padded.ptr(), x.C, padded.T, padded.F, w, b, C_out, 4, 3, sT, sF, out.ptr(), T_out, F_out);
    return out;
}

// ── Block ops (verified against PyTorch) ───────────────────────────────────

static void fe(const float* stft, float* out, int F, int T, float c) {
    const float eps = 1e-12f;
    for (int f = 0; f < F; f++) {
        for (int t = 0; t < T; t++) {
            int idx = f * T * 2 + t * 2;
            float re = stft[idx], im = stft[idx + 1];
            float mag = std::sqrt(re * re + im * im + eps);
            float s = std::pow(mag, c - 1.0f) / (1.0f + eps);
            out[0 * T * F + t * F + f] = re * s;
            out[1 * T * F + t * F + f] = im * s;
        }
    }
}

static Buf encoder_block(const Buf& x, const std::string& prefix, const deepvqe_model& m) {
    auto& cw = W(m, prefix + ".conv.weight");
    auto& cb = W(m, prefix + ".conv.bias");
    auto& rw = W(m, prefix + ".resblock.conv.weight");
    auto& rb = W(m, prefix + ".resblock.conv.bias");
    int C_out = (int)cw.dim(0);

    // Main: pad -> conv(stride 1,2) -> elu
    Buf y = causal_conv(x, cw.data.data(), cb.data.data(), C_out, 1, 2);
    for (auto& v : y.data) v = elu_f(v);

    // Residual: pad -> conv(stride 1,1) -> elu + skip
    Buf res = causal_conv(y, rw.data.data(), rb.data.data(), C_out, 1, 1);
    for (int64_t i = 0; i < res.numel(); i++)
        res.data[i] = elu_f(res.data[i]) + y.data[i];
    return res;
}

static Buf bottleneck(const Buf& x, const deepvqe_model& m) {
    int C = x.C, T = x.T, F = x.F;
    int input_size = C * F;

    auto& wih = W(m, "bottleneck.gru.weight_ih_l0");
    auto& whh = W(m, "bottleneck.gru.weight_hh_l0");
    auto& bih = W(m, "bottleneck.gru.bias_ih_l0");
    auto& bhh = W(m, "bottleneck.gru.bias_hh_l0");
    auto& fc_w = W(m, "bottleneck.fc.weight");
    auto& fc_b = W(m, "bottleneck.fc.bias");
    int hidden_size = (int)whh.dim(1);

    // Reshape: (C,T,F) -> (T, C*F)
    std::vector<float> flat(T * input_size);
    for (int t = 0; t < T; t++)
        for (int c = 0; c < C; c++)
            for (int f = 0; f < F; f++)
                flat[t * input_size + c * F + f] = x(c, t, f);

    // GRU forward
    std::vector<float> hidden(hidden_size, 0.0f);
    std::vector<float> gru_out(T * hidden_size);
    std::vector<float> gih(3 * hidden_size), ghh(3 * hidden_size);

    for (int t = 0; t < T; t++) {
        const float* xt = flat.data() + t * input_size;
        for (int i = 0; i < 3 * hidden_size; i++) {
            float s = bih.data[i];
            for (int j = 0; j < input_size; j++) s += wih.data[i * input_size + j] * xt[j];
            gih[i] = s;
        }
        for (int i = 0; i < 3 * hidden_size; i++) {
            float s = bhh.data[i];
            for (int j = 0; j < hidden_size; j++) s += whh.data[i * hidden_size + j] * hidden[j];
            ghh[i] = s;
        }
        for (int i = 0; i < hidden_size; i++) {
            float r = sigmoid_f(gih[i] + ghh[i]);
            float z = sigmoid_f(gih[hidden_size + i] + ghh[hidden_size + i]);
            float n = std::tanh(gih[2 * hidden_size + i] + r * ghh[2 * hidden_size + i]);
            hidden[i] = (1.0f - z) * n + z * hidden[i];
        }
        std::memcpy(gru_out.data() + t * hidden_size, hidden.data(), hidden_size * sizeof(float));
    }

    // Linear: (T, H) -> (T, C*F)
    int out_features = (int)fc_w.dim(0);
    std::vector<float> fc_out(T * out_features);
    for (int t = 0; t < T; t++)
        for (int o = 0; o < out_features; o++) {
            float s = fc_b.data[o];
            for (int j = 0; j < hidden_size; j++) s += fc_w.data[o * hidden_size + j] * gru_out[t * hidden_size + j];
            fc_out[t * out_features + o] = s;
        }

    // Reshape back: (T, C*F) -> (C,T,F)
    Buf out(C, T, F);
    for (int t = 0; t < T; t++)
        for (int c = 0; c < C; c++)
            for (int f = 0; f < F; f++)
                out(c, t, f) = fc_out[t * out_features + c * F + f];
    return out;
}

static Buf align_block(const Buf& x_mic, const Buf& x_ref, const deepvqe_model& m) {
    auto& hp = m.hparams;
    int C = x_mic.C, T = x_mic.T, F = x_mic.F;
    int H = hp.align_hidden, dmax = hp.dmax;
    float temperature = hp.align_temperature;

    auto& pmw = W(m, "align.pconv_mic.weight");
    auto& pmb = W(m, "align.pconv_mic.bias");
    auto& prw = W(m, "align.pconv_ref.weight");
    auto& prb = W(m, "align.pconv_ref.bias");
    auto& sw = W(m, "align.conv.1.weight");
    auto& sb = W(m, "align.conv.1.bias");

    // Q = pconv_mic(x_mic), K = pconv_ref(x_ref) — 1x1 convs
    Buf Q(H, T, F), K(H, T, F);
    for (int h = 0; h < H; h++)
        for (int t = 0; t < T; t++)
            for (int f = 0; f < F; f++) {
                float sq = pmb.data[h], sk = prb.data[h];
                for (int ci = 0; ci < C; ci++) {
                    sq += pmw.data[h * C + ci] * x_mic(ci, t, f);
                    sk += prw.data[h * C + ci] * x_ref(ci, t, f);
                }
                Q(h, t, f) = sq;
                K(h, t, f) = sk;
            }

    // Pad K: (H, T, F) -> (H, T+dmax-1, F)
    int Tp = T + dmax - 1;
    Buf Kp(H, Tp, F);
    for (int h = 0; h < H; h++)
        for (int t = 0; t < T; t++)
            for (int f = 0; f < F; f++)
                Kp(h, t + dmax - 1, f) = K(h, t, f);

    // Similarity: V[h,t,d] = sum_f Q[h,t,f] * Kp[h,t+d,f] / sqrt(F)
    float scale = 1.0f / std::sqrt((float)F);
    std::vector<float> V(H * T * dmax, 0.0f);
    for (int h = 0; h < H; h++)
        for (int t = 0; t < T; t++)
            for (int d = 0; d < dmax; d++) {
                float s = 0.0f;
                for (int f = 0; f < F; f++)
                    s += Q(h, t, f) * Kp(h, t + d, f);
                V[(h * T + t) * dmax + d] = s * scale;
            }

    // Smooth: pad(1,1,4,0) + Conv2d(H,1,(5,3)) -> (1,T,dmax)
    int Tv = T + 4, Dv = dmax + 2;
    std::vector<float> Vp(H * Tv * Dv, 0.0f);
    for (int h = 0; h < H; h++)
        for (int t = 0; t < T; t++)
            for (int d = 0; d < dmax; d++)
                Vp[h * Tv * Dv + (t + 4) * Dv + (d + 1)] = V[(h * T + t) * dmax + d];

    std::vector<float> Vc(T * dmax);
    conv2d(Vp.data(), H, Tv, Dv, sw.data.data(), sb.data.data(), 1, 5, 3, 1, 1, Vc.data(), T, dmax);

    // Softmax over delay
    for (auto& v : Vc) v /= temperature;
    for (int t = 0; t < T; t++) {
        float* row = Vc.data() + t * dmax;
        float mx = row[0];
        for (int d = 1; d < dmax; d++) if (row[d] > mx) mx = row[d];
        float sm = 0.0f;
        for (int d = 0; d < dmax; d++) { row[d] = std::exp(row[d] - mx); sm += row[d]; }
        for (int d = 0; d < dmax; d++) row[d] /= sm;
    }

    // Pad x_ref and weighted sum
    Buf rp(C, Tp, F);
    for (int c = 0; c < C; c++)
        for (int t = 0; t < T; t++)
            for (int f = 0; f < F; f++)
                rp(c, t + dmax - 1, f) = x_ref(c, t, f);

    Buf aligned(C, T, F);
    for (int c = 0; c < C; c++)
        for (int t = 0; t < T; t++)
            for (int d = 0; d < dmax; d++) {
                float a = Vc[t * dmax + d];
                for (int f = 0; f < F; f++)
                    aligned(c, t, f) += a * rp(c, t + d, f);
            }
    return aligned;
}

static Buf decoder_block(const Buf& x, const Buf& x_en,
                         const std::string& prefix, const deepvqe_model& m, bool is_last) {
    int C = x.C, T = x.T, F = x.F;

    auto& skw = W(m, prefix + ".skip_conv.weight");
    auto& skb = W(m, prefix + ".skip_conv.bias");
    auto& rw = W(m, prefix + ".resblock.conv.weight");
    auto& rb = W(m, prefix + ".resblock.conv.bias");
    auto& dw = W(m, prefix + ".deconv.conv.weight");
    auto& db = W(m, prefix + ".deconv.conv.bias");
    int C_out = (int)dw.dim(0) / 2;

    // y = x + skip_conv(x_en) — 1x1 conv
    Buf y(C, T, F);
    for (int co = 0; co < C; co++)
        for (int t = 0; t < T; t++)
            for (int f = 0; f < F; f++) {
                float s = skb.data[co];
                for (int ci = 0; ci < C; ci++)
                    s += skw.data[co * C + ci] * x_en(ci, t, f);
                y(co, t, f) = x(co, t, f) + s;
            }

    // ResidualBlock: pad -> conv -> elu + skip
    Buf res = causal_conv(y, rw.data.data(), rb.data.data(), C, 1, 1);
    for (int64_t i = 0; i < res.numel(); i++)
        res.data[i] = elu_f(res.data[i]) + y.data[i];

    // SubpixelConv2d: pad -> conv(C_out*2) -> pixel shuffle
    Buf padded = zero_pad(res, 1, 1, 3, 0);
    int Tc = padded.T - 4 + 1;
    int Fc = padded.F - 3 + 1;
    int C_conv = C_out * 2;
    Buf conv_out(C_conv, Tc, Fc);
    conv2d(padded.ptr(), C, padded.T, padded.F,
           dw.data.data(), db.data.data(), C_conv, 4, 3, 1, 1,
           conv_out.ptr(), Tc, Fc);

    // Pixel shuffle: rearrange("(r c) t f -> c t (r f)", r=2)
    int F_out = 2 * Fc;
    Buf deconv(C_out, Tc, F_out);
    for (int c = 0; c < C_out; c++)
        for (int t = 0; t < Tc; t++)
            for (int r = 0; r < 2; r++)
                for (int f = 0; f < Fc; f++)
                    deconv(c, t, r * Fc + f) = conv_out(r * C_out + c, t, f);

    // ChannelAffine + ELU (if not is_last)
    if (!is_last) {
        auto& bns = W(m, prefix + ".bn.scale");
        auto& bnb = W(m, prefix + ".bn.bias");
        for (int c = 0; c < C_out; c++)
            for (int t = 0; t < Tc; t++)
                for (int f = 0; f < F_out; f++)
                    deconv(c, t, f) = elu_f(deconv(c, t, f) * bns.data[c] + bnb.data[c]);
    }
    return deconv;
}

// Trim frequency dimension to target
static Buf freq_trim(const Buf& x, int target_F) {
    if (x.F <= target_F) return x;
    Buf out(x.C, x.T, target_F);
    for (int c = 0; c < x.C; c++)
        for (int t = 0; t < x.T; t++)
            std::memcpy(&out(c, t, 0), &x(c, t, 0), target_F * sizeof(float));
    return out;
}

static Buf concat_channels(const Buf& a, const Buf& b) {
    Buf out(a.C + b.C, a.T, a.F);
    for (int c = 0; c < a.C; c++)
        std::memcpy(out.data.data() + c * a.T * a.F, a.data.data() + c * a.T * a.F, a.T * a.F * sizeof(float));
    for (int c = 0; c < b.C; c++)
        std::memcpy(out.data.data() + (a.C + c) * a.T * a.F, b.data.data() + c * b.T * b.F, b.T * b.F * sizeof(float));
    return out;
}

static NpyArray ccm(const Buf& mask, const float* stft, int F_stft, int T, const deepvqe_model& /*m*/) {
    int F = mask.F;

    // H_real, H_imag: (9, T, F)
    std::vector<float> Hr(9 * T * F, 0.0f), Hi(9 * T * F, 0.0f);
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 9; c++) {
            int ch = r * 9 + c;
            for (int t = 0; t < T; t++)
                for (int f = 0; f < F; f++) {
                    float val = mask(ch, t, f);
                    Hr[c * T * F + t * F + f] += VR[r] * val;
                    Hi[c * T * F + t * F + f] += VI[r] * val;
                }
        }

    // Pad input: stft (F,T,2) -> x_perm (2,T,F) -> padded (2, T+2, F+2)
    int Tp = T + 2, Fp = F + 2;
    std::vector<float> xp(2 * Tp * Fp, 0.0f);
    for (int c = 0; c < 2; c++)
        for (int t = 0; t < T; t++)
            for (int f = 0; f < F; f++)
                xp[c * Tp * Fp + (t + 2) * Fp + (f + 1)] = stft[f * T * 2 + t * 2 + c];

    // Complex multiply and sum
    std::vector<float> er(T * F, 0.0f), ei(T * F, 0.0f);
    for (int mn = 0; mn < 3; mn++)
        for (int nn = 0; nn < 3; nn++) {
            int ki = mn * 3 + nn;
            for (int t = 0; t < T; t++)
                for (int f = 0; f < F; f++) {
                    int tf = t * F + f;
                    float mr = Hr[ki * T * F + tf], mi = Hi[ki * T * F + tf];
                    float xr = xp[0 * Tp * Fp + (t + mn) * Fp + (f + nn)];
                    float xi = xp[1 * Tp * Fp + (t + mn) * Fp + (f + nn)];
                    er[tf] += mr * xr - mi * xi;
                    ei[tf] += mr * xi + mi * xr;
                }
        }

    // Output: (1, F, T, 2)
    NpyArray out;
    out.shape = {1, (int64_t)F, (int64_t)T, 2};
    out.data.resize(F * T * 2);
    for (int f = 0; f < F; f++)
        for (int t = 0; t < T; t++) {
            out.data[f * T * 2 + t * 2 + 0] = er[t * F + f];
            out.data[f * T * 2 + t * 2 + 1] = ei[t * F + f];
        }
    return out;
}

// ── Full forward pass ──────────────────────────────────────────────────────

NpyArray forward(const NpyArray& mic_stft, const NpyArray& ref_stft,
                 const deepvqe_model& m, const std::string& dump_dir, bool verbose) {
    auto& hp = m.hparams;
    int F = (int)mic_stft.dim(1);
    int T = (int)mic_stft.dim(2);
    bool dump = !dump_dir.empty();

    auto save = [&](const std::string& name, const Buf& b) {
        if (!dump) return;
        npy_save(dump_dir + "/" + name + ".npy", b.ptr(),
                 {1, (int64_t)b.C, (int64_t)b.T, (int64_t)b.F});
    };

    if (verbose) printf("Forward: F=%d T=%d\n", F, T);

    // 1. Feature extraction
    Buf mic_fe(2, T, F), ref_fe(2, T, F);
    fe(mic_stft.data.data(), mic_fe.ptr(), F, T, hp.power_law_c);
    fe(ref_stft.data.data(), ref_fe.ptr(), F, T, hp.power_law_c);
    save("fe_mic", mic_fe);
    save("fe_ref", ref_fe);

    // 2. Mic encoder 1-2
    Buf mic_e1 = encoder_block(mic_fe, "mic_enc1", m);
    Buf mic_e2 = encoder_block(mic_e1, "mic_enc2", m);
    save("mic_enc1", mic_e1);
    save("mic_enc2", mic_e2);

    // 3. Far-end encoder 1-2
    Buf far_e1 = encoder_block(ref_fe, "far_enc1", m);
    Buf far_e2 = encoder_block(far_e1, "far_enc2", m);
    save("far_enc1", far_e1);
    save("far_enc2", far_e2);

    // 4. Alignment
    Buf aligned = align_block(mic_e2, far_e2, m);
    save("align", aligned);

    // 5. Concat + encoder 3-5
    Buf cat = concat_channels(mic_e2, aligned);
    Buf mic_e3 = encoder_block(cat, "mic_enc3", m);
    Buf mic_e4 = encoder_block(mic_e3, "mic_enc4", m);
    Buf mic_e5 = encoder_block(mic_e4, "mic_enc5", m);
    save("mic_enc3", mic_e3);
    save("mic_enc4", mic_e4);
    save("mic_enc5", mic_e5);

    // 6. Bottleneck
    Buf bn = bottleneck(mic_e5, m);
    save("bottleneck", bn);

    // 7. Decoder with skip connections + frequency trimming
    Buf d5_raw = decoder_block(bn, mic_e5, "dec5", m, false);
    save("dec5", d5_raw);
    Buf d5 = freq_trim(d5_raw, mic_e4.F);

    Buf d4_raw = decoder_block(d5, mic_e4, "dec4", m, false);
    save("dec4", d4_raw);
    Buf d4 = freq_trim(d4_raw, mic_e3.F);

    Buf d3_raw = decoder_block(d4, mic_e3, "dec3", m, false);
    save("dec3", d3_raw);
    Buf d3 = freq_trim(d3_raw, mic_e2.F);

    Buf d2_raw = decoder_block(d3, mic_e2, "dec2", m, false);
    save("dec2", d2_raw);
    Buf d2 = freq_trim(d2_raw, mic_e1.F);

    Buf d1_raw = decoder_block(d2, mic_e1, "dec1", m, true);
    save("dec1", d1_raw);
    Buf d1 = freq_trim(d1_raw, mic_fe.F);

    if (verbose) printf("Decoder output: C=%d T=%d F=%d\n", d1.C, d1.T, d1.F);

    // 8. CCM
    NpyArray enhanced = ccm(d1, mic_stft.data.data(), F, T, m);
    if (dump)
        npy_save(dump_dir + "/output.npy", enhanced);

    if (verbose)
        printf("Enhanced: (%lld, %lld, %lld, %lld)\n",
               (long long)enhanced.dim(0), (long long)enhanced.dim(1),
               (long long)enhanced.dim(2), (long long)enhanced.dim(3));
    return enhanced;
}

// ── Streaming frame ops ──────────────────────────────────────────────────

// Frequency dimension after encoder conv with stride 2.
static int enc_f_out(int f_in) { return (f_in - 1) / 2 + 1; }

// Single-frame causal conv: (C_in, 1, F) + history → (C_out, 1, F_out)
static Buf causal_conv_frame(const Buf& x, conv_history& hist,
                              const float* w, const float* b,
                              int C_out, int sF) {
    int C_in = x.C, F = x.F, Fp = F + 2;
    int hT = hist.history_T, kT = hT + 1;

    // Freq-pad current frame: (C_in, Fp)
    std::vector<float> cur(C_in * Fp, 0.0f);
    for (int c = 0; c < C_in; c++)
        for (int f = 0; f < F; f++)
            cur[c * Fp + f + 1] = x(c, 0, f);

    // Virtual input: (C_in, kT, Fp) = history + current
    std::vector<float> vi(C_in * kT * Fp);
    for (int c = 0; c < C_in; c++) {
        std::memcpy(&vi[c * kT * Fp],
                     &hist.buf[c * hT * Fp], hT * Fp * sizeof(float));
        std::memcpy(&vi[c * kT * Fp + hT * Fp],
                     &cur[c * Fp], Fp * sizeof(float));
    }

    int F_out = (Fp - 3) / sF + 1;
    Buf out(C_out, 1, F_out);
    conv2d(vi.data(), C_in, kT, Fp, w, b, C_out, kT, 3, 1, sF,
           out.ptr(), 1, F_out);

    // Update history: shift left, store current
    for (int c = 0; c < C_in; c++) {
        float* base = &hist.buf[c * hT * Fp];
        std::memmove(base, base + Fp, (hT - 1) * Fp * sizeof(float));
        std::memcpy(base + (hT - 1) * Fp, &cur[c * Fp], Fp * sizeof(float));
    }
    return out;
}

// Feature extraction for single STFT frame (F*2 interleaved) → (2, 1, F)
static Buf fe_frame(const float* stft_frame, int F, float c) {
    Buf out(2, 1, F);
    const float eps = 1e-12f;
    for (int f = 0; f < F; f++) {
        float re = stft_frame[f * 2], im = stft_frame[f * 2 + 1];
        float mag = std::sqrt(re * re + im * im + eps);
        float s = std::pow(mag, c - 1.0f) / (1.0f + eps);
        out(0, 0, f) = re * s;
        out(1, 0, f) = im * s;
    }
    return out;
}

static Buf encoder_block_frame(const Buf& x, const std::string& prefix,
                                const deepvqe_model& m,
                                conv_history& hc, conv_history& hr) {
    auto& cw = W(m, prefix + ".conv.weight");
    auto& cb = W(m, prefix + ".conv.bias");
    auto& rw = W(m, prefix + ".resblock.conv.weight");
    auto& rb = W(m, prefix + ".resblock.conv.bias");
    int C_out = (int)cw.dim(0);

    Buf y = causal_conv_frame(x, hc, cw.data.data(), cb.data.data(), C_out, 2);
    for (auto& v : y.data) v = elu_f(v);

    Buf res = causal_conv_frame(y, hr, rw.data.data(), rb.data.data(), C_out, 1);
    for (int64_t i = 0; i < res.numel(); i++)
        res.data[i] = elu_f(res.data[i]) + y.data[i];
    return res;
}

static Buf bottleneck_frame(const Buf& x, const deepvqe_model& m,
                             std::vector<float>& hidden) {
    int C = x.C, F = x.F, input_size = C * F;
    auto& wih = W(m, "bottleneck.gru.weight_ih_l0");
    auto& whh = W(m, "bottleneck.gru.weight_hh_l0");
    auto& bih = W(m, "bottleneck.gru.bias_ih_l0");
    auto& bhh = W(m, "bottleneck.gru.bias_hh_l0");
    auto& fc_w = W(m, "bottleneck.fc.weight");
    auto& fc_b = W(m, "bottleneck.fc.bias");
    int hidden_size = (int)whh.dim(1);

    // (C, 1, F) is already contiguous as (C*F) when T=1
    const float* xt = x.ptr();

    // Single GRU step
    std::vector<float> gih(3 * hidden_size), ghh(3 * hidden_size);
    for (int i = 0; i < 3 * hidden_size; i++) {
        float s = bih.data[i];
        for (int j = 0; j < input_size; j++)
            s += wih.data[i * input_size + j] * xt[j];
        gih[i] = s;
    }
    for (int i = 0; i < 3 * hidden_size; i++) {
        float s = bhh.data[i];
        for (int j = 0; j < hidden_size; j++)
            s += whh.data[i * hidden_size + j] * hidden[j];
        ghh[i] = s;
    }
    for (int i = 0; i < hidden_size; i++) {
        float r = sigmoid_f(gih[i] + ghh[i]);
        float z = sigmoid_f(gih[hidden_size + i] + ghh[hidden_size + i]);
        float n = std::tanh(gih[2*hidden_size + i] + r * ghh[2*hidden_size + i]);
        hidden[i] = (1.0f - z) * n + z * hidden[i];
    }

    // Linear → (C*F)
    int out_features = (int)fc_w.dim(0);
    std::vector<float> fc_out(out_features);
    for (int o = 0; o < out_features; o++) {
        float s = fc_b.data[o];
        for (int j = 0; j < hidden_size; j++)
            s += fc_w.data[o * hidden_size + j] * hidden[j];
        fc_out[o] = s;
    }

    Buf out(C, 1, F);
    for (int c = 0; c < C; c++)
        for (int f = 0; f < F; f++)
            out(c, 0, f) = fc_out[c * F + f];
    return out;
}

static Buf align_block_frame(const Buf& x_mic, const Buf& x_ref,
                              const deepvqe_model& m,
                              deepvqe_stream_state& st) {
    int C = x_mic.C, F = x_mic.F;
    int H = m.hparams.align_hidden, dmax = m.hparams.dmax;

    auto& pmw = W(m, "align.pconv_mic.weight");
    auto& pmb = W(m, "align.pconv_mic.bias");
    auto& prw = W(m, "align.pconv_ref.weight");
    auto& prb = W(m, "align.pconv_ref.bias");
    auto& sw  = W(m, "align.conv.1.weight");
    auto& sb  = W(m, "align.conv.1.bias");

    // 1x1 Q, K projections
    Buf Q(H, 1, F), K(H, 1, F);
    for (int h = 0; h < H; h++)
        for (int f = 0; f < F; f++) {
            float sq = pmb.data[h], sk = prb.data[h];
            for (int ci = 0; ci < C; ci++) {
                sq += pmw.data[h * C + ci] * x_mic(ci, 0, f);
                sk += prw.data[h * C + ci] * x_ref(ci, 0, f);
            }
            Q(h, 0, f) = sq;
            K(h, 0, f) = sk;
        }

    // Push K into history: shift left, store current at dmax-1
    for (int h = 0; h < H; h++) {
        float* base = st.align_K_hist.data() + h * dmax * F;
        std::memmove(base, base + F, (dmax - 1) * F * sizeof(float));
        std::memcpy(base + (dmax - 1) * F, &K(h, 0, 0), F * sizeof(float));
    }

    // Similarity: V[h, d] = sum_f Q[h,f] * K_hist[h,d,f] / sqrt(F)
    float scale = 1.0f / std::sqrt((float)F);
    Buf V(H, 1, dmax);
    for (int h = 0; h < H; h++)
        for (int d = 0; d < dmax; d++) {
            float s = 0.0f;
            const float* kp = st.align_K_hist.data() + h * dmax * F + d * F;
            for (int f = 0; f < F; f++)
                s += Q(h, 0, f) * kp[f];
            V(h, 0, d) = s * scale;
        }

    // Smooth conv frame: kernel (5,3), history_T=4
    Buf Vc = causal_conv_frame(V, st.align_smooth,
                                sw.data.data(), sb.data.data(), 1, 1);

    // Apply softmax temperature (matches batch path; default 1.0 = no scaling)
    float temperature = m.hparams.align_temperature;
    if (temperature != 1.0f) {
        for (int d = 0; d < dmax; d++) Vc(0, 0, d) /= temperature;
    }

    // Softmax over delay
    float mx = Vc(0, 0, 0);
    for (int d = 1; d < dmax; d++)
        if (Vc(0, 0, d) > mx) mx = Vc(0, 0, d);
    float sm = 0.0f;
    for (int d = 0; d < dmax; d++) {
        Vc(0, 0, d) = std::exp(Vc(0, 0, d) - mx);
        sm += Vc(0, 0, d);
    }
    for (int d = 0; d < dmax; d++) Vc(0, 0, d) /= sm;

    // Push x_ref into ref history
    for (int c = 0; c < C; c++) {
        float* base = st.align_ref_hist.data() + c * dmax * F;
        std::memmove(base, base + F, (dmax - 1) * F * sizeof(float));
        std::memcpy(base + (dmax - 1) * F, &x_ref(c, 0, 0), F * sizeof(float));
    }

    // Weighted sum over delay
    Buf aligned(C, 1, F);
    for (int c = 0; c < C; c++)
        for (int d = 0; d < dmax; d++) {
            float a = Vc(0, 0, d);
            const float* rp = st.align_ref_hist.data() + c * dmax * F + d * F;
            for (int f = 0; f < F; f++)
                aligned(c, 0, f) += a * rp[f];
        }
    return aligned;
}

static Buf decoder_block_frame(const Buf& x, const Buf& x_en,
                                const std::string& prefix, const deepvqe_model& m,
                                conv_history& hr, conv_history& hd, bool is_last) {
    int C = x.C, F = x.F;
    auto& skw = W(m, prefix + ".skip_conv.weight");
    auto& skb = W(m, prefix + ".skip_conv.bias");
    auto& rw  = W(m, prefix + ".resblock.conv.weight");
    auto& rb  = W(m, prefix + ".resblock.conv.bias");
    auto& dw  = W(m, prefix + ".deconv.conv.weight");
    auto& db  = W(m, prefix + ".deconv.conv.bias");
    int C_out = (int)dw.dim(0) / 2;

    // y = x + skip_conv(x_en)
    Buf y(C, 1, F);
    for (int co = 0; co < C; co++)
        for (int f = 0; f < F; f++) {
            float s = skb.data[co];
            for (int ci = 0; ci < C; ci++)
                s += skw.data[co * C + ci] * x_en(ci, 0, f);
            y(co, 0, f) = x(co, 0, f) + s;
        }

    // Residual
    Buf res = causal_conv_frame(y, hr, rw.data.data(), rb.data.data(), C, 1);
    for (int64_t i = 0; i < res.numel(); i++)
        res.data[i] = elu_f(res.data[i]) + y.data[i];

    // SubpixelConv2d: causal conv → pixel shuffle
    Buf conv_out = causal_conv_frame(res, hd, dw.data.data(), db.data.data(),
                                      C_out * 2, 1);
    int Fc = conv_out.F, F_out = 2 * Fc;
    Buf deconv(C_out, 1, F_out);
    for (int c = 0; c < C_out; c++)
        for (int r = 0; r < 2; r++)
            for (int f = 0; f < Fc; f++)
                deconv(c, 0, r * Fc + f) = conv_out(r * C_out + c, 0, f);

    if (!is_last) {
        auto& bns = W(m, prefix + ".bn.scale");
        auto& bnb = W(m, prefix + ".bn.bias");
        for (int c = 0; c < C_out; c++)
            for (int f = 0; f < F_out; f++)
                deconv(c, 0, f) = elu_f(deconv(c, 0, f) * bns.data[c] + bnb.data[c]);
    }
    return deconv;
}

static void ccm_frame(const Buf& mask, const float* mic_stft_frame,
                       int F, deepvqe_stream_state& st,
                       float* enhanced) {
    int Fp = F + 2;

    // Build H_real, H_imag: (9, F)
    std::vector<float> Hr(9 * F, 0.0f), Hi(9 * F, 0.0f);
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 9; c++) {
            int ch = r * 9 + c;
            for (int f = 0; f < F; f++) {
                Hr[c * F + f] += VR[r] * mask(ch, 0, f);
                Hi[c * F + f] += VI[r] * mask(ch, 0, f);
            }
        }

    // Build padded input: (2, 3, Fp) from 2 history frames + current
    // History is (2, 2, Fp): [re/im][older/newer][freq_padded]
    std::vector<float> xp(2 * 3 * Fp, 0.0f);
    for (int c = 0; c < 2; c++) {
        std::memcpy(&xp[c * 3 * Fp + 0 * Fp],
                     &st.ccm_stft_hist[c * 2 * Fp + 0 * Fp], Fp * sizeof(float));
        std::memcpy(&xp[c * 3 * Fp + 1 * Fp],
                     &st.ccm_stft_hist[c * 2 * Fp + 1 * Fp], Fp * sizeof(float));
    }
    // Freq-pad current frame into position 2
    for (int f = 0; f < F; f++) {
        xp[0 * 3 * Fp + 2 * Fp + f + 1] = mic_stft_frame[f * 2 + 0];
        xp[1 * 3 * Fp + 2 * Fp + f + 1] = mic_stft_frame[f * 2 + 1];
    }

    // Complex multiply and sum
    std::vector<float> er(F, 0.0f), ei(F, 0.0f);
    for (int mn = 0; mn < 3; mn++)
        for (int nn = 0; nn < 3; nn++) {
            int ki = mn * 3 + nn;
            for (int f = 0; f < F; f++) {
                float mr = Hr[ki * F + f], mi = Hi[ki * F + f];
                float xr = xp[0 * 3 * Fp + mn * Fp + f + nn];
                float xi = xp[1 * 3 * Fp + mn * Fp + f + nn];
                er[f] += mr * xr - mi * xi;
                ei[f] += mr * xi + mi * xr;
            }
        }

    // Output: (F, 1, 2) interleaved
    for (int f = 0; f < F; f++) {
        enhanced[f * 2 + 0] = er[f];
        enhanced[f * 2 + 1] = ei[f];
    }

    // Update CCM history: shift left, store current (freq-padded)
    for (int c = 0; c < 2; c++) {
        float* base = st.ccm_stft_hist.data() + c * 2 * Fp;
        std::memcpy(base, base + Fp, Fp * sizeof(float));
        std::memset(base + Fp, 0, Fp * sizeof(float));
    }
    for (int f = 0; f < F; f++) {
        st.ccm_stft_hist[0 * 2 * Fp + 1 * Fp + f + 1] = mic_stft_frame[f * 2 + 0];
        st.ccm_stft_hist[1 * 2 * Fp + 1 * Fp + f + 1] = mic_stft_frame[f * 2 + 1];
    }
}

// ── Streaming forward pass ────────────────────────────────────────────────

void forward_frame(const float* mic_stft_frame, const float* ref_stft_frame,
                   const deepvqe_model& m, deepvqe_stream_state& st,
                   float* enhanced_stft_frame) {
    auto& hp = m.hparams;
    int F = hp.n_freq_bins;

    // 1. Feature extraction
    Buf mic_fe = fe_frame(mic_stft_frame, F, hp.power_law_c);
    Buf ref_fe = fe_frame(ref_stft_frame, F, hp.power_law_c);

    // 2. Mic encoder 1-2
    Buf mic_e1 = encoder_block_frame(mic_fe, "mic_enc1", m,
                                      st.mic_enc_conv[0], st.mic_enc_res[0]);
    Buf mic_e2 = encoder_block_frame(mic_e1, "mic_enc2", m,
                                      st.mic_enc_conv[1], st.mic_enc_res[1]);

    // 3. Far-end encoder 1-2
    Buf far_e1 = encoder_block_frame(ref_fe, "far_enc1", m,
                                      st.far_enc_conv[0], st.far_enc_res[0]);
    Buf far_e2 = encoder_block_frame(far_e1, "far_enc2", m,
                                      st.far_enc_conv[1], st.far_enc_res[1]);

    // 4. Alignment
    Buf aligned = align_block_frame(mic_e2, far_e2, m, st);

    // 5. Concat + encoder 3-5
    Buf cat = concat_channels(mic_e2, aligned);
    Buf mic_e3 = encoder_block_frame(cat, "mic_enc3", m,
                                      st.mic_enc_conv[2], st.mic_enc_res[2]);
    Buf mic_e4 = encoder_block_frame(mic_e3, "mic_enc4", m,
                                      st.mic_enc_conv[3], st.mic_enc_res[3]);
    Buf mic_e5 = encoder_block_frame(mic_e4, "mic_enc5", m,
                                      st.mic_enc_conv[4], st.mic_enc_res[4]);

    // 6. Bottleneck
    Buf bn = bottleneck_frame(mic_e5, m, st.gru_hidden);

    // 7. Decoder with skip connections + freq trim
    Buf d5 = freq_trim(decoder_block_frame(bn, mic_e5, "dec5", m,
                        st.dec_res[0], st.dec_deconv[0], false), mic_e4.F);
    Buf d4 = freq_trim(decoder_block_frame(d5, mic_e4, "dec4", m,
                        st.dec_res[1], st.dec_deconv[1], false), mic_e3.F);
    Buf d3 = freq_trim(decoder_block_frame(d4, mic_e3, "dec3", m,
                        st.dec_res[2], st.dec_deconv[2], false), mic_e2.F);
    Buf d2 = freq_trim(decoder_block_frame(d3, mic_e2, "dec2", m,
                        st.dec_res[3], st.dec_deconv[3], false), mic_e1.F);
    Buf d1 = freq_trim(decoder_block_frame(d2, mic_e1, "dec1", m,
                        st.dec_res[4], st.dec_deconv[4], true), mic_fe.F);

    // 8. CCM
    ccm_frame(d1, mic_stft_frame, F, st, enhanced_stft_frame);
}

// ── Streaming state init/reset ────────────────────────────────────────────

void init_stream_state(deepvqe_stream_state& st, const deepvqe_model& m) {
    auto& hp = m.hparams;
    int hop = hp.hop_length;

    // STFT / iSTFT buffers
    st.stft_prev_mic.assign(hop, 0.0f);
    st.stft_prev_ref.assign(hop, 0.0f);
    st.istft_prev.assign(hp.n_fft, 0.0f);

    // Compute freq dims at each encoder level
    int F[6];
    F[0] = hp.n_freq_bins;
    for (int i = 0; i < 5; i++) F[i + 1] = enc_f_out(F[i]);

    // Mic encoder conv histories
    auto& mc = hp.mic_channels;
    int mic_enc_cin[5], mic_enc_fin[5];
    mic_enc_cin[0] = mc[0];   mic_enc_fin[0] = F[0];
    mic_enc_cin[1] = mc[1];   mic_enc_fin[1] = F[1];
    mic_enc_cin[2] = mc[2]*2; mic_enc_fin[2] = F[2]; // concat doubles channels
    mic_enc_cin[3] = mc[3];   mic_enc_fin[3] = F[3];
    mic_enc_cin[4] = mc[4];   mic_enc_fin[4] = F[4];
    for (int i = 0; i < 5; i++) {
        st.mic_enc_conv[i].init(mic_enc_cin[i], mic_enc_fin[i] + 2, 3);
        st.mic_enc_res[i].init(mc[i + 1], F[i + 1] + 2, 3);
    }

    // Far encoder conv histories
    auto& fc = hp.far_channels;
    st.far_enc_conv[0].init(fc[0], F[0] + 2, 3);
    st.far_enc_res[0].init(fc[1], F[1] + 2, 3);
    st.far_enc_conv[1].init(fc[1], F[1] + 2, 3);
    st.far_enc_res[1].init(fc[2], F[2] + 2, 3);

    // AlignBlock
    int H = hp.align_hidden, dmax = hp.dmax;
    int Fa = F[2]; // align operates at encoder level 2
    int Ca = mc[2];
    st.align_smooth.init(H, dmax + 2, 4); // kernel (5,3), history_T=4
    st.align_K_hist.assign(H * dmax * Fa, 0.0f);
    st.align_ref_hist.assign(Ca * dmax * Fa, 0.0f);

    // GRU hidden
    auto& whh = W(m, "bottleneck.gru.weight_hh_l0");
    int hidden_size = whh.data.empty() ? 576 : (int)whh.dim(1);
    st.gru_hidden.assign(hidden_size, 0.0f);

    // Decoder conv histories: dec5..dec1 → indices 0..4
    // Trace decoder channel widths from weight tensors
    static const char* dec_names[5] = {"dec5","dec4","dec3","dec2","dec1"};
    int dec_F[5] = {F[5], F[4], F[3], F[2], F[1]}; // input F for each decoder
    int dec_C = mc[5]; // dec5 input C = bottleneck output = enc5 C_out
    for (int i = 0; i < 5; i++) {
        st.dec_res[i].init(dec_C, dec_F[i] + 2, 3);
        st.dec_deconv[i].init(dec_C, dec_F[i] + 2, 3);
        // Next decoder's C = this decoder's C_out
        std::string dw_name = std::string(dec_names[i]) + ".deconv.conv.weight";
        auto& dw = W(m, dw_name);
        dec_C = dw.data.empty() ? dec_C : (int)dw.dim(0) / 2;
    }

    // CCM STFT history: (2, 2, F+2) — 2 frames, freq-padded
    int Fp = hp.n_freq_bins + 2;
    st.ccm_stft_hist.assign(2 * 2 * Fp, 0.0f);

    st.frame_count = 0;
}

void reset_stream_state(deepvqe_stream_state& st) {
    std::fill(st.stft_prev_mic.begin(), st.stft_prev_mic.end(), 0.0f);
    std::fill(st.stft_prev_ref.begin(), st.stft_prev_ref.end(), 0.0f);
    std::fill(st.istft_prev.begin(), st.istft_prev.end(), 0.0f);
    for (int i = 0; i < 5; i++) {
        st.mic_enc_conv[i].reset();
        st.mic_enc_res[i].reset();
    }
    for (int i = 0; i < 2; i++) {
        st.far_enc_conv[i].reset();
        st.far_enc_res[i].reset();
    }
    st.align_smooth.reset();
    std::fill(st.align_K_hist.begin(), st.align_K_hist.end(), 0.0f);
    std::fill(st.align_ref_hist.begin(), st.align_ref_hist.end(), 0.0f);
    std::fill(st.gru_hidden.begin(), st.gru_hidden.end(), 0.0f);
    for (int i = 0; i < 5; i++) {
        st.dec_res[i].reset();
        st.dec_deconv[i].reset();
    }
    std::fill(st.ccm_stft_hist.begin(), st.ccm_stft_hist.end(), 0.0f);
    st.frame_count = 0;
}

// ── GGUF loading ───────────────────────────────────────────────────────────

static uint32_t gguf_u32(struct gguf_context* ctx, const char* key) {
    int idx = gguf_find_key(ctx, key);
    return idx >= 0 ? gguf_get_val_u32(ctx, idx) : 0;
}

bool load_model(const char* path, deepvqe_model& model, bool verbose) {
    struct ggml_context* ggml_ctx = nullptr;
    struct gguf_init_params params;
    params.no_alloc = false;
    params.ctx = &ggml_ctx;

    struct gguf_context* gctx = gguf_init_from_file(path, params);
    if (!gctx) { fprintf(stderr, "Failed to load: %s\n", path); return false; }

    auto& hp = model.hparams;
    hp.n_fft        = (int)gguf_u32(gctx, "deepvqe.n_fft");
    hp.hop_length   = (int)gguf_u32(gctx, "deepvqe.hop_length");
    hp.n_freq_bins  = (int)gguf_u32(gctx, "deepvqe.n_freq_bins");
    hp.sample_rate  = (int)gguf_u32(gctx, "deepvqe.sample_rate");
    hp.dmax         = (int)gguf_u32(gctx, "deepvqe.dmax");
    hp.align_hidden = (int)gguf_u32(gctx, "deepvqe.align_hidden");
    int idx = gguf_find_key(gctx, "deepvqe.power_law_c");
    hp.power_law_c = idx >= 0 ? gguf_get_val_f32(gctx, idx) : 0.3f;
    idx = gguf_find_key(gctx, "deepvqe.bn_folded");
    hp.bn_folded = idx >= 0 ? gguf_get_val_bool(gctx, idx) : true;
    idx = gguf_find_key(gctx, "deepvqe.align_temperature");
    hp.align_temperature = idx >= 0 ? gguf_get_val_f32(gctx, idx) : 1.0f;

    int mic_n = (int)gguf_u32(gctx, "deepvqe.mic_channels.count");
    hp.mic_channels.resize(mic_n);
    for (int i = 0; i < mic_n; i++) {
        char k[64]; snprintf(k, sizeof(k), "deepvqe.mic_channels.%d", i);
        hp.mic_channels[i] = (int)gguf_u32(gctx, k);
    }
    int far_n = (int)gguf_u32(gctx, "deepvqe.far_channels.count");
    hp.far_channels.resize(far_n);
    for (int i = 0; i < far_n; i++) {
        char k[64]; snprintf(k, sizeof(k), "deepvqe.far_channels.%d", i);
        hp.far_channels[i] = (int)gguf_u32(gctx, k);
    }

    if (verbose)
        printf("Config: n_fft=%d hop=%d dmax=%d c=%.2f\n", hp.n_fft, hp.hop_length, hp.dmax, hp.power_law_c);

    int n_tensors = gguf_get_n_tensors(gctx);
    for (int i = 0; i < n_tensors; i++) {
        const char* name = gguf_get_tensor_name(gctx, i);
        NpyArray arr = load_tensor_from_ggml(ggml_ctx, name, gctx, verbose);
        if (arr.data.empty()) continue;
        model.tensors[name] = std::move(arr);
    }
    if (verbose) printf("Loaded %zu tensors\n", model.tensors.size());

    gguf_free(gctx);
    if (ggml_ctx) ggml_free(ggml_ctx);
    return true;
}
