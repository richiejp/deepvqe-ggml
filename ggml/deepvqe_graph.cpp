/**
 * DeepVQE inference using GGML computational graphs.
 *
 * Each network block builds a small GGML graph that is dispatched
 * to the selected backend (CPU with SIMD, or CUDA).
 */

#include "deepvqe_graph.h"
#include "gguf.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <thread>

// ── Helpers ───────────────────────────────────────────────────────────────

// Allocate a compute context (for graph node metadata, not tensor data).
static struct ggml_context* make_ctx(size_t mem = 256 * 1024) {
    struct ggml_init_params p = { mem, nullptr, true };
    return ggml_init(p);
}

// Create a 3D input tensor. Marks it as input.
static struct ggml_tensor* input_3d(struct ggml_context* ctx,
                                     int64_t ne0, int64_t ne1, int64_t ne2) {
    struct ggml_tensor* t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ne0, ne1, ne2);
    ggml_set_input(t);
    return t;
}

// ── Conv2d for quantized weights ─────────────────────────────────────────

// ggml_conv_2d requires 4D kernel with ne[0] % block_size == 0, which fails
// for quantized types (KW=3 < 32). This helper does im2col + mul_mat manually.
// For F32 weights, it falls through to ggml_conv_2d.
static struct ggml_tensor* conv_2d_q(
    struct ggml_context* ctx,
    struct ggml_tensor* weight,  // 4D F32 (kW, kH, IC, OC) or 2D quantized (kW*kH*IC, OC)
    struct ggml_tensor* input,   // 4D (W, H, C, N)
    int s0, int s1, int p0, int p1, int d0, int d1,
    int kW, int kH               // kernel dims (used when weight is 2D quantized)
) {
    if (weight->type == GGML_TYPE_F32) {
        return ggml_conv_2d(ctx, weight, input, s0, s1, p0, p1, d0, d1);
    }

    int64_t row_size = weight->ne[0];
    int64_t OC = weight->ne[1];
    int64_t IC = row_size / (kW * kH);

    struct ggml_tensor* shape = ggml_new_tensor_4d(ctx, input->type, kW, kH, IC, OC);

    struct ggml_tensor* im2col = ggml_im2col(ctx, shape, input,
                                              s0, s1, p0, p1, d0, d1,
                                              true, input->type);

    struct ggml_tensor* im2col_2d = ggml_reshape_2d(ctx, im2col,
        im2col->ne[0],
        im2col->ne[3] * im2col->ne[2] * im2col->ne[1]);

    struct ggml_tensor* result = ggml_mul_mat(ctx, weight, im2col_2d);

    result = ggml_reshape_4d(ctx, result,
        im2col->ne[1], im2col->ne[2], im2col->ne[3], OC);
    result = ggml_cont(ctx, ggml_permute(ctx, result, 0, 1, 3, 2));

    return result;
}

// ── Block graphs ──────────────────────────────────────────────────────────

// Feature extraction: STFT (ne0=2, ne1=T, ne2=F) → (ne0=F, ne1=T, ne2=2)
// Applies power-law compression: out = stft * mag^(c-1) / (1+eps)
// where mag = sqrt(re² + im² + eps).
static struct ggml_tensor* build_fe(struct ggml_context* ctx,
                                     struct ggml_tensor* stft, float power_c) {
    // stft: (ne0=2, ne1=T, ne2=F) — complex pair fastest
    int64_t T = stft->ne[1];
    int64_t F = stft->ne[2];
    float eps = 1e-12f;

    // mag² = sum over re²+im² along ne0 → (1, T, F)
    struct ggml_tensor* sqr = ggml_sqr(ctx, stft);
    struct ggml_tensor* mag2 = ggml_sum_rows(ctx, sqr);  // sum ne0
    struct ggml_tensor* mag2_eps = ggml_scale_bias(ctx, mag2, 1.0f, eps);
    struct ggml_tensor* mag = ggml_sqrt(ctx, mag2_eps);

    // s = mag^(c-1) / (1+eps)
    struct ggml_tensor* log_mag = ggml_log(ctx, mag);
    struct ggml_tensor* s = ggml_exp(ctx, ggml_scale(ctx, log_mag, power_c - 1.0f));
    s = ggml_scale(ctx, s, 1.0f / (1.0f + eps));
    // s: (1, T, F) — broadcasts over ne0=2 when multiplied with stft

    // Scale both re and im by s
    struct ggml_tensor* scaled = ggml_mul(ctx, stft, s);
    // scaled: (2, T, F)

    // Permute to (F, T, 2) to match Buf(C=2, T, F) memory layout
    struct ggml_tensor* out = ggml_cont(ctx, ggml_permute(ctx, scaled, 2, 1, 0, 3));
    ggml_set_output(out);
    return out;
}

// Causal conv: pad_ext([1,1,3,0]) + conv2d(kernel 4x3, stride 1xsF) + bias.
// Input: (ne0=F, ne1=T, ne2=C_in, ne3=1)
// Output: (ne0=F_out, ne1=T, ne2=C_out, ne3=1)
static struct ggml_tensor* build_causal_conv(
    struct ggml_context* ctx,
    struct ggml_tensor* x,           // (F, T, C_in)
    struct ggml_tensor* weight,      // (kF=3, kT=4, C_in, C_out)
    struct ggml_tensor* bias,        // (C_out)
    int sF
) {
    // Add batch dim if needed
    if (ggml_n_dims(x) == 3) {
        x = ggml_reshape_4d(ctx, x, x->ne[0], x->ne[1], x->ne[2], 1);
    }

    // Asymmetric causal padding: freq (1 left, 1 right), time (3 top, 0 bottom)
    struct ggml_tensor* padded = ggml_pad_ext(ctx, x,
                                               1, 1,   // dim0 (freq): left=1, right=1
                                               3, 0,   // dim1 (time): left=3, right=0
                                               0, 0,   // dim2 (channel): none
                                               0, 0);   // dim3 (batch): none

    // Conv2d: stride (sF, 1) on (dim0=freq, dim1=time), no additional padding
    struct ggml_tensor* conv = conv_2d_q(ctx, weight, padded,
                                          sF, 1, 0, 0, 1, 1, 3, 4);

    // Add bias: bias is (C_out), conv is (F_out, T, C_out, 1)
    // Reshape bias for broadcasting: (1, 1, C_out, 1)
    struct ggml_tensor* b = ggml_reshape_4d(ctx, bias, 1, 1, bias->ne[0], 1);
    conv = ggml_add(ctx, conv, b);

    return conv;
}

// Encoder block: causal_conv(stride 1,2) → ELU → causal_conv(stride 1,1) → ELU + skip
static struct ggml_tensor* build_encoder_block(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    struct ggml_tensor* conv_w, struct ggml_tensor* conv_b,
    struct ggml_tensor* res_w, struct ggml_tensor* res_b
) {
    // Main branch: stride_F=2
    struct ggml_tensor* y = build_causal_conv(ctx, x, conv_w, conv_b, 2);
    y = ggml_elu(ctx, y);

    // Residual: stride_F=1
    struct ggml_tensor* res = build_causal_conv(ctx, y, res_w, res_b, 1);
    res = ggml_add(ctx, ggml_elu(ctx, res), y);

    return res;
}

// Decoder block: skip_conv + residual + SubpixelConv2d + ChannelAffine + ELU
static struct ggml_tensor* build_decoder_block(
    struct ggml_context* ctx,
    struct ggml_tensor* x,        // (F, T, C)
    struct ggml_tensor* x_en,     // (F, T, C) encoder skip
    struct ggml_tensor* skip_w, struct ggml_tensor* skip_b,
    struct ggml_tensor* res_w, struct ggml_tensor* res_b,
    struct ggml_tensor* deconv_w, struct ggml_tensor* deconv_b,
    struct ggml_tensor* bn_scale, struct ggml_tensor* bn_bias,  // null if is_last
    bool is_last
) {
    int64_t F = x->ne[0], T = x->ne[1], C = x->ne[2];

    // skip_conv(x_en): 1x1 conv
    struct ggml_tensor* x_en_4d = ggml_reshape_4d(ctx, x_en, F, T, C, 1);
    struct ggml_tensor* skip = conv_2d_q(ctx, skip_w, x_en_4d, 1, 1, 0, 0, 1, 1, 1, 1);
    // Bias
    struct ggml_tensor* sb = ggml_reshape_4d(ctx, skip_b, 1, 1, skip_b->ne[0], 1);
    skip = ggml_add(ctx, skip, sb);
    skip = ggml_reshape_3d(ctx, skip, F, T, C);

    // y = x + skip
    struct ggml_tensor* y = ggml_add(ctx, x, skip);

    // Residual
    struct ggml_tensor* res = build_causal_conv(ctx, y, res_w, res_b, 1);
    res = ggml_reshape_3d(ctx, res, F, T, C);
    res = ggml_add(ctx, ggml_elu(ctx, res), y);

    // SubpixelConv2d: causal_conv → pixel shuffle
    int64_t C_out = ((ggml_n_dims(deconv_w) == 4) ? deconv_w->ne[3] : deconv_w->ne[1]) / 2;
    struct ggml_tensor* deconv = build_causal_conv(ctx, res, deconv_w, deconv_b, 1);
    // deconv: (F, T, C_out*2, 1) → strip batch dim
    deconv = ggml_reshape_3d(ctx, deconv, F, T, C_out * 2);

    // Pixel shuffle: (ne0=F, ne1=T, ne2=2*C_out) → (ne0=2*F, ne1=T, ne2=C_out)
    // Split channels: reshape to (F, T, C_out, 2)
    struct ggml_tensor* r1 = ggml_reshape_4d(ctx, deconv, F, T, C_out, 2);
    // Permute to (F, 2, T, C_out): old[0]→0, old[1]→2, old[2]→3, old[3]→1
    struct ggml_tensor* r2 = ggml_permute(ctx, r1, 0, 2, 3, 1);
    // Make contiguous and reshape to (2*F, T, C_out)
    struct ggml_tensor* shuffled = ggml_reshape_3d(ctx, ggml_cont(ctx, r2), 2 * F, T, C_out);

    if (!is_last && bn_scale && bn_bias) {
        // ChannelAffine: x * scale + bias, broadcast over F and T
        struct ggml_tensor* sc = ggml_reshape_3d(ctx, bn_scale, 1, 1, C_out);
        struct ggml_tensor* bi = ggml_reshape_3d(ctx, bn_bias, 1, 1, C_out);
        shuffled = ggml_elu(ctx, ggml_add(ctx, ggml_mul(ctx, shuffled, sc), bi));
    }

    return shuffled;
}

// Freq trim: take first target_F elements of dim 0
static struct ggml_tensor* build_freq_trim(struct ggml_context* ctx,
                                            struct ggml_tensor* x,
                                            int64_t target_F) {
    if (x->ne[0] <= target_F) return x;
    // ggml_view_3d creates non-contiguous tensor; wrap in cont for downstream reshape
    return ggml_cont(ctx,
        ggml_view_3d(ctx, x, target_F, x->ne[1], x->ne[2],
                      x->nb[1], x->nb[2], 0));
}

// AlignBlock: cross-attention soft delay estimation.
// x_mic, x_ref: (ne0=F, ne1=T, ne2=C)
// Returns: (ne0=F, ne1=T, ne2=C) — aligned reference.
static struct ggml_tensor* build_align(
    struct ggml_context* ctx,
    struct ggml_tensor* x_mic,   // (F, T, C)
    struct ggml_tensor* x_ref,   // (F, T, C)
    struct ggml_tensor* pmw, struct ggml_tensor* pmb,  // pconv_mic: (1, 1, C, H)
    struct ggml_tensor* prw, struct ggml_tensor* prb,  // pconv_ref: (1, 1, C, H)
    struct ggml_tensor* sw, struct ggml_tensor* sb,    // smooth conv: (3, 5, H, 1)
    int dmax
) {
    int64_t F = x_mic->ne[0], T = x_mic->ne[1], C = x_mic->ne[2];
    int64_t H = pmw->ne[3];  // align_hidden

    // 1x1 conv projections → Q, K: (F, T, H)
    struct ggml_tensor* mic4 = ggml_reshape_4d(ctx, x_mic, F, T, C, 1);
    struct ggml_tensor* ref4 = ggml_reshape_4d(ctx, x_ref, F, T, C, 1);
    struct ggml_tensor* Q4 = ggml_add(ctx, ggml_conv_2d(ctx, pmw, mic4, 1,1,0,0,1,1),
                                       ggml_reshape_4d(ctx, pmb, 1,1,H,1));
    struct ggml_tensor* K4 = ggml_add(ctx, ggml_conv_2d(ctx, prw, ref4, 1,1,0,0,1,1),
                                       ggml_reshape_4d(ctx, prb, 1,1,H,1));
    struct ggml_tensor* Q = ggml_reshape_3d(ctx, Q4, F, T, H);
    struct ggml_tensor* K = ggml_reshape_3d(ctx, K4, F, T, H);

    // Pad K in time: prepend dmax-1 zeros → (F, T+dmax-1, H)
    struct ggml_tensor* Kp = ggml_pad_ext(ctx, K, 0,0, dmax-1,0, 0,0, 0,0);

    // Similarity: V[d,t,h] = sum_f Q[f,t,h] * Kp[f,t+d,h] / sqrt(F)
    // Build by iterating over delays and concatenating
    float scale = 1.0f / std::sqrt((float)F);
    struct ggml_tensor* V = nullptr;  // will be (1, T, H) then grow to (dmax, T, H)

    for (int d = 0; d < dmax; d++) {
        // View of Kp at time offset d: (F, T, H) — make contiguous for mul
        struct ggml_tensor* Kd = ggml_cont(ctx,
            ggml_view_3d(ctx, Kp, F, T, H,
                          Kp->nb[1], Kp->nb[2],
                          d * Kp->nb[1]));
        // Element-wise multiply Q * Kd → (F, T, H), then sum over F → (1, T, H)
        struct ggml_tensor* qk = ggml_mul(ctx, Q, Kd);
        struct ggml_tensor* sim = ggml_sum_rows(ctx, qk);  // sum over ne0 → (1, T, H)
        sim = ggml_scale(ctx, sim, scale);

        if (V == nullptr) {
            V = sim;  // (1, T, H)
        } else {
            V = ggml_concat(ctx, V, sim, 0);  // concat along dim0 → (d+1, T, H)
        }
    }
    // V: (dmax, T, H)

    // Smooth conv: pad(1,1,4,0) on (dmax, T) dims, then Conv2d(H,1,(5,3))
    // Permute V to match conv input: (dmax, T, H) = (ne0=dmax, ne1=T, ne2=H)
    // Pad: freq(dmax) +1/+1, time(T) +4/+0 → (dmax+2, T+4, H, 1)
    struct ggml_tensor* Vp = ggml_reshape_4d(ctx, V, dmax, T, H, 1);
    Vp = ggml_pad_ext(ctx, Vp, 1,1, 4,0, 0,0, 0,0);
    // Conv2d with sw: kernel (kF=3, kT=5, C_in=H, C_out=1), stride (1,1)
    struct ggml_tensor* Vc = ggml_conv_2d(ctx, sw, Vp, 1,1, 0,0, 1,1);
    // Add bias
    struct ggml_tensor* s_bias = ggml_reshape_4d(ctx, sb, 1,1,1,1);
    Vc = ggml_add(ctx, Vc, s_bias);
    // Vc: (dmax, T, 1, 1) → reshape to (dmax, T)
    Vc = ggml_reshape_2d(ctx, Vc, dmax, T);

    // Softmax over delay (dim0=dmax)
    struct ggml_tensor* attn = ggml_soft_max(ctx, Vc);  // softmax over ne0
    // attn: (dmax, T)

    // Pad x_ref in time: prepend dmax-1 zeros → (F, T+dmax-1, C)
    struct ggml_tensor* rp = ggml_pad_ext(ctx, x_ref, 0,0, dmax-1,0, 0,0, 0,0);

    // Weighted sum: aligned[f,t,c] = sum_d attn[d,t] * rp[f,t+d,c]
    struct ggml_tensor* aligned = nullptr;
    for (int d = 0; d < dmax; d++) {
        // View of rp at time offset d: (F, T, C) — make contiguous
        struct ggml_tensor* rd = ggml_cont(ctx,
            ggml_view_3d(ctx, rp, F, T, C,
                          rp->nb[1], rp->nb[2],
                          d * rp->nb[1]));
        // attn[d, t]: element at ne0 offset d. View (1, T) slice, make contiguous
        struct ggml_tensor* ad = ggml_cont(ctx,
            ggml_view_2d(ctx, attn, 1, T,
                          attn->nb[1],
                          d * sizeof(float)));
        // Reshape ad to (1, T, 1) for broadcasting over F and C
        struct ggml_tensor* ad3 = ggml_reshape_3d(ctx, ad, 1, T, 1);
        // Multiply: (F, T, C) * (1, T, 1) → (F, T, C)
        struct ggml_tensor* contrib = ggml_mul(ctx, rd, ad3);

        if (aligned == nullptr) {
            aligned = contrib;
        } else {
            aligned = ggml_add(ctx, aligned, contrib);
        }
    }
    // aligned: (F, T, C)
    return aligned;
}

// Bottleneck: flatten (C,T,F) → GRU(input_size, hidden_size) → Linear → reshape
// This uses the graph-based approach with per-timestep GRU unrolling.
// h_init_out: if non-null, receives the GRU initial hidden state tensor
// (needs to be zeroed by the caller after graph allocation)
static struct ggml_tensor* build_bottleneck(
    struct ggml_context* ctx,
    struct ggml_tensor* x,      // (F, T, C)
    struct ggml_tensor* wih,    // GRU weight_ih: GGML (input_size, 3*H)
    struct ggml_tensor* whh,    // GRU weight_hh: GGML (H, 3*H)
    struct ggml_tensor* bih,    // GRU bias_ih: (3*H)
    struct ggml_tensor* bhh,    // GRU bias_hh: (3*H)
    struct ggml_tensor* fc_w,   // Linear weight: GGML (H, out_features)
    struct ggml_tensor* fc_b,   // Linear bias: (out_features)
    struct ggml_tensor** h_init_out = nullptr
) {
    int64_t F = x->ne[0], T = x->ne[1], C = x->ne[2];
    int64_t input_size = C * F;
    int64_t hidden_size = whh->ne[0];  // H
    int64_t out_features = fc_w->ne[1]; // C*F

    // Flatten: (F, T, C) → permute to (T, C*F) for GRU input
    // x is (ne0=F, ne1=T, ne2=C), need (ne0=C*F, ne1=T)
    // Permute to (F, C, T) then reshape to (C*F, T)
    // Actually: Buf layout is data[c*T*F + t*F + f]. We need flat[t*C*F + c*F + f].
    // In GGML: (ne0=F, ne1=T, ne2=C). Element at (f,t,c) is at c*T*F + t*F + f.
    // We need (ne0=C*F, ne1=T) where element at (cf, t) = x(f, t, c) with cf = c*F + f.
    // Permute (2,0,1,3) gives (C,F,T) → cont → reshape (C*F, T)
    struct ggml_tensor* flat = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));
    flat = ggml_reshape_2d(ctx, flat, C * F, T);

    // GRU: unroll over T timesteps
    // hidden state: need to create as a zero tensor
    // We'll build this iteratively, creating a new hidden state at each timestep
    // For the GRU equations:
    //   gates_ih = wih @ x_t + bih   (3*H)
    //   gates_hh = whh @ h_{t-1} + bhh   (3*H)
    //   r = sigmoid(gates_ih[0:H] + gates_hh[0:H])
    //   z = sigmoid(gates_ih[H:2H] + gates_hh[H:2H])
    //   n = tanh(gates_ih[2H:3H] + r * gates_hh[2H:3H])
    //   h_t = (1-z) * n + z * h_{t-1}

    // GRU weights in GGUF: wih is (ne0=input_size, ne1=3*H), whh is (ne0=H, ne1=3*H)
    // ggml_mul_mat(A, B): result[i,j] = sum_k A[k,i] * B[k,j]
    // For gates_ih = wih @ x_t: A=wih (input_size, 3*H), B=x_t (input_size, 1)
    // result (3*H, 1) = what we want

    // Start with zero hidden state (caller must zero after gallocr)
    struct ggml_tensor* h = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
    ggml_set_input(h);
    if (h_init_out) *h_init_out = h;

    // Collect GRU outputs for each timestep
    struct ggml_tensor* gru_out = nullptr;

    for (int64_t t = 0; t < T; t++) {
        // x_t: view of flat at timestep t → (C*F, 1)
        struct ggml_tensor* x_t = ggml_cont(ctx,
            ggml_view_2d(ctx, flat,
                          input_size, 1,
                          flat->nb[1],
                          t * flat->nb[1]));

        // gates_ih = wih @ x_t + bih → (3*H, 1)
        struct ggml_tensor* gih = ggml_add(ctx,
            ggml_mul_mat(ctx, wih, x_t),
            ggml_reshape_2d(ctx, bih, 3 * hidden_size, 1));

        // gates_hh = whh @ h + bhh → (3*H, 1)
        struct ggml_tensor* h2d = ggml_reshape_2d(ctx, h, hidden_size, 1);
        struct ggml_tensor* ghh = ggml_add(ctx,
            ggml_mul_mat(ctx, whh, h2d),
            ggml_reshape_2d(ctx, bhh, 3 * hidden_size, 1));

        // Split into r, z, n gates
        // r = sigmoid(gih[0:H] + ghh[0:H])
        struct ggml_tensor* gih_r = ggml_view_1d(ctx, gih, hidden_size, 0);
        struct ggml_tensor* ghh_r = ggml_view_1d(ctx, ghh, hidden_size, 0);
        struct ggml_tensor* r = ggml_sigmoid(ctx, ggml_add(ctx, gih_r, ghh_r));

        // z = sigmoid(gih[H:2H] + ghh[H:2H])
        struct ggml_tensor* gih_z = ggml_view_1d(ctx, gih, hidden_size, hidden_size * sizeof(float));
        struct ggml_tensor* ghh_z = ggml_view_1d(ctx, ghh, hidden_size, hidden_size * sizeof(float));
        struct ggml_tensor* z = ggml_sigmoid(ctx, ggml_add(ctx, gih_z, ghh_z));

        // n = tanh(gih[2H:3H] + r * ghh[2H:3H])
        struct ggml_tensor* gih_n = ggml_view_1d(ctx, gih, hidden_size, 2 * hidden_size * sizeof(float));
        struct ggml_tensor* ghh_n = ggml_view_1d(ctx, ghh, hidden_size, 2 * hidden_size * sizeof(float));
        struct ggml_tensor* n = ggml_tanh(ctx, ggml_add(ctx, gih_n, ggml_mul(ctx, r, ghh_n)));

        // h_t = (1 - z) * n + z * h
        // = n - z*n + z*h = n + z*(h - n)
        struct ggml_tensor* h_new = ggml_add(ctx, n,
            ggml_mul(ctx, z, ggml_sub(ctx, h, n)));

        h = h_new;

        // Collect output: reshape to (H, 1) for concatenation
        struct ggml_tensor* h_out = ggml_reshape_2d(ctx, h, hidden_size, 1);
        if (gru_out == nullptr) {
            gru_out = h_out;
        } else {
            gru_out = ggml_concat(ctx, gru_out, h_out, 1);  // concat along dim1 → (H, t+1)
        }
    }
    // gru_out: (H, T)

    // Linear: fc_out = fc_w @ gru_out + fc_b → (out_features, T)
    struct ggml_tensor* fc_out = ggml_add(ctx,
        ggml_mul_mat(ctx, fc_w, gru_out),
        ggml_reshape_2d(ctx, fc_b, out_features, 1));

    // Reshape back: (C*F, T) → (C, F, T) → permute to (F, T, C)
    struct ggml_tensor* reshaped = ggml_reshape_3d(ctx, fc_out, F, C, T);
    struct ggml_tensor* out = ggml_cont(ctx, ggml_permute(ctx, reshaped, 0, 2, 1, 3));
    // out: (F, T, C)

    return out;
}

// Concat along channel dim: (F, T, C1) + (F, T, C2) → (F, T, C1+C2)
static struct ggml_tensor* build_concat_channels(
    struct ggml_context* ctx,
    struct ggml_tensor* a,
    struct ggml_tensor* b
) {
    return ggml_concat(ctx, a, b, 2);  // concat along ne2 (channel)
}

// DFT basis vectors for 3-tap CCM
static const float CCM_VR[3] = {1.0f, -0.5f, -0.5f};
static const float CCM_VI[3] = {0.0f, 0.86602540378f, -0.86602540378f};

// Complex Convolving Mask: mask (27ch) + mic_stft → enhanced STFT
// mask: (F, T, 27), mic_stft input: (2, T, F) — original STFT
// Returns: (F, T, 2) enhanced STFT
static struct ggml_tensor* build_ccm(
    struct ggml_context* ctx,
    struct ggml_tensor* mask,     // (F, T, 27)
    struct ggml_tensor* stft_in,  // (2, T, F) — original mic STFT
    struct ggml_tensor* vr_tensor, struct ggml_tensor* vi_tensor  // DFT basis (3)
) {
    int64_t F = mask->ne[0], T = mask->ne[1];

    // Build H_real and H_imag: (F, T, 9) from mask channels
    // mask has 27 channels = 3 (DFT) × 9 (spatial)
    // H_real[c] = sum_r VR[r] * mask[r*9 + c], H_imag[c] = sum_r VI[r] * mask[r*9 + c]

    struct ggml_tensor* Hr = nullptr;  // (F, T, 9)
    struct ggml_tensor* Hi = nullptr;

    for (int r = 0; r < 3; r++) {
        // Extract 9 channels for this DFT component: mask[:,:,r*9:(r+1)*9]
        struct ggml_tensor* m_r = ggml_cont(ctx,
            ggml_view_3d(ctx, mask, F, T, 9,
                          mask->nb[1], mask->nb[2],
                          r * 9 * mask->nb[2]));
        // Scale by VR[r] and VI[r]
        struct ggml_tensor* scaled_r = ggml_scale(ctx, m_r, CCM_VR[r]);
        struct ggml_tensor* scaled_i = ggml_scale(ctx, m_r, CCM_VI[r]);

        Hr = Hr ? ggml_add(ctx, Hr, scaled_r) : scaled_r;
        Hi = Hi ? ggml_add(ctx, Hi, scaled_i) : scaled_i;
    }
    // Hr, Hi: (F, T, 9) — 3x3 spatial filter, real and imaginary parts

    // Permute input STFT: (2, T, F) → (F, T, 2)
    struct ggml_tensor* stft = ggml_cont(ctx, ggml_permute(ctx, stft_in, 2, 1, 0, 3));

    // Pad input: (F, T, 2) → pad freq +1/+1, time +2/+0 → (F+2, T+2, 2)
    struct ggml_tensor* xp = ggml_pad_ext(ctx, stft, 1,1, 2,0, 0,0, 0,0);

    // Complex convolution: for each 3x3 position (m,n):
    //   e_r += Hr[mn] * xp_r[t+m, f+n] - Hi[mn] * xp_i[t+m, f+n]
    //   e_i += Hr[mn] * xp_i[t+m, f+n] + Hi[mn] * xp_r[t+m, f+n]
    struct ggml_tensor* er = nullptr;
    struct ggml_tensor* ei = nullptr;

    for (int m = 0; m < 3; m++) {
        for (int n = 0; n < 3; n++) {
            int ki = m * 3 + n;

            // View of Hr/Hi for this kernel position: (F, T, 1) = channel ki
            struct ggml_tensor* hr = ggml_cont(ctx,
                ggml_view_3d(ctx, Hr, F, T, 1,
                              Hr->nb[1], Hr->nb[2],
                              ki * Hr->nb[2]));
            struct ggml_tensor* hi = ggml_cont(ctx,
                ggml_view_3d(ctx, Hi, F, T, 1,
                              Hi->nb[1], Hi->nb[2],
                              ki * Hi->nb[2]));

            // View of padded input at offset (n, m): re and im channels
            // xp is (F+2, T+2, 2). We want (F, T) starting at freq=n, time=m
            struct ggml_tensor* xr = ggml_cont(ctx,
                ggml_view_3d(ctx, xp, F, T, 1,
                              xp->nb[1], xp->nb[2],
                              m * xp->nb[1] + n * xp->nb[0]));
            struct ggml_tensor* xi = ggml_cont(ctx,
                ggml_view_3d(ctx, xp, F, T, 1,
                              xp->nb[1], xp->nb[2],
                              xp->nb[2] + m * xp->nb[1] + n * xp->nb[0]));

            // Complex multiply: er += hr*xr - hi*xi, ei += hr*xi + hi*xr
            struct ggml_tensor* cr = ggml_sub(ctx, ggml_mul(ctx, hr, xr),
                                               ggml_mul(ctx, hi, xi));
            struct ggml_tensor* ci = ggml_add(ctx, ggml_mul(ctx, hr, xi),
                                               ggml_mul(ctx, hi, xr));

            er = er ? ggml_add(ctx, er, cr) : cr;
            ei = ei ? ggml_add(ctx, ei, ci) : ci;
        }
    }
    // er, ei: (F, T, 1)

    // Concat re and im → (F, T, 2)
    struct ggml_tensor* enhanced = ggml_concat(ctx, er, ei, 2);
    return enhanced;
}

// ── Model loading ─────────────────────────────────────────────────────────

static uint32_t gguf_u32(struct gguf_context* ctx, const char* key) {
    int idx = gguf_find_key(ctx, key);
    return idx >= 0 ? gguf_get_val_u32(ctx, idx) : 0;
}

bool load_graph_model(const char* path, dvqe_graph_model& model,
                      bool verbose, int n_threads) {
    // Load GGUF metadata only — we'll allocate tensors on the backend buffer
    struct gguf_init_params params;
    params.no_alloc = true;
    params.ctx = &model.weight_ctx;

    struct gguf_context* gctx = gguf_init_from_file(path, params);
    if (!gctx) { fprintf(stderr, "Failed to load: %s\n", path); return false; }

    // Read hyperparameters
    auto& hp = model.hparams;
    hp.n_fft        = (int)gguf_u32(gctx, "deepvqe.n_fft");
    hp.hop_length   = (int)gguf_u32(gctx, "deepvqe.hop_length");
    hp.n_freq_bins  = (int)gguf_u32(gctx, "deepvqe.n_freq_bins");
    hp.sample_rate  = (int)gguf_u32(gctx, "deepvqe.sample_rate");
    hp.dmax         = (int)gguf_u32(gctx, "deepvqe.dmax");
    hp.align_hidden = (int)gguf_u32(gctx, "deepvqe.align_hidden");
    int idx = gguf_find_key(gctx, "deepvqe.power_law_c");
    hp.power_law_c = idx >= 0 ? gguf_get_val_f32(gctx, idx) : 0.3f;

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

    // Index weight tensors by name
    int n_tensors = gguf_get_n_tensors(gctx);
    for (int i = 0; i < n_tensors; i++) {
        const char* name = gguf_get_tensor_name(gctx, i);
        struct ggml_tensor* t = ggml_get_tensor(model.weight_ctx, name);
        if (t) model.weights[name] = t;
    }

    if (verbose) {
        printf("Graph model: %zu tensors, n_fft=%d, dmax=%d\n",
               model.weights.size(), hp.n_fft, hp.dmax);
    }

    // Load dynamic backends (CPU variants with different ISA support).
    // Searches: GGML_BACKEND_DIR (compile-time), executable dir, current dir.
    static bool backends_loaded = false;
    if (!backends_loaded) {
        ggml_backend_load_all();
        backends_loaded = true;
    }

    // Initialize backend (try CUDA first, fall back to CPU)
#ifdef GGML_USE_CUDA
    if (ggml_backend_cuda_get_device_count() > 0) {
        model.backend = ggml_backend_cuda_init(0);
        if (verbose) printf("Using CUDA backend\n");
    }
#endif
    if (!model.backend) {
        model.backend = ggml_backend_init_by_name("CPU", nullptr);
        if (!model.backend) {
            fprintf(stderr, "Failed to init CPU backend (check .so files are next to executable)\n");
            return false;
        }
        if (n_threads <= 0) {
            n_threads = std::max(1, (int)std::thread::hardware_concurrency() - 1);
        }
        // Set thread count via dynamic proc address (works with GGML_BACKEND_DL)
        auto dev = ggml_backend_get_device(model.backend);
        if (dev) {
            auto reg = ggml_backend_dev_backend_reg(dev);
            auto fn = (ggml_backend_set_n_threads_t)
                ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
            if (fn) fn(model.backend, n_threads);
        }
        if (verbose) printf("Using CPU backend (%d threads)\n", n_threads);
    }

    // Allocate weight tensors on the backend buffer (GPU if CUDA, CPU otherwise)
    model.weight_buf = ggml_backend_alloc_ctx_tensors(model.weight_ctx, model.backend);
    if (!model.weight_buf) {
        fprintf(stderr, "Failed to allocate weight buffer\n");
        gguf_free(gctx);
        return false;
    }

    // Read tensor data from GGUF file into backend buffers
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open: %s\n", path);
        gguf_free(gctx);
        return false;
    }
    size_t data_offset = gguf_get_data_offset(gctx);
    for (int i = 0; i < n_tensors; i++) {
        const char* name = gguf_get_tensor_name(gctx, i);
        struct ggml_tensor* t = ggml_get_tensor(model.weight_ctx, name);
        if (!t) continue;
        size_t offset = gguf_get_tensor_offset(gctx, i);
        size_t nbytes = ggml_nbytes(t);
        std::vector<uint8_t> buf(nbytes);
        fseek(f, data_offset + offset, SEEK_SET);
        if (fread(buf.data(), 1, nbytes, f) != nbytes) {
            fprintf(stderr, "Short read for tensor: %s\n", name);
            fclose(f);
            gguf_free(gctx);
            return false;
        }
        ggml_backend_tensor_set(t, buf.data(), 0, nbytes);
    }
    fclose(f);

    gguf_free(gctx);
    return true;
}

void free_graph_model(dvqe_graph_model& model) {
    if (model.weight_buf) { ggml_backend_buffer_free(model.weight_buf); model.weight_buf = nullptr; }
    if (model.backend) { ggml_backend_free(model.backend); model.backend = nullptr; }
    if (model.weight_ctx) { ggml_free(model.weight_ctx); model.weight_ctx = nullptr; }
    model.weights.clear();
}

// ── Full forward pass ─────────────────────────────────────────────────────


NpyArray forward_graph(const NpyArray& mic_stft, const NpyArray& ref_stft,
                       dvqe_graph_model& m,
                       std::vector<block_timing>* timings,
                       bool verbose,
                       const std::string& dump_dir) {
    (void)dump_dir;
    auto& hp = m.hparams;
    int F = (int)mic_stft.dim(1);
    int T = (int)mic_stft.dim(2);

    if (verbose) printf("forward_graph: F=%d T=%d (streaming)\n", F, T);

    // Build streaming graph, process frame-by-frame (matches PyTorch exactly)
    dvqe_stream_graph sg;
    if (!build_stream_graph(m, sg)) {
        fprintf(stderr, "ERROR: build_stream_graph failed\n");
        return {};
    }

    // Output STFT: (F, T, 2) layout matching input
    NpyArray result;
    result.shape = {1, (int64_t)F, (int64_t)T, 2};
    result.data.resize(F * T * 2, 0.0f);

    std::vector<float> mic_frame(F * 2);
    std::vector<float> ref_frame(F * 2);
    std::vector<float> enh_frame(F * 2);

    int64_t t0 = ggml_time_us();

    for (int t = 0; t < T; t++) {
        extract_stft_frame(mic_stft.data.data(), F, T, t, mic_frame.data());
        extract_stft_frame(ref_stft.data.data(), F, T, t, ref_frame.data());

        process_frame_graph(sg, m, mic_frame.data(), ref_frame.data(), enh_frame.data());

        scatter_stft_frame(enh_frame.data(), result.data.data(), F, T, t);
    }

    int64_t t1 = ggml_time_us();

    if (timings) timings->push_back({"total_graph", (double)(t1 - t0)});
    if (verbose) {
        printf("  %d frames in %lld us (%.1f us/frame)\n",
               T, (long long)(t1 - t0), (double)(t1 - t0) / T);
    }

    free_stream_graph(sg);
    return result;
}

// ── Streaming graph (T=1 with history buffers) ──────────────────────────────

// Causal conv for streaming: concat history + current frame, then conv.
// Pushes history in/out tensors to sg.conv_hist_in/conv_hist_out.
static struct ggml_tensor* build_causal_conv_s(
    struct ggml_context* ctx,
    dvqe_stream_graph& sg,
    struct ggml_tensor* x,           // (F, 1, C_in) — current frame
    struct ggml_tensor* weight,      // (kF=3, kT=4, C_in, C_out)
    struct ggml_tensor* bias,        // (C_out)
    int sF
) {
    int64_t F = x->ne[0], C_in = x->ne[2];

    // History input: last 3 frames (F, 3, C_in)
    struct ggml_tensor* hist = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, F, 3, C_in);
    ggml_set_input(hist);
    sg.conv_hist_in.push_back(hist);

    // Concat: (F, 3, C_in) + (F, 1, C_in) → (F, 4, C_in)
    struct ggml_tensor* cat = ggml_concat(ctx, hist, x, 1);

    // New history = [hist[1], hist[2], x] — built from inputs, not from cat
    // (avoids relying on cat's buffer surviving through the conv path)
    struct ggml_tensor* hist_tail = ggml_cont(ctx,
        ggml_view_3d(ctx, hist, F, 2, C_in,
                      hist->nb[1], hist->nb[2],
                      1 * hist->nb[1]));
    struct ggml_tensor* new_hist = ggml_concat(ctx, hist_tail, x, 1);
    ggml_set_output(new_hist);
    sg.conv_hist_out.push_back(new_hist);

    // Reshape to 4D, pad freq, conv
    // TODO: p0=1 should eliminate pad_ext but produces wrong results in full model
    // despite passing isolated tests. See ARCHITECTURE.md or git log for details.
    struct ggml_tensor* x4d = ggml_reshape_4d(ctx, cat, F, 4, C_in, 1);
    struct ggml_tensor* padded = ggml_pad_ext(ctx, x4d,
                                               1, 1,   // freq: left=1, right=1
                                               0, 0,   // time: none
                                               0, 0, 0, 0);
    struct ggml_tensor* conv = conv_2d_q(ctx, weight, padded,
                                          sF, 1, 0, 0, 1, 1, 3, 4);
    struct ggml_tensor* b = ggml_reshape_4d(ctx, bias, 1, 1, bias->ne[0], 1);
    conv = ggml_add(ctx, conv, b);

    return conv;
}

// Streaming encoder block: main conv (stride 2) + residual conv (stride 1).
static struct ggml_tensor* build_encoder_block_s(
    struct ggml_context* ctx,
    dvqe_stream_graph& sg,
    struct ggml_tensor* x,
    struct ggml_tensor* conv_w, struct ggml_tensor* conv_b,
    struct ggml_tensor* res_w, struct ggml_tensor* res_b
) {
    struct ggml_tensor* y = build_causal_conv_s(ctx, sg, x, conv_w, conv_b, 2);
    y = ggml_elu(ctx, y);
    y = ggml_reshape_3d(ctx, y, y->ne[0], y->ne[1], y->ne[2]);

    struct ggml_tensor* res = build_causal_conv_s(ctx, sg, y, res_w, res_b, 1);
    res = ggml_reshape_3d(ctx, res, res->ne[0], res->ne[1], res->ne[2]);
    res = ggml_add(ctx, ggml_elu(ctx, res), y);

    return res;
}

// Streaming bottleneck: single GRU step + linear.
static struct ggml_tensor* build_bottleneck_s(
    struct ggml_context* ctx,
    dvqe_stream_graph& sg,
    struct ggml_tensor* x,       // (F, 1, C)
    struct ggml_tensor* wih,     // (input_size, 3*H)
    struct ggml_tensor* whh,     // (H, 3*H)
    struct ggml_tensor* bih,     // (3*H)
    struct ggml_tensor* bhh,     // (3*H)
    struct ggml_tensor* fc_w,    // (H, out_features)
    struct ggml_tensor* fc_b     // (out_features)
) {
    int64_t F = x->ne[0], C = x->ne[2];
    int64_t input_size = C * F;
    int64_t hidden_size = whh->ne[0];
    int64_t out_features = fc_w->ne[1];

    // Flatten: (F, 1, C) → (C*F, 1)
    struct ggml_tensor* flat = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));
    flat = ggml_reshape_2d(ctx, flat, C * F, 1);

    // Hidden state input
    sg.gru_h_in = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
    ggml_set_input(sg.gru_h_in);
    struct ggml_tensor* h = sg.gru_h_in;

    // Single GRU step
    struct ggml_tensor* gih = ggml_add(ctx,
        ggml_mul_mat(ctx, wih, flat),
        ggml_reshape_2d(ctx, bih, 3 * hidden_size, 1));

    struct ggml_tensor* h2d = ggml_reshape_2d(ctx, h, hidden_size, 1);
    struct ggml_tensor* ghh = ggml_add(ctx,
        ggml_mul_mat(ctx, whh, h2d),
        ggml_reshape_2d(ctx, bhh, 3 * hidden_size, 1));

    struct ggml_tensor* gih_r = ggml_view_1d(ctx, gih, hidden_size, 0);
    struct ggml_tensor* ghh_r = ggml_view_1d(ctx, ghh, hidden_size, 0);
    struct ggml_tensor* r = ggml_sigmoid(ctx, ggml_add(ctx, gih_r, ghh_r));

    struct ggml_tensor* gih_z = ggml_view_1d(ctx, gih, hidden_size, hidden_size * sizeof(float));
    struct ggml_tensor* ghh_z = ggml_view_1d(ctx, ghh, hidden_size, hidden_size * sizeof(float));
    struct ggml_tensor* z = ggml_sigmoid(ctx, ggml_add(ctx, gih_z, ghh_z));

    struct ggml_tensor* gih_n = ggml_view_1d(ctx, gih, hidden_size, 2 * hidden_size * sizeof(float));
    struct ggml_tensor* ghh_n = ggml_view_1d(ctx, ghh, hidden_size, 2 * hidden_size * sizeof(float));
    struct ggml_tensor* n = ggml_tanh(ctx, ggml_add(ctx, gih_n, ggml_mul(ctx, r, ghh_n)));

    struct ggml_tensor* h_new = ggml_add(ctx, n,
        ggml_mul(ctx, z, ggml_sub(ctx, h, n)));

    // Output hidden state
    sg.gru_h_out = ggml_cont(ctx, h_new);
    ggml_set_output(sg.gru_h_out);

    // Linear: (H, 1) → (out_features, 1) → reshape to (F, 1, C)
    struct ggml_tensor* h_2d = ggml_reshape_2d(ctx, h_new, hidden_size, 1);
    struct ggml_tensor* fc_out = ggml_add(ctx,
        ggml_mul_mat(ctx, fc_w, h_2d),
        ggml_reshape_2d(ctx, fc_b, out_features, 1));

    // Reshape: (C*F, 1) → (F, C, 1) → permute to (F, 1, C)
    struct ggml_tensor* reshaped = ggml_reshape_3d(ctx, fc_out, F, C, 1);
    struct ggml_tensor* out = ggml_cont(ctx, ggml_permute(ctx, reshaped, 0, 2, 1, 3));

    return out;
}

// Streaming align block: cross-attention with K/ref/smooth histories.
static struct ggml_tensor* build_align_s(
    struct ggml_context* ctx,
    dvqe_stream_graph& sg,
    struct ggml_tensor* x_mic,   // (F, 1, C)
    struct ggml_tensor* x_ref,   // (F, 1, C)
    struct ggml_tensor* pmw, struct ggml_tensor* pmb,   // pconv_mic: (1, 1, C, H)
    struct ggml_tensor* prw, struct ggml_tensor* prb,   // pconv_ref: (1, 1, C, H)
    struct ggml_tensor* sw, struct ggml_tensor* sb,     // smooth conv: (3, 5, H, 1)
    int dmax
) {
    int64_t F = x_mic->ne[0], C = x_mic->ne[2];
    int64_t H = pmw->ne[3];

    // 1x1 projections → Q, K: (F, 1, H)
    struct ggml_tensor* mic4 = ggml_reshape_4d(ctx, x_mic, F, 1, C, 1);
    struct ggml_tensor* ref4 = ggml_reshape_4d(ctx, x_ref, F, 1, C, 1);
    struct ggml_tensor* Q4 = ggml_add(ctx, ggml_conv_2d(ctx, pmw, mic4, 1,1,0,0,1,1),
                                       ggml_reshape_4d(ctx, pmb, 1,1,H,1));
    struct ggml_tensor* K4 = ggml_add(ctx, ggml_conv_2d(ctx, prw, ref4, 1,1,0,0,1,1),
                                       ggml_reshape_4d(ctx, prb, 1,1,H,1));
    struct ggml_tensor* Q = ggml_reshape_3d(ctx, Q4, F, 1, H);
    struct ggml_tensor* K_cur = ggml_reshape_3d(ctx, K4, F, 1, H);

    // K history: (F, dmax-1, H)
    sg.align_K_hist_in = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, F, dmax - 1, H);
    ggml_set_input(sg.align_K_hist_in);

    // K_full = concat(K_hist, K_cur) → (F, dmax, H)
    struct ggml_tensor* K_full = ggml_concat(ctx, sg.align_K_hist_in, K_cur, 1);

    // New K history = [K_hist[1:], K_cur] — built from inputs
    struct ggml_tensor* K_hist_tail = ggml_cont(ctx,
        ggml_view_3d(ctx, sg.align_K_hist_in, F, dmax - 2, H,
                      sg.align_K_hist_in->nb[1], sg.align_K_hist_in->nb[2],
                      1 * sg.align_K_hist_in->nb[1]));
    sg.align_K_hist_out = ggml_concat(ctx, K_hist_tail, K_cur, 1);
    ggml_set_output(sg.align_K_hist_out);

    // Similarity: for each delay d, sim = sum_f(Q * K_full[:, d:d+1, :]) / sqrt(F)
    float scale = 1.0f / std::sqrt((float)F);
    struct ggml_tensor* V = nullptr;

    for (int d = 0; d < dmax; d++) {
        struct ggml_tensor* Kd = ggml_cont(ctx,
            ggml_view_3d(ctx, K_full, F, 1, H,
                          K_full->nb[1], K_full->nb[2],
                          d * K_full->nb[1]));
        struct ggml_tensor* qk = ggml_mul(ctx, Q, Kd);
        struct ggml_tensor* sim = ggml_sum_rows(ctx, qk);  // (1, 1, H)
        sim = ggml_scale(ctx, sim, scale);

        V = V ? ggml_concat(ctx, V, sim, 0) : sim;
    }
    // V: (dmax, 1, H)

    // Smooth conv with history
    sg.align_smooth_hist_in = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, dmax, 4, H);
    ggml_set_input(sg.align_smooth_hist_in);

    // V_full = concat(smooth_hist, V) → (dmax, 5, H)
    struct ggml_tensor* V_full = ggml_concat(ctx, sg.align_smooth_hist_in, V, 1);

    // New smooth history = [smooth_hist[1:], V] — built from inputs
    struct ggml_tensor* smooth_tail = ggml_cont(ctx,
        ggml_view_3d(ctx, sg.align_smooth_hist_in, dmax, 3, H,
                      sg.align_smooth_hist_in->nb[1], sg.align_smooth_hist_in->nb[2],
                      1 * sg.align_smooth_hist_in->nb[1]));
    sg.align_smooth_hist_out = ggml_concat(ctx, smooth_tail, V, 1);
    ggml_set_output(sg.align_smooth_hist_out);

    // Pad freq(dmax) +1/+1, no time padding → (dmax+2, 5, H, 1)
    struct ggml_tensor* Vp = ggml_reshape_4d(ctx, V_full, dmax, 5, H, 1);
    Vp = ggml_pad_ext(ctx, Vp, 1,1, 0,0, 0,0, 0,0);

    // Conv2d: kernel (3, 5, H, 1) → (dmax, 1, 1, 1)
    struct ggml_tensor* Vc = ggml_conv_2d(ctx, sw, Vp, 1,1, 0,0, 1,1);
    struct ggml_tensor* s_bias = ggml_reshape_4d(ctx, sb, 1,1,1,1);
    Vc = ggml_add(ctx, Vc, s_bias);
    Vc = ggml_reshape_2d(ctx, Vc, dmax, 1);

    // Softmax over delay dim
    struct ggml_tensor* attn = ggml_soft_max(ctx, Vc);  // (dmax, 1)

    // Ref history: (F, dmax-1, C)
    sg.align_ref_hist_in = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, F, dmax - 1, C);
    ggml_set_input(sg.align_ref_hist_in);

    // ref_full = concat(ref_hist, x_ref) → (F, dmax, C)
    struct ggml_tensor* ref_full = ggml_concat(ctx, sg.align_ref_hist_in, x_ref, 1);

    // New ref history = [ref_hist[1:], x_ref] — built from inputs
    struct ggml_tensor* ref_tail = ggml_cont(ctx,
        ggml_view_3d(ctx, sg.align_ref_hist_in, F, dmax - 2, C,
                      sg.align_ref_hist_in->nb[1], sg.align_ref_hist_in->nb[2],
                      1 * sg.align_ref_hist_in->nb[1]));
    sg.align_ref_hist_out = ggml_concat(ctx, ref_tail, x_ref, 1);
    ggml_set_output(sg.align_ref_hist_out);

    // Weighted sum: aligned = sum_d attn[d] * ref_full[:, d:d+1, :]
    struct ggml_tensor* aligned = nullptr;
    for (int d = 0; d < dmax; d++) {
        struct ggml_tensor* rd = ggml_cont(ctx,
            ggml_view_3d(ctx, ref_full, F, 1, C,
                          ref_full->nb[1], ref_full->nb[2],
                          d * ref_full->nb[1]));
        struct ggml_tensor* ad = ggml_cont(ctx,
            ggml_view_2d(ctx, attn, 1, 1,
                          attn->nb[1],
                          d * sizeof(float)));
        struct ggml_tensor* ad3 = ggml_reshape_3d(ctx, ad, 1, 1, 1);
        struct ggml_tensor* contrib = ggml_mul(ctx, rd, ad3);

        aligned = aligned ? ggml_add(ctx, aligned, contrib) : contrib;
    }

    return aligned;  // (F, 1, C)
}

// Streaming decoder block: skip + residual + SubpixelConv + optional BN.
static struct ggml_tensor* build_decoder_block_s(
    struct ggml_context* ctx,
    dvqe_stream_graph& sg,
    struct ggml_tensor* x,        // (F, 1, C)
    struct ggml_tensor* x_en,     // (F, 1, C) encoder skip
    struct ggml_tensor* skip_w, struct ggml_tensor* skip_b,
    struct ggml_tensor* res_w, struct ggml_tensor* res_b,
    struct ggml_tensor* deconv_w, struct ggml_tensor* deconv_b,
    struct ggml_tensor* bn_scale, struct ggml_tensor* bn_bias,
    bool is_last
) {
    int64_t F = x->ne[0], C = x->ne[2];

    // Skip conv (1x1, no history)
    struct ggml_tensor* x_en_4d = ggml_reshape_4d(ctx, x_en, F, 1, C, 1);
    struct ggml_tensor* skip = conv_2d_q(ctx, skip_w, x_en_4d, 1, 1, 0, 0, 1, 1, 1, 1);
    struct ggml_tensor* sb = ggml_reshape_4d(ctx, skip_b, 1, 1, skip_b->ne[0], 1);
    skip = ggml_add(ctx, skip, sb);
    skip = ggml_reshape_3d(ctx, skip, F, 1, C);

    struct ggml_tensor* y = ggml_add(ctx, x, skip);

    // Residual (causal conv with history)
    struct ggml_tensor* res = build_causal_conv_s(ctx, sg, y, res_w, res_b, 1);
    res = ggml_reshape_3d(ctx, res, F, 1, C);
    res = ggml_add(ctx, ggml_elu(ctx, res), y);

    // SubpixelConv2d (causal conv with history) → pixel shuffle
    int64_t C_out = ((ggml_n_dims(deconv_w) == 4) ? deconv_w->ne[3] : deconv_w->ne[1]) / 2;
    struct ggml_tensor* deconv = build_causal_conv_s(ctx, sg, res, deconv_w, deconv_b, 1);
    deconv = ggml_reshape_3d(ctx, deconv, F, 1, C_out * 2);

    // Pixel shuffle: (F, 1, 2*C_out) → (2*F, 1, C_out)
    struct ggml_tensor* r1 = ggml_reshape_4d(ctx, deconv, F, 1, C_out, 2);
    struct ggml_tensor* r2 = ggml_permute(ctx, r1, 0, 2, 3, 1);
    struct ggml_tensor* shuffled = ggml_reshape_3d(ctx, ggml_cont(ctx, r2), 2 * F, 1, C_out);

    if (!is_last && bn_scale && bn_bias) {
        struct ggml_tensor* sc = ggml_reshape_3d(ctx, bn_scale, 1, 1, C_out);
        struct ggml_tensor* bi = ggml_reshape_3d(ctx, bn_bias, 1, 1, C_out);
        shuffled = ggml_elu(ctx, ggml_add(ctx, ggml_mul(ctx, shuffled, sc), bi));
    }

    return shuffled;
}

// Streaming CCM: complex convolving mask with STFT history.
static struct ggml_tensor* build_ccm_s(
    struct ggml_context* ctx,
    dvqe_stream_graph& sg,
    struct ggml_tensor* mask,       // (F, 1, 27)
    struct ggml_tensor* stft_in     // (2, 1, F) — original mic STFT
) {
    int64_t F = mask->ne[0];

    // Permute STFT: (2, 1, F) → (F, 1, 2)
    struct ggml_tensor* stft_cur = ggml_cont(ctx, ggml_permute(ctx, stft_in, 2, 1, 0, 3));

    // STFT history: (F, 2, 2)
    sg.ccm_hist_in = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, F, 2, 2);
    ggml_set_input(sg.ccm_hist_in);

    // Concat: (F, 2, 2) + (F, 1, 2) → (F, 3, 2)
    struct ggml_tensor* stft_full = ggml_concat(ctx, sg.ccm_hist_in, stft_cur, 1);

    // New history = [ccm_hist[1:], stft_cur] — built from inputs
    struct ggml_tensor* ccm_tail = ggml_cont(ctx,
        ggml_view_3d(ctx, sg.ccm_hist_in, F, 1, 2,
                      sg.ccm_hist_in->nb[1], sg.ccm_hist_in->nb[2],
                      1 * sg.ccm_hist_in->nb[1]));
    sg.ccm_hist_out = ggml_concat(ctx, ccm_tail, stft_cur, 1);
    ggml_set_output(sg.ccm_hist_out);

    // Pad freq: (F, 3, 2) → (F+2, 3, 2)
    struct ggml_tensor* xp = ggml_pad_ext(ctx, stft_full, 1,1, 0,0, 0,0, 0,0);

    // Build H_real, H_imag from mask (same as batch CCM)
    struct ggml_tensor* Hr = nullptr;
    struct ggml_tensor* Hi = nullptr;

    for (int r = 0; r < 3; r++) {
        struct ggml_tensor* m_r = ggml_cont(ctx,
            ggml_view_3d(ctx, mask, F, 1, 9,
                          mask->nb[1], mask->nb[2],
                          r * 9 * mask->nb[2]));
        struct ggml_tensor* scaled_r = ggml_scale(ctx, m_r, CCM_VR[r]);
        struct ggml_tensor* scaled_i = ggml_scale(ctx, m_r, CCM_VI[r]);

        Hr = Hr ? ggml_add(ctx, Hr, scaled_r) : scaled_r;
        Hi = Hi ? ggml_add(ctx, Hi, scaled_i) : scaled_i;
    }
    // Hr, Hi: (F, 1, 9)

    // Complex convolution over 3x3 kernel
    struct ggml_tensor* er = nullptr;
    struct ggml_tensor* ei = nullptr;

    for (int m = 0; m < 3; m++) {
        for (int n = 0; n < 3; n++) {
            int ki = m * 3 + n;

            struct ggml_tensor* hr = ggml_cont(ctx,
                ggml_view_3d(ctx, Hr, F, 1, 1,
                              Hr->nb[1], Hr->nb[2],
                              ki * Hr->nb[2]));
            struct ggml_tensor* hi = ggml_cont(ctx,
                ggml_view_3d(ctx, Hi, F, 1, 1,
                              Hi->nb[1], Hi->nb[2],
                              ki * Hi->nb[2]));

            // xp at time=m, freq offset=n: (F, 1, 1)
            struct ggml_tensor* xr = ggml_cont(ctx,
                ggml_view_3d(ctx, xp, F, 1, 1,
                              xp->nb[1], xp->nb[2],
                              m * xp->nb[1] + n * xp->nb[0]));
            struct ggml_tensor* xi = ggml_cont(ctx,
                ggml_view_3d(ctx, xp, F, 1, 1,
                              xp->nb[1], xp->nb[2],
                              xp->nb[2] + m * xp->nb[1] + n * xp->nb[0]));

            struct ggml_tensor* cr = ggml_sub(ctx, ggml_mul(ctx, hr, xr),
                                               ggml_mul(ctx, hi, xi));
            struct ggml_tensor* ci = ggml_add(ctx, ggml_mul(ctx, hr, xi),
                                               ggml_mul(ctx, hi, xr));

            er = er ? ggml_add(ctx, er, cr) : cr;
            ei = ei ? ggml_add(ctx, ei, ci) : ci;
        }
    }

    // (F, 1, 1) + (F, 1, 1) → (F, 1, 2)
    return ggml_concat(ctx, er, ei, 2);
}

// ── Build/process/reset/free ────────────────────────────────────────────────

bool build_stream_graph(dvqe_graph_model& m, dvqe_stream_graph& sg) {
    auto& hp = m.hparams;
    int F = hp.n_freq_bins;  // 257

    // Allocate context — much smaller than batch (no GRU unrolling)
    sg.ctx = make_ctx(64 * 1024 * 1024);
    auto* ctx = sg.ctx;

    // Inputs: (2, 1, F)
    sg.mic_in = input_3d(ctx, 2, 1, F);
    sg.ref_in = input_3d(ctx, 2, 1, F);

    // 1. Feature extraction (no time dependency)
    struct ggml_tensor* mic_fe = build_fe(ctx, sg.mic_in, hp.power_law_c);
    struct ggml_tensor* ref_fe = build_fe(ctx, sg.ref_in, hp.power_law_c);

    // 2. Mic encoder 1-2
    struct ggml_tensor* mic_e1 = build_encoder_block_s(ctx, sg, mic_fe,
        m.w("mic_enc1.conv.weight"), m.w("mic_enc1.conv.bias"),
        m.w("mic_enc1.resblock.conv.weight"), m.w("mic_enc1.resblock.conv.bias"));

    struct ggml_tensor* mic_e2 = build_encoder_block_s(ctx, sg, mic_e1,
        m.w("mic_enc2.conv.weight"), m.w("mic_enc2.conv.bias"),
        m.w("mic_enc2.resblock.conv.weight"), m.w("mic_enc2.resblock.conv.bias"));

    // 3. Far-end encoder 1-2
    struct ggml_tensor* far_e1 = build_encoder_block_s(ctx, sg, ref_fe,
        m.w("far_enc1.conv.weight"), m.w("far_enc1.conv.bias"),
        m.w("far_enc1.resblock.conv.weight"), m.w("far_enc1.resblock.conv.bias"));

    struct ggml_tensor* far_e2 = build_encoder_block_s(ctx, sg, far_e1,
        m.w("far_enc2.conv.weight"), m.w("far_enc2.conv.bias"),
        m.w("far_enc2.resblock.conv.weight"), m.w("far_enc2.resblock.conv.bias"));

    // 4. Alignment
    struct ggml_tensor* aligned = build_align_s(ctx, sg, mic_e2, far_e2,
        m.w("align.pconv_mic.weight"), m.w("align.pconv_mic.bias"),
        m.w("align.pconv_ref.weight"), m.w("align.pconv_ref.bias"),
        m.w("align.conv.1.weight"), m.w("align.conv.1.bias"),
        hp.dmax);

    // 5. Concat + encoder 3-5
    struct ggml_tensor* cat = build_concat_channels(ctx, mic_e2, aligned);
    struct ggml_tensor* mic_e3 = build_encoder_block_s(ctx, sg, cat,
        m.w("mic_enc3.conv.weight"), m.w("mic_enc3.conv.bias"),
        m.w("mic_enc3.resblock.conv.weight"), m.w("mic_enc3.resblock.conv.bias"));

    struct ggml_tensor* mic_e4 = build_encoder_block_s(ctx, sg, mic_e3,
        m.w("mic_enc4.conv.weight"), m.w("mic_enc4.conv.bias"),
        m.w("mic_enc4.resblock.conv.weight"), m.w("mic_enc4.resblock.conv.bias"));

    struct ggml_tensor* mic_e5 = build_encoder_block_s(ctx, sg, mic_e4,
        m.w("mic_enc5.conv.weight"), m.w("mic_enc5.conv.bias"),
        m.w("mic_enc5.resblock.conv.weight"), m.w("mic_enc5.resblock.conv.bias"));

    // 6. Bottleneck (single GRU step)
    struct ggml_tensor* bn = build_bottleneck_s(ctx, sg, mic_e5,
        m.w("bottleneck.gru.weight_ih_l0"), m.w("bottleneck.gru.weight_hh_l0"),
        m.w("bottleneck.gru.bias_ih_l0"), m.w("bottleneck.gru.bias_hh_l0"),
        m.w("bottleneck.fc.weight"), m.w("bottleneck.fc.bias"));

    // 7. Decoder with skip connections + frequency trimming
    struct ggml_tensor* d5 = build_decoder_block_s(ctx, sg, bn, mic_e5,
        m.w("dec5.skip_conv.weight"), m.w("dec5.skip_conv.bias"),
        m.w("dec5.resblock.conv.weight"), m.w("dec5.resblock.conv.bias"),
        m.w("dec5.deconv.conv.weight"), m.w("dec5.deconv.conv.bias"),
        m.w("dec5.bn.scale"), m.w("dec5.bn.bias"), false);
    d5 = build_freq_trim(ctx, d5, mic_e4->ne[0]);

    struct ggml_tensor* d4 = build_decoder_block_s(ctx, sg, d5, mic_e4,
        m.w("dec4.skip_conv.weight"), m.w("dec4.skip_conv.bias"),
        m.w("dec4.resblock.conv.weight"), m.w("dec4.resblock.conv.bias"),
        m.w("dec4.deconv.conv.weight"), m.w("dec4.deconv.conv.bias"),
        m.w("dec4.bn.scale"), m.w("dec4.bn.bias"), false);
    d4 = build_freq_trim(ctx, d4, mic_e3->ne[0]);

    struct ggml_tensor* d3 = build_decoder_block_s(ctx, sg, d4, mic_e3,
        m.w("dec3.skip_conv.weight"), m.w("dec3.skip_conv.bias"),
        m.w("dec3.resblock.conv.weight"), m.w("dec3.resblock.conv.bias"),
        m.w("dec3.deconv.conv.weight"), m.w("dec3.deconv.conv.bias"),
        m.w("dec3.bn.scale"), m.w("dec3.bn.bias"), false);
    d3 = build_freq_trim(ctx, d3, mic_e2->ne[0]);

    struct ggml_tensor* d2 = build_decoder_block_s(ctx, sg, d3, mic_e2,
        m.w("dec2.skip_conv.weight"), m.w("dec2.skip_conv.bias"),
        m.w("dec2.resblock.conv.weight"), m.w("dec2.resblock.conv.bias"),
        m.w("dec2.deconv.conv.weight"), m.w("dec2.deconv.conv.bias"),
        m.w("dec2.bn.scale"), m.w("dec2.bn.bias"), false);
    d2 = build_freq_trim(ctx, d2, mic_e1->ne[0]);

    struct ggml_tensor* d1 = build_decoder_block_s(ctx, sg, d2, mic_e1,
        m.w("dec1.skip_conv.weight"), m.w("dec1.skip_conv.bias"),
        m.w("dec1.resblock.conv.weight"), m.w("dec1.resblock.conv.bias"),
        m.w("dec1.deconv.conv.weight"), m.w("dec1.deconv.conv.bias"),
        nullptr, nullptr, true);
    d1 = build_freq_trim(ctx, d1, mic_fe->ne[0]);

    // 8. CCM
    struct ggml_tensor* ccm_out = build_ccm_s(ctx, sg, d1, sg.mic_in);
    // ccm_out: (F, 1, 2) → permute to (2, 1, F) to match I/O format
    sg.enhanced_out = ggml_cont(ctx,
        ggml_permute(ctx, ccm_out, 2, 1, 0, 3));
    ggml_set_output(sg.enhanced_out);

    // Build graph with all outputs
    sg.graph = ggml_new_graph_custom(ctx, 8192, false);
    ggml_build_forward_expand(sg.graph, sg.enhanced_out);
    // Add all history outputs to graph
    for (auto* h : sg.conv_hist_out)
        ggml_build_forward_expand(sg.graph, h);
    if (sg.gru_h_out)
        ggml_build_forward_expand(sg.graph, sg.gru_h_out);
    if (sg.align_K_hist_out)
        ggml_build_forward_expand(sg.graph, sg.align_K_hist_out);
    if (sg.align_ref_hist_out)
        ggml_build_forward_expand(sg.graph, sg.align_ref_hist_out);
    if (sg.align_smooth_hist_out)
        ggml_build_forward_expand(sg.graph, sg.align_smooth_hist_out);
    if (sg.ccm_hist_out)
        ggml_build_forward_expand(sg.graph, sg.ccm_hist_out);

    // Allocate
    sg.galloc = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(m.backend));
    if (!ggml_gallocr_alloc_graph(sg.galloc, sg.graph)) {
        fprintf(stderr, "ERROR: stream graph allocation failed\n");
        free_stream_graph(sg);
        return false;
    }

    // Size persistent scratch to the largest history tensor
    size_t max_hist_bytes = 0;
    for (auto* h : sg.conv_hist_out)
        max_hist_bytes = std::max(max_hist_bytes, ggml_nbytes(h));
    if (sg.gru_h_out) max_hist_bytes = std::max(max_hist_bytes, ggml_nbytes(sg.gru_h_out));
    if (sg.align_K_hist_out) max_hist_bytes = std::max(max_hist_bytes, ggml_nbytes(sg.align_K_hist_out));
    if (sg.align_ref_hist_out) max_hist_bytes = std::max(max_hist_bytes, ggml_nbytes(sg.align_ref_hist_out));
    if (sg.align_smooth_hist_out) max_hist_bytes = std::max(max_hist_bytes, ggml_nbytes(sg.align_smooth_hist_out));
    if (sg.ccm_hist_out) max_hist_bytes = std::max(max_hist_bytes, ggml_nbytes(sg.ccm_hist_out));
    sg.hist_scratch.resize(max_hist_bytes, 0);

    // Zero all history
    reset_stream_graph(sg, m);

    return true;
}

void process_frame_graph(dvqe_stream_graph& sg, dvqe_graph_model& m,
                         const float* mic_stft_frame,
                         const float* ref_stft_frame,
                         float* enhanced_stft_frame) {
    int F = m.hparams.n_freq_bins;

    // Set input STFT frames
    // Layout: mic_stft_frame is [f0_re, f0_im, f1_re, f1_im, ...] = F*2 floats
    // Tensor mic_in is (ne0=2, ne1=1, ne2=F), which in memory is: 2 fastest
    // So memory layout is: [f0_re, f0_im, f1_re, f1_im, ...] — matches!
    // Direct memcpy for CPU backend (avoids backend dispatch overhead)
    memcpy(sg.mic_in->data, mic_stft_frame, F * 2 * sizeof(float));
    memcpy(sg.ref_in->data, ref_stft_frame, F * 2 * sizeof(float));

    // Compute
    ggml_backend_graph_compute(m.backend, sg.graph);

    // Read enhanced output (direct pointer for CPU)
    memcpy(enhanced_stft_frame, sg.enhanced_out->data, F * 2 * sizeof(float));

    // Update histories: copy each output back to corresponding input.
    // For CPU backend, direct memcpy (avoids double-copy through scratch).
    auto copy_hist = [](struct ggml_tensor* in, struct ggml_tensor* out) {
        if (!in || !out) return;
        memcpy(in->data, out->data, ggml_nbytes(in));
    };

    for (size_t i = 0; i < sg.conv_hist_in.size(); i++)
        copy_hist(sg.conv_hist_in[i], sg.conv_hist_out[i]);
    copy_hist(sg.gru_h_in, sg.gru_h_out);
    copy_hist(sg.align_K_hist_in, sg.align_K_hist_out);
    copy_hist(sg.align_ref_hist_in, sg.align_ref_hist_out);
    copy_hist(sg.align_smooth_hist_in, sg.align_smooth_hist_out);
    copy_hist(sg.ccm_hist_in, sg.ccm_hist_out);
}

void reset_stream_graph(dvqe_stream_graph& sg, dvqe_graph_model& m) {
    // Direct memset for CPU backend
    auto zero_tensor = [](struct ggml_tensor* t) {
        if (!t) return;
        memset(t->data, 0, ggml_nbytes(t));
    };

    for (auto* h : sg.conv_hist_in)
        zero_tensor(h);
    zero_tensor(sg.gru_h_in);
    zero_tensor(sg.align_K_hist_in);
    zero_tensor(sg.align_ref_hist_in);
    zero_tensor(sg.align_smooth_hist_in);
    zero_tensor(sg.ccm_hist_in);
}

void free_stream_graph(dvqe_stream_graph& sg) {
    if (sg.galloc) { ggml_gallocr_free(sg.galloc); sg.galloc = nullptr; }
    if (sg.ctx) { ggml_free(sg.ctx); sg.ctx = nullptr; }
    sg.conv_hist_in.clear();
    sg.conv_hist_out.clear();
    sg.graph = nullptr;
    sg.mic_in = sg.ref_in = sg.enhanced_out = nullptr;
    sg.gru_h_in = sg.gru_h_out = nullptr;
    sg.align_K_hist_in = sg.align_K_hist_out = nullptr;
    sg.align_ref_hist_in = sg.align_ref_hist_out = nullptr;
    sg.align_smooth_hist_in = sg.align_smooth_hist_out = nullptr;
    sg.ccm_hist_in = sg.ccm_hist_out = nullptr;
}
