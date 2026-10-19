"""Custom Metal kernels.

gated_delta: recurrent Gated DeltaNet update with the gates (g = exp(-exp(A_log) *
softplus(a + dt_bias)), beta = sigmoid(b)) computed inside the kernel, so a
linear-attention layer needs one dispatch for the whole recurrence instead of
several small elementwise ops around it.
"""

import mlx.core as mx

_GATED_DELTA_SRC = """
    const uint n = thread_position_in_grid.z;      // b * Hv + hv
    const uint b_idx = n / Hv;
    const uint hv = n % Hv;
    const uint hk = hv / (Hv / Hk);
    const uint lane = thread_position_in_threadgroup.x;
    const uint dv = thread_position_in_grid.y;
    constexpr int NPT = Dk / 32;                    // state elements per thread

    auto q_ = q + (b_idx * T * Hk + hk) * Dk + lane * NPT;
    auto k_ = k + (b_idx * T * Hk + hk) * Dk + lane * NPT;
    auto v_ = v + (b_idx * T * Hv + hv) * Dv + dv;
    auto y_ = y + (b_idx * T * Hv + hv) * Dv + dv;
    auto a_ = a + b_idx * T * Hv + hv;
    auto b_ = b + b_idx * T * Hv + hv;

    const float A = metal::exp(static_cast<float>(A_log[hv]));
    const float dt = static_cast<float>(dt_bias[hv]);

    float st[NPT];
    auto s_in = state_in + (n * Dv + dv) * Dk + lane * NPT;
    for (int i = 0; i < NPT; ++i) st[i] = static_cast<float>(s_in[i]);

    for (int t = 0; t < T; ++t) {
        SAVE_MID
        const float x = static_cast<float>(*a_) + dt;
        const float sp = x > 20.0f ? x : metal::log(1.0f + metal::exp(x));
        const float g = metal::exp(-A * sp);
        const float beta = 1.0f / (1.0f + metal::exp(-static_cast<float>(*b_)));

        float kk[NPT], qq[NPT];
        float kv = 0.0f;
        for (int i = 0; i < NPT; ++i) {
            kk[i] = static_cast<float>(k_[i]);
            qq[i] = static_cast<float>(q_[i]);
            st[i] *= g;
            kv += st[i] * kk[i];
        }
        kv = simd_sum(kv);
        const float delta = (static_cast<float>(*v_) - kv) * beta;

        float o = 0.0f;
        for (int i = 0; i < NPT; ++i) {
            st[i] += kk[i] * delta;
            o += st[i] * qq[i];
        }
        o = simd_sum(o);
        if (lane == 0) *y_ = static_cast<InT>(o);

        q_ += Hk * Dk; k_ += Hk * Dk;
        v_ += Hv * Dv; y_ += Hv * Dv;
        a_ += Hv; b_ += Hv;
    }

    auto s_out = state_out + (n * Dv + dv) * Dk + lane * NPT;
    for (int i = 0; i < NPT; ++i) s_out[i] = static_cast<StT>(st[i]);
"""

# state before the last step, so a rejected speculative token can be rolled back
_SAVE_MID_SRC = """
        if (t == T - 1) {
            auto s_mid = state_mid + (n * Dv + dv) * Dk + lane * NPT;
            for (int i = 0; i < NPT; ++i) s_mid[i] = static_cast<StT>(st[i]);
        }
"""

_INPUTS = ["q", "k", "v", "a", "b", "A_log", "dt_bias", "state_in", "T"]
_gated_delta_kernel = mx.fast.metal_kernel(
    name="engine_gated_delta",
    input_names=_INPUTS,
    output_names=["y", "state_out"],
    source=_GATED_DELTA_SRC.replace("SAVE_MID", ""),
)
_gated_delta_kernel_mid = mx.fast.metal_kernel(
    name="engine_gated_delta_mid",
    input_names=_INPUTS,
    output_names=["y", "state_out", "state_mid"],
    source=_GATED_DELTA_SRC.replace("SAVE_MID", _SAVE_MID_SRC),
)


def gated_delta(q, k, v, a, b, A_log, dt_bias, state, keep_mid=False):
    """
    q, k:  [B, T, Hk, Dk]   v: [B, T, Hv, Dv]   a, b: [B, T, Hv]
    state: [B, Hv, Dv, Dk] float32
    returns y [B, T, Hv, Dv], new state (+ state before the last step if keep_mid)
    """
    B, T, Hk, Dk = k.shape
    Hv, Dv = v.shape[2:]
    kernel = _gated_delta_kernel_mid if keep_mid else _gated_delta_kernel
    extra = [state.shape] if keep_mid else []
    return kernel(
        inputs=[q, k, v, a, b, A_log, dt_bias, state, T],
        template=[
            ("InT", q.dtype),
            ("StT", state.dtype),
            ("Dk", Dk),
            ("Dv", Dv),
            ("Hk", Hk),
            ("Hv", Hv),
        ],
        grid=(32, Dv, B * Hv),
        threadgroup=(32, 4, 1),
        output_shapes=[(B, T, Hv, Dv), state.shape, *extra],
        output_dtypes=[q.dtype, state.dtype, *[state.dtype] * len(extra)],
    )


# ------------------------------------------------------------------ multi-row quantized matvec
#
# y[m, n] = sum_k x[m, k] * (scale[n, g] * q[n, k] + bias[n, g])     (4-bit, group 64)
#
# MLX's kernel handles extra rows by redoing the dequantization per row, so energy
# grows ~linearly with M even though memory traffic barely changes. Here each
# weight is unpacked once and used for all M rows; the bias term is factored out
# as bias * sum(x) per group.
#
# Each simdgroup owns ROWS output rows; each lane handles 16 consecutive k per
# iteration (two packed uint32), so a simdgroup sweeps 512 k per step with
# 128-byte coalesced loads per row.

_QMV_MULTI_SRC = """
    constexpr int ROWS = 4;
    constexpr int VPT = 16;                 // values per lane per step
    constexpr int STEP = 32 * VPT;          // k per simdgroup step
    constexpr int GROUPS = K / 64;

    const uint lane = thread_position_in_threadgroup.x;
    const uint row0 = thread_position_in_grid.y * ROWS;
    if (row0 >= N) return;

    float acc[ROWS][M];
    for (int r = 0; r < ROWS; ++r)
        for (int m = 0; m < M; ++m) acc[r][m] = 0.0f;

    for (int k0 = lane * VPT; k0 < K; k0 += STEP) {
        float xv[M][VPT];
        float xs[M];
        for (int m = 0; m < M; ++m) {
            const device InT* xp = x + m * K + k0;
            float s = 0.0f;
            for (int i = 0; i < VPT; ++i) { xv[m][i] = static_cast<float>(xp[i]); s += xv[m][i]; }
            xs[m] = s;
        }
        const int g = k0 / 64;
        for (int r = 0; r < ROWS; ++r) {
            const uint row = row0 + r;
            if (row >= N) break;
            const device uint32_t* wp = w + row * (K / 8) + k0 / 8;
            const uint32_t w0 = wp[0], w1 = wp[1];
            const float sc = static_cast<float>(scales[row * GROUPS + g]);
            const float bi = static_cast<float>(biases[row * GROUPS + g]);
            float q[VPT];
            for (int i = 0; i < 8; ++i) {
                q[i] = static_cast<float>((w0 >> (4 * i)) & 0xF);
                q[i + 8] = static_cast<float>((w1 >> (4 * i)) & 0xF);
            }
            for (int m = 0; m < M; ++m) {
                float d = 0.0f;
                for (int i = 0; i < VPT; ++i) d += q[i] * xv[m][i];
                acc[r][m] += sc * d + bi * xs[m];
            }
        }
    }

    for (int r = 0; r < ROWS; ++r) {
        const uint row = row0 + r;
        if (row >= N) break;
        for (int m = 0; m < M; ++m) {
            const float v = simd_sum(acc[r][m]);
            if (lane == 0) y[m * N + row] = static_cast<InT>(v);
        }
    }
"""

_qmv_multi_kernel = mx.fast.metal_kernel(
    name="engine_qmv_multi",
    input_names=["x", "w", "scales", "biases"],
    output_names=["y"],
    source=_QMV_MULTI_SRC,
)

QMV_ROWS_PER_SIMDGROUP = 4
QMV_SIMDGROUPS = 2


def qmv_multi(x, w, scales, biases):
    """x [M, K] (M small), w [N, K/8] uint32 4-bit/group-64 -> y [M, N]"""
    M, K = x.shape
    N = w.shape[0]
    assert K % 512 == 0, "K must be a multiple of 512"
    n_sg = (N + QMV_ROWS_PER_SIMDGROUP - 1) // QMV_ROWS_PER_SIMDGROUP
    n_sg = (n_sg + QMV_SIMDGROUPS - 1) // QMV_SIMDGROUPS * QMV_SIMDGROUPS
    return _qmv_multi_kernel(
        inputs=[x, w, scales, biases],
        template=[("InT", x.dtype), ("M", M), ("K", K), ("N", N)],
        grid=(32, n_sg, 1),
        threadgroup=(32, QMV_SIMDGROUPS, 1),
        output_shapes=[(M, N)],
        output_dtypes=[x.dtype],
    )[0]
