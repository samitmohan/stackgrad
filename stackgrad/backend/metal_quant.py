import mlx.core as mx

SIMDGROUPS = 8  # output columns per threadgroup

# one simdgroup per output column; s * sum(x q) + b * sum(x) applies scale and bias once per word
_SOURCE = """
    constexpr int PER_WORD = 32 / BITS, MASK = (1 << BITS) - 1;
    const int M = x_shape[0], K = x_shape[1], N = w_shape[0];
    const int words = K / PER_WORD, groups = K / GROUP;
    const int n = threadgroup_position_in_grid.x * SG + simdgroup_index_in_threadgroup;
    const int lane = thread_index_in_simdgroup;
    if (n >= N) return;                                    // the whole simdgroup leaves together
    for (int m = 0; m < M; m++) {
        float acc = 0.0f;
        for (int wi = lane; wi < words; wi += 32) {
            const uint word = w[n * words + wi];
            const int k0 = wi * PER_WORD, g = k0 / GROUP;
            float xq = 0.0f, xs = 0.0f;
            for (int j = 0; j < PER_WORD; j++) {
                const float xv = float(x[m * K + k0 + j]);
                xq += xv * float((word >> (j * BITS)) & MASK);
                xs += xv;
            }
            acc += float(scales[n * groups + g]) * xq + float(biases[n * groups + g]) * xs;
        }
        acc = simd_sum(acc);
        if (lane == 0) out[m * N + n] = InT(acc);
    }
"""

_KERNEL = mx.fast.metal_kernel(name="stackgrad_quant_matmul", input_names=["x", "w", "scales", "biases"],
                               output_names=["out"], source=_SOURCE)


def quant_matmul(x, packed, scales, biases, bits, group):
    M, N = x.shape[0], packed.shape[0]
    return _KERNEL(
        inputs=[x, packed, scales, biases],
        template=[("InT", x.dtype), ("BITS", bits), ("GROUP", group), ("SG", SIMDGROUPS)],
        grid=(-(-N // SIMDGROUPS) * SIMDGROUPS * 32, 1, 1),
        threadgroup=(SIMDGROUPS * 32, 1, 1),
        output_shapes=[(M, N)],
        output_dtypes=[x.dtype],
    )[0]
