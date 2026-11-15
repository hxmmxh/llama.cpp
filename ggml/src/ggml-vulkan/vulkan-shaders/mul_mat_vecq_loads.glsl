#if defined(MMVQ_Q4K_VEC128)
layout(binding = 0) readonly buffer A_QUANTS128 { uvec4 data_a_quants128[]; };
#elif defined(MMVQ_Q6K_WORD32)
layout(binding = 0) readonly buffer A_QUANTS32 { uint data_a_quants32[]; };

uint load_q6_word(uint byte_address) {
    const uint word = byte_address / 4;
#if defined(MMVQ_Q6K_BRANCHLESS)
    const uint halfword = (byte_address & 2u) >> 1u;
    const uint shift = halfword * 16u;
    const uint lo = data_a_quants32[word];
    const uint hi = data_a_quants32[word + halfword];
    return (lo >> shift) | (hi << ((32u - shift) & 31u));
#else
    const uint lo = data_a_quants32[word];
    if ((byte_address & 2u) == 0) {
        return lo;
    }
    return (lo >> 16) | (data_a_quants32[word + 1] << 16);
#endif
}

uvec4 load_q6_four(uint byte_address) {
#if defined(MMVQ_Q6K_FIVE_WORDS)
    const uint word = byte_address / 4u;
    const uvec4 lo = uvec4(data_a_quants32[word], data_a_quants32[word + 1u],
                            data_a_quants32[word + 2u], data_a_quants32[word + 3u]);
    if ((byte_address & 2u) == 0u) {
        return lo;
    }
    const uint tail = data_a_quants32[word + 4u];
    return (lo >> 16) | (uvec4(lo.yzw, tail) << 16);
#else
    return uvec4(load_q6_word(byte_address), load_q6_word(byte_address + 4),
                 load_q6_word(byte_address + 8), load_q6_word(byte_address + 12));
#endif
}
#endif
