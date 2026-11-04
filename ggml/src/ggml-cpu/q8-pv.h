#pragma once

#include "ggml-quants.h"
#include "simd-mappings.h"

#if defined(__aarch64__) && defined(__ARM_NEON) && !defined(__ARM_FEATURE_SVE)
static inline void ggml_vec_mad_q8_0_f32(
        const int64_t n, float * GGML_RESTRICT y,
        const block_q8_0 * GGML_RESTRICT x, const float v) {
    GGML_ASSERT(n % QK8_0 == 0);
    for (int64_t i = 0; i < n / QK8_0; ++i) {
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d);
        for (int j = 0; j < QK8_0; j += 16) {
            const int8x16_t q = vld1q_s8(x[i].qs + j);
            const int16x8_t lo = vmovl_s8(vget_low_s8(q));
            const int16x8_t hi = vmovl_s8(vget_high_s8(q));
            // Keep the dequantization rounding before the original F32 FMA.
            const float32x4_t a = vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo))), d);
            const float32x4_t b = vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo))), d);
            const float32x4_t c = vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi))), d);
            const float32x4_t e = vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi))), d);
            float * dst = y + i * QK8_0 + j;
            vst1q_f32(dst,      vfmaq_n_f32(vld1q_f32(dst),      a, v));
            vst1q_f32(dst + 4,  vfmaq_n_f32(vld1q_f32(dst + 4),  b, v));
            vst1q_f32(dst + 8,  vfmaq_n_f32(vld1q_f32(dst + 8),  c, v));
            vst1q_f32(dst + 12, vfmaq_n_f32(vld1q_f32(dst + 12), e, v));
        }
    }
}
#endif
