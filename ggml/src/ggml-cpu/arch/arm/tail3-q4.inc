#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)

static inline void decode_q_Kx8_6bit_scales(const uint8_t * scales_in, int16x8_t * out_mins, int8_t * out_scales);

static inline void ggml_q4_K_tail3_dot4_8x8(int32x4_t * acc,
                                          int8x16_t w0, int8x16_t w1, int8x16_t w2, int8x16_t w3,
                                          const block_q8_K * q8, int nb, int b, int qs_offset) {
    for (int r = 0; r < 3; r++) {
        const int8_t * q = q8[(size_t) r * nb + b].qs + qs_offset;
        acc[r] = ggml_vdotq_s32(acc[r], w0, (int8x16_t) vld1q_dup_s64((const int64_t *) (q + 0)));
        acc[r] = ggml_vdotq_s32(acc[r], w1, (int8x16_t) vld1q_dup_s64((const int64_t *) (q + 8)));
        acc[r] = ggml_vdotq_s32(acc[r], w2, (int8x16_t) vld1q_dup_s64((const int64_t *) (q + 16)));
        acc[r] = ggml_vdotq_s32(acc[r], w3, (int8x16_t) vld1q_dup_s64((const int64_t *) (q + 24)));
    }
}

static void ggml_gemv_q4_K_8x8_q8_K_tail3(int n, float * GGML_RESTRICT s, size_t bs,
                                        const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nc) {
    constexpr int ncols_interleaved = 8;
    const int nb = n / QK_K;
    const uint8x16_t m4b = vdupq_n_u8(0x0f);
    const block_q8_K * GGML_RESTRICT q8_ptr = (const block_q8_K *) vy;

    assert(n >= 0 && n % QK_K == 0);
    assert(nc >= 0 && nc % ncols_interleaved == 0);
    assert(bs >= (size_t) nc);

    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const block_q4_Kx8 * GGML_RESTRICT q4_ptr = (const block_q4_Kx8 *) vx + (size_t) x * nb;

        // Process four columns at a time to limit the three-row register footprint.
        for (int c = 0; c < 2; c++) {
            float32x4_t acc_f32[3] = { vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0) };

            for (int b = 0; b < nb; b++) {
                const float32x4_t q4_d = vcvt_f32_f16(vld1_f16((const __fp16 *) q4_ptr[b].d + 4 * c));
                float32x4_t sb_scale[3];
                int32x4_t bias_acc[3] = { vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0) };
                int16_t bsums_arr[3][8];
                for (int r = 0; r < 3; r++) {
                    const block_q8_K & q8 = q8_ptr[(size_t) r * nb + b];
                    sb_scale[r] = vmulq_f32(q4_d, vdupq_n_f32(q8.d));
                    const int16x8_t bsums = vpaddq_s16(vld1q_s16(q8.bsums), vld1q_s16(q8.bsums + 8));
                    vst1q_s16(bsums_arr[r], bsums);
                }

                for (int sb = 0; sb < QK_K / 64; sb++) {
                    int16x8_t q4sb_mins[2];
                    int16x8_t q4sb_scales[2];
                    for (int i = 0; i < 2; i++) {
                        int8_t aux_q4sb[8];
                        const int offset = sb * 24 + i * 12;
                        decode_q_Kx8_6bit_scales(&q4_ptr[b].scales[offset], &q4sb_mins[i], aux_q4sb);
                        q4sb_scales[i] = vmovl_s8(vld1_s8(aux_q4sb));
                    }

                    uint8x16_t q4_cols[2][4];
                    for (int cp = 0; cp < 2; cp++) {
                        for (int i = 0; i < 4; i++) {
                            q4_cols[cp][i] = vld1q_u8(q4_ptr[b].qs + sb * QK_K + 16 * (2 * c + cp) + 64 * i);
                        }
                    }

                    // Keep the original low-then-high FMA sequence for each subblock.
                    for (int hi = 0; hi < 2; hi++) {
                        int32x4_t acc[2][3];
                        for (int cp = 0; cp < 2; cp++) {
                            for (int r = 0; r < 3; r++) {
                                acc[cp][r] = vdupq_n_s32(0);
                            }
                            int8x16_t weights[4];
                            for (int i = 0; i < 4; i++) {
                                weights[i] = vreinterpretq_s8_u8(hi == 0 ? vandq_u8(q4_cols[cp][i], m4b) : vshrq_n_u8(q4_cols[cp][i], 4));
                            }
                            ggml_q4_K_tail3_dot4_8x8(acc[cp], weights[0], weights[1], weights[2], weights[3],
                                q8_ptr, nb, b, sb * 64 + hi * 32);
                        }
                        const int16x4_t scales = c == 0 ? vget_low_s16(q4sb_scales[hi]) : vget_high_s16(q4sb_scales[hi]);
                        for (int r = 0; r < 3; r++) {
                            const float32x4_t sumf = vcvtq_f32_s32(vmulq_s32(vmovl_s16(scales), vpaddq_s32(acc[0][r], acc[1][r])));
                            acc_f32[r] = vfmaq_f32(acc_f32[r], sb_scale[r], sumf);
                        }
                    }

                    const int16x4_t min_lo = c == 0 ? vget_low_s16(q4sb_mins[0]) : vget_high_s16(q4sb_mins[0]);
                    const int16x4_t min_hi = c == 0 ? vget_low_s16(q4sb_mins[1]) : vget_high_s16(q4sb_mins[1]);
                    for (int r = 0; r < 3; r++) {
                        const int16x4_t bsums_lo = vdup_n_s16(bsums_arr[r][2 * sb + 0]);
                        const int16x4_t bsums_hi = vdup_n_s16(bsums_arr[r][2 * sb + 1]);
                        bias_acc[r] = vmlal_s16(bias_acc[r], bsums_lo, min_lo);
                        bias_acc[r] = vmlal_s16(bias_acc[r], bsums_hi, min_hi);
                    }
                }

                const float32x4_t q4_dmin = vcvt_f32_f16(vld1_f16((const __fp16 *) q4_ptr[b].dmin + 4 * c));
                for (int r = 0; r < 3; r++) {
                    const float32x4_t q8_d = vdupq_n_f32(q8_ptr[(size_t) r * nb + b].d);
                    const float32x4_t sb_min = vmulq_f32(q4_dmin, q8_d);
                    acc_f32[r] = vmlsq_f32(acc_f32[r], vcvtq_f32_s32(bias_acc[r]), sb_min);
                }
            }

            const int base = x * ncols_interleaved + 4 * c;
            for (int r = 0; r < 3; r++) {
                vst1q_f32(s + (size_t) r * bs + base, acc_f32[r]);
            }
        }
    }
}

#endif
