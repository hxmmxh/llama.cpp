#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
static void ggml_gemv_q6_K_8x8_q8_K_tail3(int                        n,
                                        float * GGML_RESTRICT      s,
                                        size_t                     bs,
                                        const void * GGML_RESTRICT vx,
                                        const void * GGML_RESTRICT vy,
                                        int                        nc) {
    constexpr int nrows             = 3;
    constexpr int ncols_interleaved = 8;
    constexpr int col_pairs         = ncols_interleaved / 2;

    assert(n >= 0 && n % QK_K == 0);
    assert(nc >= 0 && nc % ncols_interleaved == 0);
    assert(bs >= (size_t) nc);

    const int nb = n / QK_K;
    const uint8x16_t m4b     = vdupq_n_u8(0x0f);
    const uint8x16_t mask_lo = vdupq_n_u8(0x03);
    const uint8x16_t mask_hi = vdupq_n_u8(0x30);

    const block_q8_K * GGML_RESTRICT q8_ptr = (const block_q8_K *) vy;

    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const block_q6_Kx8 * GGML_RESTRICT q6_ptr = (const block_q6_Kx8 *) vx + (size_t) x * nb;
        float32x4_t acc_f32[nrows][2];
        for (int row = 0; row < nrows; row++) {
            for (int g = 0; g < 2; g++) {
                acc_f32[row][g] = vdupq_n_f32(0);
            }
        }

        for (int b = 0; b < nb; b++) {
            int16_t q6_scales[QK_K / 16 * ncols_interleaved];
            for (int i = 0; i < QK_K / 16; i++) {
                vst1q_s16(q6_scales + i * 8, vmovl_s8(vld1_s8(q6_ptr[b].scales + i * 8)));
            }

            int32x2_t acc[nrows][col_pairs];
            for (int row = 0; row < nrows; row++) {
                for (int cp = 0; cp < col_pairs; cp++) {
                    acc[row][cp] = vdup_n_s32(0);
                }
            }

            for (int half = 0; half < 2; half++) {
                const uint8_t * ql_base = q6_ptr[b].ql + half * 512;
                const uint8_t * qh_base = q6_ptr[b].qh + half * 256;

                for (int sb = 0; sb < QK_K / 64; sb++) {
                    const int ql_off_base = sb * QK_K / 2;
                    const int qh_off_base = ql_off_base & 255;

                    for (int cp = 0; cp < col_pairs; cp++) {
                        int32x4_t sb_acc_l[nrows];
                        int32x4_t sb_acc_h[nrows];
                        for (int row = 0; row < nrows; row++) {
                            sb_acc_l[row] = vdupq_n_s32(0);
                            sb_acc_h[row] = vdupq_n_s32(0);
                        }

                        for (int chunk = 0; chunk < 2; chunk++) {
                            const int idx = chunk * col_pairs + cp;
                            const uint8x16_t q6_ql = vld1q_u8(ql_base + ql_off_base + idx * 16);
                            uint8x16_t q6_qh = vld1q_u8(qh_base + qh_off_base + idx * 16);
                            if (sb > 1) {
                                q6_qh = vshrq_n_u8(q6_qh, 2);
                            }

                            const int8x16_t q6_l = vreinterpretq_s8_u8(vsliq_n_u8(vandq_u8(q6_ql, m4b), vandq_u8(q6_qh, mask_lo), 4));
                            const int8x16_t q6_h = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6_ql, 4), vandq_u8(q6_qh, mask_hi)));

                            for (int row = 0; row < nrows; row++) {
                                const int8_t * q8_base_l = q8_ptr[(size_t) row * nb + b].qs + half * 128 + sb * 16 + chunk * 8;
                                const int8_t * q8_base_h = q8_base_l + 64;
                                const int8x16_t q8_l = (int8x16_t) vld1q_dup_s64((const int64_t *) q8_base_l);
                                const int8x16_t q8_h = (int8x16_t) vld1q_dup_s64((const int64_t *) q8_base_h);
                                sb_acc_l[row] = vdotq_s32(sb_acc_l[row], q6_l, q8_l);
                                sb_acc_h[row] = vdotq_s32(sb_acc_h[row], q6_h, q8_h);
                            }
                        }

                        const int scale_idx_l = half * 8 + sb;
                        const int scale_idx_h = half * 8 + sb + 4;
                        const int32x2_t scale_vec_l = { (int32_t) q6_scales[scale_idx_l * 8 + cp * 2],
                                                        (int32_t) q6_scales[scale_idx_l * 8 + cp * 2 + 1] };
                        const int32x2_t scale_vec_h = { (int32_t) q6_scales[scale_idx_h * 8 + cp * 2],
                                                        (int32_t) q6_scales[scale_idx_h * 8 + cp * 2 + 1] };

                        for (int row = 0; row < nrows; row++) {
                            const int32x2_t sum_l = vpadd_s32(vget_low_s32(sb_acc_l[row]), vget_high_s32(sb_acc_l[row]));
                            const int32x2_t sum_h = vpadd_s32(vget_low_s32(sb_acc_h[row]), vget_high_s32(sb_acc_h[row]));
                            acc[row][cp] = vmla_s32(acc[row][cp], sum_l, scale_vec_l);
                            acc[row][cp] = vmla_s32(acc[row][cp], sum_h, scale_vec_h);
                        }
                    }
                }
            }

            const float32x4_t q6_d_0 = vcvt_f32_f16(vld1_f16((const __fp16 *) q6_ptr[b].d));
            const float32x4_t q6_d_1 = vcvt_f32_f16(vld1_f16((const __fp16 *) q6_ptr[b].d + 4));

            for (int row = 0; row < nrows; row++) {
                const block_q8_K & q8 = q8_ptr[(size_t) row * nb + b];
                int32x4_t bias_lo = vdupq_n_s32(0);
                int32x4_t bias_hi = vdupq_n_s32(0);

                // Keep the unsigned 6-bit dots and apply the original -32 correction.
                for (int i = 0; i < QK_K / 16; i += 4) {
                    const int16x4_t bsums_vec = vld1_s16(q8.bsums + i);
                    const int16x4_t scales_lo_0 = vld1_s16(q6_scales + (i + 0) * 8);
                    const int16x4_t scales_hi_0 = vld1_s16(q6_scales + (i + 0) * 8 + 4);
                    const int16x4_t scales_lo_1 = vld1_s16(q6_scales + (i + 1) * 8);
                    const int16x4_t scales_hi_1 = vld1_s16(q6_scales + (i + 1) * 8 + 4);
                    const int16x4_t scales_lo_2 = vld1_s16(q6_scales + (i + 2) * 8);
                    const int16x4_t scales_hi_2 = vld1_s16(q6_scales + (i + 2) * 8 + 4);
                    const int16x4_t scales_lo_3 = vld1_s16(q6_scales + (i + 3) * 8);
                    const int16x4_t scales_hi_3 = vld1_s16(q6_scales + (i + 3) * 8 + 4);

                    bias_lo = vmlal_lane_s16(bias_lo, scales_lo_0, bsums_vec, 0);
                    bias_hi = vmlal_lane_s16(bias_hi, scales_hi_0, bsums_vec, 0);
                    bias_lo = vmlal_lane_s16(bias_lo, scales_lo_1, bsums_vec, 1);
                    bias_hi = vmlal_lane_s16(bias_hi, scales_hi_1, bsums_vec, 1);
                    bias_lo = vmlal_lane_s16(bias_lo, scales_lo_2, bsums_vec, 2);
                    bias_hi = vmlal_lane_s16(bias_hi, scales_hi_2, bsums_vec, 2);
                    bias_lo = vmlal_lane_s16(bias_lo, scales_lo_3, bsums_vec, 3);
                    bias_hi = vmlal_lane_s16(bias_hi, scales_hi_3, bsums_vec, 3);
                }

                bias_lo = vshlq_n_s32(bias_lo, 5);
                bias_hi = vshlq_n_s32(bias_hi, 5);
                acc[row][0] = vsub_s32(acc[row][0], vget_low_s32(bias_lo));
                acc[row][1] = vsub_s32(acc[row][1], vget_high_s32(bias_lo));
                acc[row][2] = vsub_s32(acc[row][2], vget_low_s32(bias_hi));
                acc[row][3] = vsub_s32(acc[row][3], vget_high_s32(bias_hi));

                const float32x4_t q8_d = vdupq_n_f32(q8.d);
                const float32x4_t sb_scale_0 = vmulq_f32(q6_d_0, q8_d);
                const float32x4_t sb_scale_1 = vmulq_f32(q6_d_1, q8_d);
                const float32x2_t w_01 = vmul_f32(vcvt_f32_s32(acc[row][0]), vget_low_f32(sb_scale_0));
                const float32x2_t w_23 = vmul_f32(vcvt_f32_s32(acc[row][1]), vget_high_f32(sb_scale_0));
                const float32x2_t w_45 = vmul_f32(vcvt_f32_s32(acc[row][2]), vget_low_f32(sb_scale_1));
                const float32x2_t w_67 = vmul_f32(vcvt_f32_s32(acc[row][3]), vget_high_f32(sb_scale_1));
                acc_f32[row][0] = vaddq_f32(acc_f32[row][0], vcombine_f32(w_01, w_23));
                acc_f32[row][1] = vaddq_f32(acc_f32[row][1], vcombine_f32(w_45, w_67));
            }
        }

        const int base = x * ncols_interleaved;
        for (int row = 0; row < nrows; row++) {
            vst1q_f32(s + (size_t) row * bs + base, acc_f32[row][0]);
            vst1q_f32(s + (size_t) row * bs + base + 4, acc_f32[row][1]);
        }
    }
}
#endif
