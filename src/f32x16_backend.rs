use crate::backend_imports::*;

#[cfg(not(target_feature = "avx512f"))]
#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub(crate) struct Inner {
    pub lo: f32x8,
    pub hi: f32x8,
}

safe_target_feature! {
impl SimdBackend for f32x16 {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        type Inner = __m512;
      } else {
        type Inner = Inner;
      }
    }

    #[inline]
    pub(crate) fn simd_eq_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="avx512f")] {
            Self(_mm512_maskz_mov_ps(
              _mm512_cmp_ps_mask::<{_CMP_EQ_OQ}>(self.0, rhs.0),
              _mm512_castsi512_ps(_mm512_set1_epi32(-1)),
            ))
          } else {
            Self(Inner {
              lo: self.0.lo.simd_eq(rhs.0.lo),
              hi: self.0.hi.simd_eq(rhs.0.hi),
            })
          }
        }
    }

    #[inline]
    pub(crate) fn simd_ne_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="avx512f")] {
            Self(_mm512_maskz_mov_ps(
              _mm512_cmp_ps_mask::<{_CMP_NEQ_UQ}>(self.0, rhs.0),
              _mm512_castsi512_ps(_mm512_set1_epi32(-1)),
            ))
          } else {
            Self(Inner {
              lo: self.0.lo.simd_ne(rhs.0.lo),
              hi: self.0.hi.simd_ne(rhs.0.hi),
            })
          }
        }
    }

    #[inline]
    pub(crate) fn simd_lt_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="avx512f")] {
            Self(_mm512_maskz_mov_ps(
              _mm512_cmp_ps_mask::<{_CMP_LT_OQ}>(self.0, rhs.0),
              _mm512_castsi512_ps(_mm512_set1_epi32(-1)),
            ))
          } else {
            Self(Inner {
              lo: self.0.lo.simd_lt(rhs.0.lo),
              hi: self.0.hi.simd_lt(rhs.0.hi),
            })
          }
        }
    }

    #[inline]
    pub(crate) fn simd_gt_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="avx512f")] {
            Self(_mm512_maskz_mov_ps(
              _mm512_cmp_ps_mask::<{_CMP_GT_OQ}>(self.0, rhs.0),
              _mm512_castsi512_ps(_mm512_set1_epi32(-1)),
            ))
          } else {
            Self(Inner {
              lo: self.0.lo.simd_gt(rhs.0.lo),
              hi: self.0.hi.simd_gt(rhs.0.hi),
            })
          }
        }
    }

    #[inline]
    pub(crate) fn simd_le_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="avx512f")] {
            Self(_mm512_maskz_mov_ps(
              _mm512_cmp_ps_mask::<{_CMP_LE_OQ}>(self.0, rhs.0),
              _mm512_castsi512_ps(_mm512_set1_epi32(-1)),
            ))
          } else {
            Self(Inner {
              lo: self.0.lo.simd_le(rhs.0.lo),
              hi: self.0.hi.simd_le(rhs.0.hi),
            })
          }
        }
    }

    #[inline]
    pub(crate) fn simd_ge_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="avx512f")] {
            Self(_mm512_maskz_mov_ps(
              _mm512_cmp_ps_mask::<{_CMP_GE_OQ}>(self.0, rhs.0),
              _mm512_castsi512_ps(_mm512_set1_epi32(-1)),
            ))
          } else {
            Self(Inner {
              lo: self.0.lo.simd_ge(rhs.0.lo),
              hi: self.0.hi.simd_ge(rhs.0.hi),
            })
          }
        }
    }

    #[inline]
    pub(crate) fn select_backend(self, if_true: Self, if_false: Self) -> Self {
        pick! {
          if #[cfg(target_feature="avx512dq")] {
            Self(_mm512_mask_blend_ps(
              _mm512_movepi32_mask(_mm512_castps_si512(self.0)),
              if_false.0,
              if_true.0,
            ))
          } else {
            let [self_lo, self_hi] = cast::<f32x16, [f32x8; 2]>(self);
            let [if_true_lo, if_true_hi] = cast::<f32x16, [f32x8; 2]>(if_true);
            let [if_false_lo, if_false_hi] = cast::<f32x16, [f32x8; 2]>(if_false);
            cast([
              self_lo.select(if_true_lo, if_false_lo),
              self_hi.select(if_true_hi, if_false_hi),
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn to_bitmask_backend(self) -> u32 {
        pick! {
          if #[cfg(target_feature="avx512dq")] {
            _mm512_movepi32_mask(_mm512_castps_si512(self.0)) as u32
          } else {
            let [self_lo, self_hi] = cast::<f32x16, [f32x8; 2]>(self);
            (self_hi.to_bitmask() << 8) | self_lo.to_bitmask()
          }
        }
    }

    #[inline]
    pub(crate) fn any_backend(self) -> bool {
        pick! {
          if #[cfg(target_feature="avx512dq")] {
            _mm512_movepi32_mask(_mm512_castps_si512(self.0)) != 0
          } else {
            let [self_lo, self_hi] = cast::<f32x16, [f32x8; 2]>(self);
            self_lo.any() || self_hi.any()
          }
        }
    }

    #[inline]
    pub(crate) fn all_backend(self) -> bool {
        pick! {
          if #[cfg(target_feature="avx512dq")] {
            _mm512_movepi32_mask(_mm512_castps_si512(self.0)) == !0_u16
          } else {
            let [self_lo, self_hi] = cast::<f32x16, [f32x8; 2]>(self);
            self_lo.all() && self_hi.all()
          }
        }
    }

    #[inline]
    pub(crate) fn interleave_low_backend(self, other: Self) -> Self {
        Self::from_bits(self.to_bits().interleave_low(other.to_bits()))
    }

    #[inline]
    pub(crate) fn interleave_high_backend(self, other: Self) -> Self {
        Self::from_bits(self.to_bits().interleave_high(other.to_bits()))
    }

    #[inline]
    pub(crate) fn transpose_backend(data: [f32x16; 16]) -> [f32x16; 16] {
        // TODO(perf): Try to add an optimized `avx512` implementation using
        // `_mm512_unpackhi_ps`.

        #[inline(always)]
        fn transpose_column(data: &[f32x16; 16], index: usize) -> f32x16 {
            f32x16::from_array([
                data[0].as_array()[index],
                data[1].as_array()[index],
                data[2].as_array()[index],
                data[3].as_array()[index],
                data[4].as_array()[index],
                data[5].as_array()[index],
                data[6].as_array()[index],
                data[7].as_array()[index],
                data[8].as_array()[index],
                data[9].as_array()[index],
                data[10].as_array()[index],
                data[11].as_array()[index],
                data[12].as_array()[index],
                data[13].as_array()[index],
                data[14].as_array()[index],
                data[15].as_array()[index],
            ])
        }

        [
            transpose_column(&data, 0),
            transpose_column(&data, 1),
            transpose_column(&data, 2),
            transpose_column(&data, 3),
            transpose_column(&data, 4),
            transpose_column(&data, 5),
            transpose_column(&data, 6),
            transpose_column(&data, 7),
            transpose_column(&data, 8),
            transpose_column(&data, 9),
            transpose_column(&data, 10),
            transpose_column(&data, 11),
            transpose_column(&data, 12),
            transpose_column(&data, 13),
            transpose_column(&data, 14),
            transpose_column(&data, 15),
        ]
    }
}
}

safe_target_feature! {
// Backend of SIMD float functions.
impl f32x16 {
    #[inline]
    pub(crate) fn neg_backend(self) -> Self {
        pick! {
          if #[cfg(target_feature="avx512dq")] {
            Self(_mm512_xor_ps(self.0, Self::splat(-0.0).0))
          } else {
            let [self_lo, self_hi] = cast::<f32x16, [f32x8; 2]>(self);
            cast([-self_lo, -self_hi])
          }
        }
    }

    #[inline]
    pub(crate) fn not_backend(self) -> Self {
        pick! {
          if #[cfg(target_feature="avx512dq")] {
            Self(_mm512_xor_ps(self.0, _mm512_set1_ps(f32::from_bits(u32::MAX))))
          } else {
            let [self_lo, self_hi] = cast::<f32x16, [f32x8; 2]>(self);
            cast([!self_lo, !self_hi])
          }
        }
    }

    #[inline]
    pub(crate) fn add_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="avx512f")] {
            Self(_mm512_add_ps(self.0, rhs.0))
          } else {
            Self(Inner {
              lo: self.0.lo + rhs.0.lo,
              hi: self.0.hi + rhs.0.hi,
            })
          }
        }
    }

    #[inline]
    pub(crate) fn sub_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="avx512f")] {
            Self(_mm512_sub_ps(self.0, rhs.0))
          } else {
            Self(Inner {
              lo: self.0.lo - rhs.0.lo,
              hi: self.0.hi - rhs.0.hi,
            })
          }
        }
    }

    #[inline]
    pub(crate) fn mul_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="avx512f")] {
            Self(_mm512_mul_ps(self.0, rhs.0))
          } else {
            Self(Inner { lo: self.0.lo * rhs.0.lo, hi: self.0.hi * rhs.0.hi })
          }
        }
    }

    #[inline]
    pub(crate) fn div_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="avx512f")] {
            Self(_mm512_div_ps(self.0, rhs.0))
          } else {
            Self(Inner { lo: self.0.lo / rhs.0.lo, hi: self.0.hi / rhs.0.hi })
          }
        }
    }

    #[inline]
    pub(crate) fn rem_backend(self, rhs: Self) -> Self {
        Self::from_array([
            self.to_array()[0] % rhs.to_array()[0],
            self.to_array()[1] % rhs.to_array()[1],
            self.to_array()[2] % rhs.to_array()[2],
            self.to_array()[3] % rhs.to_array()[3],
            self.to_array()[4] % rhs.to_array()[4],
            self.to_array()[5] % rhs.to_array()[5],
            self.to_array()[6] % rhs.to_array()[6],
            self.to_array()[7] % rhs.to_array()[7],
            self.to_array()[8] % rhs.to_array()[8],
            self.to_array()[9] % rhs.to_array()[9],
            self.to_array()[10] % rhs.to_array()[10],
            self.to_array()[11] % rhs.to_array()[11],
            self.to_array()[12] % rhs.to_array()[12],
            self.to_array()[13] % rhs.to_array()[13],
            self.to_array()[14] % rhs.to_array()[14],
            self.to_array()[15] % rhs.to_array()[15],
        ])
    }

    #[inline]
    pub(crate) fn bitand_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="avx512dq")] {
            Self(_mm512_and_ps(self.0, rhs.0))
          } else {
            let [self_lo, self_hi] = cast::<f32x16, [f32x8; 2]>(self);
            let [rhs_lo, rhs_hi] = cast::<f32x16, [f32x8; 2]>(rhs);
            cast([self_lo & rhs_lo, self_hi & rhs_hi])
          }
        }
    }

    #[inline]
    pub(crate) fn bitor_backend(self, rhs: Self) -> Self {
        pick! {
        if #[cfg(target_feature="avx512dq")] {
            Self(_mm512_or_ps(self.0, rhs.0))
          } else {
            let [self_lo, self_hi] = cast::<f32x16, [f32x8; 2]>(self);
            let [rhs_lo, rhs_hi] = cast::<f32x16, [f32x8; 2]>(rhs);
            cast([self_lo | rhs_lo, self_hi | rhs_hi])
          }
        }
    }

    #[inline]
    pub(crate) fn bitxor_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="avx512dq")] {
            Self(_mm512_xor_ps(self.0, rhs.0))
          } else {
            let [self_lo, self_hi] = cast::<f32x16, [f32x8; 2]>(self);
            let [rhs_lo, rhs_hi] = cast::<f32x16, [f32x8; 2]>(rhs);
            cast([self_lo ^ rhs_lo, self_hi ^ rhs_hi])
          }
        }
    }

    #[inline]
    pub(crate) fn reduce_add_backend(self) -> f32 {
        pick! {
          if #[cfg(target_feature="avx512f")]{
            _mm512_reduce_add_ps(self.0)
          } else {
            self.0.lo.reduce_add() + self.0.hi.reduce_add()
          }
        }
    }

    #[inline]
    pub(crate) fn reduce_mul_backend(self) -> f32 {
        pick! {
          if #[cfg(target_feature="avx512f")] {
            _mm512_reduce_mul_ps(self.0)
          } else {
            self.0.lo.reduce_mul() * self.0.hi.reduce_mul()
          }
        }
    }

    #[inline]
    pub(crate) fn bitselect_backend(self, if_one: Self, if_zero: Self) -> Self {
        pick! {
          if #[cfg(target_feature="avx512dq")] {
            Self(_mm512_or_ps(
                _mm512_and_ps(if_one.0, self.0),
                _mm512_andnot_ps(self.0, if_zero.0),
            ))
          } else {
            let [self_lo, self_hi] = cast::<f32x16, [f32x8; 2]>(self);
            let [if_one_lo, if_one_hi] = cast::<f32x16, [f32x8; 2]>(if_one);
            let [if_zero_lo, if_zero_hi] = cast::<f32x16, [f32x8; 2]>(if_zero);
            cast([
              self_lo.bitselect(if_one_lo, if_zero_lo),
              self_hi.bitselect(if_one_hi, if_zero_hi),
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn is_nan_backend(self) -> Self {
        pick! {
          if #[cfg(target_feature = "avx512f")] {
            Self(_mm512_maskz_mov_ps(
              _mm512_cmp_ps_mask::<{_CMP_UNORD_Q}>(self.0, self.0),
              _mm512_castsi512_ps(_mm512_set1_epi32(-1)),
            ))
          } else {
            Self(Inner {
              lo: self.0.lo.is_nan(),
              hi: self.0.hi.is_nan(),
            })
          }
        }
    }

    #[inline]
    pub(crate) fn is_infinite_backend(self) -> Self {
        let shifted_inf = u32x16::from(0xFF000000);
        let u: u32x16 = cast(self);
        let shift_u = u << 1_u64;
        let out = (shift_u).simd_eq(shifted_inf);
        cast(out)
    }

    #[inline]
    pub(crate) fn is_finite_backend(self) -> Self {
        let shifted_exp_mask = u32x16::splat(0xFF000000);
        let u: u32x16 = cast(self);
        let shift_u = u << 1_u32;
        let out = !(shift_u & shifted_exp_mask).simd_eq(shifted_exp_mask);
        cast(out)
    }

    #[inline]
    pub(crate) fn is_sign_positive_backend(self) -> Self {
        const SIGN_MASK: u32x16 = u32x16::splat((-0.0_f32).to_bits());

        let bits = cast::<f32x16, u32x16>(self);
        let sign = bits & SIGN_MASK;
        let result = sign.simd_eq(u32x16::ZERO);
        cast::<u32x16, f32x16>(result)
    }

    #[inline]
    pub(crate) fn recip_backend(self) -> Self {
        pick! {
          if #[cfg(target_feature="avx512f")] {
            // TODO(perf): Consider using `_mm512_rcp14_ps` if the relative
            // error is acceptable.
            1.0 / self
          } else {
            Self(Inner {
              lo: self.0.lo.recip(),
              hi: self.0.hi.recip(),
            })
          }
        }
    }

    #[inline]
    pub(crate) fn recip_sqrt_backend(self) -> Self {
        pick! {
          if #[cfg(target_feature="avx512f")] {
            // TODO(perf): Consider using `_mm512_rsqrt14_ps` if the relative
            // error is acceptable.
            self.sqrt().recip()
          } else {
            Self(Inner {
              lo: self.0.lo.recip_sqrt(),
              hi: self.0.hi.recip_sqrt(),
            })
          }
        }
    }

    #[inline]
    pub(crate) fn max_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="avx512f")] {
            // _mm512_max_ps seems to do rhs < self ? self : rhs. So if there's any NaN
            // involved, it chooses rhs, so we need to specifically check rhs for
            // NaN.
            rhs.is_nan().select(self, Self(_mm512_max_ps(self.0, rhs.0)))
          } else {
            Self(Inner {
              lo: self.0.lo.max(rhs.0.lo),
              hi: self.0.hi.max(rhs.0.hi),
            })
          }
        }
    }

    #[inline]
    pub(crate) fn fast_max_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="avx512f")] {
            Self(_mm512_max_ps(self.0, rhs.0))
          } else {
            Self(Inner {
              lo: self.0.lo.fast_max(rhs.0.lo),
              hi: self.0.hi.fast_max(rhs.0.hi),
            })
          }
        }
    }

    #[inline]
    pub(crate) fn min_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="avx512f")] {
            // _mm512_min_ps seems to do rhs > self ? self : rhs. So if there's any NaN
            // involved, it chooses rhs, so we need to specifically check rhs for
            // NaN.
            rhs.is_nan().select(self, Self(_mm512_min_ps(self.0, rhs.0)))
          } else {
            Self(Inner {
              lo: self.0.lo.min(rhs.0.lo),
              hi: self.0.hi.min(rhs.0.hi),
            })
          }
        }
    }

    #[inline]
    pub(crate) fn fast_min_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="avx512f")] {
            Self(_mm512_min_ps(self.0, rhs.0))
          } else {
            Self(Inner {
              lo: self.0.lo.fast_min(rhs.0.lo),
              hi: self.0.hi.fast_min(rhs.0.hi),
            })
          }
        }
    }

    #[inline]
    pub(crate) fn clamp_backend(self, min: Self, max: Self) -> Self {
        pick! {
          if #[cfg(target_feature="avx512f")] {
            // This works since all bits set is NaN.
            self.fast_clamp(min, max) | min.is_nan() | max.is_nan()
          } else {
            // Some targets have better implementations than the above one.
            Self(Inner {
              lo: self.0.lo.clamp(min.0.lo, max.0.lo),
              hi: self.0.hi.clamp(min.0.hi, max.0.hi),
            })
          }
        }
    }

    #[inline]
    pub(crate) fn fast_clamp_backend(self, min: Self, max: Self) -> Self {
        pick! {
          if #[cfg(target_feature="avx512f")] {
            // For both `_mm512_min_ps` and `_mm512_max_ps` if any input is NaN, `rhs` gets
            // chosen. For `self` to be chosen, `self` must be the second argument.
            Self(_mm512_max_ps(min.0, _mm512_min_ps(max.0, self.0)))
          } else {
            Self(Inner {
              lo: self.0.lo.fast_clamp(min.0.lo, max.0.lo),
              hi: self.0.hi.fast_clamp(min.0.hi, max.0.hi),
            })
          }
        }
    }

    #[inline]
    pub(crate) fn reduce_max_backend(self) -> f32 {
        let [self_a, self_b] = cast::<f32x16, [f32x8; 2]>(self);
        self_a.max(self_b).reduce_max()
    }

    #[inline]
    pub(crate) fn fast_reduce_max_backend(self) -> f32 {
        let [self_a, self_b] = cast::<f32x16, [f32x8; 2]>(self);
        self_a.fast_max(self_b).fast_reduce_max()
    }

    #[inline]
    pub(crate) fn reduce_min_backend(self) -> f32 {
        let [self_a, self_b] = cast::<f32x16, [f32x8; 2]>(self);
        self_a.min(self_b).reduce_min()
    }

    #[inline]
    pub(crate) fn fast_reduce_min_backend(self) -> f32 {
        let [self_a, self_b] = cast::<f32x16, [f32x8; 2]>(self);
        self_a.fast_min(self_b).fast_reduce_min()
    }

    #[inline]
    pub(crate) fn abs_backend(self) -> Self {
        pick! {
          if #[cfg(target_feature="avx512f")] {
            let non_sign_bits = f32x16::from(f32::from_bits(i32::MAX as u32));
            self & non_sign_bits
          } else {
            Self(Inner {
              lo: self.0.lo.abs(),
              hi: self.0.hi.abs(),
            })
          }
        }
    }

    #[inline]
    pub(crate) fn floor_backend(self) -> Self {
        pick! {
          if #[cfg(target_feature="avx512f")] {
            Self(_mm512_roundscale_ps::<{_MM_FROUND_NO_EXC | _MM_FROUND_TO_NEG_INF}>(self.0))
          } else {
            Self(Inner {
              lo: self.0.lo.floor(),
              hi: self.0.hi.floor(),
            })
          }
        }
    }

    #[inline]
    pub(crate) fn ceil_backend(self) -> Self {
        pick! {
          if #[cfg(target_feature="avx512f")] {
            Self(_mm512_roundscale_ps::<{_MM_FROUND_NO_EXC | _MM_FROUND_TO_POS_INF}>(self.0))
          } else {
            Self(Inner {
              lo: self.0.lo.ceil(),
              hi: self.0.hi.ceil(),
            })
          }
        }
    }

    #[inline]
    pub(crate) fn round_backend(self) -> Self {
        pick! {
          if #[cfg(target_feature="avx512f")] {
            const_f32_as_f32x16!(HALF_NEXT_DOWN, 0.5_f32.next_down());
            const_f32_as_f32x16!(BOUNDS_LIMIT, 8388608.0);

            let self_abs = self.abs();

            let adjusted_self = self_abs + Self::HALF;
            let result_abs = Self(_mm512_roundscale_ps::<{_MM_FROUND_NO_EXC | _MM_FROUND_TO_ZERO}>(adjusted_self.0));
            // The addition breaks for `0.5.next_down()` which incorrectly rounds to
            // `1.0`. This resets the result back to `0.0`.
            let result_abs = result_abs & self_abs.simd_ne(HALF_NEXT_DOWN);

            // Large value, infinity and NaN need special handling.
            let bounds_mask: Self = cast(_mm512_maskz_mov_epi32(
              _mm512_cmp_epi32_mask::<{_MM_CMPINT_LT}>(cast(self_abs), cast(BOUNDS_LIMIT)),
              _mm512_set1_epi32(-1),
            ));

            // `abs` keeps the original sign.
            bounds_mask.abs().bitselect(result_abs, self)
          } else {
            Self(Inner {
              lo: self.0.lo.round(),
              hi: self.0.hi.round(),
            })
          }
        }
    }

    #[inline]
    pub(crate) fn round_int_backend(self) -> i32x16 {
        pick! {
          if #[cfg(target_feature="avx512f")] {
            // Based on: https://github.com/v8/v8/blob/210987a552a2bf2a854b0baa9588a5959ff3979d/src/codegen/shared-ia32-x64/macro-assembler-shared-ia32-x64.h#L489-L504
            let non_nan_mask = self.simd_eq(self);
            let non_nan = self & non_nan_mask;
            let flip_to_max: i32x16 = cast(self.simd_ge(Self::splat(2147483648.0)));
            let cast: i32x16 = cast(_mm512_cvtps_epi32(non_nan.0));
            flip_to_max ^ cast
          } else {
            i32x16(crate::i32x16_backend::Inner {
              lo: self.0.lo.round_int(),
              hi: self.0.hi.round_int(),
            })
          }
        }
    }

    #[inline]
    pub(crate) fn fast_round_int_backend(self) -> i32x16 {
        pick! {
          if #[cfg(target_feature="avx512f")] {
            cast(_mm512_cvtps_epi32(self.0))
          } else {
            i32x16(crate::i32x16_backend::Inner {
              lo: self.0.lo.fast_round_int(),
              hi: self.0.hi.fast_round_int(),
            })
          }
        }
    }

    #[inline]
    pub(crate) fn round_ties_even_backend(self) -> Self {
        pick! {
          if #[cfg(target_feature="avx512f")] {
            Self(_mm512_roundscale_ps::<{_MM_FROUND_NO_EXC | _MM_FROUND_TO_NEAREST_INT}>(self.0))
          } else {
            Self(Inner {
              lo: self.0.lo.round_ties_even(),
              hi: self.0.hi.round_ties_even(),
            })
          }
        }
    }

    #[inline]
    pub(crate) fn trunc_backend(self) -> Self {
        pick! {
          if #[cfg(target_feature="avx512f")] {
            Self(_mm512_roundscale_ps::<{_MM_FROUND_NO_EXC | _MM_FROUND_TO_ZERO}>(self.0))
          } else {
            Self(Inner {
              lo: self.0.lo.trunc(),
              hi: self.0.hi.trunc(),
            })
          }
        }
    }

    #[inline]
    pub(crate) fn trunc_int_backend(self) -> i32x16 {
        pick! {
            if #[cfg(target_feature="avx512f")] {
            // Based on: https://github.com/v8/v8/blob/210987a552a2bf2a854b0baa9588a5959ff3979d/src/codegen/shared-ia32-x64/macro-assembler-shared-ia32-x64.h#L489-L504
            let non_nan_mask = self.simd_eq(self);
            let non_nan = self & non_nan_mask;
            let flip_to_max: i32x16 = cast(self.simd_ge(Self::splat(2147483648.0)));
            let cast: i32x16 = cast(_mm512_cvttps_epi32(non_nan.0));
            flip_to_max ^ cast
          } else {
            cast([
              self.0.lo.trunc_int(),
              self.0.hi.trunc_int(),
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn fast_trunc_int_backend(self) -> i32x16 {
        pick! {
          if #[cfg(all(target_feature="avx512f"))] {
            cast(_mm512_cvttps_epi32(self.0))
          } else {
            cast([
              self.0.lo.fast_trunc_int(),
              self.0.hi.fast_trunc_int(),
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn fast_mul_add_backend(self, a: Self, b: Self) -> Self {
        pick! {
          if #[cfg(all(target_feature="avx512f",target_feature="fma"))] {
            Self(_mm512_fmadd_ps(self.0, a.0, b.0))
          } else if #[cfg(target_feature="avx512f")] {
            // still want to use 512 bit ops
            (self * a) + b
          } else {
            Self(Inner {
              lo: self.0.lo.fast_mul_add(a.0.lo, b.0.lo),
              hi: self.0.hi.fast_mul_add(a.0.hi, b.0.hi),
            })
          }
        }
    }

    #[inline]
    pub(crate) fn fast_mul_sub_backend(self, a: Self, b: Self) -> Self {
        pick! {
          if #[cfg(all(target_feature="avx512f",target_feature="fma"))] {
            Self(_mm512_fmsub_ps(self.0, a.0, b.0))
          } else if #[cfg(target_feature="avx512f")] {
            // still want to use 512 bit ops
            (self * a) - b
          } else {
            Self(Inner {
              lo: self.0.lo.fast_mul_sub(a.0.lo, b.0.lo),
              hi: self.0.hi.fast_mul_sub(a.0.hi, b.0.hi),
            })
          }
        }
    }

    #[inline]
    pub(crate) fn fast_mul_neg_add_backend(self, a: Self, b: Self) -> Self {
        pick! {
          if #[cfg(all(target_feature="avx512f",target_feature="fma"))] {
            Self(_mm512_fnmadd_ps(self.0, a.0, b.0))
          } else if #[cfg(target_feature="avx512f")] {
            // still want to use 512 bit ops
            b - (self * a)
          } else {
            Self(Inner {
              lo: self.0.lo.fast_mul_neg_add(a.0.lo, b.0.lo),
              hi: self.0.hi.fast_mul_neg_add(a.0.hi, b.0.hi),
            })
          }
        }
    }

    #[inline]
    pub(crate) fn fast_mul_neg_sub_backend(self, a: Self, b: Self) -> Self {
        pick! {
          if #[cfg(all(target_feature="avx512f",target_feature="fma"))] {
            Self(_mm512_fnmsub_ps(self.0, a.0, b.0))
          } else if #[cfg(target_feature="avx512f")] {
            // still want to use 512 bit ops
            -(self * a) - b
          } else {
            Self(Inner {
              lo: self.0.lo.fast_mul_neg_sub(a.0.lo, b.0.lo),
              hi: self.0.hi.fast_mul_neg_sub(a.0.hi, b.0.hi),
            })
          }
        }
    }

    #[inline]
    pub(crate) fn powf_simd_backend(self, n: Self) -> Self {
        const_f32_as_f32x16!(ln2f_hi, 0.693359375);
        const_f32_as_f32x16!(ln2f_lo, -2.12194440e-4);
        const_f32_as_f32x16!(P0logf, 3.3333331174E-1);
        const_f32_as_f32x16!(P1logf, -2.4999993993E-1);
        const_f32_as_f32x16!(P2logf, 2.0000714765E-1);
        const_f32_as_f32x16!(P3logf, -1.6668057665E-1);
        const_f32_as_f32x16!(P4logf, 1.4249322787E-1);
        const_f32_as_f32x16!(P5logf, -1.2420140846E-1);
        const_f32_as_f32x16!(P6logf, 1.1676998740E-1);
        const_f32_as_f32x16!(P7logf, -1.1514610310E-1);
        const_f32_as_f32x16!(P8logf, 7.0376836292E-2);

        const_f32_as_f32x16!(p2expf, 1.0 / 2.0); // coefficients for Taylor expansion of exp
        const_f32_as_f32x16!(p3expf, 1.0 / 6.0);
        const_f32_as_f32x16!(p4expf, 1.0 / 24.0);
        const_f32_as_f32x16!(p5expf, 1.0 / 120.0);
        const_f32_as_f32x16!(p6expf, 1.0 / 720.0);
        const_f32_as_f32x16!(p7expf, 1.0 / 5040.0);

        let x1 = self.abs();
        let x = x1.fraction_2();
        let mask = x.simd_gt(f32x16::SQRT_2 * f32x16::HALF);
        let x = (!mask).select(x + x, x);

        let x = x - f32x16::ONE;
        let x2 = x * x;
        let lg1 = polynomial_8!(
            x, P0logf, P1logf, P2logf, P3logf, P4logf, P5logf, P6logf, P7logf, P8logf
        );
        let lg1 = lg1 * x2 * x;

        let ef = x1.exponent();
        let ef = mask.select(ef + f32x16::ONE, ef);
        let e1 = (ef * n).round_ties_even();
        let yr = ef.fast_mul_sub(n, e1);

        let lg = f32x16::HALF.fast_mul_neg_add(x2, x) + lg1;
        let x2_err = (f32x16::HALF * x).fast_mul_sub(x, f32x16::HALF * x2);
        let lg_err = f32x16::HALF.fast_mul_add(x2, lg - x) - lg1;

        let e2 = (lg * n * f32x16::LOG2_E).round_ties_even();
        let v = lg.fast_mul_sub(n, e2 * ln2f_hi);
        let v = e2.fast_mul_neg_add(ln2f_lo, v);
        let v = v - (lg_err + x2_err).fast_mul_sub(n, yr * f32x16::LN_2);

        let x = v;
        let e3 = (x * f32x16::LOG2_E).round_ties_even();
        let x = e3.fast_mul_neg_add(f32x16::LN_2, x);
        let x2 = x * x;
        let z = x2.fast_mul_add(
            polynomial_5!(x, p2expf, p3expf, p4expf, p5expf, p6expf, p7expf),
            x + f32x16::ONE,
        );

        let ee = e1 + e2 + e3;
        let ei = cast::<_, i32x16>(ee.round_int());
        let ej = cast::<_, i32x16>(ei + (cast::<_, i32x16>(z) >> 23));

        let overflow = cast::<_, f32x16>(ej.simd_gt(i32x16::splat(0x0FF)))
            | (ee.simd_gt(f32x16::splat(300.0)));
        let underflow = cast::<_, f32x16>(ej.simd_lt(i32x16::splat(0x000)))
            | (ee.simd_lt(f32x16::splat(-300.0)));

        // Add exponent by integer addition
        let z = cast::<_, f32x16>(cast::<_, i32x16>(z) + (ei << 23));
        // Check for overflow/underflow
        let z = underflow.select(f32x16::ZERO, z);
        let z = overflow.select(Self::infinity(), z);

        // Check for self == 0
        let x_zero = self.is_zero_or_subnormal();
        let z = x_zero.select(
            n.simd_lt(f32x16::ZERO).select(
                Self::infinity(),
                n.simd_eq(f32x16::ZERO).select(f32x16::ONE, f32x16::ZERO),
            ),
            z,
        );

        let x_sign = self.is_sign_negative();
        let z = if x_sign.any() {
            // Y into an integer
            let yi = n.simd_eq(n.round_ties_even());

            // Is y odd? If yes flip the sign of the result.
            let y_odd = cast::<i32x16, f32x16>(n.round_int() << 31);

            let z1 = yi.select(
                z | y_odd,
                self.simd_eq(Self::ZERO).select(z, Self::nan_pow()),
            );

            x_sign.select(z1, z)
        } else {
            z
        };

        let x_finite = self.is_finite();
        let y_finite = n.is_finite();
        let e_finite = ee.is_finite();
        if (x_finite & y_finite & (e_finite | x_zero)).all() {
            return z;
        }

        (self.is_nan() | n.is_nan()).select(self + n, z)
    }

    #[inline]
    pub(crate) fn sqrt_backend(self) -> Self {
        pick! {
          if #[cfg(target_feature="avx512f")] {
            Self(_mm512_sqrt_ps(self.0))
          } else {
            Self(Inner {
              lo: self.0.lo.sqrt(),
              hi: self.0.hi.sqrt(),
            })
          }
        }
    }

    #[inline]
    pub(crate) fn exp_backend(self) -> Self {
        const_f32_as_f32x16!(P0, 1.0 / 2.0);
        const_f32_as_f32x16!(P1, 1.0 / 6.0);
        const_f32_as_f32x16!(P2, 1.0 / 24.0);
        const_f32_as_f32x16!(P3, 1.0 / 120.0);
        const_f32_as_f32x16!(P4, 1.0 / 720.0);
        const_f32_as_f32x16!(P5, 1.0 / 5040.0);
        // LN2D_HI/LO: double-double decomposition of ln(2) for exp range reduction,
        // following the approach from fdlibm's e_exp.c (Sun Microsystems,
        // https://www.netlib.org/fdlibm/). The f32 split uses f32-precision constants
        // (0.693359375, -2.12194440e-4) summing to ln(2) with single-precision
        // accuracy; the f64 variants use a full f64 double-double
        // decomposition.
        const_f32_as_f32x16!(LN2D_HI, 0.693359375);
        const_f32_as_f32x16!(LN2D_LO, -2.12194440e-4);
        // max_x = ln(f32::MAX) ≈ 88.7229, max_r = 127 (IEEE max normal exponent)
        // min_x = -149.5 ln(2) ≈ -103.63: min r for vm_pow2n subnormal
        let max_x = f32x16::from(88.723);
        let min_x = f32x16::from(-103.63);
        // x < min_x: e^x underflows to 0 -- skip the entire pipeline
        let finite = self.is_finite();
        let neg_underflow = self.simd_lt(min_x) & finite;
        if neg_underflow.all() {
            return Self::ZERO;
        }
        let max_r = f32x16::from(127.0);
        let r = (self * Self::LOG2_E).round_ties_even();
        let big = r.simd_gt(max_r);
        let r_safe = big.select(max_r, r);
        let excess = r - max_r;
        let excess = big.select(excess, Self::ZERO);
        let scale = Self::vm_pow2n(excess);
        let x = r.fast_mul_neg_add(LN2D_HI, self);
        let x = r.fast_mul_neg_add(LN2D_LO, x);
        let z = polynomial_5!(x, P0, P1, P2, P3, P4, P5);
        let x2 = x * x;
        let z = z.fast_mul_add(x2, x);
        let n2 = Self::vm_pow2n(r_safe);
        let z = (z + Self::ONE) * scale * n2;
        let nan_mask = self.is_nan();
        let mut result = nan_mask.select(Self::nan_pow(), z);
        let pos_overflow = self.simd_gt(max_x) & finite;
        result = pos_overflow.select(Self::infinity(), result);
        result = neg_underflow.select(Self::ZERO, result);
        let pos_inf = !finite & !self.is_sign_negative() & !nan_mask;
        result = pos_inf.select(Self::infinity(), result);
        let neg_inf = !finite & self.is_sign_negative() & !nan_mask;
        result = neg_inf.select(Self::ZERO, result);
        result
    }

    #[inline]
    pub(crate) fn exp2_backend(self) -> Self {
        const_f32_as_f32x16!(P2, 1.0 / 2.0);
        const_f32_as_f32x16!(P3, 1.0 / 6.0);
        const_f32_as_f32x16!(P4, 1.0 / 24.0);
        const_f32_as_f32x16!(P5, 1.0 / 120.0);
        const_f32_as_f32x16!(P6, 1.0 / 720.0);
        const_f32_as_f32x16!(P7, 1.0 / 5040.0);

        // max_x = log2(f32::MAX) ≈ 127.99999
        // min_x = log2(f32::MIN_POSITIVE) - 23 ≈ -126 - 23 = -149
        let max_x = f32x16::from(127.99999);
        let min_x = f32x16::from(-149.5);
        let finite = self.is_finite();
        let neg_underflow = self.simd_lt(min_x) & finite;
        if neg_underflow.all() {
            return Self::ZERO;
        }

        let round = self.round_ties_even();
        let max_r = f32x16::from(127.0);
        let big = round.simd_gt(max_r);
        let r_safe = big.select(max_r, round);
        let excess = round - max_r;
        let excess = big.select(excess, Self::ZERO);
        let scale = Self::vm_pow2n(excess);

        let fract = (self - round) * Self::LN_2;
        let fract_partial_exp2 = polynomial_5!(fract, P2, P3, P4, P5, P6, P7);
        let fract2 = fract * fract;
        let fract_exp2 = fract_partial_exp2.fast_mul_add(fract2, fract) + Self::ONE;

        let n2 = Self::vm_pow2n(r_safe);
        let result = fract_exp2 * scale * n2;

        let nan_mask = self.is_nan();
        let mut result = nan_mask.select(Self::nan_pow(), result);
        let pos_overflow = self.simd_gt(max_x) & finite;
        result = pos_overflow.select(Self::infinity(), result);
        result = neg_underflow.select(Self::ZERO, result);
        let pos_inf = !finite & !self.is_sign_negative() & !nan_mask;
        result = pos_inf.select(Self::infinity(), result);
        let neg_inf = !finite & self.is_sign_negative() & !nan_mask;
        result = neg_inf.select(Self::ZERO, result);
        result
    }

    #[inline]
    pub(crate) fn ln_backend(self) -> Self {
        const_f32_as_f32x16!(HALF, 0.5);
        const_f32_as_f32x16!(P0, 3.3333331174E-1);
        const_f32_as_f32x16!(P1, -2.4999993993E-1);
        const_f32_as_f32x16!(P2, 2.0000714765E-1);
        const_f32_as_f32x16!(P3, -1.6668057665E-1);
        const_f32_as_f32x16!(P4, 1.4249322787E-1);
        const_f32_as_f32x16!(P5, -1.2420140846E-1);
        const_f32_as_f32x16!(P6, 1.1676998740E-1);
        const_f32_as_f32x16!(P7, -1.1514610310E-1);
        const_f32_as_f32x16!(P8, 7.0376836292E-2);
        const_f32_as_f32x16!(LN2F_HI, 0.693359375);
        const_f32_as_f32x16!(LN2F_LO, -2.12194440e-4);
        const_f32_as_f32x16!(VM_SMALLEST_NORMAL, 1.17549435E-38);

        let x1 = self;
        let x = Self::fraction_2(x1);
        let e = Self::exponent(x1);
        let mask = x.simd_gt(Self::SQRT_2 * HALF);
        let x = (!mask).select(x + x, x);
        let fe = mask.select(e + Self::ONE, e);
        let x = x - Self::ONE;
        let res = polynomial_8!(x, P0, P1, P2, P3, P4, P5, P6, P7, P8);
        let x2 = x * x;
        let res = x2 * x * res;
        let res = fe.fast_mul_add(LN2F_LO, res);
        let res = res + x2.fast_mul_neg_add(HALF, x);
        let res = fe.fast_mul_add(LN2F_HI, res);
        let overflow = !self.is_finite();
        let underflow = x1.simd_lt(VM_SMALLEST_NORMAL);
        let mask = overflow | underflow;
        if !mask.any() {
            res
        } else {
            let is_zero = self.is_zero_or_subnormal();
            let res = underflow.select(Self::nan_log(), res);
            // Note: is_zero_or_subnormal() lumps subnormals (exponent==0) with zero.
            // Both get -Inf here. True subnormal inputs (~1.4e-45..1.175e-38) should
            // produce a finite negative result, but are vanishingly rare in
            // practice.
            let res = is_zero.select(-Self::infinity(), res);
            let res = overflow.select(self, res);
            // This must come *after* overflow.blend to overwrite ln(-∞) = -∞ to NaN
            let res = (!self.is_finite() & self.is_sign_negative()).select(Self::nan_log(), res);
            res
        }
    }

    #[inline]
    pub(crate) fn cbrt_backend(self) -> Self {
        let a = self.abs();
        let zero = a.simd_eq(Self::ZERO);
        if zero.all() {
            return self; // preserves -0.0
        }
        let inf = a.is_infinite();
        let nan = self.is_nan();

        let tiny = a.simd_lt(Self::from(f32::MIN_POSITIVE));
        let a_work = tiny.select(a * Self::from(16777216.0), a);

        let e = Self::exponent(a_work) + Self::ONE;
        let d = Self::fraction_2(a_work);

        // C0..C5 from SLEEF's minimax polynomial for 1/cbrt(d) on [0.5, 1.0)
        // Naoki Shibata et al., "SLEEF: A Portable Vectorized Library of C99
        // Mathematical Functions", https://sleef.org / https://github.com/shibatch/sleef
        // Licensed under the Boost Software License 1.0.
        // These are the f32-precision coefficients; our f64 variants use the f64
        // set.
        const_f32_as_f32x16!(C0, 2.2241257);
        const_f32_as_f32x16!(C1, -3.8095417);
        const_f32_as_f32x16!(C2, 5.8982625);
        const_f32_as_f32x16!(C3, -5.532182);
        const_f32_as_f32x16!(C4, 2.8208892);
        const_f32_as_f32x16!(C5, -0.60156447);
        let mut x = polynomial_5!(d, C0, C1, C2, C3, C4, C5);

        let x2 = x * x;
        let x4 = x2 * x2;
        x = x - d.fast_mul_add(x4, -x) * Self::from(1.0 / 3.0);
        // cbrt(d) = d * x² with refinement
        let mut y = (d * x) * x;
        let yx = y * x;
        let t = Self::from(2.0 / 3.0);
        y = y - t * y * (yx - Self::ONE);

        // Scale by 2^(e/3)
        let three = Self::from(3.0);
        let two = Self::from(2.0);
        let neg = e.simd_lt(Self::ZERO);
        let e_adj = neg.select(e - two, e);
        let k = (e_adj / three).trunc();
        let r = e - three * k;
        const_f32_as_f32x16!(CBRT2, 1.259921);
        const_f32_as_f32x16!(CBRT4, 1.587401);
        y = r.simd_eq(Self::ONE).select(y * CBRT2, y);
        y = r.simd_eq(two).select(y * CBRT4, y);
        y *= Self::vm_pow2n(k);
        y = tiny.select(y / Self::from(256.0_f32), y);

        let result = y.flipsign(self);
        let result = nan.select(self, result);
        let result = zero.select(self, result);
        let result = inf.select(self, result);
        result
    }

    #[inline]
    pub(crate) fn asin_backend(self) -> Self {
        // Based on the Agner Fog "vector class library":
        // https://github.com/vectorclass/version2/blob/master/vectormath_trig.h
        const_f32_as_f32x16!(P4asinf, 4.2163199048E-2);
        const_f32_as_f32x16!(P3asinf, 2.4181311049E-2);
        const_f32_as_f32x16!(P2asinf, 4.5470025998E-2);
        const_f32_as_f32x16!(P1asinf, 7.4953002686E-2);
        const_f32_as_f32x16!(P0asinf, 1.6666752422E-1);

        let xa = self.abs();
        let big = xa.simd_ge(f32x16::splat(0.5));

        let x1 = f32x16::splat(0.5) * (f32x16::ONE - xa);
        let x2 = xa * xa;
        let x3 = big.select(x1, x2);

        let xb = x1.sqrt();

        let x4 = big.select(xb, xa);

        let z = polynomial_4!(x3, P0asinf, P1asinf, P2asinf, P3asinf, P4asinf);
        let z = z.fast_mul_add(x3 * x4, x4);

        let z1 = z + z;

        // asin
        let z3 = f32x16::FRAC_PI_2 - z1;
        let asin = big.select(z3, z);
        let asin = asin.flipsign(self);

        asin
    }

    #[inline]
    pub(crate) fn acos_backend(self) -> Self {
        // Based on the Agner Fog "vector class library":
        // https://github.com/vectorclass/version2/blob/master/vectormath_trig.h
        const_f32_as_f32x16!(P4asinf, 4.2163199048E-2);
        const_f32_as_f32x16!(P3asinf, 2.4181311049E-2);
        const_f32_as_f32x16!(P2asinf, 4.5470025998E-2);
        const_f32_as_f32x16!(P1asinf, 7.4953002686E-2);
        const_f32_as_f32x16!(P0asinf, 1.6666752422E-1);

        let xa = self.abs();
        let big = xa.simd_ge(f32x16::splat(0.5));

        let x1 = f32x16::splat(0.5) * (f32x16::ONE - xa);
        let x2 = xa * xa;
        let x3 = big.select(x1, x2);

        let xb = x1.sqrt();

        let x4 = big.select(xb, xa);

        let z = polynomial_4!(x3, P0asinf, P1asinf, P2asinf, P3asinf, P4asinf);
        let z = z.fast_mul_add(x3 * x4, x4);

        let z1 = z + z;

        // acos
        let z3 = self.simd_lt(f32x16::ZERO).select(f32x16::PI - z1, z1);
        let z4 = f32x16::FRAC_PI_2 - z.flipsign(self);
        let acos = big.select(z3, z4);

        acos
    }

    #[inline]
    pub(crate) fn atan_backend(self) -> Self {
        // Based on the Agner Fog "vector class library":
        // https://github.com/vectorclass/version2/blob/master/vectormath_trig.h
        const_f32_as_f32x16!(P3atanf, 8.05374449538E-2);
        const_f32_as_f32x16!(P2atanf, -1.38776856032E-1);
        const_f32_as_f32x16!(P1atanf, 1.99777106478E-1);
        const_f32_as_f32x16!(P0atanf, -3.33329491539E-1);

        let t = self.abs();

        // small:  z = t / 1.0;
        // medium: z = (t-1.0) / (t+1.0);
        // big:    z = -1.0 / t;
        let notsmal = t.simd_ge(Self::SQRT_2 - Self::ONE);
        let notbig = t.simd_le(Self::SQRT_2 + Self::ONE);

        let mut s = notbig.select(Self::FRAC_PI_4, Self::FRAC_PI_2);
        s = notsmal & s;

        let mut a = notbig & t;
        a = notsmal.select(a - Self::ONE, a);
        let mut b = notbig & Self::ONE;
        b = notsmal.select(b + t, b);
        let z = a / b;

        let zz = z * z;

        // Taylor expansion
        let mut re = polynomial_3!(zz, P0atanf, P1atanf, P2atanf, P3atanf);
        re = re.fast_mul_add(zz * z, z) + s;

        // get sign bit
        re = (self.is_sign_negative()).select(-re, re);

        re
    }

    #[inline]
    pub(crate) fn atan2_backend(self, x: Self) -> Self {
        // Based on the Agner Fog "vector class library":
        // https://github.com/vectorclass/version2/blob/master/vectormath_trig.h
        const_f32_as_f32x16!(P3atanf, 8.05374449538E-2);
        const_f32_as_f32x16!(P2atanf, -1.38776856032E-1);
        const_f32_as_f32x16!(P1atanf, 1.99777106478E-1);
        const_f32_as_f32x16!(P0atanf, -3.33329491539E-1);

        let y = self;

        // move in first octant
        let x1 = x.abs();
        let y1 = y.abs();
        let swapxy = y1.simd_gt(x1);
        // swap x and y if y1 > x1
        let mut x2 = swapxy.select(y1, x1);
        let mut y2 = swapxy.select(x1, y1);

        // check for special case: x and y are both +/- INF
        let both_infinite = x.is_infinite() & y.is_infinite();
        if both_infinite.any() {
            let minus_one = -Self::ONE;
            x2 = both_infinite.select(x2 & minus_one, x2);
            y2 = both_infinite.select(y2 & minus_one, y2);
        }

        // x = y = 0 will produce NAN. No problem, fixed below
        let t = y2 / x2;

        // small:  z = t / 1.0;
        // medium: z = (t-1.0) / (t+1.0);
        let notsmal = t.simd_ge(Self::SQRT_2 - Self::ONE);

        let a = notsmal.select(t - Self::ONE, t);
        let b = notsmal.select(t + Self::ONE, Self::ONE);
        let s = notsmal & Self::FRAC_PI_4;
        let z = a / b;

        let zz = z * z;

        // Taylor expansion
        let mut re = polynomial_3!(zz, P0atanf, P1atanf, P2atanf, P3atanf);
        re = re.fast_mul_add(zz * z, z) + s;

        // move back in place
        re = swapxy.select(Self::FRAC_PI_2 - re, re);
        re = ((x | y).simd_eq(Self::ZERO)).select(Self::ZERO, re);
        re = (x.is_sign_negative()).select(Self::PI - re, re);

        // get sign bit
        re = (y.is_sign_negative()).select(-re, re);

        re
    }

    #[inline]
    pub(crate) fn sin_cos_backend(self) -> (Self, Self) {
        // Based on the Agner Fog "vector class library":
        // https://github.com/vectorclass/version2/blob/master/vectormath_trig.h

        const_f32_as_f32x16!(DP1F, 0.78515625_f32 * 2.0);
        const_f32_as_f32x16!(DP2F, 2.4187564849853515625E-4_f32 * 2.0);
        const_f32_as_f32x16!(DP3F, 3.77489497744594108E-8_f32 * 2.0);

        const_f32_as_f32x16!(P0sinf, -1.6666654611E-1);
        const_f32_as_f32x16!(P1sinf, 8.3321608736E-3);
        const_f32_as_f32x16!(P2sinf, -1.9515295891E-4);

        const_f32_as_f32x16!(P0cosf, 4.166664568298827E-2);
        const_f32_as_f32x16!(P1cosf, -1.388731625493765E-3);
        const_f32_as_f32x16!(P2cosf, 2.443315711809948E-5);

        const_f32_as_f32x16!(TWO_OVER_PI, 2.0 / core::f32::consts::PI);

        let xa = self.abs();

        // Find quadrant
        let y = (xa * TWO_OVER_PI).round_ties_even();
        let q: i32x16 = y.round_int();

        let x = y.fast_mul_neg_add(DP3F, y.fast_mul_neg_add(DP2F, y.fast_mul_neg_add(DP1F, xa)));

        let x2 = x * x;
        let mut s = polynomial_2!(x2, P0sinf, P1sinf, P2sinf) * (x * x2) + x;
        let mut c = polynomial_2!(x2, P0cosf, P1cosf, P2cosf) * (x2 * x2)
            + f32x16::from(0.5).fast_mul_neg_add(x2, f32x16::from(1.0));

        let swap = !(q & i32x16::from(1)).simd_eq(i32x16::from(0));

        let mut overflow: f32x16 = cast(q.simd_gt(i32x16::from(0x2000000)));
        overflow &= xa.is_finite();
        s = overflow.select(f32x16::from(0.0), s);
        c = overflow.select(f32x16::from(1.0), c);

        // calc sin
        let mut sin1 = cast::<_, f32x16>(swap).select(c, s);
        let sign_sin: i32x16 = (q << 30) ^ cast::<_, i32x16>(self);
        sin1 = sin1.flipsign(cast(sign_sin));

        // calc cos
        let mut cos1 = cast::<_, f32x16>(swap).select(s, c);
        let sign_cos: i32x16 = ((q + i32x16::from(1)) & i32x16::from(2)) << 30;
        cos1 ^= cast::<_, f32x16>(sign_cos);

        // IEEE 754: sin/cos(±∞) = NaN, sin/cos(NaN) = NaN
        let finite = self.is_finite();
        let nan = Self::splat(f32::NAN);
        let sin_final = finite.select(sin1, nan);
        let cos_final = finite.select(cos1, nan);

        (sin_final, cos_final)
    }

    #[inline]
    pub(crate) fn asin_acos_backend(self) -> (Self, Self) {
        // Based on the Agner Fog "vector class library":
        // https://github.com/vectorclass/version2/blob/master/vectormath_trig.h
        const_f32_as_f32x16!(P4asinf, 4.2163199048E-2);
        const_f32_as_f32x16!(P3asinf, 2.4181311049E-2);
        const_f32_as_f32x16!(P2asinf, 4.5470025998E-2);
        const_f32_as_f32x16!(P1asinf, 7.4953002686E-2);
        const_f32_as_f32x16!(P0asinf, 1.6666752422E-1);

        let xa = self.abs();
        let big = xa.simd_ge(f32x16::splat(0.5));

        let x1 = f32x16::splat(0.5) * (f32x16::ONE - xa);
        let x2 = xa * xa;
        let x3 = big.select(x1, x2);

        let xb = x1.sqrt();

        let x4 = big.select(xb, xa);

        let z = polynomial_4!(x3, P0asinf, P1asinf, P2asinf, P3asinf, P4asinf);
        let z = z.fast_mul_add(x3 * x4, x4);

        let z1 = z + z;

        // acos
        let z3 = self.simd_lt(f32x16::ZERO).select(f32x16::PI - z1, z1);
        let z4 = f32x16::FRAC_PI_2 - z.flipsign(self);
        let acos = big.select(z3, z4);

        // asin
        let z3 = f32x16::FRAC_PI_2 - z1;
        let asin = big.select(z3, z);
        let asin = asin.flipsign(self);

        (asin, acos)
    }

    #[inline]
    pub(crate) fn exp_m1_backend(self) -> Self {
        // x < -17.329: e^x < 2⁻²⁵, exp_m1(x) = -1.0 exactly (mantissa exhaustion)
        // IEEE simd_lt returns false for NaN, so NaN lanes can't reach here.
        // -inf is < -17.329, and exp_m1(-inf) = -1.0, also correct.
        if self.simd_lt(f32x16::from(-17.329)).all() {
            return f32x16::from(-1.0);
        }
        const_f32_as_f32x16!(P0, 1.0 / 2.0);
        const_f32_as_f32x16!(P1, 1.0 / 6.0);
        const_f32_as_f32x16!(P2, 1.0 / 24.0);
        const_f32_as_f32x16!(P3, 1.0 / 120.0);
        const_f32_as_f32x16!(P4, 1.0 / 720.0);
        const_f32_as_f32x16!(P5, 1.0 / 5040.0);
        // LN2D_HI/LO: double-double decomposition of ln(2) for exp range reduction,
        // following the approach from fdlibm's e_exp.c (Sun Microsystems,
        // https://www.netlib.org/fdlibm/). The f32 split uses f32-precision constants
        // (0.693359375, -2.12194440e-4) summing to ln(2) with single-precision
        // accuracy; the f64 variants use a full f64 double-double
        // decomposition.
        const_f32_as_f32x16!(LN2D_HI, 0.693359375);
        const_f32_as_f32x16!(LN2D_LO, -2.12194440e-4);
        // max_x = ln(f32::MAX) ≈ 88.7229, max_r = 127 (IEEE max normal exponent)
        // min_x = -149.5 ln(2) ≈ -103.63: min r for vm_pow2n subnormal
        let max_x = f32x16::from(88.723);
        let min_x = f32x16::from(-103.63);
        let max_r = f32x16::from(127.0);
        let r = (self * Self::LOG2_E).round_ties_even();
        let big = r.simd_gt(max_r);
        let r_safe = big.select(max_r, r);
        let excess = r - max_r;
        let excess = big.select(excess, Self::ZERO);
        let scale = Self::vm_pow2n(excess);
        let x = r.fast_mul_neg_add(LN2D_HI, self);
        let x = r.fast_mul_neg_add(LN2D_LO, x);
        let z = polynomial_5!(x, P0, P1, P2, P3, P4, P5);
        let x2 = x * x;
        let z = z.fast_mul_add(x2, x);
        let n2 = Self::vm_pow2n(r_safe);
        let exp_val = (z + Self::ONE) * scale * n2;
        let r_is_zero = r.simd_eq(Self::ZERO);
        let z = r_is_zero.select(z, exp_val - Self::ONE);
        let nan_mask = self.is_nan();
        let finite = self.is_finite();
        let mut result = nan_mask.select(Self::nan_pow(), z);
        let pos_overflow = self.simd_gt(max_x) & finite;
        result = pos_overflow.select(Self::infinity(), result);
        let neg_underflow = self.simd_lt(min_x) & finite;
        result = neg_underflow.select(-Self::ONE, result);
        let pos_inf = !finite & !self.is_sign_negative() & !nan_mask;
        result = pos_inf.select(Self::infinity(), result);
        let neg_inf = !finite & self.is_sign_negative() & !nan_mask;
        result = neg_inf.select(-Self::ONE, result);
        let is_zero = self.simd_eq(Self::ZERO);
        result = is_zero.select(self, result);
        result
    }

    #[inline]
    pub(crate) fn ln_1p_backend(self) -> Self {
        // Based on the identity ln(1+x) = x·ln(1+x)/((1+x)-1), i.e. x·ln(u)/(u-1)
        // where u = 1+x. From MUSL libc (Rich Felker et al., https://musl.libc.org) src/math/log1pf.c
        // and fdlibm (Sun Microsystems, https://www.netlib.org/fdlibm/) s_log1p.c.
        // When 1+x rounds to 1 exactly (subnormal x), return x directly.
        // When 1+x overflows (+inf), return ln(u) without correction.
        // Mathematically exact: compensates for the rounding loss in 1+x without
        // needing a series threshold.
        let u = self + Self::ONE;
        let eq = u.simd_eq(Self::ONE);
        let ln_u = Self::ln(u);
        let correction = self * (ln_u / (u - Self::ONE));
        let result = eq.select(self, correction);
        let over = u.is_infinite();
        over.select(ln_u, result)
    }

    #[inline]
    pub(crate) fn sinh_backend(self) -> Self {
        const_f32_as_f32x16!(P0, 1.0);
        const_f32_as_f32x16!(P1, 1.0 / 6.0);
        const_f32_as_f32x16!(P2, 1.0 / 120.0);
        const_f32_as_f32x16!(P3, 1.0 / 5040.0);
        let a = self.abs();
        // |x| < 0.5: Taylor poly; last truncation term < 1 ULP at x=0.5 for both types
        let small = a.simd_lt(f32x16::from(0.5));
        let t = a * a;
        let poly = a * polynomial_3!(t, P0, P1, P2, P3);
        let exp_based = {
            let e = a.exp();
            (e - Self::ONE / e) * Self::HALF
        };
        let result = small.select(poly, exp_based);
        result.flipsign(self)
    }

    #[inline]
    pub(crate) fn cosh_backend(self) -> Self {
        const_f32_as_f32x16!(P0, 1.0);
        const_f32_as_f32x16!(P1, 1.0 / 2.0);
        const_f32_as_f32x16!(P2, 1.0 / 24.0);
        const_f32_as_f32x16!(P3, 1.0 / 720.0);
        let a = self.abs();
        // |x| < 0.5: Taylor poly; last truncation term < 1 ULP at x=0.5 for both types
        let small = a.simd_lt(f32x16::from(0.5));
        let t = a * a;
        let poly = polynomial_3!(t, P0, P1, P2, P3);
        let exp_based = {
            let e = a.exp();
            (e + Self::ONE / e) * Self::HALF
        };
        small.select(poly, exp_based)
    }

    #[inline]
    pub(crate) fn tanh_backend(self) -> Self {
        // |x| < 2e-4: tanh(x) ≈ x, error x³/3 < 16·ULP(x)
        // bound: x² < 48·2⁻²³ → x < 2.39e-3; 2e-4 has 10× margin
        // |x| > 9.011: tanh(x) = ±1 to f32 precision (e⁻²ˣ < 2⁻²⁴)
        let a = self.abs();
        let large = a.simd_gt(f32x16::from(9.011));
        if large.all() {
            return Self::ONE.flipsign(self);
        }
        let small = a.simd_lt(f32x16::from(2e-4));
        let exp_based = {
            let t = (Self::from(-2.0) * a).exp_m1();
            let pos = -t / (t + Self::from(2.0));
            pos.flipsign(self)
        };
        let result = small.select(self, exp_based);
        large.select(Self::ONE.flipsign(self), result)
    }
}
}

safe_target_feature! {
// Internal helpers and backend of inconsistent functions.
impl f32x16 {
    #[inline]
    fn vm_pow2n(self) -> Self {
        const_f32_as_f32x16!(pow2_23, 8388608.0);
        const_f32_as_f32x16!(bias, 127.0);
        let a = self + (bias + pow2_23);
        let c = cast::<_, i32x16>(a) << 23;
        let std_result = cast::<_, f32x16>(c);

        let min_exp = f32x16::from(-126.0);
        let is_sub = self.simd_lt(min_exp);
        if is_sub.any() {
            let valid = self.simd_ge(f32x16::from(-149.0));
            let shift_f = self + f32x16::from(149.0);
            let mut shift_i = shift_f.trunc_int();
            shift_i = cast::<_, i32x16>(valid).select(shift_i, i32x16::ZERO);
            let mantissa = i32x16::ONE << shift_i;
            let sub_result = cast::<_, f32x16>(mantissa);
            let sub_result = valid.select(sub_result, f32x16::ZERO);
            is_sub.select(sub_result, std_result)
        } else {
            std_result
        }
    }

    #[inline]
    fn exponent(self) -> Self {
        const_f32_as_f32x16!(pow2_23, 8388608.0);
        const_f32_as_f32x16!(bias, 127.0);
        let a = cast::<_, u32x16>(self);
        let b = a >> 23;
        let c = b | cast::<_, u32x16>(pow2_23);
        let d = cast::<_, f32x16>(c);
        let e = d - (pow2_23 + bias);
        e
    }

    #[inline]
    fn fraction_2(self) -> Self {
        let t1 = cast::<_, u32x16>(self);
        let t2 = cast::<_, u32x16>((t1 & u32x16::from(0x007FFFFF)) | u32x16::from(0x3F000000));
        cast::<_, f32x16>(t2)
    }

    #[inline]
    fn is_zero_or_subnormal(self) -> Self {
        let t = cast::<_, i32x16>(self);
        let t = t & i32x16::splat(0x7F800000);
        let mask = t.simd_eq(i32x16::splat(0));
        cast::<_, f32x16>(mask)
    }

    #[inline]
    fn infinity() -> Self {
        cast::<_, f32x16>(i32x16::splat(0x7F800000))
    }

    #[inline]
    fn nan_log() -> Self {
        cast::<_, f32x16>(i32x16::splat(0x7FC00000 | 0x101 & 0x003FFFFF))
    }

    #[inline]
    fn nan_pow() -> Self {
        cast::<_, f32x16>(i32x16::splat(0x7FC00000 | 0x101 & 0x003FFFFF))
    }

    #[inline]
    pub(crate) fn from_i32x16_backend(v: i32x16) -> Self {
        pick! {
          if #[cfg(target_feature="avx512f")] {
            Self(_mm512_cvtepi32_ps(v.0))
          } else {
            Self(Inner {
              lo: f32x8::from_i32x8(v.0.lo),
              hi: f32x8::from_i32x8(v.0.hi),
            })
          }
        }
    }
}
}
