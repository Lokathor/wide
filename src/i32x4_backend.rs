use crate::backend_imports::*;

#[cfg(not(any(
    target_feature = "sse2",
    target_feature = "simd128",
    all(target_feature = "neon", target_arch = "aarch64")
)))]
#[repr(C, align(16))]
#[derive(Clone, Copy)]
pub(crate) struct Inner {
    pub arr: [i32; 4],
}

safe_target_feature! {
impl SimdBackend for i32x4 {
    pick! {
      if #[cfg(target_feature="sse2")] {
        type Inner = __m128i;
      } else if #[cfg(target_feature="simd128")] {
        type Inner = v128;
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        type Inner = int32x4_t;
      } else {
        type Inner = Inner;
      }
    }

    #[inline]
    pub(crate) fn simd_lt_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse2")] {
            Self(_mm_cmplt_epi32(self.0, rhs.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(i32x4_lt(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vreinterpretq_s32_u32(vcltq_s32(self.0, rhs.0)))
          } else {
            Self::from_array([
              if self.0.arr[0] < rhs.0.arr[0] { -1 } else { 0 },
              if self.0.arr[1] < rhs.0.arr[1] { -1 } else { 0 },
              if self.0.arr[2] < rhs.0.arr[2] { -1 } else { 0 },
              if self.0.arr[3] < rhs.0.arr[3] { -1 } else { 0 },
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn simd_gt_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse2")] {
            Self(_mm_cmpgt_epi32(self.0, rhs.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(i32x4_gt(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vreinterpretq_s32_u32(vcgtq_s32(self.0, rhs.0)))
          } else {
            Self::from_array([
              if self.0.arr[0] > rhs.0.arr[0] { -1 } else { 0 },
              if self.0.arr[1] > rhs.0.arr[1] { -1 } else { 0 },
              if self.0.arr[2] > rhs.0.arr[2] { -1 } else { 0 },
              if self.0.arr[3] > rhs.0.arr[3] { -1 } else { 0 },
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn simd_le_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse2")] {
            !self.simd_gt(rhs)
          } else if #[cfg(target_feature="simd128")] {
            Self(i32x4_le(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            !self.simd_gt(rhs)
          } else {
            Self::from_array([
              if self.0.arr[0] <= rhs.0.arr[0] { -1 } else { 0 },
              if self.0.arr[1] <= rhs.0.arr[1] { -1 } else { 0 },
              if self.0.arr[2] <= rhs.0.arr[2] { -1 } else { 0 },
              if self.0.arr[3] <= rhs.0.arr[3] { -1 } else { 0 },
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn simd_ge_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse2")] {
            !self.simd_lt(rhs)
          } else if #[cfg(target_feature="simd128")] {
            Self(i32x4_ge(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            !self.simd_lt(rhs)
          } else {
            Self::from_array([
              if self.0.arr[0] >= rhs.0.arr[0] { -1 } else { 0 },
              if self.0.arr[1] >= rhs.0.arr[1] { -1 } else { 0 },
              if self.0.arr[2] >= rhs.0.arr[2] { -1 } else { 0 },
              if self.0.arr[3] >= rhs.0.arr[3] { -1 } else { 0 },
            ])
          }
        }
    }
}
}

safe_target_feature! {
// Backend of SIMD integer functions.
impl i32x4 {
    #[inline]
    pub(crate) fn shr_backend(self, rhs: u32x4) -> Self {
        pick! {
          if #[cfg(target_feature="avx2")] {
            // mask the shift count to 31 to have same behavior on all platforms
            let shift_by = _mm_and_si128(rhs.0, _mm_set1_epi32(31));
            Self(_mm_srav_epi32(self.0, shift_by))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            // mask the shift count to 31 to have same behavior on all platforms
            // no right shift, have to pass negative value to left shift on neon
            let shift_by = vnegq_s32(vreinterpretq_s32_u32(vandq_u32(rhs.0, vmovq_n_u32(31))));
            Self(vshlq_s32(self.0, shift_by))
          } else {
            let arr: [i32; 4] = cast(self);
            let rhs: [u32; 4] = cast(rhs);
            cast([
              arr[0].wrapping_shr(rhs[0]),
              arr[1].wrapping_shr(rhs[1]),
              arr[2].wrapping_shr(rhs[2]),
              arr[3].wrapping_shr(rhs[3]),
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn shr_scalar_backend(self, rhs: u32) -> Self {
        pick! {
          if #[cfg(target_feature="sse2")] {
            // Use `rhs % 32` to perform wrapping shift and not unbounded shift.
            let shift = cast([rhs as u64 & 31, 0]);
            Self(_mm_sra_epi32(self.0, shift))
          } else if #[cfg(target_feature="simd128")] {
            Self(i32x4_shr(self.0, rhs))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            // Use `rhs % 32` to perform wrapping shift and not unbounded shift.
            Self(vshlq_s32(self.0, vmovq_n_s32( -(rhs as i32 & 31))))
          } else {
            Self::from_array([
              self.0.arr[0].wrapping_shr(rhs),
              self.0.arr[1].wrapping_shr(rhs),
              self.0.arr[2].wrapping_shr(rhs),
              self.0.arr[3].wrapping_shr(rhs),
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn max_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse4.1")] {
            Self(_mm_max_epi32(self.0, rhs.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(i32x4_max(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vmaxq_s32(self.0, rhs.0))
          } else {
            self.simd_lt(rhs).select(rhs, self)
          }
        }
    }

    #[inline]
    pub(crate) fn min_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse4.1")] {
            Self(_mm_min_epi32(self.0, rhs.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(i32x4_min(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vminq_s32(self.0, rhs.0))
          } else {
            self.simd_lt(rhs).select(self, rhs)
          }
        }
    }

    #[inline]
    pub(crate) fn reduce_max_backend(self) -> i32 {
        let arr: [i32; 4] = cast(self);
        arr[0].max(arr[1]).max(arr[2].max(arr[3]))
    }

    #[inline]
    pub(crate) fn reduce_min_backend(self) -> i32 {
        let arr: [i32; 4] = cast(self);
        arr[0].min(arr[1]).min(arr[2].min(arr[3]))
    }

    #[inline]
    pub(crate) fn unbounded_shr_backend(self, rhs: u32x4) -> Self {
        pick! {
          if #[cfg(target_feature="avx2")] {
            Self(_mm_srav_epi32(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            // Negate `rhs` because there is no direct shift-right intrinsic, and
            // restrict it to prevent overflow.
            Self(vshlq_s32(self.0, vnegq_s32(vreinterpretq_s32_u32(rhs.min(u32x4::splat(32)).0))))
          } else {
            let self_array = self.to_array();
            let rhs_array = rhs.to_array();

            Self::from_array([
              self_array[0].unbounded_shr(rhs_array[0]),
              self_array[1].unbounded_shr(rhs_array[1]),
              self_array[2].unbounded_shr(rhs_array[2]),
              self_array[3].unbounded_shr(rhs_array[3]),
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn unbounded_shr_scalar_backend(self, rhs: u32) -> Self {
        pick! {
          if #[cfg(target_feature="sse2")] {
            Self(_mm_sra_epi32(self.0, cast([rhs as u64, 0])))
          } else if #[cfg(target_feature="simd128")] {
            if rhs < 32 { Self(i32x4_shr(self.0, rhs)) } else { self.is_negative() }
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            // Negate `rhs` because there is no direct shift-right intrinsic, and
            // restrict it to prevent overflow.
            Self(vshlq_s32(self.0, vmovq_n_s32(-rhs.min(32).cast_signed())))
          } else {
            Self::from_array([
                self.0.arr[0].unbounded_shr(rhs),
                self.0.arr[1].unbounded_shr(rhs),
                self.0.arr[2].unbounded_shr(rhs),
                self.0.arr[3].unbounded_shr(rhs),
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn saturating_add_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(any(target_feature="sse2", target_feature="simd128"))] {
            let result = self + rhs;
            let overflow = (!(self ^ rhs) & (self ^ result)).is_negative();
            let negative = self.is_negative();

            // If overflow occurs return `MAX` if positive or `MIN` if negative.
            overflow.select(Self::MAX ^ negative, result)
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vqaddq_s32(self.0, rhs.0))
          } else {
            Self::from_array([
                self.0.arr[0].saturating_add(rhs.0.arr[0]),
                self.0.arr[1].saturating_add(rhs.0.arr[1]),
                self.0.arr[2].saturating_add(rhs.0.arr[2]),
                self.0.arr[3].saturating_add(rhs.0.arr[3]),
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn saturating_sub_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(any(target_feature="sse2", target_feature="simd128"))] {
            let result = self - rhs;
            let overflow = ((self ^ rhs) & (self ^ result)).is_negative();
            let negative = self.is_negative();

            // If overflow occurs return `MAX` if positive or `MIN` if negative.
            overflow.select(Self::MAX ^ negative, result)
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vqsubq_s32(self.0, rhs.0))
          } else {
            Self::from_array([
                self.0.arr[0].saturating_sub(rhs.0.arr[0]),
                self.0.arr[1].saturating_sub(rhs.0.arr[1]),
                self.0.arr[2].saturating_sub(rhs.0.arr[2]),
                self.0.arr[3].saturating_sub(rhs.0.arr[3]),
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn overflowing_mul_backend(self, rhs: Self) -> (Self, Self) {
        let (low, high) = self.mul_keep_low_high(rhs);
        let low = cast::<u32x4, i32x4>(low);

        let overflow = high.simd_ne(low.is_negative());
        (low, overflow)
    }

    #[inline]
    pub(crate) fn widening_mul_backend(self, rhs: Self) -> i64x4 {
        pick! {
          if #[cfg(target_feature="avx2")] {
            let a = _mm256_cvtepi32_epi64(self.0);
            let b = _mm256_cvtepi32_epi64(rhs.0);
            cast(_mm256_mul_epi32(a, b))
          } else if #[cfg(target_feature="sse4.1")] {
              let evenp = _mm_mul_epi32(self.0, rhs.0);

              let oddp = _mm_mul_epi32(
                _mm_srli_epi64::<32>(self.0),
                _mm_srli_epi64::<32>(rhs.0));

              i64x4(crate::i64x4_backend::Inner {
                lo: i64x2(_mm_unpacklo_epi64(evenp, oddp)),
                hi: i64x2(_mm_unpackhi_epi64(evenp, oddp))
              })
          } else if #[cfg(target_feature="simd128")] {
              i64x4(crate::i64x4_backend::Inner {
                lo: i64x2(i64x2_extmul_low_i32x4(self.0, rhs.0)),
                hi: i64x2(i64x2_extmul_high_i32x4(self.0, rhs.0)),
              })
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))] {
            i64x4(crate::i64x4_backend::Inner {
              lo: i64x2(vmull_s32(vget_low_s32(self.0), vget_low_s32(rhs.0))),
              hi: i64x2(vmull_s32(vget_high_s32(self.0), vget_high_s32(rhs.0))),
            })
          } else {
            let a = self.as_array();
            let b = rhs.as_array();

            cast([
              i64::from(a[0]) * i64::from(b[0]),
              i64::from(a[1]) * i64::from(b[1]),
              i64::from(a[2]) * i64::from(b[2]),
              i64::from(a[3]) * i64::from(b[3]),
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn mul_keep_low_high_backend(self, rhs: Self) -> (u32x4, i32x4) {
        pick! {
          if #[cfg(target_feature="sse4.1")] {
            let even_wide_mul = _mm_mul_epi32(self.0, rhs.0);
            let odd_wide_mul = _mm_mul_epi32(
              _mm_shuffle_epi32::<0b_00_11_00_01>(self.0),
              _mm_shuffle_epi32::<0b_00_11_00_01>(rhs.0),
            );
            let ll_hh_1 = _mm_unpacklo_epi32(even_wide_mul, odd_wide_mul);
            let ll_hh_2 = _mm_unpackhi_epi32(even_wide_mul, odd_wide_mul);

            (
              u32x4(_mm_unpacklo_epi64(ll_hh_1, ll_hh_2)),
              i32x4(_mm_unpackhi_epi64(ll_hh_1, ll_hh_2)),
            )
          } else if #[cfg(target_feature="simd128")] {
            let low_wide_mul = i64x2_extmul_low_i32x4(self.0, rhs.0);
            let high_wide_mul = i64x2_extmul_high_i32x4(self.0, rhs.0);

            (
              u32x4(i32x4_shuffle::<0, 2, 4, 6>(low_wide_mul, high_wide_mul)),
              i32x4(i32x4_shuffle::<1, 3, 5, 7>(low_wide_mul, high_wide_mul)),
            )
          } else if #[cfg(all(target_feature="neon", target_arch="aarch64"))] {
            let low_wide_mul = vreinterpretq_s32_s64(
              vmull_s32(vget_low_s32(self.0), vget_low_s32(rhs.0)),
            );
            let high_wide_mul = vreinterpretq_s32_s64(
              vmull_s32(vget_high_s32(self.0), vget_high_s32(rhs.0)),
            );
            let low_high = vuzpq_s32(low_wide_mul, high_wide_mul);

            (
              u32x4(vreinterpretq_u32_s32(low_high.0)),
              i32x4(low_high.1),
            )
          } else {
            // TODO(perf): This implementation looks quite bad. Is there a better
            // one?

            let self_array = self.to_array();
            let rhs_array = rhs.to_array();

            let widening_mul = [
              (self_array[0] as i64).wrapping_mul(rhs_array[0] as i64),
              (self_array[1] as i64).wrapping_mul(rhs_array[1] as i64),
              (self_array[2] as i64).wrapping_mul(rhs_array[2] as i64),
              (self_array[3] as i64).wrapping_mul(rhs_array[3] as i64),
            ];

            (
              u32x4::from_array([
                widening_mul[0] as u32,
                widening_mul[1] as u32,
                widening_mul[2] as u32,
                widening_mul[3] as u32,
              ]),
              i32x4::from_array([
                (widening_mul[0] >> 32) as i32,
                (widening_mul[1] >> 32) as i32,
                (widening_mul[2] >> 32) as i32,
                (widening_mul[3] >> 32) as i32,
              ]),
            )
          }
        }
    }

    #[inline]
    pub(crate) fn mul_keep_high_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse4.1")] {
            let even_wide_mul = _mm_mul_epi32(self.0, rhs.0);
            let odd_wide_mul = _mm_mul_epi32(
              _mm_shuffle_epi32::<0b_00_11_00_01>(self.0),
              _mm_shuffle_epi32::<0b_00_11_00_01>(rhs.0),
            );
            let ll_hh_1 = _mm_unpacklo_epi32(even_wide_mul, odd_wide_mul);
            let ll_hh_2 = _mm_unpackhi_epi32(even_wide_mul, odd_wide_mul);

            Self(_mm_unpackhi_epi64(ll_hh_1, ll_hh_2))
          } else if #[cfg(target_feature="simd128")] {
            let low_wide_mul = i64x2_extmul_low_i32x4(self.0, rhs.0);
            let high_wide_mul = i64x2_extmul_high_i32x4(self.0, rhs.0);

            Self(i32x4_shuffle::<1, 3, 5, 7>(low_wide_mul, high_wide_mul))
          } else if #[cfg(all(target_feature="neon", target_arch="aarch64"))] {
            let low_wide_mul = vreinterpretq_s32_s64(
              vmull_s32(vget_low_s32(self.0), vget_low_s32(rhs.0)),
            );
            let high_wide_mul = vreinterpretq_s32_s64(
              vmull_s32(vget_high_s32(self.0), vget_high_s32(rhs.0)),
            );

            Self(vuzpq_s32(low_wide_mul, high_wide_mul).1)
          } else {
            let self_array = self.to_array();
            let rhs_array = rhs.to_array();

            Self::from_array([
              ((self_array[0] as i64).wrapping_mul(rhs_array[0] as i64) >> 32) as i32,
              ((self_array[1] as i64).wrapping_mul(rhs_array[1] as i64) >> 32) as i32,
              ((self_array[2] as i64).wrapping_mul(rhs_array[2] as i64) >> 32) as i32,
              ((self_array[3] as i64).wrapping_mul(rhs_array[3] as i64) >> 32) as i32,
            ])
          }
        }
    }
}
}

safe_target_feature! {
// Backend of SIMD signed integer functions.
impl i32x4 {
    #[inline]
    pub(crate) fn abs_backend(self) -> Self {
        pick! {
          if #[cfg(target_feature="ssse3")] {
            Self(_mm_abs_epi32(self.0))
          } else if #[cfg(target_feature="sse2")] {
            let is_negative = self.is_negative();
            // This trick efficiently performs `is_negative.select(!self + 1, self)`
            (self ^ is_negative) - is_negative
          } else if #[cfg(target_feature="simd128")] {
            Self(i32x4_abs(self.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vabsq_s32(self.0))
          } else {
            let arr: [i32; 4] = cast(self);
            cast([
              arr[0].wrapping_abs(),
              arr[1].wrapping_abs(),
              arr[2].wrapping_abs(),
              arr[3].wrapping_abs(),
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn is_positive_backend(self) -> Self {
        pick! {
          if #[cfg(all(target_feature="neon", target_arch="aarch64"))] {
            Self(vreinterpretq_s32_u32(vcgtzq_s32(self.0)))
          } else {
            self.simd_gt(Self::ZERO)
          }
        }
    }

    #[inline]
    pub(crate) fn is_negative_backend(self) -> Self {
        pick! {
          if #[cfg(all(target_feature="neon", target_arch="aarch64"))] {
            Self(vreinterpretq_s32_u32(vcltzq_s32(self.0)))
          } else if #[cfg(target_feature = "sse2")] {
            Self(_mm_srai_epi32::<31>(self.0))
          } else {
            self.simd_lt(Self::ZERO)
          }
        }
    }

    #[inline]
    pub(crate) fn saturating_neg_backend(self) -> Self {
        pick! {
            if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
                Self(vqnegq_s32(self.0))
            } else {
                // Efficient computation of
                // `-self + self.simd_eq(Self::MIN).select(-Self::ONE, Self::ZERO)`
                self.simd_eq(Self::MIN) - self
            }
        }
    }

    #[inline]
    pub(crate) fn saturating_abs_backend(self) -> Self {
        pick! {
            if #[cfg(all(target_feature = "neon", target_arch = "aarch64"))] {
                Self(vqabsq_s32(self.0))
            } else {
                // If `self == MIN`, flip all bits to change the result from
                // `MIN` to `MAX`
                self.abs() ^ self.simd_eq(Self::MIN)
            }
        }
    }

    #[inline]
    pub(crate) fn saturating_add_unsigned_backend(self, rhs: u32x4) -> Self {
        pick! {
            if #[cfg(all(target_feature = "neon", target_arch = "aarch64"))] {
                Self(vuqaddq_s32(self.0, rhs.0))
            } else {
                let (result, overflow) = self.overflowing_add_unsigned(rhs);
                overflow.select(Self::MAX, result)
            }
        }
    }
}
}

safe_target_feature! {
// Internal helpers and backend of inconsistent functions.
impl i32x4 {
    #[inline]
    pub(crate) fn round_float_backend(self) -> f32x4 {
        pick! {
          if #[cfg(target_feature="sse2")] {
            cast(_mm_cvtepi32_ps(self.0))
          } else if #[cfg(target_feature="simd128")] {
            cast(Self(f32x4_convert_i32x4(self.0)))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            cast(Self(vreinterpretq_s32_f32(vcvtq_f32_s32(self.0))))
          } else {
            let arr: [i32; 4] = cast(self);
            cast([
              arr[0] as f32,
              arr[1] as f32,
              arr[2] as f32,
              arr[3] as f32,
            ])
          }
        }
    }
}
}
