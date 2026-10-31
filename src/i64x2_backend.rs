use crate::backend_imports::*;

#[cfg(not(any(
    target_feature = "sse2",
    target_feature = "simd128",
    all(target_feature = "neon", target_arch = "aarch64")
)))]
#[repr(C, align(16))]
#[derive(Clone, Copy)]
pub(crate) struct Inner {
    pub arr: [i64; 2],
}

safe_target_feature! {
impl SimdBackend for i64x2 {
    pick! {
      if #[cfg(target_feature="sse2")] {
        type Inner = __m128i;
      } else if #[cfg(target_feature="simd128")] {
        type Inner = v128;
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        type Inner = int64x2_t;
      } else {
        type Inner = Inner;
      }
    }

    #[inline]
    pub(crate) fn simd_lt_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse4.2")] {
            // only has gt, so flip arguments around to get lt
            Self(_mm_cmpgt_epi64( rhs.0, self.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(i64x2_lt(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vreinterpretq_s64_u64(vcltq_s64(self.0, rhs.0)))
          } else {
            let s: [i64;2] = cast(self);
            let r: [i64;2] = cast(rhs);
            cast([
              if s[0] < r[0] { -1_i64 } else { 0 },
              if s[1] < r[1] { -1_i64 } else { 0 },
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn simd_gt_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse4.2")] {
            Self(_mm_cmpgt_epi64(self.0, rhs.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(i64x2_gt(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vreinterpretq_s64_u64(vcgtq_s64(self.0, rhs.0)))
          } else {
            let s: [i64;2] = cast(self);
            let r: [i64;2] = cast(rhs);
            cast([
              if s[0] > r[0] { -1_i64 } else { 0 },
              if s[1] > r[1] { -1_i64 } else { 0 },
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn simd_le_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse4.1")] {
            !self.simd_gt(rhs)
          } else if #[cfg(target_feature="simd128")] {
            Self(i64x2_le(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            !self.simd_gt(rhs)
          } else {
            let s: [i64;2] = cast(self);
            let r: [i64;2] = cast(rhs);
            cast([
              if s[0] <= r[0] { -1_i64 } else { 0 },
              if s[1] <= r[1] { -1_i64 } else { 0 },
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn simd_ge_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse4.1")] {
            !self.simd_lt(rhs)
          } else if #[cfg(target_feature="simd128")] {
            Self(i64x2_ge(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            !self.simd_lt(rhs)
          } else {
            let s: [i64;2] = cast(self);
            let r: [i64;2] = cast(rhs);
            cast([
              if s[0] >= r[0] { -1_i64 } else { 0 },
              if s[1] >= r[1] { -1_i64 } else { 0 },
            ])
          }
        }
    }
}
}

safe_target_feature! {
// Backend of SIMD integer functions.
impl i64x2 {
    #[inline]
    pub(crate) fn shr_backend(self, rhs: u64x2) -> Self {
        pick! {
          if #[cfg(all(target_feature="neon", target_arch="aarch64"))] {
            // mask the shift count to 63 to have same behavior on all platforms
            // no right shift, have to pass negative value to left shift on neon
            let shift_by = vnegq_s64(vreinterpretq_s64_u64(vandq_u64(rhs.0, vmovq_n_u64(63))));
            Self(vshlq_s64(self.0, shift_by))
          } else {
            let arr: [i64; 2] = cast(self);
            let rhs: [u64; 2] = cast(rhs);
            cast([
              arr[0].wrapping_shr(rhs[0] as u32),
              arr[1].wrapping_shr(rhs[1] as u32),
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn shr_scalar_backend(self, rhs: u32) -> Self {
        pick! {
          if #[cfg(target_feature="simd128")] {
            Self(i64x2_shr(self.0, rhs))
          } else {
            let arr: [i64; 2] = cast(self);
            cast([
              arr[0].wrapping_shr(rhs),
              arr[1].wrapping_shr(rhs),
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn max_backend(self, rhs: Self) -> Self {
        self.simd_gt(rhs).select(self, rhs)
    }

    #[inline]
    pub(crate) fn min_backend(self, rhs: Self) -> Self {
        self.simd_lt(rhs).select(self, rhs)
    }

    #[inline]
    pub(crate) fn reduce_max_backend(self) -> i64 {
        pick! {
          if #[cfg(any(target_feature="sse2", target_feature="simd128"))] {
            let array: [i64; 2] = cast(self);
            array[0].max(array[1])
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            vgetq_lane_s64(self.0, 0).max(vgetq_lane_s64(self.0, 1))
          } else {
            self.0.arr[0].max(self.0.arr[1])
          }
        }
    }

    #[inline]
    pub(crate) fn reduce_min_backend(self) -> i64 {
        pick! {
          if #[cfg(any(target_feature="sse2", target_feature="simd128"))] {
            let array: [i64; 2] = cast(self);
            array[0].min(array[1])
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            vgetq_lane_s64(self.0, 0).min(vgetq_lane_s64(self.0, 1))
          } else {
            self.0.arr[0].min(self.0.arr[1])
          }
        }
    }

    #[inline]
    pub(crate) fn unbounded_shr_backend(self, rhs: u64x2) -> Self {
        pick! {
          if #[cfg(all(target_feature="neon", target_arch="aarch64"))] {
            // Negate `rhs` because there is no direct shift-right intrinsic, and
            // restrict it to prevent overflow.
            Self(vshlq_s64(self.0, vnegq_s64(vreinterpretq_s64_u64(rhs.min(u64x2::splat(64)).0))))
          } else {
            // Cannot use scalar `unbounded_shl` because it takes `u32`, which is
            // smaller than `u64`.
            cast::<u64x2, i64x2>(rhs.simd_lt(64)).select(
              self >> cast::<u64x2, i64x2>(rhs),
              self.is_negative(),
            )
          }
        }
    }

    #[inline]
    pub(crate) fn unbounded_shr_scalar_backend(self, rhs: u32) -> Self {
        pick! {
          if #[cfg(target_feature="simd128")] {
            if rhs < 64 { Self(i64x2_shr(self.0, rhs)) } else { self.is_negative() }
          } else {
            let self_array = self.to_array();

            Self::from_array([
              self_array[0].unbounded_shr(rhs),
              self_array[1].unbounded_shr(rhs),
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
            Self(vqaddq_s64(self.0, rhs.0))
          } else {
            Self::from_array([
                self.0.arr[0].saturating_add(rhs.0.arr[0]),
                self.0.arr[1].saturating_add(rhs.0.arr[1]),
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
            Self(vqsubq_s64(self.0, rhs.0))
          } else {
            Self::from_array([
                self.0.arr[0].saturating_sub(rhs.0.arr[0]),
                self.0.arr[1].saturating_sub(rhs.0.arr[1]),
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn overflowing_mul_backend(self, rhs: Self) -> (Self, Self) {
        // TODO(perf): This implementation looks quite bad. Is there a better
        // one? This intentionally avoids `mul_keep_low_high` because getting the
        // high bits of 64-bit multiplication could be slow.

        let self_array = self.to_array();
        let rhs_array = rhs.to_array();

        let result = [
            self_array[0].overflowing_mul(rhs_array[0]),
            self_array[1].overflowing_mul(rhs_array[1]),
        ];
        (
            Self::from_array([result[0].0, result[1].0]),
            Self::from_array([-(result[0].1 as i64), -(result[1].1 as i64)]),
        )
    }

    #[inline]
    pub(crate) fn mul_keep_low_high_backend(self, rhs: Self) -> (u64x2, i64x2) {
        // TODO(perf): This implementation looks quite bad. Is there a better
        // one?

        let self_array = self.to_array();
        let rhs_array = rhs.to_array();

        let widening_mul = [
            (self_array[0] as i128).wrapping_mul(rhs_array[0] as i128),
            (self_array[1] as i128).wrapping_mul(rhs_array[1] as i128),
        ];

        (
            u64x2::from_array([widening_mul[0] as u64, widening_mul[1] as u64]),
            i64x2::from_array([
                (widening_mul[0] >> 64) as i64,
                (widening_mul[1] >> 64) as i64,
            ]),
        )
    }

    #[inline]
    pub(crate) fn mul_keep_high_backend(self, rhs: Self) -> Self {
        let self_array = self.to_array();
        let rhs_array = rhs.to_array();

        Self::from_array([
            ((self_array[0] as i128).wrapping_mul(rhs_array[0] as i128) >> 64) as i64,
            ((self_array[1] as i128).wrapping_mul(rhs_array[1] as i128) >> 64) as i64,
        ])
    }
}
}

safe_target_feature! {
// Backend of SIMD signed integer functions.
impl i64x2 {
    #[inline]
    pub(crate) fn abs_backend(self) -> Self {
        pick! {
          if #[cfg(all(target_feature = "avx512f", target_feature = "avx512vl"))] {
            Self(_mm_abs_epi64(self.0))
          } else if #[cfg(target_feature="sse2")] {
            let is_negative = self.is_negative();
            // This trick efficiently performs `is_negative.select(!self + 1, self)`
            (self ^ is_negative) - is_negative
          } else if #[cfg(target_feature="simd128")] {
            Self(i64x2_abs(self.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vabsq_s64(self.0))
          } else {
            let arr: [i64; 2] = cast(self);
            cast(
              [
                arr[0].wrapping_abs(),
                arr[1].wrapping_abs(),
              ])
          }
        }
    }

    #[inline]
    pub(crate) fn is_positive_backend(self) -> Self {
        pick! {
          if #[cfg(all(target_feature="neon", target_arch="aarch64"))] {
            Self(vreinterpretq_s64_u64(vcgtzq_s64(self.0)))
          } else {
            self.simd_gt(Self::ZERO)
          }
        }
    }

    #[inline]
    pub(crate) fn is_negative_backend(self) -> Self {
        pick! {
          if #[cfg(all(target_feature="neon", target_arch="aarch64"))] {
            Self(vreinterpretq_s64_u64(vcltzq_s64(self.0)))
          } else if #[cfg(all(target_feature = "avx512f", target_feature = "avx512vl"))] {
            Self(_mm_srai_epi64::<63>(self.0))
          } else {
            self.simd_lt(Self::ZERO)
          }
        }
    }

    #[inline]
    pub(crate) fn saturating_neg_backend(self) -> Self {
        pick! {
            if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
                Self(vqnegq_s64(self.0))
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
                Self(vqabsq_s64(self.0))
            } else {
                // If `self == MIN`, flip all bits to change the result from
                // `MIN` to `MAX`
                self.abs() ^ self.simd_eq(Self::MIN)
            }
        }
    }

    #[inline]
    pub(crate) fn saturating_add_unsigned_backend(self, rhs: u64x2) -> Self {
        pick! {
            if #[cfg(all(target_feature = "neon", target_arch = "aarch64"))] {
                Self(vuqaddq_s64(self.0, rhs.0))
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
impl i64x2 {
    #[inline]
    pub(crate) fn round_float_backend(self) -> f64x2 {
        let arr: [i64; 2] = cast(self);
        cast([arr[0] as f64, arr[1] as f64])
    }
}
}
