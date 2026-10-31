use crate::backend_imports::*;

#[cfg(not(any(
    target_feature = "sse2",
    target_feature = "simd128",
    all(target_feature = "neon", target_arch = "aarch64")
)))]
#[repr(C, align(16))]
#[derive(Clone, Copy)]
pub(crate) struct Inner {
    pub arr: [f64; 2],
}

safe_target_feature! {
impl SimdBackend for f64x2 {
    pick! {
      if #[cfg(target_feature="sse2")] {
        type Inner = __m128d;
      } else if #[cfg(target_feature="simd128")] {
        type Inner = v128;
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        type Inner = float64x2_t;
      } else {
        type Inner = Inner;
      }
    }

    #[inline]
    pub(crate) fn simd_eq_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse2")] {
            Self(_mm_cmpeq_pd(self.0, rhs.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(f64x2_eq(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vreinterpretq_f64_u64(vceqq_f64(self.0, rhs.0)))
          } else {
            Self::from_array([
              if self.0.arr[0] == rhs.0.arr[0] { f64::from_bits(u64::MAX) } else { 0.0 },
              if self.0.arr[1] == rhs.0.arr[1] { f64::from_bits(u64::MAX) } else { 0.0 },
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn simd_ne_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse2")] {
            Self(_mm_cmpneq_pd(self.0, rhs.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(f64x2_ne(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            !Self(vreinterpretq_f64_u64(vceqq_f64(self.0, rhs.0)))
          } else {
            Self::from_array([
              if self.0.arr[0] != rhs.0.arr[0] { f64::from_bits(u64::MAX) } else { 0.0 },
              if self.0.arr[1] != rhs.0.arr[1] { f64::from_bits(u64::MAX) } else { 0.0 },
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn simd_lt_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse2")] {
            Self(_mm_cmplt_pd(self.0, rhs.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(f64x2_lt(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vreinterpretq_f64_u64(vcltq_f64(self.0, rhs.0)))
          } else {
            Self::from_array([
              if self.0.arr[0] < rhs.0.arr[0] { f64::from_bits(u64::MAX) } else { 0.0 },
              if self.0.arr[1] < rhs.0.arr[1] { f64::from_bits(u64::MAX) } else { 0.0 },
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn simd_gt_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="avx")] {
            Self(_mm_cmp_pd::<{_CMP_GT_OQ}>(self.0, rhs.0))
          } else if #[cfg(target_feature="sse2")] {
            Self(_mm_cmpgt_pd(self.0, rhs.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(f64x2_gt(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vreinterpretq_f64_u64(vcgtq_f64(self.0, rhs.0)))
          } else {
            Self::from_array([
              if self.0.arr[0] > rhs.0.arr[0] { f64::from_bits(u64::MAX) } else { 0.0 },
              if self.0.arr[1] > rhs.0.arr[1] { f64::from_bits(u64::MAX) } else { 0.0 },
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn simd_le_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse2")] {
            Self(_mm_cmple_pd(self.0, rhs.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(f64x2_le(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vreinterpretq_f64_u64(vcleq_f64(self.0, rhs.0)))
          } else {
            Self::from_array([
              if self.0.arr[0] <= rhs.0.arr[0] { f64::from_bits(u64::MAX) } else { 0.0 },
              if self.0.arr[1] <= rhs.0.arr[1] { f64::from_bits(u64::MAX) } else { 0.0 },
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn simd_ge_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse2")] {
            Self(_mm_cmpge_pd(self.0, rhs.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(f64x2_ge(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vreinterpretq_f64_u64(vcgeq_f64(self.0, rhs.0)))
          } else {
            Self::from_array([
              if self.0.arr[0] >= rhs.0.arr[0] { f64::from_bits(u64::MAX) } else { 0.0 },
              if self.0.arr[1] >= rhs.0.arr[1] { f64::from_bits(u64::MAX) } else { 0.0 },
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn select_backend(self, if_true: Self, if_false: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse4.1")] {
            Self(_mm_blendv_pd(if_false.0, if_true.0, self.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(v128_bitselect(if_true.0, if_false.0, self.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vbslq_f64(vreinterpretq_u64_f64(self.0), if_true.0, if_false.0))
          } else {
            generic_bit_blend(self, if_true, if_false)
          }
        }
    }

    #[inline]
    pub(crate) fn to_bitmask_backend(self) -> u32 {
        pick! {
          if #[cfg(target_feature="sse2")] {
            _mm_movemask_pd(self.0) as u32
          } else if #[cfg(target_feature="simd128")] {
            u64x2_bitmask(self.0) as u32
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            let e = vreinterpretq_u64_f64(self.0);

            (vgetq_lane_u64(e,0) >> 63 | ((vgetq_lane_u64(e,1) >> 62) & 0x2)) as u32
          } else {
            (((self.0.arr[0].to_bits() as i64) < 0) as u32) |
            (((self.0.arr[1].to_bits() as i64) < 0) as u32) << 1
          }
        }
    }

    #[inline]
    pub(crate) fn any_backend(self) -> bool {
        pick! {
          if #[cfg(target_feature="simd128")] {
            v128_any_true(self.0)
          } else {
            self.to_bitmask() != 0
          }
        }
    }

    #[inline]
    pub(crate) fn all_backend(self) -> bool {
        pick! {
          if #[cfg(target_feature="simd128")] {
            u64x2_all_true(self.0)
          } else {
            // two lanes
            self.to_bitmask() == 0b11
          }
        }
    }

    #[inline]
    pub(crate) fn interleave_low_backend(self, other: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse2")] {
            Self(_mm_unpacklo_pd(self.0, other.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(i64x2_shuffle::<0, 2>(self.0, other.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))] {
            Self(vzip1q_f64(self.0, other.0))
          } else {
            Self::from_array([self.as_array()[0], other.as_array()[0]])
          }
        }
    }

    #[inline]
    pub(crate) fn interleave_high_backend(self, other: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse2")] {
            Self(_mm_unpackhi_pd(self.0, other.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(i64x2_shuffle::<1, 3>(self.0, other.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))] {
            Self(vzip2q_f64(self.0, other.0))
          } else {
            Self::from_array([self.as_array()[1], other.as_array()[1]])
          }
        }
    }

    #[inline]
    pub(crate) fn transpose_backend(data: [f64x2; 2]) -> [f64x2; 2] {
        pick! {
          if #[cfg(any(
            target_feature="sse2",
            all(target_feature="neon",target_arch="aarch64"),
            target_feature="simd128",
          ))] {
            [data[0].interleave_low(data[1]), data[0].interleave_high(data[1])]
          } else {
            let [x, y, z, w]: [f64; 4] = cast(data);
            cast([x, z, y, w])
          }
        }
    }
}
}

safe_target_feature! {
// Backend of SIMD float functions.
impl f64x2 {
    #[inline]
    pub(crate) fn neg_backend(self) -> Self {
        pick! {
          if #[cfg(target_feature="sse")] {
            Self(_mm_xor_pd(self.0, Self::splat(-0.0).0))
          } else if #[cfg(target_feature="simd128")] {
            Self(f64x2_neg(self.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vnegq_f64(self.0))
          } else {
            Self::from_array([
              -self.0.arr[0],
              -self.0.arr[1],
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn not_backend(self) -> Self {
        pick! {
          if #[cfg(target_feature="sse2")] {
            let all_bits = const { Self::from_bits(u64x2::MAX) };
            self ^ all_bits
          } else if #[cfg(target_feature="simd128")] {
            Self(v128_not(self.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vreinterpretq_f64_u32(vmvnq_u32(vreinterpretq_u32_f64(self.0))))
          } else {
            Self::from_array([
              f64::from_bits(!self.0.arr[0].to_bits()),
              f64::from_bits(!self.0.arr[1].to_bits()),
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn add_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse2")] {
            Self(_mm_add_pd(self.0, rhs.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(f64x2_add(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vaddq_f64(self.0, rhs.0))
          } else {
            Self::from_array([
              self.0.arr[0] + rhs.0.arr[0],
              self.0.arr[1] + rhs.0.arr[1],
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn sub_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse2")] {
            Self(_mm_sub_pd(self.0, rhs.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(f64x2_sub(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vsubq_f64(self.0, rhs.0))
          } else {
            Self::from_array([
              self.0.arr[0] - rhs.0.arr[0],
              self.0.arr[1] - rhs.0.arr[1],
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn mul_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse2")] {
            Self(_mm_mul_pd(self.0, rhs.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(f64x2_mul(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vmulq_f64(self.0, rhs.0))
          } else {
            Self::from_array([
              self.0.arr[0] * rhs.0.arr[0],
              self.0.arr[1] * rhs.0.arr[1],
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn div_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse2")] {
            Self(_mm_div_pd(self.0, rhs.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(f64x2_div(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vdivq_f64(self.0, rhs.0))
          } else {
            Self::from_array([
              self.0.arr[0] / rhs.0.arr[0],
              self.0.arr[1] / rhs.0.arr[1],
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn rem_backend(self, rhs: Self) -> Self {
        Self::from_array([
            self.to_array()[0] % rhs.to_array()[0],
            self.to_array()[1] % rhs.to_array()[1],
        ])
    }

    #[inline]
    pub(crate) fn bitand_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse2")] {
            Self(_mm_and_pd(self.0, rhs.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(v128_and(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vreinterpretq_f64_u64(vandq_u64(vreinterpretq_u64_f64(self.0), vreinterpretq_u64_f64(rhs.0))))
          } else {
            Self::from_array([
              f64::from_bits(self.0.arr[0].to_bits() & rhs.0.arr[0].to_bits()),
              f64::from_bits(self.0.arr[1].to_bits() & rhs.0.arr[1].to_bits()),
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn bitor_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse2")] {
            Self(_mm_or_pd(self.0, rhs.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(v128_or(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vreinterpretq_f64_u64(vorrq_u64(vreinterpretq_u64_f64(self.0), vreinterpretq_u64_f64(rhs.0))))
          } else {
            Self::from_array([
              f64::from_bits(self.0.arr[0].to_bits() | rhs.0.arr[0].to_bits()),
              f64::from_bits(self.0.arr[1].to_bits() | rhs.0.arr[1].to_bits()),
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn bitxor_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse2")] {
            Self(_mm_xor_pd(self.0, rhs.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(v128_xor(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vreinterpretq_f64_u64(veorq_u64(vreinterpretq_u64_f64(self.0), vreinterpretq_u64_f64(rhs.0))))
          } else {
            Self::from_array([
              f64::from_bits(self.0.arr[0].to_bits() ^ rhs.0.arr[0].to_bits()),
              f64::from_bits(self.0.arr[1].to_bits() ^ rhs.0.arr[1].to_bits()),
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn reduce_add_backend(self) -> f64 {
        pick! {
          if #[cfg(target_feature="ssse3")] {
            let a = _mm_hadd_pd(self.0, self.0);
            cast::<__m128d, [f64; 2]>(a)[0]
          } else if #[cfg(any(target_feature="sse2", target_feature="simd128"))] {
            let a: [f64;2] = cast(self);
            a.iter().sum()
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            vgetq_lane_f64(self.0,0) + vgetq_lane_f64(self.0,1)
          } else {
            self.0.arr.iter().sum()
          }
        }
    }

    #[inline]
    pub(crate) fn reduce_mul_backend(self) -> f64 {
        let arr: [f64; 2] = cast(self);
        arr.iter().product()
    }

    #[inline]
    pub(crate) fn bitselect_backend(self, if_one: Self, if_zero: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse2")] {
            Self(_mm_or_pd(_mm_and_pd(if_one.0, self.0), _mm_andnot_pd(self.0, if_zero.0)))
          } else if #[cfg(target_feature="simd128")] {
            Self(v128_bitselect(if_one.0, if_zero.0, self.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vbslq_f64(vreinterpretq_u64_f64(self.0), if_one.0, if_zero.0))
          } else {
            generic_bit_blend(self, if_one, if_zero)
          }
        }
    }

    #[inline]
    pub(crate) fn is_nan_backend(self) -> Self {
        pick! {
          if #[cfg(target_feature="sse2")] {
            Self(_mm_cmpunord_pd(self.0, self.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(f64x2_ne(self.0, self.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            !Self(vreinterpretq_f64_u64(vceqq_f64(self.0, self.0)))
          } else {
            Self::from_array([
              if self.0.arr[0].is_nan() { f64::from_bits(u64::MAX) } else { 0.0 },
              if self.0.arr[1].is_nan() { f64::from_bits(u64::MAX) } else { 0.0 },
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn is_infinite_backend(self) -> Self {
        let shifted_inf = u64x2::from(0xFFE0000000000000);
        let u: u64x2 = cast(self);
        let shift_u = u << 1_u64;
        let out = (shift_u).simd_eq(shifted_inf);
        cast(out)
    }

    #[inline]
    pub(crate) fn is_finite_backend(self) -> Self {
        let shifted_exp_mask = u64x2::from(0xFFE0000000000000);
        let u: u64x2 = cast(self);
        let shift_u = u << 1_u64;
        let out = !(shift_u & shifted_exp_mask).simd_eq(shifted_exp_mask);
        cast(out)
    }

    #[inline]
    pub(crate) fn is_sign_positive_backend(self) -> Self {
        pick! {
          // Integer equality is slow without `sse4.1`.
          if #[cfg(any(target_feature = "sse4.1", not(target_feature = "sse2")))] {
            const SIGN_MASK: u64x2 = u64x2::splat((-0.0_f64).to_bits());

            let bits = cast::<f64x2, u64x2>(self);
            let sign = bits & SIGN_MASK;
            let result = sign.simd_eq(u64x2::ZERO);
            cast::<u64x2, f64x2>(result)
          } else {
            let bits = cast::<f64x2, u64x2>(self);
            let sign = bits >> 63;
            let sign = cast::<u64x2, f64x2>(sign);
            sign.simd_eq(f64x2::ZERO)
          }
        }
    }

    #[inline]
    pub(crate) fn recip_backend(self) -> Self {
        // There does not seem to be a `recip` intrinsic for any architecture. The
        // closest is `_mm_rcp14_pd` which has relative error.
        Self::ONE / self
    }

    #[inline]
    pub(crate) fn recip_sqrt_backend(self) -> Self {
        // There does not seem to be a `recip_sqrt` intrinsic for any architecture.
        // The closest is `_mm_rsqrt14_pd` which has relative error.
        Self::ONE / self.sqrt()
    }

    #[inline]
    pub(crate) fn max_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse2")] {
            // _mm_max_pd seems to do rhs < self ? self : rhs. So if there's any NaN
            // involved, it chooses rhs, so we need to specifically check rhs for
            // NaN.
            rhs.is_nan().select(self, Self(_mm_max_pd(self.0, rhs.0)))
          } else if #[cfg(target_feature="simd128")] {
            // WASM has two max intrinsics:
            // - max: This propagates NaN, that's the opposite of what we need.
            // - pmax: This is defined as self < rhs ? rhs : self, which basically
            //   chooses self if either is NaN.
            //
            // pmax is what we want, but we need to specifically check self for NaN.
            Self(v128_bitselect(
              rhs.0,
              f64x2_pmax(self.0, rhs.0),
              f64x2_ne(self.0, self.0), // NaN check
            ))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vmaxnmq_f64(self.0, rhs.0))
          } else {
            Self::from_array([
              self.0.arr[0].max(rhs.0.arr[0]),
              self.0.arr[1].max(rhs.0.arr[1]),
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn fast_max_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse2")] {
            Self(_mm_max_pd(self.0, rhs.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(f64x2_pmax(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vmaxq_f64(self.0, rhs.0))
          } else {
            Self::from_array([
              if self.0.arr[0] < rhs.0.arr[0] { rhs.0.arr[0] } else { self.0.arr[0] },
              if self.0.arr[1] < rhs.0.arr[1] { rhs.0.arr[1] } else { self.0.arr[1] },
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn min_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse2")] {
            // _mm_min_pd seems to do rhs < self ? rhs : self. So if there's any NaN
            // involved, it chooses rhs, so we need to specifically check rhs for
            // NaN.
            rhs.is_nan().select(self, Self(_mm_min_pd(self.0, rhs.0)))
          } else if #[cfg(target_feature="simd128")] {
            // WASM has two min intrinsics:
            // - min: This propagates NaN, that's the opposite of what we need.
            // - pmin: This is defined as rhs < self ? rhs : self, which basically
            //   chooses self if either is NaN.
            //
            // pmin is what we want, but we need to specifically check self for NaN.
            Self(v128_bitselect(
              rhs.0,
              f64x2_pmin(self.0, rhs.0),
              f64x2_ne(self.0, self.0), // NaN check
            ))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vminnmq_f64(self.0, rhs.0))
          } else {
            Self::from_array([
              self.0.arr[0].min(rhs.0.arr[0]),
              self.0.arr[1].min(rhs.0.arr[1]),
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn fast_min_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse2")] {
            Self(_mm_min_pd(self.0, rhs.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(f64x2_pmin(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vminq_f64(self.0, rhs.0))
          } else {
            Self::from_array([
              if self.0.arr[0] < rhs.0.arr[0] { self.0.arr[0] } else { rhs.0.arr[0] },
              if self.0.arr[1] < rhs.0.arr[1] { self.0.arr[1] } else { rhs.0.arr[1] },
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn clamp_backend(self, min: Self, max: Self) -> Self {
        pick! {
          if #[cfg(any(
            target_feature="simd128",
            all(target_feature="neon",target_arch="aarch64"),
          ))] {
            // `fast_clamp` already works.
            self.fast_clamp(min, max)
          } else {
            // This works since all bits set is NaN.
            self.fast_clamp(min, max) | min.is_nan() | max.is_nan()
          }
        }
    }

    #[inline]
    pub(crate) fn fast_clamp_backend(self, min: Self, max: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse2")] {
            // For both `_mm_min_pd` and `_mm_max_pd` if any input is NaN, `rhs` gets
            // chosen. For `self` to be chosen, `self` must be the second argument.
            Self(_mm_max_pd(min.0, _mm_min_pd(max.0, self.0)))
          } else if #[cfg(target_feature="simd128")] {
            Self(f64x2_max(f64x2_min(self.0, max.0), min.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))] {
            Self(vmaxq_f64(vminq_f64(self.0, max.0), min.0))
          } else {
            // The standard library does not have NaN propagating `min` and `max`
            // functions.
            let mut result = self;
            result = result.simd_gt(max).select(max, result);
            result = result.simd_lt(min).select(min, result);
            result
          }
        }
    }

    #[inline]
    pub(crate) fn reduce_max_backend(self) -> f64 {
        pick! {
          if #[cfg(all(target_arch = "aarch64", target_feature = "neon"))] {
            vmaxnmvq_f64(self.0)
          } else {
            let self_array = self.to_array();
            self_array[0].max(self_array[1])
          }
        }
    }

    #[inline]
    pub(crate) fn fast_reduce_max_backend(self) -> f64 {
        pick! {
          if #[cfg(all(target_arch = "aarch64", target_feature = "neon"))] {
            vmaxnmvq_f64(self.0)
          } else {
            let [e0, e1] = self.to_array();
            if e0 > e1 { e0 } else { e1 }
          }
        }
    }

    #[inline]
    pub(crate) fn reduce_min_backend(self) -> f64 {
        pick! {
          if #[cfg(all(target_arch = "aarch64", target_feature = "neon"))] {
            vminnmvq_f64(self.0)
          } else {
            let self_array = self.to_array();
            self_array[0].min(self_array[1])
          }
        }
    }

    #[inline]
    pub(crate) fn fast_reduce_min_backend(self) -> f64 {
        pick! {
          if #[cfg(all(target_arch = "aarch64", target_feature = "neon"))] {
            vminnmvq_f64(self.0)
          } else {
            let [e0, e1] = self.to_array();
            if e0 < e1 { e0 } else { e1 }
          }
        }
    }

    #[inline]
    pub(crate) fn abs_backend(self) -> Self {
        pick! {
          if #[cfg(target_feature="simd128")] {
            Self(f64x2_abs(self.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vabsq_f64(self.0))
          } else {
            let non_sign_bits = f64x2::from(f64::from_bits(i64::MAX as u64));
            self & non_sign_bits
          }
        }
    }

    #[inline]
    pub(crate) fn floor_backend(self) -> Self {
        pick! {
          if #[cfg(target_feature="simd128")] {
            Self(f64x2_floor(self.0))
          } else if #[cfg(target_feature="sse4.1")] {
            Self(_mm_floor_pd(self.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vrndmq_f64(self.0))
          } else if #[cfg(feature="std")] {
            let base: [f64; 2] = cast(self);
            cast(base.map(|val| val.floor()))
          } else {
            let base: [f64; 2] = cast(self);
            let rounded: [f64; 2] = cast(self.round());
            cast([
              if base[0] < rounded[0] { rounded[0] - 1.0 } else { rounded[0] },
              if base[1] < rounded[1] { rounded[1] - 1.0 } else { rounded[1] },
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn ceil_backend(self) -> Self {
        pick! {
          if #[cfg(target_feature="simd128")] {
            Self(f64x2_ceil(self.0))
          } else if #[cfg(target_feature="sse4.1")] {
            Self(_mm_ceil_pd(self.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vrndpq_f64(self.0))
          } else if #[cfg(feature="std")] {
            let base: [f64; 2] = cast(self);
            cast(base.map(|val| val.ceil()))
          } else {
            let base: [f64; 2] = cast(self);
            let rounded: [f64; 2] = cast(self.round());
            cast([
              if base[0] > rounded[0] { rounded[0] + 1.0 } else { rounded[0] },
              if base[1] > rounded[1] { rounded[1] + 1.0 } else { rounded[1] },
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn round_backend(self) -> Self {
        pick! {
          if #[cfg(target_feature="sse4.2")] {
            const_f64_as_f64x2!(HALF_NEXT_DOWN, 0.5_f64.next_down());
            const_f64_as_f64x2!(BOUNDS_LIMIT, 4503599627370496.0);

            let self_abs = self.abs();

            let adjusted_self = self_abs + Self::HALF;
            let result_abs = Self(_mm_round_pd::<{_MM_FROUND_NO_EXC | _MM_FROUND_TO_ZERO}>(adjusted_self.0));
            // The addition breaks for `0.5.next_down()` which incorrectly rounds to
            // `1.0`. This resets the result back to `0.0`.
            let result_abs = result_abs & self_abs.simd_ne(HALF_NEXT_DOWN);

            // Large value, infinity and NaN need special handling.
            let bounds_mask: Self = cast(_mm_cmpgt_epi64(cast(BOUNDS_LIMIT), cast(self_abs)));

            // `abs` keeps the original sign.
            bounds_mask.abs().bitselect(result_abs, self)
          } else if #[cfg(target_feature="simd128")] {
            const_f64_as_f64x2!(HALF_NEXT_DOWN, 0.5_f64.next_down());
            const_f64_as_f64x2!(BOUNDS_LIMIT, 4503599627370496.0);

            let self_abs = self.abs();

            let adjusted_self = self_abs + Self::HALF;
            let result_abs = Self(f64x2_trunc(adjusted_self.0));
            // The addition breaks for `0.5.next_down()` which incorrectly rounds to
            // `1.0`. This resets the result back to `0.0`.
            let result_abs = result_abs & self_abs.simd_ne(HALF_NEXT_DOWN);

            // Large value, infinity and NaN need special handling.
            let bounds_mask = Self(i64x2_lt(self_abs.0, BOUNDS_LIMIT.0));

            // `abs` keeps the original sign.
            bounds_mask.abs().bitselect(result_abs, self)
          } else if #[cfg(all(target_feature = "neon", target_arch = "aarch64"))] {
            Self(vrndaq_f64(self.0))
          } else {
            const_f64_as_f64x2!(HALF_NEXT_DOWN, 0.5_f64.next_down());
            const_f64_as_f64x2!(BOUNDS_LIMIT, 4503599627370496.0);

            let self_abs = self.abs();

            let adjusted_self = (self_abs + Self::HALF).to_array();
            let result_abs = Self::from_array([
              adjusted_self[0] as u64 as f64,
              adjusted_self[1] as u64 as f64,
            ]);
            // The addition breaks for `0.5.next_down()` which incorrectly rounds to
            // `1.0`. This resets the result back to `0.0`.
            let result_abs = result_abs & self_abs.simd_ne(HALF_NEXT_DOWN);

            // Large value, infinity and NaN need special handling.
            let bounds_mask: Self = cast(cast::<_, i64x2>(self_abs).simd_lt(cast::<_, i64x2>(BOUNDS_LIMIT)));

            // `abs` keeps the original sign.
            bounds_mask.abs().bitselect(result_abs, self)
          }
        }
    }

    #[inline]
    pub(crate) fn round_int_backend(self) -> i64x2 {
        pick! {
          if #[cfg(all(target_feature="avx512dq", target_feature="avx512vl"))] {
            // Based on: https://github.com/v8/v8/blob/210987a552a2bf2a854b0baa9588a5959ff3979d/src/codegen/shared-ia32-x64/macro-assembler-shared-ia32-x64.h#L489-L504
            let non_nan_mask = self.simd_eq(self);
            let non_nan = self & non_nan_mask;
            let flip_to_max: i64x2 = cast(self.simd_ge(Self::splat(9223372036854775808.0)));

            let cast: i64x2 = cast(_mm_cvtpd_epi64(non_nan.0));
            flip_to_max ^ cast
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            cast(Self(vreinterpretq_f64_s64(vcvtnq_s64_f64(self.0))))
          } else {
            let rounded: [f64; 2] = cast(self.round_ties_even());
            cast([rounded[0] as i64, rounded[1] as i64])
          }
        }
    }

    #[inline]
    pub(crate) fn fast_round_int_backend(self) -> i64x2 {
        pick! {
          if #[cfg(all(target_feature="avx512dq", target_feature="avx512vl"))] {
            cast(_mm_cvtpd_epi64(self.0))
          } else {
            self.round_int()
          }
        }
    }

    #[inline]
    pub(crate) fn round_ties_even_backend(self) -> Self {
        pick! {
          if #[cfg(target_feature="sse4.1")] {
            Self(_mm_round_pd::<{_MM_FROUND_NO_EXC | _MM_FROUND_TO_NEAREST_INT}>(self.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(f64x2_nearest(self.0))
          } else if #[cfg(all(target_feature = "neon", target_arch = "aarch64"))] {
            Self(vrndnq_f64(self.0))
          } else {
            const SIGN_MASK: f64x2 = f64x2::splat(-0.0);
            const MAGIC_VALUE: f64x2 = f64x2::splat(f64::from_bits(0x43300000_00000000));

            let self_sign = self & SIGN_MASK;
            let magic_value = MAGIC_VALUE | self_sign;
            let result = self + magic_value - magic_value;

            let bounds_mask = self.abs().simd_le(MAGIC_VALUE);
            bounds_mask.abs().select(result, self)
          }
        }
    }

    #[inline]
    pub(crate) fn trunc_backend(self) -> Self {
        pick! {
          if #[cfg(target_feature="sse4.1")] {
            Self(_mm_round_pd::<{_MM_FROUND_NO_EXC | _MM_FROUND_TO_ZERO}>(self.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(f64x2_trunc(self.0))
          } else if #[cfg(all(target_feature = "neon", target_arch = "aarch64"))] {
            Self(vrndq_f64(self.0))
          } else {
            // There does not seem to be an SSE2 intrinsic for this.
            // `_mm_cvttpd_epi32` truncates to `i32` values which cannot
            // represent all possible outputs.

            let array: [f64; 2] = cast(self);
            let result: Self = cast([
              array[0] as i64 as f64,
              array[1] as i64 as f64,
            ]);

            // Out of range values are either already round, infinite or NaN. Values
            // in range can all be represented by `i64`.
            const BOUNDS_LIMIT: i64 = 18e15_f64.to_bits().cast_signed();
            let bounds_mask: Self = cast(cast::<f64x2, i64x2>(self.abs()).simd_lt(i64x2::splat(BOUNDS_LIMIT)));

            // Reset the sign bit of the mask to preverse the sign of `self`.
            bounds_mask.abs().select(result, self)
          }
        }
    }

    #[inline]
    pub(crate) fn trunc_int_backend(self) -> i64x2 {
        pick! {
          if #[cfg(all(target_feature="avx512dq", target_feature="avx512vl"))] {
            // Based on: https://github.com/v8/v8/blob/210987a552a2bf2a854b0baa9588a5959ff3979d/src/codegen/shared-ia32-x64/macro-assembler-shared-ia32-x64.h#L489-L504
            let non_nan_mask = self.simd_eq(self);
            let non_nan = self & non_nan_mask;
            let flip_to_max: i64x2 = cast(self.simd_ge(Self::splat(9223372036854775808.0)));

            let cast: i64x2 = cast(_mm_cvttpd_epi64(non_nan.0));
            flip_to_max ^ cast
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            cast(Self(vreinterpretq_f64_s64(vcvtq_s64_f64(self.0))))
          } else {
            // There does not seem to be an intrinsic for `wasm32`.
            let n: [f64;2] = cast(self);
            cast([n[0] as i64, n[1] as i64])
          }
        }
    }

    #[inline]
    pub(crate) fn fast_trunc_int_backend(self) -> i64x2 {
        pick! {
          if #[cfg(all(target_feature="avx512dq", target_feature="avx512vl"))] {
            cast(_mm_cvttpd_epi64(self.0))
          } else {
            self.trunc_int()
          }
        }
    }

    #[inline]
    pub(crate) fn fast_mul_add_backend(self, a: Self, b: Self) -> Self {
        pick! {
          if #[cfg(all(target_feature="fma"))] {
            Self(_mm_fmadd_pd(self.0, a.0, b.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))] {
            Self(vfmaq_f64(b.0, self.0, a.0))
          } else {
            (self * a) + b
          }
        }
    }

    #[inline]
    pub(crate) fn fast_mul_sub_backend(self, a: Self, b: Self) -> Self {
        pick! {
          if #[cfg(all(target_feature="fma"))] {
            Self(_mm_fmsub_pd(self.0, a.0, b.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))] {
            Self(vfmaq_f64(vnegq_f64(b.0), self.0, a.0))
          } else {
            (self * a) - b
          }
        }
    }

    #[inline]
    pub(crate) fn fast_mul_neg_add_backend(self, a: Self, b: Self) -> Self {
        pick! {
            if #[cfg(all(target_feature="fma"))] {
              Self(_mm_fnmadd_pd(self.0, a.0, b.0))
            } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))] {
              Self(vfmsq_f64(b.0, self.0, a.0))
            } else {
              b - (self * a)
            }
        }
    }

    #[inline]
    pub(crate) fn fast_mul_neg_sub_backend(self, a: Self, b: Self) -> Self {
        pick! {
            if #[cfg(all(target_feature="fma"))] {
              Self(_mm_fnmsub_pd(self.0, a.0, b.0))
            } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))] {
              Self(vnegq_f64(vfmaq_f64(b.0, self.0, a.0)))
            } else {
              -(self * a) - b
            }
        }
    }

    #[inline]
    pub(crate) fn powf_simd_backend(self, n: Self) -> Self {
        const_f64_as_f64x2!(ln2d_hi, 0.693145751953125);
        const_f64_as_f64x2!(ln2d_lo, 1.42860682030941723212E-6);
        const_f64_as_f64x2!(P0log, 2.0039553499201281259648E1);
        const_f64_as_f64x2!(P1log, 5.7112963590585538103336E1);
        const_f64_as_f64x2!(P2log, 6.0949667980987787057556E1);
        const_f64_as_f64x2!(P3log, 2.9911919328553073277375E1);
        const_f64_as_f64x2!(P4log, 6.5787325942061044846969E0);
        const_f64_as_f64x2!(P5log, 4.9854102823193375972212E-1);
        const_f64_as_f64x2!(P6log, 4.5270000862445199635215E-5);
        const_f64_as_f64x2!(Q0log, 6.0118660497603843919306E1);
        const_f64_as_f64x2!(Q1log, 2.1642788614495947685003E2);
        const_f64_as_f64x2!(Q2log, 3.0909872225312059774938E2);
        const_f64_as_f64x2!(Q3log, 2.2176239823732856465394E2);
        const_f64_as_f64x2!(Q4log, 8.3047565967967209469434E1);
        const_f64_as_f64x2!(Q5log, 1.5062909083469192043167E1);

        // Taylor expansion constants
        const_f64_as_f64x2!(p2, 1.0 / 2.0); // coefficients for Taylor expansion of exp
        const_f64_as_f64x2!(p3, 1.0 / 6.0);
        const_f64_as_f64x2!(p4, 1.0 / 24.0);
        const_f64_as_f64x2!(p5, 1.0 / 120.0);
        const_f64_as_f64x2!(p6, 1.0 / 720.0);
        const_f64_as_f64x2!(p7, 1.0 / 5040.0);
        const_f64_as_f64x2!(p8, 1.0 / 40320.0);
        const_f64_as_f64x2!(p9, 1.0 / 362880.0);
        const_f64_as_f64x2!(p10, 1.0 / 3628800.0);
        const_f64_as_f64x2!(p11, 1.0 / 39916800.0);
        const_f64_as_f64x2!(p12, 1.0 / 479001600.0);
        const_f64_as_f64x2!(p13, 1.0 / 6227020800.0);

        let x1 = self.abs();
        let x = x1.fraction_2();
        let mask = x.simd_gt(f64x2::SQRT_2 * f64x2::HALF);
        let x = (!mask).select(x + x, x);
        let x = x - f64x2::ONE;
        let x2 = x * x;
        let px = polynomial_6!(x, P0log, P1log, P2log, P3log, P4log, P5log, P6log);
        let px = px * x * x2;
        let qx = polynomial_6n!(x, Q0log, Q1log, Q2log, Q3log, Q4log, Q5log);
        let lg1 = px / qx;

        let ef = x1.exponent();
        let ef = mask.select(ef + f64x2::ONE, ef);
        let e1 = (ef * n).round_ties_even();
        let yr = ef.fast_mul_sub(n, e1);

        let lg = f64x2::HALF.fast_mul_neg_add(x2, x) + lg1;
        let x2err = (f64x2::HALF * x).fast_mul_sub(x, f64x2::HALF * x2);
        let lg_err = f64x2::HALF.fast_mul_add(x2, lg - x) - lg1;

        let e2 = (lg * n * f64x2::LOG2_E).round_ties_even();
        let v = lg.fast_mul_sub(n, e2 * ln2d_hi);
        let v = e2.fast_mul_neg_add(ln2d_lo, v);
        let v = v - (lg_err + x2err).fast_mul_sub(n, yr * f64x2::LN_2);

        let x = v;
        let e3 = (x * f64x2::LOG2_E).round_ties_even();
        let x = e3.fast_mul_neg_add(f64x2::LN_2, x);
        let z = polynomial_13!(x, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13) + f64x2::ONE;
        let ee = e1 + e2 + e3;
        let ei = cast::<_, i64x2>(ee.round_int());
        let ej = cast::<_, i64x2>(ei + (cast::<_, i64x2>(z) >> 52));

        let overflow =
            cast::<_, f64x2>(!ej.simd_lt(i64x2::splat(0x07FF))) | ee.simd_gt(f64x2::splat(3000.0));
        let underflow =
            cast::<_, f64x2>(!ej.simd_gt(i64x2::splat(0x000))) | ee.simd_lt(f64x2::splat(-3000.0));

        // Add exponent by integer addition
        let z = cast::<_, f64x2>(cast::<_, i64x2>(z) + (ei << 52));

        // Check for overflow/underflow
        let z = if (overflow | underflow).any() {
            let z = underflow.select(f64x2::ZERO, z);
            overflow.select(Self::infinity(), z)
        } else {
            z
        };

        // Check for self == 0
        let x_zero = self.is_zero_or_subnormal();
        let z = x_zero.select(
            n.simd_lt(f64x2::ZERO).select(
                Self::infinity(),
                n.simd_eq(f64x2::ZERO).select(f64x2::ONE, f64x2::ZERO),
            ),
            z,
        );

        let x_sign = self.is_sign_negative();
        let z = if x_sign.any() {
            // Y into an integer
            let yi = n.simd_eq(n.round_ties_even());
            // Is y odd? If yes flip the sign of the result.
            let y_odd = cast::<i64x2, f64x2>(n.round_int() << 63);

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
          if #[cfg(target_feature="sse2")] {
            Self(_mm_sqrt_pd(self.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(f64x2_sqrt(self.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vsqrtq_f64(self.0))
          } else if #[cfg(feature="std")] {
            Self::from_array([
              self.0.arr[0].sqrt(),
              self.0.arr[1].sqrt(),
            ])
          } else {
            Self::from_array([
              software_sqrt(self.0.arr[0]),
              software_sqrt(self.0.arr[1]),
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn exp_backend(self) -> Self {
        const_f64_as_f64x2!(P2, 1.0 / 2.0);
        const_f64_as_f64x2!(P3, 1.0 / 6.0);
        const_f64_as_f64x2!(P4, 1.0 / 24.0);
        const_f64_as_f64x2!(P5, 1.0 / 120.0);
        const_f64_as_f64x2!(P6, 1.0 / 720.0);
        const_f64_as_f64x2!(P7, 1.0 / 5040.0);
        const_f64_as_f64x2!(P8, 1.0 / 40320.0);
        const_f64_as_f64x2!(P9, 1.0 / 362880.0);
        const_f64_as_f64x2!(P10, 1.0 / 3628800.0);
        const_f64_as_f64x2!(P11, 1.0 / 39916800.0);
        const_f64_as_f64x2!(P12, 1.0 / 479001600.0);
        const_f64_as_f64x2!(P13, 1.0 / 6227020800.0);
        // LN2D_HI/LO: double-double decomposition of ln(2) for exp range reduction,
        // following fdlibm's approach (Sun Microsystems, https://www.netlib.org/fdlibm/ e_exp.c).
        // Values chosen so LN2D_HI + LN2D_LO = ln(2) to full f64 precision.
        const_f64_as_f64x2!(LN2D_HI, 0.693145751953125);
        const_f64_as_f64x2!(LN2D_LO, 1.42860682030941723212E-6);
        let max_x = f64x2::from(709.783);
        let min_x = f64x2::from(-744.79);
        let finite = self.is_finite();
        // x < min_x: e^x underflows to 0 -- skip the entire pipeline
        let neg_underflow = self.simd_lt(min_x) & finite;
        if neg_underflow.all() {
            return Self::ZERO;
        }
        let max_r = f64x2::from(1023.0);
        let r = (self * Self::LOG2_E).round_ties_even();
        let big = r.simd_gt(max_r);
        let r_safe = big.select(max_r, r);
        let excess = r - max_r;
        let excess = big.select(excess, Self::ZERO);
        let scale = Self::vm_pow2n(excess);
        let x = r.fast_mul_neg_add(LN2D_HI, self);
        let x = r.fast_mul_neg_add(LN2D_LO, x);
        let z = polynomial_13!(x, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13);
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
        const_f64_as_f64x2!(P2, 1.0 / 2.0);
        const_f64_as_f64x2!(P3, 1.0 / 6.0);
        const_f64_as_f64x2!(P4, 1.0 / 24.0);
        const_f64_as_f64x2!(P5, 1.0 / 120.0);
        const_f64_as_f64x2!(P6, 1.0 / 720.0);
        const_f64_as_f64x2!(P7, 1.0 / 5040.0);
        const_f64_as_f64x2!(P8, 1.0 / 40320.0);
        const_f64_as_f64x2!(P9, 1.0 / 362880.0);
        const_f64_as_f64x2!(P10, 1.0 / 3628800.0);

        // max_x = log2(f64::MAX) ≈ 1023.9999999999999
        // min_x = log2(f64::MIN_POSITIVE) - 52 ≈ -1022 - 52 = -1074
        let max_x = f64x2::from(1023.9999999999999);
        let min_x = f64x2::from(-1074.5);
        let finite = self.is_finite();
        let neg_underflow = self.simd_lt(min_x) & finite;
        if neg_underflow.all() {
            return Self::ZERO;
        }

        let round = self.round_ties_even();
        let max_r = f64x2::from(1023.0);
        let big = round.simd_gt(max_r);
        let r_safe = big.select(max_r, round);
        let excess = round - max_r;
        let excess = big.select(excess, Self::ZERO);
        let scale = Self::vm_pow2n(excess);

        let fract = (self - round) * Self::LN_2;
        let fract_partial_exp2 = polynomial_8!(fract, P2, P3, P4, P5, P6, P7, P8, P9, P10);
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
        const_f64_as_f64x2!(P0, 7.70838733755885391666E0);
        const_f64_as_f64x2!(P1, 1.79368678507819816313E1);
        const_f64_as_f64x2!(P2, 1.44989225341610930846E1);
        const_f64_as_f64x2!(P3, 4.70579119878881725854E0);
        const_f64_as_f64x2!(P4, 4.97494994976747001425E-1);
        const_f64_as_f64x2!(P5, 1.01875663804580931796E-4);

        const_f64_as_f64x2!(Q0, 2.31251620126765340583E1);
        const_f64_as_f64x2!(Q1, 7.11544750618563894466E1);
        const_f64_as_f64x2!(Q2, 8.29875266912776603211E1);
        const_f64_as_f64x2!(Q3, 4.52279145837532221105E1);
        const_f64_as_f64x2!(Q4, 1.12873587189167450590E1);
        // LN2F_HI/LO from fdlibm (Freely Distributable LIBM)
        // Sun Microsystems, Inc. https://www.netlib.org/fdlibm/
        // e_log.c: bit-exact double-double decomposition of ln(2) for f64.
        // Replaced the original f32-literals (0.693359375, -2.12194440e-4)
        // which had ~10 significant digits, causing ~630 ULP error in f64 ln.
        const_f64_as_f64x2!(LN2F_HI, f64::from_bits(0x3FE62E42FEE00000));
        const_f64_as_f64x2!(LN2F_LO, f64::from_bits(0x3DEA39EF35793C76));
        const_f64_as_f64x2!(VM_SQRT2, 1.414213562373095048801);
        const_f64_as_f64x2!(VM_SMALLEST_NORMAL, 2.2250738585072014E-308);

        let x1 = self;
        let x = Self::fraction_2(x1);
        let e = Self::exponent(x1);
        let mask = x.simd_gt(VM_SQRT2 * f64x2::HALF);
        let x = (!mask).select(x + x, x);
        let fe = mask.select(e + Self::ONE, e);
        let x = x - Self::ONE;
        let px = polynomial_5!(x, P0, P1, P2, P3, P4, P5);
        let x2 = x * x;
        let px = x2 * x * px;
        let qx = polynomial_5n!(x, Q0, Q1, Q2, Q3, Q4);
        let res = px / qx;
        let res = fe.fast_mul_add(LN2F_LO, res);
        let res = res + x2.fast_mul_neg_add(f64x2::HALF, x);
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
            // Both get -Inf here. True subnormal inputs (~5e-324..2.225e-308) should
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

        const SUBN_SCALE: f64 = 1.8014398509481984e16;
        const SUBN_CBRT: f64 = 262144.0;
        let tiny = a.simd_lt(Self::from(f64::MIN_POSITIVE));
        let a = tiny.select(a * Self::from(SUBN_SCALE), a);

        let e = Self::exponent(a) + Self::ONE;
        let d = Self::fraction_2(a);

        // C0..C5 from SLEEF's minimax polynomial for 1/cbrt(d) on [0.5, 1.0)
        // Naoki Shibata et al., "SLEEF: A Portable Vectorized Library of C99
        // Mathematical Functions", https://sleef.org / https://github.com/shibatch/sleef
        // Licensed under the Boost Software License 1.0.
        const_f64_as_f64x2!(C0, 2.2307275302496609725722);
        const_f64_as_f64x2!(C1, -3.85841935510444988821632);
        const_f64_as_f64x2!(C2, 6.03990368989458747961407);
        const_f64_as_f64x2!(C3, -5.73353060922947843636166);
        const_f64_as_f64x2!(C4, 2.96155103020039511818595);
        const_f64_as_f64x2!(C5, -0.640245898480692909870982);
        let mut x = polynomial_5!(d, C0, C1, C2, C3, C4, C5);

        // Newton for 1/cbrt: x = x - (d * x^4 - x) / 3.
        let x2 = x * x;
        let x4 = x2 * x2;
        x = x - d.fast_mul_add(x4, -x) * Self::from(1.0 / 3.0);

        // cbrt(d) = d * x^2, then polish
        let mut y = (d * x) * x;
        let yx = y * x;
        let t = Self::from(2.0 / 3.0);
        y = y - t * y * (yx - Self::ONE);

        // Scale by 2^(e/3) = 2^k * 2^(r/3)
        let three = Self::from(3.0);
        let two = Self::from(2.0);
        let neg = e.simd_lt(Self::ZERO);
        let e_adj = neg.select(e - two, e);
        let k = (e_adj / three).trunc();
        let r = e - three * k;
        const_f64_as_f64x2!(CBRT2, 1.2599210498948732);
        const_f64_as_f64x2!(CBRT4, 1.5874010519681994);
        y = r.simd_eq(Self::ONE).select(y * CBRT2, y);
        y = r.simd_eq(two).select(y * CBRT4, y);
        y *= Self::vm_pow2n(k);
        y = tiny.select(y / Self::from(SUBN_CBRT), y);

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
        const_f64_as_f64x2!(R4asin, 2.967721961301243206100E-3);
        const_f64_as_f64x2!(R3asin, -5.634242780008963776856E-1);
        const_f64_as_f64x2!(R2asin, 6.968710824104713396794E0);
        const_f64_as_f64x2!(R1asin, -2.556901049652824852289E1);
        const_f64_as_f64x2!(R0asin, 2.853665548261061424989E1);

        const_f64_as_f64x2!(S3asin, -2.194779531642920639778E1);
        const_f64_as_f64x2!(S2asin, 1.470656354026814941758E2);
        const_f64_as_f64x2!(S1asin, -3.838770957603691357202E2);
        const_f64_as_f64x2!(S0asin, 3.424398657913078477438E2);

        const_f64_as_f64x2!(P5asin, 4.253011369004428248960E-3);
        const_f64_as_f64x2!(P4asin, -6.019598008014123785661E-1);
        const_f64_as_f64x2!(P3asin, 5.444622390564711410273E0);
        const_f64_as_f64x2!(P2asin, -1.626247967210700244449E1);
        const_f64_as_f64x2!(P1asin, 1.956261983317594739197E1);
        const_f64_as_f64x2!(P0asin, -8.198089802484824371615E0);

        const_f64_as_f64x2!(Q4asin, -1.474091372988853791896E1);
        const_f64_as_f64x2!(Q3asin, 7.049610280856842141659E1);
        const_f64_as_f64x2!(Q2asin, -1.471791292232726029859E2);
        const_f64_as_f64x2!(Q1asin, 1.395105614657485689735E2);
        const_f64_as_f64x2!(Q0asin, -4.918853881490881290097E1);

        let xa = self.abs();

        let big = xa.simd_ge(f64x2::splat(0.625));

        let x1 = big.select(f64x2::splat(1.0) - xa, xa * xa);

        let x2 = x1 * x1;
        let x3 = x2 * x1;
        let x4 = x2 * x2;
        let x5 = x4 * x1;

        let do_big = big.any();
        let do_small = !big.all();

        let mut rx = f64x2::default();
        let mut sx = f64x2::default();
        let mut px = f64x2::default();
        let mut qx = f64x2::default();

        if do_big {
            rx = x3.fast_mul_add(R3asin, x2 * R2asin)
                + x4.fast_mul_add(R4asin, x1.fast_mul_add(R1asin, R0asin));
            sx = x3.fast_mul_add(S3asin, x4)
                + x2.fast_mul_add(S2asin, x1.fast_mul_add(S1asin, S0asin));
        }
        if do_small {
            px = x3.fast_mul_add(P3asin, P0asin)
                + x4.fast_mul_add(P4asin, x1 * P1asin)
                + x5.fast_mul_add(P5asin, x2 * P2asin);
            qx = x4.fast_mul_add(Q4asin, x5)
                + x3.fast_mul_add(Q3asin, x1 * Q1asin)
                + x2.fast_mul_add(Q2asin, Q0asin);
        };

        let vx = big.select(rx, px);
        let wx = big.select(sx, qx);

        let y1 = vx / wx * x1;

        let mut z1 = f64x2::default();
        let mut z2 = f64x2::default();
        if do_big {
            let xb = (x1 + x1).sqrt();
            z1 = xb.fast_mul_add(y1, xb);
        }

        if do_small {
            z2 = xa.fast_mul_add(y1, xa);
        }

        // asin
        let z3 = f64x2::FRAC_PI_2 - z1;
        let asin = big.select(z3, z2);
        let asin = asin.flipsign(self);

        asin
    }

    #[inline]
    pub(crate) fn acos_backend(self) -> Self {
        // Based on the Agner Fog "vector class library":
        // https://github.com/vectorclass/version2/blob/master/vectormath_trig.h
        const_f64_as_f64x2!(R4asin, 2.967721961301243206100E-3);
        const_f64_as_f64x2!(R3asin, -5.634242780008963776856E-1);
        const_f64_as_f64x2!(R2asin, 6.968710824104713396794E0);
        const_f64_as_f64x2!(R1asin, -2.556901049652824852289E1);
        const_f64_as_f64x2!(R0asin, 2.853665548261061424989E1);

        const_f64_as_f64x2!(S3asin, -2.194779531642920639778E1);
        const_f64_as_f64x2!(S2asin, 1.470656354026814941758E2);
        const_f64_as_f64x2!(S1asin, -3.838770957603691357202E2);
        const_f64_as_f64x2!(S0asin, 3.424398657913078477438E2);

        const_f64_as_f64x2!(P5asin, 4.253011369004428248960E-3);
        const_f64_as_f64x2!(P4asin, -6.019598008014123785661E-1);
        const_f64_as_f64x2!(P3asin, 5.444622390564711410273E0);
        const_f64_as_f64x2!(P2asin, -1.626247967210700244449E1);
        const_f64_as_f64x2!(P1asin, 1.956261983317594739197E1);
        const_f64_as_f64x2!(P0asin, -8.198089802484824371615E0);

        const_f64_as_f64x2!(Q4asin, -1.474091372988853791896E1);
        const_f64_as_f64x2!(Q3asin, 7.049610280856842141659E1);
        const_f64_as_f64x2!(Q2asin, -1.471791292232726029859E2);
        const_f64_as_f64x2!(Q1asin, 1.395105614657485689735E2);
        const_f64_as_f64x2!(Q0asin, -4.918853881490881290097E1);

        let xa = self.abs();

        let big = xa.simd_ge(f64x2::splat(0.625));

        let x1 = big.select(f64x2::splat(1.0) - xa, xa * xa);

        let x2 = x1 * x1;
        let x3 = x2 * x1;
        let x4 = x2 * x2;
        let x5 = x4 * x1;

        let do_big = big.any();
        let do_small = !big.all();

        let mut rx = f64x2::default();
        let mut sx = f64x2::default();
        let mut px = f64x2::default();
        let mut qx = f64x2::default();

        if do_big {
            rx = x3.fast_mul_add(R3asin, x2 * R2asin)
                + x4.fast_mul_add(R4asin, x1.fast_mul_add(R1asin, R0asin));
            sx = x3.fast_mul_add(S3asin, x4)
                + x2.fast_mul_add(S2asin, x1.fast_mul_add(S1asin, S0asin));
        }
        if do_small {
            px = x3.fast_mul_add(P3asin, P0asin)
                + x4.fast_mul_add(P4asin, x1 * P1asin)
                + x5.fast_mul_add(P5asin, x2 * P2asin);
            qx = x4.fast_mul_add(Q4asin, x5)
                + x3.fast_mul_add(Q3asin, x1 * Q1asin)
                + x2.fast_mul_add(Q2asin, Q0asin);
        };

        let vx = big.select(rx, px);
        let wx = big.select(sx, qx);

        let y1 = vx / wx * x1;

        let mut z1 = f64x2::default();
        let mut z2 = f64x2::default();
        if do_big {
            let xb = (x1 + x1).sqrt();
            z1 = xb.fast_mul_add(y1, xb);
        }

        if do_small {
            z2 = xa.fast_mul_add(y1, xa);
        }

        // acos
        let z3 = self.simd_lt(f64x2::ZERO).select(f64x2::PI - z1, z1);
        let z4 = f64x2::FRAC_PI_2 - z2.flipsign(self);
        let acos = big.select(z3, z4);

        acos
    }

    #[inline]
    pub(crate) fn atan_backend(self) -> Self {
        // Based on the Agner Fog "vector class library":
        // https://github.com/vectorclass/version2/blob/master/vectormath_trig.h
        const_f64_as_f64x2!(MORE_BITS, 6.123233995736765886130E-17);
        const_f64_as_f64x2!(MORE_BITS_O2, 6.123233995736765886130E-17 * 0.5);
        const_f64_as_f64x2!(T3PO8, core::f64::consts::SQRT_2 + 1.0);

        const_f64_as_f64x2!(P4atan, -8.750608600031904122785E-1);
        const_f64_as_f64x2!(P3atan, -1.615753718733365076637E1);
        const_f64_as_f64x2!(P2atan, -7.500855792314704667340E1);
        const_f64_as_f64x2!(P1atan, -1.228866684490136173410E2);
        const_f64_as_f64x2!(P0atan, -6.485021904942025371773E1);

        const_f64_as_f64x2!(Q4atan, 2.485846490142306297962E1);
        const_f64_as_f64x2!(Q3atan, 1.650270098316988542046E2);
        const_f64_as_f64x2!(Q2atan, 4.328810604912902668951E2);
        const_f64_as_f64x2!(Q1atan, 4.853903996359136964868E2);
        const_f64_as_f64x2!(Q0atan, 1.945506571482613964425E2);

        let t = self.abs();

        // small:  t < 0.66
        // medium: t <= t <= 2.4142 (1+sqrt(2))
        // big:    t > 2.4142
        let notbig = t.simd_le(T3PO8);
        let notsmal = t.simd_ge(Self::splat(0.66));

        let mut s = notbig.select(Self::FRAC_PI_4, Self::FRAC_PI_2);
        s = notsmal & s;
        let mut fac = notbig.select(MORE_BITS_O2, MORE_BITS);
        fac = notsmal & fac;

        // small:  z = t / 1.0;
        // medium: z = (t-1.0) / (t+1.0);
        // big:    z = -1.0 / t;
        let mut a = notbig & t;
        a = notsmal.select(a - Self::ONE, a);
        let mut b = notbig & Self::ONE;
        b = notsmal.select(b + t, b);
        let z = a / b;

        let zz = z * z;

        let px = polynomial_4!(zz, P0atan, P1atan, P2atan, P3atan, P4atan);
        let qx = polynomial_5n!(zz, Q0atan, Q1atan, Q2atan, Q3atan, Q4atan);

        let mut re = (px / qx).fast_mul_add(z * zz, z);
        re += s + fac;

        // get sign bit
        re = (self.is_sign_negative()).select(-re, re);

        re
    }

    #[inline]
    pub(crate) fn atan2_backend(self, x: Self) -> Self {
        // Based on the Agner Fog "vector class library":
        // https://github.com/vectorclass/version2/blob/master/vectormath_trig.h
        const_f64_as_f64x2!(MORE_BITS, 6.123233995736765886130E-17);
        const_f64_as_f64x2!(MORE_BITS_O2, 6.123233995736765886130E-17 * 0.5);
        const_f64_as_f64x2!(T3PO8, core::f64::consts::SQRT_2 + 1.0);

        const_f64_as_f64x2!(P4atan, -8.750608600031904122785E-1);
        const_f64_as_f64x2!(P3atan, -1.615753718733365076637E1);
        const_f64_as_f64x2!(P2atan, -7.500855792314704667340E1);
        const_f64_as_f64x2!(P1atan, -1.228866684490136173410E2);
        const_f64_as_f64x2!(P0atan, -6.485021904942025371773E1);

        const_f64_as_f64x2!(Q4atan, 2.485846490142306297962E1);
        const_f64_as_f64x2!(Q3atan, 1.650270098316988542046E2);
        const_f64_as_f64x2!(Q2atan, 4.328810604912902668951E2);
        const_f64_as_f64x2!(Q1atan, 4.853903996359136964868E2);
        const_f64_as_f64x2!(Q0atan, 1.945506571482613964425E2);

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

        // x = y = 0 gives NAN here
        let t = y2 / x2;

        // small:  t < 0.66
        // medium: t <= t <= 2.4142 (1+sqrt(2))
        // big:    t > 2.4142
        let notbig = t.simd_le(T3PO8);
        let notsmal = t.simd_ge(Self::splat(0.66));

        let mut s = notbig.select(Self::FRAC_PI_4, Self::FRAC_PI_2);
        s = notsmal & s;
        let mut fac = notbig.select(MORE_BITS_O2, MORE_BITS);
        fac = notsmal & fac;

        // small:  z = t / 1.0;
        // medium: z = (t-1.0) / (t+1.0);
        // big:    z = -1.0 / t;
        let mut a = notbig & t;
        a = notsmal.select(a - Self::ONE, a);
        let mut b = notbig & Self::ONE;
        b = notsmal.select(b + t, b);
        let z = a / b;

        let zz = z * z;

        let px = polynomial_4!(zz, P0atan, P1atan, P2atan, P3atan, P4atan);
        let qx = polynomial_5n!(zz, Q0atan, Q1atan, Q2atan, Q3atan, Q4atan);

        let mut re = (px / qx).fast_mul_add(z * zz, z);
        re += s + fac;

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

        const_f64_as_f64x2!(P0sin, -1.66666666666666307295E-1);
        const_f64_as_f64x2!(P1sin, 8.33333333332211858878E-3);
        const_f64_as_f64x2!(P2sin, -1.98412698295895385996E-4);
        const_f64_as_f64x2!(P3sin, 2.75573136213857245213E-6);
        const_f64_as_f64x2!(P4sin, -2.50507477628578072866E-8);
        const_f64_as_f64x2!(P5sin, 1.58962301576546568060E-10);

        const_f64_as_f64x2!(P0cos, 4.16666666666665929218E-2);
        const_f64_as_f64x2!(P1cos, -1.38888888888730564116E-3);
        const_f64_as_f64x2!(P2cos, 2.48015872888517045348E-5);
        const_f64_as_f64x2!(P3cos, -2.75573141792967388112E-7);
        const_f64_as_f64x2!(P4cos, 2.08757008419747316778E-9);
        const_f64_as_f64x2!(P5cos, -1.13585365213876817300E-11);

        const_f64_as_f64x2!(DP1, 7.853981554508209228515625E-1 * 2.);
        const_f64_as_f64x2!(DP2, 7.94662735614792836714E-9 * 2.);
        const_f64_as_f64x2!(DP3, 3.06161699786838294307E-17 * 2.);

        const_f64_as_f64x2!(TWO_OVER_PI, 2.0 / core::f64::consts::PI);

        let xa = self.abs();

        let y = (xa * TWO_OVER_PI).round_ties_even();
        let q = y.round_int();

        let x = y.fast_mul_neg_add(DP3, y.fast_mul_neg_add(DP2, y.fast_mul_neg_add(DP1, xa)));

        let x2 = x * x;
        let mut s = polynomial_5!(x2, P0sin, P1sin, P2sin, P3sin, P4sin, P5sin);
        let mut c = polynomial_5!(x2, P0cos, P1cos, P2cos, P3cos, P4cos, P5cos);
        s = (x * x2).fast_mul_add(s, x);
        c = (x2 * x2).fast_mul_add(c, x2.fast_mul_neg_add(f64x2::from(0.5), f64x2::from(1.0)));

        let swap = !((q & i64x2::from(1)).simd_eq(i64x2::from(0)));

        let mut overflow: f64x2 = cast(q.simd_gt(i64x2::from(0x80000000000000)));
        overflow &= xa.is_finite();
        s = overflow.select(f64x2::from(0.0), s);
        c = overflow.select(f64x2::from(1.0), c);

        // calc sin
        let mut sin1 = cast::<_, f64x2>(swap).select(c, s);
        let sign_sin: i64x2 = (q << 62) ^ cast::<_, i64x2>(self);
        sin1 = sin1.flipsign(cast(sign_sin));

        // calc cos
        let mut cos1 = cast::<_, f64x2>(swap).select(s, c);
        let sign_cos: i64x2 = ((q + i64x2::from(1)) & i64x2::from(2)) << 62;
        cos1 ^= cast::<_, f64x2>(sign_cos);

        // IEEE 754: sin/cos(±∞) = NaN, sin/cos(NaN) = NaN
        let finite = self.is_finite();
        let nan = Self::splat(f64::NAN);
        let sin_final = finite.select(sin1, nan);
        let cos_final = finite.select(cos1, nan);

        (sin_final, cos_final)
    }

    #[inline]
    pub(crate) fn asin_acos_backend(self) -> (Self, Self) {
        // Based on the Agner Fog "vector class library":
        // https://github.com/vectorclass/version2/blob/master/vectormath_trig.h
        const_f64_as_f64x2!(R4asin, 2.967721961301243206100E-3);
        const_f64_as_f64x2!(R3asin, -5.634242780008963776856E-1);
        const_f64_as_f64x2!(R2asin, 6.968710824104713396794E0);
        const_f64_as_f64x2!(R1asin, -2.556901049652824852289E1);
        const_f64_as_f64x2!(R0asin, 2.853665548261061424989E1);

        const_f64_as_f64x2!(S3asin, -2.194779531642920639778E1);
        const_f64_as_f64x2!(S2asin, 1.470656354026814941758E2);
        const_f64_as_f64x2!(S1asin, -3.838770957603691357202E2);
        const_f64_as_f64x2!(S0asin, 3.424398657913078477438E2);

        const_f64_as_f64x2!(P5asin, 4.253011369004428248960E-3);
        const_f64_as_f64x2!(P4asin, -6.019598008014123785661E-1);
        const_f64_as_f64x2!(P3asin, 5.444622390564711410273E0);
        const_f64_as_f64x2!(P2asin, -1.626247967210700244449E1);
        const_f64_as_f64x2!(P1asin, 1.956261983317594739197E1);
        const_f64_as_f64x2!(P0asin, -8.198089802484824371615E0);

        const_f64_as_f64x2!(Q4asin, -1.474091372988853791896E1);
        const_f64_as_f64x2!(Q3asin, 7.049610280856842141659E1);
        const_f64_as_f64x2!(Q2asin, -1.471791292232726029859E2);
        const_f64_as_f64x2!(Q1asin, 1.395105614657485689735E2);
        const_f64_as_f64x2!(Q0asin, -4.918853881490881290097E1);

        let xa = self.abs();

        let big = xa.simd_ge(f64x2::splat(0.625));

        let x1 = big.select(f64x2::splat(1.0) - xa, xa * xa);

        let x2 = x1 * x1;
        let x3 = x2 * x1;
        let x4 = x2 * x2;
        let x5 = x4 * x1;

        let do_big = big.any();
        let do_small = !big.all();

        let mut rx = f64x2::default();
        let mut sx = f64x2::default();
        let mut px = f64x2::default();
        let mut qx = f64x2::default();

        if do_big {
            rx = x3.fast_mul_add(R3asin, x2 * R2asin)
                + x4.fast_mul_add(R4asin, x1.fast_mul_add(R1asin, R0asin));
            sx = x3.fast_mul_add(S3asin, x4)
                + x2.fast_mul_add(S2asin, x1.fast_mul_add(S1asin, S0asin));
        }
        if do_small {
            px = x3.fast_mul_add(P3asin, P0asin)
                + x4.fast_mul_add(P4asin, x1 * P1asin)
                + x5.fast_mul_add(P5asin, x2 * P2asin);
            qx = x4.fast_mul_add(Q4asin, x5)
                + x3.fast_mul_add(Q3asin, x1 * Q1asin)
                + x2.fast_mul_add(Q2asin, Q0asin);
        };

        let vx = big.select(rx, px);
        let wx = big.select(sx, qx);

        let y1 = vx / wx * x1;

        let mut z1 = f64x2::default();
        let mut z2 = f64x2::default();
        if do_big {
            let xb = (x1 + x1).sqrt();
            z1 = xb.fast_mul_add(y1, xb);
        }

        if do_small {
            z2 = xa.fast_mul_add(y1, xa);
        }

        // asin
        let z3 = f64x2::FRAC_PI_2 - z1;
        let asin = big.select(z3, z2);
        let asin = asin.flipsign(self);

        // acos
        let z3 = self.simd_lt(f64x2::ZERO).select(f64x2::PI - z1, z1);
        let z4 = f64x2::FRAC_PI_2 - z2.flipsign(self);
        let acos = big.select(z3, z4);

        (asin, acos)
    }

    #[inline]
    pub(crate) fn exp_m1_backend(self) -> Self {
        const_f64_as_f64x2!(P2, 1.0 / 2.0);
        const_f64_as_f64x2!(P3, 1.0 / 6.0);
        const_f64_as_f64x2!(P4, 1.0 / 24.0);
        const_f64_as_f64x2!(P5, 1.0 / 120.0);
        const_f64_as_f64x2!(P6, 1.0 / 720.0);
        const_f64_as_f64x2!(P7, 1.0 / 5040.0);
        const_f64_as_f64x2!(P8, 1.0 / 40320.0);
        const_f64_as_f64x2!(P9, 1.0 / 362880.0);
        const_f64_as_f64x2!(P10, 1.0 / 3628800.0);
        const_f64_as_f64x2!(P11, 1.0 / 39916800.0);
        const_f64_as_f64x2!(P12, 1.0 / 479001600.0);
        const_f64_as_f64x2!(P13, 1.0 / 6227020800.0);
        // LN2D_HI/LO: double-double decomposition of ln(2) for exp range reduction,
        // following fdlibm's approach (Sun Microsystems, https://www.netlib.org/fdlibm/ e_exp.c).
        const_f64_as_f64x2!(LN2D_HI, 0.693145751953125);
        const_f64_as_f64x2!(LN2D_LO, 1.42860682030941723212E-6);
        // x < -37.429: e^x < 2⁻⁵⁴, exp_m1(x) = -1.0 exactly (mantissa exhaustion)
        // IEEE simd_lt returns false for NaN, so NaN lanes can't reach here.
        // -inf is < -37.429, and exp_m1(-inf) = -1.0, also correct.
        if self.simd_lt(Self::from(-37.429)).all() {
            return Self::from(-1.0);
        }
        // max_x = ln(f64::MAX) ≈ 709.7827129, max_r = 1023 (IEEE max normal
        // exponent) min_x = -1074.5 ln(2) ≈ -744.79: min r for vm_pow2n to
        // construct subnormal
        let max_x = Self::from(709.783);
        let min_x = Self::from(-744.79);
        let max_r = Self::from(1023.0);
        let r = (self * Self::LOG2_E).round_ties_even();
        let big = r.simd_gt(max_r);
        let r_safe = big.select(max_r, r);
        let excess = r - max_r;
        let excess = big.select(excess, Self::ZERO);
        let scale = Self::vm_pow2n(excess);
        let x = r.fast_mul_neg_add(LN2D_HI, self);
        let x = r.fast_mul_neg_add(LN2D_LO, x);
        let z = polynomial_13!(x, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13);
        let n2 = Self::vm_pow2n(r_safe);
        let exp_val = (z + Self::ONE) * scale * n2;
        // When r == 0, z is already e^x - 1 from the Taylor poly.
        // Computing (z+1) - 1 would lose low bits for small x (catastrophic
        // cancellation at z ~ 0), so keep z directly.
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
        // where u = 1+x. From MUSL libc (Rich Felker et al., https://musl.libc.org) src/math/log1p.c
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
        const_f64_as_f64x2!(P0, 1.0);
        const_f64_as_f64x2!(P1, 1.0 / 6.0);
        const_f64_as_f64x2!(P2, 1.0 / 120.0);
        const_f64_as_f64x2!(P3, 1.0 / 5040.0);
        const_f64_as_f64x2!(P4, 1.0 / 362880.0);
        const_f64_as_f64x2!(P5, 1.0 / 39916800.0);
        const_f64_as_f64x2!(P6, 1.0 / 6227020800.0);
        let a = self.abs();
        // |x| < 0.5: Taylor poly; last truncation term < 1 ULP at x=0.5 for both types
        let small = a.simd_lt(f64x2::from(0.5));
        let t = a * a;
        let poly = a * polynomial_6!(t, P0, P1, P2, P3, P4, P5, P6);
        let exp_based = {
            let e = a.exp();
            (e - Self::ONE / e) * Self::HALF
        };
        let result = small.select(poly, exp_based);
        result.flipsign(self)
    }

    #[inline]
    pub(crate) fn cosh_backend(self) -> Self {
        const_f64_as_f64x2!(P0, 1.0);
        const_f64_as_f64x2!(P1, 1.0 / 2.0);
        const_f64_as_f64x2!(P2, 1.0 / 24.0);
        const_f64_as_f64x2!(P3, 1.0 / 720.0);
        const_f64_as_f64x2!(P4, 1.0 / 40320.0);
        const_f64_as_f64x2!(P5, 1.0 / 3628800.0);
        const_f64_as_f64x2!(P6, 1.0 / 479001600.0);
        const_f64_as_f64x2!(P7, 1.0 / 87178291200.0);
        let a = self.abs();
        // |x| < 0.5: Taylor poly; last truncation term < 1 ULP at x=0.5 for both types
        let small = a.simd_lt(f64x2::from(0.5));
        let t = a * a;
        let poly = polynomial_7!(t, P0, P1, P2, P3, P4, P5, P6, P7);
        let exp_based = {
            let e = a.exp();
            (e + Self::ONE / e) * Self::HALF
        };
        small.select(poly, exp_based)
    }

    #[inline]
    pub(crate) fn tanh_backend(self) -> Self {
        // |x| < 5e-8: tanh(x) ≈ x, error x³/3 < 16·ULP(x)
        // bound: x² < 48·2⁻⁵² → x < 1.03e-7; 5e-8 has 2× margin
        // |x| > 19.062: tanh(x) = ±1 to f64 precision (e⁻²ˣ < 2⁻⁵⁴)
        let a = self.abs();
        let large = a.simd_gt(f64x2::from(19.062));
        if large.all() {
            return Self::ONE.flipsign(self);
        }
        let small = a.simd_lt(f64x2::from(5e-8));
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
impl f64x2 {
    #[inline]
    fn vm_pow2n(self) -> Self {
        const_f64_as_f64x2!(pow2_52, 4503599627370496.0);
        const_f64_as_f64x2!(bias, 1023.0);
        let a = self + (bias + pow2_52);
        let c = cast::<_, i64x2>(a) << 52;
        let std_result = cast::<_, f64x2>(c);

        let min_exp = f64x2::from(-1022.0);
        let is_sub = self.simd_lt(min_exp);
        if is_sub.any() {
            let valid = self.simd_ge(f64x2::from(-1074.0));
            let shift_f = self + f64x2::from(1074.0);
            let mut shift_i = shift_f.trunc_int();
            shift_i = cast::<_, i64x2>(valid).select(shift_i, i64x2::ZERO);
            let mantissa = i64x2::ONE << shift_i;
            let sub_result = cast::<_, f64x2>(mantissa);
            let sub_result = valid.select(sub_result, f64x2::ZERO);
            is_sub.select(sub_result, std_result)
        } else {
            std_result
        }
    }

    #[inline]
    fn exponent(self) -> f64x2 {
        const_f64_as_f64x2!(pow2_52, 4503599627370496.0);
        const_f64_as_f64x2!(bias, 1023.0);
        let a = cast::<_, u64x2>(self);
        let b = a >> 52;
        let c = b | cast::<_, u64x2>(pow2_52);
        let d = cast::<_, f64x2>(c);
        let e = d - (pow2_52 + bias);
        e
    }

    #[inline]
    fn fraction_2(self) -> Self {
        let t1 = cast::<_, u64x2>(self);
        let t2 = cast::<_, u64x2>(
            (t1 & u64x2::from(0x000FFFFFFFFFFFFF)) | u64x2::from(0x3FE0000000000000),
        );
        cast::<_, f64x2>(t2)
    }

    #[inline]
    fn is_zero_or_subnormal(self) -> Self {
        let t = cast::<_, i64x2>(self);
        let t = t & i64x2::splat(0x7FF0000000000000);
        let mask = t.simd_eq(i64x2::splat(0));
        cast::<_, f64x2>(mask)
    }

    #[inline]
    fn infinity() -> Self {
        cast::<_, f64x2>(i64x2::splat(0x7FF0000000000000))
    }

    #[inline]
    fn nan_log() -> Self {
        cast::<_, f64x2>(i64x2::splat(0x7FF8000000000000 | 0x101 << 29))
    }

    #[inline]
    fn nan_pow() -> Self {
        cast::<_, f64x2>(i64x2::splat(0x7FF8000000000000 | 0x101 << 29))
    }

    #[inline]
    pub(crate) fn from_i32x4_lower2_backend(v: i32x4) -> Self {
        pick! {
          if #[cfg(target_feature="sse2")] {
            Self(_mm_cvtepi32_pd(v.0) )
          } else if #[cfg(target_feature="simd128")] {
            Self(f64x2_convert_low_i32x4(v.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))] {
            Self(vcvtq_f64_s64(vmovl_s32(vget_low_s32(v.0))))
          } else {
            Self::from_array([
                v.as_array()[0] as f64,
                v.as_array()[1] as f64,
            ])
          }
        }
    }
}
}
