use crate::backend_imports::*;

#[cfg(not(any(
    target_feature = "sse",
    target_feature = "simd128",
    all(target_feature = "neon", target_arch = "aarch64")
)))]
#[repr(C, align(16))]
#[derive(Clone, Copy)]
pub(crate) struct Inner {
    pub arr: [f32; 4],
}

safe_target_feature! {
impl SimdBackend for f32x4 {
    pick! {
      if #[cfg(target_feature="sse")] {
        type Inner = __m128;
      } else if #[cfg(target_feature="simd128")] {
        type Inner = v128;
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))] {
        type Inner = float32x4_t;
      } else {
        type Inner = Inner;
      }
    }

    #[inline]
    pub(crate) fn simd_eq_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse")] {
            Self(_mm_cmpeq_ps(self.0, rhs.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(f32x4_eq(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vreinterpretq_f32_u32(vceqq_f32(self.0, rhs.0)))
          } else {
            Self::from_array([
              if self.0.arr[0] == rhs.0.arr[0] { f32::from_bits(u32::MAX) } else { 0.0 },
              if self.0.arr[1] == rhs.0.arr[1] { f32::from_bits(u32::MAX) } else { 0.0 },
              if self.0.arr[2] == rhs.0.arr[2] { f32::from_bits(u32::MAX) } else { 0.0 },
              if self.0.arr[3] == rhs.0.arr[3] { f32::from_bits(u32::MAX) } else { 0.0 },
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn simd_ne_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse")] {
            Self(_mm_cmpneq_ps(self.0, rhs.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(f32x4_ne(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vreinterpretq_f32_u32(vmvnq_u32(vceqq_f32(self.0, rhs.0))))
          } else {
            Self::from_array([
              if self.0.arr[0] != rhs.0.arr[0] { f32::from_bits(u32::MAX) } else { 0.0 },
              if self.0.arr[1] != rhs.0.arr[1] { f32::from_bits(u32::MAX) } else { 0.0 },
              if self.0.arr[2] != rhs.0.arr[2] { f32::from_bits(u32::MAX) } else { 0.0 },
              if self.0.arr[3] != rhs.0.arr[3] { f32::from_bits(u32::MAX) } else { 0.0 },
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn simd_lt_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse")] {
            Self(_mm_cmplt_ps(self.0, rhs.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(f32x4_lt(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vreinterpretq_f32_u32(vcltq_f32(self.0, rhs.0)))
          } else {
            Self::from_array([
              if self.0.arr[0] < rhs.0.arr[0] { f32::from_bits(u32::MAX) } else { 0.0 },
              if self.0.arr[1] < rhs.0.arr[1] { f32::from_bits(u32::MAX) } else { 0.0 },
              if self.0.arr[2] < rhs.0.arr[2] { f32::from_bits(u32::MAX) } else { 0.0 },
              if self.0.arr[3] < rhs.0.arr[3] { f32::from_bits(u32::MAX) } else { 0.0 },
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn simd_gt_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse")] {
            Self(_mm_cmpgt_ps(self.0, rhs.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(f32x4_gt(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vreinterpretq_f32_u32(vcgtq_f32(self.0, rhs.0)))
          } else {
            Self::from_array([
              if self.0.arr[0] > rhs.0.arr[0] { f32::from_bits(u32::MAX) } else { 0.0 },
              if self.0.arr[1] > rhs.0.arr[1] { f32::from_bits(u32::MAX) } else { 0.0 },
              if self.0.arr[2] > rhs.0.arr[2] { f32::from_bits(u32::MAX) } else { 0.0 },
              if self.0.arr[3] > rhs.0.arr[3] { f32::from_bits(u32::MAX) } else { 0.0 },
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn simd_le_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse")] {
            Self(_mm_cmple_ps(self.0, rhs.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(f32x4_le(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vreinterpretq_f32_u32(vcleq_f32(self.0, rhs.0)))
          } else {
            Self::from_array([
              if self.0.arr[0] <= rhs.0.arr[0] { f32::from_bits(u32::MAX) } else { 0.0 },
              if self.0.arr[1] <= rhs.0.arr[1] { f32::from_bits(u32::MAX) } else { 0.0 },
              if self.0.arr[2] <= rhs.0.arr[2] { f32::from_bits(u32::MAX) } else { 0.0 },
              if self.0.arr[3] <= rhs.0.arr[3] { f32::from_bits(u32::MAX) } else { 0.0 },
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn simd_ge_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse")] {
            Self(_mm_cmpge_ps(self.0, rhs.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(f32x4_ge(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vreinterpretq_f32_u32(vcgeq_f32(self.0, rhs.0)))
          } else {
            Self::from_array([
              if self.0.arr[0] >= rhs.0.arr[0] { f32::from_bits(u32::MAX) } else { 0.0 },
              if self.0.arr[1] >= rhs.0.arr[1] { f32::from_bits(u32::MAX) } else { 0.0 },
              if self.0.arr[2] >= rhs.0.arr[2] { f32::from_bits(u32::MAX) } else { 0.0 },
              if self.0.arr[3] >= rhs.0.arr[3] { f32::from_bits(u32::MAX) } else { 0.0 },
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn select_backend(self, if_true: Self, if_false: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse4.1")] {
            Self(_mm_blendv_ps(if_false.0, if_true.0, self.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(v128_bitselect(if_true.0, if_false.0, self.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vbslq_f32(vreinterpretq_u32_f32(self.0), if_true.0, if_false.0))
          } else {
            generic_bit_blend(self, if_true, if_false)
          }
        }
    }

    #[inline]
    pub(crate) fn to_bitmask_backend(self) -> u32 {
        pick! {
          if #[cfg(target_feature="sse")] {
            _mm_movemask_ps(self.0) as u32
          } else if #[cfg(target_feature="simd128")] {
            u32x4_bitmask(self.0) as u32
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            // set all to 1 if top bit is set, else 0
            let masked = vcltq_s32( vreinterpretq_s32_f32(self.0), vdupq_n_s32(0));

            // select the right bit out of each lane
            let selectbit = u32x4::from_array([1, 2, 4, 8]).0;
            let r = vandq_u32(masked, selectbit);

            // horizontally add the 16-bit lanes
            vaddvq_u32(r) as u32
          } else {
            (((self.0.arr[0].to_bits() as i32) < 0) as u32) |
            (((self.0.arr[1].to_bits() as i32) < 0) as u32) << 1 |
            (((self.0.arr[2].to_bits() as i32) < 0) as u32) << 2 |
            (((self.0.arr[3].to_bits() as i32) < 0) as u32) << 3
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
            u32x4_all_true(self.0)
          } else {
            // four lanes
            self.to_bitmask() == 0b1111
          }
        }
    }

    #[inline]
    pub(crate) fn interleave_low_backend(self, other: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse")] {
            Self(_mm_unpacklo_ps(self.0, other.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(u32x4_shuffle::<0, 4, 1, 5>(self.0, other.0))
          } else if #[cfg(all(target_feature="neon", target_arch="aarch64"))]{
            Self(vzip1q_f32(self.0, other.0))
          } else {
            Self::from_array([
              self.0.arr[0],
              other.0.arr[0],
              self.0.arr[1],
              other.0.arr[1],
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn interleave_high_backend(self, other: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse")] {
            Self(_mm_unpackhi_ps(self.0, other.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(u32x4_shuffle::<2, 6, 3, 7>(self.0, other.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vzip2q_f32(self.0, other.0))
          } else {
            Self::from_array([
              self.0.arr[2],
              other.0.arr[2],
              self.0.arr[3],
              other.0.arr[3],
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn transpose_backend(data: [f32x4; 4]) -> [f32x4; 4] {
        pick! {
          if #[cfg(target_feature="sse")] {
            let mut e0 = data[0];
            let mut e1 = data[1];
            let mut e2 = data[2];
            let mut e3 = data[3];

            _MM_TRANSPOSE4_PS(&mut e0.0, &mut e1.0, &mut e2.0, &mut e3.0);

            [e0, e1, e2, e3]
          } else if #[cfg(any(all(target_feature="neon",target_arch="aarch64"), target_feature="simd128"))] {
            let a = data[0].interleave_low(data[2]);
            let b = data[1].interleave_low(data[3]);
            let c = data[0].interleave_high(data[2]);
            let d = data[1].interleave_high(data[3]);

            [
              a.interleave_low(b),
              a.interleave_high(b),
              c.interleave_low(d),
              c.interleave_high(d),
            ]
          } else {
            #[inline(always)]
            fn transpose_column(data: &[f32x4; 4], index: usize) -> f32x4 {
              f32x4::from_array([
                data[0].as_array()[index],
                data[1].as_array()[index],
                data[2].as_array()[index],
                data[3].as_array()[index],
              ])
            }

            [
              transpose_column(&data, 0),
              transpose_column(&data, 1),
              transpose_column(&data, 2),
              transpose_column(&data, 3),
            ]
          }
        }
    }
}
}

safe_target_feature! {
// Backend of SIMD float functions.
impl f32x4 {
    #[inline]
    pub(crate) fn neg_backend(self) -> Self {
        pick! {
          if #[cfg(target_feature="sse")] {
            Self(_mm_xor_ps(self.0, Self::splat(-0.0).0))
          } else if #[cfg(target_feature="simd128")] {
            Self(f32x4_neg(self.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vnegq_f32(self.0))
          } else {
            Self::from_array([
              -self.0.arr[0],
              -self.0.arr[1],
              -self.0.arr[2],
              -self.0.arr[3],
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn not_backend(self) -> Self {
        self ^ cast::<u128, f32x4>(u128::MAX)
    }

    #[inline]
    pub(crate) fn add_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse")] {
            Self(_mm_add_ps(self.0, rhs.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(f32x4_add(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vaddq_f32(self.0, rhs.0))
          } else {
            Self::from_array([
              self.0.arr[0] + rhs.0.arr[0],
              self.0.arr[1] + rhs.0.arr[1],
              self.0.arr[2] + rhs.0.arr[2],
              self.0.arr[3] + rhs.0.arr[3],
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn sub_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse")] {
            Self(_mm_sub_ps(self.0, rhs.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(f32x4_sub(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vsubq_f32(self.0, rhs.0))
          } else {
            Self::from_array([
              self.0.arr[0] - rhs.0.arr[0],
              self.0.arr[1] - rhs.0.arr[1],
              self.0.arr[2] - rhs.0.arr[2],
              self.0.arr[3] - rhs.0.arr[3],
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn mul_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse")] {
            Self(_mm_mul_ps(self.0, rhs.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(f32x4_mul(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vmulq_f32(self.0, rhs.0))
          } else {
            Self::from_array([
              self.0.arr[0] * rhs.0.arr[0],
              self.0.arr[1] * rhs.0.arr[1],
              self.0.arr[2] * rhs.0.arr[2],
              self.0.arr[3] * rhs.0.arr[3],
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn div_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse")] {
            Self(_mm_div_ps(self.0, rhs.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(f32x4_div(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vdivq_f32(self.0, rhs.0))
          } else {
            Self::from_array([
              self.0.arr[0] / rhs.0.arr[0],
              self.0.arr[1] / rhs.0.arr[1],
              self.0.arr[2] / rhs.0.arr[2],
              self.0.arr[3] / rhs.0.arr[3],
            ])
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
        ])
    }

    #[inline]
    pub(crate) fn bitand_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse")] {
            Self(_mm_and_ps(self.0, rhs.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(v128_and(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(self.0), vreinterpretq_u32_f32(rhs.0))))
          } else {
            Self::from_array([
              f32::from_bits(self.0.arr[0].to_bits() & rhs.0.arr[0].to_bits()),
              f32::from_bits(self.0.arr[1].to_bits() & rhs.0.arr[1].to_bits()),
              f32::from_bits(self.0.arr[2].to_bits() & rhs.0.arr[2].to_bits()),
              f32::from_bits(self.0.arr[3].to_bits() & rhs.0.arr[3].to_bits()),
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn bitor_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse")] {
            Self(_mm_or_ps(self.0, rhs.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(v128_or(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(self.0), vreinterpretq_u32_f32(rhs.0))))
          } else {
            Self::from_array([
              f32::from_bits(self.0.arr[0].to_bits() | rhs.0.arr[0].to_bits()),
              f32::from_bits(self.0.arr[1].to_bits() | rhs.0.arr[1].to_bits()),
              f32::from_bits(self.0.arr[2].to_bits() | rhs.0.arr[2].to_bits()),
              f32::from_bits(self.0.arr[3].to_bits() | rhs.0.arr[3].to_bits()),
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn bitxor_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse")] {
            Self(_mm_xor_ps(self.0, rhs.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(v128_xor(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(self.0), vreinterpretq_u32_f32(rhs.0))))
          } else {
            Self::from_array([
              f32::from_bits(self.0.arr[0].to_bits() ^ rhs.0.arr[0].to_bits()),
              f32::from_bits(self.0.arr[1].to_bits() ^ rhs.0.arr[1].to_bits()),
              f32::from_bits(self.0.arr[2].to_bits() ^ rhs.0.arr[2].to_bits()),
              f32::from_bits(self.0.arr[3].to_bits() ^ rhs.0.arr[3].to_bits()),
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn reduce_add_backend(self) -> f32 {
        let arr: [f32; 4] = cast(self);
        arr.iter().sum()
    }

    #[inline]
    pub(crate) fn reduce_mul_backend(self) -> f32 {
        let arr: [f32; 4] = cast(self);
        arr.iter().product()
    }

    #[inline]
    pub(crate) fn bitselect_backend(self, if_one: Self, if_zero: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse2")] {
            Self(_mm_or_ps(_mm_and_ps(if_one.0, self.0), _mm_andnot_ps(self.0, if_zero.0)))
          } else if #[cfg(target_feature="simd128")] {
            Self(v128_bitselect(if_one.0, if_zero.0, self.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vbslq_f32(vreinterpretq_u32_f32(self.0), if_one.0, if_zero.0))
          } else {
            generic_bit_blend(self, if_one, if_zero)
          }
        }
    }

    #[inline]
    pub(crate) fn is_nan_backend(self) -> Self {
        pick! {
          if #[cfg(target_feature="sse")] {
            Self(_mm_cmpunord_ps(self.0, self.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(f32x4_ne(self.0, self.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vreinterpretq_f32_u32(vmvnq_u32(vceqq_f32(self.0, self.0))))
          } else {
            Self::from_array([
              if self.0.arr[0].is_nan() { f32::from_bits(u32::MAX) } else { 0.0 },
              if self.0.arr[1].is_nan() { f32::from_bits(u32::MAX) } else { 0.0 },
              if self.0.arr[2].is_nan() { f32::from_bits(u32::MAX) } else { 0.0 },
              if self.0.arr[3].is_nan() { f32::from_bits(u32::MAX) } else { 0.0 },
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn is_infinite_backend(self) -> Self {
        let shifted_inf = u32x4::from(0xFF000000);
        let u: u32x4 = cast(self);
        let shift_u = u << 1_u64;
        let out = (shift_u).simd_eq(shifted_inf);
        cast(out)
    }

    #[inline]
    pub(crate) fn is_finite_backend(self) -> Self {
        let shifted_exp_mask = const { u32x4::splat(0xFF000000) };
        let u: u32x4 = cast(self);
        let shift_u = u << 1_u64;
        let out = !(shift_u & shifted_exp_mask).simd_eq(shifted_exp_mask);
        cast(out)
    }

    #[inline]
    pub(crate) fn is_sign_positive_backend(self) -> Self {
        const SIGN_MASK: u32x4 = u32x4::splat((-0.0_f32).to_bits());

        let bits = cast::<f32x4, u32x4>(self);
        let sign = bits & SIGN_MASK;
        let result = sign.simd_eq(u32x4::ZERO);
        cast::<u32x4, f32x4>(result)
    }

    #[inline]
    pub(crate) fn recip_backend(self) -> Self {
        pick! {
          if #[cfg(target_feature="sse")] {
            Self(_mm_rcp_ps(self.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(f32x4_div(f32x4_splat(1.0), self.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vdivq_f32(vdupq_n_f32(1.0), self.0))
          } else {
            Self::from_array([
              1.0 / self.0.arr[0],
              1.0 / self.0.arr[1],
              1.0 / self.0.arr[2],
              1.0 / self.0.arr[3],
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn recip_sqrt_backend(self) -> Self {
        pick! {
          if #[cfg(target_feature="sse")] {
            Self(_mm_rsqrt_ps(self.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(f32x4_div(f32x4_splat(1.0), f32x4_sqrt(self.0)))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vdivq_f32(vdupq_n_f32(1.0), vsqrtq_f32(self.0)))
          } else if #[cfg(feature="std")] {
            Self::from_array([
              1.0 / self.0.arr[0].sqrt(),
              1.0 / self.0.arr[1].sqrt(),
              1.0 / self.0.arr[2].sqrt(),
              1.0 / self.0.arr[3].sqrt(),
            ])
          } else {
            Self::from_array([
              1.0 / software_sqrt(self.0.arr[0] as f64) as f32,
              1.0 / software_sqrt(self.0.arr[1] as f64) as f32,
              1.0 / software_sqrt(self.0.arr[2] as f64) as f32,
              1.0 / software_sqrt(self.0.arr[3] as f64) as f32,
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn max_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse")] {
            // _mm_max_ps seems to do rhs < self ? self : rhs. So if there's any NaN
            // involved, it chooses rhs, so we need to specifically check rhs for
            // NaN.
            rhs.is_nan().select(self, Self(_mm_max_ps(self.0, rhs.0)))
          } else if #[cfg(target_feature="simd128")] {
            // WASM has two max intrinsics:
            // - max: This propagates NaN, that's the opposite of what we need.
            // - pmax: This is defined as self < rhs ? rhs : self, which basically
            //   chooses self if either is NaN.
            //
            // pmax is what we want, but we need to specifically check self for NaN.
            Self(v128_bitselect(
              rhs.0,
              f32x4_pmax(self.0, rhs.0),
              f32x4_ne(self.0, self.0), // NaN check
            ))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vmaxnmq_f32(self.0, rhs.0))
          } else {
            Self::from_array([
              self.0.arr[0].max(rhs.0.arr[0]),
              self.0.arr[1].max(rhs.0.arr[1]),
              self.0.arr[2].max(rhs.0.arr[2]),
              self.0.arr[3].max(rhs.0.arr[3]),
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn fast_max_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse")] {
            Self(_mm_max_ps(self.0, rhs.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(f32x4_pmax(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vmaxq_f32(self.0, rhs.0))
          } else {
            Self::from_array([
              if self.0.arr[0] < rhs.0.arr[0] { rhs.0.arr[0] } else { self.0.arr[0] },
              if self.0.arr[1] < rhs.0.arr[1] { rhs.0.arr[1] } else { self.0.arr[1] },
              if self.0.arr[2] < rhs.0.arr[2] { rhs.0.arr[2] } else { self.0.arr[2] },
              if self.0.arr[3] < rhs.0.arr[3] { rhs.0.arr[3] } else { self.0.arr[3] },
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn min_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse")] {
            // _mm_min_ps seems to do self < rhs ? self : rhs. So if there's any NaN
            // involved, it chooses rhs, so we need to specifically check rhs for
            // NaN.
            rhs.is_nan().select(self, Self(_mm_min_ps(self.0, rhs.0)))
          } else if #[cfg(target_feature="simd128")] {
            // WASM has two min intrinsics:
            // - min: This propagates NaN, that's the opposite of what we need.
            // - pmin: This is defined as rhs < self ? rhs : self, which basically
            //   chooses self if either is NaN.
            //
            // pmin is what we want, but we need to specifically check self for NaN.
            Self(v128_bitselect(
              rhs.0,
              f32x4_pmin(self.0, rhs.0),
              f32x4_ne(self.0, self.0), // NaN check
            ))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vminnmq_f32(self.0, rhs.0))
          } else {
            Self::from_array([
              self.0.arr[0].min(rhs.0.arr[0]),
              self.0.arr[1].min(rhs.0.arr[1]),
              self.0.arr[2].min(rhs.0.arr[2]),
              self.0.arr[3].min(rhs.0.arr[3]),
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn fast_min_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse")] {
            Self(_mm_min_ps(self.0, rhs.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(f32x4_pmin(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vminq_f32(self.0, rhs.0))
          } else {
            Self::from_array([
              if self.0.arr[0] < rhs.0.arr[0] { self.0.arr[0] } else { rhs.0.arr[0] },
              if self.0.arr[1] < rhs.0.arr[1] { self.0.arr[1] } else { rhs.0.arr[1] },
              if self.0.arr[2] < rhs.0.arr[2] { self.0.arr[2] } else { rhs.0.arr[2] },
              if self.0.arr[3] < rhs.0.arr[3] { self.0.arr[3] } else { rhs.0.arr[3] },
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
          if #[cfg(target_feature="sse")] {
            // For both `_mm_min_ps` and `_mm_max_ps` if any input is NaN, `rhs` gets
            // chosen. For `self` to be chosen, `self` must be the second argument.
            Self(_mm_max_ps(min.0, _mm_min_ps(max.0, self.0)))
          } else if #[cfg(target_feature="simd128")] {
            Self(f32x4_max(f32x4_min(self.0, max.0), min.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))] {
            Self(vmaxq_f32(vminq_f32(self.0, max.0), min.0))
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
    pub(crate) fn reduce_max_backend(self) -> f32 {
        pick! {
          if #[cfg(all(target_arch = "aarch64", target_feature = "neon"))] {
            vmaxnmvq_f32(self.0)
          } else if #[cfg(any(target_feature = "sse2", target_feature = "simd128"))] {
            let reduce_2 = self.max(self.shuffle_consts::<2, 3, 0, 1>());
            let reduce_1 = reduce_2.max(reduce_2.shuffle_consts::<1, 0, 1, 0>());
            reduce_1.extract_const::<0>()
          } else {
            let self_array = self.to_array();
            self_array[0].max(self_array[1]).max(self_array[2]).max(self_array[3])
          }
        }
    }

    #[inline]
    pub(crate) fn fast_reduce_max_backend(self) -> f32 {
        pick! {
          if #[cfg(all(target_arch = "aarch64", target_feature = "neon"))] {
            vmaxnmvq_f32(self.0)
          } else if #[cfg(any(target_feature = "sse2", target_feature = "simd128"))] {
            let reduce_2 = self.fast_max(self.shuffle_consts::<2, 3, 0, 1>());
            let reduce_1 = reduce_2.fast_max(reduce_2.shuffle_consts::<1, 0, 1, 0>());
            reduce_1.extract_const::<0>()
          } else {
            let self_array = self.to_array();
            self_array[0].max(self_array[1]).max(self_array[2]).max(self_array[3])
          }
        }
    }

    #[inline]
    pub(crate) fn reduce_min_backend(self) -> f32 {
        pick! {
          if #[cfg(all(target_arch = "aarch64", target_feature = "neon"))] {
            vminnmvq_f32(self.0)
          } else if #[cfg(any(target_feature = "sse2", target_feature = "simd128"))] {
            let reduce_2 = self.min(self.shuffle_consts::<2, 3, 0, 1>());
            let reduce_1 = reduce_2.min(reduce_2.shuffle_consts::<1, 0, 1, 0>());
            reduce_1.extract_const::<0>()
          } else {
            let self_array = self.to_array();
            self_array[0].min(self_array[1]).min(self_array[2]).min(self_array[3])
          }
        }
    }

    #[inline]
    pub(crate) fn fast_reduce_min_backend(self) -> f32 {
        pick! {
          if #[cfg(all(target_arch = "aarch64", target_feature = "neon"))] {
            vminnmvq_f32(self.0)
          } else if #[cfg(any(target_feature = "sse2", target_feature = "simd128"))] {
            let reduce_2 = self.fast_min(self.shuffle_consts::<2, 3, 0, 1>());
            let reduce_1 = reduce_2.fast_min(reduce_2.shuffle_consts::<1, 0, 1, 0>());
            reduce_1.extract_const::<0>()
          } else {
            let self_array = self.to_array();
            self_array[0].min(self_array[1]).min(self_array[2]).min(self_array[3])
          }
        }
    }

    #[inline]
    pub(crate) fn abs_backend(self) -> Self {
        pick! {
          if #[cfg(target_feature="simd128")] {
            Self(f32x4_abs(self.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vabsq_f32(self.0))
          } else {
            let non_sign_bits = f32x4::from(f32::from_bits(i32::MAX as u32));
            self & non_sign_bits
          }
        }
    }

    #[inline]
    pub(crate) fn floor_backend(self) -> Self {
        pick! {
          if #[cfg(target_feature="simd128")] {
            Self(f32x4_floor(self.0))
          } else if #[cfg(target_feature="sse4.1")] {
            Self(_mm_floor_ps(self.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vrndmq_f32(self.0))
          } else if #[cfg(target_feature = "sse2")] {
            // Based on https://github.com/bitshifter/glam-rs/blob/main/src/sse2.rs `m128_floor`
            // Based on https://github.com/microsoft/DirectXMath `XMVectorFloor`

            const BOUNDS_LIMIT: i32x4 = i32x4::splat(8388608.0_f32.to_bits().cast_signed());

            // This evalutes to `false` for NaNs, positive and negative infinity
            // and values large enough that their precision guarantees they are
            // whole numbers.
            let in_bounds = Self::from_bits(
              self.abs().to_bits().cast_signed().simd_lt(BOUNDS_LIMIT).cast_unsigned(),
            );

            let self_trunc = Self(_mm_cvtepi32_ps(_mm_cvttps_epi32(self.0)));

            let gt_self = self_trunc.simd_gt(self);
            // This is a faster way to compute `gt_self.select(0.0, -1.0)`
            let offset = Self(_mm_cvtepi32_ps(cast(gt_self)));

            let result = self_trunc + offset;
            in_bounds.abs().bitselect(result, self)
          } else if #[cfg(feature="std")] {
            let base: [f32; 4] = cast(self);
            cast(base.map(|val| val.floor()))
          } else {
            let base: [f32; 4] = cast(self);
            let rounded: [f32; 4] = cast(self.round());
            cast([
              if base[0] < rounded[0] { rounded[0] - 1.0 } else { rounded[0] },
              if base[1] < rounded[1] { rounded[1] - 1.0 } else { rounded[1] },
              if base[2] < rounded[2] { rounded[2] - 1.0 } else { rounded[2] },
              if base[3] < rounded[3] { rounded[3] - 1.0 } else { rounded[3] },
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn ceil_backend(self) -> Self {
        pick! {
          if #[cfg(target_feature="simd128")] {
            Self(f32x4_ceil(self.0))
          } else if #[cfg(target_feature="sse4.1")] {
            Self(_mm_ceil_ps(self.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vrndpq_f32(self.0))
          } else if #[cfg(target_feature = "sse2")] {
            // Based on https://github.com/bitshifter/glam-rs/blob/main/src/sse2.rs `m128_ceil`
            // Based on https://github.com/microsoft/DirectXMath `XMVectorCeil`

            const BOUNDS_LIMIT: i32x4 = i32x4::splat(8388608.0_f32.to_bits().cast_signed());

            // This evalutes to `false` for NaNs, positive and negative infinity
            // and values large enough that their precision guarantees they are
            // whole numbers.
            let in_bounds = Self::from_bits(
              self.abs().to_bits().cast_signed().simd_lt(BOUNDS_LIMIT).cast_unsigned(),
            );

            let self_trunc = Self(_mm_cvtepi32_ps(_mm_cvttps_epi32(self.0)));

            let lt_self = self_trunc.simd_lt(self);
            // This is a faster way to compute `lt_self.select(0.0, -1.0)`
            let offset = Self(_mm_cvtepi32_ps(cast(lt_self)));

            let result = self_trunc - offset;
            in_bounds.abs().bitselect(result, self)
          } else if #[cfg(feature="std")] {
            let base: [f32; 4] = cast(self);
            cast(base.map(|val| val.ceil()))
          } else {
            let base: [f32; 4] = cast(self);
            let rounded: [f32; 4] = cast(self.round());
            cast([
              if base[0] > rounded[0] { rounded[0] + 1.0 } else { rounded[0] },
              if base[1] > rounded[1] { rounded[1] + 1.0 } else { rounded[1] },
              if base[2] > rounded[2] { rounded[2] + 1.0 } else { rounded[2] },
              if base[3] > rounded[3] { rounded[3] + 1.0 } else { rounded[3] },
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn round_backend(self) -> Self {
        pick! {
          if #[cfg(target_feature="sse4.1")] {
            const_f32_as_f32x4!(HALF_NEXT_DOWN, 0.5_f32.next_down());
            const_f32_as_f32x4!(BOUNDS_LIMIT, 8388608.0);

            let self_abs = self.abs();

            let adjusted_self = self_abs + Self::HALF;
            let result_abs = Self(_mm_round_ps::<{_MM_FROUND_NO_EXC | _MM_FROUND_TO_ZERO}>(adjusted_self.0));
            // The addition breaks for `0.5.next_down()` which incorrectly rounds to
            // `1.0`. This resets the result back to `0.0`.
            let result_abs = result_abs & self_abs.simd_ne(HALF_NEXT_DOWN);

            // Large value, infinity and NaN need special handling.
            let bounds_mask: Self = cast(_mm_cmplt_epi32(cast(self_abs), cast(BOUNDS_LIMIT)));

            // `abs` keeps the original sign.
            bounds_mask.abs().bitselect(result_abs, self)
          } else if #[cfg(target_feature="sse2")] {
            const_f32_as_f32x4!(HALF_NEXT_DOWN, 0.5_f32.next_down());
            const_f32_as_f32x4!(BOUNDS_LIMIT, 8388608.0);

            let self_abs = self.abs();

            let adjusted_self = self_abs + Self::HALF;
            let result_abs = Self(
              _mm_cvtepi32_ps(_mm_cvttps_epi32(adjusted_self.0)),
            );
            // The addition breaks for `0.5.next_down()` which incorrectly rounds to
            // `1.0`. This resets the result back to `0.0`.
            let result_abs = result_abs & self_abs.simd_ne(HALF_NEXT_DOWN);

            // Large value, infinity and NaN need special handling.
            let bounds_mask: Self = cast(_mm_cmplt_epi32(cast(self_abs), cast(BOUNDS_LIMIT)));

            // `abs` keeps the original sign.
            bounds_mask.abs().bitselect(result_abs, self)
          } else if #[cfg(target_feature="simd128")] {
            const_f32_as_f32x4!(HALF_NEXT_DOWN, 0.5_f32.next_down());
            const_f32_as_f32x4!(BOUNDS_LIMIT, 8388608.0);

            let self_abs = self.abs();

            let adjusted_self = self_abs + Self::HALF;
            let result_abs = Self(f32x4_trunc(adjusted_self.0));
            // The addition breaks for `0.5.next_down()` which incorrectly rounds to
            // `1.0`. This resets the result back to `0.0`.
            let result_abs = result_abs & self_abs.simd_ne(HALF_NEXT_DOWN);

            // Large value, infinity and NaN need special handling.
            let bounds_mask = Self(i32x4_lt(self_abs.0, BOUNDS_LIMIT.0));

            // `abs` keeps the original sign.
            bounds_mask.abs().bitselect(result_abs, self)
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vrndaq_f32(self.0))
          } else {
            const_f32_as_f32x4!(HALF_NEXT_DOWN, 0.5_f32.next_down());
            const_f32_as_f32x4!(BOUNDS_LIMIT, 8388608.0);

            let self_abs = self.abs();

            let adjusted_self = (self_abs + Self::HALF).to_array();
            let result_abs = Self::from_array([
              adjusted_self[0] as u32 as f32,
              adjusted_self[1] as u32 as f32,
              adjusted_self[2] as u32 as f32,
              adjusted_self[3] as u32 as f32,
            ]);
            // The addition breaks for `0.5.next_down()` which incorrectly rounds to
            // `1.0`. This resets the result back to `0.0`.
            let result_abs = result_abs & self_abs.simd_ne(HALF_NEXT_DOWN);

            // Large value, infinity and NaN need special handling.
            let bounds_mask: Self = cast(cast::<_, i32x4>(self_abs).simd_lt(cast::<_, i32x4>(BOUNDS_LIMIT)));

            // `abs` keeps the original sign.
            bounds_mask.abs().bitselect(result_abs, self)
          }
        }
    }

    #[inline]
    pub(crate) fn round_int_backend(self) -> i32x4 {
        pick! {
          if #[cfg(target_feature="sse2")] {
            // Based on: https://github.com/v8/v8/blob/210987a552a2bf2a854b0baa9588a5959ff3979d/src/codegen/shared-ia32-x64/macro-assembler-shared-ia32-x64.h#L489-L504
            let non_nan_mask = self.simd_eq(self);
            let non_nan = self & non_nan_mask;
            let flip_to_max: i32x4 = cast(self.simd_ge(Self::splat(2147483648.0)));
            let cast: i32x4 = cast(_mm_cvtps_epi32(non_nan.0));
            flip_to_max ^ cast
          } else if #[cfg(target_feature="simd128")] {
            cast(Self(i32x4_trunc_sat_f32x4(f32x4_nearest(self.0))))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            cast(Self(vreinterpretq_f32_s32(vcvtnq_s32_f32(self.0))))
          } else {
            let rounded: [f32; 4] = cast(self.round());
            cast([
              rounded[0] as i32,
              rounded[1] as i32,
              rounded[2] as i32,
              rounded[3] as i32,
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn fast_round_int_backend(self) -> i32x4 {
        pick! {
          if #[cfg(target_feature="sse2")] {
            cast(_mm_cvtps_epi32(self.0))
          } else {
            self.round_int()
          }
        }
    }

    #[inline]
    pub(crate) fn round_ties_even_backend(self) -> Self {
        pick! {
          if #[cfg(target_feature="sse4.1")] {
            Self(_mm_round_ps::<{_MM_FROUND_NO_EXC | _MM_FROUND_TO_NEAREST_INT}>(self.0))
          } else if #[cfg(target_feature="sse2")] {
            let mi: __m128i = _mm_cvtps_epi32(self.0);
            let f: f32x4 = f32x4(_mm_cvtepi32_ps(mi));
            let i: i32x4 = cast(mi);
            let mask: f32x4 = cast(i.simd_eq(i32x4::from(0x80000000_u32 as i32)));
            mask.select(self, f)
          } else if #[cfg(target_feature="simd128")] {
            Self(f32x4_nearest(self.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vrndnq_f32(self.0))
          } else {
            // Note(Lokathor): This software fallback is probably very slow compared
            // to having a hardware option available, even just the sse2 version is
            // better than this. Oh well.
            let to_int = f32x4::from(1.0 / f32::EPSILON);
            let u: u32x4 = cast(self);
            let e: i32x4 = cast((u >> 23) & u32x4::from(0xff));
            let mut y: f32x4;

            let no_op_magic = i32x4::from(0x7f + 23);
            let no_op_mask: f32x4 = cast(e.simd_gt(no_op_magic) | e.simd_eq(no_op_magic));
            let no_op_val: f32x4 = self;

            let zero_magic = i32x4::from(0x7f - 1);
            let zero_mask: f32x4 = cast(e.simd_lt(zero_magic));
            let zero_val: f32x4 = self * f32x4::from(0.0);

            let neg_bit: f32x4 = cast(cast::<u32x4, i32x4>(u).simd_lt(i32x4::default()));
            let x: f32x4 = neg_bit.select(-self, self);
            y = x + to_int - to_int - x;
            y = y.simd_gt(f32x4::from(0.5)).select(
              y + x - f32x4::from(-1.0),
              y.simd_lt(f32x4::from(-0.5)).select(y + x + f32x4::from(1.0), y + x),
            );
            y = neg_bit.select(-y, y);

            no_op_mask.select(no_op_val, zero_mask.select(zero_val, y))
          }
        }
    }

    #[inline]
    pub(crate) fn trunc_backend(self) -> Self {
        pick! {
          if #[cfg(target_feature="sse4.1")] {
            Self(_mm_round_ps::<{_MM_FROUND_NO_EXC | _MM_FROUND_TO_ZERO}>(self.0))
          } else if #[cfg(target_feature="sse2")] {
            // Ported from https://docs.rs/glam/latest/glam/f32/struct.Vec4.html#method.trunc
            // Based on https://github.com/microsoft/DirectXMath `XMVectorTruncate`
            let result: Self = cast(_mm_cvtepi32_ps(_mm_cvttps_epi32(self.0)));

            // Out of range values are either already round, infinite or NaN.
            let bounds_mask: Self = cast(_mm_cmplt_epi32(
                cast(self.abs()),
                _mm_set1_epi32(8388608_f32.to_bits() as i32),
            ));

            // Reset the sign bit of the mask to preverse the sign of `self`.
            bounds_mask.abs().select(result, self)
          } else if #[cfg(target_feature="simd128")] {
            Self(f32x4_trunc(self.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vrndq_f32(self.0))
          } else {
            let array: [f32; 4] = cast(self);
            let result: Self = cast([
              array[0] as i32 as f32,
              array[1] as i32 as f32,
              array[2] as i32 as f32,
              array[3] as i32 as f32,
            ]);

            // Out of range values are either already round, infinite or NaN.
            const BOUNDS_LIMIT: i32 = 8388608_f32.to_bits() as i32;
            let bounds_mask: Self = cast(cast::<f32x4, i32x4>(self.abs()).simd_lt(i32x4::splat(BOUNDS_LIMIT)));

            // Reset the sign bit of the mask to preverse the sign of `self`.
            bounds_mask.abs().select(result, self)
          }
        }
    }

    #[inline]
    pub(crate) fn trunc_int_backend(self) -> i32x4 {
        pick! {
          if #[cfg(target_feature="sse2")] {
            // Based on: https://github.com/v8/v8/blob/210987a552a2bf2a854b0baa9588a5959ff3979d/src/codegen/shared-ia32-x64/macro-assembler-shared-ia32-x64.h#L489-L504
            let non_nan_mask = self.simd_eq(self);
            let non_nan = self & non_nan_mask;
            let flip_to_max: i32x4 = cast(self.simd_ge(Self::splat(2147483648.0)));
            let cast: i32x4 = cast(_mm_cvttps_epi32(non_nan.0));
            flip_to_max ^ cast
          } else if #[cfg(target_feature="simd128")] {
            cast(Self(i32x4_trunc_sat_f32x4(self.0)))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            cast(Self(vreinterpretq_f32_s32(vcvtq_s32_f32(self.0))))
          } else {
            let n: [f32;4] = cast(self);
            cast([
              n[0] as i32,
              n[1] as i32,
              n[2] as i32,
              n[3] as i32,
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn fast_trunc_int_backend(self) -> i32x4 {
        pick! {
          if #[cfg(target_feature="sse2")] {
            cast(_mm_cvttps_epi32(self.0))
          } else {
            self.trunc_int()
          }
        }
    }

    #[inline]
    pub(crate) fn fast_mul_add_backend(self, a: Self, b: Self) -> Self {
        pick! {
          if #[cfg(all(target_feature="sse2",target_feature="fma"))] {
            Self(_mm_fmadd_ps(self.0, a.0, b.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))] {
            Self(vfmaq_f32(b.0, self.0, a.0))
          } else {
            (self * a) + b
          }
        }
    }

    #[inline]
    pub(crate) fn fast_mul_sub_backend(self, a: Self, b: Self) -> Self {
        pick! {
          if #[cfg(all(target_feature="sse2",target_feature="fma"))] {
            Self(_mm_fmsub_ps(self.0, a.0, b.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))] {
            Self(vfmaq_f32(vnegq_f32(b.0), self.0, a.0))
          } else {
            (self * a) - b
          }
        }
    }

    #[inline]
    pub(crate) fn fast_mul_neg_add_backend(self, a: Self, b: Self) -> Self {
        pick! {
          if #[cfg(all(target_feature="sse2",target_feature="fma"))] {
            Self(_mm_fnmadd_ps(self.0, a.0, b.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))] {
            Self(vfmsq_f32(b.0, self.0, a.0))
          } else {
            b - (self * a)
          }
        }
    }

    #[inline]
    pub(crate) fn fast_mul_neg_sub_backend(self, a: Self, b: Self) -> Self {
        pick! {
          if #[cfg(all(target_feature="sse2",target_feature="fma"))] {
            Self(_mm_fnmsub_ps(self.0, a.0, b.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))] {
            Self(vnegq_f32(vfmaq_f32(b.0, self.0, a.0)))
          } else {
            -(self * a) - b
          }
        }
    }

    #[inline]
    pub(crate) fn powf_simd_backend(self, n: Self) -> Self {
        const_f32_as_f32x4!(ln2f_hi, 0.693359375);
        const_f32_as_f32x4!(ln2f_lo, -2.12194440e-4);
        const_f32_as_f32x4!(P0logf, 3.3333331174E-1);
        const_f32_as_f32x4!(P1logf, -2.4999993993E-1);
        const_f32_as_f32x4!(P2logf, 2.0000714765E-1);
        const_f32_as_f32x4!(P3logf, -1.6668057665E-1);
        const_f32_as_f32x4!(P4logf, 1.4249322787E-1);
        const_f32_as_f32x4!(P5logf, -1.2420140846E-1);
        const_f32_as_f32x4!(P6logf, 1.1676998740E-1);
        const_f32_as_f32x4!(P7logf, -1.1514610310E-1);
        const_f32_as_f32x4!(P8logf, 7.0376836292E-2);

        const_f32_as_f32x4!(p2expf, 1.0 / 2.0); // coefficients for Taylor expansion of exp
        const_f32_as_f32x4!(p3expf, 1.0 / 6.0);
        const_f32_as_f32x4!(p4expf, 1.0 / 24.0);
        const_f32_as_f32x4!(p5expf, 1.0 / 120.0);
        const_f32_as_f32x4!(p6expf, 1.0 / 720.0);
        const_f32_as_f32x4!(p7expf, 1.0 / 5040.0);

        let x1 = self.abs();
        let x = x1.fraction_2();

        let mask = x.simd_gt(f32x4::SQRT_2 * f32x4::HALF);
        let x = (!mask).select(x + x, x);

        let x = x - f32x4::ONE;
        let x2 = x * x;
        let lg1 = polynomial_8!(
            x, P0logf, P1logf, P2logf, P3logf, P4logf, P5logf, P6logf, P7logf, P8logf
        );
        let lg1 = lg1 * x2 * x;

        let ef = x1.exponent();
        let ef = mask.select(ef + f32x4::ONE, ef);

        let e1 = (ef * n).round_ties_even();
        let yr = ef.fast_mul_sub(n, e1);

        let lg = f32x4::HALF.fast_mul_neg_add(x2, x) + lg1;
        let x2_err = (f32x4::HALF * x).fast_mul_sub(x, f32x4::HALF * x2);
        let lg_err = f32x4::HALF.fast_mul_add(x2, lg - x) - lg1;

        let e2 = (lg * n * f32x4::LOG2_E).round_ties_even();
        let v = lg.fast_mul_sub(n, e2 * ln2f_hi);
        let v = e2.fast_mul_neg_add(ln2f_lo, v);
        let v = v - (lg_err + x2_err).fast_mul_sub(n, yr * f32x4::LN_2);

        let x = v;
        let e3 = (x * f32x4::LOG2_E).round_ties_even();
        let x = e3.fast_mul_neg_add(f32x4::LN_2, x);
        let x2 = x * x;
        let z = x2.fast_mul_add(
            polynomial_5!(x, p2expf, p3expf, p4expf, p5expf, p6expf, p7expf),
            x + f32x4::ONE,
        );

        let ee = e1 + e2 + e3;
        let ei = cast::<_, i32x4>(ee.round_int());
        let ej = cast::<_, i32x4>(ei + (cast::<_, i32x4>(z) >> 23));

        let overflow =
            cast::<_, f32x4>(ej.simd_gt(i32x4::splat(0x0FF))) | (ee.simd_gt(f32x4::splat(300.0)));
        let underflow =
            cast::<_, f32x4>(ej.simd_lt(i32x4::splat(0x000))) | (ee.simd_lt(f32x4::splat(-300.0)));

        // Add exponent by integer addition
        let z = cast::<_, f32x4>(cast::<_, i32x4>(z) + (ei << 23));

        // Check for overflow/underflow
        let z = if (overflow | underflow).any() {
            let z = underflow.select(f32x4::ZERO, z);
            overflow.select(Self::infinity(), z)
        } else {
            z
        };

        // Check for self == 0
        let x_zero = self.is_zero_or_subnormal();
        let z = x_zero.select(
            n.simd_lt(f32x4::ZERO).select(
                Self::infinity(),
                n.simd_eq(f32x4::ZERO).select(f32x4::ONE, f32x4::ZERO),
            ),
            z,
        );

        let x_sign = self.is_sign_negative();
        let z = if x_sign.any() {
            // Y into an integer
            let yi = n.simd_eq(n.round_ties_even());
            // Is y odd? If yes flip the sign of the result.
            let y_odd = cast::<i32x4, f32x4>(n.round_int() << 31);

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
          if #[cfg(target_feature="sse")] {
            Self(_mm_sqrt_ps(self.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(f32x4_sqrt(self.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vsqrtq_f32(self.0))
          } else if #[cfg(feature="std")] {
            Self::from_array([
              self.0.arr[0].sqrt(),
              self.0.arr[1].sqrt(),
              self.0.arr[2].sqrt(),
              self.0.arr[3].sqrt(),
            ])
          } else {
            Self::from_array([
              software_sqrt(self.0.arr[0] as f64) as f32,
              software_sqrt(self.0.arr[1] as f64) as f32,
              software_sqrt(self.0.arr[2] as f64) as f32,
              software_sqrt(self.0.arr[3] as f64) as f32,
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn exp_backend(self) -> Self {
        const_f32_as_f32x4!(P0, 1.0 / 2.0);
        const_f32_as_f32x4!(P1, 1.0 / 6.0);
        const_f32_as_f32x4!(P2, 1.0 / 24.0);
        const_f32_as_f32x4!(P3, 1.0 / 120.0);
        const_f32_as_f32x4!(P4, 1.0 / 720.0);
        const_f32_as_f32x4!(P5, 1.0 / 5040.0);
        // LN2D_HI/LO: double-double decomposition of ln(2) for exp range reduction,
        // following the approach from fdlibm's e_exp.c (Sun Microsystems,
        // https://www.netlib.org/fdlibm/). The f32 split uses f32-precision constants
        // (0.693359375, -2.12194440e-4) summing to ln(2) with single-precision
        // accuracy; the f64 variants use a full f64 double-double
        // decomposition.
        const_f32_as_f32x4!(LN2D_HI, 0.693359375);
        const_f32_as_f32x4!(LN2D_LO, -2.12194440e-4);
        // max_x = ln(f32::MAX) ≈ 88.7229, max_r = 127 (IEEE max normal exponent)
        // min_x = -149.5 ln(2) ≈ -103.63: min r for vm_pow2n subnormal
        let max_x = f32x4::from(88.723);
        let min_x = f32x4::from(-103.63);
        // x < min_x: e^x underflows to 0 -- skip the entire pipeline
        let finite = self.is_finite();
        let neg_underflow = self.simd_lt(min_x) & finite;
        if neg_underflow.all() {
            return Self::ZERO;
        }
        let max_r = f32x4::from(127.0);
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
        const_f32_as_f32x4!(P2, 1.0 / 2.0);
        const_f32_as_f32x4!(P3, 1.0 / 6.0);
        const_f32_as_f32x4!(P4, 1.0 / 24.0);
        const_f32_as_f32x4!(P5, 1.0 / 120.0);
        const_f32_as_f32x4!(P6, 1.0 / 720.0);
        const_f32_as_f32x4!(P7, 1.0 / 5040.0);

        // max_x = log2(f32::MAX) ≈ 127.99999
        // min_x = log2(f32::MIN_POSITIVE) - 23 ≈ -126 - 23 = -149
        let max_x = f32x4::from(127.99999);
        let min_x = f32x4::from(-149.5);
        let finite = self.is_finite();
        let neg_underflow = self.simd_lt(min_x) & finite;
        if neg_underflow.all() {
            return Self::ZERO;
        }

        let round = self.round_ties_even();
        let max_r = f32x4::from(127.0);
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
        const_f32_as_f32x4!(HALF, 0.5);
        const_f32_as_f32x4!(P0, 3.3333331174E-1);
        const_f32_as_f32x4!(P1, -2.4999993993E-1);
        const_f32_as_f32x4!(P2, 2.0000714765E-1);
        const_f32_as_f32x4!(P3, -1.6668057665E-1);
        const_f32_as_f32x4!(P4, 1.4249322787E-1);
        const_f32_as_f32x4!(P5, -1.2420140846E-1);
        const_f32_as_f32x4!(P6, 1.1676998740E-1);
        const_f32_as_f32x4!(P7, -1.1514610310E-1);
        const_f32_as_f32x4!(P8, 7.0376836292E-2);
        const_f32_as_f32x4!(LN2F_HI, 0.693359375);
        const_f32_as_f32x4!(LN2F_LO, -2.12194440e-4);
        const_f32_as_f32x4!(VM_SMALLEST_NORMAL, 1.17549435E-38);

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
        const_f32_as_f32x4!(C0, 2.2241257);
        const_f32_as_f32x4!(C1, -3.8095417);
        const_f32_as_f32x4!(C2, 5.8982625);
        const_f32_as_f32x4!(C3, -5.532182);
        const_f32_as_f32x4!(C4, 2.8208892);
        const_f32_as_f32x4!(C5, -0.60156447);
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
        const_f32_as_f32x4!(CBRT2, 1.259921);
        const_f32_as_f32x4!(CBRT4, 1.587401);
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
        const_f32_as_f32x4!(P4asinf, 4.2163199048E-2);
        const_f32_as_f32x4!(P3asinf, 2.4181311049E-2);
        const_f32_as_f32x4!(P2asinf, 4.5470025998E-2);
        const_f32_as_f32x4!(P1asinf, 7.4953002686E-2);
        const_f32_as_f32x4!(P0asinf, 1.6666752422E-1);

        let xa = self.abs();
        let big = xa.simd_ge(f32x4::splat(0.5));

        let x1 = f32x4::splat(0.5) * (f32x4::ONE - xa);
        let x2 = xa * xa;
        let x3 = big.select(x1, x2);

        let xb = x1.sqrt();

        let x4 = big.select(xb, xa);

        let z = polynomial_4!(x3, P0asinf, P1asinf, P2asinf, P3asinf, P4asinf);
        let z = z.fast_mul_add(x3 * x4, x4);

        let z1 = z + z;

        // asin
        let z3 = f32x4::FRAC_PI_2 - z1;
        let asin = big.select(z3, z);
        let asin = asin.flipsign(self);

        asin
    }

    #[inline]
    pub(crate) fn acos_backend(self) -> Self {
        // Based on the Agner Fog "vector class library":
        // https://github.com/vectorclass/version2/blob/master/vectormath_trig.h
        const_f32_as_f32x4!(P4asinf, 4.2163199048E-2);
        const_f32_as_f32x4!(P3asinf, 2.4181311049E-2);
        const_f32_as_f32x4!(P2asinf, 4.5470025998E-2);
        const_f32_as_f32x4!(P1asinf, 7.4953002686E-2);
        const_f32_as_f32x4!(P0asinf, 1.6666752422E-1);

        let xa = self.abs();
        let big = xa.simd_ge(f32x4::splat(0.5));

        let x1 = f32x4::splat(0.5) * (f32x4::ONE - xa);
        let x2 = xa * xa;
        let x3 = big.select(x1, x2);

        let xb = x1.sqrt();

        let x4 = big.select(xb, xa);

        let z = polynomial_4!(x3, P0asinf, P1asinf, P2asinf, P3asinf, P4asinf);
        let z = z.fast_mul_add(x3 * x4, x4);

        let z1 = z + z;

        // acos
        let z3 = self.simd_lt(f32x4::ZERO).select(f32x4::PI - z1, z1);
        let z4 = f32x4::FRAC_PI_2 - z.flipsign(self);
        let acos = big.select(z3, z4);

        acos
    }

    #[inline]
    pub(crate) fn atan_backend(self) -> Self {
        // Based on the Agner Fog "vector class library":
        // https://github.com/vectorclass/version2/blob/master/vectormath_trig.h
        const_f32_as_f32x4!(P3atanf, 8.05374449538E-2);
        const_f32_as_f32x4!(P2atanf, -1.38776856032E-1);
        const_f32_as_f32x4!(P1atanf, 1.99777106478E-1);
        const_f32_as_f32x4!(P0atanf, -3.33329491539E-1);

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
        const_f32_as_f32x4!(P3atanf, 8.05374449538E-2);
        const_f32_as_f32x4!(P2atanf, -1.38776856032E-1);
        const_f32_as_f32x4!(P1atanf, 1.99777106478E-1);
        const_f32_as_f32x4!(P0atanf, -3.33329491539E-1);

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

        const_f32_as_f32x4!(DP1F, 0.78515625_f32 * 2.0);
        const_f32_as_f32x4!(DP2F, 2.4187564849853515625E-4_f32 * 2.0);
        const_f32_as_f32x4!(DP3F, 3.77489497744594108E-8_f32 * 2.0);

        const_f32_as_f32x4!(P0sinf, -1.6666654611E-1);
        const_f32_as_f32x4!(P1sinf, 8.3321608736E-3);
        const_f32_as_f32x4!(P2sinf, -1.9515295891E-4);

        const_f32_as_f32x4!(P0cosf, 4.166664568298827E-2);
        const_f32_as_f32x4!(P1cosf, -1.388731625493765E-3);
        const_f32_as_f32x4!(P2cosf, 2.443315711809948E-5);

        const_f32_as_f32x4!(TWO_OVER_PI, 2.0 / core::f32::consts::PI);

        let xa = self.abs();

        // Find quadrant
        let y = (xa * TWO_OVER_PI).round_ties_even();
        let q: i32x4 = y.round_int();

        let x = y.fast_mul_neg_add(DP3F, y.fast_mul_neg_add(DP2F, y.fast_mul_neg_add(DP1F, xa)));

        let x2 = x * x;
        let mut s = polynomial_2!(x2, P0sinf, P1sinf, P2sinf) * (x * x2) + x;
        let mut c = polynomial_2!(x2, P0cosf, P1cosf, P2cosf) * (x2 * x2)
            + f32x4::from(0.5).fast_mul_neg_add(x2, f32x4::from(1.0));

        let swap = !(q & i32x4::from(1)).simd_eq(i32x4::from(0));

        let mut overflow: f32x4 = cast(q.simd_gt(i32x4::from(0x2000000)));
        overflow &= xa.is_finite();
        s = overflow.select(f32x4::from(0.0), s);
        c = overflow.select(f32x4::from(1.0), c);

        // calc sin
        let mut sin1 = cast::<_, f32x4>(swap).select(c, s);
        let sign_sin: i32x4 = (q << 30) ^ cast::<_, i32x4>(self);
        sin1 = sin1.flipsign(cast(sign_sin));

        // calc cos
        let mut cos1 = cast::<_, f32x4>(swap).select(s, c);
        let sign_cos: i32x4 = ((q + i32x4::from(1)) & i32x4::from(2)) << 30;
        cos1 ^= cast::<_, f32x4>(sign_cos);

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
        const_f32_as_f32x4!(P4asinf, 4.2163199048E-2);
        const_f32_as_f32x4!(P3asinf, 2.4181311049E-2);
        const_f32_as_f32x4!(P2asinf, 4.5470025998E-2);
        const_f32_as_f32x4!(P1asinf, 7.4953002686E-2);
        const_f32_as_f32x4!(P0asinf, 1.6666752422E-1);

        let xa = self.abs();
        let big = xa.simd_ge(f32x4::splat(0.5));

        let x1 = f32x4::splat(0.5) * (f32x4::ONE - xa);
        let x2 = xa * xa;
        let x3 = big.select(x1, x2);

        let xb = x1.sqrt();

        let x4 = big.select(xb, xa);

        let z = polynomial_4!(x3, P0asinf, P1asinf, P2asinf, P3asinf, P4asinf);
        let z = z.fast_mul_add(x3 * x4, x4);

        let z1 = z + z;

        // acos
        let z3 = self.simd_lt(f32x4::ZERO).select(f32x4::PI - z1, z1);
        let z4 = f32x4::FRAC_PI_2 - z.flipsign(self);
        let acos = big.select(z3, z4);

        // asin
        let z3 = f32x4::FRAC_PI_2 - z1;
        let asin = big.select(z3, z);
        let asin = asin.flipsign(self);

        (asin, acos)
    }

    #[inline]
    pub(crate) fn exp_m1_backend(self) -> Self {
        // x < -17.329: e^x < 2⁻²⁵, exp_m1(x) = -1.0 exactly (mantissa exhaustion)
        // IEEE simd_lt returns false for NaN, so NaN lanes can't reach here.
        // -inf is < -17.329, and exp_m1(-inf) = -1.0, also correct.
        if self.simd_lt(f32x4::from(-17.329)).all() {
            return f32x4::from(-1.0);
        }
        const_f32_as_f32x4!(P0, 1.0 / 2.0);
        const_f32_as_f32x4!(P1, 1.0 / 6.0);
        const_f32_as_f32x4!(P2, 1.0 / 24.0);
        const_f32_as_f32x4!(P3, 1.0 / 120.0);
        const_f32_as_f32x4!(P4, 1.0 / 720.0);
        const_f32_as_f32x4!(P5, 1.0 / 5040.0);
        // LN2D_HI/LO: double-double decomposition of ln(2) for exp range reduction,
        // following the approach from fdlibm's e_exp.c (Sun Microsystems,
        // https://www.netlib.org/fdlibm/). The f32 split uses f32-precision constants
        // (0.693359375, -2.12194440e-4) summing to ln(2) with single-precision
        // accuracy; the f64 variants use a full f64 double-double
        // decomposition.
        const_f32_as_f32x4!(LN2D_HI, 0.693359375);
        const_f32_as_f32x4!(LN2D_LO, -2.12194440e-4);
        // max_x = ln(f32::MAX) ≈ 88.7229, max_r = 127 (IEEE max normal exponent)
        // min_x = -149.5 ln(2) ≈ -103.63: min r for vm_pow2n subnormal
        let max_x = f32x4::from(88.723);
        let min_x = f32x4::from(-103.63);
        let max_r = f32x4::from(127.0);
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
        const_f32_as_f32x4!(P0, 1.0);
        const_f32_as_f32x4!(P1, 1.0 / 6.0);
        const_f32_as_f32x4!(P2, 1.0 / 120.0);
        const_f32_as_f32x4!(P3, 1.0 / 5040.0);
        let a = self.abs();
        // |x| < 0.5: Taylor poly; last truncation term < 1 ULP at x=0.5 for both types
        let small = a.simd_lt(f32x4::from(0.5));
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
        const_f32_as_f32x4!(P0, 1.0);
        const_f32_as_f32x4!(P1, 1.0 / 2.0);
        const_f32_as_f32x4!(P2, 1.0 / 24.0);
        const_f32_as_f32x4!(P3, 1.0 / 720.0);
        let a = self.abs();
        // |x| < 0.5: Taylor poly; last truncation term < 1 ULP at x=0.5 for both types
        let small = a.simd_lt(f32x4::from(0.5));
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
        let large = a.simd_gt(f32x4::from(9.011));
        if large.all() {
            return Self::ONE.flipsign(self);
        }
        let small = a.simd_lt(f32x4::from(2e-4));
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
impl f32x4 {
    #[inline]
    fn vm_pow2n(self) -> Self {
        const_f32_as_f32x4!(pow2_23, 8388608.0);
        const_f32_as_f32x4!(bias, 127.0);
        let a = self + (bias + pow2_23);
        let c = cast::<_, i32x4>(a) << 23;
        let std_result = cast::<_, f32x4>(c);

        let min_exp = f32x4::from(-126.0);
        let is_sub = self.simd_lt(min_exp);
        if is_sub.any() {
            let valid = self.simd_ge(f32x4::from(-149.0));
            let shift_f = self + f32x4::from(149.0);
            let mut shift_i = shift_f.trunc_int();
            shift_i = cast::<_, i32x4>(valid).select(shift_i, i32x4::ZERO);
            let mantissa = i32x4::ONE << shift_i;
            let sub_result = cast::<_, f32x4>(mantissa);
            let sub_result = valid.select(sub_result, f32x4::ZERO);
            is_sub.select(sub_result, std_result)
        } else {
            std_result
        }
    }

    #[inline]
    fn exponent(self) -> f32x4 {
        const_f32_as_f32x4!(pow2_23, 8388608.0);
        const_f32_as_f32x4!(bias, 127.0);
        let a = cast::<_, u32x4>(self);
        let b = a >> 23;
        let c = b | cast::<_, u32x4>(pow2_23);
        let d = cast::<_, f32x4>(c);
        let e = d - (pow2_23 + bias);
        e
    }

    #[inline]
    fn fraction_2(self) -> Self {
        let t1 = cast::<_, u32x4>(self);
        let t2 = cast::<_, u32x4>((t1 & u32x4::from(0x007FFFFF)) | u32x4::from(0x3F000000));
        cast::<_, f32x4>(t2)
    }
    #[inline]
    fn is_zero_or_subnormal(self) -> Self {
        let t = cast::<_, i32x4>(self);
        let t = t & i32x4::splat(0x7F800000);
        let mask = t.simd_eq(i32x4::splat(0));
        cast::<_, f32x4>(mask)
    }
    #[inline]
    fn infinity() -> Self {
        cast::<_, f32x4>(i32x4::splat(0x7F800000))
    }
    #[inline]
    fn nan_log() -> Self {
        cast::<_, f32x4>(i32x4::splat(0x7FC00000 | 0x101 & 0x003FFFFF))
    }
    #[inline]
    fn nan_pow() -> Self {
        cast::<_, f32x4>(i32x4::splat(0x7FC00000 | 0x101 & 0x003FFFFF))
    }

    #[inline]
    pub(crate) fn from_i32x4_backend(v: i32x4) -> Self {
        pick! {
          if #[cfg(target_feature="sse2")] {
            Self(_mm_cvtepi32_ps(v.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(f32x4_convert_i32x4(v.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))] {
            Self(vcvtq_f32_s32(v.0))
          } else {
            Self::from_array([
                v.as_array()[0] as f32,
                v.as_array()[1] as f32,
                v.as_array()[2] as f32,
                v.as_array()[3] as f32,
            ])
          }
        }
    }
}
}
