use crate::backend_imports::*;

#[cfg(not(any(
    target_feature = "sse2",
    target_feature = "simd128",
    all(target_feature = "neon", target_arch = "aarch64")
)))]
#[repr(C, align(16))]
#[derive(Clone, Copy)]
pub(crate) struct Inner {
    pub arr: [i16; 8],
}

safe_target_feature! {
impl SimdBackend for i16x8 {
    pick! {
      if #[cfg(target_feature="sse2")] {
        type Inner = __m128i;
      } else if #[cfg(target_feature="simd128")] {
        type Inner = v128;
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        type Inner = int16x8_t;
      } else {
        type Inner = Inner;
      }
    }

    #[inline]
    pub(crate) fn simd_lt_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse2")] {
            Self(_mm_cmplt_epi16(self.0, rhs.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(i16x8_lt(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vreinterpretq_s16_u16(vcltq_s16(self.0, rhs.0)))
          } else {
            Self::from_array([
              if self.0.arr[0] < rhs.0.arr[0] { -1 } else { 0 },
              if self.0.arr[1] < rhs.0.arr[1] { -1 } else { 0 },
              if self.0.arr[2] < rhs.0.arr[2] { -1 } else { 0 },
              if self.0.arr[3] < rhs.0.arr[3] { -1 } else { 0 },
              if self.0.arr[4] < rhs.0.arr[4] { -1 } else { 0 },
              if self.0.arr[5] < rhs.0.arr[5] { -1 } else { 0 },
              if self.0.arr[6] < rhs.0.arr[6] { -1 } else { 0 },
              if self.0.arr[7] < rhs.0.arr[7] { -1 } else { 0 },
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn simd_gt_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse2")] {
            Self(_mm_cmpgt_epi16(self.0, rhs.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(i16x8_gt(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vreinterpretq_s16_u16(vcgtq_s16(self.0, rhs.0)))
          } else {
            Self::from_array([
              if self.0.arr[0] > rhs.0.arr[0] { -1 } else { 0 },
              if self.0.arr[1] > rhs.0.arr[1] { -1 } else { 0 },
              if self.0.arr[2] > rhs.0.arr[2] { -1 } else { 0 },
              if self.0.arr[3] > rhs.0.arr[3] { -1 } else { 0 },
              if self.0.arr[4] > rhs.0.arr[4] { -1 } else { 0 },
              if self.0.arr[5] > rhs.0.arr[5] { -1 } else { 0 },
              if self.0.arr[6] > rhs.0.arr[6] { -1 } else { 0 },
              if self.0.arr[7] > rhs.0.arr[7] { -1 } else { 0 },
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
            Self(i16x8_le(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            !self.simd_gt(rhs)
          } else {
            Self::from_array([
              if self.0.arr[0] <= rhs.0.arr[0] { -1 } else { 0 },
              if self.0.arr[1] <= rhs.0.arr[1] { -1 } else { 0 },
              if self.0.arr[2] <= rhs.0.arr[2] { -1 } else { 0 },
              if self.0.arr[3] <= rhs.0.arr[3] { -1 } else { 0 },
              if self.0.arr[4] <= rhs.0.arr[4] { -1 } else { 0 },
              if self.0.arr[5] <= rhs.0.arr[5] { -1 } else { 0 },
              if self.0.arr[6] <= rhs.0.arr[6] { -1 } else { 0 },
              if self.0.arr[7] <= rhs.0.arr[7] { -1 } else { 0 },
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
            Self(i16x8_ge(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            !self.simd_lt(rhs)
          } else {
            Self::from_array([
              if self.0.arr[0] >= rhs.0.arr[0] { -1 } else { 0 },
              if self.0.arr[1] >= rhs.0.arr[1] { -1 } else { 0 },
              if self.0.arr[2] >= rhs.0.arr[2] { -1 } else { 0 },
              if self.0.arr[3] >= rhs.0.arr[3] { -1 } else { 0 },
              if self.0.arr[4] >= rhs.0.arr[4] { -1 } else { 0 },
              if self.0.arr[5] >= rhs.0.arr[5] { -1 } else { 0 },
              if self.0.arr[6] >= rhs.0.arr[6] { -1 } else { 0 },
              if self.0.arr[7] >= rhs.0.arr[7] { -1 } else { 0 },
            ])
          }
        }
    }
}
}

safe_target_feature! {
// Backend of SIMD integer functions.
impl i16x8 {
    #[inline]
    pub(crate) fn shr_backend(self, rhs: u16x8) -> Self {
        pick! {
          if #[cfg(all(target_feature="avx512bw", target_feature="avx512vl"))] {
            // Mask `rhs` to 15 to match `wrapping_shr`.
            let rhs = _mm_and_si128(rhs.0, _mm_set1_epi16(15));
            cast(_mm_srav_epi16(self.0, rhs))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))] {
            // Mask `rhs` to 15 to match `wrapping_shr`, and negate it because
            // there is no shift-right intrinsic.
            let neg_rhs = vnegq_s16(vreinterpretq_s16_u16(vandq_u16(rhs.0, vmovq_n_u16(15))));
            Self(vshlq_s16(self.0, neg_rhs))
          } else {
            let self_array = self.to_array();
            let rhs_array = rhs.to_array();

            Self::from_array([
              self_array[0].wrapping_shr(rhs_array[0] as u32),
              self_array[1].wrapping_shr(rhs_array[1] as u32),
              self_array[2].wrapping_shr(rhs_array[2] as u32),
              self_array[3].wrapping_shr(rhs_array[3] as u32),
              self_array[4].wrapping_shr(rhs_array[4] as u32),
              self_array[5].wrapping_shr(rhs_array[5] as u32),
              self_array[6].wrapping_shr(rhs_array[6] as u32),
              self_array[7].wrapping_shr(rhs_array[7] as u32),
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn shr_scalar_backend(self, rhs: u32) -> Self {
        pick! {
          if #[cfg(target_feature="sse2")] {
            // Use `rhs % 16` to perform wrapping shift and not unbounded shift.
            let shift = cast([rhs as u64 & 15, 0]);
            Self(_mm_sra_epi16(self.0, shift))
          } else if #[cfg(target_feature="simd128")] {
            Self(i16x8_shr(self.0, rhs))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            // Use `rhs % 16` to perform wrapping shift and not unbounded shift.
            Self(vshlq_s16(self.0, vmovq_n_s16( -(rhs as i16 & 15))))
          } else {
            Self::from_array([
              self.0.arr[0].wrapping_shr(rhs),
              self.0.arr[1].wrapping_shr(rhs),
              self.0.arr[2].wrapping_shr(rhs),
              self.0.arr[3].wrapping_shr(rhs),
              self.0.arr[4].wrapping_shr(rhs),
              self.0.arr[5].wrapping_shr(rhs),
              self.0.arr[6].wrapping_shr(rhs),
              self.0.arr[7].wrapping_shr(rhs),
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn max_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse2")] {
            Self(_mm_max_epi16(self.0, rhs.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(i16x8_max(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vmaxq_s16(self.0, rhs.0))
          } else {
            self.simd_lt(rhs).select(rhs, self)
          }
        }
    }

    #[inline]
    pub(crate) fn min_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse2")] {
            Self(_mm_min_epi16(self.0, rhs.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(i16x8_min(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vminq_s16(self.0, rhs.0))
          } else {
            self.simd_lt(rhs).select(self, rhs)
          }
        }
    }

    #[inline]
    pub(crate) fn reduce_max_backend(self) -> i16 {
        pick! {
            if #[cfg(target_feature="sse2")] {
              let hi64 = _mm_shuffle_epi32::<0b01_00_11_10>(self.0);
              let sum64 = _mm_max_epi16(self.0, hi64);
              let hi32 = _mm_shuffle_epi32::<0b11_10_00_01>(sum64);
              let sum32 = _mm_max_epi16(sum64, hi32);
              let lo16 = _mm_srli_epi32::<16>(sum32);
              let sum16 = _mm_max_epi16(sum32, lo16);
              _mm_extract_epi16::<0>(sum16) as i16
            } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
              vmaxvq_s16(self.0)
            } else {
            let arr: [i16; 8] = cast(self);

            // most boring implementation possible so optimizer doesn't overthink this
            let mut r = arr[0];
            r = r.max(arr[1]);
            r = r.max(arr[2]);
            r = r.max(arr[3]);
            r = r.max(arr[4]);
            r = r.max(arr[5]);
            r = r.max(arr[6]);
            r.max(arr[7])
          }
        }
    }

    #[inline]
    pub(crate) fn reduce_min_backend(self) -> i16 {
        pick! {
            if #[cfg(target_feature="sse2")] {
              let hi64 = _mm_shuffle_epi32::<0b01_00_11_10>(self.0);
              let sum64 = _mm_min_epi16(self.0, hi64);
              let hi32 = _mm_shuffle_epi32::<0b11_10_00_01>(sum64);
              let sum32 = _mm_min_epi16(sum64, hi32);
              let lo16 = _mm_srli_epi32::<16>(sum32);
              let sum16 = _mm_min_epi16(sum32, lo16);
              _mm_extract_epi16::<0>(sum16) as i16
            } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
              vminvq_s16(self.0)
            } else {
            let arr: [i16; 8] = cast(self);

            // most boring implementation possible so optimizer doesn't overthink this
            let mut r = arr[0];
            r = r.min(arr[1]);
            r = r.min(arr[2]);
            r = r.min(arr[3]);
            r = r.min(arr[4]);
            r = r.min(arr[5]);
            r = r.min(arr[6]);
            r.min(arr[7])
          }
        }
    }

    #[inline]
    pub(crate) fn unbounded_shr_backend(self, rhs: u16x8) -> Self {
        pick! {
          if #[cfg(all(target_feature="avx512bw", target_feature="avx512vl"))] {
            cast(_mm_srav_epi16(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))] {
            // Negate `rhs` because there is no direct shift-right intrinsic, and
            // restrict it to prevent overflow.
            Self(vshlq_s16(self.0, vnegq_s16(vreinterpretq_s16_u16(rhs.min(u16x8::splat(16)).0))))
          } else {
            let self_array = self.to_array();
            let rhs_array = rhs.to_array();

            Self::from_array([
              self_array[0].unbounded_shr(rhs_array[0] as u32),
              self_array[1].unbounded_shr(rhs_array[1] as u32),
              self_array[2].unbounded_shr(rhs_array[2] as u32),
              self_array[3].unbounded_shr(rhs_array[3] as u32),
              self_array[4].unbounded_shr(rhs_array[4] as u32),
              self_array[5].unbounded_shr(rhs_array[5] as u32),
              self_array[6].unbounded_shr(rhs_array[6] as u32),
              self_array[7].unbounded_shr(rhs_array[7] as u32),
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn unbounded_shr_scalar_backend(self, rhs: u32) -> Self {
        pick! {
          if #[cfg(target_feature="sse2")] {
            Self(_mm_sra_epi16(self.0, cast([rhs as u64, 0])))
          } else if #[cfg(target_feature="simd128")] {
            if rhs < 16 { Self(i16x8_shr(self.0, rhs)) } else { self.is_negative() }
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            // Negate `rhs` because there is no direct shift-right intrinsic, and
            // restrict it to prevent overflow.
            Self(vshlq_s16(self.0, vmovq_n_s16(-rhs.min(16).cast_signed() as i16)))
          } else {
            Self::from_array([
                self.0.arr[0].unbounded_shr(rhs),
                self.0.arr[1].unbounded_shr(rhs),
                self.0.arr[2].unbounded_shr(rhs),
                self.0.arr[3].unbounded_shr(rhs),
                self.0.arr[4].unbounded_shr(rhs),
                self.0.arr[5].unbounded_shr(rhs),
                self.0.arr[6].unbounded_shr(rhs),
                self.0.arr[7].unbounded_shr(rhs),
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn saturating_add_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse2")] {
            Self(_mm_adds_epi16(self.0, rhs.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(i16x8_add_sat(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vqaddq_s16(self.0, rhs.0))
          } else {
            Self::from_array([
              self.0.arr[0].saturating_add(rhs.0.arr[0]),
              self.0.arr[1].saturating_add(rhs.0.arr[1]),
              self.0.arr[2].saturating_add(rhs.0.arr[2]),
              self.0.arr[3].saturating_add(rhs.0.arr[3]),
              self.0.arr[4].saturating_add(rhs.0.arr[4]),
              self.0.arr[5].saturating_add(rhs.0.arr[5]),
              self.0.arr[6].saturating_add(rhs.0.arr[6]),
              self.0.arr[7].saturating_add(rhs.0.arr[7]),
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn saturating_sub_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse2")] {
            Self(_mm_subs_epi16(self.0, rhs.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(i16x8_sub_sat(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vqsubq_s16(self.0, rhs.0))
          } else {
            Self::from_array([
              self.0.arr[0].saturating_sub(rhs.0.arr[0]),
              self.0.arr[1].saturating_sub(rhs.0.arr[1]),
              self.0.arr[2].saturating_sub(rhs.0.arr[2]),
              self.0.arr[3].saturating_sub(rhs.0.arr[3]),
              self.0.arr[4].saturating_sub(rhs.0.arr[4]),
              self.0.arr[5].saturating_sub(rhs.0.arr[5]),
              self.0.arr[6].saturating_sub(rhs.0.arr[6]),
              self.0.arr[7].saturating_sub(rhs.0.arr[7]),
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn overflowing_mul_backend(self, rhs: Self) -> (Self, Self) {
        let (low, high) = self.mul_keep_low_high(rhs);
        let low = cast::<u16x8, i16x8>(low);

        let overflow = high.simd_ne(low.is_negative());
        (low, overflow)
    }

    #[inline]
    pub(crate) fn widening_mul_backend(self, rhs: Self) -> i32x8 {
        pick! {
          if #[cfg(target_feature="avx2")] {
            let a = _mm256_cvtepi16_epi32(self.0);
            let b = _mm256_cvtepi16_epi32(rhs.0);
            i32x8(_mm256_mullo_epi32(a,b))
          } else if #[cfg(target_feature="sse2")] {
            let low = _mm_mullo_epi16(self.0, rhs.0);
            let high = _mm_mulhi_epi16(self.0, rhs.0);
            i32x8(crate::i32x8_backend::Inner {
              lo: i32x4(_mm_unpacklo_epi16(low, high)),
              hi: i32x4(_mm_unpackhi_epi16(low, high))
            })
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))] {
            let lhs_low = vget_low_s16(self.0);
            let rhs_low = vget_low_s16(rhs.0);

            let lhs_high = vget_high_s16(self.0);
            let rhs_high = vget_high_s16(rhs.0);

            let low = vmull_s16(lhs_low, rhs_low);
            let high = vmull_s16(lhs_high, rhs_high);

            i32x8(crate::i32x8_backend::Inner { lo: i32x4(low), hi: i32x4(high) })
          } else {
            let a = self.as_array();
            let b = rhs.as_array();

            i32x8::from_array([
              i32::from(a[0]) * i32::from(b[0]),
              i32::from(a[1]) * i32::from(b[1]),
              i32::from(a[2]) * i32::from(b[2]),
              i32::from(a[3]) * i32::from(b[3]),
              i32::from(a[4]) * i32::from(b[4]),
              i32::from(a[5]) * i32::from(b[5]),
              i32::from(a[6]) * i32::from(b[6]),
              i32::from(a[7]) * i32::from(b[7]),
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn mul_keep_low_high_backend(self, rhs: Self) -> (u16x8, i16x8) {
        pick! {
          if #[cfg(target_feature="simd128")] {
            let low_wide_mul = i32x4_extmul_low_i16x8(self.0, rhs.0);
            let high_wide_mul = i32x4_extmul_high_i16x8(self.0, rhs.0);
            (
              u16x8(i16x8_shuffle::<0, 2, 4, 6, 8, 10, 12, 14>(low_wide_mul, high_wide_mul)),
              i16x8(i16x8_shuffle::<1, 3, 5, 7, 9, 11, 13, 15>(low_wide_mul, high_wide_mul)),
            )
          } else if #[cfg(all(target_feature="neon", target_arch="aarch64"))] {
            let low_wide_mul = vreinterpretq_s16_s32(
              vmull_s16(vget_low_s16(self.0), vget_low_s16(rhs.0)),
            );
            let high_wide_mul = vreinterpretq_s16_s32(
              vmull_s16(vget_high_s16(self.0), vget_high_s16(rhs.0)),
            );
            let low_high = vuzpq_s16(low_wide_mul, high_wide_mul);
            (
              u16x8(vreinterpretq_u16_s16(low_high.0)),
              i16x8(low_high.1),
            )
          } else {
            // TODO(perf): This implementation looks quite bad. Is there a better
            // one?

            let self_array = self.to_array();
            let rhs_array = rhs.to_array();

            let widening_mul = [
              (self_array[0] as i32).wrapping_mul(rhs_array[0] as i32),
              (self_array[1] as i32).wrapping_mul(rhs_array[1] as i32),
              (self_array[2] as i32).wrapping_mul(rhs_array[2] as i32),
              (self_array[3] as i32).wrapping_mul(rhs_array[3] as i32),
              (self_array[4] as i32).wrapping_mul(rhs_array[4] as i32),
              (self_array[5] as i32).wrapping_mul(rhs_array[5] as i32),
              (self_array[6] as i32).wrapping_mul(rhs_array[6] as i32),
              (self_array[7] as i32).wrapping_mul(rhs_array[7] as i32),
            ];

            (
              u16x8::from_array([
                widening_mul[0] as u16,
                widening_mul[1] as u16,
                widening_mul[2] as u16,
                widening_mul[3] as u16,
                widening_mul[4] as u16,
                widening_mul[5] as u16,
                widening_mul[6] as u16,
                widening_mul[7] as u16,
              ]),
              i16x8::from_array([
                (widening_mul[0] >> 16) as i16,
                (widening_mul[1] >> 16) as i16,
                (widening_mul[2] >> 16) as i16,
                (widening_mul[3] >> 16) as i16,
                (widening_mul[4] >> 16) as i16,
                (widening_mul[5] >> 16) as i16,
                (widening_mul[6] >> 16) as i16,
                (widening_mul[7] >> 16) as i16,
              ]),
            )
          }
        }
    }

    #[inline]
    pub(crate) fn mul_keep_high_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse2")] {
            Self(_mm_mulhi_epi16(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))] {
            let lhs_low = vget_low_s16(self.0);
            let rhs_low = vget_low_s16(rhs.0);

            let lhs_high = vget_high_s16(self.0);
            let rhs_high = vget_high_s16(rhs.0);

            let low = vmull_s16(lhs_low, rhs_low);
            let high = vmull_s16(lhs_high, rhs_high);

            Self(vreinterpretq_s16_u16(
              vuzpq_u16(vreinterpretq_u16_s32(low),
              vreinterpretq_u16_s32(high)).1
            ))
          } else if #[cfg(target_feature="simd128")] {
            let low =  i32x4_extmul_low_i16x8(self.0, rhs.0);
            let high = i32x4_extmul_high_i16x8(self.0, rhs.0);

            Self(i16x8_shuffle::<1, 3, 5, 7, 9, 11, 13, 15>(low, high))
          } else {
            i16x8::from_array([
              ((i32::from(rhs.as_array()[0]) * i32::from(self.as_array()[0])) >> 16) as i16,
              ((i32::from(rhs.as_array()[1]) * i32::from(self.as_array()[1])) >> 16) as i16,
              ((i32::from(rhs.as_array()[2]) * i32::from(self.as_array()[2])) >> 16) as i16,
              ((i32::from(rhs.as_array()[3]) * i32::from(self.as_array()[3])) >> 16) as i16,
              ((i32::from(rhs.as_array()[4]) * i32::from(self.as_array()[4])) >> 16) as i16,
              ((i32::from(rhs.as_array()[5]) * i32::from(self.as_array()[5])) >> 16) as i16,
              ((i32::from(rhs.as_array()[6]) * i32::from(self.as_array()[6])) >> 16) as i16,
              ((i32::from(rhs.as_array()[7]) * i32::from(self.as_array()[7])) >> 16) as i16,
            ])
          }
        }
    }
}
}

safe_target_feature! {
// Backend of SIMD signed integer functions.
impl i16x8 {
    #[inline]
    pub(crate) fn abs_backend(self) -> Self {
        pick! {
          if #[cfg(target_feature="ssse3")] {
            Self(_mm_abs_epi16(self.0))
          } else if #[cfg(target_feature="sse2")] {
            let mask = _mm_srai_epi16::<15>(self.0);
            Self(_mm_xor_si128(_mm_add_epi16(self.0, mask), mask))
          } else if #[cfg(target_feature="simd128")] {
            Self(i16x8_abs(self.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vabsq_s16(self.0))
          } else {
            let arr: [i16; 8] = cast(self);
            cast(
              [
                arr[0].wrapping_abs(),
                arr[1].wrapping_abs(),
                arr[2].wrapping_abs(),
                arr[3].wrapping_abs(),
                arr[4].wrapping_abs(),
                arr[5].wrapping_abs(),
                arr[6].wrapping_abs(),
                arr[7].wrapping_abs(),
              ])
          }
        }
    }

    #[inline]
    pub(crate) fn is_positive_backend(self) -> Self {
        pick! {
          if #[cfg(all(target_feature="neon", target_arch="aarch64"))] {
            Self(vreinterpretq_s16_u16(vcgtzq_s16(self.0)))
          } else {
            self.simd_gt(Self::ZERO)
          }
        }
    }

    #[inline]
    pub(crate) fn is_negative_backend(self) -> Self {
        pick! {
          if #[cfg(all(target_feature="neon", target_arch="aarch64"))] {
            Self(vreinterpretq_s16_u16(vcltzq_s16(self.0)))
          } else if #[cfg(target_feature = "sse2")] {
            Self(_mm_srai_epi16::<15>(self.0))
          } else {
            self.simd_lt(Self::ZERO)
          }
        }
    }

    #[inline]
    pub(crate) fn saturating_neg_backend(self) -> Self {
        pick! {
            if #[cfg(target_feature="sse2")] {
                Self(_mm_subs_epi16(_mm_setzero_si128(), self.0))
            } else if #[cfg(target_feature="simd128")] {
                Self(i16x8_sub_sat(Self::ZERO.0, self.0))
            } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
                Self(vqnegq_s16(self.0))
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
                Self(vqabsq_s16(self.0))
            } else {
                // If `self == MIN`, flip all bits to change the result from
                // `MIN` to `MAX`
                self.abs() ^ self.simd_eq(Self::MIN)
            }
        }
    }

    #[inline]
    pub(crate) fn saturating_add_unsigned_backend(self, rhs: u16x8) -> Self {
        pick! {
            if #[cfg(all(target_feature = "neon", target_arch = "aarch64"))] {
                Self(vuqaddq_s16(self.0, rhs.0))
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
impl i16x8 {
    #[inline]
    pub(crate) fn from_u8x16_low_backend(u: u8x16) -> Self {
        pick! {
          if #[cfg(target_feature="sse2")] {
            Self(_mm_unpacklo_epi8(u.0, __m128i::zeroed()))
          } else {
            let u_arr: [u8; 16] = cast(u);
            cast([
              u_arr[0] as u16 as i16,
              u_arr[1] as u16 as i16,
              u_arr[2] as u16 as i16,
              u_arr[3] as u16 as i16,
              u_arr[4] as u16 as i16,
              u_arr[5] as u16 as i16,
              u_arr[6] as u16 as i16,
              u_arr[7] as u16 as i16,
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn from_u8x16_high_backend(u: u8x16) -> Self {
        pick! {
          if #[cfg(target_feature="sse2")] {
            Self(_mm_unpackhi_epi8(u.0, __m128i::zeroed()))
          } else {
            let u_arr: [u8; 16] = cast(u);
            cast([
              u_arr[8] as u16 as i16,
              u_arr[9] as u16 as i16,
              u_arr[10] as u16 as i16,
              u_arr[11] as u16 as i16,
              u_arr[12] as u16 as i16,
              u_arr[13] as u16 as i16,
              u_arr[14] as u16 as i16,
              u_arr[15] as u16 as i16,
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn from_i32x8_saturate_backend(v: i32x8) -> Self {
        pick! {
          if #[cfg(target_feature="avx2")] {
            i16x8(_mm_packs_epi32( _mm256_extractf128_si256::<0>(v.0), _mm256_extractf128_si256::<1>(v.0)))
          } else if #[cfg(target_feature="sse2")] {
            i16x8(_mm_packs_epi32( v.0.lo.0, v.0.hi.0 ))
          } else if #[cfg(target_feature="simd128")] {
            i16x8(i16x8_narrow_i32x4(v.0.lo.0, v.0.hi.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))] {
            i16x8(vcombine_s16(vqmovn_s32(v.0.lo.0), vqmovn_s32(v.0.hi.0)))
          } else {
            fn clamp(a : i32) -> i16 {
                if a < i16::MIN as i32 {
                    i16::MIN
                }
                else if a > i16::MAX as i32 {
                    i16::MAX
                } else {
                    a as i16
                }
            }

            i16x8::from_array([
              clamp(v.as_array()[0]),
              clamp(v.as_array()[1]),
              clamp(v.as_array()[2]),
              clamp(v.as_array()[3]),
              clamp(v.as_array()[4]),
              clamp(v.as_array()[5]),
              clamp(v.as_array()[6]),
              clamp(v.as_array()[7]),
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn from_i32x8_truncate_backend(v: i32x8) -> Self {
        pick! {
          if #[cfg(target_feature="avx2")] {
            let a = _mm256_and_si256(v.0, _mm256_set1_epi32(0xffff));
            i16x8(_mm_packus_epi32( _mm256_extractf128_si256::<0>(a), _mm256_extractf128_si256::<1>(a) ))
          } else if #[cfg(target_feature="sse2")] {
            let a = _mm_srai_epi32::<16>(_mm_slli_epi32::<16>(v.0.lo.0));
            let b = _mm_srai_epi32::<16>(_mm_slli_epi32::<16>(v.0.hi.0));

            i16x8(_mm_packs_epi32( a, b))
          } else {
          i16x8::from_array([
            v.as_array()[0] as i16,
            v.as_array()[1] as i16,
            v.as_array()[2] as i16,
            v.as_array()[3] as i16,
            v.as_array()[4] as i16,
            v.as_array()[5] as i16,
            v.as_array()[6] as i16,
            v.as_array()[7] as i16,
          ])
          }
        }
    }

    #[inline]
    pub(crate) fn from_slice_unaligned_backend(input: &[i16]) -> Self {
        assert!(input.len() >= 8);

        pick! {
          if #[cfg(target_feature="sse2")] {
            unsafe { Self(_mm_loadu_si128(input.as_ptr() as *const __m128i)) }
          } else if #[cfg(target_feature="simd128")] {
            unsafe { Self(v128_load(input.as_ptr() as *const v128 )) }
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            unsafe { Self(vld1q_s16( input.as_ptr() as *const i16 )) }
          } else {
            // 2018 edition doesn't have try_into
            unsafe { Self::from_array( *(input.as_ptr() as * const [i16;8]) ) }
          }
        }
    }

    #[inline]
    pub(crate) fn dot_backend(self, rhs: Self) -> i32x4 {
        pick! {
          if #[cfg(target_feature="sse2")] {
            i32x4(_mm_madd_epi16(self.0, rhs.0))
          } else if #[cfg(target_feature="simd128")] {
            i32x4(i32x4_dot_i16x8(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            let pl = vmull_s16(vget_low_s16(self.0),  vget_low_s16(rhs.0));
            let ph = vmull_high_s16(self.0, rhs.0);
            i32x4(vpaddq_s32(pl, ph))
          } else {
            i32x4::from_array([
              (i32::from(self.0.arr[0]) * i32::from(rhs.0.arr[0])) + (i32::from(self.0.arr[1]) * i32::from(rhs.0.arr[1])),
              (i32::from(self.0.arr[2]) * i32::from(rhs.0.arr[2])) + (i32::from(self.0.arr[3]) * i32::from(rhs.0.arr[3])),
              (i32::from(self.0.arr[4]) * i32::from(rhs.0.arr[4])) + (i32::from(self.0.arr[5]) * i32::from(rhs.0.arr[5])),
              (i32::from(self.0.arr[6]) * i32::from(rhs.0.arr[6])) + (i32::from(self.0.arr[7]) * i32::from(rhs.0.arr[7])),
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn mul_scale_round_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="ssse3")] {
            Self(_mm_mulhrs_epi16(self.0, rhs.0))
          } else if #[cfg(target_feature="sse2")] {
            // unfortunately _mm_mulhrs_epi16 only got added in sse3
            let hi = _mm_mulhi_epi16(self.0, rhs.0);
            let lo = _mm_mullo_epi16(self.0, rhs.0);
            let mut v1 = _mm_unpacklo_epi16(lo, hi);
            let mut v2 = _mm_unpackhi_epi16(lo, hi);
            let a = _mm_set1_epi32(0x4000);
            v1 = _mm_srai_epi32::<15>(_mm_add_epi32(v1, a));
            v2 = _mm_srai_epi32::<15>(_mm_add_epi32(v2, a));
            let s = _mm_packs_epi32(v1, v2);
            Self(s)
          } else if #[cfg(target_feature="simd128")] {
            Self(i16x8_q15mulr_sat(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vqrdmulhq_s16(self.0, rhs.0))
          } else {
            // compiler does a surprisingly good job of vectorizing this
            Self::from_array([
              ((i32::from(self.0.arr[0]) * i32::from(rhs.0.arr[0]) + 0x4000) >> 15) as i16,
              ((i32::from(self.0.arr[1]) * i32::from(rhs.0.arr[1]) + 0x4000) >> 15) as i16,
              ((i32::from(self.0.arr[2]) * i32::from(rhs.0.arr[2]) + 0x4000) >> 15) as i16,
              ((i32::from(self.0.arr[3]) * i32::from(rhs.0.arr[3]) + 0x4000) >> 15) as i16,
              ((i32::from(self.0.arr[4]) * i32::from(rhs.0.arr[4]) + 0x4000) >> 15) as i16,
              ((i32::from(self.0.arr[5]) * i32::from(rhs.0.arr[5]) + 0x4000) >> 15) as i16,
              ((i32::from(self.0.arr[6]) * i32::from(rhs.0.arr[6]) + 0x4000) >> 15) as i16,
              ((i32::from(self.0.arr[7]) * i32::from(rhs.0.arr[7]) + 0x4000) >> 15) as i16,
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn mul_scale_round_n_backend(self, rhs: i16) -> Self {
        pick! {
          if #[cfg(target_feature="ssse3")] {
            Self( _mm_mulhrs_epi16(self.0, _mm_set1_epi16(rhs)))
          } else if #[cfg(target_feature="sse2")] {
            // unfortunately _mm_mulhrs_epi16 only got added in sse3
            let r = _mm_set1_epi16(rhs);
            let hi = _mm_mulhi_epi16(self.0, r);
            let lo = _mm_mullo_epi16(self.0, r);
            let mut v1 = _mm_unpacklo_epi16(lo, hi);
            let mut v2 = _mm_unpackhi_epi16(lo, hi);
            let a = _mm_set1_epi32(0x4000);
            v1 = _mm_srai_epi32::<15>(_mm_add_epi32(v1, a));
            v2 = _mm_srai_epi32::<15>(_mm_add_epi32(v2, a));
            let s = _mm_packs_epi32(v1, v2);
            Self(s)
          } else if #[cfg(target_feature="simd128")] {
            Self(i16x8_q15mulr_sat(self.0, i16x8_splat(rhs)))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vqrdmulhq_n_s16(self.0, rhs))
          } else {
            // compiler does a surprisingly good job of vectorizing this
            Self::from_array([
              ((i32::from(self.0.arr[0]) * i32::from(rhs) + 0x4000) >> 15) as i16,
              ((i32::from(self.0.arr[1]) * i32::from(rhs) + 0x4000) >> 15) as i16,
              ((i32::from(self.0.arr[2]) * i32::from(rhs) + 0x4000) >> 15) as i16,
              ((i32::from(self.0.arr[3]) * i32::from(rhs) + 0x4000) >> 15) as i16,
              ((i32::from(self.0.arr[4]) * i32::from(rhs) + 0x4000) >> 15) as i16,
              ((i32::from(self.0.arr[5]) * i32::from(rhs) + 0x4000) >> 15) as i16,
              ((i32::from(self.0.arr[6]) * i32::from(rhs) + 0x4000) >> 15) as i16,
              ((i32::from(self.0.arr[7]) * i32::from(rhs) + 0x4000) >> 15) as i16,
            ])
          }
        }
    }
}
}
