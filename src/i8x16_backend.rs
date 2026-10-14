use crate::imports::*;

pick! {
  if #[cfg(target_feature="sse2")] {
    pub(crate) type Inner = m128i;
  } else if #[cfg(target_feature="simd128")] {
    pub(crate) type Inner = v128;
  } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
    pub(crate) type Inner = int8x16_t;
  } else {
    #[repr(C, align(16))]
    #[derive(Clone, Copy)]
    pub(crate) struct Inner {
      pub arr: [i8; 16],
    }
  }
}

impl i8x16 {
    #[inline]
    pub(crate) fn simd_lt_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse2")] {
            Self(cmp_lt_mask_i8_m128i(self.0, rhs.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(i8x16_lt(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            unsafe {Self(vreinterpretq_s8_u8(vcltq_s8(self.0, rhs.0)))}
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
              if self.0.arr[8] < rhs.0.arr[8] { -1 } else { 0 },
              if self.0.arr[9] < rhs.0.arr[9] { -1 } else { 0 },
              if self.0.arr[10] < rhs.0.arr[10] { -1 } else { 0 },
              if self.0.arr[11] < rhs.0.arr[11] { -1 } else { 0 },
              if self.0.arr[12] < rhs.0.arr[12] { -1 } else { 0 },
              if self.0.arr[13] < rhs.0.arr[13] { -1 } else { 0 },
              if self.0.arr[14] < rhs.0.arr[14] { -1 } else { 0 },
              if self.0.arr[15] < rhs.0.arr[15] { -1 } else { 0 },
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn simd_gt_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse2")] {
            Self(cmp_gt_mask_i8_m128i(self.0, rhs.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(i8x16_gt(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            unsafe {Self(vreinterpretq_s8_u8(vcgtq_s8(self.0, rhs.0)))}
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
              if self.0.arr[8] > rhs.0.arr[8] { -1 } else { 0 },
              if self.0.arr[9] > rhs.0.arr[9] { -1 } else { 0 },
              if self.0.arr[10] > rhs.0.arr[10] { -1 } else { 0 },
              if self.0.arr[11] > rhs.0.arr[11] { -1 } else { 0 },
              if self.0.arr[12] > rhs.0.arr[12] { -1 } else { 0 },
              if self.0.arr[13] > rhs.0.arr[13] { -1 } else { 0 },
              if self.0.arr[14] > rhs.0.arr[14] { -1 } else { 0 },
              if self.0.arr[15] > rhs.0.arr[15] { -1 } else { 0 },
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
            Self(i8x16_le(self.0, rhs.0))
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
              if self.0.arr[8] <= rhs.0.arr[8] { -1 } else { 0 },
              if self.0.arr[9] <= rhs.0.arr[9] { -1 } else { 0 },
              if self.0.arr[10] <= rhs.0.arr[10] { -1 } else { 0 },
              if self.0.arr[11] <= rhs.0.arr[11] { -1 } else { 0 },
              if self.0.arr[12] <= rhs.0.arr[12] { -1 } else { 0 },
              if self.0.arr[13] <= rhs.0.arr[13] { -1 } else { 0 },
              if self.0.arr[14] <= rhs.0.arr[14] { -1 } else { 0 },
              if self.0.arr[15] <= rhs.0.arr[15] { -1 } else { 0 },
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
            Self(i8x16_ge(self.0, rhs.0))
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
              if self.0.arr[8] >= rhs.0.arr[8] { -1 } else { 0 },
              if self.0.arr[9] >= rhs.0.arr[9] { -1 } else { 0 },
              if self.0.arr[10] >= rhs.0.arr[10] { -1 } else { 0 },
              if self.0.arr[11] >= rhs.0.arr[11] { -1 } else { 0 },
              if self.0.arr[12] >= rhs.0.arr[12] { -1 } else { 0 },
              if self.0.arr[13] >= rhs.0.arr[13] { -1 } else { 0 },
              if self.0.arr[14] >= rhs.0.arr[14] { -1 } else { 0 },
              if self.0.arr[15] >= rhs.0.arr[15] { -1 } else { 0 },
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn shr_backend(self, rhs: u8x16) -> Self {
        // For x86, this technically can be done explicitly by converting
        // to `i16` or `i32` then converting back after multiplication, but that may
        // not actually be faster than auto-vectorization.
        pick! {
          if #[cfg(all(target_feature="neon", target_arch="aarch64"))] {
            unsafe {
              // Mask `rhs` to 7 to match `wrapping_shr`, and negate it because
              // there is no shift-right intrinsic.
              let neg_rhs = vnegq_s8(vreinterpretq_s8_u8(vandq_u8(rhs.0, vmovq_n_u8(7))));
              Self(vshlq_s8(self.0, neg_rhs))
            }
          } else {
            let self_array: [i8; 16] = cast(self);
            let rhs_array: [u8; 16] = cast(rhs);

            Self::from_array([
              self_array[0].wrapping_shr(rhs_array[0] as u32),
              self_array[1].wrapping_shr(rhs_array[1] as u32),
              self_array[2].wrapping_shr(rhs_array[2] as u32),
              self_array[3].wrapping_shr(rhs_array[3] as u32),
              self_array[4].wrapping_shr(rhs_array[4] as u32),
              self_array[5].wrapping_shr(rhs_array[5] as u32),
              self_array[6].wrapping_shr(rhs_array[6] as u32),
              self_array[7].wrapping_shr(rhs_array[7] as u32),
              self_array[8].wrapping_shr(rhs_array[8] as u32),
              self_array[9].wrapping_shr(rhs_array[9] as u32),
              self_array[10].wrapping_shr(rhs_array[10] as u32),
              self_array[11].wrapping_shr(rhs_array[11] as u32),
              self_array[12].wrapping_shr(rhs_array[12] as u32),
              self_array[13].wrapping_shr(rhs_array[13] as u32),
              self_array[14].wrapping_shr(rhs_array[14] as u32),
              self_array[15].wrapping_shr(rhs_array[15] as u32),
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn shr_scalar_backend(self, rhs: u32) -> Self {
        // For x86, this technically can be done explicitly by converting
        // to `i16` or `i32` then converting back after multiplication, but that
        // may not actually be faster than auto-vectorization.
        pick! {
          if #[cfg(target_feature="simd128")] {
            // Mask `rhs` to 7 to match `wrapping_shr`.
            Self(i8x16_shr(self.0, rhs & 7))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            // Mask `rhs` to 7 to match `wrapping_shr`, and negate it because
            // there is no shift-right intrinsic.
            unsafe { Self(vshlq_s8(self.0, vmovq_n_s8(-(rhs as i8 & 7)))) }
          } else {
            let self_array = self.to_array();

            cast([
              self_array[0].wrapping_shr(rhs),
              self_array[1].wrapping_shr(rhs),
              self_array[2].wrapping_shr(rhs),
              self_array[3].wrapping_shr(rhs),
              self_array[4].wrapping_shr(rhs),
              self_array[5].wrapping_shr(rhs),
              self_array[6].wrapping_shr(rhs),
              self_array[7].wrapping_shr(rhs),
              self_array[8].wrapping_shr(rhs),
              self_array[9].wrapping_shr(rhs),
              self_array[10].wrapping_shr(rhs),
              self_array[11].wrapping_shr(rhs),
              self_array[12].wrapping_shr(rhs),
              self_array[13].wrapping_shr(rhs),
              self_array[14].wrapping_shr(rhs),
              self_array[15].wrapping_shr(rhs),
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn max_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse4.1")] {
            Self(max_i8_m128i(self.0, rhs.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(i8x16_max(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            unsafe {Self(vmaxq_s8(self.0, rhs.0))}
          } else {
            self.simd_lt(rhs).select(rhs, self)
          }
        }
    }

    #[inline]
    pub(crate) fn min_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse4.1")] {
            Self(min_i8_m128i(self.0, rhs.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(i8x16_min(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            unsafe {Self(vminq_s8(self.0, rhs.0))}
          } else {
            self.simd_lt(rhs).select(self, rhs)
          }
        }
    }

    #[inline]
    pub(crate) fn reduce_max_backend(self) -> i8 {
        #[allow(dead_code)]
        const SHUFFLE_1: [i8; 16] = [8, 9, 10, 11, 12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0];
        #[allow(dead_code)]
        const SHUFFLE_2: [i8; 16] = [4, 5, 6, 7, 0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0];
        #[allow(dead_code)]
        const SHUFFLE_3: [i8; 16] = [2, 3, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        #[allow(dead_code)]
        const SHUFFLE_4: [i8; 16] = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

        pick! {
          if #[cfg(all(target_feature="ssse3", target_feature="sse4.1"))] {
            let rhs = shuffle_av_i8z_all_m128i(self.0, m128i::from(SHUFFLE_1));
            let max = max_i8_m128i(self.0, rhs);
            let rhs = shuffle_av_i8z_all_m128i(max, m128i::from(SHUFFLE_2));
            let max = max_i8_m128i(max, rhs);
            let rhs = shuffle_av_i8z_all_m128i(max, m128i::from(SHUFFLE_3));
            let max = max_i8_m128i(max, rhs);
            let rhs = shuffle_av_i8z_all_m128i(max, m128i::from(SHUFFLE_4));
            let max = max_i8_m128i(max, rhs);
            get_i32_from_m128i_s(max) as i8
          } else if #[cfg(target_feature="simd128")] {
            let rhs = i8x16_shuffle::<8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7>(self.0, self.0);
            let max = i8x16_max(self.0, rhs);
            let rhs = i8x16_shuffle::<4, 5, 6, 7, 0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0>(max, max);
            let max = i8x16_max(max, rhs);
            let rhs = i8x16_shuffle::<2, 3, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>(max, max);
            let max = i8x16_max(max, rhs);
            let rhs = i8x16_shuffle::<1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>(max, max);
            let max = i8x16_max(max, rhs);
            i8x16_extract_lane::<0>(max)
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            unsafe {
              // Use `transmute` instead of `cast` because `int8x16_t` does not
              // implement `bytemuck::Pod`.
              let rhs = vqtbl1q_s8(self.0, core::mem::transmute(SHUFFLE_1));
              let max = vmaxq_s8(self.0, rhs);
              let rhs = vqtbl1q_s8(max, core::mem::transmute(SHUFFLE_2));
              let max = vmaxq_s8(max, rhs);
              let rhs = vqtbl1q_s8(max, core::mem::transmute(SHUFFLE_3));
              let max = vmaxq_s8(max, rhs);
              let rhs = vqtbl1q_s8(max, core::mem::transmute(SHUFFLE_4));
              let max = vmaxq_s8(max, rhs);
              vgetq_lane_s8(max, 0)
            }
          } else {
            let array: [i8; 16] = cast(self);
            array.into_iter().reduce(i8::max).unwrap()
          }
        }
    }

    #[inline]
    pub(crate) fn reduce_min_backend(self) -> i8 {
        #[allow(dead_code)]
        const SHUFFLE_1: [i8; 16] = [8, 9, 10, 11, 12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0];
        #[allow(dead_code)]
        const SHUFFLE_2: [i8; 16] = [4, 5, 6, 7, 0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0];
        #[allow(dead_code)]
        const SHUFFLE_3: [i8; 16] = [2, 3, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        #[allow(dead_code)]
        const SHUFFLE_4: [i8; 16] = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

        pick! {
          if #[cfg(all(target_feature="ssse3", target_feature="sse4.1"))] {
            let rhs = shuffle_av_i8z_all_m128i(self.0, m128i::from(SHUFFLE_1));
            let min = min_i8_m128i(self.0, rhs);
            let rhs = shuffle_av_i8z_all_m128i(min, m128i::from(SHUFFLE_2));
            let min = min_i8_m128i(min, rhs);
            let rhs = shuffle_av_i8z_all_m128i(min, m128i::from(SHUFFLE_3));
            let min = min_i8_m128i(min, rhs);
            let rhs = shuffle_av_i8z_all_m128i(min, m128i::from(SHUFFLE_4));
            let min = min_i8_m128i(min, rhs);
            get_i32_from_m128i_s(min) as i8
          } else if #[cfg(target_feature="simd128")] {
            let rhs = i8x16_shuffle::<8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7>(self.0, self.0);
            let min = i8x16_min(self.0, rhs);
            let rhs = i8x16_shuffle::<4, 5, 6, 7, 0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0>(min, min);
            let min = i8x16_min(min, rhs);
            let rhs = i8x16_shuffle::<2, 3, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>(min, min);
            let min = i8x16_min(min, rhs);
            let rhs = i8x16_shuffle::<1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>(min, min);
            let min = i8x16_min(min, rhs);
            i8x16_extract_lane::<0>(min)
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            unsafe {
              // Use `transmute` instead of `cast` because `int8x16_t` does not
              // implement `bytemuck::Pod`.
              let rhs = vqtbl1q_s8(self.0, core::mem::transmute(SHUFFLE_1));
              let min = vminq_s8(self.0, rhs);
              let rhs = vqtbl1q_s8(min, core::mem::transmute(SHUFFLE_2));
              let min = vminq_s8(min, rhs);
              let rhs = vqtbl1q_s8(min, core::mem::transmute(SHUFFLE_3));
              let min = vminq_s8(min, rhs);
              let rhs = vqtbl1q_s8(min, core::mem::transmute(SHUFFLE_4));
              let min = vminq_s8(min, rhs);
              vgetq_lane_s8(min, 0)
            }
          } else {
            let array: [i8; 16] = cast(self);
            array.into_iter().reduce(i8::min).unwrap()
          }
        }
    }

    #[inline]
    pub(crate) fn unbounded_shr_backend(self, rhs: u8x16) -> Self {
        // For x86, this technically can be done explicitly by converting
        // to `i16` or `i32` then converting back after multiplication, but that may
        // not actually be faster than auto-vectorization.
        pick! {
          if #[cfg(all(target_feature="neon", target_arch="aarch64"))] {
            unsafe {
              // Negate `rhs` because there is no direct shift-right intrinsic, and
              // restrict it to prevent overflow.
              let neg_rhs = vnegq_s8(vreinterpretq_s8_u8(rhs.min(u8x16::splat(8)).0));
              Self(vshlq_s8(self.0, neg_rhs))
            }
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
              self_array[8].unbounded_shr(rhs_array[8] as u32),
              self_array[9].unbounded_shr(rhs_array[9] as u32),
              self_array[10].unbounded_shr(rhs_array[10] as u32),
              self_array[11].unbounded_shr(rhs_array[11] as u32),
              self_array[12].unbounded_shr(rhs_array[12] as u32),
              self_array[13].unbounded_shr(rhs_array[13] as u32),
              self_array[14].unbounded_shr(rhs_array[14] as u32),
              self_array[15].unbounded_shr(rhs_array[15] as u32),
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn unbounded_shr_scalar_backend(self, rhs: u32) -> Self {
        // For x86, this technically can be done explicitly by converting
        // to `i16` or `i32` then converting back after multiplication, but that
        // may not actually be faster than auto-vectorization.
        pick! {
          if #[cfg(target_feature="simd128")] {
            if rhs < 8 { Self(i8x16_shr(self.0, rhs)) } else { self.is_negative() }
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            unsafe {
              // Negate `rhs` because there is no direct shift-right intrinsic, and
              // restrict it to prevent overflow.
              Self(vshlq_s8(self.0, vmovq_n_s8(-rhs.min(8).cast_signed() as i8)))
            }
          } else {
            let self_array = self.to_array();

            cast([
              self_array[0].unbounded_shr(rhs),
              self_array[1].unbounded_shr(rhs),
              self_array[2].unbounded_shr(rhs),
              self_array[3].unbounded_shr(rhs),
              self_array[4].unbounded_shr(rhs),
              self_array[5].unbounded_shr(rhs),
              self_array[6].unbounded_shr(rhs),
              self_array[7].unbounded_shr(rhs),
              self_array[8].unbounded_shr(rhs),
              self_array[9].unbounded_shr(rhs),
              self_array[10].unbounded_shr(rhs),
              self_array[11].unbounded_shr(rhs),
              self_array[12].unbounded_shr(rhs),
              self_array[13].unbounded_shr(rhs),
              self_array[14].unbounded_shr(rhs),
              self_array[15].unbounded_shr(rhs),
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn saturating_add_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse2")] {
            Self(add_saturating_i8_m128i(self.0, rhs.0) )
          } else if #[cfg(target_feature="simd128")] {
            Self(i8x16_add_sat(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            unsafe {Self(vqaddq_s8(self.0, rhs.0))}
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
              self.0.arr[8].saturating_add(rhs.0.arr[8]),
              self.0.arr[9].saturating_add(rhs.0.arr[9]),
              self.0.arr[10].saturating_add(rhs.0.arr[10]),
              self.0.arr[11].saturating_add(rhs.0.arr[11]),
              self.0.arr[12].saturating_add(rhs.0.arr[12]),
              self.0.arr[13].saturating_add(rhs.0.arr[13]),
              self.0.arr[14].saturating_add(rhs.0.arr[14]),
              self.0.arr[15].saturating_add(rhs.0.arr[15]),
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn saturating_sub_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse2")] {
            Self(sub_saturating_i8_m128i(self.0, rhs.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(i8x16_sub_sat(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            unsafe { Self(vqsubq_s8(self.0, rhs.0)) }
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
              self.0.arr[8].saturating_sub(rhs.0.arr[8]),
              self.0.arr[9].saturating_sub(rhs.0.arr[9]),
              self.0.arr[10].saturating_sub(rhs.0.arr[10]),
              self.0.arr[11].saturating_sub(rhs.0.arr[11]),
              self.0.arr[12].saturating_sub(rhs.0.arr[12]),
              self.0.arr[13].saturating_sub(rhs.0.arr[13]),
              self.0.arr[14].saturating_sub(rhs.0.arr[14]),
              self.0.arr[15].saturating_sub(rhs.0.arr[15]),
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn overflowing_mul_backend(self, rhs: Self) -> (Self, Self) {
        let (low, high) = self.mul_keep_low_high(rhs);
        let low = cast::<u8x16, i8x16>(low);

        let overflow = high.simd_ne(low.is_negative());
        (low, overflow)
    }

    #[inline]
    pub(crate) fn widening_mul_backend(self, rhs: Self) -> i16x16 {
        pick! {
          if #[cfg(all(target_feature="neon", target_arch="aarch64"))] {
            unsafe {
              let low_wide_mul = vmull_s8(vget_low_s8(self.0), vget_low_s8(rhs.0));
              let high_wide_mul = vmull_s8(vget_high_s8(self.0), vget_high_s8(rhs.0));

              i16x16(crate::i16x16_backend::Inner {
                lo: i16x8(low_wide_mul),
                hi: i16x8(high_wide_mul),
              })
            }
          } else {
            let self_array = self.to_array();
            let rhs_array = rhs.to_array();

            i16x16::from_array([
              (self_array[0] as i16).wrapping_mul(rhs_array[0] as i16),
              (self_array[1] as i16).wrapping_mul(rhs_array[1] as i16),
              (self_array[2] as i16).wrapping_mul(rhs_array[2] as i16),
              (self_array[3] as i16).wrapping_mul(rhs_array[3] as i16),
              (self_array[4] as i16).wrapping_mul(rhs_array[4] as i16),
              (self_array[5] as i16).wrapping_mul(rhs_array[5] as i16),
              (self_array[6] as i16).wrapping_mul(rhs_array[6] as i16),
              (self_array[7] as i16).wrapping_mul(rhs_array[7] as i16),
              (self_array[8] as i16).wrapping_mul(rhs_array[8] as i16),
              (self_array[9] as i16).wrapping_mul(rhs_array[9] as i16),
              (self_array[10] as i16).wrapping_mul(rhs_array[10] as i16),
              (self_array[11] as i16).wrapping_mul(rhs_array[11] as i16),
              (self_array[12] as i16).wrapping_mul(rhs_array[12] as i16),
              (self_array[13] as i16).wrapping_mul(rhs_array[13] as i16),
              (self_array[14] as i16).wrapping_mul(rhs_array[14] as i16),
              (self_array[15] as i16).wrapping_mul(rhs_array[15] as i16),
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn mul_keep_low_high_backend(self, rhs: Self) -> (u8x16, i8x16) {
        pick! {
          if #[cfg(all(target_feature="neon", target_arch="aarch64"))] {
            unsafe {
              let low_wide_mul = vreinterpretq_s8_s16(
                vmull_s8(vget_low_s8(self.0), vget_low_s8(rhs.0)),
              );
              let high_wide_mul = vreinterpretq_s8_s16(
                vmull_s8(vget_high_s8(self.0), vget_high_s8(rhs.0)),
              );
              let low_high = vuzpq_s8(low_wide_mul, high_wide_mul);

              (
                u8x16(vreinterpretq_u8_s8(low_high.0)),
                i8x16(low_high.1),
              )
            }
          } else {
            // TODO(perf): This implementation looks quite bad. Is there a better
            // one?

            let self_array = self.to_array();
            let rhs_array = rhs.to_array();

            let widening_mul = [
              (self_array[0] as i16).wrapping_mul(rhs_array[0] as i16),
              (self_array[1] as i16).wrapping_mul(rhs_array[1] as i16),
              (self_array[2] as i16).wrapping_mul(rhs_array[2] as i16),
              (self_array[3] as i16).wrapping_mul(rhs_array[3] as i16),
              (self_array[4] as i16).wrapping_mul(rhs_array[4] as i16),
              (self_array[5] as i16).wrapping_mul(rhs_array[5] as i16),
              (self_array[6] as i16).wrapping_mul(rhs_array[6] as i16),
              (self_array[7] as i16).wrapping_mul(rhs_array[7] as i16),
              (self_array[8] as i16).wrapping_mul(rhs_array[8] as i16),
              (self_array[9] as i16).wrapping_mul(rhs_array[9] as i16),
              (self_array[10] as i16).wrapping_mul(rhs_array[10] as i16),
              (self_array[11] as i16).wrapping_mul(rhs_array[11] as i16),
              (self_array[12] as i16).wrapping_mul(rhs_array[12] as i16),
              (self_array[13] as i16).wrapping_mul(rhs_array[13] as i16),
              (self_array[14] as i16).wrapping_mul(rhs_array[14] as i16),
              (self_array[15] as i16).wrapping_mul(rhs_array[15] as i16),
            ];

            (
              u8x16::from_array([
                widening_mul[0] as u8,
                widening_mul[1] as u8,
                widening_mul[2] as u8,
                widening_mul[3] as u8,
                widening_mul[4] as u8,
                widening_mul[5] as u8,
                widening_mul[6] as u8,
                widening_mul[7] as u8,
                widening_mul[8] as u8,
                widening_mul[9] as u8,
                widening_mul[10] as u8,
                widening_mul[11] as u8,
                widening_mul[12] as u8,
                widening_mul[13] as u8,
                widening_mul[14] as u8,
                widening_mul[15] as u8,
              ]),
              i8x16::from_array([
                (widening_mul[0] >> 8) as i8,
                (widening_mul[1] >> 8) as i8,
                (widening_mul[2] >> 8) as i8,
                (widening_mul[3] >> 8) as i8,
                (widening_mul[4] >> 8) as i8,
                (widening_mul[5] >> 8) as i8,
                (widening_mul[6] >> 8) as i8,
                (widening_mul[7] >> 8) as i8,
                (widening_mul[8] >> 8) as i8,
                (widening_mul[9] >> 8) as i8,
                (widening_mul[10] >> 8) as i8,
                (widening_mul[11] >> 8) as i8,
                (widening_mul[12] >> 8) as i8,
                (widening_mul[13] >> 8) as i8,
                (widening_mul[14] >> 8) as i8,
                (widening_mul[15] >> 8) as i8,
              ])
            )
          }
        }
    }

    #[inline]
    pub(crate) fn mul_keep_high_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(all(target_feature="neon", target_arch="aarch64"))] {
            unsafe {
              let low_wide_mul = vreinterpretq_s8_s16(
                vmull_s8(vget_low_s8(self.0), vget_low_s8(rhs.0)),
              );
              let high_wide_mul = vreinterpretq_s8_s16(
                vmull_s8(vget_high_s8(self.0), vget_high_s8(rhs.0)),
              );

              Self(vuzpq_s8(low_wide_mul, high_wide_mul).1)
            }
          } else {
            let self_array = self.to_array();
            let rhs_array = rhs.to_array();

            Self::from_array([
              ((self_array[0] as i16).wrapping_mul(rhs_array[0] as i16) >> 8) as i8,
              ((self_array[1] as i16).wrapping_mul(rhs_array[1] as i16) >> 8) as i8,
              ((self_array[2] as i16).wrapping_mul(rhs_array[2] as i16) >> 8) as i8,
              ((self_array[3] as i16).wrapping_mul(rhs_array[3] as i16) >> 8) as i8,
              ((self_array[4] as i16).wrapping_mul(rhs_array[4] as i16) >> 8) as i8,
              ((self_array[5] as i16).wrapping_mul(rhs_array[5] as i16) >> 8) as i8,
              ((self_array[6] as i16).wrapping_mul(rhs_array[6] as i16) >> 8) as i8,
              ((self_array[7] as i16).wrapping_mul(rhs_array[7] as i16) >> 8) as i8,
              ((self_array[8] as i16).wrapping_mul(rhs_array[8] as i16) >> 8) as i8,
              ((self_array[9] as i16).wrapping_mul(rhs_array[9] as i16) >> 8) as i8,
              ((self_array[10] as i16).wrapping_mul(rhs_array[10] as i16) >> 8) as i8,
              ((self_array[11] as i16).wrapping_mul(rhs_array[11] as i16) >> 8) as i8,
              ((self_array[12] as i16).wrapping_mul(rhs_array[12] as i16) >> 8) as i8,
              ((self_array[13] as i16).wrapping_mul(rhs_array[13] as i16) >> 8) as i8,
              ((self_array[14] as i16).wrapping_mul(rhs_array[14] as i16) >> 8) as i8,
              ((self_array[15] as i16).wrapping_mul(rhs_array[15] as i16) >> 8) as i8,
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn abs_backend(self) -> Self {
        pick! {
          if #[cfg(target_feature="ssse3")] {
            Self(abs_i8_m128i(self.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(i8x16_abs(self.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            unsafe {Self(vabsq_s8(self.0))}
          } else {
            let arr: [i8; 16] = cast(self);
            cast([
              arr[0].wrapping_abs(),
              arr[1].wrapping_abs(),
              arr[2].wrapping_abs(),
              arr[3].wrapping_abs(),
              arr[4].wrapping_abs(),
              arr[5].wrapping_abs(),
              arr[6].wrapping_abs(),
              arr[7].wrapping_abs(),
              arr[8].wrapping_abs(),
              arr[9].wrapping_abs(),
              arr[10].wrapping_abs(),
              arr[11].wrapping_abs(),
              arr[12].wrapping_abs(),
              arr[13].wrapping_abs(),
              arr[14].wrapping_abs(),
              arr[15].wrapping_abs(),
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn is_positive_backend(self) -> Self {
        pick! {
          if #[cfg(all(target_feature="neon", target_arch="aarch64"))] {
            Self(unsafe { vreinterpretq_s8_u8(vcgtzq_s8(self.0)) })
          } else {
            self.simd_gt(Self::ZERO)
          }
        }
    }

    #[inline]
    pub(crate) fn is_negative_backend(self) -> Self {
        pick! {
          if #[cfg(all(target_feature="neon", target_arch="aarch64"))] {
            Self(unsafe { vreinterpretq_s8_u8(vcltzq_s8(self.0)) })
          } else {
            self.simd_lt(Self::ZERO)
          }
        }
    }

    #[inline]
    pub(crate) fn from_i16x16_saturate_backend(v: i16x16) -> i8x16 {
        pick! {
          if #[cfg(target_feature="avx2")] {
            i8x16(pack_i16_to_i8_m128i( extract_m128i_from_m256i::<0>(v.0), extract_m128i_from_m256i::<1>(v.0)))
          } else if #[cfg(target_feature="sse2")] {
            i8x16(pack_i16_to_i8_m128i( v.0.lo.0, v.0.hi.0 ))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))] {
            use core::arch::aarch64::*;

            unsafe {
              i8x16(vcombine_s8(vqmovn_s16(v.0.lo.0), vqmovn_s16(v.0.hi.0)))
            }
          } else if #[cfg(target_feature="simd128")] {
            i8x16(i8x16_narrow_i16x8(v.0.lo.0, v.0.hi.0))
          } else {
            fn clamp(a : i16) -> i8 {
                if a < i8::MIN as i16 {
                  i8::MIN
                }
                else if a > i8::MAX as i16 {
                  i8::MAX
                } else {
                    a as i8
                }
            }

            i8x16::from_array([
              clamp(v.as_array()[0]),
              clamp(v.as_array()[1]),
              clamp(v.as_array()[2]),
              clamp(v.as_array()[3]),
              clamp(v.as_array()[4]),
              clamp(v.as_array()[5]),
              clamp(v.as_array()[6]),
              clamp(v.as_array()[7]),
              clamp(v.as_array()[8]),
              clamp(v.as_array()[9]),
              clamp(v.as_array()[10]),
              clamp(v.as_array()[11]),
              clamp(v.as_array()[12]),
              clamp(v.as_array()[13]),
              clamp(v.as_array()[14]),
              clamp(v.as_array()[15]),
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn from_i16x16_truncate_backend(v: i16x16) -> i8x16 {
        pick! {
          if #[cfg(target_feature="avx2")] {
            let a = v.0 & set_splat_i16_m256i(0xff);
            i8x16(pack_i16_to_u8_m128i( extract_m128i_from_m256i::<0>(a), extract_m128i_from_m256i::<1>(a)))
          } else if #[cfg(target_feature="sse2")] {
            let mask = set_splat_i16_m128i(0xff);
            i8x16(pack_i16_to_u8_m128i( v.0.lo.0 & mask, v.0.hi.0 & mask ))
          } else {
            // no super good intrinsics on other platforms... plain old codegen does a reasonable job
            i8x16::from_array([
              v.as_array()[0] as i8,
              v.as_array()[1] as i8,
              v.as_array()[2] as i8,
              v.as_array()[3] as i8,
              v.as_array()[4] as i8,
              v.as_array()[5] as i8,
              v.as_array()[6] as i8,
              v.as_array()[7] as i8,
              v.as_array()[8] as i8,
              v.as_array()[9] as i8,
              v.as_array()[10] as i8,
              v.as_array()[11] as i8,
              v.as_array()[12] as i8,
              v.as_array()[13] as i8,
              v.as_array()[14] as i8,
              v.as_array()[15] as i8,
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn from_slice_unaligned_backend(input: &[i8]) -> Self {
        assert!(input.len() >= 16);

        pick! {
          if #[cfg(target_feature="sse2")] {
            unsafe { Self(load_unaligned_m128i( &*(input.as_ptr() as * const [u8;16]) )) }
          } else if #[cfg(target_feature="simd128")] {
            unsafe { Self(v128_load(input.as_ptr() as *const v128 )) }
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            unsafe { Self(vld1q_s8( input.as_ptr() as *const i8 )) }
          } else {
            // 2018 edition doesn't have try_into
            unsafe { Self::from_array( *(input.as_ptr() as * const [i8;16]) ) }
          }
        }
    }
}
