use crate::backend_imports::*;

#[cfg(not(any(
    target_feature = "sse2",
    target_feature = "simd128",
    all(target_feature = "neon", target_arch = "aarch64")
)))]
#[repr(C, align(16))]
#[derive(Clone, Copy)]
pub(crate) struct Inner {
    pub arr: [u8; 16],
}

safe_target_feature! {
impl SimdBackend for u8x16 {
    pick! {
      if #[cfg(target_feature="sse2")] {
        type Inner = __m128i;
      } else if #[cfg(target_feature="simd128")] {
        type Inner = v128;
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        type Inner = uint8x16_t;
      } else {
        type Inner = Inner;
      }
    }

    #[inline]
    pub(crate) fn simd_eq_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse2")] {
            Self(_mm_cmpeq_epi8(self.0, rhs.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(u8x16_eq(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vceqq_u8(self.0, rhs.0))
          } else {
            Self::from_array([
              if self.0.arr[0] == rhs.0.arr[0] { u8::MAX } else { 0 },
              if self.0.arr[1] == rhs.0.arr[1] { u8::MAX } else { 0 },
              if self.0.arr[2] == rhs.0.arr[2] { u8::MAX } else { 0 },
              if self.0.arr[3] == rhs.0.arr[3] { u8::MAX } else { 0 },
              if self.0.arr[4] == rhs.0.arr[4] { u8::MAX } else { 0 },
              if self.0.arr[5] == rhs.0.arr[5] { u8::MAX } else { 0 },
              if self.0.arr[6] == rhs.0.arr[6] { u8::MAX } else { 0 },
              if self.0.arr[7] == rhs.0.arr[7] { u8::MAX } else { 0 },
              if self.0.arr[8] == rhs.0.arr[8] { u8::MAX } else { 0 },
              if self.0.arr[9] == rhs.0.arr[9] { u8::MAX } else { 0 },
              if self.0.arr[10] == rhs.0.arr[10] { u8::MAX } else { 0 },
              if self.0.arr[11] == rhs.0.arr[11] { u8::MAX } else { 0 },
              if self.0.arr[12] == rhs.0.arr[12] { u8::MAX } else { 0 },
              if self.0.arr[13] == rhs.0.arr[13] { u8::MAX } else { 0 },
              if self.0.arr[14] == rhs.0.arr[14] { u8::MAX } else { 0 },
              if self.0.arr[15] == rhs.0.arr[15] { u8::MAX } else { 0 },
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn simd_ne_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse2")] {
            !self.simd_eq(rhs)
          } else if #[cfg(target_feature="simd128")] {
            Self(u8x16_ne(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            !self.simd_eq(rhs)
          } else {
            Self::from_array([
              if self.0.arr[0] != rhs.0.arr[0] { u8::MAX } else { 0 },
              if self.0.arr[1] != rhs.0.arr[1] { u8::MAX } else { 0 },
              if self.0.arr[2] != rhs.0.arr[2] { u8::MAX } else { 0 },
              if self.0.arr[3] != rhs.0.arr[3] { u8::MAX } else { 0 },
              if self.0.arr[4] != rhs.0.arr[4] { u8::MAX } else { 0 },
              if self.0.arr[5] != rhs.0.arr[5] { u8::MAX } else { 0 },
              if self.0.arr[6] != rhs.0.arr[6] { u8::MAX } else { 0 },
              if self.0.arr[7] != rhs.0.arr[7] { u8::MAX } else { 0 },
              if self.0.arr[8] != rhs.0.arr[8] { u8::MAX } else { 0 },
              if self.0.arr[9] != rhs.0.arr[9] { u8::MAX } else { 0 },
              if self.0.arr[10] != rhs.0.arr[10] { u8::MAX } else { 0 },
              if self.0.arr[11] != rhs.0.arr[11] { u8::MAX } else { 0 },
              if self.0.arr[12] != rhs.0.arr[12] { u8::MAX } else { 0 },
              if self.0.arr[13] != rhs.0.arr[13] { u8::MAX } else { 0 },
              if self.0.arr[14] != rhs.0.arr[14] { u8::MAX } else { 0 },
              if self.0.arr[15] != rhs.0.arr[15] { u8::MAX } else { 0 },
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn simd_lt_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse2")] {
            // Convert from u8 to i8.
            let offset = Self::splat(0x80);
            let self_i8 = (self ^ offset).0;
            let rhs_i8 = (rhs ^ offset).0;
            Self(_mm_cmplt_epi8(self_i8, rhs_i8))
          } else if #[cfg(target_feature="simd128")] {
            Self(u8x16_lt(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vcltq_u8(self.0, rhs.0))
          } else {
            Self::from_array([
              if self.0.arr[0] < rhs.0.arr[0] { u8::MAX } else { 0 },
              if self.0.arr[1] < rhs.0.arr[1] { u8::MAX } else { 0 },
              if self.0.arr[2] < rhs.0.arr[2] { u8::MAX } else { 0 },
              if self.0.arr[3] < rhs.0.arr[3] { u8::MAX } else { 0 },
              if self.0.arr[4] < rhs.0.arr[4] { u8::MAX } else { 0 },
              if self.0.arr[5] < rhs.0.arr[5] { u8::MAX } else { 0 },
              if self.0.arr[6] < rhs.0.arr[6] { u8::MAX } else { 0 },
              if self.0.arr[7] < rhs.0.arr[7] { u8::MAX } else { 0 },
              if self.0.arr[8] < rhs.0.arr[8] { u8::MAX } else { 0 },
              if self.0.arr[9] < rhs.0.arr[9] { u8::MAX } else { 0 },
              if self.0.arr[10] < rhs.0.arr[10] { u8::MAX } else { 0 },
              if self.0.arr[11] < rhs.0.arr[11] { u8::MAX } else { 0 },
              if self.0.arr[12] < rhs.0.arr[12] { u8::MAX } else { 0 },
              if self.0.arr[13] < rhs.0.arr[13] { u8::MAX } else { 0 },
              if self.0.arr[14] < rhs.0.arr[14] { u8::MAX } else { 0 },
              if self.0.arr[15] < rhs.0.arr[15] { u8::MAX } else { 0 },
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn simd_gt_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse2")] {
            // Convert from u8 to i8.
            let offset = Self::splat(0x80);
            let self_i8 = (self ^ offset).0;
            let rhs_i8 = (rhs ^ offset).0;
            Self(_mm_cmpgt_epi8(self_i8, rhs_i8))
          } else if #[cfg(target_feature="simd128")] {
            Self(u8x16_gt(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vcgtq_u8(self.0, rhs.0))
          } else {
            Self::from_array([
              if self.0.arr[0] > rhs.0.arr[0] { u8::MAX } else { 0 },
              if self.0.arr[1] > rhs.0.arr[1] { u8::MAX } else { 0 },
              if self.0.arr[2] > rhs.0.arr[2] { u8::MAX } else { 0 },
              if self.0.arr[3] > rhs.0.arr[3] { u8::MAX } else { 0 },
              if self.0.arr[4] > rhs.0.arr[4] { u8::MAX } else { 0 },
              if self.0.arr[5] > rhs.0.arr[5] { u8::MAX } else { 0 },
              if self.0.arr[6] > rhs.0.arr[6] { u8::MAX } else { 0 },
              if self.0.arr[7] > rhs.0.arr[7] { u8::MAX } else { 0 },
              if self.0.arr[8] > rhs.0.arr[8] { u8::MAX } else { 0 },
              if self.0.arr[9] > rhs.0.arr[9] { u8::MAX } else { 0 },
              if self.0.arr[10] > rhs.0.arr[10] { u8::MAX } else { 0 },
              if self.0.arr[11] > rhs.0.arr[11] { u8::MAX } else { 0 },
              if self.0.arr[12] > rhs.0.arr[12] { u8::MAX } else { 0 },
              if self.0.arr[13] > rhs.0.arr[13] { u8::MAX } else { 0 },
              if self.0.arr[14] > rhs.0.arr[14] { u8::MAX } else { 0 },
              if self.0.arr[15] > rhs.0.arr[15] { u8::MAX } else { 0 },
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn simd_le_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse2")] {
            // Convert from u8 to i8.
            let offset = Self::splat(0x80);
            let self_i8 = (self ^ offset).0;
            let rhs_i8 = (rhs ^ offset).0;
            // a <= b  is equivalent to  !(b < a)  or  !(a > b)
            let gt_mask = u8x16(_mm_cmpgt_epi8(self_i8, rhs_i8));
            gt_mask ^ u8x16::splat(0xFF)
          } else if #[cfg(target_feature="simd128")] {
            Self(u8x16_le(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vcleq_u8(self.0, rhs.0))
          } else {
            Self::from_array([
              if self.0.arr[0] <= rhs.0.arr[0] { u8::MAX } else { 0 },
              if self.0.arr[1] <= rhs.0.arr[1] { u8::MAX } else { 0 },
              if self.0.arr[2] <= rhs.0.arr[2] { u8::MAX } else { 0 },
              if self.0.arr[3] <= rhs.0.arr[3] { u8::MAX } else { 0 },
              if self.0.arr[4] <= rhs.0.arr[4] { u8::MAX } else { 0 },
              if self.0.arr[5] <= rhs.0.arr[5] { u8::MAX } else { 0 },
              if self.0.arr[6] <= rhs.0.arr[6] { u8::MAX } else { 0 },
              if self.0.arr[7] <= rhs.0.arr[7] { u8::MAX } else { 0 },
              if self.0.arr[8] <= rhs.0.arr[8] { u8::MAX } else { 0 },
              if self.0.arr[9] <= rhs.0.arr[9] { u8::MAX } else { 0 },
              if self.0.arr[10] <= rhs.0.arr[10] { u8::MAX } else { 0 },
              if self.0.arr[11] <= rhs.0.arr[11] { u8::MAX } else { 0 },
              if self.0.arr[12] <= rhs.0.arr[12] { u8::MAX } else { 0 },
              if self.0.arr[13] <= rhs.0.arr[13] { u8::MAX } else { 0 },
              if self.0.arr[14] <= rhs.0.arr[14] { u8::MAX } else { 0 },
              if self.0.arr[15] <= rhs.0.arr[15] { u8::MAX } else { 0 },
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn simd_ge_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse2")] {
            // Convert from u8 to i8.
            let offset = Self::splat(0x80);
            let self_i8 = (self ^ offset).0;
            let rhs_i8 = (rhs ^ offset).0;
            // a >= b  is equivalent to  !(b > a)  or  !(a < b)
            let lt_mask = u8x16(_mm_cmplt_epi8(self_i8, rhs_i8));
            lt_mask ^ u8x16::splat(0xFF)
          } else if #[cfg(target_feature="simd128")] {
            Self(u8x16_ge(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vcgeq_u8(self.0, rhs.0))
          } else {
            Self::from_array([
              if self.0.arr[0] >= rhs.0.arr[0] { u8::MAX } else { 0 },
              if self.0.arr[1] >= rhs.0.arr[1] { u8::MAX } else { 0 },
              if self.0.arr[2] >= rhs.0.arr[2] { u8::MAX } else { 0 },
              if self.0.arr[3] >= rhs.0.arr[3] { u8::MAX } else { 0 },
              if self.0.arr[4] >= rhs.0.arr[4] { u8::MAX } else { 0 },
              if self.0.arr[5] >= rhs.0.arr[5] { u8::MAX } else { 0 },
              if self.0.arr[6] >= rhs.0.arr[6] { u8::MAX } else { 0 },
              if self.0.arr[7] >= rhs.0.arr[7] { u8::MAX } else { 0 },
              if self.0.arr[8] >= rhs.0.arr[8] { u8::MAX } else { 0 },
              if self.0.arr[9] >= rhs.0.arr[9] { u8::MAX } else { 0 },
              if self.0.arr[10] >= rhs.0.arr[10] { u8::MAX } else { 0 },
              if self.0.arr[11] >= rhs.0.arr[11] { u8::MAX } else { 0 },
              if self.0.arr[12] >= rhs.0.arr[12] { u8::MAX } else { 0 },
              if self.0.arr[13] >= rhs.0.arr[13] { u8::MAX } else { 0 },
              if self.0.arr[14] >= rhs.0.arr[14] { u8::MAX } else { 0 },
              if self.0.arr[15] >= rhs.0.arr[15] { u8::MAX } else { 0 },
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn select_backend(self, if_true: Self, if_false: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse4.1")] {
            Self(_mm_blendv_epi8(if_false.0, if_true.0, self.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(v128_bitselect(if_true.0, if_false.0, self.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vbslq_u8(self.0, if_true.0, if_false.0))
          } else {
            generic_bit_blend(self, if_true, if_false)
          }
        }
    }

    #[inline]
    pub(crate) fn to_bitmask_backend(self) -> u32 {
        pick! {
          if #[cfg(target_feature="sse2")] {
            _mm_movemask_epi8(self.0) as u32
          } else if #[cfg(target_feature="simd128")] {
            u8x16_bitmask(self.0) as u32
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            // set all to 1 if top bit is set, else 0
            let masked = vcltzq_s8(self.cast_signed().0);

            // select the right bit out of each lane
            let selectbit = Self::from_array([
              1u8, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128,
            ]).0;
            let out = vandq_u8(masked, selectbit);

            // interleave the lanes so that a 16-bit sum accumulates the bits in the right order
            let table = Self::from_array([
              0u8, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15,
            ]).0;
            let r = vqtbl1q_u8(out, table);

            // horizontally add the 16-bit lanes
            vaddvq_u16(vreinterpretq_u16_u8(r)) as u32
           } else {
            ((self.0.arr[0].cast_signed() < 0) as u32) |
            ((self.0.arr[1].cast_signed() < 0) as u32) << 1 |
            ((self.0.arr[2].cast_signed() < 0) as u32) << 2 |
            ((self.0.arr[3].cast_signed() < 0) as u32) << 3 |
            ((self.0.arr[4].cast_signed() < 0) as u32) << 4 |
            ((self.0.arr[5].cast_signed() < 0) as u32) << 5 |
            ((self.0.arr[6].cast_signed() < 0) as u32) << 6 |
            ((self.0.arr[7].cast_signed() < 0) as u32) << 7 |
            ((self.0.arr[8].cast_signed() < 0) as u32) << 8 |
            ((self.0.arr[9].cast_signed() < 0) as u32) << 9 |
            ((self.0.arr[10].cast_signed() < 0) as u32) << 10 |
            ((self.0.arr[11].cast_signed() < 0) as u32) << 11 |
            ((self.0.arr[12].cast_signed() < 0) as u32) << 12 |
            ((self.0.arr[13].cast_signed() < 0) as u32) << 13 |
            ((self.0.arr[14].cast_signed() < 0) as u32) << 14 |
            ((self.0.arr[15].cast_signed() < 0) as u32) << 15
          }
        }
    }

    #[inline]
    pub(crate) fn any_backend(self) -> bool {
        pick! {
          if #[cfg(target_feature="sse2")] {
            _mm_movemask_epi8(self.0) != 0
          } else if #[cfg(target_feature="simd128")] {
            u8x16_bitmask(self.0) != 0
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))] {
            vminvq_s8(self.cast_signed().0) < 0
          } else {
            let v : [u64;2] = cast(self);
            ((v[0] | v[1]) & 0x8080808080808080) != 0
          }
        }
    }

    #[inline]
    pub(crate) fn all_backend(self) -> bool {
        pick! {
          if #[cfg(target_feature="sse2")] {
            _mm_movemask_epi8(self.0) == 0b1111_1111_1111_1111
          } else if #[cfg(target_feature="simd128")] {
            u8x16_bitmask(self.0) == 0b1111_1111_1111_1111
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))] {
            vmaxvq_s8(self.cast_signed().0) < 0
          } else {
            let v : [u64;2] = cast(self);
            (v[0] & v[1] & 0x8080808080808080) == 0x8080808080808080
          }
        }
    }

    #[inline]
    pub(crate) fn interleave_low_backend(self, other: Self) -> Self {
        pick! {
            if #[cfg(target_feature = "sse2")] {
                u8x16(_mm_unpacklo_epi8(self.0, other.0))
            } else if #[cfg(target_feature = "simd128")] {
              u8x16(u8x16_shuffle::<0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23>(self.0, other.0))
            } else if #[cfg(all(target_feature = "neon", target_arch = "aarch64"))] {
                let lhs = vget_low_u8(self.0);
                let rhs = vget_low_u8(other.0);

                let zipped = vzip_u8(lhs, rhs);
                u8x16(vcombine_u8(zipped.0, zipped.1))
            } else {
                u8x16::from_array([
                    self.as_array()[0], other.as_array()[0],
                    self.as_array()[1], other.as_array()[1],
                    self.as_array()[2], other.as_array()[2],
                    self.as_array()[3], other.as_array()[3],
                    self.as_array()[4], other.as_array()[4],
                    self.as_array()[5], other.as_array()[5],
                    self.as_array()[6], other.as_array()[6],
                    self.as_array()[7], other.as_array()[7],
                ])
            }
        }
    }

    #[inline]
    pub(crate) fn interleave_high_backend(self, other: Self) -> Self {
        pick! {
            if #[cfg(target_feature = "sse2")] {
                u8x16(_mm_unpackhi_epi8(self.0, other.0))
            } else if #[cfg(target_feature = "simd128")] {
                u8x16(u8x16_shuffle::<8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31>(self.0, other.0))
            } else if #[cfg(all(target_feature = "neon", target_arch = "aarch64"))] {
                let lhs = vget_high_u8(self.0);
                let rhs = vget_high_u8(other.0);

                let zipped = vzip_u8(lhs, rhs);
                u8x16(vcombine_u8(zipped.0, zipped.1))
            } else {
                u8x16::from_array([
                    self.as_array()[8], other.as_array()[8],
                    self.as_array()[9], other.as_array()[9],
                    self.as_array()[10], other.as_array()[10],
                    self.as_array()[11], other.as_array()[11],
                    self.as_array()[12], other.as_array()[12],
                    self.as_array()[13], other.as_array()[13],
                    self.as_array()[14], other.as_array()[14],
                    self.as_array()[15], other.as_array()[15],
                ])
            }
        }
    }

    #[inline]
    pub(crate) fn shuffle_backend(self, indices: u8x16) -> Self {
        pick! {
          if #[cfg(target_feature="ssse3")] {
            Self(_mm_shuffle_epi8(self.0, indices.0))
          } else if #[cfg(target_feature="relaxed-simd")] {
            Self(u8x16_relaxed_swizzle(self.0, indices.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(u8x16_swizzle(self.0, indices.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))] {
            Self(vqtbl1q_u8(self.0, indices.0))
          } else {
            let self_array = self.to_array();
            let indices_array = indices.to_array();

            let mut result = [0; 16];
            for i in 0..16 {
              let index = indices_array[i] as usize;
              if index < 16 {
                result[i] = self_array[index];
              }
            }

            Self::from_array(result)
          }
        }
    }

    #[inline]
    pub(crate) fn shuffle_zeroing_backend(self, indices: u8x16) -> Self {
        pick! {
          if #[cfg(target_feature="ssse3")] {
            Self(_mm_shuffle_epi8(
              self.0,
              _mm_adds_epu8(indices.0, _mm_set1_epi8(0x70)),
            ))
          } else if #[cfg(target_feature="simd128")] {
            Self(i8x16_swizzle(self.0, indices.0))
          } else {
            // The remaining branches of `shuffle` already have the behavior we want
            self.shuffle(indices)
          }
        }
    }

    #[inline]
    pub(crate) fn shuffle_wrapping_backend(self, indices: u8x16) -> Self {
        pick! {
          if #[cfg(all(target_feature = "avx512vbmi", target_feature = "avx512vl"))] {
            Self(_mm_permutexvar_epi8(indices.0, self.0))
          } else {
            self.shuffle(indices & 15)
          }
        }
    }

    #[inline]
    pub(crate) fn shuffle_2_backend(inputs: [Self; 2], indices: u8x16) -> u8x16 {
        pick! {
          if #[cfg(all(target_feature = "avx512vbmi", target_feature = "avx512vl"))] {
            u8x16(_mm_permutex2var_epi8(inputs[0].0, indices.0, inputs[1].0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))] {
            let table = uint8x16x2_t(inputs[0].0, inputs[1].0);
            u8x16(vqtbl2q_u8(table, indices.0))
          } else {
            inputs[0].shuffle_zeroing(indices) | inputs[1].shuffle_zeroing(indices - 16)
          }
        }
    }

    #[inline]
    pub(crate) fn shuffle_zeroing_2_backend(inputs: [Self; 2], indices: u8x16) -> u8x16 {
        pick! {
          if #[cfg(all(target_feature = "avx512vbmi", target_feature = "avx512vl"))] {
            inputs.shuffle(indices) & indices.simd_lt(32)
          } else {
            inputs.shuffle(indices)
          }
        }
    }

    #[inline]
    pub(crate) fn shuffle_wrapping_2_backend(inputs: [Self; 2], indices: u8x16) -> u8x16 {
        pick! {
          if #[cfg(all(target_feature = "avx512vbmi", target_feature = "avx512vl"))] {
            // `avx512` shuffle intrinsics are wrapping
            inputs.shuffle(indices)
          } else {
            inputs.shuffle(indices & 31)
          }
        }
    }

    #[inline]
    pub(crate) fn shuffle_3_backend(inputs: [Self; 3], indices: u8x16) -> u8x16 {
        pick! {
          if #[cfg(all(target_feature="neon",target_arch="aarch64"))] {
            let table = uint8x16x3_t(inputs[0].0, inputs[1].0, inputs[2].0);
            u8x16(vqtbl3q_u8(table, indices.0))
          } else {
            [inputs[0], inputs[1]].shuffle_zeroing(indices) | inputs[2].shuffle_zeroing(indices - 32)
          }
        }
    }

    #[inline]
    pub(crate) fn shuffle_zeroing_3_backend(inputs: [Self; 3], indices: u8x16) -> u8x16 {
        inputs.shuffle(indices)
    }

    #[inline]
    pub(crate) fn shuffle_wrapping_3_backend(inputs: [Self; 3], indices: u8x16) -> u8x16 {
        inputs.shuffle(indices % 48)
    }

    #[inline]
    pub(crate) fn shuffle_4_backend(inputs: [Self; 4], indices: u8x16) -> u8x16 {
        pick! {
          if #[cfg(all(target_feature="neon",target_arch="aarch64"))] {
            let table = uint8x16x4_t(inputs[0].0, inputs[1].0, inputs[2].0, inputs[3].0);
            u8x16(vqtbl4q_u8(table, indices.0))
          } else {
            [inputs[0], inputs[1]].shuffle_zeroing(indices)
              | [inputs[2], inputs[3]].shuffle_zeroing(indices - 32)
          }
        }
    }

    #[inline]
    pub(crate) fn shuffle_zeroing_4_backend(inputs: [Self; 4], indices: u8x16) -> u8x16 {
        inputs.shuffle(indices)
    }

    #[inline]
    pub(crate) fn shuffle_wrapping_4_backend(inputs: [Self; 4], indices: u8x16) -> u8x16 {
        inputs.shuffle(indices & 63)
    }

    #[inline]
    pub(crate) fn transpose_backend(data: [Self; 16]) -> [Self; 16] {
        // Can this be optimized?

        #[inline(always)]
        fn transpose_column(data: &[u8x16; 16], index: usize) -> u8x16 {
            u8x16::from_array([
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

    #[inline]
    pub(crate) fn replace_const_backend<const INDEX: usize>(self, value: u8) -> Self {
        /// Because of limitations in const generics, this must have separate
        /// branches per index. This macro is used to avoid duplication.
        macro_rules! use_indices {
      (0..16 = [$($INDEX:literal),*]) => {
        pick! {
          if #[cfg(target_feature = "sse4.1")] {
            match INDEX {
              $($INDEX => Self(_mm_insert_epi8::<$INDEX>(self.0, value as i32)),)*
              16.. => unreachable!()
            }
          } else if #[cfg(target_feature = "sse2")] {
            let replace_mask = const {
              let mut replace_mask = Self::ZERO;
              if INDEX < 16 {
                replace_mask.as_mut_array()[INDEX] = u8::MAX;
              }
              replace_mask
            };

            match INDEX {
              $($INDEX => {
                let value = _mm_cvtsi32_si128(value as i32);
                let shifted_value = Self(_mm_slli_si128::<$INDEX>(value));

                replace_mask.select(shifted_value, self)
              })*
              16.. => unreachable!(),
            }
          } else if #[cfg(all(
            target_arch = "aarch64",
            target_feature = "neon",
            target_endian = "little",
          ))] {
            match INDEX {
              $($INDEX => Self(vsetq_lane_u8::<$INDEX>(value, self.0)),)*
              16.. => unreachable!(),
            }
          } else if #[cfg(target_feature = "simd128")] {
            Self(u8x16_replace_lane::<INDEX>(self.0, value))
          } else {
            let mut result = self;
            result.as_mut_array()[INDEX] = value;
            result
          }
        }
      };
    }
        use_indices! { 0..16 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] }
    }

    #[inline]
    pub(crate) fn extract_const_backend<const INDEX: usize>(self) -> u8 {
        /// Because of limitations in const generics, this must have separate
        /// branches per index. This macro is used to avoid duplication.
        macro_rules! use_indices {
      (
        0..16 = [$($INDEX:literal),*],
        1..16 = [$($INDEX_FROM_1:literal),*],
        4..16 = [$($INDEX_FROM_4:literal),*]
      ) => {
        pick! {
          if #[cfg(target_feature = "sse4.1")] {
            match INDEX {
              $($INDEX => _mm_extract_epi8::<$INDEX>(self.0).cast_unsigned() as u8,)*
              16.. => unreachable!()
            }
          } else if #[cfg(all(target_feature = "sse2"))] {
            // Shift bytes so that the element at `INDEX` becomes the lowest one
            let shifted_self = match INDEX {
              0 => self.0,
              $($INDEX_FROM_1 => _mm_srli_si128::<$INDEX_FROM_1>(self.0),)*
              16.. => unreachable!(),
            };

            _mm_cvtsi128_si32(shifted_self).cast_unsigned() as u8
          } else if #[cfg(all(
            target_arch = "aarch64",
            target_feature = "neon",
            target_endian = "little",
          ))] {
            match INDEX {
              $($INDEX => vgetq_lane_u8::<$INDEX>(self.0),)*
              16.. => unreachable!(),
            }
          } else if #[cfg(target_feature = "simd128")] {
            u8x16_extract_lane::<INDEX>(self.0)
          } else {
            self.to_array()[INDEX]
          }
        }
      };
    }
        use_indices! {
          0..16 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
          1..16 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
          4..16 = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        }
    }
}
}

safe_target_feature! {
// Backend of SIMD integer functions.
impl u8x16 {
    #[inline]
    pub(crate) fn neg_backend(self) -> Self {
        pick! {
            if #[cfg(all(target_arch = "aarch64", target_feature = "neon"))] {
                Self(vreinterpretq_u8_s8(vnegq_s8(vreinterpretq_s8_u8(self.0))))
            } else {
                Self::default() - self
            }
        }
    }

    #[inline]
    pub(crate) fn not_backend(self) -> Self {
        self ^ cast::<u128, u8x16>(u128::MAX)
    }

    #[inline]
    pub(crate) fn add_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse2")] {
            Self(_mm_add_epi8(self.0, rhs.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(u8x16_add(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vaddq_u8(self.0, rhs.0))
          } else {
            Self::from_array([
              self.0.arr[0].wrapping_add(rhs.0.arr[0]),
              self.0.arr[1].wrapping_add(rhs.0.arr[1]),
              self.0.arr[2].wrapping_add(rhs.0.arr[2]),
              self.0.arr[3].wrapping_add(rhs.0.arr[3]),
              self.0.arr[4].wrapping_add(rhs.0.arr[4]),
              self.0.arr[5].wrapping_add(rhs.0.arr[5]),
              self.0.arr[6].wrapping_add(rhs.0.arr[6]),
              self.0.arr[7].wrapping_add(rhs.0.arr[7]),
              self.0.arr[8].wrapping_add(rhs.0.arr[8]),
              self.0.arr[9].wrapping_add(rhs.0.arr[9]),
              self.0.arr[10].wrapping_add(rhs.0.arr[10]),
              self.0.arr[11].wrapping_add(rhs.0.arr[11]),
              self.0.arr[12].wrapping_add(rhs.0.arr[12]),
              self.0.arr[13].wrapping_add(rhs.0.arr[13]),
              self.0.arr[14].wrapping_add(rhs.0.arr[14]),
              self.0.arr[15].wrapping_add(rhs.0.arr[15]),
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn sub_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse2")] {
            Self(_mm_sub_epi8(self.0, rhs.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(u8x16_sub(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vsubq_u8(self.0, rhs.0))
          } else {
            Self::from_array([
              self.0.arr[0].wrapping_sub(rhs.0.arr[0]),
              self.0.arr[1].wrapping_sub(rhs.0.arr[1]),
              self.0.arr[2].wrapping_sub(rhs.0.arr[2]),
              self.0.arr[3].wrapping_sub(rhs.0.arr[3]),
              self.0.arr[4].wrapping_sub(rhs.0.arr[4]),
              self.0.arr[5].wrapping_sub(rhs.0.arr[5]),
              self.0.arr[6].wrapping_sub(rhs.0.arr[6]),
              self.0.arr[7].wrapping_sub(rhs.0.arr[7]),
              self.0.arr[8].wrapping_sub(rhs.0.arr[8]),
              self.0.arr[9].wrapping_sub(rhs.0.arr[9]),
              self.0.arr[10].wrapping_sub(rhs.0.arr[10]),
              self.0.arr[11].wrapping_sub(rhs.0.arr[11]),
              self.0.arr[12].wrapping_sub(rhs.0.arr[12]),
              self.0.arr[13].wrapping_sub(rhs.0.arr[13]),
              self.0.arr[14].wrapping_sub(rhs.0.arr[14]),
              self.0.arr[15].wrapping_sub(rhs.0.arr[15]),
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn mul_backend(self, rhs: Self) -> Self {
        // For x86 and wasm, this technically can be done explicitly by converting
        // to `i16` then converting back after multiplication, but that may not
        // actually be faster than auto-vectorization.
        pick! {
          if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vmulq_u8(self.0, rhs.0))
          } else {
            let self_array: [u8; 16] = cast(self);
            let rhs_array: [u8; 16] = cast(rhs);

            Self::from_array([
              self_array[0].wrapping_mul(rhs_array[0]),
              self_array[1].wrapping_mul(rhs_array[1]),
              self_array[2].wrapping_mul(rhs_array[2]),
              self_array[3].wrapping_mul(rhs_array[3]),
              self_array[4].wrapping_mul(rhs_array[4]),
              self_array[5].wrapping_mul(rhs_array[5]),
              self_array[6].wrapping_mul(rhs_array[6]),
              self_array[7].wrapping_mul(rhs_array[7]),
              self_array[8].wrapping_mul(rhs_array[8]),
              self_array[9].wrapping_mul(rhs_array[9]),
              self_array[10].wrapping_mul(rhs_array[10]),
              self_array[11].wrapping_mul(rhs_array[11]),
              self_array[12].wrapping_mul(rhs_array[12]),
              self_array[13].wrapping_mul(rhs_array[13]),
              self_array[14].wrapping_mul(rhs_array[14]),
              self_array[15].wrapping_mul(rhs_array[15]),
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn bitand_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse2")] {
            Self(_mm_and_si128(self.0, rhs.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(v128_and(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vandq_u8(self.0, rhs.0))
          } else {
            Self::from_array([
              self.0.arr[0] & rhs.0.arr[0],
              self.0.arr[1] & rhs.0.arr[1],
              self.0.arr[2] & rhs.0.arr[2],
              self.0.arr[3] & rhs.0.arr[3],
              self.0.arr[4] & rhs.0.arr[4],
              self.0.arr[5] & rhs.0.arr[5],
              self.0.arr[6] & rhs.0.arr[6],
              self.0.arr[7] & rhs.0.arr[7],
              self.0.arr[8] & rhs.0.arr[8],
              self.0.arr[9] & rhs.0.arr[9],
              self.0.arr[10] & rhs.0.arr[10],
              self.0.arr[11] & rhs.0.arr[11],
              self.0.arr[12] & rhs.0.arr[12],
              self.0.arr[13] & rhs.0.arr[13],
              self.0.arr[14] & rhs.0.arr[14],
              self.0.arr[15] & rhs.0.arr[15],
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn bitor_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse2")] {
            Self(_mm_or_si128(self.0, rhs.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(v128_or(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vorrq_u8(self.0, rhs.0))
          } else {
            Self::from_array([
              self.0.arr[0] | rhs.0.arr[0],
              self.0.arr[1] | rhs.0.arr[1],
              self.0.arr[2] | rhs.0.arr[2],
              self.0.arr[3] | rhs.0.arr[3],
              self.0.arr[4] | rhs.0.arr[4],
              self.0.arr[5] | rhs.0.arr[5],
              self.0.arr[6] | rhs.0.arr[6],
              self.0.arr[7] | rhs.0.arr[7],
              self.0.arr[8] | rhs.0.arr[8],
              self.0.arr[9] | rhs.0.arr[9],
              self.0.arr[10] | rhs.0.arr[10],
              self.0.arr[11] | rhs.0.arr[11],
              self.0.arr[12] | rhs.0.arr[12],
              self.0.arr[13] | rhs.0.arr[13],
              self.0.arr[14] | rhs.0.arr[14],
              self.0.arr[15] | rhs.0.arr[15],
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn bitxor_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse2")] {
            Self(_mm_xor_si128(self.0, rhs.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(v128_xor(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(veorq_u8(self.0, rhs.0))
          } else {
            Self::from_array([
              self.0.arr[0] ^ rhs.0.arr[0],
              self.0.arr[1] ^ rhs.0.arr[1],
              self.0.arr[2] ^ rhs.0.arr[2],
              self.0.arr[3] ^ rhs.0.arr[3],
              self.0.arr[4] ^ rhs.0.arr[4],
              self.0.arr[5] ^ rhs.0.arr[5],
              self.0.arr[6] ^ rhs.0.arr[6],
              self.0.arr[7] ^ rhs.0.arr[7],
              self.0.arr[8] ^ rhs.0.arr[8],
              self.0.arr[9] ^ rhs.0.arr[9],
              self.0.arr[10] ^ rhs.0.arr[10],
              self.0.arr[11] ^ rhs.0.arr[11],
              self.0.arr[12] ^ rhs.0.arr[12],
              self.0.arr[13] ^ rhs.0.arr[13],
              self.0.arr[14] ^ rhs.0.arr[14],
              self.0.arr[15] ^ rhs.0.arr[15],
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn reduce_add_backend(self) -> u8 {
        #[allow(dead_code)]
        const SHUFFLE_1: u8x16
          = u8x16::from_array([8, 9, 10, 11, 12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0]);
        #[allow(dead_code)]
        const SHUFFLE_2: u8x16
          = u8x16::from_array([4, 5, 6, 7, 0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0]);
        #[allow(dead_code)]
        const SHUFFLE_3: u8x16
          = u8x16::from_array([2, 3, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        #[allow(dead_code)]
        const SHUFFLE_4: u8x16
          = u8x16::from_array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);

        pick! {
          if #[cfg(target_feature="ssse3")] {
            let rhs = _mm_shuffle_epi8(self.0, SHUFFLE_1.0);
            let sum = _mm_add_epi8(self.0, rhs);
            let rhs = _mm_shuffle_epi8(sum, SHUFFLE_2.0);
            let sum = _mm_add_epi8(sum, rhs);
            let rhs = _mm_shuffle_epi8(sum, SHUFFLE_3.0);
            let sum = _mm_add_epi8(sum, rhs);
            let rhs = _mm_shuffle_epi8(sum, SHUFFLE_4.0);
            let sum = _mm_add_epi8(sum, rhs);
            _mm_cvtsi128_si32(sum) as u8
          } else if #[cfg(target_feature="simd128")] {
            let rhs = u8x16_shuffle::<8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7>(self.0, self.0);
            let sum = u8x16_add(self.0, rhs);
            let rhs = u8x16_shuffle::<4, 5, 6, 7, 0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0>(sum, sum);
            let sum = u8x16_add(sum, rhs);
            let rhs = u8x16_shuffle::<2, 3, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>(sum, sum);
            let sum = u8x16_add(sum, rhs);
            let rhs = u8x16_shuffle::<1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>(sum, sum);
            let sum = u8x16_add(sum, rhs);
            u8x16_extract_lane::<0>(sum)
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            let rhs = vqtbl1q_u8(self.0, SHUFFLE_1.0);
            let sum = vaddq_u8(self.0, rhs);
            let rhs = vqtbl1q_u8(sum, SHUFFLE_2.0);
            let sum = vaddq_u8(sum, rhs);
            let rhs = vqtbl1q_u8(sum, SHUFFLE_3.0);
            let sum = vaddq_u8(sum, rhs);
            let rhs = vqtbl1q_u8(sum, SHUFFLE_4.0);
            let sum = vaddq_u8(sum, rhs);
            vgetq_lane_u8(sum, 0)
          } else {
            let array: [u8; 16] = cast(self);
            array.into_iter().reduce(u8::wrapping_add).unwrap()
          }
        }
    }

    #[inline]
    pub(crate) fn reduce_mul_backend(self) -> u8 {
        pick! {
          if #[cfg(all(target_feature="neon", target_arch="aarch64"))] {
            const HIGH_64: u8x16
              = u8x16::from_array([8, 9, 10, 11, 12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0]);
            const HIGH_32: u8x16
              = u8x16::from_array([4, 5, 6, 7, 0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0]);
            const HIGH_16: u8x16
              = u8x16::from_array([2, 3, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
            const HIGH_8: u8x16
              = u8x16::from_array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);

            let high_64 = vqtbl1q_u8(self.0, HIGH_64.0);
            let reduce_64 = vmulq_u8(self.0, high_64);
            let high_32 = vqtbl1q_u8(reduce_64, HIGH_32.0);
            let reduce_32 = vmulq_u8(reduce_64, high_32);
            let high_16 = vqtbl1q_u8(reduce_32, HIGH_16.0);
            let reduce_16 = vmulq_u8(reduce_32, high_16);
            let high_8 = vqtbl1q_u8(reduce_16, HIGH_8.0);
            let reduce_8 = vmulq_u8(reduce_16, high_8);
            vgetq_lane_u8::<0>(reduce_8)
          } else {
            self.to_array().into_iter().reduce(u8::wrapping_mul).unwrap()
          }
        }
    }

    #[inline]
    pub(crate) fn bitselect_backend(self, if_one: Self, if_zero: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse2")] {
            Self(_mm_or_si128(_mm_and_si128(if_one.0, self.0), _mm_andnot_si128(self.0, if_zero.0)))
          } else if #[cfg(target_feature="simd128")] {
            Self(v128_bitselect(if_one.0, if_zero.0, self.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vbslq_u8(self.0, if_one.0, if_zero.0))
          } else {
            generic_bit_blend(self, if_one, if_zero)
          }
        }
    }

    #[inline]
    pub(crate) fn shl_backend(self, rhs: Self) -> Self {
        // For x86, this technically can be done explicitly by converting
        // to `u16` or `u32` then converting back after multiplication, but that may
        // not actually be faster than auto-vectorization.
        pick! {
          if #[cfg(all(target_feature="neon", target_arch="aarch64"))] {
            // Mask `rhs` to 7 to match `wrapping_shl`.
            let shift_by = vreinterpretq_s8_u8(vandq_u8(rhs.0, vmovq_n_u8(7)));
            Self(vshlq_u8(self.0, shift_by))
          } else {
            let self_array: [u8; 16] = cast(self);
            let rhs_array: [u8; 16] = cast(rhs);

            Self::from_array([
              self_array[0].wrapping_shl(rhs_array[0] as u32),
              self_array[1].wrapping_shl(rhs_array[1] as u32),
              self_array[2].wrapping_shl(rhs_array[2] as u32),
              self_array[3].wrapping_shl(rhs_array[3] as u32),
              self_array[4].wrapping_shl(rhs_array[4] as u32),
              self_array[5].wrapping_shl(rhs_array[5] as u32),
              self_array[6].wrapping_shl(rhs_array[6] as u32),
              self_array[7].wrapping_shl(rhs_array[7] as u32),
              self_array[8].wrapping_shl(rhs_array[8] as u32),
              self_array[9].wrapping_shl(rhs_array[9] as u32),
              self_array[10].wrapping_shl(rhs_array[10] as u32),
              self_array[11].wrapping_shl(rhs_array[11] as u32),
              self_array[12].wrapping_shl(rhs_array[12] as u32),
              self_array[13].wrapping_shl(rhs_array[13] as u32),
              self_array[14].wrapping_shl(rhs_array[14] as u32),
              self_array[15].wrapping_shl(rhs_array[15] as u32),
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn shl_scalar_backend(self, rhs: u32) -> Self {
        // For x86, this technically can be done explicitly by converting
        // to `u16` or `u32` then converting back after multiplication, but that
        // may not actually be faster than auto-vectorization.
        pick! {
          if #[cfg(target_feature="simd128")] {
            // Mask `rhs` to 7 to match `wrapping_shl`.
            Self(u8x16_shl(self.0, rhs & 7))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            // Mask `rhs` to 7 to match `wrapping_shl`.
            Self(vshlq_u8(self.0, vmovq_n_s8(rhs as i8 & 7)))
          } else {
            let values: u16x8 = cast(self);
            let shift = rhs & 7;
            let shifted = values << shift;
            let crossed = u16x8::splat(!((0x00FFu16 << shift) & 0xFF00));
            cast(shifted & crossed)
          }
        }
    }

    #[inline]
    pub(crate) fn shr_backend(self, rhs: Self) -> Self {
        // For x86, this technically can be done explicitly by converting
        // to `u16` or `u32` then converting back after multiplication, but that may
        // not actually be faster than auto-vectorization.
        pick! {
          if #[cfg(all(target_feature="neon", target_arch="aarch64"))] {
            // Mask `rhs` to 7 to match `wrapping_shr`, and negate it because
            // there is no shift-right intrinsic.
            let neg_rhs = vnegq_s8(vreinterpretq_s8_u8(vandq_u8(rhs.0, vmovq_n_u8(7))));
            Self(vshlq_u8(self.0, neg_rhs))
          } else {
            let self_array: [u8; 16] = cast(self);
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
        // to `u16` or `u32` then converting back after multiplication, but that
        // may not actually be faster than auto-vectorization.
        pick! {
          if #[cfg(target_feature="simd128")] {
            // Mask `rhs` to 7 to match `wrapping_shr`.
            Self(u8x16_shr(self.0, rhs & 7))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            // Mask `rhs` to 7 to match `wrapping_shr`, and negate it because
            // there is no shift-right intrinsic.
            Self(vshlq_u8(self.0, vmovq_n_s8(-(rhs as i8 & 7))))
          } else {
            let values: u16x8 = cast(self);
            let shift = rhs & 7;
            let shifted = values >> shift;
            let crossed = u16x8::splat(!((0xFF00u16 >> shift) & 0x00FF));
            cast(shifted & crossed)
          }
        }
    }

    #[inline]
    pub(crate) fn max_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse2")] {
            Self(_mm_max_epu8(self.0, rhs.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(u8x16_max(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vmaxq_u8(self.0, rhs.0))
          } else {
            Self::from_array([
              self.0.arr[0].max(rhs.0.arr[0]),
              self.0.arr[1].max(rhs.0.arr[1]),
              self.0.arr[2].max(rhs.0.arr[2]),
              self.0.arr[3].max(rhs.0.arr[3]),
              self.0.arr[4].max(rhs.0.arr[4]),
              self.0.arr[5].max(rhs.0.arr[5]),
              self.0.arr[6].max(rhs.0.arr[6]),
              self.0.arr[7].max(rhs.0.arr[7]),
              self.0.arr[8].max(rhs.0.arr[8]),
              self.0.arr[9].max(rhs.0.arr[9]),
              self.0.arr[10].max(rhs.0.arr[10]),
              self.0.arr[11].max(rhs.0.arr[11]),
              self.0.arr[12].max(rhs.0.arr[12]),
              self.0.arr[13].max(rhs.0.arr[13]),
              self.0.arr[14].max(rhs.0.arr[14]),
              self.0.arr[15].max(rhs.0.arr[15]),
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn min_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse2")] {
            Self(_mm_min_epu8(self.0, rhs.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(u8x16_min(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vminq_u8(self.0, rhs.0))
          } else {
            Self::from_array([
              self.0.arr[0].min(rhs.0.arr[0]),
              self.0.arr[1].min(rhs.0.arr[1]),
              self.0.arr[2].min(rhs.0.arr[2]),
              self.0.arr[3].min(rhs.0.arr[3]),
              self.0.arr[4].min(rhs.0.arr[4]),
              self.0.arr[5].min(rhs.0.arr[5]),
              self.0.arr[6].min(rhs.0.arr[6]),
              self.0.arr[7].min(rhs.0.arr[7]),
              self.0.arr[8].min(rhs.0.arr[8]),
              self.0.arr[9].min(rhs.0.arr[9]),
              self.0.arr[10].min(rhs.0.arr[10]),
              self.0.arr[11].min(rhs.0.arr[11]),
              self.0.arr[12].min(rhs.0.arr[12]),
              self.0.arr[13].min(rhs.0.arr[13]),
              self.0.arr[14].min(rhs.0.arr[14]),
              self.0.arr[15].min(rhs.0.arr[15]),
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn reduce_max_backend(self) -> u8 {
        #[allow(dead_code)]
        const SHUFFLE_1: u8x16
          = u8x16::from_array([8, 9, 10, 11, 12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0]);
        #[allow(dead_code)]
        const SHUFFLE_2: u8x16
          = u8x16::from_array([4, 5, 6, 7, 0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0]);
        #[allow(dead_code)]
        const SHUFFLE_3: u8x16
          = u8x16::from_array([2, 3, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        #[allow(dead_code)]
        const SHUFFLE_4: u8x16
          = u8x16::from_array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);

        pick! {
          if #[cfg(target_feature="ssse3")] {
            let rhs = _mm_shuffle_epi8(self.0, SHUFFLE_1.0);
            let max = _mm_max_epu8(self.0, rhs);
            let rhs = _mm_shuffle_epi8(max, SHUFFLE_2.0);
            let max = _mm_max_epu8(max, rhs);
            let rhs = _mm_shuffle_epi8(max, SHUFFLE_3.0);
            let max = _mm_max_epu8(max, rhs);
            let rhs = _mm_shuffle_epi8(max, SHUFFLE_4.0);
            let max = _mm_max_epu8(max, rhs);
            _mm_cvtsi128_si32(max) as u8
          } else if #[cfg(target_feature="simd128")] {
            let rhs = u8x16_shuffle::<8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7>(self.0, self.0);
            let max = u8x16_max(self.0, rhs);
            let rhs = u8x16_shuffle::<4, 5, 6, 7, 0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0>(max, max);
            let max = u8x16_max(max, rhs);
            let rhs = u8x16_shuffle::<2, 3, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>(max, max);
            let max = u8x16_max(max, rhs);
            let rhs = u8x16_shuffle::<1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>(max, max);
            let max = u8x16_max(max, rhs);
            u8x16_extract_lane::<0>(max)
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            let rhs = vqtbl1q_u8(self.0, SHUFFLE_1.0);
            let max = vmaxq_u8(self.0, rhs);
            let rhs = vqtbl1q_u8(max, SHUFFLE_2.0);
            let max = vmaxq_u8(max, rhs);
            let rhs = vqtbl1q_u8(max, SHUFFLE_3.0);
            let max = vmaxq_u8(max, rhs);
            let rhs = vqtbl1q_u8(max, SHUFFLE_4.0);
            let max = vmaxq_u8(max, rhs);
            vgetq_lane_u8(max, 0)
          } else {
            let array: [u8; 16] = cast(self);
            array.into_iter().reduce(u8::max).unwrap()
          }
        }
    }

    #[inline]
    pub(crate) fn reduce_min_backend(self) -> u8 {
        #[allow(dead_code)]
        const SHUFFLE_1: u8x16
          = u8x16::from_array([8, 9, 10, 11, 12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0]);
        #[allow(dead_code)]
        const SHUFFLE_2: u8x16
          = u8x16::from_array([4, 5, 6, 7, 0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0]);
        #[allow(dead_code)]
        const SHUFFLE_3: u8x16
          = u8x16::from_array([2, 3, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        #[allow(dead_code)]
        const SHUFFLE_4: u8x16
          = u8x16::from_array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);

        pick! {
          if #[cfg(target_feature="ssse3")] {
            let rhs = _mm_shuffle_epi8(self.0, SHUFFLE_1.0);
            let min = _mm_min_epu8(self.0, rhs);
            let rhs = _mm_shuffle_epi8(min, SHUFFLE_2.0);
            let min = _mm_min_epu8(min, rhs);
            let rhs = _mm_shuffle_epi8(min, SHUFFLE_3.0);
            let min = _mm_min_epu8(min, rhs);
            let rhs = _mm_shuffle_epi8(min, SHUFFLE_4.0);
            let min = _mm_min_epu8(min, rhs);
            _mm_cvtsi128_si32(min) as u8
          } else if #[cfg(target_feature="simd128")] {
            let rhs = u8x16_shuffle::<8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7>(self.0, self.0);
            let min = u8x16_min(self.0, rhs);
            let rhs = u8x16_shuffle::<4, 5, 6, 7, 0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0>(min, min);
            let min = u8x16_min(min, rhs);
            let rhs = u8x16_shuffle::<2, 3, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>(min, min);
            let min = u8x16_min(min, rhs);
            let rhs = u8x16_shuffle::<1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>(min, min);
            let min = u8x16_min(min, rhs);
            u8x16_extract_lane::<0>(min)
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            let rhs = vqtbl1q_u8(self.0, SHUFFLE_1.0);
            let min = vminq_u8(self.0, rhs);
            let rhs = vqtbl1q_u8(min, SHUFFLE_2.0);
            let min = vminq_u8(min, rhs);
            let rhs = vqtbl1q_u8(min, SHUFFLE_3.0);
            let min = vminq_u8(min, rhs);
            let rhs = vqtbl1q_u8(min, SHUFFLE_4.0);
            let min = vminq_u8(min, rhs);
            vgetq_lane_u8(min, 0)
          } else {
            let array: [u8; 16] = cast(self);
            array.into_iter().reduce(u8::min).unwrap()
          }
        }
    }

    #[inline]
    pub(crate) fn unbounded_shl_backend(self, rhs: Self) -> Self {
        // For x86, this technically can be done explicitly by converting to `u16`
        // or `u32` then converting back after multiplication, but that may not
        // actually be faster than auto-vectorization.
        pick! {
          if #[cfg(all(target_feature="neon", target_arch="aarch64"))] {
            Self(vshlq_u8(self.0, vreinterpretq_s8_u8(rhs.0))) & rhs.simd_lt(8)
          } else {
            let self_array = self.to_array();
            let rhs_array = rhs.to_array();

            Self::from_array([
              self_array[0].unbounded_shl(rhs_array[0] as u32),
              self_array[1].unbounded_shl(rhs_array[1] as u32),
              self_array[2].unbounded_shl(rhs_array[2] as u32),
              self_array[3].unbounded_shl(rhs_array[3] as u32),
              self_array[4].unbounded_shl(rhs_array[4] as u32),
              self_array[5].unbounded_shl(rhs_array[5] as u32),
              self_array[6].unbounded_shl(rhs_array[6] as u32),
              self_array[7].unbounded_shl(rhs_array[7] as u32),
              self_array[8].unbounded_shl(rhs_array[8] as u32),
              self_array[9].unbounded_shl(rhs_array[9] as u32),
              self_array[10].unbounded_shl(rhs_array[10] as u32),
              self_array[11].unbounded_shl(rhs_array[11] as u32),
              self_array[12].unbounded_shl(rhs_array[12] as u32),
              self_array[13].unbounded_shl(rhs_array[13] as u32),
              self_array[14].unbounded_shl(rhs_array[14] as u32),
              self_array[15].unbounded_shl(rhs_array[15] as u32),
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn unbounded_shl_scalar_backend(self, rhs: u32) -> Self {
        // For x86, this technically can be done explicitly by converting
        // to `u16` or `u32` then converting back after multiplication, but that
        // may not actually be faster than auto-vectorization.
        pick! {
          if #[cfg(target_feature="simd128")] {
            // The intrinsic performs wrapping shift so we need to mask the result.
            if rhs >= 8 { Self::ZERO } else { Self(u8x16_shl(self.0, rhs)) }
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            // The intrinsic has different semantics so we need to saturate `rhs`.
            Self(vshlq_u8(self.0, vmovq_n_s8(rhs.min(i8::MAX as u32) as i8)))
          } else {
            let values: u16x8 = cast(self);
            let shifted = values.unbounded_shl_scalar(rhs);
            let crossed = u16x8::splat(!(0x00FFu16.wrapping_shl(rhs) & 0xFF00));
            cast(shifted & crossed)
          }
        }
    }

    #[inline]
    pub(crate) fn unbounded_shr_backend(self, rhs: Self) -> Self {
        // For x86, this technically can be done explicitly by converting
        // to `u16` or `u32` then converting back after multiplication, but that may
        // not actually be faster than auto-vectorization.
        pick! {
          if #[cfg(all(target_feature="neon", target_arch="aarch64"))] {
            // Negate `rhs` because there is no direct shift-right intrinsic, and
            // mask to hide `rhs` overflow.
            Self(vshlq_u8(self.0, vnegq_s8(vreinterpretq_s8_u8(rhs.0)))) & rhs.simd_lt(8)
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
        // to `u16` or `u32` then converting back after multiplication, but that
        // may not actually be faster than auto-vectorization.
        pick! {
          if #[cfg(target_feature="simd128")] {
            if rhs < 8 { Self(u8x16_shr(self.0, rhs)) } else { Self::ZERO }
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            // Negate `rhs` because there is no direct shift-right intrinsic, and
            // restrict it to prevent overflow.
            Self(vshlq_u8(self.0, vmovq_n_s8(-rhs.min(8).cast_signed() as i8)))
          } else {
            let values: u16x8 = cast(self);
            let shifted = values.unbounded_shr_scalar(rhs);
            let crossed = u16x8::splat(!(0xFF00u16.wrapping_shr(rhs) & 0x00FF));
            cast(shifted & crossed)
          }
        }
    }

    #[inline]
    pub(crate) fn saturating_add_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse2")] {
            Self(_mm_adds_epu8(self.0, rhs.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(u8x16_add_sat(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vqaddq_u8(self.0, rhs.0))
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
            Self(_mm_subs_epu8(self.0, rhs.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(u8x16_sub_sat(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vqsubq_u8(self.0, rhs.0))
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
        let overflow = high.simd_ne(Self::ZERO);
        (low, overflow)
    }

    #[inline]
    pub(crate) fn widening_mul_backend(self, rhs: Self) -> u16x16 {
        pick! {
          if #[cfg(all(target_feature="neon", target_arch="aarch64"))] {
            let low_wide_mul = vmull_u8(vget_low_u8(self.0), vget_low_u8(rhs.0));
            let high_wide_mul = vmull_u8(vget_high_u8(self.0), vget_high_u8(rhs.0));

            u16x16(crate::u16x16_backend::Inner {
              lo: u16x8(low_wide_mul),
              hi: u16x8(high_wide_mul),
            })
          } else {
            let self_array = self.to_array();
            let rhs_array = rhs.to_array();

            u16x16::from_array([
              (self_array[0] as u16).wrapping_mul(rhs_array[0] as u16),
              (self_array[1] as u16).wrapping_mul(rhs_array[1] as u16),
              (self_array[2] as u16).wrapping_mul(rhs_array[2] as u16),
              (self_array[3] as u16).wrapping_mul(rhs_array[3] as u16),
              (self_array[4] as u16).wrapping_mul(rhs_array[4] as u16),
              (self_array[5] as u16).wrapping_mul(rhs_array[5] as u16),
              (self_array[6] as u16).wrapping_mul(rhs_array[6] as u16),
              (self_array[7] as u16).wrapping_mul(rhs_array[7] as u16),
              (self_array[8] as u16).wrapping_mul(rhs_array[8] as u16),
              (self_array[9] as u16).wrapping_mul(rhs_array[9] as u16),
              (self_array[10] as u16).wrapping_mul(rhs_array[10] as u16),
              (self_array[11] as u16).wrapping_mul(rhs_array[11] as u16),
              (self_array[12] as u16).wrapping_mul(rhs_array[12] as u16),
              (self_array[13] as u16).wrapping_mul(rhs_array[13] as u16),
              (self_array[14] as u16).wrapping_mul(rhs_array[14] as u16),
              (self_array[15] as u16).wrapping_mul(rhs_array[15] as u16),
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn mul_keep_low_high_backend(self, rhs: Self) -> (Self, Self) {
        pick! {
          if #[cfg(all(target_feature="neon", target_arch="aarch64"))] {
            let low_wide_mul = vreinterpretq_u8_u16(
              vmull_u8(vget_low_u8(self.0), vget_low_u8(rhs.0)),
            );
            let high_wide_mul = vreinterpretq_u8_u16(
              vmull_u8(vget_high_u8(self.0), vget_high_u8(rhs.0)),
            );
            let low_high = vuzpq_u8(low_wide_mul, high_wide_mul);
            (
              Self(low_high.0),
              Self(low_high.1),
            )
          } else {
            // TODO(perf): This implementation looks quite bad. Is there a better
            // one?

            let self_array = self.to_array();
            let rhs_array = rhs.to_array();

            let widening_mul = [
              (self_array[0] as u16).wrapping_mul(rhs_array[0] as u16),
              (self_array[1] as u16).wrapping_mul(rhs_array[1] as u16),
              (self_array[2] as u16).wrapping_mul(rhs_array[2] as u16),
              (self_array[3] as u16).wrapping_mul(rhs_array[3] as u16),
              (self_array[4] as u16).wrapping_mul(rhs_array[4] as u16),
              (self_array[5] as u16).wrapping_mul(rhs_array[5] as u16),
              (self_array[6] as u16).wrapping_mul(rhs_array[6] as u16),
              (self_array[7] as u16).wrapping_mul(rhs_array[7] as u16),
              (self_array[8] as u16).wrapping_mul(rhs_array[8] as u16),
              (self_array[9] as u16).wrapping_mul(rhs_array[9] as u16),
              (self_array[10] as u16).wrapping_mul(rhs_array[10] as u16),
              (self_array[11] as u16).wrapping_mul(rhs_array[11] as u16),
              (self_array[12] as u16).wrapping_mul(rhs_array[12] as u16),
              (self_array[13] as u16).wrapping_mul(rhs_array[13] as u16),
              (self_array[14] as u16).wrapping_mul(rhs_array[14] as u16),
              (self_array[15] as u16).wrapping_mul(rhs_array[15] as u16),
            ];

            (
              Self::from_array([
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
              Self::from_array([
                (widening_mul[0] >> 8) as u8,
                (widening_mul[1] >> 8) as u8,
                (widening_mul[2] >> 8) as u8,
                (widening_mul[3] >> 8) as u8,
                (widening_mul[4] >> 8) as u8,
                (widening_mul[5] >> 8) as u8,
                (widening_mul[6] >> 8) as u8,
                (widening_mul[7] >> 8) as u8,
                (widening_mul[8] >> 8) as u8,
                (widening_mul[9] >> 8) as u8,
                (widening_mul[10] >> 8) as u8,
                (widening_mul[11] >> 8) as u8,
                (widening_mul[12] >> 8) as u8,
                (widening_mul[13] >> 8) as u8,
                (widening_mul[14] >> 8) as u8,
                (widening_mul[15] >> 8) as u8,
              ]),
            )
          }
        }
    }

    #[inline]
    pub(crate) fn mul_keep_high_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(all(target_feature="neon", target_arch="aarch64"))] {
            let low_wide_mul = vreinterpretq_u8_u16(
              vmull_u8(vget_low_u8(self.0), vget_low_u8(rhs.0)),
            );
            let high_wide_mul = vreinterpretq_u8_u16(
              vmull_u8(vget_high_u8(self.0), vget_high_u8(rhs.0)),
            );
            Self(vuzpq_u8(low_wide_mul, high_wide_mul).1)
          } else {
            let self_array = self.to_array();
            let rhs_array = rhs.to_array();

            Self::from_array([
              ((self_array[0] as u16).wrapping_mul(rhs_array[0] as u16) >> 8) as u8,
              ((self_array[1] as u16).wrapping_mul(rhs_array[1] as u16) >> 8) as u8,
              ((self_array[2] as u16).wrapping_mul(rhs_array[2] as u16) >> 8) as u8,
              ((self_array[3] as u16).wrapping_mul(rhs_array[3] as u16) >> 8) as u8,
              ((self_array[4] as u16).wrapping_mul(rhs_array[4] as u16) >> 8) as u8,
              ((self_array[5] as u16).wrapping_mul(rhs_array[5] as u16) >> 8) as u8,
              ((self_array[6] as u16).wrapping_mul(rhs_array[6] as u16) >> 8) as u8,
              ((self_array[7] as u16).wrapping_mul(rhs_array[7] as u16) >> 8) as u8,
              ((self_array[8] as u16).wrapping_mul(rhs_array[8] as u16) >> 8) as u8,
              ((self_array[9] as u16).wrapping_mul(rhs_array[9] as u16) >> 8) as u8,
              ((self_array[10] as u16).wrapping_mul(rhs_array[10] as u16) >> 8) as u8,
              ((self_array[11] as u16).wrapping_mul(rhs_array[11] as u16) >> 8) as u8,
              ((self_array[12] as u16).wrapping_mul(rhs_array[12] as u16) >> 8) as u8,
              ((self_array[13] as u16).wrapping_mul(rhs_array[13] as u16) >> 8) as u8,
              ((self_array[14] as u16).wrapping_mul(rhs_array[14] as u16) >> 8) as u8,
              ((self_array[15] as u16).wrapping_mul(rhs_array[15] as u16) >> 8) as u8,
            ])
          }
        }
    }
}
}

safe_target_feature! {
// Backend of SIMD unsigned integer functions.
impl u8x16 {
    #[inline]
    pub(crate) fn saturating_add_signed_backend(self, rhs: i8x16) -> Self {
        pick! {
            if #[cfg(all(target_feature = "neon", target_arch = "aarch64"))] {
                Self(vsqaddq_u8(self.0, rhs.0))
            } else {
                let (result, overflow) = self.overflowing_add_signed(rhs);
                overflow.select(rhs.is_positive().cast_unsigned(), result)
            }
        }
    }
}
}

safe_target_feature! {
// Internal helpers and backend of inconsistent functions.
impl u8x16 {
    #[inline]
    pub(crate) fn narrow_i16x8_backend(lhs: i16x8, rhs: i16x8) -> Self {
        pick! {
            if #[cfg(target_feature = "sse2")] {
                u8x16(_mm_packus_epi16(lhs.0, rhs.0))
            } else if #[cfg(target_feature = "simd128")] {
                u8x16(u8x16_narrow_i16x8(lhs.0, rhs.0))
            } else if #[cfg(all(target_feature = "neon", target_arch = "aarch64"))] {
                let lhs = vqmovun_s16(lhs.0);
                let rhs = vqmovun_s16(rhs.0);
                u8x16(vcombine_u8(lhs, rhs))
            } else {
                fn clamp(a: i16) -> u8 {
                      if a < u8::MIN as i16 {
                          u8::MIN
                      } else if a > u8::MAX as i16 {
                          u8::MAX
                      } else {
                          a as u8
                      }
                }

                Self::from_array([
                    clamp(lhs.as_array()[0]),
                    clamp(lhs.as_array()[1]),
                    clamp(lhs.as_array()[2]),
                    clamp(lhs.as_array()[3]),
                    clamp(lhs.as_array()[4]),
                    clamp(lhs.as_array()[5]),
                    clamp(lhs.as_array()[6]),
                    clamp(lhs.as_array()[7]),
                    clamp(rhs.as_array()[0]),
                    clamp(rhs.as_array()[1]),
                    clamp(rhs.as_array()[2]),
                    clamp(rhs.as_array()[3]),
                    clamp(rhs.as_array()[4]),
                    clamp(rhs.as_array()[5]),
                    clamp(rhs.as_array()[6]),
                    clamp(rhs.as_array()[7]),
                ])
            }
        }
    }
}
}
