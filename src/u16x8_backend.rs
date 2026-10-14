use crate::imports::*;

pick! {
  if #[cfg(target_feature="sse2")] {
    pub(crate) type Inner = m128i;
  } else if #[cfg(target_feature="simd128")] {
    pub(crate) type Inner = v128;
  } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
    pub(crate) type Inner = uint16x8_t;
  } else {
    #[repr(C, align(16))]
    #[derive(Clone, Copy)]
    pub(crate) struct Inner {
      pub arr: [u16; 8],
    }
  }
}

impl u16x8 {
    #[inline]
    pub(crate) fn not_backend(self) -> Self {
        self ^ cast::<u128, u16x8>(u128::MAX)
    }

    #[inline]
    pub(crate) fn add_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse2")] {
            Self(add_i16_m128i(self.0, rhs.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(u16x8_add(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            unsafe { Self(vaddq_u16(self.0, rhs.0)) }
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
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn sub_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse2")] {
            Self(sub_i16_m128i(self.0, rhs.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(u16x8_sub(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            unsafe {Self(vsubq_u16(self.0, rhs.0))}
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
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn mul_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse2")] {
            Self(mul_i16_keep_low_m128i(self.0, rhs.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(u16x8_mul(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            unsafe {Self(vmulq_u16(self.0, rhs.0))}
          } else {
            Self::from_array([
              self.0.arr[0].wrapping_mul(rhs.0.arr[0]),
              self.0.arr[1].wrapping_mul(rhs.0.arr[1]),
              self.0.arr[2].wrapping_mul(rhs.0.arr[2]),
              self.0.arr[3].wrapping_mul(rhs.0.arr[3]),
              self.0.arr[4].wrapping_mul(rhs.0.arr[4]),
              self.0.arr[5].wrapping_mul(rhs.0.arr[5]),
              self.0.arr[6].wrapping_mul(rhs.0.arr[6]),
              self.0.arr[7].wrapping_mul(rhs.0.arr[7]),
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn bitand_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse2")] {
            Self(bitand_m128i(self.0, rhs.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(v128_and(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            unsafe {Self(vandq_u16(self.0, rhs.0))}
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
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn bitor_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse2")] {
            Self(bitor_m128i(self.0, rhs.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(v128_or(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            unsafe {Self(vorrq_u16(self.0, rhs.0))}
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
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn bitxor_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse2")] {
            Self(bitxor_m128i(self.0, rhs.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(v128_xor(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            unsafe {Self(veorq_u16(self.0, rhs.0))}
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
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn simd_eq_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse2")] {
            Self(cmp_eq_mask_i16_m128i(self.0, rhs.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(u16x8_eq(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            unsafe {Self(vceqq_u16(self.0, rhs.0))}
          } else {
            Self::from_array([
              if self.0.arr[0] == rhs.0.arr[0] { u16::MAX } else { 0 },
              if self.0.arr[1] == rhs.0.arr[1] { u16::MAX } else { 0 },
              if self.0.arr[2] == rhs.0.arr[2] { u16::MAX } else { 0 },
              if self.0.arr[3] == rhs.0.arr[3] { u16::MAX } else { 0 },
              if self.0.arr[4] == rhs.0.arr[4] { u16::MAX } else { 0 },
              if self.0.arr[5] == rhs.0.arr[5] { u16::MAX } else { 0 },
              if self.0.arr[6] == rhs.0.arr[6] { u16::MAX } else { 0 },
              if self.0.arr[7] == rhs.0.arr[7] { u16::MAX } else { 0 },
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
            Self(u16x8_ne(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            !self.simd_eq(rhs)
          } else {
            Self::from_array([
              if self.0.arr[0] != rhs.0.arr[0] { u16::MAX } else { 0 },
              if self.0.arr[1] != rhs.0.arr[1] { u16::MAX } else { 0 },
              if self.0.arr[2] != rhs.0.arr[2] { u16::MAX } else { 0 },
              if self.0.arr[3] != rhs.0.arr[3] { u16::MAX } else { 0 },
              if self.0.arr[4] != rhs.0.arr[4] { u16::MAX } else { 0 },
              if self.0.arr[5] != rhs.0.arr[5] { u16::MAX } else { 0 },
              if self.0.arr[6] != rhs.0.arr[6] { u16::MAX } else { 0 },
              if self.0.arr[7] != rhs.0.arr[7] { u16::MAX } else { 0 },
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn simd_lt_backend(self, rhs: Self) -> Self {
        // no lt, so reverse gt
        Self::simd_gt(rhs, self)
    }

    #[inline]
    pub(crate) fn simd_gt_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature = "sse2")] {
            use safe_arch::*;

            let bias = m128i::from([0x8000u16; 8]);

            let a_biased = sub_i16_m128i(self.0, bias);
            let b_biased = sub_i16_m128i(rhs.0, bias);
            let mask = cmp_gt_mask_i16_m128i(a_biased, b_biased);

            Self(mask)
          } else if #[cfg(target_feature="simd128")] {
            Self(u16x8_gt(self.0, rhs.0))
          } else if #[cfg(all(target_feature = "neon", target_arch = "aarch64"))] {
            unsafe {
              use core::arch::aarch64::*;
              Self(vcgtq_u16(self.0, rhs.0))
            }
          } else {
            Self::from_array([
                if self.0.arr[0] > rhs.0.arr[0] { u16::MAX } else { 0 },
                if self.0.arr[1] > rhs.0.arr[1] { u16::MAX } else { 0 },
                if self.0.arr[2] > rhs.0.arr[2] { u16::MAX } else { 0 },
                if self.0.arr[3] > rhs.0.arr[3] { u16::MAX } else { 0 },
                if self.0.arr[4] > rhs.0.arr[4] { u16::MAX } else { 0 },
                if self.0.arr[5] > rhs.0.arr[5] { u16::MAX } else { 0 },
                if self.0.arr[6] > rhs.0.arr[6] { u16::MAX } else { 0 },
                if self.0.arr[7] > rhs.0.arr[7] { u16::MAX } else { 0 },
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
            Self(u16x8_le(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            !self.simd_gt(rhs)
          } else {
            Self::from_array([
              if self.0.arr[0] <= rhs.0.arr[0] { u16::MAX } else { 0 },
              if self.0.arr[1] <= rhs.0.arr[1] { u16::MAX } else { 0 },
              if self.0.arr[2] <= rhs.0.arr[2] { u16::MAX } else { 0 },
              if self.0.arr[3] <= rhs.0.arr[3] { u16::MAX } else { 0 },
              if self.0.arr[4] <= rhs.0.arr[4] { u16::MAX } else { 0 },
              if self.0.arr[5] <= rhs.0.arr[5] { u16::MAX } else { 0 },
              if self.0.arr[6] <= rhs.0.arr[6] { u16::MAX } else { 0 },
              if self.0.arr[7] <= rhs.0.arr[7] { u16::MAX } else { 0 },
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
            Self(u16x8_ge(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            !self.simd_lt(rhs)
          } else {
            Self::from_array([
              if self.0.arr[0] >= rhs.0.arr[0] { u16::MAX } else { 0 },
              if self.0.arr[1] >= rhs.0.arr[1] { u16::MAX } else { 0 },
              if self.0.arr[2] >= rhs.0.arr[2] { u16::MAX } else { 0 },
              if self.0.arr[3] >= rhs.0.arr[3] { u16::MAX } else { 0 },
              if self.0.arr[4] >= rhs.0.arr[4] { u16::MAX } else { 0 },
              if self.0.arr[5] >= rhs.0.arr[5] { u16::MAX } else { 0 },
              if self.0.arr[6] >= rhs.0.arr[6] { u16::MAX } else { 0 },
              if self.0.arr[7] >= rhs.0.arr[7] { u16::MAX } else { 0 },
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn replace_const_backend<const INDEX: usize>(self, value: u16) -> Self {
        /// Because of limitations in const generics, this must have separate
        /// branches per index. This macro is used to avoid duplication.
        macro_rules! use_indices {
      (0..8 = [$($INDEX:literal),*]) => {
        pick! {
          if #[cfg(target_feature = "sse2")] {
            match INDEX {
              $($INDEX => Self(insert_i16_from_i32_m128i::<$INDEX>(self.0, value as i32)),)*
              8.. => unreachable!(),
            }
          } else if #[cfg(all(
            target_arch = "aarch64",
            target_feature = "neon",
            target_endian = "little",
          ))] {
            unsafe {
              match INDEX {
                $($INDEX => Self(vsetq_lane_u16::<$INDEX>(value, self.0)),)*
                8.. => unreachable!(),
              }
            }
          } else if #[cfg(target_feature = "simd128")] {
            Self(u16x8_replace_lane::<INDEX>(self.0, value))
          } else {
            let mut result = self;
            result.as_mut_array()[INDEX] = value;
            result
          }
        }
      };
    }
        use_indices! { 0..8 = [0, 1, 2, 3, 4, 5, 6, 7] }
    }

    #[inline]
    pub(crate) fn extract_const_backend<const INDEX: usize>(self) -> u16 {
        /// Because of limitations in const generics, this must have separate
        /// branches per index. This macro is used to avoid duplication.
        macro_rules! use_indices {
      (0..8 = [$($INDEX:literal),*]) => {
        pick! {
          if #[cfg(target_feature = "sse2")] {
            match INDEX {
              $($INDEX => extract_i16_as_i32_m128i::<$INDEX>(self.0).cast_unsigned() as u16,)*
              8.. => unreachable!(),
            }
          } else if #[cfg(all(
            target_arch = "aarch64",
            target_feature = "neon",
            target_endian = "little",
          ))] {
            unsafe {
              match INDEX {
                $($INDEX => vgetq_lane_u16::<$INDEX>(self.0),)*
                8.. => unreachable!(),
              }
            }
          } else if #[cfg(target_feature = "simd128")] {
            u16x8_extract_lane::<INDEX>(self.0)
          } else {
            self.to_array()[INDEX]
          }
        }
      };
    }
        use_indices! { 0..8 = [0, 1, 2, 3, 4, 5, 6, 7] }
    }

    #[inline]
    pub(crate) fn reduce_add_backend(self) -> u16 {
        pick! {
          if #[cfg(target_feature="sse2")] {
            // there is a horizontal add instruction on ssse3, but apparently it is very slow on some AMD CPUs
            let hi64 = shuffle_ai_f32_all_m128i::<0b01_00_11_10>(self.0);
            let sum64 = add_i16_m128i(self.0, hi64);
            let hi32 = shuffle_ai_f32_all_m128i::<0b11_10_00_01>(sum64);
            let sum32 = add_i16_m128i(sum64, hi32);
            let lo16 = shr_imm_u32_m128i::<16>(sum32);
            let sum16 = add_i16_m128i(sum32, lo16);
            extract_i16_as_i32_m128i::<0>(sum16) as u16
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            unsafe { vaddvq_u16(self.0) }
          } else {
            let arr: [u16; 8] = cast(self);

            // most boring implementation possible so optimizer doesn't overthink this
            let mut r = arr[0];
            r = r.wrapping_add(arr[1]);
            r = r.wrapping_add(arr[2]);
            r = r.wrapping_add(arr[3]);
            r = r.wrapping_add(arr[4]);
            r = r.wrapping_add(arr[5]);
            r = r.wrapping_add(arr[6]);
            r.wrapping_add(arr[7])
          }
        }
    }

    #[inline]
    pub(crate) fn reduce_mul_backend(self) -> u16 {
        pick! {
          if #[cfg(target_feature="sse2")] {
            let high_64 = shuffle_ai_f32_all_m128i::<0b01_00_11_10>(self.0);
            let reduce_64 = mul_i16_keep_low_m128i(self.0, high_64);
            let high_32 = shuffle_ai_f32_all_m128i::<0b11_10_00_01>(reduce_64);
            let reduce_32 = mul_i16_keep_low_m128i(reduce_64, high_32);
            let high_16 = shr_imm_u32_m128i::<16>(reduce_32);
            let reduce_16 = mul_i16_keep_low_m128i(reduce_32, high_16);
            extract_i16_as_i32_m128i::<0>(reduce_16) as u16
          } else if #[cfg(target_feature="simd128")] {
            let high_64 = u64x2_shuffle::<1, 0>(self.0, self.0);
            let reduce_64 = u16x8_mul(self.0, high_64);
            let high_32 = u32x4_shuffle::<1, 0, 0, 0>(reduce_64, reduce_64);
            let reduce_32 = u16x8_mul(reduce_64, high_32);
            let high_16 = u16x8_shuffle::<1, 0, 0, 0, 0, 0, 0, 0>(reduce_32, reduce_32);
            let reduce_16 = u16x8_mul(reduce_32, high_16);
            u16x8_extract_lane::<0>(reduce_16)
          } else if #[cfg(all(target_feature="neon", target_arch="aarch64"))] {
            unsafe {
              let high_64 = vextq_u16::<4>(self.0, self.0);
              let reduce_64 = vmulq_u16(self.0, high_64);
              let high_32 = vrev64q_u16(reduce_64);
              let reduce_32 = vmulq_u16(reduce_64, high_32);
              let high_16 = vrev32q_u16(reduce_32);
              let reduce_16 = vmulq_u16(reduce_32, high_16);
              vgetq_lane_u16::<0>(reduce_16)
            }
          } else {
            let array = self.to_array();

            // most boring implementation possible so optimizer doesn't overthink this
            let mut result = array[0];
            result = result.wrapping_mul(array[1]);
            result = result.wrapping_mul(array[2]);
            result = result.wrapping_mul(array[3]);
            result = result.wrapping_mul(array[4]);
            result = result.wrapping_mul(array[5]);
            result = result.wrapping_mul(array[6]);
            result.wrapping_mul(array[7])
          }
        }
    }

    #[inline]
    pub(crate) fn bitselect_backend(self, if_one: Self, if_zero: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse2")] {
            Self(bitor_m128i(bitand_m128i(if_one.0, self.0), bitandnot_m128i(self.0, if_zero.0)))
          } else if #[cfg(target_feature="simd128")] {
            Self(v128_bitselect(if_one.0, if_zero.0, self.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            unsafe {Self(vbslq_u16(self.0, if_one.0, if_zero.0))}
          } else {
            generic_bit_blend(self, if_one, if_zero)
          }
        }
    }

    #[inline]
    pub(crate) fn select_backend(self, if_true: Self, if_false: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse4.1")] {
            Self(blend_varying_i8_m128i(if_false.0, if_true.0, self.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(v128_bitselect(if_true.0, if_false.0, self.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            unsafe {Self(vbslq_u16(self.0, if_true.0, if_false.0))}
          } else {
            generic_bit_blend(self, if_true, if_false)
          }
        }
    }

    #[inline]
    pub(crate) fn to_bitmask_backend(self) -> u32 {
        pick! {
          if #[cfg(target_feature="sse2")] {
            (move_mask_i8_m128i( pack_i16_to_i8_m128i(self.0,self.0)) as u32) & 0xff
          } else if #[cfg(target_feature="simd128")] {
            u16x8_bitmask(self.0) as u32
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            unsafe
            {
              // set all to 1 if top bit is set, else 0
              let masked = vcltzq_s16(self.cast_signed().0);

              // select the right bit out of each lane
              let selectbit : uint16x8_t = core::mem::transmute([1u16, 2, 4, 8, 16, 32, 64, 128]);
              let r = vandq_u16(masked, selectbit);

              // horizontally add the 16-bit lanes
              vaddvq_u16(r) as u32
             }
           } else {
            ((self.0.arr[0].cast_signed() < 0) as u32) |
            ((self.0.arr[1].cast_signed() < 0) as u32) << 1 |
            ((self.0.arr[2].cast_signed() < 0) as u32) << 2 |
            ((self.0.arr[3].cast_signed() < 0) as u32) << 3 |
            ((self.0.arr[4].cast_signed() < 0) as u32) << 4 |
            ((self.0.arr[5].cast_signed() < 0) as u32) << 5 |
            ((self.0.arr[6].cast_signed() < 0) as u32) << 6 |
            ((self.0.arr[7].cast_signed() < 0) as u32) << 7
          }
        }
    }

    #[inline]
    pub(crate) fn any_backend(self) -> bool {
        pick! {
          if #[cfg(target_feature="sse2")] {
            (move_mask_i8_m128i(self.0) & 0b1010101010101010) != 0
          } else if #[cfg(target_feature="simd128")] {
            u16x8_bitmask(self.0) != 0
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))] {
            unsafe {
              vminvq_s16(self.cast_signed().0) < 0
            }
          } else {
            let v : [u64;2] = cast(self);
            ((v[0] | v[1]) & 0x8000800080008000) != 0
          }
        }
    }

    #[inline]
    pub(crate) fn all_backend(self) -> bool {
        pick! {
          if #[cfg(target_feature="sse2")] {
            (move_mask_i8_m128i(self.0) & 0b1010101010101010) == 0b1010101010101010
          } else if #[cfg(target_feature="simd128")] {
            u16x8_bitmask(self.0) == 0b11111111
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))] {
            unsafe {
              vmaxvq_s16(self.cast_signed().0) < 0
            }
          } else {
            let v : [u64;2] = cast(self);
            (v[0] & v[1] & 0x8000800080008000) == 0x8000800080008000
          }
        }
    }

    #[inline]
    pub(crate) fn interleave_low_backend(self, other: Self) -> Self {
        pick! {
          if #[cfg(target_feature = "sse2")] {
            Self(unpack_low_i16_m128i(self.0, other.0))
          } else if #[cfg(target_feature = "simd128")] {
            Self(u16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(self.0, other.0))
          } else if #[cfg(all(target_feature = "neon", target_arch = "aarch64"))] {
            let lhs = unsafe { vget_low_u16(self.0) };
            let rhs = unsafe { vget_low_u16(other.0) };

            let zipped = unsafe { vzip_u16(lhs, rhs) };
            Self(unsafe { vcombine_u16(zipped.0, zipped.1) })
          } else {
            Self::from_array([
              self.as_array()[0],
              other.as_array()[0],
              self.as_array()[1],
              other.as_array()[1],
              self.as_array()[2],
              other.as_array()[2],
              self.as_array()[3],
              other.as_array()[3],
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn interleave_high_backend(self, other: Self) -> Self {
        pick! {
          if #[cfg(target_feature = "sse2")] {
            Self(unpack_high_i16_m128i(self.0, other.0))
          } else if #[cfg(target_feature = "simd128")] {
            Self(u16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15>(self.0, other.0))
          } else if #[cfg(all(target_feature = "neon", target_arch = "aarch64"))] {
            let lhs = unsafe { vget_high_u16(self.0) };
            let rhs = unsafe { vget_high_u16(other.0) };

            let zipped = unsafe { vzip_u16(lhs, rhs) };
            Self(unsafe { vcombine_u16(zipped.0, zipped.1) })
          } else {
            Self::from_array([
              self.as_array()[4],
              other.as_array()[4],
              self.as_array()[5],
              other.as_array()[5],
              self.as_array()[6],
              other.as_array()[6],
              self.as_array()[7],
              other.as_array()[7],
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn shuffle_backend(self, indices: u16x8) -> Self {
        pick! {
          if #[cfg(all(target_feature = "avx512bw", target_feature = "avx512vl"))] {
            #[cfg(target_arch = "x86")]
            use core::arch::x86::_mm_permutexvar_epi16;
            #[cfg(target_arch = "x86_64")]
            use core::arch::x86_64::_mm_permutexvar_epi16;
            // TODO(safe_arch): Add `_mm_permutexvar_epi16`
            Self(unsafe { m128i(_mm_permutexvar_epi16(indices.0.0, self.0.0)) })
          } else if #[cfg(any(
            target_feature = "ssse3",
            all(target_arch = "aarch64", target_feature = "neon"),
            target_feature = "simd128",
          ))] {
            let self_bytes = cast::<u16x8, u8x16>(self);
            let byte_indices = indices.to_byte_indices();

            cast::<u8x16, u16x8>(self_bytes.shuffle(byte_indices))
          } else {
            let self_array = self.to_array();
            let indices_array = indices.to_array();

            let mut result = [0; 8];
            for i in 0..8 {
              let index = indices_array[i] as usize;
              if index < 8 {
                result[i] = self_array[index];
              }
            }

            Self::from_array(result)
          }
        }
    }

    #[inline]
    pub(crate) fn shuffle_zeroing_backend(self, indices: u16x8) -> Self {
        pick! {
          if #[cfg(any(
            target_feature = "ssse3",
            all(target_arch = "aarch64", target_feature = "neon"),
            target_feature = "simd128",
          ))] {
            // Even if the `u8x16::shuffle` implementation is zeroing, our 16-bit to
            // 8-bit can trigger an overflow causing incorrect behavior
            self.shuffle(indices) & indices.simd_lt(8)
          } else {
            // The fallback branch of `shuffle` already has the behavior we want
            self.shuffle(indices)
          }
        }
    }

    #[inline]
    pub(crate) fn shuffle_wrapping_backend(self, indices: u16x8) -> Self {
        pick! {
          if #[cfg(all(target_feature = "avx512bw", target_feature = "avx512vl"))] {
            // `avx512` shuffle intrinsics are wrapping
            self.shuffle(indices)
          } else {
            self.shuffle(indices & 7)
          }
        }
    }

    #[inline]
    pub(crate) fn shuffle_2_backend(inputs: [Self; 2], indices: u16x8) -> u16x8 {
        pick! {
          if #[cfg(all(target_feature = "avx512bw", target_feature = "avx512vl"))] {
            #[cfg(target_arch = "x86")]
            use core::arch::x86::_mm_permutex2var_epi16;
            #[cfg(target_arch = "x86_64")]
            use core::arch::x86_64::_mm_permutex2var_epi16;
            // TODO(safe_arch): add `_mm_permutex2var_epi16`.
            u16x8(unsafe { m128i(_mm_permutex2var_epi16(inputs[0].0.0, indices.0.0, inputs[1].0.0)) })
          } else {
            let self_bytes = cast::<[u16x8; 2], [u8x16; 2]>(inputs);
            let byte_indices = indices.to_byte_indices();

            cast::<u8x16, u16x8>(self_bytes.shuffle(byte_indices))
          }
        }
    }

    #[inline]
    pub(crate) fn shuffle_zeroing_2_backend(inputs: [Self; 2], indices: u16x8) -> u16x8 {
        // Even if the `u8x16` shuffle is zeroing, our 16-bit to 8-bit index
        // conversion breaks for out of bounds indices.
        inputs.shuffle(indices) & indices.simd_lt(16)
    }

    #[inline]
    pub(crate) fn shuffle_wrapping_2_backend(inputs: [Self; 2], indices: u16x8) -> u16x8 {
        pick! {
          if #[cfg(all(target_feature = "avx512bw", target_feature = "avx512vl"))] {
            // `avx512` shuffle intrinsics are wrapping
            inputs.shuffle(indices)
          } else {
            inputs.shuffle(indices & 15)
          }
        }
    }

    #[inline]
    pub(crate) fn shuffle_3_backend(inputs: [Self; 3], indices: u16x8) -> u16x8 {
        let self_bytes = cast::<[u16x8; 3], [u8x16; 3]>(inputs);
        let byte_indices = indices.to_byte_indices();

        cast::<u8x16, u16x8>(self_bytes.shuffle(byte_indices))
    }

    #[inline]
    pub(crate) fn shuffle_zeroing_3_backend(inputs: [Self; 3], indices: u16x8) -> u16x8 {
        // Even if the `u8x16` shuffle is zeroing, our 16-bit to 8-bit index
        // conversion breaks for out of bounds indices.
        inputs.shuffle(indices) & indices.simd_lt(24)
    }

    #[inline]
    pub(crate) fn shuffle_wrapping_3_backend(inputs: [Self; 3], indices: u16x8) -> u16x8 {
        inputs.shuffle(indices % 24)
    }

    #[inline]
    pub(crate) fn shuffle_4_backend(inputs: [Self; 4], indices: u16x8) -> u16x8 {
        let self_bytes = cast::<[u16x8; 4], [u8x16; 4]>(inputs);
        let byte_indices = indices.to_byte_indices();

        cast::<u8x16, u16x8>(self_bytes.shuffle(byte_indices))
    }

    #[inline]
    pub(crate) fn shuffle_zeroing_4_backend(inputs: [Self; 4], indices: u16x8) -> u16x8 {
        // Even if the `u8x16` shuffle is zeroing, our 16-bit to 8-bit index
        // conversion breaks for out of bounds indices.
        inputs.shuffle(indices) & indices.simd_lt(32)
    }

    #[inline]
    pub(crate) fn shuffle_wrapping_4_backend(inputs: [Self; 4], indices: u16x8) -> u16x8 {
        inputs.shuffle(indices & 31)
    }

    #[inline]
    pub(crate) fn transpose_backend(data: [Self; 8]) -> [Self; 8] {
        pick! {
          if #[cfg(target_feature="sse2")] {
            let a1 = unpack_low_i16_m128i(data[0].0, data[1].0);
            let a2 = unpack_high_i16_m128i(data[0].0, data[1].0);
            let a3 = unpack_low_i16_m128i(data[2].0, data[3].0);
            let a4 = unpack_high_i16_m128i(data[2].0, data[3].0);
            let a5 = unpack_low_i16_m128i(data[4].0, data[5].0);
            let a6 = unpack_high_i16_m128i(data[4].0, data[5].0);
            let a7 = unpack_low_i16_m128i(data[6].0, data[7].0);
            let a8 = unpack_high_i16_m128i(data[6].0, data[7].0);

            let b1 = unpack_low_i32_m128i(a1, a3);
            let b2 = unpack_high_i32_m128i(a1, a3);
            let b3 = unpack_low_i32_m128i(a2, a4);
            let b4 = unpack_high_i32_m128i(a2, a4);
            let b5 = unpack_low_i32_m128i(a5, a7);
            let b6 = unpack_high_i32_m128i(a5, a7);
            let b7 = unpack_low_i32_m128i(a6, a8);
            let b8 = unpack_high_i32_m128i(a6, a8);

            [
              Self(unpack_low_i64_m128i(b1, b5)),
              Self(unpack_high_i64_m128i(b1, b5)),
              Self(unpack_low_i64_m128i(b2, b6)),
              Self(unpack_high_i64_m128i(b2, b6)),
              Self(unpack_low_i64_m128i(b3, b7)),
              Self(unpack_high_i64_m128i(b3, b7)),
              Self(unpack_low_i64_m128i(b4, b8)),
              Self(unpack_high_i64_m128i(b4, b8)),
            ]
         } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{

              #[inline] fn vtrq32(a : uint16x8_t, b : uint16x8_t) -> (uint16x8_t, uint16x8_t)
              {
                  unsafe {
                    let r = vtrnq_u32(vreinterpretq_u32_u16(a),vreinterpretq_u32_u16(b));
                    (vreinterpretq_u16_u32(r.0), vreinterpretq_u16_u32(r.1))
                  }
              }

            unsafe {
              let (q0,q2) = vtrq32(data[0].0, data[2].0);
              let (q1,q3) = vtrq32(data[1].0, data[3].0);
              let (q4,q6) = vtrq32(data[4].0, data[6].0);
              let (q5,q7) = vtrq32(data[5].0, data[7].0);

              let b1 = vtrnq_u16(q0, q1);
              let b2 = vtrnq_u16(q2, q3);
              let b3 = vtrnq_u16(q4, q5);
              let b4 = vtrnq_u16(q6, q7);

              // There is no vtrnq_u64 unfortunately, so there's this mess
              // which does a somewhat reasonable job, but not as good as the
              // assembly versions which just swap the 64 bit register aliases.
              [
                Self(vcombine_u16(vget_low_u16(b1.0), vget_low_u16(b3.0))),
                Self(vcombine_u16(vget_low_u16(b1.1), vget_low_u16(b3.1))),
                Self(vcombine_u16(vget_low_u16(b2.0), vget_low_u16(b4.0))),
                Self(vcombine_u16(vget_low_u16(b2.1), vget_low_u16(b4.1))),
                Self(vcombine_u16(vget_high_u16(b1.0), vget_high_u16(b3.0))),
                Self(vcombine_u16(vget_high_u16(b1.1), vget_high_u16(b3.1))),
                Self(vcombine_u16(vget_high_u16(b2.0), vget_high_u16(b4.0))),
                Self(vcombine_u16(vget_high_u16(b2.1), vget_high_u16(b4.1))),
              ]
            }
          } else if #[cfg(target_feature="simd128")] {
            #[inline] fn lo_i16(a : v128, b : v128) -> v128 { u16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(a,b) }
            #[inline] fn hi_i16(a : v128, b : v128) -> v128 { u16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15>(a,b) }
            #[inline] fn lo_i32(a : v128, b : v128) -> v128 { u32x4_shuffle::<0, 4, 1, 5>(a,b) }
            #[inline] fn hi_i32(a : v128, b : v128) -> v128 { u32x4_shuffle::<2, 6, 3, 7>(a,b) }
            #[inline] fn lo_i64(a : v128, b : v128) -> v128 { u64x2_shuffle::<0, 2>(a,b) }
            #[inline] fn hi_i64(a : v128, b : v128) -> v128 { u64x2_shuffle::<1, 3>(a,b) }

            let a1 = lo_i16(data[0].0, data[1].0);
            let a2 = hi_i16(data[0].0, data[1].0);
            let a3 = lo_i16(data[2].0, data[3].0);
            let a4 = hi_i16(data[2].0, data[3].0);
            let a5 = lo_i16(data[4].0, data[5].0);
            let a6 = hi_i16(data[4].0, data[5].0);
            let a7 = lo_i16(data[6].0, data[7].0);
            let a8 = hi_i16(data[6].0, data[7].0);

            let b1 = lo_i32(a1, a3);
            let b2 = hi_i32(a1, a3);
            let b3 = lo_i32(a2, a4);
            let b4 = hi_i32(a2, a4);
            let b5 = lo_i32(a5, a7);
            let b6 = hi_i32(a5, a7);
            let b7 = lo_i32(a6, a8);
            let b8 = hi_i32(a6, a8);

            [
              Self(lo_i64(b1, b5)),
              Self(hi_i64(b1, b5)),
              Self(lo_i64(b2, b6)),
              Self(hi_i64(b2, b6)),
              Self(lo_i64(b3, b7)),
              Self(hi_i64(b3, b7)),
              Self(lo_i64(b4, b8)),
              Self(hi_i64(b4, b8)),
            ]

          } else {
            #[inline(always)]
            fn transpose_column(data: &[u16x8; 8], index: usize) -> u16x8 {
              u16x8::from_array([
                data[0].as_array()[index],
                data[1].as_array()[index],
                data[2].as_array()[index],
                data[3].as_array()[index],
                data[4].as_array()[index],
                data[5].as_array()[index],
                data[6].as_array()[index],
                data[7].as_array()[index],
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
            ]
          }
        }
    }

    #[inline]
    pub(crate) fn shl_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(all(target_feature="avx512bw", target_feature="avx512vl"))] {
            #[cfg(target_arch = "x86")]
            use core::arch::x86::_mm_sllv_epi16;
            #[cfg(target_arch = "x86_64")]
            use core::arch::x86_64::_mm_sllv_epi16;

            // Mask `rhs` to 15 to match `wrapping_shl`.
            let rhs = bitand_m128i(rhs.0, set_splat_i16_m128i(15));
            // TODO(safe_arch): Add `_mm_sllv_epi16`.
            cast(unsafe { _mm_sllv_epi16(self.0.0, rhs.0) })
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))] {
            unsafe {
              // Mask `rhs` to 15 to match `wrapping_shl`.
              let rhs = vreinterpretq_s16_u16(vandq_u16(rhs.0, vmovq_n_u16(15)));
              Self(vshlq_u16(self.0, rhs))
            }
          } else {
            let self_array = self.to_array();
            let rhs_array = rhs.to_array();

            Self::from_array([
              self_array[0].wrapping_shl(rhs_array[0] as u32),
              self_array[1].wrapping_shl(rhs_array[1] as u32),
              self_array[2].wrapping_shl(rhs_array[2] as u32),
              self_array[3].wrapping_shl(rhs_array[3] as u32),
              self_array[4].wrapping_shl(rhs_array[4] as u32),
              self_array[5].wrapping_shl(rhs_array[5] as u32),
              self_array[6].wrapping_shl(rhs_array[6] as u32),
              self_array[7].wrapping_shl(rhs_array[7] as u32),
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn shl_scalar_backend(self, rhs: u32) -> Self {
        pick! {
          if #[cfg(target_feature="sse2")] {
            // Use `rhs % 16` to perform wrapping shift and not unbounded shift.
            let shift = cast([rhs as u64 & 15, 0]);
            Self(shl_all_u16_m128i(self.0, shift))
          } else if #[cfg(target_feature="simd128")] {
            Self(u16x8_shl(self.0, rhs))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            // Use `rhs % 16` to perform wrapping shift and not unbounded shift.
            unsafe {Self(vshlq_u16(self.0, vmovq_n_s16(rhs as i16 & 15)))}
          } else {
            Self::from_array([
              self.0.arr[0].wrapping_shl(rhs),
              self.0.arr[1].wrapping_shl(rhs),
              self.0.arr[2].wrapping_shl(rhs),
              self.0.arr[3].wrapping_shl(rhs),
              self.0.arr[4].wrapping_shl(rhs),
              self.0.arr[5].wrapping_shl(rhs),
              self.0.arr[6].wrapping_shl(rhs),
              self.0.arr[7].wrapping_shl(rhs),
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn shr_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(all(target_feature="avx512bw", target_feature="avx512vl"))] {
            #[cfg(target_arch = "x86")]
            use core::arch::x86::_mm_srlv_epi16;
            #[cfg(target_arch = "x86_64")]
            use core::arch::x86_64::_mm_srlv_epi16;

            // Mask `rhs` to 15 to match `wrapping_shr`.
            let rhs = bitand_m128i(rhs.0, set_splat_i16_m128i(15));
            // TODO(safe_arch): Add `_mm_srlv_epi16`.
            cast(unsafe { _mm_srlv_epi16(self.0.0, rhs.0) })
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))] {
            unsafe {
              // Mask `rhs` to 15 to match `wrapping_shr`, and negate it because
              // there is no shift-right intrinsic.
              let neg_rhs = vnegq_s16(vreinterpretq_s16_u16(vandq_u16(rhs.0, vmovq_n_u16(15))));
              Self(vshlq_u16(self.0, neg_rhs))
            }
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
            Self(shr_all_u16_m128i(self.0, shift))
          } else if #[cfg(target_feature="simd128")] {
            Self(u16x8_shr(self.0, rhs))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            // Use `rhs % 16` to perform wrapping shift and not unbounded shift.
            unsafe {Self(vshlq_u16(self.0, vmovq_n_s16( -(rhs as i16 & 15))))}
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
          if #[cfg(target_feature="sse4.1")] {
            Self(max_u16_m128i(self.0, rhs.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(u16x8_max(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            unsafe {Self(vmaxq_u16(self.0, rhs.0))}
          } else {
            let arr: [u16; 8] = cast(self);
            let rhs: [u16; 8] = cast(rhs);
            cast([
              arr[0].max(rhs[0]),
              arr[1].max(rhs[1]),
              arr[2].max(rhs[2]),
              arr[3].max(rhs[3]),
              arr[4].max(rhs[4]),
              arr[5].max(rhs[5]),
              arr[6].max(rhs[6]),
              arr[7].max(rhs[7]),
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn min_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse4.1")] {
            Self(min_u16_m128i(self.0, rhs.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(u16x8_min(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            unsafe {Self(vminq_u16(self.0, rhs.0))}
          } else {
            let arr: [u16; 8] = cast(self);
            let rhs: [u16; 8] = cast(rhs);
            cast([
              arr[0].min(rhs[0]),
              arr[1].min(rhs[1]),
              arr[2].min(rhs[2]),
              arr[3].min(rhs[3]),
              arr[4].min(rhs[4]),
              arr[5].min(rhs[5]),
              arr[6].min(rhs[6]),
              arr[7].min(rhs[7]),
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn reduce_max_backend(self) -> u16 {
        pick! {
          if #[cfg(all(target_feature="ssse3", target_feature="sse4.1"))] {
            let hi64 = shuffle_ai_f32_all_m128i::<0b01_00_11_10>(self.0);
            let sum64 = max_u16_m128i(self.0, hi64);
            let hi32 = shuffle_ai_f32_all_m128i::<0b11_10_00_01>(sum64);
            let sum32 = max_u16_m128i(sum64, hi32);
            let lo16 = shr_imm_u32_m128i::<16>(sum32);
            let sum16 = max_u16_m128i(sum32, lo16);
            extract_i16_as_i32_m128i::<0>(sum16) as u16
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            unsafe { vmaxvq_u16(self.0) }
          } else {
            let arr: [u16; 8] = cast(self);

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
    pub(crate) fn reduce_min_backend(self) -> u16 {
        pick! {
          if #[cfg(all(target_feature="ssse3", target_feature="sse4.1"))] {
            let hi64 = shuffle_ai_f32_all_m128i::<0b01_00_11_10>(self.0);
            let sum64 = min_u16_m128i(self.0, hi64);
            let hi32 = shuffle_ai_f32_all_m128i::<0b11_10_00_01>(sum64);
            let sum32 = min_u16_m128i(sum64, hi32);
            let lo16 = shr_imm_u32_m128i::<16>(sum32);
            let sum16 = min_u16_m128i(sum32, lo16);
            extract_i16_as_i32_m128i::<0>(sum16) as u16
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            unsafe { vminvq_u16(self.0) }
          } else {
            let arr: [u16; 8] = cast(self);

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
    pub(crate) fn unbounded_shl_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(all(target_feature="avx512bw", target_feature="avx512vl"))] {
            #[cfg(target_arch = "x86")]
            use core::arch::x86::_mm_sllv_epi16;
            #[cfg(target_arch = "x86_64")]
            use core::arch::x86_64::_mm_sllv_epi16;

            // TODO(safe_arch): Add `_mm_sllv_epi16`.
            cast(unsafe { _mm_sllv_epi16(self.0.0, rhs.0.0) })
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))] {
            unsafe {
              // The intrinsic has different semantics so we need to mask ourselves.
              Self(vshlq_u16(self.0, vreinterpretq_s16_u16(rhs.0))) & rhs.simd_lt(16)
            }
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
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn unbounded_shl_scalar_backend(self, rhs: u32) -> Self {
        pick! {
          if #[cfg(target_feature="sse2")] {
            Self(shl_all_u16_m128i(self.0, cast([rhs as u64, 0])))
          } else if #[cfg(target_feature="simd128")] {
            // The intrinsic performs wrapping shift so we need to mask the result.
            if rhs >= 16 { Self::ZERO } else { Self(u16x8_shl(self.0, rhs)) }
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            // The intrinsic has different semantics so we need to saturate `rhs`.
            unsafe { Self(vshlq_u16(self.0, vmovq_n_s16(rhs.min(16) as i16))) }
          } else {
            Self::from_array([
              self.0.arr[0].unbounded_shl(rhs),
              self.0.arr[1].unbounded_shl(rhs),
              self.0.arr[2].unbounded_shl(rhs),
              self.0.arr[3].unbounded_shl(rhs),
              self.0.arr[4].unbounded_shl(rhs),
              self.0.arr[5].unbounded_shl(rhs),
              self.0.arr[6].unbounded_shl(rhs),
              self.0.arr[7].unbounded_shl(rhs),
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn unbounded_shr_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(all(target_feature="avx512bw", target_feature="avx512vl"))] {
            #[cfg(target_arch = "x86")]
            use core::arch::x86::_mm_srlv_epi16;
            #[cfg(target_arch = "x86_64")]
            use core::arch::x86_64::_mm_srlv_epi16;

            // TODO(safe_arch): Add `_mm_srlv_epi16`.
            cast(unsafe { _mm_srlv_epi16(self.0.0, rhs.0.0) })
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))] {
            unsafe {
              // Negate `rhs` because there is no direct shift-right intrinsic, and
              // mask to hide `rhs` overflow.
              Self(vshlq_u16(self.0, vnegq_s16(vreinterpretq_s16_u16(rhs.0)))) & rhs.simd_lt(16)
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
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn unbounded_shr_scalar_backend(self, rhs: u32) -> Self {
        pick! {
          if #[cfg(target_feature="sse2")] {
            Self(shr_all_u16_m128i(self.0, cast([rhs as u64, 0])))
          } else if #[cfg(target_feature="simd128")] {
            if rhs < 16 { Self(u16x8_shr(self.0, rhs)) } else { Self::ZERO }
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            unsafe {
              // Negate `rhs` because there is no direct shift-right intrinsic, and
              // restrict it to prevent overflow.
              Self(vshlq_u16(self.0, vmovq_n_s16(-rhs.min(16).cast_signed() as i16)))
            }
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
            Self(add_saturating_u16_m128i(self.0, rhs.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(u16x8_add_sat(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            unsafe {Self(vqaddq_u16(self.0, rhs.0))}
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
            Self(sub_saturating_u16_m128i(self.0, rhs.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(u16x8_sub_sat(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            unsafe {Self(vqsubq_u16(self.0, rhs.0))}
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
        let overflow = high.simd_ne(Self::ZERO);
        (low, overflow)
    }

    #[inline]
    pub(crate) fn widening_mul_backend(self, rhs: Self) -> u32x8 {
        pick! {
          if #[cfg(target_feature="avx2")] {
            let a = convert_to_i32_m256i_from_u16_m128i(self.0);
            let b = convert_to_i32_m256i_from_u16_m128i(rhs.0);
            u32x8(mul_i32_keep_low_m256i(a,b))
          } else if #[cfg(target_feature="sse2")] {
            let low = mul_i16_keep_low_m128i(self.0, rhs.0);
            let high = mul_u16_keep_high_m128i(self.0, rhs.0);
            u32x8(crate::u32x8_backend::Inner {
              lo: u32x4(unpack_low_i16_m128i(low, high)),
              hi: u32x4(unpack_high_i16_m128i(low, high))
            })
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))] {
            let lhs_low = unsafe { vget_low_u16(self.0) };
            let rhs_low = unsafe { vget_low_u16(rhs.0) };

            let lhs_high = unsafe { vget_high_u16(self.0) };
            let rhs_high = unsafe { vget_high_u16(rhs.0) };

            let low = unsafe { vmull_u16(lhs_low, rhs_low) };
            let high = unsafe { vmull_u16(lhs_high, rhs_high) };

            u32x8(crate::u32x8_backend::Inner { lo: u32x4(low), hi: u32x4(high) })
          } else {
            let a = self.as_array();
            let b = rhs.as_array();
            u32x8::from_array([
              u32::from(a[0]) * u32::from(b[0]),
              u32::from(a[1]) * u32::from(b[1]),
              u32::from(a[2]) * u32::from(b[2]),
              u32::from(a[3]) * u32::from(b[3]),
              u32::from(a[4]) * u32::from(b[4]),
              u32::from(a[5]) * u32::from(b[5]),
              u32::from(a[6]) * u32::from(b[6]),
              u32::from(a[7]) * u32::from(b[7]),
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn mul_keep_low_high_backend(self, rhs: Self) -> (Self, Self) {
        pick! {
          if #[cfg(target_feature="simd128")] {
            let low_wide_mul = u32x4_extmul_low_u16x8(self.0, rhs.0);
            let high_wide_mul = u32x4_extmul_high_u16x8(self.0, rhs.0);
            (
              Self(u16x8_shuffle::<0, 2, 4, 6, 8, 10, 12, 14>(low_wide_mul, high_wide_mul)),
              Self(u16x8_shuffle::<1, 3, 5, 7, 9, 11, 13, 15>(low_wide_mul, high_wide_mul)),
            )
          } else if #[cfg(all(target_feature="neon", target_arch="aarch64"))] {
            unsafe {
              let low_wide_mul = vreinterpretq_u16_u32(
                vmull_u16(vget_low_u16(self.0), vget_low_u16(rhs.0)),
              );
              let high_wide_mul = vreinterpretq_u16_u32(
                vmull_u16(vget_high_u16(self.0), vget_high_u16(rhs.0)),
              );
              let low_high = vuzpq_u16(low_wide_mul, high_wide_mul);
              (
                Self(low_high.0),
                Self(low_high.1),
              )
            }
          } else {
            // TODO(perf): This implementation looks quite bad. Is there a better
            // one?

            let self_array = self.to_array();
            let rhs_array = rhs.to_array();

            let widening_mul = [
              (self_array[0] as u32).wrapping_mul(rhs_array[0] as u32),
              (self_array[1] as u32).wrapping_mul(rhs_array[1] as u32),
              (self_array[2] as u32).wrapping_mul(rhs_array[2] as u32),
              (self_array[3] as u32).wrapping_mul(rhs_array[3] as u32),
              (self_array[4] as u32).wrapping_mul(rhs_array[4] as u32),
              (self_array[5] as u32).wrapping_mul(rhs_array[5] as u32),
              (self_array[6] as u32).wrapping_mul(rhs_array[6] as u32),
              (self_array[7] as u32).wrapping_mul(rhs_array[7] as u32),
            ];

            (
              Self::from_array([
                widening_mul[0] as u16,
                widening_mul[1] as u16,
                widening_mul[2] as u16,
                widening_mul[3] as u16,
                widening_mul[4] as u16,
                widening_mul[5] as u16,
                widening_mul[6] as u16,
                widening_mul[7] as u16,
              ]),
              Self::from_array([
                (widening_mul[0] >> 16) as u16,
                (widening_mul[1] >> 16) as u16,
                (widening_mul[2] >> 16) as u16,
                (widening_mul[3] >> 16) as u16,
                (widening_mul[4] >> 16) as u16,
                (widening_mul[5] >> 16) as u16,
                (widening_mul[6] >> 16) as u16,
                (widening_mul[7] >> 16) as u16,
              ]),
            )
          }
        }
    }

    #[inline]
    pub(crate) fn mul_keep_high_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse2")] {
            Self(mul_u16_keep_high_m128i(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))] {
            let lhs_low = unsafe { vget_low_u16(self.0) };
            let rhs_low = unsafe { vget_low_u16(rhs.0) };

            let lhs_high = unsafe { vget_high_u16(self.0) };
            let rhs_high = unsafe { vget_high_u16(rhs.0) };

            let low = unsafe { vmull_u16(lhs_low, rhs_low) };
            let high = unsafe { vmull_u16(lhs_high, rhs_high) };

            u16x8(unsafe { vuzpq_u16(vreinterpretq_u16_u32(low), vreinterpretq_u16_u32(high)).1 })
          } else if #[cfg(target_feature="simd128")] {
            let low =  u32x4_extmul_low_u16x8(self.0, rhs.0);
            let high = u32x4_extmul_high_u16x8(self.0, rhs.0);

            Self(u16x8_shuffle::<1, 3, 5, 7, 9, 11, 13, 15>(low, high))
          } else {
            u16x8::from_array([
              ((u32::from(rhs.as_array()[0]) * u32::from(self.as_array()[0])) >> 16) as u16,
              ((u32::from(rhs.as_array()[1]) * u32::from(self.as_array()[1])) >> 16) as u16,
              ((u32::from(rhs.as_array()[2]) * u32::from(self.as_array()[2])) >> 16) as u16,
              ((u32::from(rhs.as_array()[3]) * u32::from(self.as_array()[3])) >> 16) as u16,
              ((u32::from(rhs.as_array()[4]) * u32::from(self.as_array()[4])) >> 16) as u16,
              ((u32::from(rhs.as_array()[5]) * u32::from(self.as_array()[5])) >> 16) as u16,
              ((u32::from(rhs.as_array()[6]) * u32::from(self.as_array()[6])) >> 16) as u16,
              ((u32::from(rhs.as_array()[7]) * u32::from(self.as_array()[7])) >> 16) as u16,
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn from_u8x16_low_backend(u: u8x16) -> Self {
        pick! {
          if #[cfg(target_feature="sse2")] {
            Self(unpack_low_i8_m128i(u.0, m128i::zeroed()))
          } else {
            let u_arr: [u8; 16] = cast(u);
            cast([
              u_arr[0] as u16,
              u_arr[1] as u16,
              u_arr[2] as u16,
              u_arr[3] as u16,
              u_arr[4] as u16,
              u_arr[5] as u16,
              u_arr[6] as u16,
              u_arr[7] as u16,
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn from_u8x16_high_backend(u: u8x16) -> Self {
        pick! {
          if #[cfg(target_feature="sse2")] {
            Self(unpack_high_i8_m128i(u.0, m128i::zeroed()))
          } else {
            let u_arr: [u8; 16] = cast(u);
            cast([
              u_arr[8] as u16,
              u_arr[9] as u16,
              u_arr[10] as u16,
              u_arr[11] as u16,
              u_arr[12] as u16,
              u_arr[13] as u16,
              u_arr[14] as u16,
              u_arr[15] as u16,
            ])
          }
        }
    }

    /// A helper for shuffle functions that turns indices of 16-bit lanes into
    /// byte indices that can be used with 8-bit shuffle intrinsics.
    ///
    /// This turns each 16-bit lane `i` into two 8-bit lanes `[2*i, 2*i + 1]`.
    ///
    /// This assumes `self` has already been reduced to the table's lane count,
    /// which may be at most 128 lanes so that `2 * i` still fits in a byte.
    #[allow(dead_code)]
    #[inline]
    fn to_byte_indices(self) -> u8x16 {
        // The byte offset of the lane, broadcast to every byte of the lane.
        let base = self.unbounded_shl_scalar(1);
        let base = base | base.unbounded_shl_scalar(8);

        // Then the offset of each byte within its lane. These bits are free because
        // every byte of `base` is a multiple of two. `from_ne_bytes` keeps this
        // correct on big endian, where the bytes of a lane are the other way around.
        const WITHIN_LANE: u16x8 = u16x8::splat(u16::from_ne_bytes([0, 1]));

        cast::<u16x8, u8x16>(base | WITHIN_LANE)
    }
}
