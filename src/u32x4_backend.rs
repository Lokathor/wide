use crate::backend_imports::*;

#[cfg(not(any(
    target_feature = "sse2",
    target_feature = "simd128",
    all(target_feature = "neon", target_arch = "aarch64")
)))]
#[repr(C, align(16))]
#[derive(Clone, Copy)]
pub(crate) struct Inner {
    pub arr: [u32; 4],
}

safe_target_feature! {
impl SimdBackend for u32x4 {
    pick! {
      if #[cfg(target_feature="sse2")] {
        type Inner = __m128i;
      } else if #[cfg(target_feature="simd128")] {
        type Inner = v128;
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        type Inner = uint32x4_t;
      } else {
        type Inner = Inner;
      }
    }

    #[inline]
    pub(crate) fn simd_eq_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse2")] {
            Self(_mm_cmpeq_epi32(self.0, rhs.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(u32x4_eq(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vceqq_u32(self.0, rhs.0))
          } else {
            Self::from_array([
              if self.0.arr[0] == rhs.0.arr[0] { u32::MAX } else { 0 },
              if self.0.arr[1] == rhs.0.arr[1] { u32::MAX } else { 0 },
              if self.0.arr[2] == rhs.0.arr[2] { u32::MAX } else { 0 },
              if self.0.arr[3] == rhs.0.arr[3] { u32::MAX } else { 0 },
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
            Self(u32x4_ne(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            !self.simd_eq(rhs)
          } else {
            Self::from_array([
              if self.0.arr[0] != rhs.0.arr[0] { u32::MAX } else { 0 },
              if self.0.arr[1] != rhs.0.arr[1] { u32::MAX } else { 0 },
              if self.0.arr[2] != rhs.0.arr[2] { u32::MAX } else { 0 },
              if self.0.arr[3] != rhs.0.arr[3] { u32::MAX } else { 0 },
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn simd_lt_backend(self, rhs: Self) -> Self {
        // lt is just gt the other way around
        rhs.simd_gt(self)
    }

    #[inline]
    pub(crate) fn simd_gt_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse2")] {
            // no unsigned less than so inverting the high bit will get the correct result
            let h = u32x4::splat(1 << 31);
            Self(_mm_cmpgt_epi32((self ^ h).0, (rhs ^ h).0))
          } else if #[cfg(target_feature="simd128")] {
            Self(u32x4_gt(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))] {
            Self(vcgtq_u32(self.0, rhs.0))
          } else {
            Self::from_array([
              if self.0.arr[0] > rhs.0.arr[0] { u32::MAX } else { 0 },
              if self.0.arr[1] > rhs.0.arr[1] { u32::MAX } else { 0 },
              if self.0.arr[2] > rhs.0.arr[2] { u32::MAX } else { 0 },
              if self.0.arr[3] > rhs.0.arr[3] { u32::MAX } else { 0 },
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
            Self(u32x4_le(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            !self.simd_gt(rhs)
          } else {
            Self::from_array([
              if self.0.arr[0] <= rhs.0.arr[0] { u32::MAX } else { 0 },
              if self.0.arr[1] <= rhs.0.arr[1] { u32::MAX } else { 0 },
              if self.0.arr[2] <= rhs.0.arr[2] { u32::MAX } else { 0 },
              if self.0.arr[3] <= rhs.0.arr[3] { u32::MAX } else { 0 },
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
            Self(u32x4_ge(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            !self.simd_lt(rhs)
          } else {
            Self::from_array([
              if self.0.arr[0] >= rhs.0.arr[0] { u32::MAX } else { 0 },
              if self.0.arr[1] >= rhs.0.arr[1] { u32::MAX } else { 0 },
              if self.0.arr[2] >= rhs.0.arr[2] { u32::MAX } else { 0 },
              if self.0.arr[3] >= rhs.0.arr[3] { u32::MAX } else { 0 },
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
            Self(vbslq_u32(self.0, if_true.0, if_false.0))
          } else {
            generic_bit_blend(self, if_true, if_false)
          }
        }
    }

    #[inline]
    pub(crate) fn to_bitmask_backend(self) -> u32 {
        pick! {
          if #[cfg(target_feature="sse2")] {
            // use f32 move_mask since it is the same size as i32
            _mm_movemask_ps(cast(self.0)) as u32
          } else if #[cfg(target_feature="simd128")] {
            u32x4_bitmask(self.0) as u32
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            // set all to 1 if top bit is set, else 0
            let masked = vcltq_s32(self.cast_signed().0, vdupq_n_s32(0));

            // select the right bit out of each lane
            let selectbit = Self::from_array([1, 2, 4, 8]).0;
            let r = vandq_u32(masked, selectbit);

            // horizontally add the 32-bit lanes
            vaddvq_u32(r) as u32
          } else {
            self.0.arr[0] >> 31 |
            self.0.arr[1] >> 31 << 1 |
            self.0.arr[2] >> 31 << 2 |
            self.0.arr[3] >> 31 << 3
          }
        }
    }

    #[inline]
    pub(crate) fn any_backend(self) -> bool {
        pick! {
          if #[cfg(target_feature="sse2")] {
            // use f32 move_mask since it is the same size as i32
            _mm_movemask_ps(cast(self.0)) != 0
          } else if #[cfg(target_feature="simd128")] {
            u32x4_bitmask(self.0) != 0
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))] {
            // some lanes are negative
            vminvq_s32(self.cast_signed().0) < 0
          } else {
            let v : [u64;2] = cast(self);
            ((v[0] | v[1]) & 0x8000000080000000) != 0
          }
        }
    }

    #[inline]
    pub(crate) fn all_backend(self) -> bool {
        pick! {
          if #[cfg(target_feature="sse2")] {
            // use f32 move_mask since it is the same size as i32
            _mm_movemask_ps(cast(self.0)) == 0b1111
          } else if #[cfg(target_feature="simd128")] {
            u32x4_bitmask(self.0) == 0b1111
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            // all lanes are negative
            vmaxvq_s32(self.cast_signed().0) < 0
          } else {
            let v : [u64;2] = cast(self);
            (v[0] & v[1] & 0x8000000080000000) == 0x8000000080000000
          }
        }
    }

    #[inline]
    pub(crate) fn interleave_low_backend(self, other: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse2")] {
            Self(_mm_unpacklo_epi32(self.0, other.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(u32x4_shuffle::<0, 4, 1, 5>(self.0, other.0))
          } else if #[cfg(all(target_feature="neon", target_arch="aarch64"))] {
            Self(vzip1q_u32(self.0, other.0))
          } else {
            let s = self.as_array();
            let b = other.as_array();
            Self::from_array([s[0], b[0], s[1], b[1]])
          }
        }
    }

    #[inline]
    pub(crate) fn interleave_high_backend(self, other: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse2")] {
            Self(_mm_unpackhi_epi32(self.0, other.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(u32x4_shuffle::<2, 6, 3, 7>(self.0, other.0))
          } else if #[cfg(all(target_feature="neon", target_arch="aarch64"))] {
            Self(vzip2q_u32(self.0, other.0))
          } else {
            let s = self.as_array();
            let b = other.as_array();
            Self::from_array([s[2], b[2], s[3], b[3]])
          }
        }
    }

    #[inline]
    pub(crate) fn shuffle_backend(self, indices: u32x4) -> Self {
        pick! {
          if #[cfg(target_feature = "avx")] {
            cast(_mm_permutevar_ps(cast(self), indices.0))
          } else if #[cfg(any(
            target_feature = "ssse3",
            all(target_arch = "aarch64", target_feature = "neon"),
            target_feature = "simd128",
          ))] {
            let self_bytes = cast::<u32x4, u8x16>(self);
            let byte_indices = indices.to_byte_indices();

            cast::<u8x16, u32x4>(self_bytes.shuffle(byte_indices))
          } else {
            let self_array = self.to_array();
            let indices_array = indices.to_array();

            let mut result = [0; 4];
            for i in 0..4 {
              let index = indices_array[i] as usize;
              if index < 4 {
                result[i] = self_array[index];
              }
            }

            Self::from_array(result)
          }
        }
    }

    #[inline]
    pub(crate) fn shuffle_zeroing_backend(self, indices: u32x4) -> Self {
        pick! {
          if #[cfg(any(
            target_feature = "avx",
            target_feature = "ssse3",
            all(target_arch = "aarch64", target_feature = "neon"),
            target_feature = "simd128",
          ))] {
            // Even if the `u8x16::shuffle` implementation is zeroing, our 32-bit to
            // 8-bit can trigger an overflow causing incorrect behavior
            self.shuffle(indices) & indices.simd_lt(4)
          } else {
            // The fallback branch of `shuffle` already has the behavior we want
            self.shuffle(indices)
          }
        }
    }

    #[inline]
    pub(crate) fn shuffle_wrapping_backend(self, indices: u32x4) -> Self {
        self.shuffle(indices & 3)
    }

    #[inline]
    pub(crate) fn shuffle_2_backend(inputs: [Self; 2], indices: u32x4) -> u32x4 {
        pick! {
          if #[cfg(all(target_feature = "avx512f", target_feature = "avx512vl"))] {
            u32x4(_mm_permutex2var_epi32(inputs[0].0, indices.0, inputs[1].0))
          } else {
            let self_bytes = cast::<[u32x4; 2], [u8x16; 2]>(inputs);
            let byte_indices = indices.to_byte_indices();

            cast::<u8x16, u32x4>(self_bytes.shuffle(byte_indices))
          }
        }
    }

    #[inline]
    pub(crate) fn shuffle_zeroing_2_backend(inputs: [Self; 2], indices: u32x4) -> u32x4 {
        // Even if the `u8x16::shuffle` implementation is zeroing, our 32-bit to
        // 8-bit can trigger an overflow causing incorrect behavior
        inputs.shuffle(indices) & indices.simd_lt(8)
    }

    #[inline]
    pub(crate) fn shuffle_wrapping_2_backend(inputs: [Self; 2], indices: u32x4) -> u32x4 {
        pick! {
          if #[cfg(all(target_feature = "avx512f", target_feature = "avx512vl"))] {
            // `avx512` shuffle intrinsics are wrapping
            inputs.shuffle(indices)
          } else {
            inputs.shuffle(indices & 7)
          }
        }
    }

    #[inline]
    pub(crate) fn shuffle_3_backend(inputs: [Self; 3], indices: u32x4) -> u32x4 {
        let self_bytes = cast::<[u32x4; 3], [u8x16; 3]>(inputs);
        let byte_indices = indices.to_byte_indices();

        cast::<u8x16, u32x4>(self_bytes.shuffle(byte_indices))
    }

    #[inline]
    pub(crate) fn shuffle_zeroing_3_backend(inputs: [Self; 3], indices: u32x4) -> u32x4 {
        // Even if the `u8x16::shuffle` implementation is zeroing, our 32-bit to
        // 8-bit can trigger an overflow causing incorrect behavior
        inputs.shuffle(indices) & indices.simd_lt(12)
    }

    #[inline]
    pub(crate) fn shuffle_wrapping_3_backend(inputs: [Self; 3], indices: u32x4) -> u32x4 {
        inputs.shuffle(indices % 12)
    }

    #[inline]
    pub(crate) fn shuffle_4_backend(inputs: [Self; 4], indices: u32x4) -> u32x4 {
        let self_bytes = cast::<[u32x4; 4], [u8x16; 4]>(inputs);
        let byte_indices = indices.to_byte_indices();

        cast::<u8x16, u32x4>(self_bytes.shuffle(byte_indices))
    }

    #[inline]
    pub(crate) fn shuffle_zeroing_4_backend(inputs: [Self; 4], indices: u32x4) -> u32x4 {
        // Even if the `u8x16::shuffle` implementation is zeroing, our 32-bit to
        // 8-bit can trigger an overflow causing incorrect behavior
        inputs.shuffle(indices) & indices.simd_lt(16)
    }

    #[inline]
    pub(crate) fn shuffle_wrapping_4_backend(inputs: [Self; 4], indices: u32x4) -> u32x4 {
        inputs.shuffle(indices & 15)
    }

    #[inline]
    pub(crate) fn transpose_backend(data: [Self; 4]) -> [Self; 4] {
        pick! {
          if #[cfg(target_feature="sse")] {
            let mut e0 = data[0];
            let mut e1 = data[1];
            let mut e2 = data[2];
            let mut e3 = data[3];

            _MM_TRANSPOSE4_PS(
              cast_mut(&mut e0.0),
              cast_mut(&mut e1.0),
              cast_mut(&mut e2.0),
              cast_mut(&mut e3.0),
            );

            [e0, e1, e2, e3]
          } else {
            #[inline(always)]
            fn transpose_column(data: &[u32x4; 4], index: usize) -> u32x4 {
              u32x4::from_array([
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

    #[inline]
    pub(crate) fn replace_const_backend<const INDEX: usize>(self, value: u32) -> Self {
        /// Because of limitations in const generics, this must have separate
        /// branches per index. This macro is used to avoid duplication.
        macro_rules! use_indices {
      (0..4 = [$($INDEX:literal),*]) => {
        pick! {
          if #[cfg(target_feature = "sse4.1")] {
            match INDEX {
              $($INDEX => Self(_mm_insert_epi32::<$INDEX>(self.0, value as i32)),)*
              4.. => unreachable!()
            }
          } else if #[cfg(target_feature = "sse2")] {
            match INDEX {
              $($INDEX => {
                const INDEX_A: i32 = $INDEX * 2;
                const INDEX_B: i32 = $INDEX * 2 + 1;

                let [value_low, value_high] = [value, value >> 16];

                let replace_a = _mm_insert_epi16::<INDEX_A>(self.0, value_low.cast_signed());
                let replace_b = _mm_insert_epi16::<INDEX_B>(replace_a, value_high.cast_signed());
                Self(replace_b)
              })*
              4.. => unreachable!(),
            }
          } else if #[cfg(all(
            target_arch = "aarch64",
            target_feature = "neon",
            target_endian = "little",
          ))] {
            match INDEX {
              $($INDEX => Self(vsetq_lane_u32::<$INDEX>(value, self.0)),)*
              4.. => unreachable!(),
            }
          } else if #[cfg(target_feature = "simd128")] {
            Self(u32x4_replace_lane::<INDEX>(self.0, value))
          } else {
            let mut result = self;
            result.as_mut_array()[INDEX] = value;
            result
          }
        }
      };
    }
        use_indices! { 0..4 = [0, 1, 2, 3] }
    }

    #[inline]
    pub(crate) fn extract_const_backend<const INDEX: usize>(self) -> u32 {
        /// Because of limitations in const generics, this must have separate
        /// branches per index. This macro is used to avoid duplication.
        macro_rules! use_indices {
      (0..4 = [$($INDEX:literal),*], 1..4 = [$($INDEX_FROM_1:literal),*]) => {
        pick! {
          if #[cfg(target_feature = "sse4.1")] {
            match INDEX {
              $($INDEX => _mm_extract_epi32::<$INDEX>(self.0).cast_unsigned(),)*
              4.. => unreachable!()
            }
          } else if #[cfg(all(target_feature = "sse2"))] {
            _mm_cvtsi128_si32(self.shuffle_consts::<INDEX, 0, 0, 0>().0).cast_unsigned()
          } else if #[cfg(all(
            target_arch = "aarch64",
            target_feature = "neon",
            target_endian = "little",
          ))] {
            match INDEX {
              $($INDEX => vgetq_lane_u32::<$INDEX>(self.0),)*
              4.. => unreachable!(),
            }
          } else if #[cfg(target_feature = "simd128")] {
            u32x4_extract_lane::<INDEX>(self.0)
          } else {
            self.to_array()[INDEX]
          }
        }
      };
    }
        use_indices! { 0..4 = [0, 1, 2, 3], 1..4 = [1, 2, 3] }
    }
}
}

safe_target_feature! {
// Backend of SIMD integer functions.
impl u32x4 {
    #[inline]
    pub(crate) fn neg_backend(self) -> Self {
        pick! {
            if #[cfg(all(target_arch = "aarch64", target_feature = "neon"))] {
                Self(vreinterpretq_u32_s32(vnegq_s32(vreinterpretq_s32_u32(self.0))))
            } else {
                Self::default() - self
            }
        }
    }

    #[inline]
    pub(crate) fn not_backend(self) -> Self {
        self ^ cast::<u128, u32x4>(u128::MAX)
    }

    #[inline]
    pub(crate) fn add_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse2")] {
            Self(_mm_add_epi32(self.0, rhs.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(u32x4_add(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vaddq_u32(self.0, rhs.0))
          } else {
            Self::from_array([
              self.0.arr[0].wrapping_add(rhs.0.arr[0]),
              self.0.arr[1].wrapping_add(rhs.0.arr[1]),
              self.0.arr[2].wrapping_add(rhs.0.arr[2]),
              self.0.arr[3].wrapping_add(rhs.0.arr[3]),
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn sub_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse2")] {
            Self(_mm_sub_epi32(self.0, rhs.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(u32x4_sub(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vsubq_u32(self.0, rhs.0))
          } else {
            Self::from_array([
              self.0.arr[0].wrapping_sub(rhs.0.arr[0]),
              self.0.arr[1].wrapping_sub(rhs.0.arr[1]),
              self.0.arr[2].wrapping_sub(rhs.0.arr[2]),
              self.0.arr[3].wrapping_sub(rhs.0.arr[3]),
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn mul_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse4.1")] {
            Self(_mm_mullo_epi32(self.0, rhs.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(u32x4_mul(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vmulq_u32(self.0, rhs.0))
          } else {
            let arr1: [u32; 4] = cast(self);
            let arr2: [u32; 4] = cast(rhs);
            cast([
              arr1[0].wrapping_mul(arr2[0]),
              arr1[1].wrapping_mul(arr2[1]),
              arr1[2].wrapping_mul(arr2[2]),
              arr1[3].wrapping_mul(arr2[3]),
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
            Self(vandq_u32(self.0, rhs.0))
          } else {
            Self::from_array([
              self.0.arr[0] & rhs.0.arr[0],
              self.0.arr[1] & rhs.0.arr[1],
              self.0.arr[2] & rhs.0.arr[2],
              self.0.arr[3] & rhs.0.arr[3],
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
            Self(vorrq_u32(self.0, rhs.0))
          } else {
            Self::from_array([
              self.0.arr[0] | rhs.0.arr[0],
              self.0.arr[1] | rhs.0.arr[1],
              self.0.arr[2] | rhs.0.arr[2],
              self.0.arr[3] | rhs.0.arr[3],
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
            Self(veorq_u32(self.0, rhs.0))
          } else {
            Self::from_array([
              self.0.arr[0] ^ rhs.0.arr[0],
              self.0.arr[1] ^ rhs.0.arr[1],
              self.0.arr[2] ^ rhs.0.arr[2],
              self.0.arr[3] ^ rhs.0.arr[3],
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn reduce_add_backend(self) -> u32 {
        pick! {
          if #[cfg(target_feature="sse2")] {
            let hi64  = _mm_unpackhi_epi64(self.0, self.0);
            let sum64 = _mm_add_epi32(hi64, self.0);
            let hi32  = _mm_shuffle_epi32::<0b10_11_00_01>(sum64);    // Swap the low two elements
            let sum32 = _mm_add_epi32(sum64, hi32);
            _mm_cvtsi128_si32(sum32).cast_unsigned()
          } else {
            let arr: [u32; 4] = cast(self);
            arr[0].wrapping_add(arr[1]).wrapping_add(
            arr[2].wrapping_add(arr[3]))
          }
        }
    }

    #[inline]
    pub(crate) fn reduce_mul_backend(self) -> u32 {
        pick! {
          if #[cfg(target_feature="sse4.1")] {
            let high_64  = _mm_unpackhi_epi64(self.0, self.0);
            let reduce_64 = _mm_mullo_epi32(high_64, self.0);
            let high_32  = _mm_shuffle_epi32::<0b10_11_00_01>(reduce_64);
            let reduce_32 = _mm_mullo_epi32(reduce_64, high_32);
            _mm_cvtsi128_si32(reduce_32).cast_unsigned()
          } else if #[cfg(target_feature="simd128")] {
            let high_64 = u64x2_shuffle::<1, 0>(self.0, self.0);
            let reduce_64 = u32x4_mul(self.0, high_64);
            let high_32 = u32x4_shuffle::<1, 0, 0, 0>(reduce_64, reduce_64);
            let reduce_32 = u32x4_mul(reduce_64, high_32);
            u32x4_extract_lane::<0>(reduce_32)
          } else if #[cfg(all(target_feature="neon", target_arch="aarch64"))] {
            let high_64 = vextq_u32::<2>(self.0, self.0);
            let reduce_64 = vmulq_u32(self.0, high_64);
            let high_32 = vrev64q_u32(reduce_64);
            let reduce_32 = vmulq_u32(reduce_64, high_32);
            vgetq_lane_u32::<0>(reduce_32)
          } else {
            let array = self.to_array();
            array[0].wrapping_mul(array[1]).wrapping_mul(array[2].wrapping_mul(array[3]))
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
            Self(vbslq_u32(self.0, if_one.0, if_zero.0))
          } else {
            generic_bit_blend(self, if_one, if_zero)
          }
        }
    }

    #[inline]
    pub(crate) fn shl_backend(self, rhs: u32x4) -> Self {
        pick! {
          if #[cfg(target_feature="avx2")] {
            // mask the shift count to 31 to have same behavior on all platforms
            let shift_by = _mm_and_si128(rhs.0, _mm_set1_epi32(31));
            Self(_mm_sllv_epi32(self.0, shift_by))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            // mask the shift count to 31 to have same behavior on all platforms
            let shift_by = vreinterpretq_s32_u32(vandq_u32(rhs.0, vmovq_n_u32(31)));
            Self(vshlq_u32(self.0, shift_by))
          } else {
            let arr: [u32; 4] = cast(self);
            let rhs: [u32; 4] = cast(rhs);
            cast([
              arr[0].wrapping_shl(rhs[0]),
              arr[1].wrapping_shl(rhs[1]),
              arr[2].wrapping_shl(rhs[2]),
              arr[3].wrapping_shl(rhs[3]),
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn shl_scalar_backend(self, rhs: u32) -> Self {
        pick! {
          if #[cfg(target_feature="sse2")] {
            // Use `rhs % 32` to perform wrapping shift and not unbounded shift.
            let shift = cast([rhs as u64 & 31, 0]);
            Self(_mm_sll_epi32(self.0, shift))
          } else if #[cfg(target_feature="simd128")] {
            Self(u32x4_shl(self.0, rhs))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            // Use `rhs % 32` to perform wrapping shift and not unbounded shift.
            Self(vshlq_u32(self.0, vmovq_n_s32(rhs as i32 & 31)))
          } else {
            Self::from_array([
              self.0.arr[0].wrapping_shl(rhs),
              self.0.arr[1].wrapping_shl(rhs),
              self.0.arr[2].wrapping_shl(rhs),
              self.0.arr[3].wrapping_shl(rhs),
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn shr_backend(self, rhs: u32x4) -> Self {
        pick! {
          if #[cfg(target_feature="avx2")] {
            // mask the shift count to 31 to have same behavior on all platforms
            let shift_by = _mm_and_si128(rhs.0, _mm_set1_epi32(31));
            Self(_mm_srlv_epi32(self.0, shift_by))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            // mask the shift count to 31 to have same behavior on all platforms
            // no right shift, have to pass negative value to left shift on neon
            let shift_by = vnegq_s32(vreinterpretq_s32_u32(vandq_u32(rhs.0, vmovq_n_u32(31))));
            Self(vshlq_u32(self.0, shift_by))
          } else {
            let arr: [u32; 4] = cast(self);
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
            Self(_mm_srl_epi32(self.0, shift))
          } else if #[cfg(target_feature="simd128")] {
            Self(u32x4_shr(self.0, rhs))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            // Use `rhs % 32` to perform wrapping shift and not unbounded shift.
            Self(vshlq_u32(self.0, vmovq_n_s32( -(rhs as i32 & 31))))
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
            Self(_mm_max_epu32(self.0, rhs.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(u32x4_max(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vmaxq_u32(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vmaxq_u16(self.0, rhs.0))
          } else {
            let arr: [u32; 4] = cast(self);
            let rhs: [u32; 4] = cast(rhs);
            cast([
              arr[0].max(rhs[0]),
              arr[1].max(rhs[1]),
              arr[2].max(rhs[2]),
              arr[3].max(rhs[3]),
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn min_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse4.1")] {
            Self(_mm_min_epu32(self.0, rhs.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(u32x4_min(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vminq_u32(self.0, rhs.0))
          } else {
            let arr: [u32; 4] = cast(self);
            let rhs: [u32; 4] = cast(rhs);
            cast([
              arr[0].min(rhs[0]),
              arr[1].min(rhs[1]),
              arr[2].min(rhs[2]),
              arr[3].min(rhs[3]),
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn reduce_max_backend(self) -> u32 {
        let arr: [u32; 4] = cast(self);
        arr[0].max(arr[1]).max(arr[2].max(arr[3]))
    }

    #[inline]
    pub(crate) fn reduce_min_backend(self) -> u32 {
        let arr: [u32; 4] = cast(self);
        arr[0].min(arr[1]).min(arr[2].min(arr[3]))
    }

    #[inline]
    pub(crate) fn unbounded_shl_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="avx2")] {
            Self(_mm_sllv_epi32(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            // The intrinsic has different semantics so we need to mask ourselves.
            Self(vshlq_u32(self.0, vreinterpretq_s32_u32(rhs.0))) & rhs.simd_lt(32)
          } else {
            let self_array = self.to_array();
            let rhs_array = rhs.to_array();

            cast([
              self_array[0].unbounded_shl(rhs_array[0]),
              self_array[1].unbounded_shl(rhs_array[1]),
              self_array[2].unbounded_shl(rhs_array[2]),
              self_array[3].unbounded_shl(rhs_array[3]),
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn unbounded_shl_scalar_backend(self, rhs: u32) -> Self {
        pick! {
          if #[cfg(target_feature="sse2")] {
            Self(_mm_sll_epi32(self.0, cast([rhs as u64, 0])))
          } else if #[cfg(target_feature="simd128")] {
            // The intrinsic performs wrapping shift so we need to mask the result.
            Self(u32x4_shl(self.0, rhs)) & Self::splat(rhs).simd_lt(32)
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            // The intrinsic has different semantics so we need to saturate `rhs`.
            Self(vshlq_u32(self.0, vmovq_n_s32(rhs.min(32) as i32)))
          } else {
            Self::from_array([
              self.0.arr[0].unbounded_shl(rhs),
              self.0.arr[1].unbounded_shl(rhs),
              self.0.arr[2].unbounded_shl(rhs),
              self.0.arr[3].unbounded_shl(rhs),
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn unbounded_shr_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="avx2")] {
            Self(_mm_srlv_epi32(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            // Negate `rhs` because there is no direct shift-right intrinsic, and
            // mask to hide `rhs` overflow.
            Self(vshlq_u32(self.0, vnegq_s32(vreinterpretq_s32_u32(rhs.0)))) & rhs.simd_lt(32)
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
            Self(_mm_srl_epi32(self.0, cast([rhs as u64, 0])))
          } else if #[cfg(target_feature="simd128")] {
            if rhs < 32 { Self(u32x4_shr(self.0, rhs)) } else { Self::ZERO }
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            // Negate `rhs` because there is no direct shift-right intrinsic, and
            // restrict it to prevent overflow.
            Self(vshlq_u32(self.0, vmovq_n_s32(-rhs.min(32).cast_signed())))
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
            let overflow = result.simd_lt(self);
            // Return `MAX` (all bits set) if overflow occurs.
            result | overflow
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vqaddq_u32(self.0, rhs.0))
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
            let no_overflow = result.simd_le(self);
            // Return `0` (no bits set) if overflow occurs.
            result & no_overflow
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vqsubq_u32(self.0, rhs.0))
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
        let overflow = high.simd_ne(Self::ZERO);
        (low, overflow)
    }

    #[inline]
    pub(crate) fn widening_mul_backend(self, rhs: Self) -> u64x4 {
        pick! {
          if #[cfg(target_feature="avx2")] {
            // ok to sign extend since we are throwing away the high half of the result anyway
            let a = _mm256_cvtepi32_epi64(self.0);
            let b = _mm256_cvtepi32_epi64(rhs.0);
            cast(_mm256_mul_epu32(a, b))
          } else if #[cfg(target_feature="sse2")] {
            let evenp = _mm_mul_epu32(self.0, rhs.0);

            let oddp = _mm_mul_epu32(
              _mm_srli_epi64::<32>(self.0),
              _mm_srli_epi64::<32>(rhs.0));

            u64x4(crate::u64x4_backend::Inner {
              lo: u64x2(_mm_unpacklo_epi64(evenp, oddp)),
              hi: u64x2(_mm_unpackhi_epi64(evenp, oddp))
            })
          } else if #[cfg(target_feature="simd128")] {
            u64x4(crate::u64x4_backend::Inner {
              lo: u64x2(u64x2_extmul_low_u32x4(self.0, rhs.0)),
              hi: u64x2(u64x2_extmul_high_u32x4(self.0, rhs.0)),
            })
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))] {
            u64x4(crate::u64x4_backend::Inner {
              lo: u64x2(vmull_u32(vget_low_u32(self.0), vget_low_u32(rhs.0))),
              hi: u64x2(vmull_u32(vget_high_u32(self.0), vget_high_u32(rhs.0))),
            })
          } else {
            let a: [u32; 4] = cast(self);
            let b: [u32; 4] = cast(rhs);
            cast([
              u64::from(a[0]) * u64::from(b[0]),
              u64::from(a[1]) * u64::from(b[1]),
              u64::from(a[2]) * u64::from(b[2]),
              u64::from(a[3]) * u64::from(b[3]),
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn mul_keep_low_high_backend(self, rhs: Self) -> (Self, Self) {
        pick! {
          if #[cfg(target_feature="sse4.1")] {
            let even_wide_mul = _mm_mul_epu32(self.0, rhs.0);
            let odd_wide_mul = _mm_mul_epu32(
              _mm_shuffle_epi32::<0b_00_11_00_01>(self.0),
              _mm_shuffle_epi32::<0b_00_11_00_01>(rhs.0),
            );

            let ll_hh_1 = _mm_unpacklo_epi32(even_wide_mul, odd_wide_mul);
            let ll_hh_2 = _mm_unpackhi_epi32(even_wide_mul, odd_wide_mul);
            (
              Self(_mm_unpacklo_epi64(ll_hh_1, ll_hh_2)),
              Self(_mm_unpackhi_epi64(ll_hh_1, ll_hh_2)),
            )
          } else if #[cfg(target_feature="simd128")] {
            let low_wide_mul = u64x2_extmul_low_u32x4(self.0, rhs.0);
            let high_wide_mul = u64x2_extmul_high_u32x4(self.0, rhs.0);
            (
              Self(u32x4_shuffle::<0, 2, 4, 6>(low_wide_mul, high_wide_mul)),
              Self(u32x4_shuffle::<1, 3, 5, 7>(low_wide_mul, high_wide_mul)),
            )
          } else if #[cfg(all(target_feature="neon", target_arch="aarch64"))] {
            let low_wide_mul = vreinterpretq_u32_u64(
              vmull_u32(vget_low_u32(self.0), vget_low_u32(rhs.0)),
            );
            let high_wide_mul = vreinterpretq_u32_u64(
              vmull_u32(vget_high_u32(self.0), vget_high_u32(rhs.0)),
            );
            let low_high = vuzpq_u32(low_wide_mul, high_wide_mul);
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
              (self_array[0] as u64).wrapping_mul(rhs_array[0] as u64),
              (self_array[1] as u64).wrapping_mul(rhs_array[1] as u64),
              (self_array[2] as u64).wrapping_mul(rhs_array[2] as u64),
              (self_array[3] as u64).wrapping_mul(rhs_array[3] as u64),
            ];

            (
              Self::from_array([
                widening_mul[0] as u32,
                widening_mul[1] as u32,
                widening_mul[2] as u32,
                widening_mul[3] as u32,
              ]),
              Self::from_array([
                (widening_mul[0] >> 32) as u32,
                (widening_mul[1] >> 32) as u32,
                (widening_mul[2] >> 32) as u32,
                (widening_mul[3] >> 32) as u32,
              ]),
            )
          }
        }
    }

    #[inline]
    pub(crate) fn mul_keep_high_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="avx2")] {
            let a = _mm256_cvtepu32_epi64(self.0);
            let b = _mm256_cvtepu32_epi64(rhs.0);
            let r = _mm256_mul_epu32(a, b);

            // the compiler does a good job shuffling the lanes around
            let b : [u32;8] = cast(r);
            cast([b[1],b[3],b[5],b[7]])
          } else if #[cfg(target_feature="sse2")] {
            let evenp = _mm_mul_epu32(self.0, rhs.0);

            let oddp = _mm_mul_epu32(
              _mm_srli_epi64::<32>(self.0),
              _mm_srli_epi64::<32>(rhs.0));

            // the compiler does a good job shuffling the lanes around
            let a : [u32;4]= cast(evenp);
            let b : [u32;4]= cast(oddp);
            cast([a[1],b[1],a[3],b[3]])

          } else if #[cfg(target_feature="simd128")] {
            let low =  u64x2_extmul_low_u32x4(self.0, rhs.0);
            let high = u64x2_extmul_high_u32x4(self.0, rhs.0);

            Self(u32x4_shuffle::<1, 3, 5, 7>(low, high))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))] {
            let l = vmull_u32(vget_low_u32(self.0), vget_low_u32(rhs.0));
            let h = vmull_u32(vget_high_u32(self.0), vget_high_u32(rhs.0));
            u32x4(vcombine_u32(vshrn_n_u64(l,32), vshrn_n_u64(h,32)))
          } else {
            let a: [u32; 4] = cast(self);
            let b: [u32; 4] = cast(rhs);
            cast([
              ((u64::from(a[0]) * u64::from(b[0])) >> 32) as u32,
              ((u64::from(a[1]) * u64::from(b[1])) >> 32) as u32,
              ((u64::from(a[2]) * u64::from(b[2])) >> 32) as u32,
              ((u64::from(a[3]) * u64::from(b[3])) >> 32) as u32,
            ])
          }
        }
    }
}
}

safe_target_feature! {
// Backend of SIMD unsigned integer functions.
impl u32x4 {
    #[inline]
    pub(crate) fn saturating_add_signed_backend(self, rhs: i32x4) -> Self {
        pick! {
            if #[cfg(all(target_feature = "neon", target_arch = "aarch64"))] {
                Self(vsqaddq_u32(self.0, rhs.0))
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
impl u32x4 {
    #[inline]
    pub(crate) fn shuffle_consts_backend<
        const I0: usize,
        const I1: usize,
        const I2: usize,
        const I3: usize,
    >(
        self,
    ) -> Self {
        if const { I0 == 0 && I1 == 1 && I2 == 2 && I3 == 3 } {
            self
        } else {
            pick! {
              if #[cfg(target_feature = "sse2")] {
                // This code path effectively calls `_mm_shuffle_epi32`, but must use
                // inline assembly instead of the intrinsic. This operation takes an
                // `i32` constant for indices, while we have a list of `usize`
                // constants. The intrinsic takes the constant as a const-generic
                // parameter, which does not let us perform the format conversion due
                // to language limitations. However, inline assembly does support
                // arbitrary `const` blocks for values in immediates.

                use core::arch::asm;

                let value: __m128i = self.0;
                let result: __m128i;

                // SAFETY:
                // - `pshufd` is guaranteed with `sse2`
                // - This operation is `pure` because it only depends on the input
                // - This operation is `nomem` because it only uses registers
                // - This operation is `nostack` because it only uses registers
                // - `result` always gets initialized
                unsafe {
                  asm!(
                    "pshufd {result}, {value}, {immediate}",
                    result = lateout(xmm_reg) result,
                    value = in(xmm_reg) value,
                    immediate = const {
                      let i0 = I0 as u32;
                      let i1 = (I1 as u32) << 2;
                      let i2 = (I2 as u32) << 4;
                      let i3 = (I3 as u32) << 6;

                      (i0 | i1 | i2 | i3).cast_signed()
                    },
                    options(pure, nomem, nostack),
                  );
                }

                Self(result)
              } else if #[cfg(all(
                target_feature = "neon",
                target_arch = "aarch64",
                target_endian = "little",
              ))] {
                // Special case for duplicate
                if const { I0 == 1 && I1 == 1 && I2 == 1 && I3 == 1 } {
                  Self(vdupq_laneq_u32::<1>(self.0))
                } else if const { I0 == 2 && I1 == 2 && I2 == 2 && I3 == 2 } {
                  Self(vdupq_laneq_u32::<2>(self.0))
                } else if const { I0 == 3 && I1 == 3 && I2 == 3 && I3 == 3 } {
                  Self(vdupq_laneq_u32::<3>(self.0))
                }
                // Special case for rotate
                else if const { I0 == 1 && I1 == 2 && I2 == 3 && I3 == 0 } {
                  Self(vextq_u32::<1>(self.0, self.0))
                } else if const { I0 == 2 && I1 == 3 && I2 == 0 && I3 == 1 } {
                  Self(vextq_u32::<2>(self.0, self.0))
                } else if const { I0 == 3 && I1 == 0 && I2 == 1 && I3 == 2 } {
                  Self(vextq_u32::<3>(self.0, self.0))
                }
                // Special case for zip
                else if const { I0 == 0 && I1 == 0 && I2 == 1 && I3 == 1 } {
                  Self(vzip1q_u32(self.0, self.0))
                } else if const { I0 == 2 && I1 == 2 && I2 == 3 && I3 == 3 } {
                  Self(vzip2q_u32(self.0, self.0))
                }
                // Special case for unzip
                else if const { I0 == 0 && I1 == 2 && I2 == 0 && I3 == 2 } {
                  Self(vuzp1q_u32(self.0, self.0))
                } else if const { I0 == 1 && I1 == 3 && I2 == 1 && I3 == 3 } {
                  Self(vuzp2q_u32(self.0, self.0))
                }
                // Special case for transpose
                else if const { I0 == 0 && I1 == 0 && I2 == 2 && I3 == 2 } {
                  Self(vtrn1q_u32(self.0, self.0))
                } else if const { I0 == 1 && I1 == 1 && I2 == 3 && I3 == 3 } {
                  Self(vtrn2q_u32(self.0, self.0))
                }
                // Special case for reverse
                else if const { I0 == 1 && I1 == 0 && I2 == 3 && I3 == 2 } {
                  Self(vrev64q_u32(self.0))
                }
                // Fallback
                else {
                  Self(vreinterpretq_u32_u8(vqtbl1q_u8(
                    vreinterpretq_u8_u32(self.0),
                    const {
                      u8x16::from_array([
                        I0 as u8 * 4,
                        I0 as u8 * 4 + 1,
                        I0 as u8 * 4 + 2,
                        I0 as u8 * 4 + 3,
                        I1 as u8 * 4,
                        I1 as u8 * 4 + 1,
                        I1 as u8 * 4 + 2,
                        I1 as u8 * 4 + 3,
                        I2 as u8 * 4,
                        I2 as u8 * 4 + 1,
                        I2 as u8 * 4 + 2,
                        I2 as u8 * 4 + 3,
                        I3 as u8 * 4,
                        I3 as u8 * 4 + 1,
                        I3 as u8 * 4 + 2,
                        I3 as u8 * 4 + 3,
                      ]).0
                    },
                  )))
                }
              } else if #[cfg(target_feature = "simd128")] {
                Self(u32x4_shuffle::<I0, I1, I2, I3>(self.0, self.0))
              } else {
                let self_array = self.to_array();
                Self::from_array([
                  self_array[I0],
                  self_array[I1],
                  self_array[I2],
                  self_array[I3],
                ])
              }
            }
        }
    }

    #[inline]
    pub(crate) fn add_mul_lo_backend<const W: u32>(self, a: Self, b: Self) -> Self {
        if W <= 16 {
            let mask = Self::splat(add_mul_operand_mask_u32::<W>());
            return self + (((a & mask) * (b & mask)) & mask);
        }

        let acc = self.to_array();
        let a = a.to_array();
        let b = b.to_array();
        Self::from_array(core::array::from_fn(|i| {
            add_mul_lo_lane_u32::<W>(acc[i], a[i], b[i])
        }))
    }

    #[inline]
    pub(crate) fn add_mul_hi_backend<const W: u32>(self, a: Self, b: Self) -> Self {
        // See `add_mul_lo`: the whole product is in the lane, so the high half is a
        // shift.
        if W <= 16 {
            let mask = Self::splat(add_mul_operand_mask_u32::<W>());
            return self + (((a & mask) * (b & mask)) >> W);
        }

        let acc = self.to_array();
        let a = a.to_array();
        let b = b.to_array();
        Self::from_array(core::array::from_fn(|i| {
            add_mul_hi_lane_u32::<W>(acc[i], a[i], b[i])
        }))
    }

    /// A helper for shuffle functions that turns indices of 32-bit lanes into
    /// byte indices that can be used with 8-bit shuffle intrinsics.
    ///
    /// This turns each 32-bit lane `i` into four 8-bit lanes
    /// `[4*i, 4*i + 1, 4*i + 2, 4*i + 3]`.
    ///
    /// This assumes `self` has already been reduced to the table's lane count,
    /// which may be at most 64 lanes so that `4 * i` still fits in a byte.
    #[allow(dead_code)]
    #[inline]
    fn to_byte_indices(self) -> u8x16 {
        // The byte offset of the lane, broadcast to every byte of the lane.
        let base = self.unbounded_shl_scalar(2);
        let base = base | base.unbounded_shl_scalar(8);
        let base = base | base.unbounded_shl_scalar(16);

        // Then the offset of each byte within its lane. These bits are free because
        // every byte of `base` is a multiple of four. `from_ne_bytes` keeps this
        // correct on big endian, where the bytes of a lane are the other way around.
        const WITHIN_LANE: u32x4 = u32x4::splat(u32::from_ne_bytes([0, 1, 2, 3]));

        cast::<u32x4, u8x16>(base | WITHIN_LANE)
    }
}
}
