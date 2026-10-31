use crate::backend_imports::*;

#[cfg(not(any(
    target_feature = "sse2",
    target_feature = "simd128",
    all(target_feature = "neon", target_arch = "aarch64")
)))]
#[repr(C, align(16))]
#[derive(Clone, Copy)]
pub(crate) struct Inner {
    pub arr: [u64; 2],
}

safe_target_feature! {
impl SimdBackend for u64x2 {
    pick! {
      if #[cfg(target_feature="sse2")] {
        type Inner = __m128i;
      } else if #[cfg(target_feature="simd128")] {
        type Inner = v128;
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        type Inner = uint64x2_t;
      } else {
        type Inner = Inner;
      }
    }

    #[inline]
    pub(crate) fn simd_eq_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse4.1")] {
            Self(_mm_cmpeq_epi64(self.0, rhs.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(u64x2_eq(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vceqq_u64(self.0, rhs.0))
          } else {
            let s: [u64;2] = cast(self);
            let r: [u64;2] = cast(rhs);
            cast([
              if s[0] == r[0] { -1_i64 } else { 0 },
              if s[1] == r[1] { -1_i64 } else { 0 },
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn simd_ne_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse4.1")] {
            !self.simd_eq(rhs)
          } else if #[cfg(target_feature="simd128")] {
            Self(u64x2_ne(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            !self.simd_eq(rhs)
          } else {
            let s: [u64;2] = cast(self);
            let r: [u64;2] = cast(rhs);
            cast([
              if s[0] != r[0] { -1_i64 } else { 0 },
              if s[1] != r[1] { -1_i64 } else { 0 },
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
          if #[cfg(target_feature="sse4.2")] {
            // no unsigned gt so inverting the high bit will get the correct result
            let highbit = u64x2::splat(1 << 63);
            Self(_mm_cmpgt_epi64((self ^ highbit).0, (rhs ^ highbit).0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vcgtq_u64(self.0, rhs.0))
          } else {
            // u64x2_gt on WASM is not a thing. https://github.com/WebAssembly/simd/pull/414
            let s: [u64;2] = cast(self);
            let r: [u64;2] = cast(rhs);
            cast([
              if s[0] > r[0] { u64::MAX } else { 0 },
              if s[1] > r[1] { u64::MAX } else { 0 },
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn simd_le_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse4.1")] {
            !self.simd_gt(rhs)
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            !self.simd_gt(rhs)
          } else {
            let s: [u64;2] = cast(self);
            let r: [u64;2] = cast(rhs);
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
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            !self.simd_lt(rhs)
          } else {
            let s: [u64;2] = cast(self);
            let r: [u64;2] = cast(rhs);
            cast([
              if s[0] >= r[0] { -1_i64 } else { 0 },
              if s[1] >= r[1] { -1_i64 } else { 0 },
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
            Self(vbslq_u64(self.0, if_true.0, if_false.0))
          } else {
            generic_bit_blend(self, if_true, if_false)
          }
        }
    }

    #[inline]
    pub(crate) fn to_bitmask_backend(self) -> u32 {
        pick! {
          if #[cfg(target_feature="sse")] {
            // use f64 move_mask since it is the same size as i64
            _mm_movemask_pd(cast(self.0)) as u32
          } else if #[cfg(target_feature="simd128")] {
            i64x2_bitmask(self.0) as u32
          } else {
            // nothing amazingly efficient for neon
            let arr: [u64; 2] = cast(self);
            (arr[0] >> 63 | ((arr[1] >> 62) & 2)) as u32
          }
        }
    }

    #[inline]
    pub(crate) fn any_backend(self) -> bool {
        pick! {
          if #[cfg(target_feature="sse")] {
            // use f64 move_mask since it is the same size as i64
            _mm_movemask_pd(cast(self.0)) != 0
          } else if #[cfg(target_feature="simd128")] {
            i64x2_bitmask(self.0) != 0
          } else {
            let v : [u64;2] = cast(self);
            ((v[0] | v[1]) & 0x8000000000000000) != 0
          }
        }
    }

    #[inline]
    pub(crate) fn all_backend(self) -> bool {
        pick! {
          if #[cfg(target_feature="avx2")] {
            // use f64 move_mask since it is the same size as i64
            _mm_movemask_pd(cast(self.0)) == 0b11
          }  else if #[cfg(target_feature="simd128")] {
            i64x2_bitmask(self.0) == 0b11
          } else {
            let v : [u64;2] = cast(self);
            ((v[0] & v[1]) & 0x8000000000000000) == 0x8000000000000000
          }
        }
    }

    #[inline]
    pub(crate) fn interleave_low_backend(self, other: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse2")] {
            Self(_mm_unpacklo_epi64(self.0, other.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(i64x2_shuffle::<0, 2>(self.0, other.0))
          } else if #[cfg(all(target_feature="neon", target_arch="aarch64"))] {
            Self(vzip1q_u64(self.0, other.0))
          } else {
            Self::from_array([self.as_array()[0], other.as_array()[0]])
          }
        }
    }

    #[inline]
    pub(crate) fn interleave_high_backend(self, other: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse2")] {
            Self(_mm_unpackhi_epi64(self.0, other.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(i64x2_shuffle::<1, 3>(self.0, other.0))
          } else if #[cfg(all(target_feature="neon", target_arch="aarch64"))] {
            Self(vzip2q_u64(self.0, other.0))
          } else {
            Self::from_array([self.as_array()[1], other.as_array()[1]])
          }
        }
    }

    #[inline]
    pub(crate) fn shuffle_backend(self, indices: u64x2) -> Self {
        pick! {
          if #[cfg(target_feature = "sse2")] {
            let e0 = Self(_mm_shuffle_epi32::<0b01_00_01_00>(self.0));
            let e1 = Self(_mm_shuffle_epi32::<0b11_10_11_10>(self.0));

            // We can assume that each index is either `0` or `1`, and negating that
            // always gives us either all bits zero or all bits one.
            (-indices).select(e1, e0)
          } else if #[cfg(target_feature = "simd128")] {
            let e0 = Self(u64x2_shuffle::<0, 0>(self.0, self.0));
            let e1 = Self(u64x2_shuffle::<1, 1>(self.0, self.0));

            // We can assume that each index is either `0` or `1`, and negating that
            // always gives us either all bits zero or all bits one.
            (-indices).select(e1, e0)
          } else if #[cfg(all(target_feature = "neon", target_arch = "aarch64"))]{
            let e0 = Self(vdupq_n_u64(vget_lane_u64::<0>(vget_low_u64(self.0))));
            let e1 = Self(vdupq_n_u64(vget_lane_u64::<0>(vget_high_u64(self.0))));

            // We can assume that each index is either `0` or `1`, and negating that
            // always gives us either all bits zero or all bits one.
            (-indices).select(e1, e0)
          } else {
            let e0 = Self::splat(self.as_array()[0]);
            let e1 = Self::splat(self.as_array()[1]);

            // We can assume that each index is either `0` or `1`, and negating that
            // always gives us either all bits zero or all bits one.
            (-indices).select(e1, e0)
          }
        }
    }

    #[inline]
    pub(crate) fn shuffle_zeroing_backend(self, indices: u64x2) -> Self {
        // Since all implementations use the `select` trick, we always need to mask
        self.shuffle(indices) & indices.simd_lt(2)
    }

    #[inline]
    pub(crate) fn shuffle_wrapping_backend(self, indices: u64x2) -> Self {
        // Since all implementations use the `select` trick, we always need to mask
        self.shuffle(indices & 1)
    }

    #[inline]
    pub(crate) fn shuffle_2_backend(inputs: [Self; 2], indices: u64x2) -> u64x2 {
        pick! {
          if #[cfg(all(target_feature = "avx512f", target_feature = "avx512vl"))] {
            u64x2(_mm_permutex2var_epi64(inputs[0].0, indices.0, inputs[1].0))
          } else {
            let self_bytes = cast::<[u64x2; 2], [u8x16; 2]>(inputs);
            let byte_indices = indices.to_byte_indices();

            cast::<u8x16, u64x2>(self_bytes.shuffle(byte_indices))
          }
        }
    }

    #[inline]
    pub(crate) fn shuffle_zeroing_2_backend(inputs: [Self; 2], indices: u64x2) -> u64x2 {
        // Even if the `u8x16::shuffle` implementation is zeroing, our 64-bit to
        // 8-bit can trigger an overflow causing incorrect behavior
        inputs.shuffle(indices) & indices.simd_lt(4)
    }

    #[inline]
    pub(crate) fn shuffle_wrapping_2_backend(inputs: [Self; 2], indices: u64x2) -> u64x2 {
        pick! {
          if #[cfg(all(target_feature = "avx512f", target_feature = "avx512vl"))] {
            // `avx512` shuffle intrinsics are wrapping
            inputs.shuffle(indices)
          } else {
            inputs.shuffle(indices & 3)
          }
        }
    }

    #[inline]
    pub(crate) fn shuffle_3_backend(inputs: [Self; 3], indices: u64x2) -> u64x2 {
        let self_bytes = cast::<[u64x2; 3], [u8x16; 3]>(inputs);
        let byte_indices = indices.to_byte_indices();

        cast::<u8x16, u64x2>(self_bytes.shuffle(byte_indices))
    }

    #[inline]
    pub(crate) fn shuffle_zeroing_3_backend(inputs: [Self; 3], indices: u64x2) -> u64x2 {
        // Even if the `u8x16::shuffle` implementation is zeroing, our 64-bit to
        // 8-bit can trigger an overflow causing incorrect behavior
        inputs.shuffle(indices) & indices.simd_lt(6)
    }

    #[inline]
    pub(crate) fn shuffle_wrapping_3_backend(inputs: [Self; 3], indices: u64x2) -> u64x2 {
        inputs.shuffle(indices % 6)
    }

    #[inline]
    pub(crate) fn shuffle_4_backend(inputs: [Self; 4], indices: u64x2) -> u64x2 {
        let self_bytes = cast::<[u64x2; 4], [u8x16; 4]>(inputs);
        let byte_indices = indices.to_byte_indices();

        cast::<u8x16, u64x2>(self_bytes.shuffle(byte_indices))
    }

    #[inline]
    pub(crate) fn shuffle_zeroing_4_backend(inputs: [Self; 4], indices: u64x2) -> u64x2 {
        // Even if the `u8x16::shuffle` implementation is zeroing, our 64-bit to
        // 8-bit can trigger an overflow causing incorrect behavior
        inputs.shuffle(indices) & indices.simd_lt(8)
    }

    #[inline]
    pub(crate) fn shuffle_wrapping_4_backend(inputs: [Self; 4], indices: u64x2) -> u64x2 {
        inputs.shuffle(indices & 7)
    }

    #[inline]
    pub(crate) fn transpose_backend(data: [Self; 2]) -> [Self; 2] {
        pick! {
          if #[cfg(any(
            target_feature="sse2",
            all(target_feature="neon",target_arch="aarch64"),
            target_feature="simd128",
          ))] {
            // TODO: Remove the casts once unpack functions exist for `Self`.
            // [data[0].interleave_low(data[1]), data[0].interleave_high(data[1])]
            [
              data[0].cast_signed().interleave_low(data[1].cast_signed()).cast_unsigned(),
              data[0].cast_signed().interleave_high(data[1].cast_signed()).cast_unsigned(),
            ]
          } else {
            let [x, y, z, w]: [u64; 4] = cast(data);
            cast([x, z, y, w])
          }
        }
    }

    #[inline]
    pub(crate) fn replace_const_backend<const INDEX: usize>(self, value: u64) -> Self {
        /// Because of limitations in const generics, this must have separate
        /// branches per index. This macro is used to avoid duplication.
        macro_rules! use_indices {
      (0..2 = [$($INDEX:literal),*]) => {
        pick! {
          if #[cfg(all(target_arch = "x86_64", target_feature = "sse4.1"))] {
            match INDEX {
              $($INDEX => Self(_mm_insert_epi64::<$INDEX>(self.0, value as i64)),)*
              2.. => unreachable!()
            }
          } else if #[cfg(target_feature = "sse2")] {
            #[cfg(target_arch = "x86")]
            let value = cast::<[u64; 2], __m128i>([value, 0]);
            #[cfg(target_arch = "x86_64")]
            let value = _mm_cvtsi64_si128(value.cast_signed());

            match INDEX {
              0 => cast(_mm_move_sd(cast(self.0), cast(value))),
              1 => cast(_mm_unpacklo_epi64(cast(self.0), cast(value))),
              2.. => unreachable!(),
            }
          } else if #[cfg(all(
            target_arch = "aarch64",
            target_feature = "neon",
            target_endian = "little",
          ))] {
            match INDEX {
              $($INDEX => Self(vsetq_lane_u64::<$INDEX>(value, self.0)),)*
              2.. => unreachable!(),
            }
          } else if #[cfg(target_feature = "simd128")] {
            Self(u64x2_replace_lane::<INDEX>(self.0, value))
          } else {
            let mut result = self;
            result.as_mut_array()[INDEX] = value;
            result
          }
        }
      };
    }
        use_indices! { 0..2 = [0, 1] }
    }

    #[inline]
    pub(crate) fn extract_const_backend<const INDEX: usize>(self) -> u64 {
        /// Because of limitations in const generics, this must have separate
        /// branches per index. This macro is used to avoid duplication.
        macro_rules! use_indices {
      (0..2 = [$($INDEX:literal),*]) => {
        pick! {
          if #[cfg(all(target_arch = "x86_64", target_feature = "sse4.1"))] {
            match INDEX {
              $($INDEX => _mm_extract_epi64::<$INDEX>(self.0).cast_unsigned(),)*
              2.. => unreachable!()
            }
          } else if #[cfg(target_arch = "x86_64")] {
            match INDEX {
              0 => _mm_cvtsi128_si64(self.0).cast_unsigned(),
              1 => _mm_cvtsi128_si64(_mm_unpackhi_epi64(self.0, self.0)).cast_unsigned(),
              2.. => unreachable!(),
            }
          } else if #[cfg(all(
            target_arch = "aarch64",
            target_feature = "neon",
            target_endian = "little",
          ))] {
            match INDEX {
              $($INDEX => vgetq_lane_u64::<$INDEX>(self.0),)*
              2.. => unreachable!(),
            }
          } else if #[cfg(target_feature = "simd128")] {
            u64x2_extract_lane::<INDEX>(self.0)
          } else {
            self.to_array()[INDEX]
          }
        }
      };
    }
        use_indices! { 0..2 = [0, 1] }
    }
}
}

safe_target_feature! {
// Backend of SIMD integer functions.
impl u64x2 {
    #[inline]
    pub(crate) fn neg_backend(self) -> Self {
        pick! {
            if #[cfg(all(target_arch = "aarch64", target_feature = "neon"))] {
                Self(vreinterpretq_u64_s64(vnegq_s64(vreinterpretq_s64_u64(self.0))))
            } else {
                Self::default() - self
            }
        }
    }

    #[inline]
    pub(crate) fn not_backend(self) -> Self {
        self ^ cast::<u128, u64x2>(u128::MAX)
    }

    #[inline]
    pub(crate) fn add_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse2")] {
            Self(_mm_add_epi64(self.0, rhs.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(u64x2_add(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vaddq_u64(self.0, rhs.0))
          } else {
            Self::from_array([
              self.0.arr[0].wrapping_add(rhs.0.arr[0]),
              self.0.arr[1].wrapping_add(rhs.0.arr[1]),
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn sub_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse2")] {
            Self(_mm_sub_epi64(self.0, rhs.0))
          } else if #[cfg(target_feature="simd128")] {
            Self(u64x2_sub(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vsubq_u64(self.0, rhs.0))
          } else {
            Self::from_array([
              self.0.arr[0].wrapping_sub(rhs.0.arr[0]),
              self.0.arr[1].wrapping_sub(rhs.0.arr[1]),
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn mul_backend(self, rhs: Self) -> Self {
        //we should try to implement this on sse2
        pick! {
          if #[cfg(target_feature="simd128")] {
            Self(u64x2_mul(self.0, rhs.0))
          } else {
            let arr1: [u64; 2] = cast(self);
            let arr2: [u64; 2] = cast(rhs);
            cast([
              arr1[0].wrapping_mul(arr2[0]),
              arr1[1].wrapping_mul(arr2[1]),
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
            Self(vandq_u64(self.0, rhs.0))
          } else {
            Self::from_array([
              self.0.arr[0] & rhs.0.arr[0],
              self.0.arr[1] & rhs.0.arr[1],
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
            Self(vorrq_u64(self.0, rhs.0))
          } else {
            Self::from_array([
              self.0.arr[0] | rhs.0.arr[0],
              self.0.arr[1] | rhs.0.arr[1],
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
            Self(veorq_u64(self.0, rhs.0))
          } else {
            Self::from_array([
              self.0.arr[0] ^ rhs.0.arr[0],
              self.0.arr[1] ^ rhs.0.arr[1],
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn reduce_add_backend(self) -> u64 {
        pick! {
          if #[cfg(any(target_feature="sse2", target_feature="simd128"))] {
            let array: [u64; 2] = cast(self);
            array[0].wrapping_add(array[1])
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            vgetq_lane_u64(self.0, 0).wrapping_add(vgetq_lane_u64(self.0, 1))
          } else {
            self.0.arr[0].wrapping_add(self.0.arr[1])
          }
        }
    }

    #[inline]
    pub(crate) fn reduce_mul_backend(self) -> u64 {
        pick! {
          if #[cfg(any(target_feature="sse2", target_feature="simd128"))] {
            let array: [u64; 2] = cast(self);
            array[0].wrapping_mul(array[1])
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            vgetq_lane_u64(self.0, 0).wrapping_mul(vgetq_lane_u64(self.0, 1))
          } else {
            self.0.arr[0].wrapping_mul(self.0.arr[1])
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
            Self(vbslq_u64(self.0, if_one.0, if_zero.0))
          } else {
            generic_bit_blend(self, if_one, if_zero)
          }
        }
    }

    #[inline]
    pub(crate) fn shl_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="avx2")] {
            // mask the shift count to 63 to have same behavior on all platforms
            let shift_by = rhs & Self::splat(63);
            Self(_mm_sllv_epi64(self.0, shift_by.0))
          } else if #[cfg(all(target_feature="neon", target_arch="aarch64"))] {
            // mask the shift count to 63 to have same behavior on all platforms
            let shift_by = vreinterpretq_s64_u64(vandq_u64(rhs.0, vmovq_n_u64(63)));
            Self(vshlq_u64(self.0, shift_by))
          } else {
            let arr: [u64; 2] = cast(self);
            let rhs: [u64; 2] = cast(rhs);
            cast([
              arr[0].wrapping_shl(rhs[0] as u32),
              arr[1].wrapping_shl(rhs[1] as u32),
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn shl_scalar_backend(self, rhs: u32) -> Self {
        pick! {
          if #[cfg(target_feature="sse2")] {
            // Use `rhs % 64` to perform wrapping shift and not unbounded shift.
            let shift = cast([rhs as u64 & 63, 0]);
            Self(_mm_sll_epi64(self.0, shift))
          } else if #[cfg(target_feature="simd128")] {
            Self(u64x2_shl(self.0, rhs))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            // Use `rhs % 64` to perform wrapping shift and not unbounded shift.
            Self(vshlq_u64(self.0, vmovq_n_s64(rhs as i64 & 63)))
          } else {
            Self::from_array([
              self.0.arr[0].wrapping_shl(rhs),
              self.0.arr[1].wrapping_shl(rhs),
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn shr_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="avx2")] {
            // mask the shift count to 63 to have same behavior on all platforms
            let shift_by = rhs & Self::splat(63);
            Self(_mm_srlv_epi64(self.0, shift_by.0))
          } else if #[cfg(all(target_feature="neon", target_arch="aarch64"))] {
            // mask the shift count to 63 to have same behavior on all platforms
            // no right shift, have to pass negative value to left shift on neon
            let shift_by = vnegq_s64(vreinterpretq_s64_u64(vandq_u64(rhs.0, vmovq_n_u64(63))));
            Self(vshlq_u64(self.0, shift_by))
          } else {
            let arr: [u64; 2] = cast(self);
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
          if #[cfg(target_feature="sse2")] {
            // Use `rhs % 64` to perform wrapping shift and not unbounded shift.
            let shift = cast([rhs as u64 & 63, 0]);
            Self(_mm_srl_epi64(self.0, shift))
          } else if #[cfg(target_feature="simd128")] {
            Self(u64x2_shr(self.0, rhs))
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            // Use `rhs % 64` to perform wrapping shift and not unbounded shift.
            Self(vshlq_u64(self.0, vmovq_n_s64(-(rhs as i64 & 63))))
          } else {
            Self::from_array([
              self.0.arr[0].wrapping_shr(rhs),
              self.0.arr[1].wrapping_shr(rhs),
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
    pub(crate) fn reduce_max_backend(self) -> u64 {
        pick! {
          if #[cfg(any(target_feature="sse2", target_feature="simd128"))] {
            let array: [u64; 2] = cast(self);
            array[0].max(array[1])
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            vgetq_lane_u64(self.0, 0).max(vgetq_lane_u64(self.0, 1))
          } else {
            self.0.arr[0].max(self.0.arr[1])
          }
        }
    }

    #[inline]
    pub(crate) fn reduce_min_backend(self) -> u64 {
        pick! {
          if #[cfg(any(target_feature="sse2", target_feature="simd128"))] {
            let array: [u64; 2] = cast(self);
            array[0].min(array[1])
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            vgetq_lane_u64(self.0, 0).min(vgetq_lane_u64(self.0, 1))
          } else {
            self.0.arr[0].min(self.0.arr[1])
          }
        }
    }

    #[inline]
    pub(crate) fn unbounded_shl_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="avx2")] {
            Self(_mm_sllv_epi64(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon", target_arch="aarch64"))] {
            // The intrinsic has different semantics so we need to mask ourselves.
            Self(vshlq_u64(self.0, vreinterpretq_s64_u64(rhs.0))) & rhs.simd_lt(64)
          } else {
            // Cannot use scalar `unbounded_shl` because it takes `u32`, which is
            // smaller than `u64`.
            (self << rhs) & rhs.simd_lt(64)
          }
        }
    }

    #[inline]
    pub(crate) fn unbounded_shl_scalar_backend(self, rhs: u32) -> Self {
        pick! {
          if #[cfg(target_feature="sse2")] {
            Self(_mm_sll_epi64(self.0, cast([rhs as u64, 0])))
          } else if #[cfg(target_feature="simd128")] {
            // The intrinsic performs wrapping shift so we need to mask the result.
            Self(u64x2_shl(self.0, rhs)) & Self::splat(rhs as u64).simd_lt(64)
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vshlq_u64(self.0, vmovq_n_s64(rhs.min(64) as i64)))
          } else {
            Self::from_array([
              self.0.arr[0].unbounded_shl(rhs),
              self.0.arr[1].unbounded_shl(rhs),
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn unbounded_shr_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="avx2")] {
            Self(_mm_srlv_epi64(self.0, rhs.0))
          } else if #[cfg(all(target_feature="neon", target_arch="aarch64"))] {
            // Negate `rhs` because there is no direct shift-right intrinsic, and
            // mask to hide `rhs` overflow.
            Self(vshlq_u64(self.0, vnegq_s64(vreinterpretq_s64_u64(rhs.0)))) & rhs.simd_lt(64)
          } else {
            // Cannot use scalar `unbounded_shr` because it takes `u32`, which is
            // smaller than `u64`.
            (self >> rhs) & rhs.simd_lt(64)
          }
        }
    }

    #[inline]
    pub(crate) fn unbounded_shr_scalar_backend(self, rhs: u32) -> Self {
        pick! {
          if #[cfg(target_feature="sse2")] {
            Self(_mm_srl_epi64(self.0, cast([rhs as u64, 0])))
          } else if #[cfg(target_feature="simd128")] {
            if rhs < 64 { Self(u64x2_shr(self.0, rhs)) } else { Self::ZERO }
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            // Negate `rhs` because there is no direct shift-right intrinsic, and
            // restrict it to prevent overflow.
            Self(vshlq_u64(self.0, vmovq_n_s64(-rhs.min(64).cast_signed() as i64)))
          } else {
            Self::from_array([
                self.0.arr[0].unbounded_shr(rhs),
                self.0.arr[1].unbounded_shr(rhs),
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
            Self(vqaddq_u64(self.0, rhs.0))
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
            let no_overflow = result.simd_le(self);
            // Return `0` (no bits set) if overflow occurs.
            result & no_overflow
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            Self(vqsubq_u64(self.0, rhs.0))
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
            Self::from_array([-(result[0].1 as i64) as u64, -(result[1].1 as i64) as u64]),
        )
    }

    #[inline]
    pub(crate) fn mul_keep_low_high_backend(self, rhs: Self) -> (Self, Self) {
        // TODO(perf): This implementation looks quite bad. Is there a better
        // one?

        let self_array = self.to_array();
        let rhs_array = rhs.to_array();

        let widening_mul = [
            (self_array[0] as u128).wrapping_mul(rhs_array[0] as u128),
            (self_array[1] as u128).wrapping_mul(rhs_array[1] as u128),
        ];

        (
            Self::from_array([widening_mul[0] as u64, widening_mul[1] as u64]),
            Self::from_array([
                (widening_mul[0] >> 64) as u64,
                (widening_mul[1] >> 64) as u64,
            ]),
        )
    }

    #[inline]
    pub(crate) fn mul_keep_high_backend(self, rhs: Self) -> Self {
        let arr1: [u64; 2] = cast(self);
        let arr2: [u64; 2] = cast(rhs);
        cast([
            ((arr1[0] as u128 * arr2[0] as u128) >> 64) as u64,
            ((arr1[1] as u128 * arr2[1] as u128) >> 64) as u64,
        ])
    }
}
}

safe_target_feature! {
// Backend of SIMD unsigned integer functions.
impl u64x2 {
    #[inline]
    pub(crate) fn saturating_add_signed_backend(self, rhs: i64x2) -> Self {
        pick! {
            if #[cfg(all(target_feature = "neon", target_arch = "aarch64"))] {
                Self(vsqaddq_u64(self.0, rhs.0))
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
impl u64x2 {
    #[inline]
    pub(crate) fn shuffle_consts_backend<const I0: usize, const I1: usize>(self) -> Self {
        if const { I0 == 0 && I1 == 0 } {
            pick! {
                if #[cfg(target_feature = "sse3")] {
                    Self(_mm_castpd_si128(_mm_movedup_pd(_mm_castsi128_pd(self.0))))
                } else if #[cfg(target_feature = "sse2")] {
                    Self(_mm_unpacklo_epi64(self.0, self.0))
                } else if #[cfg(all(
                    target_arch = "aarch64",
                    target_feature = "neon",
                    target_endian = "little",
                ))] {
                    Self(vdupq_laneq_u64::<0>(self.0))
                } else if #[cfg(target_feature = "simd128")] {
                    Self(u64x2_shuffle::<0, 0>(self.0, self.0))
                } else {
                    let self_array = self.to_array();
                    Self::from_array([self_array[0], self_array[0]])
                }
            }
        } else if const { I0 == 0 && I1 == 1 } {
            self
        } else if const { I0 == 1 && I1 == 0 } {
            pick! {
                if #[cfg(target_feature = "sse2")] {
                    let input = _mm_castsi128_pd(self.0);
                    Self(_mm_castpd_si128(_mm_shuffle_pd::<0b01>(input, input)))
                } else if #[cfg(all(
                    target_arch = "aarch64",
                    target_feature = "neon",
                    target_endian = "little",
                ))] {
                    Self(vextq_u64::<1>(self.0, self.0))
                } else if #[cfg(target_feature = "simd128")] {
                    Self(u64x2_shuffle::<1, 0>(self.0, self.0))
                } else {
                    let self_array = self.to_array();
                    Self::from_array([self_array[1], self_array[0]])
                }
            }
        } else if const { I0 == 1 && I1 == 1 } {
            pick! {
                if #[cfg(target_feature = "sse2")] {
                    Self(_mm_unpackhi_epi64(self.0, self.0))
                } else if #[cfg(all(
                    target_arch = "aarch64",
                    target_feature = "neon",
                    target_endian = "little",
                ))] {
                    Self(vdupq_laneq_u64::<1>(self.0))
                } else if #[cfg(target_feature = "simd128")] {
                    Self(u64x2_shuffle::<1, 1>(self.0, self.0))
                } else {
                    let self_array = self.to_array();
                    Self::from_array([self_array[1], self_array[1]])
                }
            }
        } else {
          unreachable!();
        }
    }

    /// A helper for shuffle functions that turns indices of 64-bit lanes into
    /// byte indices that can be used with 8-bit shuffle intrinsics.
    ///
    /// This turns each 64-bit lane `i` into eight 8-bit lanes
    /// `[8*i, 8*i + 1, 8*i + 2, 8*i + 3, ...]`.
    ///
    /// This assumes `self` has already been reduced to the table's lane count,
    /// which may be at most 32 lanes so that `8 * i` still fits in a byte.
    #[allow(dead_code)]
    #[inline]
    fn to_byte_indices(self) -> u8x16 {
        // The byte offset of the lane, broadcast to every byte of the lane.
        let base = self.unbounded_shl_scalar(3);
        let base = base | base.unbounded_shl_scalar(8);
        let base = base | base.unbounded_shl_scalar(16);
        let base = base | base.unbounded_shl_scalar(32);

        // Then the offset of each byte within its lane. These bits are free because
        // every byte of `base` is a multiple of eight. `from_ne_bytes` keeps this
        // correct on big endian, where the bytes of a lane are the other way around.
        const WITHIN_LANE: u64x2 = u64x2::splat(u64::from_ne_bytes([0, 1, 2, 3, 4, 5, 6, 7]));

        cast::<u64x2, u8x16>(base | WITHIN_LANE)
    }

    /// The exact per-lane product of `a` and `b` masked to `W` bits.
    ///
    /// Only exact for `W <= 32`, where the product still fits a lane, and
    /// callers guard on that. It is instantiated for wider `W` too, since the
    /// guard is a runtime `if` on a const, so it cannot assert the bound itself.
    #[inline]
    fn mul_masked<const W: u32>(a: Self, b: Self) -> Self {
        pick! {
          if #[cfg(target_feature="sse2")] {
            // `pmuludq` reads the low 32 bits of a lane anyway, so at `W == 32` the
            // operand masks are already implied and can be dropped. A scalar
            // multiply has no such truncation, so the other arm always masks.
            let (a, b) = if W == 32 {
              (a, b)
            } else {
              let mask = Self::splat(add_mul_operand_mask_u64::<W>());
              (a & mask, b & mask)
            };

            Self(_mm_mul_epu32(a.0, b.0))
          } else {
            let mask = add_mul_operand_mask_u64::<W>();
            let a = a.to_array();
            let b = b.to_array();
            Self::from_array(core::array::from_fn(|i| (a[i] & mask) * (b[i] & mask)))
          }
        }
    }

    #[inline]
    pub(crate) fn add_mul_lo_backend<const W: u32>(self, a: Self, b: Self) -> Self {
        pick! {
          if #[cfg(all(target_feature="avx512ifma", target_feature="avx512vl"))] {
            // IFMA is fixed at 52 bits; any other width takes the generic path.
            if W == 52 {
              return Self(_mm_madd52lo_epu64(self.0, a.0, b.0));
            }
          }
        }

        // Below 33 bits the whole product fits a lane, so one widening multiply
        // yields both halves and the split is a mask rather than an instruction.
        if W <= 32 {
            let mask = Self::splat(add_mul_operand_mask_u64::<W>());
            return self + (Self::mul_masked::<W>(a, b) & mask);
        }

        let acc = self.to_array();
        let a = a.to_array();
        let b = b.to_array();
        Self::from_array(core::array::from_fn(|i| {
            add_mul_lo_lane_u64::<W>(acc[i], a[i], b[i])
        }))
    }

    #[inline]
    pub(crate) fn add_mul_hi_backend<const W: u32>(self, a: Self, b: Self) -> Self {
        pick! {
          if #[cfg(all(target_feature="avx512ifma", target_feature="avx512vl"))] {
            // IFMA is fixed at 52 bits; any other width takes the generic path.
            if W == 52 {
              return Self(_mm_madd52hi_epu64(self.0, a.0, b.0));
            }
          }
        }

        // See `add_mul_lo`: the whole product is in the lane, so the high half is a
        // shift.
        if W <= 32 {
            return self + (Self::mul_masked::<W>(a, b) >> W);
        }

        let acc = self.to_array();
        let a = a.to_array();
        let b = b.to_array();
        Self::from_array(core::array::from_fn(|i| {
            add_mul_hi_lane_u64::<W>(acc[i], a[i], b[i])
        }))
    }
}
}
