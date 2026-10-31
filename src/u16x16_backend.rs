use crate::backend_imports::*;

#[cfg(not(target_feature = "avx2"))]
#[repr(C, align(32))]
#[derive(Clone, Copy)]
pub(crate) struct Inner {
    pub lo: u16x8,
    pub hi: u16x8,
}

safe_target_feature! {
impl SimdBackend for u16x16 {
    pick! {
      if #[cfg(target_feature="avx2")] {
        type Inner = __m256i;
      } else {
        type Inner = Inner;
      }
    }

    #[inline]
    pub(crate) fn simd_eq_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="avx2")] {
            Self(_mm256_cmpeq_epi16(self.0, rhs.0))
          } else {
            Self(Inner {
              lo: self.0.lo.simd_eq(rhs.0.lo),
              hi: self.0.hi.simd_eq(rhs.0.hi),
            })
          }
        }
    }

    #[inline]
    pub(crate) fn simd_ne_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="avx2")] {
            !self.simd_eq(rhs)
          } else {
            Self(Inner {
              lo: self.0.lo.simd_ne(rhs.0.lo),
              hi: self.0.hi.simd_ne(rhs.0.hi),
            })
          }
        }
    }

    #[inline]
    pub(crate) fn simd_lt_backend(self, rhs: Self) -> Self {
        // no gt, so just reverse to get same answer
        Self::simd_gt(rhs, self)
    }

    #[inline]
    pub(crate) fn simd_gt_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature = "avx2")] {
            let bias = Self::from_array([0x8000u16; 16]).0;
            let a_biased = _mm256_sub_epi16(self.0, bias);
            let b_biased = _mm256_sub_epi16(rhs.0, bias);
            let mask = _mm256_cmpgt_epi16(a_biased, b_biased);

            Self(mask)
          } else {
            Self(Inner {
              lo: self.0.lo.simd_gt(rhs.0.lo),
              hi: self.0.hi.simd_gt(rhs.0.hi),
            })
          }
        }
    }

    #[inline]
    pub(crate) fn simd_le_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="avx2")] {
            !self.simd_gt(rhs)
          } else {
            Self(Inner {
              lo: self.0.lo.simd_le(rhs.0.lo),
              hi: self.0.hi.simd_le(rhs.0.hi),
            })
          }
        }
    }

    #[inline]
    pub(crate) fn simd_ge_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="avx2")] {
            !self.simd_lt(rhs)
          } else {
            Self(Inner {
              lo: self.0.lo.simd_ge(rhs.0.lo),
              hi: self.0.hi.simd_ge(rhs.0.hi),
            })
          }
        }
    }

    #[inline]
    pub(crate) fn select_backend(self, if_true: Self, if_false: Self) -> Self {
        pick! {
          if #[cfg(target_feature="avx2")] {
            Self(_mm256_blendv_epi8(if_false.0, if_true.0, self.0))
          } else {
            Self(Inner {
              lo: self.0.lo.select(if_true.0.lo, if_false.0.lo),
              hi: self.0.hi.select(if_true.0.hi, if_false.0.hi),
            })
          }
        }
    }

    #[inline]
    pub(crate) fn to_bitmask_backend(self) -> u32 {
        pick! {
          if #[cfg(target_feature="sse2")] {
              let [a,b] = cast::<_,[__m128i;2]>(self);
              _mm_movemask_epi8( _mm_packs_epi16(a,b)) as u32
            } else {
            self.0.lo.to_bitmask() | (self.0.hi.to_bitmask() << 8)
          }
        }
    }

    #[inline]
    pub(crate) fn any_backend(self) -> bool {
        pick! {
          if #[cfg(target_feature="avx2")] {
            ((_mm256_movemask_epi8(self.0) as u32) & 0b10101010101010101010101010101010) != 0
          } else {
            (self.0.lo | self.0.hi).any()
          }
        }
    }

    #[inline]
    pub(crate) fn all_backend(self) -> bool {
        pick! {
          if #[cfg(target_feature="avx2")] {
            ((_mm256_movemask_epi8(self.0) as u32) & 0b10101010101010101010101010101010) == 0b10101010101010101010101010101010
          } else {
            (self.0.lo & self.0.hi).all()
          }
        }
    }

    #[inline]
    pub(crate) fn interleave_low_backend(self, other: Self) -> Self {
        // `_mm256_unpacklo_epi16` cannot be used because it acts within each
        // 128-bit lane, which is a different operation.
        let [self_a, _] = cast::<u16x16, [u16x8; 2]>(self);
        let [other_a, _] = cast::<u16x16, [u16x8; 2]>(other);
        cast([
            self_a.interleave_low(other_a),
            self_a.interleave_high(other_a),
        ])
    }

    #[inline]
    pub(crate) fn interleave_high_backend(self, other: Self) -> Self {
        // `_mm256_unpackhi_epi16` cannot be used because it acts within each
        // 128-bit lane, which is a different operation.
        let [_, self_b] = cast::<u16x16, [u16x8; 2]>(self);
        let [_, other_b] = cast::<u16x16, [u16x8; 2]>(other);
        cast([
            self_b.interleave_low(other_b),
            self_b.interleave_high(other_b),
        ])
    }

    #[inline]
    pub(crate) fn shuffle_backend(self, indices: u16x16) -> Self {
        pick! {
          if #[cfg(all(target_feature = "avx512bw", target_feature = "avx512vl"))] {
            Self(_mm256_permutexvar_epi16(indices.0, self.0))
          } else if #[cfg(any(
            target_feature = "ssse3",
            all(target_arch = "aarch64", target_feature = "neon"),
            target_feature = "simd128",
          ))] {
            let self_bytes = cast::<u16x16, u8x32>(self);
            let byte_indices = indices.to_byte_indices();

            cast::<u8x32, u16x16>(self_bytes.shuffle(byte_indices))
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
    pub(crate) fn shuffle_zeroing_backend(self, indices: u16x16) -> Self {
        pick! {
          if #[cfg(any(
            target_feature = "ssse3",
            all(target_arch = "aarch64", target_feature = "neon"),
            target_feature = "simd128",
          ))] {
            // Even if the `u8x32::shuffle` implementation is zeroing, our 16-bit to
            // 8-bit can trigger an overflow causing incorrect behavior
            self.shuffle(indices) & indices.simd_lt(16)
          } else {
            // The fallback branch of `shuffle` already has the behavior we want
            self.shuffle(indices)
          }
        }
    }

    #[inline]
    pub(crate) fn shuffle_wrapping_backend(self, indices: u16x16) -> Self {
        pick! {
          if #[cfg(all(target_feature = "avx512bw", target_feature = "avx512vl"))] {
            // `avx512` shuffle intrinsics are wrapping
            self.shuffle(indices)
          } else {
            self.shuffle(indices & 15)
          }
        }
    }

    #[inline]
    pub(crate) fn shuffle_2_backend(inputs: [Self; 2], indices: u16x16) -> u16x16 {
        pick! {
          if #[cfg(all(target_feature = "avx512bw", target_feature = "avx512vl"))] {
            u16x16(_mm256_permutex2var_epi16(inputs[0].0, indices.0, inputs[1].0))
          } else {
            let self_bytes = cast::<[u16x16; 2], [u8x32; 2]>(inputs);
            let byte_indices = indices.to_byte_indices();

            cast::<u8x32, u16x16>(self_bytes.shuffle(byte_indices))
          }
        }
    }

    #[inline]
    pub(crate) fn shuffle_zeroing_2_backend(inputs: [Self; 2], indices: u16x16) -> u16x16 {
        // Even if the `u8x32` shuffle is zeroing, our 16-bit to 8-bit index
        // conversion breaks for out of bounds indices.
        inputs.shuffle(indices) & indices.simd_lt(32)
    }

    #[inline]
    pub(crate) fn shuffle_wrapping_2_backend(inputs: [Self; 2], indices: u16x16) -> u16x16 {
        pick! {
          if #[cfg(all(target_feature = "avx512bw", target_feature = "avx512vl"))] {
            // `avx512` shuffle intrinsics are wrapping
            inputs.shuffle(indices)
          } else {
            inputs.shuffle(indices & 31)
          }
        }
    }

    #[inline]
    pub(crate) fn shuffle_3_backend(inputs: [Self; 3], indices: u16x16) -> u16x16 {
        let self_bytes = cast::<[u16x16; 3], [u8x32; 3]>(inputs);
        let byte_indices = indices.to_byte_indices();

        cast::<u8x32, u16x16>(self_bytes.shuffle(byte_indices))
    }

    #[inline]
    pub(crate) fn shuffle_zeroing_3_backend(inputs: [Self; 3], indices: u16x16) -> u16x16 {
        // Even if the `u8x32` shuffle is zeroing, our 16-bit to 8-bit index
        // conversion breaks for out of bounds indices.
        inputs.shuffle(indices) & indices.simd_lt(48)
    }

    #[inline]
    pub(crate) fn shuffle_wrapping_3_backend(inputs: [Self; 3], indices: u16x16) -> u16x16 {
        inputs.shuffle(indices % 48)
    }

    #[inline]
    pub(crate) fn shuffle_4_backend(inputs: [Self; 4], indices: u16x16) -> u16x16 {
        let self_bytes = cast::<[u16x16; 4], [u8x32; 4]>(inputs);
        let byte_indices = indices.to_byte_indices();

        cast::<u8x32, u16x16>(self_bytes.shuffle(byte_indices))
    }

    #[inline]
    pub(crate) fn shuffle_zeroing_4_backend(inputs: [Self; 4], indices: u16x16) -> u16x16 {
        // Even if the `u8x32` shuffle is zeroing, our 16-bit to 8-bit index
        // conversion breaks for out of bounds indices.
        inputs.shuffle(indices) & indices.simd_lt(64)
    }

    #[inline]
    pub(crate) fn shuffle_wrapping_4_backend(inputs: [Self; 4], indices: u16x16) -> u16x16 {
        inputs.shuffle(indices & 63)
    }

    #[inline]
    pub(crate) fn transpose_backend(data: [Self; 16]) -> [Self; 16] {
        // Can this be optimized?

        #[inline(always)]
        fn transpose_column(data: &[u16x16; 16], index: usize) -> u16x16 {
            u16x16::from_array([
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
    pub(crate) fn replace_const_backend<const INDEX: usize>(self, value: u16) -> Self {
        /// Because of limitations in const generics, this must have separate
        /// branches per index. This macro is used to avoid duplication.
        macro_rules! use_indices {
      (
        0..16 = [$($INDEX:literal),*],
        0..8 = [$($LOW_INDEX:literal),*],
        8..16 = [$($HIGH_INDEX:literal),*]
      ) => {
        pick! {
          if #[cfg(target_feature = "avx2")] {
            match INDEX {
              $($INDEX => Self(_mm256_insert_epi16::<$INDEX>(self.0, value.cast_signed())),)*
              16.. => unreachable!(),
            }
          } else {
            let [self_a, self_b] = cast::<u16x16, [u16x8; 2]>(self);

            if const { INDEX < 8 } {
              let result_a = match INDEX {
                $($LOW_INDEX => self_a.replace_const::<$LOW_INDEX>(value),)*
                8.. => unreachable!(),
              };

              cast([result_a, self_b])
            } else {
              let result_b = match INDEX {
                $($HIGH_INDEX => self_b.replace_const::<{ $HIGH_INDEX - 8 }>(value),)*
                ..8 | 16.. => unreachable!(),
              };

              cast([self_a, result_b])
            }
          }
        }
      };
    }
        use_indices! {
          0..16 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
          0..8 = [0, 1, 2, 3, 4, 5, 6, 7],
          8..16 = [8, 9, 10, 11, 12, 13, 14, 15]
        }
    }

    #[inline]
    pub(crate) fn extract_const_backend<const INDEX: usize>(self) -> u16 {
        /// Because of limitations in const generics, this must have separate
        /// branches per index. This macro is used to avoid duplication.
        macro_rules! use_indices {
      (
        0..16 = [$($INDEX:literal),*],
        0..8 = [$($LOW_INDEX:literal),*],
        8..16 = [$($HIGH_INDEX:literal),*]
      ) => {
        pick! {
          if #[cfg(target_feature = "avx2")] {
            match INDEX {
              $($INDEX => _mm256_extract_epi16::<$INDEX>(self.0).cast_unsigned() as u16,)*
              16.. => unreachable!(),
            }
          } else {
            let [self_a, self_b] = cast::<u16x16, [u16x8; 2]>(self);

            if const { INDEX < 8 } {
              match INDEX {
                $($LOW_INDEX => self_a.extract_const::<$LOW_INDEX>(),)*
                8.. => unreachable!(),
              }
            } else {
              match INDEX {
                $($HIGH_INDEX => self_b.extract_const::<{ $HIGH_INDEX - 8 }>(),)*
                ..8 | 16.. => unreachable!(),
              }
            }
          }
        }
      };
    }
        use_indices! {
          0..16 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
          0..8 = [0, 1, 2, 3, 4, 5, 6, 7],
          8..16 = [8, 9, 10, 11, 12, 13, 14, 15]
        }
    }
}
}

safe_target_feature! {
// Backend of SIMD integer functions.
impl u16x16 {
    #[inline]
    pub(crate) fn neg_backend(self) -> Self {
        pick! {
            if #[cfg(target_feature = "avx2")] {
                Self::default() - self
            } else {
                Self(Inner {
                    lo: -self.0.lo,
                    hi: -self.0.hi,
                })
            }
        }
    }

    #[inline]
    pub(crate) fn not_backend(self) -> Self {
        pick! {
          if #[cfg(target_feature="avx2")] {
            self ^ Self::MAX
          } else {
            Self(Inner {
              lo: !self.0.lo,
              hi: !self.0.hi,
            })
          }
        }
    }

    #[inline]
    pub(crate) fn add_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="avx2")] {
            Self(_mm256_add_epi16(self.0, rhs.0))
          } else {
            Self(Inner {
              lo: self.0.lo + rhs.0.lo,
              hi: self.0.hi + rhs.0.hi,
            })
          }
        }
    }

    #[inline]
    pub(crate) fn sub_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="avx2")] {
            Self(_mm256_sub_epi16(self.0, rhs.0))
          } else {
            Self(Inner {
              lo: self.0.lo - rhs.0.lo,
              hi: self.0.hi - rhs.0.hi,
            })
          }
        }
    }

    #[inline]
    pub(crate) fn mul_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="avx2")] {
            // non-widening multiplication is the same for unsigned and signed
            Self(_mm256_mullo_epi16(self.0, rhs.0))
          } else {
            Self(Inner {
              lo: self.0.lo * rhs.0.lo,
              hi: self.0.hi * rhs.0.hi,
            })
          }
        }
    }

    #[inline]
    pub(crate) fn bitand_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="avx2")] {
            Self(_mm256_and_si256(self.0, rhs.0))
          } else {
            Self(Inner {
              lo: self.0.lo & rhs.0.lo,
              hi: self.0.hi & rhs.0.hi,
            })
          }
        }
    }

    #[inline]
    pub(crate) fn bitor_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="avx2")] {
            Self(_mm256_or_si256(self.0, rhs.0))
          } else {
            Self(Inner {
              lo: self.0.lo | rhs.0.lo,
              hi: self.0.hi | rhs.0.hi,
            })
          }
        }
    }

    #[inline]
    pub(crate) fn bitxor_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="avx2")] {
            Self(_mm256_xor_si256(self.0, rhs.0))
          } else {
            Self(Inner {
              lo: self.0.lo ^ rhs.0.lo,
              hi: self.0.hi ^ rhs.0.hi,
            })
          }
        }
    }

    #[inline]
    pub(crate) fn reduce_add_backend(self) -> u16 {
        let array: [u16x8; 2] = cast(self);
        (array[0] + array[1]).reduce_add()
    }

    #[inline]
    pub(crate) fn reduce_mul_backend(self) -> u16 {
        let array: [u16x8; 2] = cast(self);
        (array[0] * array[1]).reduce_mul()
    }

    #[inline]
    pub(crate) fn bitselect_backend(self, if_one: Self, if_zero: Self) -> Self {
        pick! {
          if #[cfg(target_feature="avx2")] {
            Self(_mm256_or_si256(
                _mm256_and_si256(if_one.0, self.0),
                _mm256_andnot_si256(self.0, if_zero.0),
            ))
          } else {
            Self(Inner {
              lo: self.0.lo.bitselect(if_one.0.lo, if_zero.0.lo),
              hi: self.0.hi.bitselect(if_one.0.hi, if_zero.0.hi),
            })
          }
        }
    }

    #[inline]
    pub(crate) fn shl_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(all(target_feature="avx512bw", target_feature="avx512vl"))] {
            // Mask `rhs` to 15 to match `wrapping_shl`.
            let rhs = _mm256_and_si256(rhs.0, _mm256_set1_epi16(15));
            cast(_mm256_sllv_epi16(self.0, rhs))
          } else {
            let [self_a, self_b]: [u16x8; 2] = cast(self);
            let [rhs_a, rhs_b]: [u16x8; 2] = cast(rhs);

            cast([self_a << rhs_a, self_b << rhs_b])
          }
        }
    }

    #[inline]
    pub(crate) fn shl_scalar_backend(self, rhs: u32) -> Self {
        pick! {
          if #[cfg(target_feature="avx2")] {
            // Use `rhs % 16` to perform wrapping shift and not unbounded shift.
            let shift = cast([rhs as u64 & 15, 0]);
            Self(_mm256_sll_epi16(self.0, shift))
          } else {
            Self(Inner {
              lo: self.0.lo << rhs,
              hi: self.0.hi << rhs,
            })
          }
        }
    }

    #[inline]
    pub(crate) fn shr_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(all(target_feature="avx512bw", target_feature="avx512vl"))] {
            // Mask `rhs` to 15 to match `wrapping_shr`.
            let rhs = _mm256_and_si256(rhs.0, _mm256_set1_epi16(15));
            cast(_mm256_srlv_epi16(self.0, rhs))
          } else {
            let [self_a, self_b]: [u16x8; 2] = cast(self);
            let [rhs_a, rhs_b]: [u16x8; 2] = cast(rhs);

            cast([self_a >> rhs_a, self_b >> rhs_b])
          }
        }
    }

    #[inline]
    pub(crate) fn shr_scalar_backend(self, rhs: u32) -> Self {
        pick! {
          if #[cfg(target_feature="avx2")] {
            // Use `rhs % 16` to perform wrapping shift and not unbounded shift.
            let shift = cast([rhs as u64 & 15, 0]);
            Self(_mm256_srl_epi16(self.0, shift))
          } else {
            Self(Inner {
              lo: self.0.lo >> rhs,
              hi: self.0.hi >> rhs,
            })
          }
        }
    }

    #[inline]
    pub(crate) fn max_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="avx2")] {
            Self(_mm256_max_epu16(self.0, rhs.0))
          } else {
            Self(Inner {
              lo: self.0.lo.max(rhs.0.lo),
              hi: self.0.hi.max(rhs.0.hi),
            })
          }
        }
    }

    #[inline]
    pub(crate) fn min_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="avx2")] {
            Self(_mm256_min_epu16(self.0, rhs.0))
          } else {
            Self(Inner {
              lo: self.0.lo.min(rhs.0.lo),
              hi: self.0.hi.min(rhs.0.hi),
            })
          }
        }
    }

    #[inline]
    pub(crate) fn reduce_max_backend(self) -> u16 {
        let array: [u16x8; 2] = cast(self);
        array[0].max(array[1]).reduce_max()
    }

    #[inline]
    pub(crate) fn reduce_min_backend(self) -> u16 {
        let array: [u16x8; 2] = cast(self);
        array[0].min(array[1]).reduce_min()
    }

    #[inline]
    pub(crate) fn unbounded_shl_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(all(target_feature="avx512bw", target_feature="avx512vl"))] {
            cast(_mm256_sllv_epi16(self.0, rhs.0))
          } else {
            let [self_a, self_b] = cast::<u16x16, [u16x8; 2]>(self);
            let [rhs_a, rhs_b] = cast::<u16x16, [u16x8; 2]>(rhs);

            cast([self_a.unbounded_shl(rhs_a), self_b.unbounded_shl(rhs_b)])
          }
        }
    }

    #[inline]
    pub(crate) fn unbounded_shl_scalar_backend(self, rhs: u32) -> Self {
        pick! {
          if #[cfg(target_feature="avx2")] {
            Self(_mm256_sll_epi16(self.0, cast([rhs as u64, 0])))
          } else {
            Self(Inner {
              lo: self.0.lo.unbounded_shl_scalar(rhs),
              hi: self.0.hi.unbounded_shl_scalar(rhs),
            })
          }
        }
    }

    #[inline]
    pub(crate) fn unbounded_shr_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(all(target_feature="avx512bw", target_feature="avx512vl"))] {
            cast(_mm256_srlv_epi16(self.0, rhs.0))
          } else {
            let [self_a, self_b] = cast::<u16x16, [u16x8; 2]>(self);
            let [rhs_a, rhs_b] = cast::<u16x16, [u16x8; 2]>(rhs);

            cast([self_a.unbounded_shr(rhs_a), self_b.unbounded_shr(rhs_b)])
          }
        }
    }

    #[inline]
    pub(crate) fn unbounded_shr_scalar_backend(self, rhs: u32) -> Self {
        pick! {
          if #[cfg(target_feature="avx2")] {
            Self(_mm256_srl_epi16(self.0, cast([rhs as u64, 0])))
          } else {
            Self(Inner {
              lo: self.0.lo.unbounded_shr_scalar(rhs),
              hi: self.0.hi.unbounded_shr_scalar(rhs),
            })
          }
        }
    }

    #[inline]
    pub(crate) fn saturating_add_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="avx2")] {
            Self(_mm256_adds_epu16(self.0, rhs.0))
          } else {
            Self(Inner {
              lo: self.0.lo.saturating_add(rhs.0.lo),
              hi: self.0.hi.saturating_add(rhs.0.hi),
            })
          }
        }
    }

    #[inline]
    pub(crate) fn saturating_sub_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="avx2")] {
            Self(_mm256_subs_epu16(self.0, rhs.0))
          } else {
            Self(Inner {
              lo: self.0.lo.saturating_sub(rhs.0.lo),
              hi: self.0.hi.saturating_sub(rhs.0.hi),
            })
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
    pub(crate) fn widening_mul_backend(self, rhs: Self) -> u32x16 {
        // x86 has no `_mm256_mul_epu16` intrinsic so there is no `avx2`
        // optimization.

        let [self_a, self_b] = cast::<u16x16, [u16x8; 2]>(self);
        let [rhs_a, rhs_b] = cast::<u16x16, [u16x8; 2]>(rhs);

        cast([self_a.widening_mul(rhs_a), self_b.widening_mul(rhs_b)])
    }

    #[inline]
    pub(crate) fn mul_keep_low_high_backend(self, rhs: Self) -> (Self, Self) {
        // x86 has no `_mm256_mul_epu16` intrinsic so there is no `avx2`
        // optimization.

        let [self_a, self_b] = cast::<u16x16, [u16x8; 2]>(self);
        let [rhs_a, rhs_b] = cast::<u16x16, [u16x8; 2]>(rhs);

        let result_a = self_a.mul_keep_low_high(rhs_a);
        let result_b = self_b.mul_keep_low_high(rhs_b);
        (
            cast([result_a.0, result_b.0]),
            cast([result_a.1, result_b.1]),
        )
    }

    #[inline]
    pub(crate) fn mul_keep_high_backend(self, rhs: Self) -> Self {
        // x86 has no `_mm256_mul_epu16` intrinsic so there is no `avx2`
        // optimization.

        let [self_a, self_b] = cast::<u16x16, [u16x8; 2]>(self);
        let [rhs_a, rhs_b] = cast::<u16x16, [u16x8; 2]>(rhs);

        cast([self_a.mul_keep_high(rhs_a), self_b.mul_keep_high(rhs_b)])
    }
}
}

safe_target_feature! {
// Backend of SIMD unsigned integer functions.
impl u16x16 {
    #[inline]
    pub(crate) fn saturating_add_signed_backend(self, rhs: i16x16) -> Self {
        pick! {
            if #[cfg(target_feature="avx2")] {
                let (result, overflow) = self.overflowing_add_signed(rhs);
                overflow.select(rhs.is_positive().cast_unsigned(), result)
            } else {
                Self(Inner {
                    lo: self.0.lo.saturating_add_signed(rhs.0.lo),
                    hi: self.0.hi.saturating_add_signed(rhs.0.hi),
                })
            }
        }
    }
}
}

safe_target_feature! {
// Internal helpers and backend of inconsistent functions.
impl u16x16 {
    /// A helper for shuffle functions that turns indices of 16-bit lanes into
    /// byte indices that can be used with 8-bit shuffle intrinsics.
    ///
    /// This turns each 16-bit lane `i` into two 8-bit lanes `[2*i, 2*i + 1]`.
    ///
    /// This assumes `self` has already been reduced to the table's lane count,
    /// which may be at most 128 lanes so that `2 * i` still fits in a byte.
    #[allow(dead_code)]
    #[inline]
    fn to_byte_indices(self) -> u8x32 {
        // The byte offset of the lane, broadcast to every byte of the lane.
        let base = self.unbounded_shl_scalar(1);
        let base = base | base.unbounded_shl_scalar(8);

        // Then the offset of each byte within its lane. These bits are free because
        // every byte of `base` is a multiple of two. `from_ne_bytes` keeps this
        // correct on big endian, where the bytes of a lane are the other way around.
        const WITHIN_LANE: u16x16 = u16x16::splat(u16::from_ne_bytes([0, 1]));

        cast::<u16x16, u8x32>(base | WITHIN_LANE)
    }

    #[inline]
    pub(crate) fn from_u8x16_backend(v: u8x16) -> Self {
        pick! {
          if #[cfg(target_feature="avx2")] {
            u16x16(_mm256_cvtepu8_epi16(v.0))
          } else if #[cfg(target_feature="sse2")] {
            Self(Inner {
              lo: u16x8(_mm_srli_epi16::<8>(_mm_unpacklo_epi8(v.0, v.0))),
              hi: u16x8(_mm_srli_epi16::<8>(_mm_unpackhi_epi8(v.0, v.0))),
            })
          } else {

            u16x16::from_array([
              v.as_array()[0] as u16,
              v.as_array()[1] as u16,
              v.as_array()[2] as u16,
              v.as_array()[3] as u16,
              v.as_array()[4] as u16,
              v.as_array()[5] as u16,
              v.as_array()[6] as u16,
              v.as_array()[7] as u16,
              v.as_array()[8] as u16,
              v.as_array()[9] as u16,
              v.as_array()[10] as u16,
              v.as_array()[11] as u16,
              v.as_array()[12] as u16,
              v.as_array()[13] as u16,
              v.as_array()[14] as u16,
              v.as_array()[15] as u16,
              ])
          }
        }
    }
}
}
