use crate::backend_imports::*;

#[cfg(not(target_feature = "avx512bw"))]
#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub(crate) struct Inner {
    pub lo: u8x32,
    pub hi: u8x32,
}

safe_target_feature! {
impl SimdBackend for u8x64 {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
        type Inner = __m512i;
      } else {
        type Inner = Inner;
      }
    }

    #[inline]
    pub(crate) fn simd_eq_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="avx512bw")] {
            Self(_mm512_maskz_mov_epi8(
              _mm512_cmp_epu8_mask::<{_MM_CMPINT_EQ}>(self.0, rhs.0),
              _mm512_set1_epi8(-1),
            ))
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
          if #[cfg(target_feature="avx512bw")] {
            Self(_mm512_maskz_mov_epi8(
              _mm512_cmp_epu8_mask::<{_MM_CMPINT_NE}>(self.0, rhs.0),
              _mm512_set1_epi8(-1),
            ))
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
        pick! {
          if #[cfg(target_feature="avx512bw")] {
            Self(_mm512_maskz_mov_epi8(
              _mm512_cmp_epu8_mask::<{_MM_CMPINT_LT}>(self.0, rhs.0),
              _mm512_set1_epi8(-1),
            ))
          } else {
            Self(Inner { lo: self.0.lo.simd_lt(rhs.0.lo), hi: self.0.hi.simd_lt(rhs.0.hi) })
          }
        }
    }

    #[inline]
    pub(crate) fn simd_gt_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="avx512bw")] {
            Self(_mm512_maskz_mov_epi8(
              _mm512_cmp_epu8_mask::<{_MM_CMPINT_NLE}>(self.0, rhs.0),
              _mm512_set1_epi8(-1),
            ))
          } else {
            Self(Inner { lo: self.0.lo.simd_gt(rhs.0.lo), hi: self.0.hi.simd_gt(rhs.0.hi) })
          }
        }
    }

    #[inline]
    pub(crate) fn simd_le_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="avx512bw")] {
            Self(_mm512_maskz_mov_epi8(
              _mm512_cmp_epu8_mask::<{_MM_CMPINT_LE}>(self.0, rhs.0),
              _mm512_set1_epi8(-1),
            ))
          } else {
            Self(Inner { lo: self.0.lo.simd_le(rhs.0.lo), hi: self.0.hi.simd_le(rhs.0.hi) })
          }
        }
    }

    #[inline]
    pub(crate) fn simd_ge_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="avx512bw")] {
            Self(_mm512_maskz_mov_epi8(
              _mm512_cmp_epu8_mask::<{_MM_CMPINT_NLT}>(self.0, rhs.0),
              _mm512_set1_epi8(-1),
            ))
          } else {
            Self(Inner { lo: self.0.lo.simd_ge(rhs.0.lo), hi: self.0.hi.simd_ge(rhs.0.hi) })
          }
        }
    }

    #[inline]
    pub(crate) fn select_backend(self, if_true: Self, if_false: Self) -> Self {
        pick! {
          if #[cfg(target_feature="avx512bw")] {
            Self(_mm512_mask_blend_epi8(_mm512_movepi8_mask(self.0), if_false.0, if_true.0))
          } else {
            Self(Inner {
              lo: self.0.lo.select(if_true.0.lo, if_false.0.lo),
              hi: self.0.hi.select(if_true.0.hi, if_false.0.hi),
            })
          }
        }
    }

    #[inline]
    pub(crate) fn to_bitmask_backend(self) -> u64 {
        pick! {
          if #[cfg(target_feature="avx512bw")] {
            _mm512_movepi8_mask(self.0) as u64
          } else {
            self.0.lo.to_bitmask() as u64 | ((self.0.hi.to_bitmask() as u64) << 32)
          }
        }
    }

    #[inline]
    pub(crate) fn any_backend(self) -> bool {
        pick! {
          if #[cfg(target_feature="avx512bw")] {
            _mm512_movepi8_mask(self.0) != 0
          } else {
            (self.0.lo | self.0.hi).any()
          }
        }
    }

    #[inline]
    pub(crate) fn all_backend(self) -> bool {
        pick! {
          if #[cfg(target_feature="avx512bw")] {
            _mm512_movepi8_mask(self.0) == u64::MAX
          } else {
            (self.0.lo & self.0.hi).all()
          }
        }
    }

    #[inline]
    pub(crate) fn interleave_low_backend(self, other: Self) -> Self {
        // `_mm512_unpacklo_epi8` cannot be used because it acts within each
        // 128-bit lane, which is a different operation.
        let [self_a, _] = cast::<u8x64, [u8x32; 2]>(self);
        let [other_a, _] = cast::<u8x64, [u8x32; 2]>(other);
        cast([
            self_a.interleave_low(other_a),
            self_a.interleave_high(other_a),
        ])
    }

    #[inline]
    pub(crate) fn interleave_high_backend(self, other: Self) -> Self {
        // `_mm512_unpackhi_epi8` cannot be used because it acts within each
        // 128-bit lane, which is a different operation.
        let [_, self_b] = cast::<u8x64, [u8x32; 2]>(self);
        let [_, other_b] = cast::<u8x64, [u8x32; 2]>(other);
        cast([
            self_b.interleave_low(other_b),
            self_b.interleave_high(other_b),
        ])
    }

    #[inline]
    pub(crate) fn shuffle_backend(self, indices: u8x64) -> Self {
        pick! {
          if #[cfg(target_feature = "avx512vbmi")] {
            Self(_mm512_permutexvar_epi8(indices.0, self.0))
          } else {
            let [self_a, self_b] = cast::<u8x64, [u8x32; 2]>(self);
            let [indices_a, indices_b] = cast::<u8x64, [u8x32; 2]>(indices);

            cast([[self_a, self_b].shuffle(indices_a), [self_a, self_b].shuffle(indices_b)])
          }
        }
    }

    #[inline]
    pub(crate) fn shuffle_zeroing_backend(self, indices: u8x64) -> Self {
        pick! {
          if #[cfg(target_feature = "avx512vbmi")] {
            self.shuffle(indices) & indices.simd_lt(64)
          } else {
            let [self_a, self_b] = cast::<u8x64, [u8x32; 2]>(self);
            let [indices_a, indices_b] = cast::<u8x64, [u8x32; 2]>(indices);

            cast([
              [self_a, self_b].shuffle_zeroing(indices_a),
              [self_a, self_b].shuffle_zeroing(indices_b),
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn shuffle_wrapping_backend(self, indices: u8x64) -> Self {
        pick! {
          if #[cfg(target_feature = "avx512vbmi")] {
            // `avx512` shuffle intrinsics are wrapping
            self.shuffle(indices)
          } else {
            let [self_a, self_b] = cast::<u8x64, [u8x32; 2]>(self);
            let [indices_a, indices_b] = cast::<u8x64, [u8x32; 2]>(indices);

            cast([
              [self_a, self_b].shuffle_wrapping(indices_a),
              [self_a, self_b].shuffle_wrapping(indices_b),
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn shuffle_2_backend(inputs: [Self; 2], indices: u8x64) -> u8x64 {
        pick! {
          if #[cfg(target_feature = "avx512vbmi")] {
            u8x64(_mm512_permutex2var_epi8(inputs[0].0, indices.0, inputs[1].0))
          } else {
            inputs[0].shuffle_zeroing(indices) | inputs[1].shuffle_zeroing(indices - 64)
          }
        }
    }

    #[inline]
    pub(crate) fn shuffle_zeroing_2_backend(inputs: [Self; 2], indices: u8x64) -> u8x64 {
        pick! {
          if #[cfg(target_feature = "avx512vbmi")] {
            inputs.shuffle(indices) & indices.simd_lt(128)
          } else {
            inputs.shuffle(indices)
          }
        }
    }

    #[inline]
    pub(crate) fn shuffle_wrapping_2_backend(inputs: [Self; 2], indices: u8x64) -> u8x64 {
        pick! {
          if #[cfg(target_feature = "avx512vbmi")] {
            // `avx512` shuffle intrinsics are wrapping
            inputs.shuffle(indices)
          } else {
            inputs.shuffle(indices & 127)
          }
        }
    }

    #[inline]
    pub(crate) fn shuffle_3_backend(inputs: [Self; 3], indices: u8x64) -> u8x64 {
        [inputs[0], inputs[1]].shuffle_zeroing(indices) | inputs[2].shuffle_zeroing(indices - 128)
    }

    #[inline]
    pub(crate) fn shuffle_zeroing_3_backend(inputs: [Self; 3], indices: u8x64) -> u8x64 {
        inputs.shuffle(indices)
    }

    #[inline]
    pub(crate) fn shuffle_wrapping_3_backend(inputs: [Self; 3], indices: u8x64) -> u8x64 {
        inputs.shuffle(indices % 192)
    }

    #[inline]
    pub(crate) fn shuffle_4_backend(inputs: [Self; 4], indices: u8x64) -> u8x64 {
        [inputs[0], inputs[1]].shuffle_zeroing(indices)
            | [inputs[2], inputs[3]].shuffle_zeroing(indices - 128)
    }

    #[inline]
    pub(crate) fn shuffle_zeroing_4_backend(inputs: [Self; 4], indices: u8x64) -> u8x64 {
        inputs.shuffle(indices)
    }

    #[inline]
    pub(crate) fn shuffle_wrapping_4_backend(inputs: [Self; 4], indices: u8x64) -> u8x64 {
        // There are 64*4=256 lanes to pick from, and that is exactly how many
        // indices `u8` can represent, so out of bounds indices are impossible.
        inputs.shuffle(indices)
    }

    #[inline]
    pub(crate) fn transpose_backend(data: [Self; 64]) -> [Self; 64] {
        // Can this be optimized?

        #[inline(always)]
        fn transpose_column(data: &[u8x64; 64], index: usize) -> u8x64 {
            u8x64::from_array([
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
                data[16].as_array()[index],
                data[17].as_array()[index],
                data[18].as_array()[index],
                data[19].as_array()[index],
                data[20].as_array()[index],
                data[21].as_array()[index],
                data[22].as_array()[index],
                data[23].as_array()[index],
                data[24].as_array()[index],
                data[25].as_array()[index],
                data[26].as_array()[index],
                data[27].as_array()[index],
                data[28].as_array()[index],
                data[29].as_array()[index],
                data[30].as_array()[index],
                data[31].as_array()[index],
                data[32].as_array()[index],
                data[33].as_array()[index],
                data[34].as_array()[index],
                data[35].as_array()[index],
                data[36].as_array()[index],
                data[37].as_array()[index],
                data[38].as_array()[index],
                data[39].as_array()[index],
                data[40].as_array()[index],
                data[41].as_array()[index],
                data[42].as_array()[index],
                data[43].as_array()[index],
                data[44].as_array()[index],
                data[45].as_array()[index],
                data[46].as_array()[index],
                data[47].as_array()[index],
                data[48].as_array()[index],
                data[49].as_array()[index],
                data[50].as_array()[index],
                data[51].as_array()[index],
                data[52].as_array()[index],
                data[53].as_array()[index],
                data[54].as_array()[index],
                data[55].as_array()[index],
                data[56].as_array()[index],
                data[57].as_array()[index],
                data[58].as_array()[index],
                data[59].as_array()[index],
                data[60].as_array()[index],
                data[61].as_array()[index],
                data[62].as_array()[index],
                data[63].as_array()[index],
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
            transpose_column(&data, 16),
            transpose_column(&data, 17),
            transpose_column(&data, 18),
            transpose_column(&data, 19),
            transpose_column(&data, 20),
            transpose_column(&data, 21),
            transpose_column(&data, 22),
            transpose_column(&data, 23),
            transpose_column(&data, 24),
            transpose_column(&data, 25),
            transpose_column(&data, 26),
            transpose_column(&data, 27),
            transpose_column(&data, 28),
            transpose_column(&data, 29),
            transpose_column(&data, 30),
            transpose_column(&data, 31),
            transpose_column(&data, 32),
            transpose_column(&data, 33),
            transpose_column(&data, 34),
            transpose_column(&data, 35),
            transpose_column(&data, 36),
            transpose_column(&data, 37),
            transpose_column(&data, 38),
            transpose_column(&data, 39),
            transpose_column(&data, 40),
            transpose_column(&data, 41),
            transpose_column(&data, 42),
            transpose_column(&data, 43),
            transpose_column(&data, 44),
            transpose_column(&data, 45),
            transpose_column(&data, 46),
            transpose_column(&data, 47),
            transpose_column(&data, 48),
            transpose_column(&data, 49),
            transpose_column(&data, 50),
            transpose_column(&data, 51),
            transpose_column(&data, 52),
            transpose_column(&data, 53),
            transpose_column(&data, 54),
            transpose_column(&data, 55),
            transpose_column(&data, 56),
            transpose_column(&data, 57),
            transpose_column(&data, 58),
            transpose_column(&data, 59),
            transpose_column(&data, 60),
            transpose_column(&data, 61),
            transpose_column(&data, 62),
            transpose_column(&data, 63),
        ]
    }

    #[inline]
    pub(crate) fn replace_const_backend<const INDEX: usize>(self, value: u8) -> Self {
        /// Because of limitations in const generics, this must have separate
        /// branches per index. This macro is used to avoid duplication.
        macro_rules! use_indices {
      (
        0..64 = [$($INDEX:literal),*],
        0..32 = [$($LOW_INDEX:literal),*],
        32..64 = [$($HIGH_INDEX:literal),*]
      ) => {
        pick! {
          if #[cfg(target_feature = "avx512bw")] {
            Self(_mm512_mask_set1_epi8(
                self.0,
                const { 1 << INDEX },
                value.cast_signed(),
            ))
          } else {
            let [self_a, self_b] = cast::<u8x64, [u8x32; 2]>(self);

            if const { INDEX < 32 } {
              let result_a = match INDEX {
                $($LOW_INDEX => self_a.replace_const::<$LOW_INDEX>(value),)*
                32.. => unreachable!(),
              };

              cast([result_a, self_b])
            } else {
              let result_b = match INDEX {
                $($HIGH_INDEX => self_b.replace_const::<{ $HIGH_INDEX - 32 }>(value),)*
                ..32 | 64.. => unreachable!(),
              };

              cast([self_a, result_b])
            }
          }
        }
      };
    }
        use_indices! {
          0..64 = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
            48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63
          ],
          0..32 = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31
          ],
          32..64 = [
            32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
            55, 56, 57, 58, 59, 60, 61, 62, 63
          ]
        }
    }

    #[inline]
    pub(crate) fn extract_const_backend<const INDEX: usize>(self) -> u8 {
        /// Because of limitations in const generics, this must have separate
        /// branches per index. This macro is used to avoid duplication.
        macro_rules! use_indices {
      (
        0..64 = [$($INDEX:literal),*],
        0..32 = [$($LOW_INDEX:literal),*],
        32..64 = [$($HIGH_INDEX:literal),*]
      ) => {
        pick! {
          if #[cfg(all(target_feature = "avx512f", target_feature = "avx512bw"))] {
            match INDEX {
              $($INDEX => {
                let quarter = u8x16(_mm512_extracti32x4_epi32::<{ $INDEX / 16 }>(self.0));
                quarter.extract_const::<{ $INDEX % 16 }>()
              })*
              64.. => unreachable!()
            }
          } else {
            let [self_a, self_b] = cast::<u8x64, [u8x32; 2]>(self);

            if const { INDEX < 32 } {
              match INDEX {
                $($LOW_INDEX => self_a.extract_const::<$LOW_INDEX>(),)*
                32.. => unreachable!(),
              }
            } else {
              match INDEX {
                $($HIGH_INDEX => self_b.extract_const::<{ $HIGH_INDEX - 32 }>(),)*
                ..32 | 64.. => unreachable!(),
              }
            }
          }
        }
      };
    }
        use_indices! {
          0..64 = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
            48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63
          ],
          0..32 = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31
          ],
          32..64 = [
            32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
            55, 56, 57, 58, 59, 60, 61, 62, 63
          ]
        }
    }
}
}

safe_target_feature! {
// Backend of SIMD integer functions.
impl u8x64 {
    #[inline]
    pub(crate) fn neg_backend(self) -> Self {
        pick! {
            if #[cfg(target_feature = "avx512bw")] {
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
          if #[cfg(target_feature="avx512bw")] {
            Self(_mm512_xor_si512(self.0, _mm512_set1_epi8(-1)))
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
          if #[cfg(target_feature="avx512bw")] {
            Self(_mm512_add_epi8(self.0, rhs.0))
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
          if #[cfg(target_feature="avx512bw")] {
            Self(_mm512_sub_epi8(self.0, rhs.0))
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
        // There is no `_mm512_mullo_epi8` intrinsic, so there is no `avx512bw`
        // optimization. This technically can be done explicitly by converting to
        // `i16` then converting back after multiplication, but that may not
        // actually be faster than auto-vectorization.
        let [self_a, self_b]: [u8x32; 2] = cast(self);
        let [rhs_a, rhs_b]: [u8x32; 2] = cast(rhs);
        cast([self_a * rhs_a, self_b * rhs_b])
    }

    #[inline]
    pub(crate) fn bitand_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="avx512bw")] {
            Self(_mm512_and_si512(self.0, rhs.0))
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
          if #[cfg(target_feature="avx512bw")] {
            Self(_mm512_or_si512(self.0, rhs.0))
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
          if #[cfg(target_feature="avx512bw")] {
            Self(_mm512_xor_si512(self.0, rhs.0))
          } else {
            Self(Inner {
              lo: self.0.lo ^ rhs.0.lo,
              hi: self.0.hi ^ rhs.0.hi,
            })
          }
        }
    }

    #[inline]
    pub(crate) fn reduce_add_backend(self) -> u8 {
        // `_mm512_reduce_add_epi8` does not exist.
        let array: [u8x32; 2] = cast(self);
        (array[0] + array[1]).reduce_add()
    }

    #[inline]
    pub(crate) fn reduce_mul_backend(self) -> u8 {
        // `_mm512_reduce_mul_epi8` does not exist.
        let array: [u8x32; 2] = cast(self);
        (array[0] * array[1]).reduce_mul()
    }

    #[inline]
    pub(crate) fn bitselect_backend(self, if_one: Self, if_zero: Self) -> Self {
        pick! {
          if #[cfg(target_feature="avx512bw")] {
            Self(_mm512_or_si512(
                _mm512_and_si512(if_one.0, self.0),
                _mm512_andnot_si512(self.0, if_zero.0),
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
        // There's no dedicated `u8` shift instruction, so we split into halves
        // and let `u8x32` handle it (it widens to `u16` internally where
        // supported).
        let [self_a, self_b]: [u8x32; 2] = cast(self);
        let [rhs_a, rhs_b]: [u8x32; 2] = cast(rhs);
        cast([self_a << rhs_a, self_b << rhs_b])
    }

    #[inline]
    pub(crate) fn shl_scalar_backend(self, rhs: u32) -> Self {
        pick! {
          if #[cfg(target_feature="avx512bw")] {
            let values: u16x32 = cast(self);
            let shift = rhs & 7;
            let shifted = values << shift;
            let crossed = u16x32::splat(!((0x00FFu16 << shift) & 0xFF00));
            cast(shifted & crossed)
          } else {
            let [self_a, self_b]: [u8x32; 2] = cast(self);
            cast([self_a << rhs, self_b << rhs])
          }
        }
    }

    #[inline]
    pub(crate) fn shr_backend(self, rhs: Self) -> Self {
        let [self_a, self_b]: [u8x32; 2] = cast(self);
        let [rhs_a, rhs_b]: [u8x32; 2] = cast(rhs);
        cast([self_a >> rhs_a, self_b >> rhs_b])
    }

    #[inline]
    pub(crate) fn shr_scalar_backend(self, rhs: u32) -> Self {
        pick! {
          if #[cfg(target_feature="avx512bw")] {
            let values: u16x32 = cast(self);
            let shift = rhs & 7;
            let shifted = values >> shift;
            let crossed = u16x32::splat(!((0xFF00u16 >> shift) & 0x00FF));
            cast(shifted & crossed)
          } else {
            let [self_a, self_b]: [u8x32; 2] = cast(self);
            cast([self_a >> rhs, self_b >> rhs])
          }
        }
    }

    #[inline]
    pub(crate) fn max_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="avx512bw")] {
            Self(_mm512_max_epu8(self.0, rhs.0))
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
          if #[cfg(target_feature="avx512bw")] {
            Self(_mm512_min_epu8(self.0, rhs.0))
          } else {
            Self(Inner {
              lo: self.0.lo.min(rhs.0.lo),
              hi: self.0.hi.min(rhs.0.hi),
            })
          }
        }
    }

    #[inline]
    pub(crate) fn reduce_max_backend(self) -> u8 {
        let array: [u8x32; 2] = cast(self);
        array[0].max(array[1]).reduce_max()
    }

    #[inline]
    pub(crate) fn reduce_min_backend(self) -> u8 {
        let array: [u8x32; 2] = cast(self);
        array[0].min(array[1]).reduce_min()
    }

    #[inline]
    pub(crate) fn unbounded_shl_backend(self, rhs: Self) -> Self {
        let [self_a, self_b] = cast::<u8x64, [u8x32; 2]>(self);
        let [rhs_a, rhs_b] = cast::<u8x64, [u8x32; 2]>(rhs);
        cast([self_a.unbounded_shl(rhs_a), self_b.unbounded_shl(rhs_b)])
    }

    #[inline]
    pub(crate) fn unbounded_shl_scalar_backend(self, rhs: u32) -> Self {
        pick! {
          if #[cfg(target_feature="avx512bw")] {
            let values: u16x32 = cast(self);
            let shifted = values.unbounded_shl_scalar(rhs);
            let crossed = u16x32::splat(!(0x00FFu16.wrapping_shl(rhs) & 0xFF00));
            cast(shifted & crossed)
          } else {
            let [self_a, self_b] = cast::<u8x64, [u8x32; 2]>(self);
            cast([self_a.unbounded_shl_scalar(rhs), self_b.unbounded_shl_scalar(rhs)])
          }
        }
    }

    #[inline]
    pub(crate) fn unbounded_shr_backend(self, rhs: Self) -> Self {
        let [self_a, self_b] = cast::<u8x64, [u8x32; 2]>(self);
        let [rhs_a, rhs_b] = cast::<u8x64, [u8x32; 2]>(rhs);
        cast([self_a.unbounded_shr(rhs_a), self_b.unbounded_shr(rhs_b)])
    }

    #[inline]
    pub(crate) fn unbounded_shr_scalar_backend(self, rhs: u32) -> Self {
        pick! {
          if #[cfg(target_feature="avx512bw")] {
            let values: u16x32 = cast(self);
            let shifted = values.unbounded_shr_scalar(rhs);
            let crossed = u16x32::splat(!(0xFF00u16.wrapping_shr(rhs) & 0x00FF));
            cast(shifted & crossed)
          } else {
            let [self_a, self_b] = cast::<u8x64, [u8x32; 2]>(self);
            cast([self_a.unbounded_shr_scalar(rhs), self_b.unbounded_shr_scalar(rhs)])
          }
        }
    }

    #[inline]
    pub(crate) fn saturating_add_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="avx512bw")] {
            Self(_mm512_adds_epu8(self.0, rhs.0))
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
          if #[cfg(target_feature="avx512bw")] {
            Self(_mm512_subs_epu8(self.0, rhs.0))
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
    pub(crate) fn mul_keep_low_high_backend(self, rhs: Self) -> (Self, Self) {
        // There is no `_mm512_mullo_epi8`/`_mm512_mulhi_epu8` intrinsic, so there
        // is no `avx512bw` optimization.

        let [self_a, self_b] = cast::<u8x64, [u8x32; 2]>(self);
        let [rhs_a, rhs_b] = cast::<u8x64, [u8x32; 2]>(rhs);

        let result_a = self_a.mul_keep_low_high(rhs_a);
        let result_b = self_b.mul_keep_low_high(rhs_b);
        (
            cast([result_a.0, result_b.0]),
            cast([result_a.1, result_b.1]),
        )
    }

    #[inline]
    pub(crate) fn mul_keep_high_backend(self, rhs: Self) -> Self {
        // There is no `_mm512_mulhi_epu8` intrinsic, so there is no `avx512bw`
        // optimization.

        let [self_a, self_b] = cast::<u8x64, [u8x32; 2]>(self);
        let [rhs_a, rhs_b] = cast::<u8x64, [u8x32; 2]>(rhs);

        cast([self_a.mul_keep_high(rhs_a), self_b.mul_keep_high(rhs_b)])
    }
}
}

safe_target_feature! {
// Backend of SIMD unsigned integer functions.
impl u8x64 {
    #[inline]
    pub(crate) fn saturating_add_signed_backend(self, rhs: i8x64) -> Self {
        pick! {
            if #[cfg(target_feature="avx512bw")] {
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
