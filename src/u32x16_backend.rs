use crate::backend_imports::*;

#[cfg(not(target_feature = "avx512f"))]
#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub(crate) struct Inner {
    pub lo: u32x8,
    pub hi: u32x8,
}

safe_target_feature! {
impl SimdBackend for u32x16 {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        type Inner = __m512i;
      } else {
        type Inner = Inner;
      }
    }

    #[inline]
    pub(crate) fn simd_eq_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="avx512f")] {
            Self(_mm512_maskz_mov_epi32(
              _mm512_cmp_epu32_mask::<{_MM_CMPINT_EQ}>(self.0, rhs.0),
              _mm512_set1_epi32(-1),
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
          if #[cfg(target_feature="avx512f")] {
            Self(_mm512_maskz_mov_epi32(
              _mm512_cmp_epu32_mask::<{_MM_CMPINT_NE}>(self.0, rhs.0),
              _mm512_set1_epi32(-1),
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
          if #[cfg(target_feature="avx512f")] {
            Self(_mm512_maskz_mov_epi32(
              _mm512_cmp_epu32_mask::<{_MM_CMPINT_LT}>(self.0, rhs.0),
              _mm512_set1_epi32(-1),
            ))
          } else {
            Self(Inner {
              lo: rhs.0.lo.simd_gt(self.0.lo),
              hi: rhs.0.hi.simd_gt(self.0.hi),
            })
          }
        }
    }

    #[inline]
    pub(crate) fn simd_gt_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="avx512f")] {
            Self(_mm512_maskz_mov_epi32(
              _mm512_cmp_epu32_mask::<{_MM_CMPINT_NLE}>(self.0, rhs.0),
              _mm512_set1_epi32(-1),
            ))
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
          if #[cfg(target_feature="avx512f")] {
            Self(_mm512_maskz_mov_epi32(
              _mm512_cmp_epu32_mask::<{_MM_CMPINT_LE}>(self.0, rhs.0),
              _mm512_set1_epi32(-1),
            ))
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
          if #[cfg(target_feature="avx512f")] {
            Self(_mm512_maskz_mov_epi32(
              _mm512_cmp_epu32_mask::<{_MM_CMPINT_NLT}>(self.0, rhs.0),
              _mm512_set1_epi32(-1),
            ))
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
          if #[cfg(target_feature="avx512bw")] {
            Self(_mm512_mask_blend_epi8(_mm512_movepi8_mask(self.0), if_false.0,if_true.0))
          } else {
            let [self_lo, self_hi] = cast::<u32x16, [u32x8; 2]>(self);
            let [if_true_lo, if_true_hi] = cast::<u32x16, [u32x8; 2]>(if_true);
            let [if_false_lo, if_false_hi] = cast::<u32x16, [u32x8; 2]>(if_false);
            cast([
              self_lo.select(if_true_lo, if_false_lo),
              self_hi.select(if_true_hi, if_false_hi),
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn to_bitmask_backend(self) -> u32 {
        pick! {
          if #[cfg(target_feature="avx512dq")] {
            _mm512_movepi32_mask(self.0) as u32
          } else {
            f32x16::to_bitmask(cast(self))
          }
        }
    }

    #[inline]
    pub(crate) fn any_backend(self) -> bool {
        pick! {
          if #[cfg(target_feature="avx512dq")] {
            _mm512_movepi32_mask(self.0) != 0
          } else {
            let [a, b]: [u32x8; 2] = cast(self);
            (a | b).any()
          }
        }
    }

    #[inline]
    pub(crate) fn all_backend(self) -> bool {
        pick! {
          if #[cfg(target_feature="avx512dq")] {
            _mm512_movepi32_mask(self.0) == 0xFFFF
          } else {
            let [a, b]: [u32x8; 2] = cast(self);
            (a & b).all()
          }
        }
    }

    #[inline]
    pub(crate) fn interleave_low_backend(self, other: Self) -> Self {
        pick! {
          if #[cfg(target_feature="avx512f")] {
            // `_mm512_unpacklo_epi32` cannot be used because it acts within each
            // 128-bit lane, which is a different operation.
            let [aa, _]: [u32x8; 2] = cast(self);
            let [ba, _]: [u32x8; 2] = cast(other);
            cast([aa.interleave_low(ba), aa.interleave_high(ba)])
          } else {
            Self(Inner { lo: self.0.lo.interleave_low(other.0.lo), hi: self.0.lo.interleave_high(other.0.lo) })
          }
        }
    }

    #[inline]
    pub(crate) fn interleave_high_backend(self, other: Self) -> Self {
        pick! {
          if #[cfg(target_feature="avx512f")] {
            // `_mm512_unpackhi_epi32` cannot be used because it acts within each
            // 128-bit lane, which is a different operation.
            let [_, ab]: [u32x8; 2] = cast(self);
            let [_, bb]: [u32x8; 2] = cast(other);
            cast([ab.interleave_low(bb), ab.interleave_high(bb)])
          } else {
            Self(Inner { lo: self.0.hi.interleave_low(other.0.hi), hi: self.0.hi.interleave_high(other.0.hi) })
          }
        }
    }

    #[inline]
    pub(crate) fn shuffle_backend(self, indices: u32x16) -> Self {
        pick! {
          if #[cfg(all(target_feature = "avx512f"))] {
            Self(_mm512_permutexvar_epi32(indices.0, self.0))
          } else {
            let self_halfs = cast::<u32x16, [u32x8; 2]>(self);
            let [indices_a, indices_b] = cast::<u32x16, [u32x8; 2]>(indices);

            cast([self_halfs.shuffle(indices_a), self_halfs.shuffle(indices_b)])
          }
        }
    }

    #[inline]
    pub(crate) fn shuffle_zeroing_backend(self, indices: u32x16) -> Self {
        pick! {
          if #[cfg(all(target_feature = "avx512f"))] {
            self.shuffle(indices) & indices.simd_lt(16)
          } else {
            let self_halfs = cast::<u32x16, [u32x8; 2]>(self);
            let [indices_a, indices_b] = cast::<u32x16, [u32x8; 2]>(indices);

            cast([self_halfs.shuffle_zeroing(indices_a), self_halfs.shuffle_zeroing(indices_b)])
          }
        }
    }

    #[inline]
    pub(crate) fn shuffle_wrapping_backend(self, indices: u32x16) -> Self {
        pick! {
          if #[cfg(all(target_feature = "avx512f"))] {
            // `avx512` shuffle intrinsics are wrapping
            self.shuffle(indices)
          } else {
            let self_halfs = cast::<u32x16, [u32x8; 2]>(self);
            let [indices_a, indices_b] = cast::<u32x16, [u32x8; 2]>(indices);

            cast([self_halfs.shuffle_wrapping(indices_a), self_halfs.shuffle_wrapping(indices_b)])
          }
        }
    }

    #[inline]
    pub(crate) fn shuffle_2_backend(inputs: [Self; 2], indices: u32x16) -> u32x16 {
        pick! {
          if #[cfg(all(target_feature = "avx512f"))] {
            u32x16(_mm512_permutex2var_epi32(inputs[0].0, indices.0, inputs[1].0))
          } else {
            inputs[0].shuffle_zeroing(indices) | inputs[1].shuffle_zeroing(indices - 16)
          }
        }
    }

    #[inline]
    pub(crate) fn shuffle_zeroing_2_backend(inputs: [Self; 2], indices: u32x16) -> u32x16 {
        pick! {
          if #[cfg(all(target_feature = "avx512f"))] {
            inputs.shuffle(indices) & indices.simd_lt(32)
          } else {
            inputs.shuffle(indices)
          }
        }
    }

    #[inline]
    pub(crate) fn shuffle_wrapping_2_backend(inputs: [Self; 2], indices: u32x16) -> u32x16 {
        pick! {
          if #[cfg(all(target_feature = "avx512f", target_feature = "avx512vl"))] {
            // `avx512` shuffle intrinsics are wrapping
            inputs.shuffle(indices)
          } else {
            inputs.shuffle(indices & 31)
          }
        }
    }

    #[inline]
    pub(crate) fn shuffle_3_backend(inputs: [Self; 3], indices: u32x16) -> u32x16 {
        [inputs[0], inputs[1]].shuffle_zeroing(indices) | inputs[2].shuffle_zeroing(indices - 32)
    }

    #[inline]
    pub(crate) fn shuffle_zeroing_3_backend(inputs: [Self; 3], indices: u32x16) -> u32x16 {
        inputs.shuffle(indices)
    }

    #[inline]
    pub(crate) fn shuffle_wrapping_3_backend(inputs: [Self; 3], indices: u32x16) -> u32x16 {
        inputs.shuffle(indices % 48)
    }

    #[inline]
    pub(crate) fn shuffle_4_backend(inputs: [Self; 4], indices: u32x16) -> u32x16 {
        [inputs[0], inputs[1]].shuffle_zeroing(indices)
            | [inputs[2], inputs[3]].shuffle_zeroing(indices - 32)
    }

    #[inline]
    pub(crate) fn shuffle_zeroing_4_backend(inputs: [Self; 4], indices: u32x16) -> u32x16 {
        inputs.shuffle(indices)
    }

    #[inline]
    pub(crate) fn shuffle_wrapping_4_backend(inputs: [Self; 4], indices: u32x16) -> u32x16 {
        inputs.shuffle(indices & 63)
    }

    #[inline]
    pub(crate) fn transpose_backend(data: [Self; 16]) -> [Self; 16] {
        // Can this be optimized?

        #[inline(always)]
        fn transpose_column(data: &[u32x16; 16], index: usize) -> u32x16 {
            u32x16::from_array([
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
    pub(crate) fn replace_const_backend<const INDEX: usize>(self, value: u32) -> Self {
        /// Because of limitations in const generics, this must have separate
        /// branches per index. This macro is used to avoid duplication.
        macro_rules! use_indices {
      (
        0..16 = [$($INDEX:literal),*],
        0..8 = [$($LOW_INDEX:literal),*],
        8..16 = [$($HIGH_INDEX:literal),*]
      ) => {
        pick! {
          if #[cfg(target_feature = "avx512f")] {
            Self(_mm512_mask_set1_epi32(
                self.0,
                const { 1 << INDEX },
                value.cast_signed(),
            ))
          } else {
            let [self_a, self_b] = cast::<u32x16, [u32x8; 2]>(self);

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
    pub(crate) fn extract_const_backend<const INDEX: usize>(self) -> u32 {
        /// Because of limitations in const generics, this must have separate
        /// branches per index. This macro is used to avoid duplication.
        macro_rules! use_indices {
      (
        0..16 = [$($INDEX:literal),*],
        0..8 = [$($LOW_INDEX:literal),*],
        8..16 = [$($HIGH_INDEX:literal),*]
      ) => {
        pick! {
          if #[cfg(all(target_feature = "avx512f", target_feature = "avx512bw"))] {
            match INDEX {
              $($INDEX => {
                let quarter = u32x4(_mm512_extracti32x4_epi32::<{ $INDEX / 4 }>(self.0));
                quarter.extract_const::<{ $INDEX % 4 }>()
              })*
              16.. => unreachable!()
            }
          } else {
            let [self_a, self_b] = cast::<u32x16, [u32x8; 2]>(self);

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
impl u32x16 {
    #[inline]
    pub(crate) fn neg_backend(self) -> Self {
        pick! {
            if #[cfg(target_feature = "avx512f")] {
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
          if #[cfg(target_feature="avx512f")] {
            Self(_mm512_xor_si512(self.0, _mm512_set1_epi32(-1)))
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
          if #[cfg(target_feature="avx512f")] {
            Self(_mm512_add_epi32(self.0, rhs.0))
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
          if #[cfg(target_feature="avx512f")] {
            Self(_mm512_sub_epi32(self.0, rhs.0))
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
          if #[cfg(target_feature="avx512f")] {
            Self(_mm512_mullo_epi32(self.0, rhs.0))
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
          if #[cfg(target_feature="avx512f")] {
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
        if #[cfg(target_feature="avx512f")] {
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
          if #[cfg(target_feature="avx512f")] {
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
    pub(crate) fn reduce_add_backend(self) -> u32 {
        let array: [u32x8; 2] = cast(self);
        (array[0] + array[1]).reduce_add()
    }

    #[inline]
    pub(crate) fn reduce_mul_backend(self) -> u32 {
        let array: [u32x8; 2] = cast(self);
        (array[0] * array[1]).reduce_mul()
    }

    #[inline]
    pub(crate) fn bitselect_backend(self, if_one: Self, if_zero: Self) -> Self {
        pick! {
          if #[cfg(target_feature="avx512f")] {
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
    pub(crate) fn shl_backend(self, rhs: u32x16) -> Self {
        pick! {
          if #[cfg(target_feature="avx512f")] {
            let shift_by = _mm512_and_si512(rhs.0, _mm512_set1_epi32(31));
            Self(_mm512_sllv_epi32(self.0, shift_by))
          } else {
            Self(Inner {
              lo: self.0.lo << rhs.0.lo,
              hi: self.0.hi << rhs.0.hi,
            })
          }
        }
    }

    #[inline]
    pub(crate) fn shl_scalar_backend(self, rhs: u32) -> Self {
        pick! {
          if #[cfg(target_feature="avx512f")] {
            // Use `rhs % 32` to perform wrapping shift and not unbounded shift.
            let shift = rhs & 31;
            Self(_mm512_sllv_epi32(self.0, _mm512_set1_epi32(shift as i32)))
          } else {
            Self(Inner {
              lo: self.0.lo << rhs,
              hi: self.0.hi << rhs,
            })
          }
        }
    }

    #[inline]
    pub(crate) fn shr_backend(self, rhs: u32x16) -> Self {
        pick! {
          if #[cfg(target_feature="avx512f")] {
            let shift_by = _mm512_and_si512(rhs.0, _mm512_set1_epi32(31));
            Self(_mm512_srlv_epi32(self.0, shift_by ))
          } else {
            Self(Inner {
              lo: self.0.lo >> rhs.0.lo,
              hi: self.0.hi >> rhs.0.hi,
            })
          }
        }
    }

    #[inline]
    pub(crate) fn shr_scalar_backend(self, rhs: u32) -> Self {
        pick! {
          if #[cfg(target_feature="avx512f")] {
            // Use `rhs % 32` to perform wrapping shift and not unbounded shift.
            let shift = rhs & 31;
            Self(_mm512_srlv_epi32(self.0, _mm512_set1_epi32(shift as i32)))
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
          if #[cfg(target_feature="avx512f")] {
            Self(_mm512_max_epu32(self.0, rhs.0))
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
          if #[cfg(target_feature="avx512f")] {
            Self(_mm512_min_epu32(self.0, rhs.0))
          } else {
            Self(Inner {
              lo: self.0.lo.min(rhs.0.lo),
              hi: self.0.hi.min(rhs.0.hi),
            })
          }
        }
    }

    #[inline]
    pub(crate) fn reduce_max_backend(self) -> u32 {
        let array: [u32x8; 2] = cast(self);
        array[0].max(array[1]).reduce_max()
    }

    #[inline]
    pub(crate) fn reduce_min_backend(self) -> u32 {
        let array: [u32x8; 2] = cast(self);
        array[0].min(array[1]).reduce_min()
    }

    #[inline]
    pub(crate) fn unbounded_shl_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="avx512f")] {
            Self(_mm512_sllv_epi32(self.0, rhs.0))
          } else {
            Self(Inner {
              lo: self.0.lo.unbounded_shl(rhs.0.lo),
              hi: self.0.hi.unbounded_shl(rhs.0.hi),
            })
          }
        }
    }

    #[inline]
    pub(crate) fn unbounded_shl_scalar_backend(self, rhs: u32) -> Self {
        pick! {
          if #[cfg(target_feature="avx512f")] {
            Self(_mm512_sllv_epi32(self.0, _mm512_set1_epi32(rhs as i32)))
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
          if #[cfg(target_feature="avx512f")] {
            Self(_mm512_srlv_epi32(self.0, rhs.0))
          } else {
            Self(Inner {
              lo: self.0.lo.unbounded_shr(rhs.0.lo),
              hi: self.0.hi.unbounded_shr(rhs.0.hi),
            })
          }
        }
    }

    #[inline]
    pub(crate) fn unbounded_shr_scalar_backend(self, rhs: u32) -> Self {
        pick! {
          if #[cfg(target_feature="avx512f")] {
            Self(_mm512_srlv_epi32(self.0, _mm512_set1_epi32(rhs as i32)))
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
          if #[cfg(target_feature="avx512f")] {
            let result = self + rhs;
            let overflow = result.simd_lt(self);
            // Return `MAX` (all bits set) if overflow occurs.
            result | overflow
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
          if #[cfg(target_feature="avx512f")] {
            let result = self - rhs;
            let no_overflow = result.simd_le(self);
            // Return `0` (no bits set) if overflow occurs.
            result & no_overflow
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
        pick! {
          if #[cfg(all(target_feature="avx512f", target_feature="avx512dq"))] {
            let even_wide_mul = _mm512_mul_epu32(self.0, rhs.0);
            let odd_wide_mul = _mm512_mul_epu32(
              _mm512_shuffle_epi32::<0b_00_11_00_01>(self.0),
              _mm512_shuffle_epi32::<0b_00_11_00_01>(rhs.0),
            );

            let ll_hh_1 = _mm512_unpacklo_epi32(even_wide_mul, odd_wide_mul);
            let ll_hh_2 = _mm512_unpackhi_epi32(even_wide_mul, odd_wide_mul);
            (
              Self(_mm512_unpacklo_epi64(ll_hh_1, ll_hh_2)),
              Self(_mm512_unpackhi_epi64(ll_hh_1, ll_hh_2)),
            )
          } else {
            let [self_a, self_b] = cast::<u32x16, [u32x8; 2]>(self);
            let [rhs_a, rhs_b] = cast::<u32x16, [u32x8; 2]>(rhs);

            let result_a = self_a.mul_keep_low_high(rhs_a);
            let result_b = self_b.mul_keep_low_high(rhs_b);
            (
              cast([result_a.0, result_b.0]),
              cast([result_a.1, result_b.1]),
            )
          }
        }
    }

    #[inline]
    pub(crate) fn mul_keep_high_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="avx512dq")] {
            let alo = _mm512_extracti32x8_epi32::<0>(self.0);
            let ahi = _mm512_extracti32x8_epi32::<1>(self.0);
            let blo = _mm512_extracti32x8_epi32::<0>(rhs.0);
            let bhi = _mm512_extracti32x8_epi32::<1>(rhs.0);

            let lo_res: __m256i = {
              let a8 = u32x8(alo);
              let b8 = u32x8(blo);
              a8.mul_keep_high(b8).0
            };
            let hi_res: __m256i = {
              let a8 = u32x8(ahi);
              let b8 = u32x8(bhi);
              a8.mul_keep_high(b8).0
            };

            let zero = _mm512_setzero_si512();
            let with_lo = _mm512_inserti32x8::<0>(zero, lo_res);
            let combined = _mm512_inserti32x8::<1>(with_lo, hi_res);

            Self(combined)
          } else {
            let [self_lo, self_hi] = cast::<u32x16, [u32x8; 2]>(self);
            let [rhs_lo, rhs_hi] = cast::<u32x16, [u32x8; 2]>(rhs);
            cast([self_lo.mul_keep_high(rhs_lo), self_hi.mul_keep_high(rhs_hi)])
          }
        }
    }
}
}

safe_target_feature! {
// Backend of SIMD unsigned integer functions.
impl u32x16 {
    #[inline]
    pub(crate) fn saturating_add_signed_backend(self, rhs: i32x16) -> Self {
        pick! {
            if #[cfg(target_feature="avx512f")] {
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
impl u32x16 {
    #[inline]
    pub(crate) fn from_u16x16_backend(v: u16x16) -> Self {
        pick! {
          if #[cfg(target_feature = "avx512f")] {
            Self(_mm512_cvtepu16_epi32(v.0))
          } else if #[cfg(target_feature = "avx2")] {
            let lo: __m128i = _mm256_extractf128_si256::<0>(v.0);
            let hi: __m128i = _mm256_extractf128_si256::<1>(v.0);
            Self(Inner {
              lo: u32x8(_mm256_cvtepu16_epi32(lo)),
              hi: u32x8(_mm256_cvtepu16_epi32(hi)),
            })
          } else if #[cfg(target_feature = "sse2")] {
            Self(Inner {
              lo: u32x8(crate::u32x8_backend::Inner {
                lo: u32x4(_mm_srli_epi32::<16>(_mm_unpacklo_epi16(v.0.lo.0, v.0.lo.0))),
                hi: u32x4(_mm_srli_epi32::<16>(_mm_unpackhi_epi16(v.0.lo.0, v.0.lo.0))),
              }),
              hi: u32x8(crate::u32x8_backend::Inner {
                lo: u32x4(_mm_srli_epi32::<16>(_mm_unpacklo_epi16(v.0.hi.0, v.0.hi.0))),
                hi: u32x4(_mm_srli_epi32::<16>(_mm_unpackhi_epi16(v.0.hi.0, v.0.hi.0))),
              }),
            })
          } else {
            // Portable fallback
            let arr = v.as_array();
            Self::from_array([
              arr[0] as u32,  arr[1] as u32,  arr[2] as u32,  arr[3] as u32,
              arr[4] as u32,  arr[5] as u32,  arr[6] as u32,  arr[7] as u32,
              arr[8] as u32,  arr[9] as u32,  arr[10] as u32, arr[11] as u32,
              arr[12] as u32, arr[13] as u32, arr[14] as u32, arr[15] as u32,
            ])
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
}
}
