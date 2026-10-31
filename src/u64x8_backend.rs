use crate::backend_imports::*;

#[cfg(not(target_feature = "avx512f"))]
#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub(crate) struct Inner {
    pub lo: u64x4,
    pub hi: u64x4,
}

safe_target_feature! {
impl SimdBackend for u64x8 {
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
            Self(_mm512_maskz_mov_epi64(
              _mm512_cmp_epu64_mask::<{_MM_CMPINT_EQ}>(self.0, rhs.0),
              _mm512_set1_epi64(-1),
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
            Self(_mm512_maskz_mov_epi64(
              _mm512_cmp_epu64_mask::<{_MM_CMPINT_NE}>(self.0, rhs.0),
              _mm512_set1_epi64(-1),
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
            Self(_mm512_maskz_mov_epi64(
              _mm512_cmp_epu64_mask::<{_MM_CMPINT_LT}>(self.0, rhs.0),
              _mm512_set1_epi64(-1),
            ))
          } else {
            Self(Inner {
              lo: self.0.lo.simd_lt(rhs.0.lo),
              hi: self.0.hi.simd_lt(rhs.0.hi),
            })
          }
        }
    }

    #[inline]
    pub(crate) fn simd_gt_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="avx512f")] {
            Self(_mm512_maskz_mov_epi64(
              _mm512_cmp_epu64_mask::<{_MM_CMPINT_NLE}>(self.0, rhs.0),
              _mm512_set1_epi64(-1),
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
            Self(_mm512_maskz_mov_epi64(
              _mm512_cmp_epu64_mask::<{_MM_CMPINT_LE}>(self.0, rhs.0),
              _mm512_set1_epi64(-1),
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
            Self(_mm512_maskz_mov_epi64(
              _mm512_cmp_epu64_mask::<{_MM_CMPINT_NLT}>(self.0, rhs.0),
              _mm512_set1_epi64(-1),
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
            let [self_lo, self_hi] = cast::<u64x8, [u64x4; 2]>(self);
            let [if_true_lo, if_true_hi] = cast::<u64x8, [u64x4; 2]>(if_true);
            let [if_false_lo, if_false_hi] = cast::<u64x8, [u64x4; 2]>(if_false);
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
            // use f64 move_mask since it is the same size as i64
            _mm512_movepi64_mask(self.0) as u32
          } else {
            f64x8::to_bitmask(cast(self))
          }
        }
    }

    #[inline]
    pub(crate) fn any_backend(self) -> bool {
        pick! {
          if #[cfg(target_feature="avx512dq")] {
            _mm512_movepi64_mask(self.0) != 0
          } else {
            let [a, b]: [i64x4; 2] = cast(self);
            (a | b).any()
          }
        }
    }

    #[inline]
    pub(crate) fn all_backend(self) -> bool {
        pick! {
          if #[cfg(target_feature="avx512dq")] {
            _mm512_movepi64_mask(self.0) == 0b11111111
          } else {
            let [a, b]: [i64x4; 2] = cast(self);
            (a & b).all()
          }
        }
    }

    #[inline]
    pub(crate) fn interleave_low_backend(self, other: Self) -> Self {
        pick! {
          if #[cfg(target_feature="avx512f")] {
            // `_mm512_unpacklo_epi64` cannot be used because it acts within each
            // 128-bit lane, which is a different operation.
            let [aa, _]: [u64x4; 2] = cast(self);
            let [ba, _]: [u64x4; 2] = cast(other);
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
            // `_mm512_unpackhi_epi64` cannot be used because it acts within each
            // 128-bit lane, which is a different operation.
            let [_, ab]: [u64x4; 2] = cast(self);
            let [_, bb]: [u64x4; 2] = cast(other);
            cast([ab.interleave_low(bb), ab.interleave_high(bb)])
          } else {
            Self(Inner { lo: self.0.hi.interleave_low(other.0.hi), hi: self.0.hi.interleave_high(other.0.hi) })
          }
        }
    }

    #[inline]
    pub(crate) fn shuffle_backend(self, indices: u64x8) -> Self {
        pick! {
          if #[cfg(all(target_feature = "avx512f"))] {
            Self(_mm512_permutexvar_epi64(indices.0, self.0))
          } else {
            let self_halfs = cast::<u64x8, [u64x4; 2]>(self);
            let [indices_a, indices_b] = cast::<u64x8, [u64x4; 2]>(indices);

            cast([self_halfs.shuffle(indices_a), self_halfs.shuffle(indices_b)])
          }
        }
    }

    #[inline]
    pub(crate) fn shuffle_zeroing_backend(self, indices: u64x8) -> Self {
        pick! {
          if #[cfg(all(target_feature = "avx512f"))] {
            self.shuffle(indices) & indices.simd_lt(8)
          } else {
            let self_halfs = cast::<u64x8, [u64x4; 2]>(self);
            let [indices_a, indices_b] = cast::<u64x8, [u64x4; 2]>(indices);

            cast([self_halfs.shuffle_zeroing(indices_a), self_halfs.shuffle_zeroing(indices_b)])
          }
        }
    }

    #[inline]
    pub(crate) fn shuffle_wrapping_backend(self, indices: u64x8) -> Self {
        pick! {
          if #[cfg(all(target_feature = "avx512f"))] {
            // `avx512` shuffle intrinsics are wrapping
            self.shuffle(indices)
          } else {
            let self_halfs = cast::<u64x8, [u64x4; 2]>(self);
            let [indices_a, indices_b] = cast::<u64x8, [u64x4; 2]>(indices);

            cast([self_halfs.shuffle_wrapping(indices_a), self_halfs.shuffle_wrapping(indices_b)])
          }
        }
    }

    #[inline]
    pub(crate) fn shuffle_2_backend(inputs: [Self; 2], indices: u64x8) -> u64x8 {
        pick! {
          if #[cfg(all(target_feature = "avx512f"))] {
            u64x8(_mm512_permutex2var_epi64(inputs[0].0, indices.0, inputs[1].0))
          } else {
            inputs[0].shuffle_zeroing(indices) | inputs[1].shuffle_zeroing(indices - 8)
          }
        }
    }

    #[inline]
    pub(crate) fn shuffle_zeroing_2_backend(inputs: [Self; 2], indices: u64x8) -> u64x8 {
        pick! {
          if #[cfg(all(target_feature = "avx512f"))] {
            inputs.shuffle(indices) & indices.simd_lt(16)
          } else {
            inputs.shuffle(indices)
          }
        }
    }

    #[inline]
    pub(crate) fn shuffle_wrapping_2_backend(inputs: [Self; 2], indices: u64x8) -> u64x8 {
        pick! {
          if #[cfg(target_feature = "avx512f")] {
            // `avx512` shuffle intrinsics are wrapping
            inputs.shuffle(indices)
          } else {
            inputs.shuffle(indices & 15)
          }
        }
    }

    #[inline]
    pub(crate) fn shuffle_3_backend(inputs: [Self; 3], indices: u64x8) -> u64x8 {
        [inputs[0], inputs[1]].shuffle_zeroing(indices) | inputs[2].shuffle_zeroing(indices - 16)
    }

    #[inline]
    pub(crate) fn shuffle_zeroing_3_backend(inputs: [Self; 3], indices: u64x8) -> u64x8 {
        inputs.shuffle(indices)
    }

    #[inline]
    pub(crate) fn shuffle_wrapping_3_backend(inputs: [Self; 3], indices: u64x8) -> u64x8 {
        inputs.shuffle(indices % 24)
    }

    #[inline]
    pub(crate) fn shuffle_4_backend(inputs: [Self; 4], indices: u64x8) -> u64x8 {
        [inputs[0], inputs[1]].shuffle_zeroing(indices)
            | [inputs[2], inputs[3]].shuffle_zeroing(indices - 16)
    }

    #[inline]
    pub(crate) fn shuffle_zeroing_4_backend(inputs: [Self; 4], indices: u64x8) -> u64x8 {
        inputs.shuffle(indices)
    }

    #[inline]
    pub(crate) fn shuffle_wrapping_4_backend(inputs: [Self; 4], indices: u64x8) -> u64x8 {
        inputs.shuffle(indices & 31)
    }

    #[inline]
    pub(crate) fn transpose_backend(data: [Self; 8]) -> [Self; 8] {
        // Can this be optimized?

        #[inline(always)]
        fn transpose_column(data: &[u64x8; 8], index: usize) -> u64x8 {
            u64x8::from_array([
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

    #[inline]
    pub(crate) fn replace_const_backend<const INDEX: usize>(self, value: u64) -> Self {
        /// Because of limitations in const generics, this must have separate
        /// branches per index. This macro is used to avoid duplication.
        macro_rules! use_indices {
      (
        0..8 = [$($INDEX:literal),*],
        0..4 = [$($LOW_INDEX:literal),*],
        4..8 = [$($HIGH_INDEX:literal),*]
      ) => {
        pick! {
          if #[cfg(target_feature = "avx512f")] {
            Self(_mm512_mask_set1_epi64(
                self.0,
                const { 1 << INDEX },
                value.cast_signed(),
            ))
          } else {
            let [self_a, self_b] = cast::<u64x8, [u64x4; 2]>(self);

            if const { INDEX < 4 } {
              let result_a = match INDEX {
                $($LOW_INDEX => self_a.replace_const::<$LOW_INDEX>(value),)*
                4.. => unreachable!(),
              };

              cast([result_a, self_b])
            } else {
              let result_b = match INDEX {
                $($HIGH_INDEX => self_b.replace_const::<{ $HIGH_INDEX - 4 }>(value),)*
                ..4 | 8.. => unreachable!(),
              };

              cast([self_a, result_b])
            }
          }
        }
      };
    }
        use_indices! { 0..8 = [0, 1, 2, 3, 4, 5, 6, 7], 0..4 = [0, 1, 2, 3], 4..8 = [4, 5, 6, 7] }
    }

    #[inline]
    pub(crate) fn extract_const_backend<const INDEX: usize>(self) -> u64 {
        /// Because of limitations in const generics, this must have separate
        /// branches per index. This macro is used to avoid duplication.
        macro_rules! use_indices {
      (
        0..8 = [$($INDEX:literal),*],
        0..4 = [$($LOW_INDEX:literal),*],
        4..8 = [$($HIGH_INDEX:literal),*]
      ) => {
        pick! {
          if #[cfg(all(target_feature = "avx512f", target_feature = "avx512bw"))] {
            match INDEX {
              $($INDEX => {
                let quarter = u64x2(_mm512_extracti32x4_epi32::<{ $INDEX / 2 }>(self.0));
                quarter.extract_const::<{ $INDEX % 2 }>()
              })*
              8.. => unreachable!()
            }
          } else {
            let [self_a, self_b] = cast::<u64x8, [u64x4; 2]>(self);

            if const { INDEX < 4 } {
              match INDEX {
                $($LOW_INDEX => self_a.extract_const::<$LOW_INDEX>(),)*
                4.. => unreachable!(),
              }
            } else {
              match INDEX {
                $($HIGH_INDEX => self_b.extract_const::<{ $HIGH_INDEX - 4 }>(),)*
                ..4 | 8.. => unreachable!(),
              }
            }
          }
        }
      };
    }
        use_indices! { 0..8 = [0, 1, 2, 3, 4, 5, 6, 7], 0..4 = [0, 1, 2, 3], 4..8 = [4, 5, 6, 7] }
    }
}
}

safe_target_feature! {
// Backend of SIMD integer functions.
impl u64x8 {
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
            Self(_mm512_xor_si512(self.0, _mm512_set1_epi64(-1)))
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
            Self(_mm512_add_epi64(self.0, rhs.0))
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
            Self(_mm512_sub_epi64(self.0, rhs.0))
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
            let arr1: [u64; 8] = cast(self);
            let arr2: [u64; 8] = cast(rhs);
            cast([
              arr1[0].wrapping_mul(arr2[0]),
              arr1[1].wrapping_mul(arr2[1]),
              arr1[2].wrapping_mul(arr2[2]),
              arr1[3].wrapping_mul(arr2[3]),
              arr1[4].wrapping_mul(arr2[4]),
              arr1[5].wrapping_mul(arr2[5]),
              arr1[6].wrapping_mul(arr2[6]),
              arr1[7].wrapping_mul(arr2[7]),
            ])
          } else {
            Self(Inner { lo: self.0.lo * rhs.0.lo, hi: self.0.hi * rhs.0.hi })
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
    pub(crate) fn reduce_add_backend(self) -> u64 {
        let array: [u64x4; 2] = cast(self);
        (array[0] + array[1]).reduce_add()
    }

    #[inline]
    pub(crate) fn reduce_mul_backend(self) -> u64 {
        let array: [u64x4; 2] = cast(self);
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
    pub(crate) fn shl_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="avx512f")] {
            // Use `rhs % 64` to perform wrapping shift and not unbounded shift.
            let rhs = _mm512_and_si512(rhs.0, _mm512_set1_epi64(63));
            Self(_mm512_sllv_epi64(self.0, rhs))
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
            // Use `rhs % 64` to perform wrapping shift and not unbounded shift.
            let shift = rhs as u64 & 63;
            Self(_mm512_sllv_epi64(self.0, _mm512_set1_epi64(shift as i64)))
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
          if #[cfg(target_feature="avx512f")] {
            // Use `rhs % 64` to perform wrapping shift and not unbounded shift.
            let rhs = _mm512_and_si512(rhs.0, _mm512_set1_epi64(63));
            Self(_mm512_srlv_epi64(self.0, rhs))
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
            // Use `rhs % 64` to perform wrapping shift and not unbounded shift.
            let shift = rhs as u64 & 63;
            Self(_mm512_srlv_epi64(self.0, _mm512_set1_epi64(shift as i64)))
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
            Self(_mm512_max_epu64(self.0, rhs.0))
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
            Self(_mm512_min_epu64(self.0, rhs.0))
          } else {
            Self(Inner {
              lo: self.0.lo.min(rhs.0.lo),
              hi: self.0.hi.min(rhs.0.hi),
            })
          }
        }
    }

    #[inline]
    pub(crate) fn reduce_max_backend(self) -> u64 {
        let array: [u64x4; 2] = cast(self);
        array[0].max(array[1]).reduce_max()
    }

    #[inline]
    pub(crate) fn reduce_min_backend(self) -> u64 {
        let array: [u64x4; 2] = cast(self);
        array[0].min(array[1]).reduce_min()
    }

    #[inline]
    pub(crate) fn unbounded_shl_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="avx512f")] {
            Self(_mm512_sllv_epi64(self.0, rhs.0))
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
            Self(_mm512_sllv_epi64(self.0, _mm512_set1_epi64(rhs as i64)))
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
            Self(_mm512_srlv_epi64(self.0, rhs.0))
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
            Self(_mm512_srlv_epi64(self.0, _mm512_set1_epi64(rhs as i64)))
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
        // TODO(perf): This implementation looks quite bad. Is there a better
        // one? This intentionally avoids `mul_keep_low_high` because getting the
        // high bits of 64-bit multiplication could be slow.

        let self_array = self.to_array();
        let rhs_array = rhs.to_array();

        let result = [
            self_array[0].overflowing_mul(rhs_array[0]),
            self_array[1].overflowing_mul(rhs_array[1]),
            self_array[2].overflowing_mul(rhs_array[2]),
            self_array[3].overflowing_mul(rhs_array[3]),
            self_array[4].overflowing_mul(rhs_array[4]),
            self_array[5].overflowing_mul(rhs_array[5]),
            self_array[6].overflowing_mul(rhs_array[6]),
            self_array[7].overflowing_mul(rhs_array[7]),
        ];
        (
            Self::from_array([
                result[0].0,
                result[1].0,
                result[2].0,
                result[3].0,
                result[4].0,
                result[5].0,
                result[6].0,
                result[7].0,
            ]),
            Self::from_array([
                -(result[0].1 as i64) as u64,
                -(result[1].1 as i64) as u64,
                -(result[2].1 as i64) as u64,
                -(result[3].1 as i64) as u64,
                -(result[4].1 as i64) as u64,
                -(result[5].1 as i64) as u64,
                -(result[6].1 as i64) as u64,
                -(result[7].1 as i64) as u64,
            ]),
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
            (self_array[2] as u128).wrapping_mul(rhs_array[2] as u128),
            (self_array[3] as u128).wrapping_mul(rhs_array[3] as u128),
            (self_array[4] as u128).wrapping_mul(rhs_array[4] as u128),
            (self_array[5] as u128).wrapping_mul(rhs_array[5] as u128),
            (self_array[6] as u128).wrapping_mul(rhs_array[6] as u128),
            (self_array[7] as u128).wrapping_mul(rhs_array[7] as u128),
        ];

        (
            Self::from_array([
                widening_mul[0] as u64,
                widening_mul[1] as u64,
                widening_mul[2] as u64,
                widening_mul[3] as u64,
                widening_mul[4] as u64,
                widening_mul[5] as u64,
                widening_mul[6] as u64,
                widening_mul[7] as u64,
            ]),
            Self::from_array([
                (widening_mul[0] >> 64) as u64,
                (widening_mul[1] >> 64) as u64,
                (widening_mul[2] >> 64) as u64,
                (widening_mul[3] >> 64) as u64,
                (widening_mul[4] >> 64) as u64,
                (widening_mul[5] >> 64) as u64,
                (widening_mul[6] >> 64) as u64,
                (widening_mul[7] >> 64) as u64,
            ]),
        )
    }

    #[inline]
    pub(crate) fn mul_keep_high_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="avx512f")] {
            let arr1: [u64; 8] = cast(self);
            let arr2: [u64; 8] = cast(rhs);
            cast([
              (arr1[0] as u128 * arr2[0] as u128 >> 64) as u64,
              (arr1[1] as u128 * arr2[1] as u128 >> 64) as u64,
              (arr1[2] as u128 * arr2[2] as u128 >> 64) as u64,
              (arr1[3] as u128 * arr2[3] as u128 >> 64) as u64,
              (arr1[4] as u128 * arr2[4] as u128 >> 64) as u64,
              (arr1[5] as u128 * arr2[5] as u128 >> 64) as u64,
              (arr1[6] as u128 * arr2[6] as u128 >> 64) as u64,
              (arr1[7] as u128 * arr2[7] as u128 >> 64) as u64,
            ])
          } else {
            Self(Inner {
              lo: self.0.lo.mul_keep_high(rhs.0.lo),
              hi: self.0.hi.mul_keep_high(rhs.0.hi),
            })
          }
        }
    }
}
}

safe_target_feature! {
// Backend of SIMD unsigned integer functions.
impl u64x8 {
    #[inline]
    pub(crate) fn saturating_add_signed_backend(self, rhs: i64x8) -> Self {
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
impl u64x8 {
    /// The exact per-lane product of `a` and `b` masked to `W` bits.
    ///
    /// Only exact for `W <= 32`, where the product still fits a lane, and
    /// callers guard on that. It is instantiated for wider `W` too, since the
    /// guard is a runtime `if` on a const, so it cannot assert the bound itself.
    #[inline]
    fn mul_masked<const W: u32>(a: Self, b: Self) -> Self {
        pick! {
          if #[cfg(target_feature="avx512f")] {
            // `vpmuludq` reads the low 32 bits of a lane anyway, so at `W == 32` the
            // operand masks are already implied and can be dropped.
            let (a, b) = if W == 32 {
              (a, b)
            } else {
              let mask = Self::splat(add_mul_operand_mask_u64::<W>());
              (a & mask, b & mask)
            };

            Self(_mm512_mul_epu32(a.0, b.0))
          } else {
            // Lane-wise, so each half is independent.
            Self(Inner {
              lo: u64x4::mul_masked::<W>(a.0.lo, b.0.lo),
              hi: u64x4::mul_masked::<W>(a.0.hi, b.0.hi),
            })
          }
        }
    }

    #[inline]
    pub(crate) fn add_mul_lo_backend<const W: u32>(self, a: Self, b: Self) -> Self {
        pick! {
          if #[cfg(target_feature="avx512ifma")] {
            // IFMA is fixed at 52 bits; any other width takes the generic path.
            if W == 52 {
              return Self(_mm512_madd52lo_epu64(self.0, a.0, b.0));
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
          if #[cfg(target_feature="avx512ifma")] {
            // IFMA is fixed at 52 bits; any other width takes the generic path.
            if W == 52 {
              return Self(_mm512_madd52hi_epu64(self.0, a.0, b.0));
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
