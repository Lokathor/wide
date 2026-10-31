use crate::backend_imports::*;

#[cfg(not(target_feature = "avx2"))]
#[repr(C, align(32))]
#[derive(Clone, Copy)]
pub(crate) struct Inner {
    pub lo: u8x16,
    pub hi: u8x16,
}

safe_target_feature! {
impl SimdBackend for u8x32 {
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
            Self(_mm256_cmpeq_epi8(self.0,rhs.0))
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
        pick! {
          if #[cfg(target_feature="avx2")] {
            // Convert from u8 to i8.
            let offset = Self::splat(0x80);
            let self_i8 = (self ^ offset).0;
            let rhs_i8 = (rhs ^ offset).0;
            Self(_mm256_cmpgt_epi8(rhs_i8, self_i8))
          } else {
            Self(Inner { lo: self.0.lo.simd_lt(rhs.0.lo), hi: self.0.hi.simd_lt(rhs.0.hi) })
          }
        }
    }

    #[inline]
    pub(crate) fn simd_gt_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="avx2")] {
            // Convert from u8 to i8.
            let offset = Self::splat(0x80);
            let self_i8 = (self ^ offset).0;
            let rhs_i8 = (rhs ^ offset).0;
            Self(_mm256_cmpgt_epi8(self_i8,rhs_i8))
          } else {
            Self(Inner { lo: self.0.lo.simd_gt(rhs.0.lo), hi: self.0.hi.simd_gt(rhs.0.hi) })
          }
        }
    }

    #[inline]
    pub(crate) fn simd_le_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="avx2")] {
            // Convert from u8 to i8.
            let offset = Self::splat(0x80);
            let self_i8 = (self ^ offset).0;
            let rhs_i8 = (rhs ^ offset).0;
            let gt_mask = Self(_mm256_cmpgt_epi8(self_i8,rhs_i8));
            gt_mask ^ Self::splat(0xFF)
          } else {
            Self(Inner { lo: self.0.lo.simd_le(rhs.0.lo), hi: self.0.hi.simd_le(rhs.0.hi) })
          }
        }
    }

    #[inline]
    pub(crate) fn simd_ge_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="avx2")] {
            // Convert from u8 to i8.
            let offset = Self::splat(0x80);
            let self_i8 = (self ^ offset).0;
            let rhs_i8 = (rhs ^ offset).0;
            let lt_mask = Self(_mm256_cmpgt_epi8(rhs_i8, self_i8));
            lt_mask ^ Self::splat(0xFF)
          } else {
            Self(Inner { lo: self.0.lo.simd_ge(rhs.0.lo), hi: self.0.hi.simd_ge(rhs.0.hi) })
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
          if #[cfg(target_feature="avx2")] {
            _mm256_movemask_epi8(self.0) as u32
          } else {
            self.0.lo.to_bitmask() | (self.0.hi.to_bitmask() << 16)
          }
        }
    }

    #[inline]
    pub(crate) fn any_backend(self) -> bool {
        pick! {
          if #[cfg(target_feature="avx2")] {
            _mm256_movemask_epi8(self.0) != 0
          } else {
            (self.0.lo | self.0.hi).any()
          }
        }
    }

    #[inline]
    pub(crate) fn all_backend(self) -> bool {
        pick! {
          if #[cfg(target_feature="avx2")] {
            _mm256_movemask_epi8(self.0) == -1
          } else {
            (self.0.lo & self.0.hi).all()
          }
        }
    }

    #[inline]
    pub(crate) fn interleave_low_backend(self, other: Self) -> Self {
        // `_mm256_unpacklo_epi8` cannot be used because it acts within each
        // 128-bit lane, which is a different operation.
        let [self_a, _] = cast::<u8x32, [u8x16; 2]>(self);
        let [other_a, _] = cast::<u8x32, [u8x16; 2]>(other);
        cast([
            self_a.interleave_low(other_a),
            self_a.interleave_high(other_a),
        ])
    }

    #[inline]
    pub(crate) fn interleave_high_backend(self, other: Self) -> Self {
        // `_mm256_unpackhi_epi8` cannot be used because it acts within each
        // 128-bit lane, which is a different operation.
        let [_, self_b] = cast::<u8x32, [u8x16; 2]>(self);
        let [_, other_b] = cast::<u8x32, [u8x16; 2]>(other);
        cast([
            self_b.interleave_low(other_b),
            self_b.interleave_high(other_b),
        ])
    }

    #[inline]
    pub(crate) fn shuffle_backend(self, indices: u8x32) -> Self {
        pick! {
          if #[cfg(all(target_feature="avx512vbmi", target_feature="avx512vl"))] {
            Self(_mm256_permutexvar_epi8(indices.0, self.0))
          } else if #[cfg(target_feature="avx2")] {
            // Same broadcast+blend as strict, but skip the 0x60 zeroing fold.
            let tbl_lo = _mm256_permute2x128_si256::<0x00>(self.0, self.0);
            let tbl_hi = _mm256_permute2x128_si256::<0x11>(self.0, self.0);
            let res_lo = _mm256_shuffle_epi8(tbl_lo, indices.0);
            let res_hi = _mm256_shuffle_epi8(tbl_hi, indices.0);
            let sel = _mm256_slli_epi16::<3>(indices.0);
            Self(_mm256_blendv_epi8(res_lo, res_hi, sel))
          } else {
            Self(Inner {
              lo: [self.0.lo, self.0.hi].shuffle(indices.0.lo),
              hi: [self.0.lo, self.0.hi].shuffle(indices.0.hi),
            })
          }
        }
    }

    #[inline]
    pub(crate) fn shuffle_zeroing_backend(self, indices: u8x32) -> Self {
        pick! {
          if #[cfg(all(target_feature="avx512vbmi", target_feature="avx512vl"))] {
            // vpermb takes the index mod 32 and never zeroes, so zero the
            // out-of-range lanes ourselves: (rhs & 0xE0) == 0  <=>  rhs < 32.
            let permuted = _mm256_permutexvar_epi8(indices.0, self.0);
            let hi_bits = _mm256_and_si256(indices.0, _mm256_set1_epi8(0xE0_u8 as i8));
            let in_range = _mm256_cmpeq_epi8(hi_bits, _mm256_setzero_si256());
            Self(_mm256_and_si256(permuted, in_range))
          } else if #[cfg(target_feature="avx2")] {
            // Broadcast each 16-byte table half into both 128-bit lanes, pshufb
            // each by the index, blend by index bit 4. Fold the >=32 zeroing into
            // pshufb with an unsigned saturating add of 0x60 (0x60 + 32 = 0x80).
            let idx = _mm256_adds_epu8(indices.0, _mm256_set1_epi8(0x60));
            let tbl_lo = _mm256_permute2x128_si256::<0x00>(self.0, self.0);
            let tbl_hi = _mm256_permute2x128_si256::<0x11>(self.0, self.0);
            let res_lo = _mm256_shuffle_epi8(tbl_lo, idx);
            let res_hi = _mm256_shuffle_epi8(tbl_hi, idx);
            // move index bit 4 into the sign bit (bit 7) for blendv.
            let sel = _mm256_slli_epi16::<3>(indices.0);
            Self(_mm256_blendv_epi8(res_lo, res_hi, sel))
          } else {
            Self(Inner {
              lo: [self.0.lo, self.0.hi].shuffle_zeroing(indices.0.lo),
              hi: [self.0.lo, self.0.hi].shuffle_zeroing(indices.0.hi),
            })
          }
        }
    }

    #[inline]
    pub(crate) fn shuffle_wrapping_backend(self, indices: u8x32) -> Self {
        pick! {
          if #[cfg(all(target_feature="avx512vbmi", target_feature="avx512vl"))] {
            Self(_mm256_permutexvar_epi8(indices.0, self.0))
          } else if #[cfg(target_feature="avx2")] {
            self.shuffle(indices & 31)
          } else {
            Self(Inner {
              lo: [self.0.lo, self.0.hi].shuffle_wrapping(indices.0.lo),
              hi: [self.0.lo, self.0.hi].shuffle_wrapping(indices.0.hi),
            })
          }
        }
    }

    #[inline]
    pub(crate) fn shuffle_2_backend(inputs: [Self; 2], indices: u8x32) -> u8x32 {
        pick! {
          if #[cfg(all(target_feature = "avx512vbmi", target_feature = "avx512vl"))] {
            u8x32(_mm256_permutex2var_epi8(inputs[0].0, indices.0, inputs[1].0))
          } else if #[cfg(target_feature="avx2")] {
            inputs[0].shuffle_zeroing(indices) | inputs[1].shuffle_zeroing(indices - 32)
          } else {
            u8x32(Inner {
              lo: [inputs[0].0.lo, inputs[0].0.hi, inputs[1].0.lo, inputs[1].0.hi].shuffle(indices.0.lo),
              hi: [inputs[0].0.lo, inputs[0].0.hi, inputs[1].0.lo, inputs[1].0.hi].shuffle(indices.0.hi),
            })
          }
        }
    }

    #[inline]
    pub(crate) fn shuffle_zeroing_2_backend(inputs: [Self; 2], indices: u8x32) -> u8x32 {
        pick! {
          if #[cfg(all(target_feature = "avx512vbmi", target_feature = "avx512vl"))] {
            inputs.shuffle(indices) & indices.simd_lt(64)
          } else if #[cfg(target_feature="avx2")] {
            inputs.shuffle(indices)
          } else {
            u8x32(Inner {
              lo: [inputs[0].0.lo, inputs[0].0.hi, inputs[1].0.lo, inputs[1].0.hi].shuffle_zeroing(indices.0.lo),
              hi: [inputs[0].0.lo, inputs[0].0.hi, inputs[1].0.lo, inputs[1].0.hi].shuffle_zeroing(indices.0.hi),
            })
          }
        }
    }

    #[inline]
    pub(crate) fn shuffle_wrapping_2_backend(inputs: [Self; 2], indices: u8x32) -> u8x32 {
        pick! {
          if #[cfg(all(target_feature = "avx512vbmi", target_feature = "avx512vl"))] {
            // `avx512` shuffle intrinsics are wrapping
            inputs.shuffle(indices)
          } else {
            inputs.shuffle(indices & 63)
          }
        }
    }

    #[inline]
    pub(crate) fn shuffle_3_backend(inputs: [Self; 3], indices: u8x32) -> u8x32 {
        [inputs[0], inputs[1]].shuffle_zeroing(indices) | inputs[2].shuffle_zeroing(indices - 64)
    }

    #[inline]
    pub(crate) fn shuffle_zeroing_3_backend(inputs: [Self; 3], indices: u8x32) -> u8x32 {
        inputs.shuffle(indices)
    }

    #[inline]
    pub(crate) fn shuffle_wrapping_3_backend(inputs: [Self; 3], indices: u8x32) -> u8x32 {
        inputs.shuffle(indices % 96)
    }

    #[inline]
    pub(crate) fn shuffle_4_backend(inputs: [Self; 4], indices: u8x32) -> u8x32 {
        [inputs[0], inputs[1]].shuffle_zeroing(indices)
            | [inputs[2], inputs[3]].shuffle_zeroing(indices - 64)
    }

    #[inline]
    pub(crate) fn shuffle_zeroing_4_backend(inputs: [Self; 4], indices: u8x32) -> u8x32 {
        inputs.shuffle(indices)
    }

    #[inline]
    pub(crate) fn shuffle_wrapping_4_backend(inputs: [Self; 4], indices: u8x32) -> u8x32 {
        inputs.shuffle(indices & 127)
    }

    #[inline]
    pub(crate) fn transpose_backend(data: [Self; 32]) -> [Self; 32] {
        // Can this be optimized?

        #[inline(always)]
        fn transpose_column(data: &[u8x32; 32], index: usize) -> u8x32 {
            u8x32::from_array([
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
        ]
    }

    #[inline]
    pub(crate) fn replace_const_backend<const INDEX: usize>(self, value: u8) -> Self {
        /// Because of limitations in const generics, this must have separate
        /// branches per index. This macro is used to avoid duplication.
        macro_rules! use_indices {
      (
        0..32 = [$($INDEX:literal),*],
        0..16 = [$($LOW_INDEX:literal),*],
        16..32 = [$($HIGH_INDEX:literal),*]
      ) => {
        pick! {
          if #[cfg(target_feature = "avx2")] {
            match INDEX {
              $($INDEX => Self(_mm256_insert_epi8::<$INDEX>(self.0, value.cast_signed())),)*
              32.. => unreachable!(),
            }
          } else {
            let [self_a, self_b] = cast::<u8x32, [u8x16; 2]>(self);

            if const { INDEX < 16 } {
              let result_a = match INDEX {
                $($LOW_INDEX => self_a.replace_const::<$LOW_INDEX>(value),)*
                16.. => unreachable!(),
              };

              cast([result_a, self_b])
            } else {
              let result_b = match INDEX {
                $($HIGH_INDEX => self_b.replace_const::<{ $HIGH_INDEX - 16 }>(value),)*
                ..16 | 32.. => unreachable!(),
              };

              cast([self_a, result_b])
            }
          }
        }
      };
    }
        use_indices! {
          0..32 = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31
          ],
          0..16 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
          16..32 = [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
        }
    }

    #[inline]
    pub(crate) fn extract_const_backend<const INDEX: usize>(self) -> u8 {
        /// Because of limitations in const generics, this must have separate
        /// branches per index. This macro is used to avoid duplication.
        macro_rules! use_indices {
      (
        0..32 = [$($INDEX:literal),*],
        0..16 = [$($LOW_INDEX:literal),*],
        16..32 = [$($HIGH_INDEX:literal),*]
      ) => {
        pick! {
          if #[cfg(target_feature = "avx2")] {
            match INDEX {
              $($INDEX => _mm256_extract_epi8::<$INDEX>(self.0).cast_unsigned() as u8,)*
              32.. => unreachable!(),
            }
          } else {
            let [self_a, self_b] = cast::<u8x32, [u8x16; 2]>(self);

            if const { INDEX < 16 } {
              match INDEX {
                $($LOW_INDEX => self_a.extract_const::<$LOW_INDEX>(),)*
                16.. => unreachable!(),
              }
            } else {
              match INDEX {
                $($HIGH_INDEX => self_b.extract_const::<{ $HIGH_INDEX - 16 }>(),)*
                ..16 | 32.. => unreachable!(),
              }
            }
          }
        }
      };
    }
        use_indices! {
          0..32 = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31
          ],
          0..16 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
          16..32 = [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
        }
    }
}
}

safe_target_feature! {
// Backend of SIMD integer functions.
impl u8x32 {
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
            Self(_mm256_add_epi8(self.0,rhs.0))
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
            Self(_mm256_sub_epi8(self.0,rhs.0))
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
        // For x86, this technically can be done explicitly by converting to `i16`
        // then converting back after multiplication, but that may not actually be
        // faster than auto-vectorization.
        let [self_a, self_b]: [u8x16; 2] = cast(self);
        let [rhs_a, rhs_b]: [u8x16; 2] = cast(rhs);
        cast([self_a * rhs_a, self_b * rhs_b])
    }

    #[inline]
    pub(crate) fn bitand_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="avx2")] {
              Self(_mm256_and_si256(self.0,rhs.0))
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
            Self(_mm256_or_si256(self.0,rhs.0))
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
            Self(_mm256_xor_si256(self.0,rhs.0))
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
        let array: [u8x16; 2] = cast(self);
        (array[0] + array[1]).reduce_add()
    }

    #[inline]
    pub(crate) fn reduce_mul_backend(self) -> u8 {
        let array: [u8x16; 2] = cast(self);
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
            self.shift_each_u16(rhs.0, false, false)
          } else {
            let [self_a, self_b]: [u8x16; 2] = cast(self);
            let [rhs_a, rhs_b]: [u8x16; 2] = cast(rhs);
            cast([self_a << rhs_a, self_b << rhs_b])
          }
        }
    }

    #[inline]
    pub(crate) fn shl_scalar_backend(self, rhs: u32) -> Self {
        pick! {
          if #[cfg(target_feature="avx2")] {
            let values: u16x16 = cast(self);
            let shift = rhs & 7;
            let shifted = values << shift;
            let crossed = u16x16::splat(!((0x00FFu16 << shift) & 0xFF00));
            cast(shifted & crossed)
          } else {
            let [self_a, self_b]: [u8x16; 2] = cast(self);
            cast([self_a << rhs, self_b << rhs])
          }
        }
    }

    #[inline]
    pub(crate) fn shr_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(all(target_feature="avx512bw", target_feature="avx512vl"))] {
            self.shift_each_u16(rhs.0, true, false)
          } else {
            let [self_a, self_b]: [u8x16; 2] = cast(self);
            let [rhs_a, rhs_b]: [u8x16; 2] = cast(rhs);
            cast([self_a >> rhs_a, self_b >> rhs_b])
          }
        }
    }

    #[inline]
    pub(crate) fn shr_scalar_backend(self, rhs: u32) -> Self {
        pick! {
          if #[cfg(target_feature="avx2")] {
            let values: u16x16 = cast(self);
            let shift = rhs & 7;
            let shifted = values >> shift;
            let crossed = u16x16::splat(!((0xFF00u16 >> shift) & 0x00FF));
            cast(shifted & crossed)
          } else {
            let [self_a, self_b]: [u8x16; 2] = cast(self);
            cast([self_a >> rhs, self_b >> rhs])
          }
        }
    }

    #[inline]
    pub(crate) fn max_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="avx2")] {
            Self(_mm256_max_epu8(self.0,rhs.0))
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
            Self(_mm256_min_epu8(self.0,rhs.0))
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
        let array: [u8x16; 2] = cast(self);
        array[0].max(array[1]).reduce_max()
    }

    #[inline]
    pub(crate) fn reduce_min_backend(self) -> u8 {
        let array: [u8x16; 2] = cast(self);
        array[0].min(array[1]).reduce_min()
    }

    #[inline]
    pub(crate) fn unbounded_shl_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(all(target_feature="avx512bw", target_feature="avx512vl"))] {
            self.shift_each_u16(rhs.0, false, true)
          } else {
            let [self_a, self_b] = cast::<u8x32, [u8x16; 2]>(self);
            let [rhs_a, rhs_b] = cast::<u8x32, [u8x16; 2]>(rhs);
            cast([self_a.unbounded_shl(rhs_a), self_b.unbounded_shl(rhs_b)])
          }
        }
    }

    #[inline]
    pub(crate) fn unbounded_shl_scalar_backend(self, rhs: u32) -> Self {
        pick! {
          if #[cfg(target_feature="avx2")] {
            let values: u16x16 = cast(self);
            let shifted = values.unbounded_shl_scalar(rhs);
            let crossed = u16x16::splat(!(0x00FFu16.wrapping_shl(rhs) & 0xFF00));
            cast(shifted & crossed)
          } else {
            let [self_a, self_b] = cast::<u8x32, [u8x16; 2]>(self);
            cast([self_a.unbounded_shl_scalar(rhs), self_b.unbounded_shl_scalar(rhs)])
          }
        }
    }

    #[inline]
    pub(crate) fn unbounded_shr_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(all(target_feature="avx512bw", target_feature="avx512vl"))] {
            self.shift_each_u16(rhs.0, true, true)
          } else {
            let [self_a, self_b] = cast::<u8x32, [u8x16; 2]>(self);
            let [rhs_a, rhs_b] = cast::<u8x32, [u8x16; 2]>(rhs);
            cast([self_a.unbounded_shr(rhs_a), self_b.unbounded_shr(rhs_b)])
          }
        }
    }

    #[inline]
    pub(crate) fn unbounded_shr_scalar_backend(self, rhs: u32) -> Self {
        pick! {
          if #[cfg(target_feature="avx2")] {
            let values: u16x16 = cast(self);
            let shifted = values.unbounded_shr_scalar(rhs);
            let crossed = u16x16::splat(!(0xFF00u16.wrapping_shr(rhs) & 0x00FF));
            cast(shifted & crossed)
          } else {
            let [self_a, self_b] = cast::<u8x32, [u8x16; 2]>(self);
            cast([self_a.unbounded_shr_scalar(rhs), self_b.unbounded_shr_scalar(rhs)])
          }
        }
    }

    #[inline]
    pub(crate) fn saturating_add_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="avx2")] {
            Self(_mm256_adds_epu8(self.0, rhs.0))
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
            Self(_mm256_subs_epu8(self.0, rhs.0))
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
    pub(crate) fn widening_mul_backend(self, rhs: Self) -> u16x32 {
        // x86 has no `_mm256_mul_epu8` intrinsic so there is no `avx2`
        // optimization.

        let [self_a, self_b] = cast::<u8x32, [u8x16; 2]>(self);
        let [rhs_a, rhs_b] = cast::<u8x32, [u8x16; 2]>(rhs);

        cast([self_a.widening_mul(rhs_a), self_b.widening_mul(rhs_b)])
    }

    #[inline]
    pub(crate) fn mul_keep_low_high_backend(self, rhs: Self) -> (Self, Self) {
        // x86 has no `_mm256_mul_epu8` intrinsic so there is no `avx2`
        // optimization.

        let [self_a, self_b] = cast::<u8x32, [u8x16; 2]>(self);
        let [rhs_a, rhs_b] = cast::<u8x32, [u8x16; 2]>(rhs);

        let result_a = self_a.mul_keep_low_high(rhs_a);
        let result_b = self_b.mul_keep_low_high(rhs_b);
        (
            cast([result_a.0, result_b.0]),
            cast([result_a.1, result_b.1]),
        )
    }

    #[inline]
    pub(crate) fn mul_keep_high_backend(self, rhs: Self) -> Self {
        // x86 has no `_mm256_mul_epu8` intrinsic so there is no `avx2`
        // optimization.

        let [self_a, self_b] = cast::<u8x32, [u8x16; 2]>(self);
        let [rhs_a, rhs_b] = cast::<u8x32, [u8x16; 2]>(rhs);

        cast([self_a.mul_keep_high(rhs_a), self_b.mul_keep_high(rhs_b)])
    }
}
}

safe_target_feature! {
// Backend of SIMD unsigned integer functions.
impl u8x32 {
    #[inline]
    pub(crate) fn saturating_add_signed_backend(self, rhs: i8x32) -> Self {
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
impl u8x32 {
    #[inline]
    pub(crate) fn swizzle_half_backend(self, rhs: i8x32) -> i8x32 {
        cast(i8x32::swizzle_half(cast(self), cast(rhs)))
    }

    #[inline]
    pub(crate) fn swizzle_half_relaxed_backend(self, rhs: u8x32) -> u8x32 {
        cast(i8x32::swizzle_half_relaxed(cast(self), cast(rhs)))
    }

    // There's no `u8` shift instruction, so we cheat: widen every byte to a
    // `u16`, shift in the wider lane, then shrink back down to `u8`. Shifting
    // left can push bits past `u8::MAX`, so we keep only the low 8 bits before
    // the saturating pack (shifting right can't overflow, so the mask is a
    // no-op there). For the unbounded variants we cap the count at 8, because
    // shifting a byte by 8 or more gets rid of everything anyway.
    #[cfg(all(target_feature = "avx512bw", target_feature = "avx512vl"))]
    #[inline]
    fn shift_each_u16(self, rhs: __m256i, right: bool, unbounded: bool) -> Self {
        let self16 = _mm512_cvtepu8_epi16(self.0);
        let count16 = if unbounded {
            _mm512_cvtepu8_epi16(_mm256_min_epu8(rhs, _mm256_set1_epi8(8)))
        } else {
            _mm512_and_si512(
                _mm512_cvtepu8_epi16(rhs),
                _mm512_set1_epi16(7),
            )
        };
        let shifted = if right {
            _mm512_srlv_epi16(self16, count16)
        } else {
            _mm512_sllv_epi16(self16, count16)
        };
        let shifted = _mm512_and_si512(shifted, _mm512_set1_epi16(0xFF));
        Self(Self::pack_u16_halves(
            _mm512_extracti64x4_epi64::<0>(shifted),
            _mm512_extracti64x4_epi64::<1>(shifted),
        ))
    }

    // `_mm256_packus_epi16` packs 128 bits at a time, which scrambles the lane
    // order. This un-scrambles it: split the packed result in half, interleave
    // the 64-bit chunks back together, and reassemble.
    #[cfg(all(target_feature = "avx512bw", target_feature = "avx512vl"))]
    #[inline]
    fn pack_u16_halves(low: __m256i, high: __m256i) -> __m256i {
        let packed = _mm256_packus_epi16(low, high);
        let packed_low = _mm256_extracti128_si256::<0>(packed);
        let packed_high = _mm256_extracti128_si256::<1>(packed);
        let combined_low = _mm_unpacklo_epi64(packed_low, packed_high);
        let combined_high = _mm_unpackhi_epi64(packed_low, packed_high);
        _mm256_inserti128_si256::<1>(
            _mm256_inserti128_si256::<0>(_mm256_setzero_si256(), combined_low),
            combined_high,
        )
    }
}
}
