use crate::backend_imports::*;

#[cfg(not(target_feature = "avx2"))]
#[repr(C, align(32))]
#[derive(Clone, Copy)]
pub(crate) struct Inner {
    pub lo: u64x2,
    pub hi: u64x2,
}

safe_target_feature! {
impl SimdBackend for u64x4 {
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
            Self(_mm256_cmpeq_epi64(self.0, rhs.0))
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
        // lt is just gt the other way around
        rhs.simd_gt(self)
    }

    #[inline]
    pub(crate) fn simd_gt_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="avx2")] {
            // no unsigned gt than so inverting the high bit will get the correct result
            let highbit = u64x4::splat(1 << 63);
            Self(_mm256_cmpgt_epi64((self ^ highbit).0, (rhs ^ highbit).0))
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
            Self(_mm256_blendv_epi8(if_false.0,if_true.0,self.0))
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
            // use f64 move_mask since it is the same size as i64
            _mm256_movemask_pd(cast(self.0)) as u32
          } else {
            self.0.lo.to_bitmask() | (self.0.hi.to_bitmask() << 2)
          }
        }
    }

    #[inline]
    pub(crate) fn any_backend(self) -> bool {
        pick! {
          if #[cfg(target_feature="avx2")] {
            _mm256_movemask_pd(cast(self.0)) != 0
          } else {
            (self.0.lo | self.0.hi).any()
          }
        }
    }

    #[inline]
    pub(crate) fn all_backend(self) -> bool {
        pick! {
          if #[cfg(target_feature="avx2")] {
            _mm256_movemask_pd(cast(self.0)) == 0b1111
          } else {
            (self.0.lo & self.0.hi).all()
          }
        }
    }

    #[inline]
    pub(crate) fn interleave_low_backend(self, other: Self) -> Self {
        pick! {
          if #[cfg(target_feature="avx2")] {
            // `_mm256_unpacklo_epi64` cannot be used because it acts within each
            // 128-bit lane, which is a different operation.
            let [aa, _]: [u64x2; 2] = cast(self);
            let [ba, _]: [u64x2; 2] = cast(other);
            cast([aa.interleave_low(ba), aa.interleave_high(ba)])
          } else {
            Self(Inner { lo: self.0.lo.interleave_low(other.0.lo), hi: self.0.lo.interleave_high(other.0.lo) })
          }
        }
    }

    #[inline]
    pub(crate) fn interleave_high_backend(self, other: Self) -> Self {
        pick! {
          if #[cfg(target_feature="avx2")] {
            // `_mm256_unpackhi_epi64` cannot be used because it acts within each
            // 128-bit lane, which is a different operation.
            let [_, ab]: [u64x2; 2] = cast(self);
            let [_, bb]: [u64x2; 2] = cast(other);
            cast([ab.interleave_low(bb), ab.interleave_high(bb)])
          } else {
            Self(Inner { lo: self.0.hi.interleave_low(other.0.hi), hi: self.0.hi.interleave_high(other.0.hi) })
          }
        }
    }

    #[inline]
    pub(crate) fn shuffle_backend(self, indices: u64x4) -> Self {
        pick! {
          if #[cfg(all(target_feature="avx512f", target_feature="avx512vl"))] {
            Self(_mm256_permutexvar_epi64(indices.0, self.0))
          } else if #[cfg(any(
            target_feature = "ssse3",
            all(target_arch = "aarch64", target_feature = "neon"),
            target_feature = "simd128",
          ))] {
            let self_bytes = cast::<u64x4, u8x32>(self);
            let byte_indices = indices.to_byte_indices();

            cast::<u8x32, u64x4>(self_bytes.shuffle(byte_indices))
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
    pub(crate) fn shuffle_zeroing_backend(self, indices: u64x4) -> Self {
        pick! {
          if #[cfg(any(
            target_feature = "ssse3",
            all(target_arch = "aarch64", target_feature = "neon"),
            target_feature = "simd128",
          ))] {
            // Even if the `u8x32::shuffle` implementation is zeroing, our 64-bit to
            // 8-bit can trigger an overflow causing incorrect behavior
            self.shuffle(indices) & indices.simd_lt(4)
          } else {
            // The fallback branch of `shuffle` already has the behavior we want
            self.shuffle(indices)
          }
        }
    }

    #[inline]
    pub(crate) fn shuffle_wrapping_backend(self, indices: u64x4) -> Self {
        pick! {
          if #[cfg(all(target_feature = "avx512f", target_feature = "avx512vl"))] {
            // `avx512` shuffle intrinsics are wrapping
            self.shuffle(indices)
          } else {
            self.shuffle(indices & 3)
          }
        }
    }

    #[inline]
    pub(crate) fn shuffle_2_backend(inputs: [Self; 2], indices: u64x4) -> u64x4 {
        pick! {
          if #[cfg(all(target_feature = "avx512f", target_feature = "avx512vl"))] {
            u64x4(_mm256_permutex2var_epi64(inputs[0].0, indices.0, inputs[1].0))
          } else {
            let self_bytes = cast::<[u64x4; 2], [u8x32; 2]>(inputs);
            let byte_indices = indices.to_byte_indices();

            cast::<u8x32, u64x4>(self_bytes.shuffle(byte_indices))
          }
        }
    }

    #[inline]
    pub(crate) fn shuffle_zeroing_2_backend(inputs: [Self; 2], indices: u64x4) -> u64x4 {
        // Even if the `u8x32::shuffle` implementation is zeroing, our 32-bit to
        // 8-bit can trigger an overflow causing incorrect behavior
        inputs.shuffle(indices) & indices.simd_lt(8)
    }

    #[inline]
    pub(crate) fn shuffle_wrapping_2_backend(inputs: [Self; 2], indices: u64x4) -> u64x4 {
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
    pub(crate) fn shuffle_3_backend(inputs: [Self; 3], indices: u64x4) -> u64x4 {
        let self_bytes = cast::<[u64x4; 3], [u8x32; 3]>(inputs);
        let byte_indices = indices.to_byte_indices();

        cast::<u8x32, u64x4>(self_bytes.shuffle(byte_indices))
    }

    #[inline]
    pub(crate) fn shuffle_zeroing_3_backend(inputs: [Self; 3], indices: u64x4) -> u64x4 {
        // Even if the `u8x32::shuffle` implementation is zeroing, our 32-bit to
        // 8-bit can trigger an overflow causing incorrect behavior
        inputs.shuffle(indices) & indices.simd_lt(12)
    }

    #[inline]
    pub(crate) fn shuffle_wrapping_3_backend(inputs: [Self; 3], indices: u64x4) -> u64x4 {
        inputs.shuffle(indices % 12)
    }

    #[inline]
    pub(crate) fn shuffle_4_backend(inputs: [Self; 4], indices: u64x4) -> u64x4 {
        let self_bytes = cast::<[u64x4; 4], [u8x32; 4]>(inputs);
        let byte_indices = indices.to_byte_indices();

        cast::<u8x32, u64x4>(self_bytes.shuffle(byte_indices))
    }

    #[inline]
    pub(crate) fn shuffle_zeroing_4_backend(inputs: [Self; 4], indices: u64x4) -> u64x4 {
        // Even if the `u8x32::shuffle` implementation is zeroing, our 64-bit to
        // 8-bit can trigger an overflow causing incorrect behavior
        inputs.shuffle(indices) & indices.simd_lt(16)
    }

    #[inline]
    pub(crate) fn shuffle_wrapping_4_backend(inputs: [Self; 4], indices: u64x4) -> u64x4 {
        inputs.shuffle(indices & 15)
    }

    #[inline]
    pub(crate) fn transpose_backend(data: [Self; 4]) -> [Self; 4] {
        pick! {
          if #[cfg(target_feature="avx2")] {
            // Can this be optimized?
            // TODO: Once unpack functions are added, remove these casts
            let a = data[0].cast_signed().interleave_low(data[2].cast_signed());
            let b = data[1].cast_signed().interleave_low(data[3].cast_signed());
            let c = data[0].cast_signed().interleave_high(data[2].cast_signed());
            let d = data[1].cast_signed().interleave_high(data[3].cast_signed());
            [
              a.interleave_low(b).cast_unsigned(),
              a.interleave_high(b).cast_unsigned(),
              c.interleave_low(d).cast_unsigned(),
              c.interleave_high(d).cast_unsigned(),
            ]
          } else {
            #[inline(always)]
            fn transpose_column(data: &[u64x4; 4], index: usize) -> u64x4 {
              u64x4::from_array([
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
    pub(crate) fn replace_const_backend<const INDEX: usize>(self, value: u64) -> Self {
        /// Because of limitations in const generics, this must have separate
        /// branches per index. This macro is used to avoid duplication.
        macro_rules! use_indices {
      (
        0..4 = [$($INDEX:literal),*],
        0..2 = [$($LOW_INDEX:literal),*],
        2..4 = [$($HIGH_INDEX:literal),*]
      ) => {
        pick! {
          if #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))] {
            match INDEX {
              $($INDEX => Self(_mm256_insert_epi64::<$INDEX>(self.0, value.cast_signed())),)*
              4.. => unreachable!(),
            }
          } else {
            let [self_a, self_b] = cast::<u64x4, [u64x2; 2]>(self);

            if const { INDEX < 2 } {
              let result_a = match INDEX {
                $($LOW_INDEX => self_a.replace_const::<$LOW_INDEX>(value),)*
                2.. => unreachable!(),
              };

              cast([result_a, self_b])
            } else {
              let result_b = match INDEX {
                $($HIGH_INDEX => self_b.replace_const::<{ $HIGH_INDEX - 2 }>(value),)*
                ..2 | 4.. => unreachable!(),
              };

              cast([self_a, result_b])
            }
          }
        }
      };
    }
        use_indices! { 0..4 = [0, 1, 2, 3], 0..2 = [0, 1], 2..4 = [2, 3] }
    }

    #[inline]
    pub(crate) fn extract_const_backend<const INDEX: usize>(self) -> u64 {
        /// Because of limitations in const generics, this must have separate
        /// branches per index. This macro is used to avoid duplication.
        macro_rules! use_indices {
      (
        0..4 = [$($INDEX:literal),*],
        0..2 = [$($LOW_INDEX:literal),*],
        2..4 = [$($HIGH_INDEX:literal),*]
      ) => {
        pick! {
          if #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))] {
            match INDEX {
              $($INDEX => _mm256_extract_epi64::<$INDEX>(self.0).cast_unsigned(),)*
              4.. => unreachable!(),
            }
          } else {
            let [self_a, self_b] = cast::<u64x4, [u64x2; 2]>(self);

            if const { INDEX < 2 } {
              match INDEX {
                $($LOW_INDEX => self_a.extract_const::<$LOW_INDEX>(),)*
                2.. => unreachable!(),
              }
            } else {
              match INDEX {
                $($HIGH_INDEX => self_b.extract_const::<{ $HIGH_INDEX - 2 }>(),)*
                ..2 | 4.. => unreachable!(),
              }
            }
          }
        }
      };
    }
        use_indices! { 0..4 = [0, 1, 2, 3], 0..2 = [0, 1], 2..4 = [2, 3] }
    }
}
}

safe_target_feature! {
// Backend of SIMD integer functions.
impl u64x4 {
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
            Self(_mm256_add_epi64(self.0, rhs.0))
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
            Self(_mm256_sub_epi64(self.0, rhs.0))
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
            let arr1: [i64; 4] = cast(self);
            let arr2: [i64; 4] = cast(rhs);
            cast([
              arr1[0].wrapping_mul(arr2[0]),
              arr1[1].wrapping_mul(arr2[1]),
              arr1[2].wrapping_mul(arr2[2]),
              arr1[3].wrapping_mul(arr2[3]),
            ])
          } else {
            Self(Inner { lo: self.0.lo * rhs.0.lo, hi: self.0.hi * rhs.0.hi })
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
    pub(crate) fn reduce_add_backend(self) -> u64 {
        pick! {
          if #[cfg(all(target_arch="x86_64", target_feature="avx2"))] {
            let zwxx  = _mm256_permute4x64_epi64::<0b00_00_11_10>(self.0);
            let xz_yw = _mm256_add_epi64(zwxx, self.0);
            let yw_xz  = _mm256_permute4x64_epi64::<0b00_00_00_01>(xz_yw);
            let sum = _mm256_add_epi64(xz_yw, yw_xz);
            _mm256_extract_epi64::<0>(sum).cast_unsigned()
          } else {
            let array: [u64; 4] = cast(self);
            array[0]
              .wrapping_add(array[1])
              .wrapping_add(array[2])
              .wrapping_add(array[3])
          }
        }
    }

    #[inline]
    pub(crate) fn reduce_mul_backend(self) -> u64 {
        let array: [u64; 4] = cast(self);
        array[0]
            .wrapping_mul(array[1])
            .wrapping_mul(array[2])
            .wrapping_mul(array[3])
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
          if #[cfg(target_feature="avx2")] {
            // mask the shift count to 63 to have same behavior on all platforms
            let shift_by = rhs & Self::splat(63);
            Self(_mm256_sllv_epi64(self.0, shift_by.0))
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
          if #[cfg(target_feature="avx2")] {
            // Use `rhs % 64` to perform wrapping shift and not unbounded shift.
            let shift = cast([rhs as u64 & 63, 0]);
            Self(_mm256_sll_epi64(self.0, shift))
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
          if #[cfg(target_feature="avx2")] {
            // mask the shift count to 63 to have same behavior on all platforms
            let shift_by = rhs & Self::splat(63);
            Self(_mm256_srlv_epi64(self.0, shift_by.0))
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
          if #[cfg(target_feature="avx2")] {
            // Use `rhs % 64` to perform wrapping shift and not unbounded shift.
            let shift = cast([rhs as u64 & 63, 0]);
            Self(_mm256_srl_epi64(self.0, shift))
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
        self.simd_gt(rhs).select(self, rhs)
    }

    #[inline]
    pub(crate) fn min_backend(self, rhs: Self) -> Self {
        self.simd_lt(rhs).select(self, rhs)
    }

    #[inline]
    pub(crate) fn reduce_max_backend(self) -> u64 {
        let array: [u64; 4] = cast(self);
        array[0].max(array[1]).max(array[2]).max(array[3])
    }

    #[inline]
    pub(crate) fn reduce_min_backend(self) -> u64 {
        let array: [u64; 4] = cast(self);
        array[0].min(array[1]).min(array[2]).min(array[3])
    }

    #[inline]
    pub(crate) fn unbounded_shl_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="avx2")] {
            Self(_mm256_sllv_epi64(self.0, rhs.0))
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
          if #[cfg(target_feature="avx2")] {
            Self(_mm256_sll_epi64(self.0, cast([rhs as u64, 0])))
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
          if #[cfg(target_feature="avx2")] {
            Self(_mm256_srlv_epi64(self.0, rhs.0))
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
          if #[cfg(target_feature="avx2")] {
            Self(_mm256_srl_epi64(self.0, cast([rhs as u64, 0])))
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
          if #[cfg(target_feature="avx2")] {
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
        ];
        (
            Self::from_array([result[0].0, result[1].0, result[2].0, result[3].0]),
            Self::from_array([
                -(result[0].1 as i64) as u64,
                -(result[1].1 as i64) as u64,
                -(result[2].1 as i64) as u64,
                -(result[3].1 as i64) as u64,
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
        ];

        (
            Self::from_array([
                widening_mul[0] as u64,
                widening_mul[1] as u64,
                widening_mul[2] as u64,
                widening_mul[3] as u64,
            ]),
            Self::from_array([
                (widening_mul[0] >> 64) as u64,
                (widening_mul[1] >> 64) as u64,
                (widening_mul[2] >> 64) as u64,
                (widening_mul[3] >> 64) as u64,
            ]),
        )
    }

    #[inline]
    pub(crate) fn mul_keep_high_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="avx2")] {
            let arr1: [u64; 4] = cast(self);
            let arr2: [u64; 4] = cast(rhs);
            cast([
              (arr1[0] as u128 * arr2[0] as u128 >> 64) as u64,
              (arr1[1] as u128 * arr2[1] as u128 >> 64) as u64,
              (arr1[2] as u128 * arr2[2] as u128 >> 64) as u64,
              (arr1[3] as u128 * arr2[3] as u128 >> 64) as u64,
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
impl u64x4 {
    #[inline]
    pub(crate) fn saturating_add_signed_backend(self, rhs: i64x4) -> Self {
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
impl u64x4 {
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
    fn to_byte_indices(self) -> u8x32 {
        // The byte offset of the lane, broadcast to every byte of the lane.
        let base = self.unbounded_shl_scalar(3);
        let base = base | base.unbounded_shl_scalar(8);
        let base = base | base.unbounded_shl_scalar(16);
        let base = base | base.unbounded_shl_scalar(32);

        // Then the offset of each byte within its lane. These bits are free because
        // every byte of `base` is a multiple of eight. `from_ne_bytes` keeps this
        // correct on big endian, where the bytes of a lane are the other way around.
        const WITHIN_LANE: u64x4 = u64x4::splat(u64::from_ne_bytes([0, 1, 2, 3, 4, 5, 6, 7]));

        cast::<u64x4, u8x32>(base | WITHIN_LANE)
    }

    /// The exact per-lane product of `a` and `b` masked to `W` bits.
    ///
    /// Only exact for `W <= 32`, where the product still fits a lane, and
    /// callers guard on that. It is instantiated for wider `W` too, since the
    /// guard is a runtime `if` on a const, so it cannot assert the bound itself.
    #[inline]
    pub(crate) fn mul_masked<const W: u32>(a: Self, b: Self) -> Self {
        pick! {
          if #[cfg(target_feature="avx2")] {
            // `vpmuludq` reads the low 32 bits of a lane anyway, so at `W == 32` the
            // operand masks are already implied and can be dropped. A scalar
            // multiply has no such truncation, so the other arm always masks.
            let (a, b) = if W == 32 {
              (a, b)
            } else {
              let mask = Self::splat(add_mul_operand_mask_u64::<W>());
              (a & mask, b & mask)
            };

            Self(_mm256_mul_epu32(a.0, b.0))
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
              return Self(_mm256_madd52lo_epu64(self.0, a.0, b.0));
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
              return Self(_mm256_madd52hi_epu64(self.0, a.0, b.0));
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
