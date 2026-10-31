use crate::backend_imports::*;

#[cfg(not(target_feature = "avx2"))]
#[repr(C, align(32))]
#[derive(Clone, Copy)]
pub(crate) struct Inner {
    pub lo: u32x4,
    pub hi: u32x4,
}

safe_target_feature! {
impl SimdBackend for u32x8 {
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
            Self(_mm256_cmpeq_epi32(self.0, rhs.0 ))
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
            let highbit = u32x8::splat(1 << 31);
            Self(_mm256_cmpgt_epi32((self ^ highbit).0, (rhs ^ highbit).0 ))
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
        self.simd_eq(rhs) | self.simd_lt(rhs)
    }

    #[inline]
    pub(crate) fn simd_ge_backend(self, rhs: Self) -> Self {
        self.simd_eq(rhs) | self.simd_gt(rhs)
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
            // use f32 move_mask since it is the same size as i32
            _mm256_movemask_ps(cast(self.0)) as u32
          } else {
            self.0.lo.to_bitmask() | (self.0.hi.to_bitmask() << 4)
          }
        }
    }

    #[inline]
    pub(crate) fn any_backend(self) -> bool {
        pick! {
          if #[cfg(target_feature="avx2")] {
            _mm256_movemask_ps(cast(self.0)) != 0
          } else {
            (self.0.lo | self.0.hi).any()
          }
        }
    }

    #[inline]
    pub(crate) fn all_backend(self) -> bool {
        pick! {
          if #[cfg(target_feature="avx2")] {
            _mm256_movemask_ps(cast(self.0)) == 0b11111111
          } else {
            (self.0.lo & self.0.hi).all()
          }
        }
    }

    #[inline]
    pub(crate) fn interleave_low_backend(self, other: Self) -> Self {
        pick! {
          if #[cfg(target_feature="avx2")] {
            // `_mm256_unpacklo_epi32` cannot be used because it acts within each
            // 128-bit lane, which is a different operation.
            let [aa, _]: [u32x4; 2] = cast(self);
            let [ba, _]: [u32x4; 2] = cast(other);
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
            // `_mm256_unpackhi_epi32` cannot be used because it acts within each
            // 128-bit lane, which is a different operation.
            let [_, ab]: [u32x4; 2] = cast(self);
            let [_, bb]: [u32x4; 2] = cast(other);
            cast([ab.interleave_low(bb), ab.interleave_high(bb)])
          } else {
            Self(Inner { lo: self.0.hi.interleave_low(other.0.hi), hi: self.0.hi.interleave_high(other.0.hi) })
          }
        }
    }

    #[inline]
    pub(crate) fn shuffle_backend(self, indices: u32x8) -> Self {
        pick! {
          if #[cfg(all(target_feature = "avx512f", target_feature = "avx512vl"))] {
            Self(_mm256_permutexvar_epi32(indices.0, self.0))
          } else if #[cfg(target_feature = "avx2")] {
            Self(_mm256_permutevar8x32_epi32(self.0, indices.0))
          } else if #[cfg(any(
            target_feature = "ssse3",
            all(target_arch = "aarch64", target_feature = "neon"),
            target_feature = "simd128",
          ))] {
            let self_bytes = cast::<u32x8, u8x32>(self);
            let byte_indices = indices.to_byte_indices();

            cast::<u8x32, u32x8>(self_bytes.shuffle(byte_indices))
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
    pub(crate) fn shuffle_zeroing_backend(self, indices: u32x8) -> Self {
        pick! {
          if #[cfg(any(
            target_feature = "ssse3",
            all(target_arch = "aarch64", target_feature = "neon"),
            target_feature = "simd128",
          ))] {
            // Even if the `u8x32::shuffle` implementation is zeroing, our 32-bit to
            // 8-bit can trigger an overflow causing incorrect behavior
            self.shuffle(indices) & indices.simd_lt(8)
          } else {
            // The fallback branch of `shuffle` already has the behavior we want
            self.shuffle(indices)
          }
        }
    }

    #[inline]
    pub(crate) fn shuffle_wrapping_backend(self, indices: u32x8) -> Self {
        pick! {
          if #[cfg(all(target_feature = "avx512f", target_feature = "avx512vl"))] {
            // `avx512` shuffle intrinsics are wrapping
            self.shuffle(indices)
          } else {
            self.shuffle(indices & 7)
          }
        }
    }

    #[inline]
    pub(crate) fn shuffle_2_backend(inputs: [Self; 2], indices: u32x8) -> u32x8 {
        pick! {
          if #[cfg(all(target_feature = "avx512f", target_feature = "avx512vl"))] {
            u32x8(_mm256_permutex2var_epi32(inputs[0].0, indices.0, inputs[1].0))
          } else {
            let self_bytes = cast::<[u32x8; 2], [u8x32; 2]>(inputs);
            let byte_indices = indices.to_byte_indices();

            cast::<u8x32, u32x8>(self_bytes.shuffle(byte_indices))
          }
        }
    }

    #[inline]
    pub(crate) fn shuffle_zeroing_2_backend(inputs: [Self; 2], indices: u32x8) -> u32x8 {
        // Even if the `u8x32::shuffle` implementation is zeroing, our 32-bit to
        // 8-bit can trigger an overflow causing incorrect behavior
        inputs.shuffle(indices) & indices.simd_lt(16)
    }

    #[inline]
    pub(crate) fn shuffle_wrapping_2_backend(inputs: [Self; 2], indices: u32x8) -> u32x8 {
        pick! {
          if #[cfg(all(target_feature = "avx512f", target_feature = "avx512vl"))] {
            // `avx512` shuffle intrinsics are wrapping
            inputs.shuffle(indices)
          } else {
            inputs.shuffle(indices & 15)
          }
        }
    }

    #[inline]
    pub(crate) fn shuffle_3_backend(inputs: [Self; 3], indices: u32x8) -> u32x8 {
        let self_bytes = cast::<[u32x8; 3], [u8x32; 3]>(inputs);
        let byte_indices = indices.to_byte_indices();

        cast::<u8x32, u32x8>(self_bytes.shuffle(byte_indices))
    }

    #[inline]
    pub(crate) fn shuffle_zeroing_3_backend(inputs: [Self; 3], indices: u32x8) -> u32x8 {
        // Even if the `u8x32::shuffle` implementation is zeroing, our 32-bit to
        // 8-bit can trigger an overflow causing incorrect behavior
        inputs.shuffle(indices) & indices.simd_lt(24)
    }

    #[inline]
    pub(crate) fn shuffle_wrapping_3_backend(inputs: [Self; 3], indices: u32x8) -> u32x8 {
        inputs.shuffle(indices % 24)
    }

    #[inline]
    pub(crate) fn shuffle_4_backend(inputs: [Self; 4], indices: u32x8) -> u32x8 {
        let self_bytes = cast::<[u32x8; 4], [u8x32; 4]>(inputs);
        let byte_indices = indices.to_byte_indices();

        cast::<u8x32, u32x8>(self_bytes.shuffle(byte_indices))
    }

    #[inline]
    pub(crate) fn shuffle_zeroing_4_backend(inputs: [Self; 4], indices: u32x8) -> u32x8 {
        // Even if the `u8x32::shuffle` implementation is zeroing, our 32-bit to
        // 8-bit can trigger an overflow causing incorrect behavior
        inputs.shuffle(indices) & indices.simd_lt(32)
    }

    #[inline]
    pub(crate) fn shuffle_wrapping_4_backend(inputs: [Self; 4], indices: u32x8) -> u32x8 {
        inputs.shuffle(indices & 31)
    }

    #[inline]
    pub(crate) fn transpose_backend(data: [Self; 8]) -> [Self; 8] {
        pick! {
          if #[cfg(target_feature="avx2")] {
            let a0 = _mm256_unpacklo_epi32(data[0].0, data[1].0);
            let a1 = _mm256_unpackhi_epi32(data[0].0, data[1].0);
            let a2 = _mm256_unpacklo_epi32(data[2].0, data[3].0);
            let a3 = _mm256_unpackhi_epi32(data[2].0, data[3].0);
            let a4 = _mm256_unpacklo_epi32(data[4].0, data[5].0);
            let a5 = _mm256_unpackhi_epi32(data[4].0, data[5].0);
            let a6 = _mm256_unpacklo_epi32(data[6].0, data[7].0);
            let a7 = _mm256_unpackhi_epi32(data[6].0, data[7].0);

            pub const fn mm_shuffle(z: i32, y: i32, x: i32, w: i32) -> i32 {
              (z << 6) | (y << 4) | (x << 2) | w
            }

            const SHUFF_LO : i32 = mm_shuffle(1,0,1,0);
            const SHUFF_HI : i32 = mm_shuffle(3,2,3,2);

            // possible todo: intel performance manual suggests alternative with blend to avoid port 5 pressure
            // (since blend runs on a different port than shuffle)
            let b0 = cast::<__m256,__m256i>(_mm256_shuffle_ps::<SHUFF_LO>(cast(a0),cast(a2)));
            let b1 = cast::<__m256,__m256i>(_mm256_shuffle_ps::<SHUFF_HI>(cast(a0),cast(a2)));
            let b2 = cast::<__m256,__m256i>(_mm256_shuffle_ps::<SHUFF_LO>(cast(a1),cast(a3)));
            let b3 = cast::<__m256,__m256i>(_mm256_shuffle_ps::<SHUFF_HI>(cast(a1),cast(a3)));
            let b4 = cast::<__m256,__m256i>(_mm256_shuffle_ps::<SHUFF_LO>(cast(a4),cast(a6)));
            let b5 = cast::<__m256,__m256i>(_mm256_shuffle_ps::<SHUFF_HI>(cast(a4),cast(a6)));
            let b6 = cast::<__m256,__m256i>(_mm256_shuffle_ps::<SHUFF_LO>(cast(a5),cast(a7)));
            let b7 = cast::<__m256,__m256i>(_mm256_shuffle_ps::<SHUFF_HI>(cast(a5),cast(a7)));

            [
              u32x8(_mm256_permute2f128_si256::<0x20>(b0, b4)),
              u32x8(_mm256_permute2f128_si256::<0x20>(b1, b5)),
              u32x8(_mm256_permute2f128_si256::<0x20>(b2, b6)),
              u32x8(_mm256_permute2f128_si256::<0x20>(b3, b7)),
              u32x8(_mm256_permute2f128_si256::<0x31>(b0, b4)),
              u32x8(_mm256_permute2f128_si256::<0x31>(b1, b5)),
              u32x8(_mm256_permute2f128_si256::<0x31>(b2, b6)),
              u32x8(_mm256_permute2f128_si256::<0x31>(b3, b7))
            ]
          } else {
            // possible todo: not sure that 128bit SIMD gives us a a lot of speedup here

            #[inline(always)]
            fn transpose_column(data: &[u32x8; 8], index: usize) -> u32x8 {
              u32x8::from_array([
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
    pub(crate) fn replace_const_backend<const INDEX: usize>(self, value: u32) -> Self {
        /// Because of limitations in const generics, this must have separate
        /// branches per index. This macro is used to avoid duplication.
        macro_rules! use_indices {
      (
        0..8 = [$($INDEX:literal),*],
        0..4 = [$($LOW_INDEX:literal),*],
        4..8 = [$($HIGH_INDEX:literal),*]
      ) => {
        pick! {
          if #[cfg(target_feature = "avx2")] {
            match INDEX {
              $($INDEX => Self(_mm256_insert_epi32::<$INDEX>(self.0, value.cast_signed())),)*
              8.. => unreachable!(),
            }
          } else {
            let [self_a, self_b] = cast::<u32x8, [u32x4; 2]>(self);

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
        use_indices! {
          0..8 = [0, 1, 2, 3, 4, 5, 6, 7],
          0..4 = [0, 1, 2, 3],
          4..8 = [4, 5, 6, 7]
        }
    }

    #[inline]
    pub(crate) fn extract_const_backend<const INDEX: usize>(self) -> u32 {
        /// Because of limitations in const generics, this must have separate
        /// branches per index. This macro is used to avoid duplication.
        macro_rules! use_indices {
      (
        0..8 = [$($INDEX:literal),*],
        0..4 = [$($LOW_INDEX:literal),*],
        4..8 = [$($HIGH_INDEX:literal),*]
      ) => {
        pick! {
          if #[cfg(target_feature = "avx2")] {
            match INDEX {
              $($INDEX => _mm256_extract_epi32::<$INDEX>(self.0).cast_unsigned(),)*
              8.. => unreachable!(),
            }
          } else {
            let [self_a, self_b] = cast::<u32x8, [u32x4; 2]>(self);

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
        use_indices! {
          0..8 = [0, 1, 2, 3, 4, 5, 6, 7],
          0..4 = [0, 1, 2, 3],
          4..8 = [4, 5, 6, 7]
        }
    }
}
}

safe_target_feature! {
// Backend of SIMD integer functions.
impl u32x8 {
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
            Self(_mm256_add_epi32(self.0, rhs.0))
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
            Self(_mm256_sub_epi32(self.0, rhs.0))
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
            Self(_mm256_mullo_epi32(self.0, rhs.0))
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
    pub(crate) fn reduce_add_backend(self) -> u32 {
        let array: [u32x4; 2] = cast(self);
        (array[0] + array[1]).reduce_add()
    }

    #[inline]
    pub(crate) fn reduce_mul_backend(self) -> u32 {
        let array: [u32x4; 2] = cast(self);
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
    pub(crate) fn shl_backend(self, rhs: u32x8) -> Self {
        pick! {
          if #[cfg(target_feature="avx2")] {
            // ensure same behavior as scalar wrapping_shl
            let shift_by = _mm256_and_si256(rhs.0, _mm256_set1_epi32(31));
            Self(_mm256_sllv_epi32(self.0, shift_by))
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
            // Use `rhs % 32` to perform wrapping shift and not unbounded shift.
            let shift = cast([rhs as u64 & 31, 0]);
            Self(_mm256_sll_epi32(self.0, shift))
          } else {
            Self(Inner {
              lo: self.0.lo << rhs,
              hi: self.0.hi << rhs,
            })
          }
        }
    }

    #[inline]
    pub(crate) fn shr_backend(self, rhs: u32x8) -> Self {
        pick! {
          if #[cfg(target_feature="avx2")] {
            // ensure same behavior as scalar wrapping_shr
            let shift_by = _mm256_and_si256(rhs.0, _mm256_set1_epi32(31));
            Self(_mm256_srlv_epi32(self.0, shift_by ))
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
            // Use `rhs % 32` to perform wrapping shift and not unbounded shift.
            let shift = cast([rhs as u64 & 31, 0]);
            Self(_mm256_srl_epi32(self.0, shift))
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
            Self(_mm256_max_epu32(self.0, rhs.0 ))
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
            Self(_mm256_min_epu32(self.0, rhs.0 ))
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
        let array: [u32x4; 2] = cast(self);
        array[0].max(array[1]).reduce_max()
    }

    #[inline]
    pub(crate) fn reduce_min_backend(self) -> u32 {
        let array: [u32x4; 2] = cast(self);
        array[0].min(array[1]).reduce_min()
    }

    #[inline]
    pub(crate) fn unbounded_shl_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="avx2")] {
            Self(_mm256_sllv_epi32(self.0, rhs.0))
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
            Self(_mm256_sll_epi32(self.0, cast([rhs as u64, 0])))
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
            Self(_mm256_srlv_epi32(self.0, rhs.0))
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
            Self(_mm256_srl_epi32(self.0, cast([rhs as u64, 0])))
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
        let (low, high) = self.mul_keep_low_high(rhs);
        let overflow = high.simd_ne(Self::ZERO);
        (low, overflow)
    }

    #[inline]
    pub(crate) fn widening_mul_backend(self, rhs: Self) -> u64x8 {
        pick! {
          if #[cfg(all(target_feature="avx512f", target_feature="avx2"))] {
            const SHUFFLE_INDICES: __m512i = i64x8::from_array([0, 4, 1, 5, 2, 6, 3, 7]).0;

            let even_wide_mul = _mm256_mul_epu32(self.0, rhs.0);
            let odd_wide_mul = _mm256_mul_epu32(
              _mm256_shuffle_epi32::<0b_00_11_00_01>(self.0),
              _mm256_shuffle_epi32::<0b_00_11_00_01>(rhs.0),
            );
            let even_then_odd = cast::<[__m256i; 2], __m512i>([even_wide_mul, odd_wide_mul]);
            u64x8(_mm512_permutexvar_epi64(SHUFFLE_INDICES, even_then_odd))
          } else {
            let [self_a, self_b] = cast::<u32x8, [u32x4; 2]>(self);
            let [rhs_a, rhs_b] = cast::<u32x8, [u32x4; 2]>(rhs);

            cast([self_a.widening_mul(rhs_a), self_b.widening_mul(rhs_b)])
          }
        }
    }

    #[inline]
    pub(crate) fn mul_keep_low_high_backend(self, rhs: Self) -> (Self, Self) {
        pick! {
          if #[cfg(target_feature="avx2")] {
            let even_wide_mul = _mm256_mul_epu32(self.0, rhs.0);
            let odd_wide_mul = _mm256_mul_epu32(
              _mm256_shuffle_epi32::<0b_00_11_00_01>(self.0),
              _mm256_shuffle_epi32::<0b_00_11_00_01>(rhs.0),
            );
            let ll_hh_1 = _mm256_unpacklo_epi32(even_wide_mul, odd_wide_mul);
            let ll_hh_2 = _mm256_unpackhi_epi32(even_wide_mul, odd_wide_mul);
            (
              Self(_mm256_unpacklo_epi64(ll_hh_1, ll_hh_2)),
              Self(_mm256_unpackhi_epi64(ll_hh_1, ll_hh_2)),
            )
          } else {
            let [self_a, self_b] = cast::<u32x8, [u32x4; 2]>(self);
            let [rhs_a, rhs_b] = cast::<u32x8, [u32x4; 2]>(rhs);

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
    pub(crate) fn mul_keep_high_backend(self, rhs: u32x8) -> u32x8 {
        pick! {
          if #[cfg(target_feature="avx2")] {
            let a : [u32;8]= cast(self);
            let b : [u32;8]= cast(rhs);

            // let the compiler shuffle the values around, it does the right thing
            let r1 : [u32;8] = cast(_mm256_mul_epu32(cast([a[0], 0, a[1], 0, a[2], 0, a[3], 0]), cast([b[0], 0, b[1], 0, b[2], 0, b[3], 0])));
            let r2 : [u32;8] = cast(_mm256_mul_epu32(cast([a[4], 0, a[5], 0, a[6], 0, a[7], 0]), cast([b[4], 0, b[5], 0, b[6], 0, b[7], 0])));

            cast([r1[1], r1[3], r1[5], r1[7], r2[1], r2[3], r2[5], r2[7]])
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
impl u32x8 {
    #[inline]
    pub(crate) fn saturating_add_signed_backend(self, rhs: i32x8) -> Self {
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
impl u32x8 {
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
    fn to_byte_indices(self) -> u8x32 {
        // The byte offset of the lane, broadcast to every byte of the lane.
        let base = self.unbounded_shl_scalar(2);
        let base = base | base.unbounded_shl_scalar(8);
        let base = base | base.unbounded_shl_scalar(16);

        // Then the offset of each byte within its lane. These bits are free because
        // every byte of `base` is a multiple of four. `from_ne_bytes` keeps this
        // correct on big endian, where the bytes of a lane are the other way around.
        const WITHIN_LANE: u32x8 = u32x8::splat(u32::from_ne_bytes([0, 1, 2, 3]));

        cast::<u32x8, u8x32>(base | WITHIN_LANE)
    }

    #[inline]
    pub(crate) fn from_u16x8_backend(v: u16x8) -> Self {
        pick! {
          if #[cfg(target_feature="avx2")] {
            Self(_mm256_cvtepu16_epi32(v.0))
          } else if #[cfg(target_feature="sse2")] {
            Self(Inner {
              lo: u32x4(_mm_srli_epi32::<16>( _mm_unpacklo_epi16(v.0, v.0))),
              hi: u32x4(_mm_srli_epi32::<16>( _mm_unpackhi_epi16(v.0, v.0))),
            })
          } else {
            u32x8::from_array([
              u32::from(v.as_array()[0]),
              u32::from(v.as_array()[1]),
              u32::from(v.as_array()[2]),
              u32::from(v.as_array()[3]),
              u32::from(v.as_array()[4]),
              u32::from(v.as_array()[5]),
              u32::from(v.as_array()[6]),
              u32::from(v.as_array()[7]),
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
