use crate::imports::*;

pick! {
  if #[cfg(target_feature="avx512f")] {
    type Inner = m512i;
  } else {
    #[repr(C, align(64))]
    #[derive(Clone, Copy)]
    pub(crate) struct Inner {
      pub lo: u32x8,
      pub hi: u32x8,
    }
  }
}

define_simd_unsigned! {
  unsafe {
    T = u32,
    N = 16,
    N_STR = "16",
    Simd = u32x16,
    SignedSimd = i32x16,
    T_BITS = 32,
    T_BITS_MUL_2 = 64,
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    optional_type_x86_inner { X86Inner = __m512i },
    optional_type_arm_inner {},
    optional_type_wasm_inner {},
  }

  #[inline]
  fn not(self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self(bitxor_m512i(self.0, set_splat_i32_m512i(-1)))
      } else {
        Self(Inner {
          lo: self.0.lo.not(),
          hi: self.0.hi.not(),
        })
      }
    }
  }

  #[inline]
  fn add(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self(add_i32_m512i(self.0, rhs.0))
      } else {
        Self(Inner {
          lo: self.0.lo.add(rhs.0.lo),
          hi: self.0.hi.add(rhs.0.hi),
        })
      }
    }
  }

  #[inline]
  fn sub(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self(sub_i32_m512i(self.0, rhs.0))
      } else {
        Self(Inner {
          lo: self.0.lo.sub(rhs.0.lo),
          hi: self.0.hi.sub(rhs.0.hi),
        })
      }
    }
  }

  #[inline]
  fn mul(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self(mul_i32_keep_low_m512i(self.0, rhs.0))
      } else {
        Self(Inner {
          lo: self.0.lo.mul(rhs.0.lo),
          hi: self.0.hi.mul(rhs.0.hi),
        })
      }
    }
  }

  #[inline]
  fn bitand(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self(bitand_m512i(self.0, rhs.0))
      } else {
        Self(Inner {
          lo: self.0.lo.bitand(rhs.0.lo),
          hi: self.0.hi.bitand(rhs.0.hi),
        })
      }
    }
  }

  #[inline]
  fn bitor(self, rhs: Self) -> Self::Output {
    pick! {
    if #[cfg(target_feature="avx512f")] {
        Self(bitor_m512i(self.0, rhs.0))
      } else {
        Self(Inner {
          lo: self.0.lo.bitor(rhs.0.lo),
          hi: self.0.hi.bitor(rhs.0.hi),
        })
      }
    }
  }

  #[inline]
  fn bitxor(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self(bitxor_m512i(self.0, rhs.0))
      } else {
        Self(Inner {
          lo: self.0.lo.bitxor(rhs.0.lo),
          hi: self.0.hi.bitxor(rhs.0.hi),
        })
      }
    }
  }

  #[inline]
  fn simd_eq(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self(cmp_op_mask_u32_m512i::<{cmp_int_op!(Eq)}>(self.0, rhs.0))
      } else {
        Self(Inner {
          lo: self.0.lo.simd_eq(rhs.0.lo),
          hi: self.0.hi.simd_eq(rhs.0.hi),
        })
      }
    }
  }

  #[inline]
  fn simd_ne(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self(cmp_op_mask_u32_m512i::<{cmp_int_op!(Ne)}>(self.0, rhs.0))
      } else {
        Self(Inner {
          lo: self.0.lo.simd_ne(rhs.0.lo),
          hi: self.0.hi.simd_ne(rhs.0.hi),
        })
      }
    }
  }

  #[inline]
  fn simd_lt(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self(cmp_op_mask_u32_m512i::<{cmp_int_op!(Lt)}>(self.0, rhs.0))
      } else {
        Self(Inner {
          lo: rhs.0.lo.simd_gt(self.0.lo),
          hi: rhs.0.hi.simd_gt(self.0.hi),
        })
      }
    }
  }

  #[inline]
  fn simd_gt(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self(cmp_op_mask_u32_m512i::<{cmp_int_op!(Nle)}>(self.0, rhs.0))
      } else {
        Self(Inner {
          lo: self.0.lo.simd_gt(rhs.0.lo),
          hi: self.0.hi.simd_gt(rhs.0.hi),
        })
      }
    }
  }

  #[inline]
  fn simd_le(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self(cmp_op_mask_u32_m512i::<{cmp_int_op!(Le)}>(self.0, rhs.0))
      } else {
        Self(Inner {
          lo: self.0.lo.simd_le(rhs.0.lo),
          hi: self.0.hi.simd_le(rhs.0.hi),
        })
      }
    }
  }

  #[inline]
  fn simd_ge(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self(cmp_op_mask_u32_m512i::<{cmp_int_op!(Nlt)}>(self.0, rhs.0))
      } else {
        Self(Inner {
          lo: self.0.lo.simd_ge(rhs.0.lo),
          hi: self.0.hi.simd_ge(rhs.0.hi),
        })
      }
    }
  }

  #[inline]
  pub fn replace_const<const INDEX: usize>(self, value: u32) -> Self {
    const {
      assert!(INDEX < 16, "attempt to call `Simd::replace_const` with an out of bounds index");
    }

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
            #[cfg(target_arch = "x86")]
            use core::arch::x86::_mm512_mask_set1_epi32;
            #[cfg(target_arch = "x86_64")]
            use core::arch::x86_64::_mm512_mask_set1_epi32;

            // TODO(safe_arch): Add `_mm512_mask_set1_epi32`
            unsafe {
              Self(m512i(_mm512_mask_set1_epi32(
                  self.0.0,
                  const { 1 << INDEX },
                  value.cast_signed(),
              )))
            }
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
  pub fn extract_const<const INDEX: usize>(self) -> u32 {
    const {
      assert!(INDEX < 16, "attempt to call `Simd::extract_const` with an out of bounds index");
    }

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
            #[cfg(target_arch = "x86")]
            use core::arch::x86::_mm512_extracti32x4_epi32;
            #[cfg(target_arch = "x86_64")]
            use core::arch::x86_64::_mm512_extracti32x4_epi32;

            // TODO(safe_arch): Add `_mm512_extracti32x4_epi32`
            unsafe {
              match INDEX {
                $($INDEX => {
                  let quarter = cast::<m128i, u32x4>(m128i(
                    _mm512_extracti32x4_epi32::<{ $INDEX / 4 }>(self.0.0),
                  ));
                  quarter.extract_const::<{ $INDEX % 4 }>()
                })*
                16.. => unreachable!()
              }
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

  #[inline]
  pub fn reduce_add(self) -> u32 {
    let array: [u32x8; 2] = cast(self);
    (array[0] + array[1]).reduce_add()
  }

  #[inline]
  pub fn reduce_mul(self) -> u32 {
    let array: [u32x8; 2] = cast(self);
    (array[0] * array[1]).reduce_mul()
  }

  #[inline]
  pub fn bitselect(self, if_one: Self, if_zero: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self(bitor_m512i(
            bitand_m512i(if_one.0, self.0),
            bitandnot_m512i(self.0, if_zero.0),
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
  fn select(self, if_true: Self, if_false: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self(blend_varying_i8_m512i(if_false.0,if_true.0,movepi8_mask_m512i(self.0)))
      } else {
        Self(Inner {
          lo: self.0.lo.select(if_true.0.lo, if_false.0.lo),
          hi: self.0.hi.select(if_true.0.hi, if_false.0.hi),
        })
      }
    }
  }

  #[inline]
  pub fn to_bitmask(self) -> u32 {
    pick! {
      if #[cfg(target_feature="avx512dq")] {
        movepi32_mask_m512i(self.0) as u32
      } else {
        f32x16::to_bitmask(cast(self))
      }
    }
  }

  #[inline]
  pub fn any(self) -> bool {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
        movepi32_mask_m512i(self.0) != 0
      } else {
        let [a, b]: [u32x8; 2] = cast(self);
        (a | b).any()
      }
    }
  }

  #[inline]
  pub fn all(self) -> bool {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
        movepi32_mask_m512i(self.0) == 0xFFFF
      } else {
        let [a, b]: [u32x8; 2] = cast(self);
        (a & b).all()
      }
    }
  }

  #[inline]
  pub fn interleave_low(self, other: Self) -> Self {
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
  pub fn interleave_high(self, other: Self) -> Self {
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
  pub fn shuffle(self, indices: u32x16) -> Self {
    pick! {
      if #[cfg(all(target_feature = "avx512f"))] {
        Self(permute_i32_m512i(indices.0, self.0))
      } else {
        let self_halfs = cast::<u32x16, [u32x8; 2]>(self);
        let [indices_a, indices_b] = cast::<u32x16, [u32x8; 2]>(indices);

        cast([self_halfs.shuffle(indices_a), self_halfs.shuffle(indices_b)])
      }
    }
  }

  #[inline]
  pub fn shuffle_zeroing(self, indices: u32x16) -> Self {
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
  pub fn shuffle_wrapping(self, indices: u32x16) -> Self {
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
  fn shuffle(self: [u32x16; 2], indices: u32x16) -> u32x16 {
    pick! {
      if #[cfg(all(target_feature = "avx512f"))] {
        u32x16(shuffle_abv_i32_all_m512i(self[0].0, indices.0, self[1].0))
      } else {
        self[0].shuffle_zeroing(indices) | self[1].shuffle_zeroing(indices - 16)
      }
    }
  }

  #[inline]
  fn shuffle_zeroing(self: [u32x16; 2], indices: u32x16) -> u32x16 {
    pick! {
      if #[cfg(all(target_feature = "avx512f"))] {
        self.shuffle(indices) & indices.simd_lt(32)
      } else {
        self.shuffle(indices)
      }
    }
  }

  #[inline]
  fn shuffle_wrapping(self: [u32x16; 2], indices: u32x16) -> u32x16 {
    pick! {
      if #[cfg(all(target_feature = "avx512f", target_feature = "avx512vl"))] {
        // `avx512` shuffle intrinsics are wrapping
        self.shuffle(indices)
      } else {
        self.shuffle(indices & 31)
      }
    }
  }

  #[inline]
  fn shuffle(self: [u32x16; 3], indices: u32x16) -> u32x16 {
    [self[0], self[1]].shuffle_zeroing(indices) | self[2].shuffle_zeroing(indices - 32)
  }

  #[inline]
  fn shuffle_zeroing(self: [u32x16; 3], indices: u32x16) -> u32x16 {
    self.shuffle(indices)
  }

  #[inline]
  fn shuffle_wrapping(self: [u32x16; 3], indices: u32x16) -> u32x16 {
    self.shuffle(indices % 48)
  }

  #[inline]
  fn shuffle(self: [u32x16; 4], indices: u32x16) -> u32x16 {
    [self[0], self[1]].shuffle_zeroing(indices) | [self[2], self[3]].shuffle_zeroing(indices - 32)
  }

  #[inline]
  fn shuffle_zeroing(self: [u32x16; 4], indices: u32x16) -> u32x16 {
    self.shuffle(indices)
  }

  #[inline]
  fn shuffle_wrapping(self: [u32x16; 4], indices: u32x16) -> u32x16 {
    self.shuffle(indices & 63)
  }

  ///
  /// Currently this function is never accelerated.
  #[inline]
  pub fn transpose(data: [Self; 16]) -> [Self; 16] {
    // Can this be optimized?

    #[inline(always)]
    fn transpose_column(data: &[u32x16; 16], index: usize) -> u32x16 {
      u32x16::new([
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
  fn shl(self, rhs: u32x16) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        let shift_by = bitand_m512i(rhs.0, set_splat_i32_m512i(31));
        Self(shl_each_u32_m512i(self.0, shift_by))
      } else {
        Self(Inner {
          lo: self.0.lo.shl(rhs.0.lo),
          hi: self.0.hi.shl(rhs.0.hi),
        })
      }
    }
  }

  #[inline]
  fn shl(self, rhs: u32) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        // Use `rhs % 32` to perform wrapping shift and not unbounded shift.
        #[expect(clippy::suspicious_arithmetic_impl)]
        let shift = rhs & 31;
        Self(shl_all_u32_m512i(self.0, shift))
      } else {
        Self(Inner {
          lo: self.0.lo.shl(rhs),
          hi: self.0.hi.shl(rhs),
        })
      }
    }
  }

  #[inline]
  fn shr(self, rhs: u32x16) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        let shift_by = bitand_m512i(rhs.0, set_splat_i32_m512i(31));
        Self(shr_each_u32_m512i(self.0, shift_by ))
      } else {
        Self(Inner {
          lo: self.0.lo.shr(rhs.0.lo),
          hi: self.0.hi.shr(rhs.0.hi),
        })
      }
    }
  }

  #[inline]
  fn shr(self, rhs: u32) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        // Use `rhs % 32` to perform wrapping shift and not unbounded shift.
        #[expect(clippy::suspicious_arithmetic_impl)]
        let shift = rhs & 31;
        Self(shr_all_u32_m512i(self.0, shift))
      } else {
        Self(Inner {
          lo: self.0.lo.shr(rhs),
          hi: self.0.hi.shr(rhs),
        })
      }
    }
  }

  #[inline]
  pub fn max(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self(max_u32_m512i(self.0, rhs.0))
      } else {
        Self(Inner {
          lo: self.0.lo.max(rhs.0.lo),
          hi: self.0.hi.max(rhs.0.hi),
        })
      }
    }
  }

  #[inline]
  pub fn min(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self(min_u32_m512i(self.0, rhs.0))
      } else {
        Self(Inner {
          lo: self.0.lo.min(rhs.0.lo),
          hi: self.0.hi.min(rhs.0.hi),
        })
      }
    }
  }

  #[inline]
  pub fn reduce_max(self) -> u32 {
    let array: [u32x8; 2] = cast(self);
    array[0].max(array[1]).reduce_max()
  }

  #[inline]
  pub fn reduce_min(self) -> u32 {
    let array: [u32x8; 2] = cast(self);
    array[0].min(array[1]).reduce_min()
  }

  #[inline]
  pub fn unbounded_shl(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self(shl_each_u32_m512i(self.0, rhs.0))
      } else {
        Self(Inner {
          lo: self.0.lo.unbounded_shl(rhs.0.lo),
          hi: self.0.hi.unbounded_shl(rhs.0.hi),
        })
      }
    }
  }

  #[inline]
  pub fn unbounded_shl_scalar(self, rhs: u32) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self(shl_all_u32_m512i(self.0, rhs))
      } else {
        Self(Inner {
          lo: self.0.lo.unbounded_shl_scalar(rhs),
          hi: self.0.hi.unbounded_shl_scalar(rhs),
        })
      }
    }
  }

  #[inline]
  pub fn unbounded_shr(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self(shr_each_u32_m512i(self.0, rhs.0))
      } else {
        Self(Inner {
          lo: self.0.lo.unbounded_shr(rhs.0.lo),
          hi: self.0.hi.unbounded_shr(rhs.0.hi),
        })
      }
    }
  }

  #[inline]
  pub fn unbounded_shr_scalar(self, rhs: u32) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self(shr_all_u32_m512i(self.0, rhs))
      } else {
        Self(Inner {
          lo: self.0.lo.unbounded_shr_scalar(rhs),
          hi: self.0.hi.unbounded_shr_scalar(rhs),
        })
      }
    }
  }

  #[inline]
  pub fn saturating_add(self, rhs: Self) -> Self {
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
  pub fn saturating_sub(self, rhs: Self) -> Self {
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
  pub fn overflowing_mul(self, rhs: Self) -> (Self, Self) {
    let (low, high) = self.mul_keep_low_high(rhs);
    let overflow = high.simd_ne(Self::ZERO);
    (low, overflow)
  }

  optional_fn_widening_mul {
    // Cannot have `widening_mul` because there is no `u64x16` type.
  }

  #[inline]
  pub fn mul_keep_low_high(self, rhs: Self) -> (Self, Self) {
    pick! {
      if #[cfg(all(target_feature="avx512f", target_feature="avx512dq"))] {
        #[cfg(target_arch = "x86")]
        use core::arch::x86::{_mm512_unpackhi_epi64, _mm512_unpacklo_epi64};
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::{_mm512_unpackhi_epi64, _mm512_unpacklo_epi64};

        let even_wide_mul = mul_u32_wide_m512i(self.0, rhs.0);
        let odd_wide_mul = mul_u32_wide_m512i(
          shuffle_i32_m512i::<0b_00_11_00_01>(self.0),
          shuffle_i32_m512i::<0b_00_11_00_01>(rhs.0),
        );

        let ll_hh_1 = unpack_low_i32_m512i(even_wide_mul, odd_wide_mul);
        let ll_hh_2 = unpack_high_i32_m512i(even_wide_mul, odd_wide_mul);
        // TODO(safe_arch): Add `_mm512_unpacklo_epi64` and `_mm512_unpackhi_epi64`.
        (
          Self(m512i(unsafe { _mm512_unpacklo_epi64(ll_hh_1.0, ll_hh_2.0) })),
          Self(m512i(unsafe { _mm512_unpackhi_epi64(ll_hh_1.0, ll_hh_2.0) })),
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
  pub fn mul_keep_high(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        let alo = extract_m256i32_from_m512i::<0>(self.0);
        let ahi = extract_m256i32_from_m512i::<1>(self.0);
        let blo = extract_m256i32_from_m512i::<0>(rhs.0);
        let bhi = extract_m256i32_from_m512i::<1>(rhs.0);

        let lo_res: m256i = {
          let a8 = u32x8(alo);
          let b8 = u32x8(blo);
          a8.mul_keep_high(b8).0
        };
        let hi_res: m256i = {
          let a8 = u32x8(ahi);
          let b8 = u32x8(bhi);
          a8.mul_keep_high(b8).0
        };

        let zero = zeroed_m512i();
        let with_lo = insert_m256i32_to_m512i::<0>(zero, lo_res);
        let combined = insert_m256i32_to_m512i::<1>(with_lo, hi_res);

        Self(combined)
      } else {
        Self(Inner {
          lo: self.0.lo.mul_keep_high(rhs.0.lo),
          hi: self.0.hi.mul_keep_high(rhs.0.hi),
        })
      }
    }
  }
}

impl From<u16x16> for u32x16 {
    /// Widens and zero-extends each u16 lane to u32
    #[inline]
    fn from(v: u16x16) -> Self {
        pick! {
          if #[cfg(target_feature = "avx512f")] {
            Self(convert_to_u32_m512i_from_u16_m256i(v.0))
          } else if #[cfg(target_feature = "avx2")] {
            let lo: m128i = extract_m128i_from_m256i::<0>(v.0);
            let hi: m128i = extract_m128i_from_m256i::<1>(v.0);
            Self(Inner {
              lo: u32x8(convert_to_i32_m256i_from_u16_m128i(lo)),
              hi: u32x8(convert_to_i32_m256i_from_u16_m128i(hi)),
            })
          } else if #[cfg(target_feature = "sse2")] {
            Self(Inner {
              lo: u32x8(crate::u32x8_::Inner {
                lo: u32x4(shr_imm_u32_m128i::<16>(unpack_low_i16_m128i(v.0.lo.0, v.0.lo.0))),
                hi: u32x4(shr_imm_u32_m128i::<16>(unpack_high_i16_m128i(v.0.lo.0, v.0.lo.0))),
              }),
              hi: u32x8(crate::u32x8_::Inner {
                lo: u32x4(shr_imm_u32_m128i::<16>(unpack_low_i16_m128i(v.0.hi.0, v.0.hi.0))),
                hi: u32x4(shr_imm_u32_m128i::<16>(unpack_high_i16_m128i(v.0.hi.0, v.0.hi.0))),
              }),
            })
          } else {
            // Portable fallback
            let arr = v.as_array();
            Self::new([
              arr[0] as u32,  arr[1] as u32,  arr[2] as u32,  arr[3] as u32,
              arr[4] as u32,  arr[5] as u32,  arr[6] as u32,  arr[7] as u32,
              arr[8] as u32,  arr[9] as u32,  arr[10] as u32, arr[11] as u32,
              arr[12] as u32, arr[13] as u32, arr[14] as u32, arr[15] as u32,
            ])
          }
        }
    }
}

/// The following functionality exists only for [`u32x16`], or only for
/// particular types inconsistently.
impl u32x16 {
    /// `self + ((a * b) mod 2^W)`, reading only the low `W` bits of each lane of
    /// `a` and `b`. `W` must be in `1..=32`.
    ///
    /// There is no IFMA equivalent at this width, and below 17 bits no widening
    /// multiply is needed either: the whole product fits a lane, so the ordinary
    /// lane multiply already yields both halves.
    #[inline]
    #[must_use]
    pub fn add_mul_lo<const W: u32>(self, a: Self, b: Self) -> Self {
        if W <= 16 {
            let mask = Self::splat(add_mul_operand_mask_u32::<W>());
            return self + (((a & mask) * (b & mask)) & mask);
        }

        let acc = self.to_array();
        let a = a.to_array();
        let b = b.to_array();
        Self::new(core::array::from_fn(|i| {
            add_mul_lo_lane_u32::<W>(acc[i], a[i], b[i])
        }))
    }

    /// `self + ((a * b) >> W)`, reading only the low `W` bits of each lane of `a`
    /// and `b`. `W` must be in `1..=32`.
    #[inline]
    #[must_use]
    pub fn add_mul_hi<const W: u32>(self, a: Self, b: Self) -> Self {
        // See `add_mul_lo`: the whole product is in the lane, so the high half is a
        // shift.
        if W <= 16 {
            let mask = Self::splat(add_mul_operand_mask_u32::<W>());
            return self + (((a & mask) * (b & mask)) >> W);
        }

        let acc = self.to_array();
        let a = a.to_array();
        let b = b.to_array();
        Self::new(core::array::from_fn(|i| {
            add_mul_hi_lane_u32::<W>(acc[i], a[i], b[i])
        }))
    }
}
