use crate::imports::*;

pick! {
  if #[cfg(target_feature="avx512f")] {
    type Inner = m512i;
  } else {
    #[repr(C, align(64))]
    #[derive(Clone, Copy)]
    pub(crate) struct Inner {
      pub lo: u64x4,
      pub hi: u64x4,
    }
  }
}

define_simd_unsigned! {
  unsafe {
    T = u64,
    N = 8,
    N_STR = "eight",
    SIZE_OF_SELF = 64,
    Simd = u64x8,
    SignedSimd = i64x8,
    T_BITS = 64,
    T_BITS_MUL_2 = 128,
    [0, 1, 2, 3, 4, 5, 6, 7],
    optional_type_x86_inner { X86Inner = __m512i },
    optional_type_arm_inner {},
    optional_type_wasm_inner {},
  }

  #[inline]
  fn not(self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self(bitxor_m512i(self.0, set_splat_i64_m512i(-1)))
      } else {
        Self(Inner {
          a : self.0.lo.not(),
          b : self.0.hi.not(),
        })
      }
    }
  }

  #[inline]
  fn add(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self(add_i64_m512i(self.0, rhs.0))
      } else {
        Self(Inner {
          a : self.0.lo.add(rhs.0.lo),
          b : self.0.hi.add(rhs.0.hi),
        })
      }
    }
  }

  #[inline]
  fn sub(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self(sub_i64_m512i(self.0, rhs.0))
      } else {
        Self(Inner {
          a : self.0.lo.sub(rhs.0.lo),
          b : self.0.hi.sub(rhs.0.hi),
        })
      }
    }
  }

  #[inline]
  fn mul(self, rhs: Self) -> Self::Output {
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
        Self(Inner { a: self.0.lo.mul(rhs.0.lo), b: self.0.hi.mul(rhs.0.hi) })
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
          a : self.0.lo.bitand(rhs.0.lo),
          b : self.0.hi.bitand(rhs.0.hi),
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
          a : self.0.lo.bitor(rhs.0.lo),
          b : self.0.hi.bitor(rhs.0.hi),
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
          a : self.0.lo.bitxor(rhs.0.lo),
          b : self.0.hi.bitxor(rhs.0.hi),
        })
      }
    }
  }

  #[inline]
  fn simd_eq(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self(cmp_op_mask_u64_m512i::<{cmp_int_op!(Eq)}>(self.0, rhs.0))
      } else {
        Self(Inner {
          a : self.0.lo.simd_eq(rhs.0.lo),
          b : self.0.hi.simd_eq(rhs.0.hi),
        })
      }
    }
  }

  #[inline]
  fn simd_ne(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self(cmp_op_mask_u64_m512i::<{cmp_int_op!(Ne)}>(self.0, rhs.0))
      } else {
        Self(Inner {
          a : self.0.lo.simd_ne(rhs.0.lo),
          b : self.0.hi.simd_ne(rhs.0.hi),
        })
      }
    }
  }

  #[inline]
  fn simd_lt(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self(cmp_op_mask_u64_m512i::<{cmp_int_op!(Lt)}>(self.0, rhs.0))
      } else {
        Self(Inner {
          a : self.0.lo.simd_lt(rhs.0.lo),
          b : self.0.hi.simd_lt(rhs.0.hi),
        })
      }
    }
  }

  #[inline]
  fn simd_gt(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self(cmp_op_mask_u64_m512i::<{cmp_int_op!(Nle)}>(self.0, rhs.0))
      } else {
        Self(Inner {
          a : self.0.lo.simd_gt(rhs.0.lo),
          b : self.0.hi.simd_gt(rhs.0.hi),
        })
      }
    }
  }

  #[inline]
  fn simd_le(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self(cmp_op_mask_u64_m512i::<{cmp_int_op!(Le)}>(self.0, rhs.0))
      } else {
        Self(Inner {
          a : self.0.lo.simd_le(rhs.0.lo),
          b : self.0.hi.simd_le(rhs.0.hi),
        })
      }
    }
  }

  #[inline]
  fn simd_ge(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self(cmp_op_mask_u64_m512i::<{cmp_int_op!(Nlt)}>(self.0, rhs.0))
      } else {
        Self(Inner {
          a : self.0.lo.simd_ge(rhs.0.lo),
          b : self.0.hi.simd_ge(rhs.0.hi),
        })
      }
    }
  }

  #[inline]
  pub fn replace_const<const INDEX: usize>(self, value: u64) -> Self {
    const {
      assert!(INDEX < 8, "attempt to call `Simd::replace_const` with an out of bounds index");
    }

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
            #[cfg(target_arch = "x86")]
            use core::arch::x86::_mm512_mask_set1_epi64;
            #[cfg(target_arch = "x86_64")]
            use core::arch::x86_64::_mm512_mask_set1_epi64;

            // TODO(safe_arch): Add `_mm512_mask_set1_epi64`
            unsafe {
              Self(m512i(_mm512_mask_set1_epi64(
                  self.0.0,
                  const { 1 << INDEX },
                  value.cast_signed(),
              )))
            }
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
  pub fn extract_const<const INDEX: usize>(self) -> u64 {
    const {
      assert!(INDEX < 8, "attempt to call `Simd::extract_const` with an out of bounds index");
    }

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
            #[cfg(target_arch = "x86")]
            use core::arch::x86::_mm512_extracti32x4_epi32;
            #[cfg(target_arch = "x86_64")]
            use core::arch::x86_64::_mm512_extracti32x4_epi32;

            // TODO(safe_arch): Add `_mm512_extracti32x4_epi32`
            unsafe {
              match INDEX {
                $($INDEX => {
                  let quarter = cast::<m128i, u64x2>(m128i(
                    _mm512_extracti32x4_epi32::<{ $INDEX / 2 }>(self.0.0),
                  ));
                  quarter.extract_const::<{ $INDEX % 2 }>()
                })*
                8.. => unreachable!()
              }
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

  #[inline]
  pub fn reduce_add(self) -> u64 {
    let array: [u64x4; 2] = cast(self);
    (array[0] + array[1]).reduce_add()
  }


  #[inline]
  pub fn reduce_mul(self) -> u64 {
    let array: [u64x4; 2] = cast(self);
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
          a: self.0.lo.bitselect(if_one.0.lo, if_zero.0.lo),
          b: self.0.hi.bitselect(if_one.0.hi, if_zero.0.hi),
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
          a : self.0.lo.select(if_true.0.lo, if_false.0.lo),
          b : self.0.hi.select(if_true.0.hi, if_false.0.hi),
        })
      }
    }
  }

  #[inline]
  pub fn to_bitmask(self) -> u32 {
    pick! {
      if #[cfg(target_feature="avx512dq")] {
        // use f64 move_mask since it is the same size as i64
        movepi64_mask_m512d(cast(self.0)) as u32
      } else {
        f64x8::to_bitmask(cast(self))
      }
    }
  }

  #[inline]
  pub fn any(self) -> bool {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        movepi64_mask_m512d(cast(self.0)) != 0
      } else {
        let [a, b]: [i64x4; 2] = cast(self);
        (a | b).any()
      }
    }
  }

  #[inline]
  pub fn all(self) -> bool {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
        movepi64_mask_m512d(cast(self.0)) == 0b11111111
      } else {
        let [a, b]: [i64x4; 2] = cast(self);
        (a & b).all()
      }
    }
  }

  #[inline]
  pub fn unpack_lo(self, other: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        // `_mm512_unpacklo_epi64` cannot be used because it acts within each
        // 128-bit lane, which is a different operation.
        let [aa, _]: [u64x4; 2] = cast(self);
        let [ba, _]: [u64x4; 2] = cast(other);
        cast([aa.unpack_lo(ba), aa.unpack_hi(ba)])
      } else {
        Self(Inner { a: self.0.lo.unpack_lo(other.0.lo), b: self.0.lo.unpack_hi(other.0.lo) })
      }
    }
  }

  #[inline]
  pub fn unpack_hi(self, other: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        // `_mm512_unpackhi_epi64` cannot be used because it acts within each
        // 128-bit lane, which is a different operation.
        let [_, ab]: [u64x4; 2] = cast(self);
        let [_, bb]: [u64x4; 2] = cast(other);
        cast([ab.unpack_lo(bb), ab.unpack_hi(bb)])
      } else {
        Self(Inner { a: self.0.hi.unpack_lo(other.0.hi), b: self.0.hi.unpack_hi(other.0.hi) })
      }
    }
  }

  #[inline]
  pub fn shuffle(self, indices: u64x8) -> Self {
    pick! {
      if #[cfg(all(target_feature = "avx512f"))] {
        Self(permute_i64_m512i(indices.0, self.0))
      } else {
        let self_halfs = cast::<u64x8, [u64x4; 2]>(self);
        let [indices_a, indices_b] = cast::<u64x8, [u64x4; 2]>(indices);

        cast([self_halfs.shuffle(indices_a), self_halfs.shuffle(indices_b)])
      }
    }
  }

  #[inline]
  pub fn shuffle_zeroing(self, indices: u64x8) -> Self {
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
  pub fn shuffle_wrapping(self, indices: u64x8) -> Self {
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
  fn shuffle(self: [u64x8; 2], indices: u64x8) -> u64x8 {
    pick! {
      if #[cfg(all(target_feature = "avx512f"))] {
        u64x8(shuffle_abv_i64_all_m512i(self[0].0, indices.0, self[1].0))
      } else {
        self[0].shuffle_zeroing(indices) | self[1].shuffle_zeroing(indices - 8)
      }
    }
  }

  #[inline]
  fn shuffle_zeroing(self: [u64x8; 2], indices: u64x8) -> u64x8 {
    pick! {
      if #[cfg(all(target_feature = "avx512f"))] {
        self.shuffle(indices) & indices.simd_lt(16)
      } else {
        self.shuffle(indices)
      }
    }
  }

  #[inline]
  fn shuffle_wrapping(self: [u64x8; 2], indices: u64x8) -> u64x8 {
    pick! {
      if #[cfg(target_feature = "avx512f")] {
        // `avx512` shuffle intrinsics are wrapping
        self.shuffle(indices)
      } else {
        self.shuffle(indices & 15)
      }
    }
  }

  #[inline]
  fn shuffle(self: [u64x8; 3], indices: u64x8) -> u64x8 {
    [self[0], self[1]].shuffle_zeroing(indices) | self[2].shuffle_zeroing(indices - 16)
  }

  #[inline]
  fn shuffle_zeroing(self: [u64x8; 3], indices: u64x8) -> u64x8 {
    self.shuffle(indices)
  }

  #[inline]
  fn shuffle_wrapping(self: [u64x8; 3], indices: u64x8) -> u64x8 {
    self.shuffle(indices % 24)
  }

  #[inline]
  fn shuffle(self: [u64x8; 4], indices: u64x8) -> u64x8 {
    [self[0], self[1]].shuffle_zeroing(indices) | [self[2], self[3]].shuffle_zeroing(indices - 16)
  }

  #[inline]
  fn shuffle_zeroing(self: [u64x8; 4], indices: u64x8) -> u64x8 {
    self.shuffle(indices)
  }

  #[inline]
  fn shuffle_wrapping(self: [u64x8; 4], indices: u64x8) -> u64x8 {
    self.shuffle(indices & 31)
  }

  ///
  /// Currently this function is never accelerated.
  #[inline]
  pub fn transpose(data: [Self; 8]) -> [Self; 8] {
    // Can this be optimized?

    #[inline(always)]
    fn transpose_column(data: &[u64x8; 8], index: usize) -> u64x8 {
      u64x8::new([
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
  fn shl(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        // Use `rhs % 64` to perform wrapping shift and not unbounded shift.
        let rhs = bitand_m512i(rhs.0, set_splat_i64_m512i(63));
        Self(shl_each_u64_m512i(self.0, rhs))
      } else {
        Self(Inner {
          a : self.0.lo.shl(rhs.0.lo),
          b : self.0.hi.shl(rhs.0.hi),
        })
      }
    }
  }

  #[inline]
  fn shl(self, rhs: u32) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        // Use `rhs % 64` to perform wrapping shift and not unbounded shift.
        #[expect(clippy::suspicious_arithmetic_impl)]
        let shift = rhs as u64 & 63;
        Self(shl_all_u64_m512i(self.0, shift))
      } else {
        Self(Inner {
          a : self.0.lo.shl(rhs),
          b : self.0.hi.shl(rhs),
        })
      }
    }
  }

  #[inline]
  fn shr(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        // Use `rhs % 64` to perform wrapping shift and not unbounded shift.
        let rhs = bitand_m512i(rhs.0, set_splat_i64_m512i(63));
        Self(shr_each_u64_m512i(self.0, rhs))
      } else {
        Self(Inner {
          a : self.0.lo.shr(rhs.0.lo),
          b : self.0.hi.shr(rhs.0.hi),
        })
      }
    }
  }

  #[inline]
  fn shr(self, rhs: u32) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        // Use `rhs % 64` to perform wrapping shift and not unbounded shift.
        #[expect(clippy::suspicious_arithmetic_impl)]
        let shift = rhs as u64 & 63;
        Self(shr_all_u64_m512i(self.0, shift))
      } else {
        Self(Inner {
          a : self.0.lo.shr(rhs),
          b : self.0.hi.shr(rhs),
        })
      }
    }
  }

  #[inline]
  pub fn max(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self(max_u64_m512i(self.0, rhs.0))
      } else {
        Self(Inner {
          a: self.0.lo.max(rhs.0.lo),
          b: self.0.hi.max(rhs.0.hi),
        })
      }
    }
  }

  #[inline]
  pub fn min(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self(min_u64_m512i(self.0, rhs.0))
      } else {
        Self(Inner {
          a: self.0.lo.min(rhs.0.lo),
          b: self.0.hi.min(rhs.0.hi),
        })
      }
    }
  }

  #[inline]
  pub fn reduce_max(self) -> u64 {
    let array: [u64x4; 2] = cast(self);
    array[0].max(array[1]).reduce_max()
  }

  #[inline]
  pub fn reduce_min(self) -> u64 {
    let array: [u64x4; 2] = cast(self);
    array[0].min(array[1]).reduce_min()
  }

  #[inline]
  pub fn unbounded_shl(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self(shl_each_u64_m512i(self.0, rhs.0))
      } else {
        Self(Inner {
          a: self.0.lo.unbounded_shl(rhs.0.lo),
          b: self.0.hi.unbounded_shl(rhs.0.hi),
        })
      }
    }
  }

  #[inline]
  pub fn unbounded_shl_scalar(self, rhs: u32) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self(shl_all_u64_m512i(self.0, rhs as u64))
      } else {
        Self(Inner {
          a: self.0.lo.unbounded_shl_scalar(rhs),
          b: self.0.hi.unbounded_shl_scalar(rhs),
        })
      }
    }
  }

  #[inline]
  pub fn unbounded_shr(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self(shr_each_u64_m512i(self.0, rhs.0))
      } else {
        Self(Inner {
          a: self.0.lo.unbounded_shr(rhs.0.lo),
          b: self.0.hi.unbounded_shr(rhs.0.hi),
        })
      }
    }
  }

  #[inline]
  pub fn unbounded_shr_scalar(self, rhs: u32) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self(shr_all_u64_m512i(self.0, rhs as u64))
      } else {
        Self(Inner {
          a: self.0.lo.unbounded_shr_scalar(rhs),
          b: self.0.hi.unbounded_shr_scalar(rhs),
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
          a: self.0.lo.saturating_add(rhs.0.lo),
          b: self.0.hi.saturating_add(rhs.0.hi),
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
          a: self.0.lo.saturating_sub(rhs.0.lo),
          b: self.0.hi.saturating_sub(rhs.0.hi),
        })
      }
    }
  }

  #[inline]
  pub fn overflowing_mul(self, rhs: Self) -> (Self, Self) {
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
      Self::new([
        result[0].0,
        result[1].0,
        result[2].0,
        result[3].0,
        result[4].0,
        result[5].0,
        result[6].0,
        result[7].0,
      ]),
      Self::new([
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

  optional_fn_widening_mul {
    // Cannot have `widening_mul` because there is no `u128x8` type.
  }

  #[inline]
  pub fn mul_keep_low_high(self, rhs: Self) -> (Self, Self) {
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
      Self::new([
        widening_mul[0] as u64,
        widening_mul[1] as u64,
        widening_mul[2] as u64,
        widening_mul[3] as u64,
        widening_mul[4] as u64,
        widening_mul[5] as u64,
        widening_mul[6] as u64,
        widening_mul[7] as u64,
      ]),
      Self::new([
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
  pub fn mul_keep_high(self, rhs: Self) -> Self {
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
          a: self.0.lo.mul_keep_high(rhs.0.lo),
          b: self.0.hi.mul_keep_high(rhs.0.hi),
        })
      }
    }
  }
}

/// The following functionality exists only for [`u64x8`], or only for
/// particular types inconsistently.
impl u64x8 {
    /// The exact per-lane product of `a` and `b` masked to `W` bits.
    ///
    /// Only exact for `W <= 32`, where the product still fits a lane, and
    /// callers guard on that. It is instantiated for wider `W` too, since the
    /// guard is a runtime `if` on a const, so it cannot assert the bound itself.
    #[inline]
    #[must_use]
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

            Self(mul_u32_wide_m512i(a.0, b.0))
          } else {
            // Lane-wise, so each half is independent.
            Self(Inner {
              a: u64x4::mul_masked::<W>(a.0.lo, b.0.lo),
              b: u64x4::mul_masked::<W>(a.0.hi, b.0.hi),
            })
          }
        }
    }

    /// `self + ((a * b) mod 2^W)`, reading only the low `W` bits of each lane of
    /// `a` and `b`. `W` must be in `1..=64`.
    #[inline]
    #[must_use]
    pub fn add_mul_lo<const W: u32>(self, a: Self, b: Self) -> Self {
        pick! {
          if #[cfg(target_feature="avx512ifma")] {
            // IFMA is fixed at 52 bits; any other width takes the generic path.
            if W == 52 {
              return Self(add_mul_low_u52_m512i(self.0, a.0, b.0));
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
        Self::new(core::array::from_fn(|i| {
            add_mul_lo_lane_u64::<W>(acc[i], a[i], b[i])
        }))
    }

    /// `self + ((a * b) >> W)`, reading only the low `W` bits of each lane of `a`
    /// and `b`. `W` must be in `1..=64`.
    #[inline]
    #[must_use]
    pub fn add_mul_hi<const W: u32>(self, a: Self, b: Self) -> Self {
        pick! {
          if #[cfg(target_feature="avx512ifma")] {
            // IFMA is fixed at 52 bits; any other width takes the generic path.
            if W == 52 {
              return Self(add_mul_high_u52_m512i(self.0, a.0, b.0));
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
        Self::new(core::array::from_fn(|i| {
            add_mul_hi_lane_u64::<W>(acc[i], a[i], b[i])
        }))
    }
}
