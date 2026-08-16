use super::*;

pick! {
  if #[cfg(target_feature="avx2")] {
    /// A SIMD vector with 32 elements of type [`u8`].
    ///
    /// See the [crate level documentation] for more information about SIMD
    /// vectors.
    ///
    /// [crate level documentation]: crate
    #[derive(Default, Clone, Copy, PartialEq, Eq)]
    #[repr(C, align(32))]
    pub struct u8x32 { pub(crate) avx: m256i }
  } else {
    /// A SIMD vector with 32 elements of type [`u8`].
    ///
    /// See the [crate level documentation] for more information about SIMD
    /// vectors.
    ///
    /// [crate level documentation]: crate
    #[derive(Default, Clone, Copy, PartialEq, Eq)]
    #[repr(C, align(32))]
    pub struct u8x32 { pub(crate) a : u8x16, pub(crate) b : u8x16 }
  }
}

impl_simd_uint! {
  unsafe {
    T = u8,
    N = 32,
    Simd = u8x32,
    IntSimd = i8x32,
    T_BITS = 8,
    T_BITS_MUL_2 = 16,
    [
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
      21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
    ],
    optional_type_x86_inner { X86Inner = __m256i },
    optional_type_arm_inner {},
    optional_type_wasm_inner {},
  }

  #[inline]
  fn not(self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self { avx: self.avx.not()  }
      } else {
        Self {
          a : self.a.not(),
          b : self.b.not(),
        }
      }
    }
  }

  #[inline]
  fn add(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self { avx: add_i8_m256i(self.avx,rhs.avx) }
      } else {
        Self {
          a : self.a.add(rhs.a),
          b : self.b.add(rhs.b),
        }
      }
    }
  }

  #[inline]
  fn sub(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self { avx: sub_i8_m256i(self.avx,rhs.avx) }
      } else {
        Self {
          a : self.a.sub(rhs.a),
          b : self.b.sub(rhs.b),
        }
      }
    }
  }

  #[inline]
  fn mul(self, rhs: Self) -> Self::Output {
    // For x86, this technically can be done explicitly by converting to `i16`
    // then converting back after multiplication, but that may not actually be
    // faster than auto-vectorization.
    let [self_a, self_b]: [u8x16; 2] = cast(self);
    let [rhs_a, rhs_b]: [u8x16; 2] = cast(rhs);
    cast([self_a * rhs_a, self_b * rhs_b])
  }

  #[inline]
  fn bitand(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
          Self { avx : bitand_m256i(self.avx,rhs.avx) }
      } else {
          Self {
            a : self.a.bitand(rhs.a),
            b : self.b.bitand(rhs.b),
          }
      }
    }
  }

  #[inline]
  fn bitor(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self { avx : bitor_m256i(self.avx,rhs.avx) }
      } else {
        Self {
          a : self.a.bitor(rhs.a),
          b : self.b.bitor(rhs.b),
        }
      }
    }
  }

  #[inline]
  fn bitxor(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self { avx : bitxor_m256i(self.avx,rhs.avx) }
      } else {
        Self {
          a : self.a.bitxor(rhs.a),
          b : self.b.bitxor(rhs.b),
        }
      }
    }
  }

  #[inline]
  fn simd_eq(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self { avx : cmp_eq_mask_i8_m256i(self.avx,rhs.avx) }
      } else {
        Self {
          a : self.a.simd_eq(rhs.a),
          b : self.b.simd_eq(rhs.b),
        }
      }
    }
  }

  #[inline]
  fn simd_ne(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        !self.simd_eq(rhs)
      } else {
        Self {
          a : self.a.simd_ne(rhs.a),
          b : self.b.simd_ne(rhs.b),
        }
      }
    }
  }

  #[inline]
  fn simd_lt(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        // Convert from u8 to i8.
        let offset = Self::splat(0x80);
        let self_i8 = self.bitxor(offset).avx;
        let rhs_i8 = rhs.bitxor(offset).avx;
        Self { avx: cmp_gt_mask_i8_m256i(rhs_i8, self_i8)}
      } else {
        Self { a: self.a.simd_lt(rhs.a), b: self.b.simd_lt(rhs.b) }
      }
    }
  }

  #[inline]
  fn simd_gt(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        // Convert from u8 to i8.
        let offset = Self::splat(0x80);
        let self_i8 = self.bitxor(offset).avx;
        let rhs_i8 = rhs.bitxor(offset).avx;
        Self { avx : cmp_gt_mask_i8_m256i(self_i8,rhs_i8) }
      } else {
        Self { a: self.a.simd_gt(rhs.a), b: self.b.simd_gt(rhs.b) }
      }
    }
  }

  #[inline]
  fn simd_le(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        // Convert from u8 to i8.
        let offset = Self::splat(0x80);
        let self_i8 = self.bitxor(offset).avx;
        let rhs_i8 = rhs.bitxor(offset).avx;
        let gt_mask = Self { avx : cmp_gt_mask_i8_m256i(self_i8,rhs_i8) };
        Self { avx: gt_mask.bitxor(Self::splat(0xFF)).avx }
      } else {
        Self { a: self.a.simd_le(rhs.a), b: self.b.simd_le(rhs.b) }
      }
    }
  }

  #[inline]
  fn simd_ge(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        // Convert from u8 to i8.
        let offset = Self::splat(0x80);
        let self_i8 = self.bitxor(offset).avx;
        let rhs_i8 = rhs.bitxor(offset).avx;
        let lt_mask = Self { avx: cmp_gt_mask_i8_m256i(rhs_i8, self_i8)};
        Self { avx: lt_mask.bitxor(Self::splat(0xFF)).avx }
      } else {
        Self { a: self.a.simd_ge(rhs.a), b: self.b.simd_ge(rhs.b) }
      }
    }
  }

  #[inline]
  pub fn reduce_add(self) -> u8 {
    let array: [u8x16; 2] = cast(self);
    (array[0] + array[1]).reduce_add()
  }

  #[inline]
  pub fn reduce_mul(self) -> u8 {
    let array: [u8x16; 2] = cast(self);
    (array[0] * array[1]).reduce_mul()
  }

  #[inline]
  pub fn bitselect(self, if_one: Self, if_zero: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self {
          avx: bitor_m256i(
            bitand_m256i(if_one.avx, self.avx),
            bitandnot_m256i(self.avx, if_zero.avx),
          ),
        }
      } else {
        Self {
          a: self.a.bitselect(if_one.a, if_zero.a),
          b: self.b.bitselect(if_one.b, if_zero.b),
        }
      }
    }
  }

  #[inline]
  pub fn select(self, if_true: Self, if_false: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self { avx: blend_varying_i8_m256i(if_false.avx, if_true.avx, self.avx) }
      } else {
        Self {
          a : self.a.select(if_true.a, if_false.a),
          b : self.b.select(if_true.b, if_false.b),
        }
      }
    }
  }

  #[inline]
  pub fn to_bitmask(self) -> u32 {
    pick! {
      if #[cfg(target_feature="avx2")] {
        move_mask_i8_m256i(self.avx) as u32
      } else {
        self.a.to_bitmask() | (self.b.to_bitmask() << 16)
      }
    }
  }

  #[inline]
  pub fn any(self) -> bool {
    pick! {
      if #[cfg(target_feature="avx2")] {
        move_mask_i8_m256i(self.avx) != 0
      } else {
        (self.a | self.b).any()
      }
    }
  }

  #[inline]
  pub fn all(self) -> bool {
    pick! {
      if #[cfg(target_feature="avx2")] {
        move_mask_i8_m256i(self.avx) == -1
      } else {
        (self.a & self.b).all()
      }
    }
  }

  ///
  /// Currently this function is never accelerated.
  #[inline]
  pub fn transpose(data: [Self; 32]) -> [Self; 32] {
    // Can this be optimized?

    #[inline(always)]
    fn transpose_column(data: &[u8x32; 32], index: usize) -> u8x32 {
      u8x32::new([
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
  fn shl(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(all(target_feature="avx512bw", target_feature="avx512vl"))] {
        self.shift_each_u16(rhs.avx, false, false)
      } else {
        let [self_a, self_b]: [u8x16; 2] = cast(self);
        let [rhs_a, rhs_b]: [u8x16; 2] = cast(rhs);
        cast([self_a << rhs_a, self_b << rhs_b])
      }
    }
  }

  #[inline]
  fn shl(self, rhs: u32) -> Self::Output {
    // For x86, this technically can be done explicitly by converting
    // to `u16` or `u32` then converting back after multiplication, but that
    // may not actually be faster than auto-vectorization.
    let [self_a, self_b]: [u8x16; 2] = cast(self);
    cast([self_a << rhs, self_b << rhs])
  }

  #[inline]
  fn shr(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(all(target_feature="avx512bw", target_feature="avx512vl"))] {
        self.shift_each_u16(rhs.avx, true, false)
      } else {
        let [self_a, self_b]: [u8x16; 2] = cast(self);
        let [rhs_a, rhs_b]: [u8x16; 2] = cast(rhs);
        cast([self_a >> rhs_a, self_b >> rhs_b])
      }
    }
  }

  #[inline]
  fn shr(self, rhs: u32) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        // Use `rhs % 8` to perform wrapping shift and not unbounded shift.
        #[expect(clippy::suspicious_arithmetic_impl)]
        self.shift_all_u16(m128i::from((rhs & 7) as u128), true)
      } else {
        let [self_a, self_b]: [u8x16; 2] = cast(self);
        cast([self_a >> rhs, self_b >> rhs])
      }
    }
  }

  #[inline]
  pub fn max(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self { avx: max_u8_m256i(self.avx,rhs.avx) }
      } else {
        Self {
          a : self.a.max(rhs.a),
          b : self.b.max(rhs.b),
        }
      }
    }
  }

  #[inline]
  pub fn min(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self { avx: min_u8_m256i(self.avx,rhs.avx) }
      } else {
        Self {
          a : self.a.min(rhs.a),
          b : self.b.min(rhs.b),
        }
      }
    }
  }

  #[inline]
  pub fn reduce_max(self) -> u8 {
    let array: [u8x16; 2] = cast(self);
    array[0].max(array[1]).reduce_max()
  }

  #[inline]
  pub fn reduce_min(self) -> u8 {
    let array: [u8x16; 2] = cast(self);
    array[0].min(array[1]).reduce_min()
  }

  #[inline]
  pub fn unbounded_shl(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(all(target_feature="avx512bw", target_feature="avx512vl"))] {
        self.shift_each_u16(rhs.avx, false, true)
      } else {
        let [self_a, self_b] = cast::<u8x32, [u8x16; 2]>(self);
        let [rhs_a, rhs_b] = cast::<u8x32, [u8x16; 2]>(rhs);
        cast([self_a.unbounded_shl(rhs_a), self_b.unbounded_shl(rhs_b)])
      }
    }
  }

  #[inline]
  pub fn unbounded_shl_scalar(self, rhs: u32) -> Self {
    // For x86, this technically can be done explicitly by converting
    // to `u16` or `u32` then converting back after multiplication, but that
    // may not actually be faster than auto-vectorization.
    let [self_a, self_b] = cast::<u8x32, [u8x16; 2]>(self);
    cast([self_a.unbounded_shl_scalar(rhs), self_b.unbounded_shl_scalar(rhs)])
  }

  #[inline]
  pub fn unbounded_shr(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(all(target_feature="avx512bw", target_feature="avx512vl"))] {
        self.shift_each_u16(rhs.avx, true, true)
      } else {
        let [self_a, self_b] = cast::<u8x32, [u8x16; 2]>(self);
        let [rhs_a, rhs_b] = cast::<u8x32, [u8x16; 2]>(rhs);
        cast([self_a.unbounded_shr(rhs_a), self_b.unbounded_shr(rhs_b)])
      }
    }
  }

  #[inline]
  pub fn unbounded_shr_scalar(self, rhs: u32) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        // `rhs >= 8` shifts out the whole byte.
        if rhs < 8 {
          self.shift_all_u16(m128i::from(rhs as u128), true)
        } else {
          Self::ZERO
        }
      } else {
        let [self_a, self_b] = cast::<u8x32, [u8x16; 2]>(self);
        cast([self_a.unbounded_shr_scalar(rhs), self_b.unbounded_shr_scalar(rhs)])
      }
    }
  }

  #[inline]
  pub fn saturating_add(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self { avx: add_saturating_u8_m256i(self.avx, rhs.avx) }
      } else {
        Self {
          a : self.a.saturating_add(rhs.a),
          b : self.b.saturating_add(rhs.b),
        }
      }
    }
  }

  #[inline]
  pub fn saturating_sub(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self { avx: sub_saturating_u8_m256i(self.avx, rhs.avx) }
      } else {
        Self {
          a : self.a.saturating_sub(rhs.a),
          b : self.b.saturating_sub(rhs.b),
        }
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
    #[inline]
    pub fn widening_mul(self, rhs: Self) -> u16x32 {
      // x86 has no `_mm256_mul_epu8` intrinsic so there is no `avx2`
      // optimization.

      let [self_a, self_b] = cast::<u8x32, [u8x16; 2]>(self);
      let [rhs_a, rhs_b] = cast::<u8x32, [u8x16; 2]>(rhs);

      cast([self_a.widening_mul(rhs_a), self_b.widening_mul(rhs_b)])
    }
  }

  #[inline]
  pub fn mul_keep_low_high(self, rhs: Self) -> (Self, Self) {
    // x86 has no `_mm256_mul_epu8` intrinsic so there is no `avx2`
    // optimization.

    let [self_a, self_b] = cast::<u8x32, [u8x16; 2]>(self);
    let [rhs_a, rhs_b] = cast::<u8x32, [u8x16; 2]>(rhs);

    let result_a = self_a.mul_keep_low_high(rhs_a);
    let result_b = self_b.mul_keep_low_high(rhs_b);
    (cast([result_a.0, result_b.0]), cast([result_a.1, result_b.1]))
  }

  #[inline]
  pub fn mul_keep_high(self, rhs: Self) -> Self {
    // x86 has no `_mm256_mul_epu8` intrinsic so there is no `avx2`
    // optimization.

    let [self_a, self_b] = cast::<u8x32, [u8x16; 2]>(self);
    let [rhs_a, rhs_b] = cast::<u8x32, [u8x16; 2]>(rhs);

    cast([self_a.mul_keep_high(rhs_a), self_b.mul_keep_high(rhs_b)])
  }

  optional_fn_deserialize {}
}

/// The following functionality exists only for [`u8x32`], or only for
/// particular types inconsistently.
impl u8x32 {
  /// Returns a new vector with lanes selected from the lanes of the first input
  /// vector a specified in the second input vector `rhs`.
  /// The indices i in range `[0, 15]` select the i-th element of `self`. For
  /// indices outside of the range the resulting lane is `0`.
  ///
  /// This note that is the equivalent of two parallel swizzle operations on the
  /// two halves of the vector, and the indexes each refer to the
  /// corresponding half.
  #[inline]
  pub fn swizzle_half(self, rhs: i8x32) -> i8x32 {
    cast(i8x32::swizzle_half(cast(self), cast(rhs)))
  }

  /// Indices in the range `[0, 15]` will select the i-th element of `self`. If
  /// the high bit of any element of `rhs` is set (negative) then the
  /// corresponding output lane is guaranteed to be zero. Otherwise if the
  /// element of `rhs` is within the range `[32, 127]` then the output lane is
  /// either `0` or `self[rhs[i] % 16]` depending on the implementation.
  ///
  /// This is the equivalent to two parallel swizzle operations on the two
  /// halves of the vector, and the indexes each refer to their corresponding
  /// half.
  #[inline]
  pub fn swizzle_half_relaxed(self, rhs: u8x32) -> u8x32 {
    cast(i8x32::swizzle_half_relaxed(cast(self), cast(rhs)))
  }

  /// Full 32-entry byte table lookup. An index in `[0, 31]` selects
  /// `self[index]`; any index `>= 32` yields `0`.
  #[inline]
  pub fn swizzle(self, rhs: u8x32) -> u8x32 {
    cast(i8x32::swizzle(cast(self), cast(rhs)))
  }

  /// Like [`swizzle`](Self::swizzle), but out-of-range indices yield an
  /// implementation-defined result (`0` or `self[index % 32]`).
  #[inline]
  pub fn swizzle_relaxed(self, rhs: u8x32) -> u8x32 {
    cast(i8x32::swizzle_relaxed(cast(self), cast(rhs)))
  }
}

#[cfg(target_feature = "avx2")]
impl u8x32 {
  // There's no `u8` shift instruction, so we cheat: widen every byte to a
  // `u16`, shift in the wider lane, then shrink back down to `u8`. Shifting
  // left can push bits past `u8::MAX`, so we keep only the low 8 bits before
  // the saturating pack (shifting right can't overflow, so the mask is a
  // no-op there). For the unbounded variants we cap the count at 8, because
  // shifting a byte by 8 or more gets rid of everything anyway.
  #[cfg(all(target_feature = "avx512bw", target_feature = "avx512vl"))]
  #[inline]
  fn shift_each_u16(self, rhs: m256i, right: bool, unbounded: bool) -> Self {
    let self16 = convert_to_i16_m512i_from_u8_m256i(self.avx);
    let count16 = if unbounded {
      convert_to_u16_m512i_from_u8_m256i(min_u8_m256i(
        rhs,
        set_splat_i8_m256i(8),
      ))
    } else {
      bitand_m512i(
        convert_to_u16_m512i_from_u8_m256i(rhs),
        set_splat_i16_m512i(7),
      )
    };
    let shifted = if right {
      shr_each_u16_m512i(self16, count16)
    } else {
      shl_each_u16_m512i(self16, count16)
    };
    let shifted = bitand_m512i(shifted, set_splat_i16_m512i(0xFF));
    Self {
      avx: Self::pack_u16_halves(
        extract_m256i_from_m512i::<0>(shifted),
        extract_m256i_from_m512i::<1>(shifted),
      ),
    }
  }

  // Same trick as above, but every lane is shifted by the same `count` and we
  // work on the two 128-bit halves separately (that's all AVX2 has).
  #[inline]
  fn shift_all_u16(self, count: m128i, right: bool) -> Self {
    let low =
      convert_to_i16_m256i_from_u8_m128i(extract_m128i_m256i::<0>(self.avx));
    let high =
      convert_to_i16_m256i_from_u8_m128i(extract_m128i_m256i::<1>(self.avx));
    let low = if right {
      shr_all_u16_m256i(low, count)
    } else {
      shl_all_u16_m256i(low, count)
    };
    let high = if right {
      shr_all_u16_m256i(high, count)
    } else {
      shl_all_u16_m256i(high, count)
    };
    let low = bitand_m256i(low, set_splat_i16_m256i(0xFF));
    let high = bitand_m256i(high, set_splat_i16_m256i(0xFF));
    Self { avx: Self::pack_u16_halves(low, high) }
  }

  // `pack_i16_to_u8_m256i` packs 128 bits at a time, which scrambles the lane
  // order. This un-scrambles it: split the packed result in half, interleave
  // the 64-bit chunks back together, and reassemble.
  #[inline]
  fn pack_u16_halves(low: m256i, high: m256i) -> m256i {
    let packed = pack_i16_to_u8_m256i(low, high);
    let packed_low = extract_m128i_m256i::<0>(packed);
    let packed_high = extract_m128i_m256i::<1>(packed);
    let combined_low = unpack_low_i64_m128i(packed_low, packed_high);
    let combined_high = unpack_high_i64_m128i(packed_low, packed_high);
    insert_m128i_to_m256i::<1>(
      insert_m128i_to_m256i::<0>(zeroed_m256i(), combined_low),
      combined_high,
    )
  }
}
