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

impl_simd! {
  unsafe {
    T = u8,
    N = 32,
    Simd = u8x32,
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
    i8x32::to_bitmask(cast(self)) as u32
  }

  #[inline]
  pub fn any(self) -> bool {
    i8x32::any(cast(self))
  }

  #[inline]
  pub fn all(self) -> bool {
    i8x32::all(cast(self))
  }

  ///
  /// Currently this function is never accelerated.
  #[inline]
  pub fn transpose(data: [u8x32; 32]) -> [u8x32; 32] {
    cast(i8x32::transpose(cast(data)))
  }
}

impl_simd_uint! {
  unsafe {
    T = u8,
    N = 32,
    Simd = u8x32,
    [
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
      21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
    ],
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
  fn shl(self, rhs: Self) -> Self::Output {
    // For x86, this technically can be done explicitly by converting to `u16`
    // or `u32` then converting back after multiplication, but that may not
    // actually be faster than auto-vectorization.
    let [self_a, self_b]: [u8x16; 2] = cast(self);
    let [rhs_a, rhs_b]: [u8x16; 2] = cast(rhs);
    cast([self_a << rhs_a, self_b << rhs_b])
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
    // For x86, this technically can be done explicitly by converting to `u16`
    // or `u32` then converting back after multiplication, but that may not
    // actually be faster than auto-vectorization.
    let [self_a, self_b]: [u8x16; 2] = cast(self);
    let [rhs_a, rhs_b]: [u8x16; 2] = cast(rhs);
    cast([self_a >> rhs_a, self_b >> rhs_b])
  }

  #[inline]
  fn shr(self, rhs: u32) -> Self::Output {
    // For x86, this technically can be done explicitly by converting
    // to `u16` or `u32` then converting back after multiplication, but that
    // may not actually be faster than auto-vectorization.
    let [self_a, self_b]: [u8x16; 2] = cast(self);
    cast([self_a >> rhs, self_b >> rhs])
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
  pub fn reduce_add(self) -> u8 {
    cast(i8x32::reduce_add(cast(self)))
  }

  #[inline]
  pub fn reduce_mul(self) -> u8 {
    let array: [u8x16; 2] = cast(self);
    (array[0] * array[1]).reduce_mul()
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
  pub fn saturating_mul(self, rhs: Self) -> Self {
    let [self_a, self_b]: [u8x16; 2] = cast(self);
    let [rhs_a, rhs_b]: [u8x16; 2] = cast(rhs);
    cast([self_a.saturating_mul(rhs_a), self_b.saturating_mul(rhs_b)])
  }

  #[inline]
  pub fn overflowing_mul(self, rhs: Self) -> (Self, Self) {
    // x86 has no `_mm256_mul_epu8` intrinsic so there is no `avx2`
    // optimization.

    let [self_a, self_b] = cast::<u8x32, [u8x16; 2]>(self);
    let [rhs_a, rhs_b] = cast::<u8x32, [u8x16; 2]>(rhs);

    let result_a = self_a.overflowing_mul(rhs_a);
    let result_b = self_b.overflowing_mul(rhs_b);
    (cast([result_a.0, result_b.0]), cast([result_a.1, result_b.1]))
  }
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
}
