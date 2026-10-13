use crate::imports::*;

pick! {
  if #[cfg(target_feature="avx2")] {
    pub(crate) type Inner = m256i;
  } else {
    #[repr(C, align(32))]
    #[derive(Clone, Copy)]
    pub(crate) struct Inner {
      pub lo: i8x16,
      pub hi: i8x16,
    }
  }
}

impl i8x32 {
  #[inline]
  pub(crate) fn simd_lt(self, rhs: Self) -> Self::Output {
    rhs.simd_gt(self)
  }

  #[inline]
  pub(crate) fn simd_gt(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self(cmp_gt_mask_i8_m256i(self.0,rhs.0))
      } else {
        Self(Inner {
          lo: self.0.lo.simd_gt(rhs.0.lo),
          hi: self.0.hi.simd_gt(rhs.0.hi),
        })
      }
    }
  }

  #[inline]
  pub(crate) fn simd_le(self, rhs: Self) -> Self::Output {
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
  pub(crate) fn simd_ge(self, rhs: Self) -> Self::Output {
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
  pub(crate) fn shr(self, rhs: u8x32) -> Self::Output {
    // For x86, this technically can be done explicitly by converting to `i16`
    // or `i32` then converting back after multiplication, but that may not
    // actually be faster than auto-vectorization.
    let [self_a, self_b]: [i8x16; 2] = cast(self);
    let [rhs_a, rhs_b]: [u8x16; 2] = cast(rhs);
    cast([self_a >> rhs_a, self_b >> rhs_b])
  }

  #[inline]
  pub(crate) fn shr(self, rhs: u32) -> Self::Output {
    // For x86, this technically can be done explicitly by converting
    // to `i16` or `i32` then converting back after multiplication, but that
    // may not actually be faster than auto-vectorization.
    let [self_a, self_b]: [i8x16; 2] = cast(self);
    cast([self_a >> rhs, self_b >> rhs])
  }

  #[inline]
  pub(crate) fn max(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self(max_i8_m256i(self.0,rhs.0))
      } else {
        Self(Inner {
          lo: self.0.lo.max(rhs.0.lo),
          hi: self.0.hi.max(rhs.0.hi),
        })
      }
    }
  }

  #[inline]
  pub(crate) fn min(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self(min_i8_m256i(self.0,rhs.0))
      } else {
        Self(Inner {
          lo: self.0.lo.min(rhs.0.lo),
          hi: self.0.hi.min(rhs.0.hi),
        })
      }
    }
  }

  #[inline]
  pub(crate) fn reduce_max(self) -> i8 {
    let array: [i8x16; 2] = cast(self);
    array[0].max(array[1]).reduce_max()
  }

  #[inline]
  pub(crate) fn reduce_min(self) -> i8 {
    let array: [i8x16; 2] = cast(self);
    array[0].min(array[1]).reduce_min()
  }

  #[inline]
  pub(crate) fn unbounded_shr(self, rhs: u8x32) -> Self {
    // For x86, this technically can be done explicitly by converting to `i16`
    // or `i32` then converting back after multiplication, but that may not
    // actually be faster than auto-vectorization.
    let [self_a, self_b] = cast::<i8x32, [i8x16; 2]>(self);
    let [rhs_a, rhs_b] = cast::<u8x32, [u8x16; 2]>(rhs);
    cast([self_a.unbounded_shr(rhs_a), self_b.unbounded_shr(rhs_b)])
  }

  #[inline]
  pub(crate) fn unbounded_shr_scalar(self, rhs: u32) -> Self {
    // For x86, this technically can be done explicitly by converting
    // to `i16` or `i32` then converting back after multiplication, but that
    // may not actually be faster than auto-vectorization.
    let [self_a, self_b] = cast::<i8x32, [i8x16; 2]>(self);
    cast([self_a.unbounded_shr_scalar(rhs), self_b.unbounded_shr_scalar(rhs)])
  }

  #[inline]
  pub(crate) fn saturating_add(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self(add_saturating_i8_m256i(self.0, rhs.0))
      } else {
        Self(Inner {
          lo: self.0.lo.saturating_add(rhs.0.lo),
          hi: self.0.hi.saturating_add(rhs.0.hi),
        })
      }
    }
  }

  #[inline]
  pub(crate) fn saturating_sub(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self(sub_saturating_i8_m256i(self.0, rhs.0))
      } else {
        Self(Inner {
          lo: self.0.lo.saturating_sub(rhs.0.lo),
          hi: self.0.hi.saturating_sub(rhs.0.hi),
        })
      }
    }
  }

  #[inline]
  pub(crate) fn overflowing_mul(self, rhs: Self) -> (Self, Self) {
    let (low, high) = self.mul_keep_low_high(rhs);
    let low = cast::<u8x32, i8x32>(low);

    let overflow = high.simd_ne(low.is_negative());
    (low, overflow)
  }

  optional_fn_widening_mul {
    #[inline]
    pub(crate) fn widening_mul(self, rhs: Self) -> i16x32 {
      // x86 has no `_mm256_mul_epi8` intrinsic so there is no `avx2`
      // optimization.

      let [self_a, self_b] = cast::<i8x32, [i8x16; 2]>(self);
      let [rhs_a, rhs_b] = cast::<i8x32, [i8x16; 2]>(rhs);

      cast([self_a.widening_mul(rhs_a), self_b.widening_mul(rhs_b)])
    }
  }

  #[inline]
  pub(crate) fn mul_keep_low_high(self, rhs: Self) -> (u8x32, i8x32) {
    // x86 has no `_mm256_mul_epi8` intrinsic so there is no `avx2`
    // optimization.

    let [self_a, self_b] = cast::<i8x32, [i8x16; 2]>(self);
    let [rhs_a, rhs_b] = cast::<i8x32, [i8x16; 2]>(rhs);

    let result_a = self_a.mul_keep_low_high(rhs_a);
    let result_b = self_b.mul_keep_low_high(rhs_b);
    (cast([result_a.0, result_b.0]), cast([result_a.1, result_b.1]))
  }

  #[inline]
  pub(crate) fn mul_keep_high(self, rhs: Self) -> Self {
    // x86 has no `_mm256_mul_epi8` intrinsic so there is no `avx2`
    // optimization.

    let [self_a, self_b] = cast::<i8x32, [i8x16; 2]>(self);
    let [rhs_a, rhs_b] = cast::<i8x32, [i8x16; 2]>(rhs);

    cast([self_a.mul_keep_high(rhs_a), self_b.mul_keep_high(rhs_b)])
  }

  #[inline]
  pub(crate) fn abs(self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self(abs_i8_m256i(self.0))
      } else {
        Self(Inner {
          lo: self.0.lo.abs(),
          hi: self.0.hi.abs(),
        })
      }
    }
  }

  #[inline]
  pub(crate) fn is_positive(self) -> Self {
    pick! {
      if #[cfg(all(target_feature="neon", target_arch="aarch64"))] {
        // `neon` has dedicated greater-than-zero intrinsics.
        Self(Inner {
          lo: self.0.lo.is_positive(),
          hi: self.0.hi.is_positive(),
        })
      } else {
        self.simd_gt(Self::ZERO)
      }
    }
  }

  #[inline]
  pub(crate) fn is_negative(self) -> Self {
    pick! {
      if #[cfg(all(target_feature="neon", target_arch="aarch64"))] {
        // `neon` has dedicated less-than-zero intrinsics.
        Self(Inner {
          lo: self.0.lo.is_negative(),
          hi: self.0.hi.is_negative(),
        })
      } else {
        self.simd_lt(Self::ZERO)
      }
    }
  }
}

/// The following functionality exists only for [`i8x32`], or only for
/// particular types inconsistently.
impl i8x32 {
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
        pick! {
          if #[cfg(target_feature="avx2")] {
            Self(shuffle_av_i8z_half_m256i(self.0, add_saturating_u8_m256i(rhs.0, set_splat_i8_m256i(0x70))))
          } else {
              Self(Inner {
                lo: self.0.lo.shuffle_zeroing(rhs.0.lo.cast_unsigned()),
                hi: self.0.hi.shuffle_zeroing(rhs.0.hi.cast_unsigned()),
              })
          }
        }
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
    pub fn swizzle_half_relaxed(self, rhs: i8x32) -> i8x32 {
        pick! {
          if #[cfg(target_feature="avx2")] {
            Self(shuffle_av_i8z_half_m256i(self.0, rhs.0))
          } else {
            Self(Inner {
              lo: self.0.lo.shuffle(rhs.0.lo.cast_unsigned()),
              hi: self.0.hi.shuffle(rhs.0.hi.cast_unsigned()),
            })
          }
        }
    }

    /// Full 32-entry byte table lookup.
    ///
    /// * An index (interpreted as unsigned) in `[0, 31]` selects `self[index]`.
    /// * Any index `>= 32` (including negative `i8` values) yields `0`.
    ///
    /// Unlike [`swizzle_half`](Self::swizzle_half), indices address the entire
    /// 32-byte vector, not just their own 16-byte half.
    ///
    /// This function has been deprecated and replaced with [`shuffle_zeroing`].
    ///
    /// [`shuffle_zeroing`]: Self::shuffle_zeroing
    #[inline]
    #[deprecated(since = "1.7.0", note = "replaced with `shuffle_zeroing`")]
    pub fn swizzle(self, rhs: i8x32) -> i8x32 {
        self.shuffle_zeroing(rhs.cast_unsigned())
    }

    /// Like [`swizzle`](Self::swizzle), but out-of-range indices (unsigned
    /// `>= 32`) yield an implementation-defined result (`0` or `self[index %
    /// 32]`). Prefer this when you know all indices are in range; it can be
    /// cheaper.
    ///
    /// This function has been deprecated and replaced with [`shuffle`].
    ///
    /// [`shuffle`]: Self::shuffle
    #[inline]
    #[deprecated(since = "1.7.0", note = "replaced with `shuffle`")]
    pub fn swizzle_relaxed(self, rhs: i8x32) -> i8x32 {
        self.shuffle(rhs.cast_unsigned())
    }
}
