use crate::imports::*;

pick! {
  if #[cfg(target_feature="avx512bw")] {
    pub(crate) type Inner = m512i;
  } else {
    #[repr(C, align(64))]
    #[derive(Clone, Copy)]
    pub(crate) struct Inner {
      pub lo: i8x32,
      pub hi: i8x32,
    }
  }
}

impl i8x64 {
  unsafe {
    T = i8,
    N = 64,
    N_STR = "64",
    Simd = i8x64,
    UnsignedSimd = u8x64,
    T_BITS = 8,
    T_BITS_MUL_2 = 16,
    BitmaskType = u64,
    [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
        32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
        48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63
    ],
    optional_type_x86_inner { X86Inner = __m512i },
    optional_type_arm_inner {},
    optional_type_wasm_inner {},
  }

  #[inline]
  fn simd_lt(self, rhs: Self) -> Self::Output {
    rhs.simd_gt(self)
  }

  #[inline]
  fn simd_gt(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
        Self(cmp_op_mask_i8_m512i::<{cmp_int_op!(Nle)}>(self.0, rhs.0))
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
      if #[cfg(target_feature="avx512bw")] {
        Self(cmp_op_mask_i8_m512i::<{cmp_int_op!(Le)}>(self.0, rhs.0))
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
      if #[cfg(target_feature="avx512bw")] {
        Self(cmp_op_mask_i8_m512i::<{cmp_int_op!(Nlt)}>(self.0, rhs.0))
      } else {
        Self(Inner {
          lo: self.0.lo.simd_ge(rhs.0.lo),
          hi: self.0.hi.simd_ge(rhs.0.hi),
        })
      }
    }
  }

  #[inline]
  fn shr(self, rhs: u8x64) -> Self::Output {
    // There's no dedicated `i8` shift instruction, so we split into halves
    // and let `i8x32` handle it.
    let [self_a, self_b]: [i8x32; 2] = cast(self);
    let [rhs_a, rhs_b]: [u8x32; 2] = cast(rhs);
    cast([self_a >> rhs_a, self_b >> rhs_b])
  }

  #[inline]
  fn shr(self, rhs: u32) -> Self::Output {
    let [self_a, self_b]: [i8x32; 2] = cast(self);
    cast([self_a >> rhs, self_b >> rhs])
  }

  #[inline]
  pub fn max(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
        Self(max_i8_m512i(self.0, rhs.0))
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
      if #[cfg(target_feature="avx512bw")] {
        Self(min_i8_m512i(self.0, rhs.0))
      } else {
        Self(Inner {
          lo: self.0.lo.min(rhs.0.lo),
          hi: self.0.hi.min(rhs.0.hi),
        })
      }
    }
  }

  #[inline]
  pub fn reduce_max(self) -> i8 {
    let array: [i8x32; 2] = cast(self);
    array[0].max(array[1]).reduce_max()
  }

  #[inline]
  pub fn reduce_min(self) -> i8 {
    let array: [i8x32; 2] = cast(self);
    array[0].min(array[1]).reduce_min()
  }

  #[inline]
  pub fn unbounded_shr(self, rhs: u8x64) -> Self {
    let [self_a, self_b] = cast::<i8x64, [i8x32; 2]>(self);
    let [rhs_a, rhs_b] = cast::<u8x64, [u8x32; 2]>(rhs);
    cast([self_a.unbounded_shr(rhs_a), self_b.unbounded_shr(rhs_b)])
  }

  #[inline]
  pub fn unbounded_shr_scalar(self, rhs: u32) -> Self {
    let [self_a, self_b] = cast::<i8x64, [i8x32; 2]>(self);
    cast([self_a.unbounded_shr_scalar(rhs), self_b.unbounded_shr_scalar(rhs)])
  }

  #[inline]
  pub fn saturating_add(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
        Self(add_saturating_i8_m512i(self.0, rhs.0))
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
      if #[cfg(target_feature="avx512bw")] {
        Self(sub_saturating_i8_m512i(self.0, rhs.0))
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
    let low = cast::<u8x64, i8x64>(low);

    let overflow = high.simd_ne(low.is_negative());
    (low, overflow)
  }

  optional_fn_widening_mul {
    // Cannot have `widening_mul` because there is no `i16x64` type.
  }

  #[inline]
  pub fn mul_keep_low_high(self, rhs: Self) -> (u8x64, i8x64) {
    // There is no `_mm512_mullo_epi8`/`_mm512_mulhi_epi8` intrinsic, so there
    // is no `avx512bw` optimization.

    let [self_a, self_b] = cast::<i8x64, [i8x32; 2]>(self);
    let [rhs_a, rhs_b] = cast::<i8x64, [i8x32; 2]>(rhs);

    let result_a = self_a.mul_keep_low_high(rhs_a);
    let result_b = self_b.mul_keep_low_high(rhs_b);
    (cast([result_a.0, result_b.0]), cast([result_a.1, result_b.1]))
  }

  #[inline]
  pub fn mul_keep_high(self, rhs: Self) -> Self {
    // There is no `_mm512_mulhi_epi8` intrinsic, so there is no `avx512bw`
    // optimization.

    let [self_a, self_b] = cast::<i8x64, [i8x32; 2]>(self);
    let [rhs_a, rhs_b] = cast::<i8x64, [i8x32; 2]>(rhs);

    cast([self_a.mul_keep_high(rhs_a), self_b.mul_keep_high(rhs_b)])
  }

  #[inline]
  pub fn abs(self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
        Self(abs_i8_m512i(self.0))
      } else {
        Self(Inner {
          lo: self.0.lo.abs(),
          hi: self.0.hi.abs(),
        })
      }
    }
  }

  #[inline]
  pub fn is_positive(self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
        self.simd_gt(Self::ZERO)
      } else {
        Self(Inner {
          lo: self.0.lo.is_positive(),
          hi: self.0.hi.is_positive(),
        })
      }
    }
  }

  #[inline]
  pub fn is_negative(self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
        self.simd_lt(Self::ZERO)
      } else {
        Self(Inner {
          lo: self.0.lo.is_negative(),
          hi: self.0.hi.is_negative(),
        })
      }
    }
  }
}
