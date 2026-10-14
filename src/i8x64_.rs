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
  #[inline]
  pub(crate) fn simd_lt_backend(self, rhs: Self) -> Self {
    rhs.simd_gt(self)
  }

  #[inline]
  pub(crate) fn simd_gt_backend(self, rhs: Self) -> Self {
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
  pub(crate) fn simd_le_backend(self, rhs: Self) -> Self {
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
  pub(crate) fn simd_ge_backend(self, rhs: Self) -> Self {
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
  pub(crate) fn shr_backend(self, rhs: u8x64) -> Self {
    // There's no dedicated `i8` shift instruction, so we split into halves
    // and let `i8x32` handle it.
    let [self_a, self_b]: [i8x32; 2] = cast(self);
    let [rhs_a, rhs_b]: [u8x32; 2] = cast(rhs);
    cast([self_a >> rhs_a, self_b >> rhs_b])
  }

  #[inline]
  pub(crate) fn shr_scalar_backend(self, rhs: u32) -> Self {
    let [self_a, self_b]: [i8x32; 2] = cast(self);
    cast([self_a >> rhs, self_b >> rhs])
  }

  #[inline]
  pub(crate) fn max_backend(self, rhs: Self) -> Self {
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
  pub(crate) fn min_backend(self, rhs: Self) -> Self {
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
  pub(crate) fn reduce_max_backend(self) -> i8 {
    let array: [i8x32; 2] = cast(self);
    array[0].max(array[1]).reduce_max()
  }

  #[inline]
  pub(crate) fn reduce_min_backend(self) -> i8 {
    let array: [i8x32; 2] = cast(self);
    array[0].min(array[1]).reduce_min()
  }

  #[inline]
  pub(crate) fn unbounded_shr_backend(self, rhs: u8x64) -> Self {
    let [self_a, self_b] = cast::<i8x64, [i8x32; 2]>(self);
    let [rhs_a, rhs_b] = cast::<u8x64, [u8x32; 2]>(rhs);
    cast([self_a.unbounded_shr(rhs_a), self_b.unbounded_shr(rhs_b)])
  }

  #[inline]
  pub(crate) fn unbounded_shr_scalar_backend(self, rhs: u32) -> Self {
    let [self_a, self_b] = cast::<i8x64, [i8x32; 2]>(self);
    cast([self_a.unbounded_shr_scalar(rhs), self_b.unbounded_shr_scalar(rhs)])
  }

  #[inline]
  pub(crate) fn saturating_add_backend(self, rhs: Self) -> Self {
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
  pub(crate) fn saturating_sub_backend(self, rhs: Self) -> Self {
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
  pub(crate) fn overflowing_mul_backend(self, rhs: Self) -> (Self, Self) {
    let (low, high) = self.mul_keep_low_high(rhs);
    let low = cast::<u8x64, i8x64>(low);

    let overflow = high.simd_ne(low.is_negative());
    (low, overflow)
  }

  optional_fn_widening_mul {
    // Cannot have `widening_mul` because there is no `i16x64` type.
  }

  #[inline]
  pub(crate) fn mul_keep_low_high_backend(self, rhs: Self) -> (u8x64, i8x64) {
    // There is no `_mm512_mullo_epi8`/`_mm512_mulhi_epi8` intrinsic, so there
    // is no `avx512bw` optimization.

    let [self_a, self_b] = cast::<i8x64, [i8x32; 2]>(self);
    let [rhs_a, rhs_b] = cast::<i8x64, [i8x32; 2]>(rhs);

    let result_a = self_a.mul_keep_low_high(rhs_a);
    let result_b = self_b.mul_keep_low_high(rhs_b);
    (cast([result_a.0, result_b.0]), cast([result_a.1, result_b.1]))
  }

  #[inline]
  pub(crate) fn mul_keep_high_backend(self, rhs: Self) -> Self {
    // There is no `_mm512_mulhi_epi8` intrinsic, so there is no `avx512bw`
    // optimization.

    let [self_a, self_b] = cast::<i8x64, [i8x32; 2]>(self);
    let [rhs_a, rhs_b] = cast::<i8x64, [i8x32; 2]>(rhs);

    cast([self_a.mul_keep_high(rhs_a), self_b.mul_keep_high(rhs_b)])
  }

  #[inline]
  pub(crate) fn abs_backend(self) -> Self {
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
  pub(crate) fn is_positive_backend(self) -> Self {
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
  pub(crate) fn is_negative_backend(self) -> Self {
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
