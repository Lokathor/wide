use crate::backend_imports::*;

#[cfg(not(target_feature = "avx512bw"))]
#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub(crate) struct Inner {
    pub lo: i8x32,
    pub hi: i8x32,
}

safe_target_feature! {
impl SimdBackend for i8x64 {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
        type Inner = __m512i;
      } else {
        type Inner = Inner;
      }
    }

    #[inline]
    pub(crate) fn simd_lt_backend(self, rhs: Self) -> Self {
        rhs.simd_gt(self)
    }

    #[inline]
    pub(crate) fn simd_gt_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="avx512bw")] {
            Self(_mm512_maskz_mov_epi8(
              _mm512_cmp_epi8_mask::<{_MM_CMPINT_NLE}>(self.0, rhs.0),
              _mm512_set1_epi8(-1),
            ))
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
            Self(_mm512_maskz_mov_epi8(
              _mm512_cmp_epi8_mask::<{_MM_CMPINT_LE}>(self.0, rhs.0),
              _mm512_set1_epi8(-1),
            ))
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
            Self(_mm512_maskz_mov_epi8(
              _mm512_cmp_epi8_mask::<{_MM_CMPINT_NLT}>(self.0, rhs.0),
              _mm512_set1_epi8(-1),
            ))
          } else {
            Self(Inner {
              lo: self.0.lo.simd_ge(rhs.0.lo),
              hi: self.0.hi.simd_ge(rhs.0.hi),
            })
          }
        }
    }
}
}

safe_target_feature! {
// Backend of SIMD integer functions.
impl i8x64 {
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
            Self(_mm512_max_epi8(self.0, rhs.0))
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
            Self(_mm512_min_epi8(self.0, rhs.0))
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
        cast([
            self_a.unbounded_shr_scalar(rhs),
            self_b.unbounded_shr_scalar(rhs),
        ])
    }

    #[inline]
    pub(crate) fn saturating_add_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="avx512bw")] {
            Self(_mm512_adds_epi8(self.0, rhs.0))
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
            Self(_mm512_subs_epi8(self.0, rhs.0))
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

    #[inline]
    pub(crate) fn mul_keep_low_high_backend(self, rhs: Self) -> (u8x64, i8x64) {
        // There is no `_mm512_mullo_epi8`/`_mm512_mulhi_epi8` intrinsic, so there
        // is no `avx512bw` optimization.

        let [self_a, self_b] = cast::<i8x64, [i8x32; 2]>(self);
        let [rhs_a, rhs_b] = cast::<i8x64, [i8x32; 2]>(rhs);

        let result_a = self_a.mul_keep_low_high(rhs_a);
        let result_b = self_b.mul_keep_low_high(rhs_b);
        (
            cast([result_a.0, result_b.0]),
            cast([result_a.1, result_b.1]),
        )
    }

    #[inline]
    pub(crate) fn mul_keep_high_backend(self, rhs: Self) -> Self {
        // There is no `_mm512_mulhi_epi8` intrinsic, so there is no `avx512bw`
        // optimization.

        let [self_a, self_b] = cast::<i8x64, [i8x32; 2]>(self);
        let [rhs_a, rhs_b] = cast::<i8x64, [i8x32; 2]>(rhs);

        cast([self_a.mul_keep_high(rhs_a), self_b.mul_keep_high(rhs_b)])
    }
}
}

safe_target_feature! {
// Backend of SIMD signed integer functions.
impl i8x64 {
    #[inline]
    pub(crate) fn abs_backend(self) -> Self {
        pick! {
          if #[cfg(target_feature="avx512bw")] {
            Self(_mm512_abs_epi8(self.0))
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

    #[inline]
    pub(crate) fn saturating_neg_backend(self) -> Self {
        pick! {
            if #[cfg(target_feature="avx512bw")] {
                Self(_mm512_subs_epi8(_mm512_setzero_si512(), self.0))
            } else {
                Self(Inner {
                    lo: self.0.lo.saturating_neg(),
                    hi: self.0.hi.saturating_neg(),
                })
            }
        }
    }

    #[inline]
    pub(crate) fn saturating_abs_backend(self) -> Self {
        pick! {
            if #[cfg(target_feature = "avx512bw")] {
                // If `self == MIN`, flip all bits to change the result from
                // `MIN` to `MAX`
                self.abs() ^ self.simd_eq(Self::MIN)
            } else {
                Self(Inner {
                    lo: self.0.lo.saturating_abs(),
                    hi: self.0.hi.saturating_abs(),
                })
            }
        }
    }

    #[inline]
    pub(crate) fn saturating_add_unsigned_backend(self, rhs: u8x64) -> Self {
        pick! {
            if #[cfg(target_feature = "avx512bw")] {
                let (result, overflow) = self.overflowing_add_unsigned(rhs);
                overflow.select(Self::MAX, result)
            } else {
                Self(Inner {
                    lo: self.0.lo.saturating_add_unsigned(rhs.0.lo),
                    hi: self.0.hi.saturating_add_unsigned(rhs.0.hi),
                })
            }
        }
    }
}
}
