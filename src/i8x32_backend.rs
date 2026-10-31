use crate::backend_imports::*;

#[cfg(not(target_feature = "avx2"))]
#[repr(C, align(32))]
#[derive(Clone, Copy)]
pub(crate) struct Inner {
    pub lo: i8x16,
    pub hi: i8x16,
}

safe_target_feature! {
impl SimdBackend for i8x32 {
    pick! {
      if #[cfg(target_feature="avx2")] {
        type Inner = __m256i;
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
          if #[cfg(target_feature="avx2")] {
            Self(_mm256_cmpgt_epi8(self.0,rhs.0))
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
}
}

safe_target_feature! {
// Backend of SIMD integer functions.
impl i8x32 {
    #[inline]
    pub(crate) fn shr_backend(self, rhs: u8x32) -> Self {
        // For x86, this technically can be done explicitly by converting to `i16`
        // or `i32` then converting back after multiplication, but that may not
        // actually be faster than auto-vectorization.
        let [self_a, self_b]: [i8x16; 2] = cast(self);
        let [rhs_a, rhs_b]: [u8x16; 2] = cast(rhs);
        cast([self_a >> rhs_a, self_b >> rhs_b])
    }

    #[inline]
    pub(crate) fn shr_scalar_backend(self, rhs: u32) -> Self {
        // For x86, this technically can be done explicitly by converting
        // to `i16` or `i32` then converting back after multiplication, but that
        // may not actually be faster than auto-vectorization.
        let [self_a, self_b]: [i8x16; 2] = cast(self);
        cast([self_a >> rhs, self_b >> rhs])
    }

    #[inline]
    pub(crate) fn max_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="avx2")] {
            Self(_mm256_max_epi8(self.0,rhs.0))
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
            Self(_mm256_min_epi8(self.0,rhs.0))
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
        let array: [i8x16; 2] = cast(self);
        array[0].max(array[1]).reduce_max()
    }

    #[inline]
    pub(crate) fn reduce_min_backend(self) -> i8 {
        let array: [i8x16; 2] = cast(self);
        array[0].min(array[1]).reduce_min()
    }

    #[inline]
    pub(crate) fn unbounded_shr_backend(self, rhs: u8x32) -> Self {
        // For x86, this technically can be done explicitly by converting to `i16`
        // or `i32` then converting back after multiplication, but that may not
        // actually be faster than auto-vectorization.
        let [self_a, self_b] = cast::<i8x32, [i8x16; 2]>(self);
        let [rhs_a, rhs_b] = cast::<u8x32, [u8x16; 2]>(rhs);
        cast([self_a.unbounded_shr(rhs_a), self_b.unbounded_shr(rhs_b)])
    }

    #[inline]
    pub(crate) fn unbounded_shr_scalar_backend(self, rhs: u32) -> Self {
        // For x86, this technically can be done explicitly by converting
        // to `i16` or `i32` then converting back after multiplication, but that
        // may not actually be faster than auto-vectorization.
        let [self_a, self_b] = cast::<i8x32, [i8x16; 2]>(self);
        cast([
            self_a.unbounded_shr_scalar(rhs),
            self_b.unbounded_shr_scalar(rhs),
        ])
    }

    #[inline]
    pub(crate) fn saturating_add_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="avx2")] {
            Self(_mm256_adds_epi8(self.0, rhs.0))
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
            Self(_mm256_subs_epi8(self.0, rhs.0))
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
        let low = cast::<u8x32, i8x32>(low);

        let overflow = high.simd_ne(low.is_negative());
        (low, overflow)
    }

    #[inline]
    pub(crate) fn widening_mul_backend(self, rhs: Self) -> i16x32 {
        // x86 has no `_mm256_mul_epi8` intrinsic so there is no `avx2`
        // optimization.

        let [self_a, self_b] = cast::<i8x32, [i8x16; 2]>(self);
        let [rhs_a, rhs_b] = cast::<i8x32, [i8x16; 2]>(rhs);

        cast([self_a.widening_mul(rhs_a), self_b.widening_mul(rhs_b)])
    }

    #[inline]
    pub(crate) fn mul_keep_low_high_backend(self, rhs: Self) -> (u8x32, i8x32) {
        // x86 has no `_mm256_mul_epi8` intrinsic so there is no `avx2`
        // optimization.

        let [self_a, self_b] = cast::<i8x32, [i8x16; 2]>(self);
        let [rhs_a, rhs_b] = cast::<i8x32, [i8x16; 2]>(rhs);

        let result_a = self_a.mul_keep_low_high(rhs_a);
        let result_b = self_b.mul_keep_low_high(rhs_b);
        (
            cast([result_a.0, result_b.0]),
            cast([result_a.1, result_b.1]),
        )
    }

    #[inline]
    pub(crate) fn mul_keep_high_backend(self, rhs: Self) -> Self {
        // x86 has no `_mm256_mul_epi8` intrinsic so there is no `avx2`
        // optimization.

        let [self_a, self_b] = cast::<i8x32, [i8x16; 2]>(self);
        let [rhs_a, rhs_b] = cast::<i8x32, [i8x16; 2]>(rhs);

        cast([self_a.mul_keep_high(rhs_a), self_b.mul_keep_high(rhs_b)])
    }
}
}

safe_target_feature! {
// Backend of SIMD signed integer functions.
impl i8x32 {
    #[inline]
    pub(crate) fn abs_backend(self) -> Self {
        pick! {
          if #[cfg(target_feature="avx2")] {
            Self(_mm256_abs_epi8(self.0))
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
    pub(crate) fn is_negative_backend(self) -> Self {
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

    #[inline]
    pub(crate) fn saturating_neg_backend(self) -> Self {
        pick! {
          if #[cfg(target_feature="avx2")] {
              Self(_mm256_subs_epi8(_mm256_setzero_si256(), self.0))
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
            if #[cfg(target_feature = "avx2")] {
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
    pub(crate) fn saturating_add_unsigned_backend(self, rhs: u8x32) -> Self {
        pick! {
            if #[cfg(target_feature = "avx2")] {
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

safe_target_feature! {
// Internal helpers and backend of inconsistent functions.
impl i8x32 {
    #[inline]
    pub(crate) fn swizzle_half_backend(self, rhs: i8x32) -> i8x32 {
        pick! {
          if #[cfg(target_feature="avx2")] {
            Self(_mm256_shuffle_epi8(self.0, _mm256_adds_epu8(rhs.0, _mm256_set1_epi8(0x70))))
          } else {
              Self(Inner {
                lo: self.0.lo.shuffle_zeroing(rhs.0.lo.cast_unsigned()),
                hi: self.0.hi.shuffle_zeroing(rhs.0.hi.cast_unsigned()),
              })
          }
        }
    }

    #[inline]
    pub(crate) fn swizzle_half_relaxed_backend(self, rhs: i8x32) -> i8x32 {
        pick! {
          if #[cfg(target_feature="avx2")] {
            Self(_mm256_shuffle_epi8(self.0, rhs.0))
          } else {
            Self(Inner {
              lo: self.0.lo.shuffle(rhs.0.lo.cast_unsigned()),
              hi: self.0.hi.shuffle(rhs.0.hi.cast_unsigned()),
            })
          }
        }
    }
}
}
