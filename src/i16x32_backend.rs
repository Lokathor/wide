use crate::backend_imports::*;

#[cfg(not(target_feature = "avx512bw"))]
#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub(crate) struct Inner {
    pub lo: i16x16,
    pub hi: i16x16,
}

safe_target_feature! {
impl SimdBackend for i16x32 {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
        type Inner = __m512i;
      } else {
        type Inner = Inner;
      }
    }

    #[inline]
    pub(crate) fn simd_lt_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="avx512bw")] {
            Self(_mm512_maskz_mov_epi16(
              _mm512_cmp_epi16_mask::<{_MM_CMPINT_LT}>(self.0, rhs.0),
              _mm512_set1_epi16(-1),
            ))
          } else {
            Self(Inner {
              lo: rhs.0.lo.simd_gt(self.0.lo),
              hi: rhs.0.hi.simd_gt(self.0.hi),
            })
          }
        }
    }

    #[inline]
    pub(crate) fn simd_gt_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="avx512bw")] {
            Self(_mm512_maskz_mov_epi16(_mm512_cmp_epi16_mask::<{_MM_CMPINT_NLE}>(self.0, rhs.0),
              _mm512_set1_epi16(-1),
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
            Self(_mm512_maskz_mov_epi16(_mm512_cmp_epi16_mask::<{_MM_CMPINT_LE}>(self.0, rhs.0),
              _mm512_set1_epi16(-1),
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
            Self(_mm512_maskz_mov_epi16(_mm512_cmp_epi16_mask::<{_MM_CMPINT_NLT}>(self.0, rhs.0),
              _mm512_set1_epi16(-1),
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
impl i16x32 {
    #[inline]
    pub(crate) fn shr_backend(self, rhs: u16x32) -> Self {
        pick! {
          if #[cfg(target_feature="avx512bw")] {
            // Mask `rhs` to 15 to match `wrapping_shr`.
            let rhs = _mm512_and_si512(rhs.0, _mm512_set1_epi16(15));
            Self(_mm512_srav_epi16(self.0, rhs))
          } else {
            let [self_a, self_b]: [i16x16; 2] = cast(self);
            let [rhs_a, rhs_b]: [u16x16; 2] = cast(rhs);

            cast([self_a >> rhs_a, self_b >> rhs_b])
          }
        }
    }

    #[inline]
    pub(crate) fn shr_scalar_backend(self, rhs: u32) -> Self {
        pick! {
          if #[cfg(target_feature="avx512bw")] {
            // Use `rhs % 16` to perform wrapping shift and not unbounded shift.
            let shift = rhs as u16 & 15;
            Self(_mm512_srav_epi16(self.0, _mm512_set1_epi16(shift as i16)))
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
          if #[cfg(target_feature="avx512bw")] {
            Self(_mm512_max_epi16(self.0, rhs.0))
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
            Self(_mm512_min_epi16(self.0, rhs.0))
          } else {
            Self(Inner {
              lo: self.0.lo.min(rhs.0.lo),
              hi: self.0.hi.min(rhs.0.hi),
            })
          }
        }
    }

    #[inline]
    pub(crate) fn reduce_max_backend(self) -> i16 {
        let arr: [i16x16; 2] = cast(self);
        arr[0].max(arr[1]).reduce_max()
    }

    #[inline]
    pub(crate) fn reduce_min_backend(self) -> i16 {
        let arr: [i16x16; 2] = cast(self);
        arr[0].min(arr[1]).reduce_min()
    }

    #[inline]
    pub(crate) fn unbounded_shr_backend(self, rhs: u16x32) -> Self {
        pick! {
          if #[cfg(target_feature="avx512bw")] {
            Self(_mm512_srav_epi16(self.0, rhs.0))
          } else {
            let [self_a, self_b] = cast::<i16x32, [i16x16; 2]>(self);
            let [rhs_a, rhs_b] = cast::<u16x32, [u16x16; 2]>(rhs);

            cast([self_a.unbounded_shr(rhs_a), self_b.unbounded_shr(rhs_b)])
          }
        }
    }

    #[inline]
    pub(crate) fn unbounded_shr_scalar_backend(self, rhs: u32) -> Self {
        pick! {
          if #[cfg(target_feature="avx512bw")] {
            // `u32 as u16` truncates the higher half so we need to manually
            // saturate.
            Self(_mm512_srav_epi16(self.0, _mm512_set1_epi16(rhs.min(u16::MAX as u32) as u16 as i16)))
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
          if #[cfg(target_feature="avx512bw")] {
            Self(_mm512_adds_epi16(self.0, rhs.0))
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
            Self(_mm512_subs_epi16(self.0, rhs.0))
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
        let low = cast::<u16x32, i16x32>(low);

        let overflow = high.simd_ne(low.is_negative());
        (low, overflow)
    }

    #[inline]
    pub(crate) fn mul_keep_low_high_backend(self, rhs: Self) -> (u16x32, i16x32) {
        // x86 has no `_mm512_mul_epi16` intrinsic so there is no `avx512`
        // optimization.

        let [self_a, self_b] = cast::<i16x32, [i16x16; 2]>(self);
        let [rhs_a, rhs_b] = cast::<i16x32, [i16x16; 2]>(rhs);

        let result_a = self_a.mul_keep_low_high(rhs_a);
        let result_b = self_b.mul_keep_low_high(rhs_b);
        (
            cast([result_a.0, result_b.0]),
            cast([result_a.1, result_b.1]),
        )
    }

    #[inline]
    pub(crate) fn mul_keep_high_backend(self, rhs: Self) -> Self {
        // x86 has no `_mm512_mul_epi16` intrinsic so there is no `avx512`
        // optimization.

        let [self_a, self_b] = cast::<i16x32, [i16x16; 2]>(self);
        let [rhs_a, rhs_b] = cast::<i16x32, [i16x16; 2]>(rhs);

        cast([self_a.mul_keep_high(rhs_a), self_b.mul_keep_high(rhs_b)])
    }
}
}

safe_target_feature! {
// Backend of SIMD signed integer functions.
impl i16x32 {
    #[inline]
    pub(crate) fn abs_backend(self) -> Self {
        pick! {
          if #[cfg(target_feature="avx512bw")] {
            Self(_mm512_abs_epi16(self.0))
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
          if #[cfg(target_feature = "avx512bw")] {
            Self(_mm512_srai_epi16::<15>(self.0))
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
                Self(_mm512_subs_epi16(_mm512_setzero_si512(), self.0))
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
    pub(crate) fn saturating_add_unsigned_backend(self, rhs: u16x32) -> Self {
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

safe_target_feature! {
// Internal helpers and backend of inconsistent functions.
impl i16x32 {
    #[inline]
    pub(crate) fn from_i8x32_backend(v: i8x32) -> Self {
        pick! {
          if #[cfg(target_feature="avx512bw")] {
            Self(_mm512_cvtepi8_epi16(cast(v)))
          } else {
            let [a, b]: [i8x16; 2] = cast(v);
            Self(Inner {
              lo: i16x16::from_i8x16(a),
              hi: i16x16::from_i8x16(b),
            })
          }
        }
    }

    #[inline]
    pub(crate) fn dot_backend(self, rhs: Self) -> i32x16 {
        pick! {
          if #[cfg(target_feature="avx512bw")] {
            i32x16(_mm512_madd_epi16(self.0, rhs.0))
          } else {
            let [self_lo, self_hi] = cast::<i16x32, [i16x16; 2]>(self);
            let [rhs_lo, rhs_hi] = cast::<i16x32, [i16x16; 2]>(rhs);
            cast([self_lo.dot(rhs_lo), self_hi.dot(rhs_hi)])
          }
        }
    }
}
}
