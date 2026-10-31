use crate::backend_imports::*;

#[cfg(not(target_feature = "avx512f"))]
#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub(crate) struct Inner {
    pub lo: i32x8,
    pub hi: i32x8,
}

safe_target_feature! {
impl SimdBackend for i32x16 {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        type Inner = __m512i;
      } else {
        type Inner = Inner;
      }
    }

    #[inline]
    pub(crate) fn simd_lt_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="avx512f")] {
            Self(_mm512_maskz_mov_epi32(
              _mm512_cmp_epi32_mask::<{_MM_CMPINT_LT}>(self.0, rhs.0),
              _mm512_set1_epi32(-1)
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
          if #[cfg(target_feature="avx512f")] {
            Self(_mm512_maskz_mov_epi32(
              _mm512_cmp_epi32_mask::<{_MM_CMPINT_NLE}>(self.0, rhs.0),
              _mm512_set1_epi32(-1)
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
          if #[cfg(target_feature="avx512f")] {
            Self(_mm512_maskz_mov_epi32(
              _mm512_cmp_epi32_mask::<{_MM_CMPINT_LE}>(self.0, rhs.0),
              _mm512_set1_epi32(-1)
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
          if #[cfg(target_feature="avx512f")] {
            Self(_mm512_maskz_mov_epi32(
              _mm512_cmp_epi32_mask::<{_MM_CMPINT_NLT}>(self.0, rhs.0),
              _mm512_set1_epi32(-1)
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
impl i32x16 {
    #[inline]
    pub(crate) fn shr_backend(self, rhs: u32x16) -> Self {
        pick! {
          if #[cfg(target_feature="avx512f")] {
            // Mask `rhs` to 31 to match `wrapping_shr`.
            let rhs = _mm512_and_si512(rhs.0, _mm512_set1_epi32(31));
            Self(_mm512_srav_epi32(self.0, rhs))
          } else {
            Self(Inner {
              lo: self.0.lo >> rhs.0.lo,
              hi: self.0.hi >> rhs.0.hi,
            })
          }
        }
    }

    #[inline]
    pub(crate) fn shr_scalar_backend(self, rhs: u32) -> Self {
        pick! {
          if #[cfg(target_feature="avx512f")] {
            // Use `rhs % 32` to perform wrapping shift and not unbounded shift.
            let shift = rhs & 31;
            Self(_mm512_srav_epi32(self.0, _mm512_set1_epi32(shift as i32)))
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
          if #[cfg(target_feature="avx512f")] {
            Self(_mm512_max_epi32(self.0, rhs.0))
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
          if #[cfg(target_feature="avx512f")] {
            Self(_mm512_min_epi32(self.0, rhs.0))
          } else {
            Self(Inner {
              lo: self.0.lo.min(rhs.0.lo),
              hi: self.0.hi.min(rhs.0.hi),
            })
          }
        }
    }

    #[inline]
    pub(crate) fn reduce_max_backend(self) -> i32 {
        let arr: [i32x8; 2] = cast(self);
        arr[0].max(arr[1]).reduce_max()
    }

    #[inline]
    pub(crate) fn reduce_min_backend(self) -> i32 {
        let arr: [i32x8; 2] = cast(self);
        arr[0].min(arr[1]).reduce_min()
    }

    #[inline]
    pub(crate) fn unbounded_shr_backend(self, rhs: u32x16) -> Self {
        pick! {
          if #[cfg(target_feature="avx512f")] {
            Self(_mm512_srav_epi32(self.0, rhs.0))
          } else {
            Self(Inner {
              lo: self.0.lo.unbounded_shr(rhs.0.lo),
              hi: self.0.hi.unbounded_shr(rhs.0.hi),
            })
          }
        }
    }

    #[inline]
    pub(crate) fn unbounded_shr_scalar_backend(self, rhs: u32) -> Self {
        pick! {
          if #[cfg(target_feature="avx512f")] {
            Self(_mm512_srav_epi32(self.0, _mm512_set1_epi32(rhs as i32)))
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
          if #[cfg(target_feature="avx512f")] {
            let result = self + rhs;
            let overflow = (!(self ^ rhs) & (self ^ result)).is_negative();
            let negative = self.is_negative();

            // If overflow occurs return `MAX` if positive or `MIN` if negative.
            overflow.select(Self::MAX ^ negative, result)
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
          if #[cfg(target_feature="avx512f")] {
            let result = self - rhs;
            let overflow = ((self ^ rhs) & (self ^ result)).is_negative();
            let negative = self.is_negative();

            // If overflow occurs return `MAX` if positive or `MIN` if negative.
            overflow.select(Self::MAX ^ negative, result)
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
        let low = cast::<u32x16, i32x16>(low);

        let overflow = high.simd_ne(low.is_negative());
        (low, overflow)
    }

    #[inline]
    pub(crate) fn mul_keep_low_high_backend(self, rhs: Self) -> (u32x16, i32x16) {
        pick! {
          if #[cfg(all(target_feature="avx512f", target_feature="avx512dq"))] {
            let even_wide_mul = _mm512_mul_epi32(self.0, rhs.0);
            let odd_wide_mul = _mm512_mul_epi32(
              _mm512_shuffle_epi32::<0b_00_11_00_01>(self.0),
              _mm512_shuffle_epi32::<0b_00_11_00_01>(rhs.0),
            );
            let ll_hh_1 = _mm512_unpacklo_epi32(even_wide_mul, odd_wide_mul);
            let ll_hh_2 = _mm512_unpackhi_epi32(even_wide_mul, odd_wide_mul);
            (
              u32x16(_mm512_unpacklo_epi64(ll_hh_1, ll_hh_2)),
              i32x16(_mm512_unpackhi_epi64(ll_hh_1, ll_hh_2)),
            )
          } else {
            let [self_a, self_b] = cast::<i32x16, [i32x8; 2]>(self);
            let [rhs_a, rhs_b] = cast::<i32x16, [i32x8; 2]>(rhs);

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
    pub(crate) fn mul_keep_high_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(all(target_feature="avx512f", target_feature="avx512dq"))] {
            let even_wide_mul = _mm512_mul_epi32(self.0, rhs.0);
            let odd_wide_mul = _mm512_mul_epi32(
              _mm512_shuffle_epi32::<0b_00_11_00_01>(self.0),
              _mm512_shuffle_epi32::<0b_00_11_00_01>(rhs.0),
            );
            let ll_hh_1 = _mm512_unpacklo_epi32(even_wide_mul, odd_wide_mul);
            let ll_hh_2 = _mm512_unpackhi_epi32(even_wide_mul, odd_wide_mul);
            Self(_mm512_unpackhi_epi64(ll_hh_1, ll_hh_2))
          } else {
            let [self_a, self_b] = cast::<i32x16, [i32x8; 2]>(self);
            let [rhs_a, rhs_b] = cast::<i32x16, [i32x8; 2]>(rhs);

            cast([self_a.mul_keep_high(rhs_a), self_b.mul_keep_high(rhs_b)])
          }
        }
    }
}
}

safe_target_feature! {
// Backend of SIMD signed integer functions.
impl i32x16 {
    #[inline]
    pub(crate) fn abs_backend(self) -> Self {
        pick! {
          if #[cfg(target_feature="avx512bw")] {
            Self(_mm512_abs_epi32(self.0))
          } else {
            let [self_lo, self_hi] = cast::<i32x16, [i32x8; 2]>(self);
            cast([self_lo.abs(), self_hi.abs()])
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
          if #[cfg(target_feature = "avx512f")] {
            Self(_mm512_srai_epi32::<31>(self.0))
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
          if #[cfg(target_feature="avx512f")] {
            // Efficient computation of
            // `-self + self.simd_eq(Self::MIN).select(-Self::ONE, Self::ZERO)`
            self.simd_eq(Self::MIN) - self
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
            if #[cfg(target_feature = "avx512f")] {
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
    pub(crate) fn saturating_add_unsigned_backend(self, rhs: u32x16) -> Self {
        pick! {
            if #[cfg(target_feature = "avx512f")] {
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
impl i32x16 {
    #[inline]
    pub(crate) fn from_i16x16_backend(v: i16x16) -> Self {
        pick! {
          if #[cfg(target_feature="avx512f")] {
            Self(_mm512_cvtepi16_epi32(v.0))
          } else {
            let [a, b]: [i16x8; 2] = cast(v);
            Self(Inner {
              lo: i32x8::from_i16x8(a),
              hi: i32x8::from_i16x8(b),
            })
          }
        }
    }

    #[inline]
    pub(crate) fn round_float_backend(self) -> f32x16 {
        pick! {
          if #[cfg(target_feature="avx512f")] {
            cast(_mm512_cvtepi32_ps(self.0))
          } else {
            f32x16(crate::f32x16_backend::Inner {
              lo: self.0.lo.round_float(),
              hi: self.0.hi.round_float(),
            })
          }
        }
    }
}
}
