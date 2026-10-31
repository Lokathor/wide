use crate::backend_imports::*;

#[cfg(not(target_feature = "avx2"))]
#[repr(C, align(32))]
#[derive(Clone, Copy)]
pub(crate) struct Inner {
    pub lo: i32x4,
    pub hi: i32x4,
}

safe_target_feature! {
impl SimdBackend for i32x8 {
    pick! {
      if #[cfg(target_feature="avx2")] {
        type Inner = __m256i;
      } else {
        type Inner = Inner;
      }
    }

    #[inline]
    pub(crate) fn simd_lt_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="avx2")] {
            Self(_mm256_cmpgt_epi32(rhs.0, self.0))
          } else {
            Self(Inner {
              lo: self.0.lo.simd_lt(rhs.0.lo),
              hi: self.0.hi.simd_lt(rhs.0.hi),
            })
          }
        }
    }

    #[inline]
    pub(crate) fn simd_gt_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="avx2")] {
            Self(_mm256_cmpgt_epi32(self.0, rhs.0))
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
impl i32x8 {
    #[inline]
    pub(crate) fn shr_backend(self, rhs: u32x8) -> Self {
        pick! {
          if #[cfg(target_feature="avx2")] {
            // ensure same behavior as scalar
            let shift_by = _mm256_and_si256(rhs.0, _mm256_set1_epi32(31));
            Self(_mm256_srav_epi32(self.0, shift_by ))
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
          if #[cfg(target_feature="avx2")] {
            // Use `rhs % 32` to perform wrapping shift and not unbounded shift.
            let shift = cast([rhs as u64 & 31, 0]);
            Self(_mm256_sra_epi32(self.0, shift))
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
          if #[cfg(target_feature="avx2")] {
            Self(_mm256_max_epi32(self.0, rhs.0))
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
            Self(_mm256_min_epi32(self.0, rhs.0))
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
        let arr: [i32x4; 2] = cast(self);
        arr[0].max(arr[1]).reduce_max()
    }

    #[inline]
    pub(crate) fn reduce_min_backend(self) -> i32 {
        let arr: [i32x4; 2] = cast(self);
        arr[0].min(arr[1]).reduce_min()
    }

    #[inline]
    pub(crate) fn unbounded_shr_backend(self, rhs: u32x8) -> Self {
        pick! {
          if #[cfg(target_feature="avx2")] {
            Self(_mm256_srav_epi32(self.0, rhs.0))
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
          if #[cfg(target_feature="avx2")] {
            Self(_mm256_sra_epi32(self.0, cast([rhs as u64, 0])))
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
          if #[cfg(target_feature="avx2")] {
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
          if #[cfg(target_feature="avx2")] {
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
        let low = cast::<u32x8, i32x8>(low);

        let overflow = high.simd_ne(low.is_negative());
        (low, overflow)
    }

    #[inline]
    pub(crate) fn widening_mul_backend(self, rhs: Self) -> i64x8 {
        pick! {
          if #[cfg(all(target_feature="avx512f", target_feature="avx2"))] {
            let shuffle_indices = const { i64x8::from_array([0, 4, 1, 5, 2, 6, 3, 7]).0 };

            let even_wide_mul = _mm256_mul_epi32(self.0, rhs.0);
            let odd_wide_mul = _mm256_mul_epi32(
              _mm256_shuffle_epi32::<0b_00_11_00_01>(self.0),
              _mm256_shuffle_epi32::<0b_00_11_00_01>(rhs.0),
            );
            let even_then_odd = cast::<[__m256i; 2], __m512i>([even_wide_mul, odd_wide_mul]);
            i64x8(_mm512_permutexvar_epi64(shuffle_indices, even_then_odd))
          } else if #[cfg(target_feature="avx2")] {
            let even_wide_mul = _mm256_mul_epi32(self.0, rhs.0);
            let odd_wide_mul = _mm256_mul_epi32(
              _mm256_shuffle_epi32::<0b_00_11_00_01>(self.0),
              _mm256_shuffle_epi32::<0b_00_11_00_01>(rhs.0),
            );
            let m0145 = _mm256_unpacklo_epi64(even_wide_mul, odd_wide_mul);
            let m2367 = _mm256_unpackhi_epi64(even_wide_mul, odd_wide_mul);

            cast([
              _mm256_permute2x128_si256::<0b_0010_0000>(m0145, m2367),
              _mm256_permute2x128_si256::<0b_0011_0001>(m0145, m2367),
            ])
          } else {
            let [self_a, self_b] = cast::<i32x8, [i32x4; 2]>(self);
            let [rhs_a, rhs_b] = cast::<i32x8, [i32x4; 2]>(rhs);

            cast([self_a.widening_mul(rhs_a), self_b.widening_mul(rhs_b)])
          }
        }
    }

    #[inline]
    pub(crate) fn mul_keep_low_high_backend(self, rhs: Self) -> (u32x8, i32x8) {
        pick! {
          if #[cfg(target_feature="avx2")] {
            let even_wide_mul = _mm256_mul_epi32(self.0, rhs.0);
            let odd_wide_mul = _mm256_mul_epi32(
              _mm256_shuffle_epi32::<0b_00_11_00_01>(self.0),
              _mm256_shuffle_epi32::<0b_00_11_00_01>(rhs.0),
            );
            let ll_hh_1 = _mm256_unpacklo_epi32(even_wide_mul, odd_wide_mul);
            let ll_hh_2 = _mm256_unpackhi_epi32(even_wide_mul, odd_wide_mul);

            (
              u32x8(_mm256_unpacklo_epi64(ll_hh_1, ll_hh_2)),
              i32x8(_mm256_unpackhi_epi64(ll_hh_1, ll_hh_2)),
            )
          } else {
            let [self_a, self_b] = cast::<i32x8, [i32x4; 2]>(self);
            let [rhs_a, rhs_b] = cast::<i32x8, [i32x4; 2]>(rhs);

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
          if #[cfg(target_feature="avx2")] {
            let even_wide_mul = _mm256_mul_epi32(self.0, rhs.0);
            let odd_wide_mul = _mm256_mul_epi32(
              _mm256_shuffle_epi32::<0b_00_11_00_01>(self.0),
              _mm256_shuffle_epi32::<0b_00_11_00_01>(rhs.0),
            );
            let ll_hh_1 = _mm256_unpacklo_epi32(even_wide_mul, odd_wide_mul);
            let ll_hh_2 = _mm256_unpackhi_epi32(even_wide_mul, odd_wide_mul);

            Self(_mm256_unpackhi_epi64(ll_hh_1, ll_hh_2))
          } else {
            let [self_a, self_b] = cast::<i32x8, [i32x4; 2]>(self);
            let [rhs_a, rhs_b] = cast::<i32x8, [i32x4; 2]>(rhs);

            cast([self_a.mul_keep_high(rhs_a), self_b.mul_keep_high(rhs_b)])
          }
        }
    }
}
}

safe_target_feature! {
// Backend of SIMD signed integer functions.
impl i32x8 {
    #[inline]
    pub(crate) fn abs_backend(self) -> Self {
        pick! {
          if #[cfg(target_feature="avx2")] {
            Self(_mm256_abs_epi32(self.0))
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
          if #[cfg(target_feature = "avx2")] {
            Self(_mm256_srai_epi32::<31>(self.0))
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
          if #[cfg(target_feature="avx2")] {
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
    pub(crate) fn saturating_add_unsigned_backend(self, rhs: u32x8) -> Self {
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
impl i32x8 {
    #[inline]
    pub(crate) fn from_i16x8_backend(v: i16x8) -> Self {
        pick! {
          if #[cfg(target_feature="avx2")] {
            i32x8(_mm256_cvtepi16_epi32(v.0))
          } else if #[cfg(target_feature="sse2")] {
            Self(Inner {
              lo: i32x4(_mm_srai_epi32::<16>( _mm_unpacklo_epi16(v.0, v.0))),
              hi: i32x4(_mm_srai_epi32::<16>( _mm_unpackhi_epi16(v.0, v.0))),
            })
          } else {
            i32x8::from_array([
              i32::from(v.as_array()[0]),
              i32::from(v.as_array()[1]),
              i32::from(v.as_array()[2]),
              i32::from(v.as_array()[3]),
              i32::from(v.as_array()[4]),
              i32::from(v.as_array()[5]),
              i32::from(v.as_array()[6]),
              i32::from(v.as_array()[7]),
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn from_u16x8_backend(v: u16x8) -> Self {
        pick! {
          if #[cfg(target_feature="avx2")] {
            i32x8(_mm256_cvtepu16_epi32(v.0))
          } else if #[cfg(target_feature="sse2")] {
            Self(Inner {
              lo: i32x4(_mm_srli_epi32::<16>( _mm_unpacklo_epi16(v.0, v.0))),
              hi: i32x4(_mm_srli_epi32::<16>( _mm_unpackhi_epi16(v.0, v.0))),
            })
          } else {
            i32x8::from_array([
              i32::from(v.as_array()[0]),
              i32::from(v.as_array()[1]),
              i32::from(v.as_array()[2]),
              i32::from(v.as_array()[3]),
              i32::from(v.as_array()[4]),
              i32::from(v.as_array()[5]),
              i32::from(v.as_array()[6]),
              i32::from(v.as_array()[7]),
            ])
          }
        }
    }

    #[inline]
    pub(crate) fn round_float_backend(self) -> f32x8 {
        pick! {
          if #[cfg(target_feature="avx2")] {
            cast(_mm256_cvtepi32_ps(self.0))
          } else {
            cast([
              self.0.lo.round_float(),
              self.0.hi.round_float(),
            ])
          }
        }
    }
}
}
