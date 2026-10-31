use crate::backend_imports::*;

#[cfg(not(target_feature = "avx2"))]
#[repr(C, align(32))]
#[derive(Clone, Copy)]
pub(crate) struct Inner {
    pub lo: i64x2,
    pub hi: i64x2,
}

safe_target_feature! {
impl SimdBackend for i64x4 {
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
            !Self(_mm256_xor_si256(_mm256_cmpgt_epi64(self.0, rhs.0), _mm256_cmpeq_epi64(self.0, rhs.0)))
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
            Self(_mm256_cmpgt_epi64(self.0, rhs.0))
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
impl i64x4 {
    #[inline]
    pub(crate) fn shr_backend(self, rhs: u64x4) -> Self {
        pick! {
          if #[cfg(target_feature="avx2")] {
            let arr: [i64; 4] = cast(self);
            let rhs: [u64; 4] = cast(rhs);
            cast([
              arr[0].wrapping_shr(rhs[0] as u32),
              arr[1].wrapping_shr(rhs[1] as u32),
              arr[2].wrapping_shr(rhs[2] as u32),
              arr[3].wrapping_shr(rhs[3] as u32),
            ])
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
        // there is no signed right shift in AVX2
        let [a, b]: [i64x2; 2] = cast(self);
        cast([a >> rhs, b >> rhs])
    }

    #[inline]
    pub(crate) fn max_backend(self, rhs: Self) -> Self {
        self.simd_gt(rhs).select(self, rhs)
    }

    #[inline]
    pub(crate) fn min_backend(self, rhs: Self) -> Self {
        self.simd_lt(rhs).select(self, rhs)
    }

    #[inline]
    pub(crate) fn reduce_max_backend(self) -> i64 {
        let array: [i64; 4] = cast(self);
        array[0].max(array[1]).max(array[2]).max(array[3])
    }

    #[inline]
    pub(crate) fn reduce_min_backend(self) -> i64 {
        let array: [i64; 4] = cast(self);
        array[0].min(array[1]).min(array[2]).min(array[3])
    }

    #[inline]
    pub(crate) fn unbounded_shr_backend(self, rhs: u64x4) -> Self {
        let [self_a, self_b] = cast::<i64x4, [i64x2; 2]>(self);
        let [rhs_a, rhs_b] = cast::<u64x4, [u64x2; 2]>(rhs);

        cast([self_a.unbounded_shr(rhs_a), self_b.unbounded_shr(rhs_b)])
    }

    #[inline]
    pub(crate) fn unbounded_shr_scalar_backend(self, rhs: u32) -> Self {
        // there is no signed right shift in AVX2
        let [self_a, self_b] = cast::<i64x4, [i64x2; 2]>(self);
        cast([
            self_a.unbounded_shr_scalar(rhs),
            self_b.unbounded_shr_scalar(rhs),
        ])
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
        // TODO(perf): This implementation looks quite bad. Is there a better
        // one? This intentionally avoids `mul_keep_low_high` because getting the
        // high bits of 64-bit multiplication could be slow.

        let self_array = self.to_array();
        let rhs_array = rhs.to_array();

        let result = [
            self_array[0].overflowing_mul(rhs_array[0]),
            self_array[1].overflowing_mul(rhs_array[1]),
            self_array[2].overflowing_mul(rhs_array[2]),
            self_array[3].overflowing_mul(rhs_array[3]),
        ];
        (
            Self::from_array([result[0].0, result[1].0, result[2].0, result[3].0]),
            Self::from_array([
                -(result[0].1 as i64),
                -(result[1].1 as i64),
                -(result[2].1 as i64),
                -(result[3].1 as i64),
            ]),
        )
    }

    #[inline]
    pub(crate) fn mul_keep_low_high_backend(self, rhs: Self) -> (u64x4, i64x4) {
        // TODO(perf): This implementation looks quite bad. Is there a better
        // one?

        let self_array = self.to_array();
        let rhs_array = rhs.to_array();

        let widening_mul = [
            (self_array[0] as i128).wrapping_mul(rhs_array[0] as i128),
            (self_array[1] as i128).wrapping_mul(rhs_array[1] as i128),
            (self_array[2] as i128).wrapping_mul(rhs_array[2] as i128),
            (self_array[3] as i128).wrapping_mul(rhs_array[3] as i128),
        ];

        (
            u64x4::from_array([
                widening_mul[0] as u64,
                widening_mul[1] as u64,
                widening_mul[2] as u64,
                widening_mul[3] as u64,
            ]),
            i64x4::from_array([
                (widening_mul[0] >> 64) as i64,
                (widening_mul[1] >> 64) as i64,
                (widening_mul[2] >> 64) as i64,
                (widening_mul[3] >> 64) as i64,
            ]),
        )
    }

    #[inline]
    pub(crate) fn mul_keep_high_backend(self, rhs: Self) -> Self {
        let self_array = self.to_array();
        let rhs_array = rhs.to_array();

        Self::from_array([
            ((self_array[0] as i128).wrapping_mul(rhs_array[0] as i128) >> 64) as i64,
            ((self_array[1] as i128).wrapping_mul(rhs_array[1] as i128) >> 64) as i64,
            ((self_array[2] as i128).wrapping_mul(rhs_array[2] as i128) >> 64) as i64,
            ((self_array[3] as i128).wrapping_mul(rhs_array[3] as i128) >> 64) as i64,
        ])
    }
}
}

safe_target_feature! {
// Backend of SIMD signed integer functions.
impl i64x4 {
    #[inline]
    pub(crate) fn abs_backend(self) -> Self {
        pick! {
          if #[cfg(target_feature="avx2")] {
            // avx x86 doesn't have this builtin
            let arr: [i64; 4] = cast(self);
            cast(
              [
                arr[0].wrapping_abs(),
                arr[1].wrapping_abs(),
                arr[2].wrapping_abs(),
                arr[3].wrapping_abs(),
              ])
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
          if #[cfg(all(target_feature = "avx512f", target_feature = "avx512vl"))] {
            Self(_mm256_srai_epi64::<63>(self.0))
          } else if #[cfg(target_feature = "avx2")] {
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
    pub(crate) fn saturating_add_unsigned_backend(self, rhs: u64x4) -> Self {
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
impl i64x4 {
    #[inline]
    pub(crate) fn round_float_backend(self) -> f64x4 {
        let arr: [i64; 4] = cast(self);
        cast([arr[0] as f64, arr[1] as f64, arr[2] as f64, arr[3] as f64])
    }
}
}
