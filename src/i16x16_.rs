use crate::imports::*;

pick! {
  if #[cfg(target_feature="avx2")] {
    pub(crate) type Inner = m256i;
  } else {
    #[repr(C, align(32))]
    #[derive(Clone, Copy)]
    pub(crate) struct Inner {
      pub lo: i16x8,
      pub hi: i16x8,
    }
  }
}

impl i16x16 {
  #[inline]
  pub(crate) fn simd_lt_backend(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self(!cmp_gt_mask_i16_m256i(self.0, rhs.0) ^ cmp_eq_mask_i16_m256i(self.0,rhs.0))
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
        Self(cmp_gt_mask_i16_m256i(self.0, rhs.0))
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

  #[inline]
  pub(crate) fn shr_backend(self, rhs: u16x16) -> Self {
    pick! {
      if #[cfg(all(target_feature="avx512bw", target_feature="avx512vl"))] {
        #[cfg(target_arch = "x86")]
        use core::arch::x86::_mm256_srav_epi16;
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::_mm256_srav_epi16;

        // Mask `rhs` to 15 to match `wrapping_shr`.
        let rhs = bitand_m256i(rhs.0, set_splat_i16_m256i(15));
        // TODO(safe_arch): Add `_mm256_srav_epi16`.
        cast(unsafe { _mm256_srav_epi16(self.0.0, rhs.0) })
      } else {
        let [self_a, self_b]: [i16x8; 2] = cast(self);
        let [rhs_a, rhs_b]: [u16x8; 2] = cast(rhs);

        cast([self_a >> rhs_a, self_b >> rhs_b])
      }
    }
  }

  #[inline]
  pub(crate) fn shr_backend(self, rhs: u32) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        // Use `rhs % 16` to perform wrapping shift and not unbounded shift.
        #[expect(clippy::suspicious_arithmetic_impl)]
        let shift = cast([rhs as u64 & 15, 0]);
        Self(shr_all_i16_m256i(self.0, shift))
      } else {
        Self(Inner {
          lo: self.0.lo.shr(rhs),
          hi: self.0.hi.shr(rhs),
        })
      }
    }
  }

  #[inline]
  pub(crate) fn max_backend(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self(max_i16_m256i(self.0, rhs.0))
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
        Self(min_i16_m256i(self.0, rhs.0))
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
    let arr: [i16x8; 2] = cast(self);

    arr[0].max(arr[1]).reduce_max()
  }

  #[inline]
  pub(crate) fn reduce_min_backend(self) -> i16 {
    let arr: [i16x8; 2] = cast(self);

    arr[0].min(arr[1]).reduce_min()
  }

  #[inline]
  pub(crate) fn unbounded_shr_backend(self, rhs: u16x16) -> Self {
    pick! {
      if #[cfg(all(target_feature="avx512bw", target_feature="avx512vl"))] {
        #[cfg(target_arch = "x86")]
        use core::arch::x86::_mm256_srav_epi16;
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::_mm256_srav_epi16;

        // TODO(safe_arch): Add `_mm256_srav_epi16`.
        cast(unsafe { _mm256_srav_epi16(self.0.0, rhs.0.0) })
      } else {
        let [self_a, self_b] = cast::<i16x16, [i16x8; 2]>(self);
        let [rhs_a, rhs_b] = cast::<u16x16, [u16x8; 2]>(rhs);

        cast([self_a.unbounded_shr(rhs_a), self_b.unbounded_shr(rhs_b)])
      }
    }
  }

  #[inline]
  pub(crate) fn unbounded_shr_scalar_backend(self, rhs: u32) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self(shr_all_i16_m256i(self.0, cast([rhs as u64, 0])))
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
        Self(add_saturating_i16_m256i(self.0, rhs.0))
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
        Self(sub_saturating_i16_m256i(self.0, rhs.0))
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
    let low = cast::<u16x16, i16x16>(low);

    let overflow = high.simd_ne(low.is_negative());
    (low, overflow)
  }

  optional_fn_widening_mul {
    #[inline]
    pub(crate) fn widening_mul_backend(self, rhs: Self) -> i32x16 {
      // x86 has no `_mm256_mul_epi16` intrinsic so there is no `avx2`
      // optimization.

      let [self_a, self_b] = cast::<i16x16, [i16x8; 2]>(self);
      let [rhs_a, rhs_b] = cast::<i16x16, [i16x8; 2]>(rhs);

      cast([self_a.widening_mul(rhs_a), self_b.widening_mul(rhs_b)])
    }
  }

  #[inline]
  pub(crate) fn mul_keep_low_high_backend(self, rhs: Self) -> (u16x16, i16x16) {
    // x86 has no `_mm256_mul_epi16` intrinsic so there is no `avx2`
    // optimization.

    let [self_a, self_b] = cast::<i16x16, [i16x8; 2]>(self);
    let [rhs_a, rhs_b] = cast::<i16x16, [i16x8; 2]>(rhs);

    let result_a = self_a.mul_keep_low_high(rhs_a);
    let result_b = self_b.mul_keep_low_high(rhs_b);
    (cast([result_a.0, result_b.0]), cast([result_a.1, result_b.1]))
  }

  #[inline]
  pub(crate) fn mul_keep_high_backend(self, rhs: Self) -> Self {
    // x86 has no `_mm256_mul_epi16` intrinsic so there is no `avx2`
    // optimization.

    let [self_a, self_b] = cast::<i16x16, [i16x8; 2]>(self);
    let [rhs_a, rhs_b] = cast::<i16x16, [i16x8; 2]>(rhs);

    cast([self_a.mul_keep_high(rhs_a), self_b.mul_keep_high(rhs_b)])
  }

  #[inline]
  pub(crate) fn abs_backend(self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self(abs_i16_m256i(self.0))
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
}

impl From<i8x16> for i16x16 {
    /// widen with sign extend from i8 to i16
    #[inline]
    fn from(i: i8x16) -> Self {
        i16x16::from_i8x16(i)
    }
}

impl From<u8x16> for i16x16 {
    /// widen with zero extend from u8 to i16
    #[inline]
    fn from(i: u8x16) -> Self {
        cast(u16x16::from(i))
    }
}

/// The following functionality exists only for [`i16x16`], or only for
/// particular types inconsistently.
impl i16x16 {
    /// Converts each element from [`i8`] to [`i16`].
    #[inline]
    #[must_use]
    pub fn from_i8x16(v: i8x16) -> Self {
        pick! {
          if #[cfg(target_feature="avx2")] {
            i16x16(convert_to_i16_m256i_from_i8_m128i(v.0))
          } else if #[cfg(target_feature="sse4.1")] {
            Self(Inner {
              lo: i16x8(convert_to_i16_m128i_from_lower8_i8_m128i(v.0)),
              hi: i16x8(convert_to_i16_m128i_from_lower8_i8_m128i(unpack_high_i64_m128i(v.0, v.0)))
            })
          } else if #[cfg(target_feature="sse2")] {
            Self(Inner {
              lo: i16x8(shr_imm_i16_m128i::<8>( unpack_low_i8_m128i(v.0, v.0))),
              hi: i16x8(shr_imm_i16_m128i::<8>( unpack_high_i8_m128i(v.0, v.0))),
            })
          } else {

            i16x16::from_array([
              v.as_array()[0] as i16,
              v.as_array()[1] as i16,
              v.as_array()[2] as i16,
              v.as_array()[3] as i16,
              v.as_array()[4] as i16,
              v.as_array()[5] as i16,
              v.as_array()[6] as i16,
              v.as_array()[7] as i16,
              v.as_array()[8] as i16,
              v.as_array()[9] as i16,
              v.as_array()[10] as i16,
              v.as_array()[11] as i16,
              v.as_array()[12] as i16,
              v.as_array()[13] as i16,
              v.as_array()[14] as i16,
              v.as_array()[15] as i16,
              ])
          }
        }
    }

    /// Partially computes the dot product.
    ///
    /// First this multiplies the input 16-bit integers, producing intermediate
    /// 32-bit integers. Then this horizontally adds adjacent pairs, resulting in
    /// eight 32-bit integers.
    #[inline]
    #[must_use]
    pub fn dot(self, rhs: Self) -> i32x8 {
        pick! {
          if #[cfg(target_feature="avx2")] {
            i32x8( mul_i16_horizontal_add_m256i(self.0, rhs.0))
          } else {
            i32x8(crate::i32x8_::Inner {
              lo: self.0.lo.dot(rhs.0.lo),
              hi: self.0.hi.dot(rhs.0.hi),
            })
          }
        }
    }

    /// Multiply and scale equivalent to `((self * rhs) + 0x4000) >> 15` on each
    /// lane, effectively multiplying by a 16 bit fixed point number between `-1`
    /// and `1`. This corresponds to the following instructions:
    /// - `vqrdmulhq_n_s16` instruction on neon
    /// - `i16x8_q15mulr_sat` on simd128
    /// - `_mm256_mulhrs_epi16` on avx2
    /// - emulated via `mul_i16_*` on sse2
    #[inline]
    #[must_use]
    pub fn mul_scale_round(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="avx2")] {
            Self(mul_i16_scale_round_m256i(self.0, rhs.0))
          } else {
            Self(Inner {
              lo: self.0.lo.mul_scale_round(rhs.0.lo),
              hi: self.0.hi.mul_scale_round(rhs.0.hi),
            })
          }
        }
    }

    /// Multiply and scale equivalent to `((self * rhs) + 0x4000) >> 15` on each
    /// lane, effectively multiplying by a 16 bit fixed point number between `-1`
    /// and `1`. This corresponds to the following instructions:
    /// - `vqrdmulhq_n_s16` instruction on neon
    /// - `i16x8_q15mulr_sat` on simd128
    /// - `_mm256_mulhrs_epi16` on avx2
    /// - emulated via `mul_i16_*` on sse2
    #[inline]
    #[must_use]
    pub fn mul_scale_round_n(self, rhs: i16) -> Self {
        pick! {
          if #[cfg(target_feature="avx2")] {
            Self(mul_i16_scale_round_m256i(self.0, set_splat_i16_m256i(rhs)))
          } else {
            Self(Inner {
              lo: self.0.lo.mul_scale_round_n(rhs),
              hi: self.0.hi.mul_scale_round_n(rhs),
            })
          }
        }
    }
}
