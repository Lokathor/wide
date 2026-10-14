use crate::imports::*;

pick! {
  if #[cfg(target_feature="avx512f")] {
    pub(crate) type Inner = m512i;
  } else {
    #[repr(C, align(64))]
    #[derive(Clone, Copy)]
    pub(crate) struct Inner {
      pub lo: i32x8,
      pub hi: i32x8,
    }
  }
}

impl i32x16 {
  #[inline]
  pub(crate) fn simd_lt_backend(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self(cmp_op_mask_i32_m512i::<{cmp_int_op!(Lt)}>(self.0, rhs.0))
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
        Self(cmp_op_mask_i32_m512i::<{cmp_int_op!(Nle)}>(self.0, rhs.0))
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
        Self(cmp_op_mask_i32_m512i::<{cmp_int_op!(Le)}>(self.0, rhs.0))
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
        Self(cmp_op_mask_i32_m512i::<{cmp_int_op!(Nlt)}>(self.0, rhs.0))
      } else {
        Self(Inner {
          lo: self.0.lo.simd_ge(rhs.0.lo),
          hi: self.0.hi.simd_ge(rhs.0.hi),
        })
      }
    }
  }

  #[inline]
  pub(crate) fn shr_backend(self, rhs: u32x16) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        #[cfg(target_arch = "x86")]
        use core::arch::x86::_mm512_srav_epi32;
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::_mm512_srav_epi32;

        // Mask `rhs` to 31 to match `wrapping_shr`.
        let rhs = bitand_m512i(rhs.0, set_splat_i32_m512i(31));
        // TODO(safe_arch): Add `_mm512_srav_epi32`.
        Self(m512i(unsafe { _mm512_srav_epi32(self.0.0, rhs.0) }))
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
        #[expect(clippy::suspicious_arithmetic_impl)]
        let shift = rhs & 31;
        Self(shr_all_i32_m512i(self.0, shift))
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
      if #[cfg(target_feature="avx512f")] {
        Self(max_i32_m512i(self.0, rhs.0))
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
        Self(min_i32_m512i(self.0, rhs.0))
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
        #[cfg(target_arch = "x86")]
        use core::arch::x86::_mm512_srav_epi32;
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::_mm512_srav_epi32;

        // TODO(safe_arch): Add `_mm512_srav_epi32`.
        Self(m512i(unsafe { _mm512_srav_epi32(self.0.0, rhs.0.0) }))
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
        Self(shr_all_i32_m512i(self.0, rhs))
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

  optional_fn_widening_mul {
    // Cannot have `widening_mul` because there is no `i64x16` type.
  }

  #[inline]
  pub(crate) fn mul_keep_low_high_backend(self, rhs: Self) -> (u32x16, i32x16) {
    pick! {
      if #[cfg(all(target_feature="avx512f", target_feature="avx512dq"))] {
        #[cfg(target_arch = "x86")]
        use core::arch::x86::{_mm512_unpackhi_epi64, _mm512_unpacklo_epi64};
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::{_mm512_unpackhi_epi64, _mm512_unpacklo_epi64};

        let even_wide_mul = mul_i32_wide_m512i(self.0, rhs.0);
        let odd_wide_mul = mul_i32_wide_m512i(
          shuffle_i32_m512i::<0b_00_11_00_01>(self.0),
          shuffle_i32_m512i::<0b_00_11_00_01>(rhs.0),
        );
        let ll_hh_1 = unpack_low_i32_m512i(even_wide_mul, odd_wide_mul);
        let ll_hh_2 = unpack_high_i32_m512i(even_wide_mul, odd_wide_mul);
        // TODO(safe_arch): Add `_mm512_unpacklo_epi64` and `_mm512_unpackhi_epi64`.
        (
          u32x16(m512i(unsafe { _mm512_unpacklo_epi64(ll_hh_1.0, ll_hh_2.0) })),
          i32x16(m512i(unsafe { _mm512_unpackhi_epi64(ll_hh_1.0, ll_hh_2.0) })),
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
        #[cfg(target_arch = "x86")]
        use core::arch::x86::_mm512_unpackhi_epi64;
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::_mm512_unpackhi_epi64;

        let even_wide_mul = mul_i32_wide_m512i(self.0, rhs.0);
        let odd_wide_mul = mul_i32_wide_m512i(
          shuffle_i32_m512i::<0b_00_11_00_01>(self.0),
          shuffle_i32_m512i::<0b_00_11_00_01>(rhs.0),
        );
        let ll_hh_1 = unpack_low_i32_m512i(even_wide_mul, odd_wide_mul);
        let ll_hh_2 = unpack_high_i32_m512i(even_wide_mul, odd_wide_mul);
        // TODO(safe_arch): Add `_mm512_unpackhi_epi64`.
        Self(m512i(unsafe { _mm512_unpackhi_epi64(ll_hh_1.0, ll_hh_2.0) }))
      } else {
        let [self_a, self_b] = cast::<i32x16, [i32x8; 2]>(self);
        let [rhs_a, rhs_b] = cast::<i32x16, [i32x8; 2]>(rhs);

        cast([self_a.mul_keep_high(rhs_a), self_b.mul_keep_high(rhs_b)])
      }
    }
  }

  #[inline]
  pub(crate) fn abs_backend(self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
        Self(abs_i32_m512i(self.0))
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

impl From<i16x16> for i32x16 {
    #[inline]
    fn from(value: i16x16) -> Self {
        i32x16::from_i16x16(value)
    }
}

/// The following functionality exists only for [`i32x16`], or only for
/// particular types inconsistently.
impl i32x16 {
    /// Converts each element from [`i16`] to [`i32`].
    #[inline]
    #[must_use]
    pub fn from_i16x16(v: i16x16) -> Self {
        pick! {
          if #[cfg(target_feature="avx512f")] {
            Self(convert_to_i32_m512i_from_i16_m256i(v.0))
          } else {
            let [a, b]: [i16x8; 2] = cast(v);
            Self(Inner {
              lo: i32x8::from_i16x8(a),
              hi: i32x8::from_i16x8(b),
            })
          }
        }
    }

    /// Converts each element from [`i32`] to [`f32`].
    #[inline]
    #[must_use]
    pub fn round_float(self) -> f32x16 {
        pick! {
          if #[cfg(target_feature="avx512f")] {
            cast(convert_to_m512_from_i32_m512i(self.0))
          } else {
            f32x16(crate::f32x16_::Inner {
              lo: self.0.lo.round_float(),
              hi: self.0.hi.round_float(),
            })
          }
        }
    }
}
