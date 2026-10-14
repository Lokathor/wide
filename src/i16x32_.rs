use crate::imports::*;

pick! {
  if #[cfg(target_feature="avx512bw")] {
    pub(crate) type Inner = m512i;
  } else {
    #[repr(C, align(64))]
    #[derive(Clone, Copy)]
    pub(crate) struct Inner {
      pub lo: i16x16,
      pub hi: i16x16,
    }
  }
}

impl i16x32 {
    #[inline]
    pub(crate) fn simd_lt_backend(self, rhs: Self) -> Self {
        pick! {
          if #[cfg(target_feature="avx512bw")] {
            Self(cmp_op_mask_i16_m512i::<{cmp_int_op!(Lt)}>(self.0, rhs.0))
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
            Self(cmp_op_mask_i16_m512i::<{cmp_int_op!(Nle)}>(self.0, rhs.0))
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
            Self(cmp_op_mask_i16_m512i::<{cmp_int_op!(Le)}>(self.0, rhs.0))
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
            Self(cmp_op_mask_i16_m512i::<{cmp_int_op!(Nlt)}>(self.0, rhs.0))
          } else {
            Self(Inner {
              lo: self.0.lo.simd_ge(rhs.0.lo),
              hi: self.0.hi.simd_ge(rhs.0.hi),
            })
          }
        }
    }

    #[inline]
    pub(crate) fn shr_backend(self, rhs: u16x32) -> Self {
        pick! {
          if #[cfg(target_feature="avx512bw")] {
            #[cfg(target_arch = "x86")]
            use core::arch::x86::_mm512_srav_epi16;
            #[cfg(target_arch = "x86_64")]
            use core::arch::x86_64::_mm512_srav_epi16;

            // Mask `rhs` to 15 to match `wrapping_shr`.
            let rhs = bitand_m512i(rhs.0, set_splat_i16_m512i(15));
            // TODO(safe_arch): Add `_mm512_srav_epi16`.
            Self(m512i(unsafe { _mm512_srav_epi16(self.0.0, rhs.0) }))
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
            Self(shr_all_i16_m512i(self.0, shift))
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
            Self(max_i16_m512i(self.0, rhs.0))
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
            Self(min_i16_m512i(self.0, rhs.0))
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
            #[cfg(target_arch = "x86")]
            use core::arch::x86::_mm512_srav_epi16;
            #[cfg(target_arch = "x86_64")]
            use core::arch::x86_64::_mm512_srav_epi16;

            // TODO(safe_arch): Add `_mm512_srav_epi16`.
            Self(m512i(unsafe { _mm512_srav_epi16(self.0.0, rhs.0.0) }))
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
            Self(shr_all_i16_m512i(self.0, rhs.min(u16::MAX as u32) as u16))
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
            Self(add_saturating_i16_m512i(self.0, rhs.0))
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
            Self(sub_saturating_i16_m512i(self.0, rhs.0))
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

    #[inline]
    pub(crate) fn abs_backend(self) -> Self {
        pick! {
          if #[cfg(target_feature="avx512bw")] {
            Self(abs_i16_m512i(self.0))
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

impl i16x32 {
    #[inline]
    pub(crate) fn from_i8x32_backend(v: i8x32) -> Self {
        pick! {
          if #[cfg(target_feature="avx512bw")] {
            Self(convert_to_i16_m512i_from_i8_m256i(cast(v)))
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
            i32x16(mul_i16_horizontal_add_m512i(self.0, rhs.0))
          } else {
            i32x16(crate::i32x16_::Inner {
              lo: self.0.lo.dot(rhs.0.lo),
              hi: self.0.hi.dot(rhs.0.hi),
            })
          }
        }
    }
}
