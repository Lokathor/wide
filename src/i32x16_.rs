use super::*;

pick! {
  if #[cfg(target_feature="avx512f")] {
    #[derive(Default, Clone, Copy, PartialEq, Eq)]
    #[repr(C, align(64))]
    pub struct i32x16 { pub(crate) avx512: m512i }
  } else {
    #[derive(Default, Clone, Copy, PartialEq, Eq)]
    #[repr(C, align(64))]
    pub struct i32x16 { pub(crate) a : i32x8, pub(crate) b : i32x8 }
  }
}

impl_simd! {
  unsafe {
    T = i32,
    N = 16,
    Simd = i32x16,
    optional_type_x86_inner { X86Inner = __m512i },
    optional_type_arm_inner {},
    optional_type_wasm_inner {},
  }

  #[inline]
  fn simd_eq(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: cmp_op_mask_i32_m512i::<{cmp_int_op!(Eq)}>(self.avx512, rhs.avx512) }
      } else {
        Self {
          a : self.a.simd_eq(rhs.a),
          b : self.b.simd_eq(rhs.b),
        }
      }
    }
  }

  #[inline]
  fn simd_ne(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: cmp_op_mask_i32_m512i::<{cmp_int_op!(Ne)}>(self.avx512, rhs.avx512) }
      } else {
        Self {
          a : self.a.simd_ne(rhs.a),
          b : self.b.simd_ne(rhs.b),
        }
      }
    }
  }

  #[inline]
  fn simd_lt(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: cmp_op_mask_i32_m512i::<{cmp_int_op!(Lt)}>(self.avx512, rhs.avx512) }
      } else {
        Self {
          a : rhs.a.simd_gt(self.a),
          b : rhs.b.simd_gt(self.b),
        }
      }
    }
  }

  #[inline]
  fn simd_gt(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: cmp_op_mask_i32_m512i::<{cmp_int_op!(Nle)}>(self.avx512, rhs.avx512) }
      } else {
        Self {
          a : self.a.simd_gt(rhs.a),
          b : self.b.simd_gt(rhs.b),
        }
      }
    }
  }

  #[inline]
  fn simd_le(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: cmp_op_mask_i32_m512i::<{cmp_int_op!(Le)}>(self.avx512, rhs.avx512) }
      } else {
        Self {
          a : self.a.simd_le(rhs.a),
          b : self.b.simd_le(rhs.b),
        }
      }
    }
  }

  #[inline]
  fn simd_ge(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: cmp_op_mask_i32_m512i::<{cmp_int_op!(Nlt)}>(self.avx512, rhs.avx512) }
      } else {
        Self {
          a : self.a.simd_ge(rhs.a),
          b : self.b.simd_ge(rhs.b),
        }
      }
    }
  }

  #[inline]
  pub fn bitselect(self, if_one: Self, if_zero: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self {
          avx512: bitor_m512i(
            bitand_m512i(if_one.avx512, self.avx512),
            bitandnot_m512i(self.avx512, if_zero.avx512),
          ),
        }
      } else {
        Self {
          a: self.a.bitselect(if_one.a, if_zero.a),
          b: self.b.bitselect(if_one.b, if_zero.b),
        }
      }
    }
  }

  #[inline]
  pub fn select(self, if_true: Self, if_false: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: blend_varying_i8_m512i(if_false.avx512,if_true.avx512,movepi8_mask_m512i(self.avx512)) }
      } else {
        Self {
          a : self.a.select(if_true.a, if_false.a),
          b : self.b.select(if_true.b, if_false.b),
        }
      }
    }
  }

  #[inline]
  pub fn to_bitmask(self) -> u32 {
    pick! {
      if #[cfg(target_feature="avx512dq")] {
        movepi32_mask_m512i(self.avx512) as u32
      } else {
        self.a.to_bitmask() | (self.b.to_bitmask() << 8)
      }
    }
  }

  #[inline]
  pub fn any(self) -> bool {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
        movepi32_mask_m512i(self.avx512) != 0
      } else {
        let [a, b]: [i32x8; 2] = cast(self);
        (a | b).any()
      }
    }
  }

  #[inline]
  pub fn all(self) -> bool {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
        movepi32_mask_m512i(self.avx512) == 0xFFFF
      } else {
        let [a, b]: [i32x8; 2] = cast(self);
        (a & b).all()
      }
    }
  }

  /// Transpose matrix of 16x16 `i32` matrix. Currently not accelerated.
  #[inline]
  pub fn transpose(data: [i32x16; 16]) -> [i32x16; 16] {
    // Can this be optimized?

    #[inline(always)]
    fn transpose_column(data: &[i32x16; 16], index: usize) -> i32x16 {
      i32x16::new([
        data[0].as_array()[index],
        data[1].as_array()[index],
        data[2].as_array()[index],
        data[3].as_array()[index],
        data[4].as_array()[index],
        data[5].as_array()[index],
        data[6].as_array()[index],
        data[7].as_array()[index],
        data[8].as_array()[index],
        data[9].as_array()[index],
        data[10].as_array()[index],
        data[11].as_array()[index],
        data[12].as_array()[index],
        data[13].as_array()[index],
        data[14].as_array()[index],
        data[15].as_array()[index],
      ])
    }

    [
      transpose_column(&data, 0),
      transpose_column(&data, 1),
      transpose_column(&data, 2),
      transpose_column(&data, 3),
      transpose_column(&data, 4),
      transpose_column(&data, 5),
      transpose_column(&data, 6),
      transpose_column(&data, 7),
      transpose_column(&data, 8),
      transpose_column(&data, 9),
      transpose_column(&data, 10),
      transpose_column(&data, 11),
      transpose_column(&data, 12),
      transpose_column(&data, 13),
      transpose_column(&data, 14),
      transpose_column(&data, 15),
    ]
  }
}

impl_simd_int! {
  unsafe {
    T = i32,
    N = 16,
    Simd = i32x16,
    UnsignedSimd = u32x16,
    T_BITS = 32,
    T_BITS_MUL_2 = 64,
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
  }

  #[inline]
  fn not(self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: bitxor_m512i(self.avx512, set_splat_i32_m512i(-1)) }
      } else {
        Self {
          a : self.a.not(),
          b : self.b.not(),
        }
      }
    }
  }

  #[inline]
  fn add(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: add_i32_m512i(self.avx512, rhs.avx512) }
      } else {
        Self {
          a : self.a.add(rhs.a),
          b : self.b.add(rhs.b),
        }
      }
    }
  }

  #[inline]
  fn sub(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: sub_i32_m512i(self.avx512, rhs.avx512) }
      } else {
        Self {
          a : self.a.sub(rhs.a),
          b : self.b.sub(rhs.b),
        }
      }
    }
  }

  #[inline]
  fn mul(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: mul_i32_keep_low_m512i(self.avx512, rhs.avx512) }
      } else {
        Self { a: self.a.mul(rhs.a), b: self.b.mul(rhs.b) }
      }
    }
  }

  #[inline]
  fn shl(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        // Mask `rhs` to 31 to match `wrapping_shl`.
        let rhs = bitand_m512i(rhs.avx512, set_splat_i32_m512i(31));
        Self { avx512: shl_each_u32_m512i(self.avx512, rhs) }
      } else {
        Self {
          a: self.a.shl(rhs.a),
          b: self.b.shl(rhs.b),
        }
      }
    }
  }

  #[inline]
  fn shl(self, rhs: u32) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        // Use `rhs % 32` to perform wrapping shift and not unbounded shift.
        #[expect(clippy::suspicious_arithmetic_impl)]
        let shift = rhs & 31;
        Self { avx512: shl_all_u32_m512i(self.avx512, shift) }
      } else {
        Self {
          a : self.a.shl(rhs),
          b : self.b.shl(rhs),
        }
      }
    }
  }

  #[inline]
  fn shr(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        #[cfg(target_arch = "x86")]
        use core::arch::x86::_mm512_srav_epi32;
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::_mm512_srav_epi32;

        // Mask `rhs` to 31 to match `wrapping_shr`.
        let rhs = bitand_m512i(rhs.avx512, set_splat_i32_m512i(31));
        // TODO(safe_arch): Add `_mm512_srav_epi32`.
        Self { avx512: m512i(unsafe { _mm512_srav_epi32(self.avx512.0, rhs.0) }) }
      } else {
        Self {
          a: self.a >> rhs.a,
          b: self.b >> rhs.b,
        }
      }
    }
  }

  #[inline]
  fn shr(self, rhs: u32) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        // Use `rhs % 32` to perform wrapping shift and not unbounded shift.
        #[expect(clippy::suspicious_arithmetic_impl)]
        let shift = rhs & 31;
        Self { avx512: shr_all_i32_m512i(self.avx512, shift) }
      } else {
        Self {
          a : self.a.shr(rhs),
          b : self.b.shr(rhs),
        }
      }
    }
  }

  #[inline]
  fn bitand(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: bitand_m512i(self.avx512, rhs.avx512) }
      } else {
        Self {
          a : self.a.bitand(rhs.a),
          b : self.b.bitand(rhs.b),
        }
      }
    }
  }

  #[inline]
  fn bitor(self, rhs: Self) -> Self::Output {
    pick! {
    if #[cfg(target_feature="avx512f")] {
        Self { avx512: bitor_m512i(self.avx512, rhs.avx512) }
      } else {
        Self {
          a : self.a.bitor(rhs.a),
          b : self.b.bitor(rhs.b),
        }
      }
    }
  }

  #[inline]
  fn bitxor(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: bitxor_m512i(self.avx512, rhs.avx512) }
      } else {
        Self {
          a : self.a.bitxor(rhs.a),
          b : self.b.bitxor(rhs.b),
        }
      }
    }
  }

  #[inline]
  pub fn max(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: max_i32_m512i(self.avx512, rhs.avx512) }
      } else {
        Self {
          a: self.a.max(rhs.a),
          b: self.b.max(rhs.b),
        }
      }
    }
  }

  #[inline]
  pub fn min(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: min_i32_m512i(self.avx512, rhs.avx512) }
      } else {
        Self {
          a: self.a.min(rhs.a),
          b: self.b.min(rhs.b),
        }
      }
    }
  }

  #[inline]
  pub fn reduce_add(self) -> i32 {
    let arr: [i32x8; 2] = cast(self);
    (arr[0] + arr[1]).reduce_add()
  }

  #[inline]
  pub fn reduce_mul(self) -> i32 {
    let array: [i32x8; 2] = cast(self);
    (array[0] * array[1]).reduce_mul()
  }

  #[inline]
  pub fn reduce_max(self) -> i32 {
    let arr: [i32x8; 2] = cast(self);
    arr[0].max(arr[1]).reduce_max()
  }

  #[inline]
  pub fn reduce_min(self) -> i32 {
    let arr: [i32x8; 2] = cast(self);
    arr[0].min(arr[1]).reduce_min()
  }

  #[inline]
  pub fn unbounded_shr(self, rhs: u32x16) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        #[cfg(target_arch = "x86")]
        use core::arch::x86::_mm512_srav_epi32;
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::_mm512_srav_epi32;

        // TODO(safe_arch): Add `_mm512_srav_epi32`.
        Self { avx512: m512i(unsafe { _mm512_srav_epi32(self.avx512.0, rhs.avx512.0) }) }
      } else {
        Self {
          a: self.a.unbounded_shr(rhs.a),
          b: self.b.unbounded_shr(rhs.b),
        }
      }
    }
  }

  #[inline]
  pub fn unbounded_shr_scalar(self, rhs: u32) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: shr_all_i32_m512i(self.avx512, rhs) }
      } else {
        Self {
          a: self.a.unbounded_shr_scalar(rhs),
          b: self.b.unbounded_shr_scalar(rhs),
        }
      }
    }
  }

  #[inline]
  pub fn saturating_add(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        let result = self + rhs;
        let overflow = (!(self ^ rhs) & (self ^ result)).is_negative();
        let negative = self.is_negative();

        // If overflow occurs return `MAX` if positive or `MIN` if negative.
        overflow.select(Self::MAX ^ negative, result)
      } else {
        Self {
          a: self.a.saturating_add(rhs.a),
          b: self.b.saturating_add(rhs.b),
        }
      }
    }
  }

  #[inline]
  pub fn saturating_sub(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        let result = self - rhs;
        let overflow = ((self ^ rhs) & (self ^ result)).is_negative();
        let negative = self.is_negative();

        // If overflow occurs return `MAX` if positive or `MIN` if negative.
        overflow.select(Self::MAX ^ negative, result)
      } else {
        Self {
          a: self.a.saturating_sub(rhs.a),
          b: self.b.saturating_sub(rhs.b),
        }
      }
    }
  }

  #[inline]
  pub fn overflowing_mul(self, rhs: Self) -> (Self, Self) {
    let (low, high) = self.mul_keep_low_high(rhs);
    let low = cast::<u32x16, i32x16>(low);

    let overflow = high.simd_ne(low.is_negative());
    (low, overflow)
  }

  optional_fn_widening_mul {
    // Cannot have `widening_mul` because there is no `i64x16` type.
  }

  #[inline]
  pub fn mul_keep_low_high(self, rhs: Self) -> (u32x16, i32x16) {
    pick! {
      if #[cfg(all(target_feature="avx512f", target_feature="avx512dq"))] {
        #[cfg(target_arch = "x86")]
        use core::arch::x86::{_mm512_unpackhi_epi64, _mm512_unpacklo_epi64};
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::{_mm512_unpackhi_epi64, _mm512_unpacklo_epi64};

        let even_wide_mul = mul_i32_wide_m512i(self.avx512, rhs.avx512);
        let odd_wide_mul = mul_i32_wide_m512i(
          shuffle_i32_m512i::<0b_00_11_00_01>(self.avx512),
          shuffle_i32_m512i::<0b_00_11_00_01>(rhs.avx512),
        );
        let ll_hh_1 = unpack_low_i32_m512i(even_wide_mul, odd_wide_mul);
        let ll_hh_2 = unpack_high_i32_m512i(even_wide_mul, odd_wide_mul);
        // TODO(safe_arch): Add `_mm512_unpacklo_epi64` and `_mm512_unpackhi_epi64`.
        (
          u32x16 {
            avx512: m512i(unsafe { _mm512_unpacklo_epi64(ll_hh_1.0, ll_hh_2.0) }),
          },
          i32x16 {
            avx512: m512i(unsafe { _mm512_unpackhi_epi64(ll_hh_1.0, ll_hh_2.0) }),
          },
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
  pub fn mul_keep_high(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(all(target_feature="avx512f", target_feature="avx512dq"))] {
        #[cfg(target_arch = "x86")]
        use core::arch::x86::_mm512_unpackhi_epi64;
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::_mm512_unpackhi_epi64;

        let even_wide_mul = mul_i32_wide_m512i(self.avx512, rhs.avx512);
        let odd_wide_mul = mul_i32_wide_m512i(
          shuffle_i32_m512i::<0b_00_11_00_01>(self.avx512),
          shuffle_i32_m512i::<0b_00_11_00_01>(rhs.avx512),
        );
        let ll_hh_1 = unpack_low_i32_m512i(even_wide_mul, odd_wide_mul);
        let ll_hh_2 = unpack_high_i32_m512i(even_wide_mul, odd_wide_mul);
        // TODO(safe_arch): Add `_mm512_unpackhi_epi64`.
        Self {
          avx512: m512i(unsafe { _mm512_unpackhi_epi64(ll_hh_1.0, ll_hh_2.0) }),
        }
      } else {
        let [self_a, self_b] = cast::<i32x16, [i32x8; 2]>(self);
        let [rhs_a, rhs_b] = cast::<i32x16, [i32x8; 2]>(rhs);

        cast([self_a.mul_keep_high(rhs_a), self_b.mul_keep_high(rhs_b)])
      }
    }
  }

  #[inline]
  pub fn abs(self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
        Self { avx512: abs_i32_m512i(self.avx512) }
      } else {
        Self {
          a : self.a.abs(),
          b : self.b.abs(),
        }
      }
    }
  }

  #[inline]
  pub fn is_positive(self) -> Self {
    pick! {
      if #[cfg(all(target_feature="neon", target_arch="aarch64"))] {
        // `neon` has dedicated greater-than-zero intrinsics.
        Self {
          a: self.a.is_positive(),
          b: self.b.is_positive(),
        }
      } else {
        self.simd_gt(Self::ZERO)
      }
    }
  }

  #[inline]
  pub fn is_negative(self) -> Self {
    pick! {
      if #[cfg(all(target_feature="neon", target_arch="aarch64"))] {
        // `neon` has dedicated less-than-zero intrinsics.
        Self {
          a: self.a.is_negative(),
          b: self.b.is_negative(),
        }
      } else {
        self.simd_lt(Self::ZERO)
      }
    }
  }
}

impl i32x16 {
  #[inline]
  #[must_use]
  pub fn round_float(self) -> f32x16 {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        cast(convert_to_m512_from_i32_m512i(self.avx512))
      } else {
        f32x16 {
          a: self.a.round_float(),
          b: self.b.round_float(),
        }
      }
    }
  }
}
