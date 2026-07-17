use super::*;

pick! {
  if #[cfg(target_feature="avx2")] {
    #[derive(Default, Clone, Copy, PartialEq, Eq)]
    #[repr(C, align(32))]
    pub struct u32x8 { pub(crate) avx2: m256i }
  } else {
    #[derive(Default, Clone, Copy, PartialEq, Eq)]
    #[repr(C, align(32))]
    pub struct u32x8 { pub(crate) a : u32x4, pub(crate) b : u32x4 }
  }
}

impl_simd! {
  unsafe {
    T = u32,
    N = 8,
    Simd = u32x8,
    optional_type_x86_inner { X86Inner = __m256i },
    optional_type_arm_inner {},
    optional_type_wasm_inner {},
  }

  #[inline]
  fn simd_eq(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self { avx2: cmp_eq_mask_i32_m256i(self.avx2, rhs.avx2 ) }
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
    !self.simd_eq(rhs)
  }

  #[inline]
  fn simd_lt(self, rhs: Self) -> Self::Output {
    // lt is just gt the other way around
    rhs.simd_gt(self)
  }

  #[inline]
  fn simd_gt(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        // no unsigned gt than so inverting the high bit will get the correct result
        let highbit = u32x8::splat(1 << 31);
        Self { avx2: cmp_gt_mask_i32_m256i((self ^ highbit).avx2, (rhs ^ highbit).avx2 ) }
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
    self.simd_eq(rhs) | self.simd_lt(rhs)
  }

  #[inline]
  fn simd_ge(self, rhs: Self) -> Self::Output {
    self.simd_eq(rhs) | self.simd_gt(rhs)
  }

  #[inline]
  pub fn bitselect(self, if_one: Self, if_zero: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self {
          avx2: bitor_m256i(
            bitand_m256i(if_one.avx2, self.avx2),
            bitandnot_m256i(self.avx2, if_zero.avx2),
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
      if #[cfg(target_feature="avx2")] {
        Self { avx2: blend_varying_i8_m256i(if_false.avx2, if_true.avx2, self.avx2) }
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
    i32x8::to_bitmask(cast(self))
  }

  #[inline]
  pub fn any(self) -> bool {
    pick! {
      if #[cfg(target_feature="avx2")] {
        ((move_mask_i8_m256i(self.avx2) as u32) & 0b10001000100010001000100010001000) != 0
      } else {
        (self.a | self.b).any()
      }
    }
  }

  #[inline]
  pub fn all(self) -> bool {
    pick! {
      if #[cfg(target_feature="avx2")] {
        ((move_mask_i8_m256i(self.avx2) as u32) & 0b10001000100010001000100010001000) == 0b10001000100010001000100010001000
      } else {
        (self.a & self.b).all()
      }
    }
  }

  /// Transpose matrix of 8x8 `u32` matrix. Currently only accelerated on AVX2.
  #[inline]
  pub fn transpose(data: [u32x8; 8]) -> [u32x8; 8] {
    cast(i32x8::transpose(cast(data)))
  }
}

impl_simd_uint! {
  unsafe {
    T = u32,
    N = 8,
    Simd = u32x8,
    T_BITS = 32,
    T_BITS_MUL_2 = 64,
    [0, 1, 2, 3, 4, 5, 6, 7],
  }

  #[inline]
  fn not(self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self { avx2: self.avx2.not()  }
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
      if #[cfg(target_feature="avx2")] {
        Self { avx2: add_i32_m256i(self.avx2, rhs.avx2) }
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
      if #[cfg(target_feature="avx2")] {
        Self { avx2: sub_i32_m256i(self.avx2, rhs.avx2) }
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
      if #[cfg(target_feature="avx2")] {
        Self { avx2: mul_i32_keep_low_m256i(self.avx2, rhs.avx2) }
      } else {
        Self {
          a : self.a.mul(rhs.a),
          b : self.b.mul(rhs.b),
        }
      }
    }
  }

  #[inline]
  fn shl(self, rhs: u32x8) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        // ensure same behavior as scalar wrapping_shl
        let shift_by = bitand_m256i(rhs.avx2, set_splat_i32_m256i(31));
        Self { avx2: shl_each_u32_m256i(self.avx2, shift_by) }
      } else {
        Self {
          a : self.a.shl(rhs.a),
          b : self.b.shl(rhs.b),
        }
      }
    }
  }

  #[inline]
  fn shl(self, rhs: u32) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        // Use `rhs % 32` to perform wrapping shift and not unbounded shift.
        #[expect(clippy::suspicious_arithmetic_impl)]
        let shift = cast([rhs as u64 & 31, 0]);
        Self { avx2: shl_all_u32_m256i(self.avx2, shift) }
      } else {
        Self {
          a : self.a.shl(rhs),
          b : self.b.shl(rhs),
        }
      }
    }
  }

  #[inline]
  fn shr(self, rhs: u32x8) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        // ensure same behavior as scalar wrapping_shr
        let shift_by = bitand_m256i(rhs.avx2, set_splat_i32_m256i(31));
        Self { avx2: shr_each_u32_m256i(self.avx2, shift_by ) }
      } else {
        Self {
          a : self.a.shr(rhs.a),
          b : self.b.shr(rhs.b),
        }
      }
    }
  }

  #[inline]
  fn shr(self, rhs: u32) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        // Use `rhs % 32` to perform wrapping shift and not unbounded shift.
        #[expect(clippy::suspicious_arithmetic_impl)]
        let shift = cast([rhs as u64 & 31, 0]);
        Self { avx2: shr_all_u32_m256i(self.avx2, shift) }
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
      if #[cfg(target_feature="avx2")] {
        Self { avx2: bitand_m256i(self.avx2, rhs.avx2) }
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
      if #[cfg(target_feature="avx2")] {
        Self { avx2: bitor_m256i(self.avx2, rhs.avx2) }
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
      if #[cfg(target_feature="avx2")] {
        Self { avx2: bitxor_m256i(self.avx2, rhs.avx2) }
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
      if #[cfg(target_feature="avx2")] {
        Self { avx2: max_u32_m256i(self.avx2, rhs.avx2 ) }
      } else {
        Self {
          a : self.a.max(rhs.a),
          b : self.b.max(rhs.b),
        }
      }
    }
  }

  #[inline]
  pub fn min(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self { avx2: min_u32_m256i(self.avx2, rhs.avx2 ) }
      } else {
        Self {
          a : self.a.min(rhs.a),
          b : self.b.min(rhs.b),
        }
      }
    }
  }

  #[inline]
  pub fn reduce_add(self) -> u32 {
    cast(i32x8::reduce_add(cast(self)))
  }

  #[inline]
  pub fn reduce_mul(self) -> u32 {
    let array: [u32x4; 2] = cast(self);
    (array[0] * array[1]).reduce_mul()
  }

  #[inline]
  pub fn reduce_max(self) -> u32 {
    let array: [u32x4; 2] = cast(self);
    array[0].max(array[1]).reduce_max()
  }

  #[inline]
  pub fn reduce_min(self) -> u32 {
    let array: [u32x4; 2] = cast(self);
    array[0].min(array[1]).reduce_min()
  }

  #[inline]
  pub fn unbounded_shl(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self { avx2: shl_each_u32_m256i(self.avx2, rhs.avx2) }
      } else {
        Self {
          a: self.a.unbounded_shl(rhs.a),
          b: self.b.unbounded_shl(rhs.b),
        }
      }
    }
  }

  #[inline]
  pub fn saturating_add(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        let result = self + rhs;
        let overflow = result.simd_lt(self);
        // Return `MAX` (all bits set) if overflow occurs.
        result | overflow
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
      if #[cfg(target_feature="avx2")] {
        let result = self - rhs;
        let no_overflow = result.simd_le(self);
        // Return `0` (no bits set) if overflow occurs.
        result & no_overflow
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
    let overflow = high.simd_ne(Self::ZERO);
    (low, overflow)
  }

  optional_fn_widening_mul {
    #[inline]
    pub fn widening_mul(self, rhs: Self) -> u64x8 {
      pick! {
        if #[cfg(all(target_feature="avx512f", target_feature="avx2"))] {
          const SHUFFLE_INDICES: m512i = i64x8::new([0, 4, 1, 5, 2, 6, 3, 7]).avx512;

          let even_wide_mul = mul_u64_low_bits_m256i(self.avx2, rhs.avx2);
          let odd_wide_mul = mul_u64_low_bits_m256i(
            shuffle_ai_i32_half_m256i::<0b_00_11_00_01>(self.avx2),
            shuffle_ai_i32_half_m256i::<0b_00_11_00_01>(rhs.avx2),
          );
          let even_then_odd = cast::<[m256i; 2], m512i>([even_wide_mul, odd_wide_mul]);
          u64x8 {
            avx512: permute_i64_m512i(SHUFFLE_INDICES, even_then_odd),
          }
        } else {
          let [self_a, self_b] = cast::<u32x8, [u32x4; 2]>(self);
          let [rhs_a, rhs_b] = cast::<u32x8, [u32x4; 2]>(rhs);

          cast([self_a.widening_mul(rhs_a), self_b.widening_mul(rhs_b)])
        }
      }
    }
  }

  #[inline]
  pub fn mul_keep_low_high(self, rhs: Self) -> (Self, Self) {
    pick! {
      if #[cfg(target_feature="avx2")] {
        let even_wide_mul = mul_u64_low_bits_m256i(self.avx2, rhs.avx2);
        let odd_wide_mul = mul_u64_low_bits_m256i(
          shuffle_ai_i32_half_m256i::<0b_00_11_00_01>(self.avx2),
          shuffle_ai_i32_half_m256i::<0b_00_11_00_01>(rhs.avx2),
        );
        let ll_hh_1 = unpack_low_i32_m256i(even_wide_mul, odd_wide_mul);
        let ll_hh_2 = unpack_high_i32_m256i(even_wide_mul, odd_wide_mul);
        (
          Self { avx2: unpack_low_i64_m256i(ll_hh_1, ll_hh_2) },
          Self { avx2: unpack_high_i64_m256i(ll_hh_1, ll_hh_2) },
        )
      } else {
        let [self_a, self_b] = cast::<u32x8, [u32x4; 2]>(self);
        let [rhs_a, rhs_b] = cast::<u32x8, [u32x4; 2]>(rhs);

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
  pub fn mul_keep_high(self, rhs: u32x8) -> u32x8 {
    pick! {
      if #[cfg(target_feature="avx2")] {
        let a : [u32;8]= cast(self);
        let b : [u32;8]= cast(rhs);

        // let the compiler shuffle the values around, it does the right thing
        let r1 : [u32;8] = cast(mul_u64_low_bits_m256i(cast([a[0], 0, a[1], 0, a[2], 0, a[3], 0]), cast([b[0], 0, b[1], 0, b[2], 0, b[3], 0])));
        let r2 : [u32;8] = cast(mul_u64_low_bits_m256i(cast([a[4], 0, a[5], 0, a[6], 0, a[7], 0]), cast([b[4], 0, b[5], 0, b[6], 0, b[7], 0])));

        cast([r1[1], r1[3], r1[5], r1[7], r2[1], r2[3], r2[5], r2[7]])
      } else {
        Self {
          a : self.a.mul_keep_high(rhs.a),
          b : self.b.mul_keep_high(rhs.b),
        }
      }
    }
  }
}

impl From<u16x8> for u32x8 {
  /// widens and zero extends to u32x8
  #[inline]
  fn from(v: u16x8) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self { avx2:convert_to_i32_m256i_from_u16_m128i(v.sse) }
      } else if #[cfg(target_feature="sse2")] {
        Self {
          a: u32x4 { sse: shr_imm_u32_m128i::<16>( unpack_low_i16_m128i(v.sse, v.sse)) },
          b: u32x4 { sse: shr_imm_u32_m128i::<16>( unpack_high_i16_m128i(v.sse, v.sse)) },
        }
      } else {
        u32x8::new([
          u32::from(v.as_array()[0]),
          u32::from(v.as_array()[1]),
          u32::from(v.as_array()[2]),
          u32::from(v.as_array()[3]),
          u32::from(v.as_array()[4]),
          u32::from(v.as_array()[5]),
          u32::from(v.as_array()[6]),
          u32::from(v.as_array()[7]),
        ])
      }
    }
  }
}
