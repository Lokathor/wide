use super::*;

pick! {
  if #[cfg(target_feature="avx2")] {
    #[derive(Default, Clone, Copy, PartialEq, Eq)]
    #[repr(C, align(32))]
    pub struct i16x16 { pub(crate) avx2: m256i }
  } else {
    #[derive(Default, Clone, Copy, PartialEq, Eq)]
    #[repr(C, align(32))]
    pub struct i16x16 { pub(crate) a : i16x8, pub(crate) b : i16x8 }
  }
}

impl_simd! {
  T = i16,
  N = 16,
  Simd = i16x16,

  #[inline]
  fn simd_eq(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self { avx2: cmp_eq_mask_i16_m256i(self.avx2, rhs.avx2) }
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
      if #[cfg(target_feature="avx2")] {
        !self.simd_eq(rhs)
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
      if #[cfg(target_feature="avx2")] {
        Self { avx2: !cmp_gt_mask_i16_m256i(self.avx2, rhs.avx2) ^ cmp_eq_mask_i16_m256i(self.avx2,rhs.avx2) }
      } else {
        Self {
          a : self.a.simd_lt(rhs.a),
          b : self.b.simd_lt(rhs.b),
        }
      }
    }
  }

  #[inline]
  fn simd_gt(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self { avx2: cmp_gt_mask_i16_m256i(self.avx2, rhs.avx2) }
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
      if #[cfg(target_feature="avx2")] {
        !self.simd_gt(rhs)
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
      if #[cfg(target_feature="avx2")] {
        !self.simd_lt(rhs)
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
    pick! {
      if #[cfg(target_feature="sse2")] {
          let [a,b] = cast::<_,[m128i;2]>(self);
          move_mask_i8_m128i( pack_i16_to_i8_m128i(a,b)) as u32
        } else {
        self.a.to_bitmask() | (self.b.to_bitmask() << 8)
      }
    }
  }

  #[inline]
  pub fn any(self) -> bool {
    pick! {
      if #[cfg(target_feature="avx2")] {
        ((move_mask_i8_m256i(self.avx2) as u32) & 0b10101010101010101010101010101010) != 0
      } else {
        (self.a | self.b).any()
      }
    }
  }

  #[inline]
  pub fn all(self) -> bool {
    pick! {
      if #[cfg(target_feature="avx2")] {
        ((move_mask_i8_m256i(self.avx2) as u32) & 0b10101010101010101010101010101010) == 0b10101010101010101010101010101010
      } else {
        (self.a & self.b).all()
      }
    }
  }

  /// Transpose matrix of 16x16 `i16` matrix. Currently not accelerated.
  #[inline]
  pub fn transpose(data: [i16x16; 16]) -> [i16x16; 16] {
    // Can this be optimized?

    #[inline(always)]
    fn transpose_column(data: &[i16x16; 16], index: usize) -> i16x16 {
      i16x16::new([
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
  T = i16,
  N = 16,
  Simd = i16x16,
  UnsignedSimd = u16x16,
  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],

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
        Self { avx2: add_i16_m256i(self.avx2, rhs.avx2) }
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
        Self { avx2: sub_i16_m256i(self.avx2, rhs.avx2) }
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
        Self { avx2: mul_i16_keep_low_m256i(self.avx2, rhs.avx2) }
      } else {
        Self {
          a : self.a.mul(rhs.a),
          b : self.b.mul(rhs.b),
        }
      }
    }
  }

  #[inline]
  fn shl(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(all(target_feature="avx512bw", target_feature="avx512vl"))] {
        #[cfg(target_arch = "x86")]
        use core::arch::x86::_mm256_sllv_epi16;
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::_mm256_sllv_epi16;

        // Mask `rhs` to 15 to match `wrapping_shl`.
        let rhs = bitand_m256i(rhs.avx2, set_splat_i16_m256i(15));
        // TODO(safe_arch): Add `_mm256_sllv_epi16`.
        cast(unsafe { _mm256_sllv_epi16(self.avx2.0, rhs.0) })
      } else {
        let [self_a, self_b]: [i16x8; 2] = cast(self);
        let [rhs_a, rhs_b]: [i16x8; 2] = cast(rhs);

        cast([self_a << rhs_a, self_b << rhs_b])
      }
    }
  }

  #[inline]
  fn shl(self, rhs: u32) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        // Use `rhs % 16` to perform wrapping shift and not unbounded shift.
        #[expect(clippy::suspicious_arithmetic_impl)]
        let shift = cast([rhs as u64 & 15, 0]);
        Self { avx2: shl_all_u16_m256i(self.avx2, shift) }
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
      if #[cfg(all(target_feature="avx512bw", target_feature="avx512vl"))] {
        #[cfg(target_arch = "x86")]
        use core::arch::x86::_mm256_srav_epi16;
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::_mm256_srav_epi16;

        // Mask `rhs` to 15 to match `wrapping_shr`.
        let rhs = bitand_m256i(rhs.avx2, set_splat_i16_m256i(15));
        // TODO(safe_arch): Add `_mm256_srav_epi16`.
        cast(unsafe { _mm256_srav_epi16(self.avx2.0, rhs.0) })
      } else {
        let [self_a, self_b]: [i16x8; 2] = cast(self);
        let [rhs_a, rhs_b]: [i16x8; 2] = cast(rhs);

        cast([self_a >> rhs_a, self_b >> rhs_b])
      }
    }
  }

  #[inline]
  fn shr(self, rhs: u32) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        // Use `rhs % 16` to perform wrapping shift and not unbounded shift.
        #[expect(clippy::suspicious_arithmetic_impl)]
        let shift = cast([rhs as u64 & 15, 0]);
        Self { avx2: shr_all_i16_m256i(self.avx2, shift) }
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
        Self { avx2: max_i16_m256i(self.avx2, rhs.avx2) }
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
        Self { avx2: min_i16_m256i(self.avx2, rhs.avx2) }
      } else {
        Self {
          a : self.a.min(rhs.a),
          b : self.b.min(rhs.b),
        }
      }
    }
  }

  #[inline]
  pub fn reduce_add(self) -> i16 {
    let arr: [i16x8; 2] = cast(self);

    (arr[0] + arr[1]).reduce_add()
  }

  #[inline]
  pub fn reduce_mul(self) -> i16 {
    let array: [i16x8; 2] = cast(self);
    (array[0] * array[1]).reduce_mul()
  }

  #[inline]
  pub fn reduce_max(self) -> i16 {
    let arr: [i16x8; 2] = cast(self);

    arr[0].max(arr[1]).reduce_max()
  }

  #[inline]
  pub fn reduce_min(self) -> i16 {
    let arr: [i16x8; 2] = cast(self);

    arr[0].min(arr[1]).reduce_min()
  }

  #[inline]
  pub fn saturating_add(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self { avx2: add_saturating_i16_m256i(self.avx2, rhs.avx2) }
      } else {
        Self {
          a : self.a.saturating_add(rhs.a),
          b : self.b.saturating_add(rhs.b),
        }
      }
    }
  }

  #[inline]
  pub fn saturating_sub(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self { avx2: sub_saturating_i16_m256i(self.avx2, rhs.avx2) }
      } else {
        Self {
          a : self.a.saturating_sub(rhs.a),
          b : self.b.saturating_sub(rhs.b),
        }
      }
    }
  }

  #[inline]
  pub fn saturating_mul(self, rhs: Self) -> Self {
    let [self_a, self_b]: [i16x8; 2] = cast(self);
    let [rhs_a, rhs_b]: [i16x8; 2] = cast(rhs);
    cast([self_a.saturating_mul(rhs_a), self_b.saturating_mul(rhs_b)])
  }

  #[inline]
  pub fn overflowing_mul(self, rhs: Self) -> (Self, Self) {
    // x86 has no `_mm256_mul_epi16` intrinsic so there is no `avx2`
    // optimization.

    let [self_a, self_b] = cast::<i16x16, [i16x8; 2]>(self);
    let [rhs_a, rhs_b] = cast::<i16x16, [i16x8; 2]>(rhs);

    let result_a = self_a.overflowing_mul(rhs_a);
    let result_b = self_b.overflowing_mul(rhs_b);
    (cast([result_a.0, result_b.0]), cast([result_a.1, result_b.1]))
  }

  #[inline]
  pub fn abs(self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self { avx2: abs_i16_m256i(self.avx2) }
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

impl AlignTo for i16x16 {
  type Elem = i16;
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

impl i16x16 {
  /// widens and sign extends to i16x16
  #[inline]
  #[must_use]
  pub fn from_i8x16(v: i8x16) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        i16x16 { avx2:convert_to_i16_m256i_from_i8_m128i(v.sse) }
      } else if #[cfg(target_feature="sse4.1")] {
        i16x16 {
          a: i16x8 { sse: convert_to_i16_m128i_from_lower8_i8_m128i(v.sse) },
          b: i16x8 { sse: convert_to_i16_m128i_from_lower8_i8_m128i(unpack_high_i64_m128i(v.sse, v.sse)) }
        }
      } else if #[cfg(target_feature="sse2")] {
        i16x16 {
          a: i16x8 { sse: shr_imm_i16_m128i::<8>( unpack_low_i8_m128i(v.sse, v.sse)) },
          b: i16x8 { sse: shr_imm_i16_m128i::<8>( unpack_high_i8_m128i(v.sse, v.sse)) },
        }
      } else {

        i16x16::new([
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

  /// Calculates partial dot product.
  /// Multiplies packed signed 16-bit integers, producing intermediate signed
  /// 32-bit integers. Horizontally add adjacent pairs of intermediate 32-bit
  /// integers.
  #[inline]
  #[must_use]
  pub fn dot(self, rhs: Self) -> i32x8 {
    pick! {
      if #[cfg(target_feature="avx2")] {
        i32x8 { avx2:  mul_i16_horizontal_add_m256i(self.avx2, rhs.avx2) }
      } else {
        i32x8 {
          a : self.a.dot(rhs.a),
          b : self.b.dot(rhs.b),
        }
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
        Self { avx2: mul_i16_scale_round_m256i(self.avx2, rhs.avx2) }
      } else {
        Self {
          a : self.a.mul_scale_round(rhs.a),
          b : self.b.mul_scale_round(rhs.b),
        }
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
        Self { avx2: mul_i16_scale_round_m256i(self.avx2, set_splat_i16_m256i(rhs)) }
      } else {
        Self {
          a : self.a.mul_scale_round_n(rhs),
          b : self.b.mul_scale_round_n(rhs),
        }
      }
    }
  }
}
