use super::*;

pick! {
  if #[cfg(target_feature="avx2")] {
    #[derive(Default, Clone, Copy, PartialEq, Eq)]
    #[repr(C, align(32))]
    pub struct i32x8 { pub(crate) avx2: m256i }
  } else {
    #[derive(Default, Clone, Copy, PartialEq, Eq)]
    #[repr(C, align(32))]
    pub struct i32x8 { pub(crate) a : i32x4, pub(crate) b : i32x4}
  }
}

impl_simd! {
  unsafe {
    T = i32,
    N = 8,
    Simd = i32x8,
  }

  #[inline]
  fn simd_eq(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self { avx2: cmp_eq_mask_i32_m256i(self.avx2, rhs.avx2) }
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
        Self { avx2: cmp_gt_mask_i32_m256i(rhs.avx2, self.avx2) }
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
        Self { avx2: cmp_gt_mask_i32_m256i(self.avx2, rhs.avx2) }
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
          b: self.b.bitselect(if_one.b, if_zero.b)
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
          b : self.b.select(if_true.b, if_false.b)
        }
      }
    }
  }

  #[inline]
  pub fn to_bitmask(self) -> u32 {
    pick! {
      if #[cfg(target_feature="avx2")] {
        // use f32 move_mask since it is the same size as i32
        move_mask_m256(cast(self.avx2)) as u32
      } else {
        self.a.to_bitmask() | (self.b.to_bitmask() << 4)
      }
    }
  }

  #[inline]
  pub fn any(self) -> bool {
    pick! {
      if #[cfg(target_feature="avx2")] {
        move_mask_m256(cast(self.avx2)) != 0
      } else {
        (self.a | self.b).any()
      }
    }
  }

  #[inline]
  pub fn all(self) -> bool {
    pick! {
      if #[cfg(target_feature="avx2")] {
        move_mask_m256(cast(self.avx2)) == 0b11111111
      } else {
        (self.a & self.b).all()
      }
    }
  }

  /// Transpose matrix of 8x8 `i32` matrix. Currently only accelerated on AVX2.
  #[inline]
  pub fn transpose(data: [i32x8; 8]) -> [i32x8; 8] {
    pick! {
      if #[cfg(target_feature="avx2")] {
        let a0 = unpack_low_i32_m256i(data[0].avx2, data[1].avx2);
        let a1 = unpack_high_i32_m256i(data[0].avx2, data[1].avx2);
        let a2 = unpack_low_i32_m256i(data[2].avx2, data[3].avx2);
        let a3 = unpack_high_i32_m256i(data[2].avx2, data[3].avx2);
        let a4 = unpack_low_i32_m256i(data[4].avx2, data[5].avx2);
        let a5 = unpack_high_i32_m256i(data[4].avx2, data[5].avx2);
        let a6 = unpack_low_i32_m256i(data[6].avx2, data[7].avx2);
        let a7 = unpack_high_i32_m256i(data[6].avx2, data[7].avx2);

        pub const fn mm_shuffle(z: i32, y: i32, x: i32, w: i32) -> i32 {
          (z << 6) | (y << 4) | (x << 2) | w
        }

        const SHUFF_LO : i32 = mm_shuffle(1,0,1,0);
        const SHUFF_HI : i32 = mm_shuffle(3,2,3,2);

        // possible todo: intel performance manual suggests alternative with blend to avoid port 5 pressure
        // (since blend runs on a different port than shuffle)
        let b0 = cast::<m256,m256i>(shuffle_m256::<SHUFF_LO>(cast(a0),cast(a2)));
        let b1 = cast::<m256,m256i>(shuffle_m256::<SHUFF_HI>(cast(a0),cast(a2)));
        let b2 = cast::<m256,m256i>(shuffle_m256::<SHUFF_LO>(cast(a1),cast(a3)));
        let b3 = cast::<m256,m256i>(shuffle_m256::<SHUFF_HI>(cast(a1),cast(a3)));
        let b4 = cast::<m256,m256i>(shuffle_m256::<SHUFF_LO>(cast(a4),cast(a6)));
        let b5 = cast::<m256,m256i>(shuffle_m256::<SHUFF_HI>(cast(a4),cast(a6)));
        let b6 = cast::<m256,m256i>(shuffle_m256::<SHUFF_LO>(cast(a5),cast(a7)));
        let b7 = cast::<m256,m256i>(shuffle_m256::<SHUFF_HI>(cast(a5),cast(a7)));

        [
          i32x8 { avx2: permute2z_m256i::<0x20>(b0, b4) },
          i32x8 { avx2: permute2z_m256i::<0x20>(b1, b5) },
          i32x8 { avx2: permute2z_m256i::<0x20>(b2, b6) },
          i32x8 { avx2: permute2z_m256i::<0x20>(b3, b7) },
          i32x8 { avx2: permute2z_m256i::<0x31>(b0, b4) },
          i32x8 { avx2: permute2z_m256i::<0x31>(b1, b5) },
          i32x8 { avx2: permute2z_m256i::<0x31>(b2, b6) },
          i32x8 { avx2: permute2z_m256i::<0x31>(b3, b7) }
        ]
      } else {
        // possible todo: not sure that 128bit SIMD gives us a a lot of speedup here

        #[inline(always)]
        fn transpose_column(data: &[i32x8; 8], index: usize) -> i32x8 {
          i32x8::new([
            data[0].as_array()[index],
            data[1].as_array()[index],
            data[2].as_array()[index],
            data[3].as_array()[index],
            data[4].as_array()[index],
            data[5].as_array()[index],
            data[6].as_array()[index],
            data[7].as_array()[index],
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
        ]
      }
    }
  }
}

impl_simd_int! {
  unsafe {
    T = i32,
    N = 8,
    Simd = i32x8,
    UnsignedSimd = u32x8,
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
  fn shl(self, rhs: i32x8) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        // ensure same behavior as scalar wrapping_shl by masking the shift count
        let shift_by = bitand_m256i(rhs.avx2, set_splat_i32_m256i(31));
        // shl is the same for unsigned and signed
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
  fn shr(self, rhs: i32x8) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        // ensure same behavior as scalar
        let shift_by = bitand_m256i(rhs.avx2, set_splat_i32_m256i(31));
        Self { avx2: shr_each_i32_m256i(self.avx2, shift_by ) }
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
        Self { avx2: shr_all_i32_m256i(self.avx2, shift) }
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
    }    }
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
        Self { avx2: max_i32_m256i(self.avx2, rhs.avx2) }
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
        Self { avx2: min_i32_m256i(self.avx2, rhs.avx2) }
      } else {
        Self {
          a : self.a.min(rhs.a),
          b : self.b.min(rhs.b),
        }
      }
    }
  }

  #[inline]
  pub fn reduce_add(self) -> i32 {
    let arr: [i32x4; 2] = cast(self);
    (arr[0] + arr[1]).reduce_add()
  }

  #[inline]
  pub fn reduce_mul(self) -> i32 {
    let array: [i32x4; 2] = cast(self);
    (array[0] * array[1]).reduce_mul()
  }

  #[inline]
  pub fn reduce_max(self) -> i32 {
    let arr: [i32x4; 2] = cast(self);
    arr[0].max(arr[1]).reduce_max()
  }

  #[inline]
  pub fn reduce_min(self) -> i32 {
    let arr: [i32x4; 2] = cast(self);
    arr[0].min(arr[1]).reduce_min()
  }

  #[inline]
  pub fn saturating_add(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
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
      if #[cfg(target_feature="avx2")] {
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
    let low = cast::<u32x8, i32x8>(low);

    let overflow = high.simd_ne(low.is_negative());
    (low, overflow)
  }

  optional_fn_widening_mul {
    #[inline]
    pub fn widening_mul(self, rhs: Self) -> i64x8 {
      pick! {
        if #[cfg(all(target_feature="avx512f", target_feature="avx2"))] {
          const SHUFFLE_INDICES: __m512i = i64x8::new([0, 4, 1, 5, 2, 6, 3, 7]);

          let even_wide_mul = mul_i64_low_bits_m256i(self.avx2, rhs.avx2);
          let odd_wide_mul = mul_i64_low_bits_m256i(
            shuffle_ai_i32_half_m256i::<0b_00_11_00_01>(self.avx2),
            shuffle_ai_i32_half_m256i::<0b_00_11_00_01>(rhs.avx2),
          );
          let even_then_odd = cast::<[m256i; 2], m512i>([even_wide_mul, odd_wide_mul]);
          i64x8 {
            avx512: permute_i64_m512i(SHUFFLE_INDICES, even_then_odd),
          }
        } else if #[cfg(target_feature="avx2")] {
          let even_wide_mul = mul_i64_low_bits_m256i(self.avx2, rhs.avx2);
          let odd_wide_mul = mul_i64_low_bits_m256i(
            shuffle_ai_i32_half_m256i::<0b_00_11_00_01>(self.avx2),
            shuffle_ai_i32_half_m256i::<0b_00_11_00_01>(rhs.avx2),
          );
          let m0145 = unpack_low_i64_m256i(even_wide_mul, odd_wide_mul);
          let m2367 = unpack_high_i64_m256i(even_wide_mul, odd_wide_mul);

          cast([
            shuffle_abi_i128z_all_m256i::<0b_1010_0000>(m0156, m2367),
            shuffle_abi_i128z_all_m256i::<0b_1011_0001>(m0156, m2367),
          ])
        } else {
          let [self_a, self_b] = cast::<i32x8, [i32x4; 2]>(self);
          let [rhs_a, rhs_b] = cast::<i32x8, [i32x4; 2]>(rhs);

          cast([self_a.widening_mul(rhs_a), self_b.widening_mul(rhs_b)])
        }
      }
    }
  }

  #[inline]
  pub fn mul_keep_low_high(self, rhs: Self) -> (u32x8, i32x8) {
    pick! {
      if #[cfg(target_feature="avx2")] {
        let even_wide_mul = mul_i64_low_bits_m256i(self.avx2, rhs.avx2);
        let odd_wide_mul = mul_i64_low_bits_m256i(
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
  pub fn mul_keep_high(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        let even_wide_mul = mul_i64_low_bits_m256i(self.avx2, rhs.avx2);
        let odd_wide_mul = mul_i64_low_bits_m256i(
          shuffle_ai_i32_half_m256i::<0b_00_11_00_01>(self.avx2),
          shuffle_ai_i32_half_m256i::<0b_00_11_00_01>(rhs.avx2),
        );
        let ll_hh_1 = unpack_low_i32_m256i(even_wide_mul, odd_wide_mul);
        let ll_hh_2 = unpack_high_i32_m256i(even_wide_mul, odd_wide_mul);

        Self { avx2: unpack_high_i64_m256i(ll_hh_1, ll_hh_2) }
      } else {
        let [self_a, self_b] = cast::<i32x8, [i32x4; 2]>(self);
        let [rhs_a, rhs_b] = cast::<i32x8, [i32x4; 2]>(rhs);

        cast([self_a.mul_keep_high(rhs_a), self_b.mul_keep_high(rhs_b)])
      }
    }
  }

  #[inline]
  pub fn abs(self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self { avx2: abs_i32_m256i(self.avx2) }
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

impl From<i16x8> for i32x8 {
  #[inline]
  fn from(value: i16x8) -> Self {
    i32x8::from_i16x8(value)
  }
}

impl i32x8 {
  /// widens and sign extends to `i32x8`
  #[inline]
  #[must_use]
  pub fn from_i16x8(v: i16x8) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        i32x8 { avx2:convert_to_i32_m256i_from_i16_m128i(v.sse) }
      } else if #[cfg(target_feature="sse2")] {
        i32x8 {
          a: i32x4 { sse: shr_imm_i32_m128i::<16>( unpack_low_i16_m128i(v.sse, v.sse)) },
          b: i32x4 { sse: shr_imm_i32_m128i::<16>( unpack_high_i16_m128i(v.sse, v.sse)) },
        }
      } else {
        i32x8::new([
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

  /// widens and zero extends to `i32x8`
  #[inline]
  #[must_use]
  pub fn from_u16x8(v: u16x8) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        i32x8 { avx2:convert_to_i32_m256i_from_u16_m128i(v.sse) }
      } else if #[cfg(target_feature="sse2")] {
        i32x8 {
          a: i32x4 { sse: shr_imm_u32_m128i::<16>( unpack_low_i16_m128i(v.sse, v.sse)) },
          b: i32x4 { sse: shr_imm_u32_m128i::<16>( unpack_high_i16_m128i(v.sse, v.sse)) },
        }
      } else {
        i32x8::new([
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
  #[must_use]
  pub fn round_float(self) -> f32x8 {
    pick! {
      if #[cfg(target_feature="avx2")] {
        cast(convert_to_m256_from_i32_m256i(self.avx2))
      } else {
        cast([
          self.a.round_float(),
          self.b.round_float(),
        ])
      }
    }
  }
}
