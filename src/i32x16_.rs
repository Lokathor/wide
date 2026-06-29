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

int_uint_consts!(i32, 16, i32x16, 512);

unsafe impl Zeroable for i32x16 {}
unsafe impl Pod for i32x16 {}

impl AlignTo for i32x16 {
  type Elem = i32;
}

impl Add for i32x16 {
  type Output = Self;
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
}

impl Sub for i32x16 {
  type Output = Self;
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
}

impl Mul for i32x16 {
  type Output = Self;
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
}

integer_impl_div_rem!(
  i32,
  i32x16,
  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
);

impl Add<i32> for i32x16 {
  type Output = Self;
  #[inline]
  fn add(self, rhs: i32) -> Self::Output {
    self.add(Self::splat(rhs))
  }
}

impl Sub<i32> for i32x16 {
  type Output = Self;
  #[inline]
  fn sub(self, rhs: i32) -> Self::Output {
    self.sub(Self::splat(rhs))
  }
}

impl Mul<i32> for i32x16 {
  type Output = Self;
  #[inline]
  fn mul(self, rhs: i32) -> Self::Output {
    self.mul(Self::splat(rhs))
  }
}

impl Add<i32x16> for i32 {
  type Output = i32x16;
  #[inline]
  fn add(self, rhs: i32x16) -> Self::Output {
    i32x16::splat(self).add(rhs)
  }
}

impl Sub<i32x16> for i32 {
  type Output = i32x16;
  #[inline]
  fn sub(self, rhs: i32x16) -> Self::Output {
    i32x16::splat(self).sub(rhs)
  }
}

impl Mul<i32x16> for i32 {
  type Output = i32x16;
  #[inline]
  fn mul(self, rhs: i32x16) -> Self::Output {
    i32x16::splat(self).mul(rhs)
  }
}

impl BitAnd for i32x16 {
  type Output = Self;
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
}

impl BitOr for i32x16 {
  type Output = Self;
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
}

impl BitXor for i32x16 {
  type Output = Self;
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
}

impl Shl for i32x16 {
  type Output = Self;

  /// Shifts lanes by the corresponding lane.
  ///
  /// Bitwise shift-left; yields `self << mask(rhs)`, where mask removes any
  /// high-order bits of `rhs` that would cause the shift to exceed the bitwidth
  /// of the type. (same as `wrapping_shl`)
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
}

impl Shr for i32x16 {
  type Output = Self;

  /// Shifts lanes by the corresponding lane.
  ///
  /// Bitwise shift-right; yields `self >> mask(rhs)`, where mask removes any
  /// high-order bits of `rhs` that would cause the shift to exceed the bitwidth
  /// of the type. (same as `wrapping_shr`)
  #[inline]
  fn shr(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        #[cfg(target_arch = "x86")]
        use core::arch::x86::_mm512_srav_epi32;
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::_mm512_srav_epi32;

        // Mask `rhs` to 31 to match `wrapping_shr`.
        let rhs = bitand_m512i(rhs.avx512, set_splat_i16_m512i(31));
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
}

macro_rules! impl_shl_t_for_i32x16 {
  ($($shift_type:ty),+ $(,)?) => {
    $(impl Shl<$shift_type> for i32x16 {
      type Output = Self;
      /// Shifts all lanes by the value given.
      #[inline]
      fn shl(self, rhs: $shift_type) -> Self::Output {
        pick! {
          if #[cfg(target_feature="avx512f")] {
            let shift = cast(rhs as u32);
            Self { avx512: shl_all_u32_m512i(self.avx512, shift) }
          } else {
            Self {
              a : self.a.shl(rhs),
              b : self.b.shl(rhs),
            }
          }
        }
      }
    })+
  };
}
impl_shl_t_for_i32x16!(i8, u8, i16, u16, i32, u32, i64, u64, i128, u128);

macro_rules! impl_shr_t_for_i32x16 {
  ($($shift_type:ty),+ $(,)?) => {
    $(impl Shr<$shift_type> for i32x16 {
      type Output = Self;
      /// Shifts all lanes by the value given.
      #[inline]
      fn shr(self, rhs: $shift_type) -> Self::Output {
        pick! {
          if #[cfg(target_feature="avx512f")] {
            let shift = cast(rhs as u32);
            Self { avx512: shr_all_i32_m512i(self.avx512, shift) }
          } else {
            Self {
              a : self.a.shr(rhs),
              b : self.b.shr(rhs),
            }
          }
        }
      }
    })+
  };
}
impl_shr_t_for_i32x16!(i8, u8, i16, u16, i32, u32, i64, u64, i128, u128);

#[expect(deprecated)]
impl CmpEq for i32x16 {
  type Output = Self;
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
}

#[expect(deprecated)]
impl CmpLt for i32x16 {
  type Output = Self;
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
}

#[expect(deprecated)]
impl CmpGt for i32x16 {
  type Output = Self;
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
}

#[expect(deprecated)]
impl CmpNe for i32x16 {
  type Output = Self;
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
}

#[expect(deprecated)]
impl CmpLe for i32x16 {
  type Output = Self;
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
}

#[expect(deprecated)]
impl CmpGe for i32x16 {
  type Output = Self;
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
}

impl i32x16 {
  #[inline]
  #[must_use]
  pub const fn new(array: [i32; 16]) -> Self {
    unsafe { core::mem::transmute(array) }
  }

  simd_comparison_fns!();

  #[inline]
  #[must_use]
  pub fn blend(self, t: Self, f: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: blend_varying_i8_m512i(f.avx512,t.avx512,movepi8_mask_m512i(self.avx512)) }
      } else {
        Self {
          a : self.a.blend(t.a, f.a),
          b : self.b.blend(t.b, f.b),
        }
      }
    }
  }

  /// Returns true for each positive element and false if it is zero or
  /// negative.
  #[inline]
  #[must_use]
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

  /// Returns true for each negative element and false if it is zero or
  /// positive.
  #[inline]
  #[must_use]
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

  #[inline]
  #[must_use]
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
  #[must_use]
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

  integer_fn_clamp!();

  #[inline]
  #[must_use]
  pub fn saturating_add(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        let result = self + rhs;
        let overflow = (!(self ^ rhs) & (self ^ result)).is_negative();
        let negative = self.is_negative();

        overflow.blend(negative.blend(Self::MIN, Self::MAX), result)
      } else {
        Self {
          a: self.a.saturating_add(rhs.a),
          b: self.b.saturating_add(rhs.b),
        }
      }
    }
  }

  #[inline]
  #[must_use]
  pub fn saturating_sub(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        let result = self - rhs;
        let overflow = ((self ^ rhs) & (self ^ result)).is_negative();
        let negative = self.is_negative();

        overflow.blend(negative.blend(Self::MIN, Self::MAX), result)
      } else {
        Self {
          a: self.a.saturating_sub(rhs.a),
          b: self.b.saturating_sub(rhs.b),
        }
      }
    }
  }

  /// Lanewise saturating multiply.
  #[inline]
  #[must_use]
  pub fn saturating_mul(self, rhs: Self) -> Self {
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
        let low = Self {
          avx512: m512i(unsafe { _mm512_unpacklo_epi64(ll_hh_1.0, ll_hh_2.0) }),
        };
        let high = Self {
          avx512: m512i(unsafe { _mm512_unpackhi_epi64(ll_hh_1.0, ll_hh_2.0) }),
        };

        let no_overflow = high.simd_eq(low.is_negative());
        let limit = Self::MAX ^ (self ^ rhs).is_negative();
        no_overflow.blend(low, limit)
      } else {
        let [self_a, self_b]: [i32x8; 2] = cast(self);
        let [rhs_a, rhs_b]: [i32x8; 2] = cast(rhs);

        cast([self_a.saturating_mul(rhs_a), self_b.saturating_mul(rhs_b)])
      }
    }
  }

  integer_fn_saturating_div!([
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
  ]);

  signed_fn_overflowing_add_sub!();

  /// horizontal add of all the elements of the vector
  #[inline]
  #[must_use]
  pub fn reduce_add(self) -> i32 {
    let arr: [i32x8; 2] = cast(self);
    (arr[0] + arr[1]).reduce_add()
  }

  /// Reducing multiply. Returns the product of the elements of the vector.
  #[inline]
  #[must_use]
  pub fn reduce_mul(self) -> i32 {
    let array: [i32x8; 2] = cast(self);
    (array[0] * array[1]).reduce_mul()
  }

  /// horizontal min of all the elements of the vector
  #[inline]
  #[must_use]
  pub fn reduce_min(self) -> i32 {
    let arr: [i32x8; 2] = cast(self);
    arr[0].min(arr[1]).reduce_min()
  }

  /// horizontal max of all the elements of the vector
  #[inline]
  #[must_use]
  pub fn reduce_max(self) -> i32 {
    let arr: [i32x8; 2] = cast(self);
    arr[0].max(arr[1]).reduce_max()
  }

  #[inline]
  #[must_use]
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
  #[must_use]
  pub fn unsigned_abs(self) -> u32x16 {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
        u32x16 { avx512: abs_i32_m512i(self.avx512) }
      } else {
        u32x16 {
          a : self.a.unsigned_abs(),
          b : self.b.unsigned_abs(),
        }
      }
    }
  }

  signed_fn_signum!();

  #[inline]
  #[must_use]
  #[doc(alias("movemask", "move_mask"))]
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
  #[must_use]
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
  #[must_use]
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

  #[inline]
  #[must_use]
  pub fn none(self) -> bool {
    !self.any()
  }

  /// Transpose matrix of 16x16 `i32` matrix. Currently not accelerated.
  #[must_use]
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

  #[inline]
  pub fn to_array(self) -> [i32; 16] {
    cast(self)
  }

  #[inline]
  pub fn as_array(&self) -> &[i32; 16] {
    cast_ref(self)
  }

  #[inline]
  pub fn as_mut_array(&mut self) -> &mut [i32; 16] {
    cast_mut(self)
  }

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

impl Not for i32x16 {
  type Output = Self;
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
}
