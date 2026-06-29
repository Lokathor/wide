use super::*;

pick! {
  if #[cfg(target_feature="avx512bw")] {
    #[derive(Default, Clone, Copy, PartialEq, Eq)]
    #[repr(C, align(64))]
    pub struct i16x32 { pub(crate) avx512: m512i }
  } else {
    #[derive(Default, Clone, Copy, PartialEq, Eq)]
    #[repr(C, align(64))]
    pub struct i16x32 { pub(crate) a : i16x16, pub(crate) b : i16x16 }
  }
}

int_uint_consts!(i16, 32, i16x32, 512);

unsafe impl Zeroable for i16x32 {}
unsafe impl Pod for i16x32 {}

impl AlignTo for i16x32 {
  type Elem = i16;
}

impl Add for i16x32 {
  type Output = Self;
  #[inline]
  fn add(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
        Self { avx512: add_i16_m512i(self.avx512, rhs.avx512) }
      } else {
        Self {
          a : self.a.add(rhs.a),
          b : self.b.add(rhs.b),
        }
      }
    }
  }
}

impl Sub for i16x32 {
  type Output = Self;
  #[inline]
  fn sub(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
        Self { avx512: sub_i16_m512i(self.avx512, rhs.avx512) }
      } else {
        Self {
          a : self.a.sub(rhs.a),
          b : self.b.sub(rhs.b),
        }
      }
    }
  }
}

impl Mul for i16x32 {
  type Output = Self;
  #[inline]
  fn mul(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
        Self { avx512: mul_i16_keep_low_m512i(self.avx512, rhs.avx512) }
      } else {
        Self { a: self.a.mul(rhs.a), b: self.b.mul(rhs.b) }
      }
    }
  }
}

integer_impl_div_rem!(
  i16,
  i16x32,
  [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
  ],
);

impl Shl for i16x32 {
  type Output = Self;

  /// Shifts lanes by the corresponding lane.
  ///
  /// Bitwise shift-left; yields `self << mask(rhs)`, where mask removes any
  /// high-order bits of `rhs` that would cause the shift to exceed the bitwidth
  /// of the type. (same as `wrapping_shl`)
  #[inline]
  fn shl(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
        // Mask `rhs` to 15 to match `wrapping_shl`.
        let rhs = bitand_m512i(rhs.avx512, set_splat_i16_m512i(15));
        Self { avx512: shl_each_u16_m512i(self.avx512, rhs) }
      } else {
        let [self_a, self_b]: [i16x16; 2] = cast(self);
        let [rhs_a, rhs_b]: [i16x16; 2] = cast(rhs);

        cast([self_a << rhs_a, self_b << rhs_b])
      }
    }
  }
}

impl Shr for i16x32 {
  type Output = Self;

  /// Shifts lanes by the corresponding lane.
  ///
  /// Bitwise shift-right; yields `self >> mask(rhs)`, where mask removes any
  /// high-order bits of `rhs` that would cause the shift to exceed the bitwidth
  /// of the type. (same as `wrapping_shr`)
  #[inline]
  fn shr(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
        #[cfg(target_arch = "x86")]
        use core::arch::x86::_mm512_srav_epi16;
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::_mm512_srav_epi16;

        // Mask `rhs` to 15 to match `wrapping_shr`.
        let rhs = bitand_m512i(rhs.avx512, set_splat_i16_m512i(15));
        // TODO(safe_arch): Add `_mm512_srav_epi16`.
        Self { avx512: m512i(unsafe { _mm512_srav_epi16(self.avx512.0, rhs.0) }) }
      } else {
        let [self_a, self_b]: [i16x16; 2] = cast(self);
        let [rhs_a, rhs_b]: [i16x16; 2] = cast(rhs);

        cast([self_a >> rhs_a, self_b >> rhs_b])
      }
    }
  }
}

impl Add<i16> for i16x32 {
  type Output = Self;
  #[inline]
  fn add(self, rhs: i16) -> Self::Output {
    self.add(Self::splat(rhs))
  }
}

impl Sub<i16> for i16x32 {
  type Output = Self;
  #[inline]
  fn sub(self, rhs: i16) -> Self::Output {
    self.sub(Self::splat(rhs))
  }
}

impl Mul<i16> for i16x32 {
  type Output = Self;
  #[inline]
  fn mul(self, rhs: i16) -> Self::Output {
    self.mul(Self::splat(rhs))
  }
}

impl Add<i16x32> for i16 {
  type Output = i16x32;
  #[inline]
  fn add(self, rhs: i16x32) -> Self::Output {
    i16x32::splat(self).add(rhs)
  }
}

impl Sub<i16x32> for i16 {
  type Output = i16x32;
  #[inline]
  fn sub(self, rhs: i16x32) -> Self::Output {
    i16x32::splat(self).sub(rhs)
  }
}

impl Mul<i16x32> for i16 {
  type Output = i16x32;
  #[inline]
  fn mul(self, rhs: i16x32) -> Self::Output {
    i16x32::splat(self).mul(rhs)
  }
}

impl BitAnd for i16x32 {
  type Output = Self;
  #[inline]
  fn bitand(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
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

impl BitOr for i16x32 {
  type Output = Self;
  #[inline]
  fn bitor(self, rhs: Self) -> Self::Output {
    pick! {
    if #[cfg(target_feature="avx512bw")] {
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

impl BitXor for i16x32 {
  type Output = Self;
  #[inline]
  fn bitxor(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
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

macro_rules! impl_shl_t_for_i16x32 {
  ($($shift_type:ty),+ $(,)?) => {
    $(impl Shl<$shift_type> for i16x32 {
      type Output = Self;
      /// Shifts all lanes by the value given.
      #[inline]
      fn shl(self, rhs: $shift_type) -> Self::Output {
        pick! {
          if #[cfg(target_feature="avx512bw")] {
            let shift = cast(rhs as u16);
            Self { avx512: shl_all_u16_m512i(self.avx512, shift) }
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
impl_shl_t_for_i16x32!(i8, u8, i16, u16, i32, u32, i64, u64, i128, u128);

macro_rules! impl_shr_t_for_i16x32 {
  ($($shift_type:ty),+ $(,)?) => {
    $(impl Shr<$shift_type> for i16x32 {
      type Output = Self;
      /// Shifts all lanes by the value given.
      #[inline]
      fn shr(self, rhs: $shift_type) -> Self::Output {
        pick! {
          if #[cfg(target_feature="avx512bw")] {
            let shift = cast(rhs as u16);
            Self { avx512: shr_all_i16_m512i(self.avx512, shift) }
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
impl_shr_t_for_i16x32!(i8, u8, i16, u16, i32, u32, i64, u64, i128, u128);

#[expect(deprecated)]
impl CmpEq for i16x32 {
  type Output = Self;
  #[inline]
  fn simd_eq(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
        Self { avx512: cmp_op_mask_i16_m512i::<{cmp_int_op!(Eq)}>(self.avx512, rhs.avx512) }
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
impl CmpLt for i16x32 {
  type Output = Self;
  #[inline]
  fn simd_lt(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
        Self { avx512: cmp_op_mask_i16_m512i::<{cmp_int_op!(Lt)}>(self.avx512, rhs.avx512) }
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
impl CmpGt for i16x32 {
  type Output = Self;
  #[inline]
  fn simd_gt(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
        Self { avx512: cmp_op_mask_i16_m512i::<{cmp_int_op!(Nle)}>(self.avx512, rhs.avx512) }
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
impl CmpNe for i16x32 {
  type Output = Self;
  #[inline]
  fn simd_ne(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
        Self { avx512: cmp_op_mask_i16_m512i::<{cmp_int_op!(Ne)}>(self.avx512, rhs.avx512) }
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
impl CmpLe for i16x32 {
  type Output = Self;
  #[inline]
  fn simd_le(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
        Self { avx512: cmp_op_mask_i16_m512i::<{cmp_int_op!(Le)}>(self.avx512, rhs.avx512) }
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
impl CmpGe for i16x32 {
  type Output = Self;
  #[inline]
  fn simd_ge(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
        Self { avx512: cmp_op_mask_i16_m512i::<{cmp_int_op!(Nlt)}>(self.avx512, rhs.avx512) }
      } else {
        Self {
          a : self.a.simd_ge(rhs.a),
          b : self.b.simd_ge(rhs.b),
        }
      }
    }
  }
}

impl i16x32 {
  #[inline]
  #[must_use]
  pub const fn new(array: [i16; 32]) -> Self {
    unsafe { core::mem::transmute(array) }
  }

  simd_comparison_fns!();

  #[inline]
  #[must_use]
  pub fn blend(self, t: Self, f: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
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

  /// horizontal add of all the elements of the vector
  #[inline]
  #[must_use]
  pub fn reduce_add(self) -> i16 {
    let arr: [i16x16; 2] = cast(self);
    (arr[0] + arr[1]).reduce_add()
  }

  /// Reducing multiply. Returns the product of the elements of the vector.
  #[inline]
  #[must_use]
  pub fn reduce_mul(self) -> i16 {
    let array: [i16x16; 2] = cast(self);
    (array[0] * array[1]).reduce_mul()
  }

  /// horizontal min of all the elements of the vector
  #[inline]
  #[must_use]
  pub fn reduce_min(self) -> i16 {
    let arr: [i16x16; 2] = cast(self);
    arr[0].min(arr[1]).reduce_min()
  }

  /// horizontal max of all the elements of the vector
  #[inline]
  #[must_use]
  pub fn reduce_max(self) -> i16 {
    let arr: [i16x16; 2] = cast(self);
    arr[0].max(arr[1]).reduce_max()
  }

  #[inline]
  #[must_use]
  pub fn abs(self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
        Self { avx512: abs_i16_m512i(self.avx512) }
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
  pub fn unsigned_abs(self) -> u16x32 {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
        u16x32 { avx512: abs_i16_m512i(self.avx512) }
      } else {
        u16x32 {
          a: self.a.unsigned_abs(),
          b: self.b.unsigned_abs(),
        }
      }
    }
  }

  signed_fn_signum!();

  #[inline]
  #[must_use]
  pub fn min(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
        Self { avx512: min_i16_m512i(self.avx512, rhs.avx512) }
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
      if #[cfg(target_feature="avx512bw")] {
        Self { avx512: max_i16_m512i(self.avx512, rhs.avx512) }
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
      if #[cfg(target_feature="avx512bw")] {
        Self { avx512: add_saturating_i16_m512i(self.avx512, rhs.avx512) }
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
      if #[cfg(target_feature="avx512bw")] {
        Self { avx512: sub_saturating_i16_m512i(self.avx512, rhs.avx512) }
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
    let [self_a, self_b]: [i16x16; 2] = cast(self);
    let [rhs_a, rhs_b]: [i16x16; 2] = cast(rhs);
    cast([self_a.saturating_mul(rhs_a), self_b.saturating_mul(rhs_b)])
  }

  integer_fn_saturating_div!([
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
  ]);

  signed_fn_overflowing_add_sub!();

  /// Calculates partial dot product.
  /// Multiplies packed signed 16-bit integers, producing intermediate signed
  /// 32-bit integers. Horizontally add adjacent pairs of intermediate 32-bit
  /// integers.
  #[inline]
  #[must_use]
  pub fn dot(self, rhs: Self) -> i32x16 {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
        i32x16 { avx512: mul_i16_horizontal_add_m512i(self.avx512, rhs.avx512) }
      } else {
        i32x16 {
          a : self.a.dot(rhs.a),
          b : self.b.dot(rhs.b),
        }
      }
    }
  }

  #[inline]
  #[must_use]
  #[doc(alias("movemask", "move_mask"))]
  pub fn to_bitmask(self) -> u32 {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
        // use f16 move_mask since it is the same size as i16
        movepi16_mask_m512i(self.avx512) as u32
      } else {
        self.a.to_bitmask() | (self.b.to_bitmask() << 16)
      }
    }
  }

  #[inline]
  #[must_use]
  pub fn any(self) -> bool {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
        movepi16_mask_m512i(self.avx512) != 0
      } else {
        (self.a | self.b).any()
      }
    }
  }

  #[inline]
  #[must_use]
  pub fn all(self) -> bool {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
        movepi16_mask_m512i(self.avx512) == 0xFFFFFFFF
      } else {
        (self.a & self.b).all()
      }
    }
  }

  #[inline]
  #[must_use]
  pub fn none(self) -> bool {
    !self.any()
  }

  /// Transpose matrix of 32x32 `i16` matrix. Currently not accelerated.
  #[must_use]
  #[inline]
  pub fn transpose(data: [i16x32; 32]) -> [i16x32; 32] {
    // Can this be optimized?

    #[inline(always)]
    fn transpose_column(data: &[i16x32; 32], index: usize) -> i16x32 {
      i16x32::new([
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
        data[16].as_array()[index],
        data[17].as_array()[index],
        data[18].as_array()[index],
        data[19].as_array()[index],
        data[20].as_array()[index],
        data[21].as_array()[index],
        data[22].as_array()[index],
        data[23].as_array()[index],
        data[24].as_array()[index],
        data[25].as_array()[index],
        data[26].as_array()[index],
        data[27].as_array()[index],
        data[28].as_array()[index],
        data[29].as_array()[index],
        data[30].as_array()[index],
        data[31].as_array()[index],
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
      transpose_column(&data, 16),
      transpose_column(&data, 17),
      transpose_column(&data, 18),
      transpose_column(&data, 19),
      transpose_column(&data, 20),
      transpose_column(&data, 21),
      transpose_column(&data, 22),
      transpose_column(&data, 23),
      transpose_column(&data, 24),
      transpose_column(&data, 25),
      transpose_column(&data, 26),
      transpose_column(&data, 27),
      transpose_column(&data, 28),
      transpose_column(&data, 29),
      transpose_column(&data, 30),
      transpose_column(&data, 31),
    ]
  }

  #[inline]
  pub fn to_array(self) -> [i16; 32] {
    cast(self)
  }

  #[inline]
  pub fn as_array(&self) -> &[i16; 32] {
    cast_ref(self)
  }

  #[inline]
  pub fn as_mut_array(&mut self) -> &mut [i16; 32] {
    cast_mut(self)
  }
}

impl Not for i16x32 {
  type Output = Self;
  #[inline]
  fn not(self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
        Self { avx512: bitxor_m512i(self.avx512, set_splat_i16_m512i(-1)) }
      } else {
        Self {
          a : self.a.not(),
          b : self.b.not(),
        }
      }
    }
  }
}
