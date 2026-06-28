use super::*;

pick! {
  if #[cfg(target_feature="avx2")] {
    #[derive(Default, Clone, Copy, PartialEq, Eq)]
    #[repr(C, align(32))]
    pub struct u16x16 { pub(crate) avx2: m256i }
  } else {
    #[derive(Default, Clone, Copy, PartialEq, Eq)]
    #[repr(C, align(32))]
    pub struct u16x16 { pub(crate) a : u16x8, pub(crate) b : u16x8 }
  }
}

int_uint_consts!(u16, 16, u16x16, 256);

unsafe impl Zeroable for u16x16 {}
unsafe impl Pod for u16x16 {}

impl AlignTo for u16x16 {
  type Elem = u16;
}

impl Add for u16x16 {
  type Output = Self;
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
}

impl Sub for u16x16 {
  type Output = Self;
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
}

impl Shl for u16x16 {
  type Output = Self;

  /// Shifts lanes by the corresponding lane.
  ///
  /// Bitwise shift-left; yields `self << mask(rhs)`, where mask removes any
  /// high-order bits of `rhs` that would cause the shift to exceed the bitwidth
  /// of the type. (same as `wrapping_shl`)
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
        let [self_a, self_b]: [u16x8; 2] = cast(self);
        let [rhs_a, rhs_b]: [u16x8; 2] = cast(rhs);

        cast([self_a << rhs_a, self_b << rhs_b])
      }
    }
  }
}

impl Shr for u16x16 {
  type Output = Self;

  /// Shifts lanes by the corresponding lane.
  ///
  /// Bitwise shift-right; yields `self >> mask(rhs)`, where mask removes any
  /// high-order bits of `rhs` that would cause the shift to exceed the bitwidth
  /// of the type. (same as `wrapping_shr`)
  #[inline]
  fn shr(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(all(target_feature="avx512bw", target_feature="avx512vl"))] {
        #[cfg(target_arch = "x86")]
        use core::arch::x86::_mm256_srlv_epi16;
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::_mm256_srlv_epi16;

        // Mask `rhs` to 15 to match `wrapping_shr`.
        let rhs = bitand_m256i(rhs.avx2, set_splat_i16_m256i(15));
        // TODO(safe_arch): Add `_mm256_srlv_epi16`.
        cast(unsafe { _mm256_srlv_epi16(self.avx2.0, rhs.0) })
      } else {
        let [self_a, self_b]: [u16x8; 2] = cast(self);
        let [rhs_a, rhs_b]: [u16x8; 2] = cast(rhs);

        cast([self_a >> rhs_a, self_b >> rhs_b])
      }
    }
  }
}

impl Add<u16> for u16x16 {
  type Output = Self;
  #[inline]
  fn add(self, rhs: u16) -> Self::Output {
    self.add(Self::splat(rhs))
  }
}

impl Sub<u16> for u16x16 {
  type Output = Self;
  #[inline]
  fn sub(self, rhs: u16) -> Self::Output {
    self.sub(Self::splat(rhs))
  }
}

impl Mul<u16> for u16x16 {
  type Output = Self;

  #[inline]
  fn mul(self, rhs: u16) -> Self::Output {
    self * Self::splat(rhs)
  }
}

impl Add<u16x16> for u16 {
  type Output = u16x16;
  #[inline]
  fn add(self, rhs: u16x16) -> Self::Output {
    u16x16::splat(self).add(rhs)
  }
}

impl Sub<u16x16> for u16 {
  type Output = u16x16;
  #[inline]
  fn sub(self, rhs: u16x16) -> Self::Output {
    u16x16::splat(self).sub(rhs)
  }
}

impl Mul<u16x16> for u16 {
  type Output = u16x16;

  #[inline]
  fn mul(self, rhs: u16x16) -> Self::Output {
    u16x16::splat(self) * rhs
  }
}

impl BitAnd for u16x16 {
  type Output = Self;
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
}

impl BitOr for u16x16 {
  type Output = Self;
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
}

impl BitXor for u16x16 {
  type Output = Self;
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
}

impl Not for u16x16 {
  type Output = Self;
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
}

macro_rules! impl_shl_t_for_u16x16 {
  ($($shift_type:ty),+ $(,)?) => {
    $(impl Shl<$shift_type> for u16x16 {
      type Output = Self;
      /// Shifts all lanes by the value given.
      #[inline]
      fn shl(self, rhs: $shift_type) -> Self::Output {
        pick! {
          if #[cfg(target_feature="avx2")] {
            let shift = cast([rhs as u64, 0]);
            Self { avx2: shl_all_u16_m256i(self.avx2, shift) }
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
impl_shl_t_for_u16x16!(i8, u8, i16, u16, i32, u32, i64, u64, i128, u128);

macro_rules! impl_shr_t_for_u16x16 {
  ($($shift_type:ty),+ $(,)?) => {
    $(impl Shr<$shift_type> for u16x16 {
      type Output = Self;
      /// Shifts all lanes by the value given.
      #[inline]
      fn shr(self, rhs: $shift_type) -> Self::Output {
        pick! {
          if #[cfg(target_feature="avx2")] {
            let shift = cast([rhs as u64, 0]);
            Self { avx2: shr_all_u16_m256i(self.avx2, shift) }
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
impl_shr_t_for_u16x16!(i8, u8, i16, u16, i32, u32, i64, u64, i128, u128);

#[expect(deprecated)]
impl CmpEq for u16x16 {
  type Output = Self;
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
}

#[expect(deprecated)]
impl CmpGt for u16x16 {
  type Output = Self;
  #[inline]
  fn simd_gt(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature = "avx2")] {
        let bias = m256i::from([0x8000u16; 16]);
        let a_biased = sub_i16_m256i(self.avx2, bias);
        let b_biased = sub_i16_m256i(rhs.avx2, bias);
        let mask = cmp_gt_mask_i16_m256i(a_biased, b_biased);

        Self { avx2: mask }
      } else {
        Self {
          a: self.a.simd_gt(rhs.a),
          b: self.b.simd_gt(rhs.b),
        }
      }
    }
  }
}

#[expect(deprecated)]
impl CmpLt for u16x16 {
  type Output = Self;
  #[inline]
  fn simd_lt(self, rhs: Self) -> Self::Output {
    // no gt, so just reverse to get same answer
    Self::simd_gt(rhs, self)
  }
}

#[expect(deprecated)]
impl CmpNe for u16x16 {
  type Output = Self;
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
}

#[expect(deprecated)]
impl CmpLe for u16x16 {
  type Output = Self;
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
}

#[expect(deprecated)]
impl CmpGe for u16x16 {
  type Output = Self;
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
}

impl Mul for u16x16 {
  type Output = Self;
  #[inline]
  fn mul(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        // non-widening multiplication is the same for unsigned and signed
        Self { avx2: mul_i16_keep_low_m256i(self.avx2, rhs.avx2) }
      } else {
        Self {
          a : self.a.mul(rhs.a),
          b : self.b.mul(rhs.b),
        }
      }
    }
  }
}

integer_impl_div_rem!(
  u16,
  u16x16,
  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
);

impl From<u8x16> for u16x16 {
  /// widens and sign extends to u16x16
  #[inline]
  fn from(v: u8x16) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        u16x16 { avx2:convert_to_i16_m256i_from_u8_m128i(v.sse) }
      } else if #[cfg(target_feature="sse2")] {
        u16x16 {
          a: u16x8 { sse: shr_imm_u16_m128i::<8>( unpack_low_i8_m128i(v.sse, v.sse)) },
          b: u16x8 { sse: shr_imm_u16_m128i::<8>( unpack_high_i8_m128i(v.sse, v.sse)) },
        }
      } else {

        u16x16::new([
          v.as_array()[0] as u16,
          v.as_array()[1] as u16,
          v.as_array()[2] as u16,
          v.as_array()[3] as u16,
          v.as_array()[4] as u16,
          v.as_array()[5] as u16,
          v.as_array()[6] as u16,
          v.as_array()[7] as u16,
          v.as_array()[8] as u16,
          v.as_array()[9] as u16,
          v.as_array()[10] as u16,
          v.as_array()[11] as u16,
          v.as_array()[12] as u16,
          v.as_array()[13] as u16,
          v.as_array()[14] as u16,
          v.as_array()[15] as u16,
          ])
      }
    }
  }
}

impl u16x16 {
  #[inline]
  #[must_use]
  pub const fn new(array: [u16; 16]) -> Self {
    unsafe { core::mem::transmute(array) }
  }

  simd_comparison_fns!();

  /// Bitwise selection.
  ///
  /// For each bit this returns `t` where `self` is `1` and `f` where `self` is
  /// `0`.
  ///
  /// If `self` is a mask, meaning each lane is either all zeros or all ones,
  /// consider using [`select`] which is faster.
  ///
  /// [`select`]: Self::select
  #[inline]
  #[must_use]
  pub fn bitselect(self, t: Self, f: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self {
          avx2: bitor_m256i(
            bitand_m256i(t.avx2, self.avx2),
            bitandnot_m256i(self.avx2, f.avx2),
          ),
        }
      } else {
        Self {
          a: self.a.bitselect(t.a, f.a),
          b: self.b.bitselect(t.b, f.b),
        }
      }
    }
  }

  /// Lanewise selection.
  ///
  /// For each lane this returns `t` where `self` is all ones and `f` where
  /// `self` is all zeros.
  ///
  /// This function assumes `self` is a mask, meaning each lane is either all
  /// zeros or all ones. For bitwise selection use [`bitselect`].
  ///
  /// [`bitselect`]: Self::bitselect
  #[inline]
  #[must_use]
  pub fn select(self, t: Self, f: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self { avx2: blend_varying_i8_m256i(f.avx2, t.avx2, self.avx2) }
      } else {
        Self {
          a : self.a.select(t.a, f.a),
          b : self.b.select(t.b, f.b),
        }
      }
    }
  }

  #[inline]
  #[must_use]
  pub fn reduce_add(self) -> u16 {
    cast(i16x16::reduce_add(cast(self)))
  }

  /// Reducing multiply. Returns the product of the elements of the vector.
  #[inline]
  #[must_use]
  pub fn reduce_mul(self) -> u16 {
    let array: [u16x8; 2] = cast(self);
    (array[0] * array[1]).reduce_mul()
  }

  #[inline]
  #[must_use]
  pub fn reduce_max(self) -> u16 {
    let array: [u16x8; 2] = cast(self);
    array[0].max(array[1]).reduce_max()
  }

  #[inline]
  #[must_use]
  pub fn reduce_min(self) -> u16 {
    let array: [u16x8; 2] = cast(self);
    array[0].min(array[1]).reduce_min()
  }

  #[inline]
  #[must_use]
  pub fn max(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self { avx2: max_u16_m256i(self.avx2, rhs.avx2) }
      } else {
        Self {
          a : self.a.max(rhs.a),
          b : self.b.max(rhs.b),
        }
      }
    }
  }
  #[inline]
  #[must_use]
  pub fn min(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self { avx2: min_u16_m256i(self.avx2, rhs.avx2) }
      } else {
        Self {
          a : self.a.min(rhs.a),
          b : self.b.min(rhs.b),
        }
      }
    }
  }

  integer_fn_clamp!();

  #[inline]
  #[must_use]
  pub fn saturating_add(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self { avx2: add_saturating_u16_m256i(self.avx2, rhs.avx2) }
      } else {
        Self {
          a : self.a.saturating_add(rhs.a),
          b : self.b.saturating_add(rhs.b),
        }
      }
    }
  }
  #[inline]
  #[must_use]
  pub fn saturating_sub(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self { avx2: sub_saturating_u16_m256i(self.avx2, rhs.avx2) }
      } else {
        Self {
          a : self.a.saturating_sub(rhs.a),
          b : self.b.saturating_sub(rhs.b),
        }
      }
    }
  }

  /// Lanewise saturating multiply.
  #[inline]
  #[must_use]
  pub fn saturating_mul(self, rhs: Self) -> Self {
    let [self_a, self_b]: [u16x8; 2] = cast(self);
    let [rhs_a, rhs_b]: [u16x8; 2] = cast(rhs);
    cast([self_a.saturating_mul(rhs_a), self_b.saturating_mul(rhs_b)])
  }

  integer_fn_saturating_div!([
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
  ]);

  #[inline]
  #[must_use]
  #[doc(alias("movemask", "move_mask"))]
  pub fn to_bitmask(self) -> u32 {
    i16x16::to_bitmask(cast(self))
  }

  #[inline]
  #[must_use]
  pub fn any(self) -> bool {
    i16x16::any(cast(self))
  }

  #[inline]
  #[must_use]
  pub fn all(self) -> bool {
    i16x16::all(cast(self))
  }

  #[inline]
  #[must_use]
  pub fn none(self) -> bool {
    !self.any()
  }

  /// Transpose matrix of 16x16 `u16` matrix. Currently not accelerated.
  #[must_use]
  #[inline]
  pub fn transpose(data: [u16x16; 16]) -> [u16x16; 16] {
    cast(i16x16::transpose(cast(data)))
  }

  #[inline]
  pub fn to_array(self) -> [u16; 16] {
    cast(self)
  }

  #[inline]
  pub fn as_array(&self) -> &[u16; 16] {
    cast_ref(self)
  }

  #[inline]
  pub fn as_mut_array(&mut self) -> &mut [u16; 16] {
    cast_mut(self)
  }
}
