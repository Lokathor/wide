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

int_uint_consts!(i16, 16, i16x16, i16x16, i16a16, const_i16_as_i16x16, 256);

unsafe impl Zeroable for i16x16 {}
unsafe impl Pod for i16x16 {}

impl Add for i16x16 {
  type Output = Self;
  #[inline]
  #[must_use]
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

impl Sub for i16x16 {
  type Output = Self;
  #[inline]
  #[must_use]
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

impl Mul for i16x16 {
  type Output = Self;
  #[inline]
  #[must_use]
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
}

impl Add<i16> for i16x16 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn add(self, rhs: i16) -> Self::Output {
    self.add(Self::splat(rhs))
  }
}

impl Sub<i16> for i16x16 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn sub(self, rhs: i16) -> Self::Output {
    self.sub(Self::splat(rhs))
  }
}

impl Mul<i16> for i16x16 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn mul(self, rhs: i16) -> Self::Output {
    self.mul(Self::splat(rhs))
  }
}

impl Add<i16x16> for i16 {
  type Output = i16x16;
  #[inline]
  #[must_use]
  fn add(self, rhs: i16x16) -> Self::Output {
    i16x16::splat(self).add(rhs)
  }
}

impl Sub<i16x16> for i16 {
  type Output = i16x16;
  #[inline]
  #[must_use]
  fn sub(self, rhs: i16x16) -> Self::Output {
    i16x16::splat(self).sub(rhs)
  }
}

impl Mul<i16x16> for i16 {
  type Output = i16x16;
  #[inline]
  #[must_use]
  fn mul(self, rhs: i16x16) -> Self::Output {
    i16x16::splat(self).mul(rhs)
  }
}

impl BitAnd for i16x16 {
  type Output = Self;
  #[inline]
  #[must_use]
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

impl BitOr for i16x16 {
  type Output = Self;
  #[inline]
  #[must_use]
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

impl BitXor for i16x16 {
  type Output = Self;
  #[inline]
  #[must_use]
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

macro_rules! impl_shl_t_for_i16x16 {
  ($($shift_type:ty),+ $(,)?) => {
    $(impl Shl<$shift_type> for i16x16 {
      type Output = Self;
      /// Shifts all lanes by the value given.
      #[inline]
      #[must_use]
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
impl_shl_t_for_i16x16!(i8, u8, i16, u16, i32, u32, i64, u64, i128, u128);

macro_rules! impl_shr_t_for_i16x16 {
  ($($shift_type:ty),+ $(,)?) => {
    $(impl Shr<$shift_type> for i16x16 {
      type Output = Self;
      /// Shifts all lanes by the value given.
      #[inline]
      #[must_use]
      fn shr(self, rhs: $shift_type) -> Self::Output {
        pick! {
          if #[cfg(target_feature="avx2")] {
            let shift = cast([rhs as u64, 0]);
            Self { avx2: shr_all_i16_m256i(self.avx2, shift) }
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
impl_shr_t_for_i16x16!(i8, u8, i16, u16, i32, u32, i64, u64, i128, u128);

impl CmpEq for i16x16 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn cmp_eq(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self { avx2: cmp_eq_mask_i16_m256i(self.avx2, rhs.avx2) }
      } else {
        Self {
          a : self.a.cmp_eq(rhs.a),
          b : self.b.cmp_eq(rhs.b),
        }
      }
    }
  }
}

impl CmpGt for i16x16 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn cmp_gt(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self { avx2: cmp_gt_mask_i16_m256i(self.avx2, rhs.avx2) }
      } else {
        Self {
          a : self.a.cmp_gt(rhs.a),
          b : self.b.cmp_gt(rhs.b),
        }
      }
    }
  }
}

impl CmpLt for i16x16 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn cmp_lt(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self { avx2: !cmp_gt_mask_i16_m256i(self.avx2, rhs.avx2) ^ cmp_eq_mask_i16_m256i(self.avx2,rhs.avx2) }
      } else {
        Self {
          a : self.a.cmp_lt(rhs.a),
          b : self.b.cmp_lt(rhs.b),
        }
      }
    }
  }
}

impl i16x16 {
  #[inline]
  #[must_use]
  pub fn new(array: [i16; 16]) -> Self {
    Self::from(array)
  }
  #[inline]
  #[must_use]
  pub fn blend(self, t: Self, f: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self { avx2: blend_varying_i8_m256i(f.avx2, t.avx2, self.avx2) }
      } else {
        Self {
          a : self.a.blend(t.a, f.a),
          b : self.b.blend(t.b, f.b),
        }
      }
    }
  }
  #[inline]
  #[must_use]
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
  #[must_use]
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
  #[must_use]
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
  #[must_use]
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
  #[must_use]
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

  /// Multiply and scale equivilent to ((self * rhs) + 0x4000) >> 15 on each
  /// lane, effectively multiplying by a 16 bit fixed point number between -1
  /// and 1. This corresponds to the following instructions:
  /// - vqrdmulhq_n_s16 instruction on neon
  /// - i16x8_q15mulr_sat on simd128
  /// - _mm256_mulhrs_epi16 on avx2
  /// - emulated via mul_i16_* on sse2
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

  /// Multiply and scale equivilent to ((self * rhs) + 0x4000) >> 15 on each
  /// lane, effectively multiplying by a 16 bit fixed point number between -1
  /// and 1. This corresponds to the following instructions:
  /// - vqrdmulhq_n_s16 instruction on neon
  /// - i16x8_q15mulr_sat on simd128
  /// - _mm256_mulhrs_epi16 on avx2
  /// - emulated via mul_i16_* on sse2
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

  /// converts i16 to i8, saturating values that are too large
  pub fn pack_to_i8_saturate(self) -> i8x16 {
    pick! {
      if #[cfg(target_feature="avx2")] {
        i8x16 { sse: pack_i16_to_i8_m128i( extract_m128i_from_m256i::<0>(self.avx2), extract_m128i_from_m256i::<1>(self.avx2))  }
      } else if #[cfg(target_feature="sse2")] {
        i8x16 { sse: pack_i16_to_i8_m128i( self.a.sse, self.b.sse ) }
      } else {
        fn clamp(a : i16) -> i8 {
            if a < i8::MIN as i32 {
              i8::MIN
            }
            else if a > i8::MAX as i32 {
              i8::MAX
            } else {
                a as i8
            }
        }

        i8x16::new([
          clamp(self.as_array_ref()[0]),
          clamp(self.as_array_ref()[1]),
          clamp(self.as_array_ref()[2]),
          clamp(self.as_array_ref()[3]),
          clamp(self.as_array_ref()[4]),
          clamp(self.as_array_ref()[5]),
          clamp(self.as_array_ref()[6]),
          clamp(self.as_array_ref()[7]),
          clamp(self.as_array_ref()[8]),
          clamp(self.as_array_ref()[9]),
          clamp(self.as_array_ref()[10]),
          clamp(self.as_array_ref()[11]),
          clamp(self.as_array_ref()[12]),
          clamp(self.as_array_ref()[13]),
          clamp(self.as_array_ref()[14]),
          clamp(self.as_array_ref()[15]),
        ])
      }
    }
  }

  /// converts i16 to i8, truncating the upper bits if they are set
  pub fn pack_to_i8_truncate(self) -> i8x16 {
    pick! {
      if #[cfg(target_feature="avx2")] {
        let a = self.avx2.bitand(set_splat_i16_m256i(0xff));
        i8x16 { sse: pack_i16_to_i8_m128i( extract_m128i_from_m256i::<0>(a), extract_m128i_from_m256i::<1>(a))  }
      } else if #[cfg(target_feature="sse2")] {
        let mask = set_splat_i16_m128i(0xff);
        i8x16 { sse: pack_i16_to_i8_m128i( self.a.sse.bitand(mask), self.b.sse.bitand(mask) ) }
      } else {
      i16x8::new([
        self.as_array_ref()[0] as i16,
        self.as_array_ref()[1] as i16,
        self.as_array_ref()[2] as i16,
        self.as_array_ref()[3] as i16,
        self.as_array_ref()[4] as i16,
        self.as_array_ref()[5] as i16,
        self.as_array_ref()[6] as i16,
        self.as_array_ref()[7] as i16,
        self.as_array_ref()[8] as i16,
        self.as_array_ref()[9] as i16,
        self.as_array_ref()[10] as i16,
        self.as_array_ref()[11] as i16,
        self.as_array_ref()[12] as i16,
        self.as_array_ref()[13] as i16,
        self.as_array_ref()[14] as i16,
        self.as_array_ref()[15] as i16,
      ])
      }
    }
  }

  #[inline]
  pub fn to_array(self) -> [i16; 16] {
    cast(self)
  }

  #[inline]
  pub fn as_array_ref(&self) -> &[i16; 16] {
    cast_ref(self)
  }
}
