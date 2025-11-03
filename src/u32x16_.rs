use super::*;

pick! {
  if #[cfg(target_feature="avx512f")] {
    #[derive(Default, Clone, Copy, PartialEq, Eq)]
    #[repr(C, align(64))]
    pub struct u32x16 { pub(crate) avx512: m512i }
  } else {
    #[derive(Default, Clone, Copy, PartialEq, Eq)]
    #[repr(C, align(64))]
    pub struct u32x16 { pub(crate) a : u32x8, pub(crate) b : u32x8 }
  }
}

int_uint_consts!(u32, 16, u32x16, 512);

unsafe impl Zeroable for u32x16 {}
unsafe impl Pod for u32x16 {}

impl AlignTo for u32x16 {
  type Elem = u32;
}

impl Add for u32x16 {
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

impl Sub for u32x16 {
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

impl Add<u32> for u32x16 {
  type Output = Self;
  #[inline]
  fn add(self, rhs: u32) -> Self::Output {
    self.add(Self::splat(rhs))
  }
}

impl Sub<u32> for u32x16 {
  type Output = Self;
  #[inline]
  fn sub(self, rhs: u32) -> Self::Output {
    self.sub(Self::splat(rhs))
  }
}

impl Add<u32x16> for u32 {
  type Output = u32x16;
  #[inline]
  fn add(self, rhs: u32x16) -> Self::Output {
    u32x16::splat(self).add(rhs)
  }
}

impl Sub<u32x16> for u32 {
  type Output = u32x16;
  #[inline]
  fn sub(self, rhs: u32x16) -> Self::Output {
    u32x16::splat(self).sub(rhs)
  }
}

impl Mul for u32x16 {
  type Output = Self;
  #[inline]
  fn mul(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: mul_i32_keep_low_m512i(self.avx512, rhs.avx512) }
      } else {
        Self {
          a : self.a.mul(rhs.a),
          b : self.b.mul(rhs.b),
        }
      }
    }
  }
}

impl BitAnd for u32x16 {
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

impl BitOr for u32x16 {
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

impl BitXor for u32x16 {
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

impl From<u16x16> for u32x16 {
  /// Widens and zero-extends each u16 lane to u32
  #[inline]
  fn from(v: u16x16) -> Self {
    pick! {
      if #[cfg(target_feature = "avx512f")] {
        Self {
          avx512: convert_to_u32_m512i_from_u16_m256i(v.avx2)
        }
      } else if #[cfg(target_feature = "avx2")] {
        let lo: m128i = extract_m128i_from_m256i::<0>(v.avx2);
        let hi: m128i = extract_m128i_from_m256i::<1>(v.avx2);
        Self {
          a: u32x8 { avx2: convert_to_i32_m256i_from_u16_m128i(lo) },
          b: u32x8 { avx2: convert_to_i32_m256i_from_u16_m128i(hi) },
        }
      } else if #[cfg(target_feature = "sse2")] {
        Self {
          a: u32x8 {
            a: u32x4 {
              sse: shr_imm_u32_m128i::<16>(unpack_low_i16_m128i(v.a.sse, v.a.sse))
            },
            b: u32x4 {
              sse: shr_imm_u32_m128i::<16>(unpack_high_i16_m128i(v.a.sse, v.a.sse))
            },
          },
          b: u32x8 {
            a: u32x4 {
              sse: shr_imm_u32_m128i::<16>(unpack_low_i16_m128i(v.b.sse, v.b.sse))
            },
            b: u32x4 {
              sse: shr_imm_u32_m128i::<16>(unpack_high_i16_m128i(v.b.sse, v.b.sse))
            },
          },
        }
      } else {
        // Portable fallback
        let arr = v.as_array();
        Self::new([
          arr[0] as u32,  arr[1] as u32,  arr[2] as u32,  arr[3] as u32,
          arr[4] as u32,  arr[5] as u32,  arr[6] as u32,  arr[7] as u32,
          arr[8] as u32,  arr[9] as u32,  arr[10] as u32, arr[11] as u32,
          arr[12] as u32, arr[13] as u32, arr[14] as u32, arr[15] as u32,
        ])
      }
    }
  }
}

macro_rules! impl_shl_t_for_u32x16 {
  ($($shift_type:ty),+ $(,)?) => {
    $(impl Shl<$shift_type> for u32x16 {
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
impl_shl_t_for_u32x16!(i8, u8, i16, u16, i32, u32, i64, u64, i128, u128);

macro_rules! impl_shr_t_for_u32x16 {
  ($($shift_type:ty),+ $(,)?) => {
    $(impl Shr<$shift_type> for u32x16 {
      type Output = Self;
      /// Shifts all lanes by the value given.
      #[inline]
      fn shr(self, rhs: $shift_type) -> Self::Output {
        pick! {
          if #[cfg(target_feature="avx512f")] {
            let shift = cast(rhs as u32);
            Self { avx512: shr_all_u32_m512i(self.avx512, shift) }
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
impl_shr_t_for_u32x16!(i8, u8, i16, u16, i32, u32, i64, u64, i128, u128);

/// Shifts lanes by the corresponding lane.
///
/// Bitwise shift-right; yields `self >> mask(rhs)`, where mask removes any
/// high-order bits of `rhs` that would cause the shift to exceed the bitwidth
/// of the type. (same as `wrapping_shr`)
impl Shr<u32x16> for u32x16 {
  type Output = Self;

  #[inline]
  fn shr(self, rhs: u32x16) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        let shift_by = bitand_m512i(rhs.avx512, set_splat_i32_m512i(31));
        Self { avx512: shr_each_u32_m512i(self.avx512, shift_by ) }
      } else {
        Self {
          a : self.a.shr(rhs.a),
          b : self.b.shr(rhs.b),
        }
      }
    }
  }
}

/// Shifts lanes by the corresponding lane.
///
/// Bitwise shift-left; yields `self << mask(rhs)`, where mask removes any
/// high-order bits of `rhs` that would cause the shift to exceed the bitwidth
/// of the type. (same as `wrapping_shl`)
impl Shl<u32x16> for u32x16 {
  type Output = Self;

  #[inline]
  fn shl(self, rhs: u32x16) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        let shift_by = bitand_m512i(rhs.avx512, set_splat_i32_m512i(31));
        Self { avx512: shl_each_u32_m512i(self.avx512, shift_by) }
      } else {
        Self {
          a : self.a.shl(rhs.a),
          b : self.b.shl(rhs.b),
        }
      }
    }
  }
}

impl CmpEq for u32x16 {
  type Output = Self;
  #[inline]
  fn simd_eq(self, rhs: Self) -> Self::Output {
    Self::simd_eq(self, rhs)
  }
}

impl CmpGt for u32x16 {
  type Output = Self;
  #[inline]
  fn simd_gt(self, rhs: Self) -> Self::Output {
    Self::simd_gt(self, rhs)
  }
}

impl CmpLt for u32x16 {
  type Output = Self;
  #[inline]
  fn simd_lt(self, rhs: Self) -> Self::Output {
    // no gt, so just reverse to get same answer
    Self::simd_gt(rhs, self)
  }
}

impl u32x16 {
  #[inline]
  #[must_use]
  pub const fn new(array: [u32; 16]) -> Self {
    unsafe { core::mem::transmute(array) }
  }

  #[inline]
  #[must_use]
  pub fn simd_eq(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: cmp_op_mask_u32_m512i::<{cmp_int_op!(Eq)}>(self.avx512, rhs.avx512) }
      } else {
        Self {
          a : self.a.simd_eq(rhs.a),
          b : self.b.simd_eq(rhs.b),
        }
      }
    }
  }

  #[inline]
  #[must_use]
  pub fn simd_gt(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: cmp_op_mask_u32_m512i::<{cmp_int_op!(Nle)}>(self.avx512, rhs.avx512) }
      } else {
        Self {
          a : self.a.simd_gt(rhs.a),
          b : self.b.simd_gt(rhs.b),
        }
      }
    }
  }

  #[inline]
  #[must_use]
  pub fn simd_lt(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: cmp_op_mask_u32_m512i::<{cmp_int_op!(Lt)}>(self.avx512, rhs.avx512) }
      } else {
        Self {
          a : rhs.a.simd_gt(self.a),
          b : rhs.b.simd_gt(self.b),
        }
      }
    }
  }

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

  #[inline]
  #[must_use]
  pub fn min(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: min_u32_m512i(self.avx512, rhs.avx512) }
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
        Self { avx512: max_u32_m512i(self.avx512, rhs.avx512) }
      } else {
        Self {
          a: self.a.max(rhs.a),
          b: self.b.max(rhs.b),
        }
      }
    }
  }

  #[inline]
  #[must_use]
  pub fn mul_keep_high(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        let alo = extract_m256i32_from_m512i::<0>(self.avx512);
        let ahi = extract_m256i32_from_m512i::<1>(self.avx512);
        let blo = extract_m256i32_from_m512i::<0>(rhs.avx512);
        let bhi = extract_m256i32_from_m512i::<1>(rhs.avx512);

        let lo_res: m256i = {
          let a8 = u32x8 { avx2: alo };
          let b8 = u32x8 { avx2: blo };
          a8.mul_keep_high(b8).avx2
        };
        let hi_res: m256i = {
          let a8 = u32x8 { avx2: ahi };
          let b8 = u32x8 { avx2: bhi };
          a8.mul_keep_high(b8).avx2
        };

        let zero = zeroed_m512i();
        let with_lo = insert_m256i32_to_m512i::<0>(zero, lo_res);
        let combined = insert_m256i32_to_m512i::<1>(with_lo, hi_res);

        Self { avx512: combined }
      } else {
        Self {
          a: self.a.mul_keep_high(rhs.a),
          b: self.b.mul_keep_high(rhs.b),
        }
      }
    }
  }
  
  #[inline]
  #[must_use]
  pub fn to_bitmask(self) -> u32 {
    i32x16::to_bitmask(cast(self))
  }

  #[inline]
  #[must_use]
  pub fn any(self) -> bool {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        ((movepi8_mask_m512i(self.avx512) as u32) &
          0b10001000100010001000100010001000) != 0
      } else {
        (self.a | self.b).any()
      }
    }
  }

  #[inline]
  #[must_use]
  pub fn all(self) -> bool {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        ((movepi8_mask_m512i(self.avx512) as u32) &
          0b10001000100010001000100010001000) ==
          0b10001000100010001000100010001000
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

  #[inline]
  pub fn to_array(self) -> [u32; 16] {
    cast(self)
  }

  #[inline]
  pub fn as_array(&self) -> &[u32; 16] {
    cast_ref(self)
  }

  #[inline]
  pub fn as_mut_array(&mut self) -> &mut [u32; 16] {
    cast_mut(self)
  }
}

impl Not for u32x16 {
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
