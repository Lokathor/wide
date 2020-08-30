use super::*;

pick! {
  if #[cfg(target_feature="avx")] {
    #[derive(Default, Clone, Copy, PartialEq)]
    #[repr(C, align(32))]
    pub struct f64x4 { avx: m256d }
  } else if #[cfg(target_feature="sse2")] {
    #[derive(Default, Clone, Copy, PartialEq)]
    #[repr(C, align(32))]
    pub struct f64x4 { sse0: m128d, sse1: m128d }
  }
  else {
    #[derive(Default, Clone, Copy, PartialEq)]
    #[repr(C, align(32))]
    pub struct f64x4 { arr: [f64;4] }
  }
}

macro_rules! const_f64_as_f64x4 {
  ($i:ident, $f:expr) => {
    pub const $i: f64x4 =
      unsafe { ConstUnionHack128bit { f64a4: [$f, $f, $f, $f] }.f64x4 };
  };
}

impl f64x4 {
  const_f64_as_f64x4!(ONE, 1.0);
  const_f64_as_f64x4!(ZERO, 0.0);
  const_f64_as_f64x4!(HALF, 0.5);
  const_f64_as_f64x4!(E, core::f64::consts::E);
  const_f64_as_f64x4!(FRAC_1_PI, core::f64::consts::FRAC_1_PI);
  const_f64_as_f64x4!(FRAC_2_PI, core::f64::consts::FRAC_2_PI);
  const_f64_as_f64x4!(FRAC_2_SQRT_PI, core::f64::consts::FRAC_2_SQRT_PI);
  const_f64_as_f64x4!(FRAC_1_SQRT_2, core::f64::consts::FRAC_1_SQRT_2);
  const_f64_as_f64x4!(FRAC_PI_2, core::f64::consts::FRAC_PI_2);
  const_f64_as_f64x4!(FRAC_PI_3, core::f64::consts::FRAC_PI_3);
  const_f64_as_f64x4!(FRAC_PI_4, core::f64::consts::FRAC_PI_4);
  const_f64_as_f64x4!(FRAC_PI_6, core::f64::consts::FRAC_PI_6);
  const_f64_as_f64x4!(FRAC_PI_8, core::f64::consts::FRAC_PI_8);
  const_f64_as_f64x4!(LN_2, core::f64::consts::LN_2);
  const_f64_as_f64x4!(LN_10, core::f64::consts::LN_10);
  const_f64_as_f64x4!(LOG2_E, core::f64::consts::LOG2_E);
  const_f64_as_f64x4!(LOG10_E, core::f64::consts::LOG10_E);
  const_f64_as_f64x4!(LOG10_2, core::f64::consts::LOG10_2);
  const_f64_as_f64x4!(LOG2_10, core::f64::consts::LOG2_10);
  const_f64_as_f64x4!(PI, core::f64::consts::PI);
  const_f64_as_f64x4!(SQRT_2, core::f64::consts::SQRT_2);
  const_f64_as_f64x4!(TAU, 6.28318530717958647692528676655900577_f64);
}

unsafe impl Zeroable for f64x4 {}
unsafe impl Pod for f64x4 {}

impl Add for f64x4 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn add(self, rhs: Self) -> Self::Output {
    pick! {
    if #[cfg(target_feature="avx")] {
      Self { avx: add_m256d(self.avx, rhs.avx) }
    } else if #[cfg(target_feature="sse2")] {
      Self { sse0: add_m128d(self.sse0, rhs.sse0), sse1: add_m128d(self.sse1, rhs.sse1) }
    } else {
          Self { arr: [
            self.arr[0] + rhs.arr[0],
            self.arr[1] + rhs.arr[1],
            self.arr[2] + rhs.arr[2],
            self.arr[3] + rhs.arr[3],
          ]}
        }
      }
  }
}

impl Sub for f64x4 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn sub(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx")] {
        Self { avx: sub_m256d(self.avx, rhs.avx) }
      } else if #[cfg(target_feature="sse2")] {
        Self { sse0: sub_m128d(self.sse0, rhs.sse0), sse1: sub_m128d(self.sse1, rhs.sse1) }
      } else {
        Self { arr: [
          self.arr[0] - rhs.arr[0],
          self.arr[1] - rhs.arr[1],
          self.arr[2] - rhs.arr[2],
          self.arr[3] - rhs.arr[3],
        ]}
      }
    }
  }
}

impl Mul for f64x4 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn mul(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx")] {
        Self { avx: mul_m256d(self.avx, rhs.avx) }
      } else if #[cfg(target_feature="sse2")] {
        Self { sse0: mul_m128d(self.sse0, rhs.sse0), sse1: mul_m128d(self.sse1, rhs.sse1) }
      } else {
        Self { arr: [
          self.arr[0] * rhs.arr[0],
          self.arr[1] * rhs.arr[1],
          self.arr[2] * rhs.arr[2],
          self.arr[3] * rhs.arr[3],
        ]}
      }
    }
  }
}

impl Div for f64x4 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn div(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx")] {
        Self { avx: div_m256d(self.avx, rhs.avx) }
      } else if #[cfg(target_feature="sse2")] {
        Self { sse0: div_m128d(self.sse0, rhs.sse0), sse1: div_m128d(self.sse1, rhs.sse1) }
      } else {
        Self { arr: [
          self.arr[0] / rhs.arr[0],
          self.arr[1] / rhs.arr[1],
          self.arr[2] / rhs.arr[2],
          self.arr[3] / rhs.arr[3],
        ]}
      }
    }
  }
}

impl BitAnd for f64x4 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn bitand(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx")] {
        Self { avx: bitand_m256d(self.avx, rhs.avx) }
      } else if #[cfg(target_feature="sse2")] {
        Self { sse0: bitand_m128d(self.sse0, rhs.sse0), sse1: bitand_m128d(self.sse1, rhs.sse1) }
      } else {
        Self { arr: [
          f64::from_bits(self.arr[0].to_bits() & rhs.arr[0].to_bits()),
          f64::from_bits(self.arr[1].to_bits() & rhs.arr[1].to_bits()),
          f64::from_bits(self.arr[2].to_bits() & rhs.arr[2].to_bits()),
          f64::from_bits(self.arr[3].to_bits() & rhs.arr[3].to_bits()),
        ]}
      }
    }
  }
}

impl BitOr for f64x4 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn bitor(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx")] {
        Self { avx: bitor_m256d(self.avx, rhs.avx) }
      } else if #[cfg(target_feature="sse2")] {
        Self { sse0: bitor_m128d(self.sse0, rhs.sse0), sse1: bitor_m128d(self.sse1, rhs.sse1) }
      } else {
        Self { arr: [
          f64::from_bits(self.arr[0].to_bits() | rhs.arr[0].to_bits()),
          f64::from_bits(self.arr[1].to_bits() | rhs.arr[1].to_bits()),
          f64::from_bits(self.arr[2].to_bits() | rhs.arr[2].to_bits()),
          f64::from_bits(self.arr[3].to_bits() | rhs.arr[3].to_bits()),
        ]}
      }
    }
  }
}

impl BitXor for f64x4 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn bitxor(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx")] {
        Self { avx: bitxor_m256d(self.avx, rhs.avx) }
      } else if #[cfg(target_feature="sse2")] {
        Self { sse0: bitxor_m128d(self.sse0, rhs.sse0), sse1: bitxor_m128d(self.sse1, rhs.sse1) }
      }
      else {
        Self { arr: [
          f64::from_bits(self.arr[0].to_bits() ^ rhs.arr[0].to_bits()),
          f64::from_bits(self.arr[1].to_bits() ^ rhs.arr[1].to_bits()),
          f64::from_bits(self.arr[2].to_bits() ^ rhs.arr[2].to_bits()),
          f64::from_bits(self.arr[3].to_bits() ^ rhs.arr[3].to_bits()),
        ]}
      }
    }
  }
}

impl f64x4 {
  #[inline]
  #[must_use]
  pub fn cmp_eq(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx")]{
        Self { avx: cmp_op_mask_m256d!(self.avx, EqualOrdered, rhs.avx) }
      } else if #[cfg(target_feature="sse2")] {
        Self { sse0: cmp_eq_mask_m128d(self.sse0, rhs.sse0), sse1: cmp_eq_mask_m128d(self.sse1, rhs.sse1) }
      } else {
        Self { arr: [
          if self.arr[0] == rhs.arr[0] { f64::from_bits(u64::MAX) } else { 0.0 },
          if self.arr[1] == rhs.arr[1] { f64::from_bits(u64::MAX) } else { 0.0 },
          if self.arr[2] == rhs.arr[2] { f64::from_bits(u64::MAX) } else { 0.0 },
          if self.arr[3] == rhs.arr[3] { f64::from_bits(u64::MAX) } else { 0.0 },
        ]}
      }
    }
  }
  #[inline]
  #[must_use]
  pub fn cmp_ne(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx")]{
        Self { avx: cmp_op_mask_m256d!(self.avx, NotEqualOrdered, rhs.avx) }
      }
      else if #[cfg(target_feature="sse2")] {
        Self { sse0: cmp_neq_mask_m128d(self.sse0, rhs.sse0), sse1: cmp_neq_mask_m128d(self.sse1, rhs.sse1) }
      } else {
        Self { arr: [
          if self.arr[0] != rhs.arr[0] { f64::from_bits(u64::MAX) } else { 0.0 },
          if self.arr[1] != rhs.arr[1] { f64::from_bits(u64::MAX) } else { 0.0 },
          if self.arr[2] != rhs.arr[2] { f64::from_bits(u64::MAX) } else { 0.0 },
          if self.arr[3] != rhs.arr[3] { f64::from_bits(u64::MAX) } else { 0.0 },
        ]}
      }
    }
  }
  #[inline]
  #[must_use]
  pub fn cmp_ge(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx")]{
        Self { avx: cmp_op_mask_m256d!(self.avx, GreaterEqualOrdered, rhs.avx) }
      }
      else if #[cfg(target_feature="sse2")] {
        Self { sse0: cmp_ge_mask_m128d(self.sse0, rhs.sse0), sse1: cmp_ge_mask_m128d(self.sse1, rhs.sse1) }
      } else {
        Self { arr: [
          if self.arr[0] >= rhs.arr[0] { f64::from_bits(u64::MAX) } else { 0.0 },
          if self.arr[1] >= rhs.arr[1] { f64::from_bits(u64::MAX) } else { 0.0 },
          if self.arr[2] >= rhs.arr[2] { f64::from_bits(u64::MAX) } else { 0.0 },
          if self.arr[3] >= rhs.arr[3] { f64::from_bits(u64::MAX) } else { 0.0 },
        ]}
      }
    }
  }
  #[inline]
  #[must_use]
  pub fn cmp_gt(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx")]{
        Self { avx: cmp_op_mask_m256d!(self.avx, GreaterThanOrdered, rhs.avx) }
      }
      else if #[cfg(target_feature="sse2")] {
        Self { sse0: cmp_gt_mask_m128d(self.sse0, rhs.sse0), sse1: cmp_ge_mask_m128d(self.sse1, rhs.sse1) }
      } else {
        Self { arr: [
          if self.arr[0] > rhs.arr[0] { f64::from_bits(u64::MAX) } else { 0.0 },
          if self.arr[1] > rhs.arr[1] { f64::from_bits(u64::MAX) } else { 0.0 },
          if self.arr[2] > rhs.arr[2] { f64::from_bits(u64::MAX) } else { 0.0 },
          if self.arr[3] > rhs.arr[3] { f64::from_bits(u64::MAX) } else { 0.0 },
        ]}
      }
    }
  }
  #[inline]
  #[must_use]
  pub fn cmp_le(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx")]{
        Self { avx: cmp_op_mask_m256d!(self.avx, LessEqualOrdered, rhs.avx) }
      }
      else if #[cfg(target_feature="sse2")] {
        Self { sse0: cmp_le_mask_m128d(self.sse0, rhs.sse0), sse1: cmp_le_mask_m128d(self.sse1, rhs.sse1) }
      } else {
        Self { arr: [
          if self.arr[0] <= rhs.arr[0] { f64::from_bits(u64::MAX) } else { 0.0 },
          if self.arr[1] <= rhs.arr[1] { f64::from_bits(u64::MAX) } else { 0.0 },
          if self.arr[2] <= rhs.arr[2] { f64::from_bits(u64::MAX) } else { 0.0 },
          if self.arr[3] <= rhs.arr[3] { f64::from_bits(u64::MAX) } else { 0.0 },
        ]}
      }
    }
  }
  #[inline]
  #[must_use]
  pub fn cmp_lt(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx")]{
        Self { avx: cmp_op_mask_m256d!(self.avx, LessThanOrdered, rhs.avx) }
      }
      else if #[cfg(target_feature="sse2")] {
        Self { sse0: cmp_lt_mask_m128d(self.sse0, rhs.sse0), sse1: cmp_lt_mask_m128d(self.sse1, rhs.sse1) }
      } else {
        Self { arr: [
          if self.arr[0] < rhs.arr[0] { f64::from_bits(u64::MAX) } else { 0.0 },
          if self.arr[1] < rhs.arr[1] { f64::from_bits(u64::MAX) } else { 0.0 },
          if self.arr[2] < rhs.arr[2] { f64::from_bits(u64::MAX) } else { 0.0 },
          if self.arr[3] < rhs.arr[3] { f64::from_bits(u64::MAX) } else { 0.0 },
        ]}
      }
    }
  }
  #[inline]
  #[must_use]
  pub fn blend(self, t: Self, f: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx")] {
        Self { avx: blend_varying_m256d(f.avx, t.avx, self.avx) }
      }
      else if  #[cfg(target_feature="sse4.1")] {
        Self { sse0: blend_varying_m128d(f.sse0, t.sse0, self.sse0), sse1: blend_varying_m128d(f.sse1, t.sse1, self.sse1) }
      } else {

        generic_bit_blend(self, t, f)
      }
    }
  }

  #[inline]
  #[must_use]
  pub fn abs(self) -> Self {
    let non_sign_bits = f64x4::from(f64::from_bits(i64::MAX as u64));
    self & non_sign_bits
  }

  #[inline]
  #[must_use]
  pub fn max(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx")] {
        Self { avx: max_m256d(self.avx, rhs.avx) }
      } else if #[cfg(target_feature="sse2")] {
        Self { sse0: max_m128d(self.sse0, rhs.sse0), sse1: max_m128d(self.sse1, rhs.sse1) }
      } else {
        Self { arr: [
          self.arr[0].max(rhs.arr[0]),
          self.arr[1].max(rhs.arr[1]),
          self.arr[2].max(rhs.arr[2]),
          self.arr[3].max(rhs.arr[3]),
        ]}
      }
    }
  }

  #[inline]
  #[must_use]
  pub fn min(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx")] {
        Self { avx: min_m256d(self.avx, rhs.avx) }
      } else if #[cfg(target_feature="sse2")] {
        Self { sse0: min_m128d(self.sse0, rhs.sse0), sse1: min_m128d(self.sse1, rhs.sse1) }
      } else {
        Self { arr: [
          self.arr[0].min(rhs.arr[0]),
          self.arr[1].min(rhs.arr[1]),
          self.arr[2].min(rhs.arr[2]),
          self.arr[3].min(rhs.arr[3]),
        ]}
      }
    }
  }

  #[inline]
  #[must_use]
  pub fn is_nan(self) -> Self {
    pick! {
      if #[cfg(target_feature="avx")] {
        Self { avx: cmp_op_mask_m256d!(self.avx, Unordered, self.avx ) }
      } else if #[cfg(target_feature="sse2")] {
        Self { sse0: cmp_unord_mask_m128d(self.sse0, self.sse0) , sse1: cmp_unord_mask_m128d(self.sse1, self.sse1) }
      } else {
        Self { arr: [
          if self.arr[0].is_nan() { f64::from_bits(u64::MAX) } else { 0.0 },
          if self.arr[1].is_nan() { f64::from_bits(u64::MAX) } else { 0.0 },
          if self.arr[2].is_nan() { f64::from_bits(u64::MAX) } else { 0.0 },
          if self.arr[3].is_nan() { f64::from_bits(u64::MAX) } else { 0.0 },
        ]}
      }
    }
  }

  #[inline]
  #[must_use]
  pub fn is_finite(self) -> Self {
    let shifted_exp_mask = u64x4::from(0xFFE0000000000000);
    let u: u64x4 = cast(self);
    let shift_u = u << 1_u64;
    let out = !(shift_u & shifted_exp_mask).cmp_eq(shifted_exp_mask);
    cast(out)
  }

  #[inline]
  #[must_use]
  pub fn round(self) -> Self {
    pick! {
      if #[cfg(target_feature="avx")] {
        Self { avx: round_m256d!(self.avx, Nearest) }
      }  else if #[cfg(target_feature="sse4.1")] {
        Self { sse0: round_m128d!(self.sse0, Nearest), sse1: round_m128d!(self.sse1, Nearest) }
      } else {
          let sign_mask = f64x4::from(-0.0);
          let magic = f64x4::from(f64::from_bits(0x43300000_00000000));
          let sign = self & sign_mask;
          let signed_magic = magic | sign;
          self + signed_magic - signed_magic
      }
    }
  }

  #[inline]
  #[must_use]
  pub fn round_int(self) -> i64x4 {
    // NOTE:No optimisation for this currently available so delegate to LLVM
    let rounded: [f64; 4] = cast(self.round());
    let rounded_ints: i64x4 = cast([
      rounded[0] as i64,
      rounded[1] as i64,
      rounded[2] as i64,
      rounded[3] as i64,
    ]);
    cast::<f64x4, i64x4>(self.is_finite())
      .blend(rounded_ints, i64x4::from(i64::MIN))
  }

  #[inline]
  #[must_use]
  pub fn mul_add(self, m: Self, a: Self) -> Self {
    pick! {
      if #[cfg(all(target_feature="avx",target_feature="fma"))] {
        Self { avx: fused_mul_add_m256d(self.avx, m.avx, a.avx) }
      } else if #[cfg(all(target_feature="avx",target_feature="fma"))]
      {
        Self { sse0: fused_mul_add_m128d(self.sse0, m.sse0, a.sse0), sse1: fused_mul_add_m128d(self.sse1, m.sse1, a.sse1) }
      } else {
        (self * m) + a
      }
    }
  }

  #[inline]
  #[must_use]
  pub fn mul_sub(self, m: Self, a: Self) -> Self {
    pick! {
      if #[cfg(all(target_feature="avx",target_feature="fma"))] {
        Self { avx: fused_mul_sub_m256d(self.avx, m.avx, a.avx) }
      } else if #[cfg(all(target_feature="avx",target_feature="fma"))]
      {
        Self { sse0: fused_mul_sub_m128d(self.sse0, m.sse0, a.sse0), sse1: fused_mul_sub_m128d(self.sse1, m.sse1, a.sse1) }
      } else {
        (self * m) - a
      }
    }
  }

  #[inline]
  #[must_use]
  pub fn mul_neg_add(self, m: Self, a: Self) -> Self {
    pick! {
      if #[cfg(all(target_feature="avx",target_feature="fma"))] {
        Self { avx: fused_mul_neg_add_m256d(self.avx, m.avx, a.avx) }
      } else if #[cfg(all(target_feature="avx",target_feature="fma"))]
      {
        Self { sse0: fused_mul_neg_add_m128d(self.sse0, m.sse0, a.sse0), sse1: fused_mul_neg_add_m128d(self.sse1, m.sse1, a.sse1) }
      }  else {
        a - (self * m)
      }
    }
  }

  #[inline]
  #[must_use]
  pub fn mul_neg_sub(self, m: Self, a: Self) -> Self {
    pick! {
        if #[cfg(all(target_feature="avx",target_feature="fma"))] {
          Self { avx: fused_mul_neg_sub_m256d(self.avx, m.avx, a.avx) }
        } else if #[cfg(all(target_feature="avx",target_feature="fma"))]
        {
          Self { sse0: fused_mul_neg_sub_m128d(self.sse0, m.sse0, a.sse0), sse1: fused_mul_neg_sub_m128d(self.sse1, m.sse1, a.sse1) }
        }  else {
           -(self * m) - a
        }
    }
  }

  #[inline]
  #[must_use]
  pub fn flip_signs(self, signs: Self) -> Self {
    self ^ (signs & Self::from(-0.0))
  }

  #[allow(non_upper_case_globals)]
  pub fn asin_acos(self) -> (Self, Self) {
    // Based on the Agner Fog "vector class library":
    // https://github.com/vectorclass/version2/blob/master/vectormath_trig.h
    const_f64_as_f64x4!(R4asin, 2.967721961301243206100E-3);
    const_f64_as_f64x4!(R3asin, -5.634242780008963776856E-1);
    const_f64_as_f64x4!(R2asin, 6.968710824104713396794E0);
    const_f64_as_f64x4!(R1asin, -2.556901049652824852289E1);
    const_f64_as_f64x4!(R0asin, 2.853665548261061424989E1);

    const_f64_as_f64x4!(S3asin, -2.194779531642920639778E1);
    const_f64_as_f64x4!(S2asin, 1.470656354026814941758E2);
    const_f64_as_f64x4!(S1asin, -3.838770957603691357202E2);
    const_f64_as_f64x4!(S0asin, 3.424398657913078477438E2);

    const_f64_as_f64x4!(P5asin, 4.253011369004428248960E-3);
    const_f64_as_f64x4!(P4asin, -6.019598008014123785661E-1);
    const_f64_as_f64x4!(P3asin, 5.444622390564711410273E0);
    const_f64_as_f64x4!(P2asin, -1.626247967210700244449E1);
    const_f64_as_f64x4!(P1asin, 1.956261983317594739197E1);
    const_f64_as_f64x4!(P0asin, -8.198089802484824371615E0);

    const_f64_as_f64x4!(Q4asin, -1.474091372988853791896E1);
    const_f64_as_f64x4!(Q3asin, 7.049610280856842141659E1);
    const_f64_as_f64x4!(Q2asin, -1.471791292232726029859E2);
    const_f64_as_f64x4!(Q1asin, 1.395105614657485689735E2);
    const_f64_as_f64x4!(Q0asin, -4.918853881490881290097E1);

    let xa = self.abs();

    let big = xa.cmp_ge(f64x4::splat(0.625));

    let x1 = big.blend(f64x4::splat(1.0) - xa, xa * xa);

    let x2 = x1 * x1;
    let x3 = x2 * x1;
    let x4 = x2 * x2;
    let x5 = x4 * x1;

    let dobig = big.any();
    let dosmall = !big.all();

    let mut rx = f64x4::default();
    let mut sx = f64x4::default();
    let mut px = f64x4::default();
    let mut qx = f64x4::default();

    if dobig {
      rx = x3.mul_add(R3asin, x2 * R2asin)
        + x4.mul_add(R4asin, x1.mul_add(R1asin, R0asin));
      sx =
        x3.mul_add(S3asin, x4) + x2.mul_add(S2asin, x1.mul_add(S1asin, S0asin));
    }

    if dosmall {
      px = x3.mul_add(P3asin, P0asin)
        + x4.mul_add(P4asin, x1 * P1asin)
        + x5.mul_add(P5asin, x2 * P2asin);
      qx = x4.mul_add(Q4asin, x5)
        + x3.mul_add(Q3asin, x1 * Q1asin)
        + x2.mul_add(Q2asin, Q0asin);
    };

    let vx = big.blend(rx, px);
    let wx = big.blend(sx, qx);

    let y1 = vx / wx * x1;

    let mut z1 = f64x4::default();
    let mut z2 = f64x4::default();
    if dobig {
      let xb = (x1 + x1).sqrt();
      z1 = xb.mul_add(y1, xb);
    }

    if dosmall {
      z2 = xa.mul_add(y1, xa);
    }

    // asin
    let z3 = f64x4::FRAC_PI_2 - z1;
    let asin = big.blend(z3, z2);
    let asin = asin.flip_signs(self);

    // acos
    let z3 = self.cmp_lt(f64x4::ZERO).blend(f64x4::PI - z1, z1);
    let z4 = f64x4::FRAC_PI_2 - z2.flip_signs(self);
    let acos = big.blend(z3, z4);

    (asin, acos)
  }

  #[allow(non_upper_case_globals)]
  pub fn acos(self) -> Self {
    // Based on the Agner Fog "vector class library":
    // https://github.com/vectorclass/version2/blob/master/vectormath_trig.h
    const_f64_as_f64x4!(R4asin, 2.967721961301243206100E-3);
    const_f64_as_f64x4!(R3asin, -5.634242780008963776856E-1);
    const_f64_as_f64x4!(R2asin, 6.968710824104713396794E0);
    const_f64_as_f64x4!(R1asin, -2.556901049652824852289E1);
    const_f64_as_f64x4!(R0asin, 2.853665548261061424989E1);

    const_f64_as_f64x4!(S3asin, -2.194779531642920639778E1);
    const_f64_as_f64x4!(S2asin, 1.470656354026814941758E2);
    const_f64_as_f64x4!(S1asin, -3.838770957603691357202E2);
    const_f64_as_f64x4!(S0asin, 3.424398657913078477438E2);

    const_f64_as_f64x4!(P5asin, 4.253011369004428248960E-3);
    const_f64_as_f64x4!(P4asin, -6.019598008014123785661E-1);
    const_f64_as_f64x4!(P3asin, 5.444622390564711410273E0);
    const_f64_as_f64x4!(P2asin, -1.626247967210700244449E1);
    const_f64_as_f64x4!(P1asin, 1.956261983317594739197E1);
    const_f64_as_f64x4!(P0asin, -8.198089802484824371615E0);

    const_f64_as_f64x4!(Q4asin, -1.474091372988853791896E1);
    const_f64_as_f64x4!(Q3asin, 7.049610280856842141659E1);
    const_f64_as_f64x4!(Q2asin, -1.471791292232726029859E2);
    const_f64_as_f64x4!(Q1asin, 1.395105614657485689735E2);
    const_f64_as_f64x4!(Q0asin, -4.918853881490881290097E1);

    let xa = self.abs();

    let big = xa.cmp_ge(f64x4::splat(0.625));

    let x1 = big.blend(f64x4::splat(1.0) - xa, xa * xa);

    let x2 = x1 * x1;
    let x3 = x2 * x1;
    let x4 = x2 * x2;
    let x5 = x4 * x1;

    let dobig = big.any();
    let dosmall = !big.all();

    let mut rx = f64x4::default();
    let mut sx = f64x4::default();
    let mut px = f64x4::default();
    let mut qx = f64x4::default();

    if dobig {
      rx = x3.mul_add(R3asin, x2 * R2asin)
        + x4.mul_add(R4asin, x1.mul_add(R1asin, R0asin));
      sx =
        x3.mul_add(S3asin, x4) + x2.mul_add(S2asin, x1.mul_add(S1asin, S0asin));
    }
    if dosmall {
      px = x3.mul_add(P3asin, P0asin)
        + x4.mul_add(P4asin, x1 * P1asin)
        + x5.mul_add(P5asin, x2 * P2asin);
      qx = x4.mul_add(Q4asin, x5)
        + x3.mul_add(Q3asin, x1 * Q1asin)
        + x2.mul_add(Q2asin, Q0asin);
    };

    let vx = big.blend(rx, px);
    let wx = big.blend(sx, qx);

    let y1 = vx / wx * x1;

    let mut z1 = f64x4::default();
    let mut z2 = f64x4::default();
    if dobig {
      let xb = (x1 + x1).sqrt();
      z1 = xb.mul_add(y1, xb);
    }

    if dosmall {
      z2 = xa.mul_add(y1, xa);
    }

    // acos
    let z3 = self.cmp_lt(f64x4::ZERO).blend(f64x4::PI - z1, z1);
    let z4 = f64x4::FRAC_PI_2 - z2.flip_signs(self);
    let acos = big.blend(z3, z4);

    acos
  }
  #[inline]
  #[must_use]
  #[allow(non_upper_case_globals)]
  pub fn asin(self) -> Self {
    // Based on the Agner Fog "vector class library":
    // https://github.com/vectorclass/version2/blob/master/vectormath_trig.h
    const_f64_as_f64x4!(R4asin, 2.967721961301243206100E-3);
    const_f64_as_f64x4!(R3asin, -5.634242780008963776856E-1);
    const_f64_as_f64x4!(R2asin, 6.968710824104713396794E0);
    const_f64_as_f64x4!(R1asin, -2.556901049652824852289E1);
    const_f64_as_f64x4!(R0asin, 2.853665548261061424989E1);

    const_f64_as_f64x4!(S3asin, -2.194779531642920639778E1);
    const_f64_as_f64x4!(S2asin, 1.470656354026814941758E2);
    const_f64_as_f64x4!(S1asin, -3.838770957603691357202E2);
    const_f64_as_f64x4!(S0asin, 3.424398657913078477438E2);

    const_f64_as_f64x4!(P5asin, 4.253011369004428248960E-3);
    const_f64_as_f64x4!(P4asin, -6.019598008014123785661E-1);
    const_f64_as_f64x4!(P3asin, 5.444622390564711410273E0);
    const_f64_as_f64x4!(P2asin, -1.626247967210700244449E1);
    const_f64_as_f64x4!(P1asin, 1.956261983317594739197E1);
    const_f64_as_f64x4!(P0asin, -8.198089802484824371615E0);

    const_f64_as_f64x4!(Q4asin, -1.474091372988853791896E1);
    const_f64_as_f64x4!(Q3asin, 7.049610280856842141659E1);
    const_f64_as_f64x4!(Q2asin, -1.471791292232726029859E2);
    const_f64_as_f64x4!(Q1asin, 1.395105614657485689735E2);
    const_f64_as_f64x4!(Q0asin, -4.918853881490881290097E1);

    let xa = self.abs();

    let big = xa.cmp_ge(f64x4::splat(0.625));

    let x1 = big.blend(f64x4::splat(1.0) - xa, xa * xa);

    let x2 = x1 * x1;
    let x3 = x2 * x1;
    let x4 = x2 * x2;
    let x5 = x4 * x1;

    let dobig = big.any();
    let dosmall = !big.all();

    let mut rx = f64x4::default();
    let mut sx = f64x4::default();
    let mut px = f64x4::default();
    let mut qx = f64x4::default();

    if dobig {
      rx = x3.mul_add(R3asin, x2 * R2asin)
        + x4.mul_add(R4asin, x1.mul_add(R1asin, R0asin));
      sx =
        x3.mul_add(S3asin, x4) + x2.mul_add(S2asin, x1.mul_add(S1asin, S0asin));
    }
    if dosmall {
      px = x3.mul_add(P3asin, P0asin)
        + x4.mul_add(P4asin, x1 * P1asin)
        + x5.mul_add(P5asin, x2 * P2asin);
      qx = x4.mul_add(Q4asin, x5)
        + x3.mul_add(Q3asin, x1 * Q1asin)
        + x2.mul_add(Q2asin, Q0asin);
    };

    let vx = big.blend(rx, px);
    let wx = big.blend(sx, qx);

    let y1 = vx / wx * x1;

    let mut z1 = f64x4::default();
    let mut z2 = f64x4::default();
    if dobig {
      let xb = (x1 + x1).sqrt();
      z1 = xb.mul_add(y1, xb);
    }

    if dosmall {
      z2 = xa.mul_add(y1, xa);
    }

    // asin
    let z3 = f64x4::FRAC_PI_2 - z1;
    let asin = big.blend(z3, z2);
    let asin = asin.flip_signs(self);

    asin
  }

  #[inline]
  #[must_use]
  #[allow(non_upper_case_globals)]
  pub fn sin_cos(self) -> (Self, Self) {
    // Based on the Agner Fog "vector class library":
    // https://github.com/vectorclass/version2/blob/master/vectormath_trig.h

    const_f64_as_f64x4!(P0sin, -1.66666666666666307295E-1);
    const_f64_as_f64x4!(P1sin, 8.33333333332211858878E-3);
    const_f64_as_f64x4!(P2sin, -1.98412698295895385996E-4);
    const_f64_as_f64x4!(P3sin, 2.75573136213857245213E-6);
    const_f64_as_f64x4!(P4sin, -2.50507477628578072866E-8);
    const_f64_as_f64x4!(P5sin, 1.58962301576546568060E-10);

    const_f64_as_f64x4!(P0cos, 4.16666666666665929218E-2);
    const_f64_as_f64x4!(P1cos, -1.38888888888730564116E-3);
    const_f64_as_f64x4!(P2cos, 2.48015872888517045348E-5);
    const_f64_as_f64x4!(P3cos, -2.75573141792967388112E-7);
    const_f64_as_f64x4!(P4cos, 2.08757008419747316778E-9);
    const_f64_as_f64x4!(P5cos, -1.13585365213876817300E-11);

    const_f64_as_f64x4!(DP1, 7.853981554508209228515625E-1 * 2.);
    const_f64_as_f64x4!(DP2, 7.94662735614792836714E-9 * 2.);
    const_f64_as_f64x4!(DP3, 3.06161699786838294307E-17 * 2.);

    const_f64_as_f64x4!(TWO_OVER_PI, 2.0 / core::f64::consts::PI);

    let xa = self.abs();

    let y = (xa * TWO_OVER_PI).round();
    let q = y.round_int();

    let x = y.mul_neg_add(DP3, y.mul_neg_add(DP2, y.mul_neg_add(DP1, xa)));

    let x2 = x * x;
    let mut s = polynomial_5!(x2, P0sin, P1sin, P2sin, P3sin, P4sin, P5sin);
    let mut c = polynomial_5!(x2, P0cos, P1cos, P2cos, P3cos, P4cos, P5cos);
    s = (x * x2).mul_add(s, x);
    c =
      (x2 * x2).mul_add(c, x2.mul_neg_add(f64x4::from(0.5), f64x4::from(1.0)));

    let swap = !((q & i64x4::from(1)).cmp_eq(i64x4::from(0)));

    let mut overflow: f64x4 = cast(q.cmp_gt(i64x4::from(0x80000000000000)));
    overflow &= xa.is_finite();
    s = overflow.blend(f64x4::from(0.0), s);
    c = overflow.blend(f64x4::from(1.0), c);

    // calc sin
    let mut sin1 = cast::<_, f64x4>(swap).blend(c, s);
    let sign_sin: i64x4 = (q << 62) ^ cast::<_, i64x4>(self);
    sin1 = sin1.flip_signs(cast(sign_sin));

    // calc cos
    let mut cos1 = cast::<_, f64x4>(swap).blend(s, c);
    let sign_cos: i64x4 = ((q + i64x4::from(1)) & i64x4::from(2)) << 62;
    cos1 ^= cast::<_, f64x4>(sign_cos);

    (sin1, cos1)
  }
  #[inline]
  #[must_use]
  pub fn sin(self) -> Self {
    let (s, _) = self.sin_cos();
    s
  }
  #[inline]
  #[must_use]
  pub fn cos(self) -> Self {
    let (_, c) = self.sin_cos();
    c
  }
  #[inline]
  #[must_use]
  pub fn tan(self) -> Self {
    let (s, c) = self.sin_cos();
    s / c
  }
  #[inline]
  #[must_use]
  pub fn to_degrees(self) -> Self {
    const_f64_as_f64x4!(RAD_TO_DEG_RATIO, 180.0_f64 / core::f64::consts::PI);
    self * RAD_TO_DEG_RATIO
  }
  #[inline]
  #[must_use]
  pub fn to_radians(self) -> Self {
    const_f64_as_f64x4!(DEG_TO_RAD_RATIO, core::f64::consts::PI / 180.0_f64);
    self * DEG_TO_RAD_RATIO
  }
  #[inline]
  #[must_use]
  pub fn sqrt(self) -> Self {
    pick! {
      if #[cfg(target_feature="avx")] {
        Self { avx: sqrt_m256d(self.avx) }
      } else if #[cfg(target_feature="sse2")] {
        Self { sse0: sqrt_m128d(self.sse0), sse1: sqrt_m128d(self.sse1) }
      }
      else if #[cfg(feature="std")] {
        Self { arr: [
          self.arr[0].sqrt(),
          self.arr[1].sqrt(),
          self.arr[2].sqrt(),
          self.arr[3].sqrt(),
        ]}
      } else {
        Self { arr: [
          software_sqrt(self.arr[0] as f64) as f64,
          software_sqrt(self.arr[1] as f64) as f64,
          software_sqrt(self.arr[2] as f64) as f64,
          software_sqrt(self.arr[3] as f64) as f64,
        ]}
      }
    }
  }
  #[inline]
  #[must_use]
  pub fn move_mask(self) -> i32 {
    pick! {
      if #[cfg(target_feature="avx")] {
        move_mask_m256d(self.avx)
      } else if #[cfg(target_feature="sse2")] {
        (move_mask_m128d(self.sse1) << 2) ^ move_mask_m128d(self.sse0)
      }
      else {
        (((self.arr[0].to_bits() as i64) < 0) as i32) << 0 |
        (((self.arr[1].to_bits() as i64) < 0) as i32) << 1 |
        (((self.arr[2].to_bits() as i64) < 0) as i32) << 2 |
        (((self.arr[3].to_bits() as i64) < 0) as i32) << 3
      }
    }
  }
  #[inline]
  #[must_use]
  pub fn any(self) -> bool {
    self.move_mask() != 0
  }
  #[inline]
  #[must_use]
  pub fn all(self) -> bool {
    // eight lanes
    self.move_mask() == 0b1111
  }
  #[inline]
  #[must_use]
  pub fn none(self) -> bool {
    !self.any()
  }

  #[inline]
  #[allow(non_upper_case_globals)]
  fn vm_pow2n(self) -> Self {
    const_f64_as_f64x4!(pow2_52, 4503599627370496.0);
    const_f64_as_f64x4!(bias, 1023.0);
    let a = self + (bias + pow2_52);
    let c = cast::<_, i64x4>(a) << 52;
    cast::<_, f64x4>(c)
  }

  /// Calculate the exponent of a packed f64x4
  #[inline]
  #[must_use]
  #[allow(non_upper_case_globals)]
  pub fn exp(self) -> Self {
    const_f64_as_f64x4!(P2, 1.0 / 2.0);
    const_f64_as_f64x4!(P3, 1.0 / 6.0);
    const_f64_as_f64x4!(P4, 1. / 24.);
    const_f64_as_f64x4!(P5, 1. / 120.);
    const_f64_as_f64x4!(P6, 1. / 720.);
    const_f64_as_f64x4!(P7, 1. / 5040.);
    const_f64_as_f64x4!(P8, 1. / 40320.);
    const_f64_as_f64x4!(P9, 1. / 362880.);
    const_f64_as_f64x4!(P10, 1. / 3628800.);
    const_f64_as_f64x4!(P11, 1. / 39916800.);
    const_f64_as_f64x4!(P12, 1. / 479001600.);
    const_f64_as_f64x4!(P13, 1. / 6227020800.);
    const_f64_as_f64x4!(LN2D_HI, 0.693145751953125);
    const_f64_as_f64x4!(LN2D_LO, 1.42860682030941723212E-6);
    let max_x = f64x4::from(708.39);
    let r = (self * Self::LOG2_E).round();
    let x = r.mul_neg_add(LN2D_HI, self);
    let x = r.mul_neg_add(LN2D_LO, x);
    let z =
      polynomial_13!(x, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13);
    let n2 = Self::vm_pow2n(r);
    let z = (z + Self::ONE) * n2;
    // check for overflow
    let in_range = self.abs().cmp_lt(max_x);
    let in_range = in_range & self.is_finite();
    in_range.blend(z, Self::ZERO)
  }

  #[inline]
  #[allow(non_upper_case_globals)]
  fn exponent(self) -> f64x4 {
    const_f64_as_f64x4!(pow2_52, 4503599627370496.0);
    const_f64_as_f64x4!(bias, 1023.0);
    let a = cast::<_, u64x4>(self);
    let b = a >> 52;
    let c = b | cast::<_, u64x4>(pow2_52);
    let d = cast::<_, f64x4>(c);
    let e = d - (pow2_52 + bias);
    e
  }

  #[inline]
  #[allow(non_upper_case_globals)]
  fn fraction_2(self) -> Self {
    let t1 = cast::<_, u64x4>(self);
    let t2 = cast::<_, u64x4>(
      (t1 & u64x4::from(0x000FFFFFFFFFFFFF)) | u64x4::from(0x3FE0000000000000),
    );
    cast::<_, f64x4>(t2)
  }

  fn is_zero_or_subnormal(self) -> Self {
    let t = cast::<_, i64x4>(self);
    let t = t & i64x4::splat(0x7FF0000000000000);
    i64x4::round_float(t.cmp_eq(i64x4::splat(0)))
  }

  fn infinity() -> Self {
    cast::<_, f64x4>(i64x4::splat(0x7FF0000000000000))
  }

  fn nan_log() -> Self {
    cast::<_, f64x4>(i64x4::splat(0x7FF8000000000000 | 0x101 << 29))
  }

  fn nan_pow() -> Self {
    cast::<_, f64x4>(i64x4::splat(0x7FF8000000000000 | 0x101 << 29))
  }

  fn sign_bit(self) -> Self {
    let t1 = cast::<_, i64x4>(self);
    let t2 = t1 >> 63;
    !cast::<_, f64x4>(t2).cmp_eq(f64x4::ZERO)
  }

  pub fn reduce_add(self) -> f64 {
    pick! {
      if #[cfg(target_feature="avx")] {
        // From https://stackoverflow.com/questions/49941645/get-sum-of-values-stored-in-m256d-with-sse-avx
        let lo = cast_to_m128d_from_m256d(self.avx);
        let hi = extract_m128d_from_m256d!(self.avx,1);
        let lo = add_m128d(lo,hi);
        let hi64 = unpack_high_m128d(lo,lo);
        let sum = add_m128d_s(lo,hi64);
        get_f64_from_m128d_s(sum)
      } else if #[cfg(target_feature="sse3")] {
        let a = add_horizontal_m128d(self.sse0, self.sse0);
        let b = add_horizontal_m128d(self.sse1, self.sse1);
        get_f64_from_m128d_s(a) + get_f64_from_m128d_s(b)
      } else if #[cfg(target_feature="sse")] {
        let v0:f64 = self.sse0.to_array().iter().sum();
        let v1:f64 = self.sse1.to_array().iter().sum();
        v0 + v1
      } else {
        self.arr.iter().sum()
      }
    }
  }

  /// Natural log (ln(x))
  #[inline]
  #[must_use]
  #[allow(non_upper_case_globals)]
  pub fn ln(self) -> Self {
    const_f64_as_f64x4!(HALF, 0.5);
    const_f64_as_f64x4!(P0, 7.70838733755885391666E0);
    const_f64_as_f64x4!(P1, 1.79368678507819816313E1);
    const_f64_as_f64x4!(P2, 1.44989225341610930846E1);
    const_f64_as_f64x4!(P3, 4.70579119878881725854E0);
    const_f64_as_f64x4!(P4, 4.97494994976747001425E-1);
    const_f64_as_f64x4!(P5, 1.01875663804580931796E-4);

    const_f64_as_f64x4!(Q0, 2.31251620126765340583E1);
    const_f64_as_f64x4!(Q1, 7.11544750618563894466E1);
    const_f64_as_f64x4!(Q2, 8.29875266912776603211E1);
    const_f64_as_f64x4!(Q3, 4.52279145837532221105E1);
    const_f64_as_f64x4!(Q4, 1.12873587189167450590E1);
    const_f64_as_f64x4!(LN2F_HI, 0.693359375);
    const_f64_as_f64x4!(LN2F_LO, -2.12194440e-4);
    const_f64_as_f64x4!(VM_SQRT2, 1.414213562373095048801);
    const_f64_as_f64x4!(VM_SMALLEST_NORMAL, 1.17549435E-38);

    let x1 = self;
    let x = Self::fraction_2(x1);
    let e = Self::exponent(x1);
    let mask = x.cmp_gt(VM_SQRT2 * HALF);
    let x = (!mask).blend(x + x, x);
    let fe = mask.blend(e + Self::ONE, e);
    let x = x - Self::ONE;
    let px = polynomial_5!(x, P0, P1, P2, P3, P4, P5);
    let x2 = x * x;
    let px = x2 * x * px;
    let qx = polynomial_5n!(x, Q0, Q1, Q2, Q3, Q4, Q5);
    let res = px / qx;
    let res = fe.mul_add(LN2F_LO, res);
    let res = res + x2.mul_neg_add(HALF, x);
    let res = fe.mul_add(LN2F_HI, res);
    let overflow = !self.is_finite();
    let underflow = x1.cmp_lt(VM_SMALLEST_NORMAL);
    let mask = overflow | underflow;
    if !mask.any() {
      res
    } else {
      let iszero = self.is_zero_or_subnormal();
      let res = underflow.blend(Self::nan_log(), res);
      let res = iszero.blend(Self::infinity(), res);
      let res = overflow.blend(self, res);
      res
    }
  }

  #[inline]
  #[must_use]
  pub fn log2(self) -> Self {
    Self::ln(self) * Self::LOG2_E
  }
  #[inline]
  #[must_use]
  pub fn log10(self) -> Self {
    Self::ln(self) * Self::LOG10_E
  }

  #[inline]
  #[must_use]
  #[allow(non_upper_case_globals)]
  pub fn pow_f64x4(self, y: Self) -> Self {
    const_f64_as_f64x4!(ln2d_hi, 0.693145751953125);
    const_f64_as_f64x4!(ln2d_lo, 1.42860682030941723212E-6);
    const_f64_as_f64x4!(P0log, 2.0039553499201281259648E1);
    const_f64_as_f64x4!(P1log, 5.7112963590585538103336E1);
    const_f64_as_f64x4!(P2log, 6.0949667980987787057556E1);
    const_f64_as_f64x4!(P3log, 2.9911919328553073277375E1);
    const_f64_as_f64x4!(P4log, 6.5787325942061044846969E0);
    const_f64_as_f64x4!(P5log, 4.9854102823193375972212E-1);
    const_f64_as_f64x4!(P6log, 4.5270000862445199635215E-5);
    const_f64_as_f64x4!(Q0log, 6.0118660497603843919306E1);
    const_f64_as_f64x4!(Q1log, 2.1642788614495947685003E2);
    const_f64_as_f64x4!(Q2log, 3.0909872225312059774938E2);
    const_f64_as_f64x4!(Q3log, 2.2176239823732856465394E2);
    const_f64_as_f64x4!(Q4log, 8.3047565967967209469434E1);
    const_f64_as_f64x4!(Q5log, 1.5062909083469192043167E1);

    // Taylor expansion constants
    const_f64_as_f64x4!(p2, 1.0 / 2.0); // coefficients for Taylor expansion of exp
    const_f64_as_f64x4!(p3, 1.0 / 6.0);
    const_f64_as_f64x4!(p4, 1.0 / 24.0);
    const_f64_as_f64x4!(p5, 1.0 / 120.0);
    const_f64_as_f64x4!(p6, 1.0 / 720.0);
    const_f64_as_f64x4!(p7, 1.0 / 5040.0);
    const_f64_as_f64x4!(p8, 1.0 / 40320.0);
    const_f64_as_f64x4!(p9, 1.0 / 362880.0);
    const_f64_as_f64x4!(p10, 1.0 / 3628800.0);
    const_f64_as_f64x4!(p11, 1.0 / 39916800.0);
    const_f64_as_f64x4!(p12, 1.0 / 479001600.0);
    const_f64_as_f64x4!(p13, 1.0 / 6227020800.0);

    let x1 = self.abs();
    let x = x1.fraction_2();
    let mask = x.cmp_gt(f64x4::SQRT_2 * f64x4::HALF);
    let x = (!mask).blend(x + x, x);
    let x = x - f64x4::ONE;
    let x2 = x * x;
    let px = polynomial_6!(x, P0log, P1log, P2log, P3log, P4log, P5log, P6log);
    let px = px * x * x2;
    let qx = polynomial_6n!(x, Q0log, Q1log, Q2log, Q3log, Q4log, Q5log);
    let lg1 = px / qx;

    let ef = x1.exponent();
    let ef = mask.blend(ef + f64x4::ONE, ef);
    let e1 = (ef * y).round();
    let yr = ef.mul_sub(y, e1);

    let lg = f64x4::HALF.mul_neg_add(x2, x) + lg1;
    let x2err = (f64x4::HALF * x).mul_sub(x, f64x4::HALF * x2);
    let lgerr = f64x4::HALF.mul_add(x2, lg - x) - lg1;

    let e2 = (lg * y * f64x4::LOG2_E).round();
    let v = lg.mul_sub(y, e2 * ln2d_hi);
    let v = e2.mul_neg_add(ln2d_lo, v);
    let v = v - (lgerr + x2err).mul_sub(y, yr * f64x4::LN_2);

    let x = v;
    let e3 = (x * f64x4::LOG2_E).round();
    let x = e3.mul_neg_add(f64x4::LN_2, x);
    let z =
      polynomial_13m!(x, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13)
        + f64x4::ONE;
    let ee = e1 + e2 + e3;
    let ei = cast::<_, i64x4>(ee.round_int());
    let ej = cast::<_, i64x4>(ei + (cast::<_, i64x4>(z) >> 52));

    let overflow = cast::<_, f64x4>(!ej.cmp_lt(i64x4::splat(0x07FF)))
      | ee.cmp_gt(f64x4::splat(3000.0));
    let underflow = cast::<_, f64x4>(!ej.cmp_gt(i64x4::splat(0x000)))
      | ee.cmp_lt(f64x4::splat(-3000.0));

    // Add exponent by integer addition
    let z = cast::<_, f64x4>(cast::<_, i64x4>(z) + (ei << 52));

    // Check for overflow/underflow
    let z = if (overflow | underflow).any() {
      let z = underflow.blend(f64x4::ZERO, z);
      overflow.blend(Self::infinity(), z)
    } else {
      z
    };

    // Check for self == 0
    let xzero = self.is_zero_or_subnormal();
    let z = xzero.blend(
      y.cmp_lt(f64x4::ZERO).blend(
        Self::infinity(),
        y.cmp_eq(f64x4::ZERO).blend(f64x4::ONE, f64x4::ZERO),
      ),
      z,
    );

    let xsign = self.sign_bit();

    let z = if xsign.any() {
      // Y into an integer
      let yi = y.cmp_eq(y.round());
      // Is y odd?
      let yodd = cast::<_, i64x4>(y.round_int() << 63).round_float();
      let z1 =
        yi.blend(z | yodd, self.cmp_eq(Self::ZERO).blend(z, Self::nan_pow()));
      xsign.blend(z1, z)
    } else {
      z
    };

    let xfinite = self.is_finite();
    let yfinite = y.is_finite();
    let efinite = ee.is_finite();

    if (xfinite & yfinite & (efinite | xzero)).all() {
      return z;
    }

    (self.is_nan() | y.is_nan()).blend(self + y, z)
  }

  pub fn powf(self, y: f64) -> Self {
    Self::pow_f64x4(self, f64x4::splat(y))
  }
}

impl Not for f64x4 {
  type Output = Self;
  fn not(self) -> Self {
    pick! {
      if #[cfg(target_feature="avx")] {
        Self { avx: self.avx.not()  }
      } else if #[cfg(target_feature="sse2")] {
        Self { sse0: self.sse0.not() , sse1: self.sse1.not() }
      } else {
        Self { arr: [
          (self.arr[0].to_bits() ^ u64::MAX) as f64,
          (self.arr[1].to_bits() ^ u64::MAX) as f64,
          (self.arr[2].to_bits() ^ u64::MAX) as f64,
          (self.arr[3].to_bits() ^ u64::MAX) as f64,
        ]}
      }
    }
  }
}
