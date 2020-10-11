use super::*;

pick! {
  if #[cfg(target_feature="sse")] {
    #[derive(Default, Clone, Copy, PartialEq)]
    #[repr(C, align(16))]
    pub struct f32x4 { sse: m128 }
  } else {
    #[derive(Default, Clone, Copy, PartialEq)]
    #[repr(C, align(16))]
    pub struct f32x4 { arr: [f32;4] }
  }
}

macro_rules! const_f32_as_f32x4 {
  ($i:ident, $f:expr) => {
    pub const $i: f32x4 =
      unsafe { ConstUnionHack128bit { f32a4: [$f, $f, $f, $f] }.f32x4 };
  };
}

impl f32x4 {
  const_f32_as_f32x4!(ONE, 1.0);
  const_f32_as_f32x4!(ZERO, 0.0);
  const_f32_as_f32x4!(HALF, 0.5);
  const_f32_as_f32x4!(E, core::f32::consts::E);
  const_f32_as_f32x4!(FRAC_1_PI, core::f32::consts::FRAC_1_PI);
  const_f32_as_f32x4!(FRAC_2_PI, core::f32::consts::FRAC_2_PI);
  const_f32_as_f32x4!(FRAC_2_SQRT_PI, core::f32::consts::FRAC_2_SQRT_PI);
  const_f32_as_f32x4!(FRAC_1_SQRT_2, core::f32::consts::FRAC_1_SQRT_2);
  const_f32_as_f32x4!(FRAC_PI_2, core::f32::consts::FRAC_PI_2);
  const_f32_as_f32x4!(FRAC_PI_3, core::f32::consts::FRAC_PI_3);
  const_f32_as_f32x4!(FRAC_PI_4, core::f32::consts::FRAC_PI_4);
  const_f32_as_f32x4!(FRAC_PI_6, core::f32::consts::FRAC_PI_6);
  const_f32_as_f32x4!(FRAC_PI_8, core::f32::consts::FRAC_PI_8);
  const_f32_as_f32x4!(LN_2, core::f32::consts::LN_2);
  const_f32_as_f32x4!(LN_10, core::f32::consts::LN_10);
  const_f32_as_f32x4!(LOG2_E, core::f32::consts::LOG2_E);
  const_f32_as_f32x4!(LOG10_E, core::f32::consts::LOG10_E);
  const_f32_as_f32x4!(LOG10_2, core::f32::consts::LOG10_2);
  const_f32_as_f32x4!(LOG2_10, core::f32::consts::LOG2_10);
  const_f32_as_f32x4!(PI, core::f32::consts::PI);
  const_f32_as_f32x4!(SQRT_2, core::f32::consts::SQRT_2);
  const_f32_as_f32x4!(TAU, 6.28318530717958647692528676655900577_f32);
}

unsafe impl Zeroable for f32x4 {}
unsafe impl Pod for f32x4 {}

impl Add for f32x4 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn add(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse")] {
        Self { sse: add_m128(self.sse, rhs.sse) }
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

impl Sub for f32x4 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn sub(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse")] {
        Self { sse: sub_m128(self.sse, rhs.sse) }
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

impl Mul for f32x4 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn mul(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse")] {
        Self { sse: mul_m128(self.sse, rhs.sse) }
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

impl Div for f32x4 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn div(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse")] {
        Self { sse: div_m128(self.sse, rhs.sse) }
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

impl Add<f32> for f32x4 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn add(self, rhs: f32) -> Self::Output {
    self.add(Self::splat(rhs))
  }
}

impl Sub<f32> for f32x4 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn sub(self, rhs: f32) -> Self::Output {
    self.sub(Self::splat(rhs))
  }
}

impl Mul<f32> for f32x4 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn mul(self, rhs: f32) -> Self::Output {
    self.mul(Self::splat(rhs))
  }
}

impl Div<f32> for f32x4 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn div(self, rhs: f32) -> Self::Output {
    self.div(Self::splat(rhs))
  }
}

impl Add<f32x4> for f32 {
  type Output = f32x4;
  #[inline]
  #[must_use]
  fn add(self, rhs: f32x4) -> Self::Output {
    f32x4::splat(self).add(rhs)
  }
}

impl Sub<f32x4> for f32 {
  type Output = f32x4;
  #[inline]
  #[must_use]
  fn sub(self, rhs: f32x4) -> Self::Output {
    f32x4::splat(self).sub(rhs)
  }
}

impl Mul<f32x4> for f32 {
  type Output = f32x4;
  #[inline]
  #[must_use]
  fn mul(self, rhs: f32x4) -> Self::Output {
    f32x4::splat(self).mul(rhs)
  }
}

impl Div<f32x4> for f32 {
  type Output = f32x4;
  #[inline]
  #[must_use]
  fn div(self, rhs: f32x4) -> Self::Output {
    f32x4::splat(self).div(rhs)
  }
}

impl BitAnd for f32x4 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn bitand(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse")] {
        Self { sse: bitand_m128(self.sse, rhs.sse) }
      } else {
        Self { arr: [
          f32::from_bits(self.arr[0].to_bits() & rhs.arr[0].to_bits()),
          f32::from_bits(self.arr[1].to_bits() & rhs.arr[1].to_bits()),
          f32::from_bits(self.arr[2].to_bits() & rhs.arr[2].to_bits()),
          f32::from_bits(self.arr[3].to_bits() & rhs.arr[3].to_bits()),
        ]}
      }
    }
  }
}

impl BitOr for f32x4 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn bitor(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse")] {
        Self { sse: bitor_m128(self.sse, rhs.sse) }
      } else {
        Self { arr: [
          f32::from_bits(self.arr[0].to_bits() | rhs.arr[0].to_bits()),
          f32::from_bits(self.arr[1].to_bits() | rhs.arr[1].to_bits()),
          f32::from_bits(self.arr[2].to_bits() | rhs.arr[2].to_bits()),
          f32::from_bits(self.arr[3].to_bits() | rhs.arr[3].to_bits()),
        ]}
      }
    }
  }
}

impl BitXor for f32x4 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn bitxor(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse")] {
        Self { sse: bitxor_m128(self.sse, rhs.sse) }
      } else {
        Self { arr: [
          f32::from_bits(self.arr[0].to_bits() ^ rhs.arr[0].to_bits()),
          f32::from_bits(self.arr[1].to_bits() ^ rhs.arr[1].to_bits()),
          f32::from_bits(self.arr[2].to_bits() ^ rhs.arr[2].to_bits()),
          f32::from_bits(self.arr[3].to_bits() ^ rhs.arr[3].to_bits()),
        ]}
      }
    }
  }
}

impl CmpEq for f32x4 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn cmp_eq(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse")] {
        Self { sse: cmp_eq_mask_m128(self.sse, rhs.sse) }
      } else {
        Self { arr: [
          if self.arr[0] == rhs.arr[0] { f32::from_bits(u32::MAX) } else { 0.0 },
          if self.arr[1] == rhs.arr[1] { f32::from_bits(u32::MAX) } else { 0.0 },
          if self.arr[2] == rhs.arr[2] { f32::from_bits(u32::MAX) } else { 0.0 },
          if self.arr[3] == rhs.arr[3] { f32::from_bits(u32::MAX) } else { 0.0 },
        ]}
      }
    }
  }
}

impl CmpGe for f32x4 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn cmp_ge(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse")] {
        Self { sse: cmp_ge_mask_m128(self.sse, rhs.sse) }
      } else {
        Self { arr: [
          if self.arr[0] >= rhs.arr[0] { f32::from_bits(u32::MAX) } else { 0.0 },
          if self.arr[1] >= rhs.arr[1] { f32::from_bits(u32::MAX) } else { 0.0 },
          if self.arr[2] >= rhs.arr[2] { f32::from_bits(u32::MAX) } else { 0.0 },
          if self.arr[3] >= rhs.arr[3] { f32::from_bits(u32::MAX) } else { 0.0 },
        ]}
      }
    }
  }
}

impl CmpGt for f32x4 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn cmp_gt(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse")] {
        Self { sse: cmp_gt_mask_m128(self.sse, rhs.sse) }
      } else {
        Self { arr: [
          if self.arr[0] > rhs.arr[0] { f32::from_bits(u32::MAX) } else { 0.0 },
          if self.arr[1] > rhs.arr[1] { f32::from_bits(u32::MAX) } else { 0.0 },
          if self.arr[2] > rhs.arr[2] { f32::from_bits(u32::MAX) } else { 0.0 },
          if self.arr[3] > rhs.arr[3] { f32::from_bits(u32::MAX) } else { 0.0 },
        ]}
      }
    }
  }
}

impl CmpNe for f32x4 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn cmp_ne(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse")] {
        Self { sse: cmp_neq_mask_m128(self.sse, rhs.sse) }
      } else {
        Self { arr: [
          if self.arr[0] != rhs.arr[0] { f32::from_bits(u32::MAX) } else { 0.0 },
          if self.arr[1] != rhs.arr[1] { f32::from_bits(u32::MAX) } else { 0.0 },
          if self.arr[2] != rhs.arr[2] { f32::from_bits(u32::MAX) } else { 0.0 },
          if self.arr[3] != rhs.arr[3] { f32::from_bits(u32::MAX) } else { 0.0 },
        ]}
      }
    }
  }
}

impl CmpLe for f32x4 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn cmp_le(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse")] {
        Self { sse: cmp_le_mask_m128(self.sse, rhs.sse) }
      } else {
        Self { arr: [
          if self.arr[0] <= rhs.arr[0] { f32::from_bits(u32::MAX) } else { 0.0 },
          if self.arr[1] <= rhs.arr[1] { f32::from_bits(u32::MAX) } else { 0.0 },
          if self.arr[2] <= rhs.arr[2] { f32::from_bits(u32::MAX) } else { 0.0 },
          if self.arr[3] <= rhs.arr[3] { f32::from_bits(u32::MAX) } else { 0.0 },
        ]}
      }
    }
  }
}

impl CmpLt for f32x4 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn cmp_lt(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse")] {
        Self { sse: cmp_lt_mask_m128(self.sse, rhs.sse) }
      } else {
        Self { arr: [
          if self.arr[0] < rhs.arr[0] { f32::from_bits(u32::MAX) } else { 0.0 },
          if self.arr[1] < rhs.arr[1] { f32::from_bits(u32::MAX) } else { 0.0 },
          if self.arr[2] < rhs.arr[2] { f32::from_bits(u32::MAX) } else { 0.0 },
          if self.arr[3] < rhs.arr[3] { f32::from_bits(u32::MAX) } else { 0.0 },
        ]}
      }
    }
  }
}

impl f32x4 {
  #[inline]
  #[must_use]
  pub fn blend(self, t: Self, f: Self) -> Self {
    pick! {
      if #[cfg(target_feature="sse4.1")] {
        Self { sse: blend_varying_m128(f.sse, t.sse, self.sse) }
      } else {
        generic_bit_blend(self, t, f)
      }
    }
  }
  #[inline]
  #[must_use]
  pub fn abs(self) -> Self {
    let non_sign_bits = f32x4::from(f32::from_bits(i32::MAX as u32));
    self & non_sign_bits
  }
  #[inline]
  #[must_use]
  pub fn max(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="sse")] {
        Self { sse: max_m128(self.sse, rhs.sse) }
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
      if #[cfg(target_feature="sse")] {
        Self { sse: min_m128(self.sse, rhs.sse) }
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
      if #[cfg(target_feature="sse")] {
        Self { sse: cmp_unord_mask_m128(self.sse, self.sse) }
      } else {
        Self { arr: [
          if self.arr[0].is_nan() { f32::from_bits(u32::MAX) } else { 0.0 },
          if self.arr[1].is_nan() { f32::from_bits(u32::MAX) } else { 0.0 },
          if self.arr[2].is_nan() { f32::from_bits(u32::MAX) } else { 0.0 },
          if self.arr[3].is_nan() { f32::from_bits(u32::MAX) } else { 0.0 },
        ]}
      }
    }
  }
  #[inline]
  #[must_use]
  pub fn is_finite(self) -> Self {
    let shifted_exp_mask = u32x4::from(0xFF000000);
    let u: u32x4 = cast(self);
    let shift_u = u << 1_u64;
    let out = !(shift_u & shifted_exp_mask).cmp_eq(shifted_exp_mask);
    cast(out)
  }

  #[inline]
  #[must_use]
  pub fn round(self) -> Self {
    pick! {
      if #[cfg(target_feature="sse4.1")] {
        Self { sse: round_m128!(self.sse, Nearest) }
      } else if #[cfg(target_feature="sse2")] {
        let mi: m128i = convert_to_i32_m128i_from_m128(self.sse);
        let f: f32x4 = f32x4 { sse: convert_to_m128_from_i32_m128i(mi) };
        let i: i32x4 = cast(mi);
        let mask: f32x4 = cast(i.cmp_eq(i32x4::from(0x80000000_u32 as i32)));
        mask.blend(self, f)
      } else {
        // Note(Lokathor): This software fallback is probably very slow compared
        // to having a hardware option available, even just the sse2 version is
        // better than this. Oh well.
        let to_int = f32x4::from(1.0 / f32::EPSILON);
        let u: u32x4 = cast(self);
        let e: i32x4 = cast((u >> 23) & u32x4::from(0xff));
        let mut y: f32x4;

        let no_op_magic = i32x4::from(0x7f + 23);
        let no_op_mask: f32x4 = cast(e.cmp_gt(no_op_magic) | e.cmp_eq(no_op_magic));
        let no_op_val: f32x4 = self;

        let zero_magic = i32x4::from(0x7f - 1);
        let zero_mask: f32x4 = cast(e.cmp_lt(zero_magic));
        let zero_val: f32x4 = self * f32x4::from(0.0);

        let neg_bit: f32x4 = cast(cast::<u32x4, i32x4>(u).cmp_lt(i32x4::default()));
        let x: f32x4 = neg_bit.blend(-self, self);
        y = x + to_int - to_int - x;
        y = y.cmp_gt(f32x4::from(0.5)).blend(
          y + x - f32x4::from(-1.0),
          y.cmp_lt(f32x4::from(-0.5)).blend(y + x + f32x4::from(1.0), y + x),
        );
        y = neg_bit.blend(-y, y);

        no_op_mask.blend(no_op_val, zero_mask.blend(zero_val, y))
      }
    }
  }
  #[inline]
  #[must_use]
  pub fn round_int(self) -> i32x4 {
    pick! {
      if #[cfg(target_feature="sse2")] {
        cast(convert_to_i32_m128i_from_m128(self.sse))
      } else {
        let rounded: [f32;4] = cast(self.round());
        let rounded_ints: i32x4 = cast([
          rounded[0] as i32,
          rounded[1] as i32,
          rounded[2] as i32,
          rounded[3] as i32,
        ]);
        cast::<f32x4, i32x4>(self.is_finite()).blend(
          rounded_ints,
          i32x4::from(i32::MIN)
        )
      }
    }
  }
  // On i586, `f32::trunc` in the standard library will give wrong results.
  #[cfg(any(
    all(
      any(target_arch = "x86", target_arch = "x86_64"),
      target_feature = "sse"
    ),
    all(
      not(any(target_arch = "x86", target_arch = "x86_64")),
      feature = "std"
    ),
  ))]
  #[inline]
  #[must_use]
  pub fn trunc_int(self) -> i32x4 {
    pick! {
      if #[cfg(target_feature="sse2")] {
        cast(truncate_m128_to_m128i(self.sse))
      } else {
        let rounded: [f32;4] = cast(self.round());
        let rounded_ints: i32x4 = cast([
          rounded[0].trunc() as i32,
          rounded[1].trunc() as i32,
          rounded[2].trunc() as i32,
          rounded[3].trunc() as i32,
        ]);
        cast::<f32x4, i32x4>(self.is_finite()).blend(
          rounded_ints,
          i32x4::from(i32::MIN)
        )
      }
    }
  }
  #[inline]
  #[must_use]
  pub fn mul_add(self, m: Self, a: Self) -> Self {
    pick! {
      if #[cfg(all(target_feature="sse2",target_feature="fma"))] {
        Self { sse: fused_mul_add_m128(self.sse, m.sse, a.sse) }
      } else {
        (self * m) + a
      }
    }
  }

  #[inline]
  #[must_use]
  pub fn mul_sub(self, m: Self, s: Self) -> Self {
    pick! {
      if #[cfg(all(target_feature="sse2",target_feature="fma"))] {
        Self { sse: fused_mul_sub_m128(self.sse, m.sse, s.sse) }
      } else {
        (self * m) - s
      }
    }
  }

  #[inline]
  #[must_use]
  pub fn mul_neg_add(self, m: Self, a: Self) -> Self {
    pick! {
      if #[cfg(all(target_feature="sse2",target_feature="fma"))] {
        Self { sse: fused_mul_neg_add_m128(self.sse, m.sse, a.sse) }
      } else {
        a - (self * m)
      }
    }
  }

  #[inline]
  #[must_use]
  pub fn mul_neg_sub(self, m: Self, a: Self) -> Self {
    pick! {
      if #[cfg(all(target_feature="sse2",target_feature="fma"))] {
        Self { sse: fused_mul_neg_sub_m128(self.sse, m.sse, a.sse) }
      } else {
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
    const_f32_as_f32x4!(P4asinf, 4.2163199048E-2);
    const_f32_as_f32x4!(P3asinf, 2.4181311049E-2);
    const_f32_as_f32x4!(P2asinf, 4.5470025998E-2);
    const_f32_as_f32x4!(P1asinf, 7.4953002686E-2);
    const_f32_as_f32x4!(P0asinf, 1.6666752422E-1);

    let xa = self.abs();
    let big = xa.cmp_ge(f32x4::splat(0.5));

    let x1 = f32x4::splat(0.5) * (f32x4::ONE - xa);
    let x2 = xa * xa;
    let x3 = big.blend(x1, x2);

    let xb = x1.sqrt();

    let x4 = big.blend(xb, xa);

    let z = polynomial_4!(x3, P0asinf, P1asinf, P2asinf, P3asinf, P4asinf);
    let z = z.mul_add(x3 * x4, x4);

    let z1 = z + z;

    // acos
    let z3 = self.cmp_lt(f32x4::ZERO).blend(f32x4::PI - z1, z1);
    let z4 = f32x4::FRAC_PI_2 - z.flip_signs(self);
    let acos = big.blend(z3, z4);

    // asin
    let z3 = f32x4::FRAC_PI_2 - z1;
    let asin = big.blend(z3, z);
    let asin = asin.flip_signs(self);

    (asin, acos)
  }

  #[allow(non_upper_case_globals)]
  pub fn asin(self) -> Self {
    // Based on the Agner Fog "vector class library":
    // https://github.com/vectorclass/version2/blob/master/vectormath_trig.h
    const_f32_as_f32x4!(P4asinf, 4.2163199048E-2);
    const_f32_as_f32x4!(P3asinf, 2.4181311049E-2);
    const_f32_as_f32x4!(P2asinf, 4.5470025998E-2);
    const_f32_as_f32x4!(P1asinf, 7.4953002686E-2);
    const_f32_as_f32x4!(P0asinf, 1.6666752422E-1);

    let xa = self.abs();
    let big = xa.cmp_ge(f32x4::splat(0.5));

    let x1 = f32x4::splat(0.5) * (f32x4::ONE - xa);
    let x2 = xa * xa;
    let x3 = big.blend(x1, x2);

    let xb = x1.sqrt();

    let x4 = big.blend(xb, xa);

    let z = polynomial_4!(x3, P0asinf, P1asinf, P2asinf, P3asinf, P4asinf);
    let z = z.mul_add(x3 * x4, x4);

    let z1 = z + z;

    // asin
    let z3 = f32x4::FRAC_PI_2 - z1;
    let asin = big.blend(z3, z);
    let asin = asin.flip_signs(self);

    asin
  }

  #[inline]
  #[must_use]
  #[allow(non_upper_case_globals)]
  pub fn acos(self) -> Self {
    // Based on the Agner Fog "vector class library":
    // https://github.com/vectorclass/version2/blob/master/vectormath_trig.h
    const_f32_as_f32x4!(P4asinf, 4.2163199048E-2);
    const_f32_as_f32x4!(P3asinf, 2.4181311049E-2);
    const_f32_as_f32x4!(P2asinf, 4.5470025998E-2);
    const_f32_as_f32x4!(P1asinf, 7.4953002686E-2);
    const_f32_as_f32x4!(P0asinf, 1.6666752422E-1);

    let xa = self.abs();
    let big = xa.cmp_ge(f32x4::splat(0.5));

    let x1 = f32x4::splat(0.5) * (f32x4::ONE - xa);
    let x2 = xa * xa;
    let x3 = big.blend(x1, x2);

    let xb = x1.sqrt();

    let x4 = big.blend(xb, xa);

    let z = polynomial_4!(x3, P0asinf, P1asinf, P2asinf, P3asinf, P4asinf);
    let z = z.mul_add(x3 * x4, x4);

    let z1 = z + z;

    // acos
    let z3 = self.cmp_lt(f32x4::ZERO).blend(f32x4::PI - z1, z1);
    let z4 = f32x4::FRAC_PI_2 - z.flip_signs(self);
    let acos = big.blend(z3, z4);

    acos
  }

  #[inline]
  #[must_use]
  #[allow(non_upper_case_globals)]
  pub fn sin_cos(self) -> (Self, Self) {
    // Based on the Agner Fog "vector class library":
    // https://github.com/vectorclass/version2/blob/master/vectormath_trig.h

    const_f32_as_f32x4!(DP1F, 0.78515625_f32 * 2.0);
    const_f32_as_f32x4!(DP2F, 2.4187564849853515625E-4_f32 * 2.0);
    const_f32_as_f32x4!(DP3F, 3.77489497744594108E-8_f32 * 2.0);

    const_f32_as_f32x4!(P0sinf, -1.6666654611E-1);
    const_f32_as_f32x4!(P1sinf, 8.3321608736E-3);
    const_f32_as_f32x4!(P2sinf, -1.9515295891E-4);

    const_f32_as_f32x4!(P0cosf, 4.166664568298827E-2);
    const_f32_as_f32x4!(P1cosf, -1.388731625493765E-3);
    const_f32_as_f32x4!(P2cosf, 2.443315711809948E-5);

    const_f32_as_f32x4!(TWO_OVER_PI, 2.0 / core::f32::consts::PI);

    let xa = self.abs();

    // Find quadrant
    let y = (xa * TWO_OVER_PI).round();
    let q: i32x4 = y.round_int();

    let x = y.mul_neg_add(DP3F, y.mul_neg_add(DP2F, y.mul_neg_add(DP1F, xa)));

    let x2 = x * x;
    let mut s = polynomial_2!(x2, P0sinf, P1sinf, P2sinf) * (x * x2) + x;
    let mut c = polynomial_2!(x2, P0cosf, P1cosf, P2cosf) * (x2 * x2)
      + f32x4::from(0.5).mul_neg_add(x2, f32x4::from(1.0));

    let swap = !(q & i32x4::from(1)).cmp_eq(i32x4::from(0));

    let mut overflow: f32x4 = cast(q.cmp_gt(i32x4::from(0x2000000)));
    overflow &= xa.is_finite();
    s = overflow.blend(f32x4::from(0.0), s);
    c = overflow.blend(f32x4::from(1.0), c);

    // calc sin
    let mut sin1 = cast::<_, f32x4>(swap).blend(c, s);
    let sign_sin: i32x4 = (q << 30) ^ cast::<_, i32x4>(self);
    sin1 = sin1.flip_signs(cast(sign_sin));

    // calc cos
    let mut cos1 = cast::<_, f32x4>(swap).blend(s, c);
    let sign_cos: i32x4 = ((q + i32x4::from(1)) & i32x4::from(2)) << 30;
    cos1 ^= cast::<_, f32x4>(sign_cos);

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
    const_f32_as_f32x4!(RAD_TO_DEG_RATIO, 180.0_f32 / core::f32::consts::PI);
    self * RAD_TO_DEG_RATIO
  }
  #[inline]
  #[must_use]
  pub fn to_radians(self) -> Self {
    const_f32_as_f32x4!(DEG_TO_RAD_RATIO, core::f32::consts::PI / 180.0_f32);
    self * DEG_TO_RAD_RATIO
  }
  #[inline]
  #[must_use]
  pub fn sqrt(self) -> Self {
    pick! {
      if #[cfg(target_feature="sse")] {
        Self { sse: sqrt_m128(self.sse) }
      } else if #[cfg(feature="std")] {
        Self { arr: [
          self.arr[0].sqrt(),
          self.arr[1].sqrt(),
          self.arr[2].sqrt(),
          self.arr[3].sqrt(),
        ]}
      } else {
        Self { arr: [
          software_sqrt(self.arr[0] as f64) as f32,
          software_sqrt(self.arr[1] as f64) as f32,
          software_sqrt(self.arr[2] as f64) as f32,
          software_sqrt(self.arr[3] as f64) as f32,
        ]}
      }
    }
  }
  #[inline]
  #[must_use]
  pub fn move_mask(self) -> i32 {
    pick! {
      if #[cfg(target_feature="sse")] {
        move_mask_m128(self.sse)
      } else {
        (((self.arr[0].to_bits() as i32) < 0) as i32) << 0 |
        (((self.arr[1].to_bits() as i32) < 0) as i32) << 1 |
        (((self.arr[2].to_bits() as i32) < 0) as i32) << 2 |
        (((self.arr[3].to_bits() as i32) < 0) as i32) << 3
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
    // four lanes
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
    const_f32_as_f32x4!(pow2_23, 8388608.0);
    const_f32_as_f32x4!(bias, 127.0);
    let a = self + (bias + pow2_23);
    let c = cast::<_, i32x4>(a) << 23;
    cast::<_, f32x4>(c)
  }

  /// Calculate the exponent of a packed f32x4
  #[inline]
  #[must_use]
  #[allow(non_upper_case_globals)]
  pub fn exp(self) -> Self {
    const_f32_as_f32x4!(P0, 1.0 / 2.0);
    const_f32_as_f32x4!(P1, 1.0 / 6.0);
    const_f32_as_f32x4!(P2, 1. / 24.);
    const_f32_as_f32x4!(P3, 1. / 120.);
    const_f32_as_f32x4!(P4, 1. / 720.);
    const_f32_as_f32x4!(P5, 1. / 5040.);
    const_f32_as_f32x4!(LN2D_HI, 0.693359375);
    const_f32_as_f32x4!(LN2D_LO, -2.12194440e-4);
    let max_x = f32x4::from(87.3);
    let r = (self * Self::LOG2_E).round();
    let x = r.mul_neg_add(LN2D_HI, self);
    let x = r.mul_neg_add(LN2D_LO, x);
    let z = polynomial_5!(x, P0, P1, P2, P3, P4, P5);
    let x2 = x * x;
    let z = z.mul_add(x2, x);
    let n2 = Self::vm_pow2n(r);
    let z = (z + Self::ONE) * n2;
    // check for overflow
    let in_range = self.abs().cmp_lt(max_x);
    let in_range = in_range & self.is_finite();
    in_range.blend(z, Self::ZERO)
  }

  #[inline]
  #[allow(non_upper_case_globals)]
  fn exponent(self) -> f32x4 {
    const_f32_as_f32x4!(pow2_23, 8388608.0);
    const_f32_as_f32x4!(bias, 127.0);
    let a = cast::<_, u32x4>(self);
    let b = a >> 23;
    let c = b | cast::<_, u32x4>(pow2_23);
    let d = cast::<_, f32x4>(c);
    let e = d - (pow2_23 + bias);
    e
  }

  #[inline]
  #[allow(non_upper_case_globals)]
  fn fraction_2(self) -> Self {
    let t1 = cast::<_, u32x4>(self);
    let t2 = cast::<_, u32x4>(
      (t1 & u32x4::from(0x007FFFFF)) | u32x4::from(0x3F000000),
    );
    cast::<_, f32x4>(t2)
  }

  fn is_zero_or_subnormal(self) -> Self {
    let t = cast::<_, i32x4>(self);
    let t = t & i32x4::splat(0x7F800000);
    i32x4::round_float(t.cmp_eq(i32x4::splat(0)))
  }

  fn infinity() -> Self {
    cast::<_, f32x4>(i32x4::splat(0x7F800000))
  }

  fn nan_log() -> Self {
    cast::<_, f32x4>(i32x4::splat(0x7FC00000 | 0x101 & 0x003FFFFF))
  }

  fn nan_pow() -> Self {
    cast::<_, f32x4>(i32x4::splat(0x7FC00000 | 0x101 & 0x003FFFFF))
  }

  pub fn sign_bit(self) -> Self {
    let t1 = cast::<_, i32x4>(self);
    let t2 = t1 >> 31;
    !cast::<_, f32x4>(t2).cmp_eq(f32x4::ZERO)
  }

  pub fn reduce_add(self) -> f32 {
    pick! {
    if #[cfg(target_feature="sse")] {
          let v = self.sse.to_array();
          v.iter().sum()
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
    const_f32_as_f32x4!(HALF, 0.5);
    const_f32_as_f32x4!(P0, 3.3333331174E-1);
    const_f32_as_f32x4!(P1, -2.4999993993E-1);
    const_f32_as_f32x4!(P2, 2.0000714765E-1);
    const_f32_as_f32x4!(P3, -1.6668057665E-1);
    const_f32_as_f32x4!(P4, 1.4249322787E-1);
    const_f32_as_f32x4!(P5, -1.2420140846E-1);
    const_f32_as_f32x4!(P6, 1.1676998740E-1);
    const_f32_as_f32x4!(P7, -1.1514610310E-1);
    const_f32_as_f32x4!(P8, 7.0376836292E-2);
    const_f32_as_f32x4!(LN2F_HI, 0.693359375);
    const_f32_as_f32x4!(LN2F_LO, -2.12194440e-4);
    const_f32_as_f32x4!(VM_SMALLEST_NORMAL, 1.17549435E-38);

    let x1 = self;
    let x = Self::fraction_2(x1);
    let e = Self::exponent(x1);
    let mask = x.cmp_gt(Self::SQRT_2 * HALF);
    let x = (!mask).blend(x + x, x);
    let fe = mask.blend(e + Self::ONE, e);
    let x = x - Self::ONE;
    let res = polynomial_8!(x, P0, P1, P2, P3, P4, P5, P6, P7, P8);
    let x2 = x * x;
    let res = x2 * x * res;
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
  pub fn pow_f32x4(self, y: f32x4) -> Self {
    const_f32_as_f32x4!(ln2f_hi, 0.693359375);
    const_f32_as_f32x4!(ln2f_lo, -2.12194440e-4);
    const_f32_as_f32x4!(P0logf, 3.3333331174E-1);
    const_f32_as_f32x4!(P1logf, -2.4999993993E-1);
    const_f32_as_f32x4!(P2logf, 2.0000714765E-1);
    const_f32_as_f32x4!(P3logf, -1.6668057665E-1);
    const_f32_as_f32x4!(P4logf, 1.4249322787E-1);
    const_f32_as_f32x4!(P5logf, -1.2420140846E-1);
    const_f32_as_f32x4!(P6logf, 1.1676998740E-1);
    const_f32_as_f32x4!(P7logf, -1.1514610310E-1);
    const_f32_as_f32x4!(P8logf, 7.0376836292E-2);

    const_f32_as_f32x4!(p2expf, 1.0 / 2.0); // coefficients for Taylor expansion of exp
    const_f32_as_f32x4!(p3expf, 1.0 / 6.0);
    const_f32_as_f32x4!(p4expf, 1.0 / 24.0);
    const_f32_as_f32x4!(p5expf, 1.0 / 120.0);
    const_f32_as_f32x4!(p6expf, 1.0 / 720.0);
    const_f32_as_f32x4!(p7expf, 1.0 / 5040.0);

    let x1 = self.abs();
    let x = x1.fraction_2();

    let mask = x.cmp_gt(f32x4::SQRT_2 * f32x4::HALF);
    let x = (!mask).blend(x + x, x);

    let x = x - f32x4::ONE;
    let x2 = x * x;
    let lg1 = polynomial_8!(
      x, P0logf, P1logf, P2logf, P3logf, P4logf, P5logf, P6logf, P7logf, P8logf
    );
    let lg1 = lg1 * x2 * x;

    let ef = x1.exponent();
    let ef = mask.blend(ef + f32x4::ONE, ef);

    let e1 = (ef * y).round();
    let yr = ef.mul_sub(y, e1);

    let lg = f32x4::HALF.mul_neg_add(x2, x) + lg1;
    let x2err = (f32x4::HALF * x).mul_sub(x, f32x4::HALF * x2);
    let lgerr = f32x4::HALF.mul_add(x2, lg - x) - lg1;

    let e2 = (lg * y * f32x4::LOG2_E).round();
    let v = lg.mul_sub(y, e2 * ln2f_hi);
    let v = e2.mul_neg_add(ln2f_lo, v);
    let v = v - (lgerr + x2err).mul_sub(y, yr * f32x4::LN_2);

    let x = v;
    let e3 = (x * f32x4::LOG2_E).round();
    let x = e3.mul_neg_add(f32x4::LN_2, x);
    let x2 = x * x;
    let z = x2.mul_add(
      polynomial_5!(x, p2expf, p3expf, p4expf, p5expf, p6expf, p7expf),
      x + f32x4::ONE,
    );

    let ee = e1 + e2 + e3;
    let ei = cast::<_, i32x4>(ee.round_int());
    let ej = cast::<_, i32x4>(ei + (cast::<_, i32x4>(z) >> 23));

    let overflow = cast::<_, f32x4>(ej.cmp_gt(i32x4::splat(0x0FF)))
      | (ee.cmp_gt(f32x4::splat(300.0)));
    let underflow = cast::<_, f32x4>(ej.cmp_lt(i32x4::splat(0x000)))
      | (ee.cmp_lt(f32x4::splat(-300.0)));

    // Add exponent by integer addition
    let z = cast::<_, f32x4>(cast::<_, i32x4>(z) + (ei << 23));

    // Check for overflow/underflow
    let z = if (overflow | underflow).any() {
      let z = underflow.blend(f32x4::ZERO, z);
      overflow.blend(Self::infinity(), z)
    } else {
      z
    };

    // Check for self == 0
    let xzero = self.is_zero_or_subnormal();
    let z = xzero.blend(
      y.cmp_lt(f32x4::ZERO).blend(
        Self::infinity(),
        y.cmp_eq(f32x4::ZERO).blend(f32x4::ONE, f32x4::ZERO),
      ),
      z,
    );

    let xsign = self.sign_bit();
    let z = if xsign.any() {
      // Y into an integer
      let yi = y.cmp_eq(y.round());
      // Is y odd?
      let yodd = cast::<_, i32x4>(y.round_int() << 31).round_float();

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

  pub fn powf(self, y: f32) -> Self {
    Self::pow_f32x4(self, f32x4::splat(y))
  }
}
