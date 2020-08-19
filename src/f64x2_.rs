use super::*;

pick! {
  if #[cfg(target_feature="sse2")] {
    #[derive(Default, Clone, Copy, PartialEq)]
    #[repr(C, align(16))]
    pub struct f64x2 { sse: m128d }
  } else {
    #[derive(Default, Clone, Copy, PartialEq)]
    #[repr(C, align(16))]
    pub struct f64x2 { arr: [f64;2] }
  }
}

macro_rules! const_f64_as_f64x2 {
  ($i:ident, $f:expr) => {
    pub const $i: f64x2 =
      unsafe { ConstUnionHack128bit { f64a2: [$f, $f] }.f64x2 };
  };
}

macro_rules! polynomial_5 {
  ($x:expr, $c0:expr, $c1:expr, $c2:expr, $c3:expr, $c4:expr, $c5:expr $(,)?) => {{
    let x = $x;
    let x2 = x * x;
    let x4 = x2 * x2;
    $c3
      .mul_add(x, $c2)
      .mul_add(x2, $c5.mul_add(x, $c4).mul_add(x4, $c1.mul_add(x, $c0)))
  }};
}

impl f64x2 {
  const_f64_as_f64x2!(E, core::f64::consts::E);
  const_f64_as_f64x2!(FRAC_1_PI, core::f64::consts::FRAC_1_PI);
  const_f64_as_f64x2!(FRAC_2_PI, core::f64::consts::FRAC_2_PI);
  const_f64_as_f64x2!(FRAC_2_SQRT_PI, core::f64::consts::FRAC_2_SQRT_PI);
  const_f64_as_f64x2!(FRAC_1_SQRT_2, core::f64::consts::FRAC_1_SQRT_2);
  const_f64_as_f64x2!(FRAC_PI_2, core::f64::consts::FRAC_PI_2);
  const_f64_as_f64x2!(FRAC_PI_3, core::f64::consts::FRAC_PI_3);
  const_f64_as_f64x2!(FRAC_PI_4, core::f64::consts::FRAC_PI_4);
  const_f64_as_f64x2!(FRAC_PI_6, core::f64::consts::FRAC_PI_6);
  const_f64_as_f64x2!(FRAC_PI_8, core::f64::consts::FRAC_PI_8);
  const_f64_as_f64x2!(LN_2, core::f64::consts::LN_2);
  const_f64_as_f64x2!(LN_10, core::f64::consts::LN_10);
  const_f64_as_f64x2!(LOG2_E, core::f64::consts::LOG2_E);
  const_f64_as_f64x2!(LOG10_E, core::f64::consts::LOG10_E);
  const_f64_as_f64x2!(LOG10_2, core::f64::consts::LOG10_2);
  const_f64_as_f64x2!(LOG2_10, core::f64::consts::LOG2_10);
  const_f64_as_f64x2!(PI, core::f64::consts::PI);
  const_f64_as_f64x2!(SQRT_2, core::f64::consts::SQRT_2);
  const_f64_as_f64x2!(TAU, 6.28318530717958647692528676655900577_f64);
}

unsafe impl Zeroable for f64x2 {}
unsafe impl Pod for f64x2 {}

impl Add for f64x2 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn add(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: add_m128d(self.sse, rhs.sse) }
      } else {
        Self { arr: [
          self.arr[0] + rhs.arr[0],
          self.arr[1] + rhs.arr[1],
        ]}
      }
    }
  }
}

impl Sub for f64x2 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn sub(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: sub_m128d(self.sse, rhs.sse) }
      } else {
        Self { arr: [
          self.arr[0] - rhs.arr[0],
          self.arr[1] - rhs.arr[1],
        ]}
      }
    }
  }
}

impl Mul for f64x2 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn mul(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: mul_m128d(self.sse, rhs.sse) }
      } else {
        Self { arr: [
          self.arr[0] * rhs.arr[0],
          self.arr[1] * rhs.arr[1],
        ]}
      }
    }
  }
}

impl Div for f64x2 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn div(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: div_m128d(self.sse, rhs.sse) }
      } else {
        Self { arr: [
          self.arr[0] / rhs.arr[0],
          self.arr[1] / rhs.arr[1],
        ]}
      }
    }
  }
}

impl BitAnd for f64x2 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn bitand(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: bitand_m128d(self.sse, rhs.sse) }
      } else {
        Self { arr: [
          f64::from_bits(self.arr[0].to_bits() & rhs.arr[0].to_bits()),
          f64::from_bits(self.arr[1].to_bits() & rhs.arr[1].to_bits()),
        ]}
      }
    }
  }
}

impl BitOr for f64x2 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn bitor(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: bitor_m128d(self.sse, rhs.sse) }
      } else {
        Self { arr: [
          f64::from_bits(self.arr[0].to_bits() | rhs.arr[0].to_bits()),
          f64::from_bits(self.arr[1].to_bits() | rhs.arr[1].to_bits()),
        ]}
      }
    }
  }
}

impl BitXor for f64x2 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn bitxor(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: bitxor_m128d(self.sse, rhs.sse) }
      } else {
        Self { arr: [
          f64::from_bits(self.arr[0].to_bits() ^ rhs.arr[0].to_bits()),
          f64::from_bits(self.arr[1].to_bits() ^ rhs.arr[1].to_bits()),
        ]}
      }
    }
  }
}

impl f64x2 {
  #[inline]
  #[must_use]
  pub fn cmp_eq(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: cmp_eq_mask_m128d(self.sse, rhs.sse) }
      } else {
        Self { arr: [
          if self.arr[0] == rhs.arr[0] { f64::from_bits(u64::MAX) } else { 0.0 },
          if self.arr[1] == rhs.arr[1] { f64::from_bits(u64::MAX) } else { 0.0 },
        ]}
      }
    }
  }
  #[inline]
  #[must_use]
  pub fn cmp_ne(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: cmp_neq_mask_m128d(self.sse, rhs.sse) }
      } else {
        Self { arr: [
          if self.arr[0] != rhs.arr[0] { f64::from_bits(u64::MAX) } else { 0.0 },
          if self.arr[1] != rhs.arr[1] { f64::from_bits(u64::MAX) } else { 0.0 },
        ]}
      }
    }
  }
  #[inline]
  #[must_use]
  pub fn cmp_ge(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: cmp_ge_mask_m128d(self.sse, rhs.sse) }
      } else {
        Self { arr: [
          if self.arr[0] >= rhs.arr[0] { f64::from_bits(u64::MAX) } else { 0.0 },
          if self.arr[1] >= rhs.arr[1] { f64::from_bits(u64::MAX) } else { 0.0 },
        ]}
      }
    }
  }
  #[inline]
  #[must_use]
  pub fn cmp_gt(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: cmp_gt_mask_m128d(self.sse, rhs.sse) }
      } else {
        Self { arr: [
          if self.arr[0] > rhs.arr[0] { f64::from_bits(u64::MAX) } else { 0.0 },
          if self.arr[1] > rhs.arr[1] { f64::from_bits(u64::MAX) } else { 0.0 },
        ]}
      }
    }
  }
  #[inline]
  #[must_use]
  pub fn cmp_le(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: cmp_le_mask_m128d(self.sse, rhs.sse) }
      } else {
        Self { arr: [
          if self.arr[0] <= rhs.arr[0] { f64::from_bits(u64::MAX) } else { 0.0 },
          if self.arr[1] <= rhs.arr[1] { f64::from_bits(u64::MAX) } else { 0.0 },
        ]}
      }
    }
  }
  #[inline]
  #[must_use]
  pub fn cmp_lt(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: cmp_lt_mask_m128d(self.sse, rhs.sse) }
      } else {
        Self { arr: [
          if self.arr[0] < rhs.arr[0] { f64::from_bits(u64::MAX) } else { 0.0 },
          if self.arr[1] < rhs.arr[1] { f64::from_bits(u64::MAX) } else { 0.0 },
        ]}
      }
    }
  }
  #[inline]
  #[must_use]
  pub fn blend(self, t: Self, f: Self) -> Self {
    pick! {
      if #[cfg(target_feature="sse4.1")] {
        Self { sse: blend_varying_m128d(f.sse, t.sse, self.sse) }
      } else {
        generic_bit_blend(self, t, f)
      }
    }
  }
  #[inline]
  #[must_use]
  pub fn abs(self) -> Self {
    let non_sign_bits = f64x2::from(f64::from_bits(i64::MAX as u64));
    self & non_sign_bits
  }
  #[inline]
  #[must_use]
  pub fn max(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: max_m128d(self.sse, rhs.sse) }
      } else {
        Self { arr: [
          self.arr[0].max(rhs.arr[0]),
          self.arr[1].max(rhs.arr[1]),
        ]}
      }
    }
  }
  #[inline]
  #[must_use]
  pub fn min(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: min_m128d(self.sse, rhs.sse) }
      } else {
        Self { arr: [
          self.arr[0].min(rhs.arr[0]),
          self.arr[1].min(rhs.arr[1]),
        ]}
      }
    }
  }
  #[inline]
  #[must_use]
  pub fn is_nan(self) -> Self {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: cmp_unord_mask_m128d(self.sse, self.sse) }
      } else {
        Self { arr: [
          if self.arr[0].is_nan() { f64::from_bits(u64::MAX) } else { 0.0 },
          if self.arr[1].is_nan() { f64::from_bits(u64::MAX) } else { 0.0 },
        ]}
      }
    }
  }
  #[inline]
  #[must_use]
  pub fn is_finite(self) -> Self {
    let shifted_exp_mask = u64x2::from(0xFFE0000000000000);
    let u: u64x2 = cast(self);
    let shift_u = u << 1_u64;
    let out = !(shift_u & shifted_exp_mask).cmp_eq(shifted_exp_mask);
    cast(out)
  }
  #[inline]
  #[must_use]
  pub fn round(self) -> Self {
    pick! {
      if #[cfg(target_feature="sse4.1")] {
        Self { sse: round_m128d!(self.sse, Nearest) }
      } else {
        let sign_mask = f64x2::from(-0.0);
        let magic = f64x2::from(f64::from_bits(0x43300000_00000000));
        let sign = self & sign_mask;
        let signed_magic = magic | sign;
        self + signed_magic - signed_magic
      }
    }
  }
  #[inline]
  #[must_use]
  pub fn round_int(self) -> i64x2 {
    let rounded: [f64; 2] = cast(self.round());
    let rounded_ints: i64x2 = cast([rounded[0] as i64, rounded[1] as i64]);
    cast::<f64x2, i64x2>(self.is_finite())
      .blend(rounded_ints, i64x2::from(i64::MIN))
  }
  #[inline]
  #[must_use]
  pub fn mul_add(self, m: Self, a: Self) -> Self {
    pick! {
      if #[cfg(all(target_feature="sse2",target_feature="fma"))] {
        Self { sse: fused_mul_add_m128d(self.sse, m.sse, a.sse) }
      } else {
        (self * m) + a
      }
    }
  }
  #[inline]
  #[must_use]
  pub fn mul_neg_add(self, m: Self, a: Self) -> Self {
    pick! {
      if #[cfg(all(target_feature="sse",target_feature="fma"))] {
        Self { sse: fused_mul_neg_add_m128d(self.sse, m.sse, a.sse) }
      } else {
        a - (self * m)
      }
    }
  }
  #[inline]
  #[must_use]
  pub fn flip_signs(self, signs: Self) -> Self {
    self ^ (signs & Self::from(-0.0))
  }
  #[inline]
  #[must_use]
  #[allow(non_upper_case_globals)]
  pub fn sin_cos(self) -> (Self, Self) {
    // Based on the Agner Fog "vector class library":
    // https://github.com/vectorclass/version2/blob/master/vectormath_trig.h

    const_f64_as_f64x2!(P0sin, -1.66666666666666307295E-1);
    const_f64_as_f64x2!(P1sin, 8.33333333332211858878E-3);
    const_f64_as_f64x2!(P2sin, -1.98412698295895385996E-4);
    const_f64_as_f64x2!(P3sin, 2.75573136213857245213E-6);
    const_f64_as_f64x2!(P4sin, -2.50507477628578072866E-8);
    const_f64_as_f64x2!(P5sin, 1.58962301576546568060E-10);

    const_f64_as_f64x2!(P0cos, 4.16666666666665929218E-2);
    const_f64_as_f64x2!(P1cos, -1.38888888888730564116E-3);
    const_f64_as_f64x2!(P2cos, 2.48015872888517045348E-5);
    const_f64_as_f64x2!(P3cos, -2.75573141792967388112E-7);
    const_f64_as_f64x2!(P4cos, 2.08757008419747316778E-9);
    const_f64_as_f64x2!(P5cos, -1.13585365213876817300E-11);

    const_f64_as_f64x2!(DP1, 7.853981554508209228515625E-1 * 2.);
    const_f64_as_f64x2!(DP2, 7.94662735614792836714E-9 * 2.);
    const_f64_as_f64x2!(DP3, 3.06161699786838294307E-17 * 2.);

    const_f64_as_f64x2!(TWO_OVER_PI, 2.0 / core::f64::consts::PI);

    let xa = self.abs();

    let y = (xa * TWO_OVER_PI).round();
    let q = y.round_int();

    let x = y.mul_neg_add(DP3, y.mul_neg_add(DP2, y.mul_neg_add(DP1, xa)));

    let x2 = x * x;
    let mut s = polynomial_5!(x2, P0sin, P1sin, P2sin, P3sin, P4sin, P5sin);
    let mut c = polynomial_5!(x2, P0cos, P1cos, P2cos, P3cos, P4cos, P5cos);
    s = (x * x2).mul_add(s, x);
    c =
      (x2 * x2).mul_add(c, x2.mul_neg_add(f64x2::from(0.5), f64x2::from(1.0)));

    let swap = !((q & i64x2::from(1)).cmp_eq(i64x2::from(0)));

    let mut overflow: f64x2 = cast(q.cmp_gt(i64x2::from(0x80000000000000)));
    overflow &= xa.is_finite();
    s = overflow.blend(f64x2::from(0.0), s);
    c = overflow.blend(f64x2::from(1.0), c);

    // calc sin
    let mut sin1 = cast::<_, f64x2>(swap).blend(c, s);
    let sign_sin: i64x2 = (q << 62) ^ cast::<_, i64x2>(self);
    sin1 = sin1.flip_signs(cast(sign_sin));

    // calc cos
    let mut cos1 = cast::<_, f64x2>(swap).blend(s, c);
    let sign_cos: i64x2 = ((q + i64x2::from(1)) & i64x2::from(2)) << 62;
    cos1 ^= cast::<_, f64x2>(sign_cos);

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
    const_f64_as_f64x2!(RAD_TO_DEG_RATIO, 180.0_f64 / core::f64::consts::PI);
    self * RAD_TO_DEG_RATIO
  }
  #[inline]
  #[must_use]
  pub fn to_radians(self) -> Self {
    const_f64_as_f64x2!(DEG_TO_RAD_RATIO, core::f64::consts::PI / 180.0_f64);
    self * DEG_TO_RAD_RATIO
  }
  #[inline]
  #[must_use]
  pub fn sqrt(self) -> Self {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: sqrt_m128d(self.sse) }
      } else {
        Self { arr: [
          software_sqrt(self.arr[0]),
          software_sqrt(self.arr[1]),
        ]}
      }
    }
  }
  #[inline]
  #[must_use]
  pub fn move_mask(self) -> i32 {
    pick! {
      if #[cfg(target_feature="sse2")] {
        move_mask_m128d(self.sse)
      } else {
        (((self.arr[0].to_bits() as i64) < 0) as i32) << 0 |
        (((self.arr[1].to_bits() as i64) < 0) as i32) << 1
      }
    }
  }
}
