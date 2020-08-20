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

impl f32x4 {
  #[inline]
  #[must_use]
  pub fn cmp_eq(self, rhs: Self) -> Self {
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
  #[inline]
  #[must_use]
  pub fn cmp_ne(self, rhs: Self) -> Self {
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
  #[inline]
  #[must_use]
  pub fn cmp_ge(self, rhs: Self) -> Self {
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
  #[inline]
  #[must_use]
  pub fn cmp_gt(self, rhs: Self) -> Self {
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
  #[inline]
  #[must_use]
  pub fn cmp_le(self, rhs: Self) -> Self {
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
  #[inline]
  #[must_use]
  pub fn cmp_lt(self, rhs: Self) -> Self {
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
  pub fn flip_signs(self, signs: Self) -> Self {
    self ^ (signs & Self::from(-0.0))
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
    let t1 = cast::<_, u32x4>(self);
    let t2 = t1 << 1;
    let t3 = t2 >> 24;
    i32x4::round_float(cast::<_, i32x4>(t3) - i32x4::from(0x7F))
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
    (!mask).blend(res, Self::ZERO)
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
}
