
use super::*;

/// Wide Methods
impl f32x4 {
  #[inline]
  pub fn andnot(self, rhs: Self) -> Self {
    cfg_if! {if #[cfg(target_feature="sse")] {
      Self { sse: self.sse.andnot(rhs.sse) }
    } else {
      (!self) & rhs
    }}
  }

  pub fn is_nan(self) -> Self {
    cfg_if! {if #[cfg(target_feature="sse")] {
      // yes it's weird, yes it's correct.
      Self { sse: self.sse.cmp_nan(self.sse) }
    } else {
      let op = |a:f32| {
        if a.is_nan() {
          f32::from_bits(u32::max_value())
        } else {
          0.0
        }
      };
      Self { arr: [
        op(self.arr[0]),
        op(self.arr[1]),
        op(self.arr[2]),
        op(self.arr[3]),
      ] }
    }}
  }

  pub fn is_ordinary(self) -> Self {
    cfg_if! {if #[cfg(target_feature="sse")] {
      // yes it's weird, yes it's correct.
      Self { sse: self.sse.cmp_ordinary(self.sse) }
    } else {
      let op = |a:f32| {
        if !a.is_nan() {
          f32::from_bits(u32::max_value())
        } else {
          0.0
        }
      };
      Self { arr: [
        op(self.arr[0]),
        op(self.arr[1]),
        op(self.arr[2]),
        op(self.arr[3]),
      ] }
    }}
  }

  /// Use `self` (a boolish value) to merge `a` and `b`.
  ///
  /// For each lane index, if the `self` lane is "true" then the `a` value will
  /// be used in the output, otherwise the `b` value will be used in the output.
  ///
  /// If `sse4.1` is enabled, then "true" _only_ checks the sign bit. With less
  /// features enabled the entire bit pattern of the lane will matter. This is
  /// not normally a problem, because the comparison methods naturally return
  /// all 1s or all 0s anyway.
  #[inline]
  pub fn merge(self, a: Self, b: Self) -> Self {
    cfg_if! {if #[cfg(target_feature="sse4.1")] {
      Self { sse: b.sse.blend_var(a.sse, self.sse) }
    } else {
      (self & a) | self.andnot(b)
    }}
  }

  /// ```rust
  /// use wide::f32x4;
  /// let a = f32x4::new(1.0, 2.0, 3.0, 4.0);
  /// let b = f32x4::from(2.5);
  /// assert_eq!(a.cmp_lt(b).move_mask(), 0b0011);
  /// ```
  #[inline]
  pub fn move_mask(self) -> i32 {
    cfg_if! {if #[cfg(target_feature="sse")] {
      self.sse.move_mask()
    } else {
      let mut out = 0_i32;
      for i in 0..4 {
        if cast::<f32, i32>(self.arr[i]) < 0 {
          out |= 1<<i;
        }
      }
      out
    }}
  }
  #[inline]
  pub fn cmp_eq(self, rhs: Self) -> Self {
    cfg_if! {if #[cfg(target_feature="sse")] {
      Self { sse: self.sse.cmp_eq(rhs.sse) }
    } else {
      let test = |a, b| {
        if a == b {
          cast::<u32, f32>(core::u32::MAX)
        } else {
          cast::<u32, f32>(0)
        }
      };
      Self { arr: [
        test(self.arr[0], rhs.arr[0]),
        test(self.arr[1], rhs.arr[1]),
        test(self.arr[2], rhs.arr[2]),
        test(self.arr[3], rhs.arr[3]),
      ] }
    }}
  }
  #[inline]
  pub fn cmp_ge(self, rhs: Self) -> Self {
    cfg_if! {if #[cfg(target_feature="sse")] {
      Self { sse: self.sse.cmp_ge(rhs.sse) }
    } else {
      let test = |a, b| {
        if a >= b {
          cast::<u32, f32>(core::u32::MAX)
        } else {
          cast::<u32, f32>(0)
        }
      };
      Self { arr: [
        test(self.arr[0], rhs.arr[0]),
        test(self.arr[1], rhs.arr[1]),
        test(self.arr[2], rhs.arr[2]),
        test(self.arr[3], rhs.arr[3]),
      ] }
    }}
  }
  #[inline]
  pub fn cmp_gt(self, rhs: Self) -> Self {
    cfg_if! {if #[cfg(target_feature="sse")] {
      Self { sse: self.sse.cmp_gt(rhs.sse) }
    } else {
      let test = |a, b| {
        if a > b {
          cast::<u32, f32>(core::u32::MAX)
        } else {
          cast::<u32, f32>(0)
        }
      };
      Self { arr: [
        test(self.arr[0], rhs.arr[0]),
        test(self.arr[1], rhs.arr[1]),
        test(self.arr[2], rhs.arr[2]),
        test(self.arr[3], rhs.arr[3]),
      ] }
    }}
  }
  #[inline]
  pub fn cmp_le(self, rhs: Self) -> Self {
    cfg_if! {if #[cfg(target_feature="sse")] {
      Self { sse: self.sse.cmp_le(rhs.sse) }
    } else {
      let test = |a, b| {
        if a <= b {
          cast::<u32, f32>(core::u32::MAX)
        } else {
          cast::<u32, f32>(0)
        }
      };
      Self { arr: [
        test(self.arr[0], rhs.arr[0]),
        test(self.arr[1], rhs.arr[1]),
        test(self.arr[2], rhs.arr[2]),
        test(self.arr[3], rhs.arr[3]),
      ] }
    }}
  }
  #[inline]
  pub fn cmp_lt(self, rhs: Self) -> Self {
    cfg_if! {if #[cfg(target_feature="sse")] {
      Self { sse: self.sse.cmp_lt(rhs.sse) }
    } else {
      let test = |a, b| {
        if a < b {
          cast::<u32, f32>(core::u32::MAX)
        } else {
          cast::<u32, f32>(0)
        }
      };
      Self { arr: [
        test(self.arr[0], rhs.arr[0]),
        test(self.arr[1], rhs.arr[1]),
        test(self.arr[2], rhs.arr[2]),
        test(self.arr[3], rhs.arr[3]),
      ] }
    }}
  }
  #[inline]
  pub fn cmp_nan(self, rhs: Self) -> Self {
    cfg_if! {if #[cfg(target_feature="sse")] {
      Self { sse: self.sse.cmp_nan(rhs.sse) }
    } else {
      let test = |a: f32, b: f32| {
        if a.is_nan() || b.is_nan() {
          cast::<u32, f32>(core::u32::MAX)
        } else {
          cast::<u32, f32>(0)
        }
      };
      Self { arr: [
        test(self.arr[0], rhs.arr[0]),
        test(self.arr[1], rhs.arr[1]),
        test(self.arr[2], rhs.arr[2]),
        test(self.arr[3], rhs.arr[3]),
      ] }
    }}
  }
  #[inline]
  pub fn cmp_ne(self, rhs: Self) -> Self {
    cfg_if! {if #[cfg(target_feature="sse")] {
      Self { sse: self.sse.cmp_ne(rhs.sse) }
    } else {
      let test = |a, b| {
        if a != b {
          cast::<u32, f32>(core::u32::MAX)
        } else {
          cast::<u32, f32>(0)
        }
      };
      Self { arr: [
        test(self.arr[0], rhs.arr[0]),
        test(self.arr[1], rhs.arr[1]),
        test(self.arr[2], rhs.arr[2]),
        test(self.arr[3], rhs.arr[3]),
      ] }
    }}
  }
  /// If you call this method it sets off [Third Impact](https://evangelion.fandom.com/wiki/Third_Impact)
  #[inline]
  pub fn cmp_nge(self, rhs: Self) -> Self {
    cfg_if! {if #[cfg(target_feature="sse")] {
      Self { sse: self.sse.cmp_nge(rhs.sse) }
    } else {
      let test = |a, b| {
        if !(a >= b) {
          cast::<u32, f32>(core::u32::MAX)
        } else {
          cast::<u32, f32>(0)
        }
      };
      Self { arr: [
        test(self.arr[0], rhs.arr[0]),
        test(self.arr[1], rhs.arr[1]),
        test(self.arr[2], rhs.arr[2]),
        test(self.arr[3], rhs.arr[3]),
      ] }
    }}
  }
  #[inline]
  pub fn cmp_ngt(self, rhs: Self) -> Self {
    cfg_if! {if #[cfg(target_feature="sse")] {
      Self { sse: self.sse.cmp_ngt(rhs.sse) }
    } else {
      let test = |a, b| {
        if !(a > b) {
          cast::<u32, f32>(core::u32::MAX)
        } else {
          cast::<u32, f32>(0)
        }
      };
      Self { arr: [
        test(self.arr[0], rhs.arr[0]),
        test(self.arr[1], rhs.arr[1]),
        test(self.arr[2], rhs.arr[2]),
        test(self.arr[3], rhs.arr[3]),
      ] }
    }}
  }
  #[inline]
  pub fn cmp_nle(self, rhs: Self) -> Self {
    cfg_if! {if #[cfg(target_feature="sse")] {
      Self { sse: self.sse.cmp_nle(rhs.sse) }
    } else {
      let test = |a, b| {
        if !(a <= b) {
          cast::<u32, f32>(core::u32::MAX)
        } else {
          cast::<u32, f32>(0)
        }
      };
      Self { arr: [
        test(self.arr[0], rhs.arr[0]),
        test(self.arr[1], rhs.arr[1]),
        test(self.arr[2], rhs.arr[2]),
        test(self.arr[3], rhs.arr[3]),
      ] }
    }}
  }
  #[inline]
  pub fn cmp_nlt(self, rhs: Self) -> Self {
    cfg_if! {if #[cfg(target_feature="sse")] {
      Self { sse: self.sse.cmp_nlt(rhs.sse) }
    } else {
      let test = |a, b| {
        if !(a < b) {
          cast::<u32, f32>(core::u32::MAX)
        } else {
          cast::<u32, f32>(0)
        }
      };
      Self { arr: [
        test(self.arr[0], rhs.arr[0]),
        test(self.arr[1], rhs.arr[1]),
        test(self.arr[2], rhs.arr[2]),
        test(self.arr[3], rhs.arr[3]),
      ] }
    }}
  }
  #[inline]
  #[deprecated(since = "0.3.1", note = "use is_ordinary")]
  pub fn cmp_not_nan(self, rhs: Self) -> Self {
    cfg_if! {if #[cfg(target_feature="sse")] {
      Self { sse: self.sse.cmp_ordinary(rhs.sse) }
    } else {
      let test = |a: f32, b: f32| {
        if (!a.is_nan()) && (!b.is_nan()) {
          cast::<u32, f32>(core::u32::MAX)
        } else {
          cast::<u32, f32>(0)
        }
      };
      Self { arr: [
        test(self.arr[0], rhs.arr[0]),
        test(self.arr[1], rhs.arr[1]),
        test(self.arr[2], rhs.arr[2]),
        test(self.arr[3], rhs.arr[3]),
      ] }
    }}
  }

  #[inline]
  pub fn ceil(self) -> Self {
    cfg_if! {if #[cfg(target_feature="sse4.1")] {
      Self { sse: self.sse.ceil() }
    } else if #[cfg(target_feature="sse2")] {
      Self { sse: self.sse.ceil_sse2() }
    } else {
      let a: [f32; 4] = cast(self);
      cast([a[0].ceil(), a[1].ceil(), a[2].ceil(), a[3].ceil()])
    }}
  }

  #[inline]
  pub fn floor(self) -> Self {
    cfg_if! {if #[cfg(target_feature="sse4.1")] {
      Self { sse: self.sse.floor() }
    } else if #[cfg(target_feature="sse2")] {
      Self { sse: self.sse.floor_sse2() }
    } else {
      let a: [f32; 4] = cast(self);
      cast([a[0].floor(), a[1].floor(), a[2].floor(), a[3].floor()])
    }}
  }

  #[inline]
  pub fn abs(self) -> Self {
    self & Self::ALL_EXCEPT_SIGN
  }

  #[inline]
  pub fn cos(self) -> Self {
    // TODO: check that once this inlines we don't pay the "calculate sin" costs.
    self.sin_cos().1
  }

  #[inline]
  pub fn round(self) -> Self {
    cfg_if! {if #[cfg(target_feature = "sse4.1")] {
      // we sometimes have a direct round instruction
      Self { sse: self.sse.round_nearest() }
    } else if #[cfg(target_feature="sse2")] {
      // or we could round to int and back
      Self { sse: self.sse.round_i32().round_f32() }
    } else if #[cfg(feature = "toolchain_nightly")] {
      // hope that the intrinsics will tell LLVM what we're doing, and we can do
      // this core-only in nightly.
      use core::intrinsics::roundf32;
      let a: [f32; 4] = cast(self);
      cast(unsafe {
        [roundf32(a[0]), roundf32(a[1]), roundf32(a[2]), roundf32(a[3])]
      })
    } else {
      // this path is like the above but needs std
      let a: [f32; 4] = cast(self);
      cast([a[0].round(), a[1].round(), a[2].round(), a[3].round()])
    }}
  }

  /// Sine function.
  ///
  /// "We called it '[Sin](https://vignette.wikia.nocookie.net/finalfantasy/images/d/de/10sin-a.jpg)'."
  #[inline]
  pub fn sin(self) -> Self {
    // TODO: check that once this inlines we don't pay the "calculate cos" costs.
    self.sin_cos().0
  }

  /// Performs a "fused multiply-add", `(self * b) + c`
  ///
  /// This is only different from a normal mul and then add if the `fma`
  /// target_feature is enabled during the build. This is **not** on by default,
  /// you must enable it yourself. To be clear, this is not a cargo feature,
  /// this is a CPU feature. [Read the
  /// guide](https://rust-lang.github.io/packed_simd/perf-guide/target-feature/rustflags.html)
  /// for info on how to enable CPU features if you haven't done that before.
  #[inline]
  pub fn mul_add(self, b: Self, c: Self) -> Self {
    cfg_if! {if #[cfg(target_feature = "fma")] {
      Self { sse: self.sse.fmadd(b.sse, c.sse) }
    } else {
      (self * b) + c
    }}
  }

  /// Negated "mul_add", `c - (self * b)`.
  ///
  /// Fused if `fma` feature is enabled, see (mul_add)[f32x4::mul_add].
  pub fn negated_mul_add(self, b: Self, c: Self) -> Self {
    cfg_if! {if #[cfg(target_feature = "fma")] {
      Self { sse: self.sse.fnmadd(b.sse, c.sse) }
    } else {
      c - (self * b)
    }}
  }

  #[inline]
  pub fn recip(self) -> Self {
    cfg_if! {if #[cfg(target_feature="sse")] {
      Self { sse: self.sse.reciprocal() }
    } else {
      f32x4::from(1.0) / self
    }}
  }

  #[inline]
  pub fn max(self, b: Self) -> Self {
    cfg_if! {if #[cfg(target_feature="sse")] {
      Self { sse: self.sse.max(b.sse) }
    } else {
      let a: [f32; 4] = cast(self);
      let b: [f32; 4] = cast(b);
      cast([
        a[0].max(b[0]),
        a[1].max(b[1]),
        a[2].max(b[2]),
        a[3].max(b[3]),
      ])
    }}
  }

  #[inline]
  pub fn min(self, b: Self) -> Self {
    cfg_if! {if #[cfg(target_feature="sse")] {
      Self { sse: self.sse.min(b.sse) }
    } else {
      let a: [f32; 4] = cast(self);
      let b: [f32; 4] = cast(b);
      cast([
        a[0].min(b[0]),
        a[1].min(b[1]),
        a[2].min(b[2]),
        a[3].min(b[3]),
      ])
    }}
  }

  #[inline]
  pub fn round_i32(self) -> i32x4 {
    cfg_if! {if #[cfg(target_feature="sse")] {
      i32x4 { sse: self.sse.round_i32() }
    } else {
      i32x4 { arr: [
        self.arr[0] as i32,
        self.arr[1] as i32,
        self.arr[2] as i32,
        self.arr[3] as i32,
      ]}
    }}
  }

  /// If it's some finite value.
  ///
  /// * True for normal, denormal, and zero.
  /// * False for +/- INF, NaN
  /// * boolish
  #[allow(clippy::unreadable_literal)]
  #[allow(bad_style)]
  pub fn is_finite(self) -> f32x4 {
    const EXPONENT_MASKu: u32 = 0xFF000000_u32;
    const EXPONENT_MASKi: i32 = EXPONENT_MASKu as i32;
    cfg_if! {if #[cfg(target_feature="sse2")] {
      let t1 = self.sse.cast_m128i();
      // TODO: use an immediate shift here?
      let t2 = t1.shift_left_i32(m128i::splat_i32(1));
      let t3 = !(t2 & m128i::splat_i32(EXPONENT_MASKi))
        .cmp_eq_i32(m128i::splat_i32(EXPONENT_MASKi));
      Self { sse: t3.cast_m128() }
    } else {
      let op = |f: f32| {
        let t1 = f.to_bits();
        let t2 = t1 << 1;
        let t3 = (t2 & EXPONENT_MASKu) != EXPONENT_MASKu;
        if t3 {
          f32::from_bits(u32::max_value())
        } else {
          0.0
        }
      };
      Self { arr: [
        op(self.arr[0]),
        op(self.arr[1]),
        op(self.arr[2]),
        op(self.arr[3]),
      ]}
    }}
  }

  #[inline]
  pub fn cast_i32x4(self) -> i32x4 {
    cfg_if! {if #[cfg(target_feature="sse2")] {
      i32x4 { sse: self.sse.cast_m128i() }
    } else {
      cast(self)
    }}
  }
  #[inline]
  pub fn copysign(self, b: Self) -> Self {
    self ^ (b & Self::NEGATIVE_ZERO)
  }

  #[inline]
  pub fn clamp(self, min: Self, max: Self) -> Self {
    self.max(min).min(max)
  }

  #[inline]
  pub fn signum(self) -> Self {
    self.is_nan().merge(f32x4::NAN, Self::ONE.copysign(self))
  }

  #[inline]
  pub fn tan(self) -> Self {
    let (s, c) = self.sin_cos();
    s / c
  }

  /// calculates polynomial c2*x^2 + c1*x + c0
  ///
  /// https://github.com/vectorclass/version2/blob/master/vectormath_common.h#L111
  #[inline]
  fn polynomial_2(self, c0: Self, c1: Self, c2: Self) -> Self {
    let self2 = self * self;
    self2.mul_add(c2, self.mul_add(c1, c0))
  }

  // /// calculates polynomial c3*x^3 + c2*x^2 + c1*x + c0
  // ///
  // /// https://github.com/vectorclass/version2/blob/master/vectormath_common.h#L120
  // #[inline]
  // fn polynomial_3(self, c0: Self, c1: Self, c2: Self, c3: Self) -> Self {
  //   let self2 = self * self;
  //   c3.mul_add(self, c2).mul_add(self2, c1.mul_add(self, c0))
  // }

  /// Sine and Cosine as a single operation.
  #[allow(clippy::unreadable_literal)]
  #[allow(clippy::excessive_precision)]
  #[allow(clippy::many_single_char_names)]
  #[allow(bad_style)]
  // Note: not inline this one is kinda huge
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

    // TODO: check where we can reduce any casting operations.

    let xa = self.abs();

    // Find quadrant
    let y = (xa * (f32x4::from(2.0) / Self::PI)).round();
    let q: i32x4 = y.round_i32();

    // Reduce by extended precision modular arithmetic
    // x = ((xa - y * DP1F) - y * DP2F) - y * DP3F;
    let x = y.negated_mul_add(DP3F, y.negated_mul_add(DP2F, y.negated_mul_add(DP1F, xa)));

    // Taylor expansion of sin and cos, valid for -pi/4 <= x <= pi/4
    let x2 = x * x;
    let mut s = x2.polynomial_2(P0sinf, P1sinf, P2sinf) * (x * x2) + x;
    let mut c = x2.polynomial_2(P0cosf, P1cosf, P2cosf) * (x2 * x2)
      + Self::HALF.negated_mul_add(x2, Self::ONE);

    // swap sin and cos if odd quadrant
    let swap = !(q & i32x4::ONE).cmp_eq(i32x4::ZERO);

    // "q big if overflow"
    const_i32_as_i32x4!(BIG_THRESHOLD, 0x2000000);
    let mut overflow: f32x4 = cast(q.cmp_gt(BIG_THRESHOLD));
    overflow &= xa.is_finite();
    s = f32x4::merge(overflow, f32x4::ZERO, s);
    c = f32x4::merge(overflow, f32x4::ONE, c);

    // calc sin
    let mut sin1 = f32x4::merge(cast(swap), c, s);
    let sign_sin: i32x4 = (q << 30) ^ self.cast_i32x4();
    sin1 = sin1.copysign(cast(sign_sin));

    // calc cos
    let mut cos1 = f32x4::merge(cast(swap), s, c);
    let sign_cos: i32x4 = ((q + i32x4::ONE) & i32x4::from(2)) << 30;
    cos1 ^= cast::<i32x4, f32x4>(sign_cos);

    (sin1, cos1)
  }

  #[inline]
  #[allow(clippy::unreadable_literal)]
  #[allow(clippy::excessive_precision)]
  pub fn to_degrees(self) -> Self {
    const_f32_as_f32x4!(
      pub RAD_TO_DEG_RATIO, 57.2957795130823208767981548141051703_f32
    );
    self * RAD_TO_DEG_RATIO
  }

  #[inline]
  pub fn to_radians(self) -> Self {
    const_f32_as_f32x4!(
      pub DEG_TO_RAD_RATIO, core::f32::consts::PI / 180.0_f32
    );
    self * DEG_TO_RAD_RATIO
  }

  #[inline]
  pub fn fract(self) -> Self {
    self - self.trunc()
  }

  #[inline]
  pub fn sqrt(self) -> Self {
    cfg_if! { if #[cfg(target_feature = "sse")] {
      Self { sse: self.sse.sqrt() }
    } else if #[cfg(feature = "toolchain_nightly")] {
      use core::intrinsics::sqrtf32;
      let a: [f32; 4] = cast(self);
      cast(unsafe {
        [sqrtf32(a[0]), sqrtf32(a[1]), sqrtf32(a[2]), sqrtf32(a[3])]
      })
    } else {
      let a: [f32; 4] = cast(self);
      cast([a[0].sqrt(), a[1].sqrt(), a[2].sqrt(), a[3].sqrt()])
    }}
  }

}