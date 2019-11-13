use super::*;

cfg_if! {
  if #[cfg(target_feature="sse")] {
    #[repr(C, align(16))]
    pub struct f32x4 {
      sse: m128
    }
  } else {
    #[repr(C, align(16))]
    pub struct f32x4 {
      arr: [f32; 4]
    }
  }
}
#[test]
fn declaration_tests_f32x4() {
  use core::mem::{align_of, size_of};
  assert_eq!(size_of::<f32x4>(), 16);
  assert_eq!(align_of::<f32x4>(), 16);
}

#[allow(non_camel_case_types)]
pub union ConstUnionHack_f32x4 {
  narrow_arr: [f32; 4],
  wide_thing: f32x4,
}
#[test]
#[allow(non_snake_case)]
fn declaration_tests_ConstUnionHack_f32x4() {
  use core::mem::{align_of, size_of};
  assert_eq!(size_of::<ConstUnionHack_f32x4>(), size_of::<f32x4>());
  assert_eq!(align_of::<ConstUnionHack_f32x4>(), align_of::<f32x4>());
}

/// Declares an `f32x4` const identifier.
///
/// ## Broadcast A Single Value
///
/// * **Usage:** `const_f32_as_f32x4!(#[meta]* vis ident, val);`
///
/// The value should be a single `f32` expression, which is then duplicated into
/// all lanes of the constant declaration.
///
/// ```rust
/// const_f32_as_f32x4!(
///   /// Machine epsilon value for `f32`.
///   pub EPSILON, core::f32::EPSILON
/// );
/// ```
///
/// ## Select Each Lane
///
/// * **Usage:** `const_f32_as_f32x4!(#[meta]* vis ident, a, b, c, d);`
///
/// Each of `a`, `b`, `c`, and `d` are an `f32` expression when are then placed
/// into the constant declaration (low lane to high lane).
///
/// ```rust
/// const_f32_as_f32x4!(
///   /// 1, 2, 3, 4
///   pub ONE_TWO_THREE_FOUR, 1.0, 2.0, 3.0, 4.0
/// );
/// ```
#[macro_export]
macro_rules! const_f32_as_f32x4 {
  // broadcast a single value
  ($(#[$attrs:meta])* $v:vis $i:ident, $val:expr) => {
    $(#[$attrs])*
    $v const $i: f32x4 = {
      let cuh = ConstUnionHack_f32x4 {
        narrow_arr: [$val, $val, $val, $val],
      };
      unsafe { cuh.wide_thing }
    };
  };
  // select each lane's value
  ($(#[$attrs:meta])* $v:vis $i:ident, $a:expr, $b:expr, $c:expr, $d:expr) => {
    $(#[$attrs])*
    $v const $i: f32x4 = {
      let cuh = ConstUnionHack_f32x4 {
        narrow_arr: [$a, $b, $c, $d],
      };
      unsafe { cuh.wide_thing }
    };
  };
}

impl f32x4 {
  //
  // core::f32
  //

  const_f32_as_f32x4!(
    /// Machine epsilon value for `f32`.
    pub EPSILON, core::f32::EPSILON
  );

  const_f32_as_f32x4!(
    /// Positive Infinity (∞).
    pub INFINITY, core::f32::INFINITY
  );

  const_f32_as_f32x4!(
    /// Largest finite `f32` value.
    pub MAX, core::f32::MAX
  );

  const_f32_as_f32x4!(
    /// Smallest finite `f32` value.
    pub MIN, core::f32::MIN
  );

  const_f32_as_f32x4!(
    /// Smallest positive normal `f32` value.
    pub MIN_POSITIVE, core::f32::MIN_POSITIVE
  );

  const_f32_as_f32x4!(
    /// Not a Number (NaN).
    ///
    /// **Reminder:** This is one possible NaN value, but there are many NaN bit
    /// patterns within the `f32` space.
    pub NAN, core::f32::NAN
  );

  const_f32_as_f32x4!(
    /// Negative infinity (-∞).
    pub NEG_INFINITY, core::f32::NEG_INFINITY
  );

  //
  // core::f32::consts
  //

  const_f32_as_f32x4!(
    /// Euler's number (e)
    pub E, core::f32::consts::E
  );

  const_f32_as_f32x4!(
    /// 1/π
    pub FRAC_1_PI, core::f32::consts::FRAC_1_PI
  );

  const_f32_as_f32x4!(
    /// 2/π
    pub FRAC_2_PI, core::f32::consts::FRAC_2_PI
  );

  const_f32_as_f32x4!(
    /// 2/sqrt(π)
    pub FRAC_2_SQRT_PI, core::f32::consts::FRAC_2_SQRT_PI
  );

  const_f32_as_f32x4!(
    /// 1/sqrt(2)
    pub FRAC_1_SQRT_2, core::f32::consts::FRAC_1_SQRT_2
  );

  const_f32_as_f32x4!(
    /// π/2
    pub FRAC_PI_2, core::f32::consts::FRAC_PI_2
  );

  const_f32_as_f32x4!(
    /// π/3
    pub FRAC_PI_3, core::f32::consts::FRAC_PI_3
  );

  const_f32_as_f32x4!(
    /// π/4
    pub FRAC_PI_4, core::f32::consts::FRAC_PI_4
  );

  const_f32_as_f32x4!(
    /// π/6
    pub FRAC_PI_6, core::f32::consts::FRAC_PI_6
  );

  const_f32_as_f32x4!(
    /// π/8
    pub FRAC_PI_8, core::f32::consts::FRAC_PI_8
  );

  const_f32_as_f32x4!(
    /// ln(2)
    pub LN_2, core::f32::consts::LN_2
  );

  const_f32_as_f32x4!(
    /// ln(10)
    pub LN_10, core::f32::consts::LN_10
  );

  const_f32_as_f32x4!(
    /// log2(e)
    pub LOG2_E, core::f32::consts::LOG2_E
  );

  const_f32_as_f32x4!(
    /// log10(e)
    pub LOG10_E, core::f32::consts::LOG10_E
  );

  const_f32_as_f32x4!(
    /// Archimedes' constant (π)
    pub PI, core::f32::consts::PI
  );

  const_f32_as_f32x4!(
    /// sqrt(2)
    pub SQRT_2, core::f32::consts::SQRT_2
  );

  //
  // others
  //

  const_f32_as_f32x4!(
    /// 0.0
    pub ZERO, 0.0
  );

  const_f32_as_f32x4!(
    /// 0.5
    pub HALF, 0.5
  );

  const_f32_as_f32x4!(
    /// 1.0
    pub ONE, 1.0
  );

  const_f32_as_f32x4!(
    /// 2.0 * π, the number of radians in a circle.
    pub TWO_PI, 2.0 * core::f32::consts::PI
  );
}

impl f32x4 {
  #[inline(always)]
  pub fn new(a: f32, b: f32, c: f32, d: f32) -> Self {
    cfg_if! {if #[cfg(target_feature="sse")] {
      Self { sse: m128::set(a,b,c,d) }
    } else {
      Self { arr: [a,b,c,d] }
    }}
  }

  /// Use the mask to bitwise merge two values.
  ///
  /// It is expected that the mask will be a "boolish" value where each lane is
  /// either all 1s or all 0s, but that's not actually required.
  ///
  /// The output will have the `tru` bit any place that `mask` has a 1, and use
  /// the `fal` bit anywhere the mask has a 0.
  ///
  /// ```rust
  /// use wide::f32x4;
  /// let a = f32x4::new(1.0, 2.0, 3.0, 4.0);
  /// let b = f32x4::from(2.5);
  /// let mask = a.cmp_gt(b);
  /// let merged = f32x4::merge(mask, f32x4::from(10.0), f32x4::from(2.0));
  /// let merged_arr: [f32; 4] = unsafe { core::mem::transmute(merged) };
  /// assert_eq!(merged_arr, [2.0, 2.0, 10.0, 10.0]);
  /// ```
  #[inline]
  pub fn merge(mask: Self, tru: Self, fal: Self) -> Self {
    cfg_if! {if #[cfg(target_feature="sse")] {
      Self { sse: tru.sse ^ ((tru.sse ^ fal.sse) & mask.sse) }
    } else {
      let op = |maskf, af, bf| {
        let ai: i32 = cast(af);
        let bi: i32 = cast(bf);
        let maski: i32 = cast(maskf);
        cast::<i32, f32>(ai ^ ((ai ^ bi) & maski))
      };
      Self { arr: [
        op(mask[0], tru.arr[0], fal.arr[0]),
        op(mask[1], tru.arr[1], fal.arr[1]),
        op(mask[2], tru.arr[2], fal.arr[2]),
        op(mask[3], tru.arr[3], fal.arr[3]),
      ] }
    }}
  }
}

impl From<f32> for f32x4 {
  #[inline]
  fn from(val: f32) -> Self {
    cfg_if! {if #[cfg(target_feature="sse")] {
      Self { sse: m128::splat(val) }
    } else {
      Self::new(val,val,val,val)
    }}
  }
}

impl Add for f32x4 {
  type Output = Self;
  #[inline]
  fn add(self, rhs: Self) -> Self {
    cfg_if! {if #[cfg(target_feature="sse")] {
      Self { sse: self.sse.add(rhs.sse) }
    } else {
      Self { arr: [
        self.arr[0] + rhs.arr[0],
        self.arr[1] + rhs.arr[1],
        self.arr[2] + rhs.arr[2],
        self.arr[3] + rhs.arr[3],
      ] }
    }}
  }
}

impl Add<&'_ f32x4> for f32x4 {
  type Output = Self;
  #[inline]
  fn add(self, rhs: &Self) -> Self {
    cfg_if! {if #[cfg(target_feature="sse")] {
      Self { sse: self.sse.add(rhs.sse) }
    } else {
      Self { arr: [
        self.arr[0] + rhs.arr[0],
        self.arr[1] + rhs.arr[1],
        self.arr[2] + rhs.arr[2],
        self.arr[3] + rhs.arr[3],
      ] }
    }}
  }
}

impl Div for f32x4 {
  type Output = Self;
  #[inline]
  fn div(self, rhs: Self) -> Self {
    cfg_if! {if #[cfg(target_feature="sse")] {
      Self { sse: self.sse.div(rhs.sse) }
    } else {
      Self { arr: [
        self.arr[0] / rhs.arr[0],
        self.arr[1] / rhs.arr[1],
        self.arr[2] / rhs.arr[2],
        self.arr[3] / rhs.arr[3],
      ] }
    }}
  }
}

impl Div<&'_ f32x4> for f32x4 {
  type Output = Self;
  #[inline]
  fn div(self, rhs: &Self) -> Self {
    cfg_if! {if #[cfg(target_feature="sse")] {
      Self { sse: self.sse.div(rhs.sse) }
    } else {
      Self { arr: [
        self.arr[0] / rhs.arr[0],
        self.arr[1] / rhs.arr[1],
        self.arr[2] / rhs.arr[2],
        self.arr[3] / rhs.arr[3],
      ] }
    }}
  }
}

impl Mul for f32x4 {
  type Output = Self;
  #[inline]
  fn mul(self, rhs: Self) -> Self {
    cfg_if! {if #[cfg(target_feature="sse")] {
      Self { sse: self.sse.mul(rhs.sse) }
    } else {
      Self { arr: [
        self.arr[0] * rhs.arr[0],
        self.arr[1] * rhs.arr[1],
        self.arr[2] * rhs.arr[2],
        self.arr[3] * rhs.arr[3],
      ] }
    }}
  }
}

impl Mul<&'_ f32x4> for f32x4 {
  type Output = Self;
  #[inline]
  fn mul(self, rhs: &Self) -> Self {
    cfg_if! {if #[cfg(target_feature="sse")] {
      Self { sse: self.sse.mul(rhs.sse) }
    } else {
      Self { arr: [
        self.arr[0] * rhs.arr[0],
        self.arr[1] * rhs.arr[1],
        self.arr[2] * rhs.arr[2],
        self.arr[3] * rhs.arr[3],
      ] }
    }}
  }
}

impl Rem for f32x4 {
  type Output = Self;
  #[inline]
  fn rem(self, rhs: Self) -> Self {
    cfg_if! {if #[cfg(target_feature="sse")] {
      let arr1: [f32; 4] = cast(self.sse);
      let arr2: [f32; 4] = cast(rhs.sse);
      Self { sse: cast([
        arr1[0] % arr2[0],
        arr1[1] % arr2[1],
        arr1[2] % arr2[2],
        arr1[3] % arr2[3],
      ]) }
    } else {
      Self { arr: [
        self.arr[0] % rhs.arr[0],
        self.arr[1] % rhs.arr[1],
        self.arr[2] % rhs.arr[2],
        self.arr[3] % rhs.arr[3],
      ] }
    }}
  }
}

impl Rem<&'_ f32x4> for f32x4 {
  type Output = Self;
  #[inline]
  fn rem(self, rhs: &Self) -> Self {
    cfg_if! {if #[cfg(target_feature="sse")] {
      let arr1: [f32; 4] = cast(self.sse);
      let arr2: [f32; 4] = cast(rhs.sse);
      Self { sse: cast([
        arr1[0] % arr2[0],
        arr1[1] % arr2[1],
        arr1[2] % arr2[2],
        arr1[3] % arr2[3],
      ]) }
    } else {
      Self { arr: [
        self.arr[0] % rhs.arr[0],
        self.arr[1] % rhs.arr[1],
        self.arr[2] % rhs.arr[2],
        self.arr[3] % rhs.arr[3],
      ] }
    }}
  }
}

impl Sub for f32x4 {
  type Output = Self;
  #[inline]
  fn sub(self, rhs: Self) -> Self {
    cfg_if! {if #[cfg(target_feature="sse")] {
      Self { sse: self.sse.sub(rhs.sse) }
    } else {
      Self { arr: [
        self.arr[0] - rhs.arr[0],
        self.arr[1] - rhs.arr[1],
        self.arr[2] - rhs.arr[2],
        self.arr[3] - rhs.arr[3],
      ] }
    }}
  }
}

impl Sub<&'_ f32x4> for f32x4 {
  type Output = Self;
  #[inline]
  fn sub(self, rhs: &Self) -> Self {
    cfg_if! {if #[cfg(target_feature="sse")] {
      Self { sse: self.sse.sub(rhs.sse) }
    } else {
      Self { arr: [
        self.arr[0] - rhs.arr[0],
        self.arr[1] - rhs.arr[1],
        self.arr[2] - rhs.arr[2],
        self.arr[3] - rhs.arr[3],
      ] }
    }}
  }
}

impl BitAnd for f32x4 {
  type Output = Self;
  #[inline]
  fn bitand(self, rhs: Self) -> Self {
    cfg_if! {if #[cfg(target_feature="sse")] {
      Self { sse: self.sse.bitand(rhs.sse) }
    } else {
      Self { arr: [
        f32::from_bits(self.arr[0].to_bits() & rhs.arr[0].to_bits()),
        f32::from_bits(self.arr[1].to_bits() & rhs.arr[1].to_bits()),
        f32::from_bits(self.arr[2].to_bits() & rhs.arr[2].to_bits()),
        f32::from_bits(self.arr[3].to_bits() & rhs.arr[3].to_bits()),
      ] }
    }}
  }
}

impl BitAnd<&'_ f32x4> for f32x4 {
  type Output = Self;
  #[inline]
  fn bitand(self, rhs: &Self) -> Self {
    self & *rhs
  }
}

impl BitOr for f32x4 {
  type Output = Self;
  #[inline]
  fn bitor(self, rhs: Self) -> Self {
    cfg_if! {if #[cfg(target_feature="sse")] {
      Self { sse: self.sse.bitor(rhs.sse) }
    } else {
      Self { arr: [
        f32::from_bits(self.arr[0].to_bits() | rhs.arr[0].to_bits()),
        f32::from_bits(self.arr[1].to_bits() | rhs.arr[1].to_bits()),
        f32::from_bits(self.arr[2].to_bits() | rhs.arr[2].to_bits()),
        f32::from_bits(self.arr[3].to_bits() | rhs.arr[3].to_bits()),
      ] }
    }}
  }
}

impl BitOr<&'_ f32x4> for f32x4 {
  type Output = Self;
  #[inline]
  fn bitor(self, rhs: &Self) -> Self {
    self | *rhs
  }
}

impl BitXor for f32x4 {
  type Output = Self;
  #[inline]
  fn bitxor(self, rhs: Self) -> Self {
    cfg_if! {if #[cfg(target_feature="sse")] {
      Self { sse: self.sse.bitxor(rhs.sse) }
    } else {
      Self { arr: [
        f32::from_bits(self.arr[0].to_bits() ^ rhs.arr[0].to_bits()),
        f32::from_bits(self.arr[1].to_bits() ^ rhs.arr[1].to_bits()),
        f32::from_bits(self.arr[2].to_bits() ^ rhs.arr[2].to_bits()),
        f32::from_bits(self.arr[3].to_bits() ^ rhs.arr[3].to_bits()),
      ] }
    }}
  }
}

impl BitXor<&'_ f32x4> for f32x4 {
  type Output = Self;
  #[inline]
  fn bitxor(self, rhs: &Self) -> Self {
    self ^ *rhs
  }
}

impl f32x4 {
  /// ```rust
  /// use wide::f32x4;
  /// let a = f32x4::new(1.0, 2.0, 3.0, 4.0);
  /// let b = f32x4::from(2.5);
  /// assert_eq!(a.cmp_lt(b).move_mask(), 0b1100);
  /// ```
  #[inline]
  pub fn move_mask(self) -> i32 {
    cfg_if! {if #[cfg(target_feature="sse")] {
      self.sse.move_mask()
    } else {
      let mut out = 0_i32;
      for i in 0..4 {
        if cast::<f32, i32>(self.arr[i]) < 0 {
          out |= (1<<i);
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
    cfg_if! {if #[cfg(feature = "toolchain_nightly")] {
      use core::intrinsics::fabsf32;
      let a: [f32; 4] = cast(self);
      cast(unsafe {
        [fabsf32(a[0]), fabsf32(a[1]), fabsf32(a[2]), fabsf32(a[3])]
      })
    } else {
      let a: [f32; 4] = cast(self);
      cast([a[0].abs(), a[1].abs(), a[2].abs(), a[3].abs()])
    }}
  }

  #[inline]
  pub fn cos(self) -> Self {
    cfg_if! {if #[cfg(feature = "toolchain_nightly")] {
      use core::intrinsics::cosf32;
      let a: [f32; 4] = cast(self);
      cast(unsafe {
        [cosf32(a[0]), cosf32(a[1]), cosf32(a[2]), cosf32(a[3])]
      })
    } else {
      let a: [f32; 4] = cast(self);
      cast([a[0].cos(), a[1].cos(), a[2].cos(), a[3].cos()])
    }}
  }

  #[inline]
  pub fn exp2(self) -> Self {
    cfg_if! {if #[cfg(feature = "toolchain_nightly")] {
      use core::intrinsics::exp2f32;
      let a: [f32; 4] = cast(self);
      cast(unsafe {
        [exp2f32(a[0]), exp2f32(a[1]), exp2f32(a[2]), exp2f32(a[3])]
      })
    } else {
      let a: [f32; 4] = cast(self);
      cast([a[0].exp2(), a[1].exp2(), a[2].exp2(), a[3].exp2()])
    }}
  }

  #[inline]
  pub fn exp(self) -> Self {
    cfg_if! {if #[cfg(feature = "toolchain_nightly")] {
      use core::intrinsics::expf32;
      let a: [f32; 4] = cast(self);
      cast(unsafe {
        [expf32(a[0]), expf32(a[1]), expf32(a[2]), expf32(a[3])]
      })
    } else {
      let a: [f32; 4] = cast(self);
      cast([a[0].exp(), a[1].exp(), a[2].exp(), a[3].exp()])
    }}
  }

  #[inline]
  pub fn log10(self) -> Self {
    cfg_if! {if #[cfg(feature = "toolchain_nightly")] {
      use core::intrinsics::log10f32;
      let a: [f32; 4] = cast(self);
      cast(unsafe {
        [log10f32(a[0]), log10f32(a[1]), log10f32(a[2]), log10f32(a[3])]
      })
    } else {
      let a: [f32; 4] = cast(self);
      cast([a[0].log10(), a[1].log10(), a[2].log10(), a[3].log10()])
    }}
  }

  #[inline]
  pub fn log2(self) -> Self {
    cfg_if! {if #[cfg(feature = "toolchain_nightly")] {
      use core::intrinsics::log2f32;
      let a: [f32; 4] = cast(self);
      cast(unsafe {
        [log2f32(a[0]), log2f32(a[1]), log2f32(a[2]), log2f32(a[3])]
      })
    } else {
      let a: [f32; 4] = cast(self);
      cast([a[0].log2(), a[1].log2(), a[2].log2(), a[3].log2()])
    }}
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

  #[inline]
  pub fn sin(self) -> Self {
    cfg_if! {if #[cfg(feature = "toolchain_nightly")] {
      use core::intrinsics::sinf32;
      let a: [f32; 4] = cast(self);
      cast(unsafe {
        [sinf32(a[0]), sinf32(a[1]), sinf32(a[2]), sinf32(a[3])]
      })
    } else {
      let a: [f32; 4] = cast(self);
      cast([a[0].sin(), a[1].sin(), a[2].sin(), a[3].sin()])
    }}
  }

  #[inline]
  pub fn sqrt(self) -> Self {
    cfg_if! {if #[cfg(feature = "toolchain_nightly")] {
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

  #[inline]
  pub fn trunc(self) -> Self {
    cfg_if! {if #[cfg(feature = "toolchain_nightly")] {
      use core::intrinsics::truncf32;
      let a: [f32; 4] = cast(self);
      cast(unsafe {
        [truncf32(a[0]), truncf32(a[1]), truncf32(a[2]), truncf32(a[3])]
      })
    } else {
      let a: [f32; 4] = cast(self);
      cast([a[0].trunc(), a[1].trunc(), a[2].trunc(), a[3].trunc()])
    }}
  }

  #[inline]
  pub fn copysign(self, b: Self) -> Self {
    cfg_if! {if #[cfg(feature = "toolchain_nightly")] {
      use core::intrinsics::copysignf32;
      let a: [f32; 4] = cast(self);
      let b: [f32; 4] = cast(b);
      cast(unsafe { [
        copysignf32(a[0], b[0]),
        copysignf32(a[1], b[1]),
        copysignf32(a[2], b[2]),
        copysignf32(a[3], b[3]),
      ]})
    } else {
      let a: [f32; 4] = cast(self);
      let b: [f32; 4] = cast(b);
      cast([
        a[0].copysign(b[0]),
        a[1].copysign(b[1]),
        a[2].copysign(b[2]),
        a[3].copysign(b[3]),
      ])
    }}
  }

  #[inline]
  pub fn ln(self) -> Self {
    cfg_if! {if #[cfg(feature = "toolchain_nightly")] {
      use core::intrinsics::logf32;
      let a: [f32; 4] = cast(self);
      cast(unsafe {
        [logf32(a[0]), logf32(a[1]), logf32(a[2]), logf32(a[3])]
      })
    } else {
      let a: [f32; 4] = cast(self);
      cast([a[0].ln(), a[1].ln(), a[2].ln(), a[3].ln()])
    }}
  }

  #[inline]
  pub fn powf(self, b: Self) -> Self {
    cfg_if! {if #[cfg(feature = "toolchain_nightly")] {
      use core::intrinsics::powf32;
      let a: [f32; 4] = cast(self);
      let b: [f32; 4] = cast(b);
      cast(unsafe { [
        powf32(a[0], b[0]),
        powf32(a[1], b[1]),
        powf32(a[2], b[2]),
        powf32(a[3], b[3]),
      ]})
    } else {
      let a: [f32; 4] = cast(self);
      let b: [f32; 4] = cast(b);
      cast([
        a[0].powf(b[0]),
        a[1].powf(b[1]),
        a[2].powf(b[2]),
        a[3].powf(b[3]),
      ])
    }}
  }

  #[inline]
  pub fn powi(self, b: [i32; 4]) -> Self {
    cfg_if! {if #[cfg(feature = "toolchain_nightly")] {
      use core::intrinsics::powif32;
      let a: [f32; 4] = cast(self);
      cast(unsafe { [
        powif32(a[0], b[0]),
        powif32(a[1], b[1]),
        powif32(a[2], b[2]),
        powif32(a[3], b[3]),
      ]})
    } else {
      let a: [f32; 4] = cast(self);
      cast([
        a[0].powi(b[0]),
        a[1].powi(b[1]),
        a[2].powi(b[2]),
        a[3].powi(b[3]),
      ])
    }}
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
  /// Fused if `fma` feature is enabled, see [`mul_add`].
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
}

// // //
// CODE AFTER HERE SHOULD NOT USE CONDITIONAL COMPILATION!
// // //
// (We want to keep per-platform code away from the universal code)
// // //

impl f32x4 {
  // METHODS AVAILABLE IN CORE

  #[inline]
  pub fn classify(self) -> [core::num::FpCategory; 4] {
    let a: [f32; 4] = cast(self);
    [
      a[0].classify(),
      a[1].classify(),
      a[2].classify(),
      a[3].classify(),
    ]
  }

  #[inline]
  pub fn to_degrees(self) -> Self {
    let a: [f32; 4] = cast(self);
    cast([
      a[0].to_degrees(),
      a[1].to_degrees(),
      a[2].to_degrees(),
      a[3].to_degrees(),
    ])
  }

  #[inline]
  pub fn to_radians(self) -> Self {
    let a: [f32; 4] = cast(self);
    cast([
      a[0].to_radians(),
      a[1].to_radians(),
      a[2].to_radians(),
      a[3].to_radians(),
    ])
  }

  // REQUIRES EXTERNAL MATH LIBS

  #[inline]
  pub fn clamp(self, min: Self, max: Self) -> Self {
    self.max(min).min(max)
  }

  #[inline]
  pub fn fract(self) -> Self {
    let a: [f32; 4] = cast(self);
    cast([a[0].fract(), a[1].fract(), a[2].fract(), a[3].fract()])
  }

  #[inline]
  pub fn acos(self) -> Self {
    let a: [f32; 4] = cast(self);
    cast([a[0].acos(), a[1].acos(), a[2].acos(), a[3].acos()])
  }

  #[inline]
  pub fn acosh(self) -> Self {
    let a: [f32; 4] = cast(self);
    cast([a[0].acosh(), a[1].acosh(), a[2].acosh(), a[3].acosh()])
  }

  #[inline]
  pub fn asin(self) -> Self {
    let a: [f32; 4] = cast(self);
    cast([a[0].asin(), a[1].asin(), a[2].asin(), a[3].asin()])
  }

  #[inline]
  pub fn asinh(self) -> Self {
    let a: [f32; 4] = cast(self);
    cast([a[0].asinh(), a[1].asinh(), a[2].asinh(), a[3].asinh()])
  }

  #[inline]
  pub fn atan(self) -> Self {
    let a: [f32; 4] = cast(self);
    cast([a[0].atan(), a[1].atan(), a[2].atan(), a[3].atan()])
  }

  #[inline]
  pub fn atanh(self) -> Self {
    let a: [f32; 4] = cast(self);
    cast([a[0].atanh(), a[1].atanh(), a[2].atanh(), a[3].atanh()])
  }

  #[inline]
  pub fn cbrt(self) -> Self {
    let a: [f32; 4] = cast(self);
    cast([a[0].cbrt(), a[1].cbrt(), a[2].cbrt(), a[3].cbrt()])
  }

  #[inline]
  pub fn cosh(self) -> Self {
    let a: [f32; 4] = cast(self);
    cast([a[0].cosh(), a[1].cosh(), a[2].cosh(), a[3].cosh()])
  }

  #[inline]
  pub fn exp_m1(self) -> Self {
    let a: [f32; 4] = cast(self);
    cast([a[0].exp_m1(), a[1].exp_m1(), a[2].exp_m1(), a[3].exp_m1()])
  }

  #[inline]
  pub fn ln_1p(self) -> Self {
    let a: [f32; 4] = cast(self);
    cast([a[0].ln_1p(), a[1].ln_1p(), a[2].ln_1p(), a[3].ln_1p()])
  }

  #[inline]
  pub fn log(self, b: Self) -> Self {
    let a: [f32; 4] = cast(self);
    let b: [f32; 4] = cast(b);
    cast([
      a[0].log(b[0]),
      a[1].log(b[1]),
      a[2].log(b[2]),
      a[3].log(b[3]),
    ])
  }

  #[inline]
  pub fn signum(self) -> Self {
    let a: [f32; 4] = cast(self);
    cast([a[0].signum(), a[1].signum(), a[2].signum(), a[3].signum()])
  }

  #[inline]
  pub fn sinh(self) -> Self {
    let a: [f32; 4] = cast(self);
    cast([a[0].sinh(), a[1].sinh(), a[2].sinh(), a[3].sinh()])
  }

  #[inline]
  pub fn tan(self) -> Self {
    let a: [f32; 4] = cast(self);
    cast([a[0].tan(), a[1].tan(), a[2].tan(), a[3].tan()])
  }

  #[inline]
  pub fn tanh(self) -> Self {
    let a: [f32; 4] = cast(self);
    cast([a[0].tanh(), a[1].tanh(), a[2].tanh(), a[3].tanh()])
  }

  #[inline]
  pub fn atan2(self, b: Self) -> Self {
    let a: [f32; 4] = cast(self);
    let b: [f32; 4] = cast(b);
    cast([
      a[0].atan2(b[0]),
      a[1].atan2(b[1]),
      a[2].atan2(b[2]),
      a[3].atan2(b[3]),
    ])
  }

  #[inline]
  pub fn hypot(self, b: Self) -> Self {
    let a: [f32; 4] = cast(self);
    let b: [f32; 4] = cast(b);
    cast([
      a[0].hypot(b[0]),
      a[1].hypot(b[1]),
      a[2].hypot(b[2]),
      a[3].hypot(b[3]),
    ])
  }

  #[inline]
  pub fn sin_cos(self) -> (Self, Self) {
    (self.sin(), self.cos())
  }

  /// calculates polynomial c2*x^2 + c1*x + c0
  ///
  /// https://github.com/vectorclass/version2/blob/master/vectormath_common.h#L111
  #[inline]
  fn polynomial_2(self, c0: Self, c1: Self, c2: Self) -> Self {
    let self2 = self * self;
    self2.mul_add(c2, self.mul_add(c1, c0))
  }

  /// calculates polynomial c3*x^3 + c2*x^2 + c1*x + c0
  ///
  /// https://github.com/vectorclass/version2/blob/master/vectormath_common.h#L120
  #[inline]
  fn polynomial_3(self, c0: Self, c1: Self, c2: Self, c3: Self) -> Self {
    let self2 = self * self;
    c3.mul_add(self, c2).mul_add(self2, c1.mul_add(self, c0))
  }

  #[allow(clippy::unreadable_literal)]
  #[allow(clippy::excessive_precision)]
  #[allow(bad_style)]
  pub fn s_c_t(xx: Self) -> (Self, Self, Self) {
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

    let xa = xx.abs();

    let y = (xa * Self::TWO_PI).round();
    let q = y.sse.round_i32();

    let x = y.negated_mul_add(DP3F, y.negated_mul_add(DP2F, y.negated_mul_add(DP1F, xa)));
    let x2 = x * x;
    let s = x2.polynomial_2(P0sinf, P1sinf, P2sinf) * (x * x2) + x;
    let c = x2.polynomial_2(P0cosf, P1cosf, P2cosf) * (x2 * x2)
      + Self::HALF.negated_mul_add(x2, Self::ONE);

    // stop it rustfmt
    unimplemented!();
  }
}

impl Clone for f32x4 {
  #[inline(always)]
  fn clone(&self) -> Self {
    *self
  }
}
impl Copy for f32x4 {}
impl Default for f32x4 {
  #[inline(always)]
  fn default() -> Self {
    Self::zeroed()
  }
}
unsafe impl Zeroable for f32x4 {}
unsafe impl Pod for f32x4 {}

impl Index<usize> for f32x4 {
  type Output = f32;
  #[inline(always)]
  fn index(&self, index: usize) -> &f32 {
    let r: &[f32; 4] = cast_ref(self);
    &r[index]
  }
}
impl IndexMut<usize> for f32x4 {
  #[inline(always)]
  fn index_mut(&mut self, index: usize) -> &mut f32 {
    let r: &mut [f32; 4] = cast_mut(self);
    &mut r[index]
  }
}

impl AddAssign for f32x4 {
  #[inline]
  fn add_assign(&mut self, rhs: Self) {
    *self = *self + rhs
  }
}

impl AddAssign<&'_ f32x4> for f32x4 {
  #[inline]
  fn add_assign(&mut self, rhs: &Self) {
    *self = *self + rhs
  }
}

impl DivAssign for f32x4 {
  #[inline]
  fn div_assign(&mut self, rhs: Self) {
    *self = *self / rhs
  }
}

impl DivAssign<&'_ f32x4> for f32x4 {
  #[inline]
  fn div_assign(&mut self, rhs: &Self) {
    *self = *self / rhs
  }
}

impl MulAssign for f32x4 {
  #[inline]
  fn mul_assign(&mut self, rhs: Self) {
    *self = *self * rhs
  }
}

impl MulAssign<&'_ f32x4> for f32x4 {
  #[inline]
  fn mul_assign(&mut self, rhs: &Self) {
    *self = *self * rhs
  }
}

impl RemAssign for f32x4 {
  #[inline]
  fn rem_assign(&mut self, rhs: Self) {
    *self = *self % rhs
  }
}

impl RemAssign<&'_ f32x4> for f32x4 {
  #[inline]
  fn rem_assign(&mut self, rhs: &Self) {
    *self = *self % rhs
  }
}

impl SubAssign for f32x4 {
  #[inline]
  fn sub_assign(&mut self, rhs: Self) {
    *self = *self - rhs
  }
}

impl SubAssign<&'_ f32x4> for f32x4 {
  #[inline]
  fn sub_assign(&mut self, rhs: &Self) {
    *self = *self - rhs
  }
}

impl BitAndAssign for f32x4 {
  #[inline]
  fn bitand_assign(&mut self, rhs: Self) {
    *self = *self & rhs
  }
}

impl BitAndAssign<&'_ f32x4> for f32x4 {
  #[inline]
  fn bitand_assign(&mut self, rhs: &Self) {
    *self = *self & rhs
  }
}

impl BitOrAssign for f32x4 {
  #[inline]
  fn bitor_assign(&mut self, rhs: Self) {
    *self = *self | rhs
  }
}

impl BitOrAssign<&'_ f32x4> for f32x4 {
  #[inline]
  fn bitor_assign(&mut self, rhs: &Self) {
    *self = *self | rhs
  }
}

impl BitXorAssign for f32x4 {
  #[inline]
  fn bitxor_assign(&mut self, rhs: Self) {
    *self = *self ^ rhs
  }
}

impl BitXorAssign<&'_ f32x4> for f32x4 {
  #[inline]
  fn bitxor_assign(&mut self, rhs: &Self) {
    *self = *self ^ rhs
  }
}

impl Debug for f32x4 {
  fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
    write!(f, "f32x4(")?;
    Debug::fmt(&self[0], f)?;
    write!(f, ", ")?;
    Debug::fmt(&self[1], f)?;
    write!(f, ", ")?;
    Debug::fmt(&self[2], f)?;
    write!(f, ", ")?;
    Debug::fmt(&self[3], f)?;
    write!(f, ")")
  }
}

impl Display for f32x4 {
  fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
    write!(f, "f32x4(")?;
    Display::fmt(&self[0], f)?;
    write!(f, ", ")?;
    Display::fmt(&self[1], f)?;
    write!(f, ", ")?;
    Display::fmt(&self[2], f)?;
    write!(f, ", ")?;
    Display::fmt(&self[3], f)?;
    write!(f, ")")
  }
}

impl LowerExp for f32x4 {
  fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
    write!(f, "f32x4(")?;
    LowerExp::fmt(&self[0], f)?;
    write!(f, ", ")?;
    LowerExp::fmt(&self[1], f)?;
    write!(f, ", ")?;
    LowerExp::fmt(&self[2], f)?;
    write!(f, ", ")?;
    LowerExp::fmt(&self[3], f)?;
    write!(f, ")")
  }
}

impl UpperExp for f32x4 {
  fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
    write!(f, "f32x4(")?;
    UpperExp::fmt(&self[0], f)?;
    write!(f, ", ")?;
    UpperExp::fmt(&self[1], f)?;
    write!(f, ", ")?;
    UpperExp::fmt(&self[2], f)?;
    write!(f, ", ")?;
    UpperExp::fmt(&self[3], f)?;
    write!(f, ")")
  }
}

impl Binary for f32x4 {
  fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
    write!(f, "f32x4(")?;
    Binary::fmt(&cast::<f32, u32>(self[0]), f)?;
    write!(f, ", ")?;
    Binary::fmt(&cast::<f32, u32>(self[0]), f)?;
    write!(f, ", ")?;
    Binary::fmt(&cast::<f32, u32>(self[0]), f)?;
    write!(f, ", ")?;
    Binary::fmt(&cast::<f32, u32>(self[0]), f)?;
    write!(f, ")")
  }
}

impl LowerHex for f32x4 {
  fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
    write!(f, "f32x4(")?;
    LowerHex::fmt(&cast::<f32, u32>(self[0]), f)?;
    write!(f, ", ")?;
    LowerHex::fmt(&cast::<f32, u32>(self[0]), f)?;
    write!(f, ", ")?;
    LowerHex::fmt(&cast::<f32, u32>(self[0]), f)?;
    write!(f, ", ")?;
    LowerHex::fmt(&cast::<f32, u32>(self[0]), f)?;
    write!(f, ")")
  }
}

impl Octal for f32x4 {
  fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
    write!(f, "f32x4(")?;
    Octal::fmt(&cast::<f32, u32>(self[0]), f)?;
    write!(f, ", ")?;
    Octal::fmt(&cast::<f32, u32>(self[0]), f)?;
    write!(f, ", ")?;
    Octal::fmt(&cast::<f32, u32>(self[0]), f)?;
    write!(f, ", ")?;
    Octal::fmt(&cast::<f32, u32>(self[0]), f)?;
    write!(f, ")")
  }
}

impl UpperHex for f32x4 {
  fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
    write!(f, "f32x4(")?;
    UpperHex::fmt(&cast::<f32, u32>(self[0]), f)?;
    write!(f, ", ")?;
    UpperHex::fmt(&cast::<f32, u32>(self[0]), f)?;
    write!(f, ", ")?;
    UpperHex::fmt(&cast::<f32, u32>(self[0]), f)?;
    write!(f, ", ")?;
    UpperHex::fmt(&cast::<f32, u32>(self[0]), f)?;
    write!(f, ")")
  }
}

impl Neg for f32x4 {
  type Output = f32x4;
  #[inline]
  fn neg(self) -> f32x4 {
    f32x4::new(0.0, 0.0, 0.0, 0.0) - self
  }
}
impl Neg for &'_ f32x4 {
  type Output = f32x4;
  #[inline]
  fn neg(self) -> f32x4 {
    f32x4::new(0.0, 0.0, 0.0, 0.0) - self
  }
}

impl core::iter::Product for f32x4 {
  #[inline]
  fn product<I: Iterator<Item = f32x4>>(iter: I) -> f32x4 {
    let mut total = f32x4::new(1.0, 1.0, 1.0, 1.0);
    for i in iter {
      total *= i;
    }
    total
  }
}
impl<'a> core::iter::Product<&'a f32x4> for f32x4 {
  #[inline]
  fn product<I: Iterator<Item = &'a f32x4>>(iter: I) -> f32x4 {
    let mut total = f32x4::new(1.0, 1.0, 1.0, 1.0);
    for i in iter {
      total *= i;
    }
    total
  }
}

impl core::iter::Sum for f32x4 {
  #[inline]
  fn sum<I: Iterator<Item = f32x4>>(iter: I) -> f32x4 {
    let mut total = f32x4::new(0.0, 0.0, 0.0, 0.0);
    for i in iter {
      total += i;
    }
    total
  }
}
impl<'a> core::iter::Sum<&'a f32x4> for f32x4 {
  #[inline]
  fn sum<I: Iterator<Item = &'a f32x4>>(iter: I) -> f32x4 {
    let mut total = f32x4::new(0.0, 0.0, 0.0, 0.0);
    for i in iter {
      total += i;
    }
    total
  }
}

impl AsRef<[f32; 4]> for f32x4 {
  #[inline(always)]
  fn as_ref(&self) -> &[f32; 4] {
    cast_ref(self)
  }
}
impl AsMut<[f32; 4]> for f32x4 {
  #[inline(always)]
  fn as_mut(&mut self) -> &mut [f32; 4] {
    cast_mut(self)
  }
}

impl From<[f32; 4]> for f32x4 {
  #[inline(always)]
  fn from(arr: [f32; 4]) -> Self {
    cast(arr)
  }
}
impl From<(f32, f32, f32, f32)> for f32x4 {
  #[inline(always)]
  fn from((a, b, c, d): (f32, f32, f32, f32)) -> Self {
    Self::new(a, b, c, d)
  }
}

impl From<[i8; 4]> for f32x4 {
  #[inline]
  fn from([a, b, c, d]: [i8; 4]) -> Self {
    Self::new(f32::from(a), f32::from(b), f32::from(c), f32::from(d))
  }
}

impl From<[u8; 4]> for f32x4 {
  #[inline]
  fn from([a, b, c, d]: [u8; 4]) -> Self {
    Self::new(f32::from(a), f32::from(b), f32::from(c), f32::from(d))
  }
}

impl From<[i16; 4]> for f32x4 {
  #[inline]
  fn from([a, b, c, d]: [i16; 4]) -> Self {
    Self::new(f32::from(a), f32::from(b), f32::from(c), f32::from(d))
  }
}

impl From<[u16; 4]> for f32x4 {
  #[inline]
  fn from([a, b, c, d]: [u16; 4]) -> Self {
    Self::new(f32::from(a), f32::from(b), f32::from(c), f32::from(d))
  }
}
