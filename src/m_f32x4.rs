use super::*;

mod wide_methods;
mod wide_trait_impls;

magic! {
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

#[allow(non_camel_case_types)]
#[repr(C, align(16))]
pub union ConstUnionHack_f32x4 {
  pub narrow_arr: [f32; 4],
  pub wide_thing: f32x4,
  pub u: u128,
  pub i32_arr: [i32; 4],
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
/// use wide::*;
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
/// use wide::*;
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

// consts
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

  /// All bits active.
  pub const ALL_BITS_ACTIVE: f32x4 = {
    let cuh = ConstUnionHack_f32x4 { u: u128::max_value() };
    unsafe { cuh.wide_thing }
  };

  /// All bits active.
  pub const ALL_EXCEPT_SIGN: f32x4 = {
    let cuh = ConstUnionHack_f32x4 {
      i32_arr: [
        i32::max_value(),
        i32::max_value(),
        i32::max_value(),
        i32::max_value(),
      ],
    };
    unsafe { cuh.wide_thing }
  };

  const_f32_as_f32x4!(
    /// 0.0
    pub ZERO, 0.0
  );

  const_f32_as_f32x4!(
    /// -0.0
    pub NEGATIVE_ZERO, -0.0
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
    /// 1.0
    pub NEGATIVE_ONE, -1.0
  );

  const_f32_as_f32x4!(
    /// 2.0 * π, the number of radians in a circle.
    pub TWO_PI, 2.0 * core::f32::consts::PI
  );
}

impl f32x4 {
  #[inline(always)]
  pub fn new(a: f32, b: f32, c: f32, d: f32) -> Self {
    magic! {if #[cfg(target_feature="sse")] {
      Self { sse: m128::set_reverse(a,b,c,d) }
    } else {
      Self { arr: [a,b,c,d] }
    }}
  }
}

impl Rem for f32x4 {
  type Output = Self;
  #[inline]
  fn rem(self, rhs: Self) -> Self {
    magic! {if #[cfg(target_feature="sse")] {
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
    magic! {if #[cfg(target_feature="sse")] {
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

impl f32x4 {
  #[inline]
  pub fn exp2(self) -> Self {
    magic! {if #[cfg(feature = "toolchain_nightly")] {
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
    magic! {if #[cfg(feature = "toolchain_nightly")] {
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
    magic! {if #[cfg(feature = "toolchain_nightly")] {
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
    magic! {if #[cfg(feature = "toolchain_nightly")] {
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
  pub fn trunc(self) -> Self {
    magic! {if #[cfg(feature = "toolchain_nightly")] {
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
  pub fn ln(self) -> Self {
    magic! {if #[cfg(feature = "toolchain_nightly")] {
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
    magic! {if #[cfg(feature = "toolchain_nightly")] {
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
    magic! {if #[cfg(feature = "toolchain_nightly")] {
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

  #[inline]
  pub fn classify(self) -> [core::num::FpCategory; 4] {
    let a: [f32; 4] = cast(self);
    [a[0].classify(), a[1].classify(), a[2].classify(), a[3].classify()]
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
    cast([a[0].log(b[0]), a[1].log(b[1]), a[2].log(b[2]), a[3].log(b[3])])
  }

  #[inline]
  pub fn sinh(self) -> Self {
    let a: [f32; 4] = cast(self);
    cast([a[0].sinh(), a[1].sinh(), a[2].sinh(), a[3].sinh()])
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
}

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
    write!(f, "[")?;
    Display::fmt(&self[0], f)?;
    write!(f, ", ")?;
    Display::fmt(&self[1], f)?;
    write!(f, ", ")?;
    Display::fmt(&self[2], f)?;
    write!(f, ", ")?;
    Display::fmt(&self[3], f)?;
    write!(f, "]")
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
    Binary::fmt(&cast::<f32, u32>(self[1]), f)?;
    write!(f, ", ")?;
    Binary::fmt(&cast::<f32, u32>(self[2]), f)?;
    write!(f, ", ")?;
    Binary::fmt(&cast::<f32, u32>(self[3]), f)?;
    write!(f, ")")
  }
}

impl LowerHex for f32x4 {
  fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
    write!(f, "f32x4(")?;
    LowerHex::fmt(&cast::<f32, u32>(self[0]), f)?;
    write!(f, ", ")?;
    LowerHex::fmt(&cast::<f32, u32>(self[1]), f)?;
    write!(f, ", ")?;
    LowerHex::fmt(&cast::<f32, u32>(self[2]), f)?;
    write!(f, ", ")?;
    LowerHex::fmt(&cast::<f32, u32>(self[3]), f)?;
    write!(f, ")")
  }
}

impl Octal for f32x4 {
  fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
    write!(f, "f32x4(")?;
    Octal::fmt(&cast::<f32, u32>(self[0]), f)?;
    write!(f, ", ")?;
    Octal::fmt(&cast::<f32, u32>(self[1]), f)?;
    write!(f, ", ")?;
    Octal::fmt(&cast::<f32, u32>(self[2]), f)?;
    write!(f, ", ")?;
    Octal::fmt(&cast::<f32, u32>(self[3]), f)?;
    write!(f, ")")
  }
}

impl UpperHex for f32x4 {
  fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
    write!(f, "f32x4(")?;
    UpperHex::fmt(&cast::<f32, u32>(self[0]), f)?;
    write!(f, ", ")?;
    UpperHex::fmt(&cast::<f32, u32>(self[1]), f)?;
    write!(f, ", ")?;
    UpperHex::fmt(&cast::<f32, u32>(self[2]), f)?;
    write!(f, ", ")?;
    UpperHex::fmt(&cast::<f32, u32>(self[3]), f)?;
    write!(f, ")")
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
