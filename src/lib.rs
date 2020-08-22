//#![no_std]
#![allow(non_camel_case_types)]

//! A crate to help you go wide.
//!
//! This crate provides SIMD-compatible data types.
//!
//! When possible, explicit SIMD is used with all the math operations here. As a
//! fallback, the fact the fact that all the lengths of a fixed length array are
//! doing the same thing will often make LLVM notice that it should use SIMD
//! instructions to complete the task. In the worst case, the code just becomes
//! totally scalar (though the math is still correct, at least).
//!
//! ## Crate Features
//!
//! * `std`: This causes the feature to link to `std`.
//!   * Currently this just improves the performance of `sqrt` when an explicit
//!     SIMD `sqrt` isn't available.

// Note(Lokathor): Due to standard library magic, the std-only methods for f32
// and f64 will automatically be available simply by declaring this.
#[cfg(feature = "std")]
extern crate std;

use core::{
  fmt::{
    Binary, Debug, Display, LowerExp, LowerHex, Octal, UpperExp, UpperHex,
  },
  ops::*,
};

#[allow(unused_imports)]
use safe_arch::*;

use bytemuck::*;

macro_rules! pick {
  ($(if #[cfg($($test:meta),*)] {
      $($if_tokens:tt)*
    })else+ else {
      $($else_tokens:tt)*
    }) => {
    pick!{
      @__forests [ ] ;
      $( [ {$($test),*} {$($if_tokens)*} ], )*
      [ { } {$($else_tokens)*} ],
    }
  };
  (if #[cfg($($if_meta:meta),*)] {
      $($if_tokens:tt)*
    } $(else if #[cfg($($else_meta:meta),*)] {
      $($else_tokens:tt)*
    })*) => {
    pick!{
      @__forests [ ] ;
      [ {$($if_meta),*} {$($if_tokens)*} ],
      $( [ {$($else_meta),*} {$($else_tokens)*} ], )*
    }
  };
  (@__forests [$($not:meta,)*];) => {
    /* halt expansion */
  };
  (@__forests [$($not:meta,)*]; [{$($m:meta),*} {$($tokens:tt)*}], $($rest:tt)*) => {
    #[cfg(all( $($m,)* not(any($($not),*)) ))]
    pick!{ @__identity $($tokens)* }
    pick!{ @__forests [ $($not,)* $($m,)* ] ; $($rest)* }
  };
  (@__identity $($tokens:tt)*) => {
    $($tokens)*
  };
}

// TODO: make these generic over `mul_add`? Worth it?

macro_rules! polynomial_2 {
  ($x:expr, $c0:expr, $c1:expr, $c2:expr $(,)?) => {{
    let x = $x;
    let x2 = x * x;
    x2.mul_add($c2, x.mul_add($c1, $c0))
  }};
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

macro_rules! polynomial_5n {
  ($x:expr, $c0:expr, $c1:expr, $c2:expr, $c3:expr, $c4:expr, $c5:expr $(,)?) => {{
    let x = $x;
    let x2 = x * x;
    let x4 = x2 * x2;
    x2.mul_add(x.mul_add($c3, $c2), (x4.mul_add($c4 + x, x.mul_add($c1, $c0))))
  }};
}

macro_rules! polynomial_8 {
  ($x:expr, $c0:expr, $c1:expr, $c2:expr, $c3:expr, $c4:expr, $c5:expr,  $c6:expr, $c7:expr, $c8:expr $(,)?) => {{
    let x = $x;
    let x2 = x * x;
    let x4 = x2 * x2;
    let x8 = x4 * x4;
    x4.mul_add(
      x2.mul_add($c7.mul_add(x, $c6), x.mul_add($c5, $c4)),
      x8.mul_add($c8, x2.mul_add(x.mul_add($c3, $c2), x.mul_add($c1, $c0))),
    )
  }};
}

macro_rules! polynomial_13 {
  ($x:expr,  $c2:expr, $c3:expr, $c4:expr, $c5:expr,$c6:expr, $c7:expr, $c8:expr,$c9:expr, $c10:expr, $c11:expr, $c12:expr, $c13:expr  $(,)?) => {{
    let x = $x;
    let x2 = x * x;
    let x4 = x2 * x2;
    let x8 = x4 * x4;
    x8.mul_add(
      x4.mul_add(
        x.mul_add($c13, $c12),
        x2.mul_add(x.mul_add($c11, $c10), x.mul_add($c9, $c8)),
      ),
      x4.mul_add(
        x2.mul_add(x.mul_add($c7, $c6), x.mul_add($c5, $c4)),
        x2.mul_add(x.mul_add($c3, $c2), x),
      ),
    )
  }};
}

mod f32x8_;
pub use f32x8_::*;

mod f32x4_;
pub use f32x4_::*;

mod f64x4_;
pub use f64x4_::*;

mod f64x2_;
pub use f64x2_::*;

mod i8x16_;
pub use i8x16_::*;

mod i16x8_;
pub use i16x8_::*;

mod i32x4_;
pub use i32x4_::*;

mod i32x8_;
pub use i32x8_::*;

mod i64x2_;
pub use i64x2_::*;

mod i64x4_;
pub use i64x4_::*;

mod u8x16_;
pub use u8x16_::*;

mod u16x8_;
pub use u16x8_::*;

mod u32x4_;
pub use u32x4_::*;

mod u32x8_;
pub use u32x8_::*;

mod u64x2_;
pub use u64x2_::*;

mod u64x4_;
pub use u64x4_::*;

#[allow(non_camel_case_types)]
#[repr(C, align(16))]
union ConstUnionHack128bit {
  f64a4: [f64; 4],
  f32a8: [f32; 8],
  f32a4: [f32; 4],
  i32a4: [i32; 4],
  i64a4: [i64; 4],
  f64a2: [f64; 2],
  f32x8: f32x8,
  f32x4: f32x4,
  f64x2: f64x2,
  f64x4: f64x4,
  u128: u128,
}

#[allow(dead_code)]
fn generic_bit_blend<T>(mask: T, y: T, n: T) -> T
where
  T: Copy + BitXor<Output = T> + BitAnd<Output = T>,
{
  n ^ ((n ^ y) & mask)
}

/// given `type.op(type)` and type is Copy, impls `type.op(&type)`
macro_rules! bulk_impl_op_ref_self_for {
  ($(($op:ident, $method:ident) => [$($t:ty),+]),+ $(,)?) => {
    $( // do each trait/list matching given
      $( // do the current trait for each type in its list.
        impl $op<&Self> for $t {
          type Output = Self;
          #[inline]
          #[must_use]
          fn $method(self, rhs: &Self) -> Self::Output {
            self.$method(*rhs)
          }
        }
      )+
    )+
  };
}

bulk_impl_op_ref_self_for! {
  (Add, add) => [f32x8, f32x4, f64x4, f64x2, i8x16, i16x8, i32x8, i32x4, i64x2, u8x16, u16x8, u32x8, u32x4, u64x4, u64x2],
  (Sub, sub) => [f32x8, f32x4, f64x4, f64x2, i8x16, i16x8, i32x8, i32x4, i64x2, u8x16, u16x8, u32x8, u32x4, u64x4, u64x2],
  (Mul, mul) => [f32x8, f32x4, f64x4, f64x2, i16x8, i32x8, i32x4],
  (Div, div) => [f32x8, f32x4, f64x4, f64x2],
  (BitAnd, bitand) => [f32x8, f32x4, f64x4, f64x2, i8x16, i16x8, i32x8, i32x4, i64x2, u8x16, u16x8, u32x8, u32x4, u64x4, u64x2],
  (BitOr, bitor) => [f32x8, f32x4, f64x4, f64x2, i8x16, i16x8, i32x8, i32x4, i64x2, u8x16, u16x8, u32x8, u32x4, u64x4, u64x2],
  (BitXor, bitxor) => [f32x8, f32x4, f64x4, f64x2, i8x16, i16x8, i32x8, i32x4, i64x2, u8x16, u16x8, u32x8, u32x4, u64x4, u64x2],
}

/// given `type.op(rhs)` and type is Copy, impls `type.op_assign(rhs)`
macro_rules! bulk_impl_op_assign_for {
  ($(($op:ident<$rhs:ty>, $method:ident, $method_assign:ident) => [$($t:ty),+]),+ $(,)?) => {
    $( // do each trait/list matching given
      $( // do the current trait for each type in its list.
        impl $op<$rhs> for $t {
          #[inline]
          fn $method_assign(&mut self, rhs: $rhs) {
            *self = self.$method(rhs);
          }
        }
      )+
    )+
  };
}

// Note: remember to update bulk_impl_op_ref_self_for first or this will give
// weird errors!
bulk_impl_op_assign_for! {
  (AddAssign<Self>, add, add_assign) => [f32x8, f32x4, f64x4, f64x2, i8x16, i16x8, i32x8, i32x4, i64x2, u8x16, u16x8, u32x8, u32x4, u64x4, u64x2],
  (AddAssign<&Self>, add, add_assign) => [f32x8, f32x4, f64x4, f64x2, i8x16, i16x8, i32x8, i32x4, i64x2, u8x16, u16x8, u32x8, u32x4, u64x4, u64x2],
  (SubAssign<Self>, sub, sub_assign) => [f32x8, f32x4, f64x4, f64x2, i8x16, i16x8, i32x8, i32x4, i64x2, u8x16, u16x8, u32x8, u32x4, u64x4, u64x2],
  (SubAssign<&Self>, sub, sub_assign) => [f32x8, f32x4, f64x4, f64x2, i8x16, i16x8, i32x8, i32x4, i64x2, u8x16, u16x8, u32x8, u32x4, u64x4, u64x2],
  (MulAssign<Self>, mul, mul_assign) => [f32x8, f32x4, f64x4, f64x2, i16x8, i32x8, i32x4],
  (MulAssign<&Self>, mul, mul_assign) => [f32x8, f32x4, f64x4, f64x2, i16x8, i32x8, i32x4],
  (DivAssign<Self>, div, div_assign) => [f32x8, f32x4, f64x4, f64x2],
  (DivAssign<&Self>, div, div_assign) => [f32x8, f32x4, f64x4, f64x2],
  (BitAndAssign<Self>, bitand, bitand_assign) => [f32x8, f32x4, f64x4, f64x2, i8x16, i16x8, i32x8, i32x4, i64x2, u8x16, u16x8, u32x8, u32x4, u64x4, u64x2],
  (BitAndAssign<&Self>, bitand, bitand_assign) => [f32x8, f32x4, f64x4, f64x2, i8x16, i16x8, i32x8, i32x4, i64x2, u8x16, u16x8, u32x8, u32x4, u64x4, u64x2],
  (BitOrAssign<Self>, bitor, bitor_assign) => [f32x8, f32x4, f64x4, f64x2, i8x16, i16x8, i32x8, i32x4, i64x2, u8x16, u16x8, u32x8, u32x4, u64x4, u64x2],
  (BitOrAssign<&Self>, bitor, bitor_assign) => [f32x8, f32x4, f64x4, f64x2, i8x16, i16x8, i32x8, i32x4, i64x2, u8x16, u16x8, u32x8, u32x4, u64x4, u64x2],
  (BitXorAssign<Self>, bitxor, bitxor_assign) => [f32x8, f32x4, f64x4, f64x2, i8x16, i16x8, i32x8, i32x4, i64x2, u8x16, u16x8, u32x8, u32x4, u64x4, u64x2],
  (BitXorAssign<&Self>, bitxor, bitxor_assign) => [f32x8, f32x4, f64x4, f64x2, i8x16, i16x8, i32x8, i32x4, i64x2, u8x16, u16x8, u32x8, u32x4, u64x4, u64x2],
}

macro_rules! impl_simple_neg {
  ($($t:ty),+ $(,)?) => {
    $(
      impl Neg for $t {
        type Output = Self;
        #[inline]
        #[must_use]
        fn neg(self) -> Self::Output {
          Self::default() - self
        }
      }
      impl Neg for &'_ $t {
        type Output = $t;
        #[inline]
        #[must_use]
        fn neg(self) -> Self::Output {
          <$t>::default() - *self
        }
      }
    )+
  };
}

impl_simple_neg! {
  f32x8, f32x4, f64x4, f64x2, i8x16, i16x8, i32x8, i32x4, i64x4, i64x2, u8x16, u16x8, u32x8, u32x4, u64x2, u64x4
}

macro_rules! impl_simple_not {
  ($($t:ty),+ $(,)?) => {
    $(
      impl Not for $t {
        type Output = Self;
        #[inline]
        #[must_use]
        fn not(self) -> Self::Output {
          self ^ cast::<u128, $t>(u128::MAX)
        }
      }
      impl Not for &'_ $t {
        type Output = $t;
        #[inline]
        #[must_use]
        fn not(self) -> Self::Output {
          *self ^ cast::<u128, $t>(u128::MAX)
        }
      }
    )+
  };
}

impl_simple_not! {
  f32x4, f64x2, i8x16, i16x8, i32x4, i64x2, u8x16, u16x8, u32x4, u64x2,
}

macro_rules! impl_simple_sum {
  ($($t:ty),+ $(,)?) => {
    $(
      impl<RHS> core::iter::Sum<RHS> for $t where $t: AddAssign<RHS> {
        fn sum<I: Iterator<Item = RHS>>(iter: I) -> Self {
          let mut total = Self::zeroed();
          for val in iter {
            total += val;
          }
          total
        }
      }
    )+
  };
}

impl_simple_sum! {
  f32x4, f64x4, f64x2, i8x16, i16x8, i32x8, i32x4, i64x4, i64x2, u8x16, u16x8, u32x8, u32x4, u64x2, u64x4
}

macro_rules! impl_floating_product {
  ($($t:ty),+ $(,)?) => {
    $(
      impl<RHS> core::iter::Product<RHS> for $t where $t: MulAssign<RHS> {
        fn product<I: Iterator<Item = RHS>>(iter: I) -> Self {
          let mut total = Self::from(1.0);
          for val in iter {
            total *= val;
          }
          total
        }
      }
    )+
  };
}

impl_floating_product! {
  f32x8, f32x4, f64x4, f64x2
}

macro_rules! impl_integer_product {
  ($($t:ty),+ $(,)?) => {
    $(
      impl<RHS> core::iter::Product<RHS> for $t where $t: MulAssign<RHS> {
        fn product<I: Iterator<Item = RHS>>(iter: I) -> Self {
          let mut total = Self::from(1);
          for val in iter {
            total *= val;
          }
          total
        }
      }
    )+
  };
}

impl_integer_product! {
  i16x8, i32x4, i32x8,
}

/// impls `From<a> for b` by just calling `cast`
macro_rules! impl_from_a_for_b_with_cast {
  ($(($arr:ty, $simd:ty)),+  $(,)?) => {
    $(impl From<$arr> for $simd {
      #[inline]
      #[must_use]
      fn from(arr: $arr) -> Self {
        cast(arr)
      }
    }
    impl From<$simd> for $arr {
      #[inline]
      #[must_use]
      fn from(simd: $simd) -> Self {
        cast(simd)
      }
    })+
  };
}

impl_from_a_for_b_with_cast! {
  ([f32;8], f32x8),
  ([f32;4], f32x4), ([f64;4], f64x4), ([f64;2], f64x2),
  ([i8;16], i8x16), ([i16;8], i16x8), ([i32;8], i32x8), ([i32;4], i32x4), ([i64;2], i64x2), ([i64;4], i64x4),
  ([u8;16], u8x16), ([u16;8], u16x8), ([u32;8], u32x8), ([u32;4], u32x4), ([u64;2], u64x2), ([u64;4], u64x4),
}

macro_rules! impl_from_single_value {
  ($(([$elem:ty;$len:expr], $simd:ty)),+  $(,)?) => {
    $(impl From<$elem> for $simd {
      /// Splats the single value given across all lanes.
      #[inline]
      #[must_use]
      fn from(elem: $elem) -> Self {
        cast([elem; $len])
      }
    }
    impl $simd {
      #[inline]
      #[must_use]
      pub fn splat(elem: $elem) -> $simd {
        cast([elem; $len])
      }
    })+
  };
}

impl_from_single_value! {
  ([f32;8], f32x8),
  ([f32;4], f32x4), ([f64;4], f64x4), ([f64;2], f64x2),
  ([i8;16], i8x16), ([i16;8], i16x8), ([i32;8], i32x8), ([i32;4], i32x4), ([i64;2], i64x2), ([i64;4], i64x4),
  ([u8;16], u8x16), ([u16;8], u16x8), ([u32;8], u32x8), ([u32;4], u32x4), ([u64;2], u64x2), ([u64;4], u64x4),
}

/// formatter => [(arr, simd)+],+
macro_rules! impl_formatter_for {
  ($($trait:ident => [$(($arr:ty, $simd:ty)),+]),+ $(,)?) => {
    $( // do per trait
      $( // do per simd type
        impl $trait for $simd {
          fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
            let a: $arr = cast(*self);
            write!(f, "(")?;
            for (x, a_ref) in a.iter().enumerate() {
              if x > 0 {
                write!(f, ", ")?;
              }
              $trait::fmt(a_ref, f)?;
            }
            write!(f, ")")
          }
        }
      )+
    )+
  }
}

impl_formatter_for! {
  Binary => [([u32;8], f32x8), ([u32;4], f32x4), ([u64;4], f64x4), ([u64;2], f64x2),
  ([i8;16], i8x16), ([i16;8], i16x8), ([i32;8], i32x8), ([i32;4], i32x4), ([i64;2], i64x2),([i64;4], i64x4),
  ([u8;16], u8x16), ([u16;8], u16x8), ([u32;8], u32x8), ([u32;4], u32x4), ([u64;2], u64x2),([u64;4], u64x4)],
  Debug => [([f32;8], f32x8), ([f32;4], f32x4), ([f64;4], f64x4), ([u64;2], f64x2),
  ([i8;16], i8x16), ([i16;8], i16x8), ([i32;8], i32x8), ([i32;4], i32x4), ([i64;2], i64x2),([i64;4], i64x4),
  ([u8;16], u8x16), ([u16;8], u16x8), ([u32;8], u32x8), ([u32;4], u32x4), ([u64;2], u64x2),([u64;4], u64x4)],
  Display => [([f32;8], f32x8), ([f32;4], f32x4), ([f64;4], f64x4), ([u64;2], f64x2),
  ([i8;16], i8x16), ([i16;8], i16x8), ([i32;8], i32x8), ([i32;4], i32x4), ([i64;2], i64x2),([i64;4], i64x4),
  ([u8;16], u8x16), ([u16;8], u16x8), ([u32;8], u32x8), ([u32;4], u32x4), ([u64;2], u64x2),([u64;4], u64x4)],
  LowerExp => [([f32;8], f32x8), ([f32;4], f32x4), ([u64;4], f64x4), ([u64;2], f64x2),
  ([i8;16], i8x16), ([i16;8], i16x8), ([i32;8], i32x8), ([i32;4], i32x4), ([i64;2], i64x2),([i64;4], i64x4),
  ([u8;16], u8x16), ([u16;8], u16x8), ([u32;8], u32x8), ([u32;4], u32x4), ([u64;2], u64x2),([u64;4], u64x4)],
  LowerHex => [([u32;8], f32x8), ([u32;4], f32x4), ([u64;4], f64x4), ([u64;2], f64x2),
  ([i8;16], i8x16), ([i16;8], i16x8), ([i32;8], i32x8), ([i32;4], i32x4), ([i64;2], i64x2),([i64;4], i64x4),
  ([u8;16], u8x16), ([u16;8], u16x8), ([u32;8], u32x8), ([u32;4], u32x4), ([u64;2], u64x2),([u64;4], u64x4)],
  Octal => [([u32;8], f32x8), ([u32;4], f32x4), ([u64;4], f64x4), ([u64;2], f64x2),
  ([i8;16], i8x16), ([i16;8], i16x8), ([i32;8], i32x8), ([i32;4], i32x4), ([i64;2], i64x2),([i64;4], i64x4),
  ([u8;16], u8x16), ([u16;8], u16x8), ([u32;8], u32x8), ([u32;4], u32x4), ([u64;2], u64x2),([u64;4], u64x4)],
  UpperExp => [([u32;8], f32x8), ([u32;4], f32x4), ([u64;4], f64x4), ([u64;2], f64x2),
  ([i8;16], i8x16), ([i16;8], i16x8), ([i32;8], i32x8), ([i32;4], i32x4), ([i64;2], i64x2),([i64;4], i64x4),
  ([u8;16], u8x16), ([u16;8], u16x8), ([u32;8], u32x8), ([u32;4], u32x4), ([u64;2], u64x2),([u64;4], u64x4)],
  UpperHex => [([u32;8], f32x8), ([u32;4], f32x4), ([u64;4], f64x4), ([u64;2], f64x2),
  ([i8;16], i8x16), ([i16;8], i16x8), ([i32;8], i32x8), ([i32;4], i32x4), ([i64;2], i64x2),([i64;4], i64x4),
  ([u8;16], u8x16), ([u16;8], u16x8), ([u32;8], u32x8), ([u32;4], u32x4), ([u64;2], u64x2),([u64;4], u64x4)],
}

#[allow(unused)]
fn software_sqrt(x: f64) -> f64 {
  use core::num::Wrapping;
  type wu32 = Wrapping<u32>;
  const fn w(u: u32) -> wu32 {
    Wrapping(u)
  }
  let mut z: f64;
  let sign: wu32 = w(0x80000000);
  let mut ix0: i32;
  let mut s0: i32;
  let mut q: i32;
  let mut m: i32;
  let mut t: i32;
  let mut i: i32;
  let mut r: wu32;
  let mut t1: wu32;
  let mut s1: wu32;
  let mut ix1: wu32;
  let mut q1: wu32;
  // extract data
  {
    let [low, high]: [u32; 2] = cast(x);
    ix0 = high as i32;
    ix1 = w(low);
  }
  // inf and nan
  {
    if ix0 & 0x7ff00000 == 0x7ff00000 {
      return x * x + x;
    }
  }
  // handle zero
  {
    if ix0 <= 0 {
      if ((ix0 & (!sign).0 as i32) | (ix1.0 as i32)) == 0 {
        return x;
      } else if ix0 < 0 {
        return (x - x) / (x - x);
      }
    }
  }
  // normalize
  {
    m = ix0 >> 20;
    if m == 0 {
      // subnormal
      while ix0 == 0 {
        m -= 21;
        ix0 |= (ix1 >> 11).0 as i32;
        ix1 <<= 21;
      }
      i = 0;
      while ix0 & 0x00100000 == 0 {
        ix0 <<= 1;
        i += 1;
      }
      m -= i - 1;
      ix0 |= (ix1.0 >> (31 - i)) as i32;
      ix1 <<= i as usize;
    }
    // un-bias exponent
    m -= 1023;
    ix0 = (ix0 & 0x000fffff) | 0x00100000;
    if (m & 1) != 0 {
      // odd m, double the input to make it even
      ix0 += ix0 + ((ix1 & sign) >> 31).0 as i32;
      ix1 += ix1;
    }
    m >>= 1;
  }
  // generate sqrt bit by bit
  {
    ix0 += ix0 + ((ix1 & sign) >> 31).0 as i32;
    ix1 += ix1;
    // q and q1 store the sqrt(x);
    q = 0;
    q1 = w(0);
    s0 = 0;
    s1 = w(0);
    // our bit that moves from right to left
    r = w(0x00200000);
    while r != w(0) {
      t = s0 + (r.0 as i32);
      if t <= ix0 {
        s0 = t + (r.0 as i32);
        ix0 -= t;
        q += (r.0 as i32);
      }
      ix0 += ix0 + ((ix1 & sign) >> 31).0 as i32;
      ix1 += ix1;
      r >>= 1;
    }
    r = sign;
    while r != w(0) {
      t1 = s1 + r;
      t = s0;
      if (t < ix0) || ((t == ix0) && (t1 <= ix1)) {
        s1 = t1 + r;
        if t1 & sign == sign && (s1 & sign) == w(0) {
          s0 += 1;
        }
        ix0 -= t;
        if ix1 < t1 {
          ix0 -= 1;
        }
        ix1 -= t1;
        q1 += r;
      }
      ix0 += ix0 + ((ix1 & sign) >> 31).0 as i32;
      ix1 += ix1;
      r >>= 1;
    }
  }
  // use floating add to find out rounding direction
  {
    if ix0 | (ix1.0 as i32) != 0 {
      z = 1.0 - 1.0e-300;
      if z >= 1.0 {
        z = 1.0 + 1.0e-300;
        if q1 == w(0xffffffff) {
          q1 = w(0);
          q += 1;
        } else if z > 1.0 {
          if q1 == w(0xfffffffe) {
            q += 1;
          }
          q1 += w(2);
        } else {
          q1 += q1 & w(1);
        }
      }
    }
  }
  // finish up
  ix0 = (q >> 1) + 0x3fe00000;
  ix1 = q1 >> 1;
  if q & 1 == 1 {
    ix1 |= sign;
  }
  ix0 += m << 20;

  cast::<[u32; 2], f64>([ix1.0, ix0 as u32])
}

#[test]
fn test_software_sqrt() {
  assert!(software_sqrt(f64::NAN).is_nan());
  assert_eq!(software_sqrt(f64::INFINITY), f64::INFINITY);
  assert_eq!(software_sqrt(0.0), 0.0);
  assert_eq!(software_sqrt(-0.0), -0.0);
  assert!(software_sqrt(-1.0).is_nan());
  assert!(software_sqrt(f64::NEG_INFINITY).is_nan());
  assert_eq!(software_sqrt(4.0), 2.0);
  assert_eq!(software_sqrt(9.0), 3.0);
  assert_eq!(software_sqrt(16.0), 4.0);
  assert_eq!(software_sqrt(25.0), 5.0);
  assert_eq!(software_sqrt(5000.0 * 5000.0), 5000.0);
}
