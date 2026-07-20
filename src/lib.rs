#![no_std]
#![allow(non_camel_case_types)]
#![warn(clippy::doc_markdown)]
#![warn(clippy::missing_inline_in_public_items)]
#![allow(clippy::eq_op)]
#![allow(clippy::excessive_precision)]
#![allow(clippy::let_and_return)]
#![allow(clippy::unusual_byte_groupings)]
#![allow(clippy::misrefactored_assign_op)]
#![allow(clippy::approx_constant)]
#![forbid(missing_docs)]

//! A crate to help you go wide.
//!
//! This crate provides SIMD-compatible data types.
//!
//! When possible, explicit SIMD is used with all the math operations here. As a
//! fallback, the fact that all the lengths of a fixed length array are doing
//! the same thing will often make LLVM notice that it should use SIMD
//! instructions to complete the task. In the worst case, the code just becomes
//! totally scalar (though the math is still correct, at least).
//!
//! # Masks
//!
//! SIMD vector masks are per-element booleans, and are represented by SIMD
//! vectors where each element's value in bits is either all zeros (`false`) or
//! all ones (`true`).
//!
//! SIMD versions of functions that regularly return `bool` return masks. For
//! example, [`f32::is_sign_positive`] returns `bool`, and
//! [`f32x4::is_sign_positive`] returns an [`f32x4`] that represents a mask.
//! The [`select`] method can be used to perform a per-element `if` statement
//! over a mask. For example, for this simple scalar code:
//!
//! ```
//! let x = 1.0_f32;
//!
//! let result = if x.is_sign_positive() {
//!     5.0
//! } else {
//!     3.0
//! };
//!
//! assert_eq!(result, 5.0);
//! ```
//!
//! This is the SIMD version:
//!
//! ```
//! # use wide::f32x4;
//! #
//! let x = f32x4::new([1.0, -1.0, -1.0, 1.0]);
//!
//! let result = x.is_sign_positive().select(
//!     f32x4::splat(5.0),
//!     f32x4::splat(3.0),
//! );
//!
//! assert_eq!(result, f32x4::new([5.0, 3.0, 3.0, 5.0]));
//! ```
//!
//! # NaN bit patterns
//!
//! Operations on SIMD vectors of floats do not make any guarantees about the
//! specific bit-patterns of output NaN values (meaning the sign bit,
//! quiet/signaling bit, and payload). This is unlike standard library
//! operations on float primitives, which do define rules for what NaN bit
//! patterns are returned.
//!
//! The reason for this is that enforcing guarantees would add substantial
//! overhead to operations, and is generally not worth it.
//!
//! # Wrapping semantics
//!
//! SIMD vectors of integers treat operators as wrapping, as if [`Wrapping<T>`]
//! was used. Thus, SIMD vectors do not implement `wrapping_*` functions,
//! because that is the default behavior. This means there are no overflow
//! checks, even in debug builds.
//!
//! The reason for this is that for most applications where SIMD is appropriate,
//! it is "not a bug" to wrap, and even debug builds are unlikely to tolerate
//! the loss of performance.
//!
//! This "no panicking" approach extends to more than just integer overflows. It
//! is also true for things like [`f32x4::clamp`]. Even though [`f32::clamp`]
//! panics for invalid inputs, the SIMD version does not, because it may be used
//! with per-element branching, where invalid elements get discarded later by
//! [`select`].
//!
//! # Casting
//!
//! The SIMD types implement the [`bytemuck::Pod`] trait, which means that it
//! is possible to do bitwise casts between SIMD types of the same size with
//! the [`bytemuck::cast()`] function and others. `bytemuck` is re-exported by
//! this crate for convenience.
//!
//! This typically does not have much, if any, runtime overhead in optimized
//! builds.
//!
//! # Feature flags
//!
//! * `std`: This causes the feature to link to `std`.
//!   * Currently this just improves the performance of `sqrt` when an explicit
//!     SIMD `sqrt` isn't available.
//!
//! [`select`]: f32x4::select
//! [`Wrapping<T>`]: core::num::Wrapping

// Note(Lokathor): Due to standard library magic, the std-only methods for f32
// and f64 will automatically be available simply by declaring this.
#[cfg(feature = "std")]
extern crate std;

// TODO
// Add/Sub/Mul/Div with constant
// Shuffle left/right/by index

use core::ops::*;

#[allow(unused_imports)]
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use safe_arch::*;

use bytemuck::*;

// Re-export so that users don't need to add a bytemuck dependency of their own
pub use bytemuck;

#[macro_use]
mod simd;
#[macro_use]
mod simd_float;
#[macro_use]
mod simd_int;
#[macro_use]
mod simd_uint;

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

macro_rules! polynomial_3 {
  ($x:expr, $c0:expr, $c1:expr, $c2:expr, $c3:expr $(,)?) => {{
    let x = $x;
    let x2 = x * x;
    $c3.mul_add(x, $c2).mul_add(x2, $c1.mul_add(x, $c0))
  }};
}

macro_rules! polynomial_4 {
  ($x:expr, $c0:expr, $c1:expr, $c2:expr ,$c3:expr, $c4:expr $(,)?) => {{
    let x = $x;
    let x2 = x * x;
    let x4 = x2 * x2;
    $c3.mul_add(x, $c2).mul_add(x2, $c1.mul_add(x, $c0)) + $c4 * x4
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
  ($x:expr, $c0:expr, $c1:expr, $c2:expr, $c3:expr, $c4:expr $(,)?) => {{
    let x = $x;
    let x2 = x * x;
    let x4 = x2 * x2;
    x2.mul_add(x.mul_add($c3, $c2), (x4.mul_add($c4 + x, x.mul_add($c1, $c0))))
  }};
}

macro_rules! polynomial_6 {
  ($x:expr, $c0:expr, $c1:expr, $c2:expr, $c3:expr, $c4:expr, $c5:expr ,$c6:expr $(,)?) => {{
    let x = $x;
    let x2 = x * x;
    let x4 = x2 * x2;
    x4.mul_add(
      x2.mul_add($c6, x.mul_add($c5, $c4)),
      x2.mul_add(x.mul_add($c3, $c2), x.mul_add($c1, $c0)),
    )
  }};
}

macro_rules! polynomial_6n {
  ($x:expr, $c0:expr, $c1:expr, $c2:expr, $c3:expr, $c4:expr, $c5:expr $(,)?) => {{
    let x = $x;
    let x2 = x * x;
    let x4 = x2 * x2;
    x4.mul_add(
      x.mul_add($c5, x2 + $c4),
      x2.mul_add(x.mul_add($c3, $c2), x.mul_add($c1, $c0)),
    )
  }};
}

macro_rules! polynomial_7 {
  ($x:expr, $c0:expr, $c1:expr, $c2:expr, $c3:expr, $c4:expr, $c5:expr, $c6:expr, $c7:expr $(,)?) => {{
    let x = $x;
    let x2 = x * x;
    let x4 = x2 * x2;
    x4.mul_add(
      x2.mul_add(x.mul_add($c7, $c6), x.mul_add($c5, $c4)),
      x2.mul_add(x.mul_add($c3, $c2), x.mul_add($c1, $c0)),
    )
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
  // calculates polynomial c13*x^13 + c12*x^12 + ... + c1*x + c0
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

mod f32x16_;
pub use f32x16_::*;

mod f32x8_;
pub use f32x8_::*;

mod f32x4_;
pub use f32x4_::*;

mod f64x8_;
pub use f64x8_::*;

mod f64x4_;
pub use f64x4_::*;

mod f64x2_;
pub use f64x2_::*;

mod i8x16_;
pub use i8x16_::*;

mod i16x16_;
pub use i16x16_::*;

mod i16x32_;
pub use i16x32_::*;

mod i8x32_;
pub use i8x32_::*;

mod i16x8_;
pub use i16x8_::*;

mod i32x4_;
pub use i32x4_::*;

mod i32x8_;
pub use i32x8_::*;

mod i32x16_;
pub use i32x16_::*;

mod i64x2_;
pub use i64x2_::*;

mod i64x4_;
pub use i64x4_::*;

mod i64x8_;
pub use i64x8_::*;

mod u8x16_;
pub use u8x16_::*;

mod u8x32_;
pub use u8x32_::*;

mod u16x8_;
pub use u16x8_::*;

mod u16x16_;
pub use u16x16_::*;

mod u16x32_;
pub use u16x32_::*;

mod u32x4_;
pub use u32x4_::*;

mod u32x8_;
pub use u32x8_::*;

mod u32x16_;
pub use u32x16_::*;

mod u64x2_;
pub use u64x2_::*;

mod u64x4_;
pub use u64x4_::*;

mod u64x8_;
pub use u64x8_::*;

#[allow(dead_code)]
fn generic_bit_blend<T>(mask: T, y: T, n: T) -> T
where
  T: Copy + BitXor<Output = T> + BitAnd<Output = T>,
{
  n ^ ((n ^ y) & mask)
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

  pick! {
    if #[cfg(target_endian = "little")]
    {
      let [low, high]: [u32; 2] = cast(x);
      ix0 = high as i32;
      ix1 = w(low);
    }
    else
    {
      let [high, low]: [u32; 2] = cast(x);
      ix0 = high as i32;
      ix1 = w(low);
    }
  }

  // inf and nan
  {
    if x.is_nan() {
      return f64::NAN;
    }
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

  pick! {
    if #[cfg(target_endian = "little")]
    {
      cast::<[u32; 2], f64>([ix1.0, ix0 as u32])
    }
    else
    {
      cast::<[u32; 2], f64>([ix0 as u32, ix1.0])
    }
  }
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

/// A deprecated trait for the [`simd_eq`] function.
///
/// [`simd_eq`]: f32x4::simd_eq
#[deprecated(
  since = "1.5.0",
  note = "use the inherent function `simd_eq` instead"
)]
pub trait CmpEq<Rhs = Self> {
  /// The type returned by [`simd_eq`].
  ///
  /// [`simd_eq`]: Self::simd_eq
  type Output;

  /// Returns a [mask] that checks if each element of `self` is equal to the
  /// corresponding element of `rhs`.
  ///
  /// This is a method of the deprecated [`CmpEq`] trait. Use the inherent
  /// function instead.
  ///
  /// [mask]: crate#masks
  fn simd_eq(self, rhs: Rhs) -> Self::Output;
}

/// A deprecated trait for the [`simd_gt`] function.
///
/// [`simd_gt`]: f32x4::simd_gt
#[deprecated(
  since = "1.5.0",
  note = "use the inherent function `simd_gt` instead"
)]
pub trait CmpGt<Rhs = Self> {
  /// The type returned by [`simd_gt`].
  ///
  /// [`simd_gt`]: Self::simd_gt
  type Output;

  /// Returns a [mask] that checks if each element of `self` is greater than the
  /// corresponding element of `rhs`.
  ///
  /// This is a method of the deprecated [`CmpGt`] trait. Use the inherent
  /// function instead.
  ///
  /// [mask]: crate#masks
  fn simd_gt(self, rhs: Rhs) -> Self::Output;
}

/// A deprecated trait for the [`simd_ge`] function.
///
/// [`simd_ge`]: f32x4::simd_ge
#[deprecated(
  since = "1.5.0",
  note = "use the inherent function `simd_ge` instead"
)]
pub trait CmpGe<Rhs = Self> {
  /// The type returned by [`simd_ge`].
  ///
  /// [`simd_ge`]: Self::simd_ge
  type Output;

  /// Returns a [mask] that checks if each element of `self` is greater than or
  /// equal to the corresponding element of `rhs`.
  ///
  /// This is a method of the deprecated [`CmpGe`] trait. Use the inherent
  /// function instead.
  ///
  /// [mask]: crate#masks
  fn simd_ge(self, rhs: Rhs) -> Self::Output;
}

/// A deprecated trait for the [`simd_ne`] function.
///
/// [`simd_ne`]: f32x4::simd_ne
#[deprecated(
  since = "1.5.0",
  note = "use the inherent function `simd_ne` instead"
)]
pub trait CmpNe<Rhs = Self> {
  /// The type returned by [`simd_ne`].
  ///
  /// [`simd_ne`]: Self::simd_ne
  type Output;

  /// Returns a [mask] that checks if each element of `self` is not equal to the
  /// corresponding element of `rhs`.
  ///
  /// This is a method of the deprecated [`CmpNe`] trait. Use the inherent
  /// function instead.
  ///
  /// [mask]: crate#masks
  fn simd_ne(self, rhs: Rhs) -> Self::Output;
}

/// A deprecated trait for the [`simd_lt`] function.
///
/// [`simd_lt`]: f32x4::simd_lt
#[deprecated(
  since = "1.5.0",
  note = "use the inherent function `simd_lt` instead"
)]
pub trait CmpLt<Rhs = Self> {
  /// The type returned by [`simd_lt`].
  ///
  /// [`simd_lt`]: Self::simd_lt
  type Output;

  /// Returns a [mask] that checks if each element of `self` is less than the
  /// corresponding element of `rhs`.
  ///
  /// This is a method of the deprecated [`CmpLt`] trait. Use the inherent
  /// function instead.
  ///
  /// [mask]: crate#masks
  fn simd_lt(self, rhs: Rhs) -> Self::Output;
}

/// A deprecated trait for the [`simd_le`] function.
///
/// [`simd_le`]: f32x4::simd_le
#[deprecated(
  since = "1.5.0",
  note = "use the inherent function `simd_le` instead"
)]
pub trait CmpLe<Rhs = Self> {
  /// The type returned by [`simd_le`].
  ///
  /// [`simd_le`]: Self::simd_le
  type Output;

  /// Returns a [mask] that checks if each element of `self` is less than or
  /// equal to the corresponding element of `rhs`.
  ///
  /// This is a method of the deprecated [`CmpLe`] trait. Use the inherent
  /// function instead.
  ///
  /// [mask]: crate#masks
  fn simd_le(self, rhs: Rhs) -> Self::Output;
}

/// A trait for SIMD variants of [`align_to`] functions.
///
/// [`align_to`]: https://doc.rust-lang.org/std/primitive.slice.html#method.align_to
pub trait AlignTo
where
  Self: Pod + Default + PartialEq + From<Self::Elem>,
  Self::Elem: Pod + Default + PartialEq,
{
  /// The element type of this SIMD vector.
  type Elem;

  /// A SIMD variant of [`align_to`].
  ///
  /// [`align_to`]: https://doc.rust-lang.org/std/primitive.slice.html#method.align_to
  #[inline]
  fn simd_align_to(
    slice: &[Self::Elem],
  ) -> (&[Self::Elem], &[Self], &[Self::Elem]) {
    pod_align_to(slice)
  }

  /// A SIMD variant of [`align_to_mut`].
  ///
  /// [`align_to_mut`]: https://doc.rust-lang.org/std/primitive.slice.html#method.align_to_mut
  #[inline]
  fn simd_align_to_mut(
    slice: &mut [Self::Elem],
  ) -> (&mut [Self::Elem], &mut [Self], &mut [Self::Elem]) {
    pod_align_to_mut(slice)
  }
}
