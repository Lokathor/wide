#![no_std]
#![allow(non_camel_case_types)]

use core::ops::*;

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

mod f32x4_;
pub use f32x4_::*;

mod f64x2_;
pub use f64x2_::*;

mod i8x16_;
pub use i8x16_::*;

mod i16x8_;
pub use i16x8_::*;

mod i32x4_;
pub use i32x4_::*;

mod i64x2_;
pub use i64x2_::*;

mod u8x16_;
pub use u8x16_::*;

mod u16x8_;
pub use u16x8_::*;

mod u32x4_;
pub use u32x4_::*;

mod u64x2_;
pub use u64x2_::*;

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
  (Add, add) => [f32x4, f64x2, i8x16, i16x8, i32x4, i64x2, u8x16, u16x8, u32x4, u64x2],
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
  (AddAssign<Self>, add, add_assign) => [f32x4, f64x2, i8x16, i16x8, i32x4, i64x2, u8x16, u16x8, u32x4, u64x2],
  (AddAssign<&Self>, add, add_assign) => [f32x4, f64x2, i8x16, i16x8, i32x4, i64x2, u8x16, u16x8, u32x4, u64x2],
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
    })+
  };
}

impl_from_a_for_b_with_cast! {
  ([f32;4], f32x4), ([f64;2], f64x2),
  ([i8;16], i8x16), ([i16;8], i16x8), ([i32;4], i32x4), ([i64;2], i64x2),
  ([u8;16], u8x16), ([u16;8], u16x8), ([u32;4], u32x4), ([u64;2], u64x2),
}
