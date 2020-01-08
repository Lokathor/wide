#![warn(missing_docs)]
#![cfg_attr(not(feature = "extern_crate_std"), no_std)]
#![cfg_attr(feature = "toolchain_nightly", feature(stdsimd))]
#![cfg_attr(feature = "toolchain_nightly", feature(core_intrinsics))]
#![allow(clippy::replace_consts)]
#![allow(clippy::inline_always)]
#![allow(clippy::expl_impl_clone_on_copy)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::use_self)]
#![warn(clippy::must_use_candidate)]

//! A crate to help you go wide.
//!
//! Specifically, this crate has data types for blocks of primitives packed
//! together and used as a single unit. This works very well with SIMD/vector
//! hardware of various targets. Both in terms of explicit SIMD usage and also
//! in terms of allowing LLVM's auto-vectorizer to do its job.
//!
//! All SIMD usage is on a _best effort_ basis. Results will vary based on
//! target, optimization level, method, and if you're using a Nightly compiler
//! or not. Otherwise you get a "fallback" implementation, which will just do
//! the normal computation on each lane individually.
//!
//! * **Note:** The crate will auto-detect if you're using Nightly and take
//!   advantage of it. You don't do anything on your part. Activate the
//!   `always_use_stable` feature if you'd like to suppress this effect such as
//!   for testing purposes.
//!
//! ### What About `packed_simd`?
//!
//! Compared to the
//! [packed_simd](https://github.com/rust-lang-nursery/packed_simd) RFC efforts,
//! this crate is less concerned with complete coverage of all possible
//! intrinsics and being totally generic across all widths. Instead, I focus on
//! having a very simple, easy to understand setup that avoids generics and
//! tries to just be plain and obvious at all times. The goal is that using a
//! wide type should be as close as possible to using the scalar version of the
//! same type. Some function designed for `f32` inputs and outputs should "just
//! work" when you change it to `f32x4` inputs and outputs.
//!
//! Also, `packed_simd` is Nightly-only, whereas this crate works on Stable.
//! Even on Stable this crate will give you _reasonable_ levels of SIMD just
//! from LLVM's auto-vectorizer being pretty good at its job when you give it
//! code that it recognizes.
//!
//! When `packed_simd` eventually makes it into Stable it _might_ make this
//! crate obsolete. However, in September of 2019 I asked the `packed_simd`
//! folks if there was any kind of ETA, 6 months, 12 months, or more, and they
//! just said "no ETA". So I'm not gonna wait around for `packed_simd`.

pub(crate) use bytemuck::{cast, cast_mut, cast_ref, Pod, Zeroable};
pub(crate) use core::{convert::*, fmt::*, ops::*};

/// Does all our conditional compilation selection.
macro_rules! magic {
  (
    $(if #[cfg($($test:meta),*)] {
      $($if_tokens:tt)*
    })else* else {
      $($else_tokens:tt)*
    }
  ) => {
    magic!{
      @__forests [ ] ;
      $( [ {$($test),*} {$($if_tokens)*} ], )*
      [ { } {$($else_tokens)*} ],
    }
  };

  (
    if #[cfg($($if_meta:meta),*)] {
      $($if_tokens:tt)*
    } $(else if #[cfg($($else_meta:meta),*)] {
      $($else_tokens:tt)*
    })*
  ) => {
    magic!{
      @__forests [ ] ;
      [ {$($if_meta),*} {$($if_tokens)*} ],
      $( [ {$($else_meta),*} {$($else_tokens)*} ], )*
    }
  };

  (
    @__forests [ $($not:meta,)* ] ;
  ) => {
    /* halt expansion */
  };

  (
    @__forests [ $($not:meta,)* ] ;
    [ { $($m:meta),* } { $($tokens:tt)* } ],
    $($rest:tt)*
  ) => {
    #[cfg(all( $($m,)* not(any($($not),*)) ))]
    magic!{ @__identity $($tokens)* }

    magic!{
      @__forests [ $($not,)* $($m,)* ] ;
      $($rest)*
    }
  };

  (
    @__identity $($tokens:tt)*
  ) => {
    $($tokens)*
  };
}

pub mod arch;

magic! {
  if #[cfg(all(target_arch="x86", target_feature="sse"))] {
    pub(crate) use arch::x86::{m128, m128i};
  } else if #[cfg(all(target_arch="x86_64", target_feature="sse"))] {
    pub(crate) use arch::x86_64::{m128, m128i};
  }
  // TODO: arm, aarch64, wasm32, maybe more?
}

mod m_f32x4;
pub use m_f32x4::*;

mod m_i32x4;
pub use m_i32x4::*;

/// A `sqrt` for just one `f32`.
/// 
/// Tries its best to be `no_std`
#[inline(always)]
#[must_use]
pub fn sqrt_f32(x: f32) -> f32 {
  magic! {
    if #[cfg(target_feature = "sse")] {
      m128::set0(x).sqrt0().extract0()
    } else if #[cfg(feature = "toolchain_nightly")] {
      unsafe { core::intrinsics::sqrtf32(x) }
    } else {
      f32::sqrt(x)
    }
  }
}

/// A `sin` for just one `f32`.
#[inline(always)]
#[must_use]
pub fn sin_f32(x: f32) -> f32 {
  magic! {
    if #[cfg(target_feature = "sse")] {
      f32x4 { sse: m128::set0(x) }.sin().sse.extract0()
    } else {
      f32x4 { arr: [x, 0.0, 0.0, 0.0] }.sin().arr[0]
    }
  }
}

/// A `cos` for just one `f32`.
#[inline(always)]
#[must_use]
pub fn cos_f32(x: f32) -> f32 {
  magic! {
    if #[cfg(target_feature = "sse")] {
      f32x4 { sse: m128::set0(x) }.cos().sse.extract0()
    } else {
      f32x4 { arr: [x, 0.0, 0.0, 0.0] }.cos().arr[0]
    }
  }
}

/// A `tan` for just one `f32`.
#[inline(always)]
#[must_use]
pub fn tan_f32(x: f32) -> f32 {
  magic! {
    if #[cfg(target_feature = "sse")] {
      f32x4 { sse: m128::set0(x) }.tan().sse.extract0()
    } else {
      f32x4 { arr: [x, 0.0, 0.0, 0.0] }.tan().arr[0]
    }
  }
}
