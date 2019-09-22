#![warn(missing_docs)]
#![cfg_attr(feature = "toolchain_nightly", feature(stdsimd))]

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

/// Works like
/// [`cfg_if!`](https://docs.rs/cfg-if/0.1.9/cfg_if/macro.cfg_if.html), but for
/// `:block` instead of `:item`.
///
/// This is inspired by the `cfg_if!` macro from the
/// [cfg-if](https://docs.rs/cfg-if) crate. While `cfg_if!` works to select
/// alternate "item" declarations, `cfg_block!` selects one of several "block"
/// declarations. In other words, instead of using this at the module level, you
/// use this macro inside of a function to select alternate function
/// implementation without any risk of the outside world seeing a different
/// function signature.
#[macro_export]
macro_rules! cfg_block {
  // Handle when there's only `if` and `else if`, no lone `else`. In this case,
  // the macro sees the whole thing as being a list of entries of the form `if
  // cfg block`, and using `else` as a "separator token" between entries. Like a
  // comma separated list, but it's an `else` separated list instead.
  ($(if #[cfg($($if_else_meta:meta),*)] $if_else_block:block)else+) => {
    $crate::cfg_block!{
      @__internal ();
      $( ([$($if_else_meta),*] [$if_else_block]), )+
    }
  };

  // Handle when there's a final `else` WITHOUT an `if` on it at the end of the
  // "list" of entries. It's basically just like the above case, except that we
  // throw the final else block onto the end of the token tree that we send to
  // the internal branch. We just give it no conditions and it all works out.
  ($(if #[cfg($($if_else_meta:meta),*)] $if_else_block:block)else+
    else $else_block:block) => {
    $crate::cfg_block!{
      @__internal ();
      $( ([$($if_else_meta),*] [$if_else_block]), )+
      ( [] [$else_block] ),
    }
  };

  // Here we have some metas that we've handled so far (starts empty), as well
  // as the current pairing of metas and a block that goes with them. We emit
  // the block configured for the negation of all previous metas combined with
  // its own metas. Then we move the current metas into the negation pile and
  // recurse to the next set of entries in the token tree.
  (@__internal ($($not:meta,)*) ; ( [$($m:meta),*] [$bl:block] ), $($rest:tt)*) => {
    #[cfg(all($($m,)* not(any($($not),*))))] $bl

    $crate::cfg_block!{ @__internal ($($not,)* $($m,)*) ; $($rest)* }
  };

  // Here we've run out of token tree to process, so we just stop.
  (@__internal ($($not:meta,)*) ; ) => {};
}

pub fn lib_function_if() {
  cfg_block!{if #[cfg(windows)] {
    println!("foo");
  }}
}

pub fn lib_function_if_else() {
  cfg_block!{if #[cfg(windows)] {
    println!("foo");
  } else {
    println!("else");
  }}
}

pub fn lib_function_if_elseif() {
  cfg_block!{if #[cfg(windows)] {
    println!("foo");
  } else if #[cfg(unix)] {
    println!("elseif");
  }}
}

pub fn lib_function_if_elseif_else() {
  cfg_block!{if #[cfg(windows)] {
    println!("foo");
  } else if #[cfg(unix)] {
    println!("elseif");
  } else {
    println!("else");
  }}
}
