#![warn(missing_docs)]
#![cfg_attr(feature = "toolchain_nightly", feature(stdsimd))]

//! A crate to help you go wide.
//!
//! Specifically, this crate has data types for blocks of primitives packed
//! together and used as a single unit. This works very well with SIMD/vector
//! hardware of various targets. Both in terms of explicit SIMD usage and also
//! in terms of allowing LLVM's auto-vectorizer to do its job.
//!
//! All SIMD support is on a _best effort_ basis. Results will vary based on
//! target, optimization level, method, and if you're using a Nightly compiler
//! or not.
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
//! intrinsics and more concerned with having a good API that's easy to
//! understand.
//!
//! Also, `packed_simd` is Nightly-only, whereas this crate works on Stable.
//! Even on Stable you'll get _reasonable_ levels of SIMD on most platforms just
//! from LLVM's auto-vectorizer being pretty good when you're doing the sort of
//! code it recognizes.
//!
//! If `packed_simd` ever completes it _might_ make this crate obsolete.
//! However, in September of 2019 I asked the `packed_simd` folks when they
//! might complete their task and get it all into Stable Rust, and they just
//! said "no ETA". I'm not gonna hold my breath.

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
  // match if/else chains with a final `else`, you must have at least 1 if.
  ($(if #[cfg($($meta:meta),*)]
      $if_block:block)else+
    else
      $else_block:block) => {
    $crate::cfg_block! {
      @__internal
      () ;
      $( ( ($($meta),*) ($if_block) ), )+
      ( () ($else_block) ),
    }
  };

  // match if/else chains lacking a final `else`
  (if #[cfg($($if_meta:meta),*)]
      $if_block:block
    $(else if #[cfg($($else_meta:meta),*)]
      $else_block:block)*) => {
    $crate::cfg_block! {
      @__internal
      () ;
      ( ($($if_meta),*) ($if_block) ),
      $( ( ($($else_meta),*) ($else_block) ), )*
      ( () () ),
    }
  };

  // Internal and recursive macro to emit all the items
  //
  // Collects all the negated cfgs in a list at the beginning and after the
  // semicolon is all the remaining items
  (@__internal ($($not:meta,)*) ; ) => {};

  (@__internal ($($not:meta,)*) ; ( ($($m:meta),*) ($bl:block) ), $($rest:tt)*) => {
    // Emit all items within one block, applying an appropriate #[cfg]. The
    // #[cfg] will require all `$m` matchers specified and must also negate
    // all previous matchers.
    $crate::cfg_block! { @__apply cfg(all($($m,)* not(any($($not),*)))), $bl }

    // Recurse to emit all other items in `$rest`, and when we do so add all
    // our `$m` matchers to the list of `$not` matchers as future emissions
    // will have to negate everything we just matched as well.
    $crate::cfg_block! { @__internal ($($not,)* $($m,)*) ; $($rest)* }
  };

  // Internal macro to Apply a cfg attribute to a list of items
  (@__apply $m:meta, $bl:block) => {
    #[$m] $bl
  };
}
