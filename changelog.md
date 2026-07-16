# `wide` Changelog

## Unreleased

* Renamed float function `pow_{simd-type-name}` to `powf_simd` and deprecated `powf`.
* Added conversions between `wide` types and native intrinsics SIMD types.
* Added `reduce_mul` for integers.
* Added integer functions `reduce_mul` and `mul_keep_low_high`.

* Added overflowing arithmetic for integers.
* Added float function `round_ties_even`.

* Fixed bugs in the fallback paths of `any`, `all`, `none` and `fast_clamp`.

* Added support for shifting SIMD vectors by values of `isize` and `usize` and
  SIMD vectors of "opposite-signedness".
* Deprecated `blend` and replaced it with `select` and `bitselect`.
* Fixed bug in `powf`.
* Fixed `clamp` and added guarantees for `fast_clamp`.
* Added small optimizations to `saturating_add/sub`.
* Fixed shift operators overflow behavior.
* Fixed `round` which previously behaved like `round_ties_even`.
* Fixed `UpperExp` formatting for floats.
* Renamed integer function `mul_widen` to `widening_mul` and added it for
  remaining types.
* Added integer function `mul_keep_high` for remaining types.
* Added missing `#[must_use]` annotations
* Updated documenattion.

## 1.5.0

* Added several functions and trait implementations that previously were only
  implemented for some types inconsistently.
* Added inherit `simd_*` comparison functions and deprecated the `Cmp*` traits.
* Added integer `Div` and `Rem` implementations.
* Added integer functions `clamp`, `saturating_mul` and `saturating_div`.
* Added signed integer functions `is_positive` and `signum`.
* Added additional float constants.
* Fixed slight bug in `f64xN` functions `round` and `round_int`.
* Fixed `simd_ne` NaN behavior for floats.
* Fixed bug in `u64x8::simd_lt`.
* Fixed bug in `i64x8` and `u64x8` function `to_bitmask`
* Fixed bugs in `u32x16` functions `any`, `all` and `none`.
* Optimized signed integer function `is_negative`.

## 1.4.0

* Added more float functions.
* Implemented `Rem` for float types.
* Renamed float function `sign_bit` to `is_sign_negative` and added
  `is_sign_positive`.
* Corrected [several bugs](https://github.com/Lokathor/wide/pull/261) with
  non-finite inputs for various math functions.

## 1.3.0

* Fixes the behaviour of `f32x16` functions `is_finite` and `round_int`. They
  previously gave incorrect output in some cases.
* Added more `f32x16` functionality.

## 1.2.0

* added reduce operations and dot to `i16x32`

## 1.1.2

* Use native NEON intrinsics for `f32x4::blend` and `f64x2::blend` on aarch64,
  improving performance by using a single `vbslq` instruction instead of the
  generic 3-operation fallback.

## 1.1.1

* Further improvements to the `Neg` impls on non-x86 targets.

## 1.1.0

* add `i64x2::min`, `i64x2::max`.
* add `u64x2::min`, `u64x2::max`.
* add `u64x8::min`.

## 1.0.3

* fix floating point negation edge cases, which also allows optimizations in
  more cases.

## 1.0.2

* fix edge case where `wide` was using the wrong avx512 sub-features to select
  when `safe_arch` functions could be called, causing build errors.

## 1.0.1

* initial stable version

## 0.8.3

* `to_bitmask` implemented for all unsigned int types.

## 0.8.2

* Fixed additional build errors on wasm targets.

## 0.8.1

* Fixed type errors in the fallback implementations of some methods that
  completely prevented compilation of the crate on some targets.

## 0.8.0

* **Breaking:**
  * `move_mask` returns unsigned values now.
  * `move_mask` and many other methods renamed to better align with the standard
    library portable simd types. See [issue
    209](https://github.com/Lokathor/wide/issues/209) for details.
