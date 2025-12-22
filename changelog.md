# `wide` Changelog

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
