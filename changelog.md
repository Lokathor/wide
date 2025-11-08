# `wide` Changelog

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
