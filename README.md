[![License:0BSD](https://img.shields.io/badge/License-0BSD-brightgreen.svg)](https://opensource.org/licenses/FPL-1.0.0)
![Minimum Rust Version](https://img.shields.io/badge/Min%20Rust-1.36-green.svg)
[![travis.ci](https://travis-ci.org/Lokathor/lokacore.svg?branch=master)](https://travis-ci.org/Lokathor/lokacore)
[![AppVeyor](https://ci.appveyor.com/api/projects/status/td70y0cavp51giai/branch/master?svg=true)](https://ci.appveyor.com/project/Lokathor/lokacore/branch/master)
[![crates.io](https://img.shields.io/crates/v/lokacore.svg)](https://crates.io/crates/lokacore)
[![docs.rs](https://docs.rs/lokacore/badge.svg)](https://docs.rs/lokacore/)

# lokacore

Lokathor's core-only odds and ends

Current features of note:

* Safe casting between plain data types. Supports `T`, `&T`, `&mut T`, `&[T]`, and
  `&mut [T]`.
* Safe versions of all stable `rdrand`, `sse`, `sse2`, `sse3`, `ssse3`,
  `sse4.1`, and `sse4.2` intrinsics. At least, all the ones that _can_ be safe
  wrapped.
* Some standard floating point utilities until more floating point math becomes
  part of `core`.

CI coverage:
* Tested on: `x86`, `x86_64`, `wasm`
* Built on: `armv7`, `aarch64`, `thumbv7neon`
