[![License:Zlib](https://img.shields.io/badge/License-Zlib-brightgreen.svg)](https://opensource.org/licenses/Zlib)
![Minimum Rust Version](https://img.shields.io/badge/Min%20Rust-1.89-green.svg)
[![crates.io](https://img.shields.io/crates/v/wide.svg)](https://crates.io/crates/wide)
[![docs.rs](https://docs.rs/wide/badge.svg)](https://docs.rs/wide/)

# wide

A crate to help you go wide.

Specifically, this has portable "wide" data types that do their best to be SIMD when possible.

On `x86`, `x86_64`, `wasm32` and `aarch64 neon` this is done with explicit
intrinsic usage (via [safe_arch](https://docs.rs/safe_arch)), and on other
architectures this is done by carefully writing functions so that LLVM hopefully
does the right thing. When Rust stabilizes more explicit intrinsics then they
can go into `safe_arch` and then they can get used here.

## Rust Version Policy

* The `rust-version` entry of the crate's `Cargo.toml` will be kept accurate to
  the required Rust compiler version.
* The `rust-version` entry may increase in *any* new release (major, minor, or patch).
* If your build uses Resolver 3 (or later) this will be fine. If you're using a
  Resolver earlier than 3 then **you are responsible** for pinning a maximum
  crate version when you're using an old Rust version.
