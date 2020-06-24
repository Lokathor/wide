[![License:Zlib](https://img.shields.io/badge/License-Zlib-brightgreen.svg)](https://opensource.org/licenses/Zlib)
![Minimum Rust Version](https://img.shields.io/badge/Min%20Rust-1.38-green.svg)
[![travis.ci](https://travis-ci.org/Lokathor/wide.svg?branch=master)](https://travis-ci.org/Lokathor/wide)
[![AppVeyor](https://ci.appveyor.com/api/projects/status/33t3nhj1rplo7t1x/branch/master?svg=true)](https://ci.appveyor.com/project/Lokathor/wide/branch/master)
[![crates.io](https://img.shields.io/crates/v/wide.svg)](https://crates.io/crates/wide)
[![docs.rs](https://docs.rs/wide/badge.svg)](https://docs.rs/wide/)

# wide

A crate to help you go wide.

For more info [see the docs](https://docs.rs/wide).

CI coverage:
* Tested on: `x86`, `x86_64`, `wasm`
* Built on: `armv7`, `aarch64`, `thumbv7neon`

# Current Status

The `wide` crate is currently in a holding pattern while I develop the [safe_arch](https://docs.rs/safe_arch) crate.

`safe_arch` is a project where I'm pulling out the "safe usage of SIMD intrinsics" modules that `wide` currently has as just internal modules, and making that into its own fully formed crate.

Once `safe_arch` is able to support it, I'll update `wide` to utilize the `safe_arch` crate. The `i32x4` type will get a proper overhaul to be completed, and we might even get additional data types added.
