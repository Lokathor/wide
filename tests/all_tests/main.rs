#![allow(clippy::approx_constant)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::assertions_on_constants)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::nonminimal_bool)]
#![allow(unused_imports)]
#![allow(clippy::precedence)]
#![allow(clippy::eq_op)]
#![allow(clippy::identity_op)]
#![allow(clippy::uninlined_format_args)]
#![allow(clippy::redundant_closure)]
#![allow(clippy::needless_return)]
#![allow(clippy::crate_in_macro_def)]

use core::fmt;
use std::{num::Wrapping, ops::ShlAssign};

use wide::AlignTo;

mod t_simd;
mod t_simd_float;
mod t_simd_integer;
mod t_simd_signed;
mod t_usefulness;
mod utils;
