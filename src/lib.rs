#![no_std]
#![warn(missing_docs)]

//! A crate to help you go wide.
//!
//! Jokes aside, the point of this crate is to offer data types that let you
//! perform data operations in SIMD batches. A single SIMD register might be,
//! for example, not just one `f32` but four `f32` packed together. When you add
//! two registers it performs all four additions just as fast as if you'd added
//! only a single `f32`. This lets your code perform much faster, but it can be
//! tricky to get used to working with your data in batches like that.std
//! * SIMD types have various lane counts, but they generally come in powers of
//!   two. If you have some other number of elements then you can just let the
//!   lane go to waste, but it takes a little time to go into and out of SIMD
//!   form, and if you want to actually store your data already in SIMD form
//!   then you have to waste some amount of space in memory, which can be a bit
//!   of a drag.
//! * Generally, for any operation that would have an `if` in the middle when
//!   performing the non-SIMD version, you have to first perform your test to
//!   determine which "lanes" within the SIMD register are true or false for
//!   that check (called a "mask" value), then you perform _both_ the true and
//!   false paths, and then finally you merge the results at the end according
//!   to your mask. This means that for code where there's a heavy amount of
//!   branching you get less of a benefit from SIMD.

// Note(Lokathor): We don't need to actually import things from here because the
// primary effect of linking in the standard library is that we can fall `f32`
// methods, and they will simply be available on floats as long as the standard
// library was linked.
#[cfg(feature = "std")]
extern crate std;
