#![cfg(target_feature="sse")]
#![cfg(target_feature="sse2")]

use super::*;
use core::ops::*;

/// # SSE2 Operations
impl m128 {
  /// This rounds each lane to `i32`.
  #[inline(always)]
  pub fn round_i32(self) -> m128i {
    m128i(unsafe { _mm_cvtps_epi32(self.0) })
  }

  /// This truncates each lane to `i32`.
  #[inline(always)]
  pub fn truncate_i32(self) -> m128i {
    m128i(unsafe { _mm_cvttps_epi32(self.0) })
  }

  /// This "rounds" the lower two lanes to `f64`.
  ///
  /// `f64` has more precision than `f32` so there's no actually rounding going
  /// on here, but I'll just call it rounding so that the naming stays
  /// consistent with other similar methods.
  #[inline(always)]
  pub fn round_f64(self) -> m128d {
    m128d(unsafe { _mm_cvtps_pd(self.0) })
  }

  /// Lane 0 is the low `f64` of `rhs` rounded to `f32`, other lanes are `self`.
  #[inline(always)]
  pub fn f64_round_copy0(self, rhs: m128d) -> Self {
    Self(unsafe { _mm_cvtsd_ss(self.0, rhs.0) })
  }

  /// Cast the bits of this `m128` directly to `m128i` without modification.
  #[inline(always)]
  pub fn cast_m128i(self) -> m128i {
    m128i(unsafe { _mm_castps_si128(self.0) })
  }

  /// `[non-intrinsic]` Lanewise "ceiling" operation (round to positive
  /// infinity).
  ///
  /// This is NOT the "ceil" intrinsic. Instead, it's a "software" version of
  /// the ceiling operation using only `sse2` operations. You should prefer the
  /// `ceil` method when `sse4.1` is available. That actually _does_ perform the
  /// ceiling operation as a single instruction (much faster).
  #[inline]
  pub fn ceil_sse2(self) -> Self {
    // Note(Lokathor): This magical value is the bit pattern for the smallest
    // `f32` that will never have a fractional part (8388608). Any number of
    // this value or more is already a whole number.
    const THRESHOLD: i32 = 0x4b00_0000;

    // clear any sign bits, then check which lanes are under the magic threshold
    // where the simple "truncate and round back" trick works.
    let signless: m128i =
      self.cast_m128i() & m128i::splat_i32(i32::max_value());
    let under_threshold: m128i =
      signless.cmp_lt_i32(m128i::splat_i32(THRESHOLD));

    // truncate, round back, check for changes.
    let truncated: m128i = self.truncate_i32();
    let float_again: Self = truncated.round_f32();
    let is_smaller: Self = float_again.cmp_lt(self);

    // convert "false" to 0.0, "true" to -1.0, then _subtract_ this value from
    // our truncate trip value, which _adds_ either 0.0 or 1.0 as appropriate.
    let diff: Self = is_smaller.cast_m128i().round_f32();
    let totals: Self = float_again - diff;

    // combine the threshold test lanes with the input
    let threshold_numbers: Self = totals & under_threshold.cast_m128();
    let weird_numbers: m128i = under_threshold.andnot(self.cast_m128i());
    threshold_numbers | weird_numbers.cast_m128()
  }

  /// `[non-intrinsic]` Lanewise "floor" operation (round to negative infinity).
  ///
  /// This is NOT the "floor" intrinsic. Instead, it's a "software" version of
  /// the floor operation using only `sse2` operations. You should prefer the
  /// `floor` method when `sse4.1` is available. That actually _does_ perform
  /// the floor operation as a single instruction (much faster).
  #[inline]
  pub fn floor_sse2(self) -> Self {
    // Note(Lokathor): This magical value is the bit pattern for the smallest
    // `f32` that will never have a fractional part (8388608). Any number of
    // this value or more is already a whole number.
    const THRESHOLD: i32 = 0x4b00_0000;

    let signless: m128i =
      self.cast_m128i() & m128i::splat_i32(i32::max_value());
    let under_threshold: m128i =
      signless.cmp_lt_i32(m128i::splat_i32(THRESHOLD));

    let truncated: m128i = self.truncate_i32();
    let float_again: Self = truncated.round_f32();
    let is_smaller: Self = float_again.cmp_gt(self);

    let diff: Self = is_smaller.cast_m128i().round_f32();
    let totals: Self = float_again + diff;

    let threshold_numbers: Self = totals & under_threshold.cast_m128();
    let weird_numbers: m128i = under_threshold.andnot(self.cast_m128i());
    threshold_numbers | weird_numbers.cast_m128()
  }
}

/// A 128-bit SIMD value. Integral data, lanes determined by each op.
///
/// * This documentation numbers the lanes based on the index you'd need to use
///   to access that lane if the value were cast to an array.
/// * This is also the way that the type is printed out using
///   [`Debug`](core::fmt::Debug), [`Display`](core::fmt::Display),
///   [`LowerExp`](core::fmt::LowerExp), and [`UpperExp`](core::fmt::UpperExp).
/// * This is not necessarily the ordering you'll see if you look an `xmm`
///   register in a debugger! Basically because of how little-endian works.
/// * Most operations work per-lane, "lanewise".
/// * Some operations work using lane 0 only. When appropriate, these have the
///   same name as the lanewise version but with a `0` on the end. Eg: `cmp_eq`
///   and `cmp_eq0`. The other lanes are simply copied forward from `self`.
/// * Comparisons give "bool-ish" output, where all bits 1 in a lane is true,
///   and all bits 0 in a lane is false. Unfortunately, all bits 1 with an `f32`
///   is one of the `NaN` values, and `NaN != NaN`, so it can be a little tricky
///   to work with until you're used to it.
#[derive(Clone, Copy)]
#[allow(bad_style)]
#[repr(transparent)]
pub struct m128i(pub __m128i);

impl Default for m128i {
  #[inline(always)]
  fn default() -> Self {
    Self::zeroed()
  }
}

#[cfg(feature = "serde")]
impl serde::Serialize for m128i {
  #[inline(always)]
  fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
  where
    S: serde::Serializer,
  {
    let bits: i128 = cast(*self);
    bits.serialize(serializer)
  }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for m128i {
  #[inline(always)]
  fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
  where
    D: serde::Deserializer<'de>,
  {
    let bits: i128 = i128::deserialize(deserializer)?;
    Ok(cast(bits))
  }
}

unsafe impl Zeroable for m128i {}
unsafe impl Pod for m128i {}

impl core::fmt::Debug for m128i {
  /// Debug formats in offset order.
  ///
  /// * Use `width` to specify the lane count you want (default 1).
  /// * Use `alternate` format specifier to give `uX` instead of `iX` output.
  ///
  /// Eg, for 4 lanes of `u32`:
  /// ```txt
  /// format!("{:#4?}", m)
  /// ```
  fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
    match f.width() {
      Some(2) => {
        if f.alternate() {
          let a: [u64; 2] = cast(self.0);
          write!(f, "m128i({:?}, {:?})", a[0], a[1])
        } else {
          let a: [i64; 2] = cast(self.0);
          write!(f, "m128i({:?}, {:?})", a[0], a[1])
        }
      }
      Some(4) => {
        if f.alternate() {
          let a: [u32; 4] = cast(self.0);
          write!(f, "m128i({:?}, {:?}, {:?}, {:?})", a[0], a[1], a[2], a[3])
        } else {
          let a: [i32; 4] = cast(self.0);
          write!(f, "m128i({:?}, {:?}, {:?}, {:?})", a[0], a[1], a[2], a[3])
        }
      }
      Some(8) => {
        if f.alternate() {
          let a: [u16; 8] = cast(self.0);
          write!(
            f,
            "m128i({:?}, {:?}, {:?}, {:?}, {:?}, {:?}, {:?}, {:?})",
            a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]
          )
        } else {
          let a: [i16; 8] = cast(self.0);
          write!(
            f,
            "m128i({:?}, {:?}, {:?}, {:?}, {:?}, {:?}, {:?}, {:?})",
            a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]
          )
        }
      }
      Some(16) => {
        if f.alternate() {
          let a: [u8; 16] = cast(self.0);
          write!(f, "m128i({:?}, {:?}, {:?}, {:?}, {:?}, {:?}, {:?}, {:?}, {:?}, {:?}, {:?}, {:?}, {:?}, {:?}, {:?}, {:?})", a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15])
        } else {
          let a: [i8; 16] = cast(self.0);
          write!(f, "m128i({:?}, {:?}, {:?}, {:?}, {:?}, {:?}, {:?}, {:?}, {:?}, {:?}, {:?}, {:?}, {:?}, {:?}, {:?}, {:?})", a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15])
        }
      }
      _ => {
        if f.alternate() {
          let a: u128 = cast(self.0);
          write!(f, "m128i({:?})", a)
        } else {
          let a: i128 = cast(self.0);
          write!(f, "m128i({:?})", a)
        }
      }
    }
  }
}

impl core::fmt::Display for m128i {
  /// Display formats in offset order.
  ///
  /// * Use `width` to specify the lane count you want (default 1).
  /// * Use `alternate` format specifier to give `uX` instead of `iX` output.
  ///
  /// Eg, for 4 lanes of `u32`:
  /// ```txt
  /// format!("{:#4?}", m)
  /// ```
  fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
    match f.width() {
      Some(2) => {
        if f.alternate() {
          let a: [u64; 2] = cast(self.0);
          write!(f, "m128i({}, {})", a[0], a[1])
        } else {
          let a: [i64; 2] = cast(self.0);
          write!(f, "m128i({}, {})", a[0], a[1])
        }
      }
      Some(4) => {
        if f.alternate() {
          let a: [u32; 4] = cast(self.0);
          write!(f, "m128i({}, {}, {}, {})", a[0], a[1], a[2], a[3])
        } else {
          let a: [i32; 4] = cast(self.0);
          write!(f, "m128i({}, {}, {}, {})", a[0], a[1], a[2], a[3])
        }
      }
      Some(8) => {
        if f.alternate() {
          let a: [u16; 8] = cast(self.0);
          write!(
            f,
            "m128i({}, {}, {}, {}, {}, {}, {}, {})",
            a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]
          )
        } else {
          let a: [i16; 8] = cast(self.0);
          write!(
            f,
            "m128i({}, {}, {}, {}, {}, {}, {}, {})",
            a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]
          )
        }
      }
      Some(16) => {
        if f.alternate() {
          let a: [u8; 16] = cast(self.0);
          write!(
            f,
            "m128i({}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {})",
            a[0],
            a[1],
            a[2],
            a[3],
            a[4],
            a[5],
            a[6],
            a[7],
            a[8],
            a[9],
            a[10],
            a[11],
            a[12],
            a[13],
            a[14],
            a[15]
          )
        } else {
          let a: [i8; 16] = cast(self.0);
          write!(
            f,
            "m128i({}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {})",
            a[0],
            a[1],
            a[2],
            a[3],
            a[4],
            a[5],
            a[6],
            a[7],
            a[8],
            a[9],
            a[10],
            a[11],
            a[12],
            a[13],
            a[14],
            a[15]
          )
        }
      }
      _ => {
        if f.alternate() {
          let a: u128 = cast(self.0);
          write!(f, "m128i({})", a)
        } else {
          let a: i128 = cast(self.0);
          write!(f, "m128i({})", a)
        }
      }
    }
  }
}

impl core::fmt::Binary for m128i {
  /// Binary formats in offset order.
  ///
  /// * Use `width` to specify the lane count you want (default 1).
  /// * Use `alternate` format specifier to give leading `0b`.
  ///
  /// Eg, for 4 lanes and leading `0b`:
  /// ```txt
  /// format!("{:#4b}", m)
  /// ```
  fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
    match f.width() {
      Some(2) => {
        let a: [u64; 2] = cast(self.0);
        if f.alternate() {
          write!(f, "m128i({:#b}, {:#b})", a[0], a[1])
        } else {
          write!(f, "m128i({:b}, {:b})", a[0], a[1])
        }
      }
      Some(4) => {
        let a: [u32; 4] = cast(self.0);
        if f.alternate() {
          write!(f, "m128i({:#b}, {:#b}, {:#b}, {:#b})", a[0], a[1], a[2], a[3])
        } else {
          write!(f, "m128i({:b}, {:b}, {:b}, {:b})", a[0], a[1], a[2], a[3])
        }
      }
      Some(8) => {
        let a: [u16; 8] = cast(self.0);
        if f.alternate() {
          write!(
            f,
            "m128i({:#b}, {:#b}, {:#b}, {:#b}, {:#b}, {:#b}, {:#b}, {:#b})",
            a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]
          )
        } else {
          write!(
            f,
            "m128i({:b}, {:b}, {:b}, {:b}, {:b}, {:b}, {:b}, {:b})",
            a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]
          )
        }
      }
      Some(16) => {
        let a: [u8; 16] = cast(self.0);
        if f.alternate() {
          write!(f, "m128i({:#b}, {:#b}, {:#b}, {:#b}, {:#b}, {:#b}, {:#b}, {:#b}, {:#b}, {:#b}, {:#b}, {:#b}, {:#b}, {:#b}, {:#b}, {:#b})", a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15])
        } else {
          write!(f, "m128i({:b}, {:b}, {:b}, {:b}, {:b}, {:b}, {:b}, {:b}, {:b}, {:b}, {:b}, {:b}, {:b}, {:b}, {:b}, {:b})", a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15])
        }
      }
      _ => {
        let a: u128 = cast(self.0);
        if f.alternate() {
          write!(f, "m128i({:#b})", a)
        } else {
          write!(f, "m128i({:b})", a)
        }
      }
    }
  }
}

impl core::fmt::LowerHex for m128i {
  /// LowerHex formats in offset order.
  ///
  /// * Use `width` to specify the lane count you want (default 1).
  /// * Use `alternate` format specifier to give leading `0x`.
  ///
  /// Eg, for 4 lanes and leading `0x`:
  /// ```txt
  /// format!("{:#4x}", m)
  /// ```
  fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
    match f.width() {
      Some(2) => {
        let a: [u64; 2] = cast(self.0);
        if f.alternate() {
          write!(f, "m128i({:#x}, {:#x})", a[0], a[1])
        } else {
          write!(f, "m128i({:x}, {:x})", a[0], a[1])
        }
      }
      Some(4) => {
        let a: [u32; 4] = cast(self.0);
        if f.alternate() {
          write!(f, "m128i({:#x}, {:#x}, {:#x}, {:#x})", a[0], a[1], a[2], a[3])
        } else {
          write!(f, "m128i({:x}, {:x}, {:x}, {:x})", a[0], a[1], a[2], a[3])
        }
      }
      Some(8) => {
        let a: [u16; 8] = cast(self.0);
        if f.alternate() {
          write!(
            f,
            "m128i({:#x}, {:#x}, {:#x}, {:#x}, {:#x}, {:#x}, {:#x}, {:#x})",
            a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]
          )
        } else {
          write!(
            f,
            "m128i({:x}, {:x}, {:x}, {:x}, {:x}, {:x}, {:x}, {:x})",
            a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]
          )
        }
      }
      Some(16) => {
        let a: [u8; 16] = cast(self.0);
        if f.alternate() {
          write!(f, "m128i({:#x}, {:#x}, {:#x}, {:#x}, {:#x}, {:#x}, {:#x}, {:#x}, {:#x}, {:#x}, {:#x}, {:#x}, {:#x}, {:#x}, {:#x}, {:#x})", a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15])
        } else {
          write!(f, "m128i({:x}, {:x}, {:x}, {:x}, {:x}, {:x}, {:x}, {:x}, {:x}, {:x}, {:x}, {:x}, {:x}, {:x}, {:x}, {:x})", a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15])
        }
      }
      _ => {
        let a: u128 = cast(self.0);
        if f.alternate() {
          write!(f, "m128i({:#x})", a)
        } else {
          write!(f, "m128i({:x})", a)
        }
      }
    }
  }
}

impl core::fmt::Octal for m128i {
  /// Octal formats in offset order.
  ///
  /// * Use `width` to specify the lane count you want (default 1).
  /// * Use `alternate` format specifier to give leading `0o`.
  ///
  /// Eg, for 4 lanes and leading `0o`:
  /// ```txt
  /// format!("{:#4o}", m)
  /// ```
  fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
    match f.width() {
      Some(2) => {
        let a: [u64; 2] = cast(self.0);
        if f.alternate() {
          write!(f, "m128i({:#o}, {:#o})", a[0], a[1])
        } else {
          write!(f, "m128i({:o}, {:o})", a[0], a[1])
        }
      }
      Some(4) => {
        let a: [u32; 4] = cast(self.0);
        if f.alternate() {
          write!(f, "m128i({:#o}, {:#o}, {:#o}, {:#o})", a[0], a[1], a[2], a[3])
        } else {
          write!(f, "m128i({:o}, {:o}, {:o}, {:o})", a[0], a[1], a[2], a[3])
        }
      }
      Some(8) => {
        let a: [u16; 8] = cast(self.0);
        if f.alternate() {
          write!(
            f,
            "m128i({:#o}, {:#o}, {:#o}, {:#o}, {:#o}, {:#o}, {:#o}, {:#o})",
            a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]
          )
        } else {
          write!(
            f,
            "m128i({:o}, {:o}, {:o}, {:o}, {:o}, {:o}, {:o}, {:o})",
            a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]
          )
        }
      }
      Some(16) => {
        let a: [u8; 16] = cast(self.0);
        if f.alternate() {
          write!(f, "m128i({:#o}, {:#o}, {:#o}, {:#o}, {:#o}, {:#o}, {:#o}, {:#o}, {:#o}, {:#o}, {:#o}, {:#o}, {:#o}, {:#o}, {:#o}, {:#o})", a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15])
        } else {
          write!(f, "m128i({:o}, {:o}, {:o}, {:o}, {:o}, {:o}, {:o}, {:o}, {:o}, {:o}, {:o}, {:o}, {:o}, {:o}, {:o}, {:o})", a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15])
        }
      }
      _ => {
        let a: u128 = cast(self.0);
        if f.alternate() {
          write!(f, "m128i({:#o})", a)
        } else {
          write!(f, "m128i({:o})", a)
        }
      }
    }
  }
}

impl core::fmt::UpperHex for m128i {
  /// UpperHex formats in offset order.
  ///
  /// * Use `width` to specify the lane count you want (default 1).
  /// * Use `alternate` format specifier to give leading `0x`.
  ///
  /// Eg, for 4 lanes and leading `0x`:
  /// ```txt
  /// format!("{:#4X}", m)
  /// ```
  fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
    match f.width() {
      Some(2) => {
        let a: [u64; 2] = cast(self.0);
        if f.alternate() {
          write!(f, "m128i({:#X}, {:#X})", a[0], a[1])
        } else {
          write!(f, "m128i({:X}, {:X})", a[0], a[1])
        }
      }
      Some(4) => {
        let a: [u32; 4] = cast(self.0);
        if f.alternate() {
          write!(f, "m128i({:#X}, {:#X}, {:#X}, {:#X})", a[0], a[1], a[2], a[3])
        } else {
          write!(f, "m128i({:X}, {:X}, {:X}, {:X})", a[0], a[1], a[2], a[3])
        }
      }
      Some(8) => {
        let a: [u16; 8] = cast(self.0);
        if f.alternate() {
          write!(
            f,
            "m128i({:#X}, {:#X}, {:#X}, {:#X}, {:#X}, {:#X}, {:#X}, {:#X})",
            a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]
          )
        } else {
          write!(
            f,
            "m128i({:X}, {:X}, {:X}, {:X}, {:X}, {:X}, {:X}, {:X})",
            a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]
          )
        }
      }
      Some(16) => {
        let a: [u8; 16] = cast(self.0);
        if f.alternate() {
          write!(f, "m128i({:#X}, {:#X}, {:#X}, {:#X}, {:#X}, {:#X}, {:#X}, {:#X}, {:#X}, {:#X}, {:#X}, {:#X}, {:#X}, {:#X}, {:#X}, {:#X})", a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15])
        } else {
          write!(f, "m128i({:X}, {:X}, {:X}, {:X}, {:X}, {:X}, {:X}, {:X}, {:X}, {:X}, {:X}, {:X}, {:X}, {:X}, {:X}, {:X})", a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15])
        }
      }
      _ => {
        let a: u128 = cast(self.0);
        if f.alternate() {
          write!(f, "m128i({:#X})", a)
        } else {
          write!(f, "m128i({:X})", a)
        }
      }
    }
  }
}

impl BitAnd for m128i {
  type Output = Self;
  /// Bitwise AND.
  #[inline(always)]
  fn bitand(self, rhs: Self) -> Self {
    Self(unsafe { _mm_and_si128(self.0, rhs.0) })
  }
}
impl BitAndAssign for m128i {
  /// Bitwise AND.
  #[inline(always)]
  fn bitand_assign(&mut self, rhs: Self) {
    self.0 = unsafe { _mm_and_si128(self.0, rhs.0) };
  }
}

impl BitOr for m128i {
  type Output = Self;
  /// Bitwise OR.
  #[inline(always)]
  fn bitor(self, rhs: Self) -> Self {
    Self(unsafe { _mm_or_si128(self.0, rhs.0) })
  }
}
impl BitOrAssign for m128i {
  /// Bitwise OR.
  #[inline(always)]
  fn bitor_assign(&mut self, rhs: Self) {
    self.0 = unsafe { _mm_or_si128(self.0, rhs.0) };
  }
}

impl BitXor for m128i {
  type Output = Self;
  /// Bitwise XOR.
  #[inline(always)]
  fn bitxor(self, rhs: Self) -> Self {
    Self(unsafe { _mm_xor_si128(self.0, rhs.0) })
  }
}
impl BitXorAssign for m128i {
  /// Bitwise XOR.
  #[inline(always)]
  fn bitxor_assign(&mut self, rhs: Self) {
    self.0 = unsafe { _mm_xor_si128(self.0, rhs.0) };
  }
}

impl Not for m128i {
  type Output = Self;
  /// Bitwise negation
  #[inline(always)]
  fn not(self) -> Self {
    let i: i64 = -1;
    let b = Self::splat_i64(i);
    self ^ b
  }
}

impl m128i {
  /// Lanewise `i8` wrapping addition
  #[inline(always)]
  pub fn add_i8(self, rhs: Self) -> Self {
    Self(unsafe { _mm_add_epi8(self.0, rhs.0) })
  }
  /// Lanewise `i16` wrapping addition
  #[inline(always)]
  pub fn add_i16(self, rhs: Self) -> Self {
    Self(unsafe { _mm_add_epi16(self.0, rhs.0) })
  }
  /// Lanewise `i32` wrapping addition
  #[inline(always)]
  pub fn add_i32(self, rhs: Self) -> Self {
    Self(unsafe { _mm_add_epi32(self.0, rhs.0) })
  }
  /// Lanewise `i64` wrapping addition
  #[inline(always)]
  pub fn add_i64(self, rhs: Self) -> Self {
    Self(unsafe { _mm_add_epi64(self.0, rhs.0) })
  }

  /// Lanewise `i8` saturating addition
  #[inline(always)]
  pub fn saturating_add_i8(self, rhs: Self) -> Self {
    Self(unsafe { _mm_adds_epi8(self.0, rhs.0) })
  }
  /// Lanewise `i16` saturating addition
  #[inline(always)]
  pub fn saturating_add_i16(self, rhs: Self) -> Self {
    Self(unsafe { _mm_adds_epi16(self.0, rhs.0) })
  }
  /// Lanewise `u8` saturating addition
  #[inline(always)]
  pub fn saturating_add_u8(self, rhs: Self) -> Self {
    Self(unsafe { _mm_adds_epu8(self.0, rhs.0) })
  }
  /// Lanewise `u16` saturating addition
  #[inline(always)]
  pub fn saturating_add_u16(self, rhs: Self) -> Self {
    Self(unsafe { _mm_adds_epu16(self.0, rhs.0) })
  }

  /// Bitwise `(!self) & rhs`
  #[inline(always)]
  pub fn andnot(self, rhs: Self) -> Self {
    Self(unsafe { _mm_andnot_si128(self.0, rhs.0) })
  }

  /// Lanewise `u8` average: `(a + b + 1) >> 1`
  #[inline(always)]
  pub fn average_u8(self, rhs: Self) -> Self {
    Self(unsafe { _mm_avg_epu8(self.0, rhs.0) })
  }
  /// Lanewise `u16` average: `(a + b + 1) >> 1`
  #[inline(always)]
  pub fn average_u16(self, rhs: Self) -> Self {
    Self(unsafe { _mm_avg_epu16(self.0, rhs.0) })
  }

  /// Cast the bits of this `m128i` directly to `m128` without modification.
  #[inline(always)]
  pub fn cast_m128(self) -> m128 {
    m128(unsafe { _mm_castsi128_ps(self.0) })
  }

  /// Lanewise `i8` equality: bool-ish output
  #[inline(always)]
  pub fn cmp_eq_i8(self, rhs: Self) -> Self {
    Self(unsafe { _mm_cmpeq_epi8(self.0, rhs.0) })
  }
  /// Lanewise `i16` equality: bool-ish output
  #[inline(always)]
  pub fn cmp_eq_i16(self, rhs: Self) -> Self {
    Self(unsafe { _mm_cmpeq_epi16(self.0, rhs.0) })
  }
  /// Lanewise `i32` equality: bool-ish output
  #[inline(always)]
  pub fn cmp_eq_i32(self, rhs: Self) -> Self {
    Self(unsafe { _mm_cmpeq_epi32(self.0, rhs.0) })
  }

  /// Lanewise `i8` greater than: bool-ish output
  #[inline(always)]
  pub fn cmp_gt_i8(self, rhs: Self) -> Self {
    Self(unsafe { _mm_cmpgt_epi8(self.0, rhs.0) })
  }
  /// Lanewise `i16` greater than: bool-ish output
  #[inline(always)]
  pub fn cmp_gt_i16(self, rhs: Self) -> Self {
    Self(unsafe { _mm_cmpgt_epi16(self.0, rhs.0) })
  }
  /// Lanewise `i32` greater than: bool-ish output
  #[inline(always)]
  pub fn cmp_gt_i32(self, rhs: Self) -> Self {
    Self(unsafe { _mm_cmpgt_epi32(self.0, rhs.0) })
  }

  /// Lanewise `i8` greater than: bool-ish output
  #[inline(always)]
  pub fn cmp_lt_i8(self, rhs: Self) -> Self {
    Self(unsafe { _mm_cmplt_epi8(self.0, rhs.0) })
  }
  /// Lanewise `i16` greater than: bool-ish output
  #[inline(always)]
  pub fn cmp_lt_i16(self, rhs: Self) -> Self {
    Self(unsafe { _mm_cmplt_epi16(self.0, rhs.0) })
  }
  /// Lanewise `i32` greater than: bool-ish output
  #[inline(always)]
  pub fn cmp_lt_i32(self, rhs: Self) -> Self {
    Self(unsafe { _mm_cmplt_epi32(self.0, rhs.0) })
  }

  /// Rounds the lower two `i32` lanes to `f64` lanes.
  #[inline(always)]
  pub fn round_low_f64(self) -> m128d {
    m128d(unsafe { _mm_cvtepi32_pd(self.0) })
  }

  /// Rounds the `i32` lanes to `f32` lanes.
  #[inline(always)]
  pub fn round_f32(self) -> m128 {
    m128(unsafe { _mm_cvtepi32_ps(self.0) })
  }

  /// Gets out the lowest `i32` lane.
  #[inline(always)]
  pub fn extract0_i32(self) -> i32 {
    unsafe { _mm_cvtsi128_si32(self.0) }
  }

  /// Gets out the lowest `i64` lane.
  #[cfg(target_arch = "x86_64")]
  #[inline(always)]
  pub fn extract0_i64(self) -> i64 {
    unsafe { _mm_cvtsi128_si64(self.0) }
  }

  /// Places the `i32` in the low lane and zeroes other lanes.
  #[inline(always)]
  pub fn set0_i32(val: i32) -> Self {
    Self(unsafe { _mm_cvtsi32_si128(val) })
  }

  /// Places the `i64` in the low lane and zeroes the other lane.
  #[cfg(target_arch = "x86_64")]
  #[inline(always)]
  pub fn set0_i64(val: i64) -> Self {
    Self(unsafe { _mm_cvtsi64_si128(val) })
  }

  /// Loads the aligned `i128` address specified.
  #[inline(always)]
  pub fn load(addr: &Align16<i128>) -> Self {
    let ptr: *const __m128i = addr as *const Align16<i128> as *const __m128i;
    Self(unsafe { _mm_load_si128(ptr) })
  }

  /// Loads the aligned `i64` address specified into the low lane.
  #[inline(always)]
  pub fn load0_i64(addr: &Align16<i64>) -> Self {
    let ptr: *const __m128i = addr as *const Align16<i64> as *const __m128i;
    Self(unsafe { _mm_loadl_epi64(ptr) })
  }

  /// Loads the `i128` address specified, no alignment requirements.
  #[inline(always)]
  #[allow(clippy::cast_ptr_alignment)]
  pub fn load_unaligned(addr: &[u8; 16]) -> Self {
    let ptr: *const __m128i = addr as *const [u8; 16] as *const __m128i;
    Self(unsafe { _mm_loadu_si128(ptr) })
  }

  /// Lanewise `i16` multiply then horizontal add into four lanes.
  ///
  /// The eight `i16` multiplies produce eight intermediate `i32` values, which
  /// then get horizontal added into four `i32` values.
  #[inline(always)]
  pub fn mul_i16_hadd(self, rhs: Self) -> Self {
    Self(unsafe { _mm_madd_epi16(self.0, rhs.0) })
  }

  /// Lanewise `u8` maximum
  #[inline(always)]
  pub fn max_u8(self, rhs: Self) -> Self {
    Self(unsafe { _mm_max_epu8(self.0, rhs.0) })
  }

  /// Lanewise `u8` minimum
  #[inline(always)]
  pub fn min_u8(self, rhs: Self) -> Self {
    Self(unsafe { _mm_min_epu8(self.0, rhs.0) })
  }

  /// Lanewise `i16` maximum
  #[inline(always)]
  pub fn max_i16(self, rhs: Self) -> Self {
    Self(unsafe { _mm_max_epi16(self.0, rhs.0) })
  }

  /// Lanewise `i16` minimum
  #[inline(always)]
  pub fn min_i16(self, rhs: Self) -> Self {
    Self(unsafe { _mm_min_epi16(self.0, rhs.0) })
  }

  /// Copies the low lane `i64` into a new value, upper lane is 0.
  #[inline(always)]
  pub fn copy0_i64(self) -> Self {
    Self(unsafe { _mm_move_epi64(self.0) })
  }

  /// Crates a move mask from the `i8` lanes.
  #[inline(always)]
  pub fn move_mask_i8(self) -> i32 {
    unsafe { _mm_movemask_epi8(self.0) }
  }

  /// Lanewise `i16` multiplication, keep high bits.
  #[inline(always)]
  pub fn mul_high_i16(self, rhs: Self) -> Self {
    Self(unsafe { _mm_mulhi_epi16(self.0, rhs.0) })
  }

  /// Lanewise `i16` multiplication, keep low bits.
  #[inline(always)]
  pub fn mul_low_i16(self, rhs: Self) -> Self {
    Self(unsafe { _mm_mullo_epi16(self.0, rhs.0) })
  }

  /// Lanewise `u16` multiplication, keep high bits.
  #[inline(always)]
  pub fn mul_high_u16(self, rhs: Self) -> Self {
    Self(unsafe { _mm_mulhi_epu16(self.0, rhs.0) })
  }

  /// Lower half of each `i64` lane is `u32` multiplied into `u64` lanes.
  #[inline(always)]
  pub fn half_mul_u32(self, rhs: Self) -> Self {
    Self(unsafe { _mm_mul_epu32(self.0, rhs.0) })
  }

  /// Pack `self` then `rhs` `i16` values into saturated `i8`s
  #[inline(always)]
  pub fn pack_i16_saturating_i8(self, rhs: Self) -> Self {
    Self(unsafe { _mm_packs_epi16(self.0, rhs.0) })
  }

  /// Pack `self` then `rhs` `i32` values into saturated `i16`s
  #[inline(always)]
  pub fn pack_i32_saturating_i16(self, rhs: Self) -> Self {
    Self(unsafe { _mm_packs_epi32(self.0, rhs.0) })
  }

  /// Pack `self` then `rhs` `i16` values into saturated `u8`s
  #[inline(always)]
  pub fn pack_i16_saturating_u8(self, rhs: Self) -> Self {
    Self(unsafe { _mm_packus_epi16(self.0, rhs.0) })
  }

  /// Sum of absolute `i8` differences, eight at a time into two `i64` lanes.
  #[inline(always)]
  pub fn signed_abs_diff_i8(self, rhs: Self) -> Self {
    Self(unsafe { _mm_sad_epu8(self.0, rhs.0) })
  }

  /// Sets the `i8` values together in standard order.
  #[allow(clippy::too_many_arguments)]
  #[allow(clippy::many_single_char_names)]
  #[inline(always)]
  pub fn set_i8(
    a: i8,
    b: i8,
    c: i8,
    d: i8,
    e: i8,
    f: i8,
    g: i8,
    h: i8,
    i: i8,
    j: i8,
    k: i8,
    l: i8,
    m: i8,
    n: i8,
    o: i8,
    p: i8,
  ) -> Self {
    Self(unsafe {
      _mm_set_epi8(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p)
    })
  }

  /// Sets the `i8` values together in reverse order.
  #[allow(clippy::too_many_arguments)]
  #[allow(clippy::many_single_char_names)]
  #[inline(always)]
  pub fn set_reverse_i8(
    a: i8,
    b: i8,
    c: i8,
    d: i8,
    e: i8,
    f: i8,
    g: i8,
    h: i8,
    i: i8,
    j: i8,
    k: i8,
    l: i8,
    m: i8,
    n: i8,
    o: i8,
    p: i8,
  ) -> Self {
    Self(unsafe {
      _mm_setr_epi8(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p)
    })
  }

  /// Sets the `i16` values together in standard order.
  #[allow(clippy::too_many_arguments)]
  #[allow(clippy::many_single_char_names)]
  #[inline(always)]
  pub fn set_i16(
    a: i16,
    b: i16,
    c: i16,
    d: i16,
    e: i16,
    f: i16,
    g: i16,
    h: i16,
  ) -> Self {
    Self(unsafe { _mm_set_epi16(a, b, c, d, e, f, g, h) })
  }

  /// Sets the `i16` values together in reverse order.
  #[allow(clippy::too_many_arguments)]
  #[allow(clippy::many_single_char_names)]
  #[inline(always)]
  pub fn set_reverse_i16(
    a: i16,
    b: i16,
    c: i16,
    d: i16,
    e: i16,
    f: i16,
    g: i16,
    h: i16,
  ) -> Self {
    Self(unsafe { _mm_setr_epi16(a, b, c, d, e, f, g, h) })
  }

  /// Sets the `i32` values together in standard order.
  #[inline(always)]
  pub fn set_i32(a: i32, b: i32, c: i32, d: i32) -> Self {
    Self(unsafe { _mm_set_epi32(a, b, c, d) })
  }

  /// Sets the `i32` values together in reverse order.
  #[inline(always)]
  pub fn set_reverse_i32(a: i32, b: i32, c: i32, d: i32) -> Self {
    Self(unsafe { _mm_setr_epi32(a, b, c, d) })
  }

  /// Sets the `i64` values together in standard order.
  #[inline(always)]
  pub fn set_i64(a: i64, b: i64) -> Self {
    Self(unsafe { _mm_set_epi64x(a, b) })
  }

  /// Splats the `i8` value across all lanes.
  #[inline(always)]
  pub fn splat_i8(a: i8) -> Self {
    Self(unsafe { _mm_set1_epi8(a) })
  }

  /// Splats the `i16` value across all lanes.
  #[inline(always)]
  pub fn splat_i16(a: i16) -> Self {
    Self(unsafe { _mm_set1_epi16(a) })
  }

  /// Splats the `i32` value across all lanes.
  #[inline(always)]
  pub fn splat_i32(a: i32) -> Self {
    Self(unsafe { _mm_set1_epi32(a) })
  }

  /// Splats the `i64` value across all lanes.
  #[inline(always)]
  pub fn splat_i64(a: i64) -> Self {
    Self(unsafe { _mm_set1_epi64x(a) })
  }

  /// Lanewise `i16` left shift using `rhs` as a `i128`: `self << rhs`
  #[inline(always)]
  pub fn shift_left_i16(self, rhs: Self) -> Self {
    Self(unsafe { _mm_sll_epi16(self.0, rhs.0) })
  }

  /// Lanewise `i32` left shift using `rhs` as a `i128`: `self << rhs`
  #[inline(always)]
  pub fn shift_left_i32(self, rhs: Self) -> Self {
    Self(unsafe { _mm_sll_epi32(self.0, rhs.0) })
  }

  /// Lanewise `i64` left shift using `rhs` as a `i128`: `self << rhs`
  #[inline(always)]
  pub fn shift_left_i64(self, rhs: Self) -> Self {
    Self(unsafe { _mm_sll_epi64(self.0, rhs.0) })
  }

  /// Lanewise `i16` right shift using `rhs` as a `i128`: `self >> rhs`
  ///
  /// Sign bit is preserved when shifting.
  #[inline(always)]
  pub fn shift_right_sign_i16(self, rhs: Self) -> Self {
    Self(unsafe { _mm_sra_epi16(self.0, rhs.0) })
  }

  /// Lanewise `i32` right shift using `rhs` as a `i128`: `self >> rhs`
  ///
  /// Sign bit is preserved when shifting.
  #[inline(always)]
  pub fn shift_right_sign_i32(self, rhs: Self) -> Self {
    Self(unsafe { _mm_sra_epi32(self.0, rhs.0) })
  }

  /// Lanewise `i16` right shift using `rhs` as a `i128`: `self >> rhs`
  ///
  /// Zeroes are shifted in regardless of the sign bit.
  #[inline(always)]
  pub fn shift_right_zero_i16(self, rhs: Self) -> Self {
    Self(unsafe { _mm_srl_epi16(self.0, rhs.0) })
  }

  /// Lanewise `i32` right shift using `rhs` as a `i128`: `self >> rhs`
  ///
  /// Zeroes are shifted in regardless of the sign bit.
  #[inline(always)]
  pub fn shift_right_zero_i32(self, rhs: Self) -> Self {
    Self(unsafe { _mm_srl_epi32(self.0, rhs.0) })
  }

  /// Lanewise `i64` right shift using `rhs` as a `i128`: `self >> rhs`
  ///
  /// Zeroes are shifted in regardless of the sign bit.
  #[inline(always)]
  pub fn shift_right_zero_i64(self, rhs: Self) -> Self {
    Self(unsafe { _mm_srl_epi64(self.0, rhs.0) })
  }

  /// Stores the data to the aligned address given.
  #[inline(always)]
  pub fn store(self, addr: &mut Align16<i128>) {
    let ptr: *mut __m128i = addr as *mut Align16<i128> as *mut __m128i;
    unsafe { _mm_store_si128(ptr, self.0) }
  }

  /// Stores the lower `i64` lane to the aligned address given.
  #[inline(always)]
  pub fn store0_i64(self, addr: &mut Align16<i64>) {
    let ptr: *mut __m128i = addr as *mut Align16<i64> as *mut __m128i;
    unsafe { _mm_storel_epi64(ptr, self.0) }
  }

  /// Stores the data to the address given, no alignment requirements.
  #[inline(always)]
  #[allow(clippy::cast_ptr_alignment)]
  pub fn store_unaligned(self, addr: &mut [u8; 16]) {
    let ptr: *mut __m128i = addr as *mut [u8; 16] as *mut __m128i;
    unsafe { _mm_storeu_si128(ptr, self.0) }
  }

  /// Lanewise `i8` wrapping subtraction
  #[inline(always)]
  pub fn sub_i8(self, rhs: Self) -> Self {
    Self(unsafe { _mm_sub_epi8(self.0, rhs.0) })
  }
  /// Lanewise `i16` wrapping subtraction
  #[inline(always)]
  pub fn sub_i16(self, rhs: Self) -> Self {
    Self(unsafe { _mm_sub_epi16(self.0, rhs.0) })
  }
  /// Lanewise `i32` wrapping subtraction
  #[inline(always)]
  pub fn sub_i32(self, rhs: Self) -> Self {
    Self(unsafe { _mm_sub_epi32(self.0, rhs.0) })
  }
  /// Lanewise `i64` wrapping subtraction
  #[inline(always)]
  pub fn sub_i64(self, rhs: Self) -> Self {
    Self(unsafe { _mm_sub_epi64(self.0, rhs.0) })
  }

  /// Lanewise `i8` saturating subtraction
  #[inline(always)]
  pub fn saturating_sub_i8(self, rhs: Self) -> Self {
    Self(unsafe { _mm_subs_epi8(self.0, rhs.0) })
  }
  /// Lanewise `i16` saturating subtraction
  #[inline(always)]
  pub fn saturating_sub_i16(self, rhs: Self) -> Self {
    Self(unsafe { _mm_subs_epi16(self.0, rhs.0) })
  }

  /// Lanewise `u8` saturating subtraction
  #[inline(always)]
  pub fn saturating_sub_u8(self, rhs: Self) -> Self {
    Self(unsafe { _mm_subs_epu8(self.0, rhs.0) })
  }
  /// Lanewise `u16` saturating subtraction
  #[inline(always)]
  pub fn saturating_sub_u16(self, rhs: Self) -> Self {
    Self(unsafe { _mm_subs_epu16(self.0, rhs.0) })
  }

  /// Unpack `i8` values from the high half of `self` and `rhs`
  #[inline(always)]
  pub fn unpack_high_i8(self, rhs: Self) -> Self {
    Self(unsafe { _mm_unpackhi_epi8(self.0, rhs.0) })
  }
  /// Unpack `i16` values from the high half of `self` and `rhs`
  #[inline(always)]
  pub fn unpack_high_i16(self, rhs: Self) -> Self {
    Self(unsafe { _mm_unpackhi_epi16(self.0, rhs.0) })
  }
  /// Unpack `i32` values from the high half of `self` and `rhs`
  #[inline(always)]
  pub fn unpack_high_i32(self, rhs: Self) -> Self {
    Self(unsafe { _mm_unpackhi_epi32(self.0, rhs.0) })
  }
  /// Unpack `i64` values from the high half of `self` and `rhs`
  #[inline(always)]
  pub fn unpack_high_i64(self, rhs: Self) -> Self {
    Self(unsafe { _mm_unpackhi_epi64(self.0, rhs.0) })
  }

  /// Unpack `i8` values from the low half of `self` and `rhs`
  #[inline(always)]
  pub fn unpack_low_i8(self, rhs: Self) -> Self {
    Self(unsafe { _mm_unpacklo_epi8(self.0, rhs.0) })
  }
  /// Unpack `i16` values from the low half of `self` and `rhs`
  #[inline(always)]
  pub fn unpack_low_i16(self, rhs: Self) -> Self {
    Self(unsafe { _mm_unpacklo_epi16(self.0, rhs.0) })
  }
  /// Unpack `i32` values from the low half of `self` and `rhs`
  #[inline(always)]
  pub fn unpack_low_i32(self, rhs: Self) -> Self {
    Self(unsafe { _mm_unpacklo_epi32(self.0, rhs.0) })
  }
  /// Unpack `i64` values from the low half of `self` and `rhs`
  #[inline(always)]
  pub fn unpack_low_i64(self, rhs: Self) -> Self {
    Self(unsafe { _mm_unpacklo_epi64(self.0, rhs.0) })
  }
}

/// A 128-bit SIMD value. Always used as `f64x2`.
///
/// * This documentation numbers the lanes based on the index you'd need to use
///   to access that lane if the value were cast to an array.
/// * This is also the way that the type is printed out using
///   [`Debug`](core::fmt::Debug), [`Display`](core::fmt::Display),
///   [`LowerExp`](core::fmt::LowerExp), and [`UpperExp`](core::fmt::UpperExp).
/// * This is not necessarily the ordering you'll see if you look an `xmm`
///   register in a debugger! Basically because of how little-endian works.
/// * Most operations work per-lane, "lanewise".
/// * Some operations work using lane 0 only. When appropriate, these have the
///   same name as the lanewise version but with a `0` on the end. Eg: `cmp_eq`
///   and `cmp_eq0`. The other lanes are simply copied forward from `self`.
/// * Comparisons give "bool-ish" output, where all bits 1 in a lane is true,
///   and all bits 0 in a lane is false. Unfortunately, all bits 1 with an `f32`
///   is one of the `NaN` values, and `NaN != NaN`, so it can be a little tricky
///   to work with until you're used to it.
#[derive(Clone, Copy)]
#[allow(bad_style)]
#[repr(transparent)]
pub struct m128d(pub __m128d);

impl Default for m128d {
  #[inline(always)]
  fn default() -> Self {
    Self::zeroed()
  }
}

#[cfg(feature = "serde")]
impl serde::Serialize for m128d {
  #[inline(always)]
  fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
  where
    S: serde::Serializer,
  {
    let doubles: [f64; 2] = cast(*self);
    doubles.serialize(serializer)
  }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for m128d {
  #[inline(always)]
  fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
  where
    D: serde::Deserializer<'de>,
  {
    let doubles: [f64; 2] = <[f64; 2]>::deserialize(deserializer)?;
    Ok(cast(doubles))
  }
}

unsafe impl Zeroable for m128d {}
unsafe impl Pod for m128d {}

impl core::fmt::Debug for m128d {
  /// Debug formats in offset order.
  ///
  /// All `Formatter` information is passed directly to each individual `f64`
  /// lane being formatted.
  fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
    let a: [f64; 2] = cast(self.0);
    f.write_str("m128d(")?;
    core::fmt::Debug::fmt(&a[0], f)?;
    f.write_str(", ")?;
    core::fmt::Debug::fmt(&a[1], f)?;
    f.write_str(")")
  }
}

impl core::fmt::Display for m128d {
  /// Display formats in offset order.
  ///
  /// All `Formatter` information is passed directly to each individual `f64`
  /// lane being formatted.
  fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
    let a: [f64; 2] = cast(self.0);
    f.write_str("m128d(")?;
    core::fmt::Display::fmt(&a[0], f)?;
    f.write_str(", ")?;
    core::fmt::Display::fmt(&a[1], f)?;
    f.write_str(")")
  }
}

impl core::fmt::LowerExp for m128d {
  /// LowerExp formats in offset order.
  ///
  /// All `Formatter` information is passed directly to each individual `f64`
  /// lane being formatted.
  fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
    let a: [f64; 2] = cast(self.0);
    f.write_str("m128d(")?;
    core::fmt::LowerExp::fmt(&a[0], f)?;
    f.write_str(", ")?;
    core::fmt::LowerExp::fmt(&a[1], f)?;
    f.write_str(")")
  }
}

impl core::fmt::UpperExp for m128d {
  /// UpperExp formats in offset order.
  ///
  /// All `Formatter` information is passed directly to each individual `f64`
  /// lane being formatted.
  fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
    let a: [f64; 2] = cast(self.0);
    f.write_str("m128d(")?;
    core::fmt::UpperExp::fmt(&a[0], f)?;
    f.write_str(", ")?;
    core::fmt::UpperExp::fmt(&a[1], f)?;
    f.write_str(")")
  }
}

impl Add for m128d {
  type Output = Self;
  /// Lanewise addition.
  #[inline(always)]
  fn add(self, rhs: Self) -> Self {
    Self(unsafe { _mm_add_pd(self.0, rhs.0) })
  }
}
impl AddAssign for m128d {
  /// Lanewise addition.
  #[inline(always)]
  fn add_assign(&mut self, rhs: Self) {
    self.0 = unsafe { _mm_add_pd(self.0, rhs.0) };
  }
}

impl BitAnd for m128d {
  type Output = Self;
  /// Bitwise AND.
  #[inline(always)]
  fn bitand(self, rhs: Self) -> Self {
    Self(unsafe { _mm_and_pd(self.0, rhs.0) })
  }
}
impl BitAndAssign for m128d {
  /// Bitwise AND.
  #[inline(always)]
  fn bitand_assign(&mut self, rhs: Self) {
    self.0 = unsafe { _mm_and_pd(self.0, rhs.0) };
  }
}

impl Div for m128d {
  type Output = Self;
  /// Lanewise division.
  #[inline(always)]
  fn div(self, rhs: Self) -> Self {
    Self(unsafe { _mm_div_pd(self.0, rhs.0) })
  }
}
impl DivAssign for m128d {
  /// Lanewise division.
  #[inline(always)]
  fn div_assign(&mut self, rhs: Self) {
    self.0 = unsafe { _mm_div_pd(self.0, rhs.0) };
  }
}

impl Mul for m128d {
  type Output = Self;
  /// Lanewise multiplication.
  #[inline(always)]
  fn mul(self, rhs: Self) -> Self {
    Self(unsafe { _mm_mul_pd(self.0, rhs.0) })
  }
}
impl MulAssign for m128d {
  /// Lanewise multiplication.
  #[inline(always)]
  fn mul_assign(&mut self, rhs: Self) {
    self.0 = unsafe { _mm_mul_pd(self.0, rhs.0) };
  }
}

impl BitOr for m128d {
  type Output = Self;
  /// Bitwise OR.
  #[inline(always)]
  fn bitor(self, rhs: Self) -> Self {
    Self(unsafe { _mm_or_pd(self.0, rhs.0) })
  }
}
impl BitOrAssign for m128d {
  /// Bitwise OR.
  #[inline(always)]
  fn bitor_assign(&mut self, rhs: Self) {
    self.0 = unsafe { _mm_or_pd(self.0, rhs.0) };
  }
}

impl Sub for m128d {
  type Output = Self;
  /// Lanewise subtraction.
  #[inline(always)]
  fn sub(self, rhs: Self) -> Self {
    Self(unsafe { _mm_sub_pd(self.0, rhs.0) })
  }
}
impl SubAssign for m128d {
  /// Lanewise subtraction.
  #[inline(always)]
  fn sub_assign(&mut self, rhs: Self) {
    self.0 = unsafe { _mm_sub_pd(self.0, rhs.0) };
  }
}

impl BitXor for m128d {
  type Output = Self;
  /// Bitwise XOR.
  #[inline(always)]
  fn bitxor(self, rhs: Self) -> Self {
    Self(unsafe { _mm_xor_pd(self.0, rhs.0) })
  }
}
impl BitXorAssign for m128d {
  /// Bitwise XOR.
  #[inline(always)]
  fn bitxor_assign(&mut self, rhs: Self) {
    self.0 = unsafe { _mm_xor_pd(self.0, rhs.0) };
  }
}

impl Neg for m128d {
  type Output = Self;
  /// Lanewise `0.0 - self`
  #[inline(always)]
  fn neg(self) -> Self {
    Self(unsafe { _mm_sub_pd(_mm_setzero_pd(), self.0) })
  }
}

impl Not for m128d {
  type Output = Self;
  /// Bitwise negation
  #[inline(always)]
  fn not(self) -> Self {
    let f: f64 = cast(-1_i64);
    let b = Self::splat(f);
    self ^ b
  }
}

/// # SSE2 Operations
impl m128d {
  /// Adds the low lane, high lane unaffected.
  #[inline(always)]
  pub fn add0(self, rhs: Self) -> Self {
    Self(unsafe { _mm_add_sd(self.0, rhs.0) })
  }

  /// Bitwise `(!self) & rhs`
  #[inline(always)]
  pub fn andnot(self, rhs: Self) -> Self {
    Self(unsafe { _mm_andnot_pd(self.0, rhs.0) })
  }

  /// Cast the bits of this `m128d` directly to `m128i` without modification.
  #[inline(always)]
  pub fn cast_m128i(self) -> m128i {
    m128i(unsafe { _mm_castpd_si128(self.0) })
  }

  /// Lanewise `self == rhs`, bool-ish output
  #[inline(always)]
  pub fn cmp_eq(self, rhs: Self) -> Self {
    Self(unsafe { _mm_cmpeq_pd(self.0, rhs.0) })
  }

  /// Lane 0: `self == rhs`, bool-ish output
  #[inline(always)]
  pub fn cmp_eq0(self, rhs: Self) -> Self {
    Self(unsafe { _mm_cmpeq_sd(self.0, rhs.0) })
  }

  /// Lanewise `self >= rhs`, bool-ish output
  #[inline(always)]
  pub fn cmp_ge(self, rhs: Self) -> Self {
    Self(unsafe { _mm_cmpge_pd(self.0, rhs.0) })
  }

  /// Lane 0: `self >= rhs`, bool-ish output
  #[inline(always)]
  pub fn cmp_ge0(self, rhs: Self) -> Self {
    Self(unsafe { _mm_cmpge_sd(self.0, rhs.0) })
  }

  /// Lanewise `self > rhs`, bool-ish output
  #[inline(always)]
  pub fn cmp_gt(self, rhs: Self) -> Self {
    Self(unsafe { _mm_cmpgt_pd(self.0, rhs.0) })
  }

  /// Lane 0: `self > rhs`, bool-ish output
  #[inline(always)]
  pub fn cmp_gt0(self, rhs: Self) -> Self {
    Self(unsafe { _mm_cmpgt_sd(self.0, rhs.0) })
  }

  /// Lanewise `self <= rhs`, bool-ish output
  #[inline(always)]
  pub fn cmp_le(self, rhs: Self) -> Self {
    Self(unsafe { _mm_cmple_pd(self.0, rhs.0) })
  }

  /// Lane 0: `self <= rhs`, bool-ish output
  #[inline(always)]
  pub fn cmp_le0(self, rhs: Self) -> Self {
    Self(unsafe { _mm_cmple_sd(self.0, rhs.0) })
  }

  /// Lanewise `self < rhs`, bool-ish output
  #[inline(always)]
  pub fn cmp_lt(self, rhs: Self) -> Self {
    Self(unsafe { _mm_cmplt_pd(self.0, rhs.0) })
  }

  /// Lane 0: `self < rhs`, bool-ish output
  #[inline(always)]
  pub fn cmp_lt0(self, rhs: Self) -> Self {
    Self(unsafe { _mm_cmplt_sd(self.0, rhs.0) })
  }

  /// Lanewise `self != rhs`, bool-ish output
  #[inline(always)]
  pub fn cmp_ne(self, rhs: Self) -> Self {
    Self(unsafe { _mm_cmpneq_pd(self.0, rhs.0) })
  }

  /// Lane 0: `self != rhs`, bool-ish output
  #[inline(always)]
  pub fn cmp_ne0(self, rhs: Self) -> Self {
    Self(unsafe { _mm_cmpneq_sd(self.0, rhs.0) })
  }

  /// Lanewise `!(self >= rhs)`, bool-ish output
  ///
  /// Also, 3rd Impact and all that, of course.
  #[inline(always)]
  pub fn cmp_nge(self, rhs: Self) -> Self {
    Self(unsafe { _mm_cmpnge_pd(self.0, rhs.0) })
  }

  /// Lane 0: `!(self >= rhs)`, bool-ish output
  #[inline(always)]
  pub fn cmp_nge0(self, rhs: Self) -> Self {
    Self(unsafe { _mm_cmpnge_sd(self.0, rhs.0) })
  }

  /// Lanewise `!(self > rhs)`, bool-ish output
  #[inline(always)]
  pub fn cmp_ngt(self, rhs: Self) -> Self {
    Self(unsafe { _mm_cmpngt_pd(self.0, rhs.0) })
  }

  /// Lane 0: `!(self > rhs)`, bool-ish output
  #[inline(always)]
  pub fn cmp_ngt0(self, rhs: Self) -> Self {
    Self(unsafe { _mm_cmpngt_sd(self.0, rhs.0) })
  }

  /// Lanewise `!(self <= rhs)`, bool-ish output
  #[inline(always)]
  pub fn cmp_nle(self, rhs: Self) -> Self {
    Self(unsafe { _mm_cmpnle_pd(self.0, rhs.0) })
  }

  /// Lane 0: `!(self <= rhs)`, bool-ish output
  #[inline(always)]
  pub fn cmp_nle0(self, rhs: Self) -> Self {
    Self(unsafe { _mm_cmpnle_sd(self.0, rhs.0) })
  }

  /// Lanewise `!(self < rhs)`, bool-ish output
  #[inline(always)]
  pub fn cmp_nlt(self, rhs: Self) -> Self {
    Self(unsafe { _mm_cmpnlt_pd(self.0, rhs.0) })
  }

  /// Lane 0: `!(self < rhs)`, bool-ish output
  #[inline(always)]
  pub fn cmp_nlt0(self, rhs: Self) -> Self {
    Self(unsafe { _mm_cmpnlt_sd(self.0, rhs.0) })
  }

  /// Lanewise `self.not_nan() & rhs.not_nan()`, bool-ish output
  #[inline(always)]
  pub fn cmp_ordinary(self, rhs: Self) -> Self {
    Self(unsafe { _mm_cmpord_pd(self.0, rhs.0) })
  }

  /// Lane 0: `self.not_nan() & rhs.not_nan()`, bool-ish output
  #[inline(always)]
  pub fn cmp_ordinary0(self, rhs: Self) -> Self {
    Self(unsafe { _mm_cmpord_sd(self.0, rhs.0) })
  }

  /// Lanewise `self.is_nan() | rhs.is_nan()`, bool-ish output
  #[inline(always)]
  pub fn cmp_nan(self, rhs: Self) -> Self {
    Self(unsafe { _mm_cmpunord_pd(self.0, rhs.0) })
  }

  /// Lane 0: `self.is_nan() | rhs.is_nan()`, bool-ish output
  #[inline(always)]
  pub fn cmp_nan0(self, rhs: Self) -> Self {
    Self(unsafe { _mm_cmpunord_sd(self.0, rhs.0) })
  }

  /// Lane 0: `self == rhs`, 0 or 1 `i32` output.
  #[inline(always)]
  pub fn cmpi_eq0(self, rhs: Self) -> i32 {
    unsafe { _mm_comieq_sd(self.0, rhs.0) }
  }

  /// Lane 0: `self >= rhs`, 0 or 1 `i32` output.
  #[inline(always)]
  pub fn cmpi_ge0(self, rhs: Self) -> i32 {
    unsafe { _mm_comige_sd(self.0, rhs.0) }
  }

  /// Lane 0: `self > rhs`, 0 or 1 `i32` output.
  #[inline(always)]
  pub fn cmpi_gt0(self, rhs: Self) -> i32 {
    unsafe { _mm_comigt_sd(self.0, rhs.0) }
  }

  /// Lane 0: `self <= rhs`, 0 or 1 `i32` output.
  #[inline(always)]
  pub fn cmpi_le0(self, rhs: Self) -> i32 {
    unsafe { _mm_comile_sd(self.0, rhs.0) }
  }

  /// Lane 0: `self < rhs`, 0 or 1 `i32` output.
  #[inline(always)]
  pub fn cmpi_lt0(self, rhs: Self) -> i32 {
    unsafe { _mm_comilt_sd(self.0, rhs.0) }
  }

  /// Lane 0: `self != rhs`, 0 or 1 `i32` output.
  #[inline(always)]
  pub fn cmpi_ne0(self, rhs: Self) -> i32 {
    unsafe { _mm_comineq_sd(self.0, rhs.0) }
  }

  /// Round the lanes to `i32` and place as the two lower lanes of an [`m128i`]
  #[inline(always)]
  pub fn round_i32x4(self) -> m128i {
    m128i(unsafe { _mm_cvtpd_epi32(self.0) })
  }

  /// Round the lanes to `f32` and place as the two lower lanes of an [`m128`]
  #[inline(always)]
  pub fn round_f32x4(self) -> m128 {
    m128(unsafe { _mm_cvtpd_ps(self.0) })
  }

  /// Get the lower lane value as `f64`.
  #[inline(always)]
  pub fn extract0(self) -> f64 {
    unsafe { _mm_cvtsd_f64(self.0) }
  }

  /// Round lower lane to `i32` and return it.
  #[inline(always)]
  pub fn round_i32_extract0(self) -> i32 {
    unsafe { _mm_cvtsd_si32(self.0) }
  }

  /// Round lower lane to `i64` and return it.
  #[cfg(target_arch = "x86_64")]
  #[inline(always)]
  pub fn round_i64_extract0(self) -> i64 {
    unsafe { _mm_cvtsd_si64(self.0) }
  }

  /// Replace lane 0 with `i32` rounded to `f64`, lane 1 unaffected.
  #[inline(always)]
  pub fn replace0_with_i32(self, rhs: i32) -> Self {
    Self(unsafe { _mm_cvtsi32_sd(self.0, rhs) })
  }

  /// Replace lane 0 with `i64` rounded to `f64`, lane 1 unaffected.
  #[cfg(target_arch = "x86_64")]
  #[inline(always)]
  pub fn replace0_with_i64(self, rhs: i64) -> Self {
    Self(unsafe { _mm_cvtsi64_sd(self.0, rhs) })
  }

  /// Replace lane 0 with `rhs` low `f32` rounded to `f64`, lane 1 unaffected.
  #[inline(always)]
  pub fn replace0_with_f32(self, rhs: m128) -> Self {
    Self(unsafe { _mm_cvtss_sd(self.0, rhs.0) })
  }

  /// Truncate the lanes to `i32` and place as the two lower lanes of an
  /// [`m128i`]
  #[inline(always)]
  pub fn truncate_i32x4(self) -> m128i {
    m128i(unsafe { _mm_cvttpd_epi32(self.0) })
  }

  /// Truncate lane 0 to `i32` and return it.
  #[inline(always)]
  pub fn truncate0_i32(self) -> i32 {
    unsafe { _mm_cvttsd_si32(self.0) }
  }

  /// Truncate lane 0 to `i64` and return it.
  #[cfg(target_arch = "x86_64")]
  #[inline(always)]
  pub fn truncate0_i64(self) -> i64 {
    unsafe { _mm_cvttsd_si64(self.0) }
  }

  /// Divides the low lane, high lane unaffected.
  #[inline(always)]
  pub fn div0(self, rhs: Self) -> Self {
    Self(unsafe { _mm_div_sd(self.0, rhs.0) })
  }

  /// Load aligned `f64` array data.
  ///
  /// Loads the 0th index as the 0th lane, and the 1st index as the 1st lane.
  #[inline(always)]
  pub fn load(addr: &Align16<[f64; 2]>) -> Self {
    let ptr: *const f64 = addr as *const Align16<[f64; 2]> as *const f64;
    Self(unsafe { _mm_load_pd(ptr) })
  }

  /// Load the 16-byte aligned `f64` address into both lanes.
  #[inline(always)]
  pub fn load_aligned_splat(addr: &Align16<f64>) -> Self {
    let ptr: *const f64 = addr as *const Align16<f64> as *const f64;
    Self(unsafe { _mm_load_pd1(ptr) })
  }

  /// Load the `f64` addressed into the low lane, high lane `0.0`.
  #[allow(clippy::trivially_copy_pass_by_ref)]
  #[inline(always)]
  pub fn load0(addr: &f64) -> Self {
    Self(unsafe { _mm_load_sd(addr) })
  }

  /// Replace high lane with the float referenced, low lane unaffected.
  #[allow(clippy::trivially_copy_pass_by_ref)]
  #[inline(always)]
  pub fn replace_high(self, addr: &f64) -> Self {
    Self(unsafe { _mm_loadh_pd(self.0, addr) })
  }

  /// Replace low lane with the float referenced, high lane unaffected.
  #[allow(clippy::trivially_copy_pass_by_ref)]
  #[inline(always)]
  pub fn replace_low(self, addr: &f64) -> Self {
    Self(unsafe { _mm_loadl_pd(self.0, addr) })
  }

  /// Load aligned `f64` array data in reverse order.
  ///
  /// Loads the 0th index as the 1st lane, and the 1st index as the 0th lane.
  #[inline(always)]
  pub fn load_reverse(addr: &Align16<[f64; 2]>) -> Self {
    let ptr: *const f64 = addr as *const Align16<[f64; 2]> as *const f64;
    Self(unsafe { _mm_loadr_pd(ptr) })
  }

  /// Load `f64` array data without alignment requirement.
  ///
  /// Loads the 0th index as the 1st lane, and the 1st index as the 0th lane.
  #[inline(always)]
  pub fn load_unaligned(addr: &[f64; 2]) -> Self {
    let ptr: *const f64 = addr as *const [f64; 2] as *const f64;
    Self(unsafe { _mm_loadu_pd(ptr) })
  }

  /// Lanewise maximum.
  #[inline(always)]
  pub fn max(self, rhs: Self) -> Self {
    Self(unsafe { _mm_max_pd(self.0, rhs.0) })
  }

  /// Lane 0 maximum, other lanes are `self`.
  #[inline(always)]
  pub fn max0(self, rhs: Self) -> Self {
    Self(unsafe { _mm_max_sd(self.0, rhs.0) })
  }

  /// Lanewise minimum.
  #[inline(always)]
  pub fn min(self, rhs: Self) -> Self {
    Self(unsafe { _mm_min_pd(self.0, rhs.0) })
  }

  /// Lane 0 minimum, other lanes are `self`.
  #[inline(always)]
  pub fn min0(self, rhs: Self) -> Self {
    Self(unsafe { _mm_min_sd(self.0, rhs.0) })
  }

  /// Copies lane 0 from `rhs`, other lane is unchanged.
  #[inline(always)]
  pub fn copy0(self, rhs: Self) -> Self {
    Self(unsafe { _mm_move_sd(self.0, rhs.0) })
  }

  /// Assumes that this is a bool-ish mask and packs it into an `i32`.
  ///
  /// Specifically, the output `i32` has bits 0/1 set to be the same as the most
  /// significant bit in lanes 0/1 of `self`.
  ///
  /// (Yes, this name is kinda stupid but I couldn't come up with a better thing
  /// to rename it to, oh well.)
  #[inline(always)]
  pub fn move_mask(self) -> i32 {
    unsafe { _mm_movemask_pd(self.0) }
  }

  /// Multiplies the low lane, high lane unaffected.
  #[inline(always)]
  pub fn mul0(self, rhs: Self) -> Self {
    Self(unsafe { _mm_mul_sd(self.0, rhs.0) })
  }

  /// Set two `f64` values into an `m128d`.
  ///
  /// Because of how little-endian works, this produces the **opposite** lane
  /// order as you'd get compared to putting the arguments in to an array and
  /// then using [`load`](m128d::load) on that array. Same with using
  /// `transmute` or similar.
  #[inline(always)]
  pub fn set(a: f64, b: f64) -> Self {
    Self(unsafe { _mm_set_pd(a, b) })
  }

  /// Set an `f64` as the value in both lanes.
  #[inline(always)]
  pub fn splat(a: f64) -> Self {
    Self(unsafe { _mm_set_pd1(a) })
  }

  /// Sets the `f64` as lane 0, other lane `0.0`.
  #[inline(always)]
  pub fn set0(a: f64) -> Self {
    Self(unsafe { _mm_set_sd(a) })
  }

  /// Set two `f64` values into an `m128d` with reverse ordering.
  ///
  /// Because of how little-endian works, this produces the **the same** lane
  /// order as you'd get compared to putting the arguments in to an array and
  /// then using [`load`](m128d::load) on that array. Same with using
  /// `transmute` or similar.
  #[inline(always)]
  pub fn set_reverse(a: f64, b: f64) -> Self {
    Self(unsafe { _mm_setr_pd(a, b) })
  }

  /// Lanewise square root.
  #[inline(always)]
  pub fn sqrt(self) -> Self {
    Self(unsafe { _mm_sqrt_pd(self.0) })
  }

  /// `rhs[0]` square root copied over top of `self[0]`, `self[1]` is
  /// unaffected.
  #[inline(always)]
  pub fn sqrt_other0(self, rhs: Self) -> Self {
    Self(unsafe { _mm_sqrt_sd(self.0, rhs.0) })
  }

  /// Stores the data here to the aligned address given.
  #[inline(always)]
  pub fn store(self, addr: &mut Align16<[f64; 2]>) {
    let ptr: *mut f64 = addr as *mut Align16<[f64; 2]> as *mut f64;
    unsafe { _mm_store_pd(ptr, self.0) }
  }

  /// Stores lane 0 to both slots of the aligned address given.
  #[inline(always)]
  pub fn store0_all(self, addr: &mut Align16<[f64; 2]>) {
    let ptr: *mut f64 = addr as *mut Align16<[f64; 2]> as *mut f64;
    unsafe { _mm_store_pd1(ptr, self.0) }
  }

  /// Stores the low lane to the address given.
  #[inline(always)]
  pub fn store_low(self, addr: &mut f64) {
    unsafe { _mm_storel_pd(addr, self.0) }
  }

  /// Stores the high lane to the address given.
  #[inline(always)]
  pub fn store_high(self, addr: &mut f64) {
    unsafe { _mm_storeh_pd(addr, self.0) }
  }

  /// Stores the data here to the aligned address given, reverse order.
  #[inline(always)]
  pub fn store_reverse(self, addr: &mut Align16<[f64; 2]>) {
    let ptr: *mut f64 = addr as *mut Align16<[f64; 2]> as *mut f64;
    unsafe { _mm_storer_pd(ptr, self.0) }
  }

  /// Stores the data here to the address given.
  #[inline(always)]
  pub fn store_unaligned(self, addr: &mut [f64; 2]) {
    let ptr: *mut f64 = addr as *mut [f64; 2] as *mut f64;
    unsafe { _mm_storeu_pd(ptr, self.0) }
  }

  /// Subtracts the low lane, high lane unaffected.
  #[inline(always)]
  pub fn sub0(self, rhs: Self) -> Self {
    Self(unsafe { _mm_sub_sd(self.0, rhs.0) })
  }

  /// Gives `m128d(self[1], rhs[1])`
  #[inline(always)]
  pub fn unpack_high(self, rhs: Self) -> Self {
    Self(unsafe { _mm_unpackhi_pd(self.0, rhs.0) })
  }

  /// Gives `m128d(self[0], rhs[0])`
  #[inline(always)]
  pub fn unpack_low(self, rhs: Self) -> Self {
    Self(unsafe { _mm_unpacklo_pd(self.0, rhs.0) })
  }
}

/// Hint to the CPU that you're doing a spin-wait loop.
#[inline(always)]
pub fn pause() {
  unsafe { _mm_pause() }
}

/// Flushes the cache line pointed to from all levels.
#[inline(always)]
pub fn cache_flush(ptr: *mut impl Sized) {
  unsafe { _mm_clflush(ptr as *mut _) }
}

/// Shuffles around some `f64` lanes into a new `m128d`
///
/// This is a macro and not a function because the shuffle pattern must be a
/// compile time constant. The macro takes some requested indexes and then makes
/// the correct shuffle pattern constant for you.
///
/// * `$a` and `$b` are any `m128d` expressions.
/// * `$i0a` and `$i1b` must both be `0` or `1`. Technically any `u32` literal
///   will work, but only the lowest bit is used so stick to `0` or `1`.
/// * The 0th lane of output comes from `$a`, and the 1st lane comes from `$b`,
///   as the names `$i0a` and `$i1b` hint.
///
/// ```txt
/// shuffle128!(a, b, [0, 1])
/// ```
///
/// Would give an output of: `a[0], b[1]`
#[macro_export]
macro_rules! shuffle128d {
  ($a:expr, $b:expr, [$i0a:literal,$i1b:literal]) => {{
    // Keep only 1 bit per index
    const I0A: u32 = $i0a & 0b1;
    const I1B: u32 = $i1b & 0b1;
    // pack it up little-endian
    const IMM: i32 = (I0A | I1B << 1) as i32;
    //
    #[cfg(target_arch = "x86")]
    use core::arch::x86::_mm_shuffle_pd;
    #[cfg(target_arch = "x86")]
    use $crate::arch::x86::m128d;
    //
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::_mm_shuffle_pd;
    #[cfg(target_arch = "x86_64")]
    use $crate::arch::x86_64::m128d;
    //
    m128d(unsafe { _mm_shuffle_pd($a.0, $b.0, IMM) })
  }};
}
