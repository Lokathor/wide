#![cfg(target_feature="sse")]

use super::*;

/// A 128-bit SIMD value. Always used as `f32x4`.
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
///   same name as the lanewise version but with a `0` on the end (example:
///   `cmp_eq` and `cmp_eq0`). With the 0 version the other lanes are simply
///   copied forward from `self`.
/// * Comparisons give "bool-ish" output, where all bits 1 in a lane is true,
///   and all bits 0 in a lane is false. Unfortunately, all bits 1 with an `f32`
///   is one of the `NaN` values, and `NaN != NaN`, so it can be a little tricky
///   to work with until you're used to it.
#[derive(Clone, Copy)]
#[allow(bad_style)]
#[repr(transparent)]
pub struct m128(pub __m128);

unsafe impl Zeroable for m128 {}
unsafe impl Pod for m128 {}

impl Default for m128 {
  #[inline(always)]
  #[must_use]
  fn default() -> Self {
    Self::zeroed()
  }
}

#[cfg(feature = "serde")]
impl serde::Serialize for m128 {
  #[inline(always)]
  fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
  where
    S: serde::Serializer,
  {
    let floats: [f32; 4] = cast(*self);
    floats.serialize(serializer)
  }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for m128 {
  #[inline(always)]
  fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
  where
    D: serde::Deserializer<'de>,
  {
    let floats: [f32; 4] = <[f32; 4]>::deserialize(deserializer)?;
    Ok(cast(floats))
  }
}

impl core::fmt::Debug for m128 {
  /// Debug formats in offset order.
  ///
  /// All `Formatter` information is passed directly to each individual `f32`
  /// lane being formatted.
  fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
    let a: [f32; 4] = cast(self.0);
    f.write_str("m128(")?;
    core::fmt::Debug::fmt(&a[0], f)?;
    f.write_str(", ")?;
    core::fmt::Debug::fmt(&a[1], f)?;
    f.write_str(", ")?;
    core::fmt::Debug::fmt(&a[2], f)?;
    f.write_str(", ")?;
    core::fmt::Debug::fmt(&a[3], f)?;
    f.write_str(")")
  }
}

impl core::fmt::Display for m128 {
  /// Display formats in offset order.
  ///
  /// All `Formatter` information is passed directly to each individual `f32`
  /// lane being formatted.
  fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
    let a: [f32; 4] = cast(self.0);
    f.write_str("m128(")?;
    core::fmt::Display::fmt(&a[0], f)?;
    f.write_str(", ")?;
    core::fmt::Display::fmt(&a[1], f)?;
    f.write_str(", ")?;
    core::fmt::Display::fmt(&a[2], f)?;
    f.write_str(", ")?;
    core::fmt::Display::fmt(&a[3], f)?;
    f.write_str(")")
  }
}

impl core::fmt::LowerExp for m128 {
  /// LowerExp formats in offset order.
  ///
  /// All `Formatter` information is passed directly to each individual `f32`
  /// lane being formatted.
  fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
    let a: [f32; 4] = cast(self.0);
    f.write_str("m128(")?;
    core::fmt::LowerExp::fmt(&a[0], f)?;
    f.write_str(", ")?;
    core::fmt::LowerExp::fmt(&a[1], f)?;
    f.write_str(", ")?;
    core::fmt::LowerExp::fmt(&a[2], f)?;
    f.write_str(", ")?;
    core::fmt::LowerExp::fmt(&a[3], f)?;
    f.write_str(")")
  }
}

impl core::fmt::UpperExp for m128 {
  /// UpperExp formats in offset order.
  ///
  /// All `Formatter` information is passed directly to each individual `f32`
  /// lane being formatted.
  fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
    let a: [f32; 4] = cast(self.0);
    f.write_str("m128(")?;
    core::fmt::UpperExp::fmt(&a[0], f)?;
    f.write_str(", ")?;
    core::fmt::UpperExp::fmt(&a[1], f)?;
    f.write_str(", ")?;
    core::fmt::UpperExp::fmt(&a[2], f)?;
    f.write_str(", ")?;
    core::fmt::UpperExp::fmt(&a[3], f)?;
    f.write_str(")")
  }
}

impl Add for m128 {
  type Output = Self;
  /// Lanewise addition.
  #[inline(always)]
  #[must_use]
  fn add(self, rhs: Self) -> Self {
    Self(unsafe { _mm_add_ps(self.0, rhs.0) })
  }
}
impl AddAssign for m128 {
  /// Lanewise addition.
  #[inline(always)]
  fn add_assign(&mut self, rhs: Self) {
    self.0 = unsafe { _mm_add_ps(self.0, rhs.0) };
  }
}

impl BitAnd for m128 {
  type Output = Self;
  /// Bitwise AND.
  #[inline(always)]
  #[must_use]
  fn bitand(self, rhs: Self) -> Self {
    Self(unsafe { _mm_and_ps(self.0, rhs.0) })
  }
}
impl BitAndAssign for m128 {
  /// Bitwise AND.
  #[inline(always)]
  fn bitand_assign(&mut self, rhs: Self) {
    self.0 = unsafe { _mm_and_ps(self.0, rhs.0) };
  }
}

impl Div for m128 {
  type Output = Self;
  /// Lanewise division.
  #[inline(always)]
  #[must_use]
  fn div(self, rhs: Self) -> Self {
    Self(unsafe { _mm_div_ps(self.0, rhs.0) })
  }
}
impl DivAssign for m128 {
  /// Lanewise division.
  #[inline(always)]
  fn div_assign(&mut self, rhs: Self) {
    self.0 = unsafe { _mm_div_ps(self.0, rhs.0) };
  }
}

impl Mul for m128 {
  type Output = Self;
  /// Lanewise multiplication.
  #[inline(always)]
  #[must_use]
  fn mul(self, rhs: Self) -> Self {
    Self(unsafe { _mm_mul_ps(self.0, rhs.0) })
  }
}
impl MulAssign for m128 {
  /// Lanewise multiplication.
  #[inline(always)]
  fn mul_assign(&mut self, rhs: Self) {
    self.0 = unsafe { _mm_mul_ps(self.0, rhs.0) };
  }
}

impl Sub for m128 {
  type Output = Self;
  /// Lanewise subtraction.
  #[inline(always)]
  #[must_use]
  fn sub(self, rhs: Self) -> Self {
    Self(unsafe { _mm_sub_ps(self.0, rhs.0) })
  }
}
impl SubAssign for m128 {
  /// Lanewise subtraction.
  #[inline(always)]
  fn sub_assign(&mut self, rhs: Self) {
    self.0 = unsafe { _mm_sub_ps(self.0, rhs.0) };
  }
}

impl BitOr for m128 {
  type Output = Self;
  /// Bitwise OR.
  #[inline(always)]
  #[must_use]
  fn bitor(self, rhs: Self) -> Self {
    Self(unsafe { _mm_or_ps(self.0, rhs.0) })
  }
}
impl BitOrAssign for m128 {
  /// Bitwise OR.
  #[inline(always)]
  fn bitor_assign(&mut self, rhs: Self) {
    self.0 = unsafe { _mm_or_ps(self.0, rhs.0) };
  }
}

impl BitXor for m128 {
  type Output = Self;
  /// Bitwise XOR.
  #[inline(always)]
  #[must_use]
  fn bitxor(self, rhs: Self) -> Self {
    Self(unsafe { _mm_xor_ps(self.0, rhs.0) })
  }
}
impl BitXorAssign for m128 {
  /// Bitwise XOR.
  #[inline(always)]
  fn bitxor_assign(&mut self, rhs: Self) {
    self.0 = unsafe { _mm_xor_ps(self.0, rhs.0) };
  }
}

impl Neg for m128 {
  type Output = Self;
  /// Lanewise `0.0 - self`
  #[inline(always)]
  #[must_use]
  fn neg(self) -> Self {
    Self(unsafe { _mm_sub_ps(_mm_setzero_ps(), self.0) })
  }
}

impl Not for m128 {
  type Output = Self;
  /// Bitwise negation
  #[inline(always)]
  #[must_use]
  fn not(self) -> Self {
    let f: f32 = cast(-1_i32);
    let b = Self::splat(f);
    self ^ b
  }
}

/// # SSE Operations
impl m128 {
  /// Adds the 0th lanes without affecting the other lanes of `self.
  #[inline(always)]
  #[must_use]
  pub fn add0(self, rhs: Self) -> Self {
    Self(unsafe { _mm_add_ss(self.0, rhs.0) })
  }

  /// Bitwise `(!self) & rhs`
  #[inline(always)]
  #[must_use]
  pub fn andnot(self, rhs: Self) -> Self {
    Self(unsafe { _mm_andnot_ps(self.0, rhs.0) })
  }

  /// Lanewise `self == rhs` check, bool-ish output.
  #[inline(always)]
  #[must_use]
  pub fn cmp_eq(self, rhs: Self) -> Self {
    Self(unsafe { _mm_cmpeq_ps(self.0, rhs.0) })
  }

  /// Lane 0: `self == rhs`, bool-ish output.
  #[inline(always)]
  #[must_use]
  pub fn cmp_eq0(self, rhs: Self) -> Self {
    Self(unsafe { _mm_cmpeq_ss(self.0, rhs.0) })
  }

  /// Lanewise `self >= rhs` check, bool-ish output.
  #[inline(always)]
  #[must_use]
  pub fn cmp_ge(self, rhs: Self) -> Self {
    Self(unsafe { _mm_cmpge_ps(self.0, rhs.0) })
  }

  /// Lane 0: `self >= rhs`, bool-ish output.
  #[inline(always)]
  #[must_use]
  pub fn cmp_ge0(self, rhs: Self) -> Self {
    Self(unsafe { _mm_cmpge_ss(self.0, rhs.0) })
  }

  /// Lanewise `self > rhs` check, bool-ish output.
  #[inline(always)]
  #[must_use]
  pub fn cmp_gt(self, rhs: Self) -> Self {
    Self(unsafe { _mm_cmpgt_ps(self.0, rhs.0) })
  }

  /// Lane 0: `self > rhs`, bool-ish output.
  #[inline(always)]
  #[must_use]
  pub fn cmp_gt0(self, rhs: Self) -> Self {
    Self(unsafe { _mm_cmpgt_ss(self.0, rhs.0) })
  }

  /// Lanewise `self <= rhs` check, bool-ish output.
  #[inline(always)]
  #[must_use]
  pub fn cmp_le(self, rhs: Self) -> Self {
    Self(unsafe { _mm_cmple_ps(self.0, rhs.0) })
  }

  /// Lane 0: `self <= rhs`, bool-ish output.
  #[inline(always)]
  #[must_use]
  pub fn cmp_le0(self, rhs: Self) -> Self {
    Self(unsafe { _mm_cmple_ss(self.0, rhs.0) })
  }

  /// Lanewise `self < rhs` check, bool-ish output.
  #[inline(always)]
  #[must_use]
  pub fn cmp_lt(self, rhs: Self) -> Self {
    Self(unsafe { _mm_cmplt_ps(self.0, rhs.0) })
  }

  /// Lane 0: `self < rhs`, bool-ish output.
  #[inline(always)]
  #[must_use]
  pub fn cmp_lt0(self, rhs: Self) -> Self {
    Self(unsafe { _mm_cmplt_ss(self.0, rhs.0) })
  }

  /// Lanewise `self != rhs` check, bool-ish output.
  #[inline(always)]
  #[must_use]
  pub fn cmp_ne(self, rhs: Self) -> Self {
    Self(unsafe { _mm_cmpneq_ps(self.0, rhs.0) })
  }

  /// Lane 0: `self != rhs`, bool-ish output.
  #[inline(always)]
  #[must_use]
  pub fn cmp_ne0(self, rhs: Self) -> Self {
    Self(unsafe { _mm_cmpneq_ss(self.0, rhs.0) })
  }

  /// Lanewise `!(self >= rhs)` check, bool-ish output.
  ///
  /// Also, this triggers 3rd Impact.
  #[inline(always)]
  #[must_use]
  pub fn cmp_nge(self, rhs: Self) -> Self {
    Self(unsafe { _mm_cmpnge_ps(self.0, rhs.0) })
  }

  /// Lane 0: `!(self >= rhs)`, bool-ish output.
  #[inline(always)]
  #[must_use]
  pub fn cmp_nge0(self, rhs: Self) -> Self {
    Self(unsafe { _mm_cmpnge_ss(self.0, rhs.0) })
  }

  /// Lanewise `!(self > rhs)` check, bool-ish output.
  #[inline(always)]
  #[must_use]
  pub fn cmp_ngt(self, rhs: Self) -> Self {
    Self(unsafe { _mm_cmpngt_ps(self.0, rhs.0) })
  }

  /// Lane 0: `!(self > rhs)`, bool-ish output.
  #[inline(always)]
  #[must_use]
  pub fn cmp_ngt0(self, rhs: Self) -> Self {
    Self(unsafe { _mm_cmpngt_ss(self.0, rhs.0) })
  }

  /// Lanewise `!(self <= rhs)` check, bool-ish output.
  #[inline(always)]
  #[must_use]
  pub fn cmp_nle(self, rhs: Self) -> Self {
    Self(unsafe { _mm_cmpnle_ps(self.0, rhs.0) })
  }

  /// Lane 0: `!(self <= rhs)`, bool-ish output.
  #[inline(always)]
  #[must_use]
  pub fn cmp_nle0(self, rhs: Self) -> Self {
    Self(unsafe { _mm_cmpnle_ss(self.0, rhs.0) })
  }

  /// Lanewise `!(self < rhs)` check, bool-ish output.
  #[inline(always)]
  #[must_use]
  pub fn cmp_nlt(self, rhs: Self) -> Self {
    Self(unsafe { _mm_cmpnlt_ps(self.0, rhs.0) })
  }

  /// Lane 0: `!(self < rhs)`, bool-ish output.
  #[inline(always)]
  #[must_use]
  pub fn cmp_nlt0(self, rhs: Self) -> Self {
    Self(unsafe { _mm_cmpnlt_ss(self.0, rhs.0) })
  }

  /// Lanewise `self.not_nan() & rhs.not_nan()` check, bool-ish output.
  #[inline(always)]
  #[must_use]
  pub fn cmp_ordinary(self, rhs: Self) -> Self {
    Self(unsafe { _mm_cmpord_ps(self.0, rhs.0) })
  }

  /// Lane 0: `self.not_nan() & rhs.not_nan()`, bool-ish output.
  #[inline(always)]
  #[must_use]
  pub fn cmp_ordinary0(self, rhs: Self) -> Self {
    Self(unsafe { _mm_cmpord_ss(self.0, rhs.0) })
  }

  /// Lanewise `self.is_nan() | rhs.is_nan()` check, bool-ish output.
  #[inline(always)]
  #[must_use]
  pub fn cmp_nan(self, rhs: Self) -> Self {
    Self(unsafe { _mm_cmpunord_ps(self.0, rhs.0) })
  }

  /// Lane 0: `self.is_nan() | rhs.is_nan()`, bool-ish output.
  #[inline(always)]
  #[must_use]
  pub fn cmp_nan0(self, rhs: Self) -> Self {
    Self(unsafe { _mm_cmpunord_ss(self.0, rhs.0) })
  }

  /// Lane 0: `self == rhs`, 0 or 1 `i32` output.
  #[inline(always)]
  #[must_use]
  pub fn cmpi_eq0(self, rhs: Self) -> i32 {
    unsafe { _mm_comieq_ss(self.0, rhs.0) }
  }

  /// Lane 0: `self >= rhs`, 0 or 1 `i32` output.
  #[inline(always)]
  #[must_use]
  pub fn cmpi_ge0(self, rhs: Self) -> i32 {
    unsafe { _mm_comige_ss(self.0, rhs.0) }
  }

  /// Lane 0: `self > rhs`, 0 or 1 `i32` output.
  #[inline(always)]
  #[must_use]
  pub fn cmpi_gt0(self, rhs: Self) -> i32 {
    unsafe { _mm_comigt_ss(self.0, rhs.0) }
  }

  /// Lane 0: `self <= rhs`, 0 or 1 `i32` output.
  #[inline(always)]
  #[must_use]
  pub fn cmpi_le0(self, rhs: Self) -> i32 {
    unsafe { _mm_comile_ss(self.0, rhs.0) }
  }

  /// Lane 0: `self < rhs`, 0 or 1 `i32` output.
  #[inline(always)]
  #[must_use]
  pub fn cmpi_lt0(self, rhs: Self) -> i32 {
    unsafe { _mm_comilt_ss(self.0, rhs.0) }
  }

  /// Lane 0: `self != rhs`, 0 or 1 `i32` output.
  #[inline(always)]
  #[must_use]
  pub fn cmpi_ne0(self, rhs: Self) -> i32 {
    unsafe { _mm_comineq_ss(self.0, rhs.0) }
  }

  /// Round the `i32` to `f32` and replace lane 0.
  ///
  /// Subject to the current thread's [rounding
  /// mode](https://doc.rust-lang.org/core/arch/x86_64/fn._mm_setcsr.html#rounding-mode)
  #[inline(always)]
  #[must_use]
  pub fn round_replace0_i32(self, rhs: i32) -> Self {
    Self(unsafe { _mm_cvt_si2ss(self.0, rhs) })
  }

  /// Round lane 0 to `i32` and return.
  ///
  /// Subject to the current thread's [rounding
  /// mode](https://doc.rust-lang.org/core/arch/x86_64/fn._mm_setcsr.html#rounding-mode)
  #[inline(always)]
  #[must_use]
  pub fn round_extract0_i32(self) -> i32 {
    unsafe { _mm_cvt_ss2si(self.0) }
  }

  /// Round the `i64` to `f32` and replace lane 0.
  ///
  /// Subject to the current thread's [rounding
  /// mode](https://doc.rust-lang.org/core/arch/x86_64/fn._mm_setcsr.html#rounding-mode)
  ///
  /// Not available to `x86`
  #[inline(always)]
  #[cfg(target_arch = "x86_64")]
  #[must_use]
  pub fn round_replace0_i64(self, rhs: i64) -> Self {
    Self(unsafe { _mm_cvtsi64_ss(self.0, rhs) })
  }

  /// Directly extracts lane 0 as `f32`.
  #[inline(always)]
  #[must_use]
  pub fn extract0(self) -> f32 {
    unsafe { _mm_cvtss_f32(self.0) }
  }

  /// Round lane 0 to `i64` and return.
  ///
  /// Subject to the current thread's [rounding
  /// mode](https://doc.rust-lang.org/core/arch/x86_64/fn._mm_setcsr.html#rounding-mode)
  #[inline(always)]
  #[cfg(target_arch = "x86_64")]
  #[must_use]
  pub fn round_extract0_i64(self) -> i64 {
    unsafe { _mm_cvtss_si64(self.0) }
  }

  /// Truncate lane 0 to `i32` and return.
  #[inline(always)]
  #[must_use]
  pub fn truncate_extract0_i32(self) -> i32 {
    unsafe { _mm_cvtt_ss2si(self.0) }
  }

  /// Truncate lane 0 to `i64` and return.
  #[inline(always)]
  #[must_use]
  #[cfg(target_arch = "x86_64")]
  pub fn truncate_extract0_i64(self) -> i64 {
    unsafe { _mm_cvttss_si64(self.0) }
  }

  /// Divides the 0th lanes without affecting the other lanes of `self.
  #[inline(always)]
  #[must_use]
  pub fn div0(self, rhs: Self) -> Self {
    Self(unsafe { _mm_div_ss(self.0, rhs.0) })
  }

  /// Loads a 16-byte aligned `f32` array address into an `m128`.
  ///
  /// This produces the same lane order as you'd get if you de-referenced the
  /// pointed to array and then used `transmute`.
  #[inline(always)]
  #[must_use]
  pub fn load(addr: &Align16<[f32; 4]>) -> Self {
    let ptr: *const f32 = addr as *const Align16<[f32; 4]> as *const f32;
    Self(unsafe { _mm_load_ps(ptr) })
  }

  /// Loads the `f32` address into all lanes.
  #[allow(clippy::trivially_copy_pass_by_ref)]
  #[inline(always)]
  #[must_use]
  pub fn load_splat(addr: &f32) -> Self {
    Self(unsafe { _mm_load_ps1(addr) })
  }

  /// Loads the `f32` address into lane 0, other lanes are `0.0`.
  #[allow(clippy::trivially_copy_pass_by_ref)]
  #[inline(always)]
  #[must_use]
  pub fn load0(addr: &f32) -> Self {
    Self(unsafe { _mm_load_ss(addr) })
  }

  /// Loads 16-byte aligned `f32`s into an `m128`.
  ///
  /// This produces the **reverse** lane order as you'd get if you used a
  /// `transmute` on the pointed to array.
  #[inline(always)]
  #[must_use]
  pub fn load_reverse(addr: &Align16<[f32; 4]>) -> Self {
    let ptr: *const f32 = addr as *const Align16<[f32; 4]> as *const f32;
    Self(unsafe { _mm_loadr_ps(ptr) })
  }

  /// Loads 16-byte `f32`s into an `m128`.
  ///
  /// This doesn't have the alignment requirements of [`load`](m128::load), but
  /// the lane ordering is the same.
  #[inline(always)]
  #[must_use]
  pub fn load_unaligned(addr: &[f32; 4]) -> Self {
    let ptr: *const f32 = addr as *const [f32; 4] as *const f32;
    Self(unsafe { _mm_loadu_ps(ptr) })
  }

  /// Lanewise maximum.
  #[inline(always)]
  #[must_use]
  pub fn max(self, rhs: Self) -> Self {
    Self(unsafe { _mm_max_ps(self.0, rhs.0) })
  }

  /// Lane 0 maximum, other lanes are `self`.
  #[inline(always)]
  #[must_use]
  pub fn max0(self, rhs: Self) -> Self {
    Self(unsafe { _mm_max_ss(self.0, rhs.0) })
  }

  /// Lanewise minimum.
  #[inline(always)]
  #[must_use]
  pub fn min(self, rhs: Self) -> Self {
    Self(unsafe { _mm_min_ps(self.0, rhs.0) })
  }

  /// Lane 0 minimum, other lanes are `self`.
  #[inline(always)]
  #[must_use]
  pub fn min0(self, rhs: Self) -> Self {
    Self(unsafe { _mm_min_ss(self.0, rhs.0) })
  }

  /// Copies lane 0 from `rhs`, other lanes are `self`.
  #[inline(always)]
  #[must_use]
  pub fn copy0(self, rhs: Self) -> Self {
    Self(unsafe { _mm_move_ss(self.0, rhs.0) })
  }

  /// Copy the high two lanes of `rhs` over top of the low two lanes of `self`,
  /// other lanes unchanged.
  ///
  /// ```txt
  /// out[0] = rhs[2]
  /// out[1] = rhs[3]
  /// out[2] = self[2]
  /// out[3] = self[3]
  /// ```
  #[inline(always)]
  #[must_use]
  pub fn copy_high_low(self, rhs: Self) -> Self {
    Self(unsafe { _mm_movehl_ps(self.0, rhs.0) })
  }

  /// Copy the low two lanes of `rhs` over top of the high two lanes of `self`,
  /// other lanes unchanged.
  ///
  /// ```txt
  /// out[0] = self[0]
  /// out[1] = self[1]
  /// out[2] = rhs[0]
  /// out[3] = rhs[1]
  /// ```
  #[inline(always)]
  #[must_use]
  pub fn copy_low_high(self, rhs: Self) -> Self {
    Self(unsafe { _mm_movelh_ps(self.0, rhs.0) })
  }

  /// Assumes that this is a bool-ish mask and packs it into an `i32`.
  ///
  /// Specifically, the output `i32` has bits 0/1/2/3 set to be the same as the
  /// most significant bit in lanes 0/1/2/3 of `self`.
  ///
  /// (Yes, this name is kinda stupid but I couldn't come up with a better thing
  /// to rename it to, oh well.)
  #[inline(always)]
  #[must_use]
  pub fn move_mask(self) -> i32 {
    unsafe { _mm_movemask_ps(self.0) }
  }

  /// Lanewise approximate reciprocal.
  ///
  /// The maximum relative error for this approximation is less than
  /// 1.5*2.0e-12.
  #[inline(always)]
  #[must_use]
  pub fn reciprocal(self) -> Self {
    Self(unsafe { _mm_rcp_ps(self.0) })
  }

  /// Lane 0 approximate reciprocal, other lanes are `self`.
  ///
  /// The maximum relative error for this approximation is less than
  /// 1.5*2.0e-12.
  #[inline(always)]
  #[must_use]
  pub fn reciprocal0(self) -> Self {
    Self(unsafe { _mm_rcp_ss(self.0) })
  }

  /// Lanewise approximate reciprocal of the square root.
  ///
  /// The maximum relative error for this approximation is less than
  /// 1.5*2.0e-12.
  #[inline(always)]
  #[must_use]
  pub fn reciprocal_sqrt(self) -> Self {
    Self(unsafe { _mm_rsqrt_ps(self.0) })
  }

  /// Lane 0 approximate reciprocal of the square root, other lanes are `self`.
  ///
  /// The maximum relative error for this approximation is less than
  /// 1.5*2.0e-12.
  #[inline(always)]
  #[must_use]
  pub fn reciprocal_sqrt0(self) -> Self {
    Self(unsafe { _mm_rsqrt_ss(self.0) })
  }

  /// Set four `f32` values into an `m128`.
  ///
  /// Because of how little-endian works, this produces the **opposite** lane
  /// order as you'd get compared to putting the arguments in to an array and
  /// then using [`load`](m128::load) on that array. Same with using `transmute`
  /// or similar.
  #[inline(always)]
  #[must_use]
  pub fn set(a: f32, b: f32, c: f32, d: f32) -> Self {
    Self(unsafe { _mm_set_ps(a, b, c, d) })
  }

  /// Set the `f32` into all lanes.
  #[inline(always)]
  #[must_use]
  pub fn splat(a: f32) -> Self {
    Self(unsafe { _mm_set1_ps(a) })
  }

  /// Set the value into lane 0, other lanes `0.0`.
  #[inline(always)]
  #[must_use]
  pub fn set0(a: f32) -> Self {
    Self(unsafe { _mm_set_ss(a) })
  }

  /// Set four `f32` values into an `m128`, order reversed from normal
  /// [`set`](m128::set).
  #[inline(always)]
  #[must_use]
  pub fn set_reverse(a: f32, b: f32, c: f32, d: f32) -> Self {
    Self(unsafe { _mm_setr_ps(a, b, c, d) })
  }

  /// Lanewise square root.
  #[inline(always)]
  #[must_use]
  pub fn sqrt(self) -> Self {
    Self(unsafe { _mm_sqrt_ps(self.0) })
  }

  /// Lane 0 square root, other lanes are `self`.
  #[inline(always)]
  #[must_use]
  pub fn sqrt0(self) -> Self {
    Self(unsafe { _mm_sqrt_ss(self.0) })
  }

  /// Stores an `m128` into a 16-byte aligned `f32` array address.
  ///
  /// This uses the same lane order as [`load`](m128::load).
  #[inline(always)]
  pub fn store(self, addr: &mut Align16<[f32; 4]>) {
    let ptr: *mut f32 = addr as *mut Align16<[f32; 4]> as *mut f32;
    unsafe { _mm_store_ps(ptr, self.0) }
  }

  /// Stores lane 0 to all indexes of the array.
  #[inline(always)]
  pub fn store0_all(self, addr: &mut Align16<[f32; 4]>) {
    let ptr: *mut f32 = addr as *mut Align16<[f32; 4]> as *mut f32;
    unsafe { _mm_store_ps1(ptr, self.0) }
  }

  /// Stores lane 0 to the address given.
  #[inline(always)]
  pub fn store0(self, addr: &mut f32) {
    unsafe { _mm_store_ss(addr, self.0) }
  }

  /// Stores an `m128` into a 16-byte aligned `f32` array address.
  ///
  /// This uses the same lane order as [`load_reverse`](m128::load_reverse).
  #[inline(always)]
  pub fn store_reverse(self, addr: &mut Align16<[f32; 4]>) {
    let ptr: *mut f32 = addr as *mut Align16<[f32; 4]> as *mut f32;
    unsafe { _mm_storer_ps(ptr, self.0) }
  }

  /// Stores an `m128` into a `f32` array address.
  ///
  /// This doesn't have the alignment requirements of [`store`](m128::store),
  /// but the lane ordering is the same.
  #[inline(always)]
  pub fn store_unaligned(self, addr: &mut [f32; 4]) {
    let ptr: *mut f32 = addr as *mut [f32; 4] as *mut f32;
    unsafe { _mm_storeu_ps(ptr, self.0) }
  }

  /// Subtracts the 0th lanes without affecting the other lanes of `self.
  #[inline(always)]
  #[must_use]
  pub fn sub0(self, rhs: Self) -> Self {
    Self(unsafe { _mm_sub_ss(self.0, rhs.0) })
  }

  /// Unpack and interleave the high lanes of `self` and `rhs`.
  ///
  /// ```txt
  /// out[0] = self[2]
  /// out[1] = rhs[2]
  /// out[2] = self[3]
  /// out[3] = rhs[3]
  /// ```
  #[inline(always)]
  #[must_use]
  pub fn unpack_high(self, rhs: Self) -> Self {
    Self(unsafe { _mm_unpackhi_ps(self.0, rhs.0) })
  }

  /// Unpack and interleave the low lanes of `self` and `rhs`.
  ///
  /// ```txt
  /// out[0] = self[0]
  /// out[1] = rhs[0]
  /// out[2] = self[1]
  /// out[3] = rhs[1]
  /// ```
  #[inline(always)]
  #[must_use]
  pub fn unpack_low(self, rhs: Self) -> Self {
    Self(unsafe { _mm_unpacklo_ps(self.0, rhs.0) })
  }
}

/// Prefetch the cache line into all cache levels.
///
/// A prefetch is just a hint to the CPU and has no effect on the correctness
/// (or not) of a program. In other words, you can prefetch literally any
/// address and it's never UB. However, if you prefetch an invalid address the
/// CPU can actually slow down for a moment as it figures out that your address
/// isn't valid. So, don't go silly with this.
///
/// See Also: [`_mm_prefetch`](core::arch::x86_64::_mm_prefetch)
#[inline(always)]
pub fn prefetch0(ptr: *const impl Sized) {
  unsafe { _mm_prefetch(ptr as *const i8, _MM_HINT_T0) }
}

/// Prefetch the cache line into L2 and higher.
///
/// A prefetch is just a hint to the CPU and has no effect on the correctness
/// (or not) of a program. In other words, you can prefetch literally any
/// address and it's never UB. However, if you prefetch an invalid address the
/// CPU can actually slow down for a moment as it figures out that your address
/// isn't valid. So, don't go silly with this.
///
/// See Also: [`_mm_prefetch`](core::arch::x86_64::_mm_prefetch)
#[inline(always)]
pub fn prefetch1(ptr: *const impl Sized) {
  unsafe { _mm_prefetch(ptr as *const i8, _MM_HINT_T1) }
}

/// Prefetch the cache line into L3 and higher (or best effort).
///
/// A prefetch is just a hint to the CPU and has no effect on the correctness
/// (or not) of a program. In other words, you can prefetch literally any
/// address and it's never UB. However, if you prefetch an invalid address the
/// CPU can actually slow down for a moment as it figures out that your address
/// isn't valid. So, don't go silly with this.
///
/// See Also: [`_mm_prefetch`](core::arch::x86_64::_mm_prefetch)
#[inline(always)]
pub fn prefetch2(ptr: *const impl Sized) {
  unsafe { _mm_prefetch(ptr as *const i8, _MM_HINT_T2) }
}

/// Prefetch with non-temporal hint.
///
/// Non-temporal access is inherently spooky with respect to the rest of the
/// memory model. When I asked a member of the Rust Language Team how they felt
/// about non-temporal access, they simply replied with [the confounded
/// emoji]https://emojipedia.org/confounded-face/). I don't expose actual
/// non-temporal store/load methods as safe operations, but a non-temporal
/// _prefetch_ is still fine to do.
#[inline(always)]
pub fn prefetch_nta(ptr: *const impl Sized) {
  unsafe { _mm_prefetch(ptr as *const i8, _MM_HINT_NTA) }
}

/// Transposes, in place, the four `m128` values as if they formed a 4x4 Matrix.
///
/// The Intel guide lists the official implementation of this as being:
/// ```txt
/// __m128 tmp3, tmp2, tmp1, tmp0;
/// tmp0 := _mm_unpacklo_ps(row0, row1);
/// tmp2 := _mm_unpacklo_ps(row2, row3);
/// tmp1 := _mm_unpackhi_ps(row0, row1);
/// tmp3 := _mm_unpackhi_ps(row2, row3);
/// row0 := _mm_movelh_ps(tmp0, tmp2);
/// row1 := _mm_movehl_ps(tmp2, tmp0);
/// row2 := _mm_movelh_ps(tmp1, tmp3);
/// row3 := _mm_movehl_ps(tmp3, tmp1);
/// ```
#[inline(always)]
pub fn transpose4(r0: &mut m128, r1: &mut m128, r2: &mut m128, r3: &mut m128) {
  unsafe { _MM_TRANSPOSE4_PS(&mut r0.0, &mut r1.0, &mut r2.0, &mut r3.0) }
}

/// Shuffles around some `f32` lanes into a new `m128`
///
/// This is a macro and not a function because the shuffle pattern must be a
/// compile time constant. The macro takes some requested indexes and then makes
/// the correct shuffle pattern constant for you.
///
/// * `$a` and `$b` are any `m128` expressions.
/// * `$i0a`, `$i1a`, `$i2b`, and `$i3b` must each be `0`, `1`, `2`, or `3`.
///   Technically any `u32` literal will work, but only the lowest two bits are
///   used so stick to `0`, `1`, `2`, or `3`.
/// * Each lane in the output uses one of the lanes from an input. Like the
///   names hint, indexes 0 and 1 will come from somewhere in `$a`, and indexes
///   2 and 3 will come from somewhere in `$b`.
///
/// ```txt
/// shuffle128!(a, b, [0, 2, 1, 3])
/// ```
///
/// Would give an output of: `a[0], a[2], b[1], b[3]`
#[macro_export]
macro_rules! shuffle128 {
  ($a:expr, $b:expr, [$i0a:literal,$i1a:literal,$i2b:literal,$i3b:literal]) => {{
    // Keep only 2 bits per index
    const I0A: u32 = $i0a & 0b11;
    const I1A: u32 = $i1a & 0b11;
    const I2B: u32 = $i2b & 0b11;
    const I3B: u32 = $i3b & 0b11;
    // pack it up little-endian
    const IMM: i32 = (I0A | I1A << 2 | I2B << 4 | I3B << 6) as i32;
    //
    #[cfg(target_arch = "x86")]
    use core::arch::x86::_mm_shuffle_ps;
    #[cfg(target_arch = "x86")]
    use $crate::arch::x86::m128;
    //
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::_mm_shuffle_ps;
    #[cfg(target_arch = "x86_64")]
    use $crate::arch::x86_64::m128;
    //
    m128(unsafe { _mm_shuffle_ps($a.0, $b.0, IMM) })
  }};
}

//
// EXTRA FUNCTIONS THAT COMBINE INTRINSICS TO MAKE A USEFUL OP
//

/// # SSE Extras
impl m128 {
  /// `[non-intrinsic]` Lanewise absolute value.
  ///
  /// This is not an official Intel intrinsic, instead it's a `bitand` operation
  /// with a mask so that the sign bit is cleared in all lanes.
  #[inline(always)]
  #[must_use]
  pub fn abs(self) -> Self {
    self & Self::splat(cast(i32::max_value()))
  }
}
