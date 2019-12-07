#![cfg(target_feature = "sse")]
#![allow(bad_style)]
#![allow(clippy::unreadable_literal)]

#[path = "intel-simd-help.rs"]
mod intel_simd_help;
use intel_simd_help::*;

#[test]
fn m128i_debug() {
  let m: m128i = cast(core::u128::MAX);
  let max = core::u128::MAX;
  assert_eq!(&format!("{:?}", m), "m128i(-1)");
  assert_eq!(&format!("{:#?}", m), &format!("m128i({})", max));
  assert_eq!(&format!("{:1?}", m), "m128i(-1)");
  assert_eq!(&format!("{:#1?}", m), &format!("m128i({})", max));
  let max = core::u64::MAX;
  assert_eq!(&format!("{:2?}", m), "m128i(-1, -1)");
  assert_eq!(&format!("{:#2?}", m), &format!("m128i({}, {})", max, max));
  let max = core::u32::MAX;
  assert_eq!(&format!("{:4?}", m), "m128i(-1, -1, -1, -1)");
  assert_eq!(
    &format!("{:#4?}", m),
    &format!("m128i({}, {}, {}, {})", max, max, max, max)
  );
  let max = core::u16::MAX;
  assert_eq!(&format!("{:8?}", m), "m128i(-1, -1, -1, -1, -1, -1, -1, -1)");
  assert_eq!(
    &format!("{:#8?}", m),
    &format!(
      "m128i({}, {}, {}, {}, {}, {}, {}, {})",
      max, max, max, max, max, max, max, max
    )
  );
  let max = core::u8::MAX;
  assert_eq!(
    &format!("{:16?}", m),
    "m128i(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1)"
  );
  assert_eq!(
    &format!("{:#16?}", m),
    &format!(
      "m128i({}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {})",
      max,
      max,
      max,
      max,
      max,
      max,
      max,
      max,
      max,
      max,
      max,
      max,
      max,
      max,
      max,
      max
    )
  );
}

#[test]
fn m128i_display() {
  let m: m128i = cast(core::u128::MAX);
  let max = core::u128::MAX;
  assert_eq!(&format!("{:}", m), "m128i(-1)");
  assert_eq!(&format!("{:#}", m), &format!("m128i({})", max));
  assert_eq!(&format!("{:1}", m), "m128i(-1)");
  assert_eq!(&format!("{:#1}", m), &format!("m128i({})", max));
  let max = core::u64::MAX;
  assert_eq!(&format!("{:2}", m), "m128i(-1, -1)");
  assert_eq!(&format!("{:#2}", m), &format!("m128i({}, {})", max, max));
  let max = core::u32::MAX;
  assert_eq!(&format!("{:4}", m), "m128i(-1, -1, -1, -1)");
  assert_eq!(
    &format!("{:#4}", m),
    &format!("m128i({}, {}, {}, {})", max, max, max, max)
  );
  let max = core::u16::MAX;
  assert_eq!(&format!("{:8}", m), "m128i(-1, -1, -1, -1, -1, -1, -1, -1)");
  assert_eq!(
    &format!("{:#8}", m),
    &format!(
      "m128i({}, {}, {}, {}, {}, {}, {}, {})",
      max, max, max, max, max, max, max, max
    )
  );
  let max = core::u8::MAX;
  assert_eq!(
    &format!("{:16}", m),
    "m128i(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1)"
  );
  assert_eq!(
    &format!("{:#16}", m),
    &format!(
      "m128i({}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {})",
      max,
      max,
      max,
      max,
      max,
      max,
      max,
      max,
      max,
      max,
      max,
      max,
      max,
      max,
      max,
      max
    )
  );
}

#[test]
fn m128i_binary() {
  let m: m128i = cast(core::u128::MAX);
  let max = core::u128::MAX;
  assert_eq!(&format!("{:b}", m), &format!("m128i({:b})", max));
  assert_eq!(&format!("{:#b}", m), &format!("m128i({:#b})", max));
  assert_eq!(&format!("{:1b}", m), &format!("m128i({:b})", max));
  assert_eq!(&format!("{:#1b}", m), &format!("m128i({:#b})", max));
  let max = core::u64::MAX;
  assert_eq!(&format!("{:2b}", m), &format!("m128i({:b}, {:b})", max, max));
  assert_eq!(&format!("{:#2b}", m), &format!("m128i({:#b}, {:#b})", max, max));
  let max = core::u32::MAX;
  assert_eq!(
    &format!("{:4b}", m),
    &format!("m128i({:b}, {:b}, {:b}, {:b})", max, max, max, max)
  );
  assert_eq!(
    &format!("{:#4b}", m),
    &format!("m128i({:#b}, {:#b}, {:#b}, {:#b})", max, max, max, max)
  );
  let max = core::u16::MAX;
  assert_eq!(
    &format!("{:8b}", m),
    &format!(
      "m128i({:b}, {:b}, {:b}, {:b}, {:b}, {:b}, {:b}, {:b})",
      max, max, max, max, max, max, max, max
    )
  );
  assert_eq!(
    &format!("{:#8b}", m),
    &format!(
      "m128i({:#b}, {:#b}, {:#b}, {:#b}, {:#b}, {:#b}, {:#b}, {:#b})",
      max, max, max, max, max, max, max, max
    )
  );
  let max = core::u8::MAX;
  assert_eq!(
    &format!("{:16b}", m),
    &format!(
      "m128i({:b}, {:b}, {:b}, {:b}, {:b}, {:b}, {:b}, {:b}, {:b}, {:b}, {:b}, {:b}, {:b}, {:b}, {:b}, {:b})",
      max, max, max, max, max, max, max, max, max, max, max, max, max, max, max, max
    )
  );
  assert_eq!(
    &format!("{:#16b}", m),
    &format!(
      "m128i({:#b}, {:#b}, {:#b}, {:#b}, {:#b}, {:#b}, {:#b}, {:#b}, {:#b}, {:#b}, {:#b}, {:#b}, {:#b}, {:#b}, {:#b}, {:#b})",
      max, max, max, max, max, max, max, max, max, max, max, max, max, max, max, max
    )
  );
}

#[test]
fn m128i_lower_hex() {
  let m: m128i = cast(core::u128::MAX);
  let max = core::u128::MAX;
  assert_eq!(&format!("{:x}", m), &format!("m128i({:x})", max));
  assert_eq!(&format!("{:#x}", m), &format!("m128i({:#x})", max));
  assert_eq!(&format!("{:1x}", m), &format!("m128i({:x})", max));
  assert_eq!(&format!("{:#1x}", m), &format!("m128i({:#x})", max));
  let max = core::u64::MAX;
  assert_eq!(&format!("{:2x}", m), &format!("m128i({:x}, {:x})", max, max));
  assert_eq!(&format!("{:#2x}", m), &format!("m128i({:#x}, {:#x})", max, max));
  let max = core::u32::MAX;
  assert_eq!(
    &format!("{:4x}", m),
    &format!("m128i({:x}, {:x}, {:x}, {:x})", max, max, max, max)
  );
  assert_eq!(
    &format!("{:#4x}", m),
    &format!("m128i({:#x}, {:#x}, {:#x}, {:#x})", max, max, max, max)
  );
  let max = core::u16::MAX;
  assert_eq!(
    &format!("{:8x}", m),
    &format!(
      "m128i({:x}, {:x}, {:x}, {:x}, {:x}, {:x}, {:x}, {:x})",
      max, max, max, max, max, max, max, max
    )
  );
  assert_eq!(
    &format!("{:#8x}", m),
    &format!(
      "m128i({:#x}, {:#x}, {:#x}, {:#x}, {:#x}, {:#x}, {:#x}, {:#x})",
      max, max, max, max, max, max, max, max
    )
  );
  let max = core::u8::MAX;
  assert_eq!(
    &format!("{:16x}", m),
    &format!(
      "m128i({:x}, {:x}, {:x}, {:x}, {:x}, {:x}, {:x}, {:x}, {:x}, {:x}, {:x}, {:x}, {:x}, {:x}, {:x}, {:x})",
      max, max, max, max, max, max, max, max, max, max, max, max, max, max, max, max
    )
  );
  assert_eq!(
    &format!("{:#16x}", m),
    &format!(
      "m128i({:#x}, {:#x}, {:#x}, {:#x}, {:#x}, {:#x}, {:#x}, {:#x}, {:#x}, {:#x}, {:#x}, {:#x}, {:#x}, {:#x}, {:#x}, {:#x})",
      max, max, max, max, max, max, max, max, max, max, max, max, max, max, max, max
    )
  );
}

#[test]
fn m128i_octal() {
  let m: m128i = cast(core::u128::MAX);
  let max = core::u128::MAX;
  assert_eq!(&format!("{:o}", m), &format!("m128i({:o})", max));
  assert_eq!(&format!("{:#o}", m), &format!("m128i({:#o})", max));
  assert_eq!(&format!("{:1o}", m), &format!("m128i({:o})", max));
  assert_eq!(&format!("{:#1o}", m), &format!("m128i({:#o})", max));
  let max = core::u64::MAX;
  assert_eq!(&format!("{:2o}", m), &format!("m128i({:o}, {:o})", max, max));
  assert_eq!(&format!("{:#2o}", m), &format!("m128i({:#o}, {:#o})", max, max));
  let max = core::u32::MAX;
  assert_eq!(
    &format!("{:4o}", m),
    &format!("m128i({:o}, {:o}, {:o}, {:o})", max, max, max, max)
  );
  assert_eq!(
    &format!("{:#4o}", m),
    &format!("m128i({:#o}, {:#o}, {:#o}, {:#o})", max, max, max, max)
  );
  let max = core::u16::MAX;
  assert_eq!(
    &format!("{:8o}", m),
    &format!(
      "m128i({:o}, {:o}, {:o}, {:o}, {:o}, {:o}, {:o}, {:o})",
      max, max, max, max, max, max, max, max
    )
  );
  assert_eq!(
    &format!("{:#8o}", m),
    &format!(
      "m128i({:#o}, {:#o}, {:#o}, {:#o}, {:#o}, {:#o}, {:#o}, {:#o})",
      max, max, max, max, max, max, max, max
    )
  );
  let max = core::u8::MAX;
  assert_eq!(
    &format!("{:16o}", m),
    &format!(
      "m128i({:o}, {:o}, {:o}, {:o}, {:o}, {:o}, {:o}, {:o}, {:o}, {:o}, {:o}, {:o}, {:o}, {:o}, {:o}, {:o})",
      max, max, max, max, max, max, max, max, max, max, max, max, max, max, max, max
    )
  );
  assert_eq!(
    &format!("{:#16o}", m),
    &format!(
      "m128i({:#o}, {:#o}, {:#o}, {:#o}, {:#o}, {:#o}, {:#o}, {:#o}, {:#o}, {:#o}, {:#o}, {:#o}, {:#o}, {:#o}, {:#o}, {:#o})",
      max, max, max, max, max, max, max, max, max, max, max, max, max, max, max, max
    )
  );
}

#[test]
fn m128i_upper_hex() {
  let m: m128i = cast(core::u128::MAX);
  let max = core::u128::MAX;
  assert_eq!(&format!("{:X}", m), &format!("m128i({:X})", max));
  assert_eq!(&format!("{:#X}", m), &format!("m128i({:#X})", max));
  assert_eq!(&format!("{:1X}", m), &format!("m128i({:X})", max));
  assert_eq!(&format!("{:#1X}", m), &format!("m128i({:#X})", max));
  let max = core::u64::MAX;
  assert_eq!(&format!("{:2X}", m), &format!("m128i({:X}, {:X})", max, max));
  assert_eq!(&format!("{:#2X}", m), &format!("m128i({:#X}, {:#X})", max, max));
  let max = core::u32::MAX;
  assert_eq!(
    &format!("{:4X}", m),
    &format!("m128i({:X}, {:X}, {:X}, {:X})", max, max, max, max)
  );
  assert_eq!(
    &format!("{:#4X}", m),
    &format!("m128i({:#X}, {:#X}, {:#X}, {:#X})", max, max, max, max)
  );
  let max = core::u16::MAX;
  assert_eq!(
    &format!("{:8X}", m),
    &format!(
      "m128i({:X}, {:X}, {:X}, {:X}, {:X}, {:X}, {:X}, {:X})",
      max, max, max, max, max, max, max, max
    )
  );
  assert_eq!(
    &format!("{:#8X}", m),
    &format!(
      "m128i({:#X}, {:#X}, {:#X}, {:#X}, {:#X}, {:#X}, {:#X}, {:#X})",
      max, max, max, max, max, max, max, max
    )
  );
  let max = core::u8::MAX;
  assert_eq!(
    &format!("{:16X}", m),
    &format!(
      "m128i({:X}, {:X}, {:X}, {:X}, {:X}, {:X}, {:X}, {:X}, {:X}, {:X}, {:X}, {:X}, {:X}, {:X}, {:X}, {:X})",
      max, max, max, max, max, max, max, max, max, max, max, max, max, max, max, max
    )
  );
  assert_eq!(
    &format!("{:#16X}", m),
    &format!(
      "m128i({:#X}, {:#X}, {:#X}, {:#X}, {:#X}, {:#X}, {:#X}, {:#X}, {:#X}, {:#X}, {:#X}, {:#X}, {:#X}, {:#X}, {:#X}, {:#X})",
      max, max, max, max, max, max, max, max, max, max, max, max, max, max, max, max
    )
  );
}

#[test]
fn m128d_debug() {
  let m: m128d = cast([5.0_f64, 6.0]);
  assert_eq!(&format!("{:?}", m), "m128d(5.0, 6.0)");
}

#[test]
fn m128d_display() {
  let m: m128d = cast([5.0_f64, 6.0]);
  assert_eq!(&format!("{}", m), "m128d(5, 6)");
}

#[test]
fn m128d_lower_exp() {
  let m: m128d = cast([5.0_f64, 6.0]);
  assert_eq!(&format!("{:e}", m), "m128d(5e0, 6e0)");
}

#[test]
fn m128d_upper_exp() {
  let m: m128d = cast([5.0_f64, 6.0]);
  assert_eq!(&format!("{:E}", m), "m128d(5E0, 6E0)");
}

#[test]
fn m128_round_i32() {
  let m: m128 = cast([5.0_f32, 6.1, 7.9, 8.5]);
  let mi: m128i = m.round_i32();
  let mi_arr: [i32; 4] = cast(mi);
  assert_eq!(mi_arr, [5, 6, 8, 8]);
}

#[test]
fn m128_truncate_i32() {
  let m: m128 = cast([5.0_f32, 6.1, 7.9, 8.5]);
  let mi: m128i = m.truncate_i32();
  let mi_arr: [i32; 4] = cast(mi);
  assert_eq!(mi_arr, [5, 6, 7, 8]);
}

#[test]
fn m128_round_f64() {
  let m: m128 = cast([5.0_f32, 6.5, 7.9, 8.5]);
  let md: m128d = m.round_f64();
  let md_arr: [f64; 2] = cast(md);
  assert_eq!(md_arr, [5.0_f64, 6.5]);
}

#[test]
fn m128i_bitand() {
  let max = core::u32::MAX;
  let a: m128i = cast([0, max, max, 0]);
  let b: m128i = cast([max, 0, max, 0]);
  let out: [u32; 4] = cast(a & b);
  assert_eq!(out, [0, 0, max, 0]);
}

#[test]
fn m128i_bitand_assign() {
  let max = core::u32::MAX;
  let mut a: m128i = cast([0, max, max, 0]);
  let b: m128i = cast([max, 0, max, 0]);
  a &= b;
  let out: [u32; 4] = cast(a);
  assert_eq!(out, [0, 0, max, 0]);
}

#[test]
fn m128_cast_m128i() {
  let m: m128 = cast(12_345_678_u128);
  let mi: m128i = m.cast_m128i();
  let mi_bits: u128 = cast(mi);
  assert_eq!(mi_bits, 12_345_678_u128);
}

#[test]
fn m128d_add() {
  let a: m128d = cast([5.0_f64, 6.0]);
  let b: m128d = cast([-8.0_f64, 4.0]);
  let out: [f64; 2] = cast(a + b);
  assert_eq!(out, [-3.0, 10.0]);
}

#[test]
fn m128d_add_assign() {
  let mut a: m128d = cast([5.0_f64, 6.0]);
  let b: m128d = cast([-8.0_f64, 4.0]);
  a += b;
  let out: [f64; 2] = cast(a);
  assert_eq!(out, [-3.0, 10.0]);
}

#[test]
fn m128d_add0() {
  let a: m128d = cast([5.0_f64, 6.0]);
  let b: m128d = cast([-8.0_f64, 4.0]);
  let out: [f64; 2] = cast(a.add0(b));
  assert_eq!(out, [-3.0, 6.0]);
}

#[test]
fn m128d_bitand() {
  let a: m128d = cast([core::u64::MAX, 0]);
  let b: m128d = cast([core::u64::MAX, core::u64::MAX]);
  let out: [u64; 2] = cast(a & b);
  assert_eq!(out, [core::u64::MAX, 0]);
}

#[test]
fn m128d_bitand_assign() {
  let mut a: m128d = cast([core::u64::MAX, 0]);
  let b: m128d = cast([core::u64::MAX, core::u64::MAX]);
  a &= b;
  let out: [u64; 2] = cast(a);
  assert_eq!(out, [core::u64::MAX, 0]);
}

#[test]
fn m128d_andnot() {
  let a: m128d = cast([core::u64::MAX, 0]);
  let b: m128d = cast([core::u64::MAX, core::u64::MAX]);
  let out: [u64; 2] = cast(a.andnot(b));
  assert_eq!(out, [0, core::u64::MAX]);
}

#[test]
fn m128d_cast_m128i() {
  let m: m128d = cast(12_345_678_u128);
  let mi: m128i = m.cast_m128i();
  let mi_bits: u128 = cast(mi);
  assert_eq!(mi_bits, 12_345_678_u128);
}

#[test]
fn m128d_cmp_eq() {
  let max = core::u64::MAX;
  let a: m128d = cast([5.0, 6.0]);
  let b: m128d = cast([5.0, 7.0]);
  assert_eq!(cast::<m128d, [u64; 2]>(a.cmp_eq(b)), [max, 0]);
}

#[test]
fn m128d_cmp_eq0() {
  let max = core::u64::MAX;
  let a: m128d = cast([5.0, 6.0]);
  let b: m128d = cast([5.0, 7.0]);
  assert_eq!(cast::<m128d, [u64; 2]>(a.cmp_eq0(b)), [max, 6.0_f64.to_bits()]);
}

#[test]
fn m128d_cmp_ge() {
  let max = core::u64::MAX;
  let a: m128d = cast([5.0, 12.0]);
  let b: m128d = cast([5.0, 7.0]);
  assert_eq!(cast::<m128d, [u64; 2]>(a.cmp_ge(b)), [max, max]);
}

#[test]
fn m128d_cmp_ge0() {
  let max = core::u64::MAX;
  let a: m128d = cast([5.0, 6.0]);
  let b: m128d = cast([5.0, 7.0]);
  assert_eq!(cast::<m128d, [u64; 2]>(a.cmp_ge0(b)), [max, 6.0_f64.to_bits()]);
}

#[test]
fn m128d_cmp_gt() {
  let max = core::u64::MAX;
  let a: m128d = cast([5.0, 12.0]);
  let b: m128d = cast([5.0, 7.0]);
  assert_eq!(cast::<m128d, [u64; 2]>(a.cmp_gt(b)), [0, max]);
}

#[test]
fn m128d_cmp_gt0() {
  let a: m128d = cast([5.0, 6.0]);
  let b: m128d = cast([5.0, 7.0]);
  assert_eq!(cast::<m128d, [u64; 2]>(a.cmp_gt0(b)), [0, 6.0_f64.to_bits()]);
}

#[test]
fn m128d_cmp_le() {
  let max = core::u64::MAX;
  let a: m128d = cast([4.0, 7.0]);
  let b: m128d = cast([5.0, 7.0]);
  assert_eq!(cast::<m128d, [u64; 2]>(a.cmp_le(b)), [max, max]);
}

#[test]
fn m128d_cmp_le0() {
  let max = core::u64::MAX;
  let a: m128d = cast([4.0, 7.0]);
  let b: m128d = cast([5.0, 7.0]);
  assert_eq!(cast::<m128d, [u64; 2]>(a.cmp_le0(b)), [max, 7.0_f64.to_bits()]);
}

#[test]
fn m128d_cmp_lt() {
  let max = core::u64::MAX;
  let a: m128d = cast([4.0, 7.0]);
  let b: m128d = cast([5.0, 7.0]);
  assert_eq!(cast::<m128d, [u64; 2]>(a.cmp_lt(b)), [max, 0]);
}

#[test]
fn m128d_cmp_lt0() {
  let max = core::u64::MAX;
  let a: m128d = cast([4.0, 7.0]);
  let b: m128d = cast([5.0, 7.0]);
  assert_eq!(cast::<m128d, [u64; 2]>(a.cmp_lt0(b)), [max, 7.0_f64.to_bits()]);
}

#[test]
fn m128d_cmp_ne() {
  let max = core::u64::MAX;
  let a: m128d = cast([4.0, 7.0]);
  let b: m128d = cast([5.0, 7.0]);
  assert_eq!(cast::<m128d, [u64; 2]>(a.cmp_ne(b)), [max, 0]);
}

#[test]
fn m128d_cmp_ne0() {
  let max = core::u64::MAX;
  let a: m128d = cast([4.0, 7.0]);
  let b: m128d = cast([5.0, 7.0]);
  assert_eq!(cast::<m128d, [u64; 2]>(a.cmp_ne0(b)), [max, 7.0_f64.to_bits()]);
}

#[test]
fn m128d_cmp_nge() {
  let max = core::u64::MAX;
  let a: m128d = cast([4.0, 7.0]);
  let b: m128d = cast([5.0, 7.0]);
  assert_eq!(cast::<m128d, [u64; 2]>(a.cmp_nge(b)), [max, 0]);
}

#[test]
fn m128d_cmp_nge0() {
  let max = core::u64::MAX;
  let a: m128d = cast([4.0, 7.0]);
  let b: m128d = cast([5.0, 7.0]);
  assert_eq!(cast::<m128d, [u64; 2]>(a.cmp_nge0(b)), [max, 7.0_f64.to_bits()]);
}

#[test]
fn m128d_cmp_ngt() {
  let max = core::u64::MAX;
  let a: m128d = cast([4.0, 7.0]);
  let b: m128d = cast([5.0, 7.0]);
  assert_eq!(cast::<m128d, [u64; 2]>(a.cmp_ngt(b)), [max, max]);
}

#[test]
fn m128d_cmp_ngt0() {
  let max = core::u64::MAX;
  let a: m128d = cast([4.0, 7.0]);
  let b: m128d = cast([5.0, 7.0]);
  assert_eq!(cast::<m128d, [u64; 2]>(a.cmp_ngt0(b)), [max, 7.0_f64.to_bits()]);
}

#[test]
fn m128d_cmp_nle() {
  let a: m128d = cast([4.0, 7.0]);
  let b: m128d = cast([5.0, 7.0]);
  assert_eq!(cast::<m128d, [u64; 2]>(a.cmp_nle(b)), [0, 0]);
}

#[test]
fn m128d_cmp_nle0() {
  let a: m128d = cast([4.0, 7.0]);
  let b: m128d = cast([5.0, 7.0]);
  assert_eq!(cast::<m128d, [u64; 2]>(a.cmp_nle0(b)), [0, 7.0_f64.to_bits()]);
}

#[test]
fn m128d_cmp_nlt() {
  let max = core::u64::MAX;
  let a: m128d = cast([4.0, 7.0]);
  let b: m128d = cast([5.0, 7.0]);
  assert_eq!(cast::<m128d, [u64; 2]>(a.cmp_nlt(b)), [0, max]);
}

#[test]
fn m128d_cmp_nlt0() {
  let a: m128d = cast([4.0, 7.0]);
  let b: m128d = cast([5.0, 7.0]);
  assert_eq!(cast::<m128d, [u64; 2]>(a.cmp_nlt0(b)), [0, 7.0_f64.to_bits()]);
}

#[test]
fn m128d_cmp_ordinary() {
  let max = core::u64::MAX;
  let a: m128d = cast([4.0, 7.0]);
  let b: m128d = cast([5.0, 7.0]);
  assert_eq!(cast::<m128d, [u64; 2]>(a.cmp_ordinary(b)), [max, max]);
}

#[test]
fn m128d_cmp_ordinary0() {
  let max = core::u64::MAX;
  let a: m128d = cast([4.0, 7.0]);
  let b: m128d = cast([5.0, 7.0]);
  assert_eq!(
    cast::<m128d, [u64; 2]>(a.cmp_ordinary0(b)),
    [max, 7.0_f64.to_bits()]
  );
}

#[test]
fn m128d_cmp_nan() {
  let max = core::u64::MAX;
  let a: m128d = cast([4.0_f64.to_bits(), max]);
  let b: m128d = cast([5.0, 7.0]);
  assert_eq!(cast::<m128d, [u64; 2]>(a.cmp_nan(b)), [0, max]);
}

#[test]
fn m128d_cmp_nan0() {
  let max = core::u64::MAX;
  let a: m128d = cast([4.0, 7.0]);
  let b: m128d = cast([max, 7.0_f64.to_bits()]);
  assert_eq!(cast::<m128d, [u64; 2]>(a.cmp_nan0(b)), [max, 7.0_f64.to_bits()]);
}

#[test]
fn m128d_cmpi_eq0() {
  let a: m128d = cast([4.0, 7.0]);
  let b: m128d = cast([4.0, 5.0]);
  assert_eq!(a.cmpi_eq0(b), 1);
}

#[test]
fn m128d_cmpi_ge0() {
  let a: m128d = cast([4.0, 7.0]);
  let b: m128d = cast([4.0, 5.0]);
  assert_eq!(a.cmpi_ge0(b), 1);
}

#[test]
fn m128d_cmpi_gt0() {
  let a: m128d = cast([4.0, 7.0]);
  let b: m128d = cast([4.0, 5.0]);
  assert_eq!(a.cmpi_gt0(b), 0);
}

#[test]
fn m128d_cmpi_le0() {
  let a: m128d = cast([4.0, 7.0]);
  let b: m128d = cast([4.0, 5.0]);
  assert_eq!(a.cmpi_le0(b), 1);
}

#[test]
fn m128d_cmpi_lt0() {
  let a: m128d = cast([4.0, 7.0]);
  let b: m128d = cast([4.0, 5.0]);
  assert_eq!(a.cmpi_lt0(b), 0);
}

#[test]
fn m128d_cmpi_ne0() {
  let a: m128d = cast([4.0, 7.0]);
  let b: m128d = cast([4.0, 5.0]);
  assert_eq!(a.cmpi_ne0(b), 0);
}

#[test]
fn m128d_round_i32x4() {
  let a: m128d = cast([4.8, 7.1]);
  let r: m128i = a.round_i32x4();
  let r_i32s: [i32; 4] = cast(r);
  assert_eq!(r_i32s, [5, 7, 0, 0]);
}

#[test]
fn m128d_round_f32x4() {
  let a: m128d = cast([4.5, 7.1]);
  let r: m128 = a.round_f32x4();
  let r_f32s: [f32; 4] = cast(r);
  assert_eq!(r_f32s, [4.5, 7.1, 0.0, 0.0]);
}

#[test]
#[allow(clippy::float_cmp)]
fn m128d_extract0() {
  let a: m128d = cast([4.5, 7.1]);
  assert_eq!(a.extract0(), 4.5_f64);
}

#[test]
fn m128d_round_i32_extract0() {
  let a: m128d = cast([4.5, 7.1]);
  assert_eq!(a.round_i32_extract0(), 4_i32);
}

#[test]
#[cfg(target_arch = "x86_64")]
fn m128d_round_i64_extract0() {
  let a: m128d = cast([4.5, 7.1]);
  assert_eq!(a.round_i64_extract0(), 4_i64);
}

#[test]
fn m128d_replace0_with_i32() {
  let a: m128d = cast([4.5, 7.1]);
  let b: m128d = a.replace0_with_i32(20_i32);
  let c: [f64; 2] = cast(b);
  assert_eq!(c, [20.0, 7.1]);
}

#[test]
#[cfg(target_arch = "x86_64")]
fn m128d_replace0_with_i64() {
  let a: m128d = cast([4.5, 7.1]);
  let b: m128d = a.replace0_with_i64(20_i64);
  let c: [f64; 2] = cast(b);
  assert_eq!(c, [20.0, 7.1]);
}

#[test]
fn m128d_replace0_with_f32() {
  let a: m128d = cast([4.5, 7.1]);
  let b: m128d = a.replace0_with_f32(m128::set0(8.0));
  let c: [f64; 2] = cast(b);
  assert_eq!(c, [8.0, 7.1]);
}

#[test]
fn m128d_truncate_i32x4() {
  let a: m128d = cast([4.8, 7.1]);
  let r: m128i = a.truncate_i32x4();
  let r_i32s: [i32; 4] = cast(r);
  assert_eq!(r_i32s, [4, 7, 0, 0]);
}

#[test]
fn m128d_truncate0_i32() {
  let a: m128d = cast([4.5, 7.1]);
  assert_eq!(a.truncate0_i32(), 4_i32);
}

#[test]
#[cfg(target_arch = "x86_64")]
fn m128d_truncate0_i64() {
  let a: m128d = cast([4.5, 7.1]);
  assert_eq!(a.truncate0_i64(), 4_i64);
}

#[test]
fn m128d_div() {
  let a: m128d = cast([5.0_f64, 6.0]);
  let b: m128d = cast([-2.0_f64, 3.0]);
  let out: [f64; 2] = cast(a / b);
  assert_eq!(out, [-2.5, 2.0]);
}

#[test]
fn m128d_div_assign() {
  let mut a: m128d = cast([5.0_f64, 6.0]);
  let b: m128d = cast([-2.0_f64, 3.0]);
  a /= b;
  let out: [f64; 2] = cast(a);
  assert_eq!(out, [-2.5, 2.0]);
}

#[test]
fn m128d_div0() {
  let a: m128d = cast([5.0_f64, 6.0]);
  let b: m128d = cast([2.0_f64, 4.0]);
  let out: [f64; 2] = cast(a.div0(b));
  assert_eq!(out, [2.5, 6.0]);
}

#[test]
fn m128d_load() {
  let aligned_array: Align16<[f64; 2]> = Align16([5.0, 6.0]);
  let md: m128d = m128d::load(&aligned_array);
  let md_f64s: [f64; 2] = cast(md);
  assert_eq!(aligned_array.0, md_f64s);
}

#[test]
fn m128d_load_aligned_splat() {
  let aligned_f64: Align16<f64> = Align16(5.0_f64);
  let md: m128d = m128d::load_aligned_splat(&aligned_f64);
  let md_f64s: [f64; 2] = cast(md);
  assert_eq!(md_f64s, [5.0_f64, 5.0]);
}

#[test]
fn m128d_load0() {
  let f: f64 = 5.0;
  let md: m128d = m128d::load0(&f);
  let md_f64s: [f64; 2] = cast(md);
  assert_eq!(md_f64s, [5.0_f64, 0.0]);
}

#[test]
fn m128d_replace_high() {
  let md: m128d = cast([5.0_f64, 6.0]);
  let md2: m128d = md.replace_high(&7.0);
  let md2_f64s: [f64; 2] = cast(md2);
  assert_eq!(md2_f64s, [5.0_f64, 7.0]);
}

#[test]
fn m128d_replace_low() {
  let md: m128d = cast([5.0_f64, 6.0]);
  let md2: m128d = md.replace_low(&7.0);
  let md2_f64s: [f64; 2] = cast(md2);
  assert_eq!(md2_f64s, [7.0_f64, 6.0]);
}

#[test]
fn m128d_load_reverse() {
  let aligned_array: Align16<[f64; 2]> = Align16([5.0, 6.0]);
  let md: m128d = m128d::load_reverse(&aligned_array);
  let md_f64s: [f64; 2] = cast(md);
  assert_eq!([6.0_f64, 5.0], md_f64s);
}

#[test]
fn m128d_load_unaligned() {
  let array: [f64; 2] = [5.0, 6.0];
  let md: m128d = m128d::load_unaligned(&array);
  let md_f64s: [f64; 2] = cast(md);
  assert_eq!(array, md_f64s);
}

#[test]
fn m128d_max() {
  let a: m128d = cast([5.0_f64, 6.0]);
  let b: m128d = cast([-8.0_f64, 4.0]);
  let out: [f64; 2] = cast(a.max(b));
  assert_eq!(out, [5.0_f64, 6.0]);
}

#[test]
fn m128d_max0() {
  let a: m128d = cast([5.0_f64, 6.0]);
  let b: m128d = cast([-8.0_f64, 4.0]);
  let out: [f64; 2] = cast(a.max0(b));
  assert_eq!(out, [5.0_f64, 6.0]);
}

#[test]
fn m128d_min() {
  let a: m128d = cast([5.0_f64, 6.0]);
  let b: m128d = cast([-8.0_f64, 4.0]);
  let out: [f64; 2] = cast(a.min(b));
  assert_eq!(out, [-8.0_f64, 4.0]);
}

#[test]
fn m128d_min0() {
  let a: m128d = cast([5.0_f64, 6.0]);
  let b: m128d = cast([-8.0_f64, 4.0]);
  let out: [f64; 2] = cast(a.min0(b));
  assert_eq!(out, [-8.0_f64, 6.0]);
}

#[test]
fn m128d_copy0() {
  let a: m128d = cast([5.0_f64, 6.0]);
  let b: m128d = cast([-8.0_f64, 4.0]);
  let out: [f64; 2] = cast(a.copy0(b));
  assert_eq!(out, [-8.0_f64, 6.0]);
}

#[test]
fn m128d_move_mask() {
  let max = core::u64::MAX;
  for target_mask in 0..4 {
    let arr: [u64; 2] =
      [max * (target_mask & 0b1), max * ((target_mask & 0b10) >> 1)];
    let m: m128d = cast(arr);
    let out_mask: i32 = m.move_mask();
    assert_eq!(out_mask as u64, target_mask);
  }
}

#[test]
fn m128d_mul() {
  let a: m128d = cast([5.0_f64, 6.0]);
  let b: m128d = cast([-8.0_f64, 4.0]);
  let out: [f64; 2] = cast(a * b);
  assert_eq!(out, [-40.0, 24.0]);
}

#[test]
fn m128d_mul_assign() {
  let mut a: m128d = cast([5.0_f64, 6.0]);
  let b: m128d = cast([-8.0_f64, 4.0]);
  a *= b;
  let out: [f64; 2] = cast(a);
  assert_eq!(out, [-40.0, 24.0]);
}

#[test]
fn m128d_mul0() {
  let a: m128d = cast([5.0_f64, 6.0]);
  let b: m128d = cast([-8.0_f64, 4.0]);
  let out: [f64; 2] = cast(a.mul0(b));
  assert_eq!(out, [-40.0, 6.0]);
}

#[test]
fn m128d_bitor() {
  let a: m128d = cast([core::u64::MAX, 0]);
  let b: m128d = cast([core::u64::MAX, core::u64::MAX]);
  let out: [u64; 2] = cast(a | b);
  assert_eq!(out, [core::u64::MAX, core::u64::MAX]);
}

#[test]
fn m128d_bitor_assign() {
  let mut a: m128d = cast([core::u64::MAX, 0]);
  let b: m128d = cast([core::u64::MAX, core::u64::MAX]);
  a |= b;
  let out: [u64; 2] = cast(a);
  assert_eq!(out, [core::u64::MAX, core::u64::MAX]);
}

#[test]
fn m128d_set() {
  let a: m128d = m128d::set(5.0, 6.0);
  let a_f64s: [f64; 2] = cast(a);
  assert_eq!(a_f64s, [6.0_f64, 5.0]);
}

#[test]
fn m128d_splat() {
  let a: m128d = m128d::splat(5.0);
  let a_f64s: [f64; 2] = cast(a);
  assert_eq!(a_f64s, [5.0_f64, 5.0]);
}

#[test]
fn m128d_set0() {
  let a: m128d = m128d::set0(5.0);
  let a_f64s: [f64; 2] = cast(a);
  assert_eq!(a_f64s, [5.0_f64, 0.0]);
}

#[test]
fn m128d_set_reverse() {
  let a: m128d = m128d::set_reverse(5.0, 6.0);
  let a_f64s: [f64; 2] = cast(a);
  assert_eq!(a_f64s, [5.0_f64, 6.0]);
}

#[test]
fn test_shuffle128d() {
  let a: m128d = m128d::set_reverse(5.0, 6.0);
  let b: m128d = m128d::set_reverse(15.0, 16.0);
  //
  let c: m128d = shuffle128d!(a, b, [0, 1]);
  let c_arr: [f64; 2] = cast(c);
  assert_eq!(c_arr, [5.0_f64, 16.0]);
  //
  let c: m128d = shuffle128d!(a, b, [1, 0]);
  let c_arr: [f64; 2] = cast(c);
  assert_eq!(c_arr, [6.0_f64, 15.0]);
  //
  let c: m128d = shuffle128d!(a, b, [1, 1]);
  let c_arr: [f64; 2] = cast(c);
  assert_eq!(c_arr, [6.0_f64, 16.0]);
  //
  let c: m128d = shuffle128d!(a, b, [0, 0]);
  let c_arr: [f64; 2] = cast(c);
  assert_eq!(c_arr, [5.0_f64, 15.0]);
}

#[test]
fn m128d_sqrt() {
  let a: m128d = m128d::set_reverse(25.0, 36.0);
  let a: m128d = a.sqrt();
  let a_f64s: [f64; 2] = cast(a);
  assert_eq!(a_f64s, [5.0_f64, 6.0]);
}

#[test]
fn m128d_sqrt_other0() {
  let a: m128d = m128d::set_reverse(25.0, 36.0);
  let b: m128d = m128d::set0(4.0);
  let a: m128d = a.sqrt_other0(b);
  let a_f64s: [f64; 2] = cast(a);
  assert_eq!(a_f64s, [2.0_f64, 36.0]);
}

#[test]
fn m128d_store() {
  let mut t: Align16<[f64; 2]> = Align16([0.0, 0.0]);
  let a: m128d = m128d::set_reverse(25.0, 36.0);
  a.store(&mut t);
  assert_eq!(t.0, [25.0_f64, 36.0]);
}

#[test]
fn m128d_store0_all() {
  let mut t: Align16<[f64; 2]> = Align16([0.0, 0.0]);
  let a: m128d = m128d::set_reverse(25.0, 36.0);
  a.store0_all(&mut t);
  assert_eq!(t.0, [25.0_f64, 25.0]);
}

#[test]
#[allow(clippy::float_cmp)]
fn m128d_store_low() {
  let mut f: f64 = 0.0;
  let a: m128d = m128d::set_reverse(25.0, 36.0);
  a.store_low(&mut f);
  assert_eq!(f, 25.0_f64);
}

#[test]
#[allow(clippy::float_cmp)]
fn m128d_store_high() {
  let mut f: f64 = 0.0;
  let a: m128d = m128d::set_reverse(25.0, 36.0);
  a.store_high(&mut f);
  assert_eq!(f, 36.0_f64);
}

#[test]
fn m128d_store_reverse() {
  let mut t: Align16<[f64; 2]> = Align16([0.0, 0.0]);
  let a: m128d = m128d::set_reverse(25.0, 36.0);
  a.store_reverse(&mut t);
  assert_eq!(t.0, [36.0_f64, 25.0]);
}

#[test]
fn m128d_store_unaligned() {
  let mut t: [f64; 2] = [0.0, 0.0];
  let a: m128d = m128d::set_reverse(25.0, 36.0);
  a.store_unaligned(&mut t);
  assert_eq!(t, [25.0_f64, 36.0]);
}

#[test]
fn m128d_sub() {
  let a: m128d = cast([5.0_f64, 6.0]);
  let b: m128d = cast([-8.0_f64, 4.0]);
  let out: [f64; 2] = cast(a - b);
  assert_eq!(out, [13.0, 2.0]);
}

#[test]
fn m128d_sub_assign() {
  let mut a: m128d = cast([5.0_f64, 6.0]);
  let b: m128d = cast([-8.0_f64, 4.0]);
  a -= b;
  let out: [f64; 2] = cast(a);
  assert_eq!(out, [13.0, 2.0]);
}

#[test]
fn m128d_sub0() {
  let a: m128d = cast([5.0_f64, 6.0]);
  let b: m128d = cast([-8.0_f64, 4.0]);
  let out: [f64; 2] = cast(a.sub0(b));
  assert_eq!(out, [13.0, 6.0]);
}

#[test]
fn m128d_unpack_high() {
  let a: m128d = cast([5.0_f64, 6.0]);
  let b: m128d = cast([15.0_f64, 16.0]);
  let c: m128d = a.unpack_high(b);
  let c_f64s: [f64; 2] = cast(c);
  assert_eq!(c_f64s, [6.0_f64, 16.0]);
}

#[test]
fn m128d_unpack_low() {
  let a: m128d = cast([5.0_f64, 6.0]);
  let b: m128d = cast([15.0_f64, 16.0]);
  let c: m128d = a.unpack_low(b);
  let c_f64s: [f64; 2] = cast(c);
  assert_eq!(c_f64s, [5.0_f64, 15.0]);
}

#[test]
fn m128d_bitxor() {
  let a: m128d = cast([core::u64::MAX, 0]);
  let b: m128d = cast([core::u64::MAX, core::u64::MAX]);
  let out: [u64; 2] = cast(a ^ b);
  assert_eq!(out, [0, core::u64::MAX]);
}

#[test]
fn m128d_bitxor_assign() {
  let mut a: m128d = cast([core::u64::MAX, 0]);
  let b: m128d = cast([core::u64::MAX, core::u64::MAX]);
  a ^= b;
  let out: [u64; 2] = cast(a);
  assert_eq!(out, [0, core::u64::MAX]);
}

#[test]
fn m128d_neg() {
  let md: m128d = cast([5.0_f64, 6.0]);
  let a: [f64; 2] = cast(md);
  let neg_md: m128d = -md;
  let neg_a: [f64; 2] = cast(neg_md);
  assert_eq!([-5.0_f64, -6.0], neg_a);
  let neg_neg_md: m128d = -neg_md;
  let neg_neg_a: [f64; 2] = cast(neg_neg_md);
  assert_eq!(a, neg_neg_a);
}

#[test]
fn m128d_not() {
  let md: m128d = cast(core::u128::MAX);
  let not_md_as_doubles: [f64; 2] = cast(!md);
  assert_eq!(not_md_as_doubles, [0.0_f64, 0.0]);
  let md: m128 = cast(0_u128);
  let not_md_as_u128: u128 = cast(!md);
  assert_eq!(not_md_as_u128, core::u128::MAX);
}

#[test]
fn m128i_add_i8() {
  let ai: m128i = cast([
    0_i8, 1, -1, 127, -128, 100, 7, 2, 3, 1, -126, -125, 103, 10, 15, 16,
  ]);
  let bi: m128i =
    cast([1_i8, 2, 2, 1, -1, 8, 9, 20, 60, 40, 2, 9, -27, 26, 30, 31]);
  let out: [i8; 16] = cast(ai.add_i8(bi));
  assert_eq!(
    out,
    [1, 3, 1, -128, 127, 108, 16, 22, 63, 41, -124, -116, 76, 36, 45, 47]
  );
}

#[test]
fn m128i_add_i16() {
  let ai: m128i =
    cast([0_i16, 1, -1, core::i16::MAX, core::i16::MIN, 100, 7, 2]);
  let bi: m128i = cast([1_i16, 2, 2, 1, -1, 8, 9, 20]);
  let out: [i16; 8] = cast(ai.add_i16(bi));
  assert_eq!(out, [1, 3, 1, core::i16::MIN, core::i16::MAX, 108, 16, 22]);
}

#[test]
fn m128i_add_i32() {
  let ai: m128i = cast([0_i32, -1, core::i32::MAX, core::i32::MIN]);
  let bi: m128i = cast([1_i32, 2, 2, -1]);
  let out: [i32; 4] = cast(ai.add_i32(bi));
  assert_eq!(out, [1, 1, core::i32::MIN + 1, core::i32::MAX]);
}

#[test]
fn m128i_add_i64() {
  let ai: m128i = cast([core::i64::MAX, core::i64::MIN]);
  let bi: m128i = cast([2_i64, -27]);
  let out: [i64; 2] = cast(ai.add_i64(bi));
  assert_eq!(out, [core::i64::MIN + 1, core::i64::MAX - 26]);
}

#[test]
fn m128i_saturating_add_i8() {
  let ai: m128i = cast([
    0_i8, 1, -1, 127, -128, 100, 7, 2, 3, 1, -126, -125, 103, 10, 15, 16,
  ]);
  let bi: m128i =
    cast([1_i8, 2, 2, 1, -1, 8, 9, 20, 60, 40, 2, 9, -27, 26, 30, 31]);
  let out: [i8; 16] = cast(ai.saturating_add_i8(bi));
  assert_eq!(
    out,
    [1, 3, 1, 127, -128, 108, 16, 22, 63, 41, -124, -116, 76, 36, 45, 47]
  );
}

#[test]
fn m128i_saturating_add_i16() {
  let ai: m128i =
    cast([0_i16, 1, -1, core::i16::MAX, core::i16::MIN, 100, 7, 2]);
  let bi: m128i = cast([1_i16, 2, 2, 1, -1, 8, 9, 20]);
  let out: [i16; 8] = cast(ai.saturating_add_i16(bi));
  assert_eq!(out, [1, 3, 1, core::i16::MAX, core::i16::MIN, 108, 16, 22]);
}

#[test]
fn m128i_saturating_add_u8() {
  let ai: m128i =
    cast([0_u8, 1, 255, 127, 0, 100, 7, 2, 3, 1, 255, 255, 103, 10, 15, 16]);
  let bi: m128i =
    cast([1_u8, 2, 2, 1, 255, 8, 9, 20, 60, 40, 2, 9, 255, 26, 30, 31]);
  let out: [u8; 16] = cast(ai.saturating_add_u8(bi));
  assert_eq!(
    out,
    [1, 3, 255, 128, 255, 108, 16, 22, 63, 41, 255, 255, 255, 36, 45, 47]
  );
}

#[test]
fn m128i_saturating_add_u16() {
  let ai: m128i =
    cast([0_u16, 1, core::u16::MAX, core::u16::MAX, 0, 100, 7, 2]);
  let bi: m128i = cast([1_u16, 2, 2, 1, core::u16::MAX, 8, 9, 20]);
  let out: [u16; 8] = cast(ai.saturating_add_u16(bi));
  assert_eq!(out, [1, 3, 65535, 65535, 65535, 108, 16, 22]);
}

#[test]
fn m128i_andnot() {
  let max = core::u32::MAX;
  let a: m128i = cast([0, max, max, 0]);
  let b: m128i = cast([max, 0, max, 0]);
  let out: [u32; 4] = cast(a.andnot(b));
  assert_eq!(out, [max, 0, 0, 0]);
}

#[test]
fn m128i_bitor() {
  let max = core::u32::MAX;
  let a: m128i = cast([0, max, max, 0]);
  let b: m128i = cast([max, 0, max, 0]);
  let out: [u32; 4] = cast(a | b);
  assert_eq!(out, [max, max, max, 0]);
}

#[test]
fn m128i_bitor_assign() {
  let max = core::u32::MAX;
  let mut a: m128i = cast([0, max, max, 0]);
  let b: m128i = cast([max, 0, max, 0]);
  a |= b;
  let out: [u32; 4] = cast(a);
  assert_eq!(out, [max, max, max, 0]);
}

#[test]
fn m128i_bitxor() {
  let max = core::u32::MAX;
  let a: m128i = cast([0, max, max, 0]);
  let b: m128i = cast([max, 0, max, 0]);
  let out: [u32; 4] = cast(a ^ b);
  assert_eq!(out, [max, max, 0, 0]);
}

#[test]
fn m128i_bitxor_assign() {
  let max = core::u32::MAX;
  let mut a: m128i = cast([0, max, max, 0]);
  let b: m128i = cast([max, 0, max, 0]);
  a ^= b;
  let out: [u32; 4] = cast(a);
  assert_eq!(out, [max, max, 0, 0]);
}

#[test]
fn m128i_average_u8() {
  let ai: m128i =
    cast([0_u8, 1, 255, 127, 0, 100, 7, 2, 3, 1, 255, 255, 103, 10, 15, 16]);
  let bi: m128i =
    cast([1_u8, 2, 2, 1, 255, 8, 9, 20, 60, 40, 2, 9, 255, 26, 30, 31]);
  let out: [u8; 16] = cast(ai.average_u8(bi));
  assert_eq!(
    out,
    [1, 2, 129, 64, 128, 54, 8, 11, 32, 21, 129, 132, 179, 18, 23, 24]
  );
}

#[test]
fn m128i_average_u16() {
  let ai: m128i =
    cast([0_u16, 1, core::u16::MAX, core::u16::MAX, 0, 100, 7, 2]);
  let bi: m128i = cast([1_u16, 2, 2, 1, core::u16::MAX, 8, 9, 20]);
  let out: [u16; 8] = cast(ai.average_u16(bi));
  assert_eq!(out, [1, 2, 32769, 32768, 32768, 54, 8, 11]);
}

#[test]
fn m128i_cast_m128() {
  let mi: m128i = cast(12_345_678_u128);
  let m: m128 = mi.cast_m128();
  let m_bits: u128 = cast(m);
  assert_eq!(m_bits, 12_345_678_u128);
}

#[test]
fn m128i_cmp_eq_i8() {
  let ai: m128i =
    cast([0_u8, 2, 2, 127, 0, 100, 7, 2, 3, 1, 255, 255, 103, 10, 15, 16]);
  let bi: m128i =
    cast([1_u8, 2, 2, 1, 255, 8, 9, 20, 60, 40, 2, 9, 255, 26, 30, 31]);
  let out: [i8; 16] = cast(ai.cmp_eq_i8(bi));
  assert_eq!(out, [0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
}

#[test]
fn m128i_cmp_eq_i16() {
  let ai: m128i =
    cast([0_i16, 2, core::i16::MAX, core::i16::MAX, 0, 100, 7, 2]);
  let bi: m128i = cast([1_i16, 2, 2, 1, core::i16::MAX, 8, 9, 20]);
  let out: [i16; 8] = cast(ai.cmp_eq_i16(bi));
  assert_eq!(out, [0, -1, 0, 0, 0, 0, 0, 0]);
}

#[test]
fn m128i_cmp_eq_i32() {
  let ai: m128i = cast([0_i32, -1, core::i32::MAX, core::i32::MIN]);
  let bi: m128i = cast([1_i32, -1, 2, -1]);
  let out: [i32; 4] = cast(ai.cmp_eq_i32(bi));
  assert_eq!(out, [0, -1, 0, 0]);
}

#[test]
fn m128i_cmp_gt_i8() {
  let ai: m128i =
    cast([0_i8, 2, 2, 127, 0, 100, 7, 2, 3, 1, -1, -1, 103, 10, 15, 16]);
  let bi: m128i =
    cast([1_i8, 2, 2, 1, -1, 8, 9, 20, 60, 40, 2, 9, -1, 26, 30, 31]);
  let out: [i8; 16] = cast(ai.cmp_gt_i8(bi));
  assert_eq!(out, [0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0]);
}

#[test]
fn m128i_cmp_gt_i16() {
  let ai: m128i =
    cast([0_i16, 2, core::i16::MAX, core::i16::MAX, 0, 100, 7, 2]);
  let bi: m128i = cast([1_i16, 2, 2, 1, core::i16::MAX, 8, 9, 20]);
  let out: [i16; 8] = cast(ai.cmp_gt_i16(bi));
  assert_eq!(out, [0, 0, -1, -1, 0, -1, 0, 0]);
}

#[test]
fn m128i_cmp_gt_i32() {
  let ai: m128i = cast([0_i32, -1, core::i32::MAX, core::i32::MIN]);
  let bi: m128i = cast([1_i32, -1, 2, -1]);
  let out: [i32; 4] = cast(ai.cmp_gt_i32(bi));
  assert_eq!(out, [0, 0, -1, 0]);
}

#[test]
fn m128i_cmp_lt_i8() {
  let ai: m128i =
    cast([0_i8, 2, 2, 127, 0, 100, 7, 2, 3, 1, -1, -1, 103, 10, 15, 16]);
  let bi: m128i =
    cast([1_i8, 2, 2, 1, -1, 8, 9, 20, 60, 40, 2, 9, -1, 26, 30, 31]);
  let out: [i8; 16] = cast(ai.cmp_lt_i8(bi));
  assert_eq!(out, [-1, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1]);
}

#[test]
fn m128i_cmp_lt_i16() {
  let ai: m128i =
    cast([0_i16, 2, core::i16::MAX, core::i16::MAX, 0, 100, 7, 2]);
  let bi: m128i = cast([1_i16, 2, 2, 1, core::i16::MAX, 8, 9, 20]);
  let out: [i16; 8] = cast(ai.cmp_lt_i16(bi));
  assert_eq!(out, [-1, 0, 0, 0, -1, 0, -1, -1]);
}

#[test]
fn m128i_cmp_lt_i32() {
  let ai: m128i = cast([0_i32, -1, core::i32::MAX, core::i32::MIN]);
  let bi: m128i = cast([1_i32, -1, 2, -1]);
  let out: [i32; 4] = cast(ai.cmp_lt_i32(bi));
  assert_eq!(out, [-1, 0, 0, -1]);
}

#[test]
fn m128i_round_low_f64() {
  let i: m128i = cast([-1, 23, 8, 6]);
  let out: [f64; 2] = cast(i.round_low_f64());
  assert_eq!(out, [-1.0_f64, 23.0]);
}

#[test]
fn m128i_round_f32() {
  let i: m128i = cast([-1, 23, 8, 6]);
  let out: [f32; 4] = cast(i.round_f32());
  assert_eq!(out, [-1.0_f32, 23.0, 8.0, 6.0]);
}

#[test]
fn m128i_extract0_i32() {
  let i: m128i = cast([-1, 23, 8, 6]);
  assert_eq!(i.extract0_i32(), -1_i32);
}

#[test]
#[cfg(target_arch = "x86_64")]
fn m128i_extract0_i64() {
  let i: m128i = cast([-1_i64, 23]);
  assert_eq!(i.extract0_i64(), -1_i64);
}

#[test]
fn m128i_set0_i32() {
  let i: m128i = m128i::set0_i32(-1);
  let i_i32s: [i32; 4] = cast(i);
  assert_eq!(i_i32s, [-1, 0, 0, 0]);
}

#[cfg(target_arch = "x86_64")]
#[test]
fn m128i_set0_i64() {
  let i: m128i = m128i::set0_i64(-1);
  let i_i64s: [i64; 2] = cast(i);
  assert_eq!(i_i64s, [-1, 0]);
}

#[test]
fn m128i_load() {
  let i: m128i = m128i::load(&Align16(12345));
  let i_i128: i128 = cast(i);
  assert_eq!(i_i128, 12345);
}

#[test]
fn m128i_load0_i64() {
  let i: m128i = m128i::load0_i64(&Align16(20_i64));
  let i_i128: i128 = cast(i);
  assert_eq!(i_i128, 20);
}

#[test]
fn m128i_load_unaligned() {
  let i: m128i = m128i::load_unaligned(cast_ref(&12345_i128));
  let i_i128: i128 = cast(i);
  assert_eq!(i_i128, 12345);
}

#[test]
fn m128i_mul_i16_hadd() {
  let ai: m128i =
    cast([7_i16, 1, core::i16::MAX, core::i16::MAX, 0, 100, 7, 2]);
  let bi: m128i = cast([1_i16, 2, 2, 1, core::i16::MAX, 8, 9, 20]);
  let out: [i32; 4] = cast(ai.mul_i16_hadd(bi));
  assert_eq!(out, [9, 98301, 800, 103]);
}

#[test]
fn m128i_max_u8() {
  let ai: m128i =
    cast([0_u8, 1, 255, 127, 0, 100, 7, 2, 3, 1, 255, 255, 103, 10, 15, 16]);
  let bi: m128i =
    cast([1_u8, 2, 2, 1, 255, 8, 9, 20, 60, 40, 2, 9, 255, 26, 30, 31]);
  let out: [u8; 16] = cast(ai.max_u8(bi));
  assert_eq!(
    out,
    [1, 2, 255, 127, 255, 100, 9, 20, 60, 40, 255, 255, 255, 26, 30, 31]
  );
}

#[test]
fn m128i_min_u8() {
  let ai: m128i =
    cast([0_u8, 1, 255, 127, 0, 100, 7, 2, 3, 1, 255, 255, 103, 10, 15, 16]);
  let bi: m128i =
    cast([1_u8, 2, 2, 1, 255, 8, 9, 20, 60, 40, 2, 9, 255, 26, 30, 31]);
  let out: [u8; 16] = cast(ai.min_u8(bi));
  assert_eq!(out, [0, 1, 2, 1, 0, 8, 7, 2, 3, 1, 2, 9, 103, 10, 15, 16]);
}

#[test]
fn m128i_max_i16() {
  let ai: m128i =
    cast([7_i16, 1, core::i16::MAX, core::i16::MAX, 0, 100, 7, 2]);
  let bi: m128i = cast([1_i16, 2, 2, 1, core::i16::MAX, 8, 9, 20]);
  let out: [i16; 8] = cast(ai.max_i16(bi));
  assert_eq!(out, [7, 2, 32767, 32767, 32767, 100, 9, 20]);
}

#[test]
fn m128i_min_i16() {
  let ai: m128i =
    cast([7_i16, 1, core::i16::MAX, core::i16::MAX, 0, 100, 7, 2]);
  let bi: m128i = cast([1_i16, 2, 2, 1, core::i16::MAX, 8, 9, 20]);
  let out: [i16; 8] = cast(ai.min_i16(bi));
  assert_eq!(out, [1, 1, 2, 1, 0, 8, 7, 2]);
}

#[test]
fn m128i_copy0_i64() {
  let i: m128i = cast([3_i64, 7]);
  let c: m128i = i.copy0_i64();
  let c_i64s: [i64; 2] = cast(c);
  assert_eq!(c_i64s, [3, 0]);
}

#[test]
fn m128i_move_mask_i8() {
  let i: m128i =
    cast([0_u8, 1, 255, 127, 0, 100, 7, 2, 3, 1, 255, 255, 103, 10, 15, 16]);
  let mask: i32 = i.move_mask_i8();
  assert_eq!(mask, 3076);
}

#[test]
fn m128i_mul_high_i16() {
  let ai: m128i =
    cast([7_i16, 3453, core::i16::MAX, core::i16::MAX, 0, 100, 7, 2]);
  let bi: m128i = cast([1_i16, 2, 2342, 9654, core::i16::MAX, 8, 9, 20]);
  let out: [i16; 8] = cast(ai.mul_high_i16(bi));
  assert_eq!(out, [0, 0, 1170, 4826, 0, 0, 0, 0]);
}

#[test]
fn m128i_mul_low_i16() {
  let ai: m128i =
    cast([7_i16, 1, core::i16::MAX, core::i16::MAX, 0, 100, 7, 2]);
  let bi: m128i = cast([1_i16, 2, 2, 1, core::i16::MAX, 8, 9, 20]);
  let out: [i16; 8] = cast(ai.mul_low_i16(bi));
  assert_eq!(out, [7, 2, -2, 32767, 0, 800, 63, 40]);
}

#[test]
fn mul_high_u16() {
  let ai: m128i = cast([7_u16, 5000, 17000, 9001, 3543, 100, 7, 2]);
  let bi: m128i = cast([1_u16, 8, 26, 23, core::u16::MAX, 8, 9, 20]);
  let out: [u16; 8] = cast(ai.mul_high_u16(bi));
  assert_eq!(out, [0, 0, 6, 3, 3542, 0, 0, 0]);
}

#[test]
fn m128i_half_mul_u32() {
  let ai: m128i = cast([67_777_777_i64, 8765]);
  let bi: m128i = cast([3_i64, 12]);
  let out: [u64; 2] = cast(ai.half_mul_u32(bi));
  assert_eq!(out, [203_333_331, 105_180]);
}

#[test]
fn m128i_pack_i16_saturating_i8() {
  let ai: m128i =
    cast([7_i16, 1, core::i16::MAX, core::i16::MAX, 0, 100, 7, 2]);
  let bi: m128i = cast([1_i16, 2, 2, 1, core::i16::MAX, 8, 9, 20]);
  let out: [i8; 16] = cast(ai.pack_i16_saturating_i8(bi));
  assert_eq!(out, [7, 1, 127, 127, 0, 100, 7, 2, 1, 2, 2, 1, 127, 8, 9, 20]);
}

#[test]
fn m128i_pack_i32_saturating_i16() {
  let ai: m128i = cast([-1_i32, 23, 8, 6]);
  let bi: m128i = cast([-1_i32, -800_000, 8, 6]);
  let out: [i16; 8] = cast(ai.pack_i32_saturating_i16(bi));
  assert_eq!(out, [-1, 23, 8, 6, -1, -32768, 8, 6]);
}

#[test]
fn m128i_pack_i16_saturating_u8() {
  let ai: m128i =
    cast([7_i16, 1, core::i16::MAX, core::i16::MAX, 0, 100, 7, 2]);
  let bi: m128i = cast([1_i16, 2, 2, 1, core::i16::MAX, 8, 9, 20]);
  let out: [u8; 16] = cast(ai.pack_i16_saturating_u8(bi));
  assert_eq!(out, [7, 1, 255, 255, 0, 100, 7, 2, 1, 2, 2, 1, 255, 8, 9, 20]);
}

#[test]
fn m128i_signed_abs_diff_i8() {
  let ai: m128i =
    cast([0_i8, 1, -100, 127, 0, 100, 7, 2, 3, 1, -1, -5, 103, 10, 15, 16]);
  let bi: m128i =
    cast([1_i8, 2, 2, 1, -10, 8, 9, 20, 60, 40, 2, 9, -8, 26, 30, 31]);
  let out: [i64; 2] = cast(ai.signed_abs_diff_i8(bi));
  assert_eq!(out, [640_i64, 782]);
}

#[test]
fn m128i_set_i8() {
  let i: m128i =
    m128i::set_i8(0, 1, -100, 127, 0, 100, 7, 2, 3, 1, -1, -5, 103, 10, 15, 16);
  let out: [i8; 16] = cast(i);
  assert_eq!(
    out,
    [16, 15, 10, 103, -5, -1, 1, 3, 2, 7, 100, 0, 127, -100, 1, 0]
  );
}

#[test]
fn m128i_set_reverse_i8() {
  let i: m128i = m128i::set_reverse_i8(
    0, 1, -100, 127, 0, 100, 7, 2, 3, 1, -1, -5, 103, 10, 15, 16,
  );
  let out: [i8; 16] = cast(i);
  assert_eq!(
    out,
    [0, 1, -100, 127, 0, 100, 7, 2, 3, 1, -1, -5, 103, 10, 15, 16]
  );
}

#[test]
fn m128i_set_i16() {
  let i: m128i = m128i::set_i16(0, 1, -100, 127, 0, 100, 7, 2);
  let out: [i16; 8] = cast(i);
  assert_eq!(out, [2, 7, 100, 0, 127, -100, 1, 0]);
}

#[test]
fn m128i_set_reverse_i16() {
  let i: m128i = m128i::set_reverse_i16(0, 1, -100, 127, 0, 100, 7, 2);
  let out: [i16; 8] = cast(i);
  assert_eq!(out, [0, 1, -100, 127, 0, 100, 7, 2]);
}

#[test]
fn m128i_set_i32() {
  let i: m128i = m128i::set_i32(0, 1, -100, 127);
  let out: [i32; 4] = cast(i);
  assert_eq!(out, [127, -100, 1, 0]);
}

#[test]
fn m128i_set_reverse_i32() {
  let i: m128i = m128i::set_reverse_i32(0, 1, -100, 127);
  let out: [i32; 4] = cast(i);
  assert_eq!(out, [0, 1, -100, 127]);
}

#[test]
fn m128i_set_i64() {
  let i: m128i = m128i::set_i64(0, 1);
  let out: [i64; 2] = cast(i);
  assert_eq!(out, [1, 0]);
}

#[test]
fn m128i_splat_i8() {
  let i: m128i = m128i::splat_i8(1);
  let out: [i8; 16] = cast(i);
  assert_eq!(out, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]);
}

#[test]
fn m128i_splat_i16() {
  let i: m128i = m128i::splat_i16(1);
  let out: [i16; 8] = cast(i);
  assert_eq!(out, [1, 1, 1, 1, 1, 1, 1, 1,]);
}

#[test]
fn m128i_splat_i32() {
  let i: m128i = m128i::splat_i32(1);
  let out: [i32; 4] = cast(i);
  assert_eq!(out, [1, 1, 1, 1]);
}

#[test]
fn m128i_splat_i64() {
  let i: m128i = m128i::splat_i64(1);
  let out: [i64; 2] = cast(i);
  assert_eq!(out, [1, 1]);
}

#[test]
fn m128i_shift_left_i16() {
  let ai: m128i =
    cast([5_i16, 2, core::i16::MAX, core::i16::MAX, 0, 100, 7, 2]);
  let bi: m128i = cast(2_i128);
  let out: [i16; 8] = cast(ai.shift_left_i16(bi));
  assert_eq!(out, [20, 8, -4, -4, 0, 400, 28, 8]);
}

#[test]
fn m128i_shift_left_i32() {
  let ai: m128i = cast([3_i32, 2, core::i32::MAX, core::i32::MAX]);
  let bi: m128i = cast(2_i128);
  let out: [i32; 4] = cast(ai.shift_left_i32(bi));
  assert_eq!(out, [12, 8, -4, -4]);
}

#[test]
fn m128i_shift_left_i64() {
  let ai: m128i = cast([1_i64, core::i64::MAX]);
  let bi: m128i = cast(2_i128);
  let out: [i64; 2] = cast(ai.shift_left_i64(bi));
  assert_eq!(out, [4, -4]);
}

#[test]
fn m128i_shift_right_sign_i16() {
  let ai: m128i =
    cast([5_i16, 2, core::i16::MAX, core::i16::MAX, 0, 100, 7, 2]);
  let bi: m128i = cast(2_i128);
  let out: [i16; 8] = cast(ai.shift_right_sign_i16(bi));
  assert_eq!(out, [1, 0, 8191, 8191, 0, 25, 1, 0]);
}

#[test]
fn m128i_shift_right_sign_i32() {
  let ai: m128i = cast([3_i32, 2, core::i32::MAX, core::i32::MAX]);
  let bi: m128i = cast(2_i128);
  let out: [i32; 4] = cast(ai.shift_right_sign_i32(bi));
  assert_eq!(out, [0, 0, 536_870_911, 536_870_911]);
}

#[test]
fn m128i_shift_right_zero_i16() {
  let ai: m128i =
    cast([5_i16, 2, core::i16::MAX, core::i16::MAX, 0, 100, 7, 2]);
  let bi: m128i = cast(2_i128);
  let out: [i16; 8] = cast(ai.shift_right_zero_i16(bi));
  assert_eq!(out, [1, 0, 8191, 8191, 0, 25, 1, 0]);
}

#[test]
fn m128i_shift_right_zero_i32() {
  let ai: m128i = cast([3_i32, 2, core::i32::MAX, core::i32::MAX]);
  let bi: m128i = cast(2_i128);
  let out: [i32; 4] = cast(ai.shift_right_zero_i32(bi));
  assert_eq!(out, [0, 0, 536_870_911, 536_870_911]);
}

#[test]
fn m128i_shift_right_zero_i64() {
  let ai: m128i = cast([1_i64, core::i64::MAX]);
  let bi: m128i = cast(2_i128);
  let out: [i64; 2] = cast(ai.shift_right_zero_i64(bi));
  assert_eq!(out, [0, 2_305_843_009_213_693_951]);
}

#[test]
fn m128i_store() {
  let mut target = Align16(0_i128);
  let mi: m128i = cast(5_i128);
  mi.store(&mut target);
  assert_eq!(target.0, 5);
}

#[test]
fn m128i_store0_i64() {
  let mut target = Align16(0_i64);
  let mi: m128i = cast(5_i128);
  mi.store0_i64(&mut target);
  assert_eq!(target.0, 5);
}

#[test]
fn m128i_store_unaligned() {
  let mut target = 0_i128;
  let mi: m128i = cast(5_i128);
  mi.store_unaligned(cast_mut(&mut target));
  assert_eq!(target, 5);
}

#[test]
fn m128i_sub_i8() {
  let ai: m128i = cast([
    0_i8, 1, -1, 127, -128, 100, 7, 2, 3, 1, -126, -125, 103, 10, 15, 16,
  ]);
  let bi: m128i =
    cast([1_i8, 2, 2, 1, -1, 8, 9, 20, 60, 40, 2, 9, -27, 26, 30, 31]);
  let out: [i8; 16] = cast(ai.sub_i8(bi));
  assert_eq!(
    out,
    [
      -1, -1, -3, 126, -127, 92, -2, -18, -57, -39, -128, 122, -126, -16, -15,
      -15
    ]
  );
}

#[test]
fn m128i_sub_i16() {
  let ai: m128i =
    cast([0_i16, 1, -1, core::i16::MAX, core::i16::MIN, 100, 7, 2]);
  let bi: m128i = cast([1_i16, 2, 2, 1, -1, 8, 9, 20]);
  let out: [i16; 8] = cast(ai.sub_i16(bi));
  assert_eq!(out, [-1, -1, -3, 32766, -32767, 92, -2, -18]);
}

#[test]
fn m128i_sub_i32() {
  let ai: m128i = cast([0_i32, -1, core::i32::MAX, core::i32::MIN]);
  let bi: m128i = cast([1_i32, 2, 2, -1]);
  let out: [i32; 4] = cast(ai.sub_i32(bi));
  assert_eq!(out, [-1, -3, 2147483645, -2147483647]);
}

#[test]
fn m128i_sub_i64() {
  let ai: m128i = cast([core::i64::MAX, core::i64::MIN]);
  let bi: m128i = cast([2_i64, -27]);
  let out: [i64; 2] = cast(ai.sub_i64(bi));
  assert_eq!(out, [9223372036854775805, -9223372036854775781]);
}

#[test]
fn m128i_saturating_sub_i8() {
  let ai: m128i = cast([
    0_i8, 1, -1, 127, -128, 100, 7, 2, 3, 1, -126, -125, 103, 10, 15, 16,
  ]);
  let bi: m128i =
    cast([1_i8, 2, 2, 1, -1, 8, 9, 20, 60, 40, 2, 9, -27, 26, 30, 31]);
  let out: [i8; 16] = cast(ai.saturating_sub_i8(bi));
  assert_eq!(
    out,
    [
      -1, -1, -3, 126, -127, 92, -2, -18, -57, -39, -128, -128, 127, -16, -15,
      -15
    ]
  );
}

#[test]
fn m128i_saturating_sub_i16() {
  let ai: m128i = cast([0_i16, 1, -1, 76, core::i16::MIN, 100, 7, 2]);
  let bi: m128i = cast([1_i16, 2, 2, core::i16::MAX, -1, 8, 9, 20]);
  let out: [i16; 8] = cast(ai.saturating_sub_i16(bi));
  assert_eq!(out, [-1, -1, -3, -32691, -32767, 92, -2, -18]);
}

#[test]
fn m128i_saturating_sub_u8() {
  let ai: m128i =
    cast([0_u8, 1, 17, 127, 130, 100, 7, 2, 3, 1, 255, 245, 103, 10, 15, 16]);
  let bi: m128i =
    cast([1_u8, 2, 2, 1, 255, 8, 9, 20, 60, 40, 2, 9, 250, 26, 30, 31]);
  let out: [u8; 16] = cast(ai.saturating_sub_i8(bi));
  assert_eq!(
    out,
    [
      255, 255, 15, 126, 131, 92, 254, 238, 199, 217, 253, 236, 109, 240, 241,
      241
    ]
  );
}

#[test]
fn m128i_saturating_sub_u16() {
  let ai: m128i =
    cast([0_u16, 1, 34000, core::u16::MAX, core::u16::MIN, 100, 7, 2]);
  let bi: m128i = cast([1_u16, 2, 2, 1, 34000, 8, 9, 20]);
  let out: [u16; 8] = cast(ai.saturating_sub_u16(bi));
  assert_eq!(out, [0, 0, 33998, 65534, 0, 92, 0, 0]);
}

#[test]
fn m128i_unpack_high_i8() {
  let ai: m128i = cast([
    0_i8, 1, -1, 127, -128, 100, 7, 2, 3, 1, -126, -125, 103, 10, 15, 16,
  ]);
  let bi: m128i =
    cast([1_i8, 2, 2, 1, -1, 8, 9, 20, 60, 40, 2, 9, -27, 26, 30, 31]);
  let out: [i8; 16] = cast(ai.unpack_high_i8(bi));
  assert_eq!(
    out,
    [3, 60, 1, 40, -126, 2, -125, 9, 103, -27, 10, 26, 15, 30, 16, 31]
  );
}

#[test]
fn m128i_unpack_high_i16() {
  let ai: m128i =
    cast([0_i16, 1, -1, core::i16::MAX, core::i16::MIN, 100, 7, 2]);
  let bi: m128i = cast([1_i16, 2, 2, 1, -1, 8, 9, 20]);
  let out: [i16; 8] = cast(ai.unpack_high_i16(bi));
  assert_eq!(out, [-32768, -1, 100, 8, 7, 9, 2, 20]);
}

#[test]
fn m128i_unpack_high_i32() {
  let ai: m128i = cast([0_i32, -1, core::i32::MAX, core::i32::MIN]);
  let bi: m128i = cast([1_i32, 2, 2, -1]);
  let out: [i32; 4] = cast(ai.unpack_high_i32(bi));
  assert_eq!(out, [2147483647, 2, -2147483648, -1]);
}

#[test]
fn m128i_unpack_high_i64() {
  let ai: m128i = cast([core::i64::MAX, core::i64::MIN]);
  let bi: m128i = cast([2_i64, -27]);
  let out: [i64; 2] = cast(ai.unpack_high_i64(bi));
  assert_eq!(out, [-9223372036854775808, -27]);
}

#[test]
fn m128i_unpack_low_i8() {
  let ai: m128i = cast([
    0_i8, 1, -1, 127, -128, 100, 7, 2, 3, 1, -126, -125, 103, 10, 15, 16,
  ]);
  let bi: m128i =
    cast([1_i8, 2, 2, 1, -1, 8, 9, 20, 60, 40, 2, 9, -27, 26, 30, 31]);
  let out: [i8; 16] = cast(ai.unpack_low_i8(bi));
  assert_eq!(out, [0, 1, 1, 2, -1, 2, 127, 1, -128, -1, 100, 8, 7, 9, 2, 20]);
}

#[test]
fn m128i_unpack_low_i16() {
  let ai: m128i =
    cast([0_i16, 1, -1, core::i16::MAX, core::i16::MIN, 100, 7, 2]);
  let bi: m128i = cast([1_i16, 2, 2, 1, -1, 8, 9, 20]);
  let out: [i16; 8] = cast(ai.unpack_low_i16(bi));
  assert_eq!(out, [0, 1, 1, 2, -1, 2, 32767, 1]);
}

#[test]
fn m128i_unpack_low_i32() {
  let ai: m128i = cast([0_i32, -1, core::i32::MAX, core::i32::MIN]);
  let bi: m128i = cast([1_i32, 2, 2, -1]);
  let out: [i32; 4] = cast(ai.unpack_low_i32(bi));
  assert_eq!(out, [0, 1, -1, 2]);
}

#[test]
fn m128i_unpack_low_i64() {
  let ai: m128i = cast([core::i64::MAX, core::i64::MIN]);
  let bi: m128i = cast([2_i64, -27]);
  let out: [i64; 2] = cast(ai.unpack_low_i64(bi));
  assert_eq!(out, [9223372036854775807, 2]);
}

#[test]
fn m128i_not() {
  let md: m128i = cast(core::u128::MAX);
  let not_md_as_qw: [i64; 2] = cast(!md);
  assert_eq!(not_md_as_qw, [0_i64, 0]);
  let md: m128 = cast(0_u128);
  let not_md_as_u128: u128 = cast(!md);
  assert_eq!(not_md_as_u128, core::u128::MAX);
}

#[test]
#[allow(clippy::float_cmp)]
fn m128_ceil_sse2() {
  for f_input in [
    1.0,
    1.1,
    -0.1,
    2_147_483_648.0_f32,
    2_147_483_649.0_f32,
    core::f32::NAN,
    core::f32::MAX,
    core::f32::MIN,
    core::f32::MIN_POSITIVE,
    core::f32::INFINITY,
    core::f32::NEG_INFINITY,
  ]
  .iter()
  .copied()
  {
    let f_output = m128::set0(f_input).ceil_sse2().extract0();
    if f_input.is_nan() {
      assert!(f_output.is_nan())
    } else {
      assert_eq!(f32::ceil(f_input), f_output)
    }
  }
}

#[test]
#[allow(clippy::float_cmp)]
fn m128_floor_sse2() {
  for f_input in [
    1.0,
    1.1,
    -0.1,
    2_147_483_648.0_f32,
    2_147_483_649.0_f32,
    core::f32::NAN,
    core::f32::MAX,
    core::f32::MIN,
    core::f32::MIN_POSITIVE,
    core::f32::INFINITY,
    core::f32::NEG_INFINITY,
  ]
  .iter()
  .copied()
  {
    let f_output = m128::set0(f_input).floor_sse2().extract0();
    if f_input.is_nan() {
      assert!(f_output.is_nan())
    } else {
      assert_eq!(f32::floor(f_input), f_output, "input:{}", f_input)
    }
  }
}
