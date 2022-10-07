use wide::*;

use bytemuck::*;

#[test]
fn unpack_modify_and_repack_rgba_values() {
  let mask = u32x4::from(0xFF);
  //
  let input = u32x4::from([0xFF0000FF, 0x00FF00FF, 0x0000FFFF, 0x000000FF]);

  // unpack
  let r_actual = cast::<_, i32x4>(input >> 24).round_float();
  let g_actual = cast::<_, i32x4>((input >> 16) & mask).round_float();
  let b_actual = cast::<_, i32x4>((input >> 8) & mask).round_float();
  let a_actual = cast::<_, i32x4>(input & mask).round_float();
  let r_expected = f32x4::from([255.0, 0.0, 0.0, 0.0]);
  let g_expected = f32x4::from([0.0, 255.0, 0.0, 0.0]);
  let b_expected = f32x4::from([0.0, 0.0, 255.0, 0.0]);
  let a_expected = f32x4::from([255.0, 255.0, 255.0, 255.0]);
  assert_eq!(r_expected, r_actual);
  assert_eq!(g_expected, g_actual);
  assert_eq!(b_expected, b_actual);
  assert_eq!(a_expected, a_actual);

  // modify some of the data
  let r_new = (r_actual - f32x4::from(1.0)).max(f32x4::from(0.0));
  let g_new = (g_actual - f32x4::from(1.0)).max(f32x4::from(0.0));
  let b_new = (b_actual - f32x4::from(1.0)).max(f32x4::from(0.0));
  let a_new = a_actual;

  // repack
  let r_u = cast::<i32x4, u32x4>(r_new.round_int());
  let g_u = cast::<i32x4, u32x4>(g_new.round_int());
  let b_u = cast::<i32x4, u32x4>(b_new.round_int());
  let a_u = cast::<i32x4, u32x4>(a_new.round_int());
  let output_actual = (r_u << 24) | (g_u << 16) | (b_u << 8) | (a_u);
  let output_expected =
    u32x4::from([0xFE0000FF, 0x00FE00FF, 0x0000FEFF, 0x000000FF]);
  assert_eq!(output_expected, output_actual);
}
