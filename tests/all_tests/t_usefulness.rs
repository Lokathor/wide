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

/// Implement JPEG IDCT using i16x8. This has slightly different behavior than
/// the normal 32 bit scalar implementation in libjpeg. It's a bit more accurate
/// in some ways (since the constants are encoded in 15 bits instead of 12) but
/// is more subject to hitting saturation during intermediate calculations,
/// although that should normally not be a problem for photographic JPEGs.
///
/// The main downside of this approach is that it is very slow to do saturating
/// math on scalar types on some CPUs, so if you need bit-exact behavior on
/// different architectures this is not the algorithm for you.
#[test]
fn test_dequantize_and_idct_i16() {
  fn to_fixed(x: f32) -> i16 {
    (x * 32767.0 + 0.5) as i16
  }

  fn kernel_i16(data: [i16x8; 8]) -> [i16x8; 8] {
    // kernel x
    let a2 = data[2];
    let a6 = data[6];

    let b0 = a2.saturating_add(a6).mul_scale_round_n(to_fixed(0.5411961));
    let c0 = b0
      .saturating_sub(a6)
      .saturating_sub(a6.mul_scale_round_n(to_fixed(0.847759065)));
    let c1 = b0.saturating_add(a2.mul_scale_round_n(to_fixed(0.765366865)));

    let a0 = data[0];
    let a4 = data[4];
    let b1 = a0.saturating_add(a4);
    let b2 = a0.saturating_sub(a4);

    let x0 = b1.saturating_add(c1);
    let x1 = b2.saturating_add(c0);
    let x2 = b2.saturating_sub(c0);
    let x3 = b1.saturating_sub(c1);

    // kernel t
    let t0 = data[7];
    let t1 = data[5];
    let t2 = data[3];
    let t3 = data[1];

    let p1 = t0.saturating_add(t3);
    let p2 = t1.saturating_add(t2);
    let p3 = t0.saturating_add(t2);
    let p4 = t1.saturating_add(t3);

    let p5t = p3.saturating_add(p4);
    let p5 = p5t.saturating_add(p5t.mul_scale_round_n(to_fixed(0.175875602)));

    let e0 = t0.mul_scale_round_n(to_fixed(0.298631336));
    let e1 = t1
      .saturating_add(t1)
      .saturating_add(t1.mul_scale_round_n(to_fixed(0.053119869)));

    let e2 = t2
      .saturating_add(t2)
      .saturating_add(t2)
      .saturating_add(t2.mul_scale_round_n(to_fixed(0.072711026)));
    let e3 = t3.saturating_add(t3.mul_scale_round_n(to_fixed(0.501321110)));

    let f0 = p5.saturating_sub(p1.mul_scale_round_n(to_fixed(0.899976223)));
    let f1 = p5
      .saturating_sub(p2)
      .saturating_sub(p2)
      .saturating_sub(p2.mul_scale_round_n(to_fixed(0.562915447)));

    let f2 = p3.mul_scale_round_n(to_fixed(-0.961570560)).saturating_sub(p3);
    let f3 = p4.mul_scale_round_n(to_fixed(-0.390180644));

    let t3 = f0.saturating_add(f3).saturating_add(e3);
    let t2 = f1.saturating_add(f2).saturating_add(e2);
    let t1 = f1.saturating_add(f3).saturating_add(e1);
    let t0 = f0.saturating_add(f2).saturating_add(e0);

    [
      x0.saturating_add(t3),
      x1.saturating_add(t2),
      x2.saturating_add(t1),
      x3.saturating_add(t0),
      x3.saturating_sub(t0),
      x2.saturating_sub(t1),
      x1.saturating_sub(t2),
      x0.saturating_sub(t3),
    ]
  }

  #[cfg_attr(rustfmt, rustfmt_skip)]
	    let coefficients: [i16; 8 * 8] = [
          -14, -39, 58, -2, 3, 3, 0, 1,
	        11, 27, 4, -3, 3, 0, 1, 0,
	        -6, -13, -9, -1, -2, -1, 0, 0,
	        -4, 0, -1, -2, 0, 0, 0, 0,
	        3, 0, 0, 0, 0, 0, 0, 0,		
	        -3, -2, 0, 0, 0, 0, 0, 0,		
	        0, 0, 0, 0, 0, 0, 0, 0,
	        0, 0, 0, 0, 0, 0, 0, 0
	    ];

  #[cfg_attr(rustfmt, rustfmt_skip)]
	    let quantization_table: [i16; 8 * 8] = [
	        8, 6, 5, 8, 12, 20, 26, 31,
	        6, 6, 7, 10, 13, 29, 30, 28,
	        7, 7, 8, 12, 20, 29, 35, 28,
	        7, 9, 11, 15, 26, 44, 40, 31,
	        9, 11, 19, 28, 34, 55, 52, 39,
	        12, 18, 28, 32, 41, 52, 57, 46,
	        25, 32, 39, 44, 52, 61, 60, 51,
	        36, 46, 48, 49, 56, 50, 52, 50
	    ];

  let c: [i16x8; 8] = cast(coefficients);
  let q: [i16x8; 8] = cast(quantization_table);

  // coefficients normally go up to 1024, shift up by 3 to get extra precision
  const SHIFT: i16 = 3;

  let data = [
    c[0] * q[0] << SHIFT,
    c[1] * q[1] << SHIFT,
    c[2] * q[2] << SHIFT,
    c[3] * q[3] << SHIFT,
    c[4] * q[4] << SHIFT,
    c[5] * q[5] << SHIFT,
    c[6] * q[6] << SHIFT,
    c[7] * q[7] << SHIFT,
  ];

  let pass1 = kernel_i16(data);
  let transpose1 = i16x8::transpose(pass1);
  let pass2 = kernel_i16(transpose1);
  let result = i16x8::transpose(pass2);

  // offset to recenter to 0..256 and round properly
  const ROUND_FACTOR: i16 = 0x2020;
  let round_factor = i16x8::splat(ROUND_FACTOR);

  let result_adj = [
    result[0].saturating_add(round_factor) >> (2 * SHIFT),
    result[1].saturating_add(round_factor) >> (2 * SHIFT),
    result[2].saturating_add(round_factor) >> (2 * SHIFT),
    result[3].saturating_add(round_factor) >> (2 * SHIFT),
    result[4].saturating_add(round_factor) >> (2 * SHIFT),
    result[5].saturating_add(round_factor) >> (2 * SHIFT),
    result[6].saturating_add(round_factor) >> (2 * SHIFT),
    result[7].saturating_add(round_factor) >> (2 * SHIFT),
  ];

  let output: [i16; 64] = cast(result_adj);

  #[cfg_attr(rustfmt, rustfmt_skip)]
	    let expected_output = [
	        118, 92, 110, 83, 77, 93, 144, 198,		
	        172, 116, 114, 87, 78, 93, 146, 191,		
	        194, 107, 91, 76, 71, 93, 160, 198,		
	        196, 100, 80, 74, 67, 92, 174, 209,		
	        182, 104, 88, 81, 68, 89, 178, 206,		
	        105, 64, 59, 59, 63, 94, 183, 201,		
	        35, 27, 28, 37, 72, 121, 203, 204,		
	        38, 45, 41, 47, 99, 154, 223, 208		
	    ];

  assert_eq!(expected_output, output);
}

/// Implement JPEG IDCT using i32x8. This is most similar to the scalar
/// libjpeg version which has slightly different rounding propertis than the 16
/// bit version. Some decoders are forced to use this if they want bit-by-bit
/// compability across all architectures.
#[test]
fn test_dequantize_and_idct_i32() {
  fn to_fixed(x: f32) -> i32 {
    (x * 4096.0 + 0.5) as i32
  }

  fn kernel_i32([s0, s1, s2, s3, s4, s5, s6, s7]: [i32x8; 8]) -> [i32x8; 8] {
    // Even `chunk` indicies
    let xp1 = (s2 + s6) * to_fixed(0.5411961);
    let xv2 = xp1 + s6 * to_fixed(-1.847759065);
    let xv3 = xp1 + s2 * to_fixed(0.765366865);

    let xv0 = (s0 + s4) << 12;
    let xt1 = (s0 - s4) << 12;

    let x0 = xv0 + xv3;
    let x1 = xt1 + xv2;
    let x2 = xt1 - xv2;
    let x3 = xv0 - xv3;

    // Odd `chunk` indicies
    let mut t0 = s7;
    let mut t1 = s5;
    let mut t2 = s3;
    let mut t3 = s1;

    let tp1 = t0 + t3;
    let tp2 = t1 + t2;
    let tp3 = t0 + t2;
    let tp4 = t1 + t3;
    let tp5 = (tp3 + tp4) * to_fixed(1.175875602);

    t0 = t0 * to_fixed(0.298631336);
    t1 = t1 * to_fixed(2.053119869);
    t2 = t2 * to_fixed(3.072711026);
    t3 = t3 * to_fixed(1.501321110);

    let tp1 = tp5 + tp1 * to_fixed(-0.899976223);
    let tp2 = tp5 + tp2 * to_fixed(-2.562915447);
    let tp3 = tp3 * to_fixed(-1.961570560);
    let tp4 = tp4 * to_fixed(-0.390180644);

    t0 += tp1 + tp3;
    t1 += tp2 + tp4;
    t2 += tp2 + tp3;
    t3 += tp1 + tp4;

    [x0, x1, x2, x3, t0, t1, t2, t3]
  }

  #[cfg_attr(rustfmt, rustfmt_skip)]
	    let coefficients: [i32; 8 * 8] = [
          -14, -39, 58, -2, 3, 3, 0, 1,		
	        11, 27, 4, -3, 3, 0, 1, 0,		
	        -6, -13, -9, -1, -2, -1, 0, 0,		
	        -4, 0, -1, -2, 0, 0, 0, 0,		
	        3, 0, 0, 0, 0, 0, 0, 0,		
	        -3, -2, 0, 0, 0, 0, 0, 0,		
	        0, 0, 0, 0, 0, 0, 0, 0,		
	        0, 0, 0, 0, 0, 0, 0, 0		
	    ];

  #[cfg_attr(rustfmt, rustfmt_skip)]
	    let quantization_table: [i32; 8 * 8] = [
	        8, 6, 5, 8, 12, 20, 26, 31,		
	        6, 6, 7, 10, 13, 29, 30, 28,		
	        7, 7, 8, 12, 20, 29, 35, 28,		
	        7, 9, 11, 15, 26, 44, 40, 31,		
	        9, 11, 19, 28, 34, 55, 52, 39,		
	        12, 18, 28, 32, 41, 52, 57, 46,		
	        25, 32, 39, 44, 52, 61, 60, 51,		
	        36, 46, 48, 49, 56, 50, 52, 50		
	    ];

  let c: [i32x8; 8] = cast(coefficients);
  let q: [i32x8; 8] = cast(quantization_table);

  let scaled = [
    c[0] * q[0],
    c[1] * q[1],
    c[2] * q[2],
    c[3] * q[3],
    c[4] * q[4],
    c[5] * q[5],
    c[6] * q[6],
    c[7] * q[7],
  ];

  let [mut x0, mut x1, mut x2, mut x3, t0, t1, t2, t3] = kernel_i32(scaled);

  // add rounding factor before shifting right
  const T_ROUND_FACTOR: i32 = 1 << 9;

  x0 = x0 + T_ROUND_FACTOR;
  x1 = x1 + T_ROUND_FACTOR;
  x2 = x2 + T_ROUND_FACTOR;
  x3 = x3 + T_ROUND_FACTOR;

  let pass1 = [
    (x0 + t3) >> 10,
    (x1 + t2) >> 10,
    (x2 + t1) >> 10,
    (x3 + t0) >> 10,
    (x3 - t0) >> 10,
    (x2 - t1) >> 10,
    (x1 - t2) >> 10,
    (x0 - t3) >> 10,
  ];

  let transpose1 = i32x8::transpose(pass1);

  let [mut x0, mut x1, mut x2, mut x3, t0, t1, t2, t3] = kernel_i32(transpose1);

  // add rounding factor befor shifting right (include rebasing from -128..128
  // to 0..256)
  const X_ROUND_FACTOR: i32 = 65536 + (128 << 17);

  x0 = x0 + X_ROUND_FACTOR;
  x1 = x1 + X_ROUND_FACTOR;
  x2 = x2 + X_ROUND_FACTOR;
  x3 = x3 + X_ROUND_FACTOR;

  let pass2 = [
    (x0 + t3) >> 17,
    (x1 + t2) >> 17,
    (x2 + t1) >> 17,
    (x3 + t0) >> 17,
    (x3 - t0) >> 17,
    (x2 - t1) >> 17,
    (x1 - t2) >> 17,
    (x0 - t3) >> 17,
  ];

  let result = i32x8::transpose(pass2);

  let output: [i32; 64] = cast(result);

  // same as other DCT test with some minor rounding differences
  #[cfg_attr(rustfmt, rustfmt_skip)]
	    let expected_output = [
        118, 92, 110, 83, 77, 93, 144, 198, 
        172, 116, 114, 87, 78, 93, 146, 191, 
        194, 107, 91, 76, 71, 93, 160, 198, 
        196, 100, 80, 74, 67, 92, 174, 209, 
        182, 104, 88, 81, 68, 89, 178, 206, 
        105, 64, 59, 59, 63, 94, 183, 201, 
        35, 27, 28, 37, 72, 121, 203, 204, 
        37, 45, 41, 47, 98, 154, 223, 208];

  assert_eq!(expected_output, output);
}
