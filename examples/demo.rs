use wide::*;

fn main() {
  // /*
  let mut f = 0.0;
  #[cfg(feature = "extern_crate_std")]
  while f <= 1.5 {
    println!("rad {:.2}: s/c {:?}", f, f32x4::from(f).sin_cos());
    f += 0.1;
  }
  println!("-0.53_f32x4: {}", wide::f32x4::from(-0.53f32).fract()[0]);
  println!("-1.25_f32x4: {}", wide::f32x4::from(-1.25f32).fract()[0]);
  println!("0.25_f32x4: {}", wide::f32x4::from(0.25f32).fract()[0]);
  // */
  //println!("s/c {:?}", f32x4::new(1.0, 2.0, 4.0, 5.0).sin_cos());
}
