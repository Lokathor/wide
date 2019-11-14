use wide::*;

fn main() {
  // /*
  let mut f = 0.0;
  while f <= 1.5 {
    println!("rad {:.2}: s/c {:?}", f, f32x4::from(f).sin_cos());
    f += 0.1;
  }
  // */
  //println!("s/c {:?}", f32x4::new(1.0, 2.0, 4.0, 5.0).sin_cos());
}
