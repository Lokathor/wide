fn main() {
  let mut buffer = String::new();
  //
  buffer.push_str(r#"#![warn(missing_docs)]"#);
  //
  if cfg!(feature = "toolchain_nightly") {
    println!("I think this is nightly.");
  } else {
    println!("I think this is stable.");
  }
  //
  use std::{fs::File, io::prelude::*};
  let mut file = File::create("target/f32x.rs").expect("couldn't make file");
  file
    .write_all(buffer.as_bytes())
    .expect("couldn't write all the data.");
}
