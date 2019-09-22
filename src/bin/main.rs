
use wide::cfg_block;

fn main() {
  cfg_block!{if #[cfg(feature = "toolchain_nightly")] {
    println!("Nightly was detected");
  } else if #[cfg(windows)] {
    println!("Did not detect a nightly toolchain, but we're on Windows.");
  } else {
    println!("frop");
  }}

  
  cfg_block!{if #[cfg(feature = "toolchain_nightly")] {
    println!("2Nightly was detected");
  } else if #[cfg(windows)] {
    println!("2Did not detect a nightly toolchain, but we're on Windows.");
  }}
}
