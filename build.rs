#![allow(unused_imports)]

use std::process::Command;

fn main() {
  if cfg!(not(feature = "always_use_stable")) {
    let output = Command::new("rustc")
      .arg("--version")
      .output()
      .expect("failed to execute `rustc --version`!");
    assert!(output.status.success(), "toolchain detection unsuccessful!");
    let stdout = String::from_utf8_lossy(&output.stdout);
    if stdout.contains("nightly") {
      println!(r#"cargo:rustc-cfg=feature="toolchain_nightly""#);
    }
  }
  let target = env!("TARGET");
  if !(target.contains("x86") || target.contains("x86_64")) {
    println!(r#"cargo:rustc-cfg=feature="extern_crate_std""#);
  }
}
