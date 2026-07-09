use std::{
  panic::{UnwindSafe, catch_unwind},
  println,
};

/// Asserts that the given expression panics.
macro_rules! assert_panic {
  ($expr:expr $(,)?) => {
    crate::utils::assert_panic_helper(|| {
      let _ = $expr;
    })
  };
}
pub(crate) use assert_panic;

#[doc(hidden)]
#[track_caller]
pub fn assert_panic_helper(f: impl FnOnce() + UnwindSafe) {
  match catch_unwind(f) {
    Ok(_) => panic!("assertion `panic` failed"),
    Err(_) => println!("ok: panic is expected"),
  }
}

mod tests {
  #[test]
  #[expect(clippy::diverging_sub_expression)]
  fn test_assert_panic() {
    assert_panic!(panic!());
  }

  #[test]
  #[should_panic]
  fn test_assert_panic_panic() {
    assert_panic!(());
  }
}
