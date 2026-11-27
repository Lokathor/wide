/// A custom version of [`core::assert!`] that additionaly accepts the syntax:
///
/// ```
/// assert!(condition, dbg!(variable_0, variable_1, ...));
/// ```
///
/// The optional `dbg!` section is used to print the values of appropriate
/// parameters when a test fails, with shorter syntax than what
/// [`core::assert!`] supports.
macro_rules! assert {
    ($condition:expr $(, dbg!($($ctx:expr),* $(,)?))? $(, $message:literal $($args:tt)*)?) => {
        #[allow(clippy::neg_cmp_op_on_partial_ord)]
        if !$condition {
            $($(dbg!($ctx);)*)?

            $(
                println!("\n");
                println!($message $($args)*);
            )?

            panic!("\nassert! failed");
        }
    };
}
pub(crate) use assert;

/// A custom version of [`core::assert_eq!`] that additionaly accepts the
/// syntax:
///
/// ```
/// assert_eq!(a, b, dbg!(variable_0, variable_1, ...));
/// ```
///
/// The optional `dbg!` section is used to print the values of appropriate
/// parameters when a test fails, with shorter syntax than what
/// [`core::assert_eq!`] supports.
macro_rules! assert_eq {
    ($left:expr, $right:expr $(, dbg!($($ctx:expr),* $(,)?))? $(, $message:literal $($args:tt)*)?) => {
        match (&$left, &$right) {
            (left, right) => {
                if !(*left == *right) {
                    println!("{:>10} = {left:?}", stringify!($left));
                    println!("{:>10} = {right:?}", stringify!($right));
                    $($(dbg!($ctx);)*)?

                    $(
                        println!("\n");
                        println!($message $($args)*);
                    )?

                    panic!("\nassert_eq! failed");
                }
            }
        }
    };
}
pub(crate) use assert_eq;

mod tests {
    #[test]
    fn test_assert() {
        super::assert!(true);
    }

    #[test]
    #[should_panic]
    fn test_assert_panic() {
        super::assert!(false);
    }

    #[test]
    fn test_assert_eq() {
        super::assert_eq!(1, 1);
    }

    #[test]
    #[should_panic]
    fn test_assert_eq_panic() {
        super::assert_eq!(1, 2);
    }
}
