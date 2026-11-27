use wide::{
    f32x4, f32x8, f32x16, f64x2, f64x4, f64x8, i8x16, i8x32, i8x64, i16x8, i16x16, i16x32, i32x4,
    i32x8, i32x16, i64x2, i64x4, i64x8, u8x16, u8x32, u8x64, u16x8, u16x16, u16x32, u32x4, u32x8,
    u32x16, u64x2, u64x4, u64x8,
};

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

/// Asserts bit-equality for two values.
///
/// For floats, this is different from `==` equality. This treats `-0.0` and
/// `0.0` as not equal and only treats NaNs with equal bit patterns as equal,
/// whereas `==` treats `-0.0` and `0.0` as equal and treats all NaNs as not
/// equal. For integers, this is the same as `==` equality.
///
/// This accepts the synatx:
///
/// ```
/// assert_bit_eq!(a, b, dbg!(variable_0, variable_1, ...));
/// ```
///
/// The optional `dbg!` section is used to print the values of appropriate
/// parameters when a test fails.
macro_rules! assert_bit_eq {
    ($left:expr, $right:expr $(, dbg!($($ctx:expr),* $(,)?))? $(, $message:literal $($args:tt)*)?) => {
        match (&$left, &$right) {
            (left, right) => {
                if !crate::utils::BitEq::bit_eq(left, right) {
                    println!("{:>10} = {left:?}", stringify!($left));
                    println!("{:>10} = {right:?}", stringify!($right));
                    $($(dbg!($ctx);)*)?

                    $(
                        println!("\n");
                        println!($message $($args)*);
                    )?

                    panic!("\nassert_bit_eq! failed");
                }
            }
        }
    };
}
pub(crate) use assert_bit_eq;

pub trait BitEq {
    fn bit_eq(&self, other: &Self) -> bool;
}

macro_rules! float {
    ($T:ident) => {
        impl BitEq for $T {
            fn bit_eq(&self, other: &Self) -> bool {
                self.to_bits() == other.to_bits()
            }
        }
    };
}
float!(f32);
float!(f64);

macro_rules! integer {
    ($T:ident) => {
        impl BitEq for $T {
            fn bit_eq(&self, other: &Self) -> bool {
                self == other
            }
        }
    };
}
integer!(i8);
integer!(i16);
integer!(i32);
integer!(i64);
integer!(i128);
integer!(isize);
integer!(u8);
integer!(u16);
integer!(u32);
integer!(u64);
integer!(u128);
integer!(usize);

impl<T, const N: usize> BitEq for [T; N]
where
    T: BitEq,
{
    fn bit_eq(&self, other: &Self) -> bool {
        (0..N).all(|i| self[i].bit_eq(&other[i]))
    }
}

macro_rules! simd {
    ($T:ident, $N:literal, $Simd:ident) => {
        impl BitEq for $Simd {
            fn bit_eq(&self, other: &Self) -> bool {
                self.to_array().bit_eq(&other.to_array())
            }
        }
    };
}
simd!(f32, 4, f32x4);
simd!(f32, 8, f32x8);
simd!(f32, 16, f32x16);
simd!(f64, 2, f64x2);
simd!(f64, 4, f64x4);
simd!(f64, 8, f64x8);
simd!(i8, 16, i8x16);
simd!(i8, 32, i8x32);
simd!(i8, 64, i8x64);
simd!(i16, 8, i16x8);
simd!(i16, 16, i16x16);
simd!(i16, 32, i16x32);
simd!(i32, 4, i32x4);
simd!(i32, 8, i32x8);
simd!(i32, 16, i32x16);
simd!(i64, 2, i64x2);
simd!(i64, 4, i64x4);
simd!(i64, 8, i64x8);
simd!(u8, 16, u8x16);
simd!(u8, 32, u8x32);
simd!(u8, 64, u8x64);
simd!(u16, 8, u16x8);
simd!(u16, 16, u16x16);
simd!(u16, 32, u16x32);
simd!(u32, 4, u32x4);
simd!(u32, 8, u32x8);
simd!(u32, 16, u32x16);
simd!(u64, 2, u64x2);
simd!(u64, 4, u64x4);
simd!(u64, 8, u64x8);

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

    #[test]
    fn test_assert_bit_eq() {
        super::assert_bit_eq!(1.0, 1.0);
    }

    #[test]
    #[should_panic]
    fn test_assert_bit_eq_panic() {
        super::assert_bit_eq!(1.0, 2.0);
    }
}
