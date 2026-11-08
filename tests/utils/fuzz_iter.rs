use wide::{
    f32x4, f32x8, f32x16, f64x2, f64x4, f64x8, i8x16, i8x32, i8x64, i16x8, i16x16, i16x32, i32x4,
    i32x8, i32x16, i64x2, i64x4, i64x8, u8x16, u8x32, u8x64, u16x8, u16x16, u16x32, u32x4, u32x8,
    u32x16, u64x2, u64x4, u64x8,
};

const ITERATIONS: usize = 1000;

/// An iterator used to fuzz test `wide` functions.
///
/// This aims to cover most kinds of values (and combinations of values, for
/// tuple and array `T` types) that may trigger numerical bugs. For most `wide`
/// functions, this iterator is sufficient enough that hand-written coverage is
/// unnecessary.
///
/// For functions with unique edge cases, consider checking if this iterator
/// covers those edge cases (via `.filter(_)`) then updating its implementation
/// if necessary.
#[expect(private_bounds)]
pub fn fuzz_iter<T>() -> impl Iterator<Item = T>
where
    T: Fuzz,
{
    // Constants for the LCG
    const SEED: u64 = 0x123456789abcdef0;
    const A: u64 = 6364136223846793005;
    const C: u64 = 1442695040888963407;

    let mut state = SEED;
    let mut random_bits = move || {
        // Update the state and calculate the next number (rotate to avoid lack of
        // randomness in low bits)
        state = state.wrapping_mul(A).wrapping_add(C).rotate_left(31);

        state
    };

    (0..ITERATIONS).map(move |iteration| T::get(iteration, &mut random_bits))
}

trait Fuzz {
    /// Returns the value for one iteration.
    ///
    /// `iteration` is used to select a different distribution for different
    /// ranges in `0..ITERATIONS`. Using multiple distributions is especially
    /// important for functions that take multiple parameters (e.g., `mul_add`),
    /// where one distribution cannot generate all required kinds of value
    /// combinations.
    ///
    /// `random_bits` returns a different value every call, and attempts to have
    /// uniform distribution.
    fn get(iteration: usize, random_bits: &mut impl FnMut() -> u64) -> Self;
}

macro_rules! float {
    (
        $T:ident,
        TUnsigned = $TUnsigned:ident,
        TSigned = $TSigned:ident,
        MANTISSA_BITS = $MANTISSA_BITS:literal,
        MANTISSA_MASK = $MANTISSA_MASK:literal,
        EXPONENT_MASK = $EXPONENT_MASK:literal,
        SIGN_MASK = $SIGN_MASK:literal,
    ) => {
        impl Fuzz for $T {
            fn get(iteration: usize, random_bits: &mut impl FnMut() -> u64) -> Self {
                /// Represents what kind of random value to generate.
                enum Kind {
                    AnyBitPattern,
                    /// Any value with the exponent bits of `x`. Where `x` is a
                    /// power of two, such values range from `x` to `2x`.
                    ExponentOf($T),
                    PowerOfTwo,
                    /// Halfway between two integers, meaning
                    /// `x.fract().abs() == 0.5`.
                    HalfWay,
                    Subnormal,
                    Nan,
                    SpecialConstant,
                }

                const SPECIAL_CONSTANTS: [$T; 11] = [
                    0.0,
                    0.5,
                    1.0,
                    2.0,
                    3.0,
                    10.0,
                    $T::MAX,
                    $T::MIN_POSITIVE,
                    $T::INFINITY,
                    $TUnsigned::MAX as $T,
                    $TSigned::MAX as $T,
                ];

                // Use multiple distributions, each aiming to cover a different
                // kind of value combination.
                let kind = match iteration {
                    0..200 => Kind::AnyBitPattern,
                    // Lower exponents are important for functions like
                    // `asin_acos` and `exp`, where higher exponents return
                    // fallback values such as NaN.
                    200..300 => match random_bits() % 7 {
                        0 => Kind::ExponentOf(0.125),
                        1 => Kind::ExponentOf(0.25),
                        2 => Kind::ExponentOf(0.5),
                        3 => Kind::ExponentOf(1.0),
                        4 => Kind::ExponentOf(1e2),
                        5 => Kind::ExponentOf(1e4),
                        6 => Kind::ExponentOf(1e6),
                        7.. => unreachable!(),
                    },
                    // Mixing special values with ordinary values is
                    // important for functions such as `mul_add` and `powf`.
                    300..700 => match random_bits() % 80 {
                        0..50 => Kind::SpecialConstant,
                        50..55 => Kind::PowerOfTwo,
                        55..60 => Kind::HalfWay,
                        60..73 => Kind::AnyBitPattern,
                        73 => Kind::ExponentOf(0.125),
                        74 => Kind::ExponentOf(0.25),
                        75 => Kind::ExponentOf(0.5),
                        76 => Kind::ExponentOf(1.0),
                        77 => Kind::ExponentOf(1e2),
                        78 => Kind::ExponentOf(1e4),
                        79 => Kind::ExponentOf(1e6),
                        5.. => unreachable!(),
                    },
                    // Mixing NaNs and subnormals is best done separately from
                    // the previous step so that many non-NaN, non-subnormal
                    // cases are covered.
                    700..1000 => match random_bits() % 100 {
                        0..20 => Kind::Nan,
                        20..40 => Kind::Subnormal,
                        40..55 => Kind::SpecialConstant,
                        55..70 => Kind::PowerOfTwo,
                        70..85 => Kind::HalfWay,
                        85..93 => Kind::AnyBitPattern,
                        93 => Kind::ExponentOf(0.125),
                        94 => Kind::ExponentOf(0.25),
                        95 => Kind::ExponentOf(0.5),
                        96 => Kind::ExponentOf(1.0),
                        97 => Kind::ExponentOf(1e2),
                        98 => Kind::ExponentOf(1e4),
                        99 => Kind::ExponentOf(1e6),
                        5.. => unreachable!(),
                    },
                    ITERATIONS.. => unreachable!(),
                };

                let result = match kind {
                    Kind::AnyBitPattern => Self::from_bits(random_bits() as $TUnsigned),
                    Kind::ExponentOf(x) => Self::from_bits(
                        (random_bits() as $TUnsigned & !$EXPONENT_MASK)
                            + (x.to_bits() & $EXPONENT_MASK),
                    ),
                    Kind::PowerOfTwo => {
                        Self::from_bits(random_bits() as $TUnsigned & !$MANTISSA_MASK)
                    }
                    Kind::HalfWay => {
                        let bits = random_bits() as $TUnsigned;

                        // Only pick exponents where `x.fract() == 0.5` is representable
                        let exponent_bits = ((bits % $T::MANTISSA_DIGITS as $TUnsigned)
                            << $MANTISSA_BITS)
                            + ((0.5 as $T).to_bits() & $EXPONENT_MASK);

                        let base = $T::from_bits((bits & !$EXPONENT_MASK) | exponent_bits);

                        base.floor() + 0.5
                    }
                    Kind::Subnormal => {
                        Self::from_bits(random_bits() as $TUnsigned & !$EXPONENT_MASK)
                    }
                    Kind::Nan => Self::from_bits(random_bits() as $TUnsigned | $EXPONENT_MASK),
                    Kind::SpecialConstant => {
                        let bits = random_bits();
                        let sign = if bits & 0b1000 != 0 { -1.0 } else { 1.0 };

                        SPECIAL_CONSTANTS[bits as usize % SPECIAL_CONSTANTS.len()].copysign(sign)
                    }
                };

                // Cover values that are steps near constants, power of twos and
                // halfway values. Multiple functions have edge cases for such
                // values.
                match random_bits() % 10 {
                    0..4 => result,
                    4 => result.next_up(),
                    5 => result.next_down(),
                    6 => result.next_up().next_up(),
                    7 => result.next_down().next_down(),
                    8 => result.next_up().next_up().next_up(),
                    9 => result.next_down().next_down().next_down(),
                    10.. => unreachable!(),
                }
            }
        }
    };
}
float!(
    f32,
    TUnsigned = u32,
    TSigned = i32,
    MANTISSA_BITS = 23,
    MANTISSA_MASK = 0x007f_ffff,
    EXPONENT_MASK = 0x7f80_0000,
    SIGN_MASK = 0x8000_0000,
);
float!(
    f64,
    TUnsigned = u64,
    TSigned = i64,
    MANTISSA_BITS = 52,
    MANTISSA_MASK = 0x000f_ffff_ffff_ffff,
    EXPONENT_MASK = 0x7ff0_0000_0000_0000,
    SIGN_MASK = 0x8000_0000_0000_0000,
);

macro_rules! integer {
    ($T:ident) => {
        impl Fuzz for $T {
            fn get(iteration: usize, random_bits: &mut impl FnMut() -> u64) -> Self {
                match iteration {
                    // Distribute based on exponent in order to cover
                    // non-overflowing cases for functions such as `mul` and
                    // `zeroing_shuffle`.
                    0..500 => (random_bits() as Self).wrapping_shr(random_bits() as u32),
                    // Cover large values using uniform distribution in order to
                    // cover overflow cases for functions such as `add`.
                    500..1000 => random_bits() as Self,
                    ITERATIONS.. => unreachable!(),
                }
            }
        }
    };
}
integer!(i8);
integer!(i16);
integer!(i32);
integer!(i64);
integer!(u8);
integer!(u16);
integer!(u32);
integer!(u64);
integer!(usize);

impl<T, const N: usize> Fuzz for [T; N]
where
    T: Fuzz,
{
    fn get(iteration: usize, random_bits: &mut impl FnMut() -> u64) -> Self {
        std::array::from_fn(|_| T::get(iteration, random_bits))
    }
}

impl<T0, T1> Fuzz for (T0, T1)
where
    T0: Fuzz,
    T1: Fuzz,
{
    fn get(iteration: usize, random_bits: &mut impl FnMut() -> u64) -> Self {
        (
            T0::get(iteration, random_bits),
            T1::get(iteration, random_bits),
        )
    }
}

macro_rules! simd {
    ($T:ident, $N:literal, $Simd:ident) => {
        impl Fuzz for $Simd {
            fn get(iteration: usize, random_bits: &mut impl FnMut() -> u64) -> Self {
                Self::from_array(<[$T; $N]>::get(iteration, random_bits))
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
