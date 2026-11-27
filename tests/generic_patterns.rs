//! Compilation tests for generic usage of [`Simd<T, N>`].
//!
//! The usage of type-system hacks in the internals of the crate can lead to
//! issues with the trait solver. During the initial implementation of generics,
//! the patterns tested in this module unexpectedly failed to compile.
//!
//! It is important to test these patterns in case a future change to how
//! generics are implemented accidentally breaks them.

use wide::{PrimitiveFloat, Simd, SimdElement, SupportsLaneCount};

fn _test_concrete(f: Simd<f32, 4>, i: Simd<i32, 4>, u: Simd<u32, 4>) {
    let _ = f.fast_reduce_max();
    let _ = i.abs() + u.cast_signed();
    let _ = u.saturating_add(f.to_bits());
}

fn _test_generic_over_t<T>(f: Simd<T, 4>, i: Simd<T::Signed, 4>, u: Simd<T::Unsigned, 4>)
where
    T: PrimitiveFloat + SupportsLaneCount<4>,
{
    let _ = f.fast_reduce_max();
    let _ = i.unsigned_abs() + u;
    let _ = u.saturating_add(f.to_bits());
}

fn _test_generic_over_n<const N: usize>(f: Simd<f32, N>, i: Simd<i32, N>, u: Simd<u32, N>)
where
    f32: SupportsLaneCount<N>,
{
    let _ = f.fast_reduce_max();
    let _ = i.abs() + u.cast_signed();
    let _ = u.saturating_add(f.to_bits());
}

fn _test_generic_over_t_n<T, const N: usize>(
    f: Simd<T, N>,
    i: Simd<T::Signed, N>,
    u: Simd<T::Unsigned, N>,
) where
    T: PrimitiveFloat + SupportsLaneCount<N>,
{
    let _ = f.fast_reduce_max();
    let _ = i.unsigned_abs() + u;
    let _ = u.saturating_add(f.to_bits());
}

fn _test_generic_over_t0_t1<T0, T1>(
    _: Simd<T0, 4>,
    i0: Simd<T0::Signed, 4>,
    u0: Simd<T0::Unsigned, 4>,
    _: Simd<T1, 4>,
    i1: Simd<T1::Signed, 4>,
    u1: Simd<T1::Unsigned, 4>,
) where
    T0: SimdElement + SupportsLaneCount<4>,
    T1: SimdElement + SupportsLaneCount<4>,
{
    let _ = i0.unsigned_abs() + u0;
    let _ = i1.unsigned_abs() + u1;
}

fn _test_generic_over_t0_t1_n<T0, T1, const N: usize>(
    _: Simd<T0, N>,
    i0: Simd<T0::Signed, N>,
    u0: Simd<T0::Unsigned, N>,
    _: Simd<T1, N>,
    i1: Simd<T1::Signed, N>,
    u1: Simd<T1::Unsigned, N>,
) where
    T0: SimdElement + SupportsLaneCount<N>,
    T1: SimdElement + SupportsLaneCount<N>,
{
    let _ = i0.unsigned_abs() + u0;
    let _ = i1.unsigned_abs() + u1;
}

fn _test_generic_over_t0_t1_associated_type_eq<T0, T1>(
    _: Simd<T0, 4>,
    i0: Simd<T0::Signed, 4>,
    u0: Simd<T0::Unsigned, 4>,
    _: Simd<T1, 4>,
    i1: Simd<T1::Signed, 4>,
    u1: Simd<T1::Unsigned, 4>,
) where
    T0: SimdElement + SupportsLaneCount<4>,
    T1: SimdElement<Unsigned = T0::Unsigned> + SupportsLaneCount<4>,
{
    let _ = u0 + i0.cast_unsigned() + i1.cast_unsigned() + u1;
}

fn _test_generic_over_t0_t1_n_associated_type_eq<T0, T1, const N: usize>(
    _: Simd<T0, N>,
    i0: Simd<T0::Signed, N>,
    u0: Simd<T0::Unsigned, N>,
    _: Simd<T1, N>,
    i1: Simd<T1::Signed, N>,
    u1: Simd<T1::Unsigned, N>,
) where
    T0: SimdElement + SupportsLaneCount<N>,
    T1: SimdElement<Unsigned = T0::Unsigned> + SupportsLaneCount<N>,
{
    let _ = u0 + i0.cast_unsigned() + i1.cast_unsigned() + u1;
}

// Ensure that `Simd::Bitmask` only depends on `N`, not `T`.

fn _test_generic_over_t0_t1_bitmask_eq<T0, T1>(t0: Simd<T0, 4>, t1: Simd<T1, 4>)
where
    T0: SimdElement + SupportsLaneCount<4>,
    T1: SimdElement + SupportsLaneCount<4>,
{
    let _: [u32; 2] = [t0.to_bitmask(), t1.to_bitmask()];
}

fn _test_generic_over_t0_t1_n_bitmask_eq<T0, T1, const N: usize>(t0: Simd<T0, N>, t1: Simd<T1, N>)
where
    T0: SimdElement + SupportsLaneCount<N>,
    T1: SimdElement + SupportsLaneCount<N>,
{
    let _ = [t0.to_bitmask(), t1.to_bitmask()];
}
