//! Tools to design filters.

use num_complex::Complex;
use num_traits::{Float, FloatConst, NumAssign, One, Pow};
use std::convert::TryFrom;
use std::f64::consts::PI;
use std::iter::Sum;
use std::ops::{Mul, Neg};

/// Designs a digital or analog Butterworth filter and returns the filter
/// coefficients.
#[must_use]
pub fn butter<T>(order: i32, cutoff_freq: T) -> (Vec<T>, Vec<T>)
where
    T: Sum + Float + FloatConst + NumAssign + Pow<i32, Output = T>,
{
    let mut prototype = buttap(order);
    let fs = T::one() + T::one();
    let fs2 = fs + fs;
    let warped = T::from(4).expect("T can represent 4") * (T::PI() / fs * cutoff_freq).tan();

    // Transform a low-pass filter prototype to a different frequency.
    let degree =
        i32::try_from(prototype.len()).expect("the prototype filter should have the same order");
    let mut k = warped.pow(degree);
    for v in &mut prototype {
        *v *= warped;
    }

    // Find discrete equivalent.
    let fs2 = Complex::<T>::from(fs2);
    let prod: Complex<T> = prototype.iter().map(|&v| fs2 - v).product();
    k *= (Complex::<T>::one() / prod).re;
    for v in &mut prototype {
        *v = (fs2 + *v) / (fs2 - *v);
    }
    let z = (0..degree).map(|_| -T::one()).collect::<Vec<_>>();

    // Return polynomial transfer function representation from zeros and poles.
    let b = poly(&z).into_iter().map(|v| v * k).collect::<Vec<_>>();
    let a = poly(&prototype).into_iter().map(|v| v.re).collect();
    (b, a)
}

/// Returns `(p, k)` for analog prototype of nth-order Butterworth filter.
#[allow(clippy::cast_precision_loss)] // `order` is between `-u32::max_value()` and `u32::max_value()`.
fn buttap<A>(order: i32) -> Vec<Complex<A>>
where
    A: Float,
{
    (1 - order..order)
        .step_by(2)
        .map(|v| {
            -(Complex::i()
                * A::from(PI).expect("float-to-float conversion")
                * Complex::from(A::from(v).expect("approximation"))
                / A::from(2 * order).expect("approximation"))
            .exp()
        })
        .collect::<Vec<_>>()
}

/// Finds the coefficients of a polynomial with the given sequence of roots.
fn poly<T>(roots: &[T]) -> Vec<T>
where
    T: Copy + Mul + Sum<<T as Mul>::Output> + One + Neg,
    <T as Neg>::Output: Into<T>,
{
    let mut a = vec![T::one()];
    for root in roots {
        a = convolve(&a, &[T::one(), (-*root).into()]);
    }
    a
}

/// Returns the discrete, linear convolution of two one-dimensional sequences.
fn convolve<T>(f: &[T], g: &[T]) -> Vec<T>
where
    T: Copy + Mul + Sum<<T as Mul>::Output>,
{
    let (f, g) = if f.len() >= g.len() { (f, g) } else { (g, f) };
    (0..f.len() + g.len() - 1)
        .map(|n| {
            let f_begin = if n < g.len() { 0 } else { n - g.len() + 1 };
            let g_end = if n < g.len() { n + 1 } else { g.len() };
            f[f_begin..]
                .iter()
                .zip(g[..g_end].iter().rev())
                .map(|(fv, gv)| *fv * *gv)
                .sum()
        })
        .collect()
}

#[cfg(test)]
mod test {
    use num_complex::Complex;

    #[test]
    fn buttap() {
        let p = super::buttap(4);
        assert_eq!(
            p,
            vec![
                Complex::new(-0.38268343236508984, 0.9238795325112867),
                Complex::new(-0.9238795325112867, 0.3826834323650898),
                Complex::new(-0.9238795325112867, -0.3826834323650898),
                Complex::new(-0.38268343236508984, -0.9238795325112867),
            ]
        );
    }

    #[test]
    fn butter() {
        let (b, a) = super::butter(4, 0.01);
        assert_eq!(
            b,
            vec![
                5.845142433144486e-08,
                2.3380569732577944e-07,
                3.5070854598866915e-07,
                2.3380569732577944e-07,
                5.845142433144486e-08,
            ]
        );
        assert_eq!(
            a,
            vec![
                1.0,
                -3.917907865391987,
                5.757076379118066,
                -3.7603495076945266,
                0.9211819291912362
            ]
        );
    }

    #[test]
    fn convolve() {
        let f = vec![
            Complex::new(1_f64, 0_f64),
            Complex::new(2_f64, 0_f64),
            Complex::new(3_f64, 0_f64),
        ];
        let g = vec![
            Complex::new(4_f64, 0_f64),
            Complex::new(5_f64, 0_f64),
            Complex::new(6_f64, 0_f64),
        ];
        assert_eq!(
            super::convolve(&f, &g),
            vec![
                Complex::new(4_f64, 0_f64),
                Complex::new(13_f64, 0_f64),
                Complex::new(28_f64, 0_f64),
                Complex::new(27_f64, 0_f64),
                Complex::new(18_f64, 0_f64)
            ]
        );
    }

    #[test]
    fn poly() {
        let roots = vec![
            Complex::new(1_f64, 1_f64),
            Complex::new(0_f64, 0_f64),
            Complex::new(-1_f64, -1_f64),
        ];
        let coef = super::poly(&roots);
        assert_eq!(
            coef,
            vec![
                Complex::new(1_f64, 0_f64),
                Complex::new(0_f64, 0_f64),
                Complex::new(0_f64, -2_f64),
                Complex::new(0_f64, 0_f64)
            ]
        );
    }
}
