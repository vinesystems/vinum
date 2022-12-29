//! Filters.

pub mod design;

use crate::InvalidInput;
use lair::{Real, Scalar};
use ndarray::{Array1, Array2};
use std::cmp::{max, Ordering};

/// Applies a digital filter forward and backward to a signal.
///
/// # Errors
///
/// Returns [`InvalidInput::Shape`] if `x` is too short.
///
/// [`InvalidInput::Shape`]: ../enum.InvalidInput.html#variant.Shape
pub fn filtfilt<A>(b: &[A], a: &[A], x: &[A]) -> Result<Vec<A>, InvalidInput>
where
    A: Scalar,
    A::Real: Real,
{
    let ntaps = max(b.len(), a.len());
    let edge = ntaps * 3;
    if x.len() <= edge {
        return Err(InvalidInput::Shape(format!(
            "`x` should have more than {} elements",
            edge
        )));
    }
    let ext = odd_ext(x, edge);
    let zi = lfilter_zi(b, a);
    let vi = zi.iter().map(|v| *v * ext[0]).collect::<Vec<_>>();

    let (y, _) = lfilter(b, a, &ext, Some(&vi)); // forward filter
    let y_rev = y.into_iter().rev().collect::<Vec<_>>();

    let vi = zi.iter().map(|v| *v * y_rev[0]).collect::<Vec<_>>();
    let (y, _) = lfilter(b, a, y_rev.as_slice(), Some(&vi)); // backward filter
    let y_rev = y.into_iter().rev().collect::<Vec<_>>();
    Ok(y_rev[edge..y_rev.len() - edge].to_vec())
}

fn odd_ext<A>(x: &[A], edge: usize) -> Vec<A>
where
    A: Scalar,
{
    let mut ext = vec![A::zero(); x.len() + edge * 2];
    for (ev, xv) in &mut ext[0..edge].iter_mut().zip(x[1..=edge].iter().rev()) {
        *ev = x[0] + x[0] - *xv;
    }
    ext[edge..edge + x.len()].clone_from_slice(x);
    for (ev, xv) in &mut ext[edge + x.len()..]
        .iter_mut()
        .zip(x[x.len() - 1 - edge..x.len() - 1].iter().rev())
    {
        let last = x[x.len() - 1];
        *ev = (last + last) - *xv;
    }
    ext
}

/// Constructs initial conditions for `lfilter`.
///
/// # Panics
///
/// Panics if `a` is empty or `a[0] == 0`.
fn lfilter_zi<A>(b: &[A], a: &[A]) -> Vec<A>
where
    A: Scalar,
    A::Real: Real,
{
    assert!(a[0] != A::zero());

    let mut b = b.iter().map(|&v| v / a[0]).collect::<Vec<A>>();
    let mut a = a.iter().map(|&v| v / a[0]).collect::<Vec<A>>();
    match a.len().cmp(&b.len()) {
        Ordering::Less => a.extend((a.len()..b.len()).map(|_| A::zero())),
        Ordering::Greater => b.extend((b.len()..a.len()).map(|_| A::zero())),
        Ordering::Equal => {}
    }

    let c = lair::matrix::companion(&a).unwrap();
    let lhs = Array2::<A>::eye(a.len() - 1) - c.t();
    let rhs: Array1<A> = b[1..b.len()]
        .iter()
        .zip(a[1..a.len()].iter())
        .map(|(&bv, &av)| bv - av * b[0])
        .collect();
    let x = lair::equation::solve(&lhs, &rhs).unwrap();
    x.into_raw_vec()
}

/// Filters data `x` along one-dimension using a digital filter.
///
/// # Panics
///
/// Panics if `a` and `b` are empty, or `a[0] == 0`.
#[must_use]
fn lfilter<A>(b: &[A], a: &[A], x: &[A], vi: Option<&[A]>) -> (Vec<A>, Option<Vec<A>>)
where
    A: Scalar,
{
    assert!(!a.is_empty() || !b.is_empty());
    assert!(a[0] != A::zero());

    let mut vf = vi.map_or_else(Vec::new, <[A]>::to_vec);
    let mut y = vec![A::zero(); x.len()];

    raw_filter(b, a, x, vi, &mut vf, &mut y);

    if vi.is_some() {
        (y, Some(vf))
    } else {
        (y, None)
    }
}

fn raw_filter<A>(b: &[A], a: &[A], x: &[A], zi: Option<&[A]>, zf: &mut [A], y: &mut [A])
where
    A: Scalar,
{
    let nb = b.len();
    let na = a.len();
    let nfilt = max(nb, na);

    let mut numer: Vec<A> = Vec::with_capacity(nfilt);
    numer.extend(b);
    numer.resize(nfilt, A::zero());
    let mut denom: Vec<A> = Vec::with_capacity(nfilt);
    denom.extend(a);
    denom.resize(nfilt, A::zero());
    let mut zfzfilled: Vec<A> = Vec::with_capacity(nfilt - 1);

    if let Some(zi) = zi {
        if zi.len() < nfilt - 1 {
            zfzfilled.extend(zi);
            zfzfilled.resize(nfilt - 1, A::zero());
        } else {
            zfzfilled.extend(&zi[0..nfilt - 1]);
        }
    } else {
        zfzfilled.resize(nfilt - 1, A::zero());
    }

    linear_filter1(&mut numer, &mut denom, x, y, &mut zfzfilled);
    if zi.is_some() {
        if zf.len() >= zfzfilled.len() {
            zf.copy_from_slice(&zfzfilled);
        } else {
            zf.copy_from_slice(&zfzfilled[0..zf.len()]);
        }
    }
}

/// A 1-D linear filter.
///
/// # Panics
///
/// Panics if `numer`, `denom`, or 'z' is empty, or `a` or `z` is shorter than `b`.
fn linear_filter1<A>(numer: &mut [A], denom: &mut [A], x: &[A], y: &mut [A], z: &mut [A])
where
    A: Scalar,
{
    debug_assert_eq!(numer.len(), denom.len());
    let denom0 = denom[0];
    for (numer_val, denom_val) in numer.iter_mut().zip(denom.iter_mut()) {
        *numer_val /= denom0;
        *denom_val /= denom0;
    }

    for (xv, yv) in x.iter().zip(y.iter_mut()) {
        if numer.len() > 1 {
            *yv = z[0] + numer[0] * *xv;
            for i in 1..numer.len() - 1 {
                z[i - 1] = z[i] + *xv * numer[i] - *yv * denom[i];
            }
            z[numer.len() - 2] = *xv * numer[numer.len() - 1] - *yv * denom[numer.len() - 1];
        } else {
            *yv = *xv * numer[0];
        }
    }
}
#[cfg(test)]
mod test {
    use approx::abs_diff_eq;

    #[test]
    fn lfilter_zeroth() {
        let x = vec![1_f64, 2_f64, -1_f64, 3_f64, 0_f64, -1_f64];
        let (y, vf) = super::lfilter(&[1_f64], &[1_f64], &x, None);
        assert_eq!(y, x);
        assert!(vf.is_none());
    }

    #[test]
    fn lfilter_first() {
        use super::design::butter;
        let (b, a) = butter(1, 0.5);
        let (y, vf) = super::lfilter(&b, &a, &[1_f64, 2_f64, -1_f64, 3_f64, 0_f64, -1_f64], None);
        let filtered = &[
            0.499_999_999_999_999_94,
            1.499_999_999_999_999_8,
            0.5,
            0.999_999_999_999_999_9,
            1.499_999_999_999_999_8,
            -0.499_999_999_999_999_9,
        ];
        assert!(y
            .iter()
            .zip(filtered.iter())
            .all(|(y_elem, filtered_elem)| abs_diff_eq!(y_elem, filtered_elem, epsilon = 1e-8)));
        assert!(vf.is_none());
    }

    #[test]
    fn lfilter_second() {
        use super::design::butter;
        let (b, a) = butter(2, 0.2);
        let (y, vf) = super::lfilter(&b, &a, &[1_f64, 2_f64, -1_f64, 3_f64, 0_f64, -1_f64], None);
        let filtered = &[
            0.067_455_273_889_071_9,
            0.346_921_158_404_985_6,
            0.638_499_570_670_317_6,
            0.788_948_773_220_527_4,
            0.975_455_791_582_758_9,
            0.924_158_184_245_401_2,
        ];
        assert!(y
            .iter()
            .zip(filtered.iter())
            .all(|(y_elem, filtered_elem)| abs_diff_eq!(y_elem, filtered_elem, epsilon = 1e-8)));
        assert!(vf.is_none());
    }

    #[test]
    fn lfilter_third() {
        use super::design::butter;
        let (b, a) = butter(3, 0.3);
        let (y, vf) = super::lfilter(&b, &a, &[1_f64, 2_f64, -1_f64, 3_f64, 0_f64, -1_f64], None);
        let filtered = &[
            0.049_532_996_357_253_195,
            0.305_218_236_272_406_7,
            0.716_430_245_930_999_1,
            0.973_573_112_644_040_6,
            1.070_928_413_784_704_6,
            1.012_206_630_203_253_7,
        ];
        assert!(y
            .iter()
            .zip(filtered.iter())
            .all(|(y_elem, filtered_elem)| abs_diff_eq!(y_elem, filtered_elem, epsilon = 1e-8)));
        assert!(vf.is_none());
    }

    #[test]
    fn lfilter_fourth() {
        use super::design::butter;
        let (b, a) = butter(4, 0.1);
        let (y, vf) = super::lfilter(&b, &a, &[1_f64, 2_f64, -1_f64, 3_f64, 0_f64, -1_f64], None);
        let filtered = &[
            0.000_416_599_204_406_599_37,
            0.003_824_646_715_405_765,
            0.015_972_037_942_282_198,
            0.043_162_485_605_509_74,
            0.089_757_778_135_338_61,
            0.157_136_507_891_160_93,
        ];
        assert!(y
            .iter()
            .zip(filtered.iter())
            .all(|(y_elem, filtered_elem)| abs_diff_eq!(y_elem, filtered_elem, epsilon = 1e-8)));
        assert!(vf.is_none());
    }

    #[test]
    fn odd_ext() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(
            super::odd_ext(&x, 2),
            vec![-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        );
    }

    #[test]
    fn lfilter() {
        use super::design::butter;
        let (b, a) = butter(5, 0.25);
        let zi = super::lfilter_zi(&b, &a);
        assert!(zi
            .iter()
            .zip(
                vec![
                    0.996_720_783_693_642_1,
                    -1.494_091_472_816_327,
                    1.284_122_676_031_658_2,
                    -0.452_441_727_947_415_05,
                    0.075_594_885_409_318_84
                ]
                .iter()
            )
            .all(|(p, q)| abs_diff_eq!(*p, *q, epsilon = 1e-8)));
        let x = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let (y, _) = super::lfilter(&b, &a, &x, Some(&zi));
        assert!(y
            .iter()
            .zip(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0].iter(),)
            .all(|(p, q)| abs_diff_eq!(*p, *q, epsilon = 1e-8)));

        let x = vec![0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0];
        let zi = zi.into_iter().map(|v| v * x[0]).collect::<Vec<_>>();
        let (y, _) = super::lfilter(&b, &a, &x, Some(&zi));
        assert!(y
            .iter()
            .zip(
                vec![
                    0.5,
                    0.5,
                    0.5,
                    0.498_360_39,
                    0.486_105_28,
                    0.443_993_89,
                    0.355_052_41,
                ]
                .iter(),
            )
            .all(|(p, q)| abs_diff_eq!(*p, *q, epsilon = 1e-8)));
    }
}
