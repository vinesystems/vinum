use std::ops::{Add, Mul, Sub};

use num_traits::Float;

use crate::InvalidInput;

/// Filters a real signal through cascaded second-order sections, starting at rest.
///
/// Each section is `[b0, b1, b2, a0, a1, a2]`. Coefficients are normalized by
/// `a0`. The returned signal has the same length as `x`.
///
/// # Errors
///
/// Returns [`InvalidInput::Shape`] for no sections, or [`InvalidInput::Value`]
/// for non-finite coefficients or a zero `a0`.
// The from-rest wrapper is only needed by single-pass reference tests.
#[cfg(test)]
fn sosfilt<T: Float>(sos: &[[T; 6]], x: &[T]) -> Result<Vec<T>, InvalidInput> {
    let sections = normalize(sos)?;
    let mut y = x.to_vec();
    filter(&sections, &mut y, &mut vec![[T::zero(); 2]; sections.len()]);
    Ok(y)
}

/// Applies a digital filter forward and backward using second-order sections.
///
/// Sections contain real coefficients; the signal may be real or complex, with
/// the same underlying floating-point precision as the coefficients.
/// Each section is `[b0, b1, b2, a0, a1, a2]`. Uses odd reflection padding and
/// steady-state initial conditions. The default
/// padding length is `3 * (2 * sos.len() + 1 - min(z, p))`, where `z` and `p`
/// count sections with zero `b2` and zero `a2`, respectively. This accounts for
/// the first-order section of an odd-order filter.
///
/// # Errors
///
/// Returns [`InvalidInput::Shape`] for no sections or a signal no longer than
/// the padding. Returns [`InvalidInput::Value`] for invalid coefficients or
/// non-finite/singular steady-state initialization. Floating-point accuracy
/// still limits very extreme filter designs; success is not an accuracy guarantee.
pub fn filtfilt<T, A>(sos: &[[T; 6]], x: &[A]) -> Result<Vec<A>, InvalidInput>
where
    T: Float,
    A: Copy + Add<Output = A> + Sub<Output = A> + Mul<T, Output = A>,
{
    let sections = normalize(sos)?;
    let zeros = sections.iter().filter(|s| s[2] == T::zero()).count();
    let poles = sections.iter().filter(|s| s[5] == T::zero()).count();
    let edge = 3 * (2 * sections.len() + 1 - zeros.min(poles));
    if x.len() <= edge {
        return Err(InvalidInput::Shape(format!(
            "`x` should have more than {edge} elements"
        )));
    }
    let zi = initial_state(&sections)?;
    let mut y = Vec::with_capacity(x.len() + 2 * edge);
    for &v in x[1..=edge].iter().rev() {
        y.push(x[0] + x[0] - v);
    }
    y.extend_from_slice(x);
    let last = x[x.len() - 1];
    for &v in x[x.len() - 1 - edge..x.len() - 1].iter().rev() {
        y.push(last + last - v);
    }
    let scaled_state = |value: A| {
        zi.iter()
            .map(|s| [value * s[0], value * s[1]])
            .collect::<Vec<_>>()
    };
    let mut state = scaled_state(y[0]);
    filter(&sections, &mut y, &mut state);
    y.reverse();
    let mut state = scaled_state(y[0]);
    filter(&sections, &mut y, &mut state);
    y.reverse();
    Ok(y[edge..y.len() - edge].to_vec())
}

fn normalize<T: Float>(sos: &[[T; 6]]) -> Result<Vec<[T; 6]>, InvalidInput> {
    if sos.is_empty() {
        return Err(InvalidInput::Shape("expected at least one section".into()));
    }
    sos.iter()
        .map(|section| {
            if section.iter().any(|v| !v.is_finite()) || section[3] == T::zero() {
                return Err(InvalidInput::Value(
                    "section coefficients must be finite and a0 nonzero".into(),
                ));
            }
            let mut normalized = *section;
            for v in &mut normalized {
                *v = *v / section[3];
            }
            if normalized.iter().any(|v| !v.is_finite()) {
                return Err(InvalidInput::Value(
                    "section normalization overflowed".into(),
                ));
            }
            Ok(normalized)
        })
        .collect()
}

fn initial_state<T: Float>(sos: &[[T; 6]]) -> Result<Vec<[T; 2]>, InvalidInput> {
    let mut scale = T::one();
    let mut zi = Vec::with_capacity(sos.len());
    for s in sos {
        let denom = (s[3] + s[4]) + s[5];
        if denom == T::zero() || !denom.is_finite() {
            return Err(InvalidInput::Value(
                "section has singular steady-state initialization".into(),
            ));
        }
        let gain = ((s[0] + s[1]) + s[2]) / denom;
        let state = [scale * (gain - s[0]), scale * (s[2] - s[5] * gain)];
        scale = scale * gain;
        if !scale.is_finite() || state.iter().any(|v| !v.is_finite()) {
            return Err(InvalidInput::Value(
                "section has non-finite steady-state initialization".into(),
            ));
        }
        zi.push(state);
    }
    Ok(zi)
}

fn filter<T, A>(sos: &[[T; 6]], x: &mut [A], state: &mut [[A; 2]])
where
    T: Float,
    A: Copy + Add<Output = A> + Sub<Output = A> + Mul<T, Output = A>,
{
    // Transposed direct form II, with only two state values per section.
    for sample in x {
        for (s, z) in sos.iter().zip(state.iter_mut()) {
            let input = *sample;
            let output = input * s[0] + z[0];
            z[0] = input * s[1] - output * s[4] + z[1];
            z[1] = input * s[2] - output * s[5];
            *sample = output;
        }
    }
}

#[cfg(test)]
mod tests {
    use num_complex::{Complex32, Complex64};

    use super::{filtfilt, sosfilt};
    use crate::signal::filter::design::butter;
    use crate::InvalidInput;

    fn assert_close(actual: &[f64], expected: &[f64], tolerance: f64) {
        assert_eq!(actual.len(), expected.len());
        for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
            assert!((a - e).abs() < tolerance, "sample {}: {} != {}", i, a, e);
        }
    }

    #[test]
    fn preserves_low_order_filter_reference_outputs() {
        // Reference outputs retained from the original polynomial filter tests.
        let x = [1.0, 2.0, -1.0, 3.0, 0.0, -1.0];
        let cases = [
            (1, 0.5, [0.5, 1.5, 0.5, 1.0, 1.5, -0.5]),
            (
                2,
                0.2,
                [
                    0.0674552738890719,
                    0.3469211584049856,
                    0.6384995706703176,
                    0.7889487732205274,
                    0.9754557915827589,
                    0.9241581842454012,
                ],
            ),
            (
                3,
                0.3,
                [
                    0.049532996357253195,
                    0.3052182362724067,
                    0.7164302459309991,
                    0.9735731126440406,
                    1.0709284137847046,
                    1.0122066302032537,
                ],
            ),
            (
                4,
                0.1,
                [
                    0.00041659920440659937,
                    0.003824646715405765,
                    0.015972037942282198,
                    0.04316248560550974,
                    0.08975777813533861,
                    0.15713650789116093,
                ],
            ),
        ];
        for (order, cutoff, expected) in cases {
            let sos = butter(order, cutoff).unwrap();
            assert_close(&sosfilt(&sos, &x).unwrap(), &expected, 1e-12);
        }
    }

    #[test]
    fn matches_scipy_reference_for_odd_and_even_orders() {
        // SciPy 1.18.1: x = sin(arange(32) * 0.17); x[10] += 2;
        // sos = signal.butter(order, 0.2, output="sos");
        // signal.sosfilt(sos, x) and signal.sosfiltfilt(sos, x).
        // Gain and zero distribution among sections may differ; the cascade
        // response must agree, including the default odd-order padding.
        let mut x = (0..32)
            .map(|i| (f64::from(i) * 0.17).sin())
            .collect::<Vec<_>>();
        x[10] += 2.0;
        let indices = [0, 1, 2, 4, 8, 16, 24, 31];
        let cases = [
            (
                3,
                [
                    0.0,
                    0.0030620200018174827,
                    0.020611103986627077,
                    0.15658823441020825,
                    0.7182798284424632,
                    1.0205274360544825,
                    -0.3870641092729912,
                    -1.0018134354064776,
                ],
                [
                    0.00029341332985489965,
                    0.14639414929865424,
                    0.29122106522773294,
                    0.5866259863261547,
                    1.265623807096476,
                    0.372281665836048,
                    -0.8063838923587108,
                    -0.8411837893966629,
                ],
            ),
            (
                4,
                [
                    0.0,
                    0.0008161937419641907,
                    0.0068076128936751315,
                    0.0766600308109544,
                    0.5773094361348995,
                    1.2817118904703124,
                    -0.24531928619190163,
                    -0.999076603171238,
                ],
                [
                    0.0015078659158119564,
                    0.13384563218957224,
                    0.2698156484399371,
                    0.571801948212985,
                    1.276923964717421,
                    0.3615973960730461,
                    -0.802229764475924,
                    -0.8468189207665413,
                ],
            ),
        ];
        for (order, forward, both) in cases {
            let sos = butter(order, 0.2).unwrap();
            let y = sosfilt(&sos, &x).unwrap();
            assert_close(
                &indices.iter().map(|&i| y[i]).collect::<Vec<_>>(),
                &forward,
                1e-12,
            );
            let y = filtfilt(&sos, &x).unwrap();
            assert_close(
                &indices.iter().map(|&i| y[i]).collect::<Vec<_>>(),
                &both,
                1e-12,
            );
        }
    }

    #[test]
    fn issue_two_small_cutoffs_preserve_constant_signals() {
        for (order, cutoff, tolerance) in [(3, 1e-6, 1e-4), (8, 2.5e-4, 1e-8), (8, 0.01, 1e-10)] {
            let sos = butter(order, cutoff).unwrap();
            let y = filtfilt(&sos, &[1.0; 100]).unwrap();
            assert_close(&y, &[1.0; 100], tolerance);
        }
    }

    #[test]
    fn scalar_filters_and_normalization() {
        let x = (0..20).map(f64::from).collect::<Vec<_>>();
        let identity = butter(0, 0.01).unwrap();
        assert_eq!(sosfilt(&identity, &x).unwrap(), x);
        assert_eq!(filtfilt(&identity, &x).unwrap(), x);
        let gain = [[4.0, 0.0, 0.0, 2.0, 0.0, 0.0]];
        assert_eq!(
            sosfilt(&gain, &x).unwrap(),
            x.iter().map(|v| 2.0 * v).collect::<Vec<_>>()
        );
        assert_eq!(
            filtfilt(&gain, &x).unwrap(),
            x.iter().map(|v| 4.0 * v).collect::<Vec<_>>()
        );
        let sos = butter(5, 0.2).unwrap();
        let scaled = sos.iter().map(|s| s.map(|v| v * 2.0)).collect::<Vec<_>>();
        assert_close(
            &filtfilt(&scaled, &x).unwrap(),
            &filtfilt(&sos, &x).unwrap(),
            1e-12,
        );
        assert!(sosfilt(&identity, &[]).unwrap().is_empty());
    }

    #[test]
    fn odd_and_even_padding_boundaries() {
        for order in 1..=8 {
            let sos = butter(order, 0.2_f64).unwrap();
            let edge = 3 * (order as usize + 1);
            assert!(matches!(
                filtfilt(&sos, &vec![1.0; edge]),
                Err(InvalidInput::Shape(_))
            ));
            assert_close(
                &filtfilt(&sos, &vec![1.0; edge + 1]).unwrap(),
                &vec![1.0; edge + 1],
                1e-12,
            );
        }
    }

    #[test]
    fn rejects_invalid_sections_and_initialization() {
        assert!(matches!(
            sosfilt::<f64>(&[], &[]),
            Err(InvalidInput::Shape(_))
        ));
        assert!(matches!(
            filtfilt::<f64, f64>(&[], &[]),
            Err(InvalidInput::Shape(_))
        ));
        for section in [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [f64::NAN, 0.0, 0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 1.0, f64::INFINITY, 0.0],
            [f64::MAX, 0.0, 0.0, f64::MIN_POSITIVE, 0.0, 0.0],
        ] {
            assert!(matches!(
                sosfilt(&[section], &[1.0; 100]),
                Err(InvalidInput::Value(_))
            ));
            assert!(matches!(
                filtfilt(&[section], &[1.0; 100]),
                Err(InvalidInput::Value(_))
            ));
        }
        // An integrator has no steady-state step response.
        assert!(matches!(
            filtfilt(&[[1.0, 0.0, 0.0, 1.0, -1.0, 0.0]], &[1.0; 100]),
            Err(InvalidInput::Value(_))
        ));
        // Finite coefficients can still overflow the DC gain.
        assert!(matches!(
            filtfilt(&[[f64::MAX, f64::MAX, 0.0, 1.0, 0.0, 0.0]], &[1.0; 100]),
            Err(InvalidInput::Value(_))
        ));
    }

    #[test]
    fn supports_single_precision() {
        let sos = butter(5, 0.2_f32).unwrap();
        let y = filtfilt(&sos, &[1.0_f32; 100]).unwrap();
        assert!(y.iter().all(|v| (v - 1.0).abs() < 1e-5));
    }

    #[test]
    fn complex_signals_match_scipy_reference() {
        // SciPy 1.18.1: x = sin(arange(32)*.17) + 1j*cos(arange(32)*.23);
        // x[10] += 2-1j; sosfiltfilt(butter(order, .2, output="sos"), x).
        let mut x = (0..32)
            .map(|i| Complex64::new((f64::from(i) * 0.17).sin(), (f64::from(i) * 0.23).cos()))
            .collect::<Vec<_>>();
        x[10] += Complex64::new(2.0, -1.0);
        let indices = [0, 1, 2, 4, 8, 16, 24, 31];
        let cases = [
            (
                3,
                [
                    (0.00029341332985453883, 0.9989454928617023),
                    (0.14639414929865366, 0.9476450970619033),
                    (0.2912210652277323, 0.8767111081408472),
                    (0.5866259863261544, 0.6160813909386296),
                    (1.2656238070964765, -0.4016320792342168),
                    (0.3722816658360476, -0.8386130840618946),
                    (-0.8063838923587106, 0.7263325668599968),
                    (-0.8411837893966627, 0.6508425462466358),
                ],
            ),
            (
                4,
                [
                    (0.001507865915811762, 0.9992986239448803),
                    (0.13384563218957202, 0.9565714680377573),
                    (0.269815648439937, 0.8919144604306882),
                    (0.5718019482129854, 0.6308717454847991),
                    (1.2769239647174213, -0.40770168755075614),
                    (0.36159739607304603, -0.8330220903164268),
                    (-0.8022297644759238, 0.7249433242843912),
                    (-0.8468189207665416, 0.6586599436732372),
                ],
            ),
        ];
        for (order, expected) in cases {
            let y = filtfilt(&butter(order, 0.2_f64).unwrap(), &x).unwrap();
            assert_eq!(y.len(), x.len());
            for (&i, &(re, im)) in indices.iter().zip(&expected) {
                assert!((y[i] - Complex64::new(re, im)).norm() < 1e-12);
            }
        }
    }

    #[test]
    fn complex_single_precision_filters_both_components() {
        let x = (0..100)
            .map(|i| Complex32::new((i as f32 * 0.17).sin(), (i as f32 * 0.23).cos()))
            .collect::<Vec<_>>();
        let sos = butter(5, 0.2_f32).unwrap();
        let y = filtfilt(&sos, &x).unwrap();
        let re = filtfilt(&sos, &x.iter().map(|v| v.re).collect::<Vec<_>>()).unwrap();
        let im = filtfilt(&sos, &x.iter().map(|v| v.im).collect::<Vec<_>>()).unwrap();
        for ((actual, re), im) in y.iter().zip(re).zip(im) {
            assert!((actual.re - re).abs() < 1e-5);
            assert!((actual.im - im).abs() < 1e-5);
        }
    }

    #[test]
    fn complex_scalar_filters_normalize_and_propagate_errors() {
        let x = [Complex64::new(2.0, -3.0); 100];
        assert_eq!(filtfilt(&butter(0, 0.2_f64).unwrap(), &x).unwrap(), x);
        let gain = [[4.0, 0.0, 0.0, 2.0, 0.0, 0.0]];
        assert_eq!(
            filtfilt(&gain, &x).unwrap(),
            vec![Complex64::new(8.0, -12.0); 100]
        );
        let sos = butter(5, 0.2_f64).unwrap();
        let scaled = sos.iter().map(|s| s.map(|v| v * 2.0)).collect::<Vec<_>>();
        for value in filtfilt(&scaled, &x).unwrap() {
            assert!((value - x[0]).norm() < 1e-12);
        }
        assert!(matches!(
            filtfilt(&sos, &x[..3]),
            Err(InvalidInput::Shape(_))
        ));
        assert!(matches!(
            filtfilt(&[[1.0, 0.0, 0.0, 1.0, -1.0, 0.0]], &x),
            Err(InvalidInput::Value(_))
        ));
    }
}
