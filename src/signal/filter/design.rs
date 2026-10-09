//! Tools to design filters.

use num_traits::{Float, FloatConst};

use crate::InvalidInput;

/// Designs a digital low-pass Butterworth filter as second-order sections.
///
/// Each row contains `[b0, b1, b2, a0, a1, a2]`, with `a0 = 1`. `cutoff_freq`
/// is normalized to the Nyquist frequency. Sections are calculated directly,
/// avoiding the sensitive expansion into a high-order polynomial. Order zero
/// produces an identity section. Odd orders include a first-order section with
/// `b2 = a2 = 0`.
///
/// # Errors
///
/// Returns [`InvalidInput::Value`] for a negative order, a non-finite cutoff,
/// or a cutoff outside `0 < cutoff_freq < 1`.
///
/// # Examples
///
/// ```
/// use vinum::signal::filter::{design::butter, filtfilt};
/// let sos = butter(8, 0.01_f64)?;
/// let y = filtfilt(&sos, &[1.0; 100])?;
/// assert!(y.iter().all(|v| (v - 1.0).abs() < 1e-9));
/// # Ok::<(), vinum::InvalidInput>(())
/// ```
pub fn butter<T>(order: i32, cutoff_freq: T) -> Result<Vec<[T; 6]>, InvalidInput>
where
    T: Float + FloatConst,
{
    let zero = T::zero();
    let one = T::one();
    let two = one + one;
    if order < 0 || !cutoff_freq.is_finite() || cutoff_freq <= zero || cutoff_freq >= one {
        return Err(InvalidInput::Value(
            "expected a nonnegative order and a finite cutoff strictly between 0 and 1".into(),
        ));
    }
    if order == 0 {
        return Ok(vec![[one, zero, zero, one, zero, zero]]);
    }
    let k = (T::PI() * cutoff_freq / two).tan();
    let mut sos = Vec::new();
    if order % 2 != 0 {
        let gain = k / (one + k);
        sos.push([gain, gain, zero, one, (k - one) / (k + one), zero]);
    }
    // Pair conjugate poles, placing the most damped sections first.
    for j in (0..order / 2).rev() {
        let angle = T::PI() * T::from(2 * j + 1).expect("T can represent a pole index")
            / (two * T::from(order).expect("T can represent the order"));
        let damping = two * angle.sin();
        let k2 = k * k;
        let denom = one + damping * k + k2;
        let gain = k2 / denom;
        sos.push([
            gain,
            two * gain,
            gain,
            one,
            two * (k2 - one) / denom,
            (one - damping * k + k2) / denom,
        ]);
    }
    Ok(sos)
}

#[cfg(test)]
mod tests {
    #[test]
    fn butter_rejects_invalid_parameters() {
        assert!(super::butter(-1, 0.2_f64).is_err());
        for cutoff in [0.0, 1.0, -0.1, 1.1, f64::NAN, f64::INFINITY] {
            assert!(super::butter(3, cutoff).is_err());
        }
    }
}
