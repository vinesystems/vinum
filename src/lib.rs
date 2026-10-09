//! A fast and portable numerical library.
//!
//! # Filtering
//!
//! For Butterworth low-pass filtering, use
//! [`signal::filter::design::butter`] and [`signal::filter::filtfilt`],
//! especially for high orders or small cutoff frequencies. Second-order
//! sections (SOS) reduce numerical errors by avoiding a single high-order
//! polynomial, though floating-point limits still apply at extreme cutoffs.
//!
//! ```
//! use vinum::signal::filter::{design::butter, filtfilt};
//!
//! let sos = butter(8, 0.01_f64)?;
//! let filtered = filtfilt(&sos, &[1.0; 100])?;
//! assert!(filtered.iter().all(|v| (v - 1.0).abs() < 1e-9));
//! # Ok::<(), vinum::InvalidInput>(())
//! ```
//!
//! The cutoff is normalized to the Nyquist frequency and must be strictly
//! between 0 and 1. Each SOS row contains `[b0, b1, b2, a0, a1, a2]`.
//! [`signal::filter::filtfilt`] filters forward and backward with odd
//! reflection padding and steady-state initialization.

pub mod signal;

pub use lair::{InvalidInput, Real, Scalar};
