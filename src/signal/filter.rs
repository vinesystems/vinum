//! Filters using cascaded second-order sections.

pub mod design;
mod sos;

pub use sos::filtfilt;
