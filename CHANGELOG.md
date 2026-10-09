# Changelog

This file documents recent notable changes to this project. The format of this
file is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and
this project adheres to [Semantic
Versioning](https://semver.org/spec/v2.0.0.html).

## 2.0.0 - 2026-10-09

### Changed

* `butter` now returns a `Result` containing second-order sections instead of
  numerator and denominator polynomials.
* `filtfilt` now accepts second-order sections instead of numerator and
  denominator polynomials.

## 1.0.3 - 2022-05-05

### Changed

* Updated lair to 0.6, which requires Rust 1.51 or later.

## 1.0.2 - 2021-04-14

### Changed

* Requires Rust 1.49 or later.

## 1.0.1 - 2020-10-13

### Changed

* Compatible with Rust 1.47.

## 1.0.0 - 2020-03-02

### Added

* A forward-backward digital filter.
* Butterworth digital filter design.

[2.0.0]: https://github.com/vinesystems/vinum/compare/1.0.3...2.0.0
[1.0.3]: https://github.com/vinesystems/vinum/compare/1.0.2...1.0.3
[1.0.2]: https://github.com/vinesystems/vinum/compare/1.0.1...1.0.2
[1.0.1]: https://github.com/vinesystems/vinum/compare/1.0.0...1.0.1
[1.0.0]: https://github.com/vinesystems/vinum/tree/1.0.0
