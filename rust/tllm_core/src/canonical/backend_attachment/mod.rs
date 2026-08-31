//! Rust-signed backend attachment facts for canonical execution.

mod ffi;
mod signing;
mod types;
mod validation;

pub(crate) use ffi::register;
