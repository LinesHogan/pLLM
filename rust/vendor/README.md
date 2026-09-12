# ISL build configuration

`isl-rs/` is the source of the existing `isl-rs 0.3.1` dependency, copied from
the published crate. Its crates.io checksum is
`7ffbe47f00150a69fd7ac8053db449d09346f53bfa1c6ab0b3c80e7f3a73eada`.
The upstream repository is <https://github.com/kaushikcfd/isl-rs>.

The build configuration changes `--with-int=imath` to `--with-int=imath-32`.
`build.rs` also restores executable permission on the two upstream build
helpers in Cargo's private build directory: source distributions may normalize
file modes. The Rust bindings, ISL algorithms and bundled imath
implementation remain upstream code. Registry installation marker `.cargo-ok`
is omitted. `Cargo.toml` excludes the registry-generated `Cargo.toml.orig`
from repackaging, because Cargo reserves that filename. Upstream licenses are
preserved in the source tree and included in pLLM distributions.

ISL documents this option in `isl/doc/user.pod`, under `--with-int`: small
integers use a fast representation, and larger values fall back to imath's
arbitrary-precision representation. It does **not** truncate coefficients or
restrict the planner to 32-bit values.

The published crate hard-codes the slower backend and exposes no build option
for changing it. A pinned path dependency makes this single configuration change
reproducible for normal and offline builds, without editing the Cargo registry,
requiring system GMP/ISL development packages, or adding runtime configuration.

When updating this dependency, import a complete upstream release, retain its
licenses and provenance, and review these build/packaging differences. Re-run
large-integer, schedule, capacity and execution checks; compare source-derived
plans and cold-planning time. This tree is not a place for pLLM scheduling rules.
