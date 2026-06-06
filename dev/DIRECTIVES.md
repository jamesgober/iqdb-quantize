# iqdb-quantize -- Engineering Directives

> Engineering standards and the definition of done for this project. Read alongside `REPS.md` (repository root, authoritative) and `dev/ROADMAP.md` (current phase). If anything here conflicts with `REPS.md`, `REPS.md` wins.

---

## 0. Philosophy

This library crate is built and maintained to a production standard and treated as a flagship piece of work. The full path is planned end to end, then built one verified step at a time. Cookie-cutter patterns are a starting point to improve on, not a finish line. Where the best available choice is unclear, it is settled before code is committed. "Good enough" is treated as a defect.

---

## 1. What this is

iqdb-quantize compresses f32 vectors into smaller representations while preserving search quality. A million 768-dim vectors drop from ~3 GB to as little as ~96 MB, trading a controlled amount of recall for memory. It implements the three standard schemes (scalar, product, binary) behind a common `Quantizer` trait, and reuses `iqdb-distance` for distance on quantized codes.

---

## 2. Scope and change control

Source, tests, benches, examples, docs, and build/CI configuration are all in scope for normal work. `REPS.md`, license files, and the lint configs (`clippy.toml`, `rustfmt.toml`) are stable and changed only deliberately. Work that falls outside the current phase is recorded in `dev/ROADMAP.md` with rationale before it is taken on.

---

## 3. Engineering law (non-negotiable)

- **Performance** -- peak performance is the baseline. Borrow over clone; avoid allocation on hot paths in steady state; inline small helpers. No "faster" claim ships without `criterion` numbers.
- **Concurrency** -- correctness under contention is proven, not assumed. Lock-free or shared-state paths carry `loom` model checks.
- **Correctness** -- the defining invariants (section 8) are covered by property-based tests, not only examples.
- **Security** -- all untrusted input is validated; every allocation is bounded; library code never panics on hostile input; parse and recovery paths are fuzzed.
- **Architecture** -- SOLID, KISS, YAGNI. One responsibility per unit. Trait seams are the extension points.
- **Cross-platform** -- Linux, macOS, and Windows are first-class and verified by the CI matrix.
- **Error handling** -- every fallible path returns `Result`; the domain error type is generated with `error-forge`; errors are never silently swallowed.
- **Production-ready** -- no commented-out code, no stray `println!`/`dbg!`; every public item carries rustdoc with a runnable example.

---

## 4. The tiered-API mandate

Tier 1 covers the entire common case in a handful of calls, with no builder and no generics to name. Tier 2 is the builder for tuning. Tier 3 is the trait seam for custom backends. Shipping only the advanced surface is a failure of the design, not a feature.

---

## 5. Testing

- Unit tests per module: happy path plus edge cases.
- Property-based tests (`proptest`) for every core invariant in section 8.
- `loom` model checks for every concurrent or lock-free path.
- Integration tests where I/O, persistence, or failure recovery matter.
- `criterion` benchmarks on hot paths, with tracked baselines; a regression over 5% blocks a release.

---

## 6. Documentation, versioning, release

`docs/API.md` stays current and the README leads with the Tier-1 surface. Strict SemVer; any on-disk or wire format is called out across 0.x and frozen at 1.0. The changelog follows Keep a Changelog 1.1.0. Error and config enums are `#[non_exhaustive]`. Documentation reads in a clear, human voice.

---

## 7. Definition of done (all true, or it is not done)

1. Compiles clean on Linux/macOS/Windows, on stable and MSRV.
2. `fmt`, `clippy -D warnings`, `test --all-features`, and `cargo doc -D warnings` are clean.
3. `cargo audit` and `cargo deny check` pass.
4. No `unwrap`/`expect`/`todo!`/`dbg!` in shipping code; `unsafe` only with a `// SAFETY:` note (target: zero `unsafe`).
5. A Tier-1 API exists and headlines the docs.
6. Property tests cover every section-8 invariant; `loom` covers every concurrent path.
7. Hot-path changes carry benchmarks; no regression over 5%.
8. Every public item is documented with an example; `docs/API.md`, README, and version metadata are updated.
9. Changes are recorded under `[Unreleased]` in the changelog.

---

## 8. Project-specific invariants

Stack: Rust edition 2024, MSRV 1.87. Errors via `error-forge`. Tests via `cargo test` + `criterion` + `loom` + `proptest`.

- A quantizer must be trained on a representative sample before use; this contract is documented and enforced.
- Re-ranking with full precision is the recommended quality path and is reflected in the API.

Per-phase exit criteria in `dev/ROADMAP.md` encode these; each guarantee has a property test before the phase that introduces it can close.

---

## 9. Integration points

- `iqdb-types`: vector and metric types
- `iqdb-distance`: distance on quantized codes
- `iqdb-ivf`: IVF-PQ consumes this for in-cluster compression

<sub>Copyright &copy; 2026 <strong>James Gober</strong>.</sub>
