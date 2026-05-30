# iqdb-quantize &mdash; API Reference

> Complete reference for every public item in `iqdb-quantize`, with examples.
> **Status: pre-1.0.** Sections marked _(planned)_ describe the intended surface as it lands across the 0.x series.

## Table of Contents

- [Overview](#overview)
- [Tier 1 &mdash; the lazy path](#tier-1--the-lazy-path) _(planned: 0.2)_
- [Tier 2 &mdash; the configured path](#tier-2--the-configured-path) _(planned: 0.3)_
- [Tier 3 &mdash; traits](#tier-3--traits)
- [Errors](#errors)
- [Feature flags](#feature-flags)

---

## Overview

iqdb-quantize compresses f32 vectors into smaller representations while preserving search quality. A million 768-dim vectors drop from ~3 GB to as little as ~96 MB, trading a controlled amount of recall for memory.

---

## Tier 1 &mdash; the lazy path

_Documented as the 0.2 foundation lands._

---

## Tier 2 &mdash; the configured path

_Documented at 0.3._

---

## Tier 3 &mdash; traits

_The trait seams custom backends plug into. Documented as they stabilise._

---

## Errors

_Domain error type built on `error-forge` (`#[non_exhaustive]`). Variants documented at 0.2._

---

## Feature flags

| Feature | Default | Description |
|---------|---------|-------------|
| `std` | yes | Standard library. |
| `serde` | no | Serialization support. |

---

<sub>Copyright &copy; 2026 <strong>James Gober</strong>.</sub>
