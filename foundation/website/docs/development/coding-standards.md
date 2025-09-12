---
sidebar_position: 2
title: "Coding Standards"
description: "Code quality standards and conventions for ARES ChronoFabric development"
---

# Coding Standards

- Rust 2021, MSRV pinned.
- No `unwrap`/`expect` in library code.
- Errors: `thiserror` for library types; `anyhow` at edges only.
- No panics on hot paths.
- Unsafe isolated under `unsafe/` with safety docs and tests.
- `tracing` for logs; no `println!`.
- Public items must have rustdoc.