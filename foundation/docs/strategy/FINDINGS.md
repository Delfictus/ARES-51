# Raw Findings

This document contains the raw, unprocessed output from the analysis commands.

## `cargo check --workspace --all-features`

```
error[E0432]: unresolved import `csf_core::prelude`
 --> crates/csf-bus/src/packet.rs:3:15
  |
3 | use csf_core::prelude::*;
  |               ^^^^^^^ could not find `prelude` in `csf_core`

error[E0432]: unresolved import `csf_core::prelude`
 --> crates/csf-kernel/src/scheduler/mod.rs:3:15
  |
3 | use csf_core::prelude::*;
  |               ^^^^^^^ could not find `prelude` in `csf_core`

error[E0432]: unresolved import `nix::sched`
   --> crates/csf-kernel/src/scheduler/mod.rs:200:14
    |
200 |     use nix::sched::{sched_setscheduler, CpuSet, Policy};
    |              ^^^^^ could not find `sched` in `nix`

error[E0432]: unresolved import `csf_core::types`
 --> crates/csf-kernel/src/time.rs:4:15
  |
4 | use csf_core::types::NanoTime;
  |               ^^^^^ could not find `types` in `csf_core`

...
(Additional compilation errors omitted for brevity)
```

## `grep` Scans

### Determinism (`panic!`, `unwrap`, `expect`)

```
./src/main.rs:103:    let level = level.parse().unwrap_or(tracing::Level::INFO);
./src/main.rs:108:                .unwrap_or_else(|_| format!("chronofabric={}", level).into())
./crates/csf-network/src/routing.rs:161:                let next_hop = path.get(1).copied().unwrap_or(destination);
./crates/csf-network/src/routing.rs:172:            if cost > *distances.get(&node).unwrap_or(&u32::MAX) {
./crates/csf-network/src/security.rs:141:    rand::SecureRandom::fill(&rng, &mut nonce).unwrap();
./crates/csf-network/src/lib.rs:484:            .expect("metric creation failed");
./crates/csf-network/src/quic.rs:55:        let cert = rcgen::generate_simple_self_signed(vec!["localhost".to_string()]).unwrap();
./crates/csf-clogic/src/egc/stl.rs:510:                    _ => panic!("Expected And formula"),
./crates/csf-ffi/src/types.rs:36:        let metadata = std::ffi::CString::new(metadata_str).unwrap().into_raw();
./crates/csf-sil/src/crypto.rs:5:    Keypair::from_bytes(&[0u8; 64]).unwrap_or_else(|_| panic!("stub keypair"))
./crates/csf-sil/src/crypto.rs:10:    Signature::from_bytes(&[0u8; 64]).expect("stub signature")
...
(Additional findings omitted for brevity)
```

### Unsafe Code (`unsafe`)

```
./crates/csf-kernel/src/memory/mod.rs:104:    pub unsafe fn from_raw_parts(ptr: *mut u8, size: usize) -> Self {
./crates/csf-kernel/src/scheduler/mod.rs:73:                    unsafe { libc::sched_yield(); }
./crates/csf-ffi/src/lib.rs:59:#[no_mangle]
pub unsafe extern "C" fn csf_version_free(version: *mut c_char) {
./crates/csf-hardware/src/timestamp.rs:104:        unsafe { asm!("mrs {}, cntvct_el0", out(reg) val); }
...
(Additional findings omitted for brevity)
```

### Dynamic Dispatch (`dyn`)

```
./README.md:145:async fn main() -> Result<(), Box<dyn std::error::Error>> {
./crates/csf-network/src/protocol.rs:12:    handlers: dashmap::DashMap<MessageType, Box<dyn MessageHandler>>,
./crates/csf-kernel/src/task.rs:19:    runnable: Box<dyn FnOnce() -> Result<(), anyhow::Error> + Send + 'static>,
./crates/csf-telemetry/src/tracing.rs:56:    tracer: Box<dyn OtelTracer + Send + Sync>,
...
(Additional findings omitted for brevity)
```

### Feature Usage (`cfg(feature)`)

```
crates/csf-core/src/envelope.rs:6:#[cfg(feature = "net")]
crates/csf-clogic/src/adp/quantum.rs:70:    #[cfg(feature = "cuda")]
crates/csf-ffi/src/lib.rs:17:#[cfg(feature = "python")]
crates/csf-hardware/src/lib.rs:207:    #[cfg(feature = "cuda")]
crates/csf-telemetry/src/collector.rs:49:    #[cfg(feature = "nvidia-gpu")]
...
(Additional findings omitted for brevity)
```

### Observability (`tracing`, `prometheus`, etc.)

- `grep` for `(tracing::|#[instrument])` returned no results.
- `grep` for `(prometheus|opentelemetry)` returned no results.
