# Changelog

All notable changes to this repository are documented here.

This fork extends [airawatraj/dgx-spark-nemotron-super-agent](https://github.com/airawatraj/dgx-spark-nemotron-super-agent)
with hardware instrumentation tooling for DGX Spark (GB10, SM121).

---

## [experimental] — 2026-05-17 — parallelArchitect

> Status: experimental — GB10 hardware validation pending.

### Added

- `benchmark/benchmark_spark_instrumented.py` — instrumented benchmark with
  continuous NVML hardware telemetry during Nemotron-3-Super-120B inference.
  Wraps airawatraj's benchmark_spark.py tests (baseline TPS, concurrent sessions,
  context window) with host-side GPU measurement. Captures clock, power,
  temperature, and throttle reasons throughout each test phase via NVMLDirect.
  Reports thermal trajectory, clock stability (coefficient of variation), and
  power delta. Docker runs the model inside the container; this script runs on
  the host and reads GPU state via NVML — no Docker interaction required from
  the instrumentation side. Supports `--report` for full JSON export with
  hardware timeline, `--skip-concurrent`, `--skip-context`. `NVMLDirect`
  telemetry layer ported from
  [spark-gpu-throttle-check v2.1.0](https://github.com/parallelArchitect/spark-gpu-throttle-check).
- `CHANGELOG.md` — this file.

---

## [1.0.0] — 2026-05-10 — airawatraj original

### Added

- vLLM Docker configuration for Nemotron-3-Super-120B-A12B-NVFP4 on DGX Spark
- Key fixes over prior community setups: CUDA graphs enabled, correct tool
  parser (`qwen3_coder`), Marlin backend env var, MTP speculative decoding
  (`num_speculative_tokens=1`), scheduler tuning (`--max-num-batched-tokens 16384`)
- `benchmark/benchmark_spark.py` — TPS, TTFT, concurrent sessions, context
  window benchmark via OpenAI streaming API
- `METHODOLOGY.md` — full benchmark measurement methodology
- `NEMOHERMES.md` — hardened agentic stack configuration (K3s, Landlock, seccomp)
- Docker setup: `docker/start.sh`, `docker/stop.sh`, `docker/status.sh`
- Setup scripts: `setup/install.sh`, `setup/download_parser.sh`

### Results (airawatraj, May 14 2026)

- Single session TPS (tg128): 23.45 tok/s — highest published single-node
  result for Nemotron-3-Super-120B-A12B-NVFP4 as of May 14, 2026
- Peak TPS (tg128 c5): 72.67 tok/s
- Context stability: 0 → 100K tokens, zero crashes, zero OOM
- GPU operating under 75°C throughout

### Known limitations (airawatraj)

- Nightly vLLM image — not a stable release
- NVFP4 on SM121 forces Marlin dequantization fallback — native FP4 MoE
  compute kernels not yet available for GB10
- Uncalibrated FP8 KV cache scaling factors

### References

- [NVIDIA Developer Forum post](https://forums.developer.nvidia.com/t/nvfp4-on-dgx-spark-gb10-is-broken-i-bought-9-of-these-for-this-feature-requesting-nvidias-official-roadmap-and-response/367082/46)
- [spark-arena benchmark submission](https://spark-arena.com/benchmark/sub1778644062716)
