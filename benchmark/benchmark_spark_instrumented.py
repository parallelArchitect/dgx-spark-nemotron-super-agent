#!/usr/bin/env python3
"""
benchmark_spark_instrumented.py — DGX Spark GB10 (SM121) Nemotron-3-Super
benchmark with continuous NVML hardware telemetry.

Target platform: NVIDIA DGX Spark / ASUS GX10 (GB10, SM121, LPDDR5X UMA)
Driver: 580.142, CUDA 13.0, vLLM running in Docker (spark-brain container)

Wraps airawatraj's benchmark_spark.py tests with host-side NVML telemetry.
Docker runs the model. This script runs on the host and measures the GPU.

Original benchmark_spark.py by airawatraj (airawatraj/dgx-spark-nemotron-super-agent)
Instrumentation layer by parallelArchitect (parallelArchitect/dgx-spark-nemotron-super-agent)

Usage:
  # 1. Start Docker container first
  bash docker/start.sh

  # 2. Wait for vLLM to be ready (~10 min)
  docker logs -f spark-brain | grep "Application startup complete"

  # 3. Run instrumented benchmark on host
  python3 benchmark/benchmark_spark_instrumented.py --report

  # With custom host/port/model
  python3 benchmark/benchmark_spark_instrumented.py --host localhost --port 8000 --model Cogni-Brain --report
"""

import argparse
import ctypes
import ctypes.util
import json
import math
import os
import statistics
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

# ─── ANSI ────────────────────────────────────────────────────────────────────

RED    = "\033[31m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"
DIM    = "\033[2m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

BASELINE_DIR = Path.home() / ".spark-nemotron-benchmark"
SAMPLE_INTERVAL = 0.5

# ─── NVMLDirect (from spark-gpu-throttle-check, parallelArchitect) ───────────

THROTTLE_REASONS = {
    0x0000000000000001: "GPU_IDLE",
    0x0000000000000002: "APPLICATIONS_CLOCKS_SETTING",
    0x0000000000000004: "SW_POWER_CAP",
    0x0000000000000008: "HW_SLOWDOWN",
    0x0000000000000010: "SYNC_BOOST",
    0x0000000000000020: "SW_THERMAL_SLOWDOWN",
    0x0000000000000040: "HW_THERMAL_SLOWDOWN",
    0x0000000000000080: "HW_POWER_BRAKE_SLOWDOWN",
}

PROBLEM_REASONS = {
    0x0000000000000004,
    0x0000000000000008,
    0x0000000000000020,
    0x0000000000000040,
    0x0000000000000080,
}


def decode_throttle_bitmask(bitmask: int) -> list:
    if bitmask == 0:
        return ["NONE"]
    reasons = [name for bit, name in THROTTLE_REASONS.items() if bitmask & bit]
    return reasons if reasons else [f"UNKNOWN(0x{bitmask:016x})"]


def has_problem_throttle(bitmask: int) -> bool:
    return bool(bitmask & sum(PROBLEM_REASONS))


class NVMLDirect:
    """Lightweight NVML wrapper via ctypes. No pynvml dependency.
    Copied from spark-gpu-throttle-check v2.1.0 (parallelArchitect)."""

    def __init__(self):
        self._lib = None
        self._handle = None
        self._available = False
        self._initialized = False

    def _load_lib(self) -> bool:
        if self._initialized:
            return self._lib is not None
        self._initialized = True
        try:
            path = ctypes.util.find_library("nvidia-ml")
            if not path:
                for candidate in [
                    "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1",
                    "/usr/lib/aarch64-linux-gnu/libnvidia-ml.so.1",
                    "/usr/lib64/libnvidia-ml.so.1",
                    "/usr/lib/libnvidia-ml.so.1",
                ]:
                    if os.path.exists(candidate):
                        path = candidate
                        break
            if not path:
                return False
            self._lib = ctypes.CDLL(path)
            return self._lib.nvmlInit_v2() == 0
        except (OSError, AttributeError):
            return False

    def init(self, gpu_index: int = 0) -> bool:
        if not self._load_lib():
            return False
        self._handle = ctypes.c_void_p()
        rc = self._lib.nvmlDeviceGetHandleByIndex_v2(
            ctypes.c_uint(gpu_index), ctypes.byref(self._handle)
        )
        if rc != 0:
            return False
        self._available = True
        return True

    def shutdown(self):
        if self._lib and self._initialized:
            try:
                self._lib.nvmlShutdown()
            except Exception:
                pass

    @property
    def available(self) -> bool:
        return self._available

    def get_clock_mhz(self):
        if not self._available:
            return None
        val = ctypes.c_uint()
        rc = self._lib.nvmlDeviceGetClockInfo(self._handle, 0, ctypes.byref(val))
        return val.value if rc == 0 else None

    def get_power_w(self):
        if not self._available:
            return None
        val = ctypes.c_uint()
        rc = self._lib.nvmlDeviceGetPowerUsage(self._handle, ctypes.byref(val))
        return val.value / 1000.0 if rc == 0 else None

    def get_temperature(self):
        if not self._available:
            return None
        val = ctypes.c_uint()
        rc = self._lib.nvmlDeviceGetTemperature(self._handle, 0, ctypes.byref(val))
        return val.value if rc == 0 else None

    def get_throttle_reasons(self):
        if not self._available:
            return None
        val = ctypes.c_ulonglong()
        rc = self._lib.nvmlDeviceGetCurrentClocksThrottleReasons(
            self._handle, ctypes.byref(val)
        )
        return val.value if rc == 0 else None

    def get_gpu_name(self):
        if not self._available:
            return None
        buf = ctypes.create_string_buffer(256)
        rc = self._lib.nvmlDeviceGetName(self._handle, buf, 256)
        return buf.value.decode("utf-8", errors="replace") if rc == 0 else None

    def get_driver_version(self):
        if not self._available:
            return None
        buf = ctypes.create_string_buffer(256)
        rc = self._lib.nvmlSystemGetDriverVersion(buf, 256)
        return buf.value.decode("utf-8", errors="replace") if rc == 0 else None

    def sample(self) -> dict:
        throttle_raw = self.get_throttle_reasons()
        return {
            "timestamp": time.time(),
            "clk_mhz": self.get_clock_mhz(),
            "power_w": self.get_power_w(),
            "temp_c": self.get_temperature(),
            "throttle_raw": throttle_raw,
            "throttle_reasons": decode_throttle_bitmask(throttle_raw)
            if throttle_raw is not None else [],
            "throttle_problem": has_problem_throttle(throttle_raw)
            if throttle_raw is not None else False,
        }


# ─── Thermal trajectory ───────────────────────────────────────────────────────

def compute_thermal_trajectory(samples: list) -> dict:
    temps = [
        (s.get("elapsed", 0), s.get("temp_c"))
        for s in samples
        if s.get("temp_c") is not None and not s.get("marker")
    ]
    if len(temps) < 3:
        return {"slope": 0.0, "direction": "insufficient_data", "stable": True}

    n = len(temps)
    sum_x = sum(t[0] for t in temps)
    sum_y = sum(t[1] for t in temps)
    sum_xy = sum(t[0] * t[1] for t in temps)
    sum_x2 = sum(t[0] ** 2 for t in temps)
    denom = n * sum_x2 - sum_x ** 2
    if denom == 0:
        return {"slope": 0.0, "direction": "flat", "stable": True}

    slope = (n * sum_xy - sum_x * sum_y) / denom
    direction = "rising" if slope > 0.1 else "cooling" if slope < -0.1 else "stable"

    return {
        "slope": round(slope, 3),
        "direction": direction,
        "stable": abs(slope) <= 0.1,
        "start_temp": temps[0][1],
        "end_temp": temps[-1][1],
    }


def compute_stability_score(values: list) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    if mean == 0:
        return 0.0
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return (math.sqrt(variance) / mean) * 100


# ─── Hardware sampler ─────────────────────────────────────────────────────────

class HardwareSampler:
    def __init__(self, nvml: NVMLDirect, interval: float = SAMPLE_INTERVAL):
        self._nvml = nvml
        self._interval = interval
        self._samples = []
        self._stop = threading.Event()
        self._thread = None
        self._t_start = None

    def start(self):
        self._samples = []
        self._stop.clear()
        self._t_start = time.time()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> list:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        return self._samples

    def _run(self):
        while not self._stop.is_set():
            s = self._nvml.sample()
            s["elapsed"] = time.time() - self._t_start
            self._samples.append(s)
            time.sleep(self._interval)

    def mark(self, label: str):
        self._samples.append({
            "marker": True,
            "label": label,
            "elapsed": time.time() - self._t_start,
            "timestamp": time.time(),
        })


# ─── Hardware summary ─────────────────────────────────────────────────────────

def summarize_hw(samples: list, label: str) -> dict:
    hw = [s for s in samples if not s.get("marker")]
    if not hw:
        return {}

    clocks = [s["clk_mhz"] for s in hw if s.get("clk_mhz") is not None]
    powers = [s["power_w"] for s in hw if s.get("power_w") is not None]
    temps  = [s["temp_c"]  for s in hw if s.get("temp_c")  is not None]

    throttle_reasons_seen = set()
    problem_throttle = False
    for s in hw:
        for r in s.get("throttle_reasons", []):
            if r not in ("NONE", "GPU_IDLE"):
                throttle_reasons_seen.add(r)
        if s.get("throttle_problem"):
            problem_throttle = True

    return {
        "label": label,
        "sample_count": len(hw),
        "clk_peak_mhz": max(clocks) if clocks else None,
        "clk_avg_mhz": round(sum(clocks) / len(clocks), 1) if clocks else None,
        "clk_stability_cv": round(compute_stability_score(clocks), 3),
        "power_avg_w": round(sum(powers) / len(powers), 1) if powers else None,
        "power_peak_w": round(max(powers), 1) if powers else None,
        "temp_avg_c": round(sum(temps) / len(temps), 1) if temps else None,
        "temp_peak_c": max(temps) if temps else None,
        "thermal_trajectory": compute_thermal_trajectory(hw),
        "throttle_reasons_seen": sorted(throttle_reasons_seen),
        "problem_throttle": problem_throttle,
    }


# ─── Benchmark tests (from airawatraj, instrumented) ─────────────────────────

def make_prompt(n_words):
    base = ("The quick brown fox jumps over the lazy dog. " * 50).split()
    words = (base * ((n_words // len(base)) + 1))[:n_words]
    return " ".join(words) + "\n\nSummarize the above text in one sentence."


def stream_completion(host, port, model, prompt, max_tokens=200, timeout=120):
    url = f"http://{host}:{port}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.1,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    t_start = time.perf_counter()
    t_first = None
    full_text = ""
    usage_tokens = None

    try:
        with requests.post(url, json=payload, stream=True, timeout=timeout) as resp:
            if resp.status_code != 200:
                return None, None, 0, "", f"HTTP {resp.status_code}"
            for line in resp.iter_lines():
                if not line:
                    continue
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        if chunk.get("usage"):
                            usage_tokens = chunk["usage"].get("completion_tokens")
                        delta = chunk["choices"][0]["delta"]
                        text = (delta.get("content", "") or "") + (delta.get("reasoning", "") or "")
                        if text:
                            if t_first is None:
                                t_first = time.perf_counter()
                            full_text += text
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
    except requests.exceptions.Timeout:
        return None, None, 0, "", "Timeout"
    except requests.exceptions.ConnectionError:
        return None, None, 0, "", "Connection refused"
    except Exception as e:
        return None, None, 0, "", str(e)

    t_end = time.perf_counter()
    if t_first is None:
        return None, None, 0, full_text, "No tokens generated"

    ttft_ms = (t_first - t_start) * 1000
    generation_time = t_end - t_first
    tokens = usage_tokens if usage_tokens and usage_tokens > 0 else max(1, len(full_text) // 4)
    tps = tokens / generation_time if generation_time > 0 else 0
    return round(ttft_ms), round(tps, 1), tokens, full_text, None


def test_baseline_tps(host, port, model, sampler):
    print(f"\n  {BOLD}TEST 1 — Baseline TPS{RESET}")
    prompt = "Explain quantum entanglement in simple terms."
    runs = 3
    results = []

    sampler.mark("test1_baseline_start")
    for i in range(runs):
        sampler.mark(f"test1_run{i+1}_start")
        ttft, tps, tokens, _, err = stream_completion(host, port, model, prompt, max_tokens=300)
        sampler.mark(f"test1_run{i+1}_end")
        if err:
            print(f"  {RED}Run {i+1} failed: {err}{RESET}")
            continue
        results.append({"ttft_ms": ttft, "tps": tps, "tokens": tokens})
        print(f"  Run {i+1}: TTFT={YELLOW}{ttft}ms{RESET}  TPS={GREEN}{tps}{RESET}  tokens={tokens}")
        time.sleep(1)
    sampler.mark("test1_baseline_end")

    if results:
        avg_tps = round(statistics.mean([r["tps"] for r in results]), 1)
        avg_ttft = round(statistics.mean([r["ttft_ms"] for r in results]))
        peak_tps = max([r["tps"] for r in results])
        print(f"  Average: {GREEN}{avg_tps} tok/s{RESET}  Peak: {GREEN}{peak_tps} tok/s{RESET}  TTFT avg: {YELLOW}{avg_ttft}ms{RESET}")
        return {"avg_tps": avg_tps, "peak_tps": peak_tps, "avg_ttft_ms": avg_ttft, "runs": results}
    return {}


def test_concurrent(host, port, model, sampler, max_concurrent=4):
    import threading as _threading
    print(f"\n  {BOLD}TEST 2 — Concurrent Sessions{RESET}")
    prompts_list = [
        "Explain the history of the Roman Empire in detail.",
        "Describe how neural networks learn from data.",
        "What are the key principles of thermodynamics?",
        "Explain the causes and effects of the French Revolution.",
    ]
    results_all = []

    for n in range(1, max_concurrent + 1):
        results = [None] * n
        errors = []

        def run_request(idx):
            _, tps, _, _, err = stream_completion(
                host, port, model, prompts_list[idx % len(prompts_list)], max_tokens=200
            )
            if err:
                errors.append(err)
            else:
                results[idx] = tps

        sampler.mark(f"test2_concurrent{n}_start")
        threads = [_threading.Thread(target=run_request, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        sampler.mark(f"test2_concurrent{n}_end")

        valid = [r for r in results if r is not None]
        if valid:
            total_tps = round(sum(valid), 1)
            per_session = round(statistics.mean(valid), 1)
            color = GREEN if per_session >= 10 else YELLOW if per_session >= 6 else RED
            print(f"  {n} session(s): total={color}{total_tps} tok/s{RESET}  per-session={color}{per_session} tok/s{RESET}")
            results_all.append({"sessions": n, "total_tps": total_tps, "per_session_tps": per_session})
        else:
            print(f"  {n} session(s): {RED}FAILED{RESET}")
        time.sleep(3)

    return results_all


def test_context_window(host, port, model, sampler):
    print(f"\n  {BOLD}TEST 3 — Context Window{RESET}")
    sizes = [1024, 4096, 8192, 16384, 32768, 65536, 98304, 131072]
    results = []
    last_working = 0

    sampler.mark("test3_context_start")
    for size in sizes:
        prompt = make_prompt(int(size * 0.75))
        sampler.mark(f"test3_ctx{size}_start")
        ttft, tps, gen_tokens, _, err = stream_completion(
            host, port, model, prompt, max_tokens=100, timeout=180
        )
        sampler.mark(f"test3_ctx{size}_end")

        if err:
            print(f"  ~{size} tok: {RED}✗ {err[:40]}{RESET}")
            break
        else:
            last_working = size
            color = GREEN if tps >= 12 else YELLOW if tps >= 8 else RED
            print(f"  ~{size} tok: {GREEN}✓{RESET}  {color}{tps} tok/s{RESET}")
            results.append({"context_tokens": size, "tps": tps, "ttft_ms": ttft})
            time.sleep(2)
    sampler.mark("test3_context_end")

    return {"last_working_context": last_working, "results": results}


# ─── Report export ────────────────────────────────────────────────────────────

def export_report(results: dict, gpu_index: int = 0):
    report_dir = BASELINE_DIR / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = report_dir / f"nemotron-spark-benchmark_gpu{gpu_index}_{ts}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  {GREEN}Report saved: {path}{RESET}")
    return str(path)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="DGX Spark Nemotron-3-Super benchmark with NVML host telemetry"
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--model", default="Cogni-Brain")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--skip-concurrent", action="store_true")
    parser.add_argument("--skip-context", action="store_true")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()

    # ── Initialize NVML ──
    nvml = NVMLDirect()
    nvml_ok = nvml.init(gpu_index=args.gpu)

    print("=" * 60)
    print("  DGX Spark GB10 (SM121) Nemotron-3-Super Benchmark")
    print("  Instrumented — parallelArchitect")
    print("=" * 60)

    if nvml_ok:
        print(f"  GPU:    {nvml.get_gpu_name() or 'unknown'}")
        print(f"  Driver: {nvml.get_driver_version() or 'unknown'}")
        idle = nvml.sample()
        print(f"  Idle:   CLK={idle.get('clk_mhz')} MHz  "
              f"PWR={idle.get('power_w'):.1f}W  "
              f"TMP={idle.get('temp_c')}°C")
    else:
        print(f"  {YELLOW}NVML unavailable — hardware telemetry disabled{RESET}")

    # ── Check vLLM health ──
    print(f"\n  Checking vLLM at http://{args.host}:{args.port}...")
    try:
        r = requests.get(f"http://{args.host}:{args.port}/health", timeout=5)
        if r.status_code != 200:
            print(f"  {RED}vLLM not healthy (HTTP {r.status_code}){RESET}")
            print(f"  {DIM}Start the container: bash docker/start.sh{RESET}")
            nvml.shutdown()
            sys.exit(1)
        print(f"  {GREEN}vLLM reachable{RESET}  model={args.model}")
    except Exception as e:
        print(f"  {RED}Cannot reach vLLM: {e}{RESET}")
        print(f"  {DIM}Start the container: bash docker/start.sh{RESET}")
        nvml.shutdown()
        sys.exit(1)

    print()

    # ── Start sampler ──
    sampler = HardwareSampler(nvml) if nvml_ok else None
    if sampler:
        sampler.start()

    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "gpu_index": args.gpu,
        "host": args.host,
        "port": args.port,
        "model": args.model,
    }

    if nvml_ok:
        results["gpu_name"] = nvml.get_gpu_name()
        results["driver_version"] = nvml.get_driver_version()

    # ── Run tests ──
    results["test1_baseline"] = test_baseline_tps(args.host, args.port, args.model, sampler or _NullSampler())

    if not args.skip_concurrent:
        results["test2_concurrent"] = test_concurrent(args.host, args.port, args.model, sampler or _NullSampler())

    if not args.skip_context:
        results["test3_context"] = test_context_window(args.host, args.port, args.model, sampler or _NullSampler())

    # ── Stop sampler ──
    if sampler:
        all_samples = sampler.stop()
        hw = summarize_hw(all_samples, "full run")
        results["hardware"] = hw

        print(f"\n  {BOLD}Hardware telemetry:{RESET}")
        print(f"    Clock:  peak={hw.get('clk_peak_mhz')} MHz  "
              f"avg={hw.get('clk_avg_mhz')} MHz  "
              f"stability={hw.get('clk_stability_cv'):.2f}% CV")
        print(f"    Power:  avg={hw.get('power_avg_w')} W  peak={hw.get('power_peak_w')} W")
        print(f"    Temp:   avg={hw.get('temp_avg_c')}°C  peak={hw.get('temp_peak_c')}°C")
        traj = hw.get("thermal_trajectory", {})
        t_color = GREEN if traj.get("stable") else YELLOW
        print(f"    Thermal: {t_color}{traj.get('direction')} ({traj.get('slope'):+.2f} °C/s){RESET}")
        if hw.get("problem_throttle"):
            print(f"    {RED}Problem throttle: {', '.join(hw.get('throttle_reasons_seen', []))}{RESET}")

        if args.report:
            results["hw_timeline"] = all_samples

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print(f"  {BOLD}SUMMARY{RESET}")
    print(f"{'=' * 60}")
    t1 = results.get("test1_baseline", {})
    if t1:
        avg = t1.get("avg_tps", 0)
        peak = t1.get("peak_tps", 0)
        color = GREEN if avg >= 20 else YELLOW if avg >= 15 else RED
        print(f"  Baseline TPS (avg):  {color}{avg} tok/s{RESET}")
        print(f"  Baseline TPS (peak): {color}{peak} tok/s{RESET}")

    if args.report:
        export_report(results, args.gpu)

    nvml.shutdown()


class _NullSampler:
    """No-op sampler when NVML is unavailable."""
    def mark(self, label): pass


if __name__ == "__main__":
    main()
