r"""Stage 1: Automated llama.cpp backend, KV-cache & 65K VRAM optimizer for RX 9060 XT (16 GB).

3-Phase pipeline:
  Phase 1 (VRAM Fit Planner, ~1-2s):
    Uses `llama-fit-params.exe -fit off -fitp on` to compute exact GPU MiB (weights + full
    `-c <ctx>` KV cache + compute buffer + optional LoRA) across `--kv-types` (default: q8_0,q4_0).
    Automatically detects MoE vs Dense models and binary-searches the minimum `-ncmoe` (MoE)
    or maximum `-ngl` (Dense) that fits cleanly inside `--target-vram-gb` (default: 13.0 GiB).

  Phase 2 (Throughput Sweep via `llama-bench`):
    Benchmarks fitting candidates on HIP (ROCm0) and Vulkan (Vulkan0, GGML_VK_DISABLE_COOPMAT=1)
    with `-fa 1` at depth 0 and depth 16,384 (`-d 0,16384`). Prefers `q8_0` over `q4_0` when
    both fit with the same offload and generation speed is within 5%.

  Phase 3 (Live 65K `llama-server` Verification, skip with --no-verify-server):
    Starts `llama-server.exe` with the exact winning `-c <ctx>` flags, runs a short completion
    against `/v1/chat/completions`, reads real 65K-allocated tok/s from `/metrics`, and checks
    Windows GPU shared-memory spill for the server process.

Usage:
    # Instant 65K VRAM fit plan (~2s, no benchmarking):
    python optimize.py ..\llama.models\Qwen3.8-27B-UD-IQ3_S.gguf --lora ..\llama.models\Qwen3.8-27B-pi-lora-F16.gguf --plan-only

    # Full optimize + 65K server verification + write start-*.cmd:
    python optimize.py ..\llama.models\gemma-4-26B-A4B-it-UD-IQ4_XS.gguf --alias gemma-4-26b-iq4xs --ctx 65536 --write-script
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from bench_config import (
    HIP_DIR,
    RESULTS_DIR,
    ROCM_BIN_DIR,
    ROOT_DIR,
    VULKAN_DIR,
)


@dataclass
class FitEstimate:
    backend: str
    kv_type: str
    ngl: int
    ncmoe: int
    gpu_model_mib: int
    gpu_ctx_mib: int
    gpu_compute_mib: int
    lora_mib: int
    host_total_mib: int

    @property
    def gpu_total_mib(self) -> int:
        return self.gpu_model_mib + self.gpu_ctx_mib + self.gpu_compute_mib + self.lora_mib

    @property
    def gpu_total_gib(self) -> float:
        return round(self.gpu_total_mib / 1024.0, 2)


def slugify_model(model_path: Path) -> str:
    stem = model_path.stem.lower()
    stem = re.sub(r"[^a-z0-9.]+", "-", stem).strip("-")
    return stem


def _backend_env_and_dir(backend: str) -> tuple[Path, str, dict[str, str]]:
    env = os.environ.copy()
    if backend == "hip":
        env["PATH"] = f"{ROCM_BIN_DIR};{env.get('PATH', '')}"
        return HIP_DIR, "ROCm0", env
    if backend == "vulkan":
        # Required on RDNA4 (RX 9060 XT) in b12426 to avoid FA + coopmat segfault
        env["GGML_VK_DISABLE_COOPMAT"] = "1"
        return VULKAN_DIR, "Vulkan0", env
    raise ValueError(f"Unsupported backend: {backend}")


# ---------------------------------------------------------------------------
# Phase 1: Exact full-context VRAM fit estimation via llama-fit-params.exe
# ---------------------------------------------------------------------------


def parse_fit_params_output(
    stdout_text: str,
    backend: str,
    kv_type: str,
    ngl: int,
    ncmoe: int,
    lora_mib: int = 0,
) -> FitEstimate | None:
    """Parse stdout from `llama-fit-params.exe -fit off -fitp on`."""
    gpu_vals: tuple[int, int, int] | None = None
    host_total = 0

    for line in stdout_text.splitlines():
        parts = line.strip().split()
        if len(parts) < 4:
            continue
        dev = parts[0]
        try:
            m_mib, c_mib, comp_mib = int(parts[1]), int(parts[2]), int(parts[3])
        except ValueError:
            continue
        if dev.lower() == "host":
            host_total = m_mib + c_mib + comp_mib
        elif dev.lower().startswith(("rocm", "vulkan", "cuda", "metal")):
            gpu_vals = (m_mib, c_mib, comp_mib)

    if gpu_vals is None:
        return None
    return FitEstimate(
        backend=backend,
        kv_type=kv_type,
        ngl=ngl,
        ncmoe=ncmoe,
        gpu_model_mib=gpu_vals[0],
        gpu_ctx_mib=gpu_vals[1],
        gpu_compute_mib=gpu_vals[2],
        lora_mib=lora_mib,
        host_total_mib=host_total,
    )


_LAST_FIT_ERROR: str = ""


def estimate_vram_fit(
    backend: str,
    model_path: Path,
    ctx_size: int,
    kv_type: str,
    ngl: int = 99,
    ncmoe: int = 0,
    lora_path: Path | None = None,
) -> FitEstimate | None:
    """Query `llama-fit-params.exe` for exact GPU/Host MiB with explicit flags."""
    global _LAST_FIT_ERROR
    bdir, dev, env = _backend_env_and_dir(backend)
    exe = bdir / "llama-fit-params.exe"
    if not exe.exists():
        _LAST_FIT_ERROR = f"llama-fit-params.exe not found at {exe}"
        return None

    lora_mib = int(round(lora_path.stat().st_size / (1024**2))) if (lora_path and lora_path.exists()) else 0
    cmd = [
        str(exe),
        "-m", str(model_path),
        "-dev", dev,
        "-ngl", str(ngl),
        "-fa", "on",
        "-c", str(ctx_size),
        "-np", "1",
        "-ctk", kv_type,
        "-ctv", kv_type,
        "-fit", "off",
        "-fitp", "on",
    ]
    if ncmoe > 0:
        cmd.extend(["-ncmoe", str(ncmoe)])

    proc = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    est = parse_fit_params_output(proc.stdout, backend, kv_type, ngl, ncmoe, lora_mib=lora_mib)
    if est is None:
        err_lines = [
            ln.strip()
            for ln in (proc.stderr + "\n" + proc.stdout).splitlines()
            if ln.strip() and (" E " in ln or "error" in ln.lower() or "failed" in ln.lower() or "unknown" in ln.lower())
        ]
        _LAST_FIT_ERROR = "\n".join(err_lines[:4]) if err_lines else f"exit code {proc.returncode}"
    return est


def detect_is_moe(backend: str, model_path: Path, ctx_size: int, kv_type: str) -> bool:
    """Detect MoE architecture directly via llama-fit-params (-ncmoe 1 reduces GPU model MiB)."""
    base = estimate_vram_fit(backend, model_path, ctx_size, kv_type, ngl=99, ncmoe=0)
    one = estimate_vram_fit(backend, model_path, ctx_size, kv_type, ngl=99, ncmoe=1)
    if base is None or one is None:
        return False
    return one.gpu_model_mib < base.gpu_model_mib


def find_min_ncmoe(
    backend: str,
    model_path: Path,
    ctx_size: int,
    kv_type: str,
    budget_mib: int,
    lora_path: Path | None = None,
    max_ncmoe: int = 64,
) -> FitEstimate | None:
    """Binary-search the smallest -ncmoe that fits within budget_mib."""
    hi_est = estimate_vram_fit(backend, model_path, ctx_size, kv_type, ngl=99, ncmoe=max_ncmoe, lora_path=lora_path)
    if hi_est is None or hi_est.gpu_total_mib > budget_mib:
        return hi_est

    lo, hi = 1, max_ncmoe
    best = hi_est
    while lo <= hi:
        mid = (lo + hi) // 2
        est = estimate_vram_fit(backend, model_path, ctx_size, kv_type, ngl=99, ncmoe=mid, lora_path=lora_path)
        if est is not None and est.gpu_total_mib <= budget_mib:
            best = est
            hi = mid - 1
        else:
            lo = mid + 1
    return best


def find_max_ngl(
    backend: str,
    model_path: Path,
    ctx_size: int,
    kv_type: str,
    budget_mib: int,
    lora_path: Path | None = None,
) -> FitEstimate | None:
    """Binary-search the largest -ngl (for dense models) that fits within budget_mib."""
    lo, hi = 0, 99
    best: FitEstimate | None = None
    while lo <= hi:
        mid = (lo + hi) // 2
        est = estimate_vram_fit(backend, model_path, ctx_size, kv_type, ngl=mid, ncmoe=0, lora_path=lora_path)
        if est is not None and est.gpu_total_mib <= budget_mib:
            best = est
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def plan_candidates(
    model_path: Path,
    ctx_size: int,
    kv_types: list[str],
    budget_mib: int,
    backends: list[str],
    lora_path: Path | None = None,
    explicit_ncmoe: list[int] | None = None,
) -> tuple[bool, list[FitEstimate]]:
    """Build exact per-(backend, kv_type, offload) candidates that fit in budget_mib.

    Runs the binary search on the fastest probe backend (Vulkan, ~0.15s/call vs ~3s/call
    for HIP DLL load), then verifies the resulting (ngl, ncmoe) once on each target backend.
    """
    probe_backend = "vulkan" if "vulkan" in backends else backends[0]
    is_moe = detect_is_moe(probe_backend, model_path, ctx_size, kv_types[0])

    # Step 1: Find optimal (ngl, ncmoe) per kv_type on the fast probe backend
    offload_by_kv: dict[str, list[tuple[int, int]]] = {}
    probe_cache: dict[tuple[str, int, int], FitEstimate] = {}

    for kv in kv_types:
        if explicit_ncmoe is not None:
            offload_by_kv[kv] = [(99, n) for n in explicit_ncmoe]
            continue

        full = estimate_vram_fit(probe_backend, model_path, ctx_size, kv, ngl=99, ncmoe=0, lora_path=lora_path)
        if full is None:
            continue
        probe_cache[(kv, 99, 0)] = full
        if full.gpu_total_mib <= budget_mib:
            offload_by_kv[kv] = [(99, 0)]
        elif is_moe:
            fitted = find_min_ncmoe(probe_backend, model_path, ctx_size, kv, budget_mib, lora_path=lora_path)
            if fitted:
                probe_cache[(kv, fitted.ngl, fitted.ncmoe)] = fitted
                offload_by_kv[kv] = [(fitted.ngl, fitted.ncmoe)]
        else:
            fitted = find_max_ngl(probe_backend, model_path, ctx_size, kv, budget_mib, lora_path=lora_path)
            if fitted:
                probe_cache[(kv, fitted.ngl, fitted.ncmoe)] = fitted
                offload_by_kv[kv] = [(fitted.ngl, fitted.ncmoe)]

    # Step 2: Confirm exact MiB once per target backend (adjusting +1 ncmoe / -1 ngl if HIP differs slightly)
    plans: list[FitEstimate] = []
    for backend in backends:
        for kv, pairs in offload_by_kv.items():
            for ngl, ncmoe in pairs:
                if backend == probe_backend and (kv, ngl, ncmoe) in probe_cache:
                    plans.append(probe_cache[(kv, ngl, ncmoe)])
                    continue
                est = estimate_vram_fit(backend, model_path, ctx_size, kv, ngl=ngl, ncmoe=ncmoe, lora_path=lora_path)
                if est is None:
                    continue
                if explicit_ncmoe is None and est.gpu_total_mib > budget_mib:
                    if is_moe:
                        est = find_min_ncmoe(backend, model_path, ctx_size, kv, budget_mib, lora_path=lora_path) or est
                    else:
                        est = find_max_ngl(backend, model_path, ctx_size, kv, budget_mib, lora_path=lora_path) or est
                plans.append(est)

    return is_moe, plans


# ---------------------------------------------------------------------------
# Phase 2: Throughput sweep via llama-bench
# ---------------------------------------------------------------------------


def run_llama_bench(
    plan: FitEstimate,
    model_path: Path,
    depths: tuple[int, ...] = (0, 16384),
    reps: int = 2,
) -> list[dict]:
    bdir, dev, env = _backend_env_and_dir(plan.backend)
    exe = bdir / "llama-bench.exe"
    if not exe.exists():
        raise FileNotFoundError(f"llama-bench not found at {exe}")

    cmd = [
        str(exe),
        "-m", str(model_path),
        "-dev", dev,
        "-ngl", str(plan.ngl),
        "-fa", "1",
        "-ctk", plan.kv_type,
        "-ctv", plan.kv_type,
        "-p", "512",
        "-n", "128",
        "-d", ",".join(str(d) for d in depths),
        "-r", str(reps),
        "-o", "jsonl",
    ]
    if plan.ncmoe > 0:
        cmd.extend(["-ncmoe", str(plan.ncmoe)])

    print(
        f"  -> llama-bench [{plan.backend}] kv={plan.kv_type} ngl={plan.ngl} ncmoe={plan.ncmoe} "
        f"(est {plan.gpu_total_mib} MiB) ...",
        flush=True,
    )
    proc = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=600,
    )
    if proc.returncode != 0:
        print(f"     [WARN] llama-bench failed (rc={proc.returncode}): {proc.stderr.strip()[:260]}")
        return []

    rows: list[dict] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def summarize_candidate(plan: FitEstimate, budget_mib: int, rows: list[dict]) -> dict | None:
    if not rows:
        return None
    metrics: dict[str, float] = {}
    for r in rows:
        depth = int(r.get("n_depth", 0))
        if int(r.get("n_prompt", 0)) > 0:
            metrics[f"pp_d{depth}"] = round(float(r.get("avg_ts", 0.0)), 2)
        if int(r.get("n_gen", 0)) > 0:
            metrics[f"tg_d{depth}"] = round(float(r.get("avg_ts", 0.0)), 2)
    return {
        "backend": plan.backend,
        "kv_type": plan.kv_type,
        "ngl": plan.ngl,
        "ncmoe": plan.ncmoe,
        "fits_vram": plan.gpu_total_mib <= budget_mib,
        "fit_estimate": asdict(plan) | {"gpu_total_mib": plan.gpu_total_mib, "gpu_total_gib": plan.gpu_total_gib},
        "metrics": metrics,
    }


def pick_winner(candidates: list[dict]) -> dict:
    """Select best candidate: prefer fits_vram, rank by long-context tg, tie-break toward q8_0 within 5%."""
    fitting = [c for c in candidates if c.get("fits_vram")]
    pool = fitting if fitting else candidates

    def _speed_key(c: dict) -> tuple[float, float, float]:
        m = c["metrics"]
        return (
            float(m.get("tg_d16384", 0.0)),
            float(m.get("tg_d0", 0.0)),
            float(m.get("pp_d16384", 0.0)),
        )

    best = max(pool, key=_speed_key)
    best_tg = _speed_key(best)[0] or _speed_key(best)[1]

    # Quality tie-breaker: if a q8_0 candidate with same or better offload is within 5% speed, prefer q8_0
    if best["kv_type"] != "q8_0" and best_tg > 0:
        q8_near = [
            c for c in pool
            if c["kv_type"] == "q8_0"
            and c["ncmoe"] <= best["ncmoe"]
            and c["ngl"] >= best["ngl"]
            and (_speed_key(c)[0] or _speed_key(c)[1]) >= 0.95 * best_tg
        ]
        if q8_near:
            best = max(q8_near, key=_speed_key)

    return best


# ---------------------------------------------------------------------------
# Phase 3: Live full-65K llama-server verification + Windows shared VRAM check
# ---------------------------------------------------------------------------


def _query_win_gpu_memory_mb(pid: int) -> tuple[float | None, float | None]:
    """Query Windows GPU Process Memory Dedicated & Shared Usage (MiB) for a PID."""
    if os.name != "nt":
        return None, None
    ps = (
        f"$p = Get-Counter '\\GPU Process Memory(pid_{pid}_*)\\Dedicated Usage',"
        f"'\\GPU Process Memory(pid_{pid}_*)\\Shared Usage' -ErrorAction SilentlyContinue; "
        "if ($p) { "
        "$ded = ($p.CounterSamples | Where-Object { $_.Path -like '*dedicated*' } | Measure-Object CookedValue -Sum).Sum; "
        "$shr = ($p.CounterSamples | Where-Object { $_.Path -like '*shared*' } | Measure-Object CookedValue -Sum).Sum; "
        "Write-Output ([math]::Round($ded/1MB,1).ToString() + ',' + [math]::Round($shr/1MB,1).ToString()) }"
    )
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps],
            capture_output=True,
            text=True,
            timeout=10,
        )
        line = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
        if "," in line:
            a, b = line.split(",", 1)
            return float(a), float(b)
    except Exception:
        pass
    return None, None


def _kill_proc_tree_win(proc: subprocess.Popen, job: int | None) -> None:
    if job and os.name == "nt":
        try:
            k32 = ctypes.windll.kernel32
            k32.TerminateJobObject.argtypes = [ctypes.c_void_p, ctypes.c_uint]
            k32.TerminateJobObject.restype = ctypes.c_int
            k32.CloseHandle.argtypes = [ctypes.c_void_p]
            k32.CloseHandle.restype = ctypes.c_int
            k32.TerminateJobObject(ctypes.c_void_p(job), 1)
            k32.CloseHandle(ctypes.c_void_p(job))
        except Exception:
            pass
    try:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=5)
    except Exception:
        pass


def verify_live_server(
    winner: dict,
    model_path: Path,
    alias: str,
    ctx_size: int,
    lora_path: Path | None = None,
    port: int = 8089,
    startup_timeout: int = 180,
) -> dict:
    """Start llama-server with the full -c <ctx>, probe a completion, and inspect real VRAM/spill."""
    bdir, dev, env = _backend_env_and_dir(winner["backend"])
    exe = bdir / "llama-server.exe"
    cmd = [
        str(exe),
        "-m", str(model_path),
        "--alias", alias,
        "-dev", dev,
        "-ngl", str(winner["ngl"]),
        "-fa", "on",
        "-c", str(ctx_size),
        "-np", "1",
        "-ctk", winner["kv_type"],
        "-ctv", winner["kv_type"],
        "--jinja",
        "--metrics",
        "--host", "127.0.0.1",
        "--port", str(port),
    ]
    if winner["ncmoe"] > 0:
        cmd.extend(["-ncmoe", str(winner["ncmoe"])])
    if lora_path is not None:
        cmd.extend(["--lora", str(lora_path)])

    creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creationflags,
    )
    job = None
    if os.name == "nt":
        try:
            k32 = ctypes.windll.kernel32
            k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
            k32.CreateJobObjectW.restype = ctypes.c_void_p
            k32.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            k32.AssignProcessToJobObject.restype = ctypes.c_int
            j = k32.CreateJobObjectW(None, None)
            if j and k32.AssignProcessToJobObject(j, int(proc._handle)):
                job = int(j)
        except Exception:
            job = None

    base_url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.time() + startup_timeout
        healthy = False
        while time.time() < deadline:
            if proc.poll() is not None:
                return {"ok": False, "error": f"llama-server exited early (rc={proc.returncode})"}
            try:
                with urllib.request.urlopen(f"{base_url}/health", timeout=2) as resp:
                    if resp.status == 200:
                        healthy = True
                        break
            except Exception:
                time.sleep(1.0)

        if not healthy:
            return {"ok": False, "error": f"llama-server did not become healthy within {startup_timeout}s"}

        # Run a quick 64-token completion to trigger full compute buffers and measure real speed
        payload = json.dumps({
            "model": alias,
            "messages": [{"role": "user", "content": "Count from 1 to 40 separated by commas."}],
            "max_tokens": 64,
            "temperature": 0.0,
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{base_url}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=120) as resp:
            comp = json.loads(resp.read().decode("utf-8", errors="replace"))
        wall_sec = round(time.perf_counter() - t0, 2)

        timings = comp.get("timings") or {}
        prompt_tps = round(float(timings["prompt_per_second"]), 2) if "prompt_per_second" in timings else None
        gen_tps = round(float(timings["predicted_per_second"]), 2) if "predicted_per_second" in timings else None
        dedicated_mb, shared_mb = _query_win_gpu_memory_mb(proc.pid)

        return {
            "ok": True,
            "ctx_size": ctx_size,
            "prompt_tps": prompt_tps,
            "gen_tps": gen_tps,
            "wall_sec": wall_sec,
            "gpu_dedicated_mb": dedicated_mb,
            "gpu_shared_mb": shared_mb,
            "shared_spill_warning": (shared_mb is not None and shared_mb > 300.0),
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    finally:
        _kill_proc_tree_win(proc, job)


# ---------------------------------------------------------------------------
# Launcher script generation & CLI
# ---------------------------------------------------------------------------


def render_start_script(
    model_path: Path,
    alias: str,
    winner: dict,
    ctx_size: int,
    lora_path: Path | None = None,
    server_verify: dict | None = None,
) -> str:
    backend = winner["backend"]
    ngl = int(winner["ngl"])
    ncmoe = int(winner["ncmoe"])
    kv_type = winner["kv_type"]
    m = winner.get("metrics") or {}
    fit = winner.get("fit_estimate") or {}
    size_gb = model_path.stat().st_size / (1024**3)

    lines = [
        "@echo off",
        f"rem Auto-tuned by bench/optimize.py on {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"rem Model: {model_path.name} ({size_gb:.2f} GB), backend={backend}, ngl={ngl}, ncmoe={ncmoe}, kv={kv_type}, ctx={ctx_size}",
        f"rem Estimated {ctx_size} ctx VRAM: {fit.get('gpu_total_mib', '?')} MiB ({fit.get('gpu_total_gib', '?')} GiB) | Host: {fit.get('host_total_mib', '?')} MiB",
        f"rem llama-bench: pp@0={m.get('pp_d0', 0)} t/s, tg@0={m.get('tg_d0', 0)} t/s | pp@16k={m.get('pp_d16384', 0)} t/s, tg@16k={m.get('tg_d16384', 0)} t/s",
    ]
    if server_verify and server_verify.get("ok"):
        lines.append(
            f"rem Verified live {ctx_size} ctx server: pp={server_verify.get('prompt_tps')} t/s, "
            f"tg={server_verify.get('gen_tps')} t/s, dedicated={server_verify.get('gpu_dedicated_mb')} MiB, "
            f"shared={server_verify.get('gpu_shared_mb')} MiB"
        )

    if backend == "hip":
        lines.append(f'set "PATH={ROCM_BIN_DIR};%PATH%"')
        exe_rel = r"%~dp0llama.cpp-hip\llama-server.exe"
        dev_flag = "-dev ROCm0"
    else:
        lines.append("set GGML_VK_DISABLE_COOPMAT=1")
        exe_rel = r"%~dp0llama.cpp-vulkan\llama-server.exe"
        dev_flag = "-dev Vulkan0"

    try:
        model_rel = "%~dp0" + str(model_path.resolve().relative_to(ROOT_DIR.resolve()))
    except ValueError:
        model_rel = str(model_path.resolve())

    lora_part = ""
    if lora_path is not None:
        try:
            lora_rel = "%~dp0" + str(lora_path.resolve().relative_to(ROOT_DIR.resolve()))
        except ValueError:
            lora_rel = str(lora_path.resolve())
        lora_part = f' --lora "{lora_rel}"'

    ncmoe_part = f" -ncmoe {ncmoe}" if ncmoe > 0 else ""

    lines.append(f'"{exe_rel}" -m "{model_rel}"{lora_part} --alias {alias} ^')
    lines.append(
        f"  {dev_flag} -ngl {ngl}{ncmoe_part} -fa on -c {ctx_size} -np 1 -ctk {kv_type} -ctv {kv_type} "
        "--jinja --host 127.0.0.1 --port 8080 %*"
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Optimize llama-server parameters and verify 65K VRAM fit on RX 9060 XT.")
    ap.add_argument("model", type=Path, help="Path to .gguf model file")
    ap.add_argument("--alias", default=None, help="Model alias for llama-server (default: derived from filename)")
    ap.add_argument("--lora", type=Path, default=None, help="Optional LoRA .gguf file to account for & include")
    ap.add_argument("--ctx", type=int, default=65536, help="Target context window (default: 65536)")
    ap.add_argument("--kv-types", default="q8_0,q4_0", help="Comma-separated KV cache types to sweep (default: q8_0,q4_0)")
    ap.add_argument("--backends", default="hip,vulkan", help="Comma-separated backends to sweep (default: hip,vulkan)")
    ap.add_argument("--target-vram-gb", type=float, default=13.0, help="Usable VRAM budget in GiB (default: 13.0)")
    ap.add_argument("--depths", default="0,16384", help="Comma-separated context depths for llama-bench")
    ap.add_argument("--reps", type=int, default=2, help="Repetitions per llama-bench point")
    ap.add_argument("--ncmoe-values", default=None, help="Override automatic -ncmoe search with explicit comma-separated values")
    ap.add_argument("--plan-only", action="store_true", help="Only run Phase 1 (exact 65K VRAM fit planning) and exit")
    ap.add_argument("--no-verify-server", action="store_true", help="Skip Phase 3 live llama-server 65K verification")
    ap.add_argument("--write-script", action="store_true", help="Write E:\\local-llm\\start-<alias>.cmd")
    args = ap.parse_args()

    model_path = args.model.resolve()
    if not model_path.exists():
        print(f"Error: model file not found: {model_path}", file=sys.stderr)
        return 1

    lora_path = args.lora.resolve() if args.lora else None
    if lora_path and not lora_path.exists():
        print(f"Error: LoRA file not found: {lora_path}", file=sys.stderr)
        return 1

    alias = args.alias or slugify_model(model_path)
    size_gb = model_path.stat().st_size / (1024**3)
    kv_types = [x.strip() for x in args.kv_types.split(",") if x.strip()]
    backends = [x.strip().lower() for x in args.backends.split(",") if x.strip()]
    depths = tuple(int(x.strip()) for x in args.depths.split(",") if x.strip())
    explicit_ncmoe = [int(x.strip()) for x in args.ncmoe_values.split(",") if x.strip()] if args.ncmoe_values else None
    budget_mib = int(round(args.target_vram_gb * 1024))

    print(
        f"[Phase 1/3] Planning {args.ctx} ctx VRAM fit for {model_path.name} ({size_gb:.2f} GiB) | "
        f"budget={budget_mib} MiB ({args.target_vram_gb:.1f} GiB) | kv={kv_types}"
    )
    t0 = time.time()
    is_moe, plans = plan_candidates(
        model_path=model_path,
        ctx_size=args.ctx,
        kv_types=kv_types,
        budget_mib=budget_mib,
        backends=backends,
        lora_path=lora_path,
        explicit_ncmoe=explicit_ncmoe,
    )
    if not plans:
        print("Error: llama-fit-params produced no valid configurations.", file=sys.stderr)
        if _LAST_FIT_ERROR:
            print(f"Reason from llama-fit-params:\n{_LAST_FIT_ERROR}", file=sys.stderr)
        return 1

    print(f"  Detected architecture: {'MoE' if is_moe else 'Dense'}")
    print(f"  {'Backend':<8} {'KV':<5} {'ngl':>3} {'ncmoe':>5} {'Weights':>8} {'KV MiB':>7} {'Comp':>6} {'LoRA':>5} {'GPU Total':>10} {'Host':>7} {'Status'}")
    for p in plans:
        st = "FITS" if p.gpu_total_mib <= budget_mib else f"SPILL (+{p.gpu_total_mib - budget_mib} MiB)"
        print(
            f"  {p.backend:<8} {p.kv_type:<5} {p.ngl:>3} {p.ncmoe:>5} {p.gpu_model_mib:>8} "
            f"{p.gpu_ctx_mib:>7} {p.gpu_compute_mib:>6} {p.lora_mib:>5} {p.gpu_total_mib:>10} "
            f"{p.host_total_mib:>7} {st}"
        )

    if args.plan_only:
        return 0

    print(f"\n[Phase 2/3] Benchmarking {len(plans)} configuration(s) with llama-bench (depths={depths})...")
    candidates: list[dict] = []
    for p in plans:
        rows = run_llama_bench(p, model_path, depths=depths, reps=args.reps)
        summary = summarize_candidate(p, budget_mib, rows)
        if summary:
            candidates.append(summary)
            m = summary["metrics"]
            print(
                f"     => pp@0={m.get('pp_d0', 0):.1f} tg@0={m.get('tg_d0', 0):.1f} | "
                f"pp@16k={m.get('pp_d16384', 0):.1f} tg@16k={m.get('tg_d16384', 0):.1f} t/s"
            )

    if not candidates:
        print("Error: no llama-bench configurations succeeded.", file=sys.stderr)
        return 1

    winner = pick_winner(candidates)
    print("\n=== Recommended Configuration ===")
    print(json.dumps(winner, indent=2))

    server_verify: dict | None = None
    if not args.no_verify_server:
        print(f"\n[Phase 3/3] Verifying full -c {args.ctx} allocation in live llama-server...")
        server_verify = verify_live_server(
            winner=winner,
            model_path=model_path,
            alias=alias,
            ctx_size=args.ctx,
            lora_path=lora_path,
        )
        print("  Live 65K verification:", json.dumps(server_verify))
        if server_verify.get("shared_spill_warning"):
            print(
                f"  [WARN] Windows reported {server_verify.get('gpu_shared_mb')} MiB shared GPU RAM spill! "
                "Consider lowering --target-vram-gb (e.g. 12.5) and re-running."
            )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    profile_record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "alias": alias,
        "model_file": model_path.name,
        "lora_file": lora_path.name if lora_path else None,
        "is_moe": is_moe,
        "size_gb": round(size_gb, 2),
        "target_ctx": args.ctx,
        "budget_mib": budget_mib,
        "elapsed_sec": round(time.time() - t0, 1),
        "winner": winner,
        "server_verify": server_verify,
        "candidates": candidates,
    }
    with (RESULTS_DIR / "hw_profiles.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(profile_record) + "\n")

    if args.write_script:
        out_script = ROOT_DIR / f"start-{alias}.cmd"
        out_script.write_text(
            render_start_script(model_path, alias, winner, args.ctx, lora_path, server_verify),
            encoding="utf-8",
        )
        print(f"Wrote launcher script: {out_script}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
