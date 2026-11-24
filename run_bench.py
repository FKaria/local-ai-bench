"""Stage 2: Local math & coding benchmark runner driven by the `pi` coding agent harness.

Guarantees local-only execution (never hits an external provider):
- By default, creates an isolated temporary PI_CODING_AGENT_DIR containing ONLY a
  `local-llama` provider pointing at http://127.0.0.1:<port>/v1 and strips cloud API keys.
- Alternatively, pass `--use-user-pi-config --pi-provider <name>` to use your own
  configured local provider explicitly (--pi-provider is mandatory in that mode).

Anti-snooping guarantees:
- Each task runs in a brand-new temp directory under %TEMP% (outside E:\\local-llm\\bench).
- Reference math answers and coding `hidden_tests/` are never placed in the workspace
  while `pi` runs; hidden tests are copied in ONLY after `pi` exits.
- Every tool call in `pi --mode json` is audited for paths/commands escaping the workspace.

Usage:
    python run_bench.py --script ..\\start-gpt-oss-20b.cmd
    python run_bench.py --no-start-server --model gpt-oss-20b
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from bench_config import (
    BENCH_DIR,
    CLOUD_API_KEY_ENV_VARS,
    RESULTS_DIR,
    ROOT_DIR,
    TASKS_DIR,
    LauncherInfo,
    parse_start_script,
)
from game_envs import extract_game_action, make_game_env
from game_server import LocalGameServer, render_play_client
from math_grader import grade_math


# ---------------------------------------------------------------------------
# HTTP / llama-server helpers
# ---------------------------------------------------------------------------


def http_get_text(url: str, timeout: float = 3.0) -> tuple[int, str]:
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", errors="replace")
    except Exception:
        return 0, ""


def wait_for_server(base_url: str, timeout_sec: int = 180) -> bool:
    deadline = time.time() + timeout_sec
    health_url = f"{base_url}/health"
    while time.time() < deadline:
        code, body = http_get_text(health_url, timeout=2.0)
        if code == 200 and ("ok" in body.lower() or body.strip() == ""):
            return True
        time.sleep(1.0)
    return False


def query_loaded_model_id(base_url: str) -> str | None:
    code, body = http_get_text(f"{base_url}/v1/models", timeout=5.0)
    if code != 200:
        return None
    try:
        payload = json.loads(body)
        data = payload.get("data") or []
        if data and isinstance(data[0], dict):
            return str(data[0].get("id"))
    except json.JSONDecodeError:
        return None
    return None


def scrape_llama_metrics(base_url: str) -> dict[str, float] | None:
    """Parse Prometheus counters from llama-server /metrics (requires --metrics)."""
    code, body = http_get_text(f"{base_url}/metrics", timeout=3.0)
    if code != 200 or not body:
        return None
    wanted = (
        "llamacpp:prompt_tokens_total",
        "llamacpp:prompt_seconds_total",
        "llamacpp:tokens_predicted_total",
        "llamacpp:tokens_predicted_seconds_total",
    )
    out: dict[str, float] = {}
    for line in body.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        for key in wanted:
            if line.startswith(key):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        out[key] = float(parts[-1])
                    except ValueError:
                        pass
    return out if out else None


def start_llama_server(script_path: Path, port: int, log_path: Path) -> subprocess.Popen:
    """Start llama-server via a start-*.cmd launcher, passing --metrics and --port."""
    log_fh = log_path.open("w", encoding="utf-8", errors="replace")
    cmd = ["cmd.exe", "/c", str(script_path), "--port", str(port), "--metrics"]
    creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    proc = subprocess.Popen(
        cmd,
        cwd=str(script_path.parent),
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        creationflags=creationflags,
    )
    proc._bench_log_fh = log_fh  # type: ignore[attr-defined]
    return proc


def stop_llama_server(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    try:
        if proc.poll() is None:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                    capture_output=True,
                    timeout=15,
                )
            else:
                proc.terminate()
                proc.wait(timeout=10)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    finally:
        log_fh = getattr(proc, "_bench_log_fh", None)
        if log_fh is not None:
            try:
                log_fh.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Task loading & grading
# ---------------------------------------------------------------------------


def load_all_tasks(
    selected_ids: set[str] | None = None,
    difficulties: set[str] | None = None,
    categories: set[str] | None = None,
) -> list[dict]:
    tasks: list[dict] = []

    math_file = TASKS_DIR / "math" / "tasks.json"
    if math_file.exists():
        for item in json.loads(math_file.read_text(encoding="utf-8")):
            item.setdefault("category", "math")
            item.setdefault("difficulty", "easy")
            tasks.append(item)

    coding_dir = TASKS_DIR / "coding"
    if coding_dir.exists():
        for task_json in sorted(coding_dir.glob("*/task.json")):
            item = json.loads(task_json.read_text(encoding="utf-8"))
            item.setdefault("category", "coding")
            item.setdefault("difficulty", "easy")
            item["_task_dir"] = str(task_json.parent)
            tasks.append(item)

    games_file = TASKS_DIR / "games" / "tasks.json"
    if games_file.exists():
        for item in json.loads(games_file.read_text(encoding="utf-8")):
            item.setdefault("category", "game")
            item.setdefault("difficulty", "medium")
            tasks.append(item)

    if selected_ids:
        tasks = [t for t in tasks if t["id"] in selected_ids]
    if difficulties:
        tasks = [t for t in tasks if str(t.get("difficulty", "easy")).lower() in difficulties]
    if categories:
        tasks = [t for t in tasks if str(t.get("category", "")).lower() in categories]
    return tasks


def _sha256_bytes(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot_starter_tests(workspace: Path) -> dict[str, str]:
    """Record SHA-256 hashes of starter test_*.py files before pi runs."""
    return {
        p.name: _sha256_bytes(p)
        for p in workspace.glob("test_*.py")
        if p.is_file()
    }


def verify_and_restore_canonical_tests(
    task: dict,
    workspace: Path,
    starter_hashes: dict[str, str],
) -> list[str]:
    """Detect modified/deleted starter tests, restore pristine task tests, and remove model-added test_*.py."""
    task_dir = Path(task["_task_dir"])
    starter_dir = task_dir / "starter"
    modified: list[str] = []

    for name, orig_hash in starter_hashes.items():
        ws_file = workspace / name
        if not ws_file.exists():
            modified.append(f"{name} (deleted)")
        elif _sha256_bytes(ws_file) != orig_hash:
            modified.append(f"{name} (modified)")

    # Remove any model-created test_*.py files so grading runs strictly task-defined tests
    for p in list(workspace.glob("test_*.py")):
        if p.name not in starter_hashes:
            modified.append(f"{p.name} (added by model)")
            try:
                p.unlink()
            except OSError:
                pass

    # Restore pristine starter test_*.py files unconditionally before running unittest
    if starter_dir.exists():
        for src in starter_dir.glob("test_*.py"):
            shutil.copy2(src, workspace / src.name)

    return modified


def grade_math_task(task: dict, workspace: Path, final_text: str) -> tuple[bool, str, str]:
    return grade_math(task, workspace, final_text)


def grade_coding_task(
    task: dict,
    workspace: Path,
    starter_hashes: dict[str, str] | None = None,
) -> tuple[bool, str, str, list[str]]:
    modified_tests: list[str] = []
    if starter_hashes is not None:
        modified_tests = verify_and_restore_canonical_tests(task, workspace, starter_hashes)

    task_dir = Path(task["_task_dir"])
    hidden_dir = task_dir / "hidden_tests"
    if hidden_dir.exists():
        for src in hidden_dir.iterdir():
            dst = workspace / src.name
            if src.is_dir():
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)

    verify_cmd = list(task.get("verify_cmd") or ["python", "-m", "unittest", "discover", "-s", ".", "-p", "test_*.py", "-v"])
    if verify_cmd and verify_cmd[0] == "python":
        verify_cmd[0] = sys.executable

    verify_timeout = int(task.get("verify_timeout_sec", 60))
    try:
        proc = subprocess.run(
            verify_cmd,
            cwd=str(workspace),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=verify_timeout,
        )
    except subprocess.TimeoutExpired:
        return (
            False,
            "test_timeout",
            f"Verification command timed out after {verify_timeout}s (solution too slow or infinite loop)",
            modified_tests,
        )

    output = (proc.stdout + "\n" + proc.stderr).strip()
    if proc.returncode == 0:
        return True, "passed", output[-2000:], modified_tests
    return False, "test_failed", output[-2000:], modified_tests


# ---------------------------------------------------------------------------
# Pi execution & JSONL telemetry + anti-snooping audit
# ---------------------------------------------------------------------------


def find_pi_executable() -> str:
    for name in ("pi.cmd", "pi.exe", "pi"):
        found = shutil.which(name)
        if found:
            return found
    raise FileNotFoundError("Could not find `pi` on PATH. Is @earendil-works/pi-coding-agent installed?")


def create_isolated_pi_config(config_dir: Path, base_url: str, model_id: str, ctx_size: int) -> None:
    """Write a locked-down models.json containing ONLY the local llama-server provider."""
    config_dir.mkdir(parents=True, exist_ok=True)
    models_json = {
        "providers": {
            "local-llama": {
                "name": "Local llama-server",
                "baseUrl": f"{base_url}/v1",
                "api": "openai-completions",
                "apiKey": "local-bench-key",
                "models": [
                    {
                        "id": model_id,
                        "name": model_id,
                        "reasoning": True,
                        "input": ["text"],
                        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                        "contextWindow": ctx_size,
                        "maxTokens": max(4096, ctx_size - 4096),
                    }
                ],
            }
        }
    }
    (config_dir / "models.json").write_text(json.dumps(models_json, indent=2), encoding="utf-8")


SNOOP_PATTERNS = (
    re.compile(r"hidden_tests", re.IGNORECASE),
    re.compile(r"tasks\.json", re.IGNORECASE),
    re.compile(r"local-llm[\\/]+bench", re.IGNORECASE),
    re.compile(r"bench[\\/]+tasks", re.IGNORECASE),
    re.compile(r"(?:^|[\s\"'\\/])\.\.(?:[\s\"'\\/]|$)"),
)


def audit_tool_call_for_snooping(tool_name: str, args: dict, workspace: Path) -> list[str]:
    """Flag any tool call attempting to read or traverse outside the isolated task workspace."""
    flags: list[str] = []
    ws_resolved = workspace.resolve()
    bench_str = str(BENCH_DIR.resolve()).lower()
    root_str = str(ROOT_DIR.resolve()).lower()

    # Inspect only path/command/search arguments (never file content/edits text, which can legitimately contain '..')
    inspect_keys = ("path", "file_path", "cwd", "directory", "target", "command", "pattern", "glob")
    inspected_values = [str(args[k]) for k in inspect_keys if k in args and isinstance(args[k], str)]
    combined = "\n".join(inspected_values)
    lower_combined = combined.lower()

    if bench_str in lower_combined or root_str in lower_combined:
        flags.append(f"{tool_name}: referenced benchmark/workspace root path")

    for pat in SNOOP_PATTERNS:
        if pat.search(combined):
            flags.append(f"{tool_name}: matched restricted pattern {pat.pattern!r}")

    for key in ("path", "file_path", "cwd", "directory", "target"):
        val = args.get(key)
        if isinstance(val, str) and val.strip():
            candidate = Path(val)
            try:
                resolved = candidate.resolve() if candidate.is_absolute() else (ws_resolved / candidate).resolve()
                if not resolved.is_relative_to(ws_resolved):
                    flags.append(f"{tool_name}: accessed path outside workspace ({val})")
            except Exception:
                pass

    return flags


def parse_pi_jsonl(stdout_text: str, workspace: Path) -> dict:
    """Parse `pi --mode json` JSONL output into structured trace, tokens, and snooping audit."""
    turns = 0
    input_tokens = 0
    output_tokens = 0
    cache_read_tokens = 0
    tool_calls: list[dict] = []
    tool_errors = 0
    snooping_flags: list[str] = []
    assistant_texts: list[str] = []
    trace: list[dict] = []
    stop_reason = "unknown"
    parse_errors = 0
    pending_tools: dict[str, dict] = {}

    for line in stdout_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            parse_errors += 1
            continue

        ev_type = ev.get("type", "")

        if ev_type == "tool_execution_start":
            tc_id = str(ev.get("toolCallId", len(tool_calls)))
            t_name = str(ev.get("toolName", "unknown"))
            t_args = ev.get("args") if isinstance(ev.get("args"), dict) else {}
            flags = audit_tool_call_for_snooping(t_name, t_args, workspace)
            snooping_flags.extend(flags)
            rec = {
                "id": tc_id,
                "tool": t_name,
                "args": t_args,
                "is_error": False,
                "duration_ms": None,
                "output_preview": "",
                "snooping_flags": flags,
            }
            pending_tools[tc_id] = rec
            tool_calls.append(rec)

        elif ev_type == "tool_execution_end":
            tc_id = str(ev.get("toolCallId", ""))
            is_err = bool(ev.get("isError", False))
            if is_err:
                tool_errors += 1
            rec = pending_tools.get(tc_id)
            if rec is not None:
                rec["is_error"] = is_err
                rec["duration_ms"] = ev.get("durationMs")
                res = ev.get("result")
                if isinstance(res, dict):
                    content = res.get("content") or []
                    text_parts = [c.get("text", "") for c in content if isinstance(c, dict)]
                    rec["output_preview"] = "\n".join(text_parts)[:600]
                trace.append({
                    "kind": "tool",
                    "tool": rec["tool"],
                    "args": rec["args"],
                    "is_error": rec["is_error"],
                    "duration_ms": rec["duration_ms"],
                    "output_preview": rec["output_preview"],
                    "snooping_flags": rec["snooping_flags"],
                })

        elif ev_type == "message_end":
            msg = ev.get("message") or {}
            if msg.get("role") == "assistant":
                turns += 1
                stop_reason = str(msg.get("stopReason", stop_reason))
                usage = msg.get("usage") or {}
                input_tokens += int(usage.get("input", 0) or 0)
                output_tokens += int(usage.get("output", 0) or 0)
                cache_read_tokens += int(usage.get("cacheRead", 0) or 0)

                content = msg.get("content") or []
                text_chunks: list[str] = []
                thinking_chunks: list[str] = []
                if isinstance(content, list):
                    for block in content:
                        if not isinstance(block, dict):
                            continue
                        btype = block.get("type")
                        if btype == "text" and block.get("text"):
                            text_chunks.append(str(block["text"]))
                        elif btype in ("thinking", "reasoning") and (block.get("thinking") or block.get("text")):
                            thinking_chunks.append(str(block.get("thinking") or block.get("text")))
                elif isinstance(content, str):
                    text_chunks.append(content)

                joined_text = "\n".join(text_chunks).strip()
                if joined_text:
                    assistant_texts.append(joined_text)
                trace.append({
                    "kind": "assistant",
                    "turn": turns,
                    "stop_reason": stop_reason,
                    "thinking_preview": "\n".join(thinking_chunks)[:800],
                    "text": joined_text[:1500],
                    "usage": usage,
                })

    return {
        "turns": turns,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_tokens": cache_read_tokens,
        "tool_call_count": len(tool_calls),
        "tool_error_count": tool_errors,
        "snooping_flags": snooping_flags,
        "final_text": assistant_texts[-1] if assistant_texts else "",
        "stop_reason": stop_reason,
        "parse_errors": parse_errors,
        "trace": trace,
    }


def _create_win_job_for_proc(proc: subprocess.Popen) -> int | None:
    """Attach a Windows process to a Win32 Job Object so its entire tree can be killed atomically."""
    if os.name != "nt":
        return None
    try:
        k32 = ctypes.windll.kernel32
        k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
        k32.CreateJobObjectW.restype = ctypes.c_void_p
        k32.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        k32.AssignProcessToJobObject.restype = ctypes.c_int
        k32.CloseHandle.argtypes = [ctypes.c_void_p]
        k32.CloseHandle.restype = ctypes.c_int

        job = k32.CreateJobObjectW(None, None)
        if not job:
            return None
        if not k32.AssignProcessToJobObject(job, int(proc._handle)):
            k32.CloseHandle(job)
            return None
        return int(job)
    except Exception:
        return None


def _close_win_job(job: int | None, terminate: bool = False) -> None:
    if not job or os.name != "nt":
        return
    try:
        k32 = ctypes.windll.kernel32
        if terminate:
            k32.TerminateJobObject.argtypes = [ctypes.c_void_p, ctypes.c_uint]
            k32.TerminateJobObject.restype = ctypes.c_int
            k32.TerminateJobObject(ctypes.c_void_p(job), 1)
        k32.CloseHandle.argtypes = [ctypes.c_void_p]
        k32.CloseHandle.restype = ctypes.c_int
        k32.CloseHandle(ctypes.c_void_p(job))
    except Exception:
        pass


def kill_process_tree(proc: subprocess.Popen, job: int | None = None) -> None:
    """Terminate a process and all its children (required on Windows for pi.cmd -> node.exe)."""
    if job is not None:
        _close_win_job(job, terminate=True)
        try:
            proc.wait(timeout=3)
        except Exception:
            pass
    if proc.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                capture_output=True,
                timeout=5,
            )
        else:
            proc.kill()
        proc.wait(timeout=3)
    except Exception:
        try:
            proc.kill()
            proc.wait(timeout=2)
        except Exception:
            pass


def run_with_tree_timeout(
    cmd: list[str],
    cwd: Path,
    env: dict[str, str],
    timeout_sec: int,
) -> tuple[int, str, str, bool]:
    """Run subprocess capturing stdout/stderr live and killing the full process tree on timeout."""
    creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=creationflags,
    )
    job = _create_win_job_for_proc(proc)
    out_chunks: list[str] = []
    err_chunks: list[str] = []

    def _drain(stream, bucket: list[str]) -> None:
        try:
            for chunk in iter(stream.readline, ""):
                bucket.append(chunk)
        except Exception:
            pass
        finally:
            try:
                stream.close()
            except Exception:
                pass

    t_out = threading.Thread(target=_drain, args=(proc.stdout, out_chunks), daemon=True)
    t_err = threading.Thread(target=_drain, args=(proc.stderr, err_chunks), daemon=True)
    t_out.start()
    t_err.start()

    timed_out = False
    try:
        exit_code = proc.wait(timeout=timeout_sec)
        _close_win_job(job, terminate=False)
    except subprocess.TimeoutExpired:
        timed_out = True
        kill_process_tree(proc, job=job)
        exit_code = -1

    t_out.join(timeout=3.0)
    t_err.join(timeout=3.0)
    return exit_code, "".join(out_chunks), "".join(err_chunks), timed_out


def run_rpc_game_session(
    cmd: list[str],
    cwd: Path,
    env_vars: dict[str, str],
    game_env,
    game_type: str,
    initial_prompt: str,
    timeout_sec: int,
    max_consecutive_invalid: int = 5,
) -> tuple[int, dict, str, bool]:
    """Drive a multi-turn tool-free game inside a single `pi --mode rpc --no-tools` process."""
    creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env_vars,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        creationflags=creationflags,
    )
    job = _create_win_job_for_proc(proc)
    out_q: queue.Queue[str | None] = queue.Queue()
    err_chunks: list[str] = []

    def _read_stdout() -> None:
        try:
            assert proc.stdout is not None
            for line in iter(proc.stdout.readline, ""):
                out_q.put(line)
        except Exception:
            pass
        finally:
            try:
                if proc.stdout:
                    proc.stdout.close()
            except Exception:
                pass
            out_q.put(None)

    def _read_stderr() -> None:
        try:
            assert proc.stderr is not None
            for line in iter(proc.stderr.readline, ""):
                err_chunks.append(line)
        except Exception:
            pass
        finally:
            try:
                if proc.stderr:
                    proc.stderr.close()
            except Exception:
                pass

    t_out = threading.Thread(target=_read_stdout, daemon=True)
    t_err = threading.Thread(target=_read_stderr, daemon=True)
    t_out.start()
    t_err.start()

    deadline = time.perf_counter() + timeout_sec
    timed_out = False
    turns = 0
    input_tokens = 0
    output_tokens = 0
    cache_read_tokens = 0
    stop_reason = "unknown"
    final_text = ""
    trace: list[dict] = []
    consec_invalid = 0
    prev_invalid_total = 0

    try:
        obs = game_env.reset()
        next_msg = f"{initial_prompt}\n\n{obs}"
        step_idx = 0

        while not game_env.status()["done"]:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                timed_out = True
                break

            step_idx += 1
            req = {"id": f"turn-{step_idx}", "type": "prompt", "message": next_msg}
            assert proc.stdin is not None
            try:
                proc.stdin.write(json.dumps(req) + "\n")
                proc.stdin.flush()
            except OSError:
                break

            turn_text = ""
            turn_thinking = ""
            turn_usage: dict = {}
            got_agent_end = False

            while True:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    timed_out = True
                    break
                try:
                    line = out_q.get(timeout=min(0.5, remaining))
                except queue.Empty:
                    if proc.poll() is not None:
                        break
                    continue
                if line is None:
                    break
                line_s = line.strip()
                if not line_s.startswith("{"):
                    continue
                try:
                    ev = json.loads(line_s)
                except json.JSONDecodeError:
                    continue

                ev_type = ev.get("type", "")
                if ev_type == "response" and ev.get("command") == "prompt" and ev.get("success") is False:
                    err_chunks.append(f"RPC prompt rejected: {ev.get('error', '')}\n")
                    break
                elif ev_type == "message_end":
                    msg = ev.get("message") or {}
                    if msg.get("role") == "assistant":
                        turns += 1
                        stop_reason = str(msg.get("stopReason", stop_reason))
                        usage = msg.get("usage") or {}
                        turn_usage = usage
                        input_tokens += int(usage.get("input", 0) or 0)
                        output_tokens += int(usage.get("output", 0) or 0)
                        cache_read_tokens += int(usage.get("cacheRead", 0) or 0)
                        content = msg.get("content") or []
                        t_chunks: list[str] = []
                        th_chunks: list[str] = []
                        if isinstance(content, list):
                            for blk in content:
                                if not isinstance(blk, dict):
                                    continue
                                bt = blk.get("type")
                                if bt == "text" and blk.get("text"):
                                    t_chunks.append(str(blk["text"]))
                                elif bt in ("thinking", "reasoning") and (blk.get("thinking") or blk.get("text")):
                                    th_chunks.append(str(blk.get("thinking") or blk.get("text")))
                        elif isinstance(content, str):
                            t_chunks.append(content)
                        turn_text = "\n".join(t_chunks).strip()
                        turn_thinking = "\n".join(th_chunks).strip()
                        if turn_text:
                            final_text = turn_text
                elif ev_type == "agent_end":
                    got_agent_end = True
                    break

            if timed_out or not got_agent_end:
                break

            trace.append({
                "kind": "assistant",
                "turn": turns,
                "stop_reason": stop_reason,
                "thinking_preview": turn_thinking[:800],
                "text": turn_text[:1500],
                "usage": turn_usage,
            })

            action = extract_game_action(turn_text, game_type)
            obs, done, st = game_env.step(action)
            invalid_now = int(st.get("invalid_moves", 0))
            is_invalid = invalid_now > prev_invalid_total
            prev_invalid_total = invalid_now
            consec_invalid = (consec_invalid + 1) if is_invalid else 0

            trace.append({
                "kind": "tool",
                "tool": f"game:{game_type}",
                "args": {"action": action},
                "is_error": is_invalid,
                "duration_ms": 0,
                "output_preview": obs[:600],
                "snooping_flags": [],
            })

            if done or consec_invalid >= max_consecutive_invalid:
                break

            next_msg = (
                f"{obs}\n\n"
                "Choose your next move and put it inside square brackets at the end of your response."
            )
    finally:
        try:
            if proc.stdin:
                proc.stdin.close()
        except Exception:
            pass
        kill_process_tree(proc, job=job)
        t_out.join(timeout=2.0)
        t_err.join(timeout=2.0)

    exit_code = -1 if timed_out else (proc.returncode if proc.returncode is not None else 0)
    parsed = {
        "turns": turns,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_tokens": cache_read_tokens,
        "tool_call_count": sum(1 for item in trace if item["kind"] == "tool"),
        "tool_error_count": sum(1 for item in trace if item["kind"] == "tool" and item.get("is_error")),
        "snooping_flags": [],
        "final_text": final_text,
        "stop_reason": stop_reason,
        "parse_errors": 0,
        "trace": trace,
    }
    return exit_code, parsed, "".join(err_chunks), timed_out


def run_single_task(
    task: dict,
    pi_exe: str,
    provider: str,
    model_id: str,
    base_url: str,
    pi_env: dict[str, str],
    thinking: str | None = None,
    timeout_override: int | None = None,
) -> dict:
    task_id = task["id"]
    timeout_sec = int(timeout_override if timeout_override is not None else task.get("timeout_sec", 180))

    # Isolated temp directory strictly outside E:\local-llm\bench
    workspace = Path(tempfile.mkdtemp(prefix=f"pi-bench-{task_id}-"))
    starter_test_hashes: dict[str, str] = {}
    allow_tools = bool(task.get("allow_tools", True))
    game_server: LocalGameServer | None = None
    rpc_game_env = None
    try:
        if task["category"] == "coding":
            starter_dir = Path(task["_task_dir"]) / "starter"
            if starter_dir.exists():
                for src in starter_dir.iterdir():
                    dst = workspace / src.name
                    if src.is_dir():
                        shutil.copytree(src, dst)
                    else:
                        shutil.copy2(src, dst)
            starter_test_hashes = snapshot_starter_tests(workspace)
        elif task["category"] == "game":
            env = make_game_env(task)
            if allow_tools:
                game_server = LocalGameServer(env)
                game_port = game_server.start()
                (workspace / "play.py").write_text(render_play_client(game_port), encoding="utf-8")
            else:
                rpc_game_env = env

        metrics_before = scrape_llama_metrics(base_url)

        t0 = time.perf_counter()
        if task["category"] == "game" and rpc_game_env is not None:
            cmd = [
                pi_exe,
                "--provider", provider,
                "--model", model_id,
                "--mode", "rpc",
                "--no-tools",
                "--no-session",
                "--no-extensions",
                "--no-mcp",
                "--no-skills",
                "--no-prompt-templates",
                "--no-context-files",
                "--no-approve",
                "--offline",
            ]
            if thinking:
                cmd.extend(["--thinking", thinking])
            exit_code, parsed, stderr_text, timed_out = run_rpc_game_session(
                cmd=cmd,
                cwd=workspace,
                env_vars=pi_env,
                game_env=rpc_game_env,
                game_type=str(task["game"]),
                initial_prompt=task["prompt"],
                timeout_sec=timeout_sec,
            )
        else:
            cmd = [
                pi_exe,
                "--provider", provider,
                "--model", model_id,
                "--mode", "json",
                "-p",
                "--no-session",
                "--no-extensions",
                "--no-mcp",
                "--no-skills",
                "--no-prompt-templates",
                "--no-context-files",
                "--no-approve",
                "--offline",
            ]
            if not allow_tools:
                cmd.append("--no-tools")
            if thinking:
                cmd.extend(["--thinking", thinking])
            cmd.extend(["--", task["prompt"]])

            exit_code, stdout_text, stderr_text, timed_out = run_with_tree_timeout(
                cmd=cmd,
                cwd=workspace,
                env=pi_env,
                timeout_sec=timeout_sec,
            )
            parsed = parse_pi_jsonl(stdout_text, workspace)

        wall_sec = round(time.perf_counter() - t0, 2)
        metrics_after = scrape_llama_metrics(base_url)

        # Compute real server throughput deltas from llama-server /metrics when available
        prompt_tps: float | None = None
        gen_tps: float | None = None
        server_prompt_tokens: int | None = None
        server_gen_tokens: int | None = None
        if metrics_before and metrics_after:
            dp_tok = metrics_after.get("llamacpp:prompt_tokens_total", 0.0) - metrics_before.get("llamacpp:prompt_tokens_total", 0.0)
            dp_sec = metrics_after.get("llamacpp:prompt_seconds_total", 0.0) - metrics_before.get("llamacpp:prompt_seconds_total", 0.0)
            dg_tok = metrics_after.get("llamacpp:tokens_predicted_total", 0.0) - metrics_before.get("llamacpp:tokens_predicted_total", 0.0)
            dg_sec = metrics_after.get("llamacpp:tokens_predicted_seconds_total", 0.0) - metrics_before.get("llamacpp:tokens_predicted_seconds_total", 0.0)
            if dp_tok > 0 and dp_sec > 0:
                prompt_tps = round(dp_tok / dp_sec, 2)
                server_prompt_tokens = int(round(dp_tok))
            if dg_tok > 0 and dg_sec > 0:
                gen_tps = round(dg_tok / dg_sec, 2)
                server_gen_tokens = int(round(dg_tok))

        # Grade deterministically AFTER pi exits (restoring pristine task tests for coding tasks)
        modified_test_files: list[str] = []
        game_metrics: dict | None = None
        active_game_env = rpc_game_env if rpc_game_env is not None else (game_server.env if game_server is not None else None)
        if task["category"] == "game" and active_game_env is not None:
            st = active_game_env.status()
            game_metrics = st
            if st["won"]:
                passed, status, grade_detail = True, "passed", st["summary"]
            elif timed_out:
                passed, status, grade_detail = False, "timeout", f"Pi timed out after {timeout_sec}s | {st['summary']}"
            elif exit_code != 0 and parsed["turns"] == 0:
                passed, status, grade_detail = False, "harness_error", f"pi exited {exit_code}: {stderr_text.strip()[:500]}"
            else:
                passed, status, grade_detail = False, "game_lost", st["summary"]
        elif timed_out:
            passed, status, grade_detail = False, "timeout", f"Pi timed out after {timeout_sec}s"
            if task["category"] == "coding":
                modified_test_files = verify_and_restore_canonical_tests(task, workspace, starter_test_hashes)
        elif exit_code != 0 and parsed["turns"] == 0:
            passed, status, grade_detail = False, "harness_error", f"pi exited {exit_code}: {stderr_text.strip()[:500]}"
        elif task["category"] == "math":
            passed, status, grade_detail = grade_math_task(task, workspace, parsed["final_text"])
            if not passed and parsed["stop_reason"] in ("length", "maxTokens"):
                grade_detail = f"[TRUNCATED: stop_reason={parsed['stop_reason']}] {grade_detail}"
        else:
            passed, status, grade_detail, modified_test_files = grade_coding_task(
                task, workspace, starter_hashes=starter_test_hashes
            )

        return {
            "task_id": task_id,
            "title": task["title"],
            "category": task["category"],
            "allow_tools": allow_tools,
            "difficulty": task.get("difficulty", "easy"),
            "prompt": task["prompt"],
            "passed": passed,
            "status": status,
            "grade_detail": grade_detail,
            "game_metrics": game_metrics,
            "wall_sec": wall_sec,
            "pi_exit_code": exit_code,
            "pi_stderr_preview": stderr_text.strip()[:500],
            "turns": parsed["turns"],
            "input_tokens": server_prompt_tokens if server_prompt_tokens is not None else parsed["input_tokens"],
            "output_tokens": server_gen_tokens if server_gen_tokens is not None else parsed["output_tokens"],
            "cache_read_tokens": parsed["cache_read_tokens"],
            "prompt_tps": prompt_tps,
            "gen_tps": gen_tps,
            "tool_call_count": parsed["tool_call_count"],
            "tool_error_count": parsed["tool_error_count"],
            "snooping_flags": parsed["snooping_flags"],
            "snooping_detected": len(parsed["snooping_flags"]) > 0,
            "modified_test_files": modified_test_files,
            "tests_tampered": len(modified_test_files) > 0,
            "stop_reason": parsed["stop_reason"],
            "trace": parsed["trace"],
        }
    finally:
        if game_server is not None:
            game_server.stop()
        shutil.rmtree(workspace, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run local math & coding benchmarks using the `pi` harness.")
    ap.add_argument("--script", type=Path, default=None, help="Path to E:\\local-llm\\start-*.cmd launcher")
    ap.add_argument("--no-start-server", action="store_true", help="Use an already-running llama-server")
    ap.add_argument("--model", default=None, help="Model ID/alias served by llama-server (auto-detected if omitted)")
    ap.add_argument("--host", default="127.0.0.1", help="llama-server host (default: 127.0.0.1)")
    ap.add_argument("--port", type=int, default=8080, help="llama-server port (default: 8080)")
    ap.add_argument("--ctx", type=int, default=65536, help="Context window advertised in isolated Pi models.json")
    ap.add_argument("--thinking", default=None, help="Optional Pi --thinking level (off, minimal, low, medium, high)")
    ap.add_argument("--timeout", type=int, default=None, help="Override per-task timeout in seconds")
    ap.add_argument("--tasks", default=None, help="Comma-separated task IDs to run (default: all)")
    ap.add_argument("--difficulty", default=None, help="Filter by difficulty: easy, medium, hard (comma-separated)")
    ap.add_argument("--category", default=None, help="Filter by category: math, coding, game (comma-separated)")
    ap.add_argument(
        "--use-user-pi-config",
        action="store_true",
        help="Use your ~/.pi/agent config instead of an isolated temp config (REQUIRES --pi-provider)",
    )
    ap.add_argument(
        "--pi-provider",
        default=None,
        help="Explicit Pi provider name (required with --use-user-pi-config; defaults to isolated 'local-llama')",
    )
    ap.add_argument("--startup-timeout", type=int, default=180, help="Seconds to wait for llama-server /health")
    ap.add_argument("--no-report", action="store_true", help="Skip rebuilding report.html after the run")
    args = ap.parse_args()

    if args.use_user_pi_config and not args.pi_provider:
        print(
            "Error: --pi-provider <name> is REQUIRED when --use-user-pi-config is set "
            "(prevents accidental fallback to an external default provider).",
            file=sys.stderr,
        )
        return 2

    if not args.no_start_server and not args.script:
        print("Error: pass --script <path-to-start-*.cmd> or --no-start-server.", file=sys.stderr)
        return 2

    launcher: LauncherInfo | None = None
    if args.script:
        launcher = parse_start_script(args.script)
        port = args.port or launcher.port
        host = args.host or launcher.host
        ctx_size = launcher.ctx_size or args.ctx
    else:
        port = args.port
        host = args.host
        ctx_size = args.ctx

    base_url = f"http://{host}:{port}"
    selected_ids = {t.strip() for t in args.tasks.split(",") if t.strip()} if args.tasks else None
    difficulties = {d.strip().lower() for d in args.difficulty.split(",") if d.strip()} if args.difficulty else None
    categories = {c.strip().lower() for c in args.category.split(",") if c.strip()} if args.category else None
    tasks = load_all_tasks(selected_ids=selected_ids, difficulties=difficulties, categories=categories)
    if not tasks:
        print("Error: no matching tasks found.", file=sys.stderr)
        return 1

    pi_exe = find_pi_executable()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    server_proc: subprocess.Popen | None = None
    isolated_pi_dir: Path | None = None

    try:
        if not args.no_start_server and launcher is not None:
            log_path = RESULTS_DIR / f"server-{launcher.alias}.log"
            print(f"[1/4] Starting llama-server via {launcher.script_path.name} on {base_url} (log: {log_path.name})...")
            server_proc = start_llama_server(launcher.script_path, port, log_path)
            if not wait_for_server(base_url, timeout_sec=args.startup_timeout):
                print(f"Error: llama-server did not become healthy within {args.startup_timeout}s. Check {log_path}", file=sys.stderr)
                return 1
            print("      llama-server is healthy.")
        else:
            print(f"[1/4] Checking already-running llama-server at {base_url}...")
            if not wait_for_server(base_url, timeout_sec=5):
                print(f"Error: llama-server is not reachable at {base_url}/health", file=sys.stderr)
                return 1

        detected_model = query_loaded_model_id(base_url)
        model_id = args.model or (launcher.alias if launcher else None) or detected_model or "local-model"
        print(f"      Target model: {model_id} (server /v1/models reported: {detected_model})")

        # Build locked-down subprocess environment for Pi
        pi_env = os.environ.copy()
        pi_env["PI_OFFLINE"] = "1"
        if args.use_user_pi_config:
            provider = args.pi_provider
            print(f"[2/4] Using user Pi config with explicit provider: --provider {provider} --model {model_id}")
        else:
            provider = "local-llama"
            isolated_pi_dir = Path(tempfile.mkdtemp(prefix="pi-bench-cfg-"))
            create_isolated_pi_config(isolated_pi_dir, base_url, model_id, ctx_size)
            pi_env["PI_CODING_AGENT_DIR"] = str(isolated_pi_dir)
            for k in CLOUD_API_KEY_ENV_VARS:
                pi_env.pop(k, None)
            print(f"[2/4] Using isolated local-only Pi config ({isolated_pi_dir}) -> {base_url}/v1")

        print(f"[3/4] Running {len(tasks)} task(s) via `{Path(pi_exe).name}`...")
        run_started = datetime.now(timezone.utc)
        task_results: list[dict] = []

        for idx, task in enumerate(tasks, 1):
            t_lim = args.timeout if args.timeout is not None else task.get("timeout_sec", 180)
            print(f"  [{idx}/{len(tasks)}] {task['id']} ({task['category']}/{task.get('difficulty','easy')}, {t_lim}s) ...", end=" ", flush=True)
            res = run_single_task(
                task=task,
                pi_exe=pi_exe,
                provider=provider,
                model_id=model_id,
                base_url=base_url,
                pi_env=pi_env,
                thinking=args.thinking,
                timeout_override=args.timeout,
            )
            task_results.append(res)
            tps_str = f"{res['gen_tps']:.1f} t/s" if res["gen_tps"] is not None else "n/a t/s"
            snoop_str = " [SNOOPING!]" if res["snooping_detected"] else ""
            tamper_str = " [TESTS MODIFIED->RESTORED]" if res["tests_tampered"] else ""
            print(
                f"{'PASS' if res['passed'] else 'FAIL'} ({res['status']}) | "
                f"{res['wall_sec']:.1f}s | {tps_str} | tools={res['tool_call_count']}{snoop_str}{tamper_str}"
            )

        # Aggregate run summary
        total = len(task_results)
        passed_n = sum(1 for r in task_results if r["passed"])
        math_res = [r for r in task_results if r["category"] == "math"]
        code_res = [r for r in task_results if r["category"] == "coding"]
        game_res = [r for r in task_results if r["category"] == "game"]
        gen_speeds = [r["gen_tps"] for r in task_results if r["gen_tps"] is not None]
        pp_speeds = [r["prompt_tps"] for r in task_results if r["prompt_tps"] is not None]

        run_id = f"{run_started.strftime('%Y%m%d-%H%M%S')}_{re.sub(r'[^a-zA-Z0-9._-]+', '-', model_id)}"
        run_payload = {
            "run_id": run_id,
            "timestamp": run_started.isoformat(),
            "model_id": model_id,
            "server_reported_model": detected_model,
            "provider": provider,
            "thinking": args.thinking,
            "launcher": {
                "script": launcher.script_path.name if launcher else None,
                "backend": launcher.backend if launcher else "unknown",
                "model_file": launcher.model_file if launcher else None,
                "lora_file": launcher.lora_file if launcher else None,
                "ctx_size": ctx_size,
                "raw_flags": launcher.raw_flags if launcher else "",
            },
            "summary": {
                "total": total,
                "passed": passed_n,
                "pass_rate": round(100.0 * passed_n / total, 1) if total else 0.0,
                "math_passed": sum(1 for r in math_res if r["passed"]),
                "math_total": len(math_res),
                "coding_passed": sum(1 for r in code_res if r["passed"]),
                "coding_total": len(code_res),
                "game_passed": sum(1 for r in game_res if r["passed"]),
                "game_total": len(game_res),
                "avg_gen_tps": round(sum(gen_speeds) / len(gen_speeds), 2) if gen_speeds else None,
                "avg_prompt_tps": round(sum(pp_speeds) / len(pp_speeds), 2) if pp_speeds else None,
                "avg_wall_sec": round(sum(r["wall_sec"] for r in task_results) / total, 2) if total else 0.0,
                "avg_output_tokens": round(sum(r["output_tokens"] for r in task_results) / total, 1) if total else 0.0,
                "avg_tool_calls": round(sum(r["tool_call_count"] for r in task_results) / total, 2) if total else 0.0,
                "total_tool_errors": sum(r["tool_error_count"] for r in task_results),
                "snooping_incidents": sum(1 for r in task_results if r["snooping_detected"]),
                "tests_tampered_count": sum(1 for r in task_results if r["tests_tampered"]),
            },
            "tasks": task_results,
        }

        out_file = RESULTS_DIR / f"{run_id}.json"
        out_file.write_text(json.dumps(run_payload, indent=2), encoding="utf-8")
        print(f"\n[4/4] Saved results: {out_file}")
        print(
            f"      Score: {passed_n}/{total} ({run_payload['summary']['pass_rate']}%) | "
            f"Avg gen: {run_payload['summary']['avg_gen_tps']} t/s | "
            f"Snooping alerts: {run_payload['summary']['snooping_incidents']}"
        )

        if not args.no_report:
            import build_report
            report_path = build_report.build_report()
            print(f"      Updated dashboard: {report_path}")

        return 0
    finally:
        stop_llama_server(server_proc)
        if isolated_pi_dir is not None:
            shutil.rmtree(isolated_pi_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
