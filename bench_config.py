"""Shared paths and start-*.cmd parsing utilities for E:\\local-llm\\bench."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
ROOT_DIR = BENCH_DIR.parent
MODELS_DIR = ROOT_DIR / "llama.models"
HIP_DIR = ROOT_DIR / "llama.cpp-hip"
VULKAN_DIR = ROOT_DIR / "llama.cpp-vulkan"
TASKS_DIR = BENCH_DIR / "tasks"
RESULTS_DIR = BENCH_DIR / "results"
REPORT_PATH = BENCH_DIR / "report.html"

ROCM_BIN_DIR = Path(r"C:\Program Files\AMD\ROCm\7.2\bin")

# Cloud API key env vars stripped when running isolated local benchmarks
CLOUD_API_KEY_ENV_VARS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_OAUTH_TOKEN",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "OPENROUTER_API_KEY",
    "DEEPSEEK_API_KEY",
    "GROQ_API_KEY",
    "XAI_API_KEY",
    "MISTRAL_API_KEY",
    "TOGETHER_API_KEY",
    "FIREWORKS_API_KEY",
    "CEREBRAS_API_KEY",
    "NVIDIA_API_KEY",
    "COHERE_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_BEARER_TOKEN_BEDROCK",
)


@dataclass
class LauncherInfo:
    script_path: Path
    alias: str
    backend: str
    model_file: str
    lora_file: str | None = None
    ctx_size: int | None = None
    port: int = 8080
    host: str = "127.0.0.1"
    raw_flags: str = ""
    extra: dict = field(default_factory=dict)


def parse_start_script(script_path: Path) -> LauncherInfo:
    """Parse a start-*.cmd launcher in E:\\local-llm to extract alias, model, backend, and flags."""
    script_path = script_path.resolve()
    text = script_path.read_text(encoding="utf-8", errors="replace")

    # Join Windows cmd line continuations (^)
    joined_lines: list[str] = []
    buf = ""
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.lower().startswith("rem ") or stripped.startswith("@"):
            continue
        if stripped.endswith("^"):
            buf += stripped[:-1] + " "
        else:
            joined_lines.append((buf + stripped).strip())
            buf = ""
    if buf:
        joined_lines.append(buf.strip())

    cmd_line = next((l for l in joined_lines if "llama-server" in l.lower()), "")

    backend = "hip" if "llama.cpp-hip" in cmd_line.lower() or "rocm" in cmd_line.lower() else (
        "vulkan" if "llama.cpp-vulkan" in cmd_line.lower() or "vulkan" in cmd_line.lower() else "unknown"
    )

    def _find_flag(pattern: str) -> str | None:
        m = re.search(pattern, cmd_line)
        return m.group(1).strip('"') if m else None

    alias = _find_flag(r"--alias\s+([^\s]+)") or script_path.stem.removeprefix("start-")
    model_raw = _find_flag(r"-m\s+(\"[^\"]+\"|[^\s]+)") or ""
    model_file = Path(model_raw.replace("%~dp0", str(script_path.parent) + "\\")).name if model_raw else ""
    lora_raw = _find_flag(r"--lora\s+(\"[^\"]+\"|[^\s]+)")
    lora_file = Path(lora_raw.replace("%~dp0", str(script_path.parent) + "\\")).name if lora_raw else None
    ctx_raw = _find_flag(r"(?:-c|--ctx-size)\s+(\d+)")
    port_raw = _find_flag(r"--port\s+(\d+)")
    host = _find_flag(r"--host\s+([^\s]+)") or "127.0.0.1"

    return LauncherInfo(
        script_path=script_path,
        alias=alias,
        backend=backend,
        model_file=model_file,
        lora_file=lora_file,
        ctx_size=int(ctx_raw) if ctx_raw else None,
        port=int(port_raw) if port_raw else 8080,
        host=host,
        raw_flags=cmd_line,
    )


def discover_start_scripts() -> list[Path]:
    return sorted(ROOT_DIR.glob("start-*.cmd"))
