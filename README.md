# Local LLM Benchmark Harness (`pi` + `llama.cpp`)

Lightweight, self-contained local benchmark & optimizer for evaluating GGUF models on math and coding tasks using the `pi` coding agent harness against `llama-server`.

## Directory Layout

- `optimize.py` — Sweeps HIP vs Vulkan (no-coopmat) and `-ncmoe`/KV settings via `llama-bench` at depth 0 and 16K, logs hardware profiles, and generates `E:\local-llm\start-<model>.cmd`.
- `run_bench.py` — Launches a `start-*.cmd` (with `--metrics`), runs math & coding tasks via `pi --mode json --no-session` inside isolated temp workspaces, runs hidden grading tests *after* `pi` exits, audits tool calls for snooping, and rebuilds `report.html`.
- `build_report.py` — Builds a self-contained, offline interactive `report.html` dashboard from `results/`.
- `tasks/math/tasks.json` — Deterministic math problems (exact integer verification).
- `tasks/coding/<task_id>/` — Coding problems with `starter/` (copied to temp dir) and `hidden_tests/` (never exposed to the model).

## Quickstart

### 1. Run the PoC benchmark against a model script
```powershell
cd E:\local-llm\bench
python run_bench.py --script ..\start-gpt-oss-20b.cmd
```

This automatically:
1. Starts `llama-server` using `start-gpt-oss-20b.cmd` with `--metrics` enabled on `127.0.0.1:8080`.
2. Creates an isolated temporary `PI_CODING_AGENT_DIR` containing **only** a `local-llama` provider pointing to `http://127.0.0.1:8080/v1` (and strips cloud API keys from the environment) so `pi` can **never** hit an external provider.
3. Runs each task in a clean temp directory under `%TEMP%`, grades it deterministically, audits tool calls for snooping, saves results to `results/`, stops the server, and generates `report.html`.

### 2. Run against an already-running `llama-server`
```powershell
python run_bench.py --no-start-server --model gpt-oss-20b
```

### 3. Use your own configured local provider in `~/.pi/agent`
```powershell
python run_bench.py --script ..\start-gpt-oss-20b.cmd --use-user-pi-config --pi-provider <your-local-provider-name>
```
*(When `--use-user-pi-config` is set, `--pi-provider` is mandatory so `pi` cannot fall back to an external default provider.)*

### 4. Run a subset of tasks (quick smoke test)
```powershell
python run_bench.py --script ..\start-gpt-oss-20b.cmd --tasks math_01_sum_squares,code_01_bugfix_rle
```

### 5. Optimize a newly downloaded `.gguf`
```powershell
python optimize.py ..\llama.models\YourNewModel.gguf --ctx 65536 --write-script
```

### 6. View the HTML dashboard
Open `E:\local-llm\bench\report.html` in your browser (or regenerate manually with `python build_report.py`).
