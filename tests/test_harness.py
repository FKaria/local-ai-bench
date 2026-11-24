"""Self-tests for the benchmark harness, math symbolic grader, test-tampering protection, and report builder."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent.parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

import bench_config
import build_report
import game_envs
import game_server
import math_grader
import optimize
import run_bench


class TestBenchHarness(unittest.TestCase):
    def test_scripts_compile(self):
        for name in (
            "bench_config.py",
            "game_envs.py",
            "game_server.py",
            "math_grader.py",
            "optimize.py",
            "run_bench.py",
            "build_report.py",
        ):
            src_path = BENCH_DIR / name
            compile(src_path.read_text(encoding="utf-8"), str(src_path), "exec")

    def test_start_script_parsing(self):
        scripts = bench_config.discover_start_scripts()
        self.assertGreaterEqual(len(scripts), 1)
        for s in scripts:
            info = bench_config.parse_start_script(s)
            self.assertTrue(info.alias)
            self.assertIn(info.backend, ("hip", "vulkan"))
            self.assertTrue(info.model_file.endswith(".gguf"))

    def test_tasks_loaded_and_coding_starters_fail_both_visible_and_hidden_tests(self):
        tasks = run_bench.load_all_tasks()
        categories = {t["category"] for t in tasks}
        self.assertEqual(categories, {"math", "coding", "game"})
        self.assertEqual(len(tasks), 23)  # 8 no-tools math + 3 tool math + 7 coding + 5 games
        self.assertEqual(len(run_bench.load_all_tasks(difficulties={"medium"})), 7)
        self.assertEqual(len(run_bench.load_all_tasks(difficulties={"hard"})), 6)
        self.assertEqual(len(run_bench.load_all_tasks(categories={"game"})), 5)

        for t in tasks:
            if t["category"] == "coding":
                ws = Path(tempfile.mkdtemp(prefix="verify-starter-"))
                try:
                    for src in (Path(t["_task_dir"]) / "starter").iterdir():
                        shutil.copy2(src, ws / src.name)
                    # Visible starter tests must fail out of the box on buggy starter code
                    vis = subprocess.run(
                        [sys.executable, "-m", "unittest", "test_basic.py"],
                        cwd=str(ws),
                        capture_output=True,
                        text=True,
                    )
                    self.assertNotEqual(vis.returncode, 0, f"Starter test_basic.py for {t['id']} should fail before fix")

                    hashes = run_bench.snapshot_starter_tests(ws)
                    passed, status, _, modified = run_bench.grade_coding_task(t, ws, starter_hashes=hashes)
                    self.assertFalse(passed)
                    self.assertEqual(status, "test_failed")
                    self.assertEqual(modified, [])
                finally:
                    shutil.rmtree(ws, ignore_errors=True)

    def test_tampered_starter_tests_are_detected_and_restored(self):
        tasks = {t["id"]: t for t in run_bench.load_all_tasks()}
        task = tasks["code_01_bugfix_rle"]
        ws = Path(tempfile.mkdtemp(prefix="verify-tamper-"))
        try:
            for src in (Path(task["_task_dir"]) / "starter").iterdir():
                shutil.copy2(src, ws / src.name)
            hashes = run_bench.snapshot_starter_tests(ws)

            # Simulate a cheating model replacing test_basic.py with a trivial pass and adding test_fake.py
            (ws / "test_basic.py").write_text("import unittest\nclass T(unittest.TestCase):\n    def test_ok(self): pass\n", encoding="utf-8")
            (ws / "test_fake.py").write_text("import unittest\nclass F(unittest.TestCase):\n    def test_fail(self): self.fail()\n", encoding="utf-8")

            passed, status, _, modified = run_bench.grade_coding_task(task, ws, starter_hashes=hashes)
            self.assertFalse(passed)
            self.assertEqual(status, "test_failed")
            self.assertTrue(any("test_basic.py (modified)" in m for m in modified))
            self.assertTrue(any("test_fake.py (added by model)" in m for m in modified))
            self.assertFalse((ws / "test_fake.py").exists())
        finally:
            shutil.rmtree(ws, ignore_errors=True)

    def test_math_graders_all_modes(self):
        ws = Path(tempfile.mkdtemp(prefix="verify-math-"))
        try:
            # 1. Integer (no tools -> ignores answer.txt, uses boxed)
            t_int = {"grader": "integer", "allow_tools": False, "expected": "455832"}
            ok, _, _ = math_grader.grade_math(t_int, ws, r"Thus 487*936 = \boxed{455,832}.")
            self.assertTrue(ok)

            # 2. Root set (unordered, LaTeX fractions, braces)
            t_set = {"grader": "set", "allow_tools": False, "expected_set": [-1.5, 5.0]}
            for resp in (
                r"\boxed{5, -\frac{3}{2}}",
                r"\boxed{5,\;-\tfrac32}",
                r"\boxed{\{-1.5, 5\}}",
                r"\boxed{x = -3/2 \text{ or } x = 5}",
            ):
                ok, _, msg = math_grader.grade_math(t_set, ws, resp)
                self.assertTrue(ok, f"Failed on {resp}: {msg}")
            t_qset = {"grader": "set", "allow_tools": False, "expected_set": [-3.0, -1.0, 1.0]}
            ok_q, _, msg_q = math_grader.grade_math(t_qset, ws, r"\boxed{-3,\,-1,\;1}")
            self.assertTrue(ok_q, f"Failed on thin-space set: {msg_q}")
            ok_bad, _, _ = math_grader.grade_math(t_set, ws, r"\boxed{1.5, 5}")
            self.assertFalse(ok_bad)

            # 3. Antiderivative (up to arbitrary + C, various notations including new trig/exp integral)
            t_intg = {
                "grader": "antiderivative",
                "allow_tools": False,
                "var": "x",
                "integrand": "exp(x) * (1 + sin(x))/(1 + cos(x))",
                "expected_expr": "tan(x/2) * exp(x)",
            }
            for resp in (
                r"\boxed{e^x \tan(x/2) + C}",
                r"\boxed{\tan\left(\frac{x}{2}\right) e^x + C}",
                r"\boxed{\frac{\sin x}{1 + \cos x} e^x + C}",
                r"\boxed{e^{x} \tan\frac{x}{2}}",
            ):
                ok, _, msg = math_grader.grade_math(t_intg, ws, resp)
                self.assertTrue(ok, f"Failed on {resp}: {msg}")
            ok_bad_intg, _, _ = math_grader.grade_math(t_intg, ws, r"\boxed{e^x \cot(x/2) + C}")
            self.assertFalse(ok_bad_intg)

            # 4. Implicit ODE general solution (any equivalent form / reparameterized constant C)
            t_ode_imp = {
                "grader": "ode_implicit",
                "allow_tools": False,
                "ode_rhs": "y / (2*y*log(y) + y - x)",
                "y_positive": True,
            }
            for resp in (
                r"\boxed{x = y \ln y + \frac{C}{y}}",
                r"\boxed{x = y \ln|y| + \frac{C}{y}}",
                r"\boxed{x y = y^2 \ln(y) + C}",
                r"\boxed{y(x - y\ln y) = C}",
                r"\boxed{x = y \log(y) - \frac{2C}{y}}",
            ):
                ok, _, msg = math_grader.grade_math(t_ode_imp, ws, resp)
                self.assertTrue(ok, f"Failed on implicit ODE {resp}: {msg}")
            for bad_resp in (
                r"\boxed{x = y \ln y + C}",
                r"\boxed{x = y^2 \ln y + C/y}",
                r"\boxed{x = x + 0*C}",
            ):
                ok_bad, _, _ = math_grader.grade_math(t_ode_imp, ws, bad_resp)
                self.assertFalse(ok_bad, f"Should have rejected {bad_resp}")

            # 5. Symbolic expression (explicit formula)
            t_ode = {
                "grader": "expr",
                "allow_tools": False,
                "var": "x",
                "expected_expr": "3 * exp(x**2)",
            }
            for resp in (
                r"\boxed{y(x) = 3 e^{x^2}}",
                r"\boxed{3\exp(x^2)}",
                r"\boxed{3 \cdot e^{x^{2}}}",
            ):
                ok, _, msg = math_grader.grade_math(t_ode, ws, resp)
                self.assertTrue(ok, f"Failed on {resp}: {msg}")
            ok_bad_ode, _, _ = math_grader.grade_math(t_ode, ws, r"\boxed{3 e^{2x}}")
            self.assertFalse(ok_bad_ode)

            # 5. Safe AST blocks arbitrary code execution even when SymPy is installed
            canary = ws / "pwned.txt"
            ok_inject, _, _ = math_grader.grade_math(
                t_ode, ws, rf"\boxed{{__import__('pathlib').Path({str(canary)!r}).write_text('x')}}"
            )
            self.assertFalse(ok_inject)
            self.assertFalse(canary.exists(), "Code injection executed during math grading!")
        finally:
            shutil.rmtree(ws, ignore_errors=True)

    def test_anti_snooping_audit(self):
        ws = Path(tempfile.mkdtemp(prefix="verify-snoop-"))
        try:
            self.assertEqual(run_bench.audit_tool_call_for_snooping("read", {"path": "rle.py"}, ws), [])
            # '..' inside edited source text must NOT trigger a false-positive snooping alert
            self.assertEqual(
                run_bench.audit_tool_call_for_snooping(
                    "edit",
                    {"path": "solution.py", "edits": [{"oldText": "pass", "newText": '# range 1..10 """.."""'}]},
                    ws,
                ),
                [],
            )
            self.assertTrue(
                run_bench.audit_tool_call_for_snooping("read", {"path": str(BENCH_DIR / "tasks" / "math" / "tasks.json")}, ws)
            )
            self.assertTrue(
                run_bench.audit_tool_call_for_snooping("bash", {"command": "cat ../hidden_tests/test_hidden.py"}, ws)
            )
        finally:
            shutil.rmtree(ws, ignore_errors=True)

    def test_report_builder(self):
        tmp_html = Path(tempfile.mkdtemp(prefix="verify-report-")) / "report.html"
        try:
            build_report.build_report(tmp_html)
            html = tmp_html.read_text(encoding="utf-8")
            self.assertIn("Local LLM Benchmark Dashboard", html)
            self.assertIn("15 tok/s target", html)
        finally:
            shutil.rmtree(tmp_html.parent, ignore_errors=True)

    def test_tree_timeout_kills_children(self):
        ws = Path(tempfile.mkdtemp(prefix="verify-timeout-"))
        try:
            cmd = ["cmd.exe", "/c", sys.executable, "-c", "import time; print('started', flush=True); time.sleep(30)"]
            t0 = time.perf_counter()
            rc, out, _err, timed_out = run_bench.run_with_tree_timeout(cmd, ws, dict(run_bench.os.environ), timeout_sec=1)
            elapsed = time.perf_counter() - t0
            self.assertTrue(timed_out)
            self.assertEqual(rc, -1)
            self.assertLess(elapsed, 6.0, f"Tree kill took too long: {elapsed:.2f}s")
            self.assertIn("started", out)
        finally:
            shutil.rmtree(ws, ignore_errors=True)

    def test_game_envs_and_http_play_client(self):
        # 1. Mastermind end-to-end via LocalGameServer + play.py
        mm_env = game_envs.MastermindEnv(code_len=5, num_colors=8, max_turns=12, seed=42)
        srv = game_server.LocalGameServer(mm_env)
        port = srv.start()
        ws = Path(tempfile.mkdtemp(prefix="verify-game-"))
        try:
            play_src = game_server.render_play_client(port)
            # Secret must NEVER appear in play.py
            secret_str = " ".join(str(x) for x in mm_env._secret)
            self.assertNotIn(secret_str, play_src)
            (ws / "play.py").write_text(play_src, encoding="utf-8")

            p_status = subprocess.run([sys.executable, "play.py", "status"], cwd=str(ws), capture_output=True, text=True)
            self.assertEqual(p_status.returncode, 0)
            self.assertIn("MASTERMIND", p_status.stdout)

            p_win = subprocess.run(
                [sys.executable, "play.py", "guess", *[str(x) for x in mm_env._secret]],
                cwd=str(ws),
                capture_output=True,
                text=True,
            )
            self.assertEqual(p_win.returncode, 0)
            self.assertIn("WON", p_win.stdout)
            self.assertTrue(mm_env.status()["won"])
        finally:
            srv.stop()
            shutil.rmtree(ws, ignore_errors=True)

        # 2. Minesweeper safe opening, reveal all safe cells -> win, hit mine -> lose
        ms_env = game_envs.MinesweeperEnv(rows=7, cols=7, num_mines=8, max_turns=60, seed=1337, start_cell=(3, 3))
        self.assertGreater(len(ms_env.revealed), 5)
        for r in range(ms_env.rows):
            for c in range(ms_env.cols):
                if (r, c) not in ms_env._mines and (r, c) not in ms_env.revealed and not ms_env.done:
                    ms_env.step(f"reveal {r} {c}")
        self.assertTrue(ms_env.status()["won"])

        ms_lose = game_envs.MinesweeperEnv(rows=7, cols=7, num_mines=8, max_turns=60, seed=1337, start_cell=(3, 3))
        mr, mc = next(iter(ms_lose._mines))
        obs, done, st = ms_lose.step(f"reveal {mr} {mc}")
        self.assertTrue(done)
        self.assertFalse(st["won"])
        self.assertIn("BOOM", obs)

        # 3. Sokoban verified 13-move solution via LocalGameServer + play.py
        sk_env = game_envs.SokobanEnv(max_turns=80)
        srv2 = game_server.LocalGameServer(sk_env)
        port2 = srv2.start()
        ws2 = Path(tempfile.mkdtemp(prefix="verify-soko-"))
        try:
            (ws2 / "play.py").write_text(game_server.render_play_client(port2), encoding="utf-8")
            dir_map = {"U": "up", "D": "down", "L": "left", "R": "right"}
            for ch in "DULLURDRRURUL":
                p = subprocess.run([sys.executable, "play.py", dir_map[ch]], cwd=str(ws2), capture_output=True, text=True)
                self.assertEqual(p.returncode, 0)
            self.assertTrue(sk_env.status()["won"])
            self.assertEqual(sk_env.status()["turns"], 13)
        finally:
            srv2.stop()
            shutil.rmtree(ws2, ignore_errors=True)

    def test_optimizer_fit_parser_and_winner_selection(self):
        sample = "ROCm0 10616 2329 589 \nHost 520 0 104 \n"
        est = optimize.parse_fit_params_output(sample, "hip", "q8_0", ngl=99, ncmoe=0, lora_mib=304)
        self.assertIsNotNone(est)
        assert est is not None
        self.assertEqual(est.gpu_total_mib, 10616 + 2329 + 589 + 304)
        self.assertEqual(est.host_total_mib, 624)

        # Prefer fitting candidate over spilling candidate, and q8_0 over q4_0 within 5% speed
        c_spill = {"backend": "vulkan", "kv_type": "q8_0", "ngl": 99, "ncmoe": 0, "fits_vram": False, "metrics": {"tg_d16384": 50.0, "tg_d0": 55.0}}
        c_q4 = {"backend": "hip", "kv_type": "q4_0", "ngl": 99, "ncmoe": 0, "fits_vram": True, "metrics": {"tg_d16384": 20.0, "tg_d0": 22.0}}
        c_q8 = {"backend": "hip", "kv_type": "q8_0", "ngl": 99, "ncmoe": 0, "fits_vram": True, "metrics": {"tg_d16384": 19.4, "tg_d0": 21.5}}
        winner = optimize.pick_winner([c_spill, c_q4, c_q8])
        self.assertEqual(winner["kv_type"], "q8_0")
        self.assertTrue(winner["fits_vram"])

    def test_rpc_tool_free_game_loop_and_action_extraction(self):
        # Verify action extraction from realistic CoT + bracketed moves
        self.assertEqual(
            game_envs.extract_game_action("Based on clue [1] I guess [guess 1 2 3 4].", "mastermind"),
            "guess 1 2 3 4",
        )
        self.assertEqual(
            game_envs.extract_game_action("Cell (2,3) must be safe.\n[reveal 2 3]", "minesweeper"),
            "reveal 2 3",
        )
        self.assertEqual(
            game_envs.extract_game_action("Push box down first.\n[down]", "sokoban"),
            "down",
        )

        # End-to-end mock `pi --mode rpc` subprocess solving Sokoban in 13 turns
        ws = Path(tempfile.mkdtemp(prefix="verify-rpc-game-"))
        try:
            mock_rpc = ws / "mock_pi_rpc.py"
            mock_rpc.write_text(
                "import json, sys\n"
                "moves = ['down','up','left','left','up','right','down','right','right','up','right','up','left']\n"
                "i = 0\n"
                "for line in sys.stdin:\n"
                "    req = json.loads(line)\n"
                "    print(json.dumps({'id': req.get('id'), 'type': 'response', 'command': 'prompt', 'success': True}), flush=True)\n"
                "    mv = moves[i] if i < len(moves) else 'up'\n"
                "    i += 1\n"
                "    print(json.dumps({'type': 'message_end', 'message': {'role': 'assistant', 'stopReason': 'stop', 'content': [{'type': 'text', 'text': f'Next move: [{mv}]'}], 'usage': {'input': 50, 'output': 10}}}), flush=True)\n"
                "    print(json.dumps({'type': 'agent_end', 'messages': []}), flush=True)\n",
                encoding="utf-8",
            )
            sk_env = game_envs.SokobanEnv(max_turns=80)
            rc, parsed, err, timed_out = run_bench.run_rpc_game_session(
                cmd=[sys.executable, str(mock_rpc)],
                cwd=ws,
                env_vars=dict(run_bench.os.environ),
                game_env=sk_env,
                game_type="sokoban",
                initial_prompt="Solve Sokoban",
                timeout_sec=15,
            )
            self.assertFalse(timed_out, f"RPC game timed out: {err}")
            self.assertTrue(sk_env.status()["won"])
            self.assertEqual(sk_env.status()["turns"], 13)
            self.assertEqual(parsed["turns"], 13)
            self.assertEqual(parsed["tool_call_count"], 13)
        finally:
            shutil.rmtree(ws, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
