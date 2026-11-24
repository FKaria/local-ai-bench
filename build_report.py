"""Stage 3: Build a self-contained, offline interactive HTML benchmark dashboard.

Reads all run files in E:\\local-llm\\bench\\results\\*.json and optional hardware
profiles in hw_profiles.jsonl, producing E:\\local-llm\\bench\\report.html.

Usage:
    python build_report.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from bench_config import REPORT_PATH, RESULTS_DIR


def load_runs() -> list[dict]:
    runs: list[dict] = []
    if not RESULTS_DIR.exists():
        return runs
    for path in sorted(RESULTS_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if "run_id" in data and "tasks" in data:
                runs.append(data)
        except Exception:
            continue
    return runs


def load_hw_profiles() -> list[dict]:
    profiles: list[dict] = []
    p = RESULTS_DIR / "hw_profiles.jsonl"
    if not p.exists():
        return profiles
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            profiles.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return profiles


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Local LLM Benchmark Dashboard (Pi + llama.cpp)</title>
<style>
  :root {
    --bg: #0d1117;
    --panel: #161b22;
    --panel-2: #1c222b;
    --border: #30363d;
    --text: #e6edf3;
    --muted: #8b949e;
    --accent: #58a6ff;
    --green: #3fb950;
    --red: #f85149;
    --amber: #d29922;
    --purple: #bc8cff;
    --cyan: #39c5cf;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Inter, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.5;
  }
  header {
    padding: 20px 28px;
    border-bottom: 1px solid var(--border);
    background: var(--panel);
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 12px;
  }
  header h1 { margin: 0; font-size: 20px; font-weight: 600; }
  header .sub { color: var(--muted); font-size: 13px; }
  main { max-width: 1400px; margin: 0 auto; padding: 24px; display: flex; flex-direction: column; gap: 22px; }
  .kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 14px; }
  .kpi {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px 18px;
  }
  .kpi .label { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; }
  .kpi .val { font-size: 26px; font-weight: 700; margin-top: 6px; }
  .kpi .hint { color: var(--muted); font-size: 12px; margin-top: 4px; }
  .charts-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(460px, 1fr)); gap: 18px; }
  .card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 18px 20px;
  }
  .card h2 { margin: 0 0 4px 0; font-size: 15px; font-weight: 600; }
  .card .desc { color: var(--muted); font-size: 12px; margin-bottom: 14px; }
  svg { width: 100%; height: auto; display: block; overflow: visible; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th, td { padding: 10px 12px; border-bottom: 1px solid var(--border); text-align: left; }
  th { color: var(--muted); font-weight: 600; font-size: 12px; text-transform: uppercase; letter-spacing: 0.03em; }
  tr:hover td { background: rgba(88, 166, 255, 0.04); }
  .pill {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 600;
  }
  .pill-pass { background: rgba(63, 185, 80, 0.16); color: var(--green); border: 1px solid rgba(63, 185, 80, 0.4); }
  .pill-fail { background: rgba(248, 81, 73, 0.16); color: var(--red); border: 1px solid rgba(248, 81, 73, 0.4); }
  .pill-warn { background: rgba(210, 153, 34, 0.16); color: var(--amber); border: 1px solid rgba(210, 153, 34, 0.4); }
  .pill-info { background: rgba(88, 166, 255, 0.14); color: var(--accent); border: 1px solid rgba(88, 166, 255, 0.35); }
  .matrix-cell { cursor: pointer; text-align: center; transition: transform 0.08s ease; }
  .matrix-cell:hover { filter: brightness(1.2); }
  .empty { text-align: center; padding: 48px 20px; color: var(--muted); }
  code { background: var(--panel-2); padding: 2px 6px; border-radius: 4px; font-size: 12px; }
  .tooltip {
    position: fixed;
    pointer-events: none;
    background: #0b0f14;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 12px;
    font-size: 12px;
    color: var(--text);
    box-shadow: 0 8px 24px rgba(0,0,0,0.45);
    display: none;
    z-index: 999;
    max-width: 280px;
  }
  .modal-backdrop {
    position: fixed; inset: 0; background: rgba(0, 0, 0, 0.7);
    display: none; align-items: center; justify-content: center; z-index: 1000; padding: 24px;
  }
  .modal {
    background: var(--panel); border: 1px solid var(--border); border-radius: 12px;
    max-width: 920px; width: 100%; max-height: 88vh; overflow: auto; padding: 22px 24px;
  }
  .modal-hdr { display: flex; justify-content: space-between; align-items: flex-start; gap: 12px; margin-bottom: 14px; }
  .close-btn {
    background: var(--panel-2); color: var(--text); border: 1px solid var(--border);
    border-radius: 6px; padding: 6px 12px; cursor: pointer; font-size: 13px;
  }
  .trace-step {
    background: var(--panel-2); border: 1px solid var(--border); border-radius: 8px;
    padding: 10px 12px; margin-top: 8px; font-size: 12px;
  }
  .trace-step pre {
    margin: 6px 0 0 0; white-space: pre-wrap; word-break: break-word;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 11.5px; color: #c9d1d9;
  }
</style>
</head>
<body>
<header>
  <div>
    <h1>Local LLM Benchmark Dashboard</h1>
    <div class="sub">Hardware: RX 9060 XT (16 GB) + 32 GB RAM &middot; Harness: <code>pi</code> + <code>llama.cpp</code> &middot; Generated <span id="gen-time"></span></div>
  </div>
  <div class="sub" id="run-count"></div>
</header>

<main id="app"></main>
<div class="tooltip" id="tooltip"></div>
<div class="modal-backdrop" id="modal-backdrop">
  <div class="modal">
    <div class="modal-hdr">
      <div>
        <h2 id="modal-title" style="margin:0"></h2>
        <div class="sub" id="modal-sub"></div>
      </div>
      <button class="close-btn" id="modal-close">Close</button>
    </div>
    <div id="modal-body"></div>
  </div>
</div>

<script>
const DATA = __BENCH_DATA_JSON__;

const PALETTE = ["#58a6ff", "#3fb950", "#bc8cff", "#39c5cf", "#f0883e", "#f778ba", "#d29922", "#a5d6ff"];
const fmt = (v, suffix = "") => (v === null || v === undefined ? "—" : `${v}${suffix}`);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, m => ({ "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;" }[m]));

document.getElementById("gen-time").textContent = DATA.generated_at;
document.getElementById("run-count").textContent = `${DATA.runs.length} benchmark run(s) loaded`;

const tooltip = document.getElementById("tooltip");
function showTip(evt, html) {
  tooltip.innerHTML = html;
  tooltip.style.display = "block";
  tooltip.style.left = Math.min(evt.clientX + 14, window.innerWidth - 300) + "px";
  tooltip.style.top = Math.min(evt.clientY + 14, window.innerHeight - 160) + "px";
}
function hideTip() { tooltip.style.display = "none"; }

const modalBackdrop = document.getElementById("modal-backdrop");
document.getElementById("modal-close").onclick = () => { modalBackdrop.style.display = "none"; };
modalBackdrop.onclick = (e) => { if (e.target === modalBackdrop) modalBackdrop.style.display = "none"; };

function openTraceModal(run, task) {
  document.getElementById("modal-title").textContent = `${run.model_id} — ${task.task_id}`;
  document.getElementById("modal-sub").textContent =
    `${task.title} | ${task.category} | ${task.status} | ${task.wall_sec}s | gen=${fmt(task.gen_tps, " t/s")} | tools=${task.tool_call_count}`;

  const snoopHtml = task.snooping_detected
    ? `<div class="trace-step" style="border-color: var(--red);"><strong style="color:var(--red)">Snooping Flags Detected</strong><pre>${esc(task.snooping_flags.join("\n"))}</pre></div>`
    : `<div class="trace-step" style="border-color: rgba(63,185,80,0.4);"><strong style="color:var(--green)">Anti-Snooping Audit: Clean</strong> (all tool calls stayed within temp workspace)</div>`;

  const tamperHtml = task.tests_tampered
    ? `<div class="trace-step" style="border-color: var(--amber);"><strong style="color:var(--amber)">Starter Tests Modified by Model (Restored Canonical Tests Before Grading)</strong><pre>${esc((task.modified_test_files || []).join("\n"))}</pre></div>`
    : "";

  const gameHtml = task.game_metrics
    ? `<div class="trace-step" style="border-color: var(--purple);"><strong style="color:var(--purple)">Game Metrics (${esc(task.game_metrics.game)})</strong><pre>progress=${Math.round((task.game_metrics.progress || 0) * 100)}% | game_turns=${task.game_metrics.turns}/${task.game_metrics.max_turns} | invalid_moves=${task.game_metrics.invalid_moves}\n${esc(task.game_metrics.summary)}</pre></div>`
    : "";

  const traceSteps = (task.trace || []).map((step, i) => {
    if (step.kind === "tool") {
      const badge = step.is_error ? `<span class="pill pill-fail">error</span>` : `<span class="pill pill-info">ok</span>`;
      return `<div class="trace-step">
        <div><strong>#${i+1} Tool: <code>${esc(step.tool)}</code></strong> ${badge} <span style="color:var(--muted)">${fmt(step.duration_ms, " ms")}</span></div>
        <pre>args: ${esc(JSON.stringify(step.args, null, 2))}</pre>
        ${step.output_preview ? `<pre>output: ${esc(step.output_preview)}</pre>` : ""}
      </div>`;
    }
    return `<div class="trace-step">
      <div><strong>#${i+1} Assistant (turn ${step.turn})</strong> <span class="pill pill-info">${esc(step.stop_reason)}</span></div>
      ${step.thinking_preview ? `<pre style="color:var(--muted)">[thinking] ${esc(step.thinking_preview)}</pre>` : ""}
      ${step.text ? `<pre>${esc(step.text)}</pre>` : ""}
    </div>`;
  }).join("");

  document.getElementById("modal-body").innerHTML = `
    <div class="trace-step"><strong>Prompt</strong><pre>${esc(task.prompt)}</pre></div>
    <div class="trace-step"><strong>Grading Result (${esc(task.status)})</strong><pre>${esc(task.grade_detail)}</pre></div>
    ${snoopHtml}
    ${tamperHtml}
    ${gameHtml}
    <h3 style="margin:16px 0 6px;font-size:14px">Pi Session Trace (${(task.trace || []).length} events)</h3>
    ${traceSteps || '<div style="color:var(--muted);font-size:13px">No JSONL trace events captured.</div>'}
  `;
  modalBackdrop.style.display = "flex";
}

function renderParetoChart(runs) {
  const W = 520, H = 280, pad = { t: 20, r: 24, b: 44, l: 52 };
  const iw = W - pad.l - pad.r, ih = H - pad.t - pad.b;
  const speeds = runs.map(r => r.summary.avg_gen_tps || 0);
  const maxX = Math.max(30, ...speeds) * 1.15;
  const sx = x => pad.l + (x / maxX) * iw;
  const sy = y => pad.t + ih - (y / 100) * ih;

  let grid = "";
  for (let y = 0; y <= 100; y += 25) {
    grid += `<line x1="${pad.l}" y1="${sy(y)}" x2="${W-pad.r}" y2="${sy(y)}" stroke="#21262d" />`;
    grid += `<text x="${pad.l-8}" y="${sy(y)+4}" fill="#8b949e" font-size="11" text-anchor="end">${y}%</text>`;
  }
  const ticks = 5;
  for (let i = 0; i <= ticks; i++) {
    const xv = Math.round((maxX / ticks) * i);
    grid += `<line x1="${sx(xv)}" y1="${pad.t}" x2="${sx(xv)}" y2="${H-pad.b}" stroke="#21262d" />`;
    grid += `<text x="${sx(xv)}" y="${H-pad.b+18}" fill="#8b949e" font-size="11" text-anchor="middle">${xv}</text>`;
  }
  // 15 tok/s target line
  const x15 = sx(15);
  const targetLine = `
    <line x1="${x15}" y1="${pad.t}" x2="${x15}" y2="${H-pad.b}" stroke="#d29922" stroke-dasharray="5 4" stroke-width="1.5" />
    <text x="${x15+5}" y="${pad.t+12}" fill="#d29922" font-size="11">15 tok/s target</text>
  `;

  const dots = runs.map((r, i) => {
    const spd = r.summary.avg_gen_tps || 0;
    const acc = r.summary.pass_rate || 0;
    const col = PALETTE[i % PALETTE.length];
    return `<g class="pt" data-idx="${i}" style="cursor:pointer">
      <circle cx="${sx(spd)}" cy="${sy(acc)}" r="8" fill="${col}" fill-opacity="0.85" stroke="#fff" stroke-width="1.5" />
      <text x="${sx(spd)+11}" y="${sy(acc)+4}" fill="#e6edf3" font-size="11">${esc(r.model_id)}</text>
    </g>`;
  }).join("");

  return `<svg viewBox="0 0 ${W} ${H}">
    ${grid}${targetLine}${dots}
    <text x="${pad.l + iw/2}" y="${H-6}" fill="#8b949e" font-size="11" text-anchor="middle">Avg Generation Speed (tok/s)</text>
    <text x="14" y="${pad.t + ih/2}" fill="#8b949e" font-size="11" text-anchor="middle" transform="rotate(-90 14 ${pad.t + ih/2})">Pass Rate (%)</text>
  </svg>`;
}

function renderCategoryChart(runs) {
  const W = 520, H = 280, pad = { t: 24, r: 20, b: 64, l: 48 };
  const iw = W - pad.l - pad.r, ih = H - pad.t - pad.b;
  const sy = y => pad.t + ih - (y / 100) * ih;
  const grpW = iw / Math.max(1, runs.length);
  const hasGames = runs.some(r => (r.summary.game_total || 0) > 0);
  const nBars = hasGames ? 3 : 2;
  const barW = Math.min(24, (grpW - 16) / nBars);

  let grid = "";
  for (let y = 0; y <= 100; y += 25) {
    grid += `<line x1="${pad.l}" y1="${sy(y)}" x2="${W-pad.r}" y2="${sy(y)}" stroke="#21262d" />`;
    grid += `<text x="${pad.l-8}" y="${sy(y)+4}" fill="#8b949e" font-size="11" text-anchor="end">${y}%</text>`;
  }

  const bars = runs.map((r, i) => {
    const s = r.summary;
    const mathPct = s.math_total ? (100 * s.math_passed / s.math_total) : 0;
    const codePct = s.coding_total ? (100 * s.coding_passed / s.coding_total) : 0;
    const gamePct = s.game_total ? (100 * (s.game_passed || 0) / s.game_total) : 0;
    const cx = pad.l + grpW * i + grpW / 2;
    const startX = cx - (nBars * (barW + 3)) / 2;
    const gameRect = hasGames
      ? `<rect x="${startX + 2*(barW+3)}" y="${sy(gamePct)}" width="${barW}" height="${Math.max(1, sy(0) - sy(gamePct))}" fill="#bc8cff" rx="3" />`
      : "";
    const gameLbl = hasGames ? ` · ${s.game_passed || 0}/${s.game_total || 0}G` : "";
    return `
      <rect x="${startX}" y="${sy(mathPct)}" width="${barW}" height="${Math.max(1, sy(0) - sy(mathPct))}" fill="#58a6ff" rx="3" />
      <rect x="${startX + barW + 3}" y="${sy(codePct)}" width="${barW}" height="${Math.max(1, sy(0) - sy(codePct))}" fill="#3fb950" rx="3" />
      ${gameRect}
      <text x="${cx}" y="${H - pad.b + 16}" fill="#e6edf3" font-size="10.5" text-anchor="middle">${esc(r.model_id.slice(0, 18))}</text>
      <text x="${cx}" y="${H - pad.b + 30}" fill="#8b949e" font-size="10" text-anchor="middle">${s.math_passed}/${s.math_total}M · ${s.coding_passed}/${s.coding_total}C${gameLbl}</text>
    `;
  }).join("");

  const gameLegend = hasGames
    ? `<rect x="${W-220}" y="4" width="10" height="10" fill="#bc8cff" rx="2" /><text x="${W-205}" y="13" fill="#8b949e" font-size="11">Games</text>`
    : "";

  return `<svg viewBox="0 0 ${W} ${H}">
    ${grid}${bars}${gameLegend}
    <rect x="${W-155}" y="4" width="10" height="10" fill="#58a6ff" rx="2" /><text x="${W-140}" y="13" fill="#8b949e" font-size="11">Math</text>
    <rect x="${W-90}" y="4" width="10" height="10" fill="#3fb950" rx="2" /><text x="${W-75}" y="13" fill="#8b949e" font-size="11">Coding</text>
  </svg>`;
}

function renderApp() {
  const app = document.getElementById("app");
  const runs = DATA.runs;
  if (!runs.length) {
    app.innerHTML = `<div class="card empty">
      <h2>No benchmark runs found yet</h2>
      <p>Run your first proof-of-concept benchmark from <code>E:\\local-llm\\bench</code>:</p>
      <p><code>python run_bench.py --script ..\\start-gpt-oss-20b.cmd</code></p>
      <p style="margin-top:12px">Results saved to <code>results\\*.json</code> will automatically populate all charts and the clickable trace matrix here.</p>
    </div>`;
    return;
  }

  const bestAcc = runs.reduce((a, b) => (b.summary.pass_rate > a.summary.pass_rate ? b : a), runs[0]);
  const fastRuns = runs.filter(r => (r.summary.avg_gen_tps || 0) >= 15);
  const bestFast = fastRuns.length ? fastRuns.reduce((a, b) => (b.summary.pass_rate > a.summary.pass_rate ? b : a), fastRuns[0]) : null;
  const totalSnoops = runs.reduce((acc, r) => acc + (r.summary.snooping_incidents || 0), 0);

  const allTaskIds = [];
  const taskMeta = {};
  for (const r of runs) {
    for (const t of r.tasks) {
      if (!taskMeta[t.task_id]) {
        allTaskIds.push(t.task_id);
        taskMeta[t.task_id] = {
          title: t.title,
          category: t.category,
          difficulty: t.difficulty || "easy",
          allow_tools: t.allow_tools !== false
        };
      }
    }
  }

  const diffSummary = (tasks, diff) => {
    const sub = tasks.filter(t => (t.difficulty || "easy") === diff);
    if (!sub.length) return "—";
    return `${sub.filter(t => t.passed).length}/${sub.length}`;
  };

  const leaderboardRows = runs.map(r => {
    const s = r.summary;
    const l = r.launcher || {};
    const snoopPill = s.snooping_incidents > 0
      ? `<span class="pill pill-fail">${s.snooping_incidents} flagged</span>`
      : `<span class="pill pill-pass">clean</span>`;
    return `<tr>
      <td><strong>${esc(r.model_id)}</strong><div style="color:var(--muted);font-size:11px">${esc(l.model_file || r.run_id)}</div></td>
      <td><span class="pill pill-info">${esc(l.backend || "—")}</span></td>
      <td><strong>${s.pass_rate}%</strong> (${s.passed}/${s.total})</td>
      <td>${diffSummary(r.tasks, "easy")} / ${diffSummary(r.tasks, "medium")} / ${diffSummary(r.tasks, "hard")}</td>
      <td>${s.math_passed}/${s.math_total}</td>
      <td>${s.coding_passed}/${s.coding_total}</td>
      <td>${s.game_total ? `${s.game_passed || 0}/${s.game_total}` : "—"}</td>
      <td style="color:${(s.avg_gen_tps||0) >= 15 ? 'var(--green)' : 'var(--amber)'}"><strong>${fmt(s.avg_gen_tps, " t/s")}</strong></td>
      <td>${fmt(s.avg_prompt_tps, " t/s")}</td>
      <td>${fmt(s.avg_wall_sec, "s")}</td>
      <td>${fmt(s.avg_output_tokens)}</td>
      <td>${fmt(s.avg_tool_calls)} (${s.total_tool_errors} err)</td>
      <td>${snoopPill}</td>
    </tr>`;
  }).join("");

  const matrixHeaders = runs.map(r => `<th style="text-align:center">${esc(r.model_id)}</th>`).join("");
  const matrixRows = allTaskIds.map(tid => {
    const meta = taskMeta[tid];
    const cells = runs.map((r, ri) => {
      const t = r.tasks.find(x => x.task_id === tid);
      if (!t) return `<td style="text-align:center;color:var(--muted)">—</td>`;
      const cls = t.passed ? "pill-pass" : "pill-fail";
      const snoop = t.snooping_detected ? " ⚠" : "";
      const tamper = t.tests_tampered ? " ✎" : "";
      return `<td class="matrix-cell" data-run="${ri}" data-task="${esc(tid)}">
        <span class="pill ${cls}">${t.passed ? "PASS" : esc(t.status)}${snoop}${tamper}</span>
        <div style="color:var(--muted);font-size:11px;margin-top:2px">${t.wall_sec}s · ${fmt(t.gen_tps, " t/s")}</div>
      </td>`;
    }).join("");
    const diffCls = meta.difficulty === "hard" ? "pill-fail" : (meta.difficulty === "medium" ? "pill-warn" : "pill-pass");
    return `<tr>
      <td><code>${esc(tid)}</code> <span class="pill ${diffCls}">${esc(meta.difficulty)}</span> ${meta.allow_tools ? "" : '<span class="pill pill-info">no-tools</span>'}<div style="color:var(--muted);font-size:11px">${esc(meta.title)} (${esc(meta.category)})</div></td>
      ${cells}
    </tr>`;
  }).join("");

  app.innerHTML = `
    <section class="kpi-grid">
      <div class="kpi"><div class="label">Runs Evaluated</div><div class="val">${runs.length}</div><div class="hint">${allTaskIds.length} unique task(s)</div></div>
      <div class="kpi"><div class="label">Highest Accuracy</div><div class="val" style="color:var(--green)">${bestAcc.summary.pass_rate}%</div><div class="hint">${esc(bestAcc.model_id)} (${bestAcc.summary.passed}/${bestAcc.summary.total})</div></div>
      <div class="kpi"><div class="label">Best ≥ 15 tok/s</div><div class="val" style="color:var(--accent)">${bestFast ? bestFast.summary.pass_rate + "%" : "—"}</div><div class="hint">${bestFast ? esc(bestFast.model_id) + " @ " + bestFast.summary.avg_gen_tps + " t/s" : "No runs with ≥15 t/s metrics yet"}</div></div>
      <div class="kpi"><div class="label">Anti-Snooping Audit</div><div class="val" style="color:${totalSnoops ? 'var(--red)' : 'var(--green)'}">${totalSnoops === 0 ? "CLEAN" : totalSnoops + " ALERTS"}</div><div class="hint">Tool calls audited across all runs</div></div>
    </section>

    <section class="charts-grid">
      <div class="card" id="pareto-card">
        <h2>Accuracy vs. Generation Speed (Pareto)</h2>
        <div class="desc">Hover any point for full details. Dashed line marks your 15 tok/s target.</div>
        ${renderParetoChart(runs)}
      </div>
      <div class="card">
        <h2>Pass Rate by Category (Math / Coding / Games)</h2>
        <div class="desc">Deterministic grading after isolated Pi execution (hidden tests &amp; in-memory game server).</div>
        ${renderCategoryChart(runs)}
      </div>
    </section>

    <section class="card">
      <h2>Model Leaderboard</h2>
      <div class="desc">Speed metrics come from <code>llama-server --metrics</code> deltas per task.</div>
      <div style="overflow-x:auto">
        <table>
          <thead><tr>
            <th>Model</th><th>Backend</th><th>Overall</th><th>E / M / H</th><th>Math</th><th>Coding</th><th>Games</th>
            <th>Gen tok/s</th><th>Prompt tok/s</th><th>Avg Wall</th><th>Avg Out Tok</th><th>Tools</th><th>Snooping</th>
          </tr></thead>
          <tbody>${leaderboardRows}</tbody>
        </table>
      </div>
    </section>

    <section class="card">
      <h2>Task × Model Matrix</h2>
      <div class="desc">Click any cell to inspect the full <code>pi</code> tool-call trace, thinking preview, snooping audit, and test output.</div>
      <div style="overflow-x:auto">
        <table>
          <thead><tr><th>Task</th>${matrixHeaders}</tr></thead>
          <tbody>${matrixRows}</tbody>
        </table>
      </div>
    </section>
  `;

  // Wire Pareto hover tooltips
  document.querySelectorAll("#pareto-card .pt").forEach(el => {
    const r = runs[Number(el.dataset.idx)];
    const s = r.summary;
    el.addEventListener("mousemove", e => showTip(e,
      `<strong>${esc(r.model_id)}</strong><br>` +
      `Backend: ${esc(r.launcher?.backend || "—")}<br>` +
      `Pass: ${s.pass_rate}% (${s.passed}/${s.total})<br>` +
      `Gen: ${fmt(s.avg_gen_tps, " t/s")} · Prompt: ${fmt(s.avg_prompt_tps, " t/s")}<br>` +
      `Avg tools: ${s.avg_tool_calls} · Avg out tok: ${s.avg_output_tokens}`
    ));
    el.addEventListener("mouseleave", hideTip);
  });

  // Wire Matrix click -> trace inspector modal
  document.querySelectorAll(".matrix-cell").forEach(cell => {
    cell.addEventListener("click", () => {
      const run = runs[Number(cell.dataset.run)];
      const task = run.tasks.find(t => t.task_id === cell.dataset.task);
      if (task) openTraceModal(run, task);
    });
  });
}

renderApp();
</script>
</body>
</html>
"""


def build_report(output_path: Path = REPORT_PATH) -> Path:
    payload = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "runs": load_runs(),
        "hw_profiles": load_hw_profiles(),
    }
    html = HTML_TEMPLATE.replace("__BENCH_DATA_JSON__", json.dumps(payload))
    output_path.write_text(html, encoding="utf-8")
    return output_path


def main() -> int:
    out = build_report()
    print(f"Generated report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
