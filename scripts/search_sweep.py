"""Run one agent search per (workload, prompt framing) and summarize speedup against the measured ceiling.

Each cell is an ordinary `agent.py` run with its own RunConfig file and output directory, so every cell
keeps the full evidence (requests, attempts, raw sessions). The summary reads only those artifacts.

    uv run --locked python scripts/search_sweep.py --base configs/live_gemini_flash.json \\
        --ceilings evidence/ceilings/mac-m5-cpu-docker/results.json --out runs/sweep-01

A cell's speedup is the geometric mean of the judged sessions of its final best against the baseline;
1.0 when nothing was accepted. Share of ceiling = ln(speedup) / ln(ceiling) for headroom workloads.
Accepting a change on a control (XLA already optimal) is counted as a false positive.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from workloads import registry  # noqa: E402

FRAMINGS = ("baseline", "compiler_complement")


def cell_result(run_dir):
    """(state, speedup vs baseline, accepted count, evaluated count) from a finished run's history."""
    history_path = run_dir / "history.json"
    if not history_path.exists():
        return {"state": "NO_HISTORY", "speedup": None, "accepted": 0, "evaluated": 0}
    history = json.loads(history_path.read_text(encoding="utf-8"))
    accepted = [entry for entry in history[1:] if entry.get("state") == "ACCEPTED"]
    evaluated = sum(1 for entry in history[1:] if (entry.get("attempt") or {}).get("evaluation"))
    speedup = 1.0
    if accepted:
        sessions = (accepted[-1].get("attempt") or {}).get("evaluation", {}).get("sessions", [])
        against = "baseline" if any(s["against"] == "baseline" for s in sessions) else "incumbent"
        if against == "incumbent" and len(accepted) > 1:
            # chained promotions without a baseline session: multiply the per-step medians
            speedup = math.prod(_session_speedup(a, "incumbent") for a in accepted)
        else:
            speedup = _session_speedup(accepted[-1], against)
    last = history[-1].get("state") if len(history) > 1 else "NO_CANDIDATE"
    return {"state": "ACCEPTED" if accepted else last, "speedup": speedup,
            "accepted": len(accepted), "evaluated": evaluated}


def _session_speedup(entry, against):
    values = [s["analysis"]["speedup"]["speedup"] for s in entry["attempt"]["evaluation"]["sessions"]
              if s["against"] == against]
    return math.exp(sum(math.log(v) for v in values) / len(values))


def run_cell(base, workload, framing, out, env, calibrations=None):
    cell = f"{workload}-{framing}"
    config = {**base, "workload_id": workload, "prompt_framing": framing}
    spec = registry.get(workload)
    if calibrations is not None:
        frozen = Path(calibrations) / f"{workload}-{'x'.join(map(str, spec.development_shapes[-1]))}.json"
        if frozen.exists():
            config["calibration_path"] = str(frozen.resolve())
    config_path = out / "configs" / f"{cell}.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    run_dir = out / "runs" / cell
    log_path = out / "logs" / f"{cell}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run([sys.executable, str(REPO / "agent.py"), "--config", str(config_path),
                               "--output-dir", str(run_dir)], cwd=REPO, env=env, stdout=log,
                              stderr=subprocess.STDOUT, check=False)
    return run_dir, proc.returncode


def summarize(out, ceilings):
    rows = []
    for run_dir in sorted((out / "runs").iterdir()):
        workload, framing = run_dir.name.rsplit("-", 1)[0], run_dir.name.rsplit("-", 1)[1]
        for name in FRAMINGS:                       # framing names contain '_', not '-'
            if run_dir.name.endswith("-" + name):
                workload, framing = run_dir.name[: -len(name) - 1], name
        result = cell_result(run_dir)
        spec = registry.get(workload)
        kind = "headroom" if spec in registry.HEADROOM else "control"
        ceiling = ceilings.get(workload)
        share = (math.log(result["speedup"]) / math.log(ceiling)
                 if kind == "headroom" and ceiling and ceiling > 1 and result["speedup"] else None)
        requests = out / "runs" / run_dir.name / "requests.json"
        rows.append({"workload": workload, "class": kind, "framing": framing, **result, "ceiling": ceiling,
                     "share_of_ceiling": share, "requests_file": str(requests) if requests.exists() else None})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--base", type=Path, required=True, help="RunConfig v1 with provider, prices and cap")
    parser.add_argument("--ceilings", type=Path, required=True, help="results.json from measure_ceilings.py")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workload", action="append", help="limit to these workloads (repeatable)")
    parser.add_argument("--framing", action="append", choices=FRAMINGS, help="limit to these framings")
    parser.add_argument("--summary-only", action="store_true", help="only re-read finished cells")
    parser.add_argument("--calibrations", type=Path, default=None,
                        help="directory of frozen calibrate_aa.py records, shared with measure_ceilings.py")
    args = parser.parse_args(argv)
    base = json.loads(args.base.read_text(encoding="utf-8"))
    ceilings = {}
    for row in json.loads(args.ceilings.read_text(encoding="utf-8")):
        if row["speedup"]:
            ceilings[row["workload"]] = math.exp(sum(map(math.log, row["speedup"])) / len(row["speedup"]))
    if not args.summary_only:
        args.out.mkdir(parents=True, exist_ok=True)
        workloads = args.workload or [s.workload_id for s in (*registry.HEADROOM, *registry.CONTROLS)]
        env = dict(os.environ)
        for workload in workloads:
            for framing in args.framing or FRAMINGS:
                run_dir, code = run_cell(base, workload, framing, args.out, env, args.calibrations)
                result = cell_result(run_dir)
                print(f"{workload:20s} {framing:20s} exit={code} {result['state']:16s} "
                      f"speedup={result['speedup'] if result['speedup'] is None else round(result['speedup'], 2)}",
                      flush=True)
    rows = summarize(args.out, ceilings)
    (args.out / "summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print("\nworkload             class     framing              speedup  ceiling  share  accepted/evaluated")
    for r in rows:
        share = f"{100 * r['share_of_ceiling']:.0f}%" if r["share_of_ceiling"] is not None else "-"
        speed = f"{r['speedup']:.2f}x" if r["speedup"] else "-"
        ceil = f"{r['ceiling']:.2f}x" if r["ceiling"] else "-"
        print(f"{r['workload']:20s} {r['class']:9s} {r['framing']:20s} {speed:8s} {ceil:8s} {share:6s} "
              f"{r['accepted']}/{r['evaluated']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
