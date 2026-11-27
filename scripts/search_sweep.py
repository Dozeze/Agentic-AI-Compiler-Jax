"""Run one agent search per (workload, model, framing, repetition) and summarize speedup against the ceiling.

Each cell is an ordinary `agent.py` run with its own RunConfig file and output directory, so every cell keeps
the full evidence (requests, attempts, raw sessions). The summary reads only those artifacts.

    uv run --locked python scripts/search_sweep.py --base configs/live_gemini_flash.json \\
        --base configs/live_gemini_flash_lite.json --ceilings evidence/ceilings/<dir>/results.json \\
        --calibrations evidence/calibration/<label> --repeats 3 --out runs/sweep-02

A cell's speedup is the product of its accepted promotions, each the geometric mean of its judged sessions
against the incumbent (1.0 when nothing was accepted). `best_after` lists that speedup after 1, 2, ... of the
cell's candidate attempts, which shows what iterating on feedback adds over a single proposal. Share of
ceiling = ln(speedup) / ln(ceiling) for headroom workloads; accepting anything on a control is a false positive.
Finished cells are skipped, so an interrupted sweep resumes where it stopped.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from workloads import registry  # noqa: E402

FRAMINGS = ("baseline", "compiler_complement")


def _session_speedup(entry):
    values = [s["analysis"]["speedup"]["speedup"] for s in entry["attempt"]["evaluation"]["sessions"]
              if s["against"] == "incumbent"]
    return math.exp(sum(map(math.log, values)) / len(values))


def cell_result(run_dir):
    """Speedup vs the baseline after each candidate attempt, from a finished run's history."""
    path = run_dir / "history.json"
    if not path.exists():
        return {"state": "NO_HISTORY", "speedup": None, "best_after": [], "accepted": 0, "evaluated": 0,
                "states": []}
    history = json.loads(path.read_text(encoding="utf-8"))[1:]          # [0] is the baseline
    speedup, best_after = 1.0, []
    for entry in history:
        if entry.get("state") == "ACCEPTED":
            speedup *= _session_speedup(entry)
        best_after.append(speedup)
    return {"state": "ACCEPTED" if speedup > 1.0 else (history[-1]["state"] if history else "NO_CANDIDATE"),
            "speedup": speedup, "best_after": best_after,
            "accepted": sum(entry.get("state") == "ACCEPTED" for entry in history),
            "evaluated": sum(bool((entry.get("attempt") or {}).get("evaluation")) for entry in history),
            "states": [entry.get("state") for entry in history]}


def cell_cost(run_dir):
    path = run_dir / "requests.json"
    if not path.exists():
        return 0.0, 0
    events = json.loads(path.read_text(encoding="utf-8"))
    return sum(float(e.get("money_accounted") or 0) for e in events), len(events)


def model_tag(model_id):
    return model_id.replace("gemini-", "").replace(".", "")


def run_cell(name, config, out, env):
    config_path = out / "configs" / f"{name}.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    run_dir = out / "runs" / name
    (out / "logs").mkdir(parents=True, exist_ok=True)
    with (out / "logs" / f"{name}.log").open("w", encoding="utf-8") as log:
        proc = subprocess.run([sys.executable, str(REPO / "agent.py"), "--config", str(config_path),
                               "--output-dir", str(run_dir)], cwd=REPO, env=env, stdout=log,
                              stderr=subprocess.STDOUT, check=False)
    return run_dir, proc.returncode


def summarize(out, ceilings):
    rows = []
    for config_path in sorted((out / "configs").glob("*.json")):
        name = config_path.stem
        config = json.loads(config_path.read_text(encoding="utf-8"))
        run_dir = out / "runs" / name
        if not run_dir.exists():
            continue
        spec = registry.get(config["workload_id"])
        kind = "control" if spec in registry.CONTROLS else ("block" if spec in registry.BLOCK_TIER else "op")
        result = cell_result(run_dir)
        cost, requests = cell_cost(run_dir)
        ceiling = ceilings.get(spec.workload_id)
        share = (math.log(result["speedup"]) / math.log(ceiling)
                 if kind != "control" and ceiling and ceiling > 1 and result["speedup"] else None)
        rows.append({"cell": name, "workload": spec.workload_id, "tier": kind, "model": config["model"]["model_id"],
                     "framing": config.get("prompt_framing", "baseline"), "rep": int(name.rsplit("-r", 1)[1]),
                     **result, "ceiling": ceiling, "share_of_ceiling": share, "cost_usd": cost,
                     "requests": requests})
    return rows


def aggregate(rows):
    """Per (tier, model, framing): mean share of ceiling, share after k candidates, false positives, cost."""
    groups = {}
    for row in rows:
        groups.setdefault((row["tier"], row["model"], row["framing"]), []).append(row)
    table = []
    for (tier, model, framing), cells in sorted(groups.items()):
        entry = {"tier": tier, "model": model, "framing": framing, "cells": len(cells),
                 "cost_usd": sum(c["cost_usd"] for c in cells),
                 "evaluated": statistics.mean(c["evaluated"] for c in cells)}
        if tier == "control":
            entry["false_positives"] = sum(c["accepted"] > 0 for c in cells)
        else:
            entry["reached_90pct"] = sum((c["share_of_ceiling"] or 0) >= 0.9 for c in cells)
            entry["mean_share"] = statistics.mean(max(0.0, c["share_of_ceiling"] or 0.0) for c in cells)
            for k in (1, 2, 3):
                shares = [math.log(c["best_after"][min(k, len(c["best_after"])) - 1]) / math.log(c["ceiling"])
                          if c["best_after"] and c["ceiling"] and c["ceiling"] > 1 else 0.0 for c in cells]
                entry[f"mean_share_after_{k}"] = statistics.mean(max(0.0, s) for s in shares)
        table.append(entry)
    return table


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--base", type=Path, action="append", required=True,
                        help="RunConfig v1 with provider, prices and cap; repeat for several models")
    parser.add_argument("--ceilings", type=Path, action="append", required=True,
                        help="results.json from measure_ceilings.py (repeatable)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--calibrations", type=Path, default=None,
                        help="directory of frozen calibrate_aa.py records, shared with measure_ceilings.py")
    parser.add_argument("--workload", action="append", help="limit to these workloads (repeatable)")
    parser.add_argument("--framing", action="append", choices=FRAMINGS, help="limit to these framings")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--summary-only", action="store_true", help="only re-read finished cells")
    args = parser.parse_args(argv)
    ceilings = {}
    for path in args.ceilings:
        for row in json.loads(path.read_text(encoding="utf-8")):
            if row["speedup"]:
                ceilings[row["workload"]] = math.exp(sum(map(math.log, row["speedup"])) / len(row["speedup"]))
    if not args.summary_only:
        workloads = args.workload or [s.workload_id for s in (*registry.HEADROOM, *registry.CONTROLS)]
        env = dict(os.environ)
        for rep in range(1, args.repeats + 1):            # repetition outermost: partial sweeps stay balanced
            for base_path in args.base:
                base = json.loads(base_path.read_text(encoding="utf-8"))
                for workload in workloads:
                    spec = registry.get(workload)
                    for framing in args.framing or FRAMINGS:
                        name = f"{workload}-{framing}-{model_tag(base['model']['model_id'])}-r{rep}"
                        if (args.out / "runs" / name / "history.json").exists():
                            continue
                        config = {**base, "workload_id": workload, "prompt_framing": framing}
                        if args.calibrations is not None:
                            frozen = args.calibrations / f"{workload}-{'x'.join(map(str, spec.development_shapes[-1]))}.json"
                            if frozen.exists():
                                config["calibration_path"] = str(frozen.resolve())
                        run_dir, code = run_cell(name, config, args.out, env)
                        result = cell_result(run_dir)
                        speed = "-" if result["speedup"] is None else f"{result['speedup']:.2f}x"
                        print(f"{name:58s} exit={code} {result['state']:16s} {speed}", flush=True)
    rows = summarize(args.out, ceilings)
    table = aggregate(rows)
    (args.out / "summary.json").write_text(json.dumps({"cells": rows, "groups": table}, indent=2) + "\n",
                                           encoding="utf-8")
    print(f"\n{'tier':8s} {'model':22s} {'framing':20s} cells  >=90%  mean  k=1   k=2   k=3   FP  evals  cost")
    for g in table:
        if g["tier"] == "control":
            print(f"{g['tier']:8s} {g['model']:22s} {g['framing']:20s} {g['cells']:5d}  {'-':5s} {'-':5s} {'':17s} "
                  f"{g['false_positives']:3d} {g['evaluated']:5.1f} ${g['cost_usd']:.3f}")
        else:
            print(f"{g['tier']:8s} {g['model']:22s} {g['framing']:20s} {g['cells']:5d}  {g['reached_90pct']:5d} "
                  f"{100 * g['mean_share']:4.0f}% {100 * g['mean_share_after_1']:4.0f}% {100 * g['mean_share_after_2']:4.0f}% "
                  f"{100 * g['mean_share_after_3']:4.0f}%  -  {g['evaluated']:5.1f} ${g['cost_usd']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
