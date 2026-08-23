#!/usr/bin/env python
"""Regenerates EXPERIMENTS.md from every runs/*.json log written by
run_attack.py / evaluate.py (see io_utils.save_run_log). Pure read of runs/ --
does not touch checkpoints/data/models, so it's safe and fast to rerun after
every new run.

Usage:
    python scripts/gen_experiment_log.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
RUNS_DIR = PROJECT_DIR / "runs"
OUT_PATH = PROJECT_DIR / "EXPERIMENTS.md"


def load_runs() -> list[dict]:
    runs = []
    for path in sorted(RUNS_DIR.glob("*.json")):
        with open(path) as f:
            data = json.load(f)
        data["_file"] = path.name
        data["_run_type"] = "run_attack" if "config" in data else "evaluate"
        runs.append(data)
    runs.sort(key=lambda d: d.get("timestamp", ""))
    return runs


def target_rows(rows: list[dict], attack: str) -> list[dict]:
    """Rows for this attack, targets only (excludes the surrogate, group == "")."""
    return [r for r in rows if r["attack"] == attack and r.get("group")]


def mean(values: list[float]) -> float | None:
    values = [v for v in values if v is not None and v == v]  # drop None/NaN
    return sum(values) / len(values) if values else None


def fmt(x, pct=False) -> str:
    if x is None:
        return "—"
    return f"{x:.1f}%" if pct else f"{x:.4f}"


def summary_table(runs: list[dict]) -> str:
    header = (
        "| thời gian | loại run | attack | manifest | steps | crafted/skipped | "
        "ASR TB (target) | mAP-drop % TB (target) | log |\n"
        "|---|---|---|---|---|---|---|---|---|\n"
    )
    lines = [header]
    for run in runs:
        ts = run.get("timestamp", "?")
        rtype = run["_run_type"]
        manifest = Path(run.get("manifest", "?")).name
        results = run.get("results", {})
        rows = results.get("rows", [])

        if rtype == "run_attack":
            attacks = [run["attack"]]
            steps = run.get("config", {}).get("steps", "?")
            n_done = f"{results.get('n_crafted', '?')}/{results.get('n_skipped', '?')}"
        else:
            attacks = run.get("attacks", [])
            steps = "—"
            n_done = "—"

        asr_means, drop_means = [], []
        for attack in attacks:
            arows = target_rows(rows, attack)
            asr_means.append(mean([r["ASR"] for r in arows if r.get("ASR") not in (None, "")]))
            drop_means.append(mean([r["mAP_drop_pct"] for r in arows]))

        lines.append(
            f"| {ts} | {rtype} | {', '.join(attacks)} | {manifest} | {steps} | {n_done} | "
            f"{fmt(mean([v for v in asr_means if v is not None]), pct=True)} | "
            f"{fmt(mean([v for v in drop_means if v is not None]), pct=True)} | "
            f"`runs/{run['_file']}` |\n"
        )
    return "".join(lines)


def detail_section(run: dict) -> str:
    ts = run.get("timestamp", "?")
    rtype = run["_run_type"]
    manifest = run.get("manifest", "?")
    rows = run.get("results", {}).get("rows", [])

    out = [f"### {ts} — `{run['_file']}`\n\n"]
    out.append(f"- loại run: `{rtype}`\n")
    out.append(f"- manifest: `{manifest}`\n")
    if rtype == "run_attack":
        cfg = run.get("config", {})
        out.append(f"- attack: `{run.get('attack')}`\n")
        out.append(
            "- config: "
            + ", ".join(f"{k}={v}" for k, v in cfg.items())
            + "\n"
        )
        res = run.get("results", {})
        out.append(
            f"- crafted={res.get('n_crafted')} skipped={res.get('n_skipped')} "
            f"craft={res.get('craft_elapsed_sec')}s eval={res.get('eval_elapsed_sec')}s\n"
        )
    else:
        out.append(f"- attacks: {run.get('attacks')}\n")
        out.append(f"- score_thr={run.get('score_thr')} iou_thr={run.get('iou_thr')}\n")

    out.append(
        "\n| attack | model | nhóm | mAP_clean | mAP_adv | mAP_drop % | AP50_clean | AP50_adv | ASR |\n"
        "|---|---|---|---|---|---|---|---|---|\n"
    )
    for r in rows:
        out.append(
            f"| {r['attack']} | {r['model_name']} | {r.get('group') or '—'} | "
            f"{fmt(r.get('mAP_clean'))} | {fmt(r.get('mAP_adv'))} | {fmt(r.get('mAP_drop_pct'), pct=True) if r.get('mAP_drop_pct') is not None else '—'} | "
            f"{fmt(r.get('AP50_clean'))} | {fmt(r.get('AP50_adv'))} | "
            f"{fmt(r['ASR'], pct=True) if r.get('ASR') not in (None, '') else '—'} |\n"
        )
    out.append("\n")
    return "".join(out)


def main() -> None:
    runs = load_runs()
    if not runs:
        print(f"[gen_experiment_log] không tìm thấy run nào trong {RUNS_DIR}", file=sys.stderr)

    parts = [
        "<!-- TỰ SINH bởi scripts/gen_experiment_log.py từ runs/*.json. Không sửa tay. -->\n",
        "# Nhật ký thực nghiệm\n\n",
        f"{len(runs)} run được log trong `runs/`. Xem [RESEARCH.md](RESEARCH.md) để biết mục tiêu/metric/dataset.\n\n",
        "## Tổng hợp\n\n",
        summary_table(runs),
        "\n## Chi tiết từng run\n\n",
    ]
    for run in reversed(runs):  # mới nhất trước
        parts.append(detail_section(run))

    OUT_PATH.write_text("".join(parts))
    print(f"[gen_experiment_log] đã ghi {OUT_PATH} ({len(runs)} run)")


if __name__ == "__main__":
    main()
