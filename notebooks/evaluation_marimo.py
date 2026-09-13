"""EchoLoop evaluation notebook (marimo / molab).

Run locally:  marimo edit notebooks/evaluation_marimo.py
On molab:     upload this file, then run `python -m eval.run_eval` output files
              (data/evaluation_report.json, data/eval_curves.csv) alongside it.
"""
import marimo

__generated_with = "0.9.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    from pathlib import Path

    import marimo as mo
    import pandas as pd

    root = Path(__file__).resolve().parent.parent
    report = json.loads((root / "data" / "evaluation_report.json").read_text(encoding="utf-8"))
    curves = pd.read_csv(root / "data" / "eval_curves.csv")
    mo.md(f"# EchoLoop evaluation\n\n**{report['dataset']}**\n\n"
          f"Replayed {report['rows_replayed']} interactions · "
          f"{report['n_test']} held-out test interactions · generated {report['generated']}")
    return curves, json, mo, pd, report, root


@app.cell
def _(mo, pd, report):
    table = pd.DataFrame(report["ablations"])[
        ["name", "top1", "top3", "candidate_coverage", "avg_reward",
         "clarification_burden", "latency_ms_per_interaction"]]
    mo.ui.table(table, label="Ablations: voice → +video → +memory → +RL reranking")
    return (table,)


@app.cell
def _(mo, report):
    gain = report["personalization_gain"] * 100
    mo.md(f"## Personalization gain\n\n"
          f"**{gain:+.1f} points top-1** from the contextual-bandit reranker, "
          f"over the same system with a frozen policy.\n\n"
          f"Memory hit rate: {report['memory_hit_rate']*100:.1f}%")
    return (gain,)


@app.cell
def _(curves):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    for name, g in curves.groupby("configuration"):
        ax.plot(g["turn_bucket"], g["top1_accuracy"], marker="o", label=name)
    ax.set_xlabel("interaction number (per user)")
    ax.set_ylabel("top-1 accuracy")
    ax.set_title("Learning curve: accuracy vs interactions seen")
    ax.legend(fontsize=7)
    ax.grid(alpha=.3)
    fig
    return ax, fig, g, name, plt


@app.cell
def _(pd, plt, report):
    fig2, ax2 = plt.subplots(figsize=(8, 3.5))
    abl = pd.DataFrame(report["ablations"])
    ax2.barh(abl["name"], abl["clarification_burden"], color="#6366f1")
    ax2.set_xlabel("extra clarification turns per interaction (lower is better)")
    ax2.invert_yaxis()
    fig2.tight_layout()
    fig2
    return abl, ax2, fig2


@app.cell
def _(pd, plt, report):
    best = report["ablations"][-1]["failure_counts"]
    fig3, ax3 = plt.subplots(figsize=(6, 3))
    ax3.bar(list(best.keys()), list(best.values()), color="#f59e0b")
    ax3.set_title("Failure categories (full system)")
    fig3.tight_layout()
    fig3
    return ax3, best, fig3


@app.cell
def _(mo, report):
    mo.md(f"## Compute\n\n"
          f"- device: `{report['compute']['device']}`\n"
          f"- examples processed: {report['compute']['examples_processed']}\n"
          f"- models: {report['compute']['models']}\n\n"
          f"Integrations live this run: "
          f"{', '.join(k for k, v in report['integrations'].items() if v) or 'none'}")
    return


if __name__ == "__main__":
    app.run()
