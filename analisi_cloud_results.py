"""
Analisi ML dei risultati dei 10 modelli cloud.
Legge i results_*.csv e calcola accuracy, F1, valid/tot per ogni modello
× dataset × strategia. Genera:
  - results/analisi_cloud_summary.csv   (tabella completa)
  - results/analisi_cloud_report.txt    (report leggibile)
  - plots/cloud_accuracy_pi.png         (bar chart pain intensity)
  - plots/cloud_accuracy_etio.png       (bar chart etiology)
"""

import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── Configurazione ───────────────────────────────────────────────────────────
RESULTS_DIR = "results"
PLOTS_DIR   = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

DATASETS = {
    "DEEPSEEK": "results_MC_gill_DEEPSEEK.csv",
    "GPT":      "results_McGill_Pain_Questionnaire_GPT_Con_Dolore.csv",
    "CLAUDE":   "results_McGill_Pain_Questionnaire_CLAUDE.csv",
    "DOCTORAI": "results_McGill_Pain_Questionnaire_DOCTORAI.csv",
}

PI_STRATEGIES = [
    ("pi_zero_shot_pred",        "zero_shot"),
    ("pi_cot_simple_pred",       "cot_simple"),
    ("pi_cot_hierarchical_pred", "cot_hierarchical"),
    ("pi_cot_verification_pred", "cot_verification"),
    ("pi_cot_ensemble_pred",     "cot_ensemble"),
    ("pi_cot_decision_tree_pred","cot_decision_tree"),
    ("pi_cot_confidence_pred",   "cot_confidence"),
    ("pi_few_shot_pred",         "few_shot"),
]

ETIO_STRATEGIES = [
    ("etio_zero_shot_pred",       "zero_shot"),
    ("etio_cot_simple_pred",      "cot_simple"),
    ("etio_cot_verification_pred","cot_verification"),
    ("etio_few_shot_pred",        "few_shot"),
    ("etio_knowledge_guided_pred","knowledge_guided"),
]


def compute_metrics(y_true, y_pred_raw, average):
    pairs = [(yt, yp) for yt, yp in zip(y_true, y_pred_raw)
             if pd.notna(yp) and pd.notna(yt)]
    total = len(y_true)
    valid = len(pairs)
    if valid == 0:
        return None, None, valid, total
    yt, yp = zip(*pairs)
    yt = [int(v) for v in yt]
    yp = [int(v) for v in yp]
    acc = accuracy_score(yt, yp)
    f1  = f1_score(yt, yp, average=average, zero_division=0)
    return acc, f1, valid, total


# ─── Calcolo metriche ─────────────────────────────────────────────────────────
rows = []

for ds_name, ds_file in DATASETS.items():
    path = os.path.join(RESULTS_DIR, ds_file)
    if not os.path.exists(path):
        print(f"  [SKIP] {ds_file} non trovato")
        continue
    df = pd.read_csv(path)
    print(f"\n{ds_name}: {df['model'].nunique()} modelli, {len(df)} righe")

    for model in sorted(df["model"].unique()):
        mdf = df[df["model"] == model]

        for pred_col, strat in PI_STRATEGIES:
            if pred_col not in mdf.columns:
                continue
            acc, f1, valid, total = compute_metrics(
                mdf["true_pain_intensity"], mdf[pred_col], "macro")
            rows.append({
                "dataset": ds_name, "model": model,
                "task": "pain_intensity", "strategy": strat,
                "accuracy": round(acc, 4) if acc is not None else None,
                "f1": round(f1, 4) if f1 is not None else None,
                "valid": valid, "total": total,
            })

        for pred_col, strat in ETIO_STRATEGIES:
            if pred_col not in mdf.columns:
                continue
            acc, f1, valid, total = compute_metrics(
                mdf["true_etiology"], mdf[pred_col], "binary")
            rows.append({
                "dataset": ds_name, "model": model,
                "task": "etiology", "strategy": strat,
                "accuracy": round(acc, 4) if acc is not None else None,
                "f1": round(f1, 4) if f1 is not None else None,
                "valid": valid, "total": total,
            })

summary = pd.DataFrame(rows)
summary.to_csv(os.path.join(RESULTS_DIR, "analisi_cloud_summary.csv"), index=False)
print(f"\nSummary salvato: {len(summary)} righe")


# ─── Report leggibile ─────────────────────────────────────────────────────────
report_path = os.path.join(RESULTS_DIR, "analisi_cloud_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("=" * 72 + "\n")
    f.write("ANALISI 10 MODELLI CLOUD - McGill Pain Questionnaire\n")
    f.write("=" * 72 + "\n\n")

    for ds_name in DATASETS:
        ds_df = summary[summary["dataset"] == ds_name]
        if ds_df.empty:
            continue
        f.write(f"\n{'='*72}\n")
        f.write(f"DATASET: {ds_name}\n")
        f.write(f"{'='*72}\n")

        for model in sorted(ds_df["model"].unique()):
            mdf = ds_df[ds_df["model"] == model]
            f.write(f"\n  {'─'*60}\n")
            f.write(f"  {model}\n")
            f.write(f"  {'─'*60}\n")

            f.write(f"\n  PAIN INTENSITY (3 classi, F1-macro)\n")
            f.write(f"  {'Strategia':<25} {'Acc':>7} {'F1':>7} {'Valid/Tot':>12}\n")
            f.write(f"  {'-'*54}\n")
            pi_df = mdf[mdf["task"] == "pain_intensity"].sort_values("strategy")
            for _, r in pi_df.iterrows():
                acc_s = f"{r['accuracy']:.3f}" if pd.notna(r['accuracy']) else "  N/A"
                f1_s  = f"{r['f1']:.3f}"       if pd.notna(r['f1'])       else "  N/A"
                f.write(f"  {r['strategy']:<25} {acc_s:>7} {f1_s:>7} {r['valid']:>6}/{r['total']:<6}\n")

            f.write(f"\n  ETIOLOGY (binaria, F1-binary)\n")
            f.write(f"  {'Strategia':<25} {'Acc':>7} {'F1':>7} {'Valid/Tot':>12}\n")
            f.write(f"  {'-'*54}\n")
            et_df = mdf[mdf["task"] == "etiology"].sort_values("strategy")
            for _, r in et_df.iterrows():
                acc_s = f"{r['accuracy']:.3f}" if pd.notna(r['accuracy']) else "  N/A"
                f1_s  = f"{r['f1']:.3f}"       if pd.notna(r['f1'])       else "  N/A"
                f.write(f"  {r['strategy']:<25} {acc_s:>7} {f1_s:>7} {r['valid']:>6}/{r['total']:<6}\n")

print(f"Report salvato: {report_path}")


# ─── Grafici: best strategy per modello × dataset ─────────────────────────────
def best_per_model(df_task, metric="f1"):
    """Per ogni (dataset, model): prende la strategia con F1 massimo."""
    valid = df_task.dropna(subset=[metric])
    if valid.empty:
        return pd.DataFrame()
    idx = valid.groupby(["dataset", "model"])[metric].idxmax()
    return valid.loc[idx].reset_index(drop=True)

model_short = {m: m.replace("Cloud/","") for m in summary["model"].unique()}

for task, avg_label, fname in [
    ("pain_intensity", "F1-macro",  "cloud_f1_pi.png"),
    ("etiology",       "F1-binary", "cloud_f1_etio.png"),
]:
    task_df = summary[summary["task"] == task]
    best    = best_per_model(task_df)
    if best.empty:
        continue

    datasets = list(DATASETS.keys())
    models   = sorted(best["model"].unique())
    x        = np.arange(len(models))
    width    = 0.2
    fig, ax  = plt.subplots(figsize=(14, 5))

    colors = ["#378ADD","#1D9E75","#D85A30","#534AB7"]
    for i, ds in enumerate(datasets):
        vals = []
        for m in models:
            row = best[(best["dataset"] == ds) & (best["model"] == m)]
            vals.append(row["f1"].values[0] if not row.empty and pd.notna(row["f1"].values[0]) else 0)
        ax.bar(x + i * width, vals, width, label=ds, color=colors[i], alpha=0.85)

    ax.set_xlabel("Modello")
    ax.set_ylabel(avg_label)
    ax.set_title(f"Best {avg_label} per modello × dataset  ({task.replace('_',' ').title()})")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([model_short[m] for m in models], rotation=35, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.legend(title="Dataset")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, fname)
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Grafico salvato: {out}")

# ─── Tabella riepilogativa: best F1 per modello (media 4 dataset) ─────────────
print("\n" + "="*72)
print("BEST F1 MEDIO (4 dataset) PER MODELLO")
print("="*72)

for task, avg_label in [("pain_intensity","F1-macro"),("etiology","F1-binary")]:
    task_df = summary[summary["task"] == task]
    best    = best_per_model(task_df)
    if best.empty:
        continue
    avg = best.groupby("model")["f1"].mean().sort_values(ascending=False)
    print(f"\n  {avg_label}")
    print(f"  {'Modello':<35} {'F1 medio':>8}")
    print(f"  {'-'*45}")
    for model, val in avg.items():
        print(f"  {model_short[model]:<35} {val:.4f}")

print("\nAnalisi completata.")
