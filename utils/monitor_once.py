"""
Stampa un report di avanzamento su console.
Lanciare: python monitor_once.py
"""
import pandas as pd, os, json, datetime, sys

sys.stdout.reconfigure(encoding="utf-8")

CLOUD_MODELS = [
    "gemma4:31b", "gemma3:27b", "ministral-3:14b",
    "gemma3:4b", "gemma3:12b", "rnj-1:8b",
    "ministral-3:8b", "devstral-small-2:24b",
]
DATASETS = {
    "MC_gill_DEEPSEEK":                        500,
    "McGill_Pain_Questionnaire_GPT_Con_Dolore": 500,
    "McGill_Pain_Questionnaire_CLAUDE":         500,
    "McGill_Pain_Questionnaire_DOCTORAI":       498,
}


def main():
    now = datetime.datetime.now().strftime("%H:%M %d/%m/%Y")
    lines = [f"MPQ Monitor — {now}\n"]

    grand_done, grand_total = 0, 0
    all_complete = True

    for ds, total in DATASETS.items():
        fname = f"checkpoints/checkpoint_{ds}_cloud.csv"
        if not os.path.exists(fname):
            lines.append(f"[{ds[:30]}]: non iniziato")
            grand_total += total * len(CLOUD_MODELS)
            all_complete = False
            continue

        df = pd.read_csv(fname, on_bad_lines="skip")
        pred_cols = [c for c in df.columns if c.endswith("_pred")]
        done = 0
        for m in CLOUD_MODELS:
            rows = df[df["model"] == f"Cloud/{m}"]
            if len(rows) and len(pred_cols):
                done += int(rows[pred_cols].notna().all(axis=1).sum())
        exp  = total * len(CLOUD_MODELS)
        grand_done  += done
        grand_total += exp
        pct = done / exp * 100 if exp > 0 else 0
        if done < exp:
            all_complete = False
        status = "OK" if done >= exp else "..."
        lines.append(f"[{status}] {ds[:30]}: {done}/{exp} ({pct:.0f}%)")

    lines.append(
        f"\nTOTALE: {grand_done}/{grand_total} "
        f"({grand_done / grand_total * 100:.0f}%)"
    )

    if os.path.exists("key_state.json"):
        with open("key_state.json", encoding="utf-8") as f:
            ks = json.load(f)
        exhausted = ks.get("exhausted", {})
        if exhausted:
            names = [v["name"] for v in exhausted.values()]
            lines.append(f"\nChiavi esaurite ({len(exhausted)}): {', '.join(names)}")

    err_path = "run_err_cloud.txt"
    if os.path.exists(err_path):
        err_text = open(err_path, encoding="utf-8", errors="ignore").read().strip()
        if err_text:
            lines.append(f"\nERRORI run_err_cloud.txt:\n{err_text[-400:]}")

    if all_complete:
        lines.append("\nELABORAZIONE COMPLETATA! Tutti i dataset processati.")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
