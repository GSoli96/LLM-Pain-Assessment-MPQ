"""
Invia un report di avanzamento su Telegram.
Lanciare ogni ora: python monitor_once.py
Oppure schedulare con Task Scheduler di Windows.
"""
import pandas as pd, os, requests, json, datetime, sys

sys.stdout.reconfigure(encoding="utf-8")

TOKEN   = "8779083388:AAE9m7dlXSm2ql1kYsqcwcNy34pNYUSYFuo"
CHAT_ID = "139781098"

CLOUD_MODELS = [
    "gpt-oss:120b", "gemma4:31b", "gemma3:27b", "ministral-3:14b",
    "gemma3:4b", "gemma3:12b", "rnj-1:8b", "nemotron-3-super:latest",
    "ministral-3:8b", "devstral-small-2:24b",
]
DATASETS = {
    "MC_gill_DEEPSEEK":                        500,
    "McGill_Pain_Questionnaire_GPT_Con_Dolore": 500,
    "McGill_Pain_Questionnaire_CLAUDE":         500,
    "McGill_Pain_Questionnaire_DOCTORAI":       498,
}


def send_telegram(msg: str):
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TOKEN}/sendMessage",
            json={"chat_id": CHAT_ID, "text": msg, "parse_mode": "HTML"},
            timeout=10,
        )
        if not r.ok:
            print(f"Telegram error: {r.status_code} {r.text[:100]}")
    except Exception as e:
        print(f"Telegram non raggiungibile: {e}")


def main():
    now = datetime.datetime.now().strftime("%H:%M %d/%m/%Y")
    lines = [f"<b>MPQ Monitor</b> — {now}\n"]

    grand_done, grand_total = 0, 0
    all_complete = True

    for ds, total in DATASETS.items():
        fname = f"checkpoints/checkpoint_{ds}_cloud.csv"
        if not os.path.exists(fname):
            lines.append(f"📭 <b>{ds[:30]}</b>: non iniziato")
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
        icon = "✅" if done >= exp else "🔄"
        lines.append(f"{icon} <b>{ds[:30]}</b>: {done}/{exp} ({pct:.0f}%)")

    lines.append(
        f"\n<b>TOTALE: {grand_done}/{grand_total} "
        f"({grand_done / grand_total * 100:.0f}%)</b>"
    )

    # Chiavi esaurite
    if os.path.exists("key_state.json"):
        with open("key_state.json", encoding="utf-8") as f:
            ks = json.load(f)
        exhausted = ks.get("exhausted", {})
        if exhausted:
            names = [v["name"] for v in exhausted.values()]
            lines.append(f"\n🔑 Chiavi esaurite ({len(exhausted)}): {', '.join(names)}")
            all_keys_gone = len(exhausted) >= len(CLOUD_MODELS)  # proxy
            if len(exhausted) >= 25:  # soglia: quasi tutte le 29 chiavi
                lines.append("🔴 <b>ATTENZIONE: quasi tutte le chiavi esaurite!</b>")

    # Errori nel run_err
    err_path = "run_err_cloud.txt"
    if os.path.exists(err_path):
        err_text = open(err_path, encoding="utf-8", errors="ignore").read().strip()
        if err_text:
            lines.append(f"\n⚠️ <b>ERRORI run_err_cloud.txt:</b>\n<code>{err_text[-400:]}</code>")

    if all_complete:
        lines.append("\n🎉 <b>ELABORAZIONE COMPLETATA!</b> Tutti i dataset processati.")

    msg = "\n".join(lines)
    send_telegram(msg)
    print(msg)


if __name__ == "__main__":
    main()
