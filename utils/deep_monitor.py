"""
deep_monitor.py — Controllo qualita' predizioni per modello/dataset + report Telegram.

  python deep_monitor.py --check    → Telegram solo se invalidi o script bloccato
  python deep_monitor.py --report   → Telegram completo (dedup: max 1 ogni 45 min)
  python deep_monitor.py            → equivalente a --check
"""
import sys, os, argparse, datetime
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import requests

BASE_DIR = Path(__file__).parent
os.chdir(BASE_DIR)

TOKEN   = "8779083388:AAE9m7dlXSm2ql1kYsqcwcNy34pNYUSYFuo"
CHAT_ID = "139781098"
CHECKPOINT_DIR = "checkpoints"
LAST_REPORT_FILE = "last_report_sent.txt"

DATASETS = {
    "MC_gill_DEEPSEEK":                         500,
    "McGill_Pain_Questionnaire_GPT_Con_Dolore":  268,
    "McGill_Pain_Questionnaire_CLAUDE":          282,
    "McGill_Pain_Questionnaire_DOCTORAI":        290,
}

SHORT = {
    "MC_gill_DEEPSEEK":                        "DEEPSEEK",
    "McGill_Pain_Questionnaire_GPT_Con_Dolore": "GPT",
    "McGill_Pain_Questionnaire_CLAUDE":         "CLAUDE",
    "McGill_Pain_Questionnaire_DOCTORAI":       "DOCTORAI",
}

PI_COLS = [
    "pi_zero_shot_pred", "pi_cot_simple_pred", "pi_cot_hierarchical_pred",
    "pi_cot_verification_pred", "pi_cot_ensemble_pred", "pi_cot_decision_tree_pred",
    "pi_cot_confidence_pred", "pi_few_shot_pred",
]
ETIO_COLS = [
    "etio_zero_shot_pred", "etio_cot_simple_pred", "etio_cot_verification_pred",
    "etio_few_shot_pred", "etio_knowledge_guided_pred",
]

PI_VALID   = {0.0, 1.0, 2.0}
ETIO_VALID = {0.0, 1.0}

MODEL_ABBREV = {
    "devstral-small-2:24b":    "devstral",
    "gemma3:12b":              "g3:12b",
    "gemma3:27b":              "g3:27b",
    "gemma3:4b":               "g3:4b",
    "gemma4:31b":              "g4:31b",
    "glm-4.7":                 "glm-4.7",
    "gpt-oss:120b":            "gpt:120b",
    "gpt-oss:20b":             "gpt:20b",
    "minimax-m2":              "minimax",
    "ministral-3:14b":         "min3:14b",
    "ministral-3:3b":          "min3:3b",
    "ministral-3:8b":          "min3:8b",
    "nemotron-3-nano:30b":     "nemo:30b",
    "nemotron-3-super:latest": "nemosup",
    "rnj-1:8b":                "rnj:8b",
}


def send_telegram(text: str):
    chunks = [text[i:i+4000] for i in range(0, len(text), 4000)]
    for chunk in chunks:
        try:
            r = requests.post(
                f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                json={"chat_id": CHAT_ID, "text": chunk, "parse_mode": "HTML"},
                timeout=15,
            )
            if not r.ok:
                print(f"Telegram error: {r.status_code} {r.text[:120]}")
        except Exception as e:
            print(f"Telegram non raggiungibile: {e}")


def report_already_sent_recently(minutes: int = 45) -> bool:
    if not os.path.exists(LAST_REPORT_FILE):
        return False
    try:
        age_min = (datetime.datetime.now().timestamp() - os.path.getmtime(LAST_REPORT_FILE)) / 60
        return age_min < minutes
    except Exception:
        return False


def mark_report_sent():
    Path(LAST_REPORT_FILE).write_text(datetime.datetime.now().isoformat())


def strat_char(null_count: int, n_rows: int) -> str:
    if n_rows == 0: return "-"
    done = n_rows - null_count
    pct = done / n_rows
    if pct >= 1.0: return "✓"
    if pct == 0.0: return "□"
    digit = max(1, int(pct * 10))  # 1..9
    return str(digit)


def shorten_model(m: str) -> str:
    return m.replace("Cloud/", "")


def analyze_dataset(dataset_name: str) -> dict | None:
    path = os.path.join(CHECKPOINT_DIR, f"checkpoint_{dataset_name}_cloud.csv")
    if not os.path.exists(path):
        return None

    df = pd.read_csv(path, on_bad_lines="skip")
    present_pi   = [c for c in PI_COLS   if c in df.columns]
    present_etio = [c for c in ETIO_COLS if c in df.columns]
    all_pred = present_pi + present_etio

    results = {}
    for model, grp in df.groupby("model"):
        n = len(grp)
        complete = int(grp[all_pred].notna().all(axis=1).sum()) if all_pred else 0
        null_rows = n - complete

        invalid = 0
        for c in present_pi:
            col = grp[c].dropna()
            invalid += int((~col.isin(PI_VALID)).sum())
        for c in present_etio:
            col = grp[c].dropna()
            invalid += int((~col.isin(ETIO_VALID)).sum())

        pi_nulls   = {c: int(grp[c].isna().sum()) for c in PI_COLS}
        etio_nulls = {c: int(grp[c].isna().sum()) if c in grp.columns else n for c in ETIO_COLS}

        results[shorten_model(model)] = {
            "rows":      n,
            "complete":  complete,
            "null_rows": null_rows,
            "invalid":   invalid,
            "pi_nulls":  pi_nulls,
            "etio_nulls": etio_nulls,
        }
    return results


def build_report() -> str:
    now = datetime.datetime.now().strftime("%H:%M %d/%m/%Y")
    lines = [
        f"📊 <b>MPQ Report</b> — {now}",
        "<code>Legenda: ✓=100% 9=90% .. 1=10% □=0%</code>",
        "<code>pi : zs cs ch cv ce cd cc fs</code>",
        "<code>etio: zs cs cv fs kg</code>",
        "",
    ]

    grand_done = grand_tot = 0

    for ds, n_paz in DATASETS.items():
        short = SHORT[ds]
        stats = analyze_dataset(ds)

        if stats is None:
            lines.append(f"<b>{short}</b>: 📭 non iniziato\n")
            grand_tot += n_paz * 15
            continue

        ds_done = sum(s["complete"] for s in stats.values())
        ds_tot  = n_paz * len(stats)
        pct_ds  = ds_done / ds_tot * 100 if ds_tot else 0
        grand_done += ds_done
        grand_tot  += ds_tot

        ds_inv = sum(s["invalid"] for s in stats.values())
        inv_tag = f" ❌{ds_inv}inv" if ds_inv else ""
        lines.append(f"<b>━ {short} ({ds_done}/{ds_tot} — {pct_ds:.0f}%){inv_tag} ━</b>")
        lines.append("<code>Modello        % | pi[12345678] etio[12345]</code>")

        for model in sorted(stats.keys()):
            s = stats[model]
            pct = s["complete"] / n_paz * 100 if n_paz else 0
            pi_str   = "".join(strat_char(s["pi_nulls"].get(c, s["rows"]),   s["rows"]) for c in PI_COLS)
            etio_str = "".join(strat_char(s["etio_nulls"].get(c, s["rows"]), s["rows"]) for c in ETIO_COLS)
            name = MODEL_ABBREV.get(model, model[:9])
            inv_flag = " ❌" if s["invalid"] > 0 else ""
            lines.append(f"<code>{name:<10} {pct:3.0f}% | [{pi_str}] [{etio_str}]{inv_flag}</code>")

        lines.append("")

    grand_pct = grand_done / grand_tot * 100 if grand_tot else 0
    lines.append(f"<b>TOTALE: {grand_done}/{grand_tot} ({grand_pct:.0f}%)</b>")

    return "\n".join(lines)


def build_quick_check() -> tuple[str, bool]:
    now = datetime.datetime.now().strftime("%H:%M %d/%m/%Y")
    log_stale, log_msg = check_log_stale()
    total_invalid = 0
    ds_lines = []

    for ds, n_paz in DATASETS.items():
        short = SHORT[ds]
        stats = analyze_dataset(ds)
        if stats is None:
            ds_lines.append(f"  {short}: 📭 non iniziato")
            continue
        ds_done = sum(s["complete"] for s in stats.values())
        ds_tot  = n_paz * len(stats)
        ds_inv  = sum(s["invalid"]  for s in stats.values())
        ds_null = sum(s["null_rows"] for s in stats.values())
        pct = ds_done / ds_tot * 100 if ds_tot else 0
        total_invalid += ds_inv
        inv_tag  = f" ❌{ds_inv}inv" if ds_inv else ""
        null_tag = f" ⏳{ds_null}null" if ds_null else ""
        ds_lines.append(f"  {short}: {ds_done}/{ds_tot} ({pct:.0f}%){inv_tag}{null_tag}")

    has_problem = total_invalid > 0 or log_stale
    lines = [f"🔍 <b>MPQ Check</b> — {now}\n"]
    lines += ds_lines
    lines.append("")
    if log_stale:
        lines.append(f"🔴 <b>SCRIPT BLOCCATO</b>: {log_msg}")
    else:
        lines.append(f"✅ Script attivo ({log_msg})")
    if total_invalid > 0:
        lines.append(f"❌ <b>{total_invalid} valori INVALIDI</b> — esegui _repair_parse.py")
    elif not log_stale:
        lines.append("✅ Nessun valore invalido.")
    return "\n".join(lines), has_problem


def check_log_stale(max_minutes: int = 10) -> tuple[bool, str]:
    log_path = "run_log_cloud.txt"
    if not os.path.exists(log_path):
        return True, "run_log_cloud.txt non trovato"
    age = (datetime.datetime.now().timestamp() - os.path.getmtime(log_path)) / 60
    if age > max_minutes:
        return True, f"log non aggiornato da {age:.0f} min"
    return False, f"log aggiornato {age:.1f} min fa"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--check",  action="store_true")
    parser.add_argument("--force",  action="store_true", help="Ignora dedup per --report")
    args = parser.parse_args()

    if args.report:
        if not args.force and report_already_sent_recently(45):
            print("→ Report gia' inviato negli ultimi 45 min — skip (usa --force per ignorare).")
            return
        report_text = build_report()
        print(report_text)
        send_telegram(report_text)
        mark_report_sent()
        print("\n→ Report Telegram inviato.")
    else:
        check_text, has_problem = build_quick_check()
        print(check_text)
        if has_problem:
            send_telegram(check_text)
            print("\n→ Alert Telegram inviato.")
        else:
            print("\n→ Tutto OK. Telegram non inviato.")


if __name__ == "__main__":
    main()
