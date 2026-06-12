"""
Resoconto completo: stato dataset per modello + check chiavi API.
"""
import asyncio
import json
import os
import sys
import pandas as pd
from ollama import AsyncClient as AsyncOllamaClient

sys.stdout.reconfigure(encoding='utf-8')

CHECKPOINT_DIR = "checkpoints"
KEY_CHECK_MODEL = "gpt-oss:20b"

# Dataset attesi e numero pazienti
DATASETS = {
    "MC_gill_DEEPSEEK":                              500,
    "McGill_Pain_Questionnaire_GPT_Con_Dolore":      500,
    "McGill_Pain_Questionnaire_CLAUDE_Con_Dolore":   500,
    "McGill_Pain_Questionnaire_DOCTORAI":            500,
}

CLOUD_MODELS = [
    "Cloud/gpt-oss:20b",
    "Cloud/gpt-oss:120b",
    "Cloud/gemma4:31b",
    "Cloud/gemma3:27b",
    "Cloud/minimax-m2",
    "Cloud/glm-4.7",
    "Cloud/ministral-3:14b",
    "Cloud/gemma3:4b",
    "Cloud/gemma3:12b",
    "Cloud/rnj-1:8b",
]

# ─────────────────────────────────────────────
# DATASET REPORT
# ─────────────────────────────────────────────

def load_best_checkpoint(dataset_name: str, model_label: str) -> pd.DataFrame | None:
    """
    Per un dato dataset+modello cerca prima il checkpoint per-modello,
    poi quello combinato _cloud. Restituisce il DataFrame filtrato (solo quel modello).
    """
    model_safe = model_label.replace("Cloud/", "").replace(":", "_").replace("/", "_")
    per_model = os.path.join(CHECKPOINT_DIR, f"checkpoint_{dataset_name}_{model_safe}.csv")
    combined  = os.path.join(CHECKPOINT_DIR, f"checkpoint_{dataset_name}_cloud.csv")

    for path in [per_model, combined]:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                sub = df[df["model"] == model_label]
                if not sub.empty:
                    return sub.reset_index(drop=True)
            except Exception:
                pass
    return None


def analyze_model(dataset_name: str, model_label: str, total_patients: int) -> dict:
    df = load_best_checkpoint(dataset_name, model_label)
    if df is None or df.empty:
        return {"done": 0, "valid": 0, "null": 0, "missing": total_patients, "pct": 0}

    pred_col = "pi_zero_shot_pred"
    if pred_col in df.columns:
        valid = int(df[pred_col].notna().sum())
        null  = int(df[pred_col].isna().sum())
    else:
        valid = len(df)
        null  = 0

    missing = total_patients - valid
    pct     = round(100 * valid / total_patients)
    return {"done": len(df), "valid": valid, "null": null, "missing": missing, "pct": pct}


def print_dataset_report():
    print("\n" + "="*80)
    print("  RESOCONTO DATASET")
    print("="*80)

    grand_total = 0
    grand_valid = 0
    grand_missing = 0

    for dataset, total in DATASETS.items():
        print(f"\n{'─'*80}")
        print(f"  DATASET: {dataset}  ({total} pazienti attesi)")
        print(f"{'─'*80}")
        print(f"  {'Modello':<30} {'Valide':>7} {'NULL':>6} {'Mancanti':>9} {'%':>5}")
        print(f"  {'─'*28} {'─'*7} {'─'*6} {'─'*9} {'─'*5}")

        ds_valid   = 0
        ds_missing = 0

        for model in CLOUD_MODELS:
            r = analyze_model(dataset, model, total)
            flag = "✓" if r["pct"] == 100 else ("~" if r["pct"] >= 50 else "✗")
            mname = model.replace("Cloud/", "")
            print(f"  {flag} {mname:<28} {r['valid']:>7} {r['null']:>6} {r['missing']:>9} {r['pct']:>4}%")
            ds_valid   += r["valid"]
            ds_missing += r["missing"]

        ds_max = total * len(CLOUD_MODELS)
        print(f"  {'─'*28} {'─'*7} {'─'*6} {'─'*9} {'─'*5}")
        print(f"  {'TOTALE DATASET':<30} {ds_valid:>7}       {ds_missing:>9}  {round(100*ds_valid/ds_max):>3}%")

        grand_total   += ds_max
        grand_valid   += ds_valid
        grand_missing += ds_missing

    print(f"\n{'='*80}")
    print(f"  GRAND TOTAL:  {grand_valid}/{grand_total} predizioni valide  ({round(100*grand_valid/grand_total)}%)")
    print(f"  MANCANTI:     {grand_missing} predizioni da completare")
    print(f"{'='*80}")


# ─────────────────────────────────────────────
# KEY CHECK
# ─────────────────────────────────────────────

async def test_key(entry: dict) -> dict:
    name = entry["name"]
    key  = entry["key"]
    client = AsyncOllamaClient(
        host="https://ollama.com",
        headers={"Authorization": "Bearer " + key}
    )
    try:
        r = await asyncio.wait_for(
            client.chat(
                model=KEY_CHECK_MODEL,
                messages=[{"role": "user", "content": "Say OK"}],
                options={"num_predict": 5}
            ),
            timeout=25
        )
        txt = r.message.content.strip()[:30].encode("ascii", "replace").decode()
        return {"name": name, "status": "OK", "detail": txt}
    except Exception as e:
        msg = str(e)
        lo  = msg.lower()
        if "weekly" in lo:   status = "WEEKLY_LIMIT"
        elif "daily" in lo:  status = "DAILY_LIMIT"
        elif "session" in lo: status = "SESSION_LIMIT"
        elif "subscription" in lo: status = "SUBSCRIPTION"
        elif "401" in msg:   status = "INVALID (401)"
        elif "403" in msg:   status = "FORBIDDEN (403)"
        elif "404" in msg or "not found" in lo: status = "NOT_FOUND (404)"
        elif "timeout" in lo or "timed out" in lo: status = "TIMEOUT"
        else:                status = f"ERRORE"
        return {"name": name, "status": status, "detail": msg[:100]}


async def print_key_report():
    with open("ChaviOllama.json") as f:
        keys = json.load(f)["CLOUD_API_KEYS"]

    print(f"\n{'='*80}")
    print(f"  CHECK CHIAVI API  ({len(keys)} chiavi, modello: {KEY_CHECK_MODEL})")
    print(f"{'='*80}")
    print("  Verifica in corso (parallela)...\n")

    results = await asyncio.gather(*[test_key(e) for e in keys])

    icons = {
        "OK":             "[  OK  ]",
        "WEEKLY_LIMIT":   "[WEEK  ]",
        "DAILY_LIMIT":    "[DAY   ]",
        "SESSION_LIMIT":  "[SESS  ]",
        "SUBSCRIPTION":   "[SUB   ]",
        "INVALID (401)":  "[ 401  ]",
        "FORBIDDEN (403)":"[ 403  ]",
        "NOT_FOUND (404)":"[ 404  ]",
        "TIMEOUT":        "[TIMOUT]",
    }

    for r in results:
        icon = icons.get(r["status"], "[ ERR  ]")
        print(f"  {icon} {r['name']:<20}  {r['status']}")
        if r["status"] not in ("OK", "WEEKLY_LIMIT", "DAILY_LIMIT", "SESSION_LIMIT"):
            detail = r["detail"].encode("ascii", "replace").decode()
            print(f"           └─ {detail[:70]}")

    ok      = [r for r in results if r["status"] == "OK"]
    weekly  = [r for r in results if r["status"] == "WEEKLY_LIMIT"]
    daily   = [r for r in results if r["status"] == "DAILY_LIMIT"]
    session = [r for r in results if r["status"] == "SESSION_LIMIT"]
    broken  = [r for r in results if r["status"] not in
               ("OK", "WEEKLY_LIMIT", "DAILY_LIMIT", "SESSION_LIMIT")]

    print(f"\n{'─'*80}")
    print(f"  USABILI ORA    ({len(ok):2}): {[r['name'] for r in ok]}")
    print(f"  WEEKLY LIMIT   ({len(weekly):2}): {[r['name'] for r in weekly]}")
    print(f"  DAILY LIMIT    ({len(daily):2}): {[r['name'] for r in daily]}")
    print(f"  SESSION LIMIT  ({len(session):2}): {[r['name'] for r in session]}")
    print(f"  NON VALIDE     ({len(broken):2}): {[r['name'] for r in broken]}")
    print(f"{'='*80}\n")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

async def main():
    print_dataset_report()
    await print_key_report()

asyncio.run(main())
