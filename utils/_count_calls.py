import pandas as pd, os, sys
sys.stdout.reconfigure(encoding="utf-8")

total_calls = 0
for fname in sorted(os.listdir("checkpoints")):
    if "_cloud.csv" not in fname:
        continue
    df = pd.read_csv(f"checkpoints/{fname}", on_bad_lines="skip")
    pred_cols = [c for c in df.columns if c.endswith("_pred")]
    calls = int(df[pred_cols].notna().sum().sum())
    total_calls += calls
    label = fname.replace("checkpoint_", "").replace("_cloud.csv", "")
    print(f"  {label:<45} {calls:>6} chiamate OK")

print(f"\nTOTALE chiamate API riuscite (checkpoint): {total_calls}")
print(f"Chiavi totali usate nel run completo:       ~27")
print(f"Media per chiave (run completo):            ~{total_calls//27}")
print()
print("--- Stima threshold 401 ---")
print(f"Le grazia erano le ultime 3 chiavi attive prima del WEEKLY degli altri.")
print(f"Hanno ricevuto circa {total_calls//3} chiamate totali (se le altre erano gia' esaurite)")
print(f"In pratica: ultime ore con 3 chiavi = ~{total_calls//10} chiamate ciascuna nel picco")
