import json, sys
sys.stdout.reconfigure(encoding="utf-8")
with open("key_state.json", encoding="utf-8") as f:
    d = json.load(f)
weekly, daily = [], []
for v in d["exhausted"].values():
    entry = f"  {v['name']:<22} reset: {v['reset_at'][:16]}"
    if v["reason"] == "WEEKLY_LIMIT":
        weekly.append(entry)
    else:
        daily.append(entry)
print(f"WEEKLY_LIMIT ({len(weekly)}) — reset lunedi 00:00:")
for e in sorted(weekly): print(e)
print(f"\nDAILY_LIMIT ({len(daily)}) — reset domani 00:00:")
for e in sorted(daily): print(e)
print(f"\nChiavi totali esaurite: {len(d['exhausted'])}")
