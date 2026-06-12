"""
Controlla lo stato di tutte le chiavi in ChaviOllama.json.
Uso: python check_keys.py
"""
import asyncio
import json
from ollama import AsyncClient as AsyncOllamaClient

MODEL = "rnj-1:8b"

with open("ChaviOllama.json") as f:
    KEYS = json.load(f)["CLOUD_API_KEYS"]


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
                model=MODEL,
                messages=[{"role": "user", "content": "Say OK"}],
                options={"num_predict": 5}
            ),
            timeout=25
        )
        txt = r.message.content.strip()[:40].encode("ascii", "replace").decode()
        return {"name": name, "key": key, "status": "OK", "detail": txt}
    except Exception as e:
        msg = str(e)
        lo  = msg.lower()
        if "weekly" in lo:
            status = "WEEKLY_LIMIT"
        elif "daily" in lo:
            status = "DAILY_LIMIT"
        elif "session" in lo:
            status = "SESSION_LIMIT"
        elif "subscription" in lo:
            status = "SUBSCRIPTION"
        elif "401" in msg:
            status = "INVALID_KEY (401)"
        elif "403" in msg:
            status = "FORBIDDEN (403)"
        elif "404" in msg or "not found" in lo:
            status = "NOT_FOUND (404)"
        elif "timeout" in lo or "timed out" in lo:
            status = "TIMEOUT"
        else:
            status = "ERRORE"
        return {"name": name, "key": key, "status": status, "detail": msg[:120]}


async def main():
    raw = await asyncio.gather(*[test_key(e) for e in KEYS], return_exceptions=True)
    results = []
    for entry, r in zip(KEYS, raw):
        if isinstance(r, BaseException):
            results.append({"name": entry["name"], "key": entry["key"], "status": "ERRORE", "detail": str(r)[:120]})
        else:
            results.append(r)

    ok      = [r for r in results if r["status"] == "OK"]
    limited = [r for r in results if "LIMIT" in r["status"]]
    broken  = [r for r in results if r["status"] not in ("OK",) and "LIMIT" not in r["status"]]

    icons = {"OK": "[OK]  ", "WEEKLY_LIMIT": "[WEEK]", "DAILY_LIMIT": "[DAY] ",
             "SESSION_LIMIT": "[SESS]", "SUBSCRIPTION": "[SUB] ",
             "INVALID_KEY (401)": "[401] ", "FORBIDDEN (403)": "[403] ",
             "NOT_FOUND (404)": "[404] ", "TIMEOUT": "[TIME]", "ERRORE": "[ERR] "}

    pro_keys = {e["key"] for e in KEYS if e.get("type") == "PRO"}

    print(f"\n{'='*70}")
    print(f"  CHECK CHIAVI OLLAMA  (modello: {MODEL})  —  {len(KEYS)} chiavi totali")
    print(f"{'='*70}")
    for r in results:
        icon = icons.get(r["status"], "[???] ")
        tipo = "[PRO] " if r.get("key") in pro_keys else "[free]"
        print(f"  {icon} {tipo} {r['name']:<22} {r['status']}")
        if r["status"] != "OK":
            print(f"           {r['detail'][:80]}")

    print(f"\n{'='*70}")
    print(f"  FUNZIONANTI  ({len(ok)}):  {[r['name'] for r in ok]}")
    wk = [r for r in limited if 'WEEKLY' in r['status']]
    dy = [r for r in limited if 'DAILY'  in r['status']]
    ss = [r for r in limited if 'SESSION' in r['status']]
    print(f"  WEEKLY LIMIT ({len(wk)}):  {[r['name'] for r in wk]}")
    print(f"  DAILY LIMIT  ({len(dy)}):  {[r['name'] for r in dy]}")
    print(f"  SESSION LIMIT({len(ss)}):  {[r['name'] for r in ss]}")
    print(f"  NON VALIDE   ({len(broken)}):  {[r['name'] for r in broken]}")
    print(f"{'='*70}\n")

asyncio.run(main())
