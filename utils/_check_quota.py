"""
Ispeziona gli header HTTP di risposta di Ollama Cloud per ogni chiave,
cercando informazioni su quota/token rimanenti.
"""
import asyncio
import json
import sys
import httpx

sys.stdout.reconfigure(encoding="utf-8")

MODEL = "rnj-1:8b"
BASE_URL = "https://ollama.com"

with open("ChaviOllama.json") as f:
    ALL_KEYS = json.load(f)["CLOUD_API_KEYS"]

# Testa solo le chiavi nello script (esclude le 401 note)
INVALID = {"key4", "key5", "key22", "key26"}
KEYS = [k for k in ALL_KEYS if k["name"] not in INVALID]


async def check_key(entry: dict, semaphore: asyncio.Semaphore) -> dict:
    name = entry["name"]
    key  = entry["key"]
    async with semaphore:
        try:
            async with httpx.AsyncClient(
                base_url=BASE_URL,
                headers={"Authorization": f"Bearer {key}"},
                timeout=25
            ) as client:
                resp = await client.post(
                    "/api/chat",
                    json={
                        "model": MODEL,
                        "messages": [{"role": "user", "content": "Say OK"}],
                        "stream": False,
                        "options": {"num_predict": 5}
                    }
                )
                status = resp.status_code
                headers = dict(resp.headers)
                try:
                    body = resp.json()
                    content = body.get("message", {}).get("content", "")[:30]
                except Exception:
                    content = resp.text[:60]

                return {
                    "name": name,
                    "http_status": status,
                    "content": content,
                    "headers": headers,
                }
        except Exception as e:
            return {"name": name, "http_status": -1, "content": "", "headers": {},
                    "error": str(e)[:100]}


async def main():
    semaphore = asyncio.Semaphore(5)  # max 5 richieste contemporanee
    tasks = [check_key(k, semaphore) for k in KEYS]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Raccogli tutti gli header unici trovati nelle risposte OK
    all_header_keys = set()
    for r in results:
        if isinstance(r, dict) and r.get("http_status") == 200:
            all_header_keys.update(r["headers"].keys())

    # Filtra header interessanti (quota, rate, usage, limit, token)
    interesting = [h for h in all_header_keys
                   if any(w in h.lower() for w in
                          ["quota", "rate", "limit", "usage", "token", "credit",
                           "remain", "budget", "allow", "week", "day"])]

    print(f"\n{'='*70}")
    print(f"  HEADER INTERESSANTI trovati nelle risposte (su {len(KEYS)} chiavi)")
    print(f"{'='*70}")
    if interesting:
        print("  Trovati:", interesting)
    else:
        print("  Nessun header quota/rate trovato nelle risposte OK.")

    print(f"\n{'='*70}")
    print(f"  STATO PER CHIAVE")
    print(f"{'='*70}")
    for r in results:
        if isinstance(r, BaseException):
            print(f"  [ERR] ???  {r}")
            continue
        name  = r["name"]
        code  = r.get("http_status", "?")
        err   = r.get("error", "")

        if code == 200:
            # Mostra header quota se presenti
            quota_hdrs = {k: v for k, v in r["headers"].items()
                          if any(w in k.lower() for w in
                                 ["quota", "rate", "limit", "usage", "token",
                                  "credit", "remain", "budget", "week", "day"])}
            status_str = f"OK  resp={repr(r['content'])}"
            if quota_hdrs:
                status_str += f"  | QUOTA HEADERS: {quota_hdrs}"
            print(f"  [ OK ] {name:<22} {status_str}")
        else:
            body_preview = r.get("content", err)[:80].encode("ascii", "replace").decode()
            print(f"  [{code:>4}] {name:<22} {body_preview}")

    # Stampa tutti gli header di una risposta OK come riferimento
    ok_results = [r for r in results
                  if isinstance(r, dict) and r.get("http_status") == 200]
    if ok_results:
        print(f"\n{'='*70}")
        print(f"  TUTTI GLI HEADER di '{ok_results[0]['name']}' (risposta OK)")
        print(f"{'='*70}")
        for k, v in sorted(ok_results[0]["headers"].items()):
            print(f"  {k:<40} {v}")

asyncio.run(main())
