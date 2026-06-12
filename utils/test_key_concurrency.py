"""
Test progressivo concorrenza chiave PRO Ollama.
Misura: successo, tempo risposta, comportamento oltre soglia.
"""
import asyncio, time, sys
sys.stdout.reconfigure(encoding="utf-8")
from ollama import AsyncClient as AsyncOllamaClient

API_KEY = "b950b339fadc406ab35e44346c77bc62.-dIhpg7R1aP802xXHl0dOWtT"  # giando_pagamento (PRO)
MODEL   = "gemma3:4b"
PROMPT  = "Respond with only 0 or 1: is 4 an even number?"

def get_client():
    return AsyncOllamaClient(
        host="https://ollama.com",
        headers={"Authorization": f"Bearer {API_KEY}"}
    )

async def single_call(client, call_id: int) -> dict:
    t0 = time.perf_counter()
    try:
        resp = await client.chat(
            model=MODEL,
            messages=[{"role": "user", "content": PROMPT}],
            stream=False,
            options={"temperature": 0.0, "num_predict": 5}
        )
        elapsed = time.perf_counter() - t0
        return {"id": call_id, "ok": True, "elapsed": elapsed,
                "resp": resp.message.content.strip()[:20]}
    except Exception as e:
        elapsed = time.perf_counter() - t0
        err = str(e)
        # Estrai HTTP status se presente
        status = "ERR"
        for code in ["401", "429", "403", "500", "503"]:
            if code in err:
                status = f"HTTP{code}"
                break
        return {"id": call_id, "ok": False, "elapsed": elapsed,
                "resp": status, "err": err[:80]}

async def test_concurrent(n: int) -> dict:
    client = get_client()
    t0 = time.perf_counter()
    results = await asyncio.gather(*[single_call(client, i) for i in range(n)])
    total = time.perf_counter() - t0
    ok_list = [r for r in results if r["ok"]]
    return {"n": n, "ok": len(ok_list), "total": total,
            "avg": sum(r["elapsed"] for r in ok_list) / len(ok_list) if ok_list else 0,
            "details": results}

async def test_sequential(n: int, delay_s: float) -> dict:
    client = get_client()
    t0 = time.perf_counter()
    results = []
    for i in range(n):
        r = await single_call(client, i)
        results.append(r)
        if i < n - 1:
            await asyncio.sleep(delay_s)
    total = time.perf_counter() - t0
    ok_list = [r for r in results if r["ok"]]
    return {"n": n, "delay": delay_s, "ok": len(ok_list), "total": total,
            "avg": sum(r["elapsed"] for r in ok_list) / len(ok_list) if ok_list else 0,
            "details": results}

def print_result(label: str, res: dict):
    print(f"\n{'─'*55}")
    print(f"  {label}")
    print(f"  OK: {res['ok']}/{res['n']} | "
          f"Tempo totale: {res['total']:.1f}s | "
          f"Avg/call: {res['avg']:.1f}s")
    for r in res["details"]:
        icon = "✅" if r["ok"] else "❌"
        print(f"    [{r['id']}] {icon} {r['elapsed']:.2f}s  →  {r['resp']}")

async def main():
    print(f"{'='*55}")
    print(f"  TEST CONCORRENZA — chiave PRO")
    print(f"  Modello: {MODEL}   Prompt brevissimo")
    print(f"  Piano PRO: max 3 modelli concorrenti (fonte: ollama.com/pricing)")
    print(f"{'='*55}")

    summary = []

    # ── 1. Test concorrenza progressiva ──────────────────
    print("\n[1/2] TEST CHIAMATE SIMULTANEE (1→6)")
    for n in range(1, 7):
        print(f"\n  >>> {n} chiamate simultanee...")
        res = await test_concurrent(n)
        print_result(f"Concurrent x{n}", res)
        summary.append((f"concurrent_{n}", res["ok"], res["n"], res["total"], res["avg"]))
        await asyncio.sleep(4)  # pausa tra batch

    # ── 2. Test sequenziale con delay variabile ────────────
    print("\n[2/2] TEST SEQUENZIALE CON DELAY (5 chiamate)")
    for delay in [0.0, 0.2, 0.5, 1.0]:
        label = f"no delay" if delay == 0 else f"{int(delay*1000)}ms delay"
        print(f"\n  >>> 5 chiamate sequenziali, {label}...")
        res = await test_sequential(5, delay)
        print_result(f"Sequential x5 ({label})", res)
        summary.append((f"seq_{label}", res["ok"], res["n"], res["total"], res["avg"]))
        await asyncio.sleep(4)

    # ── Riepilogo finale ──────────────────────────────────
    print(f"\n\n{'='*55}")
    print("  RIEPILOGO FINALE")
    print(f"{'='*55}")
    print(f"  {'Test':<28} {'OK':>4} {'Tot(s)':>7} {'Avg(s)':>7}")
    print(f"  {'─'*50}")
    for label, ok, n, tot, avg in summary:
        flag = " ⚠️" if ok < n else ""
        print(f"  {label:<28} {ok}/{n:>1}  {tot:>6.1f}s  {avg:>6.1f}s{flag}")

    # ── Raccomandazione ───────────────────────────────────
    print(f"\n{'='*55}")
    print("  RACCOMANDAZIONE:")
    best = max((s for s in summary if s[1] == s[2] and "concurrent" in s[0]),
               key=lambda s: int(s[0].split("_")[1]), default=None)
    if best:
        max_ok_conc = int(best[0].split("_")[1])
        print(f"  → Max chiamate simultanee senza errori: {max_ok_conc}")
        print(f"  → Tempo medio/chiamata a quella concorrenza: {best[4]:.1f}s")
        seq_no_delay = next((s for s in summary if s[0] == "seq_no delay"), None)
        if seq_no_delay and seq_no_delay[4] > 0:
            throughput_conc = max_ok_conc / best[4]
            throughput_seq  = 1 / seq_no_delay[4]
            print(f"  → Throughput concurrent: {throughput_conc:.2f} call/s")
            print(f"  → Throughput sequenziale: {throughput_seq:.2f} call/s")
            print(f"  → Guadagno concorrenza: {throughput_conc/throughput_seq:.1f}x")
    print(f"{'='*55}")

asyncio.run(main())
