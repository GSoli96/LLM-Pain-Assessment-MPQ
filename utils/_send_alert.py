import requests
TOKEN = "8779083388:AAE9m7dlXSm2ql1kYsqcwcNy34pNYUSYFuo"
CHAT_ID = "139781098"
msg = (
    "<b>MPQ Alert</b> — script fermato alle 17:06 (chiavi esaurite).\n\n"
    "Avanzamento salvato: DEEPSEEK 31%, GPT 29%.\n"
    "Chiavi OK rimaste (10): submittedpaper0, datslabunisa, Ste1, PEPPE_KEY,\n"
    "RAFFY_INF_KEY, RAF_29_KEY, Testgsoli, grazia1, grazia2, grazia3.\n\n"
    "Azione: esegui run_cloud.bat per riprendere dal checkpoint.\n\n"
    "<i>Monitoraggio attivato: check ogni 30 min + report Telegram ogni ora.</i>"
)
r = requests.post(
    f"https://api.telegram.org/bot{TOKEN}/sendMessage",
    json={"chat_id": CHAT_ID, "text": msg, "parse_mode": "HTML"},
    timeout=10,
)
print("Telegram OK" if r.ok else f"Errore: {r.status_code} {r.text[:100]}")
