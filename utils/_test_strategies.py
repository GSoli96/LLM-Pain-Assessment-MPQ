"""
Test sistematico di tutti i modelli cloud × tutte le strategie × 5 pazienti.
Misura: lunghezza risposta, successo parsing, troncatura.
Raccomanda i max_tokens ottimali per ogni strategia.

Uso: python _test_strategies.py
"""

import asyncio, json, math, os, re, sys, unicodedata
import pandas as pd
from ollama import AsyncClient as AsyncOllamaClient

sys.stdout.reconfigure(encoding='utf-8')

# ─── CONFIGURAZIONE TEST ───────────────────────────────────────────────────────
N_PATIENTS   = 5          # pazienti su cui testare ogni coppia modello×strategia
MAX_TOKENS_TEST = 1200    # token generosi per calibrazione (non tronca)
CONCURRENCY  = 8          # chiamate simultanee

# Chiave "fresca" usata per il test — ruota automaticamente su 429
TEST_KEYS = [
    "20bf054fd20046908f013a155458f3b9.qKri9BIrRhRMlIZ0BDihRuyb",  # Testgsoli
    "5920d3664f504232bb33ff74565b4acf.XTW_rpMrXpOErbJhTWwZj8tI",  # giando.soli
    "bf514d38e1334e1fa9c92efef1b281a1.OKMalBy7tWjAExxoNl6m7LVs",  # grazia1
    "0a3b693e62ca4b538c15f82b3ae936a9.nRE1Kvf52oiSMG_PXC-89cz2",  # grazia2
    "0397cbceed6842fcb305ecdd6ad8073b.6RB9D4qTzA7LRm7TVNRyJG_t",  # grazia3
]
KEY_IDX = 0

CLOUD_MODELS = [
    "gpt-oss:120b",
    "gemma4:31b",
    "gemma3:27b",
    "ministral-3:14b",
    "gemma3:4b",
    "gemma3:12b",
    "rnj-1:8b",
    "nemotron-3-super:latest",
    "ministral-3:8b",
    "devstral-small-2:24b",      # sostituto glm-4.7
]

DATASET_PATH = "experiments/data/MC_gill_DEEPSEEK.csv"

SCORE_COLS = [
    'sensory_score', 'fear_score', 'sensory_misc_score', 'brightness_score',
    'affective_score', 'thermal_score', 'traction_score', 'tension_score',
    'evaluative_score', 'dullness_score', 'autonomic_score', 'sensory_3_score',
    'spatial_score', 'affective_evaluative_sensory_score', 'temporal_score',
    'punishment_score', 'punctate_score', 'sensory_2_score', 'incisive_score',
    'constrictive_score',
]
DESCRIPTOR_COLS = [
    'sensory', 'fear', 'sensory_misc', 'brightness', 'affective', 'thermal',
    'traction', 'tension', 'evaluative', 'dullness', 'autonomic', 'sensory_3',
    'spatial', 'affective_evaluative_sensory', 'temporal', 'punishment',
    'punctate', 'sensory_2', 'incisive', 'constrictive',
]

# Alias colonne CSV → nomi canonici usati dallo script
_COL_ALIASES = {
    'traction_pressure_score':    'traction_score',
    'punctate_pressure_score':    'punctate_score',
    'incisive_pressure_score':    'incisive_score',
    'constrictive_pressure_score':'constrictive_score',
    'sensory_miscellaneous_score':'sensory_misc_score',
    'affective_eval_sensory_score':'affective_evaluative_sensory_score',
    'sensory2_score':             'sensory_2_score',
    'sensory3_score':             'sensory_3_score',
    'traction_pressure':    'traction',
    'punctate_pressure':    'punctate',
    'incisive_pressure':    'incisive',
    'constrictive_pressure':'constrictive',
    'sensory_miscellaneous':'sensory_misc',
    'affective_eval_sensory':'affective_evaluative_sensory',
    'sensory2':             'sensory_2',
    'sensory3':             'sensory_3',
}

def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    col_lower = {c.lower(): c for c in df.columns}
    rename = {}
    for alias, canon in _COL_ALIASES.items():
        actual = col_lower.get(alias)
        if actual:
            rename[actual] = canon
    df = df.rename(columns=rename)
    if 'total_pain_score' not in df.columns:
        present = [c for c in SCORE_COLS if c in df.columns]
        if present:
            df['total_pain_score'] = df[present].sum(axis=1)
    return df
WILKIE_COEFFICIENTS = {
    'shooting': 3.382, 'tingling': 3.881, 'crushing': 0.540, 'blinding': 0.729,
    'tight': 0.171, 'cold': 0.095, 'itchy': -0.721, 'pressing': -0.827,
    'smarting': -0.777, 'flashing': -1.777, 'jumping': -1.605,
    'freezing': -0.658, 'tearing': -0.655,
}
WILKIE_INTERCEPT = 0.815

PAIN_INTENSITY_SYSTEM = (
    "Sei un esperto nella valutazione del dolore con il McGill Pain Questionnaire (MPQ). "
    "Il tuo compito è classificare l'intensità del dolore di un paziente in base al "
    "Pain Rating Index (PRI), calcolato come somma degli score delle 20 sottoscale MPQ. "
    "Le classi sono:\n"
    "  - Classe 0: PRI < 26 (dolore lieve)\n"
    "  - Classe 1: 26 ≤ PRI ≤ 45 (dolore moderato)\n"
    "  - Classe 2: PRI ≥ 46 (dolore severo)\n"
    "Rispondi SEMPRE con 'Classe 0', 'Classe 1' o 'Classe 2' alla fine della tua risposta."
)
ETIOLOGY_SYSTEM = (
    "Sei un neurologo esperto nella diagnosi differenziale del dolore oncologico. "
    "Il tuo compito è classificare l'eziologia del dolore di un paziente come:\n"
    "  - Classe 0: Nocicettivo (attivazione di afferenze nocicettive in tessuti somatici o viscerali)\n"
    "  - Classe 1: Neuropatico (dolore causato da lesione del sistema nervoso centrale o periferico)\n"
    "Rispondi SEMPRE con 'Classe 0' (nocicettivo) o 'Classe 1' (neuropatico) alla fine della tua risposta."
)


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def build_patient_context(row: pd.Series) -> str:
    lines = [f"Sesso: {row['Sex']}, Età: {row['Age']} anni", "", "Sottoscale MPQ e score:"]
    for col in SCORE_COLS:
        label = col.replace('_score', '').replace('_', ' ').capitalize()
        lines.append(f"  - {label}: {row[col]}")
    lines += ["", "Descrittori verbali scelti dal paziente (uno per sottoscala):"]
    for c in DESCRIPTOR_COLS:
        lines.append(f"  - {c.replace('_', ' ')}: {str(row[c]).lower().strip()}")
    return '\n'.join(lines)


def get_descriptors(row: pd.Series) -> list[str]:
    return [str(row[c]).lower().strip() for c in DESCRIPTOR_COLS]


def build_prompts(row: pd.Series, df: pd.DataFrame, idx: int) -> dict:
    ctx  = build_patient_context(row)
    desc = get_descriptors(row)

    # Few-shot examples (semplici, da pazienti vicini)
    ex_pi   = [{'scores': 'PRI=38', 'pri': 38, 'class': 1},
               {'scores': 'PRI=52', 'pri': 52, 'class': 2}]
    ex_etio = [{'descriptors': ['shooting','tingling','burning'], 'class': 1},
               {'descriptors': ['pressing','aching','heavy'],     'class': 0}]
    ex_pi_txt = "\n\n".join([
        f"ESEMPIO {i+1}:\nScore MPQ: {e['scores']}\nPRI totale: {e['pri']}\n"
        f"Classificazione: Classe {e['class']}"
        for i, e in enumerate(ex_pi)
    ])
    ex_etio_txt = "\n\n".join([
        f"ESEMPIO {i+1}:\nDescrittori: {', '.join(e['descriptors'])}\n"
        f"Classificazione: Classe {e['class']} ({'nocicettivo' if e['class']==0 else 'neuropatico'})"
        for i, e in enumerate(ex_etio)
    ])
    coef_text = '\n'.join([
        f"  {d}: {'presente' if d in desc else 'assente'} (coefficiente: {c:+.3f})"
        for d, c in WILKIE_COEFFICIENTS.items()
    ])

    return {
        # ── PAIN INTENSITY (8 strategie) ──
        'pi_zero_shot': (
            PAIN_INTENSITY_SYSTEM,
            f"Dati del paziente:\n{ctx}\n\nClassifica l'intensità del dolore di questo paziente."
        ),
        'pi_cot_simple': (
            PAIN_INTENSITY_SYSTEM,
            f"Dati del paziente:\n{ctx}\n\n"
            f"Segui questi passi per classificare l'intensità del dolore:\n"
            f"PASSO 1 - Somma tutti i 20 score MPQ e calcola il PRI totale.\n"
            f"PASSO 2 - Confronta il PRI con le soglie: <26=Classe 0, 26-45=Classe 1, >=46=Classe 2.\n"
            f"PASSO 3 - Indica la classe risultante.\n\nMostra il calcolo esplicito."
        ),
        'pi_cot_hierarchical': (
            PAIN_INTENSITY_SYSTEM,
            f"\nDati del paziente:\n{ctx}\n\n"
            f"Esegui una CLASSIFICAZIONE GERARCHICA:\n"
            f"LIVELLO 1 - PRI_SENSORY, PRI_AFFECTIVE, PRI_EVALUATIVE, PRI_MISCELLANEOUS\n"
            f"LIVELLO 2 - Pattern predominante\n"
            f"LIVELLO 3 - PRI TOTALE\n"
            f"LIVELLO 4 - Classificazione finale\n\n"
            f"Mostra i calcoli. Conclude con 'Classe Finale: [0/1/2]'"
        ),
        'pi_cot_verification': (
            PAIN_INTENSITY_SYSTEM,
            f"{ctx}\n\n"
            f"FASE 1 - calcola PRI e classe iniziale\n"
            f"FASE 2 - verifica critica con metodo alternativo (somma per 4 domini)\n"
            f"FASE 3 - classificazione finale\n"
            f"FASE 4 - OUTPUT: Classe Finale: [0/1/2]"
        ),
        'pi_cot_ensemble': (
            PAIN_INTENSITY_SYSTEM,
            f"{ctx}\n\n"
            f"Analizza da TRE PROSPETTIVE:\n"
            f"1. QUANTITATIVA: calcola PRI esatto → RISULTATO P1: Classe [X]\n"
            f"2. QUALITATIVA: analizza descrittori → RISULTATO P2: Classe [Y]\n"
            f"3. CLINICA: pattern dominante → RISULTATO P3: Classe [Z]\n\n"
            f"CLASSIFICAZIONE INTEGRATA: Classe Finale = [consensus]\n"
            f"OUTPUT: Classe [0/1/2]"
        ),
        'pi_cot_decision_tree': (
            PAIN_INTENSITY_SYSTEM,
            f"{ctx}\n\n"
            f"Segui l'ALBERO DECISIONALE:\n"
            f"DOMANDA 1: PRI_total ≥ 46? → Calcola, Verdetto SÌ/NO\n"
            f"  Se SÌ → CLASSE 2\n"
            f"DOMANDA 2: PRI_total ≥ 26? → Se NO → CLASSE 0\n"
            f"DOMANDA 3: sottodominio predominante → CLASSE 1\n\n"
            f"OUTPUT FINALE: Classe: [0/1/2]"
        ),
        'pi_cot_confidence': (
            PAIN_INTENSITY_SYSTEM,
            f"{ctx}\n\n"
            f"STEP 1 - PRI totale\nSTEP 2 - Classe base\nSTEP 3 - Confidenza base (%)\n"
            f"STEP 4 - Penalità (soglia, pattern, varianza)\nSTEP 5 - Confidenza adattata\n"
            f"STEP 6 - OUTPUT FINALE: Classe: [C], Confidenza: [Z]%"
        ),
        'pi_few_shot': (
            PAIN_INTENSITY_SYSTEM,
            f"Esempi di classificazione:\n\n{ex_pi_txt}\n\n---\n\n"
            f"Dati del paziente:\n{ctx}\n\nClassifica l'intensità del dolore."
        ),
        # ── ETIOLOGY (5 strategie) ──
        'etio_zero_shot': (
            ETIOLOGY_SYSTEM,
            f"Il paziente ha scelto i seguenti descrittori verbali:\n"
            f"{', '.join(desc)}\n\nClassifica l'eziologia del dolore."
        ),
        'etio_cot_simple': (
            ETIOLOGY_SYSTEM,
            f"Descrittori: {', '.join(desc)}\n\n"
            f"PASSO 1 - classifica ogni descrittore (neuropatico/nocicettivo/neutro)\n"
            f"PASSO 2 - valuta il pattern complessivo\n"
            f"PASSO 3 - concludi con la classificazione\n\nMostra il ragionamento esplicito."
        ),
        'etio_cot_verification': (
            ETIOLOGY_SYSTEM,
            f"Descrittori: {', '.join(desc)}\n\n"
            f"FASE 1 - classifica ogni descrittore [N]/[NC]/[NEUTRO], conta N vs NC\n"
            f"FASE 2 - verifica pattern specifici (shooting/tingling→neuropatico; pressione/trazione→nocicettivo)\n"
            f"FASE 3 - classificazione finale\n"
            f"OUTPUT: Classe [0/1]"
        ),
        'etio_few_shot': (
            ETIOLOGY_SYSTEM,
            f"Esempi:\n\n{ex_etio_txt}\n\n---\n\n"
            f"Descrittori del paziente: {', '.join(desc)}\n\nClassifica l'eziologia."
        ),
        'etio_knowledge_guided': (
            ETIOLOGY_SYSTEM,
            f"Formula Wilkie et al. (2001): Score = 0.815 + somma(coefficiente × presenza)\n"
            f"  Se Score > 0: Classe 0 (Nocicettivo)\n"
            f"  Se Score ≤ 0: Classe 1 (Neuropatico)\n\n"
            f"Descrittori e coefficienti:\n{coef_text}\n\n"
            f"Calcola il Predictive Score e classifica."
        ),
    }


def parse_class(response: str, task: str) -> int | None:
    response = unicodedata.normalize('NFKC', response)
    r = response.lower()

    if task == 'pi':
        for cls in ['2', '1', '0']:
            if f'classe {cls}' in r or f'class {cls}' in r:
                return int(cls)
            if f'classe finale: {cls}' in r or f'output: classe {cls}' in r:
                return int(cls)
        if 'classe integrata:' in r or 'classificazione finale:' in r:
            for cls in ['2', '1', '0']:
                if re.search(rf'(?:classe integrata|classificazione finale):\s*{cls}', r):
                    return int(cls)
        nums = re.findall(r'\b[012]\b', response)
        if nums:
            return int(nums[-1])

    elif task == 'etio':
        if any(w in r for w in ['neuropath', 'classe 1', 'class 1', 'neuropatico']):
            return 1
        if any(w in r for w in ['nociceptiv', 'classe 0', 'class 0', 'nocicettivo']):
            return 0
        if 'classe:' in r:
            after = r.split('classe:')[1][:10]
            if '1' in after: return 1
            if '0' in after: return 0
        if any(w in r for w in ['neuropatic', 'nervos', 'neural']):
            return 1
        if any(w in r for w in ['nocicet', 'tissut', 'infiamm', 'somatico']):
            return 0
    return None


def is_truncated(response: str, task: str) -> bool:
    """True se la risposta non contiene la conclusione attesa."""
    r = unicodedata.normalize('NFKC', response).lower()
    if task == 'pi':
        return not any(w in r for w in ['classe 0', 'classe 1', 'classe 2',
                                         'class 0', 'class 1', 'class 2',
                                         'classe finale', 'output finale'])
    else:
        return not any(w in r for w in ['classe 0', 'classe 1', 'class 0', 'class 1',
                                         'neuropatico', 'nocicettivo', 'neuropatica',
                                         'nociceptiv', 'neuropath'])


# ─── API CALL ──────────────────────────────────────────────────────────────────

_clients: dict = {}
_exhausted: set = set()

def get_client(key: str) -> AsyncOllamaClient:
    if key not in _clients:
        _clients[key] = AsyncOllamaClient(
            host='https://ollama.com',
            headers={'Authorization': 'Bearer ' + key}
        )
    return _clients[key]


async def call_model(model: str, system: str, user: str, semaphore: asyncio.Semaphore) -> str:
    global KEY_IDX
    active = [k for k in TEST_KEYS if k not in _exhausted]
    if not active:
        return ""

    for attempt in range(4):
        key = active[KEY_IDX % len(active)]
        KEY_IDX += 1
        try:
            async with semaphore:
                resp = await asyncio.wait_for(
                    get_client(key).chat(
                        model=model,
                        messages=[
                            {'role': 'system', 'content': system},
                            {'role': 'user',   'content': user},
                        ],
                        stream=False,
                        options={'temperature': 0.0, 'num_predict': MAX_TOKENS_TEST}
                    ),
                    timeout=60
                )
            return resp.message.content.strip()
        except (Exception, BaseException) as e:
            msg = str(e).lower()
            if 'weekly' in msg or 'daily' in msg:
                _exhausted.add(key)
                active = [k for k in TEST_KEYS if k not in _exhausted]
                if not active:
                    return ""
            elif '429' in msg or 'session' in msg or 'concurrent' in msg:
                await asyncio.sleep(3 + attempt * 2)
            elif '401' in msg or '403' in msg or '404' in msg:
                _exhausted.add(key)
                active = [k for k in TEST_KEYS if k not in _exhausted]
                if not active:
                    return ""
            elif 'cancelled' in msg or 'ssl' in msg or 'timeout' in msg:
                await asyncio.sleep(2 + attempt)
            else:
                await asyncio.sleep(2)
    return ""


# ─── MAIN TEST LOOP ────────────────────────────────────────────────────────────

async def test_model(model: str, patients: list, semaphore: asyncio.Semaphore) -> list[dict]:
    """Testa un modello su tutti i pazienti e strategie."""
    results = []
    tasks_meta = []
    coros = []

    for pid, row in patients:
        prompts = build_prompts(row, None, pid)
        for strat_name, (system, user) in prompts.items():
            task_type = 'pi' if strat_name.startswith('pi_') else 'etio'
            tasks_meta.append((pid, strat_name, task_type))
            coros.append(call_model(model, system, user, semaphore))

    responses = await asyncio.gather(*coros, return_exceptions=True)

    for (pid, strat_name, task_type), resp in zip(tasks_meta, responses):
        if isinstance(resp, BaseException):
            resp = ""
        pred       = parse_class(resp, task_type) if resp else None
        truncated  = is_truncated(resp, task_type) if resp else True
        resp_chars = len(resp)
        # stima token = chars / 3.5 per italiano
        est_tokens = math.ceil(resp_chars / 3.5)
        results.append({
            'model': model, 'patient_id': pid, 'strategy': strat_name,
            'task': task_type, 'resp_chars': resp_chars, 'est_tokens': est_tokens,
            'parsed': pred is not None, 'truncated': truncated,
            'pred': pred, 'response_tail': resp[-120:] if resp else '',
        })
    return results


async def main():
    df = normalize_df(pd.read_csv(DATASET_PATH))
    # Prendi 5 pazienti distribuiti (inizio, metà, fine + 2 random)
    indices = [0, 99, 249, 374, 499]
    patients = [(df.at[i, 'ID'] if 'ID' in df.columns else i, df.iloc[i])
                for i in indices if i < len(df)]

    semaphore = asyncio.Semaphore(CONCURRENCY)

    print(f"\n{'='*72}")
    print(f"  TEST STRATEGIE  —  {len(CLOUD_MODELS)} modelli × 13 strategie × {len(patients)} pazienti")
    print(f"  max_tokens per chiamata: {MAX_TOKENS_TEST}  |  concurrency: {CONCURRENCY}")
    print(f"{'='*72}\n")

    all_results = []
    for i, model in enumerate(CLOUD_MODELS, 1):
        print(f"  [{i:02d}/{len(CLOUD_MODELS)}] {model} ...", end=' ', flush=True)
        res = await test_model(model, patients, semaphore)
        all_results.extend(res)
        parsed = sum(1 for r in res if r['parsed'])
        trunc  = sum(1 for r in res if r['truncated'])
        total  = len(res)
        print(f"parsed={parsed}/{total}  trunc={trunc}/{total}")

    # ── ANALISI ──────────────────────────────────────────────────────────────
    results_df = pd.DataFrame(all_results)
    results_df.to_csv('_test_results.csv', index=False)

    print(f"\n{'='*72}")
    print(f"  RISULTATI PER STRATEGIA (su tutti i modelli)")
    print(f"{'='*72}")
    print(f"  {'Strategia':<30} {'Parse%':>7} {'Trunc%':>7} {'AvgChars':>9} {'P95Chars':>9} {'RecTokens':>10}")
    print(f"  {'-'*30} {'-'*7} {'-'*7} {'-'*9} {'-'*9} {'-'*10}")

    strat_recs = {}
    for strat in results_df['strategy'].unique():
        sub = results_df[results_df['strategy'] == strat]
        sub_resp = sub[sub['resp_chars'] > 0]
        parse_pct  = round(100 * sub['parsed'].sum() / len(sub))
        trunc_pct  = round(100 * sub['truncated'].sum() / len(sub))
        avg_chars  = round(sub_resp['resp_chars'].mean()) if len(sub_resp) else 0
        p95_chars  = round(sub_resp['resp_chars'].quantile(0.95)) if len(sub_resp) else 0
        rec_tokens = max(200, math.ceil(p95_chars / 3.0) + 80)  # /3 per sicurezza + buffer
        strat_recs[strat] = rec_tokens
        flag = ' ⚠' if parse_pct < 90 else ''
        print(f"  {strat:<30} {parse_pct:>6}% {trunc_pct:>6}% {avg_chars:>9} {p95_chars:>9} {rec_tokens:>10}{flag}")

    print(f"\n{'='*72}")
    print(f"  RISULTATI PER MODELLO (parse% su tutte le strategie)")
    print(f"{'='*72}")
    print(f"  {'Modello':<26} {'PI parse%':>10} {'Etio parse%':>12} {'Trunc%':>7} {'AvgChars':>9}")
    print(f"  {'-'*26} {'-'*10} {'-'*12} {'-'*7} {'-'*9}")

    for model in CLOUD_MODELS:
        sub  = results_df[results_df['model'] == model]
        pi   = sub[sub['task'] == 'pi']
        etio = sub[sub['task'] == 'etio']
        pi_pct   = round(100 * pi['parsed'].sum() / len(pi)) if len(pi) else 0
        etio_pct = round(100 * etio['parsed'].sum() / len(etio)) if len(etio) else 0
        trunc    = round(100 * sub['truncated'].sum() / len(sub))
        avg_ch   = round(sub[sub['resp_chars'] > 0]['resp_chars'].mean()) if sub['resp_chars'].sum() > 0 else 0
        flag = ' ⚠' if pi_pct < 80 or etio_pct < 80 else ''
        print(f"  {model:<26} {pi_pct:>9}% {etio_pct:>11}% {trunc:>6}% {avg_ch:>9}{flag}")

    # ── PROBLEMI SPECIFICI ────────────────────────────────────────────────────
    failures = results_df[(~results_df['parsed']) | results_df['truncated']]
    if len(failures) > 0:
        print(f"\n{'='*72}")
        print(f"  RISPOSTE PROBLEMATICHE ({len(failures)} casi)")
        print(f"{'='*72}")
        shown = 0
        for _, row in failures.iterrows():
            if shown >= 20:
                print(f"  ... e altri {len(failures)-20} casi")
                break
            tail = row['response_tail'].encode('ascii','replace').decode()
            trunc_flag = '[TRUNC]' if row['truncated'] else ''
            parse_flag = '[NOPARSE]' if not row['parsed'] else ''
            print(f"  {row['model']:<24} {row['strategy']:<28} chars={row['resp_chars']:>4} {trunc_flag}{parse_flag}")
            if row['resp_chars'] > 0 and (not row['parsed'] or row['truncated']):
                print(f"    tail: ...{tail[-80:]}")
            shown += 1

    # ── RACCOMANDAZIONI max_tokens ────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  RACCOMANDAZIONI max_tokens  (da usare in ask_llm_multiple_CoT.py)")
    print(f"{'='*72}")
    pi_keys   = [k for k in strat_recs if k.startswith('pi_')]
    etio_keys = [k for k in strat_recs if k.startswith('etio_')]
    print(f"\n  _PI_TOKENS = {{")
    for k in pi_keys:
        name = k.replace('pi_', '')
        print(f"      '{name}': {strat_recs[k]},")
    print(f"  }}")
    print(f"\n  _ETIO_TOKENS = {{")
    for k in etio_keys:
        name = k.replace('etio_', '')
        print(f"      '{name}': {strat_recs[k]},")
    print(f"  }}")
    print(f"\n  Risultati completi salvati in: _test_results.csv")
    print(f"{'='*72}\n")


if __name__ == '__main__':
    asyncio.run(main())
