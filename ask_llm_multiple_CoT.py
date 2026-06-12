"""
MPQ LLM Classifier - McGill Pain Questionnaire
================================================
Confronto tra strategie di prompting LLM per la classificazione di:
  - Pain Intensity (3 classi via PRI)
  - Etiology (Nocicettivo vs Neuropatico)

Strategie implementate:
  1. Zero-Shot
  2. Chain-of-Thought (CoT) - Semplice
  3. CoT Gerarchica (Multi-livello)
  4. CoT con Verifica (Self-Consistency)
  5. CoT con ReAct Pattern
  6. CoT Ensemble (Multi-Perspective)
  7. CoT con Albero Decisionale Clinico
  8. CoT con Calcolo della Confidenza
  9. Few-Shot
  10. Knowledge-Guided (solo Etiology, formula Wilkie et al. 2001)

Supporto modelli:
  - Ollama locale (es. llama3, mistral)
  - Ollama Cloud via API key
"""

import os
import json
import time
import math
import asyncio
import random
import hashlib
import datetime
import argparse
import requests
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report
)
from openai import AsyncOpenAI  # ollama locale compatibile OpenAI
from ollama import AsyncClient as AsyncOllamaClient
from typing import Literal

# ─────────────────────────────────────────────
# CONFIGURAZIONE
# ─────────────────────────────────────────────

# Scegli la modalità: "local" | "cloud" | "both"
MODE = "both"

# Modelli locali — uno per PC, ognuno con il proprio host Ollama
LOCAL_MODELS = [
    "phi4-mini:latest", # questo PC       — 3.8B ~2.5GB VRAM
    "mistral:latest",   # Domino11 (5.24) — 7B ~5GB VRAM
    "qwen2.5:7b",       # MSIMio (5.24) — 7B ~5GB VRAM
    "llama3.1:8b",      # Domino12 (5.15) — 8B ~5GB VRAM
]
LOCAL_MODEL_HOSTS = {
    "phi4-mini:latest": "http://localhost:11434/v1",
    "mistral:latest":   "http://localhost:11434/v1",
    "qwen2.5:7b":       "http://localhost:11434/v1",
    "llama3.1:8b":      "http://192.168.5.15:11434/v1",
}
LOCAL_API_KEY = "ollama"  # placeholder richiesto dal client OpenAI

# Modelli cloud Ollama
CLOUD_MODELS = [
    # "gpt-oss:20b"  — rimosso: restituisce risposte vuote su CoT complesse
    # "minimax-m2"   — rimosso: PI parse rate 40%, fallisce su CoT gerarchica/verifica
    # "glm-4.7"      — rimosso: PI parse rate ~39%, troppo verboso (calcola PRI esplicitamente)
    # "gpt-oss:120b"  — rimosso: richiede account premium (401 su chiavi free)
    "gemma4:31b",
    "gemma3:27b",
    "ministral-3:14b",
    "gemma3:4b",
    "gemma3:12b",
    "rnj-1:8b",
    # "nemotron-3-super:latest" — rimosso: risposta vuota con tutte le chiavi (serve tier Nvidia)
    "ministral-3:8b",            # sostituto minimax-m2   (Mistral 8B)
    "devstral-small-2:24b",      # sostituto glm-4.7      (Mistral Devstral 24B, PI 80% / Etio 83%)
]
# Chiavi API cloud — caricate da ChaviOllama.json (NON hardcodare qui per la repo pubblica)
# Copia ChaviOllama.json.example → ChaviOllama.json e inserisci le tue chiavi.
def _load_cloud_api_keys() -> list:
    """Carica CLOUD_API_KEYS da ChaviOllama.json."""
    try:
        with open("ChaviOllama.json", encoding="utf-8") as f:
            data = json.load(f)
        keys = [e["key"] for e in data.get("CLOUD_API_KEYS", [])]
        if not keys:
            raise ValueError("Nessuna chiave trovata in ChaviOllama.json")
        return keys
    except FileNotFoundError:
        raise FileNotFoundError(
            "ChaviOllama.json non trovato. "
            "Copia ChaviOllama.json.example → ChaviOllama.json e inserisci le tue chiavi API."
        )

CLOUD_API_KEYS = _load_cloud_api_keys()
CLOUD_KEY_IDX = 0        # indice chiave corrente (ruotato su 429/403)
_cloud_clients: dict = {}  # cache client per chiave
# {api_key: {"name": ..., "reason": ..., "since": ..., "reset_at": ...}}
_exhausted_keys: dict = {}
_all_keys_exhausted: bool = False  # True quando non rimane nessuna chiave attiva

KEY_STATE_FILE = "key_state.json"

# Mappa key→name per logging leggibile
_KEY_NAMES: dict = {}  # popolato da _load_key_state() o _build_key_names()

def _build_key_names():
    """Popola _KEY_NAMES da ChaviOllama.json."""
    global _KEY_NAMES
    try:
        with open("ChaviOllama.json", encoding="utf-8") as f:
            data = json.load(f)
        _KEY_NAMES = {e["key"]: e["name"] for e in data.get("CLOUD_API_KEYS", [])}
    except Exception:
        pass

def _key_name(api_key: str) -> str:
    return _KEY_NAMES.get(api_key, api_key[:12] + "…")

def _get_key_semaphore(api_key: str) -> asyncio.Semaphore:
    """Ritorna (creando se necessario) il semaforo per-chiave."""
    if api_key not in _KEY_SEMAPHORES:
        limit = MAX_CONCURRENT_PRO if api_key in PRO_API_KEYS else MAX_CONCURRENT_FREE
        _KEY_SEMAPHORES[api_key] = asyncio.Semaphore(limit)
    return _KEY_SEMAPHORES[api_key]

def _load_key_state():
    """Carica chiavi esaurite da key_state.json, scartando quelle con reset scaduto."""
    global _exhausted_keys
    _build_key_names()
    if not os.path.exists(KEY_STATE_FILE):
        return
    try:
        with open(KEY_STATE_FILE, encoding="utf-8") as f:
            data = json.load(f)
        now = datetime.datetime.now()
        loaded = 0
        for key, info in data.get("exhausted", {}).items():
            reset_at = datetime.datetime.fromisoformat(info["reset_at"])
            if now < reset_at:
                _exhausted_keys[key] = info
                loaded += 1
                print(f"  [KEY-STATE] {info['name']}: {info['reason']} — reset {reset_at.strftime('%d/%m %H:%M')}", flush=True)
        if loaded:
            print(f"  [KEY-STATE] {loaded} chiavi già esaurite caricate da {KEY_STATE_FILE}", flush=True)
    except Exception as e:
        print(f"  [KEY-STATE] Errore caricamento: {e}", flush=True)

def _save_key_state():
    """Persiste lo stato corrente di _exhausted_keys su key_state.json."""
    try:
        data = {"exhausted": _exhausted_keys, "updated": datetime.datetime.now().isoformat()}
        with open(KEY_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"  [KEY-STATE] Errore salvataggio: {e}", flush=True)

def _mark_key_exhausted(api_key: str, reason: str):
    """Marca una chiave come esaurita e persiste lo stato."""
    global _exhausted_keys
    now = datetime.datetime.now()
    if "weekly" in reason.lower():
        days = (7 - now.weekday()) % 7 or 7   # giorni al prossimo lunedì
        reset_at = (now + datetime.timedelta(days=days)).replace(hour=0, minute=0, second=0, microsecond=0)
    else:  # daily
        reset_at = (now + datetime.timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    _exhausted_keys[api_key] = {
        "name": _key_name(api_key),
        "reason": reason,
        "since": now.isoformat(),
        "reset_at": reset_at.isoformat(),
    }
    _save_key_state()

# Telegram notifiche
TELEGRAM_TOKEN   = "8779083388:AAE9m7dlXSm2ql1kYsqcwcNy34pNYUSYFuo"
TELEGRAM_CHAT_ID = "139781098"
NOTIFY_INTERVAL  = 4 * 3600  # 4 ore in secondi

# Path dataset (lista — esclude Dataset.csv che è il dataset originale)
DATASET_PATHS = [
    "Tesi_codice_Senatore/dataset/MC_gill_DEEPSEEK.csv",
    "Tesi_codice_Senatore/dataset/McGill_Pain_Questionnaire_GPT_Con_Dolore.csv",
    "Tesi_codice_Senatore/dataset/McGill_Pain_Questionnaire_CLAUDE.csv",
    "Tesi_codice_Senatore/dataset/McGill_Pain_Questionnaire_DOCTORAI.csv",
]

# Output (i nomi vengono generati automaticamente per ogni dataset)
OUTPUT_DIR = "results"
CHECKPOINT_DIR = "checkpoints"   # salvataggio incrementale per paziente
CACHE_FILE = "llm_cache.json"

# Async concurrency limit per modello
CONCURRENCY_LIMIT = 1
# Max chiamate cloud simultanee (evita 429 da rate limit)
CLOUD_CONCURRENT_CALLS = 5
CLOUD_SEMAPHORE = None  # inizializzato dentro asyncio.run()
# Chiavi PRO (type="PRO" in ChaviOllama.json): concorrenza media. Free: 1 concurrent.
def _load_pro_api_keys() -> set:
    try:
        with open("ChaviOllama.json", encoding="utf-8") as f:
            data = json.load(f)
        return {e["key"] for e in data.get("CLOUD_API_KEYS", []) if e.get("type") == "PRO"}
    except Exception:
        return set()

PRO_API_KEYS = _load_pro_api_keys()
MAX_CONCURRENT_PRO  = 2
MAX_CONCURRENT_FREE = 1
_KEY_SEMAPHORES: dict = {}  # api_key → asyncio.Semaphore, creati lazily

# ─────────────────────────────────────────────
# GRUPPI PER ESECUZIONE PARALLELA (--group)
# ─────────────────────────────────────────────
REMOTE_QWEN_HOST = "http://172.19.50.23:11434/v1"  # nuova macchina qwen2.5:7b

MODEL_GROUPS = {
    "local":  ["phi4-mini:latest", "mistral:latest"],
    "remote": ["qwen2.5:7b", "llama3.1:8b"],
}

# Seleziona quali strategie CoT usare (per risparmiare token/API)
USE_COT_STRATEGIES = {
    'simple': True,           # CoT originale
    'hierarchical': True,     # CoT gerarchica
    'verification': True,     # CoT con verifica
    'react': False,           # CoT ReAct (molto token-intensive)
    'ensemble': True,         # CoT ensemble
    'decision_tree': True,    # CoT albero decisionale
    'confidence': True,       # CoT con confidenza
}

# ─────────────────────────────────────────────
# COSTANTI CLINICHE
# ─────────────────────────────────────────────

# Colonne score per il calcolo del PRI
SCORE_COLS = [
    'sensory_score', 'fear_score', 'sensory_misc_score', 'brightness_score',
    'affective_score', 'thermal_score', 'traction_score', 'tension_score',
    'evaluative_score', 'dullness_score', 'autonomic_score', 'sensory_3_score',
    'spatial_score', 'affective_evaluative_sensory_score', 'temporal_score',
    'punishment_score', 'punctate_score', 'sensory_2_score', 'incisive_score',
    'constrictive_score'
]

# Colonne descrittori testuali MPQ (i valori sono le parole scelte dal paziente)
DESCRIPTOR_COLS = [
    'sensory', 'fear', 'sensory_misc', 'brightness', 'affective', 'thermal',
    'traction', 'tension', 'evaluative', 'dullness', 'autonomic', 'sensory_3',
    'spatial', 'affective_evaluative_sensory', 'temporal', 'punishment',
    'punctate', 'sensory_2', 'incisive', 'constrictive'
]

# Colonne fattori modulatori — popolato per ogni dataset in main()
FACTOR_COLS = []

# 13 descrittori Wilkie et al. (2001) e coefficienti beta
WILKIE_COEFFICIENTS = {
    'shooting':  3.382,
    'tingling':  3.881,
    'crushing':  0.540,
    'blinding':  0.729,
    'tight':     0.171,
    'cold':      0.095,
    'itchy':    -0.721,
    'pressing': -0.827,
    'smarting': -0.777,
    'flashing': -1.777,
    'jumping':  -1.605,
    'freezing': -0.658,
    'tearing':  -0.655,
}
WILKIE_INTERCEPT = 0.815

# Soglie PRI per Pain Intensity
PRI_CLASS_0_MAX = 25   # PRI < 26
PRI_CLASS_1_MAX = 45   # 26 <= PRI < 46
# PRI >= 46 -> Classe 2

# ─────────────────────────────────────────────
# TELEGRAM
# ─────────────────────────────────────────────

def send_telegram(message: str) -> None:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}, timeout=10)
    except Exception as e:
        print(f"[Telegram] Errore invio: {e}")


def build_progress_message(progress: dict) -> str:
    now = datetime.datetime.now()
    elapsed = now - progress['start_time']
    h, rem = divmod(int(elapsed.total_seconds()), 3600)
    m = rem // 60

    ds_done = len(progress['datasets_done'])
    ds_total = progress['datasets_total']
    ds_current = progress['dataset_current'] or '—'
    pat_total = progress.get('patients_total', 0)
    models_total = progress['models_total']
    models_done = len(progress['models_done'])
    models_left = models_total - models_done

    # pazienti per modello sul dataset corrente
    pat_lines = []
    for label, count in progress.get('patients_done', {}).items():
        pct = int(count / pat_total * 100) if pat_total else 0
        pat_lines.append(f"  • {label}: {count}/{pat_total} ({pct}%)")

    errors = progress.get('errors', [])
    err_block = ("\n\nERRORI:\n" + "\n".join(f"  ⚠ {e}" for e in errors[-5:])) if errors else ""

    lines = [
        f"<b>MPQ LLM — aggiornamento</b>",
        f"Tempo trascorso: {h}h {m}m",
        f"",
        f"Dataset: {ds_done}/{ds_total} completati",
        f"Corrente: <code>{ds_current}</code>",
        f"Pazienti totali dataset: {pat_total}",
        f"",
        f"Modelli: {models_done}/{models_total} completati, {models_left} in corso",
    ]
    if pat_lines:
        lines += ["", "Avanzamento per modello:"] + pat_lines
    if err_block:
        lines.append(err_block)
    return "\n".join(lines)


async def periodic_notify(progress: dict) -> None:
    """Task asyncio che invia un aggiornamento Telegram ogni NOTIFY_INTERVAL secondi."""
    while True:
        await asyncio.sleep(NOTIFY_INTERVAL)
        send_telegram(build_progress_message(progress))


# ─────────────────────────────────────────────
# NORMALIZZAZIONE DATASET
# ─────────────────────────────────────────────

# Mapping nomi colonna → nome canonico atteso dal codice
_COL_ALIASES = {
    # score: varianti con "_pressure_"
    'traction_pressure_score':    'traction_score',
    'punctate_pressure_score':    'punctate_score',
    'incisive_pressure_score':    'incisive_score',
    'constrictive_pressure_score':'constrictive_score',
    'sensory_miscellaneous_score':'sensory_misc_score',
    # score: abbreviazioni GPT/DOCTORAI
    'affective_eval_sensory_score':'affective_evaluative_sensory_score',
    'sensory2_score':             'sensory_2_score',
    'sensory3_score':             'sensory_3_score',
    # descrittori verbali
    'traction_pressure':    'traction',
    'punctate_pressure':    'punctate',
    'incisive_pressure':    'incisive',
    'constrictive_pressure':'constrictive',
    'sensory_miscellaneous':'sensory_misc',
    'affective_eval_sensory':'affective_evaluative_sensory',
    'sensory2':             'sensory_2',
    'sensory3':             'sensory_3',
}

def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Uniforma i nomi colonna tra dataset con convenzioni diverse.
    Non fa lowercase globale per non rompere colonne come Sex, Age, ID.
    """
    df = df.copy()

    # Mappa case-insensitive: lowercase → nome reale attuale nel df
    col_lower_map = {c.lower(): c for c in df.columns}

    rename = {}

    # 1. Applica gli alias espliciti (es. traction_pressure_score → traction_score)
    for alias, canonical in _COL_ALIASES.items():
        actual = col_lower_map.get(alias)
        if actual and actual not in rename.values():
            rename[actual] = canonical

    # 2. Normalizza in minuscolo le colonne di score, descrittori e pain_type
    #    se esistono con lettere maiuscole ma il codice si aspetta minuscolo
    for expected in SCORE_COLS + DESCRIPTOR_COLS + ['total_pain_score', 'pain_type']:
        if expected not in df.columns:
            actual = col_lower_map.get(expected)
            if actual:
                rename[actual] = expected

    df = df.rename(columns=rename)

    # 3. Calcola total_pain_score se ancora mancante (DEEPSEEK non ce l'ha)
    if 'total_pain_score' not in df.columns:
        present = [c for c in SCORE_COLS if c in df.columns]
        df['total_pain_score'] = df[present].sum(axis=1)

    # 4. Rimuovi righe con dati chiave mancanti
    df = df.dropna(subset=['total_pain_score', 'pain_type']).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────
# UTILITY
# ─────────────────────────────────────────────

def get_true_labels(df: pd.DataFrame) -> dict:
    """Calcola le etichette reali per Pain Intensity e Etiology."""
    pri = df['total_pain_score']
    # bins aperti verso l'esterno per gestire valori fuori range
    pain_intensity = pd.cut(
        pri,
        bins=[-float('inf'), 25, 45, float('inf')],
        labels=[0, 1, 2]
    ).astype(int)

    etiology = (
        df['pain_type']
        .map({'nociceptive': 0, 'neuropathic': 1})
        .fillna(-1)
        .astype(int)
    )

    return {
        'pain_intensity': pain_intensity.tolist(),
        'etiology': etiology.tolist()
    }


def get_patient_descriptors(row: pd.Series) -> list[str]:
    """Restituisce la lista di descrittori MPQ scelti dal paziente."""
    return [str(row[c]).lower().strip() for c in DESCRIPTOR_COLS]


def get_wilkie_present(descriptors: list[str]) -> list[str]:
    """Filtra i descrittori del paziente che sono nei 13 di Wilkie."""
    return [d for d in descriptors if d in WILKIE_COEFFICIENTS]


def compute_wilkie_score(descriptors: list[str]) -> float:
    """Calcola il Predictive Score di Wilkie et al."""
    score = WILKIE_INTERCEPT
    for desc, coef in WILKIE_COEFFICIENTS.items():
        if desc in descriptors:
            score += coef
    return score


def parse_class_from_response(response: str, task: str) -> int | None:
    """
    Estrae la classe predetta dalla risposta testuale del modello.
    Versione migliorata con supporto per formati CoT complessi.
    """
    import unicodedata
    # Normalizza spazi Unicode (narrow no-break space  , etc.) → spazio normale
    response = unicodedata.normalize('NFKC', response)
    response_lower = response.lower()

    import re

    if task == 'pain_intensity':
        # Cerca "Classe X", "Classe: X", "Classe finale: X", ecc.
        for cls in ['2', '1', '0']:
            patterns = [
                f'classe {cls}', f'class {cls}',
                f'classe: {cls}', f'class: {cls}',
                f'classe finale: {cls}', f'classe finale {cls}',
                f'output: classe {cls}', f'output finale: {cls}',
                f'risposta finale: {cls}',
            ]
            if any(p in response_lower for p in patterns):
                return int(cls)

        # Formati CoT ensemble / classificazione integrata
        m = re.search(
            r'(?:classe integrata|classificazione finale|classe finale|output finale)'
            r'\s*[:\-]?\s*([012])',
            response_lower
        )
        if m:
            return int(m.group(1))

        # Fallback: ultimo numero 0/1/2 isolato negli ultimi 300 caratteri
        numbers = re.findall(r'\b[012]\b', response[-300:])
        if numbers:
            return int(numbers[-1])

    elif task == 'etiology':
        if any(w in response_lower for w in [
            'neuropath', 'classe 1', 'class 1', 'classe: 1', 'class: 1',
            'neuropatico', 'neuropatica'
        ]):
            return 1
        if any(w in response_lower for w in [
            'nociceptiv', 'classe 0', 'class 0', 'classe: 0', 'class: 0',
            'nocicettivo', 'nocicettiva'
        ]):
            return 0

        # Output strutturato "classe: X"
        if 'classe:' in response_lower:
            after = response_lower.split('classe:')[1][:15]
            if re.search(r'\b1\b', after): return 1
            if re.search(r'\b0\b', after): return 0

        # Fallback semantico sugli ultimi 400 caratteri
        tail = response_lower[-400:]
        if any(w in tail for w in ['neuropat', 'nervos', 'neural', 'shooting', 'tingling']):
            return 1
        if any(w in tail for w in ['nocicet', 'tissut', 'infiamm', 'somatico', 'pressing']):
            return 0

    return None


def build_patient_context(row: pd.Series, include_factors: bool = True) -> str:
    """Costruisce il contesto testuale del paziente per il prompt."""
    descriptors = get_patient_descriptors(row)

    lines = [
        f"Sesso: {row['Sex']}, Età: {row['Age']} anni",
        f"",
        f"Sottoscale MPQ e score:",
    ]

    for col in SCORE_COLS:
        label = col.replace('_score', '').replace('_', ' ').capitalize()
        lines.append(f"  - {label}: {row[col]}")

    lines += [
        f"",
        f"Descrittori verbali scelti dal paziente (uno per sottoscala):",
    ]
    for c, d in zip(DESCRIPTOR_COLS, descriptors):
        lines.append(f"  - {c.replace('_', ' ')}: {d}")

    if include_factors:
        lines += ["", "Fattori modulatori:"]
        for f in FACTOR_COLS:
            label = f.replace('factor_', '').replace('_', ' ')
            lines.append(f"  - {label}: {row[f]}")

    return '\n'.join(lines)


# ─────────────────────────────────────────────
# PROMPT BUILDERS - PAIN INTENSITY (CoT Avanzate)
# ─────────────────────────────────────────────

PAIN_INTENSITY_SYSTEM = (
    "Sei un esperto nella valutazione del dolore con il McGill Pain Questionnaire (MPQ). "
    "Il tuo compito è classificare l'intensità del dolore di un paziente in base al "
    "Pain Rating Index (PRI), calcolato come somma degli score delle 20 sottoscale MPQ. "
    "Le classi sono:\n"
    "  - Classe 0: PRI < 26 (dolore lieve)\n"
    "  - Classe 1: 26 ≤ PRI ≤ 45 (dolore moderato)\n"
    "  - Classe 2: PRI ≥ 46 (dolore severo)\n"
    "I dati del paziente sono già forniti nel messaggio utente: non chiedere ulteriori informazioni. "
    "Rispondi SEMPRE con 'Classe 0', 'Classe 1' o 'Classe 2' alla fine della tua risposta."
)

def prompt_pi_zero_shot(row: pd.Series) -> str:
    context = build_patient_context(row, include_factors=False)
    return f"Dati del paziente:\n{context}\n\nClassifica l'intensità del dolore di questo paziente."

def prompt_pi_cot_simple(row: pd.Series) -> str:
    """CoT originale (semplice)"""
    context = build_patient_context(row, include_factors=False)
    return (
        f"Dati del paziente:\n{context}\n\n"
        f"Segui questi passi per classificare l'intensità del dolore:\n"
        f"PASSO 1 - Somma tutti i 20 score MPQ elencati sopra e calcola il PRI totale.\n"
        f"PASSO 2 - Confronta il PRI con le soglie: <26 = Classe 0, 26-45 = Classe 1, >=46 = Classe 2.\n"
        f"PASSO 3 - Indica la classe risultante.\n\n"
        f"Mostra il calcolo esplicito di ogni passo."
    )

def prompt_pi_cot_hierarchical(row: pd.Series) -> str:
    """CoT Gerarchica - analisi multi-livello"""
    context = build_patient_context(row, include_factors=False)
    return f"""
Dati del paziente:
{context}

Esegui una CLASSIFICAZIONE GERARCHICA dell'intensità del dolore:

LIVELLO 1 - ANALISI PER SOTTODOMINIO:
Calcola il PRI parziale per ciascuna categoria MPQ:
  - PRI_SENSORY = somma(sensory_score, sensory_2_score, sensory_3_score, punctate_score, incisive_score, spatial_score, constrictive_score)
  - PRI_AFFECTIVE = somma(affective_score, fear_score, punishment_score, affective_evaluative_sensory_score)
  - PRI_EVALUATIVE = somma(evaluative_score, brightness_score)
  - PRI_MISCELLANEOUS = somma(thermal_score, traction_score, tension_score, dullness_score, autonomic_score, temporal_score, sensory_misc_score)

Mostra i calcoli intermedi per ogni dominio.

LIVELLO 2 - IDENTIFICAZIONE PATTERN:
Analizza il profilo del dolore:
  - Pattern sensoriale predominante? (PRI_SENSORY > PRI_AFFECTIVE + PRI_EVALUATIVE)
  - Pattern affettivo significativo? (PRI_AFFECTIVE > 10)
  - Pattern misto/equilibrato?

LIVELLO 3 - CALCOLO PRI TOTALE:
PRI_total = PRI_SENSORY + PRI_AFFECTIVE + PRI_EVALUATIVE + PRI_MISCELLANEOUS
Verifica che la somma corrisponda alla somma diretta dei 20 score.

LIVELLO 4 - CLASSIFICAZIONE INTEGRATA:
Considera sia PRI_total che i pattern identificati:
  - Se PRI_total < 26: CLASSE 0 (dolore lieve)
  - Se PRI_total 26-45: CLASSE 1 (dolore moderato)
      * Se pattern affettivo significativo: nota la componente emotiva
      * Se pattern sensoriale predominante: potrebbe essere neuropatico
  - Se PRI_total ≥ 46: CLASSE 2 (dolore severo)

Mostra tutti i calcoli e la classificazione finale nel formato: "Classe Finale: [0/1/2]"
"""

def prompt_pi_cot_verification(row: pd.Series) -> str:
    """CoT con Verifica (Self-Consistency)"""
    context = build_patient_context(row, include_factors=False)
    return f"""
{context}

FASE 1 - RAGIONAMENTO INIZIALE:
Calcola passo-passo il PRI totale:
  [Elenca tutti i 20 score con la loro somma progressiva]
  PRI_totale_calcolato = [risultato]

Determina classe iniziale basata sulle soglie:
  <26 → Classe 0
  26-45 → Classe 1
  ≥46 → Classe 2
Classe_iniziale = [X]

FASE 2 - VERIFICA CRITICA:
Ora metti in discussione il tuo ragionamento:

a) Ho incluso tutti i 20 score? (Verifica elenco completo)
   Score mancanti: [nessuno/elencali]

b) Ricalcolo con metodo alternativo (somma per sottodomini):
   Sensory = (sensory + sensory_2 + sensory_3 + punctate + incisive + spatial + constrictive) = [valore]
   Affective = (affective + fear + punishment + affective_evaluative_sensory) = [valore]
   Evaluative = (evaluative + brightness) = [valore]
   Misc = (thermal + traction + tension + dullness + autonomic + temporal + sensory_misc) = [valore]
   PRI_verifica = [somma dei 4 domini] = [deve eguagliare PRI_iniziale]

c) Analisi casi limite:
   - Il PRI è entro 3 punti da una soglia? (Sì/No - se sì, quale soglia?)
   - Ci sono score estremi (valore 3) in sottoscale chiave? (Elenca)

FASE 3 - CLASSIFICAZIONE FINALE:
Dopo la verifica, la classificazione iniziale era corretta? (Sì/No)
Se No, correggi a: Classe [Y]
Motivazione della correzione: [spiegazione]

FASE 4 - OUTPUT:
Classe Finale: [0/1/2]
Confidenza: [Alta/Media/Bassa] (bassa se vicino a soglie o con discrepanze)
"""

def prompt_pi_cot_react(row: pd.Series) -> str:
    """CoT con ReAct Pattern (Reasoning + Acting)"""
    context = build_patient_context(row, include_factors=False)
    return f"""
{context}

Segui il pattern ReAct (REasoning + ACTing) per la classificazione:

THOUGHT 1: Devo calcolare il PRI totale sommando tutti i 20 score MPQ.
ACTION 1: [CALC] Eseguo la somma progressiva
  sensory(3) + fear(2) + sensory_misc(1) + brightness(2) + affective(3) + 
  thermal(1) + traction(2) + tension(2) + evaluative(3) + dullness(1) +
  autonomic(1) + sensory_3(2) + spatial(2) + affective_evaluative_sensory(1) +
  temporal(2) + punishment(1) + punctate(2) + sensory_2(2) + incisive(1) + constrictive(2)
OBSERVATION 1: Somma parziale = [XX]. Continuo la somma...

THOUGHT 2: Completo la somma e ottengo PRI_total.
ACTION 2: [CALC] PRI_total = [YY]
OBSERVATION 2: PRI_total = [YY]

THOUGHT 3: Devo confrontare PRI_total con le soglie cliniche.
ACTION 3: [COMPARE] Soglie: [0-25] = Classe0, [26-45] = Classe1, [46-78] = Classe2
OBSERVATION 3: [YY] rientra nell'intervallo [XX-YY] → Classe [Z]_tentativa

THOUGHT 4: Verifico la distribuzione degli score per pattern clinici.
ACTION 4: [ANALYZE] 
  - Score medi: sensory=2.1, affective=1.8, evaluative=2.5
  - Pattern predominante: [sensoriale/affettivo/misto]
OBSERVATION 4: Il pattern suggerisce [caratteristiche specifiche]

THOUGHT 5: Formulo la classificazione finale considerando anche il pattern.
ACTION 5: [OUTPUT] Classe [Z_finale] perché [motivazione]

OUTPUT FINALE: Classe [0/1/2]
"""

def prompt_pi_cot_ensemble(row: pd.Series) -> str:
    """CoT Ensemble - Multi-perspective analysis"""
    context = build_patient_context(row, include_factors=False)
    descriptors = get_patient_descriptors(row)
    factors = {f: row[f] for f in FACTOR_COLS[:5]} if FACTOR_COLS else {}
    
    return f"""
{context}

Analizza l'intensità del dolore da TRE PROSPETTIVE diverse:

┌─────────────────────────────────────────────────────────────┐
│ PROSPETTIVA 1 - QUANTITATIVA (basata su PRI esatto)        │
└─────────────────────────────────────────────────────────────┘
Calcolo PRI esatto:
  sensory(3) + fear(2) + sensory_misc(1) + brightness(2) + affective(3) + 
  thermal(1) + traction(2) + tension(2) + evaluative(3) + dullness(1) +
  autonomic(1) + sensory_3(2) + spatial(2) + affective_evaluative_sensory(1) +
  temporal(2) + punishment(1) + punctate(2) + sensory_2(2) + incisive(1) + constrictive(2)
  
  PRI_totale = [somma]

Applicazione soglie:
  - PRI_totale < 26 → Classe 0
  - 26 ≤ PRI_totale ≤ 45 → Classe 1
  - PRI_totale ≥ 46 → Classe 2

RISULTATO PROSPETTIVA 1: Classe [X]

┌─────────────────────────────────────────────────────────────┐
│ PROSPETTIVA 2 - QUALITATIVA (basata su descrittori verbali)│
└─────────────────────────────────────────────────────────────┘
Descrittori del paziente: {', '.join(descriptors[:10])}...

Analisi qualitativa:
  - Numero di descrittori ad alta intensità (punteggio 3): [conta]
  - Descrittori sensoriali predominanti: [elenca]
  - Descrittori affettivi presenti: [elenca]
  - Grado di sofferenza percepita: [basso/medio/alto] basato su descrittori affettivi

Stima severità percepita dal paziente:
  - Lieve (0-2 descrittori alta intensità) → Classe 0/1
  - Moderata (3-5 descrittori alta intensità) → Classe 1
  - Severa (6+ descrittori alta intensità) → Classe 2

RISULTATO PROSPETTIVA 2: Classe [Y]

┌─────────────────────────────────────────────────────────────┐
│ PROSPETTIVA 3 - CLINICO-FUNZIONALE (impatto sulla vita)    │
└─────────────────────────────────────────────────────────────┘
Fattori modulatori considerati:
  {chr(10).join([f'  - {k}: {v}' for k, v in factors.items()])}

Valutazione funzionale:
  - Il dolore interferisce con attività quotidiane? [Sì/No/Parzialmente]
  - Durata del dolore: [acuto/subacuto/cronico]
  - Fattori aggravanti presenti? [Sì/No]
  - Pattern di peggioramento/miglioramento: [descrizione]

Severità funzionale stimata:
  - Bassa disabilità → Classe 0/1
  - Disabilità moderata → Classe 1
  - Alta disabilità → Classe 2

RISULTATO PROSPETTIVA 3: Classe [Z]

┌─────────────────────────────────────────────────────────────┐
│ CLASSIFICAZIONE INTEGRATA                                   │
└─────────────────────────────────────────────────────────────┘
Le 3 prospettive concordano?
  - Sì, tutte danno Classe [W] → Classe Finale = [W]
  - No, discrepanze: [spiega]

Classe Finale Integrata = [media ponderata o consensus]
Motivazione clinica: [spiegazione della scelta]

OUTPUT: Classe [0/1/2]
"""

def prompt_pi_cot_decision_tree(row: pd.Series) -> str:
    """CoT con Albero Decisionale Clinico"""
    context = build_patient_context(row, include_factors=False)
    return f"""
{context}

Segui questo ALBERO DECISIONALE CLINICO per la classificazione:

                    ┌─────────────────┐
                    │  START: PRI ≥ 46? │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │ SÌ                          │ NO
              ▼                             ▼
        ┌──────────┐              ┌─────────────────┐
        │ CLASSE 2 │              │  PRI ≥ 26?      │
        │ (Severo) │              └────────┬────────┘
        └──────────┘                       │
                              ┌────────────┴────────────┐
                              │ SÌ                      │ NO
                              ▼                         ▼
                    ┌──────────────┐            ┌──────────┐
                    │ ANALISI      │            │ CLASSE 0 │
                    │ PATTERN      │            │ (Lieve)  │
                    └──────────────┘            └──────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │ Sottodominio predominante?   │
              └───────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
  ┌──────────┐        ┌──────────────┐      ┌──────────┐
  │SENSORIALE│        │  AFFETTIVO   │      │  MISTO   │
  │ > 60%    │        │   > 30%      │      │          │
  └─────┬────┘        └──────┬───────┘      └─────┬────┘
        │                    │                    │
        ▼                    ▼                    ▼
  ┌──────────┐        ┌──────────────┐      ┌──────────┐
  │Considera │        │Considera     │      │ CLASSE 1 │
  │componente│        │componente    │      │Standard  │
  │neuropatica│       │psicologica   │      └──────────┘
  └──────────┘        └──────────────┘

ORA ESEGUI L'ALBERO:

DOMANDA 1: PRI_total ≥ 46?
  ├─ Calcolo PRI_total = somma(20 score) = [calcolo] = [valore]
  ├─ Verdetto: [SÌ/NO]
  └─ Se SÌ → CLASSE 2 (dolore severo) → VAI a OUTPUT FINALE

DOMANDA 2 (se NO a D1): PRI_total ≥ 26?
  ├─ [valore] ≥ 26? [SÌ/NO]
  ├─ Se NO → CLASSE 0 (dolore lieve) → VAI a OUTPUT FINALE
  └─ Se SÌ → VAI a DOMANDA 3

DOMANDA 3 (se PRI in [26-45]): Qual è il sottodominio predominante?
  ├─ Calcolo PRI_SENSORY = [calcolo] = [valore_S]
  ├─ Calcolo PRI_AFFECTIVE = [calcolo] = [valore_A]
  ├─ Calcolo PRI_EVALUATIVE = [calcolo] = [valore_E]
  ├─ Totale = [valore_S + valore_A + valore_E] = [deve essere ≈ PRI_total]
  │
  ├─ Percentuali:
  │   ├─ %Sensory = (valore_S / PRI_total) × 100 = [X]%
  │   ├─ %Affective = (valore_A / PRI_total) × 100 = [Y]%
  │   └─ %Evaluative = (valore_E / PRI_total) × 100 = [Z]%
  │
  └─ Pattern identificato:
      ├─ Se %Sensory > 60% → "Sensoriale predominante" → CLASSE 1 con nota neuropatica
      ├─ Se %Affective > 30% → "Affettivo significativo" → CLASSE 1 con nota psicologica
      └─ Altrimenti → "Misto/equilibrato" → CLASSE 1 standard

OUTPUT FINALE:
Classe: [0/1/2]
Sottotipo: [Lieve/Moderato-Sensoriale/Moderato-Affettivo/Moderato-Misto/Severo]
Note cliniche: [se applicabile]
"""

def prompt_pi_cot_confidence(row: pd.Series) -> str:
    """CoT con Calcolo della Confidenza"""
    context = build_patient_context(row, include_factors=False)
    return f"""
{context}

PROCEDURA DI CLASSIFICAZIONE CON CALCOLO DELLA CONFIDENZA

STEP 1 - CALCOLO PRI TOTALE
PRI_totale = somma di tutti i 20 score MPQ:

[Elenca somma dettagliata]
PRI_totale = [valore]

STEP 2 - DETERMINAZIONE CLASSE BASE
Confronto con soglie:
  - PRI_totale < 26 → Classe 0
  - 26 ≤ PRI_totale ≤ 45 → Classe 1
  - PRI_totale ≥ 46 → Classe 2

Classe_base = [C] (dove C ∈ {{0,1,2}})

STEP 3 - CALCOLO CONFIDENZA BASE
Calcolo la distanza dalla soglia più vicina:

Se Classe_base = 0:
  distanza_da_soglia = 26 - PRI_totale
  confidenza_base = min(100, (distanza_da_soglia / 26) × 100)

Se Classe_base = 1:
  distanza_da_soglia_inferiore = PRI_totale - 26
  distanza_da_soglia_superiore = 45 - PRI_totale
  distanza_min = min(distanza_da_soglia_inferiore, distanza_da_soglia_superiore)
  confidenza_base = min(100, (distanza_min / 20) × 100)

Se Classe_base = 2:
  distanza_da_soglia = PRI_totale - 45
  confidenza_base = min(100, (distanza_da_soglia / 33) × 100)

Confidenza_base = [X]%

STEP 4 - ANALISI FATTORI DI DUBBIO (penalità)

Fattore 1 - Prossimità alla soglia:
  - PRI entro 3 punti dalla soglia → penalità 20%
  - PRI entro 5 punti dalla soglia → penalità 10%
  Penalità_soglia = [Y1]%

Fattore 2 - Score estremi contrastanti:
  - Presenza di score = 3 (massimo) in sottoscale a basso impatto → penalità 5%
  - Score zero in sottoscale ad alto impatto (sensory, affective) → penalità 10%
  Penalità_pattern = [Y2]%

Fattore 3 - Variabilità tra sottodomini:
  - Calcolo deviazione standard degli score dei 4 domini:
    sensory_domain = [valore], affective_domain = [valore], 
    evaluative_domain = [valore], misc_domain = [valore]
  - Alta varianza (>2.0) → penalità 15%
  - Media varianza (1.0-2.0) → penalità 8%
  Penalità_varianza = [Y3]%

STEP 5 - CONFIDENZA ADATTATA
Confidenza_adattata = confidenza_base - (penalità_soglia + penalità_pattern + penalità_varianza)
Confidenza_adattata = [Z]%

STEP 6 - CLASSIFICAZIONE E RACCOMANDAZIONE
Classe: [C]
Confidenza: [Z]%

Giudizio di confidenza:
  - ≥ 80%: Alta confidenza - classificazione robusta
  - 60-79%: Media confidenza - accettabile
  - 40-59%: Bassa confidenza - interpretare con cautela
  - < 40%: Confidenza molto bassa - richiedere rivalutazione

Raccomandazione clinica:
  [Se confidenza < 70%: "Considerare valutazione specialistica aggiuntiva"]
  [Se confidenza ≥ 70%: "Classificazione sufficientemente affidabile"]

OUTPUT FINALE:
Classe: [C]
Confidenza: [Z]%
"""

def prompt_pi_few_shot(row: pd.Series, examples: list[dict]) -> str:
    context = build_patient_context(row, include_factors=False)
    ex_text = "\n\n".join([
        f"ESEMPIO {i+1}:\nScore MPQ: {ex['scores']}\nPRI totale: {ex['pri']}\n"
        f"Classificazione: Classe {ex['class']} "
        f"({'lieve' if ex['class']==0 else 'moderato' if ex['class']==1 else 'severo'})"
        for i, ex in enumerate(examples)
    ])
    return (
        f"Ecco alcuni esempi di classificazione dell'intensità del dolore:\n\n{ex_text}\n\n---\n\n"
        f"Dati del paziente:\n{context}\n\n"
        f"Classifica l'intensità del dolore seguendo gli esempi."
    )


# ─────────────────────────────────────────────
# PROMPT BUILDERS - ETIOLOGY (CoT Avanzate)
# ─────────────────────────────────────────────

ETIOLOGY_SYSTEM = (
    "Sei un neurologo esperto nella diagnosi differenziale del dolore oncologico. "
    "Il tuo compito è classificare l'eziologia del dolore di un paziente come:\n"
    "  - Classe 0: Nocicettivo (attivazione di afferenze nocicettive in tessuti somatici o viscerali)\n"
    "  - Classe 1: Neuropatico (dolore causato da lesione del sistema nervoso centrale o periferico)\n"
    "I dati del paziente sono già forniti nel messaggio utente: non chiedere ulteriori informazioni. "
    "Rispondi SEMPRE con 'Classe 0' (nocicettivo) o 'Classe 1' (neuropatico) alla fine della tua risposta."
)

def prompt_etio_zero_shot(row: pd.Series) -> str:
    descriptors = get_patient_descriptors(row)
    return (
        f"Il paziente ha scelto i seguenti descrittori verbali per il suo dolore:\n"
        f"{', '.join(descriptors)}\n\nClassifica l'eziologia del dolore di questo paziente."
    )

def prompt_etio_cot_simple(row: pd.Series) -> str:
    """CoT semplice per eziologia"""
    descriptors = get_patient_descriptors(row)
    return (
        f"Il paziente ha scelto i seguenti descrittori verbali per il suo dolore:\n"
        f"{', '.join(descriptors)}\n\n"
        f"Segui questi passi per classificare l'eziologia:\n"
        f"PASSO 1 - Analizza ogni descrittore e indica se è tipicamente associato a "
        f"dolore nocicettivo, neuropatico, o neutro.\n"
        f"PASSO 2 - Valuta il pattern complessivo dei descrittori. "
        f"Il dolore neuropatico tende ad avere descrittori come sensazioni elettriche, "
        f"bruciore, formicolio. Il nocicettivo tende ad avere descrittori di pressione, "
        f"trazione, pulsazione.\n"
        f"PASSO 3 - Decidi la classificazione in base al profilo dominante.\n\n"
        f"Mostra il ragionamento esplicito di ogni passo."
    )

def prompt_etio_cot_verification(row: pd.Series) -> str:
    """CoT con verifica per eziologia"""
    descriptors = get_patient_descriptors(row)
    return f"""
Descrittori del paziente: {', '.join(descriptors)}

FASE 1 - ANALISI INIZIALE:
Per ogni descrittore, classificalo come:
  [N] = Neuropatico tipico
  [NC] = Nocicettivo tipico  
  [NEUTRO] = Neutro/ambiguo

Elenco dettagliato:
{chr(10).join([f'  - {d}: [N/NC/NEUTRO] perché ...' for d in descriptors[:15]])}

Conteggio: Neuropatici = [X], Nocicettivi = [Y], Neutri = [Z]
Classificazione iniziale: [Neuropatico se X > Y, altrimenti Nocicettivo]

FASE 2 - VERIFICA CRITICA:
Considera pattern specifici:
  - Descrittorio "bruciore"/"shooting"/"tingling" → forte predittori neuropatici
  - Descrittorio "pressione"/"trazione"/"pesantezza" → forti predittori nocicettivi
  - Presenza di descrittori contraddittori: [elenca]

FASE 3 - VALUTAZIONE FINALE:
La classificazione iniziale è confermata? [Sì/No]
Se No, correggi a: [Classe corretta]

OUTPUT: Classe [0/1] (nocicettivo/neuropatico)
"""

def prompt_etio_few_shot(row: pd.Series, examples: list[dict]) -> str:
    descriptors = get_patient_descriptors(row)
    ex_text = "\n\n".join([
        f"ESEMPIO {i+1}:\nDescrittori: {', '.join(ex['descriptors'])}\n"
        f"Classificazione: Classe {ex['class']} ({'nocicettivo' if ex['class'] == 0 else 'neuropatico'})"
        for i, ex in enumerate(examples)
    ])
    return (
        f"Ecco alcuni esempi di classificazione dell'eziologia del dolore:\n\n{ex_text}\n\n---\n\n"
        f"Il paziente ha scelto i seguenti descrittori:\n{', '.join(descriptors)}\n\n"
        f"Classifica l'eziologia seguendo gli esempi."
    )

def prompt_etio_knowledge_guided(row: pd.Series) -> str:
    descriptors = get_patient_descriptors(row)
    coef_text = '\n'.join([
        f"  {desc}: {'presente' if desc in descriptors else 'assente'} "
        f"(coefficiente: {coef:+.3f})"
        for desc, coef in WILKIE_COEFFICIENTS.items()
    ])
    return (
        f"Utilizza la formula predittiva di Wilkie et al. (2001) per classificare "
        f"l'eziologia del dolore.\n\n"
        f"FORMULA: Predictive Score = 0.815 + somma(coefficiente × presenza_descrittore)\n"
        f"  - Se Score > 0: Nocicettivo (Classe 0)\n"
        f"  - Se Score <= 0: Neuropatico (Classe 1)\n\n"
        f"Descrittori del paziente e loro presenza:\n{coef_text}\n\n"
        f"Calcola il Predictive Score sommando intercetta (0.815) e i coefficienti "
        f"dei descrittori PRESENTI, poi classifica."
    )

def build_few_shot_examples(df: pd.DataFrame, test_idx: int, n: int = 2) -> dict:
    train_df = df.drop(index=test_idx).reset_index(drop=True)
    true_labels = get_true_labels(train_df)
    pi_examples, etio_examples = [], []
    seen_pi, seen_etio = set(), set()

    for i, row in train_df.iterrows():
        pi_cls = true_labels['pain_intensity'][i]
        etio_cls = true_labels['etiology'][i]

        if pi_cls not in seen_pi and len(pi_examples) < n:
            scores_str = ', '.join(f"{c.replace('_score','')}: {row[c]}" for c in SCORE_COLS)
            pi_examples.append({'scores': scores_str, 'pri': row['total_pain_score'], 'class': pi_cls})
            seen_pi.add(pi_cls)

        if etio_cls not in seen_etio and len(etio_examples) < n:
            etio_examples.append({'descriptors': get_patient_descriptors(row), 'class': etio_cls})
            seen_etio.add(etio_cls)

    return {'pain_intensity': pi_examples, 'etiology': etio_examples}


# ─────────────────────────────────────────────
# LLM CLIENT & CACHING
# ─────────────────────────────────────────────

def load_cache() -> dict:
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_cache(cache: dict):
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=4)

def get_cache_key(model: str, system: str, user: str) -> str:
    s = f"{model}|{system}|{user}"
    return hashlib.md5(s.encode('utf-8')).hexdigest()

def build_client_async(mode: Literal["local", "cloud"], base_url: str = None):
    if mode == "local":
        return AsyncOpenAI(base_url=base_url, api_key=LOCAL_API_KEY, timeout=600.0)
    else:
        return None  # client cloud costruito dinamicamente in async_call_llm

def _get_cloud_client(api_key: str) -> AsyncOllamaClient:
    if api_key not in _cloud_clients:
        _cloud_clients[api_key] = AsyncOllamaClient(
            host='https://ollama.com',
            headers={'Authorization': 'Bearer ' + api_key}
        )
    return _cloud_clients[api_key]

async def async_call_llm(client, mode: str, model: str, system: str, user: str, cache_dict: dict, max_tokens: int = 2048) -> str:
    """Chiama il modello LLM con round-robin corretto delle chiavi cloud.

    FIX rispetto alla versione precedente:
    - api_key viene catturata PRIMA dell'await e riusata nell'except (elimina race condition
      che marcava chiavi sbagliate come esaurite in presenza di chiamate parallele)
    - CLOUD_KEY_IDX incrementato PRIMA della chiamata → round-robin reale, non
      tutte le coroutine sulla stessa chiave
    - Session limit gestito con pausa+retry invece di essere trattato come errore fatale
    """
    global CLOUD_SEMAPHORE, CLOUD_KEY_IDX, _exhausted_keys, _all_keys_exhausted
    cache_key = get_cache_key(model, system, user)

    if cache_key in cache_dict:
        return cache_dict[cache_key]

    n_keys = len(CLOUD_API_KEYS)
    max_attempts = 5

    for attempt in range(max_attempts):
        api_key = None  # traccia la chiave usata in questo tentativo
        try:
            if mode == "local":
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user}
                    ],
                    max_tokens=max_tokens,
                    temperature=0.0
                )
                res_text = response.choices[0].message.content.strip()
            else:
                active_keys = [k for k in CLOUD_API_KEYS if k not in _exhausted_keys]
                if not active_keys:
                    if not _all_keys_exhausted:
                        _all_keys_exhausted = True
                        msg = "🔴 <b>Tutte le chiavi cloud esaurite.</b> Script fermato."
                        print(f"  [KEY] {msg}", flush=True)
                        send_telegram(msg)
                    return ""

                # Round-robin: seleziona e incrementa PRIMA dell'await
                # così coroutine concorrenti usano chiavi diverse
                api_key = active_keys[CLOUD_KEY_IDX % len(active_keys)]
                CLOUD_KEY_IDX += 1

                async with CLOUD_SEMAPHORE:
                    async with _get_key_semaphore(api_key):
                        # Jitter solo per chiavi free (PRO non ne ha bisogno)
                        if api_key not in PRO_API_KEYS:
                            await asyncio.sleep(random.uniform(2.0, 4.0))
                        response = await _get_cloud_client(api_key).chat(
                            model=model,
                            messages=[
                                {"role": "system", "content": system},
                                {"role": "user", "content": user}
                            ],
                            stream=False,
                            options={'temperature': 0.0, 'num_predict': max_tokens}
                        )
                if hasattr(response, 'message'):
                    res_text = response.message.content.strip()
                else:
                    res_text = response.get('message', {}).get('content', '').strip()

            cache_dict[cache_key] = res_text
            return res_text

        except Exception as e:
            err_str = str(e)
            is_weekly  = 'weekly' in err_str.lower() or 'daily' in err_str.lower()
            is_session = 'session' in err_str.lower()
            is_rate_err = '429' in err_str or '403' in err_str or '401' in err_str

            if mode == "cloud" and api_key is not None:
                n_active = len([k for k in CLOUD_API_KEYS if k not in _exhausted_keys])
                if is_weekly:
                    # Esaurisce SOLO la chiave che ha effettivamente fatto la chiamata
                    reason = "DAILY_LIMIT" if "daily" in err_str.lower() else "WEEKLY_LIMIT"
                    _mark_key_exhausted(api_key, reason)
                    n_rem = len(CLOUD_API_KEYS) - len(_exhausted_keys)
                    print(f"  [KEY-ESAURITA] {_key_name(api_key)}: {reason}. Attive: {n_rem}/{n_keys}", flush=True)
                elif is_session and attempt < max_attempts - 1:
                    # Session limit: si resetta in pochi minuti, aspetta e riprova
                    print(f"  [KEY] {model}: session limit — pausa 60s ({n_active} attive)", flush=True)
                    await asyncio.sleep(60)
                    continue
                elif is_rate_err and attempt < max_attempts - 1:
                    err_tag = "429" if '429' in err_str else ("401" if '401' in err_str else "403")
                    print(f"  [KEY] {model}: {err_tag} — riprovo ({n_active} attive)", flush=True)
                    await asyncio.sleep(2)
                    continue

            print(f"  [ERRORE] {model}: {err_str[:80]}")
            return ""


# ─────────────────────────────────────────────
# CLASSIFICAZIONE (ASYNC BATCHING)
# ─────────────────────────────────────────────

# Token limit calibrati con _test_strategies.py (10 modelli × 13 strategie × 5 pazienti)
# Valori = ceil(P95_chars / 3.0) + 80, cappati a 1500 per bilanciare quota vs copertura
_PI_TOKENS = {
    'zero_shot':         250,
    'cot_simple':        510,
    'cot_hierarchical': 1100,
    'cot_verification':  850,
    'cot_react':         800,
    'cot_ensemble':      970,
    'cot_decision_tree': 350,
    'cot_confidence':   1200,
    'few_shot':          330,
}
_ETIO_TOKENS = {
    'zero_shot':        700,
    'cot_simple':      1500,
    'cot_verification':1030,
    'few_shot':         750,
    'knowledge_guided': 500,
}


def _pred_is_filled(val) -> bool:
    """True se val è una predizione valida: non None, non NaN, non stringa vuota."""
    if val is None:
        return False
    try:
        if pd.isna(val):
            return False
    except (TypeError, ValueError):
        pass
    return not (isinstance(val, str) and val.strip() == '')


async def process_patient_async(i, row, df, true_labels, client, mode, model, model_label, cache_dict, semaphore, existing_record=None):
    async with semaphore:
        # Inizializza il record: parte dai dati esistenti (se parziale) o da zero
        if existing_record:
            record = dict(existing_record)
        else:
            record = {
                'patient_id': row['ID'],
                'model': model_label,
                'true_pain_intensity': true_labels['pain_intensity'][i],
                'true_etiology': true_labels['etiology'][i],
            }

        few_shot_ex = build_few_shot_examples(df, i, n=2)

        # ── PAIN INTENSITY: costruisce solo le strategie con pred mancante ──
        all_pi = {'zero_shot': prompt_pi_zero_shot(row)}
        if USE_COT_STRATEGIES.get('simple', True):
            all_pi['cot_simple'] = prompt_pi_cot_simple(row)
        if USE_COT_STRATEGIES.get('hierarchical', True):
            all_pi['cot_hierarchical'] = prompt_pi_cot_hierarchical(row)
        if USE_COT_STRATEGIES.get('verification', True):
            all_pi['cot_verification'] = prompt_pi_cot_verification(row)
        if USE_COT_STRATEGIES.get('react', False):
            all_pi['cot_react'] = prompt_pi_cot_react(row)
        if USE_COT_STRATEGIES.get('ensemble', True):
            all_pi['cot_ensemble'] = prompt_pi_cot_ensemble(row)
        if USE_COT_STRATEGIES.get('decision_tree', True):
            all_pi['cot_decision_tree'] = prompt_pi_cot_decision_tree(row)
        if USE_COT_STRATEGIES.get('confidence', True):
            all_pi['cot_confidence'] = prompt_pi_cot_confidence(row)
        all_pi['few_shot'] = prompt_pi_few_shot(row, few_shot_ex['pain_intensity'])

        strategies_pi = {
            name: prompt for name, prompt in all_pi.items()
            if not _pred_is_filled(record.get(f'pi_{name}_pred'))
        }

        # ── ETIOLOGY: costruisce solo le strategie con pred mancante ──
        all_etio = {
            'zero_shot':        prompt_etio_zero_shot(row),
            'cot_simple':       prompt_etio_cot_simple(row),
            'cot_verification': prompt_etio_cot_verification(row),
            'few_shot':         prompt_etio_few_shot(row, few_shot_ex['etiology']),
            'knowledge_guided': prompt_etio_knowledge_guided(row),
        }

        strategies_etio = {
            name: prompt for name, prompt in all_etio.items()
            if not _pred_is_filled(record.get(f'etio_{name}_pred'))
        }

        # Se non manca nulla, restituisce il record com'è
        if not strategies_pi and not strategies_etio:
            return record

        # Lancia in parallelo SOLO le strategie mancanti
        pi_names = list(strategies_pi.keys())
        etio_names = list(strategies_etio.keys())

        all_coros = (
            [async_call_llm(client, mode, model, PAIN_INTENSITY_SYSTEM, prompt, cache_dict,
                            max_tokens=_PI_TOKENS.get(name, 700))
             for name, prompt in strategies_pi.items()]
            +
            [async_call_llm(client, mode, model, ETIOLOGY_SYSTEM, prompt, cache_dict,
                            max_tokens=_ETIO_TOKENS.get(name, 500))
             for name, prompt in strategies_etio.items()]
        )
        all_resps = await asyncio.gather(*all_coros)

        pi_resps   = all_resps[:len(pi_names)]
        etio_resps = all_resps[len(pi_names):]

        for name, resp in zip(pi_names, pi_resps):
            pred = parse_class_from_response(resp, 'pain_intensity')
            record[f'pi_{name}_pred']     = pred
            record[f'pi_{name}_response'] = resp

        for name, resp in zip(etio_names, etio_resps):
            pred = parse_class_from_response(resp, 'etiology')
            record[f'etio_{name}_pred']     = pred
            record[f'etio_{name}_response'] = resp

        print(f"  Paziente {i+1} completato.", flush=True)
        return record


# Mantieni le altre funzioni (classify_dataset_async, compute_metrics, generate_report, main)
# uguali al codice originale...

async def classify_dataset_async(
    df: pd.DataFrame, client, mode: str, model: str, model_label: str,
    checkpoint_path: str, cache_dict: dict, checkpoint_lock: asyncio.Lock,
    progress: dict
) -> pd.DataFrame:
    true_labels = get_true_labels(df)

    print(f"\n{'='*60}")
    print(f"Modello: {model_label}")
    print(f"{'='*60}")

    # Carica checkpoint esistente per questo modello.
    # Un paziente è "done" solo se TUTTE le pred sono compilate.
    # Pazienti parziali vengono riprocessati solo per le strategie mancanti.
    done_ids = set()
    partial_records = {}  # patient_id → record dict per pazienti parzialmente compilati
    results = []
    if os.path.exists(checkpoint_path):
        chk = pd.read_csv(checkpoint_path)
        existing = chk[chk['model'] == model_label]
        if not existing.empty:
            pred_cols = [c for c in existing.columns if c.endswith('_pred')]
            if pred_cols:
                fully_done_mask = existing[pred_cols].apply(
                    lambda col: col.map(_pred_is_filled)
                ).all(axis=1)
            else:
                fully_done_mask = pd.Series([False] * len(existing), index=existing.index)
            fully_done = existing[fully_done_mask]
            partial = existing[~fully_done_mask]
            done_ids = set(fully_done['patient_id'].tolist())
            results = fully_done.to_dict('records')
            partial_records = {
                r['patient_id']: r
                for r in partial.to_dict('records')
            }
            n_full = len(fully_done)
            n_part = len(partial)
            if n_part > 0:
                print(f"  Checkpoint: {n_full} completi, {n_part} parziali → solo strategie mancanti")
            else:
                print(f"  Checkpoint trovato: {n_full} pazienti completi, riprendo...")

    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    total = len(df)

    for i, row in df.iterrows():
        if row['ID'] in done_ids:
            continue

        if _all_keys_exhausted:
            print(f"  [{model_label}] Chiavi esaurite — interrompo.", flush=True)
            break

        existing_rec = partial_records.get(row['ID'])
        n_missing = 0
        if existing_rec:
            pred_cols_all = [k for k in existing_rec if k.endswith('_pred')]
            n_missing = sum(1 for c in pred_cols_all if not _pred_is_filled(existing_rec.get(c)))
            print(f"  [{model_label}] Paziente {len(results)+1}/{total} (parziale, {n_missing} pred mancanti)...", flush=True)
        else:
            print(f"  [{model_label}] Paziente {len(results)+1}/{total}...", flush=True)
        try:
            record = await process_patient_async(
                i, row, df, true_labels, client, mode, model, model_label, cache_dict, semaphore,
                existing_record=existing_rec
            )
        except Exception as e:
            err_msg = f"[{model_label}] paziente {i}: {e}"
            print(f"  ERRORE {err_msg}")
            progress['errors'].append(err_msg)
            send_telegram(f"⚠️ <b>ERRORE runtime</b>\n{err_msg}")
            continue

        results.append(record)
        progress['patients_done'][model_label] = len(results)

        # Salva checkpoint ogni 10 pazienti (o all'ultimo)
        if len(results) % 10 == 0 or len(results) + len(done_ids) == total:
            async with checkpoint_lock:
                chk_df = pd.DataFrame(results)
                if os.path.exists(checkpoint_path):
                    existing_other = pd.read_csv(checkpoint_path)
                    existing_other = existing_other[existing_other['model'] != model_label]
                    chk_df = pd.concat([existing_other, chk_df], ignore_index=True)
                chk_df.to_csv(checkpoint_path, index=False)
                save_cache(cache_dict)

    progress['models_done'].append(model_label)
    return pd.DataFrame([r for r in results if r.get('model') == model_label])


async def run_all_models_async(
    df: pd.DataFrame, models_to_test: list, checkpoint_path: str, progress: dict
) -> tuple[list, dict]:
    """Esegue tutti i modelli in parallelo con cache e checkpoint condivisi."""
    global CLOUD_SEMAPHORE, _KEY_SEMAPHORES, _cloud_clients
    CLOUD_SEMAPHORE = asyncio.Semaphore(CLOUD_CONCURRENT_CALLS)
    _KEY_SEMAPHORES = {}   # reset: i semafori del loop precedente causano "bound to a different event loop"
    _cloud_clients = {}    # reset: i client del loop precedente causano "Event loop is closed"
    cache_dict = load_cache()
    checkpoint_lock = asyncio.Lock()

    tasks = []
    for mode, model_name, model_label, base_url in models_to_test:
        client = build_client_async(mode, base_url)
        tasks.append(
            classify_dataset_async(
                df, client, mode, model_name, model_label,
                checkpoint_path, cache_dict, checkpoint_lock, progress
            )
        )

    print(f"\nAvvio {len(tasks)} modelli in parallelo...")
    notify_task = asyncio.create_task(periodic_notify(progress))
    try:
        results_list = await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        notify_task.cancel()

    all_results = []
    all_metrics = {}
    for (_, _, model_label, _), result in zip(models_to_test, results_list):
        if isinstance(result, Exception):
            err = f"{model_label}: {result}"
            print(f"  [ERRORE] {err}")
            progress['errors'].append(err)
            send_telegram(f"⚠️ <b>ERRORE modello</b>\n{err}")
            continue
        all_results.append(result)
        all_metrics[model_label] = compute_metrics(result, model_label)

    return all_results, all_metrics


def compute_metrics(results_df: pd.DataFrame, model_label: str) -> dict:
    model_df = results_df[results_df['model'] == model_label]
    metrics = {}
    
    # Dinamicamente costruisci le strategie in base a quelle presenti
    pi_strategies = [col.replace('pi_', '').replace('_pred', '') 
                     for col in model_df.columns 
                     if col.startswith('pi_') and col.endswith('_pred')]
    etio_strategies = [col.replace('etio_', '').replace('_pred', '') 
                       for col in model_df.columns 
                       if col.startswith('etio_') and col.endswith('_pred')]

    for task, strategies in [('pain_intensity', pi_strategies), ('etiology', etio_strategies)]:
        true_col = f'true_{task}'
        y_true = model_df[true_col].tolist()

        for strategy in strategies:
            pred_col = f'pi_{strategy}_pred' if task == 'pain_intensity' else f'etio_{strategy}_pred'
            if pred_col not in model_df.columns:
                continue
                
            y_pred = model_df[pred_col].tolist()

            valid = [(yt, yp) for yt, yp in zip(y_true, y_pred) if not pd.isna(yp)]
            if not valid:
                metrics[f'{task}_{strategy}'] = {'accuracy': None, 'f1': None, 'n_valid': 0}
                continue

            yt_v, yp_v = zip(*valid)
            avg = 'macro' if task == 'pain_intensity' else 'binary'
            acc = accuracy_score(yt_v, yp_v)
            f1 = f1_score(yt_v, yp_v, average=avg, zero_division=0)

            metrics[f'{task}_{strategy}'] = {
                'accuracy': round(acc, 3),
                'f1': round(f1, 3),
                'n_valid': len(valid),
                'n_total': len(y_true),
                'confusion_matrix': confusion_matrix(yt_v, yp_v).tolist(),
                'classification_report': classification_report(yt_v, yp_v, zero_division=0)
            }
    return metrics


def generate_report(all_results: pd.DataFrame, all_metrics: dict) -> str:
    lines = [
        "=" * 70,
        "REPORT: LLM vs McGill Pain Questionnaire Classification",
        "=" * 70,
        ""
    ]
    models = all_results['model'].unique()

    for model_label in models:
        lines += [f"\n{'─'*60}", f"Modello: {model_label}", f"{'─'*60}"]
        metrics = all_metrics[model_label]

        lines += ["", "PAIN INTENSITY (3 classi: 0=lieve, 1=moderato, 2=severo)"]
        lines.append(f"{'Strategia':<25} {'Accuracy':>10} {'F1-macro':>10} {'Valid/Tot':>12}")
        lines.append("-" * 57)
        
        for strategy in sorted([k.split('_',1)[1] for k in metrics.keys() if k.startswith('pain_intensity_')]):
            m = metrics.get(f'pain_intensity_{strategy}', {})
            acc = f"{m['accuracy']:.3f}" if m.get('accuracy') is not None else "N/A"
            f1  = f"{m['f1']:.3f}"       if m.get('f1') is not None else "N/A"
            n   = f"{m.get('n_valid','?')}/{m.get('n_total','?')}"
            lines.append(f"{strategy:<25} {acc:>10} {f1:>10} {n:>12}")

        lines += ["", "ETIOLOGY (0=nocicettivo, 1=neuropatico)"]
        lines.append(f"{'Strategia':<25} {'Accuracy':>10} {'F1-binary':>10} {'Valid/Tot':>12}")
        lines.append("-" * 57)
        
        for strategy in sorted([k.split('_',1)[1] for k in metrics.keys() if k.startswith('etiology_')]):
            m = metrics.get(f'etiology_{strategy}', {})
            acc = f"{m['accuracy']:.3f}" if m.get('accuracy') is not None else "N/A"
            f1  = f"{m['f1']:.3f}"       if m.get('f1') is not None else "N/A"
            n   = f"{m.get('n_valid','?')}/{m.get('n_total','?')}"
            lines.append(f"{strategy:<25} {acc:>10} {f1:>10} {n:>12}")

        lines += ["", "MATRICI DI CONFUSIONE:"]
        for task_strat, m in metrics.items():
            if m.get('confusion_matrix'):
                lines.append(f"\n  {task_strat}:")
                for row in m['confusion_matrix']:
                    lines.append(f"    {row}")

    lines += ["", "=" * 70, "Fine report", "=" * 70]
    return '\n'.join(lines)


def main():
    global FACTOR_COLS, CACHE_FILE
    import pathlib, sys
    pathlib.Path(OUTPUT_DIR).mkdir(exist_ok=True)
    pathlib.Path(CHECKPOINT_DIR).mkdir(exist_ok=True)

    _load_key_state()   # ripristina chiavi esaurite da run precedenti

    # Parsing gruppo (--group local|remote|cloud|all)
    parser = argparse.ArgumentParser(description="MPQ LLM Classifier", add_help=False)
    parser.add_argument("--group", choices=["local", "remote", "cloud", "all"], default="all")
    parser.add_argument("--model", default=None,
                        help="Esegui solo questo modello cloud (es. gpt-oss:20b)")
    parser.add_argument("--keys-idx", default=None,
                        help="Indici chiavi da usare, separati da virgola (es. 0,1,2,3)")
    args, _ = parser.parse_known_args()
    GROUP = args.group
    GROUP_SUFFIX = f"_{GROUP}" if GROUP != "all" else ""
    SINGLE_MODEL = args.model
    KEYS_IDX = [int(i) for i in args.keys_idx.split(",")] if args.keys_idx else None

    # Filtra le chiavi se specificato --keys-idx
    global CLOUD_API_KEYS
    if KEYS_IDX is not None:
        CLOUD_API_KEYS = [CLOUD_API_KEYS[i] for i in KEYS_IDX]

    # File specifici per gruppo/modello (cache, log, err)
    if SINGLE_MODEL:
        _model_safe = SINGLE_MODEL.replace(":", "_").replace("/", "_")
        _file_suffix = f"_{_model_safe}"
    else:
        _file_suffix = GROUP_SUFFIX
    CACHE_FILE = f"llm_cache{_file_suffix}.json"
    _log = open(f"run_log{_file_suffix}.txt", "w", buffering=-1, encoding="utf-8")
    _err = open(f"run_err{_file_suffix}.txt", "w", buffering=-1, encoding="utf-8")
    sys.stdout = _log
    sys.stderr = _err

    # Costruzione lista modelli per gruppo
    models_to_test = []
    if GROUP == "all":
        if MODE in ("local", "both"):
            for m in LOCAL_MODELS:
                base_url = LOCAL_MODEL_HOSTS.get(m, "http://localhost:11434/v1")
                models_to_test.append(("local", m, f"Local/{m}", base_url))
        if MODE in ("cloud", "both"):
            for m in CLOUD_MODELS:
                models_to_test.append(("cloud", m, f"Cloud/{m}", None))
    elif GROUP == "local":
        for m in MODEL_GROUPS["local"]:
            base_url = LOCAL_MODEL_HOSTS.get(m, "http://localhost:11434/v1")
            models_to_test.append(("local", m, f"Local/{m}", base_url))
    elif GROUP == "remote":
        for m in MODEL_GROUPS["remote"]:
            if m == "qwen2.5:7b":
                base_url = REMOTE_QWEN_HOST
            else:
                base_url = LOCAL_MODEL_HOSTS.get(m, "http://localhost:11434/v1")
            models_to_test.append(("local", m, f"Local/{m}", base_url))
    elif GROUP == "cloud":
        for m in CLOUD_MODELS:
            models_to_test.append(("cloud", m, f"Cloud/{m}", None))

    # Se --model è specificato, tieni solo quel modello
    if SINGLE_MODEL:
        models_to_test = [(m, n, l, h) for m, n, l, h in models_to_test if n == SINGLE_MODEL]
        if not models_to_test:
            print(f"[ERRORE] Modello '{SINGLE_MODEL}' non trovato nel gruppo '{GROUP}'.")
            import sys; sys.exit(1)

    model_labels = [label for _, _, label, _ in models_to_test]
    print(f"Strategie CoT attive: {[k for k, v in USE_COT_STRATEGIES.items() if v]}")
    print(f"Modelli: {model_labels}")
    print(f"Dataset: {len(DATASET_PATHS)}\n")

    start_time = datetime.datetime.now()
    send_telegram(
        f"🚀 <b>MPQ LLM avviato</b>\n"
        f"Dataset: {len(DATASET_PATHS)}\n"
        f"Modelli: {len(models_to_test)} ({len(LOCAL_MODELS)} locali + {len(CLOUD_MODELS)} cloud)\n"
        f"Ora: {start_time.strftime('%H:%M %d/%m/%Y')}"
    )

    datasets_done = []

    for dataset_path in DATASET_PATHS:
        dataset_name = pathlib.Path(dataset_path).stem
        print(f"\n{'#'*70}")
        print(f"DATASET: {dataset_name}")
        print(f"{'#'*70}")

        try:
            df = normalize_dataframe(pd.read_csv(dataset_path))
        except FileNotFoundError:
            print(f"  [SKIP] File non trovato: {dataset_path}")
            send_telegram(f"⚠️ File non trovato: {dataset_path}")
            continue
        except Exception as e:
            print(f"  [SKIP] Errore caricamento {dataset_path}: {e}")
            send_telegram(f"⚠️ Errore caricamento <code>{dataset_name}</code>: {e}")
            continue

        # Imposta le colonne factor per questo dataset
        FACTOR_COLS = [c for c in df.columns if c.startswith('factor_')]
        print(f"  {len(df)} pazienti, {len(df.columns)} colonne, {len(FACTOR_COLS)} fattori modulatori")

        progress = {
            'start_time': start_time,
            'dataset_current': dataset_name,
            'datasets_done': datasets_done,
            'datasets_total': len(DATASET_PATHS),
            'patients_total': len(df),
            'patients_done': {},
            'models_total': len(models_to_test),
            'models_done': [],
            'errors': [],
        }

        send_telegram(
            f"📂 <b>Nuovo dataset</b>: <code>{dataset_name}</code>\n"
            f"{len(df)} pazienti | {len(models_to_test)} modelli\n"
            f"Dataset {len(datasets_done)+1}/{len(DATASET_PATHS)}"
        )

        if SINGLE_MODEL:
            _ms = SINGLE_MODEL.replace(":", "_").replace("/", "_")
            checkpoint_path = f"{CHECKPOINT_DIR}/checkpoint_{dataset_name}_{_ms}.csv"
        else:
            checkpoint_path = f"{CHECKPOINT_DIR}/checkpoint_{dataset_name}{GROUP_SUFFIX}.csv"

        all_results, all_metrics = asyncio.run(
            run_all_models_async(df, models_to_test, checkpoint_path, progress)
        )

        if _all_keys_exhausted:
            send_telegram("🔴 <b>Script terminato</b>: tutte le chiavi cloud esaurite. Riavvia quando le chiavi si rinnovano.")
            print("\n[STOP] Tutte le chiavi esaurite. Script terminato.")
            import sys; sys.exit(0)

        if all_results:
            final_results = pd.concat(all_results, ignore_index=True)
            out_csv = f"{OUTPUT_DIR}/results_{dataset_name}.csv"
            out_report = f"{OUTPUT_DIR}/report_{dataset_name}.txt"

            final_results.to_csv(out_csv, index=False)
            print(f"\nRisultati salvati in: {out_csv}")

            report = generate_report(final_results, all_metrics)
            with open(out_report, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"Report salvato in: {out_report}")
            print("\n" + report)

            datasets_done.append(dataset_name)
            n_errors = len(progress.get('errors', []))
            send_telegram(
                f"✅ <b>Dataset completato</b>: <code>{dataset_name}</code>\n"
                f"Pazienti: {len(df)} | Modelli: {len(all_results)}/{len(models_to_test)}\n"
                f"Errori: {n_errors}\n"
                f"Progresso totale: {len(datasets_done)}/{len(DATASET_PATHS)} dataset"
            )
        else:
            send_telegram(f"⚠️ <b>Dataset fallito</b>: <code>{dataset_name}</code> — nessun risultato")

    elapsed = datetime.datetime.now() - start_time
    h, rem = divmod(int(elapsed.total_seconds()), 3600)
    send_telegram(
        f"🏁 <b>Esecuzione completata</b>\n"
        f"Durata totale: {h}h {rem//60}m\n"
        f"Dataset processati: {len(datasets_done)}/{len(DATASET_PATHS)}"
    )

if __name__ == "__main__":
    import sys, traceback, atexit

    def _flush_logs():
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass

    atexit.register(_flush_logs)

    try:
        main()
    except Exception:
        # Scrivi traceback su file di crash indipendente dal redirect
        with open("crash.log", "w", encoding="utf-8") as cf:
            traceback.print_exc(file=cf)
        try:
            traceback.print_exc()  # anche su stderr rediretto
        except Exception:
            pass
        raise