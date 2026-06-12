"""
fix_gaps.py — completa le lacune residue nei checkpoint.

Interventi:
  1. DOCTORAI: pazienti 491-498 (mai processati) per tutti i modelli
  2. gemma3:4b pi_cot_decision_tree: token 350 → 800
  3. rnj-1:8b pi_few_shot: prompt con output esplicito (modello piccolo)
  4. Tutte le lacune transient (<= tutte) per ogni modello

Uso: python fix_gaps.py
"""
import asyncio, os, pathlib, sys
import pandas as pd

os.chdir(pathlib.Path(__file__).parent)
sys.stdout.reconfigure(encoding="utf-8")

import ask_llm_multiple_CoT as M

# ── Modelli attivi ───────────────────────────────────────────────────────────
MODELS = [
    "gemma4:31b", "gemma3:27b", "ministral-3:14b", "gemma3:4b",
    "gemma3:12b", "rnj-1:8b", "ministral-3:8b", "devstral-small-2:24b",
]

DATASET_PATHS = M.DATASET_PATHS
CHECKPOINT_DIR = M.CHECKPOINT_DIR

# ── Token override per gemma3:4b ─────────────────────────────────────────────
TOKEN_OVERRIDES = {
    "gemma3:4b": {"cot_decision_tree": 800},
}

# ── Prompt pi_few_shot con output esplicito per rnj-1:8b ─────────────────────
def _pi_few_shot_explicit(row, examples):
    base = M.prompt_pi_few_shot(row, examples)
    return (
        base
        + "\n\nATTENZIONE: Devi rispondere OBBLIGATORIAMENTE con una sola riga nel formato esatto:\n"
        "Classe 0   oppure   Classe 1   oppure   Classe 2\n"
        "Scrivi SOLO la classe, nessuna spiegazione aggiuntiva."
    )

PROMPT_OVERRIDES_PI = {
    ("rnj-1:8b", "few_shot"): _pi_few_shot_explicit,
}


# ── Processa paziente: solo strategie mancanti ───────────────────────────────
async def fix_patient(
    df: pd.DataFrame,
    pid: int,
    model_short: str,
    existing: dict,
    cache_dict: dict,
    semaphore: asyncio.Semaphore,
) -> dict:
    async with semaphore:
        # Individua la riga del paziente nel DataFrame
        mask = df['ID'] == pid
        if not mask.any():
            print(f"  WARN: paziente {pid} non trovato nel dataset", flush=True)
            return existing or {}

        i   = df.index[mask][0]
        row = df.loc[i]
        true_labels = M.get_true_labels(df)
        few_shot_ex = M.build_few_shot_examples(df, i, n=2)

        record = dict(existing) if existing else {
            'patient_id':          pid,
            'model':               f"Cloud/{model_short}",
            'true_pain_intensity': true_labels['pain_intensity'][i],
            'true_etiology':       true_labels['etiology'][i],
        }

        tok_ov = TOKEN_OVERRIDES.get(model_short, {})

        # ── Strategie PI ────────────────────────────────────────────────────
        all_pi = {'zero_shot': M.prompt_pi_zero_shot(row)}
        if M.USE_COT_STRATEGIES.get('simple',        True): all_pi['cot_simple']       = M.prompt_pi_cot_simple(row)
        if M.USE_COT_STRATEGIES.get('hierarchical',  True): all_pi['cot_hierarchical'] = M.prompt_pi_cot_hierarchical(row)
        if M.USE_COT_STRATEGIES.get('verification',  True): all_pi['cot_verification'] = M.prompt_pi_cot_verification(row)
        if M.USE_COT_STRATEGIES.get('ensemble',      True): all_pi['cot_ensemble']     = M.prompt_pi_cot_ensemble(row)
        if M.USE_COT_STRATEGIES.get('decision_tree', True): all_pi['cot_decision_tree']= M.prompt_pi_cot_decision_tree(row)
        if M.USE_COT_STRATEGIES.get('confidence',    True): all_pi['cot_confidence']   = M.prompt_pi_cot_confidence(row)

        fs_key = (model_short, "few_shot")
        if fs_key in PROMPT_OVERRIDES_PI:
            all_pi['few_shot'] = PROMPT_OVERRIDES_PI[fs_key](row, few_shot_ex['pain_intensity'])
        else:
            all_pi['few_shot'] = M.prompt_pi_few_shot(row, few_shot_ex['pain_intensity'])

        strat_pi = {n: p for n, p in all_pi.items()
                    if not M._pred_is_filled(record.get(f'pi_{n}_pred'))}

        # ── Strategie Etio ───────────────────────────────────────────────────
        all_etio = {
            'zero_shot':        M.prompt_etio_zero_shot(row),
            'cot_simple':       M.prompt_etio_cot_simple(row),
            'cot_verification': M.prompt_etio_cot_verification(row),
            'few_shot':         M.prompt_etio_few_shot(row, few_shot_ex['etiology']),
            'knowledge_guided': M.prompt_etio_knowledge_guided(row),
        }
        strat_etio = {n: p for n, p in all_etio.items()
                      if not M._pred_is_filled(record.get(f'etio_{n}_pred'))}

        if not strat_pi and not strat_etio:
            return record  # già completo

        pi_names   = list(strat_pi.keys())
        etio_names = list(strat_etio.keys())

        def _tok(name):
            return tok_ov.get(name, M._PI_TOKENS.get(name, 700))

        coros = (
            [M.async_call_llm(None, "cloud", model_short,
                              M.PAIN_INTENSITY_SYSTEM, p, cache_dict,
                              max_tokens=_tok(n))
             for n, p in strat_pi.items()]
            +
            [M.async_call_llm(None, "cloud", model_short,
                              M.ETIOLOGY_SYSTEM, p, cache_dict,
                              max_tokens=M._ETIO_TOKENS.get(n, 500))
             for n, p in strat_etio.items()]
        )
        try:
            resps = await asyncio.wait_for(asyncio.gather(*coros), timeout=90)
        except asyncio.TimeoutError:
            print(f"  TIMEOUT paziente {pid} [{model_short}] — skip", flush=True)
            return record

        for name, resp in zip(pi_names, resps[:len(pi_names)]):
            pred = M.parse_class_from_response(resp, 'pain_intensity')
            record[f'pi_{name}_pred']     = pred
            record[f'pi_{name}_response'] = resp

        for name, resp in zip(etio_names, resps[len(pi_names):]):
            pred = M.parse_class_from_response(resp, 'etiology')
            record[f'etio_{name}_pred']     = pred
            record[f'etio_{name}_response'] = resp

        missing_strat = len(strat_pi) + len(strat_etio)
        print(f"  [{model_short}] pz {pid} completato "
              f"({missing_strat} strategie richieste)", flush=True)
        return record


# ── Processa un dataset per tutti i modelli ──────────────────────────────────
async def fix_dataset(dataset_path: str, cache_dict: dict):
    dataset_name = pathlib.Path(dataset_path).stem
    chk_path     = f"{CHECKPOINT_DIR}/checkpoint_{dataset_name}_cloud.csv"

    try:
        df = M.normalize_dataframe(pd.read_csv(dataset_path))
    except Exception as e:
        print(f"[SKIP] Errore caricamento {dataset_path}: {e}")
        return

    all_ids = set(df['ID'].tolist())
    n_paz   = len(df)
    print(f"\n{'#'*65}")
    print(f"DATASET: {dataset_name}  ({n_paz} pazienti)")
    print(f"{'#'*65}")

    # Carica checkpoint corrente
    if os.path.exists(chk_path):
        chk = pd.read_csv(chk_path)
    else:
        chk = pd.DataFrame()

    # Lock per scrittura checkpoint
    lock = asyncio.Lock()

    for model_short in MODELS:
        model_label = f"Cloud/{model_short}"
        print(f"\n  Modello: {model_label}", flush=True)

        # Righe esistenti per questo modello
        if not chk.empty:
            existing_rows = chk[chk['model'] == model_label]
        else:
            existing_rows = pd.DataFrame()

        pred_cols = [c for c in existing_rows.columns if c.endswith('_pred')] if not existing_rows.empty else []

        # Pazienti completamente done (tutte le pred compilate)
        if not existing_rows.empty and pred_cols:
            done_mask = existing_rows[pred_cols].apply(
                lambda col: col.map(M._pred_is_filled)
            ).all(axis=1)
            fully_done  = existing_rows[done_mask]
            partial_rows = existing_rows[~done_mask]
        else:
            fully_done   = pd.DataFrame()
            partial_rows = existing_rows

        done_ids     = set(fully_done['patient_id'].tolist())
        partial_dict = {r['patient_id']: r for r in partial_rows.to_dict('records')} if not partial_rows.empty else {}
        results      = fully_done.to_dict('records')

        n_todo = len(all_ids - done_ids)
        if n_todo == 0:
            print(f"    Tutti i {len(done_ids)} pazienti completi — skip", flush=True)
            continue

        n_partial = len([pid for pid in all_ids if pid in partial_dict])
        n_new     = len(all_ids - done_ids) - n_partial
        print(f"    Completi: {len(done_ids)} | Parziali: {n_partial} | Nuovi: {n_new}", flush=True)

        semaphore = asyncio.Semaphore(M.CONCURRENCY_LIMIT)
        tasks = []
        for pid in sorted(all_ids - done_ids):
            existing_rec = partial_dict.get(pid)
            tasks.append(fix_patient(df, pid, model_short, existing_rec, cache_dict, semaphore))

        # Esegui in batch per salvare progressi
        BATCH = 20
        for b in range(0, len(tasks), BATCH):
            if M._all_keys_exhausted:
                print(f"  [{model_label}] Chiavi esaurite — interrompo.", flush=True)
                break
            batch_results = await asyncio.gather(*tasks[b:b+BATCH], return_exceptions=True)
            for res in batch_results:
                if isinstance(res, Exception):
                    print(f"  ERRORE batch: {res}", flush=True)
                elif res:
                    results.append(res)

            # Salva checkpoint
            async with lock:
                chk_df = pd.DataFrame(results)
                if os.path.exists(chk_path):
                    other = pd.read_csv(chk_path)
                    other = other[other['model'] != model_label]
                    chk_df = pd.concat([other, chk_df], ignore_index=True)
                chk_df.to_csv(chk_path, index=False)
                M.save_cache(cache_dict)
            print(f"    Checkpoint salvato ({len(results)} righe)", flush=True)


# ── Main ─────────────────────────────────────────────────────────────────────
async def main():
    # Inizializza i globali async del modulo principale
    M.CLOUD_SEMAPHORE    = asyncio.Semaphore(M.CLOUD_CONCURRENT_CALLS)
    M._KEY_SEMAPHORES    = {}
    M._cloud_clients     = {}
    M._exhausted_keys    = {}
    M._all_keys_exhausted = False
    M.CLOUD_KEY_IDX      = 0

    cache_dict = M.load_cache()

    for ds_path in DATASET_PATHS:
        await fix_dataset(ds_path, cache_dict)

    print("\n" + "="*65)
    print("fix_gaps.py completato.")
    print("="*65)


if __name__ == "__main__":
    asyncio.run(main())
