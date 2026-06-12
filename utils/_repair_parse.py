"""
Ri-parsa le risposte stored nei checkpoint che hanno pred=null ma response non-null.
Usa il parser migliorato (normalizzazione Unicode, formati extra).
Sicuro da eseguire mentre ask_llm_multiple_CoT.py gira.
"""
import os, re, sys, unicodedata
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

CHECKPOINT_DIR = 'checkpoints'

PI_PRED_RESP = [
    ('pi_zero_shot_pred',        'pi_zero_shot_response'),
    ('pi_cot_simple_pred',       'pi_cot_simple_response'),
    ('pi_cot_hierarchical_pred', 'pi_cot_hierarchical_response'),
    ('pi_cot_verification_pred', 'pi_cot_verification_response'),
    ('pi_cot_ensemble_pred',     'pi_cot_ensemble_response'),
    ('pi_cot_decision_tree_pred','pi_cot_decision_tree_response'),
    ('pi_cot_confidence_pred',   'pi_cot_confidence_response'),
    ('pi_few_shot_pred',         'pi_few_shot_response'),
]
ETIO_PRED_RESP = [
    ('etio_zero_shot_pred',        'etio_zero_shot_response'),
    ('etio_cot_simple_pred',       'etio_cot_simple_response'),
    ('etio_cot_verification_pred', 'etio_cot_verification_response'),
    ('etio_few_shot_pred',         'etio_few_shot_response'),
    ('etio_knowledge_guided_pred', 'etio_knowledge_guided_response'),
]


def parse_class(response: str, task: str):
    response = unicodedata.normalize('NFKC', response)
    r = response.lower()

    if task == 'pain_intensity':
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

    elif task == 'etiology':
        if any(w in r for w in ['neuropath', 'classe 1', 'class 1', 'neuropatico']):
            return 1
        if any(w in r for w in ['nociceptiv', 'classe 0', 'class 0', 'nocicettivo']):
            return 0
        if 'classe:' in r:
            after = r.split('classe:')[1][:10]
            if '1' in after: return 1
            if '0' in after: return 0
        # Fallback: ultima parola neuropatica/nocicettiva
        if any(w in r for w in ['neuropatic', 'neuropath', 'nervos', 'neural']):
            return 1
        if any(w in r for w in ['nocicet', 'tissut', 'infiamm', 'somatico']):
            return 0

    return None


def repair_file(path: str) -> int:
    df = pd.read_csv(path)
    fixed = 0

    pairs = PI_PRED_RESP + ETIO_PRED_RESP
    tasks = ['pain_intensity'] * len(PI_PRED_RESP) + ['etiology'] * len(ETIO_PRED_RESP)

    for (pred_c, resp_c), task in zip(pairs, tasks):
        if pred_c not in df.columns or resp_c not in df.columns:
            continue
        mask = df[pred_c].isna() & df[resp_c].notna()
        if mask.sum() == 0:
            continue
        for idx in df[mask].index:
            resp = str(df.at[idx, resp_c])
            result = parse_class(resp, task)
            if result is not None:
                df.at[idx, pred_c] = float(result)
                fixed += 1

    if fixed > 0:
        df.to_csv(path, index=False)

    return fixed


def main():
    files = sorted([f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.csv')])
    total_fixed = 0

    print(f"{'='*70}")
    print(f"  REPAIR PARSE — {len(files)} file da analizzare")
    print(f"{'='*70}")

    for fname in files:
        path = os.path.join(CHECKPOINT_DIR, fname)
        fixed = repair_file(path)
        short = fname.replace('checkpoint_', '').replace('.csv', '')
        if fixed > 0:
            print(f"  [FIXED] {short:<55} +{fixed}")
        else:
            print(f"  [  OK ] {short:<55} nessuna fix")
        total_fixed += fixed

    print(f"\n  TOTALE: {total_fixed} predizioni recuperate")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
