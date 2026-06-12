# Can LLMs Generate Reliable Synthetic Surveys for Pain Assessment?
### A Multidimensional Evaluation to Design a New Clinical Decision Support System

[![GitHub](https://img.shields.io/badge/GitHub-LLM--Pain--Assessment--MPQ-blue)](https://github.com/GSoli96/LLM-Pain-Assessment-MPQ)

**Authors:** Gennaro Capaldo, Marco Cascella, Stefano Cirillo, Ornella Piazza, Giuseppe Polese, Carmela Pia Senatore, Giandomenico Solimando  
**Affiliations:** Department of Computer Science, University of Salerno · Department of Medicine, Surgery, and Dentistry, University of Salerno

---

## Abstract

Pain assessment remains a major clinical challenge because it is subjective and multifactorial. Current evaluations rely on self-reports and clinician judgment, resulting in heterogeneous, poorly standardized assessments that can impair treatment planning and monitoring. Limited clinical data and strict privacy constraints also hinder the development of robust learning models and decision support systems for chronic pain. In this study, we examine whether recent large language models (LLMs) can generate clinically plausible synthetic datasets based on the McGill Pain Questionnaire (MPQ). Using an augmented prompt-engineering approach, we create 4 new large datasets of MPQs, assembled into a new dataset suite (**MPQ-SynSuite**). Optimal discretization cut-offs for pain intensity are identified through statistical testing, and the resulting labeled data are used to train classification models to automatically assess intensity. Regarding etiology, MPQ questionnaires were labeled based on clinical criteria derived from MPQ descriptor profiles. We then conducted a comprehensive evaluation of the intensity and etiology assessment capabilities of **8 cloud LLMs** (Gemma 4 31B, Gemma 3 27/12/4B, Ministral-3 14B/8B, Devstral-Small-2 24B, RNJ-1 8B) using **13 prompting strategy–task combinations** (8 for Pain Intensity, 5 for Etiology). Results show that CoT Hierarchical prompting enables near-perfect Pain Intensity classification (macro F1 up to 1.00 for 3 models). Etiology classification converges robustly at F1 ≈ 0.846 across all models, while Knowledge-Guided prompting consistently fails (F1 ≤ 0.475). Finally, we present **PainMate**, a web-based clinical decision support system integrating the best-performing models for pain classification and drug therapy planning.

---

## Repository Structure

```
TerapiaDoloreSenatore/
├── paper/
│   └── elsarticle-template-num.tex     # LaTeX source of the paper
├── Tesi_codice_Senatore/
│   └── dataset/                        # Input datasets (4 synthetic MPQ datasets)
│       ├── MC_gill_DEEPSEEK.csv
│       ├── McGill_Pain_Questionnaire_CLAUDE.csv
│       ├── McGill_Pain_Questionnaire_DOCTORAI.csv
│       └── McGill_Pain_Questionnaire_GPT_Con_Dolore.csv
├── results/                            # Output metrics and reports (large files excluded from git)
│   ├── analisi_cloud_summary.csv       # Aggregated metrics (model × strategy × dataset)
│   └── analisi_cloud_report.txt        # Human-readable analysis report
├── plots/                              # Generated figures (bar charts)
├── utils/                              # Utility and monitoring scripts
│   ├── _check_quota.py                 # API quota checker
│   ├── _count_calls.py                 # API call counter
│   ├── _status.py                      # Experiment status monitor
│   ├── deep_monitor.py                 # Detailed live monitoring
│   ├── monitor_once.py                 # Single status snapshot
│   ├── fix_gaps.py                     # Fill missing predictions
│   └── ...
├── ask_llm_multiple_CoT.py             # Main experiment script
├── analisi_cloud_results.py            # Analysis & metric computation script
├── run_cloud.bat                       # Windows launcher for cloud experiments
├── requirements.txt                    # Python dependencies
└── ChaviOllama.json.example            # API key configuration template
```

---

## Setup

### 1. Python environment

```bash
pip install -r requirements.txt
```

### 2. API keys

The experiment requires API keys for the [Ollama Cloud](https://ollama.com/search) free/PRO tier.

1. Copy the template: `cp ChaviOllama.json.example ChaviOllama.json`
2. Edit `ChaviOllama.json` and replace the placeholder keys with your own Ollama Cloud API keys.
3. Obtain keys by registering at [https://ollama.com](https://ollama.com) and navigating to Settings → API Keys.

**Note:** `ChaviOllama.json` is listed in `.gitignore` and will not be committed.

---

## Running the Experiments

### Cloud experiment (all 8 models × 4 datasets × 13 strategies)

```bash
run_cloud.bat          # Windows
# or:
python ask_llm_multiple_CoT.py --group cloud
```

The script uses automatic key rotation, per-request caching (`llm_cache_cloud.json`), and incremental checkpointing (`checkpoints/`) to allow safe resumption after interruption.

### Analyzing results

```bash
python analisi_cloud_results.py
```

Produces:
- `results/analisi_cloud_summary.csv` — complete metrics table (model × dataset × strategy)
- `results/analisi_cloud_report.txt` — human-readable summary
- `plots/cloud_f1_pi.png` and `plots/cloud_f1_etio.png` — bar charts

---

## Models

| Model | Family | Parameters |
|---|---|---|
| Gemma 4 31B | Gemma | 31B |
| Gemma 3 27B | Gemma | 27B |
| Gemma 3 12B | Gemma | 12B |
| Gemma 3 4B | Gemma | 4B |
| Ministral-3 14B | Mistral | 14B |
| Ministral-3 8B | Mistral | 8B |
| Devstral-Small-2 24B | Mistral | 24B |
| RNJ-1 8B | RNJ | 8B |

All models are accessed via the [Ollama Cloud API](https://ollama.com/search) with temperature = 0.

---

## Prompting Strategies

| Strategy | Pain Intensity | Etiology |
|---|---|---|
| Zero-Shot | ✓ | ✓ |
| CoT Simple | ✓ | ✓ |
| CoT Hierarchical | ✓ | — |
| CoT Verification | ✓ | ✓ |
| CoT Ensemble | ✓ | — |
| CoT Decision-Tree | ✓ | — |
| CoT Confidence | ✓ | — |
| Few-Shot | ✓ | ✓ |
| Knowledge-Guided (Wilkie et al.) | — | ✓ |

---

## Key Results

**Pain Intensity (macro F1, averaged over 4 datasets)**

| Model | Best Strategy | Best F1 |
|---|---|---|
| Gemma 3 27B | CoT Hierarchical | **1.000** |
| Devstral-Small-2 24B | CoT Hierarchical | **1.000** |
| Gemma 4 31B | CoT Hierarchical | 0.992 |
| Gemma 3 12B | CoT Hierarchical | 0.997 |

**Etiology (binary F1, averaged over 4 datasets)**

Most models converge at F1 ≈ 0.846 across Zero-Shot, CoT Simple, CoT Verification, and Few-Shot.  
Knowledge-Guided prompting fails universally (F1 ≤ 0.475).

---

## Citation

```bibtex
@article{capaldo2026llm_mpq,
  title   = {Can LLMs Generate Reliable Synthetic Surveys for Pain Assessment?
             A Multidimensional Evaluation to Design a New Clinical Decision Support System},
  author  = {Capaldo, Gennaro and Cascella, Marco and Cirillo, Stefano and Piazza, Ornella
             and Polese, Giuseppe and Senatore, Carmela Pia and Solimando, Giandomenico},
  journal = {(under review)},
  year    = {2026}
}
```

---

## License

This repository is shared for research reproducibility. The datasets were generated synthetically using LLMs for research purposes only. Clinical use without validation is not recommended.
