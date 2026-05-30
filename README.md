# FUNNEL-AL

Active learning for entity alignment with three orthogonal selection signals.

## Method

FUNNEL-AL selects entities for labeling using a multiplicative combination of three signals:

```
score(s | S) = U_funnel(s) * (1 + α * C_src(s|S)) * (1 + γ * T_inst(s))
```

- **U_funnel** (Funnel Uncertainty): Measures prediction ambiguity via top-K score decay pattern.
- **C_src** (Structural Coverage): Submodular PPR-based coverage gain ensuring graph-space diversity.
- **T_inst** (Temporal Instability): Cross-round ranking change detecting unconverged predictions.

## Supported Backbone Models

| Model | Reference |
|-------|-----------|
| Dual-AMN | AAAI 2021 |
| LightEA | EMNLP 2022 |
| GCN-Align | EMNLP 2018 |

## Datasets

- DBP15K: zh-en, ja-en, fr-en
- SRPRS: en-fr, en-de

## Quick Start

```bash
# Single run (DualA on zh-en, seed=42)
bash run.sh duala zh_en 42

# Run all experiments
bash run_all.sh
```

### Arguments

```
--model       {duala, lightea, gcn_align}
--data_path   path to dataset directory
--seed        random seed
--alpha       coverage booster strength (default: 4.0)
--funnel_gamma  temporal instability booster (default: 3.0)
--cov_eta     coverage discount sharpness (default: 20.0)
```

## Requirements

```bash
pip install -r requirements.txt
```

Requires GPU with CUDA support.

## Project Structure

```
FunnelAL/
├── train.py                   # Main training entry
├── run.sh                     # One-click single-run script
├── run_all.sh                 # Full experiment suite
├── requirements.txt
├── src/
│   ├── funnel_al.py           # FUNNEL-AL selection strategy
│   ├── candidate_builder.py   # CSLS candidate construction
│   ├── evaluate.py            # H@1/H@10/MRR evaluation
│   ├── layer.py               # Graph attention layer (Dual-AMN)
│   ├── utils.py               # Common utilities
│   ├── pool_loader.py         # Active-learning pool adapter
│   ├── data_loader.py         # KG data loader
│   ├── io_utils.py            # I/O helpers
│   └── models/
│       ├── duala.py           # Dual-AMN backbone
│       ├── lightea.py         # LightEA backbone
│       └── gcn_align.py       # GCN-Align backbone
└── data/
    ├── zh_en/                 # DBP15K
    ├── ja_en/
    ├── fr_en/
    ├── en_fr_15k_V1/          # SRPRS
    └── en_de_15k_V1/
```
