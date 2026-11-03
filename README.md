# SASHE-FL Prototype

This is the actual code used to generate every experimental number, table, and
result figure in the paper *"Privacy-Preserving Federated Learning for Secure
Healthcare Data."* Nothing in the paper's Results section was estimated or
invented — it all comes from running this code.

## What it does

A from-scratch (NumPy) two-hidden-layer MLP (30–256–128–1) is trained with
federated SCAFFOLD across 5 simulated institutions on a non-IID
(Dirichlet-split) partition of the Wisconsin diagnostic breast-cancer dataset.
Per-site model updates are selectively encrypted with **real CKKS
homomorphic encryption** (via the TenSEAL library — this is not a HE
simulation, ciphertexts are actually constructed, packed into 4096-slot
vectors, homomorphically summed on a "server" context that holds no secret
key, and decrypted). The encrypted fraction `p` of parameters is chosen by a
closed-form sensitivity score, matching the scheme described in Section IV of
the paper.

## Files

| File | Purpose |
|---|---|
| `sim.py` | Core library: the MLP (forward/backward by hand), SCAFFOLD local training, the sensitivity-score formula (Eq. 6), layer-wise top-*p* masking, CKKS encrypt/aggregate/decrypt helpers, the non-IID Dirichlet partitioner, and the gradient-inversion attack probe (`attack_cos`, `privacy_proxy`). |
| `run.py` | Main experiment driver. For each seed: prepares data, trains plain FedAvg / SCAFFOLD / DP-noised FedAvg baselines, runs the encrypted-sensitivity setup phase, then runs SASHE-FL at `p ∈ {100, 50, 30, 20, 10, 5}%`, and runs the privacy probe on the final model. Writes one JSON record per seed. |
| `extra.py` | Two follow-up experiments: (a) cost/accuracy of the "first-layer guard" mask variant, and (b) a 10-seed FedAvg-vs-SCAFFOLD comparison under two heterogeneity levels (Dirichlet α = 0.5 and 0.1). |
| `analyze.py` | Aggregates `res_all.json` across seeds into `summary.json` (means ± standard deviations for every quantity reported in the paper). |
| `figures/bw_diagrams.py` | Generates the four black-and-white architecture/protocol diagrams (Figs. 1–4 in the paper) — these are illustrative, not data-driven. |
| `figures/bw_results.py` | Generates the two black-and-white results figures (Figs. 5–6) directly from `summary.json` / `res_all.json` / `extra.json`. |
| `res_all.json` | Raw per-seed results from `run.py` (5 seeds) — this is the real experimental output. |
| `extra.json` | Raw output of `extra.py`. |
| `summary.json` | Aggregated means/SDs produced by `analyze.py`; every number quoted in the paper's tables and text traces back to this file. |
| `figures/fig*.png` | The actual rendered figures as they appear in the paper. |

## Requirements

```
pip install numpy scikit-learn tenseal matplotlib
```

Tested with: Python 3.12, NumPy 2.4, scikit-learn 1.8, TenSEAL 0.3.18,
matplotlib (any recent version). TenSEAL needs a C++ build toolchain on some
platforms — if `pip install tenseal` fails, see
https://github.com/OpenMined/TenSEAL for platform-specific build notes.

## Reproducing the results

```bash
# Main experiment (5 seeds) — this is what produced res_all.json.
# Takes roughly 60–75 seconds per seed on a single CPU core.
python run.py 0,1,2,3,4 res_all.json

# Aggregate into summary.json
python analyze.py

# Follow-up experiments (guard-mask cost + 10-seed SCAFFOLD study)
python extra.py

# Regenerate the figures (from inside figures/, with the three JSON
# files copied or symlinked alongside)
cd figures
python bw_diagrams.py   # Figs. 1-4 (illustrative)
python bw_results.py    # Figs. 5-6 (data-driven, reads ../res_all.json etc.)
```

Random seeds control the train/test split, the Dirichlet client partition,
and model initialization, so re-running with the same seed list reproduces
the same numbers up to floating-point nondeterminism in CKKS's approximate
arithmetic (differences on the order of 1e-8, which is itself one of the
things the paper measures — see `r['setup']['err']` and `r['he'][p]['maxerr']`
in `res_all.json`).

