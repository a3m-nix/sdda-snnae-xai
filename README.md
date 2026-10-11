# SDDA: Spike Density Divergence Attribution for SNN-AE anomaly detection

Code and experimental outputs for the article "Spike Density Divergence Attribution: A Baseline-Relative Explainability Framework for Spiking Neural Network Autoencoders in IoT Anomaly Detection" (under revision).

The repository contains the experiments only: the evaluation pipeline, the revision analyses, their configurations,
and their outputs. Large per-sample outputs and the 60 trained models are in the Zenodo record https://doi.org/10.5281/zenodo.23278327.

## Repository structure
```text
src/pipeline/      Main pipeline notebook: SNN-AE training (2 datasets x 3 seeds x 10 folds = 60 models) and the
                   evaluation of all explanation methods (outputs removed; paths relative to the repository root)
src/revision/      Revision analyses (scripts 7-11) and the two-test statistical decisions
results/paper3-faithfulness-v4-3x10/
  raw/             Per-sample outputs of the main evaluation (feature_behavior.csv and suppression.csv: Zenodo)
  tables/          Aggregate tables of the article and of Supplementary Tables S1-S6 and S14
  baselines/       Validation-normal spike baselines per seed-fold unit
  metadata/        Feature lists and run manifest
results/revision-audit-20261009T115504/   Corrected resampled t-test and detection-stratified faithfulness (Supplementary Table S9)
results/0-revisi-sdda-v1/     Revision outputs (ablation tables, conservation, suppression, score linkage, two-test decisions)
data/              How to obtain and prepare the datasets
environment/       Computing environment and package versions
```

## Article items and their sources

| Article item | Script | Output |
|---|---|---|
| Tables 4-10, Figs. 4-9 (data) | `src/pipeline/3-4-paper3_xai_v4_3x10_final_multicell_v2.ipynb` | `results/paper3-faithfulness-v4-3x10/raw`, `results/paper3-faithfulness-v4-3x10/tables` |
| Supplementary Tables S1-S6, S14 | `src/pipeline/3-4-paper3_xai_v4_3x10_final_multicell_v2.ipynb` | `results/paper3-faithfulness-v4-3x10/tables` |
| Supplementary Table S7 (462 comparisons, two tests) | `src/revision/two_test_decisions.py` | `results/0-revisi-sdda-v1/two-test-decisions/S7_*` |
| Table 11, Supplementary Table S8 (ablation of Eq. 9) | `src/revision/8-eq9-ablation-v1.py` | `results/0-revisi-sdda-v1/eq9-ablation-v1/tables` (per-job replay: Zenodo) |
| Supplementary Table S9 (detection-stratified) | `src/revision/7-revision-csv-audit.py` | `results/revision-audit-20261009T115504` |
| Supplementary Table S10 (Proposition 1, conservation) | `src/revision/9-conservation-check.py` | `results/0-revisi-sdda-v1/conservation-check` |
| Supplementary Table S11 (suppression) | `src/revision/10-suppression-check.py` | `results/0-revisi-sdda-v1/suppression-check` |
| Supplementary Table S12 (attribution mass and score) | `src/revision/11-score-link-check.py` | `results/0-revisi-sdda-v1/score-link-check` |
| Supplementary Table S13 (re-test of Table 10) | `src/revision/two_test_decisions.py` | `results/0-revisi-sdda-v1/two-test-decisions/S13_*` |

## Reproducing the analyses
1. Create the environment from `environment/requirements-freeze.txt` (see `environment/ENVIRONMENT.md`).
2. Prepare the data as described in `data/README.md`, or download the Zenodo archives and extract them into `results/`.
3. Run the pipeline notebook, then the revision scripts. The revision scripts are published unchanged; their default
   paths point to the authors' server. Pass the repository paths instead, for example:
   ```bash
   python src/revision/7-revision-csv-audit.py --base results/paper3-faithfulness-v4-3x10
   python src/revision/9-conservation-check.py results/0-revisi-sdda-v1
   python src/revision/two_test_decisions.py --pipeline results/paper3-faithfulness-v4-3x10 \
       --ablation results/0-revisi-sdda-v1/eq9-ablation-v1 --out results/rerun-two-test-decisions
   ```
   `8-eq9-ablation-v1.py` reads its root folder from the constant `ROOT` at the top of the script and needs the trained
   models from Zenodo. `two_test_decisions.py` stops if any reported number (257/108/0/97 of 462, 365, 0 of 22, Table 10)
   is not reproduced.
4. Verify file integrity with `sha256sum -c MANIFEST.sha256`.

## License
Code: MIT License (see `LICENSE`). Results and documentation: CC BY 4.0.
