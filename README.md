# Metabolite–disease association prediction: evaluation code

Analysis code and derived input data. The code evaluates metabolite–disease
association models under two cross-validation regimes (held-out metabolites and
held-out diseases) with permuted-label, disease-blind and randomised-representation
controls, and runs the robustness, substitution and audit analyses reported in
the paper.

## Contents

```
run_all.sh          runs every analysis in order, then the summary statistics
requirements.txt    Python packages
scripts/            analysis code (one script per analysis, listed below)
pucontrols/         stand-alone audit package: the diagnostic controls for any
                    positive-unlabelled association table
rebuild/            derived input data used by every analysis (described below)
```

Outputs are written to `results/` and run logs to `logs/`; neither is part of the
repository.

## Installation

Python 3.13.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running

```bash
./run_all.sh
```

Run from the repository root. To run a single script, set `PYTHONPATH=scripts:.`,
for example `PYTHONPATH=scripts:. python scripts/run_definitive.py main`. The main
benchmark (`run_definitive.py main`) takes about 15 minutes on an 8-core Apple
silicon laptop. Steps whose inputs are not available (see "Data") are skipped with
a message.

`scripts/summarize_results.py`, the last step, computes from the analysis outputs
the summary statistics reported in the paper: confidence intervals, paired tests,
equivalence tests and shares of the excess over chance.

## Data

### Provided in `rebuild/`

| File | Content |
|---|---|
| `model_positives.csv` | the 4,791 modeled metabolite–disease associations |
| `fp_index.csv`, `fp_matrix.npy` | ECFP4 fingerprints (radius 2, 1,024 bits) of the 46,432 compounds in the candidate universe |
| `unlabelled_ids.npy` | the 44,093 compounds of the unannotated pool |
| `disease_features.csv` | 52-dimensional gene-association profiles of the 71 diseases (truncated SVD of CTD direct-evidence gene membership) |
| `disease_genes.json` | the CTD direct-evidence gene set of each disease |
| `disease_kegg_features.csv` | membership of each disease in 20 KEGG pathways |
| `disease2mesh.json` | the MeSH descriptor of each disease |
| `sibling_groups.json` | assignment of the 71 diseases to 64 groups; diseases with the same MeSH descriptor share a group and are kept in one fold |
| `scaffolds.csv` | Bemis–Murcko scaffold of each modeled metabolite |
| `metabolite_classes.csv` | chemical superclass and class of each modeled metabolite (HMDB taxonomy) |
| `metabolite_status.csv` | curation-status and biospecimen fields of every HMDB record |
| `associations_clean.csv`, `associations_curated.csv`, `mesh_names.csv` | disease annotations at each curation stage, and MeSH descriptor names |
| `parse_summary.txt` | counts from parsing the HMDB release |

These files are the exact inputs of every analysis. Their construction from the
source releases is described in the Methods. The scripts that first assembled the
fingerprints, disease profiles, KEGG features, scaffolds, positive set and sibling
groups are not part of this repository.

### Rebuilt from the public releases

A few analyses also need files that are either large or verbatim source content.
They are rebuilt from the public releases. Place the releases under `raw/` and run
`./run_all.sh --from-raw`.

| File | Needed by | Built by | From |
|---|---|---|---|
| `rebuild/metabolites_all.csv` | `compute_descriptors_all.py` | `parse_hmdb.py` | HMDB 5.0, `hmdb_metabolites.xml` (https://hmdb.ca/downloads) at `raw/data_raw/HMDB/hmdb_metabolites.xml` |
| `rebuild/descriptors_all.csv` | `run_attention.py`, `run_robustness.py sampler`, `pucontrols/example_audit.py` | `compute_descriptors_all.py` | `metabolites_all.csv` |
| `rebuild/assoc_years.csv` | `run_robustness.py temporal`, `run_robustness.py corrob` | `extract_temporal.py` | HMDB 5.0 XML |
| `rebuild/disease_mesh_tree.npy`, `disease_mesh_text.npy` | `run_transfer_repr.py` | `build_ontology_repr.py` | MeSH 2026, `desc2026.xml` (https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/desc2026.xml) at `raw/data_raw/mesh/MeSH_desc2026.xml` |
| `rebuild/ctd_direct.tsv` | `run_comparator.py` | see below | CTD, `CTD_genes_diseases.tsv.gz` (https://ctdbase.org/downloads/) |

`rebuild/ctd_direct.tsv` is the CTD gene–disease file reduced to the rows with a
non-empty `DirectEvidence` field and the columns `GeneSymbol`, `GeneID`,
`DiseaseID` and `DirectEvidence`, tab-separated with that header.

`--from-raw` overwrites `rebuild/metabolite_status.csv` and
`rebuild/parse_summary.txt` with files rebuilt from the release.

## Scripts and the paper

| Script | Output in `results/` | Paper |
|---|---|---|
| `run_benchmark.py` | `benchmark_cv.csv`, `benchmark_ldo.csv`, `benchmark_ctrl.csv` | conventional protocol (Fig. 1a) |
| `run_specificity.py` | `specificity_all.csv` | matched-negative and within-metabolite designs; also the shared data loader |
| `run_final_benchmark.py` | `final_benchmark.csv` | matched-negative design |
| `run_definitive.py` | `definitive_per_disease.csv`, `definitive_oof.parquet`, `ablations.csv` | corrected protocol (Table 2, Figs. 1b, 5); feature-block ablation (Table 4, Fig. 7a) |
| `run_perm2.py` | `perm_within.csv` | within-disease permutation null |
| `run_idcontrol.py` | `id_control.csv` | random Gaussian and one-hot disease blocks (Table 4) |
| `run_calibration.py` | `discrimination_summary.csv`, `calibration.csv`, `roc_curves.csv`, `pr_curves.csv`, `topk_enrichment.csv` | calibration and decision utility (Table 3, Fig. 6) |
| `run_attention.py` | `attention_auc.csv`, `attention_effect_sizes.csv` | annotation-status audit (Fig. 4, Table S6) |
| `run_disease_meta.py` | `disease_meta.csv`, `disease_meta_model.csv` | disease-level meta-regression (Fig. 7b–c, Table S10) |
| `run_strata_network.py` | `per_metabolite_rank.csv`, `strata_*.csv`, `network_*.csv` | stratified ranking and network structure (Figs. 7d–f, 9; Tables S11, S12, S15) |
| `run_rank_baselines.py` | `ranking_baselines.csv` | within-metabolite ranking against the prevalence-ordered baseline |
| `run_analog_audit.py` | `analog_*.csv` | nearest-neighbour similarity audit |
| `run_curves.py` | `learning_curve.csv`, `label_noise.csv` | data scaling and label noise (Fig. 8, Tables S13–S14) |
| `run_transfer_repr.py` | `transfer_repr.csv` | disease-representation exchange (Table 5, Fig. 10a) |
| `run_graph_models.py` | `graph_models.csv` | graph learners (Table 5, Fig. 10b) |
| `run_robustness.py` | `sampler_grid.csv`, `temporal_holdout.csv`, `corroboration.csv`, `ldo_seeds20.csv`, `seed_stability.csv` | negative samplers, prospective split, curation depth, 20 seeds (Table 6, Figs. 10c, 11a–c, Table S16) |
| `run_comparator.py` | `comparator_controls.csv` | gene–disease comparison (Fig. 11e–f, Table S17) |
| `run_belowchance.py` | `belowchance.csv` | label-shuffle diagnostic (Fig. 5f, Table S7) |
| `run_substitution_mi.py` | `substitution_dist20.csv` | prediction-time substitution (Fig. 11d, Table S8) |
| `run_substitution_stratified.py` | `substitution_strata.csv`, `substitution_slope.csv`; with `MDA_STRAT_BLOCK=random`, `*_rand.csv` | distance-stratified substitution, real and random-Gaussian blocks (Table S9) |
| `make_substitution_block_tests.py` | `substitution_block_tests.csv` | real against random-Gaussian block |
| `run_sensitivity_checks.py` | `sensitivity_checks.csv` | sampler diagnostics, merged disease groups, learning-curve continuation |
| `run_shortcut_audit.py` | `shortcut_audit.csv` | training-fold shortcuts in the held-out-disease regime |
| `pucontrols/example_audit.py` | `toolkit_audit.csv` | the audit package applied to this dataset (Table S18) |
| `summarize_results.py` | summary files listed in its header | the reported summary statistics |

Shared modules: `evalproto.py` (fold construction for both regimes),
`degree_matched.py` (degree-matched negative sampler), `run_specificity.py` (data
loader). Data assembly: `parse_hmdb.py`, `parse_status.py`,
`compute_descriptors_all.py`, `extract_temporal.py`, `build_ontology_repr.py`.

## Reproducibility notes

- All random seeds are fixed in the scripts.
- Random forest and gradient-boosted trees reproduce the per-disease results
  exactly across platforms.
- Logistic regression and the multilayer perceptron depend on the BLAS library and
  solver of the platform. Their per-disease AUROC can differ slightly between
  machines (up to 0.004 in the per-learner means).
- Random-forest probabilities contain many exact ties, and multi-threaded
  prediction can order them differently from run to run. Within-disease AUROC is
  unaffected; pooled curve points and quantile bins can move in the last digits.

## Audit package

`pucontrols/` implements the diagnostic controls (permuted labels, target-blind,
degree-only, grouped split, duplicate targets) for any positive-unlabelled
association table. `pucontrols/example_audit.py` shows its use on this dataset.

## License

MIT; see `LICENSE`.
