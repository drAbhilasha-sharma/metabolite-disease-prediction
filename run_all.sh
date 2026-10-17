#!/usr/bin/env bash
# Run every analysis from the inputs in rebuild/, then compute the summary
# statistics. Outputs are written to results/.
#
#   ./run_all.sh              analyses from the deposited inputs
#   ./run_all.sh --from-raw   first rebuild the inputs that need the public
#                             releases in raw/ (see README, "Data")
#
# Steps whose inputs are not available are skipped with a message.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=scripts:.
mkdir -p results logs

run() {  # run <log name> <command...>
    local name=$1; shift
    echo "== $name"
    "$@" > "logs/$name.log" 2>&1 || { echo "   failed, see logs/$name.log"; exit 1; }
}
need() {  # need <file...>: true if every file exists, otherwise report and skip
    local f
    for f in "$@"; do
        if [ ! -e "$f" ]; then echo "== skipped (needs $f): $STEP"; return 1; fi
    done
}

HMDB=raw/data_raw/HMDB/hmdb_metabolites.xml
MESH=raw/data_raw/mesh/MeSH_desc2026.xml

# ---------------------------------------------------- inputs from the releases
if [ "${1:-}" = "--from-raw" ]; then
    STEP="parse HMDB";            need "$HMDB" && run parse_hmdb             python scripts/parse_hmdb.py
    STEP="curation status";       need "$HMDB" && run parse_status           python scripts/parse_status.py
    STEP="descriptors";           need rebuild/metabolites_all.csv && run descriptors python scripts/compute_descriptors_all.py
    STEP="first-evidence years";  need "$HMDB" && run extract_temporal       python scripts/extract_temporal.py
    STEP="MeSH representations";  need "$MESH" && run ontology_repr          python scripts/build_ontology_repr.py
fi

# ------------------------------------------------------------- main analyses
run benchmark_conventional  python scripts/run_benchmark.py all
run specificity             python scripts/run_specificity.py all
run benchmark_matched       python scripts/run_final_benchmark.py
run definitive              python scripts/run_definitive.py all
run perm_within             python scripts/run_perm2.py
run idcontrol               python scripts/run_idcontrol.py
run calibration             python scripts/run_calibration.py
STEP="annotation-status audit"; need rebuild/descriptors_all.csv && run attention python scripts/run_attention.py
run disease_meta            python scripts/run_disease_meta.py
run strata_network          python scripts/run_strata_network.py
run rank_baselines          python scripts/run_rank_baselines.py
run analog_audit            python scripts/run_analog_audit.py
run curves                  python scripts/run_curves.py

# ------------------------------------------------- interventions, robustness
STEP="representation exchange"; need rebuild/disease_mesh_tree.npy rebuild/disease_mesh_text.npy \
    && run transfer_repr    python scripts/run_transfer_repr.py
run graph_models            python scripts/run_graph_models.py
run robustness_seeds        python scripts/run_robustness.py seeds
STEP="sampler grid";        need rebuild/descriptors_all.csv && run robustness_sampler  python scripts/run_robustness.py sampler
STEP="prospective split";   need rebuild/assoc_years.csv     && run robustness_temporal python scripts/run_robustness.py temporal
STEP="curation depth";      need rebuild/assoc_years.csv     && run robustness_corrob   python scripts/run_robustness.py corrob
STEP="gene-disease comparison"; need rebuild/ctd_direct.tsv  && run comparator          python scripts/run_comparator.py
run belowchance             python scripts/run_belowchance.py
run substitution            env MDA_SUB_SEEDS=20 MDA_SUB_OUT=results/substitution_dist20.csv \
                            python scripts/run_substitution_mi.py
run substitution_strata     python scripts/run_substitution_stratified.py
run substitution_strata_rand env MDA_STRAT_BLOCK=random python scripts/run_substitution_stratified.py
run substitution_tests      python scripts/make_substitution_block_tests.py
run sensitivity_checks      python scripts/run_sensitivity_checks.py
run shortcut_audit          python scripts/run_shortcut_audit.py
STEP="audit package example"; need rebuild/descriptors_all.csv && run audit_example python pucontrols/example_audit.py

# ------------------------------------------------------------------ summaries
run summaries               python scripts/summarize_results.py
echo "done: results in results/, logs in logs/"
