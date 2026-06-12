# EDHEC BTC Allocation Experiments

This repository contains the experiment data, code, notebooks, and compressed reproducibility outputs for the EDHEC BTC allocation paper project.

## Repository Contents

- `data_2026/`: input datasets and validation files.
- `experiments/`: Python experiment scripts and per-part dependency files.
- `notebooks/`: Colab-oriented notebooks for the fourteen experiment parts.
- `outputs/*_outputs.zip`: compressed result bundles for each experiment part.

The unpacked local output folders are intentionally excluded from Git to avoid duplicating the compressed result bundles.

## Experiment Parts

1. `part1_btc_macro_state`
2. `part2_portfolio_risk_budget`
3. `part3_btc_state_dependence`
4. `part4_conditional_btc_allocation`
5. `part5_implementability_rebalancing`
6. `part6_robustness_analysis`
7. `part7_realtime_probabilistic_regime_robustness`
8. `part8_uncertainty_quantification`
9. `part9_regime_stability_audit`
10. `part10_benchmark_cap_sensitivity`
11. `part11_hmm_model_comparison`
12. `part12_stress_composite_sensitivity`
13. `part13_transaction_cost_sensitivity`
14. `part14_pairwise_inference`

## Setup

Create and activate a Python environment, then install the dependencies for the experiment part you want to run. For example:

```bash
pip install -r experiments/part1_btc_macro_state/requirements-part1.txt
```

Several parts share the same core dependencies:

```bash
pip install numpy pandas matplotlib scipy
```

## Running Experiments

Each experiment part has a standalone Python script under `experiments/`. For example:

```bash
python experiments/part1_btc_macro_state/run_part1_btc_macro_state.py
```

Run the corresponding `run_part*.py` script for the other parts.

## Results

The compressed outputs are stored in `outputs/`:

- `part1_btc_macro_state_outputs.zip`
- `part2_portfolio_risk_budget_outputs.zip`
- `part3_btc_state_dependence_outputs.zip`
- `part4_conditional_btc_allocation_outputs.zip`
- `part5_implementability_rebalancing_outputs.zip`
- `part6_robustness_analysis_outputs.zip`
- `part7_realtime_probabilistic_regime_robustness_outputs.zip`
- `part8_uncertainty_quantification_outputs.zip`
- `part9_regime_stability_audit_outputs.zip`
- `part10_benchmark_cap_sensitivity_outputs.zip`
- `part11_hmm_model_comparison_outputs.zip`
- `part12_stress_composite_sensitivity_outputs.zip`
- `part13_transaction_cost_sensitivity_outputs.zip`
- `part14_pairwise_inference_outputs.zip`

Unzip the relevant result bundle to inspect generated tables, figures, logs, and serialized outputs.

Parts 10--14 add the final validation layer used in the paper: benchmark and risk-cap sensitivity, HMM model comparison, stress-composite sensitivity, transaction-cost sensitivity, and pairwise bootstrap inference.

## Notes

Before using or redistributing the data, verify that the original data licenses and source terms permit the intended use.
