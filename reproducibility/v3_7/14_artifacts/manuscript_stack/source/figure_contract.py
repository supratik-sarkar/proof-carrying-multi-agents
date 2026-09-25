"""Canonical figure identities and declared input dependencies."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
NAMES=['figure_01_executive_safety_performance_atlas','figure_02_certificate_runtime_workflow','figure_03_theory_sota_geometry','figure_04_mechanism_falsification_atlas','figure_A0_full_56_cell_sota_atlas','figure_A3_snapshot_fresh_replay','figure_A4_support_path_separation','figure_A1_guarantee_boundary','figure_A2_algorithm_schematic','figure_B1_theory_dependency','figure_C1_certificate_failure_channel_atlas','figure_C2_expanded_ablations','figure_C3_audit_sampling_atlas','figure_C4_separating_witness_matrix','figure_C5_injection_common_mode','figure_C6_shift_validity_atlas','figure_C7_responsibility_open_set','figure_C8_privacy_modelled','figure_C9_scaling_modelled','figure_D1_timing_distributions','figure_D2_cost_telemetry']
STATIC={2,6,7,8,9,10}
INPUTS={1:['method_summary.csv','paired_cell_effects.csv','cell_metrics.csv'],3:['figure_03_capability_contract.json','plot_parameters.json'],4:['ablation_summary.csv','risk_coverage_summary.csv','paired_cell_effects.csv','selectivity_verification.csv','table_25_protocol_sv.csv','figure_04_protocol.json','method_summary.csv'],5:['paired_cell_effects.csv','method_dataset_summary.csv'],11:['figure_C1_certificate_failure_channel_atlas.csv','table_07_channel_ablation.csv','table_10_replay_drift_covgap.csv'],12:['ablation_summary.csv'],13:['audit_sampling.csv'],14:['witness_matrix_scoped.csv'],15:['injection_stress.csv'],16:['shift_stress.csv'],17:['responsibility.csv'],18:['privacy_frontier.csv'],19:['scaling_surface.csv'],20:['timing_samples.csv','timing_summary.csv'],21:['figure_D2_cost_telemetry.csv','table_28_protocol_cost.csv','cell_metrics.csv','cost_telemetry.csv']}

RESERVED_OUTPUTS={
    'figure_03_release_control_funnel':['release_control_funnel.csv'],
    'figure_05_adversarial_shift_robustness_atlas':['injection_stress.csv','shift_stress.csv'],
}

NAMES += list(RESERVED_OUTPUTS)
INPUTS.update({22:RESERVED_OUTPUTS[NAMES[21]],23:RESERVED_OUTPUTS[NAMES[22]]})
