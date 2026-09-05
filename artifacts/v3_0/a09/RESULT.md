# A09 — shift

- release: `v3.0`
- provenance class: `DIRECT`
- spec hash: `70804e4b2c028a1e`
- requires model calls: `True`

## Metrics
```json
{"experiment_id":"A09","metric_version":"v3.0.0","n_records":1500,"note":"D_alarm is a gate; it is never substituted for a valid D_bar","regimes":{"backend_change":{"a_lcb":0.6281900448036282,"balanced_accuracy":0.66,"d_alarm":0.25638008960725633,"d_bar":null,"n":300,"n_cal":300,"n_dep":300,"note":"D_alarm is a one-sided lower bound on restricted TV: large proves shift, small does NOT certify absence. Never substitute for D_bar.","observed_contract_bad":0.12,"triggered":true},"corruption":{"a_lcb":0.7836566536969652,"balanced_accuracy":0.81,"d_alarm":0.5673133073939305,"d_bar":null,"n":300,"n_cal":300,"n_dep":300,"note":"D_alarm is a one-sided lower bound on restricted TV: large proves shift, small does NOT certify absence. Never substitute for D_bar.","observed_contract_bad":0.15333333333333332,"triggered":true},"held_out_dataset":{"a_lcb":0.6898493350228092,"balanced_accuracy":0.72,"d_alarm":0.3796986700456184,"d_bar":null,"n":300,"n_cal":300,"n_dep":300,"note":"D_alarm is a one-sided lower bound on restricted TV: large proves shift, small does NOT certify absence. Never substitute for D_bar.","observed_contract_bad":0.11,"triggered":true},"none":{"a_lcb":0.47643128173224225,"balanced_accuracy":0.51,"d_alarm":0.0,"d_bar":null,"n":300,"n_cal":300,"n_dep":300,"note":"D_alarm is a one-sided lower bound on restricted TV: large proves shift, small does NOT certify absence. Never substitute for D_bar.","observed_contract_bad":0.023333333333333334,"triggered":false},"tool_drift":{"a_lcb":0.5468571178088736,"balanced_accuracy":0.58,"d_alarm":0.09371423561774717,"d_bar":null,"n":300,"n_cal":300,"n_dep":300,"note":"D_alarm is a one-sided lower bound on restricted TV: large proves shift, small does NOT certify absence. Never substitute for D_bar.","observed_contract_bad":0.08666666666666667,"triggered":false}},"spec_hash":"70804e4b2c028a1e8d5118d1f1f2b954c0b9dc99290b25204eb0eb2bbad7702f"}
```

## Checks
```json
{"d_bar_not_faked":true,"records_present":true}
```

## Failures and limitations

None recorded.
