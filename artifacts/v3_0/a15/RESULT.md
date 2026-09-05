# A15 — entailment_checker

- release: `v3.0`
- provenance class: `DIRECT`
- spec hash: `b80f2d9ffc1fcf32`
- requires model calls: `True`

## Metrics
```json
{"alpha_ent":0.05,"conditions":{"context_truncation":{"FN":11,"FP":7,"TN":71,"TP":211,"check_fail_rate":0.023333333333333334,"checker_fingerprint":"checker-v3-fixture","coverage":0.7266666666666667,"fnr":0.04954954954954955,"fpr":0.08974358974358974,"n":300,"pcg_accepted_harm":0.03211009174311927,"precision":0.9678899082568807,"recall":0.9504504504504504},"distractor_evidence":{"FN":10,"FP":6,"TN":73,"TP":211,"check_fail_rate":0.02,"checker_fingerprint":"checker-v3-fixture","coverage":0.7233333333333334,"fnr":0.04524886877828054,"fpr":0.0759493670886076,"n":300,"pcg_accepted_harm":0.027649769585253458,"precision":0.9723502304147466,"recall":0.9547511312217195},"negation_traps":{"FN":10,"FP":8,"TN":66,"TP":216,"check_fail_rate":0.02666666666666667,"checker_fingerprint":"checker-v3-fixture","coverage":0.7466666666666667,"fnr":0.04424778761061947,"fpr":0.10810810810810811,"n":300,"pcg_accepted_harm":0.03571428571428571,"precision":0.9642857142857143,"recall":0.9557522123893806},"nominal":{"FN":10,"FP":1,"TN":72,"TP":217,"check_fail_rate":0.0033333333333333335,"checker_fingerprint":"checker-v3-fixture","coverage":0.7266666666666667,"fnr":0.04405286343612335,"fpr":0.0136986301369863,"n":300,"pcg_accepted_harm":0.0045871559633027525,"precision":0.9954128440366973,"recall":0.9559471365638766},"weaker_checker":{"FN":6,"FP":14,"TN":65,"TP":215,"check_fail_rate":0.04666666666666667,"checker_fingerprint":"checker-v3-fixture","coverage":0.7633333333333333,"fnr":0.027149321266968326,"fpr":0.17721518987341772,"n":300,"pcg_accepted_harm":0.0611353711790393,"precision":0.9388646288209607,"recall":0.9728506787330317}},"experiment_id":"A15","metric_version":"v3.0.0","n_records":1500,"spec_hash":"b80f2d9ffc1fcf32b02b711ee434663747252ac569c517b13a051e900b6d360d","threshold_policy":"max retained coverage s.t. UCB(checker FPR) <= alpha_ent, frozen pre-eval"}
```

## Checks
```json
{"alpha_ent_frozen":true,"checker_identified":true,"degraded_conditions_present":true,"nominal_condition_present":true,"records_present":true}
```

## Failures and limitations

None recorded.
