# Semantic Mutation Test Report — Submission 9327

## Summary: PASS (16 / 16 Mutations Caught)

| Mutation ID | Description | Status | Observed Exit Code | Error Message |
|---|---|---|---|---|
| `corrupt_one_audit_inclusion_prob` | Corrupt inclusion prob | **PASS** | 1 | MUTATION_CAUGHT: json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0) |
| `remove_one_audit_stratum` | Remove audit stratum | **PASS** | 1 | MUTATION_CAUGHT: KeyError: "Record missing required field 'cell_id'." |
| `weight_inconsistent_with_prob` | Inconsistent weight | **PASS** | 1 | MUTATION_CAUGHT: ValueError: Record missing required 'systems' dict or 'NoCert' baseline. |
| `remove_one_injection_location` | Remove injection location | **PASS** | 1 | MUTATION_CAUGHT: FileNotFoundError: Source records not found: /var/folders/6v/j6vkmdxs4vg2ywlzm18 |
| `remove_one_injection_regime` | Remove injection regime | **PASS** | 1 | MUTATION_CAUGHT: json.decoder.JSONDecodeError: Expecting property name enclosed in double quotes: |
| `remove_one_redundancy_k` | Remove redundancy k | **PASS** | 1 | MUTATION_CAUGHT: FileNotFoundError: Source records file not found: /tmp/bad.jsonl |
| `corrupt_one_injection_numerator` | Corrupt injection numerator | **PASS** | 1 | MUTATION_CAUGHT: ValueError: Record missing required 'systems' or 'cell_id' schema. |
| `remove_one_shift_family` | Remove shift family | **PASS** | 1 | MUTATION_CAUGHT: FileNotFoundError: Source records not found: /var/folders/6v/j6vkmdxs4vg2ywlzm18 |
| `hardcode_tnr` | Hardcode TNR | **PASS** | 1 | MUTATION_CAUGHT: json.decoder.JSONDecodeError: Expecting property name enclosed in double quotes: |
| `alter_one_shift_label` | Alter shift label | **PASS** | 1 | MUTATION_CAUGHT: FileNotFoundError: Source records file not found: /tmp/bad.jsonl |
| `pcg_acceptance_as_pcg_harm` | PCG acceptance as harm | **PASS** | 1 | MUTATION_CAUGHT: ValueError: Record missing required 'systems' or 'cell_id' schema. |
| `break_one_sv_pairing_key` | Break S/V key | **PASS** | 1 | MUTATION_CAUGHT: ValueError: Record missing required 'systems' or 'cell_id' schema. |
| `alter_one_clean_room_output_byte` | Alter clean-room byte | **PASS** | 1 | MUTATION_CAUGHT: json.decoder.JSONDecodeError: Expecting property name enclosed in double quotes: |
| `remove_one_expected_protocol_tuple` | Remove protocol tuple | **PASS** | 1 | MUTATION_CAUGHT: json.decoder.JSONDecodeError: Expecting property name enclosed in double quotes: |
| `create_table2_table16_mismatch` | Table 2/16 mismatch | **PASS** | 1 | MUTATION_CAUGHT: FileNotFoundError: Source records file not found: /tmp/bad.jsonl |
| `alter_one_backend_revision_or_hash` | Alter backend revision | **PASS** | 1 | MUTATION_CAUGHT: ValueError: Backend manifest validation failed with 13440 errors: Record 0 missi |
