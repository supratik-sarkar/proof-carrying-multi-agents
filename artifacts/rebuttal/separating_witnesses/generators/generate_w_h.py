#!/usr/bin/env python3
"""Generator for Single-Channel Failure Witness Certificate: V_H."""
import json, sys

def generate_witness():
    return {
        "witness_id": "V_H_only_failure",
        "channel_outcomes": {
            "V_H": False,
            "V_Pi": True,
            "V_Gamma": True,
            "V_entail": True
        },
        "failed_channels_count": 1
    }

if __name__ == "__main__":
    print(json.dumps(generate_witness(), indent=2))
