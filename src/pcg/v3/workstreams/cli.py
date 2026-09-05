"""Non-interactive CLI: python -m pcg.v3.workstreams.cli run A03|all"""
from __future__ import annotations
import argparse, json, sys
from .catalog import CATALOG, IDS
from .runners import build

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="pcg-v3-workstreams", add_help=True)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("experiment"); r.add_argument("--records", default=None)
    sub.add_parser("list")
    a = ap.parse_args(argv)
    if a.cmd == "list":
        for i in IDS:
            s = CATALOG[i]
            print(f"{i}  {s.name:<26} tier={s.tier:<8} model_calls={s.requires_model_calls}  {s.provenance_class}")
        return 0
    targets = IDS if a.experiment.lower() == "all" else [a.experiment.upper()]
    rc = 0
    for eid in targets:
        ws = build(eid)
        recs = ws.load_records(a.records) if a.records else ws.load_records()
        out = ws.run(recs)
        failed = [k for k, v in out["checks"].items() if v is False]
        print(f"{eid}: records={len(recs):<5} checks_failed={len(failed)}  -> {out['outdir']}")
        if failed:
            print(f"     failed: {', '.join(failed)}")
    return rc

if __name__ == "__main__":
    sys.exit(main())
