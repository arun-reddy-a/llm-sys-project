#!/usr/bin/env python3
"""Query results.jsonl — structured logs from Modal runs.

Usage:
    python3 query_results.py                     # show all runs (summary)
    python3 query_results.py --target bench_moe  # filter by target
    python3 query_results.py --last 3            # last N runs
    python3 query_results.py --last 1 --full     # last run with full stdout
    python3 query_results.py --failed             # only failed runs
"""

import json
import sys
import argparse
from pathlib import Path

RESULTS_FILE = Path(__file__).parent / "results.jsonl"


def load_entries():
    if not RESULTS_FILE.exists():
        return []
    entries = []
    for line in RESULTS_FILE.read_text().splitlines():
        line = line.strip()
        if line:
            entries.append(json.loads(line))
    return entries


def print_summary(entries):
    print(f"{'#':>3}  {'Timestamp':>24}  {'Target':<25}  {'Status':>6}  {'GPU':<5}")
    print(f"{'─'*3}  {'─'*24}  {'─'*25}  {'─'*6}  {'─'*5}")
    for i, e in enumerate(entries):
        ts = e.get("timestamp", "?")[:19]
        target = e.get("target", "?")
        code = e.get("exit_code", "?")
        gpu = e.get("gpu", "?")
        status = "✅ OK" if code == 0 else f"❌ {code}"
        print(f"{i+1:>3}  {ts:>24}  {target:<25}  {status:>6}  {gpu:<5}")


def print_full(entries):
    for e in entries:
        print(f"\n{'='*70}")
        print(f"  Target   : {e.get('target')}")
        print(f"  Time     : {e.get('timestamp')}")
        print(f"  Exit Code: {e.get('exit_code')}")
        print(f"  GPU      : {e.get('gpu')}")
        print(f"{'='*70}")
        print(e.get("stdout", ""))


def main():
    parser = argparse.ArgumentParser(description="Query results.jsonl")
    parser.add_argument("--target", help="Filter by target name (substring match)")
    parser.add_argument("--last", type=int, help="Show only the last N runs")
    parser.add_argument("--full", action="store_true", help="Show full stdout")
    parser.add_argument("--failed", action="store_true", help="Only failed runs")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    entries = load_entries()
    if not entries:
        print(f"No results found in {RESULTS_FILE}")
        sys.exit(0)

    # Filter
    if args.target:
        entries = [e for e in entries if args.target in e.get("target", "")]
    if args.failed:
        entries = [e for e in entries if e.get("exit_code", 0) != 0]
    if args.last:
        entries = entries[-args.last:]

    if not entries:
        print("No matching results.")
        sys.exit(0)

    # Output
    if args.json:
        print(json.dumps(entries, indent=2))
    elif args.full:
        print_full(entries)
    else:
        print_summary(entries)
        print(f"\n{len(entries)} result(s). Use --full for stdout, --json for raw JSON.")


if __name__ == "__main__":
    main()
