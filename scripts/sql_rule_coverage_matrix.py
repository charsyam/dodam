#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
from pathlib import Path


SELECTED_RE = re.compile(
    r"\[dodam:sql-rule\] selected=(?P<rule>\S+) kind=(?P<kind>\S+) "
    r"cost_rank=(?P<rank>\d+) estimated_cost=(?P<cost>\d+) "
    r"estimated_scan_bytes=(?P<scan>\S+) (?:scan_source=(?P<scan_source>\S+) )?"
    r"required_features=(?P<features>\S*) "
    r"required_columns=(?P<columns>.*)$"
)
DIRECT_SELECTED_REJECT_RE = re.compile(
    r"\[dodam:direct-selected\] (?P<path>[^:]+) reject: (?P<reason>.*)$"
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run TPC-H queries with SQL rule profiling and report selected rules."
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--queries", default="tests/tpch_coverage.rs")
    parser.add_argument("--output-dir", default="/tmp/dodam-sql-rule-coverage")
    parser.add_argument("--runner", default="target/release/tpch_real_inprocess")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--only", default="")
    parser.add_argument("--json-out", default="")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--trace-direct-selected",
        action="store_true",
        help="Include direct selected/page decoder rejection reasons in JSON rows.",
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Additional environment variable for the runner. May be repeated.",
    )
    args = parser.parse_args()

    cmd = [
        args.runner,
        "--data-dir",
        args.data_dir,
        "--queries",
        args.queries,
        "--output-dir",
        args.output_dir,
        "--batch-size",
        str(args.batch_size),
        "--repeats",
        "1",
    ]
    if args.only:
        cmd.extend(["--only", args.only])

    env = os.environ.copy()
    env["DODAM_SQL_RULE_PROFILE"] = "1"
    if args.trace_direct_selected:
        env["DODAM_DIRECT_SELECTION_TRACE"] = "1"
    for item in args.env:
        name, sep, value = item.partition("=")
        if not sep or not name:
            parser.error(f"--env expects NAME=VALUE, got {item!r}")
        env[name] = value
    completed = subprocess.run(
        cmd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=args.timeout,
    )
    rows = parse_rows(completed.stdout)
    print_report(rows)
    if args.json_out:
        json_out = Path(args.json_out)
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(rows, indent=2) + "\n")
    return completed.returncode


def parse_rows(text: str) -> list[dict]:
    rows = []
    pending = []
    pending_rejects = []
    for line in text.splitlines():
        match = SELECTED_RE.search(line)
        if match:
            pending.append(
                {
                    "rule": match.group("rule"),
                    "kind": match.group("kind"),
                    "cost_rank": int(match.group("rank")),
                    "estimated_cost": int(match.group("cost")),
                    "estimated_scan_bytes": parse_scan_bytes(match.group("scan")),
                    "scan_source": match.group("scan_source") or "unknown",
                    "required_features": split_csv(match.group("features")),
                    "required_columns": split_csv(match.group("columns")),
                }
            )
            continue
        match = DIRECT_SELECTED_REJECT_RE.search(line)
        if match:
            pending_rejects.append(
                {
                    "path": match.group("path"),
                    "reason": match.group("reason"),
                }
            )
            continue
        if line.startswith("repeat=") and ": ok " in line:
            parts = line.split()
            query = parts[1].rstrip(":") if len(parts) > 1 else "unknown"
            selected = pending.pop(0) if pending else {}
            row = {"query": query, **selected}
            row["maturity_status"] = maturity_status(row)
            if pending_rejects:
                row["direct_selected_rejects"] = pending_rejects
                pending_rejects = []
            rows.append(row)
    return rows


def parse_scan_bytes(value: str):
    if value == "unknown":
        return None
    return int(value)


def split_csv(value: str) -> list[str]:
    return [part for part in value.split(",") if part]


def maturity_status(row: dict) -> str:
    rule = row.get("rule")
    if not rule:
        return "rule_absent"
    if row.get("estimated_scan_bytes") is None:
        return "cost_unknown"
    kind = row.get("kind") or ""
    if kind in ("vector-aggregate", "vector-join-aggregate", "expression-aggregate"):
        return "selected_vector_path"
    return "selected_generic_or_derived"


def print_report(rows: list[dict]) -> None:
    print("| query | rule | kind | maturity | cost | scan bytes | scan source |")
    print("|---|---|---|---|---:|---:|---|")
    for row in rows:
        scan = row.get("estimated_scan_bytes")
        scan_text = "" if scan is None else str(scan)
        print(
            f"| {row.get('query', '')} | {row.get('rule', '')} | "
            f"{row.get('kind', '')} | {row.get('maturity_status', '')} | "
            f"{row.get('estimated_cost', '')} | {scan_text} | "
            f"{row.get('scan_source', '')} |"
        )


if __name__ == "__main__":
    raise SystemExit(main())
