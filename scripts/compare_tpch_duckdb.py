#!/usr/bin/env python3
import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


TABLES = {
    "lineitem",
    "orders",
    "customer",
    "supplier",
    "partsupp",
    "part",
    "nation",
    "region",
}

KEYWORDS = {
    "and",
    "as",
    "cross",
    "from",
    "full",
    "group",
    "having",
    "inner",
    "join",
    "left",
    "limit",
    "on",
    "or",
    "order",
    "outer",
    "right",
    "select",
    "semi",
    "where",
}

TPCH_OUTPUT = re.compile(r"repeat=(\d+)\s+([^:\s]+):\s+ok\s+([0-9.]+)s")
DUCKDB_MARK = re.compile(r"^DODAM_BENCH\s+repeat=(\d+)\s+query=([^\s]+)$")
DUCKDB_TIMER = re.compile(r"^Run Time \(s\): real\s+([0-9.]+)")
TPCH_PROFILE_ELAPSED = re.compile(
    r"\[dodam:tpch-profile\]\s+(?P<label>.*?):\s+(?P<ms>[0-9.]+)\s+ms$"
)
TPCH_PROFILE_FOLD = re.compile(
    r"\[dodam:tpch-profile\]\s+(?P<label>.*?):\s+total=(?P<total>[0-9.]+)\s+ms\s+"
    r"stream_read=(?P<stream>[0-9.]+)\s+ms\s+worker_wait_merge=(?P<merge>[0-9.]+)\s+ms\s+"
    r"(?P<unit>batches|view_chunks)=(?P<count>\d+)"
)
TPCH_PROFILE_PAIR_COLLECT = re.compile(
    r"\[dodam:tpch-profile\]\s+(?P<label>.*?):\s+total=(?P<total>[0-9.]+)\s+ms\s+"
    r"stream_read=(?P<stream>[0-9.]+)\s+ms\s+worker_wait_merge=(?P<merge>[0-9.]+)\s+ms\s+"
    r"worker_wait=(?P<wait>[0-9.]+)\s+ms\s+flatten=(?P<flatten>[0-9.]+)\s+ms\s+"
    r"view_chunks=(?P<count>\d+)\s+pairs=(?P<pairs>\d+)\s+max_partial_pairs=(?P<max_partial_pairs>\d+)"
)
TPCH_PROFILE_HASH_MAP_BUILD = re.compile(
    r"\[dodam:tpch-profile\]\s+(?P<label>.*?):\s+hash_map_build=(?P<build>[0-9.]+)\s+ms\s+"
    r"pairs=(?P<pairs>\d+)\s+entries=(?P<entries>\d+)"
)
TPCH_PROFILE_DENSE_I32_MAP_BUILD = re.compile(
    r"\[dodam:tpch-profile\]\s+(?P<label>.*?):\s+dense_i32_map_build=(?P<build>[0-9.]+)\s+ms\s+"
    r"pairs=(?P<pairs>\d+)\s+entries=(?P<entries>\d+)\s+layout=(?P<layout>\S+)"
)
TPCH_PROFILE_LATE_MATERIALIZED = re.compile(
    r"\[dodam:tpch-profile\]\s+(?P<label>.*?):\s+late_materialized\s+"
    r"rows=(?P<rows>\d+)\s+selected=(?P<selected>\d+)\s+ratio=(?P<ratio>[0-9.]+)\s+"
    r"selector_runs=(?P<selector_runs>\d+)\s+(?:chunks|row_group_chunk)=(?P<chunks>\d+)"
    r"(?:\s+predicate_read=(?P<predicate_read>[0-9.]+)\s+ms\s+"
    r"payload_read=(?P<payload_read>[0-9.]+)\s+ms\s+"
    r"predicate_batches=(?P<predicate_batches>\d+)\s+"
    r"payload_batches=(?P<payload_batches>\d+)\s+payload_rows=(?P<payload_rows>\d+))?"
)
TPCH_PROFILE_ROW_GROUP_MAP = re.compile(
    r"\[dodam:tpch-profile\]\s+(?P<label>row_group_map(?:_view)?\s+[^:]+):\s+"
    r"chunk=(?P<chunk>\d+)\s+row_groups=(?P<row_groups>\d+)\s+projected_columns=(?P<columns>\d+)\s+"
    r"rows=(?P<rows>\d+)\s+batches=(?P<batches>\d+)\s+zero_batches=(?P<zero_batches>\d+)\s+"
    r"total=(?P<total>[0-9.]+)\s+ms\s+setup=(?P<setup>[0-9.]+)\s+ms\s+"
    r"metadata=(?P<metadata>[0-9.]+)\s+ms\s+planning=(?P<planning>[0-9.]+)\s+ms\s+"
    r"read_next=(?P<read_next>[0-9.]+)\s+ms\s+reader_next=(?P<reader_next>[0-9.]+)\s+ms.*?"
    r"consume=(?P<consume>[0-9.]+)\s+ms"
)
ROW_GROUP_MAP_SUMMARY = re.compile(
    r"\[dodam:scan-profile\]\s+(?P<kind>\S+_summary)\s+(?P<label>[^:]+):\s+"
    r"chunks=(?P<chunks>\d+)\s+row_groups=(?P<row_groups>\d+)\s+"
    r"(?:projected_columns=(?P<columns>\d+)\s+)?rows=(?P<rows>\d+)\s+"
    r"batches=(?P<batches>\d+)\s+zero_batches=(?P<zero_batches>\d+)\s+"
    r"total_sum=(?P<total>[0-9.]+)\s+ms\s+read_next=(?P<read_next>[0-9.]+)\s+ms\s+"
    r"reader_next=(?P<reader_next>[0-9.]+)\s+ms\s+reader_next_avg=(?P<reader_next_avg>[0-9.]+)\s+ms\s+"
    r"reader_next_p95_max=(?P<reader_next_p95_max>[0-9.]+)\s+ms\s+"
    r"reader_next_max=(?P<reader_next_max>[0-9.]+)\s+ms\s+reader_calls=(?P<reader_calls>\d+)\s+"
    r"reader_eof=(?P<reader_eof>\d+)\s+avg_batch_rows=(?P<avg_batch_rows>[0-9.]+)\s+"
    r"consume=(?P<consume>[0-9.]+)\s+ms\s+compressed=(?P<compressed_scanned>\d+)/(?P<compressed_total>\d+)"
)
SCAN_PROFILE = re.compile(
    r"\[dodam:scan-profile\]\s+(?P<label>[^:]+):\s+elapsed=(?P<elapsed>[0-9.]+)\s+ms\s+"
    r"next_wait=(?P<next_wait>[0-9.]+)\s+ms\s+consumer_gap=(?P<consumer_gap>[0-9.]+)\s+ms\s+"
    r"rows=(?P<rows>\d+)\s+batches=(?P<batches>\d+)\s+row_groups=(?P<row_groups>\d+)/(?P<total_row_groups>\d+).*?"
    r"decode=(?P<decode>[0-9.]+)\s+ms"
)
I64_SET_FILTER_PROFILE = re.compile(
    r"\[dodam:scan-profile\]\s+i64_set_filter\s+(?P<label>[^:]+):\s+"
    r"elapsed=(?P<elapsed>[0-9.]+)\s+ms\s+setup=(?P<setup>[0-9.]+)\s+ms\s+"
    r"read_loop=(?P<read_loop>[0-9.]+)\s+ms\s+row_groups=(?P<row_groups>\d+)/(?P<total_row_groups>\d+)\s+"
    r"projected_columns=(?P<columns>\d+)\s+rows=(?P<rows>\d+)\s+batches=(?P<batches>\d+)\s+"
    r"zero_batches=(?P<zero_batches>\d+)\s+bytes=(?P<bytes>\d+).*?"
    r"parquet_next=(?P<parquet_next>[0-9.]+)\s+ms\s+parquet_next_avg=(?P<parquet_next_avg>[0-9.]+)\s+ms\s+"
    r"parquet_next_max=(?P<parquet_next_max>[0-9.]+)\s+ms\s+parquet_calls=(?P<parquet_calls>\d+)"
)
PARQUET_COLUMN_PROFILE = re.compile(
    r"\[dodam:parquet-column-profile\]\s+(?P<label>[^:]+):\s+"
    r"row_groups=(?P<row_groups>\d+)/(?P<total_row_groups>\d+)\s+columns=\[(?P<columns>.*)\]$"
)
PARQUET_COLUMN_PROFILE_COLUMN = re.compile(
    r"(?P<column>[^:,\]]+):compressed=(?P<compressed_scanned>\d+)/(?P<compressed_total>\d+)\s+"
    r"uncompressed=(?P<uncompressed_scanned>\d+)\s+encodings=(?P<encodings>[^,]+)"
)
DIRECT_PRIMITIVE_PROFILE = re.compile(
    r"\[dodam:direct-primitive-profile\]\s+(?P<label>[^:\[]+)\[(?P<columns>[^\]]*)\]:\s+"
    r"(?:reader_kind=(?P<reader_kind>\S+)\s+)?"
    r"row_groups=(?P<row_groups>\d+)\s+rows=(?P<rows>\d+)\s+batches=(?P<batches>\d+)\s+"
    r"read=(?P<read>[0-9.]+)\s+ms\s+consume=(?P<consume>[0-9.]+)\s+ms\s+"
    r"column_read=\[(?P<column_read>[^\]]*)\]\s+selected_predicate=(?P<selected_predicate>[0-9.]+)\s+ms\s+"
    r"selected_payload=(?P<selected_payload>[0-9.]+)\s+ms\s+selected_dictionary=(?P<selected_dictionary>[0-9.]+)\s+ms\s+"
    r"selected_rows=(?P<selected_rows>\d+)\s+selected_ratio=(?P<selected_ratio>[0-9.]+)\s+"
    r"selected_runs=(?P<selected_runs>\d+)\s+selected_batches=(?P<selected_batches>\d+)\s+"
    r"full_batches=(?P<full_batches>\d+)\s+selected_skip_calls=(?P<selected_skip_calls>\d+)\s+"
    r"selected_skipped_rows=(?P<selected_skipped_rows>\d+)\s+selected_read_calls=(?P<selected_read_calls>\d+)\s+"
    r"selected_read_rows=(?P<selected_read_rows>\d+)"
    r"(?:\s+dictionary_range_pages=(?P<dictionary_range_pages>\d+)"
    r"\s+dictionary_block_pages=(?P<dictionary_block_pages>\d+)"
    r"\s+dictionary_block_rows=(?P<dictionary_block_rows>\d+))?"
    r"(?:\s+selected_page_skip_pages=(?P<selected_page_skip_pages>\d+)"
    r"\s+selected_page_skip_rows=(?P<selected_page_skip_rows>\d+))?"
)
DIRECT_SELECTED_REJECT = re.compile(
    r"\[dodam:direct-selected\]\s+(?P<label>[^:]+)\s+reject:\s+(?P<reason>.*)$"
)
PHYSICAL_EVENT = re.compile(r"\[dodam:physical\]\s+(?P<fields>.*?)\s+label=(?P<label>.*)$")


@dataclass(frozen=True)
class TpchQuery:
    name: str
    sql: str


@dataclass
class QueryStats:
    samples: list[float]

    def median(self, warmup: int) -> float:
        samples = self.warm_samples(warmup)
        return statistics.median(samples) if samples else float("nan")

    def min(self, warmup: int) -> float:
        samples = self.warm_samples(warmup)
        return min(samples) if samples else float("nan")

    def max(self, warmup: int) -> float:
        samples = self.warm_samples(warmup)
        return max(samples) if samples else float("nan")

    def stdev(self, warmup: int) -> float:
        samples = self.warm_samples(warmup)
        return statistics.stdev(samples) if len(samples) > 1 else 0.0

    def warm_samples(self, warmup: int) -> list[float]:
        if warmup <= 0:
            return self.samples
        return self.samples[warmup:]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare Dodam and DuckDB TPC-H Parquet-output timings."
    )
    parser.add_argument("--data-dir", default="/tmp/dodam-tpchgen-sf1")
    parser.add_argument("--queries", default="tests/tpch_coverage.rs")
    parser.add_argument("--dodam", default="target/release/tpch_real_inprocess")
    parser.add_argument("--duckdb", default="duckdb")
    parser.add_argument(
        "--duckdb-mode",
        choices=("cli", "single-process"),
        default="cli",
        help="Run DuckDB once per query (cli) or once for the whole repeated benchmark.",
    )
    parser.add_argument("--output-dir", default="/tmp/dodam-tpch-compare")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16 * 1024)
    parser.add_argument("--only", default="")
    parser.add_argument("--skip-dodam", action="store_true")
    parser.add_argument("--skip-duckdb", action="store_true")
    parser.add_argument("--json-out", default="")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--fail-slower-ratio",
        type=float,
        default=0.0,
        help="Exit non-zero when a Dodam-slower query exceeds this ratio and --fail-slower-gap-ms.",
    )
    parser.add_argument(
        "--fail-slower-gap-ms",
        type=float,
        default=0.0,
        help="Exit non-zero when a Dodam-slower query exceeds this millisecond gap and --fail-slower-ratio.",
    )
    parser.add_argument(
        "--save-dodam-output",
        action="store_true",
        help="Save Dodam runner stdout/stderr under the Dodam output directory.",
    )
    parser.add_argument(
        "--dodam-env",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Additional environment variable for the Dodam runner. May be repeated.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    validate_data_dir(data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    queries = filter_queries(load_tpch_queries(Path(args.queries)), args.only)
    if not queries:
        raise SystemExit("no queries selected")

    dodam_stats = {}
    duckdb_stats = {}
    if not args.skip_dodam:
        dodam_stats = run_dodam(args, output_dir / "dodam", queries)
    if not args.skip_duckdb:
        duckdb_stats = run_duckdb(args, output_dir / "duckdb", queries, data_dir)

    report = build_report(queries, dodam_stats, duckdb_stats, args.warmup)
    dodam_profile_summary = load_dodam_profile_summary(output_dir / "dodam")
    if dodam_profile_summary:
        report["dodam_profile_summary"] = dodam_profile_summary
        report["dodam_profile_bottlenecks"] = profile_bottleneck_rows(dodam_profile_summary)
        report["dodam_column_profiles"] = column_profile_rows(dodam_profile_summary)
        report["dodam_scan_input_hot_columns"] = scan_input_hot_column_rows(
            dodam_profile_summary
        )
        report["dodam_direct_primitive_profiles"] = direct_primitive_profile_rows(
            dodam_profile_summary
        )
        report["dodam_direct_selected_rejects"] = direct_selected_reject_rows(
            dodam_profile_summary
        )
        report["dodam_physical_maturity"] = physical_maturity_rows(dodam_profile_summary)
        report["dodam_physical_maturity_actions"] = physical_maturity_action_rows(
            dodam_profile_summary
        )
    print_report(report)
    if args.json_out:
        json_out = Path(args.json_out)
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(report, indent=2) + "\n")
    failures = slower_threshold_failures(
        report,
        ratio_threshold=args.fail_slower_ratio,
        gap_ms_threshold=args.fail_slower_gap_ms,
    )
    if failures:
        for row in failures:
            print(
                "threshold failure: "
                f"{row['query']} ratio={row['ratio']:.3f}x "
                f"gap={row['gap_s'] * 1000.0:.3f}ms",
                file=sys.stderr,
            )
        return 1
    return 0


def validate_data_dir(data_dir: Path) -> None:
    missing = [table for table in sorted(TABLES) if not (data_dir / f"{table}.parquet").exists()]
    if missing:
        raise SystemExit(f"missing parquet table(s) under {data_dir}: {', '.join(missing)}")


def load_tpch_queries(path: Path) -> list[TpchQuery]:
    source = path.read_text()
    queries = []
    for name, sql in re.findall(
        r'name: "([^"]+)",\n\s+expected_status: "[^"]+",\n\s+sql: r#"(.*?)"#,',
        source,
        re.S,
    ):
        queries.append(TpchQuery(name=name, sql=sql.strip()))
    return queries


def filter_queries(queries: list[TpchQuery], only: str) -> list[TpchQuery]:
    filters = [item.strip().lower() for item in only.split(",") if item.strip()]
    if not filters:
        return queries
    selected = []
    for query in queries:
        name = query.name.lower()
        if any(name == filter_ or name.startswith(filter_) or filter_ in name for filter_ in filters):
            selected.append(query)
    return selected


def run_dodam(args: argparse.Namespace, output_dir: Path, queries: list[TpchQuery]) -> dict[str, QueryStats]:
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        args.dodam,
        "--data-dir",
        args.data_dir,
        "--queries",
        args.queries,
        "--output-dir",
        str(output_dir),
        "--batch-size",
        str(args.batch_size),
        "--repeats",
        str(args.repeats),
    ]
    if args.only:
        command.extend(["--only", args.only])
    env = None
    if args.dodam_env:
        env = dict(os.environ)
        for item in args.dodam_env:
            name, sep, value = item.partition("=")
            if not sep or not name:
                raise SystemExit(f"--dodam-env expects NAME=VALUE, got {item!r}")
            env[name] = value
    completed = subprocess.run(
        command,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=args.timeout * max(1, args.repeats),
    )
    if args.save_dodam_output:
        (output_dir / "runner.stdout").write_text(completed.stdout)
        (output_dir / "runner.stderr").write_text(completed.stderr)
        profiles = parse_tpch_profiles(completed.stderr)
        if profiles:
            (output_dir / "tpch_profile.json").write_text(json.dumps(profiles, indent=2) + "\n")
            summary = summarize_tpch_profiles(profiles)
            (output_dir / "tpch_profile_summary.json").write_text(
                json.dumps(summary, indent=2) + "\n"
            )
    if completed.returncode != 0:
        sys.stderr.write(completed.stdout)
        sys.stderr.write(completed.stderr)
        raise SystemExit(f"Dodam runner failed with status {completed.returncode}")
    stats = {query.name: QueryStats([]) for query in queries}
    for _repeat, name, seconds in TPCH_OUTPUT.findall(completed.stdout):
        if name in stats:
            stats[name].samples.append(float(seconds))
    return stats


def load_dodam_profile_summary(output_dir: Path) -> list[dict]:
    path = output_dir / "tpch_profile_summary.json"
    if not path.exists():
        return []
    return json.loads(path.read_text())


def profile_bottleneck_rows(summary: list[dict]) -> list[dict]:
    rows = []
    for row in summary:
        if row.get("kind") == "physical_event":
            continue
        bottleneck = row.get("bottleneck")
        if not bottleneck:
            continue
        rows.append(
            {
                "label": row.get("label"),
                "kind": row.get("kind"),
                "bottleneck": bottleneck,
                "recommended_focus": row.get("recommended_focus"),
                "total_ms_median_warm": row.get("total_ms_median_warm"),
                "projected_columns_median_warm": row.get("projected_columns_median_warm"),
                "read_fraction_median_warm": row.get("read_fraction_median_warm"),
                "consume_fraction_median_warm": row.get("consume_fraction_median_warm"),
                "rows_per_chunk_median_warm": row.get("rows_per_chunk_median_warm"),
                "reader_next_avg_ms_median_warm": row.get(
                    "reader_next_avg_ms_median_warm"
                ),
                "reader_next_p95_max_ms_median_warm": row.get(
                    "reader_next_p95_max_ms_median_warm"
                ),
                "compressed_ratio_median_warm": row.get("compressed_ratio_median_warm"),
            }
        )
    return sorted(
        rows,
        key=lambda row: row.get("total_ms_median_warm") or 0.0,
        reverse=True,
    )


def column_profile_rows(summary: list[dict]) -> list[dict]:
    rows = []
    for row in summary:
        if row.get("kind") != "parquet_column":
            continue
        rows.append(
            {
                "label": row.get("label"),
                "column": row.get("column"),
                "compressed_scanned_median_warm": row.get("compressed_scanned_median_warm"),
                "compressed_total_median_warm": row.get("compressed_total_median_warm"),
                "uncompressed_scanned_median_warm": row.get(
                    "uncompressed_scanned_median_warm"
                ),
                "compressed_ratio_median_warm": row.get("compressed_ratio_median_warm"),
                "encodings": row.get("encodings"),
            }
        )
    return sorted(
        rows,
        key=lambda row: row.get("compressed_scanned_median_warm") or 0.0,
        reverse=True,
    )


def scan_input_hot_column_rows(summary: list[dict]) -> list[dict]:
    scan_rows = [
        row
        for row in summary
        if row.get("kind")
        in {
            "row_group_map_summary",
            "row_group_map_view_summary",
            "dictionary_row_group_map_summary",
            "dictionary_row_group_map_view_summary",
        }
        and row.get("bottleneck") in {"read-heavy", "read-heavy-narrow", "read-heavy-wide"}
    ]
    column_rows = [row for row in summary if row.get("kind") == "parquet_column"]
    output = []
    for scan in scan_rows:
        scan_label = scan.get("label") or ""
        source = scan_label.split("[", 1)[0]
        if not source:
            continue
        matching_columns = [
            column
            for column in column_rows
            if (column.get("label") or "").endswith(source)
            or (column.get("label") or "") == source
        ]
        total_scanned = sum(
            column.get("compressed_scanned_median_warm") or 0.0
            for column in matching_columns
        )
        for column in matching_columns:
            scanned = column.get("compressed_scanned_median_warm") or 0.0
            output.append(
                {
                    "scan_label": scan_label,
                    "source": column.get("label"),
                    "column": column.get("column"),
                    "bottleneck": scan.get("bottleneck"),
                    "recommended_focus": scan.get("recommended_focus"),
                    "scan_total_ms_median_warm": scan.get("total_ms_median_warm"),
                    "scan_read_fraction_median_warm": scan.get(
                        "read_fraction_median_warm"
                    ),
                    "compressed_scanned_median_warm": scanned,
                    "compressed_share_median_warm": (
                        scanned / total_scanned if total_scanned > 0.0 else None
                    ),
                    "encodings": column.get("encodings"),
                }
            )
    return sorted(
        output,
        key=lambda row: (
            row.get("scan_total_ms_median_warm") or 0.0,
            row.get("compressed_scanned_median_warm") or 0.0,
        ),
        reverse=True,
    )


def direct_primitive_profile_rows(summary: list[dict]) -> list[dict]:
    rows = []
    for row in summary:
        if row.get("kind") != "direct_primitive":
            continue
        rows.append(
            {
                "label": row.get("label"),
                "reader_kind": row.get("reader_kind"),
                "column": row.get("column"),
                "read_ms_median_warm": row.get("read_ms_median_warm"),
                "consume_ms_median_warm": row.get("consume_ms_median_warm"),
                "column_read_ms_median_warm": row.get("column_read_ms_median_warm"),
                "rows_median_warm": row.get("rows_median_warm"),
                "batches_median_warm": row.get("batches_median_warm"),
                "selected_ratio_median_warm": row.get("selected_ratio_median_warm"),
                "dictionary_range_pages_median_warm": row.get(
                    "dictionary_range_pages_median_warm"
                ),
                "dictionary_block_pages_median_warm": row.get(
                    "dictionary_block_pages_median_warm"
                ),
                "dictionary_block_rows_median_warm": row.get(
                    "dictionary_block_rows_median_warm"
                ),
                "selected_page_skip_pages_median_warm": row.get(
                    "selected_page_skip_pages_median_warm"
                ),
                "selected_page_skip_rows_median_warm": row.get(
                    "selected_page_skip_rows_median_warm"
                ),
            }
        )
    return sorted(
        rows,
        key=lambda row: row.get("column_read_ms_median_warm") or 0.0,
        reverse=True,
    )


def direct_selected_reject_rows(summary: list[dict]) -> list[dict]:
    rows = []
    for row in summary:
        if row.get("kind") != "direct_selected_reject":
            continue
        rows.append(
            {
                "label": row.get("label"),
                "reason": row.get("reason"),
                "count": row.get("count_median_warm"),
            }
        )
    return sorted(rows, key=lambda row: row.get("count") or 0.0, reverse=True)


def physical_maturity_rows(summary: list[dict]) -> list[dict]:
    rows = []
    for row in summary:
        kind = row.get("kind")
        if kind == "physical_event":
            event = {}
            for key, value in row.items():
                if key.endswith("_samples") or key.endswith("_min_warm") or key.endswith("_max_warm"):
                    continue
                if key.endswith("_median_warm"):
                    event[key.removesuffix("_median_warm")] = value
                else:
                    event[key] = value
            if "status" in event and "physical_status" not in event:
                event["physical_status"] = event["status"]
            rows.append(event)
            continue
        if kind in {
            "row_group_map_summary",
            "row_group_map_view_summary",
            "dictionary_row_group_map_summary",
            "dictionary_row_group_map_view_summary",
        }:
            rows.append(
                {
                    "label": row.get("label"),
                    "kind": kind,
                    "physical_status": "blocked_row_group_map",
                    "bottleneck": row.get("bottleneck"),
                    "recommended_focus": row.get("recommended_focus"),
                    "read_fraction": row.get("read_fraction_median_warm"),
                    "consume_fraction": row.get("consume_fraction_median_warm"),
                    "rows": row.get("rows_median_warm"),
                    "batches": row.get("batches_median_warm"),
                    "compressed_ratio": row.get("compressed_ratio_median_warm"),
                }
            )
            continue
        if kind == "direct_primitive":
            selected_batches = row.get("selected_batches_median_warm") or 0.0
            full_batches = row.get("full_batches_median_warm") or 0.0
            if selected_batches > 0 and full_batches == 0:
                status = "selected_payload_path"
            elif selected_batches > 0 and full_batches > 0:
                status = "mixed_selected_full_payload"
            elif full_batches > 0:
                status = "full_payload_fallback"
            else:
                status = "direct_primitive_path"
            rows.append(
                {
                    "label": row.get("label"),
                    "kind": kind,
                    "reader_kind": row.get("reader_kind"),
                    "physical_status": status,
                    "selected_ratio": row.get("selected_ratio_median_warm"),
                    "selected_runs": row.get("selected_runs_median_warm"),
                    "selected_batches": selected_batches,
                    "full_batches": full_batches,
                    "read_ms": row.get("read_ms_median_warm"),
                    "consume_ms": row.get("consume_ms_median_warm"),
                    "selected_payload_ms": row.get("selected_payload_ms_median_warm"),
                    "dictionary_range_pages": row.get(
                        "dictionary_range_pages_median_warm"
                    ),
                    "dictionary_block_pages": row.get(
                        "dictionary_block_pages_median_warm"
                    ),
                    "dictionary_block_rows": row.get(
                        "dictionary_block_rows_median_warm"
                    ),
                    "selected_page_skip_pages": row.get(
                        "selected_page_skip_pages_median_warm"
                    ),
                    "selected_page_skip_rows": row.get(
                        "selected_page_skip_rows_median_warm"
                    ),
                }
            )
            continue
        if kind == "direct_selected_reject":
            rows.append(
                {
                    "label": row.get("label"),
                    "kind": kind,
                    "physical_status": "decoder_or_cost_rejected",
                    "reason": row.get("reason"),
                    "count": row.get("count_median_warm"),
                }
            )
            continue
        if kind == "i64_set_filter":
            rows.append(
                {
                    "label": row.get("label"),
                    "kind": kind,
                    "physical_status": "i64_set_row_filter",
                    "read_fraction": row.get("read_fraction_median_warm"),
                    "rows": row.get("rows_median_warm"),
                    "batches": row.get("batches_median_warm"),
                    "compressed_scanned": row.get("compressed_scanned_median_warm"),
                }
            )
            continue
        if kind == "late_materialized":
            selector_runs = row.get("selector_runs_median_warm") or 0.0
            selected_rows = row.get("selected_rows_median_warm") or 0.0
            rows_value = row.get("rows_median_warm") or 0.0
            selector_run_ratio = selector_runs / rows_value if rows_value > 0 else None
            selector_runs_per_selected = (
                selector_runs / selected_rows if selected_rows > 0 else None
            )
            status = (
                "fragmented_late_materialized"
                if selector_runs_per_selected is not None
                and selector_runs_per_selected > 0.50
                else "late_materialized_blocked_path"
            )
            rows.append(
                {
                    "label": row.get("label"),
                    "kind": kind,
                    "physical_status": status,
                    "selected_ratio": row.get("selected_ratio_median_warm"),
                    "selector_runs": selector_runs,
                    "selector_run_ratio": selector_run_ratio,
                    "selector_runs_per_selected": selector_runs_per_selected,
                    "rows": rows_value,
                    "selected_rows": selected_rows,
                    "chunks": row.get("chunks_median_warm"),
                    "predicate_read_ms": row.get("predicate_read_ms_median_warm"),
                    "payload_read_ms": row.get("payload_read_ms_median_warm"),
                    "predicate_batches": row.get("predicate_batches_median_warm"),
                    "payload_batches": row.get("payload_batches_median_warm"),
                    "payload_rows": row.get("payload_rows_median_warm"),
                }
            )
            continue
        if kind in {"fold", "pair_collect"}:
            rows.append(
                {
                    "label": row.get("label"),
                    "kind": kind,
                    "physical_status": (
                        "parallel_pair_collect" if kind == "pair_collect" else "parallel_fold"
                    ),
                    "total_ms": row.get("total_ms_median_warm"),
                    "stream_read_ms": row.get("stream_read_ms_median_warm"),
                    "worker_wait_merge_ms": row.get("worker_wait_merge_ms_median_warm"),
                    "worker_wait_ms": row.get("worker_wait_ms_median_warm"),
                    "flatten_ms": row.get("flatten_ms_median_warm"),
                    "pairs": row.get("pairs_median_warm"),
                    "max_partial_pairs": row.get("max_partial_pairs_median_warm"),
                }
            )
            continue
        if kind in {"hash_map_build", "dense_i32_map_build"}:
            rows.append(
                {
                    "label": row.get("label"),
                    "kind": kind,
                    "physical_status": (
                        "dense_i32_map_from_pairs"
                        if kind == "dense_i32_map_build"
                        else "hash_map_from_pairs"
                    ),
                    "build_ms": row.get("build_ms_median_warm"),
                    "pairs": row.get("pairs_median_warm"),
                    "entries": row.get("entries_median_warm"),
                    "layout": row.get("layout"),
                }
            )
    return rows


def physical_maturity_action_rows(summary: list[dict]) -> list[dict]:
    rows = []
    for row in summary:
        kind = row.get("kind")
        if kind in {
            "row_group_map_summary",
            "row_group_map_view_summary",
            "dictionary_row_group_map_summary",
            "dictionary_row_group_map_view_summary",
        }:
            total_ms = row.get("total_ms_median_warm")
            read_fraction = row.get("read_fraction_median_warm")
            consume_fraction = row.get("consume_fraction_median_warm")
            if total_ms is None:
                continue
            rows.append(
                {
                    "label": row.get("label"),
                    "kind": kind,
                    "cost_ms": total_ms,
                    "physical_status": "blocked_row_group_map",
                    "focus": row_group_map_recommended_focus(
                        row.get("bottleneck") or "",
                        row.get("projected_columns_median_warm"),
                    ),
                    "read_fraction": read_fraction,
                    "consume_fraction": consume_fraction,
                    "evidence": (
                        f"read={format_percent(read_fraction)} consume={format_percent(consume_fraction)} "
                        f"cols={format_number(row.get('projected_columns_median_warm'))}"
                    ),
                }
            )
            continue
        if kind == "direct_primitive":
            read_ms = row.get("read_ms_median_warm")
            consume_ms = row.get("consume_ms_median_warm")
            selected_batches = row.get("selected_batches_median_warm") or 0.0
            full_batches = row.get("full_batches_median_warm") or 0.0
            if read_ms is None:
                continue
            focus = (
                "selected payload/read skipping"
                if full_batches > 0 and selected_batches == 0
                else "direct primitive column reader"
            )
            rows.append(
                {
                    "label": row.get("label"),
                    "kind": kind,
                    "cost_ms": read_ms + (consume_ms or 0.0),
                    "physical_status": (
                        "full_payload_fallback"
                        if full_batches > 0 and selected_batches == 0
                        else "direct_primitive_path"
                    ),
                    "focus": focus,
                    "read_fraction": (
                        read_ms / (read_ms + consume_ms) if consume_ms and read_ms + consume_ms > 0 else None
                    ),
                    "consume_fraction": (
                        consume_ms / (read_ms + consume_ms) if consume_ms and read_ms + consume_ms > 0 else None
                    ),
                    "evidence": (
                        f"column={row.get('column') or ''} "
                        f"selected={format_percent(row.get('selected_ratio_median_warm'))}"
                    ),
                }
            )
            continue
        if kind == "i64_set_filter":
            elapsed_ms = row.get("elapsed_ms_median_warm")
            if elapsed_ms is None:
                continue
            rows.append(
                {
                    "label": row.get("label"),
                    "kind": kind,
                    "cost_ms": elapsed_ms,
                    "physical_status": "i64_set_row_filter",
                    "focus": "row filter selectivity/cost model",
                    "read_fraction": row.get("read_fraction_median_warm"),
                    "consume_fraction": None,
                    "evidence": (
                        f"rows={format_number(row.get('rows_median_warm'))} "
                        f"batches={format_number(row.get('batches_median_warm'))}"
                    ),
                }
            )
            continue
        if kind == "late_materialized":
            predicate_read_ms = row.get("predicate_read_ms_median_warm")
            payload_read_ms = row.get("payload_read_ms_median_warm")
            selected_rows = row.get("selected_rows_median_warm") or 0.0
            selector_runs = row.get("selector_runs_median_warm") or 0.0
            rows_value = row.get("rows_median_warm") or 0.0
            selected_ratio = row.get("selected_ratio_median_warm")
            selector_runs_per_selected = (
                selector_runs / selected_rows if selected_rows > 0 else None
            )
            cost_ms = (predicate_read_ms or 0.0) + (payload_read_ms or 0.0)
            if cost_ms <= 0.0:
                continue
            fragmented = (
                selector_runs_per_selected is not None
                and selector_runs_per_selected > 0.50
            )
            payload_fraction = payload_read_ms / cost_ms if payload_read_ms is not None else None
            predicate_fraction = (
                predicate_read_ms / cost_ms if predicate_read_ms is not None else None
            )
            if fragmented and payload_fraction is not None and payload_fraction > 0.65:
                focus = "reject fragmented selected payload"
            elif fragmented:
                focus = "coalesce fragmented row selection"
            elif predicate_fraction is not None and predicate_fraction > 0.65:
                focus = "predicate column decode"
            elif payload_fraction is not None and payload_fraction > 0.65:
                focus = "selected payload decode"
            else:
                focus = "late selection/payload balance"
            rows.append(
                {
                    "label": row.get("label"),
                    "kind": kind,
                    "cost_ms": cost_ms,
                    "physical_status": (
                        "fragmented_late_materialized"
                        if fragmented
                        else "late_materialized_blocked_path"
                    ),
                    "focus": focus,
                    "read_fraction": predicate_fraction,
                    "consume_fraction": payload_fraction,
                    "evidence": (
                        f"selected={format_percent(selected_ratio)} "
                        f"runs/selected={format_number(selector_runs_per_selected)} "
                        f"payload_rows={format_number(row.get('payload_rows_median_warm'))} "
                        f"rows={format_number(rows_value)}"
                    ),
                }
            )
            continue
        if kind in {"pair_collect", "hash_map_build", "dense_i32_map_build"}:
            cost_ms = row.get("total_ms_median_warm") or row.get("build_ms_median_warm")
            if cost_ms is None:
                continue
            rows.append(
                {
                    "label": row.get("label"),
                    "kind": kind,
                    "cost_ms": cost_ms,
                    "physical_status": (
                        "parallel_pair_collect" if kind == "pair_collect" else "build_lookup"
                    ),
                    "focus": "build-side memory/layout",
                    "read_fraction": None,
                    "consume_fraction": None,
                    "evidence": f"pairs={format_number(row.get('pairs_median_warm'))}",
                }
            )
    return sorted(rows, key=lambda row: row.get("cost_ms") or 0.0, reverse=True)


def parse_tpch_profiles(text: str) -> list[dict]:
    rows = []
    for line in text.splitlines():
        physical = PHYSICAL_EVENT.search(line)
        if physical:
            event = parse_physical_event_fields(physical.group("fields"))
            event["label"] = physical.group("label")
            event["kind"] = "physical_event"
            rows.append(event)
            continue
        pair_collect = TPCH_PROFILE_PAIR_COLLECT.search(line)
        if pair_collect:
            rows.append(
                {
                    "label": pair_collect.group("label"),
                    "kind": "pair_collect",
                    "total_ms": float(pair_collect.group("total")),
                    "stream_read_ms": float(pair_collect.group("stream")),
                    "worker_wait_merge_ms": float(pair_collect.group("merge")),
                    "worker_wait_ms": float(pair_collect.group("wait")),
                    "flatten_ms": float(pair_collect.group("flatten")),
                    "view_chunks": int(pair_collect.group("count")),
                    "pairs": int(pair_collect.group("pairs")),
                    "max_partial_pairs": int(pair_collect.group("max_partial_pairs")),
                }
            )
            continue
        hash_map_build = TPCH_PROFILE_HASH_MAP_BUILD.search(line)
        if hash_map_build:
            rows.append(
                {
                    "label": hash_map_build.group("label"),
                    "kind": "hash_map_build",
                    "build_ms": float(hash_map_build.group("build")),
                    "pairs": int(hash_map_build.group("pairs")),
                    "entries": int(hash_map_build.group("entries")),
                }
            )
            continue
        dense_i32_map_build = TPCH_PROFILE_DENSE_I32_MAP_BUILD.search(line)
        if dense_i32_map_build:
            rows.append(
                {
                    "label": dense_i32_map_build.group("label"),
                    "kind": "dense_i32_map_build",
                    "build_ms": float(dense_i32_map_build.group("build")),
                    "pairs": int(dense_i32_map_build.group("pairs")),
                    "entries": int(dense_i32_map_build.group("entries")),
                    "layout": dense_i32_map_build.group("layout"),
                }
            )
            continue
        direct_reject = DIRECT_SELECTED_REJECT.search(line)
        if direct_reject:
            rows.append(
                {
                    "label": direct_reject.group("label"),
                    "kind": "direct_selected_reject",
                    "reason": direct_reject.group("reason"),
                    "count": 1,
                }
            )
            continue
        direct_primitive = DIRECT_PRIMITIVE_PROFILE.search(line)
        if direct_primitive:
            columns = [
                column.strip()
                for column in direct_primitive.group("columns").split(",")
                if column.strip()
            ]
            column_read = parse_direct_column_read(direct_primitive.group("column_read"))
            for index, column in enumerate(columns):
                rows.append(
                    {
                        "label": direct_primitive.group("label"),
                        "kind": "direct_primitive",
                        "reader_kind": direct_primitive.group("reader_kind") or "",
                        "column": column,
                        "row_groups": int(direct_primitive.group("row_groups")),
                        "rows": int(direct_primitive.group("rows")),
                        "batches": int(direct_primitive.group("batches")),
                        "read_ms": float(direct_primitive.group("read")),
                        "consume_ms": float(direct_primitive.group("consume")),
                        "column_read_ms": column_read.get(index, 0.0),
                        "selected_predicate_ms": float(
                            direct_primitive.group("selected_predicate")
                        ),
                        "selected_payload_ms": float(direct_primitive.group("selected_payload")),
                        "selected_dictionary_ms": float(
                            direct_primitive.group("selected_dictionary")
                        ),
                        "selected_rows": int(direct_primitive.group("selected_rows")),
                        "selected_ratio": float(direct_primitive.group("selected_ratio")),
                        "selected_runs": int(direct_primitive.group("selected_runs")),
                        "selected_batches": int(direct_primitive.group("selected_batches")),
                        "full_batches": int(direct_primitive.group("full_batches")),
                        "selected_skip_calls": int(
                            direct_primitive.group("selected_skip_calls")
                        ),
                        "selected_skipped_rows": int(
                            direct_primitive.group("selected_skipped_rows")
                        ),
                        "selected_read_calls": int(
                            direct_primitive.group("selected_read_calls")
                        ),
                        "selected_read_rows": int(direct_primitive.group("selected_read_rows")),
                        "dictionary_range_pages": int(
                            direct_primitive.group("dictionary_range_pages") or 0
                        ),
                        "dictionary_block_pages": int(
                            direct_primitive.group("dictionary_block_pages") or 0
                        ),
                        "dictionary_block_rows": int(
                            direct_primitive.group("dictionary_block_rows") or 0
                        ),
                        "selected_page_skip_pages": int(
                            direct_primitive.group("selected_page_skip_pages") or 0
                        ),
                        "selected_page_skip_rows": int(
                            direct_primitive.group("selected_page_skip_rows") or 0
                        ),
                    }
                )
            continue
        column_profile = PARQUET_COLUMN_PROFILE.search(line)
        if column_profile:
            row_groups = int(column_profile.group("row_groups"))
            total_row_groups = int(column_profile.group("total_row_groups"))
            for column in parse_parquet_column_profile_columns(column_profile.group("columns")):
                compressed_scanned = column["compressed_scanned"]
                compressed_total = column["compressed_total"]
                rows.append(
                    {
                        "label": column_profile.group("label"),
                        "kind": "parquet_column",
                        "column": column["column"],
                        "row_groups": row_groups,
                        "total_row_groups": total_row_groups,
                        "compressed_scanned": compressed_scanned,
                        "compressed_total": compressed_total,
                        "uncompressed_scanned": column["uncompressed_scanned"],
                        "compressed_ratio": (
                            compressed_scanned / compressed_total if compressed_total else None
                        ),
                        "encodings": column["encodings"],
                    }
                )
            continue
        row_group_summary = ROW_GROUP_MAP_SUMMARY.search(line)
        if row_group_summary:
            total_ms = float(row_group_summary.group("total"))
            read_next_ms = float(row_group_summary.group("read_next"))
            consume_ms = float(row_group_summary.group("consume"))
            chunks = int(row_group_summary.group("chunks"))
            rows_count = int(row_group_summary.group("rows"))
            compressed_scanned = int(row_group_summary.group("compressed_scanned"))
            compressed_total = int(row_group_summary.group("compressed_total"))
            projected_columns = int(row_group_summary.group("columns") or 0)
            read_fraction = read_next_ms / total_ms if total_ms > 0.0 else None
            consume_fraction = consume_ms / total_ms if total_ms > 0.0 else None
            bottleneck = classify_row_group_map_bottleneck(
                read_fraction, consume_fraction, projected_columns
            )
            rows.append(
                {
                    "label": row_group_summary.group("label"),
                    "kind": row_group_summary.group("kind"),
                    "chunks": chunks,
                    "row_groups": int(row_group_summary.group("row_groups")),
                    "projected_columns": projected_columns,
                    "rows": rows_count,
                    "batches": int(row_group_summary.group("batches")),
                    "zero_batches": int(row_group_summary.group("zero_batches")),
                    "total_ms": total_ms,
                    "read_next_ms": read_next_ms,
                    "reader_next_ms": float(row_group_summary.group("reader_next")),
                    "reader_next_avg_ms": float(row_group_summary.group("reader_next_avg")),
                    "reader_next_p95_max_ms": float(
                        row_group_summary.group("reader_next_p95_max")
                    ),
                    "reader_next_max_ms": float(row_group_summary.group("reader_next_max")),
                    "reader_calls": int(row_group_summary.group("reader_calls")),
                    "reader_eof": int(row_group_summary.group("reader_eof")),
                    "avg_batch_rows": float(row_group_summary.group("avg_batch_rows")),
                    "consume_ms": consume_ms,
                    "read_fraction": read_fraction,
                    "consume_fraction": consume_fraction,
                    "bottleneck": bottleneck,
                    "recommended_focus": row_group_map_recommended_focus(
                        bottleneck, projected_columns
                    ),
                    "rows_per_chunk": rows_count / chunks if chunks else None,
                    "compressed_scanned": compressed_scanned,
                    "compressed_total": compressed_total,
                    "compressed_ratio": (
                        compressed_scanned / compressed_total if compressed_total else None
                    ),
                }
            )
            continue
        row_group_map = TPCH_PROFILE_ROW_GROUP_MAP.search(line)
        if row_group_map:
            rows.append(
                {
                    "label": row_group_map.group("label"),
                    "kind": "row_group_map",
                    "chunk": int(row_group_map.group("chunk")),
                    "row_groups": int(row_group_map.group("row_groups")),
                    "projected_columns": int(row_group_map.group("columns")),
                    "rows": int(row_group_map.group("rows")),
                    "batches": int(row_group_map.group("batches")),
                    "zero_batches": int(row_group_map.group("zero_batches")),
                    "total_ms": float(row_group_map.group("total")),
                    "setup_ms": float(row_group_map.group("setup")),
                    "metadata_ms": float(row_group_map.group("metadata")),
                    "planning_ms": float(row_group_map.group("planning")),
                    "read_next_ms": float(row_group_map.group("read_next")),
                    "reader_next_ms": float(row_group_map.group("reader_next")),
                    "consume_ms": float(row_group_map.group("consume")),
                }
            )
            continue
        scan = SCAN_PROFILE.search(line)
        if scan:
            rows.append(
                {
                    "label": scan.group("label"),
                    "kind": "scan",
                    "elapsed_ms": float(scan.group("elapsed")),
                    "next_wait_ms": float(scan.group("next_wait")),
                    "consumer_gap_ms": float(scan.group("consumer_gap")),
                    "rows": int(scan.group("rows")),
                    "batches": int(scan.group("batches")),
                    "row_groups": int(scan.group("row_groups")),
                    "total_row_groups": int(scan.group("total_row_groups")),
                    "decode_ms": float(scan.group("decode")),
                }
            )
            continue
        i64_set_filter = I64_SET_FILTER_PROFILE.search(line)
        if i64_set_filter:
            elapsed_ms = float(i64_set_filter.group("elapsed"))
            read_loop_ms = float(i64_set_filter.group("read_loop"))
            rows.append(
                {
                    "label": i64_set_filter.group("label"),
                    "kind": "i64_set_filter",
                    "elapsed_ms": elapsed_ms,
                    "setup_ms": float(i64_set_filter.group("setup")),
                    "read_loop_ms": read_loop_ms,
                    "read_fraction": read_loop_ms / elapsed_ms if elapsed_ms > 0.0 else None,
                    "row_groups": int(i64_set_filter.group("row_groups")),
                    "total_row_groups": int(i64_set_filter.group("total_row_groups")),
                    "projected_columns": int(i64_set_filter.group("columns")),
                    "rows": int(i64_set_filter.group("rows")),
                    "batches": int(i64_set_filter.group("batches")),
                    "zero_batches": int(i64_set_filter.group("zero_batches")),
                    "compressed_scanned": int(i64_set_filter.group("bytes")),
                    "parquet_next_ms": float(i64_set_filter.group("parquet_next")),
                    "parquet_next_avg_ms": float(i64_set_filter.group("parquet_next_avg")),
                    "parquet_next_max_ms": float(i64_set_filter.group("parquet_next_max")),
                    "parquet_calls": int(i64_set_filter.group("parquet_calls")),
                }
            )
            continue
        fold = TPCH_PROFILE_FOLD.search(line)
        if fold:
            rows.append(
                {
                    "label": fold.group("label"),
                    "kind": "fold",
                    "total_ms": float(fold.group("total")),
                    "stream_read_ms": float(fold.group("stream")),
                    "worker_wait_merge_ms": float(fold.group("merge")),
                    fold.group("unit"): int(fold.group("count")),
                }
            )
            continue
        late_materialized = TPCH_PROFILE_LATE_MATERIALIZED.search(line)
        if late_materialized:
            row = {
                "label": late_materialized.group("label"),
                "kind": "late_materialized",
                "rows": int(late_materialized.group("rows")),
                "selected_rows": int(late_materialized.group("selected")),
                "selected_ratio": float(late_materialized.group("ratio")),
                "selector_runs": int(late_materialized.group("selector_runs")),
                "chunks": int(late_materialized.group("chunks")),
            }
            if late_materialized.group("predicate_read") is not None:
                row.update(
                    {
                        "predicate_read_ms": float(late_materialized.group("predicate_read")),
                        "payload_read_ms": float(late_materialized.group("payload_read")),
                        "predicate_batches": int(late_materialized.group("predicate_batches")),
                        "payload_batches": int(late_materialized.group("payload_batches")),
                        "payload_rows": int(late_materialized.group("payload_rows")),
                    }
                )
            rows.append(row)
            continue
        elapsed = TPCH_PROFILE_ELAPSED.search(line)
        if elapsed:
            rows.append(
                {
                    "label": elapsed.group("label"),
                    "kind": "elapsed",
                    "elapsed_ms": float(elapsed.group("ms")),
                }
            )
    return rows


def parse_physical_event_fields(fields: str) -> dict:
    row = {}
    for part in fields.split():
        key, sep, value = part.partition("=")
        if not sep:
            continue
        row[key] = parse_physical_event_value(value)
    return row


def parse_physical_event_value(value: str):
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def parse_parquet_column_profile_columns(columns: str) -> list[dict]:
    rows = []
    for column in PARQUET_COLUMN_PROFILE_COLUMN.finditer(columns):
        rows.append(
            {
                "column": column.group("column").strip(),
                "compressed_scanned": int(column.group("compressed_scanned")),
                "compressed_total": int(column.group("compressed_total")),
                "uncompressed_scanned": int(column.group("uncompressed_scanned")),
                "encodings": column.group("encodings"),
            }
        )
    return rows


def parse_direct_column_read(column_read: str) -> dict[int, float]:
    values = {}
    for item in column_read.split(","):
        index, sep, value = item.partition(":")
        if not sep:
            continue
        try:
            values[int(index)] = float(value)
        except ValueError:
            continue
    return values


def classify_row_group_map_bottleneck(read_fraction, consume_fraction, projected_columns) -> str:
    if read_fraction is not None and read_fraction >= 0.60:
        if projected_columns is not None and projected_columns <= 4:
            return "read-heavy-narrow"
        if projected_columns is not None and projected_columns >= 8:
            return "read-heavy-wide"
        return "read-heavy"
    if consume_fraction is not None and consume_fraction >= 0.50:
        return "consume-heavy"
    return "mixed"


def row_group_map_recommended_focus(bottleneck: str, projected_columns=None) -> str:
    if bottleneck == "read-heavy-narrow":
        if projected_columns is not None and projected_columns <= 4:
            return "parallel page/column decode scheduling"
        return "page/column decode scheduling"
    if bottleneck == "read-heavy-wide":
        return "scan bandwidth and projection pruning"
    if bottleneck == "read-heavy":
        return "scan/decode scheduling"
    if bottleneck == "consume-heavy":
        return "vectorized consume sink"
    return "balanced scan and consume"


def summarize_tpch_profiles(rows: list[dict]) -> list[dict]:
    grouped = {}
    for row in rows:
        label = row.get("label", "")
        kind = row.get("kind", "")
        column = row.get("column", "")
        key = (label, kind, column)
        metrics = grouped.setdefault(key, {})
        for metric, value in row.items():
            if metric in {"label", "kind", "column"} or not isinstance(value, (int, float)):
                continue
            metrics.setdefault(metric, []).append(float(value))
        for metric, value in row.items():
            if metric in {"label", "kind", "column"}:
                continue
            if isinstance(value, str):
                metrics.setdefault(metric, []).append(value)
    summary = []
    for (label, kind, column), metrics in sorted(grouped.items()):
        entry = {"label": label, "kind": kind}
        if column:
            entry["column"] = column
        for metric, values in sorted(metrics.items()):
            if values and isinstance(values[0], str):
                entry[metric] = most_common_value(values)
                entry[f"{metric}_samples"] = len(values)
                continue
            warm = values[1:] if len(values) > 1 else values
            entry[f"{metric}_samples"] = len(values)
            entry[f"{metric}_median_warm"] = statistics.median(warm)
            entry[f"{metric}_min_warm"] = min(warm)
            entry[f"{metric}_max_warm"] = max(warm)
        summary.append(entry)
    return summary


def most_common_value(values: list[str]) -> str:
    counts = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def run_duckdb(
    args: argparse.Namespace,
    output_dir: Path,
    queries: list[TpchQuery],
    data_dir: Path,
) -> dict[str, QueryStats]:
    if args.duckdb_mode == "single-process":
        return run_duckdb_single_process(args, output_dir, queries, data_dir)
    return run_duckdb_cli(args, output_dir, queries, data_dir)


def run_duckdb_cli(
    args: argparse.Namespace,
    output_dir: Path,
    queries: list[TpchQuery],
    data_dir: Path,
) -> dict[str, QueryStats]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = {query.name: QueryStats([]) for query in queries}
    for repeat in range(1, args.repeats + 1):
        repeat_dir = output_dir / f"r{repeat}"
        repeat_dir.mkdir(parents=True, exist_ok=True)
        for query in queries:
            output_path = repeat_dir / f"{query.name}.parquet"
            sql = rewrite_table_refs_for_duckdb(query.sql, data_dir)
            copy_sql = (
                f"COPY ({sql}) TO '{escape_sql_string(str(output_path))}' "
                "(FORMAT parquet, COMPRESSION snappy, ROW_GROUP_SIZE 65536)"
            )
            started = time.perf_counter()
            completed = subprocess.run(
                [args.duckdb, "-c", copy_sql],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=args.timeout,
            )
            elapsed = time.perf_counter() - started
            if completed.returncode != 0:
                sys.stderr.write(completed.stdout)
                sys.stderr.write(completed.stderr)
                raise SystemExit(
                    f"DuckDB failed for repeat={repeat} query={query.name} status={completed.returncode}"
                )
            stats[query.name].samples.append(elapsed)
    return stats


def run_duckdb_single_process(
    args: argparse.Namespace,
    output_dir: Path,
    queries: list[TpchQuery],
    data_dir: Path,
) -> dict[str, QueryStats]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = {query.name: QueryStats([]) for query in queries}
    script = [".bail on", ".timer on"]
    for repeat in range(1, args.repeats + 1):
        repeat_dir = output_dir / f"r{repeat}"
        repeat_dir.mkdir(parents=True, exist_ok=True)
        for query in queries:
            output_path = repeat_dir / f"{query.name}.parquet"
            sql = rewrite_table_refs_for_duckdb(query.sql, data_dir)
            copy_sql = (
                f"COPY ({sql}) TO '{escape_sql_string(str(output_path))}' "
                "(FORMAT parquet, COMPRESSION snappy, ROW_GROUP_SIZE 65536);"
            )
            script.append(f".print DODAM_BENCH repeat={repeat} query={query.name}")
            script.append(copy_sql)
    completed = subprocess.run(
        [args.duckdb],
        input="\n".join(script) + "\n",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=args.timeout * max(1, args.repeats) * max(1, len(queries)),
    )
    if completed.returncode != 0:
        sys.stderr.write(completed.stdout)
        raise SystemExit(f"DuckDB single-process runner failed with status {completed.returncode}")
    current = None
    for line in completed.stdout.splitlines():
        line = line.strip()
        marker = DUCKDB_MARK.match(line)
        if marker:
            current = (int(marker.group(1)), marker.group(2))
            continue
        timer = DUCKDB_TIMER.match(line)
        if timer and current is not None:
            _repeat, name = current
            if name in stats:
                stats[name].samples.append(float(timer.group(1)))
            current = None
    missing = [name for name, query_stats in stats.items() if len(query_stats.samples) != args.repeats]
    if missing:
        raise SystemExit(
            "DuckDB single-process timing parse failed for: "
            + ", ".join(f"{name}({len(stats[name].samples)}/{args.repeats})" for name in missing)
        )
    return stats


def rewrite_table_refs_for_duckdb(sql: str, data_dir: Path) -> str:
    tokens = tokenize_sql(sql)
    output = []
    index = 0
    expect_table = False
    while index < len(tokens):
        token = tokens[index]
        if token.kind == "word" and expect_table and token.value.lower() in TABLES:
            table = token.value.lower()
            alias, next_index = consume_alias(tokens, index + 1, table)
            table_path = escape_sql_string(str(data_dir / f"{table}.parquet"))
            output.append(f"read_parquet('{table_path}') AS {alias}")
            index = next_index
            expect_table = False
            continue
        if token.kind == "word":
            lower = token.value.lower()
            if lower in {"from", "join"}:
                expect_table = True
            elif expect_table:
                expect_table = False
            output.append(token.value)
        elif token.value == ",":
            expect_table = True
            output.append(token.value)
        else:
            output.append(token.value)
        index += 1
    return "".join(output).strip().rstrip(";")


def consume_alias(tokens: list["Token"], index: int, table: str) -> tuple[str, int]:
    original = index
    index = skip_ws(tokens, index)
    if token_word(tokens, index, "as"):
        alias_index = skip_ws(tokens, index + 1)
        if alias_index < len(tokens) and tokens[alias_index].kind == "word":
            return tokens[alias_index].value, alias_index + 1
        return table, index
    if index < len(tokens) and tokens[index].kind == "word" and tokens[index].value.lower() not in KEYWORDS:
        return tokens[index].value, index + 1
    return table, original


def skip_ws(tokens: list["Token"], index: int) -> int:
    while index < len(tokens) and tokens[index].kind == "ws":
        index += 1
    return index


def token_word(tokens: list["Token"], index: int, value: str) -> bool:
    return index < len(tokens) and tokens[index].kind == "word" and tokens[index].value.lower() == value


@dataclass(frozen=True)
class Token:
    kind: str
    value: str


def tokenize_sql(sql: str) -> list[Token]:
    tokens = []
    index = 0
    while index < len(sql):
        ch = sql[index]
        if ch.isspace():
            start = index
            index += 1
            while index < len(sql) and sql[index].isspace():
                index += 1
            tokens.append(Token("ws", sql[start:index]))
        elif ch == "'":
            start = index
            index += 1
            while index < len(sql):
                if sql[index] == "'":
                    index += 1
                    if index < len(sql) and sql[index] == "'":
                        index += 1
                        continue
                    break
                index += 1
            tokens.append(Token("quoted", sql[start:index]))
        elif ch.isascii() and (ch.isalnum() or ch == "_"):
            start = index
            index += 1
            while index < len(sql) and sql[index].isascii() and (sql[index].isalnum() or sql[index] == "_"):
                index += 1
            tokens.append(Token("word", sql[start:index]))
        else:
            tokens.append(Token("punct", ch))
            index += 1
    return tokens


def escape_sql_string(value: str) -> str:
    return value.replace("'", "''")


def build_report(
    queries: list[TpchQuery],
    dodam_stats: dict[str, QueryStats],
    duckdb_stats: dict[str, QueryStats],
    warmup: int,
) -> dict:
    rows = []
    total_dodam = 0.0
    total_duckdb = 0.0
    for query in queries:
        dodam = dodam_stats.get(query.name)
        duckdb = duckdb_stats.get(query.name)
        dodam_median = dodam.median(warmup) if dodam else None
        duckdb_median = duckdb.median(warmup) if duckdb else None
        ratio = None
        gap = None
        if dodam_median is not None:
            total_dodam += dodam_median
        if duckdb_median is not None:
            total_duckdb += duckdb_median
        if dodam_median is not None and duckdb_median is not None and duckdb_median > 0.0:
            ratio = dodam_median / duckdb_median
            gap = dodam_median - duckdb_median
        rows.append(
            {
                "query": query.name,
                "dodam_median_s": dodam_median,
                "duckdb_median_s": duckdb_median,
                "dodam_min_s": dodam.min(warmup) if dodam else None,
                "dodam_max_s": dodam.max(warmup) if dodam else None,
                "dodam_stdev_s": dodam.stdev(warmup) if dodam else None,
                "duckdb_min_s": duckdb.min(warmup) if duckdb else None,
                "duckdb_max_s": duckdb.max(warmup) if duckdb else None,
                "duckdb_stdev_s": duckdb.stdev(warmup) if duckdb else None,
                "ratio": ratio,
                "gap_s": gap,
                "dodam_samples_s": dodam.samples if dodam else [],
                "duckdb_samples_s": duckdb.samples if duckdb else [],
            }
        )
    comparable = [row for row in rows if row["ratio"] is not None]
    dodam_slower = [row for row in comparable if row["gap_s"] > 0.0]
    return {
        "warmup": warmup,
        "total_dodam_median_sum_s": total_dodam if dodam_stats else None,
        "total_duckdb_median_sum_s": total_duckdb if duckdb_stats else None,
        "total_ratio": (total_dodam / total_duckdb) if dodam_stats and duckdb_stats and total_duckdb > 0 else None,
        "rows": rows,
        "dodam_slower": sorted(dodam_slower, key=lambda row: row["gap_s"], reverse=True),
        "largest_absolute_gaps": sorted(comparable, key=lambda row: abs(row["gap_s"]), reverse=True),
        "slowest_by_ratio": sorted(comparable, key=lambda row: row["ratio"], reverse=True),
    }


def slower_threshold_failures(
    report: dict,
    ratio_threshold: float,
    gap_ms_threshold: float,
) -> list[dict]:
    if ratio_threshold <= 0.0 and gap_ms_threshold <= 0.0:
        return []
    failures = []
    for row in report.get("dodam_slower", []):
        ratio = row.get("ratio")
        gap = row.get("gap_s")
        if ratio is None or gap is None:
            continue
        ratio_failed = ratio_threshold <= 0.0 or ratio > ratio_threshold
        gap_failed = gap_ms_threshold <= 0.0 or gap * 1000.0 > gap_ms_threshold
        if ratio_failed and gap_failed:
            failures.append(row)
    return failures


def print_report(report: dict) -> None:
    total_ratio = report["total_ratio"]
    if total_ratio is not None:
        print(
            "median-sum: "
            f"Dodam={report['total_dodam_median_sum_s']:.6f}s "
            f"DuckDB={report['total_duckdb_median_sum_s']:.6f}s "
            f"ratio={total_ratio:.3f}x"
        )
    print(
        "| query | dodam ms | duckdb ms | ratio | gap ms | dodam min/max/stdev ms | duckdb min/max/stdev ms |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|")
    for row in report["rows"]:
        print(
            f"| {row['query']} | "
            f"{format_ms(row['dodam_median_s'])} | "
            f"{format_ms(row['duckdb_median_s'])} | "
            f"{format_ratio(row['ratio'])} | "
            f"{format_ms(row['gap_s'])} | "
            f"{format_min_max_stdev_ms(row.get('dodam_min_s'), row.get('dodam_max_s'), row.get('dodam_stdev_s'))} | "
            f"{format_min_max_stdev_ms(row.get('duckdb_min_s'), row.get('duckdb_max_s'), row.get('duckdb_stdev_s'))} |"
        )
    if report["dodam_slower"]:
        print("\nDodam slower:")
        for row in report["dodam_slower"]:
            print(
                f"- {row['query']}: gap={row['gap_s'] * 1000.0:.3f}ms "
                f"ratio={row['ratio']:.3f}x"
            )
    else:
        print("\nDodam slower: none")
    if report["largest_absolute_gaps"]:
        print("\nlargest absolute gaps:")
        for row in report["largest_absolute_gaps"][:8]:
            print(
                f"- {row['query']}: gap={row['gap_s'] * 1000.0:.3f}ms "
                f"ratio={row['ratio']:.3f}x"
            )
    if report.get("dodam_profile_bottlenecks"):
        print("\nDodam profile bottlenecks:")
        print(
            "| label | kind | total ms | cols | bottleneck | focus | read % | consume % | rows/chunk | reader avg ms | reader p95 max ms | compressed % |"
        )
        print("|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|")
        for row in report["dodam_profile_bottlenecks"][:8]:
            print(
                f"| {row.get('label') or ''} | "
                f"{row.get('kind') or ''} | "
                f"{format_number(row.get('total_ms_median_warm'))} | "
                f"{format_number(row.get('projected_columns_median_warm'))} | "
                f"{row.get('bottleneck') or ''} | "
                f"{row.get('recommended_focus') or ''} | "
                f"{format_percent(row.get('read_fraction_median_warm'))} | "
                f"{format_percent(row.get('consume_fraction_median_warm'))} | "
                f"{format_number(row.get('rows_per_chunk_median_warm'))} | "
                f"{format_number(row.get('reader_next_avg_ms_median_warm'))} | "
                f"{format_number(row.get('reader_next_p95_max_ms_median_warm'))} | "
                f"{format_percent(row.get('compressed_ratio_median_warm'))} |"
            )
    if report.get("dodam_column_profiles"):
        print("\nDodam column profiles:")
        print("| source | column | scanned MB | total MB | uncompressed MB | scanned % | encodings |")
        print("|---|---|---:|---:|---:|---:|---|")
        for row in report["dodam_column_profiles"][:8]:
            print(
                f"| {row.get('label') or ''} | "
                f"{row.get('column') or ''} | "
                f"{format_bytes_mb(row.get('compressed_scanned_median_warm'))} | "
                f"{format_bytes_mb(row.get('compressed_total_median_warm'))} | "
                f"{format_bytes_mb(row.get('uncompressed_scanned_median_warm'))} | "
                f"{format_percent(row.get('compressed_ratio_median_warm'))} | "
                f"{row.get('encodings') or ''} |"
            )
    if report.get("dodam_scan_input_hot_columns"):
        print("\nDodam scan input hot columns:")
        print("| scan | column | scanned MB | share | read % | focus | encodings |")
        print("|---|---|---:|---:|---:|---|---|")
        for row in report["dodam_scan_input_hot_columns"][:8]:
            print(
                f"| {row.get('scan_label') or ''} | "
                f"{row.get('column') or ''} | "
                f"{format_bytes_mb(row.get('compressed_scanned_median_warm'))} | "
                f"{format_percent(row.get('compressed_share_median_warm'))} | "
                f"{format_percent(row.get('scan_read_fraction_median_warm'))} | "
                f"{row.get('recommended_focus') or ''} | "
                f"{row.get('encodings') or ''} |"
            )
    if report.get("dodam_direct_primitive_profiles"):
        print("\nDodam direct primitive profiles:")
        print("| source | reader | column | column read ms | read ms | consume ms | rows | batches | selected % |")
        print("|---|---|---|---:|---:|---:|---:|---:|---:|")
        for row in report["dodam_direct_primitive_profiles"][:8]:
            print(
                f"| {row.get('label') or ''} | "
                f"{row.get('reader_kind') or ''} | "
                f"{row.get('column') or ''} | "
                f"{format_number(row.get('column_read_ms_median_warm'))} | "
                f"{format_number(row.get('read_ms_median_warm'))} | "
                f"{format_number(row.get('consume_ms_median_warm'))} | "
                f"{format_number(row.get('rows_median_warm'))} | "
                f"{format_number(row.get('batches_median_warm'))} | "
                f"{format_percent(row.get('selected_ratio_median_warm'))} |"
            )
    if report.get("dodam_direct_selected_rejects"):
        print("\nDodam direct selected rejects:")
        for row in report["dodam_direct_selected_rejects"][:8]:
            print(
                f"- {row.get('label') or ''}: "
                f"{row.get('reason') or ''} "
                f"count={format_number(row.get('count'))}"
            )
    if report.get("dodam_physical_maturity"):
        print("\nDodam physical maturity:")
        print("| label | kind | status | focus | read % | consume % | selected % |")
        print("|---|---|---|---|---:|---:|---:|")
        for row in report["dodam_physical_maturity"][:10]:
            print(
                f"| {row.get('label') or ''} | "
                f"{row.get('kind') or ''} | "
                f"{row.get('physical_status') or ''} | "
                f"{row.get('recommended_focus') or row.get('reason') or ''} | "
                f"{format_percent(row.get('read_fraction'))} | "
                f"{format_percent(row.get('consume_fraction'))} | "
                f"{format_percent(row.get('selected_ratio'))} |"
            )
    if report.get("dodam_physical_maturity_actions"):
        print("\nDodam physical maturity actions:")
        print("| label | kind | cost ms | status | focus | read % | consume % | evidence |")
        print("|---|---|---:|---|---|---:|---:|---|")
        for row in report["dodam_physical_maturity_actions"][:10]:
            print(
                f"| {row.get('label') or ''} | "
                f"{row.get('kind') or ''} | "
                f"{format_number(row.get('cost_ms'))} | "
                f"{row.get('physical_status') or ''} | "
                f"{row.get('focus') or ''} | "
                f"{format_percent(row.get('read_fraction'))} | "
                f"{format_percent(row.get('consume_fraction'))} | "
                f"{row.get('evidence') or ''} |"
            )


def format_ms(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value * 1000.0:.3f}"


def format_min_max_stdev_ms(
    min_value: float | None,
    max_value: float | None,
    stdev_value: float | None,
) -> str:
    if min_value is None or max_value is None or stdev_value is None:
        return ""
    return f"{min_value * 1000.0:.3f}/{max_value * 1000.0:.3f}/{stdev_value * 1000.0:.3f}"


def format_ratio(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.3f}x"


def format_number(value) -> str:
    if value is None:
        return ""
    return f"{float(value):.3f}"


def format_percent(value) -> str:
    if value is None:
        return ""
    return f"{float(value) * 100.0:.1f}"


def format_bytes_mb(value) -> str:
    if value is None:
        return ""
    return f"{float(value) / (1024.0 * 1024.0):.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
