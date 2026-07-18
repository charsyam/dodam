#!/usr/bin/env python3
import argparse
import json
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
    print_report(report)
    if args.json_out:
        json_out = Path(args.json_out)
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(report, indent=2) + "\n")
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
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=args.timeout * max(1, args.repeats),
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


def print_report(report: dict) -> None:
    total_ratio = report["total_ratio"]
    if total_ratio is not None:
        print(
            "median-sum: "
            f"Dodam={report['total_dodam_median_sum_s']:.6f}s "
            f"DuckDB={report['total_duckdb_median_sum_s']:.6f}s "
            f"ratio={total_ratio:.3f}x"
        )
    print("| query | dodam ms | duckdb ms | ratio | gap ms |")
    print("|---|---:|---:|---:|---:|")
    for row in report["rows"]:
        print(
            f"| {row['query']} | "
            f"{format_ms(row['dodam_median_s'])} | "
            f"{format_ms(row['duckdb_median_s'])} | "
            f"{format_ratio(row['ratio'])} | "
            f"{format_ms(row['gap_s'])} |"
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


def format_ms(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value * 1000.0:.3f}"


def format_ratio(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.3f}x"


if __name__ == "__main__":
    raise SystemExit(main())
