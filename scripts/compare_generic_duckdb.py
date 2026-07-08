#!/usr/bin/env python3
import argparse
import json
import statistics
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GenericQuery:
    name: str
    dodam_sql: str
    duckdb_sql: str


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare Dodam and DuckDB on non-TPC-H generic SQL workloads."
    )
    parser.add_argument("--work-dir", default="/tmp/dodam-generic-bench")
    parser.add_argument("--dodam", default="target/release/dodam")
    parser.add_argument("--duckdb", default="duckdb")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16 * 1024)
    parser.add_argument("--json-out", default="")
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    data_dir = work_dir / "data"
    output_dir = work_dir / "out"
    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_fixture(args.duckdb, data_dir, args.timeout)

    queries = generic_queries(data_dir)
    report = []
    for query in queries:
        dodam_samples = run_dodam(args, query, output_dir / "dodam")
        duckdb_samples = run_duckdb(args, query, output_dir / "duckdb")
        dodam_median = median_after_warmup(dodam_samples, args.warmup)
        duckdb_median = median_after_warmup(duckdb_samples, args.warmup)
        report.append(
            {
                "query": query.name,
                "dodam_median_s": dodam_median,
                "duckdb_median_s": duckdb_median,
                "ratio": dodam_median / duckdb_median if duckdb_median > 0 else None,
                "gap_s": dodam_median - duckdb_median,
                "dodam_samples_s": dodam_samples,
                "duckdb_samples_s": duckdb_samples,
            }
        )

    print_report(report)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2) + "\n")
    return 0


def ensure_fixture(duckdb: str, data_dir: Path, timeout: float) -> None:
    facts = data_dir / "facts.parquet"
    dim = data_dir / "dim.parquet"
    nested = data_dir / "nested.parquet"
    if facts.exists() and dim.exists() and nested.exists():
        return
    sql = f"""
COPY (
  SELECT
    i::INTEGER AS id,
    CASE WHEN i % 17 = 0 THEN NULL ELSE (i % 1000)::INTEGER END AS key,
    (i % 10)::INTEGER AS bucket,
    (i * 3 % 100000)::BIGINT AS value,
    CASE WHEN i % 13 = 0 THEN NULL ELSE 'label-' || (i % 64)::VARCHAR END AS label,
    CAST((i % 10000) / 100.0 AS DECIMAL(18,2)) AS amount,
    DATE '2024-01-01' + ((i % 31)::INTEGER) AS event_date
  FROM range(0, 600000) AS t(i)
) TO '{facts}' (FORMAT PARQUET, COMPRESSION ZSTD);
COPY (
  SELECT
    i::INTEGER AS key,
    'name-' || i::VARCHAR AS name,
    CASE WHEN i % 5 = 0 THEN NULL ELSE 'class-' || (i % 20)::VARCHAR END AS class
  FROM range(0, 1000) AS t(i)
) TO '{dim}' (FORMAT PARQUET, COMPRESSION ZSTD);
COPY (
  SELECT
    i::INTEGER AS id,
    [i::INTEGER, (i + 1)::INTEGER, NULL] AS tags,
    struct_pack(rank := (i % 7)::INTEGER, label := 'n-' || (i % 16)::VARCHAR) AS attrs
  FROM range(0, 20000) AS t(i)
) TO '{nested}' (FORMAT PARQUET, COMPRESSION ZSTD);
"""
    completed = subprocess.run(
        [duckdb],
        input=sql,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    if completed.returncode != 0:
        raise SystemExit(completed.stderr or completed.stdout)


def generic_queries(data_dir: Path) -> list[GenericQuery]:
    facts = data_dir / "facts.parquet"
    dim = data_dir / "dim.parquet"
    nested = data_dir / "nested.parquet"
    return [
        GenericQuery(
            "filter_group_decimal_date",
            f"SELECT bucket, count(*), sum(value), min(amount), max(event_date) FROM '{facts}' WHERE amount >= '10.00' AND event_date <= '2024-01-20' GROUP BY bucket ORDER BY bucket",
            f"SELECT bucket, count(*), sum(value), min(amount), max(event_date) FROM read_parquet('{facts}') WHERE amount >= '10.00' AND event_date <= '2024-01-20' GROUP BY bucket ORDER BY bucket",
        ),
        GenericQuery(
            "join_expression_projection",
            f"SELECT f.bucket, coalesce(d.class, 'missing') AS class_name, count(*), sum(f.value) FROM '{facts}' f JOIN '{dim}' d ON f.key = d.key WHERE f.bucket IN (1, 3, 7) GROUP BY f.bucket, coalesce(d.class, 'missing') ORDER BY f.bucket, class_name",
            f"SELECT f.bucket, coalesce(d.class, 'missing') AS class_name, count(*), sum(f.value) FROM read_parquet('{facts}') f JOIN read_parquet('{dim}') d ON f.key = d.key WHERE f.bucket IN (1, 3, 7) GROUP BY f.bucket, coalesce(d.class, 'missing') ORDER BY f.bucket, class_name",
        ),
        GenericQuery(
            "nested_projection_filter",
            f"SELECT id, attrs.rank, tags[1] FROM '{nested}' WHERE array_length(tags) = 3 AND attrs.rank IN (1, 3, 5) ORDER BY id LIMIT 2000",
            f"SELECT id, attrs.rank, tags[1] FROM read_parquet('{nested}') WHERE array_length(tags) = 3 AND attrs.rank IN (1, 3, 5) ORDER BY id LIMIT 2000",
        ),
    ]


def run_dodam(args, query: GenericQuery, output_dir: Path) -> list[float]:
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = []
    for repeat in range(args.repeats):
        output = output_dir / f"{query.name}-{repeat}.parquet"
        sql = f"COPY ({query.dodam_sql}) TO '{output}' (FORMAT PARQUET)"
        started = time.perf_counter()
        completed = subprocess.run(
            [args.dodam, "query", sql, "--batch-size", str(args.batch_size)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=args.timeout,
        )
        samples.append(time.perf_counter() - started)
        if completed.returncode != 0:
            raise SystemExit(completed.stderr or completed.stdout)
    return samples


def run_duckdb(args, query: GenericQuery, output_dir: Path) -> list[float]:
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = []
    for repeat in range(args.repeats):
        output = output_dir / f"{query.name}-{repeat}.parquet"
        sql = f"COPY ({query.duckdb_sql}) TO '{output}' (FORMAT PARQUET)"
        started = time.perf_counter()
        completed = subprocess.run(
            [args.duckdb, "-c", sql],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=args.timeout,
        )
        samples.append(time.perf_counter() - started)
        if completed.returncode != 0:
            raise SystemExit(completed.stderr or completed.stdout)
    return samples


def median_after_warmup(samples: list[float], warmup: int) -> float:
    return statistics.median(samples[warmup:] if warmup > 0 else samples)


def print_report(report: list[dict]) -> None:
    total_dodam = sum(row["dodam_median_s"] for row in report)
    total_duckdb = sum(row["duckdb_median_s"] for row in report)
    print(f"median-sum: Dodam={total_dodam:.6f}s DuckDB={total_duckdb:.6f}s ratio={total_dodam / total_duckdb:.3f}x")
    print("| query | dodam ms | duckdb ms | ratio | gap ms |")
    print("|---|---:|---:|---:|---:|")
    for row in report:
        print(
            f"| {row['query']} | {row['dodam_median_s'] * 1000:.3f} | "
            f"{row['duckdb_median_s'] * 1000:.3f} | {row['ratio']:.3f}x | "
            f"{row['gap_s'] * 1000:.3f} |"
        )


if __name__ == "__main__":
    raise SystemExit(main())
