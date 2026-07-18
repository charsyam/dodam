#!/usr/bin/env python3
import argparse
import json
import os
import re
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
    parser.add_argument(
        "--fixture-scale",
        type=int,
        default=1,
        help="Scale generic fixture row counts. 1 keeps the historical 600k facts rows.",
    )
    parser.add_argument(
        "--dodam-mode",
        choices=["query", "query-file"],
        default="query",
        help="Use one Dodam process per sample, or one query-file process per query.",
    )
    parser.add_argument(
        "--duckdb-mode",
        choices=["query", "query-file"],
        default="query",
        help="Use one DuckDB process per sample, or one .timer-enabled SQL file process per query.",
    )
    parser.add_argument("--json-out", default="")
    parser.add_argument(
        "--dodam-parquet-options",
        default="",
        help="Extra Dodam COPY FORMAT PARQUET options, for example: COMPRESSION uncompressed, ROW_GROUP_SIZE 65536",
    )
    parser.add_argument(
        "--duckdb-parquet-options",
        default="",
        help="Extra DuckDB COPY FORMAT PARQUET options.",
    )
    parser.add_argument(
        "--only",
        default="",
        help="Comma-separated query names to run. Matches are case-insensitive.",
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--show-stderr",
        action="store_true",
        help="Print Dodam/DuckDB stderr for profiling runs.",
    )
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    if args.fixture_scale <= 0:
        raise SystemExit("--fixture-scale must be positive")
    data_dir = work_dir / (
        "data" if args.fixture_scale == 1 else f"data_sf{args.fixture_scale}"
    )
    output_dir = work_dir / "out"
    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_fixture(args.duckdb, data_dir, args.timeout, args.fixture_scale)

    queries = filter_queries(generic_queries(data_dir), args.only)
    if not queries:
        raise SystemExit(f"no generic queries matched --only {args.only!r}")
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


def ensure_fixture(duckdb: str, data_dir: Path, timeout: float, scale: int) -> None:
    facts = data_dir / "facts.parquet"
    facts_scrambled = data_dir / "facts_scrambled.parquet"
    product = data_dir / "product_v2.parquet"
    dim = data_dir / "dim.parquet"
    nested = data_dir / "nested.parquet"
    if (
        facts.exists()
        and facts_scrambled.exists()
        and product.exists()
        and dim.exists()
        and nested.exists()
    ):
        return
    for path in (facts, facts_scrambled, product, dim, nested):
        if path.exists():
            path.unlink()
    facts_rows = 600_000 * scale
    nested_rows = 20_000 * scale
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
  FROM range(0, {facts_rows}) AS t(i)
) TO '{facts}' (FORMAT PARQUET, COMPRESSION ZSTD);
COPY (
  SELECT
    ((i * 15485863) % {facts_rows})::INTEGER AS id,
    CASE WHEN i % 17 = 0 THEN NULL ELSE (i % 1000)::INTEGER END AS key,
    (i % 10)::INTEGER AS bucket,
    (i * 3 % 100000)::BIGINT AS value,
    CASE WHEN i % 13 = 0 THEN NULL ELSE 'label-' || (i % 64)::VARCHAR END AS label,
    CAST((i % 10000) / 100.0 AS DECIMAL(18,2)) AS amount,
    DATE '2024-01-01' + ((i % 31)::INTEGER) AS event_date
  FROM range(0, {facts_rows}) AS t(i)
) TO '{facts_scrambled}' (FORMAT PARQUET, COMPRESSION ZSTD);
COPY (
  SELECT
    i::INTEGER AS id,
    CAST((i % 10000) / 100.0 AS DECIMAL(18,2)) AS amount,
    CAST((i % 100) / 100.0 AS DECIMAL(18,2)) AS rate,
    CAST((i % 8) / 100.0 AS DECIMAL(18,2)) AS tax,
    DATE '2024-01-01' + ((i % 31)::INTEGER) AS event_date
  FROM range(0, {facts_rows}) AS t(i)
) TO '{product}' (FORMAT PARQUET, COMPRESSION ZSTD);
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
  FROM range(0, {nested_rows}) AS t(i)
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
    facts_scrambled = data_dir / "facts_scrambled.parquet"
    product = data_dir / "product_v2.parquet"
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
        GenericQuery(
            "coalesce_type_projection",
            f"SELECT id, coalesce(key, -1) AS key_or_missing, coalesce(label, 'missing') AS label_or_missing, coalesce(amount, '0.00') AS amount_or_zero, coalesce(event_date, DATE '2024-01-01') AS event_or_default FROM '{facts}' WHERE id < 20000 ORDER BY id",
            f"SELECT id, coalesce(key, -1) AS key_or_missing, coalesce(label, 'missing') AS label_or_missing, coalesce(amount, '0.00') AS amount_or_zero, coalesce(event_date, DATE '2024-01-01') AS event_or_default FROM read_parquet('{facts}') WHERE id < 20000 ORDER BY id",
        ),
        GenericQuery(
            "three_key_expression_group",
            f"SELECT bucket, event_date, coalesce(label, 'missing') AS label_or_missing, count(*), sum(value) FROM '{facts}' WHERE amount < '50.00' GROUP BY bucket, event_date, coalesce(label, 'missing') ORDER BY bucket, event_date, label_or_missing LIMIT 2000",
            f"SELECT bucket, event_date, coalesce(label, 'missing') AS label_or_missing, count(*), sum(value) FROM read_parquet('{facts}') WHERE amount < '50.00' GROUP BY bucket, event_date, coalesce(label, 'missing') ORDER BY bucket, event_date, label_or_missing LIMIT 2000",
        ),
        GenericQuery(
            "nullable_key_group",
            f"SELECT key, count(*), sum(value) FROM '{facts}' GROUP BY key ORDER BY key NULLS FIRST",
            f"SELECT key, count(*), sum(value) FROM read_parquet('{facts}') GROUP BY key ORDER BY key NULLS FIRST",
        ),
        GenericQuery(
            "non_null_bucket_count_sum",
            f"SELECT bucket, count(*), sum(value) FROM '{facts}' GROUP BY bucket ORDER BY bucket",
            f"SELECT bucket, count(*), sum(value) FROM read_parquet('{facts}') GROUP BY bucket ORDER BY bucket",
        ),
        GenericQuery(
            "review_q1_two_key_group",
            f"SELECT bucket, event_date, count(*), sum(value), min(amount), max(amount) FROM '{facts}' WHERE event_date <= '2024-01-20' GROUP BY bucket, event_date ORDER BY bucket, event_date",
            f"SELECT bucket, event_date, count(*), sum(value), min(amount), max(amount) FROM read_parquet('{facts}') WHERE event_date <= '2024-01-20' GROUP BY bucket, event_date ORDER BY bucket, event_date",
        ),
        GenericQuery(
            "review_q1_one_key_group",
            f"SELECT bucket, count(*), sum(value), min(amount), max(amount) FROM '{facts}' WHERE event_date <= '2024-01-20' GROUP BY bucket ORDER BY bucket",
            f"SELECT bucket, count(*), sum(value), min(amount), max(amount) FROM read_parquet('{facts}') WHERE event_date <= '2024-01-20' GROUP BY bucket ORDER BY bucket",
        ),
        GenericQuery(
            "review_q1_alt_string_group",
            f"SELECT label, count(*), sum(value), min(amount), max(amount) FROM '{facts}' WHERE event_date <= '2024-01-20' GROUP BY label ORDER BY label NULLS FIRST",
            f"SELECT label, count(*), sum(value), min(amount), max(amount) FROM read_parquet('{facts}') WHERE event_date <= '2024-01-20' GROUP BY label ORDER BY label NULLS FIRST",
        ),
        GenericQuery(
            "review_q6_expression_sum",
            f"SELECT sum(value * bucket) FROM '{facts}' WHERE amount >= '10.00' AND amount < '90.00' AND event_date < '2024-01-20'",
            f"SELECT sum(value * bucket) FROM read_parquet('{facts}') WHERE amount >= '10.00' AND amount < '90.00' AND event_date < '2024-01-20'",
        ),
        GenericQuery(
            "standard_discounted_product_sum",
            f"SELECT sum(amount * (1 - rate)) FROM '{product}' WHERE event_date < '2024-01-20' AND amount >= '10.00' AND amount < '20.00'",
            f"SELECT sum(amount * (1 - rate)) FROM read_parquet('{product}') WHERE event_date < '2024-01-20' AND amount >= '10.00' AND amount < '20.00'",
        ),
        GenericQuery(
            "standard_three_factor_product_sum",
            f"SELECT sum(amount * (1 - rate) * (1 + tax)) FROM '{product}' WHERE event_date < '2024-01-20' AND amount >= '10.00' AND amount < '20.00'",
            f"SELECT sum(amount * (1 - rate) * (1 + tax)) FROM read_parquet('{product}') WHERE event_date < '2024-01-20' AND amount >= '10.00' AND amount < '20.00'",
        ),
        GenericQuery(
            "review_high_cardinality_group",
            f"SELECT id, count(*), sum(value) FROM '{facts}' GROUP BY id ORDER BY id",
            f"SELECT id, count(*), sum(value) FROM read_parquet('{facts}') GROUP BY id ORDER BY id",
        ),
        GenericQuery(
            "review_count_distinct",
            f"SELECT count(DISTINCT id) FROM '{facts}'",
            f"SELECT count(DISTINCT id) FROM read_parquet('{facts}')",
        ),
        GenericQuery(
            "review_like_group",
            f"SELECT label, count(*), sum(value) FROM '{facts}' WHERE label LIKE 'label-2%' GROUP BY label ORDER BY label",
            f"SELECT label, count(*), sum(value) FROM read_parquet('{facts}') WHERE label LIKE 'label-2%' GROUP BY label ORDER BY label",
        ),
        GenericQuery(
            "review_filter_order_limit",
            f"SELECT id, key, bucket, value FROM '{facts}' WHERE amount >= '10.00' AND event_date < '2024-01-20' ORDER BY value DESC LIMIT 5000",
            f"SELECT id, key, bucket, value FROM read_parquet('{facts}') WHERE amount >= '10.00' AND event_date < '2024-01-20' ORDER BY value DESC LIMIT 5000",
        ),
        GenericQuery(
            "optimizer_three_way_comma_join",
            f"SELECT d.class, d2.name, count(*), sum(f.value) FROM '{facts}' f, '{dim}' d, '{dim}' d2 WHERE f.key = d.key AND f.bucket = d2.key GROUP BY d.class, d2.name ORDER BY d.class NULLS FIRST, d2.name",
            f"SELECT d.class, d2.name, count(*), sum(f.value) FROM read_parquet('{facts}') f, read_parquet('{dim}') d, read_parquet('{dim}') d2 WHERE f.key = d.key AND f.bucket = d2.key GROUP BY d.class, d2.name ORDER BY d.class NULLS FIRST, d2.name",
        ),
        GenericQuery(
            "optimizer_selective_three_way_comma_join",
            f"SELECT d.class, d2.name, count(*), sum(f.value) FROM '{facts}' f, '{dim}' d, '{dim}' d2 WHERE f.key = d.key AND f.bucket = d2.key AND f.amount < '20.00' AND d.class IS NOT NULL GROUP BY d.class, d2.name ORDER BY d.class, d2.name",
            f"SELECT d.class, d2.name, count(*), sum(f.value) FROM read_parquet('{facts}') f, read_parquet('{dim}') d, read_parquet('{dim}') d2 WHERE f.key = d.key AND f.bucket = d2.key AND f.amount < '20.00' AND d.class IS NOT NULL GROUP BY d.class, d2.name ORDER BY d.class, d2.name",
        ),
        GenericQuery(
            "optimizer_four_way_comma_join",
            f"SELECT d.class, d2.name, d3.class AS bucket_class, count(*), sum(f.value) FROM '{facts}' f, '{dim}' d, '{dim}' d2, '{dim}' d3 WHERE f.key = d.key AND f.bucket = d2.key AND f.bucket = d3.key GROUP BY d.class, d2.name, d3.class ORDER BY d.class NULLS FIRST, d2.name, bucket_class NULLS FIRST",
            f"SELECT d.class, d2.name, d3.class AS bucket_class, count(*), sum(f.value) FROM read_parquet('{facts}') f, read_parquet('{dim}') d, read_parquet('{dim}') d2, read_parquet('{dim}') d3 WHERE f.key = d.key AND f.bucket = d2.key AND f.bucket = d3.key GROUP BY d.class, d2.name, d3.class ORDER BY d.class NULLS FIRST, d2.name, bucket_class NULLS FIRST",
        ),
        GenericQuery(
            "optimizer_selective_four_way_comma_join",
            f"SELECT d.class, d2.name, d3.class AS bucket_class, count(*), sum(f.value) FROM '{facts}' f, '{dim}' d, '{dim}' d2, '{dim}' d3 WHERE f.key = d.key AND f.bucket = d2.key AND f.bucket = d3.key AND f.amount < '20.00' AND d.class IS NOT NULL GROUP BY d.class, d2.name, d3.class ORDER BY d.class, d2.name, bucket_class NULLS FIRST",
            f"SELECT d.class, d2.name, d3.class AS bucket_class, count(*), sum(f.value) FROM read_parquet('{facts}') f, read_parquet('{dim}') d, read_parquet('{dim}') d2, read_parquet('{dim}') d3 WHERE f.key = d.key AND f.bucket = d2.key AND f.bucket = d3.key AND f.amount < '20.00' AND d.class IS NOT NULL GROUP BY d.class, d2.name, d3.class ORDER BY d.class, d2.name, bucket_class NULLS FIRST",
        ),
        GenericQuery(
            "string_low_cardinality_group",
            f"SELECT label, count(*), sum(value) FROM '{facts}' WHERE id < 200000 GROUP BY label ORDER BY label NULLS FIRST",
            f"SELECT label, count(*), sum(value) FROM read_parquet('{facts}') WHERE id < 200000 GROUP BY label ORDER BY label NULLS FIRST",
        ),
        GenericQuery(
            "three_key_expression_group_no_order",
            f"SELECT bucket, event_date, coalesce(label, 'missing') AS label_or_missing, count(*), sum(value) FROM '{facts}' WHERE amount < '50.00' GROUP BY bucket, event_date, coalesce(label, 'missing')",
            f"SELECT bucket, event_date, coalesce(label, 'missing') AS label_or_missing, count(*), sum(value) FROM read_parquet('{facts}') WHERE amount < '50.00' GROUP BY bucket, event_date, coalesce(label, 'missing')",
        ),
        GenericQuery(
            "decimal_date_group_no_order",
            f"SELECT bucket, count(*), sum(value), min(amount), max(event_date) FROM '{facts}' WHERE amount >= '10.00' AND event_date <= '2024-01-20' GROUP BY bucket",
            f"SELECT bucket, count(*), sum(value), min(amount), max(event_date) FROM read_parquet('{facts}') WHERE amount >= '10.00' AND event_date <= '2024-01-20' GROUP BY bucket",
        ),
        GenericQuery(
            "low_selectivity_decimal_date_group",
            f"SELECT bucket, count(*), sum(value), min(amount), max(event_date) FROM '{facts}' WHERE amount < '1.00' GROUP BY bucket",
            f"SELECT bucket, count(*), sum(value), min(amount), max(event_date) FROM read_parquet('{facts}') WHERE amount < '1.00' GROUP BY bucket",
        ),
        GenericQuery(
            "low_selectivity_three_key_expression_group_no_order",
            f"SELECT bucket, event_date, coalesce(label, 'missing') AS label_or_missing, count(*), sum(value) FROM '{facts}' WHERE amount < '1.00' GROUP BY bucket, event_date, coalesce(label, 'missing')",
            f"SELECT bucket, event_date, coalesce(label, 'missing') AS label_or_missing, count(*), sum(value) FROM read_parquet('{facts}') WHERE amount < '1.00' GROUP BY bucket, event_date, coalesce(label, 'missing')",
        ),
        GenericQuery(
            "standard_between_like_case",
            f"SELECT id, key, bucket, label FROM '{facts}' WHERE (key BETWEEN 10 AND 40 OR label LIKE 'label-2%') AND id NOT BETWEEN 1000 AND 2000 ORDER BY id LIMIT 5000",
            f"SELECT id, key, bucket, label FROM read_parquet('{facts}') WHERE (key BETWEEN 10 AND 40 OR label LIKE 'label-2%') AND id NOT BETWEEN 1000 AND 2000 ORDER BY id LIMIT 5000",
        ),
        GenericQuery(
            "standard_string_functions",
            f"SELECT id, trim(concat(' ', coalesce(label, 'missing'), ' ')) AS label_trimmed, replace(coalesce(label, 'missing'), 'label', 'tag') AS label_replaced, concat(coalesce(label, 'missing'), '-', CAST(bucket AS VARCHAR)) AS label_bucket FROM '{facts}' WHERE replace(coalesce(label, 'missing'), 'label', 'tag') LIKE 'tag-2%' ORDER BY id LIMIT 5000",
            f"SELECT id, trim(concat(' ', coalesce(label, 'missing'), ' ')) AS label_trimmed, replace(coalesce(label, 'missing'), 'label', 'tag') AS label_replaced, concat(coalesce(label, 'missing'), '-', CAST(bucket AS VARCHAR)) AS label_bucket FROM read_parquet('{facts}') WHERE replace(coalesce(label, 'missing'), 'label', 'tag') LIKE 'tag-2%' ORDER BY id LIMIT 5000",
        ),
        GenericQuery(
            "standard_ilike_filter",
            f"SELECT id, key, bucket, label FROM '{facts}' WHERE label ILIKE 'LABEL-2%' OR coalesce(label, 'missing') ILIKE 'MISSING' ORDER BY id LIMIT 5000",
            f"SELECT id, key, bucket, label FROM read_parquet('{facts}') WHERE label ILIKE 'LABEL-2%' OR coalesce(label, 'missing') ILIKE 'MISSING' ORDER BY id LIMIT 5000",
        ),
        GenericQuery(
            "standard_numeric_functions",
            f"SELECT id, abs(value - 500) AS value_distance, floor(amount / 10.0) AS amount_floor, ceil(amount / 10.0) AS amount_ceil, round(amount / 7.0) AS amount_round FROM '{facts}' WHERE abs(value - 500) <= 20 OR floor(amount / 10.0) = 3 ORDER BY id LIMIT 5000",
            f"SELECT id, abs(value - 500) AS value_distance, floor(amount / 10.0) AS amount_floor, ceil(amount / 10.0) AS amount_ceil, round(amount / 7.0) AS amount_round FROM read_parquet('{facts}') WHERE abs(value - 500) <= 20 OR floor(amount / 10.0) = 3 ORDER BY id LIMIT 5000",
        ),
        GenericQuery(
            "standard_aggregate_filter",
            f"SELECT bucket, count(*) FILTER (WHERE amount >= '50.00') AS high_amount_rows, count(value) FILTER (WHERE key IS NOT NULL) AS keyed_values, sum(value) FILTER (WHERE label LIKE 'label-2%') AS label_two_sum, avg(value) FILTER (WHERE amount < '25.00') AS low_amount_avg FROM '{facts}' GROUP BY bucket ORDER BY bucket",
            f"SELECT bucket, count(*) FILTER (WHERE amount >= '50.00') AS high_amount_rows, count(value) FILTER (WHERE key IS NOT NULL) AS keyed_values, sum(value) FILTER (WHERE label LIKE 'label-2%') AS label_two_sum, avg(value) FILTER (WHERE amount < '25.00') AS low_amount_avg FROM read_parquet('{facts}') GROUP BY bucket ORDER BY bucket",
        ),
        GenericQuery(
            "standard_aggregate_filter_sparse",
            f"SELECT bucket, count(*) FILTER (WHERE id < 1000) AS tiny_rows, sum(value) FILTER (WHERE label LIKE 'label-999%') AS tiny_label_sum, avg(value) FILTER (WHERE amount < '1.00') AS tiny_amount_avg FROM '{facts}' GROUP BY bucket ORDER BY bucket",
            f"SELECT bucket, count(*) FILTER (WHERE id < 1000) AS tiny_rows, sum(value) FILTER (WHERE label LIKE 'label-999%') AS tiny_label_sum, avg(value) FILTER (WHERE amount < '1.00') AS tiny_amount_avg FROM read_parquet('{facts}') GROUP BY bucket ORDER BY bucket",
        ),
        GenericQuery(
            "standard_aggregate_filter_dense",
            f"SELECT bucket, count(*) FILTER (WHERE id >= 0) AS all_rows, sum(value) FILTER (WHERE amount >= '0.00') AS all_amount_sum, avg(value) FILTER (WHERE label IS NOT NULL) AS labeled_avg FROM '{facts}' GROUP BY bucket ORDER BY bucket",
            f"SELECT bucket, count(*) FILTER (WHERE id >= 0) AS all_rows, sum(value) FILTER (WHERE amount >= '0.00') AS all_amount_sum, avg(value) FILTER (WHERE label IS NOT NULL) AS labeled_avg FROM read_parquet('{facts}') GROUP BY bucket ORDER BY bucket",
        ),
        GenericQuery(
            "standard_group_by_all_expression",
            f"SELECT lower(coalesce(label, 'missing')) AS label_class, bucket + 1 AS bucket_plus_one, count(*) AS rows, sum(value) AS total_value FROM '{facts}' GROUP BY ALL ORDER BY label_class, bucket_plus_one LIMIT 5000",
            f"SELECT lower(coalesce(label, 'missing')) AS label_class, bucket + 1 AS bucket_plus_one, count(*) AS rows, sum(value) AS total_value FROM read_parquet('{facts}') GROUP BY ALL ORDER BY label_class, bucket_plus_one LIMIT 5000",
        ),
        GenericQuery(
            "standard_having_simple_case",
            f"SELECT CASE bucket WHEN 1 THEN 'one' WHEN 2 THEN 'two' ELSE 'other' END AS bucket_class, count(*) AS rows, sum(value) AS total_value FROM '{facts}' GROUP BY CASE bucket WHEN 1 THEN 'one' WHEN 2 THEN 'two' ELSE 'other' END HAVING rows > 0 ORDER BY bucket_class",
            f"SELECT CASE bucket WHEN 1 THEN 'one' WHEN 2 THEN 'two' ELSE 'other' END AS bucket_class, count(*) AS rows, sum(value) AS total_value FROM read_parquet('{facts}') GROUP BY CASE bucket WHEN 1 THEN 'one' WHEN 2 THEN 'two' ELSE 'other' END HAVING rows > 0 ORDER BY bucket_class",
        ),
        GenericQuery(
            "standard_nested_in_subquery",
            f"SELECT id, key, label FROM '{facts}' WHERE key IN (SELECT key FROM (SELECT DISTINCT key FROM '{facts}' WHERE key IS NOT NULL) AS keys) ORDER BY id LIMIT 100000",
            f"SELECT id, key, label FROM read_parquet('{facts}') WHERE key IN (SELECT key FROM (SELECT DISTINCT key FROM read_parquet('{facts}') WHERE key IS NOT NULL) AS keys) ORDER BY id LIMIT 100000",
        ),
        GenericQuery(
            "standard_not_in_subquery_null_safe",
            f"SELECT id, key, label FROM '{facts}' WHERE key NOT IN (SELECT key FROM '{facts}' WHERE key IS NOT NULL AND key < 3) ORDER BY id LIMIT 100000",
            f"SELECT id, key, label FROM read_parquet('{facts}') WHERE key NOT IN (SELECT key FROM read_parquet('{facts}') WHERE key IS NOT NULL AND key < 3) ORDER BY id LIMIT 100000",
        ),
        GenericQuery(
            "standard_tuple_in_subquery",
            f"SELECT id, key, label FROM '{facts}' WHERE (id, key) IN (SELECT id, key FROM '{facts}' WHERE key >= 2 AND key IS NOT NULL) ORDER BY id LIMIT 100000",
            f"SELECT id, key, label FROM read_parquet('{facts}') WHERE (id, key) IN (SELECT id, key FROM read_parquet('{facts}') WHERE key >= 2 AND key IS NOT NULL) ORDER BY id LIMIT 100000",
        ),
        GenericQuery(
            "standard_tuple_in_direct_filter_equiv",
            f"SELECT id, key, label FROM '{facts}' WHERE key >= 2 AND key IS NOT NULL ORDER BY id LIMIT 100000",
            f"SELECT id, key, label FROM read_parquet('{facts}') WHERE key >= 2 AND key IS NOT NULL ORDER BY id LIMIT 100000",
        ),
        GenericQuery(
            "standard_tuple_in_non_key_predicate",
            f"SELECT id, key, label FROM '{facts}' WHERE (id, key) IN (SELECT id, key FROM '{facts}' WHERE label LIKE 'label-2%') ORDER BY id LIMIT 100000",
            f"SELECT id, key, label FROM read_parquet('{facts}') WHERE (id, key) IN (SELECT id, key FROM read_parquet('{facts}') WHERE label LIKE 'label-2%') ORDER BY id LIMIT 100000",
        ),
        GenericQuery(
            "standard_tuple_in_other_file",
            f"SELECT id, key, label FROM '{facts}' WHERE (id, key) IN (SELECT id, key FROM '{facts_scrambled}' WHERE key >= 2 AND key IS NOT NULL) ORDER BY id LIMIT 100000",
            f"SELECT id, key, label FROM read_parquet('{facts}') WHERE (id, key) IN (SELECT id, key FROM read_parquet('{facts_scrambled}') WHERE key >= 2 AND key IS NOT NULL) ORDER BY id LIMIT 100000",
        ),
        GenericQuery(
            "standard_tuple_not_in_null_safe",
            f"SELECT id, key, label FROM '{facts}' WHERE key IS NOT NULL AND (id, key) NOT IN (SELECT id, key FROM '{facts_scrambled}' WHERE key IS NOT NULL AND key < 3) ORDER BY id LIMIT 100000",
            f"SELECT id, key, label FROM read_parquet('{facts}') WHERE key IS NOT NULL AND (id, key) NOT IN (SELECT id, key FROM read_parquet('{facts_scrambled}') WHERE key IS NOT NULL AND key < 3) ORDER BY id LIMIT 100000",
        ),
        GenericQuery(
            "standard_tuple_in_mixed_key_other_file",
            f"SELECT id, key, label FROM '{facts}' WHERE label IS NOT NULL AND (id, label) IN (SELECT id, label FROM '{facts_scrambled}' WHERE label LIKE 'label-2%') ORDER BY id LIMIT 100000",
            f"SELECT id, key, label FROM read_parquet('{facts}') WHERE label IS NOT NULL AND (id, label) IN (SELECT id, label FROM read_parquet('{facts_scrambled}') WHERE label LIKE 'label-2%') ORDER BY id LIMIT 100000",
        ),
        GenericQuery(
            "standard_direct_cte_aggregate",
            f"WITH grouped AS (SELECT bucket, count(*) AS row_count, sum(value) AS total_value FROM '{facts}' GROUP BY bucket) SELECT bucket, row_count, total_value FROM grouped WHERE row_count >= 1 ORDER BY bucket",
            f"WITH grouped AS (SELECT bucket, count(*) AS row_count, sum(value) AS total_value FROM read_parquet('{facts}') GROUP BY bucket) SELECT bucket, row_count, total_value FROM grouped WHERE row_count >= 1 ORDER BY bucket",
        ),
        GenericQuery(
            "standard_join_simple_case_residual",
            f"SELECT f.id, d.name FROM '{facts}' f JOIN '{dim}' d ON f.key = d.key WHERE CASE d.name WHEN 'two' THEN 1 ELSE 0 END = 1 ORDER BY f.id LIMIT 100000",
            f"SELECT f.id, d.name FROM read_parquet('{facts}') f JOIN read_parquet('{dim}') d ON f.key = d.key WHERE CASE d.name WHEN 'two' THEN 1 ELSE 0 END = 1 ORDER BY f.id LIMIT 100000",
        ),
        GenericQuery(
            "window_rank_partition_order",
            f"SELECT id, bucket, value, row_number() OVER (PARTITION BY bucket ORDER BY value) AS rn, rank() OVER (PARTITION BY bucket ORDER BY value) AS rnk, dense_rank() OVER (PARTITION BY bucket ORDER BY value) AS drnk FROM '{facts}' ORDER BY bucket, value, id LIMIT 100000",
            f"SELECT id, bucket, value, row_number() OVER (PARTITION BY bucket ORDER BY value) AS rn, rank() OVER (PARTITION BY bucket ORDER BY value) AS rnk, dense_rank() OVER (PARTITION BY bucket ORDER BY value) AS drnk FROM read_parquet('{facts}') ORDER BY bucket, value, id LIMIT 100000",
        ),
        GenericQuery(
            "window_lag_lead_partition_order",
            f"SELECT id, bucket, lag(value) OVER (PARTITION BY bucket ORDER BY id) AS prev_value, lead(label, 2) OVER (PARTITION BY bucket ORDER BY id) AS next_label FROM '{facts}' ORDER BY bucket, id LIMIT 100000",
            f"SELECT id, bucket, lag(value) OVER (PARTITION BY bucket ORDER BY id) AS prev_value, lead(label, 2) OVER (PARTITION BY bucket ORDER BY id) AS next_label FROM read_parquet('{facts}') ORDER BY bucket, id LIMIT 100000",
        ),
        GenericQuery(
            "window_aggregate_partition",
            f"SELECT id, key, count(value) OVER (PARTITION BY key) AS value_count, sum(CAST(value AS DOUBLE)) OVER (PARTITION BY key) AS value_sum, avg(value) OVER (PARTITION BY key) AS value_avg FROM '{facts}' ORDER BY key NULLS FIRST, id LIMIT 100000",
            f"SELECT id, key, count(value) OVER (PARTITION BY key) AS value_count, sum(CAST(value AS DOUBLE)) OVER (PARTITION BY key) AS value_sum, avg(value) OVER (PARTITION BY key) AS value_avg FROM read_parquet('{facts}') ORDER BY key NULLS FIRST, id LIMIT 100000",
        ),
        GenericQuery(
            "window_running_aggregate",
            f"SELECT id, key, count(*) OVER (PARTITION BY key ORDER BY id) AS running_count, sum(CAST(value AS DOUBLE)) OVER (PARTITION BY key ORDER BY id) AS running_sum FROM '{facts}' ORDER BY key NULLS FIRST, id LIMIT 100000",
            f"SELECT id, key, count(*) OVER (PARTITION BY key ORDER BY id) AS running_count, sum(CAST(value AS DOUBLE)) OVER (PARTITION BY key ORDER BY id) AS running_sum FROM read_parquet('{facts}') ORDER BY key NULLS FIRST, id LIMIT 100000",
        ),
        GenericQuery(
            "union_all_append",
            f"SELECT id, bucket, value FROM '{facts}' WHERE bucket = 1 UNION ALL SELECT id, bucket, value FROM '{facts}' WHERE bucket = 7",
            f"SELECT id, bucket, value FROM read_parquet('{facts}') WHERE bucket = 1 UNION ALL SELECT id, bucket, value FROM read_parquet('{facts}') WHERE bucket = 7",
        ),
        GenericQuery(
            "union_distinct_low_cardinality",
            f"SELECT bucket FROM '{facts}' WHERE bucket IN (1, 7) UNION SELECT bucket FROM '{facts}' WHERE bucket IN (7, 9) ORDER BY bucket",
            f"SELECT bucket FROM read_parquet('{facts}') WHERE bucket IN (1, 7) UNION SELECT bucket FROM read_parquet('{facts}') WHERE bucket IN (7, 9) ORDER BY bucket",
        ),
        GenericQuery(
            "union_distinct_rows",
            f"SELECT bucket, value FROM '{facts}' WHERE bucket IN (1, 7) UNION DISTINCT SELECT bucket, value FROM '{facts}' WHERE bucket IN (7, 9) ORDER BY bucket, value LIMIT 5000",
            f"SELECT bucket, value FROM read_parquet('{facts}') WHERE bucket IN (1, 7) UNION DISTINCT SELECT bucket, value FROM read_parquet('{facts}') WHERE bucket IN (7, 9) ORDER BY bucket, value LIMIT 5000",
        ),
        GenericQuery(
            "union_distinct_i64_rows",
            f"SELECT id FROM '{facts}' WHERE id IN (1, 2, 3, 4, 5, 6, 7, 8) UNION DISTINCT SELECT id FROM '{facts}' WHERE id IN (5, 6, 7, 8, 9, 10, 11, 12) ORDER BY id",
            f"SELECT id FROM read_parquet('{facts}') WHERE id IN (1, 2, 3, 4, 5, 6, 7, 8) UNION DISTINCT SELECT id FROM read_parquet('{facts}') WHERE id IN (5, 6, 7, 8, 9, 10, 11, 12) ORDER BY id",
        ),
        GenericQuery(
            "union_distinct_i64_i32_rows",
            f"SELECT id, bucket FROM '{facts}' WHERE id IN (1, 2, 3, 4, 5, 6, 7, 8) UNION DISTINCT SELECT id, bucket FROM '{facts}' WHERE id IN (5, 6, 7, 8, 9, 10, 11, 12) ORDER BY id, bucket",
            f"SELECT id, bucket FROM read_parquet('{facts}') WHERE id IN (1, 2, 3, 4, 5, 6, 7, 8) UNION DISTINCT SELECT id, bucket FROM read_parquet('{facts}') WHERE id IN (5, 6, 7, 8, 9, 10, 11, 12) ORDER BY id, bucket",
        ),
        GenericQuery(
            "intersect_distinct_rows",
            f"SELECT bucket, value FROM '{facts}' WHERE bucket IN (1, 7) INTERSECT SELECT bucket, value FROM '{facts}' WHERE bucket IN (7, 9) ORDER BY bucket, value LIMIT 5000",
            f"SELECT bucket, value FROM read_parquet('{facts}') WHERE bucket IN (1, 7) INTERSECT SELECT bucket, value FROM read_parquet('{facts}') WHERE bucket IN (7, 9) ORDER BY bucket, value LIMIT 5000",
        ),
        GenericQuery(
            "except_distinct_rows",
            f"SELECT bucket, value FROM '{facts}' WHERE bucket IN (1, 7) EXCEPT SELECT bucket, value FROM '{facts}' WHERE bucket IN (7, 9) ORDER BY bucket, value LIMIT 5000",
            f"SELECT bucket, value FROM read_parquet('{facts}') WHERE bucket IN (1, 7) EXCEPT SELECT bucket, value FROM read_parquet('{facts}') WHERE bucket IN (7, 9) ORDER BY bucket, value LIMIT 5000",
        ),
        GenericQuery(
            "except_distinct_simple_case_rows",
            f"SELECT CASE bucket WHEN 1 THEN 'one' ELSE 'other' END AS bucket_class FROM '{facts}' WHERE bucket IN (1, 7) EXCEPT DISTINCT SELECT CASE bucket WHEN 1 THEN 'one' ELSE 'other' END AS bucket_class FROM '{facts}' WHERE bucket IN (7, 9) ORDER BY bucket_class",
            f"SELECT CASE bucket WHEN 1 THEN 'one' ELSE 'other' END AS bucket_class FROM read_parquet('{facts}') WHERE bucket IN (1, 7) EXCEPT DISTINCT SELECT CASE bucket WHEN 1 THEN 'one' ELSE 'other' END AS bucket_class FROM read_parquet('{facts}') WHERE bucket IN (7, 9) ORDER BY bucket_class",
        ),
        GenericQuery(
            "intersect_all_rows",
            f"SELECT bucket FROM '{facts}' WHERE bucket IN (1, 7) INTERSECT ALL SELECT bucket FROM '{facts}' WHERE bucket IN (7, 9) ORDER BY bucket LIMIT 5000",
            f"SELECT bucket FROM read_parquet('{facts}') WHERE bucket IN (1, 7) INTERSECT ALL SELECT bucket FROM read_parquet('{facts}') WHERE bucket IN (7, 9) ORDER BY bucket LIMIT 5000",
        ),
        GenericQuery(
            "except_all_rows",
            f"SELECT bucket FROM '{facts}' WHERE bucket IN (1, 7) EXCEPT ALL SELECT bucket FROM '{facts}' WHERE bucket IN (7, 9) ORDER BY bucket LIMIT 5000",
            f"SELECT bucket FROM read_parquet('{facts}') WHERE bucket IN (1, 7) EXCEPT ALL SELECT bucket FROM read_parquet('{facts}') WHERE bucket IN (7, 9) ORDER BY bucket LIMIT 5000",
        ),
        GenericQuery(
            "intersect_all_i64_rows",
            f"SELECT id FROM '{facts}' WHERE id IN (1, 2, 3, 4, 5, 6, 7, 8) INTERSECT ALL SELECT id FROM '{facts}' WHERE id IN (5, 6, 7, 8, 9, 10, 11, 12) ORDER BY id",
            f"SELECT id FROM read_parquet('{facts}') WHERE id IN (1, 2, 3, 4, 5, 6, 7, 8) INTERSECT ALL SELECT id FROM read_parquet('{facts}') WHERE id IN (5, 6, 7, 8, 9, 10, 11, 12) ORDER BY id",
        ),
        GenericQuery(
            "except_all_i64_rows",
            f"SELECT id FROM '{facts}' WHERE id IN (1, 2, 3, 4, 5, 6, 7, 8) EXCEPT ALL SELECT id FROM '{facts}' WHERE id IN (5, 6, 7, 8, 9, 10, 11, 12) ORDER BY id",
            f"SELECT id FROM read_parquet('{facts}') WHERE id IN (1, 2, 3, 4, 5, 6, 7, 8) EXCEPT ALL SELECT id FROM read_parquet('{facts}') WHERE id IN (5, 6, 7, 8, 9, 10, 11, 12) ORDER BY id",
        ),
        GenericQuery(
            "select_distinct_low_cardinality",
            f"SELECT DISTINCT bucket FROM '{facts}' WHERE bucket IN (1, 7, 9) ORDER BY bucket",
            f"SELECT DISTINCT bucket FROM read_parquet('{facts}') WHERE bucket IN (1, 7, 9) ORDER BY bucket",
        ),
        GenericQuery(
            "select_distinct_rows",
            f"SELECT DISTINCT bucket, value FROM '{facts}' WHERE bucket IN (1, 7, 9) ORDER BY bucket, value LIMIT 5000",
            f"SELECT DISTINCT bucket, value FROM read_parquet('{facts}') WHERE bucket IN (1, 7, 9) ORDER BY bucket, value LIMIT 5000",
        ),
        GenericQuery(
            "union_all_order_limit",
            f"SELECT id, bucket, value FROM '{facts}' WHERE bucket = 1 UNION ALL SELECT id, bucket, value FROM '{facts}' WHERE bucket = 7 ORDER BY id DESC LIMIT 5000",
            f"SELECT id, bucket, value FROM read_parquet('{facts}') WHERE bucket = 1 UNION ALL SELECT id, bucket, value FROM read_parquet('{facts}') WHERE bucket = 7 ORDER BY id DESC LIMIT 5000",
        ),
        GenericQuery(
            "union_all_order_limit_full_read",
            f"SELECT id, bucket, value FROM '{facts}' WHERE bucket = 1 UNION ALL SELECT id, bucket, value FROM '{facts}' WHERE bucket = 7 ORDER BY id DESC LIMIT {120_000 * data_scale_from_path(data_dir)}",
            f"SELECT id, bucket, value FROM read_parquet('{facts}') WHERE bucket = 1 UNION ALL SELECT id, bucket, value FROM read_parquet('{facts}') WHERE bucket = 7 ORDER BY id DESC LIMIT {120_000 * data_scale_from_path(data_dir)}",
        ),
        GenericQuery(
            "union_all_order_limit_scrambled",
            f"SELECT id, bucket, value FROM '{facts_scrambled}' WHERE bucket = 1 UNION ALL SELECT id, bucket, value FROM '{facts_scrambled}' WHERE bucket = 7 ORDER BY id DESC LIMIT 5000",
            f"SELECT id, bucket, value FROM read_parquet('{facts_scrambled}') WHERE bucket = 1 UNION ALL SELECT id, bucket, value FROM read_parquet('{facts_scrambled}') WHERE bucket = 7 ORDER BY id DESC LIMIT 5000",
        ),
    ]


def data_scale_from_path(data_dir: Path) -> int:
    match = re.search(r"data_sf(\d+)$", data_dir.name)
    return int(match.group(1)) if match else 1


def filter_queries(queries: list[GenericQuery], only: str) -> list[GenericQuery]:
    filters = [item.strip().lower() for item in only.split(",") if item.strip()]
    if not filters:
        return queries
    return [query for query in queries if query.name.lower() in filters]


def run_dodam(args, query: GenericQuery, output_dir: Path) -> list[float]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.dodam_mode == "query-file":
        return run_dodam_query_file(args, query, output_dir)
    samples = []
    for repeat in range(args.repeats):
        output = output_dir / f"{query.name}-{repeat}.parquet"
        sql = f"COPY ({query.dodam_sql}) TO '{output}' ({copy_parquet_options(args.dodam_parquet_options)})"
        started = time.perf_counter()
        completed = subprocess.run(
            [args.dodam, "query", sql, "--batch-size", str(args.batch_size)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=args.timeout,
        )
        samples.append(time.perf_counter() - started)
        if args.show_stderr and completed.stderr:
            print(completed.stderr, end="")
        if completed.returncode != 0:
            raise SystemExit(completed.stderr or completed.stdout)
    return samples


COPY_PROFILE_RE = re.compile(r"copy_profile total=(\d+)us")


def run_dodam_query_file(args, query: GenericQuery, output_dir: Path) -> list[float]:
    total_runs = args.repeats
    sql_file = output_dir / f"{query.name}.sql"
    statements = []
    for repeat in range(total_runs):
        output = output_dir / f"{query.name}-{repeat}.parquet"
        statements.append(
            f"COPY ({query.dodam_sql}) TO '{output}' ({copy_parquet_options(args.dodam_parquet_options)});"
        )
    sql_file.write_text("\n".join(statements) + "\n")
    env = os.environ.copy()
    env["DODAM_PROFILE_COPY"] = "1"
    completed = subprocess.run(
        [args.dodam, "query-file", str(sql_file), "--batch-size", str(args.batch_size)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=args.timeout * max(1, total_runs),
        env=env,
    )
    if completed.returncode != 0:
        raise SystemExit(completed.stderr or completed.stdout)
    if args.show_stderr and completed.stderr:
        print(completed.stderr, end="")
    samples = [int(match) / 1_000_000 for match in COPY_PROFILE_RE.findall(completed.stderr)]
    if len(samples) != total_runs:
        raise SystemExit(
            f"expected {total_runs} Dodam copy_profile samples for {query.name}, got {len(samples)}\n"
            f"stderr:\n{completed.stderr}"
        )
    return samples


def run_duckdb(args, query: GenericQuery, output_dir: Path) -> list[float]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.duckdb_mode == "query-file":
        return run_duckdb_query_file(args, query, output_dir)
    samples = []
    for repeat in range(args.repeats):
        output = output_dir / f"{query.name}-{repeat}.parquet"
        sql = f"COPY ({query.duckdb_sql}) TO '{output}' ({copy_parquet_options(args.duckdb_parquet_options)})"
        started = time.perf_counter()
        completed = subprocess.run(
            [args.duckdb, "-c", sql],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=args.timeout,
        )
        samples.append(time.perf_counter() - started)
        if args.show_stderr and completed.stderr:
            print(completed.stderr, end="")
        if completed.returncode != 0:
            raise SystemExit(completed.stderr or completed.stdout)
    return samples


DUCKDB_TIMER_RE = re.compile(r"Run Time \(s\): real ([0-9.]+)")


def run_duckdb_query_file(args, query: GenericQuery, output_dir: Path) -> list[float]:
    total_runs = args.repeats
    sql_file = output_dir / f"{query.name}.sql"
    statements = [".timer on"]
    for repeat in range(total_runs):
        output = output_dir / f"{query.name}-{repeat}.parquet"
        statements.append(
            f"COPY ({query.duckdb_sql}) TO '{output}' ({copy_parquet_options(args.duckdb_parquet_options)});"
        )
    sql_file.write_text("\n".join(statements) + "\n")
    started = time.perf_counter()
    with sql_file.open("r") as stdin:
        completed = subprocess.run(
            [args.duckdb],
            stdin=stdin,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=args.timeout * max(1, total_runs),
        )
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        raise SystemExit(completed.stderr or completed.stdout)
    if args.show_stderr and completed.stderr:
        print(completed.stderr, end="")
    output = completed.stdout + completed.stderr
    samples = [float(match) for match in DUCKDB_TIMER_RE.findall(output)]
    if len(samples) != total_runs:
        # Some DuckDB builds suppress .timer for non-interactive COPY. Keep this
        # mode usable as a total-process smoke measurement instead of failing.
        return [elapsed / total_runs for _ in range(total_runs)]
    return samples


def copy_parquet_options(extra_options: str) -> str:
    options = "FORMAT PARQUET"
    extra_options = extra_options.strip()
    if extra_options:
        options = f"{options}, {extra_options}"
    return options


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
