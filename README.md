# dodam

`dodam` is a Rust OLAP database engine prototype built around Iceberg-style table
planning, Parquet storage, and Arrow `RecordBatch` vectorized execution.

The first milestone is a single-node engine. The module boundaries are already
shaped so the same logical flow can later run across multiple workers:

1. `catalog`: resolves a table into immutable file fragments.
2. `execution/logical`: owns query-facing expressions such as filters,
   projections, sort keys, and aggregate calls.
3. `execution/physical`: turns fragments into physical operators that exchange
   Arrow `RecordBatch` streams.
4. `execution/aggregate`: consumes batch streams into global or grouped
   aggregate results.
5. `storage`: reads columnar files into Arrow batches.
6. `engine`: exposes the public API used by the CLI or future SQL/server layer.

## Current CLI

Scan one Parquet file:

```sh
cargo run --bin dodam -- scan ./data/part-000.parquet
```

Scan a directory recursively:

```sh
cargo run --bin dodam -- scan ./data --batch-size 8192 --limit 100000
```

Project a subset of columns:

```sh
cargo run --bin dodam -- scan ./data --columns id,payload
```

Apply a vectorized filter:

```sh
cargo run --bin dodam -- scan ./data --columns payload --filter id=42
cargo run --bin dodam -- scan ./data --columns payload --filter "id>=200000 AND id<200010"
```

Apply an explicit order before limiting:

```sh
cargo run --bin dodam -- scan ./data --columns id,payload --order-by "id desc" --limit 10
```

Compute global or grouped aggregates:

```sh
cargo run --bin dodam -- aggregate ./data --agg 'count(*)' --agg sum(id) --agg avg(id)
cargo run --bin dodam -- aggregate ./data --agg min(payload) --agg max(payload) --filter "id>=1000"
cargo run --bin dodam -- aggregate ./data --group-by payload --agg 'count(*)' --agg sum(id)
```

Register a local Parquet file or directory in the workspace catalog:

```sh
cargo run --bin dodam -- catalog register orders ./data/orders
cargo run --bin dodam -- catalog refresh orders
cargo run --bin dodam -- catalog list
```

This creates or updates `.dodam/catalog.json` under the current directory. The
catalog stores the table location, schema/statistics snapshot, partition columns,
and the registered file fragment list. Registered table names can be used in SQL:

```sh
cargo run --bin dodam -- query "SELECT id, payload FROM orders WHERE id >= 10 ORDER BY id DESC LIMIT 5"
```

Registered tables use the file fragment snapshot stored at registration time.
Run `catalog refresh <name>` after adding or removing files; direct path queries
still discover the directory live.

Hive-style partition directories such as `dt=2026-07-01/country=kr` are recorded
on file fragments. Filters on those partition columns are applied before
Parquet row-group planning. Partition columns are also materialized into scan
output when selected or grouped:

```sh
cargo run --bin dodam -- query "SELECT id FROM orders WHERE dt = '2026-07-01'"
cargo run --bin dodam -- query "SELECT dt, count(*) FROM orders GROUP BY dt ORDER BY dt"
```

Run the supported SQL subset:

```sh
cargo run --bin dodam -- query "SELECT id, payload FROM './data' WHERE id >= 10 ORDER BY id DESC LIMIT 5"
cargo run --bin dodam -- query "SELECT t.id AS selected_id FROM './data' AS t WHERE t.payload = 'a' ORDER BY t.id DESC"
cargo run --bin dodam -- query "SELECT payload, count(*), sum(id) FROM './data' GROUP BY payload"
cargo run --bin dodam -- query "SELECT payload, count(*) AS n FROM './data' GROUP BY payload ORDER BY n DESC LIMIT 5"
cargo run --bin dodam -- query "SELECT payload AS p, count(*) AS n FROM './data' GROUP BY payload HAVING n > 1 ORDER BY p"
cargo run --bin dodam -- query "SELECT id AS selected_id FROM './data' WHERE payload = 'a' OR id = 1 ORDER BY payload, id DESC"
cargo run --bin dodam -- query "SELECT id FROM './data' WHERE (payload IN ('a', 'b') OR payload IS NULL) AND NOT id = 3"
cargo run --bin dodam -- query "SELECT DISTINCT payload FROM './data' ORDER BY payload"
```

The execution path is vectorized-first:

```text
ParquetScanExec -> FilterExec -> ProjectionExec -> LimitExec -> MetricsSink
ParquetScanExec -> FilterExec -> SortExec -> ProjectionExec -> LimitExec -> MetricsSink
ParquetScanExec -> FilterExec -> ProjectionExec -> DistinctExec -> SortExec -> LimitExec -> MetricsSink
ParquetScanExec -> FilterExec -> ProjectionExec -> AggregateSink
```

Every physical operator returns a stream of Arrow `RecordBatch` values. The CLI
only counts rows at the final sink. Projection is pushed into the Parquet reader
so unneeded columns are not decoded. Filters are represented as expression trees
and currently support `AND` plus `=`, `!=`, `<`, `<=`, `>`, and `>=` comparisons
over `Int32`, `Int64`, and `Utf8` columns. Predicates are decomposed into
conjuncts; row-group-prunable conjuncts are pushed into Parquet planning while
the original predicate remains as a residual batch filter for correctness.
Predicate pushdown currently prunes Parquet row groups using exact min/max
statistics when available. Full and projected scans decode row groups in
parallel and may yield batches in worker completion order. `LIMIT` caps the
number of rows returned by the stream, but without an explicit ordering operator
it does not define which files or row groups those rows come from. `ORDER BY`
uses a blocking in-memory sort over the scanned batches and pushes `LIMIT` into
Arrow's sort indices for top-N scans. It gives deterministic `LIMIT` results for
supported Arrow-sortable columns. Selective filtered scans use a single-reader
pruning path to avoid metadata planning overhead. Parquet
footer/Arrow reader metadata is cached by path, file size, and modified
timestamp. Scan metrics include phase timings for metadata access, row-group
planning/pruning, Parquet decode, filter, projection, and limit so regressions
can be tied to a specific vectorized stage. Page-level pruning, broader
expression evaluation, external sort, and distributed/grouped partial
aggregation should be added on this batch boundary rather than row-by-row.
Aggregation currently supports global and grouped `COUNT(*)`, `COUNT(col)`,
numeric `SUM/AVG`, and numeric or `Utf8` `MIN/MAX`.
Global aggregation updates `SUM/MIN/MAX` with Arrow aggregate kernels per
`RecordBatch`, while `COUNT` uses Arrow array null counts. Grouped aggregation
encodes group keys with Arrow's row format per input batch, collects row indices
per encoded key, then updates each group's accumulators with Arrow kernels over
the group's selected `RecordBatch`.
`DISTINCT` is implemented as a blocking batch operator over the projected output:
it encodes full rows with Arrow's row format, tracks unique encoded rows, then
uses Arrow take kernels to produce the first occurrence of each distinct row.
The SQL frontend is intentionally thin: it accepts simple `SELECT` queries over
local Parquet paths or registered tables, with optional `WHERE`, `GROUP BY`,
`HAVING`, `ORDER BY`, `LIMIT`, `DISTINCT`, and equality joins. `WHERE` supports
comparisons, `AND`/`OR`/`NOT`, parentheses, `IN`, and `IS NULL`/`IS NOT NULL`.
`ORDER BY` supports multiple columns, and aliases are accepted for supported
column and aggregate items. Single-table aliases are accepted, and join columns
must be table-qualified. Plain `SELECT DISTINCT` is supported for scan outputs;
`DISTINCT ON`, aggregate DISTINCT, and JOIN DISTINCT are rejected for now. SQL
scan and aggregate results are both materialized as Arrow `RecordBatch` values,
with SELECT aliases reflected in the output schema. Aggregate result batches
support `HAVING`, `ORDER BY` over group columns, aggregate expressions, or SELECT
aliases, followed by `LIMIT`. Join aggregates are supported by aggregating the
streamed join output batches. Subqueries, arithmetic expressions, computed SQL
expressions, and DML are rejected for now.

## Benchmark

### Scan microbenchmark

Run the scan benchmark:

```sh
cargo bench --bench scan
```

Current local baseline on the generated benchmark files:

```text
scan/full/uncompressed       ~19.7 ms
scan/projected/uncompressed  ~595 us
scan/filtered/uncompressed   ~825 us

scan/full/snappy             ~19.3 ms
scan/projected/snappy        ~612 us
scan/filtered/snappy         ~1.07 ms

scan/full/zstd               ~21.9 ms
scan/projected/zstd          ~843 us
scan/filtered/zstd           ~1.34 ms
```

The generated benchmark files currently use 1,048,576 rows, 17 columns, and
65,536-row row groups. CLI scan output includes total/scanned/pruned row group
counts and compressed byte counts for checking projection and pruning
effectiveness.
Timing is also reported in microseconds for the metadata, planning, decode,
filter, projection, and limit phases.

### TPC-H SF100

The following end-to-end comparison was measured on 2026-08-23. Both engines
read the same approximately 38 GiB SF100 Parquet dataset and write every query
result as Snappy Parquet. Each query ran five times sequentially; the table uses
the median of the final four runs after discarding the first run as warmup.
Lower is better, and the ratio is `Dodam / DuckDB`.

Q09 and Q11 were remeasured with the same procedure after the dense-layout
boundary improvements in commit `35a7895`. The median sum substitutes those two
new medians into the original 22-query record.

```sh
scripts/gen_tpchgen_parquet.sh 100 /tmp/dodam-tpchgen-sf100
cargo build --release --bin tpch_real_inprocess
python3 scripts/compare_tpch_duckdb.py \
  --data-dir /tmp/dodam-tpchgen-sf100 \
  --output-dir /tmp/dodam-tpch-sf100-compare \
  --duckdb-mode cli \
  --repeats 5 \
  --warmup 1 \
  --batch-size 16384 \
  --timeout 180 \
  --json-out /tmp/dodam-tpch-sf100-compare/report.json
```

| Query | Dodam (ms) | DuckDB (ms) | Ratio |
|---|---:|---:|---:|
| Q01 | 2,593.222 | 2,814.750 | 0.921x |
| Q02 | 806.385 | 896.555 | 0.899x |
| Q03 | 2,037.043 | 3,292.490 | 0.619x |
| Q04 | 1,336.434 | 1,801.345 | 0.742x |
| Q05 | 3,803.138 | 4,243.518 | 0.896x |
| Q06 | 1,378.774 | 1,669.688 | 0.826x |
| Q07 | 3,396.210 | 4,328.756 | 0.785x |
| Q08 | 4,220.724 | 5,399.882 | 0.782x |
| Q09 | 6,899.103 | 8,848.585 | 0.780x |
| Q10 | 3,580.100 | 4,362.929 | 0.821x |
| Q11 | 336.453 | 469.825 | 0.716x |
| Q12 | 1,889.470 | 2,326.796 | 0.812x |
| Q13 | 2,354.430 | 4,082.296 | 0.577x |
| Q14 | 2,820.988 | 3,632.703 | 0.777x |
| Q15 | 1,968.926 | 3,170.654 | 0.621x |
| Q16 | 803.017 | 901.507 | 0.891x |
| Q17 | 2,474.120 | 3,776.012 | 0.655x |
| Q18 | 2,309.239 | 5,915.427 | 0.390x |
| Q19 | 2,259.660 | 3,956.652 | 0.571x |
| Q20 | 3,219.416 | 3,805.236 | 0.846x |
| Q21 | 3,356.837 | 8,421.781 | 0.399x |
| Q22 | 827.516 | 1,070.676 | 0.773x |
| **Median sum** | **54,671.202** | **79,188.064** | **0.690x** |

All original full-suite executions and the targeted Q09/Q11 remeasurements
completed successfully. Final-run row counts and non-numeric values matched
DuckDB; numeric values had at most `3.856e-14` relative error from `DOUBLE`
versus `DECIMAL` representation. The substituted median-sum elapsed time is
31.0% lower than DuckDB's, and no individual query is slower. The host used an
AMD Ryzen 9 7945HX (16 cores/32 threads), Linux 6.14, Rust 1.89.0, and DuckDB
1.5.4. Dodam used its default 16-thread Rayon pool; DuckDB used its default CLI
settings.

### TPC-H SF200 dense-boundary follow-up

Q09 and Q11 were also run individually at SF200 with one execution and no
warmup, because running the full suite in one session was not reliable on the
30 GiB host. `Dodam speedup` is the previous Dodam time divided by the improved
time, so larger is better.

| Query | Dodam before (ms) | Dodam after (ms) | Dodam speedup | Time reduction | DuckDB (ms) | After / DuckDB |
|---|---:|---:|---:|---:|---:|---:|
| Q09 | 45,570.507 | 17,145.217 | 2.658x | 62.4% | 26,793.000 | 0.640x |
| Q11 | 1,316.990 | 745.616 | 1.766x | 43.4% | 871.150 | 0.856x |

DuckDB Q09 used a 12 GiB memory limit and temporary spilling; its default
configuration was killed by the host OOM handler. Q11 used DuckDB's default CLI
settings. The final SF200 Q09 output matched the previously validated 175-row
result exactly, and the DuckDB differential TPC-H-lite test passed after both
changes.

## Roadmap

- Single node:
  - Iceberg metadata/snapshot reader
  - projection and filter pushdown
  - vectorized expressions
  - hash aggregation and sort
  - simple SQL or Substrait/DataFusion-compatible frontend
- Distributed:
  - split planning from execution scheduling
  - fragment assignment to workers
  - shuffle exchange operators
  - object-store based reads for S3/GCS/Azure/local
  - coordinator metadata service
