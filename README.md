# dodam

`dodam` is a Rust OLAP database engine prototype built around Iceberg-style table
planning, Parquet storage, and Arrow `RecordBatch` vectorized execution.

The first milestone is a single-node engine. The module boundaries are already
shaped so the same logical flow can later run across multiple workers:

1. `catalog`: resolves a table into immutable file fragments.
2. `execution/logical`: owns query-facing expressions such as filters,
   projections, sort keys, and aggregate calls.
3. `optimizer`: rewrites the shared `plan::LogicalPlan` and estimates join
   alternatives without introducing query-specific execution paths.
4. `plan`: lowers optimized logical nodes into physical pipelines, including
   scan pushdown, exchanges, and cost-based join strategy selection.
5. `execution/physical`: turns fragments into physical operators that exchange
   Arrow `RecordBatch` streams.
6. `execution/aggregate`: consumes batch streams into global or grouped
   aggregate results.
7. `storage`: reads columnar files into Arrow batches.
8. `engine`: exposes the public API used by the CLI or future SQL/server layer.

The general optimizer folds projection, filter, sort/top-N, limit, and distinct
nodes into a scan only when their relational ordering is preserved. The
physical scan reads the union of output, predicate, and ordering columns once,
sends prunable conjuncts to Parquet row-group pruning, and keeps the full
predicate as a vectorized residual filter. Joins choose the smaller estimated
input as the hash build side and switch to partitioned hash when that build
exceeds the configured memory budget. `DODAM_OPTIMIZER_TRACE=1` prints the
logical rules selected for a scan.

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

### TPC-H SF100 and SF200

The following end-to-end comparisons were measured on 2026-08-23. Both engines
read the same approximately 38 GiB SF100 or 75 GiB SF200 Parquet dataset and
write every query result as Snappy Parquet. SF100 uses the median of the final
four runs from five sequential executions after discarding the first run as
warmup. SF200 ran every query sequentially, one query per process, with one
execution and no warmup because the full-suite session was not reliable on the
30 GiB host. All times are milliseconds; lower is better, and each ratio is
`Dodam / DuckDB`.

Q09 and Q11 use the measurements taken after the dense-layout boundary
improvements in commit `35a7895`. Q06 and Q22 use the measurements taken after
removing an unprofitable late-materialization attempt and adding a direct
atomic dense-set build, respectively. The SF100 aggregate substitutes the four
new medians into the original 22-query record, while the SF200 aggregate sums
the 22 individual executions.

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

| Query | SF100 Dodam | SF100 DuckDB | SF100 ratio | SF200 Dodam | SF200 DuckDB | SF200 ratio |
|---|---:|---:|---:|---:|---:|---:|
| Q01 | 2,593.222 | 2,814.750 | 0.921x | 5,921.390 | 5,026.026 | 1.178x |
| Q02 | 806.385 | 896.555 | 0.899x | 1,832.128 | 1,728.292 | 1.060x |
| Q03 | 2,037.043 | 3,292.490 | 0.619x | 5,509.412 | 8,009.946 | 0.688x |
| Q04 | 1,336.434 | 1,801.345 | 0.742x | 3,521.010 | 3,792.233 | 0.928x |
| Q05 | 3,803.138 | 4,243.518 | 0.896x | 9,686.127 | 9,268.823 | 1.045x |
| Q06 | 987.902 | 1,640.959 | 0.602x | 3,662.726 | 3,284.201 | 1.115x |
| Q07 | 3,396.210 | 4,328.756 | 0.785x | 10,519.008 | 10,872.687 | 0.967x |
| Q08 | 4,220.724 | 5,399.882 | 0.782x | 11,934.727 | 12,576.822 | 0.949x |
| Q09 | 6,899.102 | 8,848.585 | 0.780x | 17,145.217 | 26,793.000 | 0.640x |
| Q10 | 3,580.100 | 4,362.929 | 0.821x | 8,955.118 | 9,116.241 | 0.982x |
| Q11 | 336.453 | 469.825 | 0.716x | 745.616 | 871.150 | 0.856x |
| Q12 | 1,889.470 | 2,326.796 | 0.812x | 4,857.482 | 4,746.457 | 1.023x |
| Q13 | 2,354.430 | 4,082.296 | 0.577x | 4,788.346 | 8,254.875 | 0.580x |
| Q14 | 2,820.988 | 3,632.703 | 0.777x | 7,157.254 | 7,744.794 | 0.924x |
| Q15 | 1,968.926 | 3,170.654 | 0.621x | 5,640.596 | 6,585.578 | 0.857x |
| Q16 | 803.017 | 901.507 | 0.891x | 1,792.309 | 1,809.433 | 0.991x |
| Q17 | 2,474.120 | 3,776.012 | 0.655x | 7,782.446 | 8,465.569 | 0.919x |
| Q18 | 2,309.239 | 5,915.427 | 0.390x | 4,920.321 | 10,862.545 | 0.453x |
| Q19 | 2,259.660 | 3,956.652 | 0.571x | 6,352.603 | 7,675.516 | 0.828x |
| Q20 | 3,219.416 | 3,805.236 | 0.846x | 8,202.987 | 8,076.513 | 1.016x |
| Q21 | 3,356.837 | 8,421.781 | 0.399x | 8,070.544 | 18,133.437 | 0.445x |
| Q22 | 629.042 | 913.394 | 0.689x | 2,012.324 | 2,040.707 | 0.986x |
| **Aggregate** | **54,081.856** | **79,002.053** | **0.685x** | **141,009.691** | **175,734.845** | **0.802x** |

Dodam is 31.5% faster by the SF100 aggregate and 19.8% faster by the SF200
aggregate. It wins 22 of 22 queries at SF100 and 16 of 22 at SF200. At SF200,
Q09 improved from 45,570.507 ms to 17,145.217 ms (2.658x), Q11 improved from
1,317.365 ms to 745.616 ms (1.766x), Q06 improved from 4,114.202 ms to
3,662.726 ms (1.123x), and Q22 improved from 2,666.801 ms to 2,012.324 ms
(1.325x).

Dodam completed all 22 SF200 queries with its default settings. DuckDB completed
21 with its defaults: Q09 was killed by the host OOM handler after reaching
approximately 23.5 GiB RSS. The SF200 DuckDB Q09 value in the table is a retry
with a 12 GiB memory limit, 16 threads, and temporary spilling. Final-run row
counts and non-numeric values matched DuckDB; numeric values had at most
`3.856e-14` relative error at SF100. The final SF200 Q09 output matched the
previously validated 175-row result exactly, and the DuckDB differential
TPC-H-lite test passed after the Q06, Q09, Q11, and Q22 changes.

The generalized optimizer was regression-checked at SF100 with the same
five-run, one-warmup protocol. This is a focused validation rather than a
replacement for the full-suite table above.

| Query | Dodam ms | DuckDB ms | ratio |
|---|---:|---:|---:|
| Q06 | 1,010.601 | 1,706.408 | 0.592x |
| Q09 | 6,861.022 | 9,176.985 | 0.748x |

Q06's single output differed by about `3e-16` relatively because of floating
point accumulation order. All 175 Q09 keys matched and its maximum relative
numeric difference was `2.43e-15`.

The host used an AMD Ryzen 9 7945HX (16 cores/32 threads), Linux 6.14, Rust
1.89.0, and DuckDB 1.5.4. Dodam used its default 16-thread Rayon pool; DuckDB
used its default CLI settings except for the SF200 Q09 retry described above.

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
