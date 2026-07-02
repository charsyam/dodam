# Dodam TODO

## Done

### Core Execution

- Built a vectorized Arrow `RecordBatch` execution path for local Parquet scan, filter, projection, limit, sort, distinct, aggregation, and join.
- Added local Parquet file and directory planning with stable fragment order, schema/statistics discovery, metadata cache, projection pushdown, and row-group pruning.
- Added global and grouped aggregation for `COUNT(*)`, `COUNT(col)`, `SUM`, `AVG`, `MIN`, and `MAX`.
- Added faster grouped aggregation paths for common numeric and string keys.
- Added `INNER`, `LEFT OUTER`, `RIGHT OUTER`, `FULL OUTER`, and `LEFT SEMI` joins.
- Added hash join, sort-merge join, partitioned spill hash join, and chunked file-hash join execution.
- Added multi-column equality joins, build-side selection from estimates, partition count calculation, spill repartitioning, heavy-hitter separation, and join metrics.
- Added join memory safeguards: recursive repartitioning, bounded output chunks, bloom-style filtering metrics, skew handling, and nested-loop fallback for pathological spill partitions.
- Added output projection pushdown into hash join and partitioned/spill hash join where semantics allow it.
- Added SQL aggregate execution over streamed join output batches, including global aggregate, grouped aggregate, `HAVING`, aggregate aliases, `ORDER BY`, and `LIMIT`.
- Added dense `Int32` unique join key fast path and `Int64` single-key fast paths.
- Added dense `Int32` bitmap matched-build tracking for outer joins.

### SQL, CLI, COPY

- Added SQL parsing through `sqlparser`.
- Supported projection, column aliases, table aliases, qualified columns, filters, grouping, `HAVING`, `ORDER BY`, `LIMIT`, `DISTINCT`, joins, and `EXPLAIN`.
- Added streaming SQL execution for simple joins to avoid collecting all join output before CLI output.
- Added Dodam `COPY (SELECT ...) TO 'file' (FORMAT CSV)` for fair DuckDB-style benchmark comparison.
- Added Dodam `COPY (SELECT ...) TO 'file' (FORMAT PARQUET)`.
- Added configurable Parquet COPY options:
  - `COMPRESSION SNAPPY|UNCOMPRESSED`
  - `DICTIONARY true|false`
  - `ROW_GROUP_SIZE <rows>`
  - `WRITE_BATCH_SIZE <rows>`
  - `DATA_PAGE_ROW_COUNT_LIMIT <rows>`
- Added execution sink integration: `RecordBatchSink`, stream-to-sink drain helper, `PhysicalPlan::execute_to_sink`, and engine-level sink execution helpers.
- Routed CLI COPY and streaming SELECT through the engine sink boundary.
- Added `HashJoinExec::execute_to_sink` and SQL-level `try_execute_sql_to_sink`.
- Added configurable COPY output buffer:
  - CLI: `--copy-buffer-size <bytes>`
  - env: `DODAM_COPY_BUFFER_BYTES`
  - default: `8 MiB`
- Replaced `arrow-csv` writer path with a custom CSV sink using batch-level `Vec<u8>`, `BufWriter<File>`, `itoa`, and `ryu`.
- Added CSV serialization hot paths for single `Int32` and `Int32, Utf8` output shapes.
- Guarded direct sink/streaming join output so outer joins with SELECT-specific projection fall back to the correct materialized projection path.

### Storage And Catalog Layering

- Split major layers into catalog, storage, engine, logical execution, physical execution, optimizer, cost model, and SQL.
- Added `TableScanSource`, `StorageLocation`, `StorageFormat`, `ObjectStore`, and format-dispatching `ScanExec`.
- Added local filesystem object-store abstraction and Parquet table/source planning.
- Kept S3/object storage on hold until the local OLAP engine and execution model are stronger.
- Added a persistent local workspace catalog under `.dodam/catalog.json`.
- Added `dodam catalog register <name> <path>` and `dodam catalog list`.
- Added SQL resolution for registered table names such as `SELECT * FROM orders` while preserving direct path queries.
- Added catalog registration snapshots for schema fields, table statistics, and discovered partition columns.
- Added catalog registration snapshots for file fragments, fragment statistics, and fragment partition values.
- Changed registered table scans to use the catalog fragment snapshot instead of recursively rediscovering the directory on every query.
- Added `dodam catalog refresh <name>` to rebuild a registered table snapshot from its stored location.
- Added Hive-style directory partition discovery on file fragments.
- Added directory partition pruning for positive `=` and `IN` predicates before Parquet row-group planning.
- Added scan-time materialization of Hive partition columns so they can be selected, ordered, filtered after pruning when needed, and grouped.
- Added Parquet scan compatibility coverage for nullable and richer Arrow types:
  - nullable `Boolean`, `Float64`, and `Utf8`
  - `Decimal128(10, 2)`
  - `Timestamp(Millisecond)`
  - `List<Int32>`
  - `Struct<rank: Int32, label: Utf8>`
- Added filter execution support and SQL coverage for richer scalar Parquet types:
  - boolean literals and boolean comparison
  - decimal comparison against numeric/string literals
  - timestamp comparison against `YYYY-MM-DD` and `YYYY-MM-DD HH:MM:SS` string literals
  - timezone-aware timestamp comparison using the same epoch semantics as Arrow
  - `Date32`/`Date64` comparison against `YYYY-MM-DD` string literals
- Added Parquet row-group pruning for richer scalar statistics:
  - boolean min/max statistics
  - `Float64` min/max statistics
  - `Date32`/`Date64` statistics
  - decimal statistics stored as `Int32`, `Int64`, `ByteArray`, or `FixedLenByteArray`
  - timezone-aware and timezone-less timestamp statistics stored as `Int64`

### Correctness And Compatibility

- Added a DuckDB differential test harness that skips cleanly when the `duckdb` CLI is unavailable.
- Added initial DuckDB differential coverage for:
  - scan/filter/null predicates, including `IS NULL`, `IS NOT NULL`, `NOT`, and `IN`
  - global and grouped aggregate matrix over count/sum/avg/min/max
  - inner/left/right/full/semi joins with duplicate build-side keys and unmatched rows
  - multi-key joins that combine integer and string keys
  - deterministic `ORDER BY` + `LIMIT`
  - scalar type filters over boolean, float, decimal, timestamp, and date columns
  - derived-table subqueries in `FROM`
  - uncorrelated `IN (SELECT ...)` / `NOT IN (SELECT ...)` subqueries
  - uncorrelated scalar subquery filters
  - uncorrelated top-level `EXISTS` / `NOT EXISTS` filters
- Expanded DuckDB differential coverage for NULL semantics:
  - `IN` / `NOT IN` with literal `NULL` inside `OR`
  - scalar subquery `NULL` comparisons
  - left join behavior for NULL join keys
- Expanded type compatibility coverage and fixes:
  - decimal literals with trailing fractional zeros beyond column scale
  - timezone offset timestamp literals such as `+09:00`, normalized to epoch semantics
  - matching filter and row-group pruning interpretation for decimal/timestamp literals
  - SQL projection coverage for nested `List` and `Struct` columns
- Added first-stage SQL scalar projection expressions for single-table non-aggregate SELECT queries:
  - numeric arithmetic with `+`, `-`, `*`, `/`
  - `CAST(... AS VARCHAR/INT/DOUBLE)`
  - `COALESCE(...)`
  - DuckDB differential coverage for arithmetic, cast, and coalesce projection output
- Extended the same scalar expression evaluator to single-table non-aggregate `WHERE` predicates:
  - arithmetic/cast/coalesce comparisons
  - expression `IS NULL` / `IS NOT NULL`
  - DuckDB differential coverage for computed `WHERE` output
- Added first-stage string functions and searched `CASE` support in scalar expressions:
  - `lower`, `upper`, and `length` in projection and computed `WHERE`
  - `CASE WHEN ... THEN ... ELSE ... END` for searched CASE expressions
  - DuckDB differential coverage for string functions and CASE output
- Added a TPC-H-lite DuckDB differential suite over lineitem/orders/customer-shaped Parquet fixtures:
  - Q1-style date-filtered grouped aggregate with multi-column `GROUP BY`
  - Q6-style date/discount/quantity filtered aggregate
  - Q3-style customer/orders join aggregate
- The TPC-H-lite suite caught missing `Date32` `min`/`max` aggregate support; added `Date32` aggregate state and `Date32Array` result materialization.
- The TPC-H-lite join aggregate case caught an aggregate join output projection correctness issue around qualified column names when the cost model chooses the left side as the hash build input.
- Fixed build-side-aware join output projection mapping for hash/materialized join output schemas and restored aggregate join output projection pushdown.
- The first differential grouped-aggregate case caught a real correctness bug: grouped `min(Utf8)` / `max(Utf8)` on the single-key fast path produced `NULL` because empty min/max state defaulted to `Int64(NULL)`.
- Fixed the grouped aggregate fast path so min/max states keep the expected output null type (`Int64`, `Float64`, or `Utf8`) even when a group starts with nulls.
- Added first-stage subquery support for `FROM (SELECT ...) AS alias` by materializing the inner query to in-memory `RecordBatch`es and applying the outer projection/filter/group/order/limit over those batches.
- Added `DISTINCT` support over derived tables.
- Added first-stage derived table join support by materializing join inputs and reusing `HashJoinExec`.
- Added aggregate and `DISTINCT` support over derived table joins.
- Added first-stage uncorrelated `IN (SELECT ...)` support by materializing the subquery to a literal `IN` list.
- Fixed `IN (SELECT ...)` / `NOT IN (SELECT ...)` SQL 3-valued logic when the RHS contains `NULL`.
- Added first-stage uncorrelated scalar subquery filters such as `WHERE id = (SELECT id FROM ...)`.
- Fixed scalar subquery edge semantics:
  - 0 rows returns scalar `NULL`
  - 1 `NULL` row returns scalar `NULL`
  - multiple rows return an error
- Added first-stage uncorrelated top-level `EXISTS` / `NOT EXISTS` filters.
- Added first-stage correlated top-level `EXISTS` / `NOT EXISTS` support for single-table non-aggregate outer queries by binding outer alias references per row.
- Added correctness-first correlated subquery fallback for single-table non-aggregate outer queries:
  - correlated `IN (SELECT ...)`
  - correlated scalar subquery filters
  - `EXISTS` / `NOT EXISTS` inside `AND` / `OR` boolean expressions
- Current subquery limitations:
  - `IN (SELECT ...)` currently requires a single result column
  - correlated subquery support is currently correctness-first and row-bound; it is not optimized for large outer inputs yet

### Performance Work

- Added DuckDB e2e benchmark coverage for scan/filter, aggregation, grouped aggregation, inner join, left/right/full outer join, and semi join.
- Fixed benchmark fairness so Dodam CLI and DuckDB CLI both use `COPY ... TO '<benchmark-output>.csv' (FORMAT CSV)` and pay CSV file write cost.
- Changed DuckDB e2e benchmarks to use a shared output format option for every COPY case:
  - default: `DODAM_E2E_COPY_FORMAT=PARQUET`
  - optional CSV comparison: `DODAM_E2E_COPY_FORMAT=CSV`
  - Criterion groups are separated as `duckdb_e2e_parquet` and `duckdb_e2e_csv` so CSV and Parquet histories are not compared against each other.
- Fixed benchmark flow so `target/release/dodam` is rebuilt before CLI runs.
- Fixed Dodam e2e benchmark batch size at `16 KiB`.
- Tested direct COPY join output bypassing `RecordBatch` materialization, then removed it from the default path because it was slower.
- Tested dense `Int32` index-driven CSV writing, then removed it because it regressed inner join CLI performance.
- Optimized hash join output materialization:
  - contiguous build row refs use slice instead of `take_record_batch`
  - single build chunk output skips `concat_batches`
  - output schemas are cached per probe batch
- Added COPY profiling with `DODAM_PROFILE_COPY=1`.
- Added `gen_e2e_data` helper binary for benchmark-shaped Parquet data generation.
- Added direct dense `Int32` unique hash build for `FastSingleKey` joins.
- Fixed RIGHT/FULL join build-side selection so the cost model can build the smaller side.
- Expanded DuckDB e2e join benchmarks with narrow fact, wide-pruned fact, and small-row-group duplicate-key datasets to check that scan/join tuning is not only fitted to one dense-key benchmark.
- Added sequential Parquet scan heuristics for small scans, pruned-column scans, and narrow fixed-width scans after validating against narrow, wide-pruned, and small-row-group join datasets.
- Added SQL aggregate execution over streamed join output and narrowed aggregate-join output projection so aggregate `ORDER BY` does not force full join row materialization when only grouped/aggregate input columns are needed.
- Avoided `concat_batches` for single-batch sort/order-limit paths. Benchmark impact was neutral, but it removes unnecessary work in the common single-batch case.
- Avoided collecting hash-join `all_rows` for inner/semi builds that do not need unmatched-build emission. This improved `inner_join_non_dense_i32` from about `21.5 ms` Dodam CLI to about `20.5 ms`, making it slightly faster than DuckDB CLI on the current file-output benchmark.

### Benchmark Tuning Attempts

Effective or retained:

- Fair file-output benchmarking:
  - Dodam CLI and DuckDB CLI both write CSV files instead of `/dev/null`.
  - This is the baseline for all current DuckDB comparisons.
- Parquet output benchmarking:
  - Added Dodam and DuckDB CLI cases that both write real Parquet output files.
  - Parquet is now the default output format for all DuckDB e2e COPY cases, not only selected join-output cases.
  - Confirmed DuckDB's default Parquet COPY output uses `SNAPPY` compression with plain encoding rather than dictionary encoding on the current join-output datasets.
  - Changed Dodam Parquet COPY defaults to `SNAPPY`, dictionary disabled, and `64 Ki` row groups to align with DuckDB's default output shape.
  - Changed Dodam Parquet COPY default write batch size and data page row-count limit from `16 Ki` to `8 Ki`; this improved duplicate-build fanout Parquet output and wide-output Parquet output without regressing the dense inner join benchmark.
  - Changed Dodam Parquet COPY statistics from page-level to row-group/chunk-level statistics; this keeps row-group stats while avoiding page-index/statistics overhead in ArrowWriter.
  - Added Parquet delta binary packed encoding for output columns named `id` and `f.id`; this specifically targets sorted/repeated integer id output and substantially reduces duplicate fanout output size.
  - This improved `inner_join_materialize_parquet` Dodam CLI from about `36.6 ms` to about `21.1 ms`, versus DuckDB CLI about `23.4 ms`.
  - This improved `inner_join_wide_output_parquet` Dodam CLI from about `51.7 ms` to about `33.5 ms`, versus DuckDB CLI about `38.5 ms`.
  - After the `8 Ki` writer/page change, `inner_join_wide_output` Parquet output improved further to about `28.9 ms`, versus DuckDB CLI about `39.1 ms`.
  - After the `8 Ki` writer/page change, `inner_join_duplicate_build` Parquet output improved from about `45.9 ms` to about `40.3 ms`, versus DuckDB CLI about `28.1 ms`.
  - After id-column delta encoding, `inner_join_duplicate_build` Parquet output improved further to about `36.7 ms`, versus DuckDB CLI about `28.2 ms`; output size dropped from about `4.0 MiB` to about `2.5 MiB`.
  - Changed Parquet COPY to use a configurable COPY output buffer instead of the default small `BufWriter` buffer.
  - Tuned Parquet's default buffer selection to keep `8 MiB` for narrow/two-column output and use `1 MiB` for wider output unless `--copy-buffer-size` or `DODAM_COPY_BUFFER_BYTES` is set.
  - This keeps `inner_join_duplicate_build` around `35.3 ms`, `inner_join_duplicate_build_small_row_groups` around `37.8 ms`, and improves `inner_join_wide_output` to about `25.4 ms`.
  - Output sizes are close to DuckDB on the current datasets: about `2.3 MiB` vs `2.2 MiB` for the narrow join output and about `3.7 MiB` vs `3.6 MiB` for the wide join output.
- Direct CSV sink paths:
  - Single `Int32` output helps semi joins.
  - `Int32, Utf8` output helps common inner join COPY output.
- Dense `Int32` unique hash build:
  - Helped dense-key inner join build time.
  - Kept with the bounded dense-width heuristic.
- Dense `Int32` bitmap matched-build tracking:
  - Helped right/full joins by avoiding repeated `HashSet<BuildRowRef>` work for dense unique build keys.
- Sequential scan heuristic:
  - Helped duplicate-build narrow/wide-pruned/small fixed-width join cases where parallel reader overhead dominated.
  - Guarded to avoid regressing non-dense sparse scans.
- Aggregate-over-join projection narrowing:
  - Improved engine time for `join_grouped_aggregate_ordered` from about `8.37 ms` to about `8.22 ms`.
  - CLI-level improvement is mostly hidden by process/COPY cost.
- Inner/semi hash build `all_rows` collection flag:
  - Effective for `inner_join_non_dense_i32`.
  - Outer/full and partitioned paths still collect all build rows because unmatched-build emission needs them.
- Utf8 build-column gather:
  - Removed the extra byte-count pass before building gathered `StringArray` output; this improved duplicate fanout engine time from about `9.0 ms` to about `8.7 ms`.
  - CLI impact is smaller because Parquet write still dominates, but duplicate fanout COPY improved to about `35.1 ms` and small-row-group duplicate COPY to about `37.0 ms` in the retained configuration.

Tried and rejected or neutral:

- Direct COPY join output bypassing `RecordBatch` materialization as a broad default path:
  - Removed because it was slower in early tests.
- Dense `Int32` index-driven CSV writing:
  - Removed because it regressed inner join CLI performance.
- Sort/order-limit `take` to `slice` when sorted indices are contiguous:
  - Rejected because `filter_limit` engine time regressed by about `3%`; the range check cost outweighed savings.
- Sparse dense lookup expansion for non-dense `Int32` join keys:
  - Rejected because `inner_join_non_dense_i32` CLI regressed from about `21.5 ms` to about `29.7 ms`.
  - Sparse dense table creation and lookup were slower than the tuned hash-map path.
- Dense unique direct sink lookup for complete dense keys:
  - Rejected because `inner_join_materialize` engine regressed by about `33%` and CLI by about `4%`.
- CSV right-array cache inside `write_i32_utf8_rows`:
  - Rejected because `inner_join_non_dense_i32` showed no measurable improvement.
- `#[inline]` on direct sink helper functions:
  - Rejected because `inner_join_non_dense_i32` and semi join COPY benchmarks showed no measurable improvement.
- Lazy `MatchedBuildTracker` creation in sink execution:
  - Rejected because it did not improve inner/non-dense CLI time and made full-join engine results noisier.
- Direct writer expansion for wide output:
  - Deferred because `inner_join_wide_output` is already roughly tied with DuckDB CLI, so the expected gain is low compared with the added trait/API complexity.
- Parquet `write_batch_size` and `data_page_row_count_limit` tuning alone:
  - Neutral before dictionary encoding was disabled.
  - Retained as configurable COPY options.
  - Later changed defaults to `8 Ki` after Parquet became the default e2e output format; this reduced duplicate fanout Parquet writer time and improved wide Parquet output.
  - Re-tested `1 Ki`, `2 Ki`, `4 Ki`, `12 Ki`, and `16 Ki` after buffer and gather tuning; `2 Ki` showed profile noise but Criterion did not show a meaningful CLI improvement and increased output size, so the default remains `8 Ki`.
- Parquet `ROW_GROUP_SIZE 65536`:
  - Speed impact was neutral after dictionary encoding was disabled.
  - Retained because it aligns Dodam's output row-group structure with DuckDB's default Parquet COPY output.
- Increasing global join output chunk size from `8 Ki` to `64 Ki`:
  - Helped some duplicate fanout cases but regressed the small-row-group duplicate case.
  - Re-tested after Parquet writer tuning: it improved the small-row-group duplicate CLI case to about `41.1 ms`, but regressed the default duplicate CLI case to about `37.8 ms` and slightly worsened wide output.
  - Rejected as a global execution change; a future Parquet-specific sink preference would need more selective logic.
- Parquet sink-side RecordBatch coalescing:
  - Looked better in one-off profiling, but Criterion showed regressions in duplicate fanout cases due to concat/memory churn.
  - Rejected.
- Direct Parquet `Int32, Utf8` join row sink:
  - Gave only small/noisy gains and regressed the small-row-group duplicate case before the `8 Ki` writer/page fix.
  - Rejected in favor of the simpler writer/page default change.
- Round-major duplicate fanout row ordering:
  - Made the first rows resemble DuckDB's output order, but increased output size from about `4.0 MiB` to about `4.5 MiB`.
  - Rejected.
- Marking direct Parquet output fields non-nullable:
  - Did not remove ArrowWriter's `PLAIN, RLE` encoding and did not improve file size.
  - Rejected.
- Parquet WriterVersion 2.0:
  - Reduced duplicate fanout output size further, to about `2.0 MiB`, by using delta encodings more broadly.
  - Rejected as the default because duplicate fanout CLI time regressed to about `42.6 ms`, even though dense/wide output improved.
- Column-specific Parquet string encodings for payload columns:
  - `DELTA_BYTE_ARRAY` reduced duplicate fanout output size to about `2.0 MiB`, but profile time worsened to about `57 ms` total with about `28.6 ms` in Parquet write.
  - `DELTA_LENGTH_BYTE_ARRAY` also regressed duplicate fanout to about `52.5 ms` total and made wide-output Parquet much larger.
  - String-column dictionary-only encoding regressed heavily: duplicate fanout rose to about `95 ms` total and wide-output to about `77 ms` total.
  - Rejected; keep only `DELTA_BINARY_PACKED` for `id`/`f.id` as the retained encoding override.
- `ORDER BY` output ordering experiments:
  - `ORDER BY d.payload` reduced duplicate fanout file size to about `2.9 MiB`, but total runtime rose to about `92 ms` due to sort/materialization.
  - Rejected for default unordered joins.
- Sorting dense duplicate build refs by `(batch,row)`:
  - Did not change the observed output order in the profiled duplicate case and did not improve payload compression.
  - Rejected.
- Forcing very small Parquet scans to sequential reader:
  - Restored file-order duplicate payload output, but regressed engine time for dense and duplicate joins.
  - Rejected as a broad heuristic.

### Planning And DAG Execution

- Added explicit `LogicalPlan` and `LogicalScan` descriptors.
- Converted scan, aggregate, and join engine plans into logical plans.
- Changed explain `PhysicalPlanNode` into an enum-backed declarative node while preserving existing explain output.
- Added `Partitioning`, `Distribution`, `ExchangeKind`, `PhysicalJoinStrategy`, `StagePlan`, `TaskPlan`, and `ExecutionGraphPlan`.
- Added typed `PhysicalExecutionConfig` payloads to declarative physical nodes.
- Added `PhysicalPlanner` for lowering `LogicalPlan` into declarative physical nodes.
- Added optional required-distribution planning that inserts hash repartition exchanges for partitioned joins and gather exchanges for global aggregate/sort/limit.
- Added `StagePlanner` for splitting declarative physical plans into stages at `Exchange` boundaries.
- Added `TaskPlanner` to expand stages into scan-fragment and shuffle-partition tasks.
- Added local lowering from declarative physical nodes back into executable local physical plans for scan/filter/projection/sort/limit/distinct/hash join/partitioned hash join/sort-merge join.
- Added local task execution for scan-fragment task graphs by narrowing scan nodes to task input fragments.
- Added dependency-ordered local execution graph scheduling based on stage dependencies.
- Added local gather and hash repartition shuffle materialization.
- Changed local shuffle storage from in-memory maps to temporary Arrow IPC shuffle files with cleanup on drop.
- Added `MemoryExec` and `IpcExec` so exchange consumers can use the normal physical executor path.
- Changed exchange consumers from loading shuffle partitions into memory to streaming Arrow IPC shuffle files through `IpcExec`.
- Changed local shuffle writing from one IPC file per output batch to one IPC file per task-output partition write.
- Added local shuffle writer rolling with a `64 MiB` target IPC file size.
- Added configurable local execution options for shuffle writer target file size.
- Added local execution graph metrics for stage/task counts, task output rows/batches, shuffle read/write files, batches, rows, bytes, and timing.
- Added per-stage local execution graph metrics so bottlenecks can be attributed by stage.
- Added tests for logical/physical descriptor exposure, exchange/stage planning, task planning, local graph execution, hash repartition join execution, and shuffle file rolling.
- Added task-planning invariant coverage to ensure shuffle consumer tasks read shuffle partitions rather than scan fragments directly.

## Current Performance Snapshot

- With `16 KiB` Dodam batch size and default COPY Parquet file output:
  - `filter_limit`: Dodam CLI about `4.4 ms`, DuckDB CLI about `13.4 ms`
  - `global_aggregate`: Dodam CLI about `12.8 ms`, DuckDB CLI about `13.9 ms`
  - `grouped_aggregate`: Dodam CLI about `11.3 ms`, DuckDB CLI about `14.8 ms`
  - `INNER JOIN`: Dodam engine about `5.0 ms`, Dodam CLI about `21.1 ms`, DuckDB CLI about `23.2 ms`
  - `join_grouped_aggregate_ordered`: Dodam engine about `6.8 ms`, Dodam CLI about `16.3 ms`, DuckDB CLI about `17.7 ms`
  - `LEFT OUTER JOIN`: Dodam CLI about `21.2 ms`, DuckDB CLI about `22.9 ms`
  - `RIGHT OUTER JOIN`: Dodam CLI about `22.0 ms`, DuckDB CLI about `23.2 ms`
  - `FULL OUTER JOIN`: Dodam CLI about `21.9 ms`, DuckDB CLI about `23.6 ms`
  - `LEFT SEMI JOIN`: Dodam CLI about `12.0 ms`, DuckDB CLI about `16.6 ms`
  - `INNER JOIN non-dense Int32 key`: Dodam CLI about `23.3 ms`, DuckDB CLI about `24.0 ms`
  - `INNER JOIN multi-key`: Dodam CLI about `23.8 ms`, DuckDB CLI about `24.0 ms`; effectively tied.
  - `INNER JOIN string key`: Dodam CLI about `20.8 ms`, DuckDB CLI about `21.8 ms`
  - `FULL OUTER JOIN unmatched-heavy`: Dodam CLI about `14.0 ms`, DuckDB CLI about `19.6 ms`
  - `INNER JOIN wide output`: Dodam CLI about `28.9 ms`, DuckDB CLI about `39.1 ms`
  - `LEFT SEMI JOIN duplicate build`: Dodam CLI about `12.5 ms`, DuckDB CLI about `18.1 ms`
  - `LEFT SEMI JOIN duplicate build small row groups`: Dodam CLI about `14.5 ms`, DuckDB CLI about `19.1 ms`
  - `INNER JOIN duplicate build`: Dodam engine about `9.0 ms`, Dodam CLI about `36.7 ms`, DuckDB CLI about `28.2 ms`
  - `INNER JOIN duplicate build narrow fact`: Dodam CLI about `40.1 ms`, DuckDB CLI about `27.9 ms`
  - `INNER JOIN duplicate build wide-pruned fact`: Dodam CLI about `40.7 ms`, DuckDB CLI about `28.2 ms`
  - `INNER JOIN duplicate build small row groups`: Dodam engine about `10.8 ms`, Dodam CLI about `42.3 ms`, DuckDB CLI about `30.3 ms`
- Current interpretation:
  - Under Parquet output, Dodam is faster than DuckDB for the currently supported scan/filter, aggregate, grouped aggregate, dense joins, semi joins, outer joins, string-key joins, and wide-output join benchmarks.
  - Multi-key inner join is effectively tied at CLI level.
  - The clear supported-feature gap is duplicate-build inner join with fanout output: Dodam's engine is reasonable, but CLI Parquet output is still slower than DuckDB.
  - Before the `8 Ki` writer/page change, profiling `inner_join_duplicate_build` showed `copy_profile_sink write` about `30 ms` out of about `52 ms` total for `524,288` output rows; engine join build was about `4.4 ms` and join materialization about `5.6 ms`.
  - The same duplicate fanout output produces a larger file than DuckDB on the profiled case: Dodam about `4.0 MiB`, DuckDB about `3.6 MiB`, both with `524,288` rows and `8` row groups.
  - RIGHT/FULL improved mostly because dense `Int32` bitmap matched-build tracking removed repeated `HashSet<BuildRowRef>` work.
  - Non-dense inner join is now mostly limited by probe/materialize and Parquet decode rather than CSV serialization.
  - Scan decode metrics are cumulative operator time, not wall-clock time.

## Next Priority

### 1. Fix Duplicate-Build Inner Join Fanout Output

- Investigate why ArrowWriter is slower and produces larger `id` column chunks than DuckDB for duplicate fanout output even with matching row count, row-group count, SNAPPY compression, and dictionary disabled.
- Consider a Parquet-oriented direct sink path for common join output shapes so duplicate fanout rows can be written in larger, writer-friendly batches without extra RecordBatch materialization churn.
- Revisit duplicate inner join output batching:
  - row ordering and repeated value patterns
  - batch size handed to ArrowWriter
  - row group boundaries
  - whether fanout output can preserve compression-friendly runs
- Compare ArrowWriter encoding options and statistics settings against DuckDB output metadata for the duplicate fanout case.
- Current best retained writer settings for the duplicate gap:
  - row-group/chunk statistics instead of page statistics
  - `8 Ki` write batch and data page row count
  - `DELTA_BINARY_PACKED` encoding for `id`/`f.id`
- Treat this as the top DuckDB-comparison performance gap because most other supported Parquet-output benchmarks are already faster or tied.

### 2. Add Focused COPY And Parquet Output Tests

- Add focused COPY tests:
  - CSV escaping
  - CSV header option
  - Parquet COPY output can be read back
  - Parquet COPY options: compression, dictionary, row-group size
  - numeric/null output
  - unsupported column type fallback
  - COPY join output equivalence with SELECT

### 3. Fix Local DAG Execution Quality

- Reduce duplicated local DAG task work.
- Ensure exchange consumer stages read only materialized shuffle inputs and do not accidentally keep executing producer subtrees.
- Expose stage-level metrics in CLI profiling and/or `EXPLAIN`.
- Use the new per-stage metrics to identify excessive task counts in row-group-heavy plans.
- Add configurable local execution options to CLI/profile entry points when DAG execution becomes user-facing.

### 4. Finish Declarative Plan Coverage

- Move `EXPLAIN` for scan/join/aggregate/sort/limit/copy fully onto declarative plan representation.
- Add aggregate execution lowering to declarative physical nodes.
- Split aggregation into partial/final physical operators.
- Add grouped aggregate distribution planning:
  - partial aggregate before shuffle
  - hash repartition by group keys
  - final aggregate after shuffle
- Add sort-merge join ordering requirements to physical planning.
- Preserve current fast local execution paths while moving planning state out of ad hoc execution objects.

### 5. Serialization Boundary For Future Distributed Execution

- Make plan nodes serializable.
- Make expressions, projections, aggregate expressions, join keys, table scan sources, and exchange descriptors serializable.
- Define stable task descriptors and stable partition/shuffle descriptors.
- Keep executor-facing APIs free from Rust trait objects crossing process boundaries.
- Add round-trip tests for serialized logical plan, physical plan, stage plan, and task plan.

### 6. More Realistic Large-Data Behavior

- Add larger benchmark datasets:
  - non-dense join keys
  - duplicate build keys
  - unmatched-heavy outer joins
  - multi-column joins
  - wider projected output
  - larger fact/dimension scale factors
- Add shuffle stress tests with many fragments and multiple shuffle partitions.
- Add external sort for large `ORDER BY`.
- Add partial/final aggregation spill support.
- Add grouped aggregation spill support.
- Add sort-merge join input spill/sort support.

### 7. Join Improvements Still Worth Doing

- Generalize matched-build tracking:
  - dense `Int64` bitmap tracking
  - bitmap/index tracking for unique non-dense keys
  - keep `HashSet<BuildRowRef>` for duplicate-key and generic row-key joins
- Restore direct sink/streaming optimization for outer joins with projected output by applying final projection in the sink path.
- Expand join benchmark coverage beyond the current dense-key benchmark.
- Keep anti join and cross join out for now unless query patterns justify them.
- Add non-equi join only after the planner can deliberately choose nested-loop/range join strategies.

### 8. SQL Completeness

- Add SQL tests for more alias and expression combinations.
- Extend richer Parquet type support beyond residual filtering:
  - explicit behavior for nested/list/struct projection in SQL and CLI output
  - casts and richer date/time literal forms
- Add more expression parser support only after parser structure remains clean.

### 7. Storage/Catalog Next Steps

- Add catalog table removal/rename commands.
- Expand directory partition pruning beyond positive `=` and `IN`:
  - range predicates for ordered partition values
  - `IS NULL`/`IS NOT NULL`
  - safe handling of `OR`
- Avoid reading a physical Parquet column when a query projects only partition columns.
- Add a real CSV scan implementation behind `ScanExec`.
- Add JSON and Arrow IPC scan implementations after CSV.
- Improve statistics:
  - column-level compressed bytes
  - null counts
  - row-group min/max exposure through catalog APIs
  - approximate NDV later
- Add Iceberg metadata/snapshot planning from manifests instead of local directory scans.
- Revisit S3/object storage later with:
  - range-read capable object store
  - remote metadata cache
  - listing abstraction
  - retry/backoff and request metrics

## Validation Checklist

- Run before considering the current branch healthy:
  - `cargo fmt`
  - `cargo test`
  - `cargo clippy --all-targets --all-features -- -D warnings`
- Run before claiming performance:
  - release build
  - Dodam CLI benchmark
  - DuckDB CLI benchmark
  - engine-level criterion benchmark
  - profile with `DODAM_PROFILE_COPY=1` for COPY-heavy queries
