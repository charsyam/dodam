use super::*;

pub(super) async fn try_execute_set_operation_sql_to_sink(
    engine: &DodamEngine,
    sql: &str,
    batch_size: usize,
    sink: &mut dyn RecordBatchSink,
) -> Result<Option<ScanPlanMetrics>> {
    let dialect = GenericDialect {};
    let statements = Parser::parse_sql(&dialect, sql)
        .map_err(|error| DodamError::UnsupportedSql(error.to_string()))?;
    let [Statement::Query(query)] = statements.as_slice() else {
        return Ok(None);
    };
    if !query_contains_set_operation(query.body.as_ref()) {
        return Ok(None);
    }
    if query.fetch.is_some() || !query.locks.is_empty() {
        return Ok(None);
    }
    let order_by = parse_order_by(query, &[], &[], None)?;
    let limit = parse_limit(query)?;
    let offset = parse_offset(query)?;

    if order_by.is_some() || limit.is_some() || offset != 0 {
        if try_write_same_source_union_all_ordered_desc_to_sink(
            engine,
            query.body.as_ref(),
            batch_size,
            order_by.as_ref(),
            limit,
            offset,
            sink,
        )
        .await?
        {
            sink.finish()?;
            return Ok(Some(ScanPlanMetrics::default()));
        }
        return Ok(None);
    }

    if let Some(shared) = plan_same_source_union_all_scan(query.body.as_ref())? {
        if try_write_same_source_union_all_primitive_to_sink(engine, &shared, batch_size, sink)? {
            sink.finish()?;
            return Ok(Some(ScanPlanMetrics::default()));
        }
        let mut stream = engine
            .scan_parquet_batches(
                shared.path,
                batch_size,
                None,
                shared.projection,
                Some(shared.filter),
            )
            .await?;
        write_same_source_union_all_scan_stream_to_sink(&mut stream, &shared.aliases, sink)?;
        sink.finish()?;
        return Ok(Some(stream.into_scan_plan_metrics()));
    }
    if let Some(shared) = plan_same_source_union_all_filter_scan(query.body.as_ref())? {
        let mut stream = engine
            .scan_parquet_batches(
                shared.path.clone(),
                batch_size,
                None,
                shared.scan_projection.clone(),
                Some(shared.prefilter.clone()),
            )
            .await?;
        write_same_source_union_all_filter_scan_stream_to_sink(&mut stream, &shared, sink)?;
        sink.finish()?;
        return Ok(Some(stream.into_scan_plan_metrics()));
    }

    let mut state = UnionAllSinkState::default();
    Box::pin(write_union_all_set_expr_to_sink(
        engine,
        query.body.as_ref(),
        batch_size,
        sink,
        &mut state,
    ))
    .await?;
    sink.finish()?;
    Ok(Some(ScanPlanMetrics::default()))
}

fn write_same_source_union_all_scan_stream_to_sink(
    stream: &mut SendableBatchStream,
    aliases: &[(String, String)],
    sink: &mut dyn RecordBatchSink,
) -> Result<()> {
    let target_rows = same_source_union_scan_write_coalesce_rows();
    if target_rows == 0 {
        for batch in stream.by_ref() {
            let batch = batch?;
            if aliases.is_empty() {
                sink.write_batch(&batch)?;
            } else {
                for renamed in rename_output_batches(vec![batch], aliases)? {
                    sink.write_batch(&renamed)?;
                }
            }
        }
        return Ok(());
    }
    let mut buffered = Vec::<RecordBatch>::new();
    let mut buffered_rows = 0usize;
    for batch in stream.by_ref() {
        let batch = batch?;
        let batches = if aliases.is_empty() {
            vec![batch]
        } else {
            rename_output_batches(vec![batch], aliases)?
        };
        for batch in batches {
            if batch.num_rows() == 0 {
                continue;
            }
            buffered_rows += batch.num_rows();
            buffered.push(batch);
            if buffered_rows >= target_rows {
                flush_record_batch_buffer(&mut buffered, &mut buffered_rows, sink)?;
            }
        }
    }
    flush_record_batch_buffer(&mut buffered, &mut buffered_rows, sink)?;
    Ok(())
}

fn write_same_source_union_all_filter_scan_stream_to_sink(
    stream: &mut SendableBatchStream,
    shared: &SameSourceUnionAllFilterScan,
    sink: &mut dyn RecordBatchSink,
) -> Result<()> {
    for batch in stream.by_ref() {
        let batch = batch?;
        let mut output = Vec::new();
        append_same_source_union_all_filter_batches(&mut output, &batch, shared)?;
        for batch in output {
            sink.write_batch(&batch)?;
        }
    }
    Ok(())
}

pub(super) fn append_same_source_union_all_filter_batches(
    output: &mut Vec<RecordBatch>,
    batch: &RecordBatch,
    shared: &SameSourceUnionAllFilterScan,
) -> Result<()> {
    for filter in &shared.filters {
        let filtered = filter_batch(batch.clone(), filter)?;
        if filtered.num_rows() == 0 {
            continue;
        }
        let filtered = apply_output_projection(vec![filtered], &shared.projection)?;
        let filtered = rename_output_batches(filtered, &shared.aliases)?;
        output.extend(filtered);
    }
    Ok(())
}

fn same_source_union_scan_write_coalesce_rows() -> usize {
    std::env::var("DODAM_UNION_SCAN_WRITE_COALESCE_ROWS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0)
}

fn flush_record_batch_buffer(
    buffered: &mut Vec<RecordBatch>,
    buffered_rows: &mut usize,
    sink: &mut dyn RecordBatchSink,
) -> Result<()> {
    if buffered.is_empty() {
        return Ok(());
    }
    let batch = if buffered.len() == 1 {
        buffered.pop().expect("checked non-empty buffer")
    } else {
        let schema = buffered[0].schema();
        concat_batches(&schema, buffered.iter())?
    };
    buffered.clear();
    *buffered_rows = 0;
    sink.write_batch(&batch)?;
    Ok(())
}

fn try_write_same_source_union_all_primitive_to_sink(
    engine: &DodamEngine,
    shared: &SameSourceUnionAllScan,
    batch_size: usize,
    sink: &mut dyn RecordBatchSink,
) -> Result<bool> {
    if !same_source_union_primitive_sink_enabled() {
        return Ok(false);
    }
    let Projection::Columns(output_columns) = &shared.projection else {
        return Ok(false);
    };
    let Expr::InList {
        column: filter_column,
        values,
        negated,
        ..
    } = shared.filter.expr()
    else {
        return Ok(false);
    };
    if *negated {
        return Ok(false);
    }
    let mut scan_projection = shared.projection.clone();
    add_projection_columns(&mut scan_projection, vec![filter_column.clone()]);
    let Projection::Columns(scan_columns) = &scan_projection else {
        return Ok(false);
    };
    let Some(column_types) =
        engine.parquet_direct_primitive_column_types(&shared.path, scan_columns)?
    else {
        return Ok(false);
    };
    if !column_types.iter().all(|column_type| {
        matches!(
            column_type,
            DirectPrimitiveColumnType::I32 | DirectPrimitiveColumnType::I64
        )
    }) {
        return Ok(false);
    }
    let Some(filter_index) = scan_columns
        .iter()
        .position(|column| column == filter_column)
    else {
        return Ok(false);
    };
    let filter_values = match column_types[filter_index] {
        DirectPrimitiveColumnType::I32 => PrimitiveFilterValues::I32(
            values
                .iter()
                .map(|value| value.as_i32(filter_column))
                .collect::<Result<Vec<_>>>()?,
        ),
        DirectPrimitiveColumnType::I64 => PrimitiveFilterValues::I64(
            values
                .iter()
                .map(|value| value.as_i64(filter_column))
                .collect::<Result<Vec<_>>>()?,
        ),
        DirectPrimitiveColumnType::Date32
        | DirectPrimitiveColumnType::Decimal128Int64 { .. }
        | DirectPrimitiveColumnType::Decimal128Int64Raw { .. } => return Ok(false),
    };
    let row_group_count = engine.parquet_row_group_count(&shared.path)?;
    if row_group_count == 0 {
        return Ok(true);
    }
    let specs = scan_columns
        .iter()
        .zip(column_types.iter())
        .map(|(name, column_type)| DirectPrimitiveColumnSpec {
            name,
            column_type: *column_type,
        })
        .collect::<Vec<_>>();
    let chunk_size = same_source_union_primitive_chunk_size(row_group_count);
    let row_groups = (0..row_group_count).collect::<Vec<_>>();
    let profile = ordered_sink_profile_enabled();
    let total_started = profile.then(Instant::now);
    let mut metrics = DirectPrimitiveColumnScanMetrics::default();
    let mut concat_elapsed = Duration::ZERO;
    let mut write_elapsed = Duration::ZERO;
    let mut batches_written = 0usize;
    let mut rows_written = 0usize;
    for chunk in row_groups.chunks(chunk_size) {
        let scan_results = std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(chunk.len());
            let scan_columns_ref = scan_columns;
            let column_types_ref = &column_types;
            let filter_values_ref = &filter_values;
            for (position, row_group) in chunk.iter().copied().enumerate() {
                let engine = engine.clone();
                let path = shared.path.clone();
                let specs = specs.clone();
                handles.push(scope.spawn(move || {
                    let mut row_group_batches = Vec::new();
                    let metrics = engine.scan_parquet_primitive_columns_view(
                        &path,
                        batch_size,
                        &[row_group],
                        &specs,
                        |view| {
                            if let Some(batch) = primitive_ordered_selected_batch(
                                view,
                                scan_columns_ref,
                                column_types_ref,
                                filter_index,
                                filter_values_ref,
                            )? && batch.num_rows() > 0
                            {
                                row_group_batches.push(batch);
                            }
                            Ok(())
                        },
                    )?;
                    Ok::<_, DodamError>((position, row_group_batches, metrics))
                }));
            }
            let mut results = Vec::with_capacity(handles.len());
            for handle in handles {
                match handle.join() {
                    Ok(result) => results.push(result?),
                    Err(_) => {
                        return Err(DodamError::UnsupportedSql(
                            "same-source union primitive scan worker panicked".to_string(),
                        ));
                    }
                }
            }
            Ok::<_, DodamError>(results)
        })?;
        let mut scan_results = scan_results;
        scan_results.sort_by_key(|(position, _, _)| *position);
        let mut chunk_batches = Vec::new();
        for (_, row_group_batches, row_group_metrics) in scan_results {
            if let Some(row_group_metrics) = row_group_metrics {
                metrics.merge_from(row_group_metrics);
            }
            chunk_batches.extend(row_group_batches);
        }
        if chunk_batches.is_empty() {
            continue;
        }
        let concat_started = profile.then(Instant::now);
        let batch = PrimitiveBatch::concat(chunk_batches)?;
        if let Some(started) = concat_started {
            concat_elapsed += started.elapsed();
        }
        if batch.is_empty() {
            continue;
        }
        rows_written += batch.num_rows();
        let write_started = profile.then(Instant::now);
        write_same_source_primitive_batch_to_sink(batch, &scan_projection, shared, sink)?;
        if let Some(started) = write_started {
            write_elapsed += started.elapsed();
            batches_written += 1;
        }
    }
    print_same_source_union_primitive_sink_profile(
        total_started,
        &metrics,
        concat_elapsed,
        write_elapsed,
        batches_written,
        rows_written,
        output_columns.len(),
    );
    Ok(true)
}

fn same_source_union_primitive_sink_enabled() -> bool {
    if std::env::var("DODAM_DISABLE_UNION_PRIMITIVE_SINK")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
    {
        return false;
    }
    std::env::var("DODAM_ENABLE_UNION_PRIMITIVE_SINK")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

pub(super) fn same_source_union_primitive_chunk_size(row_group_count: usize) -> usize {
    if let Some(value) = std::env::var("DODAM_UNION_PRIMITIVE_ROW_GROUP_CHUNK_SIZE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
    {
        return value;
    }
    let workers = std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(4)
        .min(row_group_count.max(1));
    workers.min(24).max(1)
}

fn print_same_source_union_primitive_sink_profile(
    total_started: Option<Instant>,
    metrics: &DirectPrimitiveColumnScanMetrics,
    concat_elapsed: Duration,
    write_elapsed: Duration,
    batches: usize,
    rows: usize,
    output_columns: usize,
) {
    let Some(total_started) = total_started else {
        return;
    };
    let column_read = metrics
        .column_read_nanos
        .iter()
        .map(|nanos| format!("{:.3}", (*nanos as f64) / 1_000_000.0))
        .collect::<Vec<_>>()
        .join(",");
    eprintln!(
        "[dodam:same-source-union-primitive-sink-profile] total={}us read={:.3}ms consume={:.3}ms concat={}us write={}us row_groups={} batches={} rows={} output_columns={} column_read_ms=[{}]",
        total_started.elapsed().as_micros(),
        (metrics.read_nanos as f64) / 1_000_000.0,
        (metrics.consume_nanos as f64) / 1_000_000.0,
        concat_elapsed.as_micros(),
        write_elapsed.as_micros(),
        metrics.row_groups,
        batches,
        rows,
        output_columns,
        column_read,
    );
}

async fn try_write_same_source_union_all_ordered_desc_to_sink(
    engine: &DodamEngine,
    expr: &SetExpr,
    batch_size: usize,
    order_by: Option<&SortKey>,
    limit: Option<usize>,
    offset: usize,
    sink: &mut dyn RecordBatchSink,
) -> Result<bool> {
    let (Some(order_by), Some(limit)) = (order_by, limit) else {
        return Ok(false);
    };
    if offset != 0 || limit == 0 {
        return Ok(false);
    }
    let [sort] = order_by.expressions.as_slice() else {
        return Ok(false);
    };
    if !sort.descending || sort.nulls_first {
        return Ok(false);
    }
    let Some(shared) = plan_same_source_union_all_scan(expr)? else {
        return Ok(false);
    };
    let row_groups_monotonic = engine
        .parquet_row_groups_monotonic_by_column(shared.path.clone(), &sort.column)
        .await?;
    let row_group_count = engine.parquet_row_group_count(&shared.path)?;
    if row_group_count == 0 {
        return Ok(true);
    }

    let mut scan_projection = shared.projection.clone();
    add_projection_columns(&mut scan_projection, shared.filter.referenced_columns());
    for sort in &order_by.expressions {
        add_projection_columns(&mut scan_projection, vec![sort.column.clone()]);
    }

    if !row_groups_monotonic {
        return try_write_same_source_union_all_streaming_primitive_topk_to_sink(
            engine,
            &shared,
            batch_size,
            &scan_projection,
            &sort.column,
            limit,
            row_group_count,
            sink,
        );
    }

    if try_write_same_source_union_all_ordered_desc_primitive_to_sink(
        engine,
        &shared,
        batch_size,
        &scan_projection,
        &sort.column,
        limit,
        row_group_count,
        sink,
    )? {
        return Ok(true);
    }

    if try_write_same_source_union_all_ordered_desc_chunked_to_sink(
        engine,
        &shared,
        batch_size,
        &scan_projection,
        order_by,
        limit,
        sink,
        row_group_count,
    )
    .await?
    {
        return Ok(true);
    }

    let row_group_batch_size = row_group_ordered_sort_batch_size(batch_size);
    let mut written = 0usize;
    let profile = ordered_sink_profile_enabled();
    let total_started = profile.then(Instant::now);
    let mut scan_sort_elapsed = Duration::ZERO;
    let mut sink_write_elapsed = Duration::ZERO;
    let mut row_groups_scanned = 0usize;
    let mut batches_written = 0usize;
    for row_group in (0..row_group_count).rev() {
        let scan_sort_started = profile.then(Instant::now);
        let sorted = scan_filter_sort_ordered_row_group(
            engine,
            shared.path.clone(),
            row_group_batch_size,
            scan_projection.clone(),
            row_group,
            &shared.filter,
            order_by,
            &sort.column,
        )
        .await?;
        if let Some(started) = scan_sort_started {
            scan_sort_elapsed += started.elapsed();
            row_groups_scanned += 1;
        }
        for batch in sorted {
            if written >= limit {
                print_ordered_sink_profile(
                    total_started,
                    scan_sort_elapsed,
                    sink_write_elapsed,
                    row_groups_scanned,
                    batches_written,
                    written,
                );
                return Ok(true);
            }
            let remaining = limit - written;
            let batch = if batch.num_rows() > remaining {
                batch.slice(0, remaining)
            } else {
                batch
            };
            written += batch.num_rows();
            let sink_write_started = profile.then(Instant::now);
            write_same_source_batch_to_sink(batch, &scan_projection, &shared, sink)?;
            if let Some(started) = sink_write_started {
                sink_write_elapsed += started.elapsed();
                batches_written += 1;
            }
            if written >= limit {
                print_ordered_sink_profile(
                    total_started,
                    scan_sort_elapsed,
                    sink_write_elapsed,
                    row_groups_scanned,
                    batches_written,
                    written,
                );
                return Ok(true);
            }
        }
    }
    print_ordered_sink_profile(
        total_started,
        scan_sort_elapsed,
        sink_write_elapsed,
        row_groups_scanned,
        batches_written,
        written,
    );
    Ok(true)
}

async fn try_write_same_source_union_all_ordered_desc_chunked_to_sink(
    engine: &DodamEngine,
    shared: &SameSourceUnionAllScan,
    batch_size: usize,
    scan_projection: &Projection,
    order_by: &SortKey,
    limit: usize,
    sink: &mut dyn RecordBatchSink,
    row_group_count: usize,
) -> Result<bool> {
    let chunk_size = ordered_sink_row_group_chunk_size(limit, row_group_count);
    if chunk_size <= 1 {
        return Ok(false);
    }
    let mut written = 0usize;
    let profile = ordered_sink_profile_enabled();
    let total_started = profile.then(Instant::now);
    let mut scan_elapsed = Duration::ZERO;
    let mut filter_elapsed = Duration::ZERO;
    let mut sort_elapsed = Duration::ZERO;
    let mut write_elapsed = Duration::ZERO;
    let mut scan_profile = RowGroupBatchScanProfile::default();
    let mut chunks = 0usize;
    let mut batches_written = 0usize;
    let row_groups = (0..row_group_count).rev().collect::<Vec<_>>();
    for chunk in row_groups.chunks(chunk_size) {
        if written >= limit {
            print_ordered_chunked_sink_profile(
                total_started,
                scan_elapsed,
                filter_elapsed,
                sort_elapsed,
                write_elapsed,
                chunks,
                batches_written,
                written,
                chunk_size,
                &scan_profile,
            );
            return Ok(true);
        }
        chunks += 1;
        let scan_started = profile.then(Instant::now);
        let use_row_filter = ordered_sink_row_filter_enabled();
        let (batches, profile_metrics) = if use_row_filter {
            engine
                .scan_parquet_row_group_batches_filtered_profiled(
                    shared.path.clone(),
                    batch_size,
                    scan_projection.clone(),
                    chunk.to_vec(),
                    vec![shared.filter.expr().clone()],
                )
                .await?
        } else {
            engine
                .scan_parquet_row_group_batches_profiled(
                    shared.path.clone(),
                    batch_size,
                    scan_projection.clone(),
                    chunk.to_vec(),
                )
                .await?
        };
        scan_profile.merge_from(profile_metrics);
        if let Some(started) = scan_started {
            scan_elapsed += started.elapsed();
        }
        let filter_started = profile.then(Instant::now);
        let sort_column = order_by
            .expressions
            .first()
            .map(|sort| sort.column.as_str())
            .unwrap_or_default();
        let filtered = if use_row_filter {
            batches
                .into_iter()
                .filter(|batch| batch.num_rows() > 0)
                .collect::<Vec<_>>()
        } else {
            let mut filtered = Vec::new();
            for batch in batches {
                let batch = filter_batch(batch, &shared.filter)?;
                if batch.num_rows() > 0 {
                    filtered.push(batch);
                }
            }
            filtered
        };
        if let Some(started) = filter_started {
            filter_elapsed += started.elapsed();
        }
        if filtered.is_empty() {
            continue;
        }
        let sort_started = profile.then(Instant::now);
        let sorted = match reverse_ascending_primitive_runs_if_ordered(&filtered, sort_column)? {
            Some(sorted) => sorted,
            None => apply_output_order_limit(filtered, Some(order_by), None, 0)?,
        };
        if let Some(started) = sort_started {
            sort_elapsed += started.elapsed();
        }
        for batch in sorted {
            if written >= limit {
                print_ordered_chunked_sink_profile(
                    total_started,
                    scan_elapsed,
                    filter_elapsed,
                    sort_elapsed,
                    write_elapsed,
                    chunks,
                    batches_written,
                    written,
                    chunk_size,
                    &scan_profile,
                );
                return Ok(true);
            }
            let remaining = limit - written;
            let batch = if batch.num_rows() > remaining {
                batch.slice(0, remaining)
            } else {
                batch
            };
            written += batch.num_rows();
            let write_started = profile.then(Instant::now);
            write_same_source_batch_to_sink(batch, scan_projection, shared, sink)?;
            if let Some(started) = write_started {
                write_elapsed += started.elapsed();
                batches_written += 1;
            }
        }
    }
    print_ordered_chunked_sink_profile(
        total_started,
        scan_elapsed,
        filter_elapsed,
        sort_elapsed,
        write_elapsed,
        chunks,
        batches_written,
        written,
        chunk_size,
        &scan_profile,
    );
    Ok(true)
}

fn reverse_ascending_primitive_runs_if_ordered(
    batches: &[RecordBatch],
    sort_column: &str,
) -> Result<Option<Vec<RecordBatch>>> {
    if sort_column.is_empty() || batches.is_empty() {
        return Ok(None);
    }
    if !batches.iter().all(record_batch_supports_fast_reverse) {
        return Ok(None);
    }
    let mut runs = Vec::<(usize, usize)>::new();
    let mut run_start = 0usize;
    let mut previous_last: Option<i128> = None;
    for (index, batch) in batches.iter().enumerate() {
        if batch.num_rows() == 0 {
            continue;
        }
        let column_index = output_batch_column_index(batch, sort_column)?;
        let Some((first, last)) = numeric_column_ascending_bounds(batch.column(column_index))?
        else {
            return Ok(None);
        };
        if previous_last.is_some_and(|previous| previous > first) {
            if run_start == index {
                return Ok(None);
            }
            runs.push((run_start, index));
            run_start = index;
        }
        previous_last = Some(last);
    }
    if run_start < batches.len() {
        runs.push((run_start, batches.len()));
    }
    if runs.is_empty() {
        return Ok(Some(Vec::new()));
    }
    let mut output = Vec::new();
    for (start, end) in runs {
        for batch in batches[start..end].iter().rev() {
            let Some(reversed) = reverse_primitive_record_batch_rows(batch)? else {
                return Ok(None);
            };
            if reversed.num_rows() > 0 {
                output.push(reversed);
            }
        }
    }
    Ok(Some(output))
}

fn ordered_sink_row_filter_enabled() -> bool {
    std::env::var("DODAM_ORDERED_SINK_ROW_FILTER")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

fn ordered_sink_row_group_chunk_size(limit: usize, row_group_count: usize) -> usize {
    std::env::var("DODAM_ORDERED_SINK_ROW_GROUP_CHUNK_SIZE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or_else(|| {
            if limit <= reverse_row_group_topk_max_limit_rows() {
                1
            } else if row_group_count >= 96 {
                8
            } else if row_group_count >= 24 {
                4
            } else if row_group_count >= 8 {
                2
            } else {
                1
            }
        })
}

fn ordered_primitive_parallel_chunk_enabled(chunk_len: usize, limit: usize) -> bool {
    if chunk_len <= 1 || limit <= ordered_primitive_sink_auto_max_limit_rows() {
        return false;
    }
    !std::env::var("DODAM_DISABLE_ORDERED_PRIMITIVE_PARALLEL_CHUNK")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

fn ordered_primitive_sink_row_group_chunk_size(limit: usize, row_group_count: usize) -> usize {
    if let Some(value) = std::env::var("DODAM_ORDERED_SINK_ROW_GROUP_CHUNK_SIZE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
    {
        return value;
    }
    choose_ordered_primitive_row_group_chunk(OrderedPrimitiveChunkCostInput {
        limit,
        row_groups: row_group_count,
        available_parallelism: std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(4),
        small_limit_rows: ordered_primitive_sink_auto_max_limit_rows(),
        max_workers: 24,
    })
}

#[allow(clippy::too_many_arguments)]
fn print_ordered_chunked_sink_profile(
    total_started: Option<Instant>,
    scan_elapsed: Duration,
    filter_elapsed: Duration,
    sort_elapsed: Duration,
    write_elapsed: Duration,
    chunks: usize,
    batches: usize,
    rows: usize,
    chunk_size: usize,
    scan_profile: &RowGroupBatchScanProfile,
) {
    let Some(total_started) = total_started else {
        return;
    };
    eprintln!(
        "[dodam:ordered-chunked-sink-profile] total={}us scan={}us filter={}us sort={}us write={}us chunks={} chunk_size={} batches={} rows={} metadata={}us planning={}us next={}us next_avg={}us next_p95={}us next_max={}us next_calls={} eof_calls={} scan_batches={} scan_rows={} row_groups={}/{} projected_columns={} bytes={}/{} zero_batches={}",
        total_started.elapsed().as_micros(),
        scan_elapsed.as_micros(),
        filter_elapsed.as_micros(),
        sort_elapsed.as_micros(),
        write_elapsed.as_micros(),
        chunks,
        chunk_size,
        batches,
        rows,
        scan_profile.metadata_nanos / 1_000,
        scan_profile.planning_nanos / 1_000,
        scan_profile.next_nanos / 1_000,
        if scan_profile.next_calls == 0 {
            0
        } else {
            (scan_profile.next_nanos / scan_profile.next_calls as u64) / 1_000
        },
        scan_profile.p95_next_nanos / 1_000,
        scan_profile.max_next_nanos / 1_000,
        scan_profile.next_calls,
        scan_profile.eof_calls,
        scan_profile.output_batches,
        scan_profile.output_rows,
        scan_profile.row_groups_scanned,
        scan_profile.row_groups_total,
        scan_profile.projected_columns,
        scan_profile.compressed_bytes_scanned,
        scan_profile.compressed_bytes_total,
        scan_profile.zero_row_batches,
    );
}

fn try_write_same_source_union_all_ordered_desc_primitive_to_sink(
    engine: &DodamEngine,
    shared: &SameSourceUnionAllScan,
    batch_size: usize,
    scan_projection: &Projection,
    sort_column: &str,
    limit: usize,
    row_group_count: usize,
    sink: &mut dyn RecordBatchSink,
) -> Result<bool> {
    let primitive_sink_enabled = ordered_primitive_sink_enabled(limit);
    let fused_page_decoder_enabled = ordered_primitive_fused_page_decoder_enabled();
    if !primitive_sink_enabled && !fused_page_decoder_enabled {
        return Ok(false);
    }
    let Projection::Columns(columns) = scan_projection else {
        return Ok(false);
    };
    let Expr::InList {
        column: filter_column,
        values,
        negated,
        ..
    } = shared.filter.expr()
    else {
        return Ok(false);
    };
    if *negated || !columns.iter().any(|column| column == sort_column) {
        return Ok(false);
    }
    let Some(column_types) = engine.parquet_direct_primitive_column_types(&shared.path, columns)?
    else {
        return Ok(false);
    };
    if !column_types.iter().all(|column_type| {
        matches!(
            column_type,
            DirectPrimitiveColumnType::I32 | DirectPrimitiveColumnType::I64
        )
    }) {
        return Ok(false);
    }
    let Some(filter_index) = columns.iter().position(|column| column == filter_column) else {
        return Ok(false);
    };
    let filter_values = match column_types[filter_index] {
        DirectPrimitiveColumnType::I32 => PrimitiveFilterValues::I32(
            values
                .iter()
                .map(|value| value.as_i32(filter_column))
                .collect::<Result<Vec<_>>>()?,
        ),
        DirectPrimitiveColumnType::I64 => PrimitiveFilterValues::I64(
            values
                .iter()
                .map(|value| value.as_i64(filter_column))
                .collect::<Result<Vec<_>>>()?,
        ),
        DirectPrimitiveColumnType::Date32
        | DirectPrimitiveColumnType::Decimal128Int64 { .. }
        | DirectPrimitiveColumnType::Decimal128Int64Raw { .. } => return Ok(false),
    };
    let specs = columns
        .iter()
        .zip(column_types.iter())
        .map(|(name, column_type)| DirectPrimitiveColumnSpec {
            name,
            column_type: *column_type,
        })
        .collect::<Vec<_>>();
    let mut written = 0usize;
    let mut primitive_metrics = DirectPrimitiveColumnScanMetrics::default();
    let profile = ordered_sink_profile_enabled();
    let total_started = profile.then(Instant::now);
    let mut write_elapsed = Duration::ZERO;
    let mut concat_elapsed = Duration::ZERO;
    let mut slice_elapsed = Duration::ZERO;
    let mut batches_written = 0usize;
    let row_groups = (0..row_group_count).rev().collect::<Vec<_>>();
    let chunk_size = ordered_primitive_sink_row_group_chunk_size(limit, row_group_count).max(1);
    if fused_page_decoder_enabled {
        let filter_i32_values;
        let filter_i64_values;
        let (filter_i32, filter_i64) = match &filter_values {
            PrimitiveFilterValues::I32(values) => {
                filter_i32_values = values.clone();
                (&filter_i32_values[..], &[][..])
            }
            PrimitiveFilterValues::I64(values) => {
                filter_i64_values = values.clone();
                (&[][..], &filter_i64_values[..])
            }
        };
        let mut fused_batches = Vec::new();
        let mut fused_metrics = DirectPrimitiveColumnScanMetrics::default();
        let mut fused_supported = true;
        for chunk in row_groups.chunks(chunk_size) {
            let mut chunk_batches = Vec::new();
            let metrics = match engine.scan_parquet_required_plain_primitive_in_list_desc(
                &shared.path,
                batch_size,
                chunk,
                &specs,
                filter_index,
                filter_i32,
                filter_i64,
                |batch| {
                    let batch = direct_ordered_primitive_batch_to_primitive_batch(
                        batch,
                        columns,
                        &column_types,
                    )?;
                    if !batch.is_empty() {
                        chunk_batches.push(batch);
                    }
                    Ok(())
                },
            )? {
                Some(metrics) => metrics,
                None => {
                    fused_supported = false;
                    break;
                }
            };
            fused_metrics.merge_from(metrics);
            if chunk_batches.is_empty() {
                continue;
            }
            fused_batches.push(PrimitiveBatch::concat(chunk_batches)?);
        }
        if fused_supported {
            primitive_metrics.merge_from(fused_metrics);
            for batch in fused_batches {
                let remaining = limit - written;
                let batch = if batch.num_rows() > remaining {
                    let slice_started = profile.then(Instant::now);
                    let batch = batch.slice(0, remaining)?;
                    if let Some(started) = slice_started {
                        slice_elapsed += started.elapsed();
                    }
                    batch
                } else {
                    batch
                };
                written += batch.num_rows();
                let write_started = profile.then(Instant::now);
                write_same_source_primitive_batch_to_sink(batch, scan_projection, shared, sink)?;
                if let Some(started) = write_started {
                    write_elapsed += started.elapsed();
                    batches_written += 1;
                }
                if written >= limit {
                    break;
                }
            }
            print_ordered_primitive_sink_profile(
                total_started,
                &primitive_metrics,
                write_elapsed,
                concat_elapsed,
                slice_elapsed,
                batches_written,
                written,
            );
            return Ok(true);
        }
        if !primitive_sink_enabled {
            return Ok(false);
        }
    }
    for chunk in row_groups.chunks(chunk_size) {
        let mut chunk_batches = Vec::new();
        if ordered_primitive_parallel_chunk_enabled(chunk.len(), limit) {
            let page_reader_enabled = ordered_primitive_sink_page_reader_enabled();
            let scan_results = std::thread::scope(|scope| {
                let mut handles = Vec::with_capacity(chunk.len());
                let columns_ref = columns;
                let column_types_ref = &column_types;
                let filter_values_ref = &filter_values;
                for (position, row_group) in chunk.iter().copied().enumerate() {
                    let engine = engine.clone();
                    let path = shared.path.clone();
                    let specs = specs.clone();
                    handles.push(scope.spawn(move || {
                        let mut row_group_batches = Vec::new();
                        let metrics = if page_reader_enabled {
                            engine.scan_parquet_primitive_columns_page_view(
                                &path,
                                batch_size,
                                &[row_group],
                                &specs,
                                |view| {
                                    if let Some(batch) = primitive_ordered_selected_batch(
                                        view,
                                        columns_ref,
                                        column_types_ref,
                                        filter_index,
                                        filter_values_ref,
                                    )? && batch.num_rows() > 0
                                    {
                                        row_group_batches.push(batch);
                                    }
                                    Ok(())
                                },
                            )?
                        } else {
                            engine.scan_parquet_primitive_columns_view(
                                &path,
                                batch_size,
                                &[row_group],
                                &specs,
                                |view| {
                                    if let Some(batch) = primitive_ordered_selected_batch(
                                        view,
                                        columns_ref,
                                        column_types_ref,
                                        filter_index,
                                        filter_values_ref,
                                    )? && batch.num_rows() > 0
                                    {
                                        row_group_batches.push(batch);
                                    }
                                    Ok(())
                                },
                            )?
                        };
                        Ok::<_, DodamError>((position, row_group_batches, metrics))
                    }));
                }
                let mut results = Vec::with_capacity(handles.len());
                for handle in handles {
                    match handle.join() {
                        Ok(result) => results.push(result?),
                        Err(_) => {
                            return Err(DodamError::UnsupportedSql(
                                "ordered primitive scan worker panicked".to_string(),
                            ));
                        }
                    }
                }
                Ok::<_, DodamError>(results)
            })?;
            let mut scan_results = scan_results;
            scan_results.sort_by_key(|(position, _, _)| *position);
            for (_, row_group_batches, metrics) in scan_results {
                if let Some(metrics) = metrics {
                    primitive_metrics.merge_from(metrics);
                }
                chunk_batches.extend(row_group_batches.into_iter().rev());
            }
        } else {
            for &row_group in chunk {
                let mut row_group_batches = Vec::new();
                let metrics = if ordered_primitive_sink_page_reader_enabled() {
                    engine.scan_parquet_primitive_columns_page_view(
                        &shared.path,
                        batch_size,
                        &[row_group],
                        &specs,
                        |view| {
                            if written >= limit {
                                return Ok(());
                            }
                            if let Some(batch) = primitive_ordered_selected_batch(
                                view,
                                columns,
                                &column_types,
                                filter_index,
                                &filter_values,
                            )? && batch.num_rows() > 0
                            {
                                row_group_batches.push(batch);
                            }
                            Ok(())
                        },
                    )?
                } else {
                    engine.scan_parquet_primitive_columns_view(
                        &shared.path,
                        batch_size,
                        &[row_group],
                        &specs,
                        |view| {
                            if written >= limit {
                                return Ok(());
                            }
                            if let Some(batch) = primitive_ordered_selected_batch(
                                view,
                                columns,
                                &column_types,
                                filter_index,
                                &filter_values,
                            )? && batch.num_rows() > 0
                            {
                                row_group_batches.push(batch);
                            }
                            Ok(())
                        },
                    )?
                };
                if let Some(metrics) = metrics {
                    primitive_metrics.merge_from(metrics);
                }
                chunk_batches.extend(row_group_batches.into_iter().rev());
                if written
                    + chunk_batches
                        .iter()
                        .map(PrimitiveBatch::num_rows)
                        .sum::<usize>()
                    >= limit
                {
                    break;
                }
            }
        }
        if chunk_batches.is_empty() {
            continue;
        }
        let concat_started = profile.then(Instant::now);
        let batch = PrimitiveBatch::concat(chunk_batches)?;
        if let Some(started) = concat_started {
            concat_elapsed += started.elapsed();
        }
        if batch.is_empty() {
            continue;
        }
        let remaining = limit - written;
        let batch = if batch.num_rows() > remaining {
            let slice_started = profile.then(Instant::now);
            let batch = batch.slice(0, remaining)?;
            if let Some(started) = slice_started {
                slice_elapsed += started.elapsed();
            }
            batch
        } else {
            batch
        };
        written += batch.num_rows();
        let write_started = profile.then(Instant::now);
        write_same_source_primitive_batch_to_sink(batch, scan_projection, shared, sink)?;
        if let Some(started) = write_started {
            write_elapsed += started.elapsed();
            batches_written += 1;
        }
        if written >= limit {
            print_ordered_primitive_sink_profile(
                total_started,
                &primitive_metrics,
                write_elapsed,
                concat_elapsed,
                slice_elapsed,
                batches_written,
                written,
            );
            return Ok(true);
        }
    }
    print_ordered_primitive_sink_profile(
        total_started,
        &primitive_metrics,
        write_elapsed,
        concat_elapsed,
        slice_elapsed,
        batches_written,
        written,
    );
    Ok(true)
}

fn ordered_primitive_sink_enabled(limit: usize) -> bool {
    if std::env::var("DODAM_ENABLE_ORDERED_PRIMITIVE_SINK")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
    {
        return true;
    }
    if std::env::var("DODAM_DISABLE_ORDERED_PRIMITIVE_SINK_AUTO")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
    {
        return false;
    }
    choose_ordered_primitive_sink(OrderedPrimitiveSinkCostInput {
        limit,
        small_limit_rows: ordered_primitive_sink_auto_max_limit_rows(),
        large_auto_enabled: ordered_primitive_large_sink_auto_enabled(limit),
    })
}

fn ordered_primitive_sink_auto_max_limit_rows() -> usize {
    std::env::var("DODAM_ORDERED_PRIMITIVE_SINK_AUTO_MAX_LIMIT_ROWS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(16 * 1024)
}

fn ordered_primitive_large_sink_auto_enabled(limit: usize) -> bool {
    if std::env::var("DODAM_DISABLE_ORDERED_PRIMITIVE_LARGE_SINK_AUTO")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
    {
        return false;
    }
    limit > ordered_primitive_sink_auto_max_limit_rows()
}

fn ordered_primitive_fused_page_decoder_enabled() -> bool {
    std::env::var("DODAM_ENABLE_ORDERED_PRIMITIVE_FUSED_PAGE_DECODER")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

fn ordered_primitive_sink_page_reader_enabled() -> bool {
    std::env::var("DODAM_ENABLE_ORDERED_PRIMITIVE_PAGE_READER")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

fn print_ordered_primitive_sink_profile(
    total_started: Option<Instant>,
    metrics: &DirectPrimitiveColumnScanMetrics,
    write_elapsed: Duration,
    concat_elapsed: Duration,
    slice_elapsed: Duration,
    batches: usize,
    rows: usize,
) {
    let Some(total_started) = total_started else {
        return;
    };
    let column_read = metrics
        .column_read_nanos
        .iter()
        .map(|nanos| format!("{:.3}", (*nanos as f64) / 1_000_000.0))
        .collect::<Vec<_>>()
        .join(",");
    eprintln!(
        "[dodam:ordered-primitive-sink-profile] total={}us read={:.3}ms consume={:.3}ms concat={}us slice={}us write={}us row_groups={} batches={} rows={} column_read_ms=[{}]",
        total_started.elapsed().as_micros(),
        (metrics.read_nanos as f64) / 1_000_000.0,
        (metrics.consume_nanos as f64) / 1_000_000.0,
        concat_elapsed.as_micros(),
        slice_elapsed.as_micros(),
        write_elapsed.as_micros(),
        metrics.row_groups,
        batches,
        rows,
        column_read,
    );
}

fn primitive_ordered_selected_batch(
    view: BatchView<'_>,
    column_names: &[String],
    column_types: &[DirectPrimitiveColumnType],
    filter_index: usize,
    filter_values: &PrimitiveFilterValues,
) -> Result<Option<PrimitiveBatch>> {
    if let Some(batch) = primitive_ordered_selected_batch_null_free_fast(
        view,
        column_names,
        column_types,
        filter_index,
        filter_values,
    )? {
        return Ok(Some(batch));
    }
    if !row_at_time_fallback_enabled() {
        return Ok(None);
    }
    let mut selected = Vec::new();
    match filter_values {
        PrimitiveFilterValues::I32(values) => {
            if !matches!(column_types[filter_index], DirectPrimitiveColumnType::I32) {
                return Ok(None);
            }
            let Some(column) = view.i32_vector(filter_index) else {
                return Ok(None);
            };
            for row in (0..view.num_rows()).rev() {
                if !column.is_null(row) && values.iter().any(|value| *value == column.value(row)) {
                    selected.push(row as u32);
                }
            }
        }
        PrimitiveFilterValues::I64(values) => {
            if !matches!(column_types[filter_index], DirectPrimitiveColumnType::I64) {
                return Ok(None);
            }
            let Some(column) = view.i64_vector(filter_index) else {
                return Ok(None);
            };
            for row in (0..view.num_rows()).rev() {
                if !column.is_null(row) && values.iter().any(|value| *value == column.value(row)) {
                    selected.push(row as u32);
                }
            }
        }
    }
    if selected.is_empty() {
        return primitive_empty_batch(column_names, column_types).map(Some);
    }
    let mut columns = Vec::with_capacity(column_types.len());
    for (index, column_type) in column_types.iter().enumerate() {
        match column_type {
            DirectPrimitiveColumnType::I32 => {
                let Some(column) = view.i32_vector(index) else {
                    return Ok(None);
                };
                let mut output = Vec::with_capacity(selected.len());
                for row in &selected {
                    let row = *row as usize;
                    if column.is_null(row) {
                        return Ok(None);
                    }
                    output.push(column.value(row));
                }
                columns.push(PrimitiveColumn {
                    name: column_names[index].clone(),
                    data_type: primitive_output_data_type(column_type)?,
                    nullable: false,
                    values: PrimitiveColumnValues::I32(output),
                });
            }
            DirectPrimitiveColumnType::I64 => {
                let Some(column) = view.i64_vector(index) else {
                    return Ok(None);
                };
                let mut output = Vec::with_capacity(selected.len());
                for row in &selected {
                    let row = *row as usize;
                    if column.is_null(row) {
                        return Ok(None);
                    }
                    output.push(column.value(row));
                }
                columns.push(PrimitiveColumn {
                    name: column_names[index].clone(),
                    data_type: primitive_output_data_type(column_type)?,
                    nullable: false,
                    values: PrimitiveColumnValues::I64(output),
                });
            }
            DirectPrimitiveColumnType::Date32
            | DirectPrimitiveColumnType::Decimal128Int64 { .. }
            | DirectPrimitiveColumnType::Decimal128Int64Raw { .. } => return Ok(None),
        }
    }
    Ok(Some(PrimitiveBatch { columns }))
}

pub(super) fn primitive_topk_filter_positions_into(
    column: NullFreePrimitiveColumn<'_>,
    filter_values: &PrimitiveFilterValues,
    selected: &mut Vec<usize>,
) {
    selected.clear();
    match (column, filter_values) {
        (NullFreePrimitiveColumn::I32(values), PrimitiveFilterValues::I32(filter_values)) => {
            reserve_selected_positions(selected, values.len());
            primitive_topk_filter_i32_positions(values, filter_values, selected);
        }
        (NullFreePrimitiveColumn::I64(values), PrimitiveFilterValues::I64(filter_values)) => {
            reserve_selected_positions(selected, values.len());
            primitive_topk_filter_i64_positions(values, filter_values, selected);
        }
        _ => {}
    }
}

pub(super) fn primitive_topk_filter_positions_with_min_key_into(
    filter_column: NullFreePrimitiveColumn<'_>,
    filter_values: &PrimitiveFilterValues,
    sort_column: NullFreePrimitiveColumn<'_>,
    min_key: i128,
    selected: &mut Vec<usize>,
) {
    selected.clear();
    match (filter_column, filter_values, sort_column) {
        (
            NullFreePrimitiveColumn::I32(filter_values_slice),
            PrimitiveFilterValues::I32(filter_values),
            NullFreePrimitiveColumn::I32(sort_values),
        ) => {
            let min_key = match i32::try_from(min_key) {
                Ok(min_key) => min_key,
                Err(_) if min_key < i128::from(i32::MIN) => {
                    primitive_topk_filter_i32_positions(
                        filter_values_slice,
                        filter_values,
                        selected,
                    );
                    return;
                }
                Err(_) => {
                    selected.clear();
                    return;
                }
            };
            reserve_selected_positions(selected, filter_values_slice.len());
            primitive_topk_filter_i32_positions_with_i32_min_key(
                filter_values_slice,
                filter_values,
                sort_values,
                min_key,
                selected,
            );
        }
        (
            NullFreePrimitiveColumn::I32(filter_values_slice),
            PrimitiveFilterValues::I32(filter_values),
            NullFreePrimitiveColumn::I64(sort_values),
        ) => {
            let min_key = match i64::try_from(min_key) {
                Ok(min_key) => min_key,
                Err(_) if min_key < i128::from(i64::MIN) => {
                    primitive_topk_filter_i32_positions(
                        filter_values_slice,
                        filter_values,
                        selected,
                    );
                    return;
                }
                Err(_) => {
                    selected.clear();
                    return;
                }
            };
            reserve_selected_positions(selected, filter_values_slice.len());
            primitive_topk_filter_i32_positions_with_i64_min_key(
                filter_values_slice,
                filter_values,
                sort_values,
                min_key,
                selected,
            );
        }
        (
            NullFreePrimitiveColumn::I64(filter_values_slice),
            PrimitiveFilterValues::I64(filter_values),
            NullFreePrimitiveColumn::I64(sort_values),
        ) => {
            let min_key = match i64::try_from(min_key) {
                Ok(min_key) => min_key,
                Err(_) if min_key < i128::from(i64::MIN) => {
                    primitive_topk_filter_i64_positions(
                        filter_values_slice,
                        filter_values,
                        selected,
                    );
                    return;
                }
                Err(_) => {
                    selected.clear();
                    return;
                }
            };
            reserve_selected_positions(selected, filter_values_slice.len());
            primitive_topk_filter_i64_positions_with_i64_min_key(
                filter_values_slice,
                filter_values,
                sort_values,
                min_key,
                selected,
            );
        }
        (
            NullFreePrimitiveColumn::I64(filter_values_slice),
            PrimitiveFilterValues::I64(filter_values),
            NullFreePrimitiveColumn::I32(sort_values),
        ) => {
            let min_key = match i32::try_from(min_key) {
                Ok(min_key) => min_key,
                Err(_) if min_key < i128::from(i32::MIN) => {
                    primitive_topk_filter_i64_positions(
                        filter_values_slice,
                        filter_values,
                        selected,
                    );
                    return;
                }
                Err(_) => {
                    selected.clear();
                    return;
                }
            };
            reserve_selected_positions(selected, filter_values_slice.len());
            primitive_topk_filter_i64_positions_with_i32_min_key(
                filter_values_slice,
                filter_values,
                sort_values,
                min_key,
                selected,
            );
        }
        _ => primitive_topk_filter_positions_into(filter_column, filter_values, selected),
    }
}

pub(super) fn primitive_topk_filter_i32_positions(
    values: &[i32],
    filter_values: &[i32],
    selected: &mut Vec<usize>,
) {
    match filter_values {
        [] => {}
        [a] => push_i32_eq_positions(values, *a, selected),
        [a, b] => push_i32_eq2_positions(values, *a, *b, selected),
        [a, b, c] => push_i32_eq3_positions(values, *a, *b, *c, selected),
        [a, b, c, d] => push_i32_eq4_positions(values, *a, *b, *c, *d, selected),
        _ => {
            for (row, value) in values.iter().copied().enumerate() {
                if filter_values.contains(&value) {
                    selected.push(row);
                }
            }
        }
    }
}

pub(super) fn primitive_topk_filter_i64_positions(
    values: &[i64],
    filter_values: &[i64],
    selected: &mut Vec<usize>,
) {
    match filter_values {
        [] => {}
        [a] => push_i64_eq_positions(values, *a, selected),
        [a, b] => push_i64_eq2_positions(values, *a, *b, selected),
        [a, b, c] => push_i64_eq3_positions(values, *a, *b, *c, selected),
        [a, b, c, d] => push_i64_eq4_positions(values, *a, *b, *c, *d, selected),
        _ => {
            for (row, value) in values.iter().copied().enumerate() {
                if filter_values.contains(&value) {
                    selected.push(row);
                }
            }
        }
    }
}

fn primitive_topk_filter_i32_positions_with_i32_min_key(
    values: &[i32],
    filter_values: &[i32],
    sort_values: &[i32],
    min_key: i32,
    selected: &mut Vec<usize>,
) {
    push_primitive_position_pairs_unrolled_with_offset(
        values,
        sort_values,
        0,
        selected,
        |value, key| key >= min_key && small_i32_filter_values_contains(filter_values, value),
    );
}

fn primitive_topk_filter_i32_positions_with_i64_min_key(
    values: &[i32],
    filter_values: &[i32],
    sort_values: &[i64],
    min_key: i64,
    selected: &mut Vec<usize>,
) {
    if let [a, b] = filter_values {
        if primitive_topk_block_max_skip_enabled() {
            push_i32_eq2_positions_with_i64_min_key_blocked(
                values,
                *a,
                *b,
                sort_values,
                min_key,
                selected,
            );
        } else {
            push_primitive_position_pairs_unrolled_with_offset(
                values,
                sort_values,
                0,
                selected,
                |value, key| key >= min_key && (value == *a || value == *b),
            );
        }
    } else {
        push_primitive_position_pairs_unrolled_with_offset(
            values,
            sort_values,
            0,
            selected,
            |value, key| key >= min_key && small_i32_filter_values_contains(filter_values, value),
        );
    }
}

fn primitive_topk_filter_i64_positions_with_i64_min_key(
    values: &[i64],
    filter_values: &[i64],
    sort_values: &[i64],
    min_key: i64,
    selected: &mut Vec<usize>,
) {
    push_primitive_position_pairs_unrolled_with_offset(
        values,
        sort_values,
        0,
        selected,
        |value, key| key >= min_key && small_i64_filter_values_contains(filter_values, value),
    );
}

fn primitive_topk_filter_i64_positions_with_i32_min_key(
    values: &[i64],
    filter_values: &[i64],
    sort_values: &[i32],
    min_key: i32,
    selected: &mut Vec<usize>,
) {
    push_primitive_position_pairs_unrolled_with_offset(
        values,
        sort_values,
        0,
        selected,
        |value, key| key >= min_key && small_i64_filter_values_contains(filter_values, value),
    );
}

fn push_i32_eq2_positions_with_i64_min_key_blocked(
    values: &[i32],
    a: i32,
    b: i32,
    keys: &[i64],
    min_key: i64,
    selected: &mut Vec<usize>,
) {
    let len = values.len().min(keys.len());
    let mut row = 0usize;
    const BLOCK: usize = 64;
    while row + BLOCK <= len {
        let block_keys = &keys[row..row + BLOCK];
        let mut max_key = block_keys[0];
        for key in block_keys.iter().copied().skip(1) {
            if key > max_key {
                max_key = key;
            }
        }
        if max_key >= min_key {
            push_primitive_position_pairs_unrolled_with_offset(
                &values[row..row + BLOCK],
                block_keys,
                row,
                selected,
                |value, key| key >= min_key && (value == a || value == b),
            );
        }
        row += BLOCK;
    }
    while row < len {
        let value = values[row];
        if keys[row] >= min_key && (value == a || value == b) {
            selected.push(row);
        }
        row += 1;
    }
}

fn small_i32_filter_values_contains(values: &[i32], value: i32) -> bool {
    match values {
        [] => false,
        [a] => value == *a,
        [a, b] => value == *a || value == *b,
        [a, b, c] => value == *a || value == *b || value == *c,
        [a, b, c, d] => value == *a || value == *b || value == *c || value == *d,
        _ => values.contains(&value),
    }
}

fn small_i64_filter_values_contains(values: &[i64], value: i64) -> bool {
    match values {
        [] => false,
        [a] => value == *a,
        [a, b] => value == *a || value == *b,
        [a, b, c] => value == *a || value == *b || value == *c,
        [a, b, c, d] => value == *a || value == *b || value == *c || value == *d,
        _ => values.contains(&value),
    }
}

fn push_i32_eq_positions(values: &[i32], a: i32, selected: &mut Vec<usize>) {
    push_primitive_positions_unrolled(values, selected, |value| value == a);
}

fn push_i32_eq2_positions(values: &[i32], a: i32, b: i32, selected: &mut Vec<usize>) {
    push_primitive_positions_unrolled(values, selected, |value| value == a || value == b);
}

fn push_i32_eq3_positions(values: &[i32], a: i32, b: i32, c: i32, selected: &mut Vec<usize>) {
    push_primitive_positions_unrolled(values, selected, |value| {
        value == a || value == b || value == c
    });
}

fn push_i32_eq4_positions(
    values: &[i32],
    a: i32,
    b: i32,
    c: i32,
    d: i32,
    selected: &mut Vec<usize>,
) {
    push_primitive_positions_unrolled(values, selected, |value| {
        value == a || value == b || value == c || value == d
    });
}

fn push_i64_eq_positions(values: &[i64], a: i64, selected: &mut Vec<usize>) {
    push_primitive_positions_unrolled(values, selected, |value| value == a);
}

fn push_i64_eq2_positions(values: &[i64], a: i64, b: i64, selected: &mut Vec<usize>) {
    push_primitive_positions_unrolled(values, selected, |value| value == a || value == b);
}

fn push_i64_eq3_positions(values: &[i64], a: i64, b: i64, c: i64, selected: &mut Vec<usize>) {
    push_primitive_positions_unrolled(values, selected, |value| {
        value == a || value == b || value == c
    });
}

fn push_i64_eq4_positions(
    values: &[i64],
    a: i64,
    b: i64,
    c: i64,
    d: i64,
    selected: &mut Vec<usize>,
) {
    push_primitive_positions_unrolled(values, selected, |value| {
        value == a || value == b || value == c || value == d
    });
}

fn push_primitive_positions_unrolled<T, F>(values: &[T], selected: &mut Vec<usize>, mut matches: F)
where
    T: Copy,
    F: FnMut(T) -> bool,
{
    let mut row = 0usize;
    let chunks = values.len() / 8 * 8;
    while row < chunks {
        let v0 = values[row];
        let v1 = values[row + 1];
        let v2 = values[row + 2];
        let v3 = values[row + 3];
        let v4 = values[row + 4];
        let v5 = values[row + 5];
        let v6 = values[row + 6];
        let v7 = values[row + 7];
        if matches(v0) {
            selected.push(row);
        }
        if matches(v1) {
            selected.push(row + 1);
        }
        if matches(v2) {
            selected.push(row + 2);
        }
        if matches(v3) {
            selected.push(row + 3);
        }
        if matches(v4) {
            selected.push(row + 4);
        }
        if matches(v5) {
            selected.push(row + 5);
        }
        if matches(v6) {
            selected.push(row + 6);
        }
        if matches(v7) {
            selected.push(row + 7);
        }
        row += 8;
    }
    while row < values.len() {
        if matches(values[row]) {
            selected.push(row);
        }
        row += 1;
    }
}

fn push_primitive_position_pairs_unrolled_with_offset<T, U, F>(
    values: &[T],
    keys: &[U],
    offset: usize,
    selected: &mut Vec<usize>,
    mut matches: F,
) where
    T: Copy,
    U: Copy,
    F: FnMut(T, U) -> bool,
{
    let len = values.len().min(keys.len());
    let mut row = 0usize;
    let chunks = len / 8 * 8;
    while row < chunks {
        let v0 = values[row];
        let v1 = values[row + 1];
        let v2 = values[row + 2];
        let v3 = values[row + 3];
        let v4 = values[row + 4];
        let v5 = values[row + 5];
        let v6 = values[row + 6];
        let v7 = values[row + 7];
        let k0 = keys[row];
        let k1 = keys[row + 1];
        let k2 = keys[row + 2];
        let k3 = keys[row + 3];
        let k4 = keys[row + 4];
        let k5 = keys[row + 5];
        let k6 = keys[row + 6];
        let k7 = keys[row + 7];
        if matches(v0, k0) {
            selected.push(offset + row);
        }
        if matches(v1, k1) {
            selected.push(offset + row + 1);
        }
        if matches(v2, k2) {
            selected.push(offset + row + 2);
        }
        if matches(v3, k3) {
            selected.push(offset + row + 3);
        }
        if matches(v4, k4) {
            selected.push(offset + row + 4);
        }
        if matches(v5, k5) {
            selected.push(offset + row + 5);
        }
        if matches(v6, k6) {
            selected.push(offset + row + 6);
        }
        if matches(v7, k7) {
            selected.push(offset + row + 7);
        }
        row += 8;
    }
    while row < len {
        if matches(values[row], keys[row]) {
            selected.push(offset + row);
        }
        row += 1;
    }
}

fn primitive_filter_i32_positions_desc(
    values: &[i32],
    filter_values: &[i32],
    selected: &mut Vec<usize>,
) {
    match filter_values {
        [] => {}
        [a] => push_i32_eq_positions_desc(values, *a, selected),
        [a, b] => push_i32_eq2_positions_desc(values, *a, *b, selected),
        [a, b, c] => push_primitive_positions_desc_unrolled(values, selected, |value| {
            value == *a || value == *b || value == *c
        }),
        [a, b, c, d] => push_primitive_positions_desc_unrolled(values, selected, |value| {
            value == *a || value == *b || value == *c || value == *d
        }),
        _ => {
            let mut row = values.len();
            while row > 0 {
                row -= 1;
                if filter_values.contains(&values[row]) {
                    selected.push(row);
                }
            }
        }
    }
}

fn primitive_filter_i64_positions_desc(
    values: &[i64],
    filter_values: &[i64],
    selected: &mut Vec<usize>,
) {
    match filter_values {
        [] => {}
        [a] => push_i64_eq_positions_desc(values, *a, selected),
        [a, b] => push_i64_eq2_positions_desc(values, *a, *b, selected),
        [a, b, c] => push_primitive_positions_desc_unrolled(values, selected, |value| {
            value == *a || value == *b || value == *c
        }),
        [a, b, c, d] => push_primitive_positions_desc_unrolled(values, selected, |value| {
            value == *a || value == *b || value == *c || value == *d
        }),
        _ => {
            let mut row = values.len();
            while row > 0 {
                row -= 1;
                if filter_values.contains(&values[row]) {
                    selected.push(row);
                }
            }
        }
    }
}

fn push_i32_eq_positions_desc(values: &[i32], a: i32, selected: &mut Vec<usize>) {
    let mut row = values.len();
    while row >= 8 {
        row -= 8;
        if values[row + 7] == a {
            selected.push(row + 7);
        }
        if values[row + 6] == a {
            selected.push(row + 6);
        }
        if values[row + 5] == a {
            selected.push(row + 5);
        }
        if values[row + 4] == a {
            selected.push(row + 4);
        }
        if values[row + 3] == a {
            selected.push(row + 3);
        }
        if values[row + 2] == a {
            selected.push(row + 2);
        }
        if values[row + 1] == a {
            selected.push(row + 1);
        }
        if values[row] == a {
            selected.push(row);
        }
    }
    while row > 0 {
        row -= 1;
        if values[row] == a {
            selected.push(row);
        }
    }
}

fn push_i32_eq2_positions_desc(values: &[i32], a: i32, b: i32, selected: &mut Vec<usize>) {
    let mut row = values.len();
    while row >= 8 {
        row -= 8;
        let v7 = values[row + 7];
        let v6 = values[row + 6];
        let v5 = values[row + 5];
        let v4 = values[row + 4];
        let v3 = values[row + 3];
        let v2 = values[row + 2];
        let v1 = values[row + 1];
        let v0 = values[row];
        if v7 == a || v7 == b {
            selected.push(row + 7);
        }
        if v6 == a || v6 == b {
            selected.push(row + 6);
        }
        if v5 == a || v5 == b {
            selected.push(row + 5);
        }
        if v4 == a || v4 == b {
            selected.push(row + 4);
        }
        if v3 == a || v3 == b {
            selected.push(row + 3);
        }
        if v2 == a || v2 == b {
            selected.push(row + 2);
        }
        if v1 == a || v1 == b {
            selected.push(row + 1);
        }
        if v0 == a || v0 == b {
            selected.push(row);
        }
    }
    while row > 0 {
        row -= 1;
        let value = values[row];
        if value == a || value == b {
            selected.push(row);
        }
    }
}

fn push_i64_eq_positions_desc(values: &[i64], a: i64, selected: &mut Vec<usize>) {
    let mut row = values.len();
    while row >= 8 {
        row -= 8;
        if values[row + 7] == a {
            selected.push(row + 7);
        }
        if values[row + 6] == a {
            selected.push(row + 6);
        }
        if values[row + 5] == a {
            selected.push(row + 5);
        }
        if values[row + 4] == a {
            selected.push(row + 4);
        }
        if values[row + 3] == a {
            selected.push(row + 3);
        }
        if values[row + 2] == a {
            selected.push(row + 2);
        }
        if values[row + 1] == a {
            selected.push(row + 1);
        }
        if values[row] == a {
            selected.push(row);
        }
    }
    while row > 0 {
        row -= 1;
        if values[row] == a {
            selected.push(row);
        }
    }
}

fn push_i64_eq2_positions_desc(values: &[i64], a: i64, b: i64, selected: &mut Vec<usize>) {
    let mut row = values.len();
    while row >= 8 {
        row -= 8;
        let v7 = values[row + 7];
        let v6 = values[row + 6];
        let v5 = values[row + 5];
        let v4 = values[row + 4];
        let v3 = values[row + 3];
        let v2 = values[row + 2];
        let v1 = values[row + 1];
        let v0 = values[row];
        if v7 == a || v7 == b {
            selected.push(row + 7);
        }
        if v6 == a || v6 == b {
            selected.push(row + 6);
        }
        if v5 == a || v5 == b {
            selected.push(row + 5);
        }
        if v4 == a || v4 == b {
            selected.push(row + 4);
        }
        if v3 == a || v3 == b {
            selected.push(row + 3);
        }
        if v2 == a || v2 == b {
            selected.push(row + 2);
        }
        if v1 == a || v1 == b {
            selected.push(row + 1);
        }
        if v0 == a || v0 == b {
            selected.push(row);
        }
    }
    while row > 0 {
        row -= 1;
        let value = values[row];
        if value == a || value == b {
            selected.push(row);
        }
    }
}

fn push_primitive_positions_desc_unrolled<T, F>(
    values: &[T],
    selected: &mut Vec<usize>,
    mut matches: F,
) where
    T: Copy,
    F: FnMut(T) -> bool,
{
    let mut row = values.len();
    while row >= 8 {
        row -= 8;
        if matches(values[row + 7]) {
            selected.push(row + 7);
        }
        if matches(values[row + 6]) {
            selected.push(row + 6);
        }
        if matches(values[row + 5]) {
            selected.push(row + 5);
        }
        if matches(values[row + 4]) {
            selected.push(row + 4);
        }
        if matches(values[row + 3]) {
            selected.push(row + 3);
        }
        if matches(values[row + 2]) {
            selected.push(row + 2);
        }
        if matches(values[row + 1]) {
            selected.push(row + 1);
        }
        if matches(values[row]) {
            selected.push(row);
        }
    }
    while row > 0 {
        row -= 1;
        if matches(values[row]) {
            selected.push(row);
        }
    }
}

pub(super) fn reserve_selected_positions(selected: &mut Vec<usize>, row_count: usize) {
    let target = row_count / 4 + 1;
    if selected.capacity() < target {
        selected.reserve(target - selected.capacity());
    }
}

pub(super) fn primitive_topk_sequence_base(location: DirectPrimitiveBatchLocation) -> Option<u64> {
    let row_group = u64::try_from(location.row_group).ok()?;
    let row_offset = u64::try_from(location.row_offset).ok()?;
    Some((row_group << 32) | row_offset)
}

fn primitive_ordered_selected_batch_null_free_fast(
    view: BatchView<'_>,
    column_names: &[String],
    column_types: &[DirectPrimitiveColumnType],
    filter_index: usize,
    filter_values: &PrimitiveFilterValues,
) -> Result<Option<PrimitiveBatch>> {
    if filter_index >= column_types.len() {
        return Ok(None);
    }
    let row_count = view.num_rows();
    let mut columns = Vec::with_capacity(column_types.len());
    for (index, column_type) in column_types.iter().enumerate() {
        match column_type {
            DirectPrimitiveColumnType::I32 => {
                let Some(column) = view.i32_vector(index) else {
                    return Ok(None);
                };
                let Some(values) = column.values_if_null_free() else {
                    return Ok(None);
                };
                columns.push(NullFreePrimitiveColumn::I32(values));
            }
            DirectPrimitiveColumnType::I64 => {
                let Some(column) = view.i64_vector(index) else {
                    return Ok(None);
                };
                let Some(values) = column.values_if_null_free() else {
                    return Ok(None);
                };
                columns.push(NullFreePrimitiveColumn::I64(values));
            }
            DirectPrimitiveColumnType::Date32
            | DirectPrimitiveColumnType::Decimal128Int64 { .. }
            | DirectPrimitiveColumnType::Decimal128Int64Raw { .. } => return Ok(None),
        }
    }
    if columns.iter().any(|column| match column {
        NullFreePrimitiveColumn::I32(values) => values.len() != row_count,
        NullFreePrimitiveColumn::I64(values) => values.len() != row_count,
    }) {
        return Ok(None);
    }
    let mut output = column_types
        .iter()
        .map(|column_type| match column_type {
            DirectPrimitiveColumnType::I32 => {
                PrimitiveColumnOutput::I32(Vec::with_capacity(row_count / 4 + 1))
            }
            DirectPrimitiveColumnType::I64 => {
                PrimitiveColumnOutput::I64(Vec::with_capacity(row_count / 4 + 1))
            }
            _ => unreachable!("checked primitive ordered column type"),
        })
        .collect::<Vec<_>>();
    let mut selected_positions = Vec::with_capacity(row_count / 4 + 1);
    match (filter_values, columns[filter_index]) {
        (PrimitiveFilterValues::I32(values), NullFreePrimitiveColumn::I32(filter_column)) => {
            primitive_filter_i32_positions_desc(filter_column, values, &mut selected_positions);
        }
        (PrimitiveFilterValues::I64(values), NullFreePrimitiveColumn::I64(filter_column)) => {
            primitive_filter_i64_positions_desc(filter_column, values, &mut selected_positions);
        }
        _ => return Ok(None),
    }
    if selected_positions.is_empty() {
        return primitive_empty_batch(column_names, column_types).map(Some);
    }
    if selected_positions.len().saturating_mul(2) >= row_count {
        for (column, output) in columns.iter().zip(output.iter_mut()) {
            gather_null_free_primitive_column(column, &selected_positions, output)?;
        }
    } else {
        for &row in &selected_positions {
            push_null_free_primitive_row(&columns, &mut output, row)?;
        }
    }
    let selected_rows = output
        .first()
        .map(|column| match column {
            PrimitiveColumnOutput::I32(values) => values.len(),
            PrimitiveColumnOutput::I64(values) => values.len(),
        })
        .unwrap_or(0);
    if selected_rows == 0 {
        return primitive_empty_batch(column_names, column_types).map(Some);
    }
    let columns = output
        .into_iter()
        .zip(column_names.iter())
        .zip(column_types.iter())
        .map(|column| match column {
            ((PrimitiveColumnOutput::I32(values), name), column_type) => Ok(PrimitiveColumn {
                name: name.clone(),
                data_type: primitive_output_data_type(column_type)?,
                nullable: false,
                values: PrimitiveColumnValues::I32(values),
            }),
            ((PrimitiveColumnOutput::I64(values), name), column_type) => Ok(PrimitiveColumn {
                name: name.clone(),
                data_type: primitive_output_data_type(column_type)?,
                nullable: false,
                values: PrimitiveColumnValues::I64(values),
            }),
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Some(PrimitiveBatch { columns }))
}

pub(super) fn row_at_time_fallback_enabled() -> bool {
    std::env::var("DODAM_ENABLE_ROW_AT_TIME_FALLBACK")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

pub(super) fn ordered_sink_profile_enabled() -> bool {
    std::env::var("DODAM_PROFILE_ORDERED_SINK")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

fn print_ordered_sink_profile(
    total_started: Option<Instant>,
    scan_sort_elapsed: Duration,
    sink_write_elapsed: Duration,
    row_groups: usize,
    batches: usize,
    rows: usize,
) {
    let Some(total_started) = total_started else {
        return;
    };
    eprintln!(
        "[dodam:ordered-sink-profile] total={}us scan_sort={}us sink_write={}us row_groups={} batches={} rows={}",
        total_started.elapsed().as_micros(),
        scan_sort_elapsed.as_micros(),
        sink_write_elapsed.as_micros(),
        row_groups,
        batches,
        rows
    );
}

fn write_same_source_batch_to_sink(
    batch: RecordBatch,
    scan_projection: &Projection,
    shared: &SameSourceUnionAllScan,
    sink: &mut dyn RecordBatchSink,
) -> Result<()> {
    if scan_projection == &shared.projection && shared.aliases.is_empty() {
        sink.write_batch(&batch)?;
        return Ok(());
    }
    let batches = apply_output_projection(vec![batch], &shared.projection)?;
    let batches = rename_output_batches(batches, &shared.aliases)?;
    for batch in batches {
        sink.write_batch(&batch)?;
    }
    Ok(())
}

pub(super) fn write_same_source_primitive_batch_to_sink(
    batch: PrimitiveBatch,
    scan_projection: &Projection,
    shared: &SameSourceUnionAllScan,
    sink: &mut dyn RecordBatchSink,
) -> Result<()> {
    let batch = project_rename_primitive_batch(
        batch,
        scan_projection,
        &shared.projection,
        &shared.aliases,
    )?;
    if !sink.write_primitive_batch(batch)? {
        return Err(DodamError::UnsupportedSql(
            "primitive sink rejected primitive batch".to_string(),
        ));
    }
    Ok(())
}

fn project_rename_primitive_batch(
    batch: PrimitiveBatch,
    scan_projection: &Projection,
    output_projection: &Projection,
    aliases: &[(String, String)],
) -> Result<PrimitiveBatch> {
    let mut columns = batch.columns;
    if scan_projection != output_projection {
        let Projection::Columns(output_columns) = output_projection else {
            return Err(DodamError::UnsupportedSql(
                "primitive sink cannot widen projected output".to_string(),
            ));
        };
        let mut projected = Vec::with_capacity(output_columns.len());
        for output_column in output_columns {
            let Some(index) = columns
                .iter()
                .position(|column| column.name == *output_column)
            else {
                return Err(DodamError::UnsupportedSql(format!(
                    "primitive output column {output_column} was not scanned"
                )));
            };
            projected.push(columns.remove(index));
        }
        columns = projected;
    }
    if !aliases.is_empty() {
        for column in &mut columns {
            if let Some((alias, _)) = aliases
                .iter()
                .find(|(alias, target)| !alias.contains('(') && target == &column.name)
            {
                column.name = alias.clone();
            }
        }
    }
    Ok(PrimitiveBatch { columns })
}

#[derive(Default)]
struct UnionAllSinkState {
    schema: Option<Arc<Schema>>,
}

async fn write_union_all_set_expr_to_sink(
    engine: &DodamEngine,
    expr: &SetExpr,
    batch_size: usize,
    sink: &mut dyn RecordBatchSink,
    state: &mut UnionAllSinkState,
) -> Result<()> {
    match expr {
        SetExpr::SetOperation {
            op,
            set_quantifier,
            left,
            right,
        } if *op == SetOperator::Union && *set_quantifier == SetQuantifier::All => {
            Box::pin(write_union_all_set_expr_to_sink(
                engine,
                left.as_ref(),
                batch_size,
                sink,
                state,
            ))
            .await?;
            Box::pin(write_union_all_set_expr_to_sink(
                engine,
                right.as_ref(),
                batch_size,
                sink,
                state,
            ))
            .await
        }
        SetExpr::SetOperation {
            op, set_quantifier, ..
        } => Err(DodamError::UnsupportedSql(format!(
            "{op} {set_quantifier} is not supported yet"
        ))),
        SetExpr::Query(query) => {
            if query_contains_set_operation(query.body.as_ref()) {
                return Box::pin(write_union_all_set_expr_to_sink(
                    engine,
                    query.body.as_ref(),
                    batch_size,
                    sink,
                    state,
                ))
                .await;
            }
            if query.order_by.is_some()
                || query.limit_clause.is_some()
                || query.fetch.is_some()
                || !query.locks.is_empty()
            {
                return Err(DodamError::UnsupportedSql(
                    "ORDER BY, LIMIT, FETCH, and locking clauses inside UNION ALL operands are not supported yet"
                        .to_string(),
                ));
            }
            let batches = query_output_batches(
                Box::pin(execute_sql(engine, &query.to_string(), batch_size)).await?,
            )?;
            write_union_all_batches_to_sink(batches, sink, state)
        }
        SetExpr::Select(_) => {
            let batches = query_output_batches(
                Box::pin(execute_sql(engine, &expr.to_string(), batch_size)).await?,
            )?;
            write_union_all_batches_to_sink(batches, sink, state)
        }
        other => Err(DodamError::UnsupportedSql(format!(
            "unsupported UNION ALL operand: {other}"
        ))),
    }
}

fn write_union_all_batches_to_sink(
    batches: Vec<RecordBatch>,
    sink: &mut dyn RecordBatchSink,
    state: &mut UnionAllSinkState,
) -> Result<()> {
    for batch in batches {
        let schema = if let Some(schema) = &state.schema {
            schema.clone()
        } else {
            let schema = batch.schema();
            state.schema = Some(schema.clone());
            schema
        };
        validate_union_all_batches(&schema, std::slice::from_ref(&batch))?;
        let batch = align_union_all_batch_schema(batch, schema)?;
        sink.write_batch(&batch)?;
    }
    Ok(())
}
