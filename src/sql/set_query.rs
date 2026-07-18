use super::*;

pub(super) async fn try_execute_set_operation_sql(
    engine: &DodamEngine,
    sql: &str,
    batch_size: usize,
) -> Result<Option<QueryOutput>> {
    let dialect = GenericDialect {};
    let statements = Parser::parse_sql(&dialect, sql)
        .map_err(|error| DodamError::UnsupportedSql(error.to_string()))?;
    let [Statement::Query(query)] = statements.as_slice() else {
        return Ok(None);
    };
    if !query_contains_set_operation(query.body.as_ref()) {
        return Ok(None);
    }
    let order_by = parse_order_by(query, &[], &[], None)?;
    let limit = parse_limit(query)?;
    let offset = parse_offset(query)?;
    if let Some(batches) = try_execute_same_source_union_all_monotonic_topk(
        engine,
        query.body.as_ref(),
        batch_size,
        order_by.as_ref(),
        limit,
        offset,
    )
    .await?
    {
        return Ok(Some(QueryOutput::Scan { batches }));
    }
    if let Some(mut batches) =
        try_execute_same_source_union_all_scan(engine, query.body.as_ref(), batch_size).await?
    {
        batches = apply_output_order_limit(batches, order_by.as_ref(), limit, offset)?;
        return Ok(Some(QueryOutput::Scan { batches }));
    }
    if order_by.is_none()
        && limit.is_none()
        && offset == 0
        && let Some(batches) =
            try_execute_same_source_union_all_filter_scan(engine, query.body.as_ref(), batch_size)
                .await?
    {
        return Ok(Some(QueryOutput::Scan { batches }));
    }
    if let Some(mut batches) = try_execute_simple_case_distinct_set_literals(query.body.as_ref())? {
        batches = apply_output_order_limit(batches, order_by.as_ref(), limit, offset)?;
        return Ok(Some(QueryOutput::Scan { batches }));
    }
    if let Some(mut batches) = try_execute_same_source_union_distinct_scan(
        engine,
        query.body.as_ref(),
        batch_size,
        order_by.as_ref(),
        limit,
        offset,
    )
    .await?
    {
        batches = apply_output_order_limit(batches, order_by.as_ref(), limit, offset)?;
        return Ok(Some(QueryOutput::Scan { batches }));
    }
    if let Some(mut batches) = try_execute_same_source_distinct_set_scan(
        engine,
        query.body.as_ref(),
        batch_size,
        order_by.as_ref(),
        limit,
        offset,
    )
    .await?
    {
        batches = apply_output_order_limit(batches, order_by.as_ref(), limit, offset)?;
        return Ok(Some(QueryOutput::Scan { batches }));
    }
    if let Some(mut batches) = try_execute_same_source_all_set_scan(
        engine,
        query.body.as_ref(),
        batch_size,
        order_by.as_ref(),
        limit,
        offset,
    )? {
        batches = apply_output_order_limit(batches, order_by.as_ref(), limit, offset)?;
        return Ok(Some(QueryOutput::Scan { batches }));
    }
    let child_topk = if offset == 0 {
        order_by.as_ref().zip(limit)
    } else {
        None
    };
    let mut batches = Box::pin(execute_set_operation_expr(
        engine,
        query.body.as_ref(),
        batch_size,
        child_topk,
        false,
    ))
    .await?;
    batches = apply_output_order_limit(batches, order_by.as_ref(), limit, offset)?;
    Ok(Some(QueryOutput::Scan { batches }))
}

async fn try_execute_same_source_union_all_monotonic_topk(
    engine: &DodamEngine,
    expr: &SetExpr,
    batch_size: usize,
    order_by: Option<&SortKey>,
    limit: Option<usize>,
    offset: usize,
) -> Result<Option<Vec<RecordBatch>>> {
    let (Some(order_by), Some(limit)) = (order_by, limit) else {
        return Ok(None);
    };
    if offset != 0 || limit == 0 {
        return Ok(None);
    }
    let [sort] = order_by.expressions.as_slice() else {
        return Ok(None);
    };
    if !sort.descending || sort.nulls_first {
        return Ok(None);
    }
    let Some(shared) = plan_same_source_union_all_scan(expr)? else {
        return Ok(None);
    };
    if let Some(mut batches) =
        try_row_group_ordered_desc_sort(engine, &shared, order_by, &sort.column, batch_size, limit)
            .await?
    {
        batches = apply_output_projection(batches, &shared.projection)?;
        batches = rename_output_batches(batches, &shared.aliases)?;
        return Ok(Some(batches));
    }
    if let Some(mut batches) =
        try_reverse_row_group_desc_tail_topk(engine, &shared, &sort.column, batch_size, limit)
            .await?
    {
        batches = apply_output_order_limit(batches, Some(order_by), Some(limit), 0)?;
        batches = apply_output_projection(batches, &shared.projection)?;
        batches = rename_output_batches(batches, &shared.aliases)?;
        return Ok(Some(batches));
    }
    if let Some(mut batches) = try_same_source_union_all_streaming_desc_topk(
        engine,
        &shared,
        &sort.column,
        batch_size,
        limit,
    )
    .await?
    {
        batches = apply_output_projection(batches, &shared.projection)?;
        batches = rename_output_batches(batches, &shared.aliases)?;
        return Ok(Some(batches));
    }
    let stream = engine
        .scan_parquet_filtered_batches_preserve_order(
            shared.path,
            batch_size,
            shared.projection,
            Some(shared.filter),
        )
        .await?;
    let Some(mut batches) = collect_monotonic_desc_tail_topk(stream, &sort.column, limit)? else {
        return Ok(None);
    };
    batches = rename_output_batches(batches, &shared.aliases)?;
    batches = apply_output_order_limit(batches, Some(order_by), Some(limit), 0)?;
    Ok(Some(batches))
}

async fn try_row_group_ordered_desc_sort(
    engine: &DodamEngine,
    shared: &SameSourceUnionAllScan,
    order_by: &SortKey,
    sort_column: &str,
    batch_size: usize,
    limit: usize,
) -> Result<Option<Vec<RecordBatch>>> {
    if limit <= reverse_row_group_topk_max_limit_rows() {
        return Ok(None);
    }
    if !engine
        .parquet_row_groups_monotonic_by_column(shared.path.clone(), sort_column)
        .await?
    {
        return Ok(None);
    }
    let row_group_count = engine.parquet_row_group_count(&shared.path)?;
    if row_group_count == 0 {
        return Ok(Some(Vec::new()));
    }

    let mut scan_projection = shared.projection.clone();
    add_projection_columns(&mut scan_projection, shared.filter.referenced_columns());
    for sort in &order_by.expressions {
        add_projection_columns(&mut scan_projection, vec![sort.column.clone()]);
    }

    if row_group_ordered_desc_bulk_global_sort_enabled() {
        let row_groups = (0..row_group_count).rev().collect::<Vec<_>>();
        let batches = engine
            .scan_parquet_row_group_batches(
                shared.path.clone(),
                batch_size,
                scan_projection,
                row_groups,
            )
            .await?;
        let mut filtered = Vec::new();
        for batch in batches {
            let batch = filter_batch(batch, &shared.filter)?;
            if batch.num_rows() > 0 {
                filtered.push(batch);
            }
        }
        return Ok(Some(apply_output_order_limit(
            filtered,
            Some(order_by),
            Some(limit),
            0,
        )?));
    }

    if row_group_ordered_desc_parallel_enabled() && row_group_count > 1 {
        let row_group_batch_size = row_group_ordered_sort_batch_size(batch_size);
        let mut handles = Vec::with_capacity(row_group_count);
        for row_group in (0..row_group_count).rev() {
            let engine = engine.clone();
            let path = shared.path.clone();
            let scan_projection = scan_projection.clone();
            let filter = shared.filter.clone();
            let order_by = order_by.clone();
            let sort_column = sort_column.to_string();
            handles.push(tokio::task::spawn(async move {
                scan_filter_sort_ordered_row_group(
                    &engine,
                    path,
                    row_group_batch_size,
                    scan_projection,
                    row_group,
                    &filter,
                    &order_by,
                    &sort_column,
                )
                .await
            }));
        }
        let mut output = Vec::new();
        let mut rows = 0usize;
        for handle in handles {
            let sorted = handle
                .await
                .map_err(|error| DodamError::UnsupportedSql(error.to_string()))??;
            for batch in sorted {
                rows += batch.num_rows();
                output.push(batch);
            }
            if rows >= limit {
                break;
            }
        }
        return Ok(Some(limit_batches(output, Some(limit), 0)));
    }

    let mut output = Vec::new();
    let mut rows = 0usize;
    let row_group_batch_size = row_group_ordered_sort_batch_size(batch_size);
    for row_group in (0..row_group_count).rev() {
        let sorted = scan_filter_sort_ordered_row_group(
            engine,
            shared.path.clone(),
            row_group_batch_size,
            scan_projection.clone(),
            row_group,
            &shared.filter,
            order_by,
            sort_column,
        )
        .await?;
        for batch in sorted {
            rows += batch.num_rows();
            output.push(batch);
        }
        if rows >= limit {
            break;
        }
    }
    Ok(Some(limit_batches(output, Some(limit), 0)))
}

pub(super) async fn scan_filter_sort_ordered_row_group(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    projection: Projection,
    row_group: usize,
    filter: &FilterExpr,
    order_by: &SortKey,
    sort_column: &str,
) -> Result<Vec<RecordBatch>> {
    let profile = ordered_sink_profile_enabled();
    let total_started = profile.then(Instant::now);
    let scan_started = profile.then(Instant::now);
    let batches = engine
        .scan_parquet_row_group_batches(path, batch_size, projection, vec![row_group])
        .await?;
    let scan_elapsed = scan_started.map(|started| started.elapsed());
    let post_started = profile.then(Instant::now);
    if let Some(sorted) =
        reverse_filter_ascending_batches_if_ordered(&batches, sort_column, filter)?
    {
        print_ordered_row_group_profile(
            total_started,
            scan_elapsed,
            post_started.map(|started| started.elapsed()),
            row_group,
            sorted.iter().map(RecordBatch::num_rows).sum(),
            sorted.len(),
            "reverse-filter",
        );
        return Ok(sorted);
    }
    let mut filtered = Vec::new();
    for batch in batches {
        let batch = filter_batch(batch, filter)?;
        if batch.num_rows() > 0 {
            filtered.push(batch);
        }
    }
    if filtered.is_empty() {
        print_ordered_row_group_profile(
            total_started,
            scan_elapsed,
            post_started.map(|started| started.elapsed()),
            row_group,
            0,
            0,
            "empty",
        );
        return Ok(Vec::new());
    }
    let result = match reverse_ascending_primitive_batches_if_ordered(&filtered, sort_column)? {
        Some(sorted) => Ok(sorted),
        None => apply_output_order_limit(filtered, Some(order_by), None, 0),
    }?;
    print_ordered_row_group_profile(
        total_started,
        scan_elapsed,
        post_started.map(|started| started.elapsed()),
        row_group,
        result.iter().map(RecordBatch::num_rows).sum(),
        result.len(),
        "filter-sort",
    );
    Ok(result)
}

fn row_group_ordered_desc_parallel_enabled() -> bool {
    std::env::var("DODAM_ROW_GROUP_ORDERED_DESC_PARALLEL")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

fn print_ordered_row_group_profile(
    total_started: Option<Instant>,
    scan_elapsed: Option<Duration>,
    post_elapsed: Option<Duration>,
    row_group: usize,
    rows: usize,
    batches: usize,
    mode: &str,
) {
    let Some(total_started) = total_started else {
        return;
    };
    eprintln!(
        "[dodam:ordered-row-group-profile] row_group={} mode={} total={}us scan={}us post={}us rows={} batches={}",
        row_group,
        mode,
        total_started.elapsed().as_micros(),
        scan_elapsed
            .map(|duration| duration.as_micros())
            .unwrap_or(0),
        post_elapsed
            .map(|duration| duration.as_micros())
            .unwrap_or(0),
        rows,
        batches
    );
}

fn reverse_filter_ascending_batches_if_ordered(
    batches: &[RecordBatch],
    sort_column: &str,
    filter: &FilterExpr,
) -> Result<Option<Vec<RecordBatch>>> {
    let mut previous_last: Option<i128> = None;
    for batch in batches {
        if batch.num_rows() == 0 {
            continue;
        }
        let index = output_batch_column_index(batch, sort_column)?;
        let Some((first, last)) = numeric_column_ascending_bounds(batch.column(index))? else {
            return Ok(None);
        };
        if previous_last.is_some_and(|previous| previous > first) {
            return Ok(None);
        }
        previous_last = Some(last);
    }

    let mut output = Vec::new();
    for batch in batches.iter().rev() {
        if batch.num_rows() == 0 {
            continue;
        }
        if let Some(batch) = reverse_primitive_in_list_selected_batch(batch, filter)? {
            if batch.num_rows() > 0 {
                output.push(batch);
            }
            continue;
        }
        let indices = match reverse_simple_in_list_indices(batch, filter)? {
            Some(indices) => indices,
            None if reverse_filter_ordered_batches_disabled() => return Ok(None),
            None => {
                let mask = evaluate_filter_mask(batch, filter)?;
                let mut indices = Vec::new();
                for row in (0..batch.num_rows()).rev() {
                    if !mask.is_null(row) && mask.value(row) {
                        indices.push(row as u32);
                    }
                }
                indices
            }
        };
        if indices.is_empty() {
            continue;
        }
        output.push(take_record_batch(batch, &UInt32Array::from(indices))?);
    }
    Ok(Some(output))
}

fn reverse_primitive_in_list_selected_batch(
    batch: &RecordBatch,
    filter: &FilterExpr,
) -> Result<Option<RecordBatch>> {
    if !reverse_primitive_in_list_ordered_batches_enabled() {
        return Ok(None);
    }
    reverse_primitive_in_list_selected_batch_unchecked(batch, filter)
}

fn reverse_primitive_in_list_selected_batch_unchecked(
    batch: &RecordBatch,
    filter: &FilterExpr,
) -> Result<Option<RecordBatch>> {
    let Some(indices) = reverse_simple_in_list_indices_unchecked(batch, filter)? else {
        return Ok(None);
    };
    if indices.is_empty() {
        return Ok(Some(RecordBatch::new_empty(batch.schema())));
    }
    let mut columns = Vec::with_capacity(batch.num_columns());
    for column in batch.columns() {
        let Some(array) = gather_primitive_array(column, &indices) else {
            return Ok(None);
        };
        columns.push(array);
    }
    Ok(Some(RecordBatch::try_new(batch.schema(), columns)?))
}

pub(super) fn gather_primitive_array(column: &ArrayRef, indices: &[u32]) -> Option<ArrayRef> {
    if column.null_count() != 0 {
        return None;
    }
    match column.data_type() {
        DataType::Int32 => {
            let values = column.as_any().downcast_ref::<Int32Array>()?;
            let mut output = Vec::with_capacity(indices.len());
            for &index in indices {
                output.push(values.value(index as usize));
            }
            Some(Arc::new(Int32Array::from(output)))
        }
        DataType::Int64 => {
            let values = column.as_any().downcast_ref::<Int64Array>()?;
            let mut output = Vec::with_capacity(indices.len());
            for &index in indices {
                output.push(values.value(index as usize));
            }
            Some(Arc::new(Int64Array::from(output)))
        }
        DataType::UInt64 => {
            let values = column.as_any().downcast_ref::<UInt64Array>()?;
            let mut output = Vec::with_capacity(indices.len());
            for &index in indices {
                output.push(values.value(index as usize));
            }
            Some(Arc::new(UInt64Array::from(output)))
        }
        DataType::Float64 => {
            let values = column.as_any().downcast_ref::<Float64Array>()?;
            let mut output = Vec::with_capacity(indices.len());
            for &index in indices {
                output.push(values.value(index as usize));
            }
            Some(Arc::new(Float64Array::from(output)))
        }
        DataType::Date32 => {
            let values = column.as_any().downcast_ref::<Date32Array>()?;
            let mut output = Vec::with_capacity(indices.len());
            for &index in indices {
                output.push(values.value(index as usize));
            }
            Some(Arc::new(Date32Array::from(output)))
        }
        _ => None,
    }
}

fn reverse_simple_in_list_indices(
    batch: &RecordBatch,
    filter: &FilterExpr,
) -> Result<Option<Vec<u32>>> {
    if !reverse_in_list_ordered_batches_enabled() {
        return Ok(None);
    }
    reverse_simple_in_list_indices_unchecked(batch, filter)
}

fn reverse_simple_in_list_indices_unchecked(
    batch: &RecordBatch,
    filter: &FilterExpr,
) -> Result<Option<Vec<u32>>> {
    let Expr::InList {
        column,
        values,
        negated,
        ..
    } = filter.expr()
    else {
        return Ok(None);
    };
    if *negated {
        return Ok(None);
    }
    let column_index = output_batch_column_index(batch, column)?;
    let array = batch.column(column_index);
    match array.data_type() {
        DataType::Int32 => {
            let values = values
                .iter()
                .map(|value| value.as_i32(column))
                .collect::<Result<Vec<_>>>()?;
            let values_array = array.as_any().downcast_ref::<Int32Array>().expect("Int32");
            let mut indices = Vec::new();
            for row in (0..values_array.len()).rev() {
                if !values_array.is_null(row)
                    && values.iter().any(|value| *value == values_array.value(row))
                {
                    indices.push(row as u32);
                }
            }
            Ok(Some(indices))
        }
        DataType::Int64 => {
            let values = values
                .iter()
                .map(|value| value.as_i64(column))
                .collect::<Result<Vec<_>>>()?;
            let values_array = array.as_any().downcast_ref::<Int64Array>().expect("Int64");
            let mut indices = Vec::new();
            for row in (0..values_array.len()).rev() {
                if !values_array.is_null(row)
                    && values.iter().any(|value| *value == values_array.value(row))
                {
                    indices.push(row as u32);
                }
            }
            Ok(Some(indices))
        }
        _ => Ok(None),
    }
}

fn reverse_primitive_in_list_ordered_batches_enabled() -> bool {
    std::env::var("DODAM_ENABLE_REVERSE_PRIMITIVE_IN_LIST_ORDERED_BATCHES")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

fn reverse_in_list_ordered_batches_enabled() -> bool {
    std::env::var("DODAM_ENABLE_REVERSE_IN_LIST_ORDERED_BATCHES")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

fn reverse_filter_ordered_batches_disabled() -> bool {
    std::env::var("DODAM_DISABLE_REVERSE_FILTER_ORDERED_BATCHES")
        .map(|value| !matches!(value.as_str(), "0" | "false" | "FALSE" | "no" | "NO"))
        .unwrap_or(true)
}

pub(super) fn row_group_ordered_sort_batch_size(default_batch_size: usize) -> usize {
    std::env::var("DODAM_ROW_GROUP_ORDERED_SORT_BATCH_SIZE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .map(|value| default_batch_size.max(value))
        .unwrap_or(default_batch_size)
}

fn reverse_ascending_primitive_batches_if_ordered(
    batches: &[RecordBatch],
    sort_column: &str,
) -> Result<Option<Vec<RecordBatch>>> {
    if row_group_primitive_reverse_materialization_disabled() {
        return Ok(None);
    }
    let mut previous_last: Option<i128> = None;
    for batch in batches {
        if !record_batch_supports_fast_reverse(batch) {
            return Ok(None);
        }
        let index = output_batch_column_index(batch, sort_column)?;
        let Some((first, last)) = numeric_column_ascending_bounds(batch.column(index))? else {
            return Ok(None);
        };
        if previous_last.is_some_and(|previous| previous > first) {
            return Ok(None);
        }
        previous_last = Some(last);
    }

    let mut output = Vec::with_capacity(batches.len());
    for batch in batches.iter().rev() {
        let Some(reversed) = reverse_primitive_record_batch_rows(batch)? else {
            return Ok(None);
        };
        output.push(reversed);
    }
    Ok(Some(output))
}

fn row_group_primitive_reverse_materialization_disabled() -> bool {
    std::env::var("DODAM_DISABLE_ROW_GROUP_PRIMITIVE_REVERSE_MATERIALIZATION")
        .map(|value| !matches!(value.as_str(), "0" | "false" | "FALSE" | "no" | "NO"))
        .unwrap_or(true)
}

fn row_group_ordered_desc_bulk_global_sort_enabled() -> bool {
    std::env::var("DODAM_ROW_GROUP_ORDERED_DESC_BULK_GLOBAL_SORT")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

pub(super) fn record_batch_supports_fast_reverse(batch: &RecordBatch) -> bool {
    batch
        .columns()
        .iter()
        .all(|column| primitive_array_supports_fast_reverse(column))
}

fn primitive_array_supports_fast_reverse(column: &ArrayRef) -> bool {
    column.null_count() == 0
        && matches!(
            column.data_type(),
            DataType::Int32
                | DataType::Int64
                | DataType::UInt64
                | DataType::Float64
                | DataType::Date32
        )
}

pub(super) fn reverse_primitive_record_batch_rows(
    batch: &RecordBatch,
) -> Result<Option<RecordBatch>> {
    let mut columns = Vec::with_capacity(batch.num_columns());
    for column in batch.columns() {
        let Some(reversed) = reverse_primitive_array(column) else {
            return Ok(None);
        };
        columns.push(reversed);
    }
    Ok(Some(RecordBatch::try_new(batch.schema(), columns)?))
}

fn reverse_primitive_array(column: &ArrayRef) -> Option<ArrayRef> {
    if column.null_count() != 0 {
        return None;
    }
    match column.data_type() {
        DataType::Int32 => {
            let values = column.as_any().downcast_ref::<Int32Array>()?;
            let mut output = Vec::with_capacity(values.len());
            for row in (0..values.len()).rev() {
                output.push(values.value(row));
            }
            Some(Arc::new(Int32Array::from(output)))
        }
        DataType::Int64 => {
            let values = column.as_any().downcast_ref::<Int64Array>()?;
            let mut output = Vec::with_capacity(values.len());
            for row in (0..values.len()).rev() {
                output.push(values.value(row));
            }
            Some(Arc::new(Int64Array::from(output)))
        }
        DataType::UInt64 => {
            let values = column.as_any().downcast_ref::<UInt64Array>()?;
            let mut output = Vec::with_capacity(values.len());
            for row in (0..values.len()).rev() {
                output.push(values.value(row));
            }
            Some(Arc::new(UInt64Array::from(output)))
        }
        DataType::Float64 => {
            let values = column.as_any().downcast_ref::<Float64Array>()?;
            let mut output = Vec::with_capacity(values.len());
            for row in (0..values.len()).rev() {
                output.push(values.value(row));
            }
            Some(Arc::new(Float64Array::from(output)))
        }
        DataType::Date32 => {
            let values = column.as_any().downcast_ref::<Date32Array>()?;
            let mut output = Vec::with_capacity(values.len());
            for row in (0..values.len()).rev() {
                output.push(values.value(row));
            }
            Some(Arc::new(Date32Array::from(output)))
        }
        _ => None,
    }
}

async fn try_reverse_row_group_desc_tail_topk(
    engine: &DodamEngine,
    shared: &SameSourceUnionAllScan,
    sort_column: &str,
    batch_size: usize,
    limit: usize,
) -> Result<Option<Vec<RecordBatch>>> {
    if limit > reverse_row_group_topk_max_limit_rows() {
        return Ok(None);
    }
    if !engine
        .parquet_row_groups_monotonic_by_column(shared.path.clone(), sort_column)
        .await?
    {
        return Ok(None);
    }
    let row_group_count = engine.parquet_row_group_count(&shared.path)?;
    if row_group_count == 0 {
        return Ok(Some(Vec::new()));
    }

    let mut scan_projection = shared.projection.clone();
    add_projection_columns(&mut scan_projection, shared.filter.referenced_columns());
    add_projection_columns(&mut scan_projection, vec![sort_column.to_string()]);

    let mut suffix = Vec::new();
    let mut suffix_rows = 0usize;
    for row_group in (0..row_group_count).rev() {
        let batches = engine
            .scan_parquet_row_group_batches(
                shared.path.clone(),
                batch_size,
                scan_projection.clone(),
                vec![row_group],
            )
            .await?;
        let mut filtered = Vec::new();
        for batch in batches {
            let batch = filter_batch(batch, &shared.filter)?;
            if batch.num_rows() > 0 {
                filtered.push(batch);
            }
        }
        suffix_rows += filtered.iter().map(RecordBatch::num_rows).sum::<usize>();
        suffix.splice(0..0, filtered);
        if suffix_rows >= limit {
            break;
        }
    }
    Ok(Some(suffix))
}

pub(super) fn reverse_row_group_topk_max_limit_rows() -> usize {
    std::env::var("DODAM_REVERSE_ROW_GROUP_TOPK_MAX_LIMIT_ROWS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(65_536)
}

fn collect_monotonic_desc_tail_topk(
    mut stream: SendableBatchStream,
    sort_column: &str,
    limit: usize,
) -> Result<Option<Vec<RecordBatch>>> {
    let mut tail = VecDeque::new();
    let mut tail_rows = 0usize;
    let mut previous_last = None;

    for batch in stream.by_ref() {
        let batch = batch?;
        if batch.num_rows() == 0 {
            continue;
        }
        let index = output_batch_column_index(&batch, sort_column)?;
        let column = batch.column(index);
        let Some((first, last)) = numeric_column_ascending_bounds(column)? else {
            return Ok(None);
        };
        if previous_last.is_some_and(|previous| previous > first) {
            return Ok(None);
        }
        previous_last = Some(last);
        tail_rows += batch.num_rows();
        tail.push_back(batch);
        while tail_rows > limit {
            let excess = tail_rows - limit;
            let front_rows = tail.front().map(RecordBatch::num_rows).unwrap_or(0);
            if excess >= front_rows {
                tail.pop_front();
                tail_rows -= front_rows;
            } else if let Some(front) = tail.pop_front() {
                let kept = front.slice(excess, front_rows - excess);
                tail_rows -= excess;
                tail.push_front(kept);
            }
        }
    }
    if tail.is_empty() {
        return Ok(Some(Vec::new()));
    }
    Ok(Some(tail.into_iter().collect()))
}

async fn try_same_source_union_all_streaming_desc_topk(
    engine: &DodamEngine,
    shared: &SameSourceUnionAllScan,
    sort_column: &str,
    batch_size: usize,
    limit: usize,
) -> Result<Option<Vec<RecordBatch>>> {
    if limit > reverse_row_group_topk_max_limit_rows() {
        return Ok(None);
    }

    let mut scan_projection = shared.projection.clone();
    add_projection_columns(&mut scan_projection, shared.filter.referenced_columns());
    add_projection_columns(&mut scan_projection, vec![sort_column.to_string()]);

    let mut stream = engine
        .scan_parquet_batches(
            shared.path.clone(),
            batch_size,
            None,
            scan_projection,
            Some(shared.filter.clone()),
        )
        .await?;
    let mut batches = Vec::new();
    let mut heap = BinaryHeap::<Reverse<(i128, u64, usize, u32)>>::with_capacity(limit + 1);
    let mut sequence = 0u64;
    let profile = ordered_sink_profile_enabled();
    let total_started = profile.then(Instant::now);
    let mut next_elapsed = Duration::ZERO;
    let mut heap_elapsed = Duration::ZERO;
    let mut selected_sort_elapsed = Duration::ZERO;
    let mut materialize_elapsed = Duration::ZERO;
    let mut scanned_batches = 0usize;
    let mut scanned_rows = 0usize;

    for batch in stream.by_ref() {
        let next_started = profile.then(Instant::now);
        let batch = batch?;
        if let Some(started) = next_started {
            next_elapsed += started.elapsed();
        }
        if batch.num_rows() == 0 {
            continue;
        }
        scanned_batches += 1;
        scanned_rows += batch.num_rows();
        let key_index = output_batch_column_index(&batch, sort_column)?;
        let key_column = batch.column(key_index);
        if !topk_sort_key_type_supported(key_column.data_type()) {
            return Ok(None);
        }
        let batch_index = batches.len();
        let heap_started = profile.then(Instant::now);
        if !update_streaming_desc_topk_heap(
            key_column,
            batch_index,
            limit,
            &mut sequence,
            &mut heap,
        )? {
            return Ok(None);
        }
        if let Some(started) = heap_started {
            heap_elapsed += started.elapsed();
        }
        batches.push(batch);
    }

    if heap.is_empty() {
        return Ok(Some(Vec::new()));
    }

    let mut selected = heap
        .into_iter()
        .map(|Reverse(item)| item)
        .collect::<Vec<_>>();
    let selected_sort_started = profile.then(Instant::now);
    selected.sort_unstable_by(|left, right| right.cmp(left));
    if let Some(started) = selected_sort_started {
        selected_sort_elapsed += started.elapsed();
    }
    let materialize_started = profile.then(Instant::now);
    let batch = materialize_topk_selected_rows(&batches, &selected)?;
    if let Some(started) = materialize_started {
        materialize_elapsed += started.elapsed();
    }
    print_streaming_topk_profile(
        total_started,
        next_elapsed,
        heap_elapsed,
        selected_sort_elapsed,
        materialize_elapsed,
        scanned_batches,
        scanned_rows,
        selected.len(),
    );
    Ok(Some(vec![batch]))
}

#[allow(clippy::too_many_arguments)]
fn print_streaming_topk_profile(
    total_started: Option<Instant>,
    next_elapsed: Duration,
    heap_elapsed: Duration,
    selected_sort_elapsed: Duration,
    materialize_elapsed: Duration,
    scanned_batches: usize,
    scanned_rows: usize,
    selected_rows: usize,
) {
    let Some(total_started) = total_started else {
        return;
    };
    eprintln!(
        "[dodam:streaming-topk-profile] total={}us next={}us heap={}us selected_sort={}us materialize={}us batches={} rows={} selected={}",
        total_started.elapsed().as_micros(),
        next_elapsed.as_micros(),
        heap_elapsed.as_micros(),
        selected_sort_elapsed.as_micros(),
        materialize_elapsed.as_micros(),
        scanned_batches,
        scanned_rows,
        selected_rows,
    );
}

#[allow(clippy::too_many_arguments)]
pub(super) fn try_write_same_source_union_all_streaming_primitive_topk_to_sink(
    engine: &DodamEngine,
    shared: &SameSourceUnionAllScan,
    batch_size: usize,
    scan_projection: &Projection,
    sort_column: &str,
    limit: usize,
    row_group_count: usize,
    sink: &mut dyn RecordBatchSink,
) -> Result<bool> {
    try_write_same_source_union_all_streaming_primitive_topk_to_sink_inner(
        engine,
        shared,
        batch_size,
        scan_projection,
        sort_column,
        limit,
        row_group_count,
        sink,
        true,
    )
}

#[allow(clippy::too_many_arguments)]
fn try_write_same_source_union_all_streaming_primitive_topk_to_sink_inner(
    engine: &DodamEngine,
    shared: &SameSourceUnionAllScan,
    batch_size: usize,
    scan_projection: &Projection,
    sort_column: &str,
    limit: usize,
    row_group_count: usize,
    sink: &mut dyn RecordBatchSink,
    allow_selected_payload: bool,
) -> Result<bool> {
    if limit > reverse_row_group_topk_max_limit_rows() {
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
    if *negated {
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
    let Some(sort_index) = columns.iter().position(|column| column == sort_column) else {
        return Ok(false);
    };
    let key_scan_columns = primitive_topk_key_columns(columns, filter_index, sort_index);
    let explicit_selected_payload = primitive_topk_selected_payload_enabled();
    let auto_selected_payload =
        primitive_topk_selected_payload_auto_enabled() && !explicit_selected_payload;
    let use_selected_payload = allow_selected_payload
        && (explicit_selected_payload || auto_selected_payload)
        && primitive_topk_selected_payload_precheck_accepts(
            engine,
            &shared.path,
            sort_column,
            limit,
            row_group_count,
            columns,
            &key_scan_columns,
            auto_selected_payload,
        )?;
    let scan_columns = if use_selected_payload {
        key_scan_columns
    } else {
        columns.clone()
    };
    let scan_column_types = scan_columns
        .iter()
        .map(|column| {
            let Some(index) = columns.iter().position(|candidate| candidate == column) else {
                return Err(DodamError::UnsupportedSql(format!(
                    "primitive top-k scan column {column} is not projected"
                )));
            };
            Ok(column_types[index])
        })
        .collect::<Result<Vec<_>>>()?;
    let Some(scan_filter_index) = scan_columns
        .iter()
        .position(|column| column == filter_column)
    else {
        return Ok(false);
    };
    let Some(scan_sort_index) = scan_columns.iter().position(|column| column == sort_column) else {
        return Ok(false);
    };
    let scan_filter_values = match scan_column_types[scan_filter_index] {
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
    let row_groups = (0..row_group_count).collect::<Vec<_>>();
    let profile = ordered_sink_profile_enabled();
    let total_started = profile.then(Instant::now);
    if primitive_topk_fused_selected_page_reader_enabled() && !use_selected_payload {
        let filter_i32_values;
        let filter_i64_values;
        let (filter_i32, filter_i64) = match &scan_filter_values {
            PrimitiveFilterValues::I32(values) => {
                filter_i32_values = values.clone();
                (&filter_i32_values[..], &[][..])
            }
            PrimitiveFilterValues::I64(values) => {
                filter_i64_values = values.clone();
                (&[][..], &filter_i64_values[..])
            }
        };
        let specs = columns
            .iter()
            .zip(column_types.iter())
            .map(|(name, column_type)| DirectPrimitiveColumnSpec {
                name,
                column_type: *column_type,
            })
            .collect::<Vec<_>>();
        let mut state = PrimitiveSelectedBatchTopkState::new(limit, &column_types);
        let mut metrics = DirectPrimitiveColumnScanMetrics::default();
        let mut supported = true;
        let chunk_size = same_source_union_primitive_chunk_size(row_group_count);
        for chunk in row_groups.chunks(chunk_size) {
            let scan_results = std::thread::scope(|scope| {
                let mut handles = Vec::with_capacity(chunk.len());
                for (position, row_group) in chunk.iter().copied().enumerate() {
                    let engine = engine.clone();
                    let path = shared.path.clone();
                    let specs = specs.clone();
                    let column_types = column_types.clone();
                    let columns = columns.to_vec();
                    handles.push(scope.spawn(move || {
                        let mut local_state =
                            PrimitiveSelectedBatchTopkState::new(limit, &column_types);
                        let metrics = engine
                            .scan_parquet_required_plain_primitive_in_list_desc_selected_pages(
                                &path,
                                batch_size,
                                &[row_group],
                                &specs,
                                filter_index,
                                filter_i32,
                                filter_i64,
                                |batch| {
                                    local_state.consume_selected_page(
                                        batch,
                                        &column_types,
                                        sort_index,
                                    )
                                },
                            )?;
                        let batch = if local_state.unsupported {
                            None
                        } else {
                            Some(local_state.into_primitive_batch(&columns, &column_types)?)
                                .filter(|batch| !batch.is_empty())
                        };
                        Ok::<_, DodamError>((position, batch, metrics))
                    }));
                }
                let mut results = Vec::with_capacity(handles.len());
                for handle in handles {
                    match handle.join() {
                        Ok(result) => results.push(result?),
                        Err(_) => {
                            return Err(DodamError::UnsupportedSql(
                                "primitive top-k fused selected page worker panicked".to_string(),
                            ));
                        }
                    }
                }
                Ok::<_, DodamError>(results)
            })?;
            let mut scan_results = scan_results;
            scan_results.sort_by_key(|(position, _, _)| *position);
            for (_, batch, row_group_metrics) in scan_results {
                let Some(row_group_metrics) = row_group_metrics else {
                    supported = false;
                    break;
                };
                metrics.merge_from(row_group_metrics);
                if let Some(batch) = batch {
                    state.consume_primitive_batch(&batch, &column_types, sort_index)?;
                }
            }
            if !supported || state.unsupported {
                break;
            }
        }
        if supported && !state.unsupported {
            let batch = state.into_primitive_batch(columns, &column_types)?;
            print_streaming_primitive_topk_profile(total_started, &metrics, batch.num_rows());
            write_same_source_primitive_batch_to_sink(batch, scan_projection, shared, sink)?;
            return Ok(true);
        }
    }
    let Some((state, metrics)) = engine
        .scan_parquet_primitive_columns_parallel_view_fold_with_location(
            shared.path.clone(),
            batch_size,
            row_groups,
            scan_columns
                .iter()
                .zip(scan_column_types.iter())
                .map(|(name, column_type)| (name.clone(), *column_type))
                .collect(),
            || PrimitiveTopkState::new(limit, &scan_column_types, use_selected_payload),
            |state, location, view| {
                state.consume_view(
                    location,
                    view,
                    &scan_column_types,
                    scan_filter_index,
                    &scan_filter_values,
                    scan_sort_index,
                )
            },
            PrimitiveTopkState::merge,
        )?
    else {
        return Ok(false);
    };
    if state.unsupported {
        return Ok(false);
    }
    let batch = if use_selected_payload {
        if let Some(row_refs) = state.selected_row_refs_sorted() {
            let base_batch = state.into_primitive_batch(&scan_columns, &scan_column_types)?;
            if primitive_topk_selected_payload_spread_accepts(
                &row_refs,
                row_group_count,
                columns,
                &scan_columns,
            ) || !primitive_topk_selected_payload_spread_gate_enabled()
            {
                read_primitive_topk_selected_payload_with_base(
                    engine,
                    &shared.path,
                    batch_size,
                    row_refs,
                    base_batch,
                    &scan_columns,
                    &scan_column_types,
                    columns,
                    &column_types,
                )?
            } else {
                return try_write_same_source_union_all_streaming_primitive_topk_to_sink_inner(
                    engine,
                    shared,
                    batch_size,
                    scan_projection,
                    sort_column,
                    limit,
                    row_group_count,
                    sink,
                    false,
                );
            }
        } else {
            state.into_primitive_batch(&scan_columns, &scan_column_types)?
        }
    } else {
        state.into_primitive_batch(columns, &column_types)?
    };
    print_streaming_primitive_topk_profile(total_started, &metrics, batch.num_rows());
    write_same_source_primitive_batch_to_sink(batch, scan_projection, shared, sink)?;
    Ok(true)
}

fn primitive_topk_key_columns(
    columns: &[String],
    filter_index: usize,
    sort_index: usize,
) -> Vec<String> {
    let mut scan_columns = Vec::with_capacity(2);
    scan_columns.push(columns[filter_index].clone());
    if sort_index != filter_index {
        scan_columns.push(columns[sort_index].clone());
    }
    scan_columns
}

fn primitive_topk_selected_payload_enabled() -> bool {
    std::env::var("DODAM_ENABLE_PRIMITIVE_TOPK_SELECTED_PAYLOAD")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

fn primitive_topk_selected_payload_auto_enabled() -> bool {
    if std::env::var("DODAM_DISABLE_PRIMITIVE_TOPK_SELECTED_PAYLOAD_AUTO")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
    {
        return false;
    }
    std::env::var("DODAM_ENABLE_PRIMITIVE_TOPK_SELECTED_PAYLOAD_AUTO")
        .map(|value| !matches!(value.as_str(), "0" | "false" | "FALSE" | "no" | "NO"))
        .unwrap_or(true)
}

fn primitive_topk_fused_selected_page_reader_enabled() -> bool {
    std::env::var("DODAM_ENABLE_PRIMITIVE_TOPK_FUSED_SELECTED_PAGE_READER")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

fn primitive_topk_selected_payload_precheck_accepts(
    engine: &DodamEngine,
    path: &Path,
    sort_column: &str,
    limit: usize,
    total_row_groups: usize,
    output_columns: &[String],
    base_columns: &[String],
    require_stats: bool,
) -> Result<bool> {
    let missing_payload_columns =
        primitive_topk_missing_payload_columns(output_columns, base_columns);
    if missing_payload_columns == 0 {
        log_primitive_topk_selected_payload_spread(
            limit,
            0,
            total_row_groups,
            missing_payload_columns,
            SelectedPayloadDecision::EmptySelection,
            "precheck",
        );
        return Ok(true);
    }
    if require_stats && missing_payload_columns < primitive_topk_selected_payload_min_auto_columns()
    {
        log_primitive_topk_selected_payload_spread(
            limit,
            0,
            total_row_groups,
            missing_payload_columns,
            SelectedPayloadDecision::PayloadColumns,
            "precheck",
        );
        return Ok(false);
    }
    let Some(ranges) = engine.parquet_primitive_column_min_max_by_row_group(path, sort_column)?
    else {
        return Ok(!require_stats);
    };
    let Some(candidate_row_groups) =
        estimate_desc_topk_candidate_row_groups_from_stats(&ranges, limit)
    else {
        return Ok(!require_stats);
    };
    let decision = choose_selected_payload_by_spread(SelectedPayloadSpreadCostInput {
        selected_rows: limit,
        selected_row_groups: candidate_row_groups,
        total_row_groups,
        missing_payload_columns,
        max_selected_row_group_ratio: primitive_topk_selected_payload_max_row_group_ratio(),
        max_selected_row_groups: primitive_topk_selected_payload_max_row_groups(),
    });
    log_primitive_topk_selected_payload_spread(
        limit,
        candidate_row_groups,
        total_row_groups,
        missing_payload_columns,
        decision,
        "precheck",
    );
    Ok(decision.accepted())
}

fn estimate_desc_topk_candidate_row_groups_from_stats(
    ranges: &[PrimitiveRowGroupMinMax],
    limit: usize,
) -> Option<usize> {
    if limit == 0 || ranges.is_empty() {
        return Some(0);
    }
    let mut sorted = ranges.to_vec();
    sorted.sort_unstable_by(|left, right| {
        right
            .max
            .cmp(&left.max)
            .then_with(|| right.min.cmp(&left.min))
            .then_with(|| left.row_group.cmp(&right.row_group))
    });
    let mut rows = 0usize;
    let mut threshold_min = None;
    for range in &sorted {
        rows = rows.saturating_add(range.rows);
        threshold_min =
            Some(threshold_min.map_or(range.min, |current: i128| current.min(range.min)));
        if rows >= limit {
            break;
        }
    }
    let threshold_min = threshold_min?;
    Some(
        ranges
            .iter()
            .filter(|range| range.max >= threshold_min)
            .count(),
    )
}

fn primitive_topk_selected_payload_spread_accepts(
    row_refs: &[PrimitiveTopkRowRef],
    total_row_groups: usize,
    output_columns: &[String],
    base_columns: &[String],
) -> bool {
    let missing_payload_columns =
        primitive_topk_missing_payload_columns(output_columns, base_columns);
    let mut row_groups = FastHashSet::default();
    for row_ref in row_refs {
        row_groups.insert(row_ref.row_group);
    }
    let decision = choose_selected_payload_by_spread(SelectedPayloadSpreadCostInput {
        selected_rows: row_refs.len(),
        selected_row_groups: row_groups.len(),
        total_row_groups,
        missing_payload_columns,
        max_selected_row_group_ratio: primitive_topk_selected_payload_max_row_group_ratio(),
        max_selected_row_groups: primitive_topk_selected_payload_max_row_groups(),
    });
    log_primitive_topk_selected_payload_spread(
        row_refs.len(),
        row_groups.len(),
        total_row_groups,
        missing_payload_columns,
        decision,
        "actual",
    );
    decision.accepted()
}

fn primitive_topk_missing_payload_columns(
    output_columns: &[String],
    base_columns: &[String],
) -> usize {
    output_columns
        .iter()
        .filter(|column| !base_columns.contains(*column))
        .count()
}

fn primitive_topk_selected_payload_max_row_group_ratio() -> f64 {
    std::env::var("DODAM_PRIMITIVE_TOPK_SELECTED_PAYLOAD_MAX_ROW_GROUP_RATIO")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite())
        .unwrap_or(0.25)
}

fn primitive_topk_selected_payload_max_row_groups() -> usize {
    std::env::var("DODAM_PRIMITIVE_TOPK_SELECTED_PAYLOAD_MAX_ROW_GROUPS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(16)
}

fn primitive_topk_selected_payload_min_auto_columns() -> usize {
    std::env::var("DODAM_PRIMITIVE_TOPK_SELECTED_PAYLOAD_MIN_AUTO_COLUMNS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(2)
}

fn primitive_topk_selected_payload_spread_gate_enabled() -> bool {
    if std::env::var("DODAM_DISABLE_PRIMITIVE_TOPK_SELECTED_PAYLOAD_SPREAD_GATE")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
    {
        return false;
    }
    std::env::var("DODAM_ENABLE_PRIMITIVE_TOPK_SELECTED_PAYLOAD_SPREAD_GATE")
        .map(|value| !matches!(value.as_str(), "0" | "false" | "FALSE" | "no" | "NO"))
        .unwrap_or(true)
}

fn log_primitive_topk_selected_payload_spread(
    selected_rows: usize,
    selected_row_groups: usize,
    total_row_groups: usize,
    missing_payload_columns: usize,
    decision: SelectedPayloadDecision,
    phase: &str,
) {
    if !std::env::var("DODAM_PRIMITIVE_TOPK_SELECTED_PAYLOAD_TRACE")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
    {
        return;
    }
    eprintln!(
        "[dodam:primitive-topk-selected-payload] phase={} decision={} selected_rows={} row_groups={}/{} missing_payload_columns={}",
        phase,
        decision.reason(),
        selected_rows,
        selected_row_groups,
        total_row_groups,
        missing_payload_columns
    );
}

fn read_primitive_topk_selected_payload(
    engine: &DodamEngine,
    path: &Path,
    batch_size: usize,
    row_refs: Vec<PrimitiveTopkRowRef>,
    column_names: &[String],
    column_types: &[DirectPrimitiveColumnType],
) -> Result<PrimitiveBatch> {
    if row_refs.is_empty() {
        return primitive_empty_batch(column_names, column_types);
    }
    let mut refs_by_row_group = FastHashMap::<usize, Vec<(usize, usize)>>::default();
    for (output_index, row_ref) in row_refs.iter().copied().enumerate() {
        refs_by_row_group
            .entry(row_ref.row_group)
            .or_default()
            .push((row_ref.row_offset, output_index));
    }
    let mut row_groups = refs_by_row_group.keys().copied().collect::<Vec<_>>();
    row_groups.sort_unstable();
    for refs in refs_by_row_group.values_mut() {
        refs.sort_unstable_by_key(|(row_offset, _)| *row_offset);
    }
    let refs_by_row_group = Arc::new(refs_by_row_group);
    let rows = row_refs.len();
    let Some((state, _metrics)) = engine
        .scan_parquet_primitive_columns_parallel_view_fold_with_location(
            path.to_path_buf(),
            batch_size,
            row_groups,
            column_names
                .iter()
                .zip(column_types.iter())
                .map(|(name, column_type)| (name.clone(), *column_type))
                .collect(),
            || SelectedPrimitivePayloadState::new(rows, column_types),
            {
                let refs_by_row_group = Arc::clone(&refs_by_row_group);
                move |state: &mut SelectedPrimitivePayloadState, location, view| {
                    state.consume(location, view, column_types, refs_by_row_group.as_ref())
                }
            },
            SelectedPrimitivePayloadState::merge,
        )?
    else {
        return Err(DodamError::UnsupportedSql(
            "primitive top-k selected payload reader is unsupported".to_string(),
        ));
    };
    state.into_batch(column_names, column_types)
}

#[allow(clippy::too_many_arguments)]
fn read_primitive_topk_selected_payload_with_base(
    engine: &DodamEngine,
    path: &Path,
    batch_size: usize,
    row_refs: Vec<PrimitiveTopkRowRef>,
    base_batch: PrimitiveBatch,
    base_column_names: &[String],
    base_column_types: &[DirectPrimitiveColumnType],
    output_column_names: &[String],
    output_column_types: &[DirectPrimitiveColumnType],
) -> Result<PrimitiveBatch> {
    if output_column_names.len() != output_column_types.len()
        || base_column_names.len() != base_column_types.len()
    {
        return Err(DodamError::UnsupportedSql(
            "primitive top-k selected payload schema mismatch".to_string(),
        ));
    }
    let mut base_columns = base_batch
        .columns
        .into_iter()
        .map(|column| (column.name.clone(), column))
        .collect::<FastHashMap<_, _>>();
    let mut missing_names = Vec::new();
    let mut missing_types = Vec::new();
    for (name, column_type) in output_column_names.iter().zip(output_column_types.iter()) {
        if base_columns.contains_key(name) {
            continue;
        }
        missing_names.push(name.clone());
        missing_types.push(*column_type);
    }
    let mut missing_columns = if missing_names.is_empty() {
        FastHashMap::default()
    } else {
        read_primitive_topk_selected_payload(
            engine,
            path,
            batch_size,
            row_refs,
            &missing_names,
            &missing_types,
        )?
        .columns
        .into_iter()
        .map(|column| (column.name.clone(), column))
        .collect::<FastHashMap<_, _>>()
    };
    let mut columns = Vec::with_capacity(output_column_names.len());
    for (name, column_type) in output_column_names.iter().zip(output_column_types.iter()) {
        let column = if let Some(column) = base_columns.remove(name) {
            column
        } else if let Some(column) = missing_columns.remove(name) {
            column
        } else {
            return Err(DodamError::UnsupportedSql(format!(
                "primitive top-k selected payload column {name} was not materialized"
            )));
        };
        if !primitive_column_matches_direct_type(&column, column_type) {
            return Err(DodamError::UnsupportedSql(format!(
                "primitive top-k selected payload column {name} type mismatch"
            )));
        }
        columns.push(column);
    }
    Ok(PrimitiveBatch { columns })
}

struct SelectedPrimitivePayloadState {
    columns: Vec<PrimitiveColumnOutput>,
    filled: Vec<bool>,
    unsupported: bool,
}

impl SelectedPrimitivePayloadState {
    fn new(rows: usize, column_types: &[DirectPrimitiveColumnType]) -> Self {
        let columns = column_types
            .iter()
            .map(|column_type| match column_type {
                DirectPrimitiveColumnType::I32 => PrimitiveColumnOutput::I32(vec![0; rows]),
                DirectPrimitiveColumnType::I64 => PrimitiveColumnOutput::I64(vec![0; rows]),
                _ => unreachable!("primitive selected payload only uses i32/i64 columns"),
            })
            .collect();
        Self {
            columns,
            filled: vec![false; rows],
            unsupported: false,
        }
    }

    fn consume(
        &mut self,
        location: DirectPrimitiveBatchLocation,
        view: BatchView<'_>,
        column_types: &[DirectPrimitiveColumnType],
        refs_by_row_group: &FastHashMap<usize, Vec<(usize, usize)>>,
    ) -> Result<()> {
        let Some(refs) = refs_by_row_group.get(&location.row_group) else {
            return Ok(());
        };
        let batch_start = location.row_offset;
        let batch_end = batch_start.saturating_add(view.num_rows());
        let first = refs.partition_point(|(row_offset, _)| *row_offset < batch_start);
        let mut index = first;
        if index >= refs.len() || refs[index].0 >= batch_end {
            return Ok(());
        }
        let Some(columns) = null_free_primitive_columns_for_topk(view, column_types) else {
            self.unsupported = true;
            return Ok(());
        };
        while index < refs.len() {
            let (row_offset, output_index) = refs[index];
            if row_offset >= batch_end {
                break;
            }
            let local_row = row_offset - batch_start;
            for (source, target) in columns.iter().zip(self.columns.iter_mut()) {
                overwrite_null_free_primitive_value(source, local_row, target, output_index)?;
            }
            self.filled[output_index] = true;
            index += 1;
        }
        Ok(())
    }

    fn merge(&mut self, source: Self) -> Result<()> {
        self.unsupported |= source.unsupported;
        for (index, filled) in source.filled.iter().copied().enumerate() {
            if !filled {
                continue;
            }
            for (source_column, target) in source.columns.iter().zip(self.columns.iter_mut()) {
                overwrite_primitive_output_slot(source_column, index, target, index)?;
            }
            self.filled[index] = true;
        }
        Ok(())
    }

    fn into_batch(
        self,
        column_names: &[String],
        column_types: &[DirectPrimitiveColumnType],
    ) -> Result<PrimitiveBatch> {
        if self.unsupported || self.filled.iter().any(|filled| !*filled) {
            return Err(DodamError::UnsupportedSql(
                "primitive top-k selected payload reader did not fill all rows".to_string(),
            ));
        }
        let columns = self
            .columns
            .into_iter()
            .zip(column_names.iter())
            .zip(column_types.iter())
            .map(|((values, name), column_type)| {
                let values = match values {
                    PrimitiveColumnOutput::I32(values) => PrimitiveColumnValues::I32(values),
                    PrimitiveColumnOutput::I64(values) => PrimitiveColumnValues::I64(values),
                };
                Ok(PrimitiveColumn {
                    name: name.clone(),
                    data_type: primitive_output_data_type(column_type)?,
                    nullable: false,
                    values,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(PrimitiveBatch { columns })
    }
}

fn print_streaming_primitive_topk_profile(
    total_started: Option<Instant>,
    metrics: &DirectPrimitiveColumnScanMetrics,
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
        "[dodam:streaming-primitive-topk-profile] total={}us read={:.3}ms consume={:.3}ms row_groups={} batches={} scanned_rows={} output_rows={} column_read_ms=[{}]",
        total_started.elapsed().as_micros(),
        (metrics.read_nanos as f64) / 1_000_000.0,
        (metrics.consume_nanos as f64) / 1_000_000.0,
        metrics.row_groups,
        metrics.batches,
        metrics.rows,
        rows,
        column_read,
    );
}

struct PrimitiveTopkState {
    limit: usize,
    unsupported: bool,
    sequence: u64,
    heap: BinaryHeap<Reverse<(i128, u64, usize)>>,
    columns: Vec<PrimitiveColumnOutput>,
    row_refs: Option<Vec<PrimitiveTopkRowRef>>,
    selected_positions: Vec<usize>,
}

struct PrimitiveSelectedBatchTopkState {
    limit: usize,
    unsupported: bool,
    sequence: u64,
    heap: BinaryHeap<Reverse<(i128, u64, usize)>>,
    columns: Vec<PrimitiveColumnOutput>,
}

#[derive(Clone, Copy)]
struct PrimitiveTopkRowRef {
    row_group: usize,
    row_offset: usize,
}

impl PrimitiveSelectedBatchTopkState {
    fn new(limit: usize, column_types: &[DirectPrimitiveColumnType]) -> Self {
        let columns = column_types
            .iter()
            .map(|column_type| match column_type {
                DirectPrimitiveColumnType::I32 => {
                    PrimitiveColumnOutput::I32(Vec::with_capacity(limit))
                }
                DirectPrimitiveColumnType::I64 => {
                    PrimitiveColumnOutput::I64(Vec::with_capacity(limit))
                }
                _ => unreachable!("primitive selected top-k only uses i32/i64 columns"),
            })
            .collect();
        Self {
            limit,
            unsupported: false,
            sequence: 0,
            heap: BinaryHeap::with_capacity(limit.saturating_add(1)),
            columns,
        }
    }

    fn consume_selected_page(
        &mut self,
        batch: DirectSelectedPrimitivePageBatch<'_>,
        column_types: &[DirectPrimitiveColumnType],
        sort_index: usize,
    ) -> Result<()> {
        if sort_index >= batch.columns.len()
            || batch.columns.len() != column_types.len()
            || batch.columns.is_empty()
        {
            self.unsupported = true;
            return Ok(());
        }
        for &row in batch.selected_positions {
            let Some(key) = batch.columns[sort_index].value_i128(row) else {
                self.unsupported = true;
                return Ok(());
            };
            self.insert_candidate_page(key, &batch.columns, column_types, row)?;
            self.sequence = self.sequence.wrapping_add(1);
        }
        Ok(())
    }

    fn consume_primitive_batch(
        &mut self,
        batch: &PrimitiveBatch,
        column_types: &[DirectPrimitiveColumnType],
        sort_index: usize,
    ) -> Result<()> {
        if sort_index >= batch.columns.len()
            || batch.columns.len() != column_types.len()
            || batch.columns.is_empty()
        {
            self.unsupported = true;
            return Ok(());
        }
        let rows = batch.num_rows();
        for column in &batch.columns {
            if column.values.len() != rows {
                self.unsupported = true;
                return Ok(());
            }
        }
        for row in 0..rows {
            let Some(key) = primitive_column_values_key(&batch.columns[sort_index].values, row)
            else {
                self.unsupported = true;
                return Ok(());
            };
            self.insert_candidate_primitive_batch(key, &batch.columns, column_types, row)?;
            self.sequence = self.sequence.wrapping_add(1);
        }
        Ok(())
    }

    fn insert_candidate_page(
        &mut self,
        key: i128,
        columns: &[DirectSelectedPrimitiveColumnPageView<'_>],
        column_types: &[DirectPrimitiveColumnType],
        row: usize,
    ) -> Result<()> {
        let sequence = self.sequence;
        if self.heap.len() < self.limit {
            let slot = self.push_slot_from_page(columns, column_types, row)?;
            self.heap.push(Reverse((key, sequence, slot)));
            return Ok(());
        }
        if let Some(worst) = self.heap.peek()
            && key < worst.0.0
        {
            return Ok(());
        }
        let mut replace_slot = None;
        {
            if let Some(mut worst) = self.heap.peek_mut()
                && (key, sequence, 0usize) > (worst.0.0, worst.0.1, 0usize)
            {
                replace_slot = Some(worst.0.2);
                *worst = Reverse((key, sequence, worst.0.2));
            }
        }
        if let Some(slot) = replace_slot {
            self.overwrite_slot_from_page(slot, columns, column_types, row)?;
        }
        Ok(())
    }

    fn insert_candidate_primitive_batch(
        &mut self,
        key: i128,
        columns: &[PrimitiveColumn],
        column_types: &[DirectPrimitiveColumnType],
        row: usize,
    ) -> Result<()> {
        let sequence = self.sequence;
        if self.heap.len() < self.limit {
            let slot = self.push_slot_from_primitive_batch(columns, column_types, row)?;
            self.heap.push(Reverse((key, sequence, slot)));
            return Ok(());
        }
        if let Some(worst) = self.heap.peek()
            && key < worst.0.0
        {
            return Ok(());
        }
        let mut replace_slot = None;
        {
            if let Some(mut worst) = self.heap.peek_mut()
                && (key, sequence, 0usize) > (worst.0.0, worst.0.1, 0usize)
            {
                replace_slot = Some(worst.0.2);
                *worst = Reverse((key, sequence, worst.0.2));
            }
        }
        if let Some(slot) = replace_slot {
            self.overwrite_slot_from_primitive_batch(slot, columns, column_types, row)?;
        }
        Ok(())
    }

    fn push_slot_from_page(
        &mut self,
        columns: &[DirectSelectedPrimitiveColumnPageView<'_>],
        column_types: &[DirectPrimitiveColumnType],
        row: usize,
    ) -> Result<usize> {
        let slot = primitive_output_len(&self.columns[0]);
        for ((source, column_type), target) in columns
            .iter()
            .zip(column_types.iter())
            .zip(self.columns.iter_mut())
        {
            push_direct_selected_page_value(source, column_type, row, target)?;
        }
        Ok(slot)
    }

    fn overwrite_slot_from_page(
        &mut self,
        slot: usize,
        columns: &[DirectSelectedPrimitiveColumnPageView<'_>],
        column_types: &[DirectPrimitiveColumnType],
        row: usize,
    ) -> Result<()> {
        for ((source, column_type), target) in columns
            .iter()
            .zip(column_types.iter())
            .zip(self.columns.iter_mut())
        {
            overwrite_direct_selected_page_value(source, column_type, row, target, slot)?;
        }
        Ok(())
    }

    fn push_slot_from_primitive_batch(
        &mut self,
        columns: &[PrimitiveColumn],
        column_types: &[DirectPrimitiveColumnType],
        row: usize,
    ) -> Result<usize> {
        let slot = primitive_output_len(&self.columns[0]);
        for ((source, column_type), target) in columns
            .iter()
            .zip(column_types.iter())
            .zip(self.columns.iter_mut())
        {
            push_primitive_batch_value(&source.values, column_type, row, target)?;
        }
        Ok(slot)
    }

    fn overwrite_slot_from_primitive_batch(
        &mut self,
        slot: usize,
        columns: &[PrimitiveColumn],
        column_types: &[DirectPrimitiveColumnType],
        row: usize,
    ) -> Result<()> {
        for ((source, column_type), target) in columns
            .iter()
            .zip(column_types.iter())
            .zip(self.columns.iter_mut())
        {
            overwrite_primitive_batch_value(&source.values, column_type, row, target, slot)?;
        }
        Ok(())
    }

    fn into_primitive_batch(
        self,
        column_names: &[String],
        column_types: &[DirectPrimitiveColumnType],
    ) -> Result<PrimitiveBatch> {
        let mut selected = self
            .heap
            .into_iter()
            .map(|Reverse(item)| item)
            .collect::<Vec<_>>();
        selected.sort_unstable_by(|left, right| right.cmp(left));
        let mut output = column_types
            .iter()
            .map(|column_type| match column_type {
                DirectPrimitiveColumnType::I32 => {
                    PrimitiveColumnOutput::I32(Vec::with_capacity(selected.len()))
                }
                DirectPrimitiveColumnType::I64 => {
                    PrimitiveColumnOutput::I64(Vec::with_capacity(selected.len()))
                }
                _ => unreachable!("primitive selected top-k only uses i32/i64 columns"),
            })
            .collect::<Vec<_>>();
        for (_, _, slot) in selected {
            for (source, target) in self.columns.iter().zip(output.iter_mut()) {
                push_primitive_output_slot(source, slot, target)?;
            }
        }
        primitive_output_batch_from_columns(output, column_names, column_types)
    }
}

impl PrimitiveTopkState {
    fn new(limit: usize, column_types: &[DirectPrimitiveColumnType], track_row_refs: bool) -> Self {
        let columns = column_types
            .iter()
            .map(|column_type| match column_type {
                DirectPrimitiveColumnType::I32 => {
                    PrimitiveColumnOutput::I32(Vec::with_capacity(limit))
                }
                DirectPrimitiveColumnType::I64 => {
                    PrimitiveColumnOutput::I64(Vec::with_capacity(limit))
                }
                _ => unreachable!("primitive top-k only uses i32/i64 columns"),
            })
            .collect();
        Self {
            limit,
            unsupported: false,
            sequence: 0,
            heap: BinaryHeap::with_capacity(limit.saturating_add(1)),
            columns,
            row_refs: (track_row_refs
                || primitive_topk_row_refs_enabled()
                || primitive_topk_selected_payload_enabled())
            .then(|| Vec::with_capacity(limit)),
            selected_positions: Vec::new(),
        }
    }

    fn consume_view(
        &mut self,
        location: DirectPrimitiveBatchLocation,
        view: BatchView<'_>,
        column_types: &[DirectPrimitiveColumnType],
        filter_index: usize,
        filter_values: &PrimitiveFilterValues,
        sort_index: usize,
    ) -> Result<()> {
        let Some(columns) = null_free_primitive_columns_for_topk(view, column_types) else {
            self.unsupported = true;
            return Ok(());
        };
        let threshold_key =
            if primitive_topk_fused_filter_threshold_enabled() && self.heap.len() >= self.limit {
                self.heap.peek().map(|worst| worst.0.0)
            } else {
                None
            };
        if let Some(threshold_key) = threshold_key {
            primitive_topk_filter_positions_with_min_key_into(
                columns[filter_index],
                filter_values,
                columns[sort_index],
                threshold_key,
                &mut self.selected_positions,
            );
        } else {
            primitive_topk_filter_positions_into(
                columns[filter_index],
                filter_values,
                &mut self.selected_positions,
            );
        }
        let base_sequence = primitive_topk_sequence_base(location).unwrap_or(self.sequence);
        self.sequence = self.sequence.wrapping_add(view.num_rows() as u64);
        for index in 0..self.selected_positions.len() {
            let row = self.selected_positions[index];
            let Some(key) = primitive_topk_key(columns[sort_index], row) else {
                return Err(DodamError::UnsupportedSql(
                    "streaming primitive top-k sort column type mismatch".to_string(),
                ));
            };
            self.insert_candidate_with_sequence(
                key,
                base_sequence.wrapping_add(row as u64),
                &columns,
                PrimitiveTopkRowRef {
                    row_group: location.row_group,
                    row_offset: location.row_offset.saturating_add(row),
                },
                row,
            )?;
        }
        self.selected_positions.clear();
        Ok(())
    }

    fn insert_candidate_with_sequence(
        &mut self,
        key: i128,
        sequence: u64,
        columns: &[NullFreePrimitiveColumn<'_>],
        row_ref: PrimitiveTopkRowRef,
        row: usize,
    ) -> Result<()> {
        if self.heap.len() < self.limit {
            let slot = self.push_slot_from_columns(columns, row_ref, row)?;
            self.heap.push(Reverse((key, sequence, slot)));
            return Ok(());
        }
        if let Some(worst) = self.heap.peek()
            && key < worst.0.0
        {
            return Ok(());
        }
        let mut replace_slot = None;
        {
            if let Some(mut worst) = self.heap.peek_mut()
                && (key, sequence, 0usize) > (worst.0.0, worst.0.1, 0usize)
            {
                replace_slot = Some(worst.0.2);
                *worst = Reverse((key, sequence, worst.0.2));
            }
        }
        if let Some(slot) = replace_slot {
            self.overwrite_slot_from_columns(slot, columns, row_ref, row)?;
        }
        Ok(())
    }

    fn merge(&mut self, other: Self) -> Result<()> {
        self.unsupported |= other.unsupported;
        if other.unsupported {
            return Ok(());
        }
        let mut selected = other
            .heap
            .iter()
            .map(|Reverse(item)| *item)
            .collect::<Vec<_>>();
        selected.sort_unstable_by(|left, right| right.cmp(left));
        for (key, _, slot) in selected {
            self.insert_candidate_from_state(key, &other, slot)?;
        }
        Ok(())
    }

    fn insert_candidate_from_state(
        &mut self,
        key: i128,
        source: &PrimitiveTopkState,
        source_slot: usize,
    ) -> Result<()> {
        let sequence = self.sequence;
        self.sequence = self.sequence.wrapping_add(1);
        if self.heap.len() < self.limit {
            let slot = self.push_slot_from_state(source, source_slot)?;
            self.heap.push(Reverse((key, sequence, slot)));
            return Ok(());
        }
        if let Some(worst) = self.heap.peek()
            && key < worst.0.0
        {
            return Ok(());
        }
        let mut replace_slot = None;
        {
            if let Some(mut worst) = self.heap.peek_mut()
                && (key, sequence, 0usize) > (worst.0.0, worst.0.1, 0usize)
            {
                replace_slot = Some(worst.0.2);
                *worst = Reverse((key, sequence, worst.0.2));
            }
        }
        if let Some(slot) = replace_slot {
            self.overwrite_slot_from_state(slot, source, source_slot)?;
        }
        Ok(())
    }

    fn into_primitive_batch(
        self,
        column_names: &[String],
        column_types: &[DirectPrimitiveColumnType],
    ) -> Result<PrimitiveBatch> {
        let mut selected = self
            .heap
            .into_iter()
            .map(|Reverse(item)| item)
            .collect::<Vec<_>>();
        selected.sort_unstable_by(|left, right| right.cmp(left));
        let mut output = column_types
            .iter()
            .map(|column_type| match column_type {
                DirectPrimitiveColumnType::I32 => {
                    PrimitiveColumnOutput::I32(Vec::with_capacity(selected.len()))
                }
                DirectPrimitiveColumnType::I64 => {
                    PrimitiveColumnOutput::I64(Vec::with_capacity(selected.len()))
                }
                _ => unreachable!("primitive top-k only uses i32/i64 columns"),
            })
            .collect::<Vec<_>>();
        for (_, _, slot) in selected {
            if let Some(row_refs) = &self.row_refs {
                let row_ref = row_refs[slot];
                let _physical_row = (row_ref.row_group, row_ref.row_offset);
            }
            for (source, target) in self.columns.iter().zip(output.iter_mut()) {
                push_primitive_output_slot(source, slot, target)?;
            }
        }
        primitive_output_batch_from_columns(output, column_names, column_types)
    }

    fn selected_row_refs_sorted(&self) -> Option<Vec<PrimitiveTopkRowRef>> {
        let row_refs = self.row_refs.as_ref()?;
        let mut selected = self
            .heap
            .iter()
            .map(|Reverse(item)| *item)
            .collect::<Vec<_>>();
        selected.sort_unstable_by(|left, right| right.cmp(left));
        Some(
            selected
                .into_iter()
                .map(|(_, _, slot)| row_refs[slot])
                .collect(),
        )
    }

    fn push_slot_from_columns(
        &mut self,
        columns: &[NullFreePrimitiveColumn<'_>],
        row_ref: PrimitiveTopkRowRef,
        row: usize,
    ) -> Result<usize> {
        let slot = primitive_output_len(&self.columns[0]);
        for (source, target) in columns.iter().zip(self.columns.iter_mut()) {
            push_null_free_primitive_value(source, row, target)?;
        }
        if let Some(row_refs) = &mut self.row_refs {
            row_refs.push(row_ref);
        }
        Ok(slot)
    }

    fn overwrite_slot_from_columns(
        &mut self,
        slot: usize,
        columns: &[NullFreePrimitiveColumn<'_>],
        row_ref: PrimitiveTopkRowRef,
        row: usize,
    ) -> Result<()> {
        for (source, target) in columns.iter().zip(self.columns.iter_mut()) {
            overwrite_null_free_primitive_value(source, row, target, slot)?;
        }
        if let Some(row_refs) = &mut self.row_refs {
            row_refs[slot] = row_ref;
        }
        Ok(())
    }

    fn push_slot_from_state(
        &mut self,
        source: &PrimitiveTopkState,
        source_slot: usize,
    ) -> Result<usize> {
        let slot = primitive_output_len(&self.columns[0]);
        for (source_column, target) in source.columns.iter().zip(self.columns.iter_mut()) {
            push_primitive_output_slot(source_column, source_slot, target)?;
        }
        if let (Some(target_refs), Some(source_refs)) = (&mut self.row_refs, &source.row_refs) {
            target_refs.push(source_refs[source_slot]);
        }
        Ok(slot)
    }

    fn overwrite_slot_from_state(
        &mut self,
        slot: usize,
        source: &PrimitiveTopkState,
        source_slot: usize,
    ) -> Result<()> {
        for (source_column, target) in source.columns.iter().zip(self.columns.iter_mut()) {
            overwrite_primitive_output_slot(source_column, source_slot, target, slot)?;
        }
        if let (Some(target_refs), Some(source_refs)) = (&mut self.row_refs, &source.row_refs) {
            target_refs[slot] = source_refs[source_slot];
        }
        Ok(())
    }
}

fn primitive_topk_row_refs_enabled() -> bool {
    std::env::var("DODAM_ENABLE_PRIMITIVE_TOPK_ROW_REFS")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

fn primitive_topk_fused_filter_threshold_enabled() -> bool {
    if std::env::var("DODAM_DISABLE_PRIMITIVE_TOPK_FUSED_FILTER_THRESHOLD")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
    {
        return false;
    }
    std::env::var("DODAM_ENABLE_PRIMITIVE_TOPK_FUSED_FILTER_THRESHOLD")
        .map(|value| !matches!(value.as_str(), "0" | "false" | "FALSE" | "no" | "NO"))
        .unwrap_or(true)
}

pub(super) fn primitive_topk_block_max_skip_enabled() -> bool {
    if std::env::var("DODAM_DISABLE_PRIMITIVE_TOPK_BLOCK_MAX_SKIP")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
    {
        return false;
    }
    std::env::var("DODAM_ENABLE_PRIMITIVE_TOPK_BLOCK_MAX_SKIP")
        .map(|value| !matches!(value.as_str(), "0" | "false" | "FALSE" | "no" | "NO"))
        .unwrap_or(true)
}

fn null_free_primitive_columns_for_topk<'a>(
    view: BatchView<'a>,
    column_types: &[DirectPrimitiveColumnType],
) -> Option<Vec<NullFreePrimitiveColumn<'a>>> {
    let mut columns = Vec::with_capacity(column_types.len());
    for (index, column_type) in column_types.iter().enumerate() {
        match column_type {
            DirectPrimitiveColumnType::I32 => {
                let values = view.i32_vector(index)?.values_if_null_free()?;
                columns.push(NullFreePrimitiveColumn::I32(values));
            }
            DirectPrimitiveColumnType::I64 => {
                let values = view.i64_vector(index)?.values_if_null_free()?;
                columns.push(NullFreePrimitiveColumn::I64(values));
            }
            _ => return None,
        }
    }
    Some(columns)
}

fn update_streaming_desc_topk_heap(
    key_column: &ArrayRef,
    batch_index: usize,
    limit: usize,
    sequence: &mut u64,
    heap: &mut BinaryHeap<Reverse<(i128, u64, usize, u32)>>,
) -> Result<bool> {
    match key_column.data_type() {
        DataType::Int32 => {
            let values = key_column
                .as_any()
                .downcast_ref::<Int32Array>()
                .expect("Int32 data type");
            update_streaming_desc_topk_heap_typed(
                values.len(),
                |row| (!values.is_null(row)).then(|| i128::from(values.value(row))),
                batch_index,
                limit,
                sequence,
                heap,
            )
        }
        DataType::Int64 => {
            let values = key_column
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("Int64 data type");
            update_streaming_desc_topk_heap_typed(
                values.len(),
                |row| (!values.is_null(row)).then(|| i128::from(values.value(row))),
                batch_index,
                limit,
                sequence,
                heap,
            )
        }
        DataType::UInt64 => {
            let values = key_column
                .as_any()
                .downcast_ref::<UInt64Array>()
                .expect("UInt64 data type");
            update_streaming_desc_topk_heap_typed(
                values.len(),
                |row| (!values.is_null(row)).then(|| i128::from(values.value(row))),
                batch_index,
                limit,
                sequence,
                heap,
            )
        }
        DataType::Date32 => {
            let values = key_column
                .as_any()
                .downcast_ref::<Date32Array>()
                .expect("Date32 data type");
            update_streaming_desc_topk_heap_typed(
                values.len(),
                |row| (!values.is_null(row)).then(|| i128::from(values.value(row))),
                batch_index,
                limit,
                sequence,
                heap,
            )
        }
        data_type => Err(DodamError::UnsupportedSql(format!(
            "unsupported streaming top-k sort type: {data_type:?}"
        ))),
    }
}

fn update_streaming_desc_topk_heap_typed<F>(
    rows: usize,
    mut key_at: F,
    batch_index: usize,
    limit: usize,
    sequence: &mut u64,
    heap: &mut BinaryHeap<Reverse<(i128, u64, usize, u32)>>,
) -> Result<bool>
where
    F: FnMut(usize) -> Option<i128>,
{
    let mut threshold = (heap.len() == limit)
        .then(|| heap.peek().map(|worst| worst.0))
        .flatten();
    for row in 0..rows {
        let Some(key) = key_at(row) else {
            return Ok(false);
        };
        if let Some(worst) = threshold
            && key < worst.0
        {
            *sequence = (*sequence).wrapping_add(1);
            continue;
        }
        let item = (key, *sequence, batch_index, row as u32);
        *sequence = (*sequence).wrapping_add(1);
        if heap.len() < limit {
            heap.push(Reverse(item));
            if heap.len() == limit {
                threshold = heap.peek().map(|worst| worst.0);
            }
        } else {
            let mut replaced = false;
            {
                if let Some(mut worst) = heap.peek_mut()
                    && item > worst.0
                {
                    *worst = Reverse(item);
                    replaced = true;
                }
            }
            if replaced {
                threshold = heap.peek().map(|worst| worst.0);
            }
        }
    }
    Ok(true)
}

pub(super) fn materialize_topk_selected_rows(
    batches: &[RecordBatch],
    selected: &[(i128, u64, usize, u32)],
) -> Result<RecordBatch> {
    if selected.is_empty() {
        let schema = batches
            .first()
            .map(RecordBatch::schema)
            .unwrap_or_else(|| Arc::new(Schema::empty()));
        return Ok(RecordBatch::new_empty(schema));
    }
    if let Some(batch) = take_topk_selected_record_batch_runs(batches, selected)? {
        return Ok(batch);
    }
    if let Some(batch) = gather_topk_selected_record_batch(batches, selected)? {
        return Ok(batch);
    }
    let mut chunks = Vec::with_capacity(selected.len());
    for &(_, _, batch_index, row) in selected {
        chunks.push(batches[batch_index].slice(row as usize, 1));
    }
    let schema = chunks[0].schema();
    Ok(concat_batches(&schema, chunks.iter())?)
}

fn take_topk_selected_record_batch_runs(
    batches: &[RecordBatch],
    selected: &[(i128, u64, usize, u32)],
) -> Result<Option<RecordBatch>> {
    if selected.len() < topk_take_materialization_min_rows() {
        return Ok(None);
    }
    let Some(first_batch) = batches.first() else {
        return Ok(None);
    };
    let schema = first_batch.schema();
    let mut chunks = Vec::new();
    let mut index = 0usize;
    while index < selected.len() {
        let batch_index = selected[index].2;
        let Some(batch) = batches.get(batch_index) else {
            return Ok(None);
        };
        if batch.schema() != schema {
            return Ok(None);
        }
        let mut rows = Vec::new();
        while index < selected.len() && selected[index].2 == batch_index {
            rows.push(selected[index].3);
            index += 1;
        }
        chunks.push(take_record_batch(batch, &UInt32Array::from(rows))?);
    }
    let Some(first) = chunks.first() else {
        return Ok(None);
    };
    if chunks.len() == 1 {
        return Ok(chunks.pop());
    }
    let schema = first.schema();
    Ok(Some(concat_batches(&schema, chunks.iter())?))
}

fn topk_take_materialization_min_rows() -> usize {
    std::env::var("DODAM_TOPK_TAKE_MATERIALIZATION_MIN_ROWS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(usize::MAX)
}

fn gather_topk_selected_record_batch(
    batches: &[RecordBatch],
    selected: &[(i128, u64, usize, u32)],
) -> Result<Option<RecordBatch>> {
    let Some(first_batch) = batches.first() else {
        return Ok(None);
    };
    let schema = first_batch.schema();
    let mut columns = Vec::with_capacity(first_batch.num_columns());
    for column_index in 0..first_batch.num_columns() {
        let Some(column) = gather_topk_selected_array(batches, selected, column_index) else {
            return Ok(None);
        };
        columns.push(column);
    }
    Ok(Some(RecordBatch::try_new(schema, columns)?))
}

fn gather_topk_selected_array(
    batches: &[RecordBatch],
    selected: &[(i128, u64, usize, u32)],
    column_index: usize,
) -> Option<ArrayRef> {
    let data_type = batches.first()?.column(column_index).data_type().clone();
    if batches
        .iter()
        .any(|batch| batch.column(column_index).data_type() != &data_type)
    {
        return None;
    }

    match data_type {
        DataType::Boolean => {
            if topk_selected_column_null_count(batches, column_index) == 0 {
                let mut output = Vec::with_capacity(selected.len());
                for &(_, _, batch_index, row) in selected {
                    let values = batches[batch_index]
                        .column(column_index)
                        .as_any()
                        .downcast_ref::<BooleanArray>()?;
                    output.push(values.value(row as usize));
                }
                Some(Arc::new(BooleanArray::from(output)))
            } else {
                let mut output = Vec::with_capacity(selected.len());
                for &(_, _, batch_index, row) in selected {
                    let values = batches[batch_index]
                        .column(column_index)
                        .as_any()
                        .downcast_ref::<BooleanArray>()?;
                    let row = row as usize;
                    output.push((!values.is_null(row)).then(|| values.value(row)));
                }
                Some(Arc::new(BooleanArray::from(output)))
            }
        }
        DataType::Int32 => {
            if topk_selected_column_null_count(batches, column_index) == 0 {
                let mut output = Vec::with_capacity(selected.len());
                for &(_, _, batch_index, row) in selected {
                    let values = batches[batch_index]
                        .column(column_index)
                        .as_any()
                        .downcast_ref::<Int32Array>()?;
                    output.push(values.value(row as usize));
                }
                Some(Arc::new(Int32Array::from(output)))
            } else {
                let mut output = Vec::with_capacity(selected.len());
                for &(_, _, batch_index, row) in selected {
                    let values = batches[batch_index]
                        .column(column_index)
                        .as_any()
                        .downcast_ref::<Int32Array>()?;
                    let row = row as usize;
                    output.push((!values.is_null(row)).then(|| values.value(row)));
                }
                Some(Arc::new(Int32Array::from(output)))
            }
        }
        DataType::Int64 => {
            if topk_selected_column_null_count(batches, column_index) == 0 {
                let mut output = Vec::with_capacity(selected.len());
                for &(_, _, batch_index, row) in selected {
                    let values = batches[batch_index]
                        .column(column_index)
                        .as_any()
                        .downcast_ref::<Int64Array>()?;
                    output.push(values.value(row as usize));
                }
                Some(Arc::new(Int64Array::from(output)))
            } else {
                let mut output = Vec::with_capacity(selected.len());
                for &(_, _, batch_index, row) in selected {
                    let values = batches[batch_index]
                        .column(column_index)
                        .as_any()
                        .downcast_ref::<Int64Array>()?;
                    let row = row as usize;
                    output.push((!values.is_null(row)).then(|| values.value(row)));
                }
                Some(Arc::new(Int64Array::from(output)))
            }
        }
        DataType::UInt64 => {
            if topk_selected_column_null_count(batches, column_index) == 0 {
                let mut output = Vec::with_capacity(selected.len());
                for &(_, _, batch_index, row) in selected {
                    let values = batches[batch_index]
                        .column(column_index)
                        .as_any()
                        .downcast_ref::<UInt64Array>()?;
                    output.push(values.value(row as usize));
                }
                Some(Arc::new(UInt64Array::from(output)))
            } else {
                let mut output = Vec::with_capacity(selected.len());
                for &(_, _, batch_index, row) in selected {
                    let values = batches[batch_index]
                        .column(column_index)
                        .as_any()
                        .downcast_ref::<UInt64Array>()?;
                    let row = row as usize;
                    output.push((!values.is_null(row)).then(|| values.value(row)));
                }
                Some(Arc::new(UInt64Array::from(output)))
            }
        }
        DataType::Float64 => {
            if topk_selected_column_null_count(batches, column_index) == 0 {
                let mut output = Vec::with_capacity(selected.len());
                for &(_, _, batch_index, row) in selected {
                    let values = batches[batch_index]
                        .column(column_index)
                        .as_any()
                        .downcast_ref::<Float64Array>()?;
                    output.push(values.value(row as usize));
                }
                Some(Arc::new(Float64Array::from(output)))
            } else {
                let mut output = Vec::with_capacity(selected.len());
                for &(_, _, batch_index, row) in selected {
                    let values = batches[batch_index]
                        .column(column_index)
                        .as_any()
                        .downcast_ref::<Float64Array>()?;
                    let row = row as usize;
                    output.push((!values.is_null(row)).then(|| values.value(row)));
                }
                Some(Arc::new(Float64Array::from(output)))
            }
        }
        DataType::Date32 => {
            if topk_selected_column_null_count(batches, column_index) == 0 {
                let mut output = Vec::with_capacity(selected.len());
                for &(_, _, batch_index, row) in selected {
                    let values = batches[batch_index]
                        .column(column_index)
                        .as_any()
                        .downcast_ref::<Date32Array>()?;
                    output.push(values.value(row as usize));
                }
                Some(Arc::new(Date32Array::from(output)))
            } else {
                let mut output = Vec::with_capacity(selected.len());
                for &(_, _, batch_index, row) in selected {
                    let values = batches[batch_index]
                        .column(column_index)
                        .as_any()
                        .downcast_ref::<Date32Array>()?;
                    let row = row as usize;
                    output.push((!values.is_null(row)).then(|| values.value(row)));
                }
                Some(Arc::new(Date32Array::from(output)))
            }
        }
        _ => None,
    }
}

fn topk_selected_column_null_count(batches: &[RecordBatch], column_index: usize) -> usize {
    let mut null_count = 0usize;
    for batch in batches {
        null_count = null_count.saturating_add(batch.column(column_index).null_count());
    }
    null_count
}

fn topk_sort_key_type_supported(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Int32 | DataType::Int64 | DataType::UInt64 | DataType::Date32
    )
}

pub(super) fn numeric_column_ascending_bounds(column: &ArrayRef) -> Result<Option<(i128, i128)>> {
    match column.data_type() {
        DataType::Int32 => {
            let values = column
                .as_any()
                .downcast_ref::<Int32Array>()
                .expect("Int32 data type");
            primitive_ascending_bounds(values.len(), |row| {
                (!values.is_null(row)).then(|| i128::from(values.value(row)))
            })
        }
        DataType::Int64 => {
            let values = column
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("Int64 data type");
            primitive_ascending_bounds(values.len(), |row| {
                (!values.is_null(row)).then(|| i128::from(values.value(row)))
            })
        }
        DataType::UInt64 => {
            let values = column
                .as_any()
                .downcast_ref::<UInt64Array>()
                .expect("UInt64 data type");
            primitive_ascending_bounds(values.len(), |row| {
                (!values.is_null(row)).then(|| i128::from(values.value(row)))
            })
        }
        DataType::Date32 => {
            let values = column
                .as_any()
                .downcast_ref::<Date32Array>()
                .expect("Date32 data type");
            primitive_ascending_bounds(values.len(), |row| {
                (!values.is_null(row)).then(|| i128::from(values.value(row)))
            })
        }
        _ => Ok(None),
    }
}

fn primitive_ascending_bounds<F>(len: usize, mut value: F) -> Result<Option<(i128, i128)>>
where
    F: FnMut(usize) -> Option<i128>,
{
    if len == 0 {
        return Ok(None);
    }
    let Some(first) = value(0) else {
        return Ok(None);
    };
    let mut previous = first;
    for row in 1..len {
        let Some(current) = value(row) else {
            return Ok(None);
        };
        if previous > current {
            return Ok(None);
        }
        previous = current;
    }
    Ok(Some((first, previous)))
}

async fn try_execute_same_source_union_all_scan(
    engine: &DodamEngine,
    expr: &SetExpr,
    batch_size: usize,
) -> Result<Option<Vec<RecordBatch>>> {
    let Some(shared) = plan_same_source_union_all_scan(expr)? else {
        return Ok(None);
    };
    let stream = engine
        .scan_parquet_batches(
            shared.path,
            batch_size,
            None,
            shared.projection,
            Some(shared.filter),
        )
        .await?;
    let batches = rename_output_batches(collect_batches(stream)?, &shared.aliases)?;
    Ok(Some(batches))
}

async fn try_execute_same_source_union_all_filter_scan(
    engine: &DodamEngine,
    expr: &SetExpr,
    batch_size: usize,
) -> Result<Option<Vec<RecordBatch>>> {
    let Some(shared) = plan_same_source_union_all_filter_scan(expr)? else {
        return Ok(None);
    };
    let mut stream = engine
        .scan_parquet_batches(
            shared.path.clone(),
            batch_size,
            None,
            shared.scan_projection.clone(),
            Some(shared.prefilter.clone()),
        )
        .await?;
    let mut output = Vec::new();
    for batch in stream.by_ref() {
        let batch = batch?;
        append_same_source_union_all_filter_batches(&mut output, &batch, &shared)?;
    }
    Ok(Some(output))
}

async fn try_execute_same_source_union_distinct_scan(
    engine: &DodamEngine,
    expr: &SetExpr,
    batch_size: usize,
    order_by: Option<&SortKey>,
    limit: Option<usize>,
    offset: usize,
) -> Result<Option<Vec<RecordBatch>>> {
    let Some(shared) = plan_same_source_union_distinct_scan(expr)? else {
        return Ok(None);
    };
    if let Some(batches) = try_execute_direct_distinct_scan(
        engine,
        DirectDistinctScan {
            path: shared.path.clone(),
            projection: shared.projection.clone(),
            aliases: shared.aliases.clone(),
            filter: Some(shared.filter.clone()),
        },
        batch_size,
    )? {
        return Ok(Some(batches));
    }
    let stream = engine
        .scan_parquet_distinct_batches(
            shared.path,
            batch_size,
            scan_limit_with_offset(limit, offset)?,
            shared.projection,
            Some(shared.filter),
            order_by.cloned(),
        )
        .await?;
    let batches = rename_output_batches(collect_batches(stream)?, &shared.aliases)?;
    Ok(Some(batches))
}

async fn try_execute_same_source_distinct_set_scan(
    engine: &DodamEngine,
    expr: &SetExpr,
    batch_size: usize,
    order_by: Option<&SortKey>,
    limit: Option<usize>,
    offset: usize,
) -> Result<Option<Vec<RecordBatch>>> {
    let Some(shared) = plan_same_source_distinct_set_scan(expr)? else {
        return Ok(None);
    };
    if let Some(batches) = try_execute_direct_distinct_scan(
        engine,
        DirectDistinctScan {
            path: shared.path.clone(),
            projection: shared.projection.clone(),
            aliases: shared.aliases.clone(),
            filter: Some(shared.filter.clone()),
        },
        batch_size,
    )? {
        return Ok(Some(batches));
    }
    let stream = engine
        .scan_parquet_distinct_batches(
            shared.path,
            batch_size,
            scan_limit_with_offset(limit, offset)?,
            shared.projection,
            Some(shared.filter),
            order_by.cloned(),
        )
        .await?;
    let batches = rename_output_batches(collect_batches(stream)?, &shared.aliases)?;
    Ok(Some(batches))
}

fn try_execute_same_source_all_set_scan(
    engine: &DodamEngine,
    expr: &SetExpr,
    batch_size: usize,
    _order_by: Option<&SortKey>,
    _limit: Option<usize>,
    _offset: usize,
) -> Result<Option<Vec<RecordBatch>>> {
    let Some(scan) = plan_same_source_all_set_primitive_scan(expr)? else {
        return Ok(None);
    };
    let Some(column_types) = engine
        .parquet_direct_primitive_column_types(&scan.path, std::slice::from_ref(&scan.column))?
    else {
        return Ok(None);
    };
    let [column_type] = column_types.as_slice() else {
        return Ok(None);
    };
    if !matches!(
        column_type,
        DirectPrimitiveColumnType::I32 | DirectPrimitiveColumnType::I64
    ) {
        return Ok(None);
    }
    let row_group_count = engine.parquet_row_group_count(&scan.path)?;
    let row_groups = (0..row_group_count).collect::<Vec<_>>();
    let candidates = scan.candidates();
    let Some((state, _metrics)) = engine.scan_parquet_primitive_columns_parallel_view_fold(
        scan.path.clone(),
        batch_size,
        row_groups,
        vec![(scan.column.clone(), *column_type)],
        || PrimitiveSetAllCounts::new(candidates.clone()),
        move |state, view| state.consume(view, *column_type),
        |state, partial| {
            state.merge(partial);
            Ok(())
        },
    )?
    else {
        return Ok(None);
    };
    if state.unsupported {
        return Ok(None);
    }
    let output = state.finish(&scan.left_values, &scan.right_values, scan.op);
    let (data_type, column): (DataType, ArrayRef) = match column_type {
        DirectPrimitiveColumnType::I32 => {
            let values = output
                .into_iter()
                .map(|value| {
                    i32::try_from(value)
                        .map_err(|_| DodamError::InvalidFilter(format!("{}={value}", scan.column)))
                })
                .collect::<Result<Vec<_>>>()?;
            (DataType::Int32, Arc::new(Int32Array::from(values)))
        }
        DirectPrimitiveColumnType::I64 => (DataType::Int64, Arc::new(Int64Array::from(output))),
        _ => return Ok(None),
    };
    let schema = Arc::new(Schema::new(vec![Field::new(&scan.column, data_type, true)]));
    let batch = RecordBatch::try_new(schema, vec![column])?;
    rename_output_batches(vec![batch], &scan.aliases).map(Some)
}

#[derive(Clone)]
pub(super) struct DirectDistinctScan {
    pub(super) path: PathBuf,
    pub(super) projection: Projection,
    pub(super) aliases: Vec<(String, String)>,
    pub(super) filter: Option<FilterExpr>,
}

#[derive(Clone)]
struct SameSourceAllSetPrimitiveScan {
    path: PathBuf,
    column: String,
    aliases: Vec<(String, String)>,
    left_values: Vec<i64>,
    right_values: Vec<i64>,
    op: SetOperator,
}

impl SameSourceAllSetPrimitiveScan {
    fn candidates(&self) -> Vec<i64> {
        let mut candidates = Vec::new();
        for value in self.left_values.iter().chain(self.right_values.iter()) {
            if !candidates.iter().any(|candidate| candidate == value) {
                candidates.push(*value);
            }
        }
        candidates
    }
}

pub(super) fn try_collect_direct_monotonic_count_distinct(
    engine: &DodamEngine,
    query: &SqlQuery,
    batch_size: usize,
) -> Result<Option<AggregateMetrics>> {
    if !query.group_by.is_empty()
        || query.filter.is_some()
        || !query.aggregate_expressions.is_empty()
        || !query.filtered_aggregates.is_empty()
    {
        return Ok(None);
    }
    let [AggregateExpr::CountDistinct(column)] = query.aggregates.as_slice() else {
        return Ok(None);
    };
    let Some(column_types) =
        engine.parquet_direct_primitive_column_types(&query.path, std::slice::from_ref(column))?
    else {
        return Ok(None);
    };
    let Some(column_type) = column_types.first().copied() else {
        return Ok(None);
    };
    if !matches!(
        column_type,
        DirectPrimitiveColumnType::I32 | DirectPrimitiveColumnType::I64
    ) {
        return Ok(None);
    }
    let row_groups = (0..engine.parquet_row_group_count(&query.path)?).collect::<Vec<_>>();
    let Some((state, scan_metrics)) = engine.scan_parquet_primitive_columns_parallel_view_fold(
        query.path.clone(),
        batch_size,
        row_groups,
        vec![(column.clone(), column_type)],
        MonotonicPrimitiveDistinctCount::default,
        move |state, view| state.consume(view, column_type),
        |state, partial| state.merge(partial),
    )?
    else {
        return Ok(None);
    };
    let Some(count) = state.finish() else {
        return Ok(None);
    };
    Ok(Some(AggregateMetrics {
        fragments: 1,
        batches: scan_metrics.batches,
        rows: scan_metrics.rows,
        values: vec![AggregateResult {
            expr: query.aggregates[0].clone(),
            value: AggregateValue::Count(count),
        }],
        ..AggregateMetrics::default()
    }))
}

#[derive(Default)]
struct MonotonicPrimitiveDistinctCount {
    ranges: Vec<(i64, i64, u64)>,
    unsupported: bool,
}

impl MonotonicPrimitiveDistinctCount {
    fn consume(
        &mut self,
        view: BatchView<'_>,
        column_type: DirectPrimitiveColumnType,
    ) -> Result<()> {
        if self.unsupported || view.num_rows() == 0 {
            return Ok(());
        }
        match column_type {
            DirectPrimitiveColumnType::I32 => {
                let Some(values) = view.i32_vector(0) else {
                    self.unsupported = true;
                    return Ok(());
                };
                self.consume_i32(values)
            }
            DirectPrimitiveColumnType::I64 => {
                let Some(values) = view.i64_vector(0) else {
                    self.unsupported = true;
                    return Ok(());
                };
                self.consume_i64(values)
            }
            _ => {
                self.unsupported = true;
                Ok(())
            }
        }
    }

    fn consume_i32(&mut self, values: I32VectorView<'_>) -> Result<()> {
        if let Some((bytes, len)) = values.raw_bytes() {
            return self.consume_len_i64(len, |row| {
                Some(i64::from(read_i32_le_unaligned(bytes, row)))
            });
        }
        if let Some(values) = values.values_if_null_free() {
            return self.consume_len_i64(values.len(), |row| Some(i64::from(values[row])));
        }
        if let Some((values, def_levels)) = values.raw_nullable() {
            let full_width_values = values.len() == def_levels.len();
            let mut value_index = 0usize;
            return self.consume_len_i64(def_levels.len(), |row| {
                if def_levels[row] == 0 {
                    None
                } else if full_width_values {
                    Some(i64::from(values[row]))
                } else {
                    let value = values.get(value_index).copied().map(i64::from);
                    value_index += 1;
                    value
                }
            });
        }
        self.unsupported = true;
        Ok(())
    }

    fn consume_i64(&mut self, values: I64VectorView<'_>) -> Result<()> {
        if let Some((bytes, len)) = values.raw_bytes() {
            return self.consume_len_i64(len, |row| Some(read_i64_le_unaligned(bytes, row)));
        }
        if let Some(values) = values.values_if_null_free() {
            return self.consume_len_i64(values.len(), |row| Some(values[row]));
        }
        if let Some((values, def_levels)) = values.raw_nullable() {
            let full_width_values = values.len() == def_levels.len();
            let mut value_index = 0usize;
            return self.consume_len_i64(def_levels.len(), |row| {
                if def_levels[row] == 0 {
                    None
                } else if full_width_values {
                    Some(values[row])
                } else {
                    let value = values.get(value_index).copied();
                    value_index += 1;
                    value
                }
            });
        }
        self.unsupported = true;
        Ok(())
    }

    fn consume_len_i64<F>(&mut self, rows: usize, mut value_at: F) -> Result<()>
    where
        F: FnMut(usize) -> Option<i64>,
    {
        let mut first = None;
        let mut last = None;
        let mut count = 0u64;
        for row in 0..rows {
            let Some(value) = value_at(row) else {
                continue;
            };
            if last.is_some_and(|previous| value <= previous) {
                self.unsupported = true;
                return Ok(());
            }
            first.get_or_insert(value);
            last = Some(value);
            count += 1;
        }
        if let (Some(first), Some(last)) = (first, last) {
            self.ranges.push((first, last, count));
        }
        Ok(())
    }

    fn merge(&mut self, partial: Self) -> Result<()> {
        if self.unsupported || partial.unsupported {
            self.unsupported = true;
            return Ok(());
        }
        self.ranges.extend(partial.ranges);
        Ok(())
    }

    fn finish(mut self) -> Option<u64> {
        if self.unsupported {
            return None;
        }
        self.ranges.sort_unstable_by_key(|range| range.0);
        let mut previous_end = None;
        let mut count = 0u64;
        for (start, end, range_count) in self.ranges {
            if previous_end.is_some_and(|previous| start <= previous) {
                return None;
            }
            previous_end = Some(end);
            count = count.saturating_add(range_count);
        }
        Some(count)
    }
}

pub(super) fn try_execute_direct_distinct_scan(
    engine: &DodamEngine,
    scan: DirectDistinctScan,
    batch_size: usize,
) -> Result<Option<Vec<RecordBatch>>> {
    if let Some(batches) =
        try_execute_direct_distinct_single_primitive_values(engine, &scan, batch_size)?
    {
        return Ok(Some(batches));
    }
    if let Some(batches) = try_execute_direct_distinct_primitive_pairs(engine, &scan, batch_size)? {
        return Ok(Some(batches));
    }
    Ok(None)
}

fn try_execute_direct_distinct_single_primitive_values(
    engine: &DodamEngine,
    scan: &DirectDistinctScan,
    batch_size: usize,
) -> Result<Option<Vec<RecordBatch>>> {
    let Projection::Columns(columns) = &scan.projection else {
        return Ok(None);
    };
    let [projected_column] = columns.as_slice() else {
        return Ok(None);
    };
    let Some(filter) = scan.filter.as_ref() else {
        return Ok(None);
    };
    let Expr::InList {
        column,
        values,
        negated: false,
        has_null: false,
    } = filter.expr()
    else {
        return Ok(None);
    };
    if column != projected_column {
        return Ok(None);
    }
    let candidates = values
        .iter()
        .map(|value| value.as_i64(column))
        .collect::<Result<Vec<_>>>()?;
    if candidates.is_empty() {
        return Ok(None);
    }
    let Some(column_types) =
        engine.parquet_direct_primitive_column_types(&scan.path, std::slice::from_ref(column))?
    else {
        return Ok(None);
    };
    let [column_type] = column_types.as_slice() else {
        return Ok(None);
    };
    if !matches!(
        column_type,
        DirectPrimitiveColumnType::I32 | DirectPrimitiveColumnType::I64
    ) {
        return Ok(None);
    }
    let row_group_count = engine.parquet_row_group_count(&scan.path)?;
    let row_groups = (0..row_group_count).collect::<Vec<_>>();
    let Some((state, _metrics)) = engine.scan_parquet_primitive_columns_parallel_view_fold(
        scan.path.clone(),
        batch_size,
        row_groups,
        vec![(column.clone(), *column_type)],
        || PrimitiveCandidatePresence::new(candidates.clone()),
        move |state, view| state.consume(view, *column_type),
        |state, partial| {
            state.merge(partial);
            Ok(())
        },
    )?
    else {
        return Ok(None);
    };
    if state.unsupported {
        return Ok(None);
    }
    let mut output = Vec::with_capacity(state.candidates.len());
    for (index, candidate) in state.candidates.iter().copied().enumerate() {
        if state.found[index] {
            output.push(candidate);
        }
    }
    let (data_type, column): (DataType, ArrayRef) = match column_type {
        DirectPrimitiveColumnType::I32 => {
            let values = output
                .into_iter()
                .map(|value| {
                    i32::try_from(value)
                        .map_err(|_| DodamError::InvalidFilter(format!("{column}={value}")))
                })
                .collect::<Result<Vec<_>>>()?;
            (DataType::Int32, Arc::new(Int32Array::from(values)))
        }
        DirectPrimitiveColumnType::I64 => (DataType::Int64, Arc::new(Int64Array::from(output))),
        _ => return Ok(None),
    };
    let schema = Arc::new(Schema::new(vec![Field::new(
        projected_column,
        data_type,
        true,
    )]));
    let batch = RecordBatch::try_new(schema, vec![column])?;
    rename_output_batches(vec![batch], &scan.aliases).map(Some)
}

fn try_execute_direct_distinct_primitive_pairs(
    engine: &DodamEngine,
    scan: &DirectDistinctScan,
    batch_size: usize,
) -> Result<Option<Vec<RecordBatch>>> {
    let Projection::Columns(columns) = &scan.projection else {
        return Ok(None);
    };
    let [first_column, second_column] = columns.as_slice() else {
        return Ok(None);
    };
    let Some(filter) = scan.filter.as_ref() else {
        return Ok(None);
    };
    let Expr::InList {
        column,
        values,
        negated: false,
        has_null: false,
    } = filter.expr()
    else {
        return Ok(None);
    };
    let Some(filter_index) = columns.iter().position(|candidate| candidate == column) else {
        return Ok(None);
    };
    if filter_index != 0 {
        return Ok(None);
    }
    let other_index = if filter_index == 0 { 1 } else { 0 };
    let candidates = values
        .iter()
        .map(|value| value.as_i64(column))
        .collect::<Result<Vec<_>>>()?;
    if candidates.is_empty() {
        return Ok(None);
    }
    let Some(column_types) = engine.parquet_direct_primitive_column_types(&scan.path, columns)?
    else {
        return Ok(None);
    };
    if !primitive_distinct_column_type_supported(column_types[filter_index])
        || !primitive_distinct_column_type_supported(column_types[other_index])
    {
        return Ok(None);
    }
    let row_group_count = engine.parquet_row_group_count(&scan.path)?;
    let row_groups = (0..row_group_count).collect::<Vec<_>>();
    let Some((state, _metrics)) = engine.scan_parquet_primitive_columns_parallel_view_fold(
        scan.path.clone(),
        batch_size,
        row_groups,
        columns
            .iter()
            .zip(column_types.iter())
            .map(|(name, column_type)| (name.clone(), *column_type))
            .collect(),
        || {
            PrimitiveDistinctPairs::new(
                candidates.clone(),
                filter_index,
                other_index,
                column_types[filter_index],
                column_types[other_index],
            )
        },
        |state, view| state.consume(view),
        |state, partial| {
            state.merge(partial);
            Ok(())
        },
    )?
    else {
        return Ok(None);
    };
    if state.unsupported {
        return Ok(None);
    }
    let mut pairs = state.pairs.into_iter().collect::<Vec<_>>();
    pairs.sort_unstable();
    let first_values = pairs.iter().map(|(key, _)| *key).collect::<Vec<_>>();
    let second_values = pairs.iter().map(|(_, value)| *value).collect::<Vec<_>>();
    let schema = Arc::new(Schema::new(vec![
        Field::new(
            first_column,
            primitive_distinct_data_type(column_types[0])?,
            true,
        ),
        Field::new(
            second_column,
            primitive_distinct_data_type(column_types[1])?,
            true,
        ),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            primitive_distinct_array(first_values, column_types[0])?,
            primitive_distinct_array(second_values, column_types[1])?,
        ],
    )?;
    rename_output_batches(vec![batch], &scan.aliases).map(Some)
}

fn primitive_distinct_column_type_supported(column_type: DirectPrimitiveColumnType) -> bool {
    matches!(
        column_type,
        DirectPrimitiveColumnType::I32 | DirectPrimitiveColumnType::I64
    )
}

fn primitive_distinct_data_type(column_type: DirectPrimitiveColumnType) -> Result<DataType> {
    match column_type {
        DirectPrimitiveColumnType::I32 => Ok(DataType::Int32),
        DirectPrimitiveColumnType::I64 => Ok(DataType::Int64),
        _ => Err(DodamError::UnsupportedSql(format!(
            "unsupported primitive distinct type {column_type:?}"
        ))),
    }
}

fn primitive_distinct_array(
    values: Vec<i64>,
    column_type: DirectPrimitiveColumnType,
) -> Result<ArrayRef> {
    match column_type {
        DirectPrimitiveColumnType::I32 => Ok(Arc::new(Int32Array::from(
            values
                .into_iter()
                .map(|value| {
                    i32::try_from(value).map_err(|_| {
                        DodamError::InvalidFilter(format!("primitive distinct value {value}"))
                    })
                })
                .collect::<Result<Vec<_>>>()?,
        ))),
        DirectPrimitiveColumnType::I64 => Ok(Arc::new(Int64Array::from(values))),
        _ => Err(DodamError::UnsupportedSql(format!(
            "unsupported primitive distinct type {column_type:?}"
        ))),
    }
}

fn primitive_distinct_column_values(
    view: &BatchView<'_>,
    index: usize,
    column_type: DirectPrimitiveColumnType,
) -> Option<Vec<i64>> {
    match column_type {
        DirectPrimitiveColumnType::I32 => {
            let column = view.i32_vector(index)?;
            if let Some(values) = column.values_if_null_free() {
                return Some(values.iter().map(|value| i64::from(*value)).collect());
            }
            let (values, def_levels) = column.raw_nullable()?;
            Some(nullable_i32_values_to_i64(values, def_levels))
        }
        DirectPrimitiveColumnType::I64 => {
            let column = view.i64_vector(index)?;
            if let Some(values) = column.values_if_null_free() {
                return Some(values.to_vec());
            }
            let (values, def_levels) = column.raw_nullable()?;
            Some(nullable_i64_values(values, def_levels))
        }
        _ => None,
    }
}

fn primitive_distinct_null_free_column_values(
    view: &BatchView<'_>,
    index: usize,
    column_type: DirectPrimitiveColumnType,
) -> Option<Vec<i64>> {
    match column_type {
        DirectPrimitiveColumnType::I32 => view
            .i32_vector(index)
            .and_then(|column| column.values_if_null_free())
            .map(|values| values.iter().map(|value| i64::from(*value)).collect()),
        DirectPrimitiveColumnType::I64 => view
            .i64_vector(index)
            .and_then(|column| column.values_if_null_free())
            .map(|values| values.to_vec()),
        _ => None,
    }
}

fn nullable_i32_values_to_i64(values: &[i32], def_levels: &[i16]) -> Vec<i64> {
    nullable_values(values.len(), def_levels, |index| i64::from(values[index]))
}

fn nullable_i64_values(values: &[i64], def_levels: &[i16]) -> Vec<i64> {
    nullable_values(values.len(), def_levels, |index| values[index])
}

fn nullable_values<F>(value_len: usize, def_levels: &[i16], value_at: F) -> Vec<i64>
where
    F: Fn(usize) -> i64,
{
    let mut output = Vec::with_capacity(def_levels.len());
    let mut value_index = 0usize;
    let full_width_values = value_len == def_levels.len();
    for (row, definition) in def_levels.iter().copied().enumerate() {
        if definition == 0 {
            continue;
        }
        let index = if full_width_values { row } else { value_index };
        output.push(value_at(index));
        value_index += 1;
    }
    output
}

struct PrimitiveCandidatePresence {
    candidates: Vec<i64>,
    found: Vec<bool>,
    unsupported: bool,
}

impl PrimitiveCandidatePresence {
    fn new(candidates: Vec<i64>) -> Self {
        let found = vec![false; candidates.len()];
        Self {
            candidates,
            found,
            unsupported: false,
        }
    }

    fn consume(
        &mut self,
        view: BatchView<'_>,
        column_type: DirectPrimitiveColumnType,
    ) -> Result<()> {
        let Some(values) = primitive_distinct_column_values(&view, 0, column_type) else {
            self.unsupported = true;
            return Ok(());
        };
        self.consume_values(values)
    }

    fn consume_values<I>(&mut self, values: I) -> Result<()>
    where
        I: IntoIterator<Item = i64>,
    {
        let values = values.into_iter();
        match self.candidates.as_slice() {
            [a] => {
                for value in values {
                    if value == *a {
                        self.found[0] = true;
                    }
                }
            }
            [a, b] => {
                for value in values {
                    if value == *a {
                        self.found[0] = true;
                    } else if value == *b {
                        self.found[1] = true;
                    }
                }
            }
            [a, b, c] => {
                for value in values {
                    if value == *a {
                        self.found[0] = true;
                    } else if value == *b {
                        self.found[1] = true;
                    } else if value == *c {
                        self.found[2] = true;
                    }
                }
            }
            candidates => {
                for value in values {
                    if let Some(index) = candidates.iter().position(|candidate| *candidate == value)
                    {
                        self.found[index] = true;
                    }
                }
            }
        }
        Ok(())
    }

    fn merge(&mut self, partial: Self) {
        self.unsupported |= partial.unsupported;
        for (found, partial_found) in self.found.iter_mut().zip(partial.found) {
            *found |= partial_found;
        }
    }
}

struct PrimitiveSetAllCounts {
    candidates: Vec<i64>,
    counts: Vec<usize>,
    unsupported: bool,
}

impl PrimitiveSetAllCounts {
    fn new(candidates: Vec<i64>) -> Self {
        let counts = vec![0; candidates.len()];
        Self {
            candidates,
            counts,
            unsupported: false,
        }
    }

    fn consume(
        &mut self,
        view: BatchView<'_>,
        column_type: DirectPrimitiveColumnType,
    ) -> Result<()> {
        match column_type {
            DirectPrimitiveColumnType::I32 => {
                let Some(values) = view
                    .i32_vector(0)
                    .and_then(|column| column.values_if_null_free())
                else {
                    self.unsupported = true;
                    return Ok(());
                };
                self.consume_values(values.iter().map(|value| i64::from(*value)))
            }
            DirectPrimitiveColumnType::I64 => {
                let Some(values) = view
                    .i64_vector(0)
                    .and_then(|column| column.values_if_null_free())
                else {
                    self.unsupported = true;
                    return Ok(());
                };
                self.consume_values(values.iter().copied())
            }
            _ => {
                self.unsupported = true;
                Ok(())
            }
        }
    }

    fn consume_values<I>(&mut self, values: I) -> Result<()>
    where
        I: IntoIterator<Item = i64>,
    {
        let values = values.into_iter();
        match self.candidates.as_slice() {
            [] => for _ in values {},
            [a] => {
                let mut count = 0usize;
                for value in values {
                    count += usize::from(value == *a);
                }
                self.counts[0] += count;
            }
            [a, b] => {
                let mut count_a = 0usize;
                let mut count_b = 0usize;
                for value in values {
                    if value == *a {
                        count_a += 1;
                    } else if value == *b {
                        count_b += 1;
                    }
                }
                self.counts[0] += count_a;
                self.counts[1] += count_b;
            }
            [a, b, c] => {
                let mut count_a = 0usize;
                let mut count_b = 0usize;
                let mut count_c = 0usize;
                for value in values {
                    if value == *a {
                        count_a += 1;
                    } else if value == *b {
                        count_b += 1;
                    } else if value == *c {
                        count_c += 1;
                    }
                }
                self.counts[0] += count_a;
                self.counts[1] += count_b;
                self.counts[2] += count_c;
            }
            candidates => {
                for value in values {
                    if let Some(index) = candidates.iter().position(|candidate| *candidate == value)
                    {
                        self.counts[index] += 1;
                    }
                }
            }
        };
        Ok(())
    }

    fn merge(&mut self, partial: Self) {
        self.unsupported |= partial.unsupported;
        for (count, partial_count) in self.counts.iter_mut().zip(partial.counts) {
            *count += partial_count;
        }
    }

    fn finish(&self, left_values: &[i64], right_values: &[i64], op: SetOperator) -> Vec<i64> {
        let mut output = Vec::new();
        for &value in left_values {
            let left_count = self.count_for(value);
            let right_count = right_values
                .iter()
                .any(|right| *right == value)
                .then(|| self.count_for(value))
                .unwrap_or(0);
            let repeats = match op {
                SetOperator::Intersect => left_count.min(right_count),
                SetOperator::Except => left_count.saturating_sub(right_count),
                _ => 0,
            };
            output.extend(std::iter::repeat_n(value, repeats));
        }
        output
    }

    fn count_for(&self, value: i64) -> usize {
        self.candidates
            .iter()
            .position(|candidate| *candidate == value)
            .map(|index| self.counts[index])
            .unwrap_or(0)
    }
}

struct PrimitiveDistinctPairs {
    candidates: Vec<i64>,
    filter_index: usize,
    other_index: usize,
    filter_type: DirectPrimitiveColumnType,
    other_type: DirectPrimitiveColumnType,
    pairs: FastHashSet<(i64, i64)>,
    unsupported: bool,
}

impl PrimitiveDistinctPairs {
    fn new(
        candidates: Vec<i64>,
        filter_index: usize,
        other_index: usize,
        filter_type: DirectPrimitiveColumnType,
        other_type: DirectPrimitiveColumnType,
    ) -> Self {
        Self {
            candidates,
            filter_index,
            other_index,
            filter_type,
            other_type,
            pairs: FastHashSet::default(),
            unsupported: false,
        }
    }

    fn consume(&mut self, view: BatchView<'_>) -> Result<()> {
        let Some(keys) =
            primitive_distinct_null_free_column_values(&view, self.filter_index, self.filter_type)
        else {
            self.unsupported = true;
            return Ok(());
        };
        let Some(values) =
            primitive_distinct_null_free_column_values(&view, self.other_index, self.other_type)
        else {
            self.unsupported = true;
            return Ok(());
        };
        if keys.len() != values.len() {
            self.unsupported = true;
            return Ok(());
        }
        match self.candidates.as_slice() {
            [a] => {
                for row in 0..keys.len() {
                    if keys[row] == *a {
                        self.pairs.insert((keys[row], values[row]));
                    }
                }
            }
            [a, b] => {
                for row in 0..keys.len() {
                    let key = keys[row];
                    if key == *a || key == *b {
                        self.pairs.insert((key, values[row]));
                    }
                }
            }
            [a, b, c] => {
                for row in 0..keys.len() {
                    let key = keys[row];
                    if key == *a || key == *b || key == *c {
                        self.pairs.insert((key, values[row]));
                    }
                }
            }
            candidates => {
                for row in 0..keys.len() {
                    let key = keys[row];
                    if candidates.contains(&key) {
                        self.pairs.insert((key, values[row]));
                    }
                }
            }
        }
        Ok(())
    }

    fn merge(&mut self, partial: Self) {
        self.unsupported |= partial.unsupported;
        self.pairs.extend(partial.pairs);
    }
}

pub(super) fn plan_same_source_union_all_scan(
    expr: &SetExpr,
) -> Result<Option<SameSourceUnionAllScan>> {
    let mut operands = Vec::new();
    if !collect_union_all_operand_queries(expr, &mut operands)? || operands.len() < 2 {
        return Ok(None);
    }
    let Some(shared) = same_source_disjoint_union_all_plan(&operands) else {
        return Ok(None);
    };
    Ok(Some(shared))
}

pub(super) fn plan_same_source_union_all_filter_scan(
    expr: &SetExpr,
) -> Result<Option<SameSourceUnionAllFilterScan>> {
    let mut operands = Vec::new();
    if !collect_union_all_operand_queries(expr, &mut operands)? || operands.len() < 2 {
        return Ok(None);
    }
    Ok(same_source_union_all_filter_scan_plan(&operands))
}

fn plan_same_source_union_distinct_scan(expr: &SetExpr) -> Result<Option<SameSourceUnionAllScan>> {
    let mut operands = Vec::new();
    if !collect_union_distinct_operand_queries(expr, &mut operands)? || operands.len() < 2 {
        return Ok(None);
    }
    let Some(shared) = same_source_union_distinct_plan(&operands) else {
        return Ok(None);
    };
    Ok(Some(shared))
}

fn plan_same_source_distinct_set_scan(expr: &SetExpr) -> Result<Option<SameSourceUnionAllScan>> {
    let SetExpr::SetOperation {
        op,
        set_quantifier,
        left,
        right,
    } = expr
    else {
        return Ok(None);
    };
    if !matches!(*op, SetOperator::Intersect | SetOperator::Except)
        || !union_quantifier_is_distinct(*set_quantifier)
    {
        return Ok(None);
    }
    let Some(left_query) = single_set_operand_query(left.as_ref())? else {
        return Ok(None);
    };
    let Some(right_query) = single_set_operand_query(right.as_ref())? else {
        return Ok(None);
    };
    Ok(same_source_distinct_set_plan(
        &left_query,
        &right_query,
        *op,
    ))
}

fn try_execute_simple_case_distinct_set_literals(
    expr: &SetExpr,
) -> Result<Option<Vec<RecordBatch>>> {
    let SetExpr::SetOperation {
        op,
        set_quantifier,
        left,
        right,
    } = expr
    else {
        return Ok(None);
    };
    if !matches!(*op, SetOperator::Intersect | SetOperator::Except)
        || !union_quantifier_is_distinct(*set_quantifier)
    {
        return Ok(None);
    }
    let Some(left_query) = single_set_operand_query(left.as_ref())? else {
        return Ok(None);
    };
    let Some(right_query) = single_set_operand_query(right.as_ref())? else {
        return Ok(None);
    };
    if left_query.path != right_query.path || left_query.aliases != right_query.aliases {
        return Ok(None);
    }
    let Some((output_name, left_values)) = simple_case_literal_projection_values(&left_query)?
    else {
        return Ok(None);
    };
    let Some((right_output_name, right_values)) =
        simple_case_literal_projection_values(&right_query)?
    else {
        return Ok(None);
    };
    if output_name != right_output_name {
        return Ok(None);
    }
    let values = match op {
        SetOperator::Intersect => intersect_literal_values(left_values, &right_values),
        SetOperator::Except => except_literal_values(left_values, &right_values),
        _ => return Ok(None),
    };
    literal_values_batch(&output_name, values).map(|batch| Some(vec![batch]))
}

fn simple_case_literal_projection_values(
    query: &SqlQuery,
) -> Result<Option<(String, Vec<LiteralValue>)>> {
    if query.join.is_some()
        || query.expression_filter.is_some()
        || query.having.is_some()
        || query.order_by.is_some()
        || query.limit.is_some()
        || query.offset != 0
        || query.distinct
        || !query.aggregates.is_empty()
        || !query.aggregate_expressions.is_empty()
        || query.group_by.len() > 0
        || query.expressions.len() != 1
        || query.qualified_wildcards.len() > 0
    {
        return Ok(None);
    }
    let expression = &query.expressions[0];
    let Some(case) = simple_case_literal_descriptor(&expression.expr) else {
        return Ok(None);
    };
    let Some(filter) = query.filter.as_ref() else {
        return Ok(None);
    };
    let Some((filter_column, filter_values)) = positive_literal_filter_values(filter) else {
        return Ok(None);
    };
    if filter_column != case.column {
        return Ok(None);
    }
    let mut output = Vec::new();
    for value in filter_values {
        append_unique_literal_values(&mut output, vec![case.result_for_literal(&value)]);
    }
    Ok(Some((expression.output_name.clone(), output)))
}

struct SimpleCaseLiteralDescriptor {
    column: String,
    branches: Vec<(LiteralValue, LiteralValue)>,
    else_value: LiteralValue,
}

impl SimpleCaseLiteralDescriptor {
    fn result_for_literal(&self, value: &LiteralValue) -> LiteralValue {
        self.branches
            .iter()
            .find_map(|(condition, result)| (condition == value).then(|| result.clone()))
            .unwrap_or_else(|| self.else_value.clone())
    }
}

fn simple_case_literal_descriptor(
    expr: &ScalarSqlExpression,
) -> Option<SimpleCaseLiteralDescriptor> {
    let GroupKeyExpr::SimpleCaseLiteral {
        column,
        branches,
        else_value,
    } = simple_case_literal_group_key(expr)?
    else {
        return None;
    };
    Some(SimpleCaseLiteralDescriptor {
        column,
        branches: branches
            .into_iter()
            .map(|(condition, result)| {
                Some((
                    literal_value_from_group_key_literal(&condition)?,
                    literal_value_from_group_key_literal(&result)?,
                ))
            })
            .collect::<Option<Vec<_>>>()?,
        else_value: literal_value_from_group_key_literal(&else_value)?,
    })
}

fn literal_value_from_group_key_literal(value: &GroupKeyLiteral) -> Option<LiteralValue> {
    Some(match value {
        GroupKeyLiteral::Null => LiteralValue::Null,
        GroupKeyLiteral::Boolean(value) => LiteralValue::Boolean(*value),
        GroupKeyLiteral::Int64(value) => LiteralValue::Int64(*value),
        GroupKeyLiteral::Float64(value) => LiteralValue::Float64(f64::from_bits(*value)),
        GroupKeyLiteral::Utf8(value) => LiteralValue::Utf8(value.clone()),
    })
}

fn literal_values_batch(name: &str, values: Vec<LiteralValue>) -> Result<RecordBatch> {
    if values
        .iter()
        .all(|value| matches!(value, LiteralValue::Utf8(_) | LiteralValue::Null))
    {
        let array = StringArray::from(
            values
                .into_iter()
                .map(|value| match value {
                    LiteralValue::Utf8(value) => Some(value),
                    LiteralValue::Null => None,
                    _ => unreachable!("checked utf8 literal values"),
                })
                .collect::<Vec<_>>(),
        );
        return RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(name, DataType::Utf8, true)])),
            vec![Arc::new(array)],
        )
        .map_err(DodamError::from);
    }
    if values
        .iter()
        .all(|value| matches!(value, LiteralValue::Int64(_) | LiteralValue::Null))
    {
        let array = Int64Array::from(
            values
                .into_iter()
                .map(|value| match value {
                    LiteralValue::Int64(value) => Some(value),
                    LiteralValue::Null => None,
                    _ => unreachable!("checked int64 literal values"),
                })
                .collect::<Vec<_>>(),
        );
        return RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(name, DataType::Int64, true)])),
            vec![Arc::new(array)],
        )
        .map_err(DodamError::from);
    }
    Err(DodamError::UnsupportedSql(
        "simple CASE set literal output type is not supported yet".to_string(),
    ))
}

fn plan_same_source_all_set_primitive_scan(
    expr: &SetExpr,
) -> Result<Option<SameSourceAllSetPrimitiveScan>> {
    let SetExpr::SetOperation {
        op,
        set_quantifier,
        left,
        right,
    } = expr
    else {
        return Ok(None);
    };
    if !matches!(*op, SetOperator::Intersect | SetOperator::Except)
        || *set_quantifier != SetQuantifier::All
    {
        return Ok(None);
    }
    let Some(left_query) = single_set_operand_query(left.as_ref())? else {
        return Ok(None);
    };
    let Some(right_query) = single_set_operand_query(right.as_ref())? else {
        return Ok(None);
    };
    Ok(same_source_all_set_primitive_plan(
        &left_query,
        &right_query,
        *op,
    ))
}

pub(super) struct SameSourceUnionAllScan {
    pub(super) path: PathBuf,
    pub(super) projection: Projection,
    pub(super) aliases: Vec<(String, String)>,
    pub(super) filter: FilterExpr,
}

pub(super) struct SameSourceUnionAllFilterScan {
    pub(super) path: PathBuf,
    pub(super) projection: Projection,
    pub(super) scan_projection: Projection,
    pub(super) aliases: Vec<(String, String)>,
    pub(super) filters: Vec<FilterExpr>,
    pub(super) prefilter: FilterExpr,
}

fn single_set_operand_query(expr: &SetExpr) -> Result<Option<SqlQuery>> {
    match expr {
        SetExpr::Query(query) => {
            if query_contains_set_operation(query.body.as_ref())
                || query.order_by.is_some()
                || query.limit_clause.is_some()
                || parse_offset(query)? != 0
                || query.fetch.is_some()
                || !query.locks.is_empty()
            {
                return Ok(None);
            }
            Ok(Some(parse_sql(&query.to_string())?))
        }
        SetExpr::Select(_) => Ok(Some(parse_sql(&expr.to_string())?)),
        _ => Ok(None),
    }
}

fn collect_union_distinct_operand_queries(
    expr: &SetExpr,
    output: &mut Vec<SqlQuery>,
) -> Result<bool> {
    match expr {
        SetExpr::SetOperation {
            op,
            set_quantifier,
            left,
            right,
        } if *op == SetOperator::Union && union_quantifier_is_distinct(*set_quantifier) => Ok(
            collect_union_distinct_operand_queries(left.as_ref(), output)?
                && collect_union_distinct_operand_queries(right.as_ref(), output)?,
        ),
        SetExpr::SetOperation { .. } => Ok(false),
        SetExpr::Query(query) => {
            if query_contains_set_operation(query.body.as_ref()) {
                return collect_union_distinct_operand_queries(query.body.as_ref(), output);
            }
            if query.order_by.is_some()
                || query.limit_clause.is_some()
                || parse_offset(query)? != 0
                || query.fetch.is_some()
                || !query.locks.is_empty()
            {
                return Ok(false);
            }
            output.push(parse_sql(&query.to_string())?);
            Ok(true)
        }
        SetExpr::Select(_) => {
            output.push(parse_sql(&expr.to_string())?);
            Ok(true)
        }
        _ => Ok(false),
    }
}

fn collect_union_all_operand_queries(expr: &SetExpr, output: &mut Vec<SqlQuery>) -> Result<bool> {
    match expr {
        SetExpr::SetOperation {
            op,
            set_quantifier,
            left,
            right,
        } if *op == SetOperator::Union && *set_quantifier == SetQuantifier::All => {
            Ok(collect_union_all_operand_queries(left.as_ref(), output)?
                && collect_union_all_operand_queries(right.as_ref(), output)?)
        }
        SetExpr::SetOperation { .. } => Ok(false),
        SetExpr::Query(query) => {
            if query_contains_set_operation(query.body.as_ref()) {
                return collect_union_all_operand_queries(query.body.as_ref(), output);
            }
            if query.order_by.is_some()
                || query.limit_clause.is_some()
                || parse_offset(query)? != 0
                || query.fetch.is_some()
                || !query.locks.is_empty()
            {
                return Ok(false);
            }
            output.push(parse_sql(&query.to_string())?);
            Ok(true)
        }
        SetExpr::Select(_) => {
            output.push(parse_sql(&expr.to_string())?);
            Ok(true)
        }
        _ => Ok(false),
    }
}

fn same_source_union_distinct_plan(operands: &[SqlQuery]) -> Option<SameSourceUnionAllScan> {
    let first = operands.first()?;
    if !same_source_union_all_operand_supported(first) {
        return None;
    }
    let (filter_column, first_values) = positive_literal_filter_values(first.filter.as_ref()?)?;
    let mut values = Vec::new();
    append_unique_literal_values(&mut values, first_values);
    for operand in operands.iter().skip(1) {
        if !same_source_union_all_operand_supported(operand)
            || operand.path != first.path
            || operand.projection != first.projection
            || operand.aliases != first.aliases
        {
            return None;
        }
        let (column, operand_values) = positive_literal_filter_values(operand.filter.as_ref()?)?;
        if column != filter_column {
            return None;
        }
        append_unique_literal_values(&mut values, operand_values);
    }
    Some(SameSourceUnionAllScan {
        path: first.path.clone(),
        projection: first.projection.clone(),
        aliases: first.aliases.clone(),
        filter: FilterExpr::new(Expr::InList {
            column: filter_column,
            values,
            negated: false,
            has_null: false,
        }),
    })
}

fn same_source_distinct_set_plan(
    left: &SqlQuery,
    right: &SqlQuery,
    op: SetOperator,
) -> Option<SameSourceUnionAllScan> {
    if !same_source_union_all_operand_supported(left)
        || !same_source_union_all_operand_supported(right)
        || left.path != right.path
        || left.projection != right.projection
        || left.aliases != right.aliases
    {
        return None;
    }
    let (left_column, left_values) = positive_literal_filter_values(left.filter.as_ref()?)?;
    let (right_column, right_values) = positive_literal_filter_values(right.filter.as_ref()?)?;
    if left_column != right_column {
        return None;
    }
    let values = match op {
        SetOperator::Intersect => intersect_literal_values(left_values, &right_values),
        SetOperator::Except => except_literal_values(left_values, &right_values),
        _ => return None,
    };
    Some(SameSourceUnionAllScan {
        path: left.path.clone(),
        projection: left.projection.clone(),
        aliases: left.aliases.clone(),
        filter: FilterExpr::new(Expr::InList {
            column: left_column,
            values,
            negated: false,
            has_null: false,
        }),
    })
}

fn same_source_all_set_primitive_plan(
    left: &SqlQuery,
    right: &SqlQuery,
    op: SetOperator,
) -> Option<SameSourceAllSetPrimitiveScan> {
    if !same_source_union_all_operand_supported(left)
        || !same_source_union_all_operand_supported(right)
        || left.path != right.path
        || left.projection != right.projection
        || left.aliases != right.aliases
    {
        return None;
    }
    let Projection::Columns(projected) = &left.projection else {
        return None;
    };
    let [projected_column] = projected.as_slice() else {
        return None;
    };
    let (left_column, left_values) = positive_literal_filter_values(left.filter.as_ref()?)?;
    let (right_column, right_values) = positive_literal_filter_values(right.filter.as_ref()?)?;
    if left_column != right_column || left_column != *projected_column {
        return None;
    }
    Some(SameSourceAllSetPrimitiveScan {
        path: left.path.clone(),
        column: projected_column.clone(),
        aliases: left.aliases.clone(),
        left_values: literal_values_to_unique_i64(left_values, projected_column).ok()?,
        right_values: literal_values_to_unique_i64(right_values, projected_column).ok()?,
        op,
    })
}

fn same_source_union_all_filter_scan_plan(
    operands: &[SqlQuery],
) -> Option<SameSourceUnionAllFilterScan> {
    let first = operands.first()?;
    if !same_source_union_all_operand_supported(first) {
        return None;
    }
    let mut filters = Vec::with_capacity(operands.len());
    for operand in operands {
        if !same_source_union_all_operand_supported(operand)
            || operand.path != first.path
            || operand.projection != first.projection
            || operand.aliases != first.aliases
        {
            return None;
        }
        filters.push(operand.filter.clone()?);
    }
    let prefilter = union_filter_or(filters.iter().map(|filter| filter.expr().clone()))?;
    let mut scan_projection = first.projection.clone();
    for filter in &filters {
        add_projection_columns(&mut scan_projection, filter.referenced_columns());
    }
    Some(SameSourceUnionAllFilterScan {
        path: first.path.clone(),
        projection: first.projection.clone(),
        scan_projection,
        aliases: first.aliases.clone(),
        filters,
        prefilter,
    })
}

fn same_source_disjoint_union_all_plan(operands: &[SqlQuery]) -> Option<SameSourceUnionAllScan> {
    let first = operands.first()?;
    if !same_source_union_all_operand_supported(first) {
        return None;
    }
    let (filter_column, first_values) = positive_literal_filter_values(first.filter.as_ref()?)?;
    let mut values = Vec::new();
    append_disjoint_literal_values(&mut values, first_values)?;
    for operand in operands.iter().skip(1) {
        if !same_source_union_all_operand_supported(operand)
            || operand.path != first.path
            || operand.projection != first.projection
            || operand.aliases != first.aliases
        {
            return None;
        }
        let (column, operand_values) = positive_literal_filter_values(operand.filter.as_ref()?)?;
        if column != filter_column {
            return None;
        }
        append_disjoint_literal_values(&mut values, operand_values)?;
    }
    Some(SameSourceUnionAllScan {
        path: first.path.clone(),
        projection: first.projection.clone(),
        aliases: first.aliases.clone(),
        filter: FilterExpr::new(Expr::InList {
            column: filter_column,
            values,
            negated: false,
            has_null: false,
        }),
    })
}

fn union_filter_or(filters: impl IntoIterator<Item = Expr>) -> Option<FilterExpr> {
    filters
        .into_iter()
        .reduce(|left, right| Expr::Or(Box::new(left), Box::new(right)))
        .map(FilterExpr::new)
}

fn positive_literal_filter_values(filter: &FilterExpr) -> Option<(String, Vec<LiteralValue>)> {
    match filter.expr() {
        Expr::Comparison(comparison)
            if comparison.op == ComparisonOp::Eq
                && !matches!(comparison.value, LiteralValue::Null) =>
        {
            Some((comparison.column.clone(), vec![comparison.value.clone()]))
        }
        Expr::InList {
            column,
            values,
            negated: false,
            has_null: false,
        } if values
            .iter()
            .all(|value| !matches!(value, LiteralValue::Null)) =>
        {
            Some((column.clone(), values.clone()))
        }
        _ => None,
    }
}

fn same_source_union_all_operand_supported(query: &SqlQuery) -> bool {
    query.join.is_none()
        && query.expression_filter.is_none()
        && query.having.is_none()
        && query.order_by.is_none()
        && query.limit.is_none()
        && query.offset == 0
        && !query.distinct
        && query.aggregates.is_empty()
        && query.aggregate_expressions.is_empty()
        && projection_expressions_are_plain_columns(&query.expressions)
        && query.group_by.is_empty()
        && query.qualified_wildcards.is_empty()
}

async fn execute_set_operation_expr(
    engine: &DodamEngine,
    expr: &SetExpr,
    batch_size: usize,
    child_topk: Option<(&SortKey, usize)>,
    child_distinct: bool,
) -> Result<Vec<RecordBatch>> {
    match expr {
        SetExpr::SetOperation {
            op,
            set_quantifier,
            left,
            right,
        } if *op == SetOperator::Union => {
            let distinct = union_quantifier_is_distinct(*set_quantifier);
            let mut left_batches = Box::pin(execute_set_operation_expr(
                engine,
                left.as_ref(),
                batch_size,
                union_child_topk_for_quantifier(*set_quantifier, child_topk),
                child_distinct || distinct,
            ))
            .await?;
            let right_batches = Box::pin(execute_set_operation_expr(
                engine,
                right.as_ref(),
                batch_size,
                union_child_topk_for_quantifier(*set_quantifier, child_topk),
                child_distinct || distinct,
            ))
            .await?;
            append_union_all_batches(&mut left_batches, right_batches)?;
            if distinct {
                left_batches = apply_output_distinct(left_batches, true)?;
            }
            Ok(left_batches)
        }
        SetExpr::SetOperation {
            op,
            set_quantifier,
            left,
            right,
        } if *op == SetOperator::Intersect || *op == SetOperator::Except => {
            let left_batches = Box::pin(execute_set_operation_expr(
                engine,
                left.as_ref(),
                batch_size,
                None,
                false,
            ))
            .await?;
            let right_batches = Box::pin(execute_set_operation_expr(
                engine,
                right.as_ref(),
                batch_size,
                None,
                false,
            ))
            .await?;
            let batches = if *set_quantifier == SetQuantifier::All {
                apply_all_row_set_operation(left_batches, right_batches, *op)?
            } else {
                apply_distinct_row_set_operation(left_batches, right_batches, *op)?
            };
            apply_output_distinct(batches, child_distinct)
        }
        SetExpr::SetOperation {
            op, set_quantifier, ..
        } => Err(DodamError::UnsupportedSql(format!(
            "{op} {set_quantifier} is not supported yet"
        ))),
        SetExpr::Query(query) => {
            if query_contains_set_operation(query.body.as_ref()) {
                return Box::pin(execute_set_operation_expr(
                    engine,
                    query.body.as_ref(),
                    batch_size,
                    child_topk,
                    child_distinct,
                ))
                .await;
            }
            if query.order_by.is_some()
                || query.limit_clause.is_some()
                || query.fetch.is_some()
                || !query.locks.is_empty()
            {
                return Err(DodamError::UnsupportedSql(
                    "ORDER BY, LIMIT, FETCH, and locking clauses inside UNION operands are not supported yet"
                        .to_string(),
                ));
            }
            let sql = union_all_operand_sql_with_child_topk(&query.to_string(), child_topk);
            let batches =
                query_output_batches(Box::pin(execute_sql(engine, &sql, batch_size)).await?)?;
            apply_output_distinct(batches, child_distinct)
        }
        SetExpr::Select(_) => {
            let sql = union_all_operand_sql_with_child_topk(&expr.to_string(), child_topk);
            let batches =
                query_output_batches(Box::pin(execute_sql(engine, &sql, batch_size)).await?)?;
            apply_output_distinct(batches, child_distinct)
        }
        other => Err(DodamError::UnsupportedSql(format!(
            "unsupported set operation operand: {other}"
        ))),
    }
}
