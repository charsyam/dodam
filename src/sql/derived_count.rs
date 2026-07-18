use super::*;

pub(super) async fn try_execute_derived_left_join_count_distribution_sql(
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
    let SetExpr::Select(select) = query.body.as_ref() else {
        return Ok(None);
    };
    let Some((subquery, alias)) = parse_derived_from(select)? else {
        return Ok(None);
    };
    reject_query_features(query)?;
    reject_select_features(select)?;

    let outer_group_by = parse_group_by(select, Some(&alias))?;
    let projection = parse_projection(select, &outer_group_by, Some(&alias))?;
    if parse_distinct(select)?
        || outer_group_by.len() != 1
        || !matches!(projection.aggregates.as_slice(), [AggregateExpr::CountStar])
        || !projection.aggregate_expressions.is_empty()
        || projection_requires_expression_path(&projection.expressions)
        || select.selection.is_some()
        || select.having.is_some()
    {
        return Ok(None);
    }
    let order_by = parse_order_by(
        query,
        &projection.aliases,
        &projection.ordinal_targets,
        Some(&alias),
    )?;
    let limit = parse_limit(query)?;

    let inner = match parse_query(subquery) {
        Ok(inner) => inner,
        Err(DodamError::UnsupportedSql(_))
        | Err(DodamError::UnknownColumn(_))
        | Err(DodamError::UnknownTableQualifier(_)) => {
            return Ok(None);
        }
        Err(error) => return Err(error),
    };
    let Some(join) = &inner.join else {
        return Ok(None);
    };
    if inner.distinct
        || inner.filter.is_some()
        || inner.having.is_some()
        || inner.order_by.is_some()
        || inner.limit.is_some()
        || !inner.aggregate_expressions.is_empty()
        || projection_requires_expression_path(&inner.expressions)
        || join.join_type != JoinType::Left
        || join.left_keys.len() != 1
        || join.right_keys.len() != 1
        || inner.group_by.len() != 1
        || !same_join_column(&inner.group_by[0], &join.left_keys[0])
    {
        return Ok(None);
    }
    let [AggregateExpr::Count(count_column)] = inner.aggregates.as_slice() else {
        return Ok(None);
    };
    if resolve_inner_output_column_index(&inner, &outer_group_by[0]) != Some(inner.group_by.len()) {
        return Ok(None);
    }
    if !inner.path.exists() {
        return Ok(None);
    }

    let dense_counts = collect_dense_right_counts(engine, join, count_column, batch_size).await?;
    if dense_counts.is_empty() {
        return Ok(None);
    }
    let groups =
        collect_left_count_distribution(engine, &inner.path, join, &dense_counts, batch_size)
            .await?;
    let rows = groups
        .iter()
        .map(|group| match group.values[0].value {
            AggregateValue::Count(value) => value as usize,
            _ => 0,
        })
        .sum();
    let metrics = AggregateMetrics {
        fragments: 2,
        batches: 1,
        rows,
        values: Vec::new(),
        groups,
        ..AggregateMetrics::default()
    };
    let mut batches =
        aggregate_metrics_to_batches(&metrics, &outer_group_by, &projection.aggregates)?;
    batches = apply_output_order_limit(batches, order_by.as_ref(), limit, 0)?;
    batches = rename_output_batches(batches, &projection.aliases)?;
    Ok(Some(QueryOutput::Aggregate { metrics, batches }))
}

pub(super) async fn collect_dense_right_counts(
    engine: &DodamEngine,
    join: &SqlJoin,
    count_column: &str,
    batch_size: usize,
) -> Result<Vec<u32>> {
    let count_column = strip_column_prefix(count_column, &join.right_alias);
    let count_column_required = count_column != join.right_keys[0]
        && !parquet_column_is_non_nullable(&join.right.path, &count_column)?;
    let mut right_projection = vec![join.right_keys[0].clone()];
    if count_column_required {
        add_column_once(&mut right_projection, count_column.clone());
    }
    if let Some(filter) = &join.right_filter {
        for column in filter.referenced_columns() {
            add_column_once(
                &mut right_projection,
                strip_column_prefix(&column, &join.right_alias),
            );
        }
    }
    let right_key_index_hint = projected_column_index(&right_projection, &join.right_keys[0]);
    let count_index_hint = count_column_required
        .then(|| projected_column_index(&right_projection, &count_column))
        .flatten();
    let right_projection_for_scan = right_projection.clone();
    let direct_filter = join
        .right_filter
        .as_ref()
        .filter(|filter| expr_is_like_only(filter.expr()));
    let fast_like_filter = direct_filter
        .filter(|_| fast_like_distribution_enabled())
        .and_then(|filter| fast_like_substrings_filter(filter.expr()));
    let fast_like_finders = fast_like_filter.as_ref().map(|filter| {
        filter
            .parts
            .iter()
            .map(|part| Finder::new(part))
            .collect::<Vec<_>>()
    });
    let eval_filter = direct_filter.filter(|_| fast_like_filter.is_none());
    if dense_right_count_row_group_map_enabled()
        && !count_column_required
        && eval_filter.is_none()
        && let Some(filter) = fast_like_filter.clone()
        && let Some(counts) = collect_dense_right_counts_fast_like_row_group_map(
            engine,
            join.right.path.clone(),
            batch_size,
            Projection::Columns(right_projection_for_scan.clone()),
            join.right_keys[0].clone(),
            filter,
        )
        .await?
    {
        return Ok(counts);
    }
    let mut right_stream = engine
        .scan_parquet_batches(
            join.right.path.clone(),
            batch_size,
            None,
            Projection::Columns(right_projection_for_scan),
            if direct_filter.is_some() {
                None
            } else {
                join.right_filter.clone()
            },
        )
        .await?;
    let mut dense_counts = Vec::<u32>::new();
    while let Some(batch) = right_stream.next() {
        let batch = batch?;
        let fast_like_strings = fast_like_filter
            .as_ref()
            .map(|filter| batch_string_column(&batch, &filter.column))
            .transpose()?;
        let key_index =
            batch_projected_column_index(&batch, right_key_index_hint, &join.right_keys[0])?;
        let count_index = if count_column_required {
            Some(batch_projected_column_index(
                &batch,
                count_index_hint,
                &count_column,
            )?)
        } else {
            None
        };
        let Some(keys) = batch
            .column(key_index)
            .as_any()
            .downcast_ref::<Int64Array>()
        else {
            return Ok(Vec::new());
        };
        let values = count_index.map(|index| batch.column(index));
        let mask = eval_filter
            .map(|filter| evaluate_filter_mask(&batch, filter))
            .transpose()?;
        if mask.is_none()
            && values.is_none()
            && keys.null_count() == 0
            && let (Some(filter), Some(strings), Some(finders)) = (
                fast_like_filter.as_ref(),
                fast_like_strings.as_ref(),
                fast_like_finders.as_ref(),
            )
        {
            let key_values = keys.values().as_ref();
            if strings.null_count() == 0 {
                if let Some((first, second)) = fast_like_two_substring_finders(finders) {
                    for (row, &key) in key_values.iter().enumerate() {
                        if !fast_like_two_substrings_row_matches_non_null(
                            strings,
                            row,
                            &filter.parts[0],
                            &filter.parts[1],
                            first,
                            second,
                            filter.negated,
                        ) {
                            continue;
                        }
                        if key < 0 || key > 10_000_000 {
                            return Ok(Vec::new());
                        }
                        let index = key as usize;
                        if dense_counts.len() <= index {
                            dense_counts.resize(index + 1, 0);
                        }
                        dense_counts[index] =
                            dense_counts[index].checked_add(1).ok_or_else(|| {
                                DodamError::InvalidFilter("dense count overflow".to_string())
                            })?;
                    }
                    continue;
                }
                for (row, &key) in key_values.iter().enumerate() {
                    if !fast_like_substrings_row_matches_non_null(
                        strings,
                        row,
                        &filter.parts,
                        finders,
                        filter.negated,
                    ) {
                        continue;
                    }
                    if key < 0 || key > 10_000_000 {
                        return Ok(Vec::new());
                    }
                    let index = key as usize;
                    if dense_counts.len() <= index {
                        dense_counts.resize(index + 1, 0);
                    }
                    dense_counts[index] = dense_counts[index].checked_add(1).ok_or_else(|| {
                        DodamError::InvalidFilter("dense count overflow".to_string())
                    })?;
                }
                continue;
            }
            if let Some((first, second)) = fast_like_two_substring_finders(finders) {
                for (row, &key) in key_values.iter().enumerate() {
                    if !fast_like_two_substrings_row_matches(
                        strings,
                        row,
                        &filter.parts[0],
                        &filter.parts[1],
                        first,
                        second,
                        filter.negated,
                    ) {
                        continue;
                    }
                    if key < 0 || key > 10_000_000 {
                        return Ok(Vec::new());
                    }
                    let index = key as usize;
                    if dense_counts.len() <= index {
                        dense_counts.resize(index + 1, 0);
                    }
                    dense_counts[index] = dense_counts[index].checked_add(1).ok_or_else(|| {
                        DodamError::InvalidFilter("dense count overflow".to_string())
                    })?;
                }
                continue;
            }
            for (row, &key) in key_values.iter().enumerate() {
                if !fast_like_substrings_row_matches(
                    strings,
                    row,
                    &filter.parts,
                    finders,
                    filter.negated,
                ) {
                    continue;
                }
                if key < 0 || key > 10_000_000 {
                    return Ok(Vec::new());
                }
                let index = key as usize;
                if dense_counts.len() <= index {
                    dense_counts.resize(index + 1, 0);
                }
                dense_counts[index] = dense_counts[index]
                    .checked_add(1)
                    .ok_or_else(|| DodamError::InvalidFilter("dense count overflow".to_string()))?;
            }
            continue;
        }
        for row in 0..batch.num_rows() {
            if let (Some(filter), Some(strings), Some(finders)) = (
                fast_like_filter.as_ref(),
                fast_like_strings.as_ref(),
                fast_like_finders.as_ref(),
            ) && !fast_like_substrings_row_matches(
                strings,
                row,
                &filter.parts,
                finders,
                filter.negated,
            ) {
                continue;
            }
            if mask
                .as_ref()
                .is_some_and(|mask| mask.is_null(row) || !mask.value(row))
            {
                continue;
            }
            if keys.is_null(row) || values.is_some_and(|values| values.is_null(row)) {
                continue;
            }
            let key = keys.value(row);
            if key < 0 || key > 10_000_000 {
                return Ok(Vec::new());
            }
            let index = key as usize;
            if dense_counts.len() <= index {
                dense_counts.resize(index + 1, 0);
            }
            dense_counts[index] = dense_counts[index]
                .checked_add(1)
                .ok_or_else(|| DodamError::InvalidFilter("dense count overflow".to_string()))?;
        }
    }
    Ok(dense_counts)
}

pub(super) async fn collect_dense_right_counts_fast_like_row_group_map(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    projection: Projection,
    key_column: String,
    filter: FastLikeSubstrings,
) -> Result<Option<Vec<u32>>> {
    let filter = Arc::new(filter);
    let Some(partials) = engine
        .parquet_row_group_map_view(
            path,
            batch_size,
            projection,
            dense_right_count_row_group_chunk(),
            Vec::<u32>::new,
            {
                let filter = filter.clone();
                let key_column = key_column.clone();
                move |view, counts| {
                    collect_dense_right_counts_fast_like_view(view, &key_column, &filter, counts)
                }
            },
            |counts| Ok(Some(counts)),
        )
        .await?
    else {
        return Ok(None);
    };
    let mut counts = Vec::<u32>::new();
    for partial in partials {
        if counts.len() < partial.len() {
            counts.resize(partial.len(), 0);
        }
        for (index, count) in partial.into_iter().enumerate() {
            counts[index] = counts[index]
                .checked_add(count)
                .ok_or_else(|| DodamError::InvalidFilter("dense count overflow".to_string()))?;
        }
    }
    Ok(Some(counts))
}

pub(super) fn collect_dense_right_counts_fast_like_batch(
    batch: RecordBatch,
    key_column: &str,
    filter: &FastLikeSubstrings,
    counts: &mut Vec<u32>,
) -> Result<Option<()>> {
    let key_index = batch_column_index(&batch, key_column)?;
    let Some(keys) = batch
        .column(key_index)
        .as_any()
        .downcast_ref::<Int64Array>()
    else {
        return Ok(None);
    };
    let strings = batch_string_column(&batch, &filter.column)?;
    let finders = filter
        .parts
        .iter()
        .map(|part| Finder::new(part))
        .collect::<Vec<_>>();
    if keys.null_count() == 0 && strings.null_count() == 0 {
        let key_values = keys.values().as_ref();
        if let Some((first, second)) = fast_like_two_substring_finders(&finders) {
            for (row, &key) in key_values.iter().enumerate() {
                if fast_like_two_substrings_row_matches_non_null(
                    strings,
                    row,
                    &filter.parts[0],
                    &filter.parts[1],
                    first,
                    second,
                    filter.negated,
                ) {
                    dense_count_increment(counts, key)?;
                }
            }
            return Ok(Some(()));
        }
        for (row, &key) in key_values.iter().enumerate() {
            if fast_like_substrings_row_matches_non_null(
                strings,
                row,
                &filter.parts,
                &finders,
                filter.negated,
            ) {
                dense_count_increment(counts, key)?;
            }
        }
        return Ok(Some(()));
    }
    if let Some((first, second)) = fast_like_two_substring_finders(&finders) {
        for row in 0..keys.len() {
            if keys.is_valid(row)
                && fast_like_two_substrings_row_matches(
                    strings,
                    row,
                    &filter.parts[0],
                    &filter.parts[1],
                    first,
                    second,
                    filter.negated,
                )
            {
                dense_count_increment(counts, keys.value(row))?;
            }
        }
        return Ok(Some(()));
    }
    for row in 0..keys.len() {
        if keys.is_valid(row)
            && fast_like_substrings_row_matches(
                strings,
                row,
                &filter.parts,
                &finders,
                filter.negated,
            )
        {
            dense_count_increment(counts, keys.value(row))?;
        }
    }
    Ok(Some(()))
}

pub(super) fn collect_dense_right_counts_fast_like_view(
    view: BatchView<'_>,
    key_column: &str,
    filter: &FastLikeSubstrings,
    counts: &mut Vec<u32>,
) -> Result<Option<()>> {
    let Some(batch) = view.try_record_batch() else {
        return Ok(None);
    };
    let key_index = batch_column_index(batch, key_column)?;
    let Some(keys) = batch
        .column(key_index)
        .as_any()
        .downcast_ref::<Int64Array>()
    else {
        return Ok(None);
    };
    let string_index = batch_column_index(batch, &filter.column)?;
    let Some(strings) = view.utf8(string_index) else {
        return collect_dense_right_counts_fast_like_batch(
            batch.clone(),
            key_column,
            filter,
            counts,
        );
    };
    let finders = filter
        .parts
        .iter()
        .map(|part| Finder::new(part))
        .collect::<Vec<_>>();
    if keys.null_count() == 0 && strings.null_count() == 0 {
        let key_values = keys.values().as_ref();
        if let Some((first, second)) = fast_like_two_substring_finders(&finders) {
            for (row, &key) in key_values.iter().enumerate() {
                if fast_like_two_substrings_row_matches_non_null(
                    strings,
                    row,
                    &filter.parts[0],
                    &filter.parts[1],
                    first,
                    second,
                    filter.negated,
                ) {
                    dense_count_increment(counts, key)?;
                }
            }
            return Ok(Some(()));
        }
        for (row, &key) in key_values.iter().enumerate() {
            if fast_like_substrings_row_matches_non_null(
                strings,
                row,
                &filter.parts,
                &finders,
                filter.negated,
            ) {
                dense_count_increment(counts, key)?;
            }
        }
        return Ok(Some(()));
    }
    if let Some((first, second)) = fast_like_two_substring_finders(&finders) {
        for row in 0..keys.len() {
            if keys.is_valid(row)
                && fast_like_two_substrings_row_matches(
                    strings,
                    row,
                    &filter.parts[0],
                    &filter.parts[1],
                    first,
                    second,
                    filter.negated,
                )
            {
                dense_count_increment(counts, keys.value(row))?;
            }
        }
        return Ok(Some(()));
    }
    for row in 0..keys.len() {
        if keys.is_valid(row)
            && fast_like_substrings_row_matches(
                strings,
                row,
                &filter.parts,
                &finders,
                filter.negated,
            )
        {
            dense_count_increment(counts, keys.value(row))?;
        }
    }
    Ok(Some(()))
}

pub(super) fn dense_count_increment(counts: &mut Vec<u32>, key: i64) -> Result<()> {
    if !(0..=10_000_000).contains(&key) {
        return Err(DodamError::InvalidFilter(
            "dense count key out of supported range".to_string(),
        ));
    }
    let index = key as usize;
    if counts.len() <= index {
        counts.resize(index + 1, 0);
    }
    counts[index] = counts[index]
        .checked_add(1)
        .ok_or_else(|| DodamError::InvalidFilter("dense count overflow".to_string()))?;
    Ok(())
}

pub(super) fn dense_right_count_row_group_map_enabled() -> bool {
    std::env::var("DODAM_DISABLE_DENSE_RIGHT_COUNT_ROW_GROUP_MAP")
        .map(|value| !matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(true)
}

pub(super) fn dense_right_count_row_group_chunk() -> usize {
    std::env::var("DODAM_DENSE_RIGHT_COUNT_ROW_GROUP_CHUNK")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(16)
}

#[derive(Clone)]
pub(super) struct FastLikeSubstrings {
    column: String,
    parts: Vec<Vec<u8>>,
    negated: bool,
}

pub(super) fn fast_like_distribution_enabled() -> bool {
    std::env::var("DODAM_DISABLE_FAST_LIKE_DISTRIBUTION")
        .map(|value| !matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(true)
}

pub(super) fn fast_like_substrings_filter(expr: &Expr) -> Option<FastLikeSubstrings> {
    match expr {
        Expr::Like {
            column,
            pattern,
            negated,
            escape,
            case_insensitive,
        } => {
            if *case_insensitive
                || escape.is_some()
                || !pattern.starts_with('%')
                || !pattern.ends_with('%')
                || pattern.contains('_')
            {
                return None;
            }
            let parts = pattern
                .split('%')
                .filter(|part| !part.is_empty())
                .map(|part| part.as_bytes().to_vec())
                .collect::<Vec<_>>();
            (!parts.is_empty()).then(|| FastLikeSubstrings {
                column: column.clone(),
                parts,
                negated: *negated,
            })
        }
        Expr::Not(expr) => {
            let mut filter = fast_like_substrings_filter(expr)?;
            filter.negated = !filter.negated;
            Some(filter)
        }
        _ => None,
    }
}

pub(super) fn fast_like_substrings_row_matches(
    strings: &StringArray,
    row: usize,
    parts: &[Vec<u8>],
    finders: &[Finder<'_>],
    negated: bool,
) -> bool {
    if strings.is_null(row) {
        return false;
    }
    let mut haystack = bytes_string_parts(strings.value_offsets(), strings.value_data(), row);
    for (part, finder) in parts.iter().zip(finders) {
        let Some(index) = finder.find(haystack) else {
            return negated;
        };
        haystack = &haystack[index + part.len()..];
    }
    !negated
}

pub(super) fn fast_like_substrings_row_matches_non_null(
    strings: &StringArray,
    row: usize,
    parts: &[Vec<u8>],
    finders: &[Finder<'_>],
    negated: bool,
) -> bool {
    let mut haystack = bytes_string_parts(strings.value_offsets(), strings.value_data(), row);
    for (part, finder) in parts.iter().zip(finders) {
        let Some(index) = finder.find(haystack) else {
            return negated;
        };
        haystack = &haystack[index + part.len()..];
    }
    !negated
}

pub(super) fn fast_like_two_substring_finders<'a>(
    finders: &'a [Finder<'a>],
) -> Option<(&'a Finder<'a>, &'a Finder<'a>)> {
    if let [first, second] = finders {
        Some((first, second))
    } else {
        None
    }
}

pub(super) fn fast_like_two_substrings_row_matches(
    strings: &StringArray,
    row: usize,
    first_part: &[u8],
    second_part: &[u8],
    first_finder: &Finder<'_>,
    second_finder: &Finder<'_>,
    negated: bool,
) -> bool {
    if strings.is_null(row) {
        return false;
    }
    fast_like_two_substrings_row_matches_non_null(
        strings,
        row,
        first_part,
        second_part,
        first_finder,
        second_finder,
        negated,
    )
}

pub(super) fn fast_like_two_substrings_row_matches_non_null(
    strings: &StringArray,
    row: usize,
    first_part: &[u8],
    _second_part: &[u8],
    first_finder: &Finder<'_>,
    second_finder: &Finder<'_>,
    negated: bool,
) -> bool {
    let haystack = bytes_string_parts(strings.value_offsets(), strings.value_data(), row);
    let matched = first_finder
        .find(haystack)
        .and_then(|index| second_finder.find(&haystack[index + first_part.len()..]))
        .is_some();
    matched != negated
}

pub(super) fn expr_is_like_only(expr: &Expr) -> bool {
    match expr {
        Expr::Like { .. } => true,
        Expr::Not(expr) => expr_is_like_only(expr),
        Expr::And(left, right) | Expr::Or(left, right) => {
            expr_is_like_only(left) && expr_is_like_only(right)
        }
        Expr::Boolean(_)
        | Expr::Comparison(_)
        | Expr::ColumnComparison { .. }
        | Expr::InList { .. }
        | Expr::IsNull { .. } => false,
    }
}

pub(super) fn parquet_column_is_non_nullable(path: &PathBuf, column: &str) -> Result<bool> {
    if !path.is_file() {
        return Ok(false);
    }
    let file = File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    Ok(builder
        .schema()
        .fields()
        .iter()
        .find(|field| field.name() == column)
        .is_some_and(|field| !field.is_nullable()))
}

pub(super) async fn collect_left_count_distribution(
    engine: &DodamEngine,
    left_path: &PathBuf,
    join: &SqlJoin,
    dense_counts: &[u32],
    batch_size: usize,
) -> Result<Vec<GroupAggregateResult>> {
    let mut left_stream = engine
        .scan_parquet_batches(
            left_path.clone(),
            batch_size,
            None,
            Projection::Columns(vec![join.left_keys[0].clone()]),
            None,
        )
        .await?;
    let left_key_index_hint = Some(0);
    let mut distribution = Vec::<u64>::new();
    while let Some(batch) = left_stream.next() {
        let batch = batch?;
        let key_index =
            batch_projected_column_index(&batch, left_key_index_hint, &join.left_keys[0])?;
        let Some(keys) = batch
            .column(key_index)
            .as_any()
            .downcast_ref::<Int64Array>()
        else {
            return Ok(Vec::new());
        };
        if keys.null_count() == 0 {
            for key in keys.values().as_ref() {
                let count = if *key >= 0 {
                    dense_counts.get(*key as usize).copied().unwrap_or(0)
                } else {
                    0
                };
                let index = count as usize;
                if distribution.len() <= index {
                    distribution.resize(index + 1, 0);
                }
                distribution[index] += 1;
            }
            continue;
        }
        for row in 0..batch.num_rows() {
            let count = if keys.is_valid(row) {
                let key = keys.value(row);
                if key >= 0 {
                    dense_counts.get(key as usize).copied().unwrap_or(0)
                } else {
                    0
                }
            } else {
                0
            };
            let index = count as usize;
            if distribution.len() <= index {
                distribution.resize(index + 1, 0);
            }
            distribution[index] += 1;
        }
    }
    Ok(distribution
        .into_iter()
        .enumerate()
        .filter(|(_, rows)| *rows > 0)
        .map(|(count, rows)| GroupAggregateResult {
            keys: vec![GroupValue::UInt64(Some(count as u64))],
            values: vec![AggregateResult {
                expr: AggregateExpr::CountStar,
                value: AggregateValue::Count(rows),
            }],
        })
        .collect())
}

pub(super) fn resolve_inner_output_column_index(inner: &SqlQuery, column: &str) -> Option<usize> {
    if let Some(index) = inner
        .group_by
        .iter()
        .position(|group| same_join_column(group, column))
    {
        return Some(index);
    }
    if let Some(index) = inner
        .aggregates
        .iter()
        .position(|aggregate| aggregate.to_string() == column)
    {
        return Some(inner.group_by.len() + index);
    }
    let (_, target) = inner.aliases.iter().find(|(alias, _)| alias == column)?;
    if let Some(index) = inner
        .group_by
        .iter()
        .position(|group| same_join_column(group, target))
    {
        return Some(index);
    }
    inner
        .aggregates
        .iter()
        .position(|aggregate| aggregate.to_string() == *target)
        .map(|index| inner.group_by.len() + index)
}

pub(super) fn try_count_derived_aggregate_groups(
    inner_metrics: &AggregateMetrics,
    inner_batches: &[RecordBatch],
    group_by: &[String],
    projection: &ParsedProjection,
    filter: Option<&FilterExpr>,
    having: Option<&FilterExpr>,
    order_by: Option<&SortKey>,
    limit: Option<usize>,
) -> Result<Option<QueryOutput>> {
    if group_by.len() != 1
        || !matches!(projection.aggregates.as_slice(), [AggregateExpr::CountStar])
        || !projection.aggregate_expressions.is_empty()
        || projection_requires_expression_path(&projection.expressions)
        || filter.is_some()
        || having.is_some()
        || inner_metrics.groups.is_empty()
    {
        return Ok(None);
    }
    let Some(schema_batch) = inner_batches.first() else {
        return Ok(None);
    };
    let Some(output_column_index) = schema_batch
        .schema()
        .fields()
        .iter()
        .position(|field| field.name() == &group_by[0])
    else {
        return Ok(None);
    };
    let inner_key_count = inner_metrics.groups[0].keys.len();
    let mut counts: HashMap<GroupValue, u64> = HashMap::new();
    for group in &inner_metrics.groups {
        let key = if output_column_index < inner_key_count {
            group.keys[output_column_index].clone()
        } else {
            let value_index = output_column_index - inner_key_count;
            let Some(value) = group.values.get(value_index) else {
                return Ok(None);
            };
            let Some(key) = aggregate_value_to_group_value(&value.value) else {
                return Ok(None);
            };
            key
        };
        *counts.entry(key).or_insert(0) += 1;
    }
    let mut groups = counts
        .into_iter()
        .map(|(key, count)| GroupAggregateResult {
            keys: vec![key],
            values: vec![AggregateResult {
                expr: AggregateExpr::CountStar,
                value: AggregateValue::Count(count),
            }],
        })
        .collect::<Vec<_>>();
    groups.sort_by(|left, right| left.keys[0].to_string().cmp(&right.keys[0].to_string()));
    let metrics = AggregateMetrics {
        fragments: 1,
        batches: 1,
        rows: inner_metrics.groups.len(),
        values: Vec::new(),
        groups,
        ..AggregateMetrics::default()
    };
    let mut batches = aggregate_metrics_to_batches(&metrics, group_by, &projection.aggregates)?;
    batches = apply_output_order_limit(batches, order_by, limit, 0)?;
    batches = rename_output_batches(batches, &projection.aliases)?;
    Ok(Some(QueryOutput::Aggregate { metrics, batches }))
}

pub(super) fn aggregate_value_to_group_value(value: &AggregateValue) -> Option<GroupValue> {
    match value {
        AggregateValue::Count(value) => Some(GroupValue::UInt64(Some(*value))),
        AggregateValue::Int64(value) => Some(GroupValue::Int64(*value)),
        AggregateValue::Date32(value) => Some(GroupValue::Date32(*value)),
        AggregateValue::Date64(value) => Some(GroupValue::Date64(*value)),
        AggregateValue::Decimal128(value, precision, scale) => {
            Some(GroupValue::Decimal128(*value, *precision, *scale))
        }
        AggregateValue::Utf8(value) => Some(GroupValue::Utf8(value.clone())),
        AggregateValue::Float64(_) | AggregateValue::TimestampMillisecond(_, _) => None,
    }
}
