use super::*;

pub(super) fn literal_date_days(expr: &SqlExpr) -> Result<i32> {
    let LiteralValue::Utf8(value) = sql_literal_value(expr)? else {
        return Err(DodamError::UnsupportedSql(format!(
            "expected DATE expression, got {expr}"
        )));
    };
    let (year, month, day) = parse_ymd(&value)?;
    let days = days_from_civil(year, month, day)?;
    i32::try_from(days).map_err(|_| DodamError::UnsupportedSql("DATE overflow".to_string()))
}

pub(super) fn should_use_i64_set_row_filter_for_keys_auto(
    default_enabled: bool,
    keys: &HashSet<i64>,
    projected_columns: usize,
) -> bool {
    let Some((min_key, max_key)) = raw_i64_key_range(keys.iter().copied()) else {
        return false;
    };
    should_use_i64_set_row_filter_for_key_stats(
        default_enabled,
        keys.len(),
        min_key,
        max_key,
        projected_columns,
    )
}

pub(super) fn should_use_i64_set_row_filter_for_key_stats(
    default_enabled: bool,
    key_count: usize,
    min_key: i64,
    max_key: i64,
    projected_columns: usize,
) -> bool {
    if !default_enabled
        || key_count == 0
        || key_count > i64_set_row_filter_max_keys()
        || projected_columns < i64_set_row_filter_min_projected_columns()
    {
        return false;
    }
    let Some(width) = max_key
        .checked_sub(min_key)
        .and_then(|width| width.checked_add(1))
        .and_then(|width| usize::try_from(width).ok())
    else {
        return false;
    };
    if width == 0 {
        return false;
    }
    let density = key_count as f64 / width as f64;
    density <= i64_set_row_filter_max_density()
        || key_count <= i64_set_row_filter_always_allow_keys()
}

pub(super) fn env_flag_enabled(name: &str) -> bool {
    std::env::var(name)
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false)
}

pub(super) fn i64_set_row_filter_max_keys() -> usize {
    i64_set_row_filter_max_keys_with_default(1_000_000)
}

pub(super) fn i64_set_row_filter_max_keys_with_default(default_max_keys: usize) -> usize {
    std::env::var("DODAM_I64_SET_ROW_FILTER_MAX_KEYS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default_max_keys)
}

pub(super) fn i64_set_row_filter_min_projected_columns() -> usize {
    std::env::var("DODAM_I64_SET_ROW_FILTER_MIN_PROJECTED_COLUMNS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(4)
}

pub(super) fn i64_set_row_filter_max_density() -> f64 {
    std::env::var("DODAM_I64_SET_ROW_FILTER_MAX_DENSITY")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite() && *value >= 0.0)
        .unwrap_or(0.35)
}

pub(super) fn i64_set_row_filter_always_allow_keys() -> usize {
    std::env::var("DODAM_I64_SET_ROW_FILTER_ALWAYS_ALLOW_KEYS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(4096)
}

pub(super) fn i64_set_row_filter_row_group_chunk(default_chunk: usize) -> usize {
    std::env::var("DODAM_I64_SET_ROW_FILTER_ROW_GROUP_CHUNK")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default_chunk)
}

pub(super) fn dense_i32_max_entries(default_bytes: usize) -> usize {
    dense_value_max_entries::<i32>("DODAM_DENSE_I32_BYTES", default_bytes)
}

pub(super) fn dense_u8_max_entries(default_bytes: usize) -> usize {
    dense_value_max_entries::<u8>("DODAM_DENSE_U8_BYTES", default_bytes)
}

fn dense_value_max_entries<T>(env_name: &str, default_bytes: usize) -> usize {
    std::env::var(env_name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .map(|bytes| bytes / std::mem::size_of::<T>())
        .filter(|entries| *entries > 0)
        .unwrap_or_else(|| default_bytes / std::mem::size_of::<T>())
}

pub(super) fn dense_max_amplification(default_amplification: f64) -> f64 {
    std::env::var("DODAM_DENSE_MAX_AMPLIFICATION")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite() && *value >= 1.0)
        .unwrap_or(default_amplification)
}

pub(super) fn dense_i64_probe_max_key(default_bytes: usize) -> usize {
    std::env::var("DODAM_DENSE_I64_PROBE_MAX_KEY")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or_else(|| {
            dense_i64_probe_bytes(default_bytes)
                .saturating_mul(8)
                .saturating_sub(1)
        })
}

pub(super) fn dense_i64_probe_bytes(default_bytes: usize) -> usize {
    std::env::var("DODAM_DENSE_I64_PROBE_BYTES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default_bytes)
}

pub(super) fn dense_i64_rank_map_bytes(default_bytes: usize) -> usize {
    std::env::var("DODAM_DENSE_I64_RANK_MAP_BYTES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|bytes| *bytes > 0)
        .unwrap_or(default_bytes)
}

pub(super) fn dense_f64_sum_bytes(default_bytes: usize) -> usize {
    std::env::var("DODAM_DENSE_F64_SUM_BYTES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|bytes| *bytes > 0)
        .unwrap_or(default_bytes)
}

pub(super) fn raw_i64_key_range(keys: impl IntoIterator<Item = i64>) -> Option<(i64, i64)> {
    let mut iter = keys.into_iter();
    let first = iter.next()?;
    let mut min_key = first;
    let mut max_key = first;
    for key in iter {
        min_key = min_key.min(key);
        max_key = max_key.max(key);
    }
    Some((min_key, max_key))
}

pub(super) fn projection_column_count(projection: &Projection) -> usize {
    match projection {
        Projection::All => usize::MAX,
        Projection::Columns(columns) => columns.len(),
    }
}

pub(super) fn string_inequality_literal(
    conjuncts: &[SqlExpr],
    column: &str,
) -> Result<Option<String>> {
    for conjunct in conjuncts {
        let SqlExpr::BinaryOp { left, op, right } = conjunct else {
            continue;
        };
        if *op != BinaryOperator::NotEq {
            continue;
        }
        if sql_expr_column_matches(left, column) {
            if let LiteralValue::Utf8(value) = sql_literal_value(right)? {
                return Ok(Some(value));
            }
        } else if sql_expr_column_matches(right, column)
            && let LiteralValue::Utf8(value) = sql_literal_value(left)?
        {
            return Ok(Some(value));
        }
    }
    Ok(None)
}

pub(super) fn numeric_i64_equality_literal(
    conjuncts: &[SqlExpr],
    column: &str,
) -> Result<Option<i64>> {
    for conjunct in conjuncts {
        let SqlExpr::BinaryOp { left, op, right } = conjunct else {
            continue;
        };
        if *op != BinaryOperator::Eq {
            continue;
        }
        if sql_expr_column_matches(left, column) {
            if let LiteralValue::Int64(value) = sql_literal_value(right)? {
                return Ok(Some(value));
            }
        } else if sql_expr_column_matches(right, column)
            && let LiteralValue::Int64(value) = sql_literal_value(left)?
        {
            return Ok(Some(value));
        }
    }
    Ok(None)
}

pub(super) fn not_like_prefix_literal(
    conjuncts: &[SqlExpr],
    column: &str,
) -> Result<Option<String>> {
    for conjunct in conjuncts {
        let SqlExpr::Like {
            expr,
            pattern,
            negated,
            ..
        } = conjunct
        else {
            continue;
        };
        if !*negated || !sql_expr_column_matches(expr, column) {
            continue;
        }
        let LiteralValue::Utf8(pattern) = sql_literal_value(pattern)? else {
            continue;
        };
        if let Some(value) = pattern.strip_suffix('%')
            && !value.contains('%')
            && !value.contains('_')
        {
            return Ok(Some(value.to_string()));
        }
    }
    Ok(None)
}

pub(super) fn like_suffix_literal(conjuncts: &[SqlExpr], column: &str) -> Result<Option<String>> {
    for conjunct in conjuncts {
        let SqlExpr::Like {
            expr,
            pattern,
            negated,
            ..
        } = conjunct
        else {
            continue;
        };
        if *negated || !sql_expr_column_matches(expr, column) {
            continue;
        }
        let LiteralValue::Utf8(pattern) = sql_literal_value(pattern)? else {
            continue;
        };
        if let Some(value) = pattern.strip_prefix('%')
            && !value.contains('%')
            && !value.contains('_')
        {
            return Ok(Some(value.to_string()));
        }
    }
    Ok(None)
}

pub(super) fn numeric_in_i64_literals(
    conjuncts: &[SqlExpr],
    column: &str,
) -> Result<Option<HashSet<i64>>> {
    for conjunct in conjuncts {
        let SqlExpr::InList {
            expr,
            list,
            negated,
        } = conjunct
        else {
            continue;
        };
        if *negated || !sql_expr_column_matches(expr, column) {
            continue;
        }
        let mut values = HashSet::new();
        for item in list {
            values.insert(literal_as_f64(&sql_literal_value(item)?)? as i64);
        }
        return Ok(Some(values));
    }
    Ok(None)
}

pub(super) fn like_substrings_literal(expr: &SqlExpr, column: &str) -> Result<Option<Vec<String>>> {
    match expr {
        SqlExpr::Like {
            expr,
            pattern,
            negated,
            ..
        } if !*negated && sql_expr_column_matches(expr, column) => {
            let LiteralValue::Utf8(pattern) = sql_literal_value(pattern)? else {
                return Ok(None);
            };
            let parts = pattern
                .split('%')
                .filter(|part| !part.is_empty())
                .map(str::to_string)
                .collect::<Vec<_>>();
            Ok((!parts.is_empty() && !pattern.contains('_')).then_some(parts))
        }
        SqlExpr::Exists { subquery, .. }
        | SqlExpr::InSubquery { subquery, .. }
        | SqlExpr::Subquery(subquery) => {
            if let SetExpr::Select(select) = subquery.body.as_ref()
                && let Some(selection) = select.selection.as_ref()
            {
                return like_substrings_literal(selection, column);
            }
            Ok(None)
        }
        SqlExpr::BinaryOp { left, right, .. } => {
            Ok(like_substrings_literal(left, column)?.or(like_substrings_literal(right, column)?))
        }
        SqlExpr::Nested(expr) | SqlExpr::UnaryOp { expr, .. } => {
            like_substrings_literal(expr, column)
        }
        SqlExpr::InList { expr, list, .. } => {
            if let Some(parts) = like_substrings_literal(expr, column)? {
                return Ok(Some(parts));
            }
            for item in list {
                if let Some(parts) = like_substrings_literal(item, column)? {
                    return Ok(Some(parts));
                }
            }
            Ok(None)
        }
        _ => Ok(None),
    }
}

pub(super) fn merge_f64_groups<K, S>(groups: &mut HashMap<K, f64, S>, batch: HashMap<K, f64, S>)
where
    K: Eq + std::hash::Hash,
    S: BuildHasher,
{
    for (key, value) in batch {
        *groups.entry(key).or_insert(0.0) += value;
    }
}

pub(super) fn merge_maps<K, V, S>(output: &mut HashMap<K, V, S>, batch: HashMap<K, V, S>)
where
    K: Eq + std::hash::Hash,
    S: BuildHasher,
{
    output.extend(batch);
}

pub(super) fn merge_sets<K: Eq + std::hash::Hash>(output: &mut HashSet<K>, batch: HashSet<K>) {
    output.extend(batch);
}

pub(super) fn selective_i64_key_range<I>(keys: I) -> Option<(i64, i64)>
where
    I: IntoIterator<Item = i64>,
{
    let mut min_key = i64::MAX;
    let mut max_key = i64::MIN;
    let mut len = 0_usize;
    for key in keys {
        if key < 0 {
            return None;
        }
        min_key = min_key.min(key);
        max_key = max_key.max(key);
        len += 1;
    }
    selective_i64_range_from_parts(min_key, max_key, len)
}

pub(super) fn selective_i64_range_from_parts(
    min_key: i64,
    max_key: i64,
    len: usize,
) -> Option<(i64, i64)> {
    if len == 0 || min_key < 0 || max_key < min_key {
        return None;
    }
    let width = usize::try_from(max_key.checked_sub(min_key)?.checked_add(1)?).ok()?;
    (width <= len.saturating_mul(8).max(1024)).then_some((min_key, max_key))
}

pub(super) fn i64_range_pruning_predicates(column: &str, min_key: i64, max_key: i64) -> Vec<Expr> {
    vec![
        Expr::Comparison(ComparisonExpr {
            column: column.to_string(),
            op: ComparisonOp::GtEq,
            value: LiteralValue::Int64(min_key),
        }),
        Expr::Comparison(ComparisonExpr {
            column: column.to_string(),
            op: ComparisonOp::LtEq,
            value: LiteralValue::Int64(max_key),
        }),
    ]
}

pub(super) fn date32_to_ymd_string(days: i32) -> Result<String> {
    let (year, month, day) = civil_from_days(i64::from(days))?;
    Ok(format!("{year:04}-{month:02}-{day:02}"))
}

pub(super) fn numeric_between_bounds(
    conjuncts: &[SqlExpr],
    column: &str,
) -> Result<Option<(f64, f64)>> {
    for conjunct in conjuncts {
        let SqlExpr::Between {
            expr,
            negated,
            low,
            high,
        } = conjunct
        else {
            continue;
        };
        if *negated || !sql_expr_column_matches(expr, column) {
            continue;
        }
        return Ok(Some((
            literal_as_f64(&sql_literal_value(low)?)?,
            literal_as_f64(&sql_literal_value(high)?)?,
        )));
    }
    Ok(None)
}

pub(super) fn upper_numeric_bound(conjuncts: &[SqlExpr], column: &str) -> Result<Option<f64>> {
    let mut bound = None;
    for conjunct in conjuncts {
        let SqlExpr::BinaryOp { left, op, right } = conjunct else {
            continue;
        };
        if matches!(op, BinaryOperator::Lt | BinaryOperator::LtEq)
            && sql_expr_column_matches(left, column)
        {
            bound = Some(literal_as_f64(&sql_literal_value(right)?)?);
        } else if matches!(op, BinaryOperator::Gt | BinaryOperator::GtEq)
            && sql_expr_column_matches(right, column)
        {
            bound = Some(literal_as_f64(&sql_literal_value(left)?)?);
        }
    }
    Ok(bound)
}

pub(super) fn lower_numeric_bound(conjuncts: &[SqlExpr], column: &str) -> Result<Option<f64>> {
    let mut bound = None;
    for conjunct in conjuncts {
        let SqlExpr::BinaryOp { left, op, right } = conjunct else {
            continue;
        };
        if matches!(op, BinaryOperator::Gt | BinaryOperator::GtEq)
            && sql_expr_column_matches(left, column)
        {
            bound = Some(literal_as_f64(&sql_literal_value(right)?)?);
        } else if matches!(op, BinaryOperator::Lt | BinaryOperator::LtEq)
            && sql_expr_column_matches(right, column)
        {
            bound = Some(literal_as_f64(&sql_literal_value(left)?)?);
        }
    }
    Ok(bound)
}

pub(super) fn scaled_f64_to_i128(value: f64, scale: f64) -> i128 {
    (value * scale).round() as i128
}

pub(super) fn date_between_bounds(
    conjuncts: &[SqlExpr],
    column: &str,
) -> Result<Option<(i32, i32)>> {
    for conjunct in conjuncts {
        let SqlExpr::Between {
            expr,
            low,
            high,
            negated,
        } = conjunct
        else {
            continue;
        };
        if *negated || !sql_expr_column_matches(expr, column) {
            continue;
        }
        return Ok(Some((literal_date_days(low)?, literal_date_days(high)?)));
    }
    Ok(None)
}

pub(super) fn select_item_alias(item: &SelectItem) -> Option<String> {
    match item {
        SelectItem::ExprWithAlias { alias, .. } => Some(alias.value.clone()),
        _ => None,
    }
}

pub(super) fn string_equality_literal(
    conjuncts: &[SqlExpr],
    column: &str,
) -> Result<Option<String>> {
    for conjunct in conjuncts {
        let SqlExpr::BinaryOp { left, op, right } = conjunct else {
            continue;
        };
        if *op != BinaryOperator::Eq {
            continue;
        }
        if sql_expr_column_matches(left, column) {
            if let LiteralValue::Utf8(value) = sql_literal_value(right)? {
                return Ok(Some(value));
            }
        } else if sql_expr_column_matches(right, column)
            && let LiteralValue::Utf8(value) = sql_literal_value(left)?
        {
            return Ok(Some(value));
        }
    }
    Ok(None)
}

pub(super) fn sql_expr_column_matches(expr: &SqlExpr, column: &str) -> bool {
    match expr {
        SqlExpr::Identifier(ident) => ident.value.eq_ignore_ascii_case(column),
        SqlExpr::CompoundIdentifier(parts) => parts
            .last()
            .is_some_and(|ident| ident.value.eq_ignore_ascii_case(column)),
        SqlExpr::Nested(expr) => sql_expr_column_matches(expr, column),
        _ => false,
    }
}

pub(super) fn single_f64_aggregate_output(name: String, value: Option<f64>) -> Result<QueryOutput> {
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(name, DataType::Float64, true)])),
        vec![Arc::new(Float64Array::from(vec![value]))],
    )?;
    Ok(QueryOutput::Aggregate {
        metrics: AggregateMetrics::default(),
        batches: vec![batch],
    })
}

pub(super) fn batch_column<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a ArrayRef> {
    let index = batch
        .schema()
        .fields()
        .iter()
        .position(|field| field.name().eq_ignore_ascii_case(name))
        .ok_or_else(|| DodamError::UnknownColumn(name.to_string()))?;
    Ok(batch.column(index))
}

pub(super) fn batch_string_column<'a>(
    batch: &'a RecordBatch,
    name: &str,
) -> Result<&'a StringArray> {
    batch_column(batch, name)?
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| DodamError::UnsupportedSql(format!("{name} must be Utf8")))
}

pub(super) fn utf8_value_is_one_byte(offsets: &[i32], data: &[u8], row: usize, byte: u8) -> bool {
    let start = offsets[row] as usize;
    let end = offsets[row + 1] as usize;
    end == start + 1 && data[start] == byte
}

pub(super) fn numeric_i64_value(column: &ArrayRef, row: usize) -> Result<Option<i64>> {
    if column.is_null(row) {
        return Ok(None);
    }
    match column.data_type() {
        DataType::Int32 => Ok(Some(i64::from(
            column
                .as_any()
                .downcast_ref::<Int32Array>()
                .expect("Int32 column")
                .value(row),
        ))),
        DataType::Int64 => Ok(Some(
            column
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("Int64 column")
                .value(row),
        )),
        data_type => Err(DodamError::UnsupportedSql(format!(
            "expected integer column, got {data_type:?}"
        ))),
    }
}

pub(super) fn numeric_f64_value(column: &ArrayRef, row: usize) -> Result<Option<f64>> {
    if column.is_null(row) {
        return Ok(None);
    }
    match column.data_type() {
        DataType::Int32 => Ok(Some(f64::from(
            column
                .as_any()
                .downcast_ref::<Int32Array>()
                .expect("Int32")
                .value(row),
        ))),
        DataType::Int64 => Ok(Some(
            column
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("Int64")
                .value(row) as f64,
        )),
        DataType::Float64 => Ok(Some(
            column
                .as_any()
                .downcast_ref::<Float64Array>()
                .expect("Float64")
                .value(row),
        )),
        DataType::Decimal128(_, scale) => Ok(Some(
            column
                .as_any()
                .downcast_ref::<Decimal128Array>()
                .expect("Decimal128")
                .value(row) as f64
                / decimal_scale_factor(*scale),
        )),
        data_type => Err(DodamError::UnsupportedSql(format!(
            "expected numeric column, got {data_type:?}"
        ))),
    }
}

pub(super) fn decimal_scale_factor(scale: i8) -> f64 {
    10_f64.powi(i32::from(scale))
}

pub(super) fn first_table_path_in_subqueries(
    expr: &SqlExpr,
    alias: &str,
) -> Result<Option<PathBuf>> {
    match expr {
        SqlExpr::Exists { subquery, .. }
        | SqlExpr::InSubquery { subquery, .. }
        | SqlExpr::Subquery(subquery) => {
            if let SetExpr::Select(select) = subquery.body.as_ref()
                && let Ok(table) = parse_from(select)
                && table_ref_alias_or_name(&table).eq_ignore_ascii_case(alias)
            {
                return Ok(Some(table.path));
            }
            if let SetExpr::Select(select) = subquery.body.as_ref()
                && let Some(selection) = select.selection.as_ref()
                && let Some(path) = first_table_path_in_subqueries(selection, alias)?
            {
                return Ok(Some(path));
            }
            Ok(None)
        }
        SqlExpr::BinaryOp { left, right, .. } => Ok(first_table_path_in_subqueries(left, alias)?
            .or(first_table_path_in_subqueries(right, alias)?)),
        SqlExpr::Nested(expr) | SqlExpr::UnaryOp { expr, .. } => {
            first_table_path_in_subqueries(expr, alias)
        }
        SqlExpr::InList { expr, list, .. } => {
            if let Some(path) = first_table_path_in_subqueries(expr, alias)? {
                return Ok(Some(path));
            }
            for item in list {
                if let Some(path) = first_table_path_in_subqueries(item, alias)? {
                    return Ok(Some(path));
                }
            }
            Ok(None)
        }
        SqlExpr::Between {
            expr, low, high, ..
        } => Ok(first_table_path_in_subqueries(expr, alias)?
            .or(first_table_path_in_subqueries(low, alias)?)
            .or(first_table_path_in_subqueries(high, alias)?)),
        _ => Ok(None),
    }
}

pub(super) fn date32_value(column: &ArrayRef, row: usize) -> Result<Option<i32>> {
    if column.is_null(row) {
        return Ok(None);
    }
    match column.data_type() {
        DataType::Date32 => Ok(Some(
            column
                .as_any()
                .downcast_ref::<Date32Array>()
                .expect("Date32")
                .value(row),
        )),
        DataType::Date64 => Ok(Some(
            (column
                .as_any()
                .downcast_ref::<Date64Array>()
                .expect("Date64")
                .value(row)
                / 86_400_000) as i32,
        )),
        DataType::Int32 => Ok(Some(
            column
                .as_any()
                .downcast_ref::<Int32Array>()
                .expect("Int32 date")
                .value(row),
        )),
        data_type => Err(DodamError::UnsupportedSql(format!(
            "expected date column, got {data_type:?}"
        ))),
    }
}

pub(super) fn bytes_string_parts<'a>(offsets: &[i32], data: &'a [u8], row: usize) -> &'a [u8] {
    let start = offsets[row] as usize;
    let end = offsets[row + 1] as usize;
    &data[start..end]
}
pub(super) fn literal_value_to_sql_expr(value: LiteralValue) -> SqlExpr {
    SqlExpr::Value(
        match value {
            LiteralValue::Null => Value::Null,
            LiteralValue::Boolean(value) => Value::Boolean(value),
            LiteralValue::Int64(value) => Value::Number(value.to_string(), false),
            LiteralValue::Float64(value) => Value::Number(value.to_string(), false),
            LiteralValue::Utf8(value) => Value::SingleQuotedString(value),
        }
        .with_empty_span(),
    )
}
