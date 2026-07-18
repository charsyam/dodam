use super::*;

#[derive(Clone)]
struct DecimalProductSumSpec {
    sum_expr: AggregateExpr,
    left_column: String,
    right_column: String,
    projection: Vec<String>,
    predicates: Vec<PrimitivePredicate>,
}

#[derive(Clone)]
enum PrimitivePredicate {
    Date32 {
        index: usize,
        min: Option<i32>,
        max: Option<i32>,
    },
    Decimal128 {
        index: usize,
        scale: i64,
        min: Option<i64>,
        max: Option<i64>,
    },
}

#[derive(Default)]
struct DecimalProductSumState {
    sum: f64,
    count: u64,
    rows: usize,
    batches: usize,
}

pub(super) async fn try_collect_filtered_decimal_product_sum_scan_fold(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    filter: Option<FilterExpr>,
    aggregates: &[AggregateExpr],
    expressions: &[ProjectionExpression],
) -> Result<Option<AggregateMetrics>> {
    let Some(filter) = filter.as_ref() else {
        return Ok(None);
    };
    if !path.exists() {
        return Ok(None);
    }
    let Some(spec) =
        DecimalProductSumSpec::try_new(engine, &path, filter, aggregates, expressions)?
    else {
        return Ok(None);
    };
    let started = Instant::now();
    let state = engine
        .parquet_scan_accumulate_chunks_view(
            path,
            batch_size,
            Projection::Columns(spec.projection.clone()),
            scan_aggregate_row_group_chunk(),
            8,
            scan_aggregate_fusion_enabled(),
            DecimalProductSumState::default,
            DecimalProductSumState::default,
            {
                let spec = spec.clone();
                move |view, state| {
                    consume_decimal_product_sum_view(view, &spec, state)?;
                    Ok(Some(()))
                }
            },
            |state, partial| {
                state.sum += partial.sum;
                state.count += partial.count;
                state.rows += partial.rows;
                state.batches += partial.batches;
            },
            "filtered decimal product sum",
        )
        .await?;
    Ok(Some(AggregateMetrics {
        fragments: 1,
        batches: state.batches,
        rows: state.rows,
        values: vec![AggregateResult {
            expr: spec.sum_expr,
            value: AggregateValue::Float64((state.count > 0).then_some(state.sum)),
        }],
        aggregate_nanos: started.elapsed().as_nanos().min(u64::MAX as u128) as u64,
        ..AggregateMetrics::default()
    }))
}

impl DecimalProductSumSpec {
    fn try_new(
        engine: &DodamEngine,
        path: &Path,
        filter: &FilterExpr,
        aggregates: &[AggregateExpr],
        expressions: &[ProjectionExpression],
    ) -> Result<Option<Self>> {
        let [AggregateExpr::Sum(sum_column)] = aggregates else {
            return Ok(None);
        };
        let [expression] = expressions else {
            return Ok(None);
        };
        if expression.output_name != *sum_column {
            return Ok(None);
        }
        let Some((left_column, right_column)) = decimal_product_columns(&expression.expr) else {
            return Ok(None);
        };
        let Some((left_precision, _)) = engine.parquet_decimal128_type(path, left_column)? else {
            return Ok(None);
        };
        let Some((right_precision, _)) = engine.parquet_decimal128_type(path, right_column)? else {
            return Ok(None);
        };
        if left_precision > 18 || right_precision > 18 {
            return Ok(None);
        }
        let mut projection = Vec::new();
        add_column_once(&mut projection, left_column.to_string());
        add_column_once(&mut projection, right_column.to_string());
        for column in filter.referenced_columns() {
            add_column_once(&mut projection, column);
        }
        let mut predicates = Vec::new();
        for column in filter.referenced_columns() {
            if let Some(predicate) =
                primitive_predicate_for_column(engine, path, filter.expr(), &projection, &column)?
            {
                predicates.push(predicate);
            } else {
                return Ok(None);
            }
        }
        if predicates.is_empty() {
            return Ok(None);
        }
        Ok(Some(Self {
            sum_expr: aggregates[0].clone(),
            left_column: left_column.to_string(),
            right_column: right_column.to_string(),
            projection,
            predicates,
        }))
    }
}

fn decimal_product_columns(expr: &ScalarSqlExpression) -> Option<(&str, &str)> {
    let ScalarSqlExpression::Binary { left, op, right } = expr else {
        return None;
    };
    if *op != BinaryOperator::Multiply {
        return None;
    }
    let ScalarSqlExpression::Column(left) = left.as_ref() else {
        return None;
    };
    let ScalarSqlExpression::Column(right) = right.as_ref() else {
        return None;
    };
    Some((left.as_str(), right.as_str()))
}

fn primitive_predicate_for_column(
    engine: &DodamEngine,
    path: &Path,
    expr: &Expr,
    projection: &[String],
    column: &str,
) -> Result<Option<PrimitivePredicate>> {
    let index = projection
        .iter()
        .position(|candidate| candidate == column)
        .expect("predicate column is projected");
    if engine.parquet_is_date32_column(path, column)? {
        let Some((min, max)) = primitive_i32_bounds(expr, column, literal_to_date32_local)? else {
            return Ok(None);
        };
        return Ok(Some(PrimitivePredicate::Date32 { index, min, max }));
    }
    if let Some((precision, scale)) = engine.parquet_decimal128_type(path, column)? {
        if precision > 18 {
            return Ok(None);
        }
        let scale_value = decimal_scale_i64_local(scale)?;
        let Some((min, max)) = primitive_i64_bounds(expr, column, |literal| {
            literal_to_decimal_i64_local(literal, scale_value)
        })?
        else {
            return Ok(None);
        };
        return Ok(Some(PrimitivePredicate::Decimal128 {
            index,
            scale: scale_value,
            min,
            max,
        }));
    }
    Ok(None)
}

fn primitive_i32_bounds<F>(
    expr: &Expr,
    column: &str,
    convert: F,
) -> Result<Option<(Option<i32>, Option<i32>)>>
where
    F: Fn(&LiteralValue) -> Option<i32> + Copy,
{
    let Some((min, max)) = primitive_bounds(expr, column, convert)? else {
        return Ok(None);
    };
    Ok(Some((min, max)))
}

fn primitive_i64_bounds<F>(
    expr: &Expr,
    column: &str,
    convert: F,
) -> Result<Option<(Option<i64>, Option<i64>)>>
where
    F: Fn(&LiteralValue) -> Option<i64> + Copy,
{
    primitive_bounds(expr, column, convert)
}

fn primitive_bounds<T, F>(
    expr: &Expr,
    column: &str,
    convert: F,
) -> Result<Option<(Option<T>, Option<T>)>>
where
    T: BoundValue,
    F: Fn(&LiteralValue) -> Option<T> + Copy,
{
    let mut min = None;
    let mut max = None;
    if !collect_primitive_bounds(expr, column, convert, &mut min, &mut max)? {
        return Ok(None);
    }
    Ok(Some((min, max)))
}

fn collect_primitive_bounds<T, F>(
    expr: &Expr,
    column: &str,
    convert: F,
    min: &mut Option<T>,
    max: &mut Option<T>,
) -> Result<bool>
where
    T: BoundValue,
    F: Fn(&LiteralValue) -> Option<T> + Copy,
{
    match expr {
        Expr::Boolean(Some(true)) => Ok(true),
        Expr::And(left, right) => Ok(collect_primitive_bounds(left, column, convert, min, max)?
            && collect_primitive_bounds(right, column, convert, min, max)?),
        Expr::Comparison(comparison) if comparison.column == column => {
            let Some(value) = convert(&comparison.value) else {
                return Ok(false);
            };
            apply_bound(comparison.op, value, min, max)
        }
        Expr::Comparison(_) => Ok(true),
        _ => Ok(false),
    }
}

trait BoundValue: Copy + Ord {
    fn checked_next(self) -> Option<Self>;
    fn checked_prev(self) -> Option<Self>;
}

impl BoundValue for i32 {
    fn checked_next(self) -> Option<Self> {
        self.checked_add(1)
    }

    fn checked_prev(self) -> Option<Self> {
        self.checked_sub(1)
    }
}

impl BoundValue for i64 {
    fn checked_next(self) -> Option<Self> {
        self.checked_add(1)
    }

    fn checked_prev(self) -> Option<Self> {
        self.checked_sub(1)
    }
}

fn apply_bound<T: BoundValue>(
    op: ComparisonOp,
    value: T,
    min: &mut Option<T>,
    max: &mut Option<T>,
) -> Result<bool> {
    match op {
        ComparisonOp::Eq => {
            *min = Some(min.map_or(value, |current| current.max(value)));
            *max = Some(max.map_or(value, |current| current.min(value)));
            Ok(true)
        }
        ComparisonOp::Gt => {
            let Some(value) = value.checked_next() else {
                return Ok(false);
            };
            *min = Some(min.map_or(value, |current| current.max(value)));
            Ok(true)
        }
        ComparisonOp::GtEq => {
            *min = Some(min.map_or(value, |current| current.max(value)));
            Ok(true)
        }
        ComparisonOp::Lt => {
            let Some(value) = value.checked_prev() else {
                return Ok(false);
            };
            *max = Some(max.map_or(value, |current| current.min(value)));
            Ok(true)
        }
        ComparisonOp::LtEq => {
            *max = Some(max.map_or(value, |current| current.min(value)));
            Ok(true)
        }
        ComparisonOp::NotEq => Ok(false),
    }
}

fn consume_decimal_product_sum_view(
    view: BatchView<'_>,
    spec: &DecimalProductSumSpec,
    state: &mut DecimalProductSumState,
) -> Result<()> {
    if view.num_columns() != spec.projection.len() {
        let Some(batch) = view.try_record_batch() else {
            return Err(DodamError::UnsupportedSql(
                "filtered decimal product sum raw vector layout mismatch".to_string(),
            ));
        };
        return consume_decimal_product_sum_batch(batch, spec, state);
    }
    let (Some(left), Some(right)) = (view.decimal128_vector(0), view.decimal128_vector(1)) else {
        let Some(batch) = view.try_record_batch() else {
            return Err(DodamError::UnsupportedSql(
                "filtered decimal product sum requires decimal inputs".to_string(),
            ));
        };
        return consume_decimal_product_sum_batch(batch, spec, state);
    };
    let Some(left_scale) = left.scale_i64() else {
        return consume_decimal_product_sum_batch_required(view, spec, state);
    };
    let Some(right_scale) = right.scale_i64() else {
        return consume_decimal_product_sum_batch_required(view, spec, state);
    };
    if left.precision() > 18 || right.precision() > 18 {
        return consume_decimal_product_sum_batch_required(view, spec, state);
    }
    let left_values = left.raw_values();
    let right_values = right.raw_values();
    let revenue_scale = 1.0 / ((left_scale as f64) * (right_scale as f64));
    state.rows += view.num_rows();
    state.batches += 1;
    if left.null_count() == 0 && right.null_count() == 0 && predicates_are_null_free(view, spec) {
        for row in 0..view.num_rows() {
            if predicates_match_row(view, spec, row)? {
                state.sum +=
                    (left_values[row] as i64 * right_values[row] as i64) as f64 * revenue_scale;
                state.count += 1;
            }
        }
        return Ok(());
    }
    for row in 0..view.num_rows() {
        if left.is_null(row) || right.is_null(row) || !predicates_match_row(view, spec, row)? {
            continue;
        }
        state.sum += (left_values[row] as i64 * right_values[row] as i64) as f64 * revenue_scale;
        state.count += 1;
    }
    Ok(())
}

fn consume_decimal_product_sum_batch_required(
    view: BatchView<'_>,
    spec: &DecimalProductSumSpec,
    state: &mut DecimalProductSumState,
) -> Result<()> {
    let Some(batch) = view.try_record_batch() else {
        return Err(DodamError::UnsupportedSql(
            "filtered decimal product sum fallback requires RecordBatch".to_string(),
        ));
    };
    consume_decimal_product_sum_batch(batch, spec, state)
}

fn consume_decimal_product_sum_batch(
    batch: &RecordBatch,
    spec: &DecimalProductSumSpec,
    state: &mut DecimalProductSumState,
) -> Result<()> {
    let left = batch_column(batch, &spec.left_column)?;
    let right = batch_column(batch, &spec.right_column)?;
    let Some(left) = decimal_input(left)? else {
        return Err(DodamError::UnsupportedSql(
            "filtered decimal product sum left input must be Decimal128".to_string(),
        ));
    };
    let Some(right) = decimal_input(right)? else {
        return Err(DodamError::UnsupportedSql(
            "filtered decimal product sum right input must be Decimal128".to_string(),
        ));
    };
    let Some(left_scale) = left.scale_i64() else {
        return Err(DodamError::UnsupportedSql(
            "filtered decimal product sum left scale is too large".to_string(),
        ));
    };
    let Some(right_scale) = right.scale_i64() else {
        return Err(DodamError::UnsupportedSql(
            "filtered decimal product sum right scale is too large".to_string(),
        ));
    };
    let revenue_scale = 1.0 / ((left_scale as f64) * (right_scale as f64));
    let left_values = left.raw_values();
    let right_values = right.raw_values();
    state.rows += batch.num_rows();
    state.batches += 1;
    for row in 0..batch.num_rows() {
        if left.is_null(row) || right.is_null(row) || !batch_predicates_match_row(batch, spec, row)?
        {
            continue;
        }
        state.sum += (left_values[row] as i64 * right_values[row] as i64) as f64 * revenue_scale;
        state.count += 1;
    }
    Ok(())
}

fn predicates_are_null_free(view: BatchView<'_>, spec: &DecimalProductSumSpec) -> bool {
    spec.predicates.iter().all(|predicate| match predicate {
        PrimitivePredicate::Date32 { index, .. } => view
            .date32_vector(*index)
            .is_some_and(|values| values.values_if_null_free().is_some()),
        PrimitivePredicate::Decimal128 { index, .. } => view
            .decimal128_vector(*index)
            .is_some_and(|values| values.null_count() == 0),
    })
}

fn predicates_match_row(
    view: BatchView<'_>,
    spec: &DecimalProductSumSpec,
    row: usize,
) -> Result<bool> {
    for predicate in &spec.predicates {
        match predicate {
            PrimitivePredicate::Date32 { index, min, max } => {
                let Some(values) = view.date32_vector(*index) else {
                    return Ok(false);
                };
                if values.is_null(row) || !i32_in_bounds(values.value(row), *min, *max) {
                    return Ok(false);
                }
            }
            PrimitivePredicate::Decimal128 {
                index,
                scale,
                min,
                max,
            } => {
                let Some(values) = view.decimal128_vector(*index) else {
                    return Ok(false);
                };
                if values.is_null(row) || values.scale_i64() != Some(*scale) {
                    return Ok(false);
                }
                let value = values.raw_values()[row] as i64;
                if !i64_in_bounds(value, *min, *max) {
                    return Ok(false);
                }
            }
        }
    }
    Ok(true)
}

fn batch_predicates_match_row(
    batch: &RecordBatch,
    spec: &DecimalProductSumSpec,
    row: usize,
) -> Result<bool> {
    for predicate in &spec.predicates {
        match predicate {
            PrimitivePredicate::Date32 { index, min, max } => {
                let Some(values) = batch.column(*index).as_any().downcast_ref::<Date32Array>()
                else {
                    return Ok(false);
                };
                if values.is_null(row) || !i32_in_bounds(values.value(row), *min, *max) {
                    return Ok(false);
                }
            }
            PrimitivePredicate::Decimal128 {
                index,
                scale,
                min,
                max,
            } => {
                let Some(input) = decimal_input(batch.column(*index))? else {
                    return Ok(false);
                };
                if input.is_null(row) || input.scale_i64() != Some(*scale) {
                    return Ok(false);
                }
                let value = input.raw_values()[row] as i64;
                if !i64_in_bounds(value, *min, *max) {
                    return Ok(false);
                }
            }
        }
    }
    Ok(true)
}

#[inline]
fn i32_in_bounds(value: i32, min: Option<i32>, max: Option<i32>) -> bool {
    min.is_none_or(|min| value >= min) && max.is_none_or(|max| value <= max)
}

#[inline]
fn i64_in_bounds(value: i64, min: Option<i64>, max: Option<i64>) -> bool {
    min.is_none_or(|min| value >= min) && max.is_none_or(|max| value <= max)
}

fn literal_to_decimal_i64_local(value: &LiteralValue, scale: i64) -> Option<i64> {
    match value {
        LiteralValue::Int64(value) => value.checked_mul(scale),
        LiteralValue::Float64(value) if value.is_finite() => {
            Some((value * scale as f64).round() as i64)
        }
        LiteralValue::Utf8(value) => parse_decimal_literal_i64(value, scale),
        _ => None,
    }
}

fn parse_decimal_literal_i64(value: &str, scale: i64) -> Option<i64> {
    let value = value.trim();
    if value.is_empty() {
        return None;
    }
    let negative = value.starts_with('-');
    let unsigned = value.strip_prefix(['-', '+']).unwrap_or(value);
    let (whole, fraction) = unsigned.split_once('.').unwrap_or((unsigned, ""));
    if whole.is_empty() && fraction.is_empty() {
        return None;
    }
    if !whole.bytes().all(|byte| byte.is_ascii_digit())
        || !fraction.bytes().all(|byte| byte.is_ascii_digit())
    {
        return None;
    }
    let mut raw = whole.parse::<i64>().ok()?.checked_mul(scale)?;
    if !fraction.is_empty() {
        let mut divisor = 1_i64;
        for _ in 0..fraction.len() {
            divisor = divisor.checked_mul(10)?;
        }
        let fraction_raw = fraction
            .parse::<i64>()
            .ok()?
            .checked_mul(scale)?
            .checked_div(divisor)?;
        raw = raw.checked_add(fraction_raw)?;
    }
    Some(if negative { -raw } else { raw })
}

fn literal_to_date32_local(value: &LiteralValue) -> Option<i32> {
    match value {
        LiteralValue::Int64(value) => i32::try_from(*value).ok(),
        LiteralValue::Utf8(value) => {
            let (year, month, day) = parse_ymd(value).ok()?;
            let days = days_from_civil(year, month, day).ok()?;
            i32::try_from(days).ok()
        }
        _ => None,
    }
}

fn decimal_scale_i64_local(scale: i8) -> Result<i64> {
    let scale = u32::try_from(scale)
        .map_err(|_| DodamError::UnsupportedSql("negative decimal scale".to_string()))?;
    10_i64
        .checked_pow(scale)
        .ok_or_else(|| DodamError::UnsupportedSql("decimal scale overflow".to_string()))
}
