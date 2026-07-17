use super::*;

pub(super) fn simplify_filtered_aggregates_with_parquet_stats(
    engine: &DodamEngine,
    path: &Path,
    specs: &[NativeFilteredAggregateSpec],
) -> Result<Vec<NativeFilteredAggregateSpec>> {
    if specs.is_empty() || !metadata_filtered_aggregate_predicate_simplify_enabled() {
        return Ok(specs.to_vec());
    }
    specs
        .iter()
        .map(|spec| {
            let mut spec = spec.clone();
            if let Some(value) = parquet_metadata_predicate_truth(engine, path, &spec.condition)? {
                spec.condition = sql_boolean_expr(value);
            }
            Ok(spec)
        })
        .collect()
}

fn metadata_filtered_aggregate_predicate_simplify_enabled() -> bool {
    !std::env::var("DODAM_DISABLE_METADATA_FILTERED_AGGREGATE_PREDICATES")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

fn sql_boolean_expr(value: bool) -> SqlExpr {
    SqlExpr::Value(Value::Boolean(value).with_empty_span())
}

fn parquet_metadata_predicate_truth(
    engine: &DodamEngine,
    path: &Path,
    expr: &SqlExpr,
) -> Result<Option<bool>> {
    match expr {
        SqlExpr::IsNotNull(inner) => {
            let column = sql_column_name(inner, None)?;
            parquet_column_is_never_null(engine, path, &column)
        }
        SqlExpr::BinaryOp { left, op, right }
            if matches!(
                op,
                BinaryOperator::Eq
                    | BinaryOperator::NotEq
                    | BinaryOperator::Gt
                    | BinaryOperator::GtEq
                    | BinaryOperator::Lt
                    | BinaryOperator::LtEq
            ) =>
        {
            parquet_metadata_column_literal_predicate_truth(engine, path, left, op, right).or_else(
                |_| {
                    parquet_metadata_column_literal_predicate_truth(
                        engine,
                        path,
                        right,
                        &reverse_binary_operator(op),
                        left,
                    )
                },
            )
        }
        _ => Ok(None),
    }
}

fn parquet_column_is_never_null(
    engine: &DodamEngine,
    path: &Path,
    column: &str,
) -> Result<Option<bool>> {
    let Some(ranges) = engine.parquet_primitive_column_min_max_by_row_group(path, column)? else {
        return Ok(None);
    };
    if ranges.iter().all(|range| range.null_count == Some(0)) {
        Ok(Some(true))
    } else {
        Ok(None)
    }
}

fn parquet_metadata_column_literal_predicate_truth(
    engine: &DodamEngine,
    path: &Path,
    column_expr: &SqlExpr,
    op: &BinaryOperator,
    literal_expr: &SqlExpr,
) -> Result<Option<bool>> {
    let Ok(column) = sql_column_name(column_expr, None) else {
        return Ok(None);
    };
    let Ok(literal) = sql_literal_value(literal_expr) else {
        return Ok(None);
    };
    if matches!(literal, LiteralValue::Null) {
        return Ok(None);
    }
    let Some(ranges) = engine.parquet_primitive_column_min_max_by_row_group(path, &column)? else {
        return Ok(None);
    };
    if ranges.is_empty() || ranges.iter().any(|range| range.null_count != Some(0)) {
        return Ok(None);
    }
    let Some(literal) = metadata_literal_to_i128_for_ranges(&literal, &ranges)? else {
        return Ok(None);
    };
    let always_true = ranges
        .iter()
        .all(|range| metadata_range_comparison_always_true(range.min, range.max, op, literal));
    if always_true {
        return Ok(Some(true));
    }
    let always_false = ranges
        .iter()
        .all(|range| metadata_range_comparison_always_false(range.min, range.max, op, literal));
    Ok(always_false.then_some(false))
}

fn metadata_literal_to_i128_for_ranges(
    literal: &LiteralValue,
    ranges: &[PrimitiveRowGroupMinMax],
) -> Result<Option<i128>> {
    let Some(first) = ranges.first() else {
        return Ok(None);
    };
    match &first.data_type {
        DataType::Decimal128(precision, scale) => {
            literal_as_decimal128_for_type(literal, *precision, *scale)
        }
        DataType::Date32 => literal_as_date32_for_type(literal).map(|value| value.map(i128::from)),
        DataType::Int32 | DataType::Int64 | DataType::UInt32 => {
            match literal_as_i64_for_type(literal) {
                Ok(value) => Ok(value.map(i128::from)),
                Err(DodamError::InvalidCast(_)) => Ok(None),
                Err(error) => Err(error),
            }
        }
        _ => Ok(None),
    }
}

fn metadata_range_comparison_always_true(
    min: i128,
    max: i128,
    op: &BinaryOperator,
    literal: i128,
) -> bool {
    match op {
        BinaryOperator::Eq => min == literal && max == literal,
        BinaryOperator::NotEq => literal < min || literal > max,
        BinaryOperator::Gt => min > literal,
        BinaryOperator::GtEq => min >= literal,
        BinaryOperator::Lt => max < literal,
        BinaryOperator::LtEq => max <= literal,
        _ => false,
    }
}

fn metadata_range_comparison_always_false(
    min: i128,
    max: i128,
    op: &BinaryOperator,
    literal: i128,
) -> bool {
    match op {
        BinaryOperator::Eq => literal < min || literal > max,
        BinaryOperator::NotEq => min == literal && max == literal,
        BinaryOperator::Gt => max <= literal,
        BinaryOperator::GtEq => max < literal,
        BinaryOperator::Lt => min >= literal,
        BinaryOperator::LtEq => min > literal,
        _ => false,
    }
}
