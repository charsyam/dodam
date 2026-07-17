use super::*;

pub(super) fn query_output_batches(output: QueryOutput) -> Result<Vec<RecordBatch>> {
    match output {
        QueryOutput::Scan { batches } | QueryOutput::Aggregate { batches, .. } => Ok(batches),
        QueryOutput::Explain { .. } => Err(DodamError::UnsupportedSql(
            "EXPLAIN cannot be used as a derived table".to_string(),
        )),
    }
}

pub(super) fn literal_values_from_single_column_batches(
    batches: Vec<RecordBatch>,
) -> Result<Vec<LiteralValue>> {
    let Some(schema) = batches.first().map(RecordBatch::schema) else {
        return Ok(Vec::new());
    };
    if schema.fields().len() != 1 {
        return Err(DodamError::UnsupportedSql(
            "IN subquery must return exactly one column".to_string(),
        ));
    }
    let mut values = Vec::new();
    for batch in batches {
        let column = batch.column(0);
        for row in 0..batch.num_rows() {
            if column.is_null(row) {
                values.push(LiteralValue::Null);
                continue;
            }
            values.push(literal_value_from_array(column, row)?);
        }
    }
    Ok(values)
}

pub(super) fn scalar_literal_value_from_batches(batches: Vec<RecordBatch>) -> Result<LiteralValue> {
    let values = literal_values_from_single_column_batches(batches)?;
    match values.as_slice() {
        [] => Ok(LiteralValue::Null),
        [value] => Ok(value.clone()),
        _ => Err(DodamError::UnsupportedSql(
            "scalar subquery must return at most one row".to_string(),
        )),
    }
}

pub(super) fn non_null_literal_values(list: &[SqlExpr]) -> Result<Vec<LiteralValue>> {
    let mut values = Vec::with_capacity(list.len());
    for expr in list {
        let value = sql_literal_value(expr)?;
        if !matches!(value, LiteralValue::Null) {
            values.push(value);
        }
    }
    Ok(values)
}

pub(super) fn literal_list_contains_null(list: &[SqlExpr]) -> Result<bool> {
    for expr in list {
        if matches!(sql_literal_value(expr)?, LiteralValue::Null) {
            return Ok(true);
        }
    }
    Ok(false)
}

pub(super) fn subquery_values_contain_null(values: &[LiteralValue]) -> bool {
    values
        .iter()
        .any(|value| matches!(value, LiteralValue::Null))
}

pub(super) fn non_null_subquery_values(values: Vec<LiteralValue>) -> Vec<LiteralValue> {
    values
        .into_iter()
        .filter(|value| !matches!(value, LiteralValue::Null))
        .collect()
}

pub(super) fn evaluate_literal_in_values(
    value: &LiteralValue,
    values: &[LiteralValue],
    negated: bool,
) -> Option<bool> {
    if matches!(value, LiteralValue::Null) {
        return None;
    }
    let has_null = subquery_values_contain_null(values);
    let matched = values
        .iter()
        .filter(|candidate| !matches!(candidate, LiteralValue::Null))
        .any(|candidate| {
            matches!(
                compare_literal_values(value, &BinaryOperator::Eq, candidate),
                Ok(Some(true))
            )
        });
    let result = if matched {
        Some(true)
    } else if has_null {
        None
    } else {
        Some(false)
    };
    if negated {
        result.map(|value| !value)
    } else {
        result
    }
}

pub(super) fn compare_literal_values(
    left: &LiteralValue,
    op: &BinaryOperator,
    right: &LiteralValue,
) -> Result<Option<bool>> {
    if matches!(left, LiteralValue::Null) || matches!(right, LiteralValue::Null) {
        return Ok(None);
    }
    let ordering = match (left, right) {
        (LiteralValue::Boolean(left), LiteralValue::Boolean(right)) => left.cmp(right),
        (LiteralValue::Int64(left), LiteralValue::Int64(right)) => left.cmp(right),
        (LiteralValue::Float64(left), LiteralValue::Float64(right)) => left
            .partial_cmp(right)
            .ok_or_else(|| DodamError::InvalidFilter(format!("{left} {op} {right}")))?,
        (LiteralValue::Int64(left), LiteralValue::Float64(right)) => (*left as f64)
            .partial_cmp(right)
            .ok_or_else(|| DodamError::InvalidFilter(format!("{left} {op} {right}")))?,
        (LiteralValue::Float64(left), LiteralValue::Int64(right)) => left
            .partial_cmp(&(*right as f64))
            .ok_or_else(|| DodamError::InvalidFilter(format!("{left} {op} {right}")))?,
        (LiteralValue::Utf8(left), LiteralValue::Utf8(right)) => left.cmp(right),
        _ => {
            return Err(DodamError::InvalidFilter(format!("{left} {op} {right}")));
        }
    };
    Ok(Some(match op {
        BinaryOperator::Eq => ordering.is_eq(),
        BinaryOperator::NotEq => !ordering.is_eq(),
        BinaryOperator::Gt => ordering.is_gt(),
        BinaryOperator::GtEq => ordering.is_gt() || ordering.is_eq(),
        BinaryOperator::Lt => ordering.is_lt(),
        BinaryOperator::LtEq => ordering.is_lt() || ordering.is_eq(),
        _ => unreachable!("validated comparison operator"),
    }))
}

pub(super) fn literal_value_from_array(column: &ArrayRef, row: usize) -> Result<LiteralValue> {
    match column.data_type() {
        DataType::Boolean => {
            let values = column
                .as_any()
                .downcast_ref::<BooleanArray>()
                .expect("Boolean data type");
            Ok(LiteralValue::Boolean(values.value(row)))
        }
        DataType::Int32 => {
            let values = column
                .as_any()
                .downcast_ref::<Int32Array>()
                .expect("Int32 data type");
            Ok(LiteralValue::Int64(i64::from(values.value(row))))
        }
        DataType::Int64 => {
            let values = column
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("Int64 data type");
            Ok(LiteralValue::Int64(values.value(row)))
        }
        DataType::Float64 => {
            let values = column
                .as_any()
                .downcast_ref::<Float64Array>()
                .expect("Float64 data type");
            Ok(LiteralValue::Float64(values.value(row)))
        }
        DataType::Decimal128(_, scale) => {
            let values = column
                .as_any()
                .downcast_ref::<Decimal128Array>()
                .expect("Decimal128 data type");
            Ok(LiteralValue::Float64(
                values.value(row) as f64 / 10_f64.powi(i32::from(*scale)),
            ))
        }
        DataType::Date32 => {
            let values = column
                .as_any()
                .downcast_ref::<Date32Array>()
                .expect("Date32 data type");
            let (year, month, day) = civil_from_days(i64::from(values.value(row)))?;
            Ok(LiteralValue::Utf8(format!("{year:04}-{month:02}-{day:02}")))
        }
        DataType::Date64 => {
            let values = column
                .as_any()
                .downcast_ref::<Date64Array>()
                .expect("Date64 data type");
            let (year, month, day) = civil_from_days(values.value(row) / 86_400_000)?;
            Ok(LiteralValue::Utf8(format!("{year:04}-{month:02}-{day:02}")))
        }
        DataType::Utf8 => {
            let values = column
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("Utf8 data type");
            Ok(LiteralValue::Utf8(values.value(row).to_string()))
        }
        data_type => Err(DodamError::UnsupportedSql(format!(
            "IN subquery result type {data_type} is not supported yet"
        ))),
    }
}
