use super::*;

pub(super) fn apply_output_filter(
    batches: Vec<RecordBatch>,
    filter: Option<&FilterExpr>,
) -> Result<Vec<RecordBatch>> {
    let Some(filter) = filter else {
        return Ok(batches);
    };

    let mut filtered = Vec::new();
    for batch in batches {
        let batch = filter_batch(batch, filter)?;
        if batch.num_rows() > 0 {
            filtered.push(batch);
        }
    }
    Ok(filtered)
}

pub(super) fn apply_output_expression_filter(
    batches: Vec<RecordBatch>,
    predicate: &SqlExpr,
    table_alias: Option<&str>,
) -> Result<Vec<RecordBatch>> {
    let mut filtered = Vec::new();
    for batch in batches {
        let mask = evaluate_scalar_predicate(&batch, predicate, table_alias)?;
        let batch = filter_record_batch(&batch, &mask)?;
        if batch.num_rows() > 0 {
            filtered.push(batch);
        }
    }
    Ok(filtered)
}

pub(super) fn apply_output_join_expression_filter(
    batches: Vec<RecordBatch>,
    predicate: &SqlExpr,
    table_aliases: &[&str],
) -> Result<Vec<RecordBatch>> {
    let mut filtered = Vec::new();
    for batch in batches {
        let mask = evaluate_join_scalar_predicate(&batch, predicate, table_aliases)?;
        let batch = filter_record_batch(&batch, &mask)?;
        if batch.num_rows() > 0 {
            filtered.push(batch);
        }
    }
    Ok(filtered)
}

pub(super) fn evaluate_join_scalar_predicate(
    batch: &RecordBatch,
    predicate: &SqlExpr,
    table_aliases: &[&str],
) -> Result<BooleanArray> {
    evaluate_scalar_predicate_with_parser(
        batch,
        predicate,
        ScalarPredicateParser::Join(table_aliases),
    )
}

pub(super) fn evaluate_scalar_predicate(
    batch: &RecordBatch,
    predicate: &SqlExpr,
    table_alias: Option<&str>,
) -> Result<BooleanArray> {
    evaluate_scalar_predicate_with_parser(
        batch,
        predicate,
        ScalarPredicateParser::Single(table_alias),
    )
}

#[derive(Clone, Copy)]
enum ScalarPredicateParser<'a> {
    Single(Option<&'a str>),
    Join(&'a [&'a str]),
}

impl ScalarPredicateParser<'_> {
    fn parse(self, expr: &SqlExpr) -> Result<ScalarSqlExpression> {
        match self {
            ScalarPredicateParser::Single(table_alias) => {
                parse_scalar_sql_expression(expr, table_alias)
            }
            ScalarPredicateParser::Join(table_aliases) => {
                parse_join_scalar_sql_expression(expr, table_aliases)
            }
        }
    }

    fn unsupported_context(self) -> &'static str {
        match self {
            ScalarPredicateParser::Single(_) => "expression",
            ScalarPredicateParser::Join(_) => "JOIN expression",
        }
    }
}

fn evaluate_scalar_predicate_with_parser(
    batch: &RecordBatch,
    predicate: &SqlExpr,
    parser: ScalarPredicateParser<'_>,
) -> Result<BooleanArray> {
    if let Some(mask) = try_evaluate_vector_scalar_predicate(batch, predicate, parser)? {
        return Ok(mask);
    }
    match predicate {
        SqlExpr::BinaryOp { left, op, right } if *op == BinaryOperator::And => {
            let left = evaluate_scalar_predicate_with_parser(batch, left, parser)?;
            let right = evaluate_scalar_predicate_with_parser(batch, right, parser)?;
            Ok(boolean_and(&left, &right))
        }
        SqlExpr::BinaryOp { left, op, right } if *op == BinaryOperator::Or => {
            let left = evaluate_scalar_predicate_with_parser(batch, left, parser)?;
            let right = evaluate_scalar_predicate_with_parser(batch, right, parser)?;
            Ok(boolean_or(&left, &right))
        }
        SqlExpr::UnaryOp { op, expr } if *op == UnaryOperator::Not => {
            let mask = evaluate_scalar_predicate_with_parser(batch, expr, parser)?;
            Ok(boolean_not(&mask))
        }
        SqlExpr::Nested(expr) => evaluate_scalar_predicate_with_parser(batch, expr, parser),
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
            let left = evaluate_scalar_expression(batch, &parser.parse(left)?)?;
            let right = evaluate_scalar_expression(batch, &parser.parse(right)?)?;
            Ok(BooleanArray::from(compare_evaluated_scalars(
                left, op, right,
            )?))
        }
        SqlExpr::IsNull(expr) | SqlExpr::IsNotNull(expr) => {
            let value = evaluate_scalar_expression(batch, &parser.parse(expr)?)?;
            let negated_null_check = matches!(predicate, SqlExpr::IsNotNull(_));
            if let EvaluatedScalar::Array(array) = &value {
                return if negated_null_check {
                    Ok(is_not_null(array.as_ref())?)
                } else {
                    Ok(is_null(array.as_ref())?)
                };
            }
            Ok(BooleanArray::from(
                scalar_null_mask(value)
                    .into_iter()
                    .map(|is_null| {
                        Some(if negated_null_check {
                            !is_null
                        } else {
                            is_null
                        })
                    })
                    .collect::<Vec<_>>(),
            ))
        }
        SqlExpr::InList {
            expr,
            list,
            negated,
        } => {
            let value = evaluate_scalar_expression(batch, &parser.parse(expr)?)?;
            let values = list
                .iter()
                .map(|expr| evaluate_scalar_expression(batch, &parser.parse(expr)?))
                .collect::<Result<Vec<_>>>()?;
            Ok(BooleanArray::from(evaluate_scalar_in_list(
                value, &values, *negated,
            )?))
        }
        SqlExpr::Between {
            expr,
            negated,
            low,
            high,
        } => {
            let lower = evaluate_scalar_predicate_with_parser(
                batch,
                &SqlExpr::BinaryOp {
                    left: expr.clone(),
                    op: if *negated {
                        BinaryOperator::Lt
                    } else {
                        BinaryOperator::GtEq
                    },
                    right: low.clone(),
                },
                parser,
            )?;
            let upper = evaluate_scalar_predicate_with_parser(
                batch,
                &SqlExpr::BinaryOp {
                    left: expr.clone(),
                    op: if *negated {
                        BinaryOperator::Gt
                    } else {
                        BinaryOperator::LtEq
                    },
                    right: high.clone(),
                },
                parser,
            )?;
            if *negated {
                Ok(boolean_or(&lower, &upper))
            } else {
                Ok(boolean_and(&lower, &upper))
            }
        }
        SqlExpr::Like {
            negated,
            any,
            expr,
            pattern,
            escape_char,
        }
        | SqlExpr::ILike {
            negated,
            any,
            expr,
            pattern,
            escape_char,
        } => {
            if *any {
                return Err(DodamError::UnsupportedSql(
                    "LIKE ANY is not supported".to_string(),
                ));
            }
            let value = scalar_as_utf8(evaluate_scalar_expression(batch, &parser.parse(expr)?)?)?;
            let case_insensitive = matches!(predicate, SqlExpr::ILike { .. });
            let mut pattern = sql_like_pattern(pattern)?;
            if case_insensitive {
                pattern = pattern.to_lowercase();
            }
            let escape = sql_like_escape(escape_char)?;
            let tokens = scalar_like_pattern_tokens(&pattern, escape)?;
            Ok(BooleanArray::from(
                value
                    .into_iter()
                    .map(|value| {
                        value.map(|value| {
                            let normalized_value;
                            let value = if case_insensitive {
                                normalized_value = value.to_lowercase();
                                normalized_value.as_str()
                            } else {
                                value.as_str()
                            };
                            let matched = scalar_like_matches(value, &tokens);
                            if *negated { !matched } else { matched }
                        })
                    })
                    .collect::<Vec<_>>(),
            ))
        }
        _ => Err(DodamError::UnsupportedSql(format!(
            "unsupported {} WHERE predicate: {predicate}",
            parser.unsupported_context()
        ))),
    }
}

fn try_evaluate_vector_scalar_predicate(
    batch: &RecordBatch,
    predicate: &SqlExpr,
    parser: ScalarPredicateParser<'_>,
) -> Result<Option<BooleanArray>> {
    match predicate {
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
            let left = parser.parse(left)?;
            let right = parser.parse(right)?;
            if let Some(mask) = vector_list_length_literal_compare(batch, &left, op, &right)? {
                return Ok(Some(mask));
            }
            if let Some(mask) = vector_list_length_literal_compare(
                batch,
                &right,
                &reverse_binary_operator(op),
                &left,
            )? {
                return Ok(Some(mask));
            }
            if let Some(mask) = vector_column_literal_compare(batch, &left, op, &right)? {
                return Ok(Some(mask));
            }
            if let Some(mask) =
                vector_column_literal_compare(batch, &right, &reverse_binary_operator(op), &left)?
            {
                return Ok(Some(mask));
            }
            Ok(None)
        }
        SqlExpr::InList {
            expr,
            list,
            negated,
        } => {
            let expr = parser.parse(expr)?;
            let values = list
                .iter()
                .map(|expr| parser.parse(expr))
                .collect::<Result<Vec<_>>>()?;
            vector_i64_expression_in_list(batch, &expr, &values, *negated)
        }
        SqlExpr::Like {
            negated,
            any,
            expr,
            pattern,
            escape_char,
        }
        | SqlExpr::ILike {
            negated,
            any,
            expr,
            pattern,
            escape_char,
        } => {
            if *any {
                return Ok(None);
            }
            let expr = parser.parse(expr)?;
            let case_insensitive = matches!(predicate, SqlExpr::ILike { .. });
            vector_string_like_literal(
                batch,
                &expr,
                pattern,
                sql_like_escape(escape_char)?,
                *negated,
                case_insensitive,
            )
        }
        _ => Ok(None),
    }
}

pub(super) fn reverse_binary_operator(op: &BinaryOperator) -> BinaryOperator {
    match op {
        BinaryOperator::Gt => BinaryOperator::Lt,
        BinaryOperator::GtEq => BinaryOperator::LtEq,
        BinaryOperator::Lt => BinaryOperator::Gt,
        BinaryOperator::LtEq => BinaryOperator::GtEq,
        _ => op.clone(),
    }
}

fn vector_list_length_literal_compare(
    batch: &RecordBatch,
    left: &ScalarSqlExpression,
    op: &BinaryOperator,
    right: &ScalarSqlExpression,
) -> Result<Option<BooleanArray>> {
    let ScalarSqlExpression::ListLength { column, field } = left else {
        return Ok(None);
    };
    let Some(literal) = scalar_i64_literal(right) else {
        return Ok(None);
    };
    let list = list_array_column(batch, column, field.as_deref())?;
    Ok(Some(BooleanArray::from(
        (0..list.len())
            .map(|row| {
                if list.is_null(row) {
                    None
                } else {
                    compare_optional_values(
                        Some(i64::from(list.value_length(row))),
                        op,
                        Some(literal),
                    )
                }
            })
            .collect::<Vec<_>>(),
    )))
}

fn vector_column_literal_compare(
    batch: &RecordBatch,
    left: &ScalarSqlExpression,
    op: &BinaryOperator,
    right: &ScalarSqlExpression,
) -> Result<Option<BooleanArray>> {
    let ScalarSqlExpression::Column(column) = left else {
        return Ok(None);
    };
    let ScalarSqlExpression::Literal(literal) = right else {
        return Ok(None);
    };
    if matches!(literal, LiteralValue::Null) {
        return Ok(Some(BooleanArray::from(vec![None; batch.num_rows()])));
    }
    let column_index = output_batch_column_index(batch, column)?;
    let array = batch.column(column_index);
    match array.data_type() {
        DataType::Int32 => {
            let values = array.as_any().downcast_ref::<Int32Array>().expect("Int32");
            let literal = literal_as_i64_for_type(literal)?;
            if values.null_count() == 0 {
                return Ok(Some(BooleanArray::from(
                    (0..values.len())
                        .map(|row| {
                            compare_optional_values(Some(i64::from(values.value(row))), op, literal)
                                .unwrap_or(false)
                        })
                        .collect::<Vec<_>>(),
                )));
            }
            Ok(Some(BooleanArray::from(
                (0..values.len())
                    .map(|row| {
                        if values.is_null(row) {
                            None
                        } else {
                            compare_optional_values(Some(i64::from(values.value(row))), op, literal)
                        }
                    })
                    .collect::<Vec<_>>(),
            )))
        }
        DataType::Int64 => {
            let values = array.as_any().downcast_ref::<Int64Array>().expect("Int64");
            let literal = literal_as_i64_for_type(literal)?;
            if values.null_count() == 0 {
                return Ok(Some(BooleanArray::from(
                    (0..values.len())
                        .map(|row| {
                            compare_optional_values(Some(values.value(row)), op, literal)
                                .unwrap_or(false)
                        })
                        .collect::<Vec<_>>(),
                )));
            }
            Ok(Some(BooleanArray::from(
                (0..values.len())
                    .map(|row| {
                        if values.is_null(row) {
                            None
                        } else {
                            compare_optional_values(Some(values.value(row)), op, literal)
                        }
                    })
                    .collect::<Vec<_>>(),
            )))
        }
        DataType::Float64 => {
            let values = array
                .as_any()
                .downcast_ref::<Float64Array>()
                .expect("Float64");
            let literal = literal_as_f64_for_type(literal)?;
            if values.null_count() == 0 {
                return Ok(Some(BooleanArray::from(
                    (0..values.len())
                        .map(|row| {
                            compare_optional_f64(Some(values.value(row)), op, literal)
                                .unwrap_or(false)
                        })
                        .collect::<Vec<_>>(),
                )));
            }
            Ok(Some(BooleanArray::from(
                (0..values.len())
                    .map(|row| {
                        if values.is_null(row) {
                            None
                        } else {
                            compare_optional_f64(Some(values.value(row)), op, literal)
                        }
                    })
                    .collect::<Vec<_>>(),
            )))
        }
        DataType::Boolean => {
            let values = array
                .as_any()
                .downcast_ref::<BooleanArray>()
                .expect("Boolean");
            let literal = literal_as_bool_for_type(literal)?;
            if values.null_count() == 0 {
                return Ok(Some(BooleanArray::from(
                    (0..values.len())
                        .map(|row| {
                            compare_optional_values(Some(values.value(row)), op, literal)
                                .unwrap_or(false)
                        })
                        .collect::<Vec<_>>(),
                )));
            }
            Ok(Some(BooleanArray::from(
                (0..values.len())
                    .map(|row| {
                        if values.is_null(row) {
                            None
                        } else {
                            compare_optional_values(Some(values.value(row)), op, literal)
                        }
                    })
                    .collect::<Vec<_>>(),
            )))
        }
        DataType::Date32 => {
            let values = array
                .as_any()
                .downcast_ref::<Date32Array>()
                .expect("Date32");
            let literal = literal_as_date32_for_type(literal)?;
            if values.null_count() == 0 {
                return Ok(Some(BooleanArray::from(
                    (0..values.len())
                        .map(|row| {
                            compare_optional_values(Some(values.value(row)), op, literal)
                                .unwrap_or(false)
                        })
                        .collect::<Vec<_>>(),
                )));
            }
            Ok(Some(BooleanArray::from(
                (0..values.len())
                    .map(|row| {
                        if values.is_null(row) {
                            None
                        } else {
                            compare_optional_values(Some(values.value(row)), op, literal)
                        }
                    })
                    .collect::<Vec<_>>(),
            )))
        }
        DataType::Decimal128(precision, scale) => {
            let values = array
                .as_any()
                .downcast_ref::<Decimal128Array>()
                .expect("Decimal128");
            let literal = literal_as_decimal128_for_type(literal, *precision, *scale)?;
            if values.null_count() == 0 {
                return Ok(Some(BooleanArray::from(
                    (0..values.len())
                        .map(|row| {
                            compare_optional_values(Some(values.value(row)), op, literal)
                                .unwrap_or(false)
                        })
                        .collect::<Vec<_>>(),
                )));
            }
            Ok(Some(BooleanArray::from(
                (0..values.len())
                    .map(|row| {
                        if values.is_null(row) {
                            None
                        } else {
                            compare_optional_values(Some(values.value(row)), op, literal)
                        }
                    })
                    .collect::<Vec<_>>(),
            )))
        }
        DataType::Utf8 => {
            let values = array.as_any().downcast_ref::<StringArray>().expect("Utf8");
            let literal = literal_as_utf8_for_type(literal)?;
            Ok(Some(BooleanArray::from(
                (0..values.len())
                    .map(|row| {
                        if values.is_null(row) {
                            None
                        } else {
                            compare_optional_values(
                                Some(values.value(row).to_string()),
                                op,
                                literal.clone(),
                            )
                        }
                    })
                    .collect::<Vec<_>>(),
            )))
        }
        _ => Ok(None),
    }
}

fn vector_i64_expression_in_list(
    batch: &RecordBatch,
    expr: &ScalarSqlExpression,
    values: &[ScalarSqlExpression],
    negated: bool,
) -> Result<Option<BooleanArray>> {
    let Some(probe) = vector_i64_expression(batch, expr)? else {
        return Ok(None);
    };
    let mut literals = Vec::new();
    let mut has_null = false;
    for value in values {
        match scalar_i64_literal(value) {
            Some(value) => literals.push(value),
            None if matches!(value, ScalarSqlExpression::Literal(LiteralValue::Null)) => {
                has_null = true
            }
            None => return Ok(None),
        }
    }
    Ok(Some(BooleanArray::from(
        probe
            .into_iter()
            .map(|value| {
                let Some(value) = value else {
                    return None;
                };
                let matched = literals.contains(&value);
                if matched {
                    Some(!negated)
                } else if has_null {
                    None
                } else {
                    Some(negated)
                }
            })
            .collect::<Vec<_>>(),
    )))
}

fn vector_i64_expression(
    batch: &RecordBatch,
    expr: &ScalarSqlExpression,
) -> Result<Option<Vec<Option<i64>>>> {
    match expr {
        ScalarSqlExpression::Column(column) => vector_i64_array(batch_column(batch, column)?),
        ScalarSqlExpression::StructField { column, field } => {
            vector_i64_array(struct_field_array(batch, column, field)?)
        }
        _ => Ok(None),
    }
}

#[derive(Clone, Copy)]
enum SimpleLikeLiteral<'a> {
    Exact(&'a str),
    Prefix(&'a str),
    Suffix(&'a str),
    Contains(&'a str),
}

fn vector_string_like_literal(
    batch: &RecordBatch,
    expr: &ScalarSqlExpression,
    pattern: &SqlExpr,
    escape: Option<char>,
    negated: bool,
    case_insensitive: bool,
) -> Result<Option<BooleanArray>> {
    let ScalarSqlExpression::Column(column) = expr else {
        return Ok(None);
    };
    let pattern = sql_like_pattern(pattern)?;
    let Some(pattern) = simple_like_literal(&pattern, escape) else {
        return Ok(None);
    };
    let values = batch_column(batch, column)?;
    let Some(values) = values.as_any().downcast_ref::<StringArray>() else {
        return Ok(None);
    };
    Ok(Some(BooleanArray::from(
        (0..values.len())
            .map(|row| {
                if values.is_null(row) {
                    return None;
                }
                let matched = simple_like_matches(values.value(row), pattern, case_insensitive);
                Some(if negated { !matched } else { matched })
            })
            .collect::<Vec<_>>(),
    )))
}

fn simple_like_literal(pattern: &str, escape: Option<char>) -> Option<SimpleLikeLiteral<'_>> {
    if escape.is_some() || pattern.as_bytes().contains(&b'_') {
        return None;
    }
    let percent_count = pattern
        .as_bytes()
        .iter()
        .filter(|byte| **byte == b'%')
        .count();
    match percent_count {
        0 => Some(SimpleLikeLiteral::Exact(pattern)),
        1 if pattern.ends_with('%') => {
            Some(SimpleLikeLiteral::Prefix(&pattern[..pattern.len() - 1]))
        }
        1 if pattern.starts_with('%') => Some(SimpleLikeLiteral::Suffix(&pattern[1..])),
        2 if pattern.starts_with('%') && pattern.ends_with('%') => {
            Some(SimpleLikeLiteral::Contains(&pattern[1..pattern.len() - 1]))
        }
        _ => None,
    }
}

fn simple_like_matches(
    value: &str,
    pattern: SimpleLikeLiteral<'_>,
    case_insensitive: bool,
) -> bool {
    if case_insensitive {
        let value = value.to_lowercase();
        return match pattern {
            SimpleLikeLiteral::Exact(pattern) => value == pattern.to_lowercase(),
            SimpleLikeLiteral::Prefix(pattern) => value.starts_with(&pattern.to_lowercase()),
            SimpleLikeLiteral::Suffix(pattern) => value.ends_with(&pattern.to_lowercase()),
            SimpleLikeLiteral::Contains(pattern) => value.contains(&pattern.to_lowercase()),
        };
    }
    match pattern {
        SimpleLikeLiteral::Exact(pattern) => value == pattern,
        SimpleLikeLiteral::Prefix(pattern) => value.starts_with(pattern),
        SimpleLikeLiteral::Suffix(pattern) => value.ends_with(pattern),
        SimpleLikeLiteral::Contains(pattern) => value.contains(pattern),
    }
}

fn vector_i64_array(array: &dyn Array) -> Result<Option<Vec<Option<i64>>>> {
    match array.data_type() {
        DataType::Int32 => {
            let values = array.as_any().downcast_ref::<Int32Array>().expect("Int32");
            if values.null_count() == 0 {
                Ok(Some(
                    values
                        .values()
                        .iter()
                        .map(|value| Some(i64::from(*value)))
                        .collect(),
                ))
            } else {
                Ok(Some(
                    values.iter().map(|value| value.map(i64::from)).collect(),
                ))
            }
        }
        DataType::Int64 => {
            let values = array.as_any().downcast_ref::<Int64Array>().expect("Int64");
            if values.null_count() == 0 {
                Ok(Some(
                    values.values().iter().map(|value| Some(*value)).collect(),
                ))
            } else {
                Ok(Some(values.iter().collect()))
            }
        }
        _ => Ok(None),
    }
}

fn scalar_i64_literal(expr: &ScalarSqlExpression) -> Option<i64> {
    match expr {
        ScalarSqlExpression::Literal(LiteralValue::Int64(value)) => Some(*value),
        _ => None,
    }
}

pub(super) fn boolean_and(left: &BooleanArray, right: &BooleanArray) -> BooleanArray {
    BooleanArray::from(
        (0..left.len())
            .map(
                |row| match (boolean_value(left, row), boolean_value(right, row)) {
                    (Some(false), _) | (_, Some(false)) => Some(false),
                    (Some(true), Some(true)) => Some(true),
                    _ => None,
                },
            )
            .collect::<Vec<_>>(),
    )
}

pub(super) fn boolean_or(left: &BooleanArray, right: &BooleanArray) -> BooleanArray {
    BooleanArray::from(
        (0..left.len())
            .map(
                |row| match (boolean_value(left, row), boolean_value(right, row)) {
                    (Some(true), _) | (_, Some(true)) => Some(true),
                    (Some(false), Some(false)) => Some(false),
                    _ => None,
                },
            )
            .collect::<Vec<_>>(),
    )
}

pub(super) fn boolean_not(mask: &BooleanArray) -> BooleanArray {
    BooleanArray::from(
        (0..mask.len())
            .map(|row| boolean_value(mask, row).map(|value| !value))
            .collect::<Vec<_>>(),
    )
}

fn boolean_value(mask: &BooleanArray, row: usize) -> Option<bool> {
    if mask.is_null(row) {
        None
    } else {
        Some(mask.value(row))
    }
}

pub(super) fn apply_output_expression_projection(
    batches: Vec<RecordBatch>,
    expressions: &[ProjectionExpression],
) -> Result<Vec<RecordBatch>> {
    if expressions.is_empty() {
        return Ok(batches);
    }
    batches
        .into_iter()
        .map(|batch| {
            let mut fields = Vec::with_capacity(expressions.len());
            let mut columns = Vec::with_capacity(expressions.len());
            for expression in expressions {
                let value = evaluate_scalar_expression(&batch, &expression.expr)?;
                fields.push(Field::new(
                    expression.output_name.clone(),
                    value.data_type(),
                    value.is_nullable(),
                ));
                columns.push(value.into_array(batch.num_rows()));
            }
            Ok(RecordBatch::try_new(
                Arc::new(Schema::new(fields)),
                columns,
            )?)
        })
        .collect()
}

#[derive(Clone)]
pub(super) enum EvaluatedScalar {
    Array(ArrayRef),
    Int64(Vec<Option<i64>>),
    Float64(Vec<Option<f64>>),
    Decimal128 {
        values: Vec<Option<i128>>,
        precision: u8,
        scale: i8,
    },
    Utf8(Vec<Option<String>>),
    Boolean(Vec<Option<bool>>),
    Date32(Vec<Option<i32>>),
    TimestampMillisecond(Vec<Option<i64>>),
}

impl EvaluatedScalar {
    pub(super) fn len(&self) -> usize {
        match self {
            Self::Array(array) => array.len(),
            Self::Int64(values) => values.len(),
            Self::Float64(values) => values.len(),
            Self::Decimal128 { values, .. } => values.len(),
            Self::Utf8(values) => values.len(),
            Self::Boolean(values) => values.len(),
            Self::Date32(values) => values.len(),
            Self::TimestampMillisecond(values) => values.len(),
        }
    }

    pub(super) fn data_type(&self) -> DataType {
        match self {
            Self::Array(array) => array.data_type().clone(),
            Self::Int64(_) => DataType::Int64,
            Self::Float64(_) => DataType::Float64,
            Self::Decimal128 {
                precision, scale, ..
            } => DataType::Decimal128(*precision, *scale),
            Self::Utf8(_) => DataType::Utf8,
            Self::Boolean(_) => DataType::Boolean,
            Self::Date32(_) => DataType::Date32,
            Self::TimestampMillisecond(_) => DataType::Timestamp(TimeUnit::Millisecond, None),
        }
    }

    pub(super) fn is_nullable(&self) -> bool {
        match self {
            Self::Array(array) => array.null_count() > 0,
            Self::Int64(values) => values.iter().any(Option::is_none),
            Self::Float64(values) => values.iter().any(Option::is_none),
            Self::Decimal128 { values, .. } => values.iter().any(Option::is_none),
            Self::Utf8(values) => values.iter().any(Option::is_none),
            Self::Boolean(values) => values.iter().any(Option::is_none),
            Self::Date32(values) => values.iter().any(Option::is_none),
            Self::TimestampMillisecond(values) => values.iter().any(Option::is_none),
        }
    }

    pub(super) fn into_array(self, _rows: usize) -> ArrayRef {
        match self {
            Self::Array(array) => array,
            Self::Int64(values) => Arc::new(Int64Array::from(values)) as ArrayRef,
            Self::Float64(values) => Arc::new(Float64Array::from(values)) as ArrayRef,
            Self::Decimal128 {
                values,
                precision,
                scale,
            } => Arc::new(
                Decimal128Array::from(values)
                    .with_precision_and_scale(precision, scale)
                    .expect("valid Decimal128 scalar expression"),
            ) as ArrayRef,
            Self::Utf8(values) => Arc::new(StringArray::from(values)) as ArrayRef,
            Self::Boolean(values) => Arc::new(BooleanArray::from(values)) as ArrayRef,
            Self::Date32(values) => Arc::new(Date32Array::from(values)) as ArrayRef,
            Self::TimestampMillisecond(values) => {
                Arc::new(TimestampMillisecondArray::from(values)) as ArrayRef
            }
        }
    }
}

pub(super) fn evaluate_scalar_expression(
    batch: &RecordBatch,
    expr: &ScalarSqlExpression,
) -> Result<EvaluatedScalar> {
    match expr {
        ScalarSqlExpression::Column(column) => evaluated_column(batch, column),
        ScalarSqlExpression::StructField { column, field } => {
            evaluated_struct_field(batch, column, field)
        }
        ScalarSqlExpression::ListIndex {
            column,
            field,
            index,
        } => {
            let index = scalar_as_i64(evaluate_scalar_expression(batch, index)?)?;
            evaluated_list_index(batch, column, field.as_deref(), &index)
        }
        ScalarSqlExpression::ListLength { column, field } => {
            evaluated_list_length(batch, column, field.as_deref())
        }
        ScalarSqlExpression::Literal(value) => Ok(evaluated_literal(value, batch.num_rows())),
        ScalarSqlExpression::Binary { left, op, right } => {
            if decimal_expression_fast_enabled()
                && let Some(values) = evaluate_decimal_product_expression(batch, left, op, right)?
            {
                return Ok(EvaluatedScalar::Float64(values));
            }
            let left = evaluate_scalar_expression(batch, left)?;
            let right = evaluate_scalar_expression(batch, right)?;
            evaluate_binary_scalar(left, op, right)
        }
        ScalarSqlExpression::Cast { expr, target } => {
            let value = evaluate_scalar_expression(batch, expr)?;
            cast_evaluated_scalar(value, target)
        }
        ScalarSqlExpression::Coalesce(values) => {
            if let Some(value) = evaluate_column_literal_coalesce(batch, values)? {
                return Ok(value);
            }
            let mut evaluated = values
                .iter()
                .map(|expr| evaluate_scalar_expression(batch, expr))
                .collect::<Result<Vec<_>>>()?;
            let Some(first) = evaluated.first().cloned() else {
                return Err(DodamError::UnsupportedSql(
                    "COALESCE requires at least one argument".to_string(),
                ));
            };
            let mut result = first;
            for value in evaluated.drain(1..) {
                result = coalesce_evaluated_scalar(result, value)?;
            }
            Ok(result)
        }
        ScalarSqlExpression::Lower(expr) => {
            let value = scalar_as_utf8(evaluate_scalar_expression(batch, expr)?)?;
            Ok(EvaluatedScalar::Utf8(
                value
                    .into_iter()
                    .map(|value| value.map(|value| value.to_lowercase()))
                    .collect(),
            ))
        }
        ScalarSqlExpression::Upper(expr) => {
            let value = scalar_as_utf8(evaluate_scalar_expression(batch, expr)?)?;
            Ok(EvaluatedScalar::Utf8(
                value
                    .into_iter()
                    .map(|value| value.map(|value| value.to_uppercase()))
                    .collect(),
            ))
        }
        ScalarSqlExpression::Length(expr) => {
            let value = scalar_as_utf8(evaluate_scalar_expression(batch, expr)?)?;
            Ok(EvaluatedScalar::Int64(
                value
                    .into_iter()
                    .map(|value| value.map(|value| value.chars().count() as i64))
                    .collect(),
            ))
        }
        ScalarSqlExpression::Trim(expr) => {
            let value = scalar_as_utf8(evaluate_scalar_expression(batch, expr)?)?;
            Ok(EvaluatedScalar::Utf8(
                value
                    .into_iter()
                    .map(|value| value.map(|value| value.trim().to_string()))
                    .collect(),
            ))
        }
        ScalarSqlExpression::Abs(expr) => {
            let value = materialize_array_scalar(evaluate_scalar_expression(batch, expr)?)?;
            evaluate_abs_scalar(value)
        }
        ScalarSqlExpression::Round(expr) => evaluate_f64_unary_scalar(batch, expr, f64::round),
        ScalarSqlExpression::Floor(expr) => evaluate_f64_unary_scalar(batch, expr, f64::floor),
        ScalarSqlExpression::Ceil(expr) => evaluate_f64_unary_scalar(batch, expr, f64::ceil),
        ScalarSqlExpression::Replace { expr, from, to } => {
            let values = scalar_as_utf8(evaluate_scalar_expression(batch, expr)?)?;
            let from_values = scalar_as_utf8(evaluate_scalar_expression(batch, from)?)?;
            let to_values = scalar_as_utf8(evaluate_scalar_expression(batch, to)?)?;
            Ok(EvaluatedScalar::Utf8(
                (0..batch.num_rows())
                    .map(
                        |row| match (&values[row], &from_values[row], &to_values[row]) {
                            (Some(value), Some(from), Some(to)) => Some(value.replace(from, to)),
                            _ => None,
                        },
                    )
                    .collect(),
            ))
        }
        ScalarSqlExpression::Concat(values) => {
            let values = values
                .iter()
                .map(|expr| scalar_as_utf8(evaluate_scalar_expression(batch, expr)?))
                .collect::<Result<Vec<_>>>()?;
            Ok(EvaluatedScalar::Utf8(
                (0..batch.num_rows())
                    .map(|row| {
                        let mut output = String::new();
                        for values in &values {
                            if let Some(value) = &values[row] {
                                output.push_str(value);
                            }
                        }
                        Some(output)
                    })
                    .collect(),
            ))
        }
        ScalarSqlExpression::ExtractYear(expr) => {
            let value = scalar_as_i64(evaluate_scalar_expression(batch, expr)?)?;
            Ok(EvaluatedScalar::Int64(
                value
                    .into_iter()
                    .map(|value| {
                        value
                            .map(|days| civil_from_days(days).map(|(year, _, _)| i64::from(year)))
                            .transpose()
                    })
                    .collect::<Result<Vec<_>>>()?,
            ))
        }
        ScalarSqlExpression::Substring {
            expr,
            start,
            length,
        } => {
            let values = scalar_as_utf8(evaluate_scalar_expression(batch, expr)?)?;
            let starts = scalar_as_i64(evaluate_scalar_expression(batch, start)?)?;
            let lengths = length
                .as_ref()
                .map(|expr| scalar_as_i64(evaluate_scalar_expression(batch, expr)?))
                .transpose()?;
            Ok(EvaluatedScalar::Utf8(
                (0..batch.num_rows())
                    .map(|row| {
                        substring_value(
                            values[row].as_deref(),
                            starts[row],
                            lengths.as_ref().map(|values| values[row]),
                        )
                    })
                    .collect(),
            ))
        }
        ScalarSqlExpression::Case {
            conditions,
            results,
            else_result,
        } => evaluate_case_expression(batch, conditions, results, else_result.as_deref()),
    }
}

fn evaluate_abs_scalar(value: EvaluatedScalar) -> Result<EvaluatedScalar> {
    match value {
        EvaluatedScalar::Array(_) => unreachable!("array scalar was materialized before abs"),
        EvaluatedScalar::Int64(values) => Ok(EvaluatedScalar::Int64(
            values
                .into_iter()
                .map(|value| {
                    value
                        .map(|value| {
                            value.checked_abs().ok_or_else(|| {
                                DodamError::UnsupportedSql("ABS integer overflow".to_string())
                            })
                        })
                        .transpose()
                })
                .collect::<Result<Vec<_>>>()?,
        )),
        EvaluatedScalar::Float64(values) => Ok(EvaluatedScalar::Float64(
            values
                .into_iter()
                .map(|value| value.map(f64::abs))
                .collect(),
        )),
        EvaluatedScalar::Decimal128 { values, scale, .. } => {
            let scale = decimal_scale_f64(scale)?;
            Ok(EvaluatedScalar::Float64(
                values
                    .into_iter()
                    .map(|value| value.map(|value| (value as f64 / scale).abs()))
                    .collect(),
            ))
        }
        other => Err(DodamError::TypeMismatch(format!(
            "cannot use {} in ABS",
            other.data_type()
        ))),
    }
}

fn evaluate_f64_unary_scalar(
    batch: &RecordBatch,
    expr: &ScalarSqlExpression,
    op: fn(f64) -> f64,
) -> Result<EvaluatedScalar> {
    let values = scalar_as_f64(evaluate_scalar_expression(batch, expr)?)?;
    Ok(EvaluatedScalar::Float64(
        values.into_iter().map(|value| value.map(op)).collect(),
    ))
}

fn evaluate_column_literal_coalesce(
    batch: &RecordBatch,
    values: &[ScalarSqlExpression],
) -> Result<Option<EvaluatedScalar>> {
    let [left, right] = values else {
        return Ok(None);
    };
    if let Some(value) = column_literal_coalesce(batch, left, right, false)? {
        return Ok(Some(value));
    }
    column_literal_coalesce(batch, right, left, true)
}

fn column_literal_coalesce(
    batch: &RecordBatch,
    column_expr: &ScalarSqlExpression,
    literal_expr: &ScalarSqlExpression,
    literal_first: bool,
) -> Result<Option<EvaluatedScalar>> {
    let ScalarSqlExpression::Column(column) = column_expr else {
        return Ok(None);
    };
    let ScalarSqlExpression::Literal(literal) = literal_expr else {
        return Ok(None);
    };
    let column_index = output_batch_column_index(batch, column)?;
    let array = batch.column(column_index);
    if literal_first {
        if !matches!(literal, LiteralValue::Null) {
            return Ok(Some(evaluated_literal(literal, batch.num_rows())));
        }
        return Ok(Some(EvaluatedScalar::Array(array.clone())));
    }
    if array.null_count() == 0 || matches!(literal, LiteralValue::Null) {
        return Ok(Some(EvaluatedScalar::Array(array.clone())));
    }
    coalesce_array_with_literal(array.as_ref(), literal)
}

fn coalesce_array_with_literal(
    array: &dyn Array,
    literal: &LiteralValue,
) -> Result<Option<EvaluatedScalar>> {
    match array.data_type() {
        DataType::Int32 => {
            let values = array.as_any().downcast_ref::<Int32Array>().expect("Int32");
            let literal = literal_as_i64_for_type(literal)?;
            if let Some(literal) = literal {
                let values = (0..values.len())
                    .map(|row| {
                        if values.is_valid(row) {
                            i64::from(values.value(row))
                        } else {
                            literal
                        }
                    })
                    .collect::<Vec<_>>();
                return Ok(Some(EvaluatedScalar::Array(Arc::new(Int64Array::from(
                    values,
                )))));
            }
            Ok(Some(EvaluatedScalar::Int64(
                (0..values.len())
                    .map(|row| {
                        if values.is_valid(row) {
                            Some(i64::from(values.value(row)))
                        } else {
                            literal
                        }
                    })
                    .collect(),
            )))
        }
        DataType::Int64 => {
            let values = array.as_any().downcast_ref::<Int64Array>().expect("Int64");
            let literal = literal_as_i64_for_type(literal)?;
            if let Some(literal) = literal {
                let values = (0..values.len())
                    .map(|row| {
                        if values.is_valid(row) {
                            values.value(row)
                        } else {
                            literal
                        }
                    })
                    .collect::<Vec<_>>();
                return Ok(Some(EvaluatedScalar::Array(Arc::new(Int64Array::from(
                    values,
                )))));
            }
            Ok(Some(EvaluatedScalar::Int64(
                (0..values.len())
                    .map(|row| {
                        if values.is_valid(row) {
                            Some(values.value(row))
                        } else {
                            literal
                        }
                    })
                    .collect(),
            )))
        }
        DataType::Float64 => {
            let values = array
                .as_any()
                .downcast_ref::<Float64Array>()
                .expect("Float64");
            let literal = literal_as_f64_for_type(literal)?;
            if let Some(literal) = literal {
                let values = (0..values.len())
                    .map(|row| {
                        if values.is_valid(row) {
                            values.value(row)
                        } else {
                            literal
                        }
                    })
                    .collect::<Vec<_>>();
                return Ok(Some(EvaluatedScalar::Array(Arc::new(Float64Array::from(
                    values,
                )))));
            }
            Ok(Some(EvaluatedScalar::Float64(
                (0..values.len())
                    .map(|row| {
                        if values.is_valid(row) {
                            Some(values.value(row))
                        } else {
                            literal
                        }
                    })
                    .collect(),
            )))
        }
        DataType::Boolean => {
            let values = array
                .as_any()
                .downcast_ref::<BooleanArray>()
                .expect("Boolean");
            let literal = literal_as_bool_for_type(literal)?;
            if let Some(literal) = literal {
                let values = (0..values.len())
                    .map(|row| {
                        if values.is_valid(row) {
                            values.value(row)
                        } else {
                            literal
                        }
                    })
                    .collect::<Vec<_>>();
                return Ok(Some(EvaluatedScalar::Array(Arc::new(BooleanArray::from(
                    values,
                )))));
            }
            Ok(Some(EvaluatedScalar::Boolean(
                (0..values.len())
                    .map(|row| {
                        if values.is_valid(row) {
                            Some(values.value(row))
                        } else {
                            literal
                        }
                    })
                    .collect(),
            )))
        }
        DataType::Date32 => {
            let values = array
                .as_any()
                .downcast_ref::<Date32Array>()
                .expect("Date32");
            let literal = literal_as_date32_for_type(literal)?;
            if let Some(literal) = literal {
                let values = (0..values.len())
                    .map(|row| {
                        if values.is_valid(row) {
                            values.value(row)
                        } else {
                            literal
                        }
                    })
                    .collect::<Vec<_>>();
                return Ok(Some(EvaluatedScalar::Array(Arc::new(Date32Array::from(
                    values,
                )))));
            }
            Ok(Some(EvaluatedScalar::Date32(
                (0..values.len())
                    .map(|row| {
                        if values.is_valid(row) {
                            Some(values.value(row))
                        } else {
                            literal
                        }
                    })
                    .collect(),
            )))
        }
        DataType::Decimal128(precision, scale) => {
            let values = array
                .as_any()
                .downcast_ref::<Decimal128Array>()
                .expect("Decimal128");
            let literal = literal_as_decimal128_for_type(literal, *precision, *scale)?;
            if let Some(literal) = literal {
                let values = (0..values.len())
                    .map(|row| {
                        if values.is_valid(row) {
                            values.value(row)
                        } else {
                            literal
                        }
                    })
                    .collect::<Vec<_>>();
                let array = Decimal128Array::from(values)
                    .with_precision_and_scale(*precision, *scale)
                    .expect("valid Decimal128 coalesce result");
                return Ok(Some(EvaluatedScalar::Array(Arc::new(array))));
            }
            Ok(Some(EvaluatedScalar::Decimal128 {
                values: (0..values.len())
                    .map(|row| {
                        if values.is_valid(row) {
                            Some(values.value(row))
                        } else {
                            literal
                        }
                    })
                    .collect(),
                precision: *precision,
                scale: *scale,
            }))
        }
        DataType::Utf8 => {
            let values = array.as_any().downcast_ref::<StringArray>().expect("Utf8");
            let literal = literal_as_utf8_for_type(literal)?;
            if let Some(literal) = literal {
                let values = (0..values.len())
                    .map(|row| {
                        if values.is_valid(row) {
                            values.value(row).to_string()
                        } else {
                            literal.clone()
                        }
                    })
                    .collect::<Vec<_>>();
                return Ok(Some(EvaluatedScalar::Array(Arc::new(StringArray::from(
                    values,
                )))));
            }
            Ok(Some(EvaluatedScalar::Utf8(
                (0..values.len())
                    .map(|row| {
                        if values.is_valid(row) {
                            Some(values.value(row).to_string())
                        } else {
                            literal.clone()
                        }
                    })
                    .collect(),
            )))
        }
        _ => Ok(None),
    }
}

pub(super) fn literal_as_i64_for_type(literal: &LiteralValue) -> Result<Option<i64>> {
    match literal {
        LiteralValue::Null => Ok(None),
        LiteralValue::Int64(value) => Ok(Some(*value)),
        LiteralValue::Float64(value) => Ok(Some(*value as i64)),
        LiteralValue::Utf8(value) => value
            .parse::<i64>()
            .map(Some)
            .map_err(|_| DodamError::InvalidCast(format!("cannot cast '{value}' to integer"))),
        LiteralValue::Boolean(value) => Ok(Some(i64::from(*value))),
    }
}

fn literal_as_f64_for_type(literal: &LiteralValue) -> Result<Option<f64>> {
    match literal {
        LiteralValue::Null => Ok(None),
        LiteralValue::Int64(value) => Ok(Some(*value as f64)),
        LiteralValue::Float64(value) => Ok(Some(*value)),
        LiteralValue::Utf8(value) => value
            .parse::<f64>()
            .map(Some)
            .map_err(|_| DodamError::InvalidCast(format!("cannot cast '{value}' to double"))),
        LiteralValue::Boolean(value) => Ok(Some(if *value { 1.0 } else { 0.0 })),
    }
}

fn literal_as_bool_for_type(literal: &LiteralValue) -> Result<Option<bool>> {
    match literal {
        LiteralValue::Null => Ok(None),
        LiteralValue::Boolean(value) => Ok(Some(*value)),
        other => Err(DodamError::InvalidCast(format!(
            "cannot cast {} to boolean",
            literal_type_name(other)
        ))),
    }
}

pub(super) fn literal_as_date32_for_type(literal: &LiteralValue) -> Result<Option<i32>> {
    match literal {
        LiteralValue::Null => Ok(None),
        LiteralValue::Utf8(value) => parse_date32_days(value).map(Some),
        LiteralValue::Int64(value) => i32::try_from(*value)
            .map(Some)
            .map_err(|_| DodamError::InvalidCast(format!("cannot cast {value} to DATE"))),
        other => Err(DodamError::InvalidCast(format!(
            "cannot cast {} to DATE",
            literal_type_name(other)
        ))),
    }
}

pub(super) fn literal_as_decimal128_for_type(
    literal: &LiteralValue,
    precision: u8,
    scale: i8,
) -> Result<Option<i128>> {
    match literal {
        LiteralValue::Null => Ok(None),
        LiteralValue::Int64(value) => {
            let factor = decimal_scale_i128(scale).ok_or_else(|| {
                DodamError::InvalidCast(format!("decimal scale {scale} overflows"))
            })?;
            i128::from(*value)
                .checked_mul(factor)
                .ok_or_else(|| DodamError::InvalidCast("decimal cast overflow".to_string()))
                .and_then(|value| validate_decimal_precision(value, precision))
                .map(Some)
        }
        LiteralValue::Float64(value) => {
            parse_decimal_literal_to_scaled(&value.to_string(), scale, precision).map(Some)
        }
        LiteralValue::Utf8(value) => {
            parse_decimal_literal_to_scaled(value, scale, precision).map(Some)
        }
        other => Err(DodamError::InvalidCast(format!(
            "cannot cast {} to DECIMAL({precision},{scale})",
            literal_type_name(other)
        ))),
    }
}

fn literal_as_utf8_for_type(literal: &LiteralValue) -> Result<Option<String>> {
    Ok(match literal {
        LiteralValue::Null => None,
        LiteralValue::Utf8(value) => Some(value.clone()),
        LiteralValue::Int64(value) => Some(value.to_string()),
        LiteralValue::Float64(value) => Some(format_f64_for_sql_varchar(*value)),
        LiteralValue::Boolean(value) => Some(value.to_string()),
    })
}

fn literal_type_name(literal: &LiteralValue) -> &'static str {
    match literal {
        LiteralValue::Null => "NULL",
        LiteralValue::Boolean(_) => "BOOLEAN",
        LiteralValue::Int64(_) => "INTEGER",
        LiteralValue::Float64(_) => "DOUBLE",
        LiteralValue::Utf8(_) => "VARCHAR",
    }
}

struct DecimalScalarColumn<'a> {
    values: &'a Decimal128Array,
    scale: f64,
}

fn decimal_expression_fast_enabled() -> bool {
    std::env::var("DODAM_DISABLE_DECIMAL_EXPR_FAST")
        .map(|value| !matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(true)
}

fn evaluate_decimal_product_expression(
    batch: &RecordBatch,
    left: &ScalarSqlExpression,
    op: &BinaryOperator,
    right: &ScalarSqlExpression,
) -> Result<Option<Vec<Option<f64>>>> {
    if *op != BinaryOperator::Multiply {
        return Ok(None);
    }
    if let Some((value, complement)) = decimal_discount_product_operands(batch, left, right)? {
        return Ok(Some(decimal_complement_product(value, complement)));
    }
    if let Some((value, complement)) = decimal_discount_product_operands(batch, right, left)? {
        return Ok(Some(decimal_complement_product(value, complement)));
    }
    Ok(None)
}

fn decimal_discount_product_operands<'a>(
    batch: &'a RecordBatch,
    value: &ScalarSqlExpression,
    complement: &ScalarSqlExpression,
) -> Result<Option<(DecimalScalarColumn<'a>, DecimalScalarColumn<'a>)>> {
    let Some(value) = decimal_scalar_column(batch, value)? else {
        return Ok(None);
    };
    let Some(complement) = decimal_one_minus_column(batch, complement)? else {
        return Ok(None);
    };
    Ok(Some((value, complement)))
}

fn decimal_one_minus_column<'a>(
    batch: &'a RecordBatch,
    expr: &ScalarSqlExpression,
) -> Result<Option<DecimalScalarColumn<'a>>> {
    let ScalarSqlExpression::Binary { left, op, right } = expr else {
        return Ok(None);
    };
    if *op != BinaryOperator::Minus || !scalar_literal_is_one(left) {
        return Ok(None);
    }
    decimal_scalar_column(batch, right)
}

fn decimal_scalar_column<'a>(
    batch: &'a RecordBatch,
    expr: &ScalarSqlExpression,
) -> Result<Option<DecimalScalarColumn<'a>>> {
    let ScalarSqlExpression::Column(column) = expr else {
        return Ok(None);
    };
    let index = batch
        .schema()
        .fields()
        .iter()
        .position(|field| field.name() == column)
        .ok_or_else(|| DodamError::UnknownColumn(column.to_string()))?;
    let array = batch.column(index);
    let DataType::Decimal128(precision, scale) = array.data_type() else {
        return Ok(None);
    };
    if *precision > 18 {
        return Ok(None);
    }
    let Some(scale_raw) = decimal_scale_i128(*scale) else {
        return Ok(None);
    };
    Ok(Some(DecimalScalarColumn {
        values: array
            .as_any()
            .downcast_ref::<Decimal128Array>()
            .expect("Decimal128 scalar input"),
        scale: scale_raw as f64,
    }))
}

fn scalar_literal_is_one(expr: &ScalarSqlExpression) -> bool {
    match expr {
        ScalarSqlExpression::Literal(LiteralValue::Int64(1)) => true,
        ScalarSqlExpression::Literal(LiteralValue::Float64(value)) => *value == 1.0,
        _ => false,
    }
}

pub(super) fn decimal_scale_i128(scale: i8) -> Option<i128> {
    let scale = u32::try_from(scale).ok()?;
    Some(10_i128.checked_pow(scale)?)
}

fn decimal_scale_f64(scale: i8) -> Result<f64> {
    let scale = decimal_scale_i128(scale)
        .ok_or_else(|| DodamError::UnsupportedSql(format!("decimal scale {scale} overflows")))?;
    Ok(scale as f64)
}

fn decimal_align_factor(from_scale: i8, to_scale: i8) -> Result<i128> {
    if to_scale < from_scale {
        return Err(DodamError::UnsupportedSql(format!(
            "cannot align decimal scale {from_scale} to lower scale {to_scale}"
        )));
    }
    decimal_scale_i128(to_scale - from_scale)
        .ok_or_else(|| DodamError::UnsupportedSql("decimal scale alignment overflow".to_string()))
}

fn align_decimal_value(value: i128, from_scale: i8, to_scale: i8) -> Result<i128> {
    value
        .checked_mul(decimal_align_factor(from_scale, to_scale)?)
        .ok_or_else(|| DodamError::UnsupportedSql("decimal scale alignment overflow".to_string()))
}

fn decimal_complement_product(
    value: DecimalScalarColumn<'_>,
    complement: DecimalScalarColumn<'_>,
) -> Vec<Option<f64>> {
    let value_raw = value.values.values();
    let complement_raw = complement.values.values();
    if value.values.null_count() == 0 && complement.values.null_count() == 0 {
        return value_raw
            .iter()
            .copied()
            .zip(complement_raw.iter().copied())
            .map(|(value_raw, complement_value)| {
                Some(
                    (value_raw as f64 / value.scale)
                        * (1.0 - complement_value as f64 / complement.scale),
                )
            })
            .collect();
    }
    (0..value.values.len())
        .map(|row| {
            if value.values.is_null(row) || complement.values.is_null(row) {
                None
            } else {
                Some(
                    (value_raw[row] as f64 / value.scale)
                        * (1.0 - complement_raw[row] as f64 / complement.scale),
                )
            }
        })
        .collect()
}

fn evaluate_case_expression(
    batch: &RecordBatch,
    conditions: &[SqlExpr],
    results: &[ScalarSqlExpression],
    else_result: Option<&ScalarSqlExpression>,
) -> Result<EvaluatedScalar> {
    if conditions.len() != results.len() {
        return Err(DodamError::UnsupportedSql(
            "CASE conditions and results length mismatch".to_string(),
        ));
    }
    if conditions.len() == 1 && else_result.is_none() {
        let mask = evaluate_scalar_predicate(batch, &conditions[0], None)?;
        let value = evaluate_scalar_expression(batch, &results[0])?;
        return mask_evaluated_scalar(value, &mask);
    }
    let evaluated_results = results
        .iter()
        .map(|expr| evaluate_scalar_expression(batch, expr))
        .collect::<Result<Vec<_>>>()?;
    let evaluated_else = else_result
        .map(|expr| evaluate_scalar_expression(batch, expr))
        .transpose()?;
    let result_kind = evaluated_results
        .iter()
        .chain(evaluated_else.iter())
        .find_map(evaluated_scalar_kind)
        .unwrap_or(EvaluatedScalarKind::Utf8);
    let mut output = empty_scalar_values(result_kind, batch.num_rows());

    let masks = conditions
        .iter()
        .map(|condition| evaluate_scalar_predicate(batch, condition, None))
        .collect::<Result<Vec<_>>>()?;
    for row in 0..batch.num_rows() {
        let mut selected = None;
        for (index, mask) in masks.iter().enumerate() {
            if boolean_value(mask, row) == Some(true) {
                selected = evaluated_results.get(index);
                break;
            }
        }
        let selected = selected.or(evaluated_else.as_ref());
        set_scalar_value_from(&mut output, row, selected)?;
    }
    Ok(output)
}

fn mask_evaluated_scalar(value: EvaluatedScalar, mask: &BooleanArray) -> Result<EvaluatedScalar> {
    if let EvaluatedScalar::Array(array) = value {
        return Ok(EvaluatedScalar::Array(mask_array_with_boolean(
            array, mask,
        )?));
    }
    let value = materialize_array_scalar(value)?;
    Ok(match value {
        EvaluatedScalar::Array(_) => unreachable!("array scalar was materialized before masking"),
        EvaluatedScalar::Int64(values) => EvaluatedScalar::Int64(mask_options(values, mask)),
        EvaluatedScalar::Float64(values) => EvaluatedScalar::Float64(mask_options(values, mask)),
        EvaluatedScalar::Decimal128 {
            values,
            precision,
            scale,
        } => EvaluatedScalar::Decimal128 {
            values: mask_options(values, mask),
            precision,
            scale,
        },
        EvaluatedScalar::Utf8(values) => EvaluatedScalar::Utf8(mask_options(values, mask)),
        EvaluatedScalar::Boolean(values) => EvaluatedScalar::Boolean(mask_options(values, mask)),
        EvaluatedScalar::Date32(values) => EvaluatedScalar::Date32(mask_options(values, mask)),
        EvaluatedScalar::TimestampMillisecond(values) => {
            EvaluatedScalar::TimestampMillisecond(mask_options(values, mask))
        }
    })
}

fn mask_array_with_boolean(array: ArrayRef, mask: &BooleanArray) -> Result<ArrayRef> {
    let mask_nulls = NullBuffer::from(
        (0..array.len())
            .map(|row| boolean_value(mask, row) == Some(true))
            .collect::<Vec<_>>(),
    );
    let data = array.to_data();
    let nulls = NullBuffer::union_many([data.nulls(), Some(&mask_nulls)]);
    let data = data.into_builder().nulls(nulls).build()?;
    Ok(make_array(data))
}

fn mask_options<T>(mut values: Vec<Option<T>>, mask: &BooleanArray) -> Vec<Option<T>> {
    for (row, value) in values.iter_mut().enumerate() {
        if boolean_value(mask, row) != Some(true) {
            *value = None;
        }
    }
    values
}

fn evaluate_scalar_in_list(
    value: EvaluatedScalar,
    values: &[EvaluatedScalar],
    negated: bool,
) -> Result<Vec<Option<bool>>> {
    let value_kind = evaluated_scalar_kind(&value).unwrap_or(EvaluatedScalarKind::Utf8);
    let mut output = Vec::with_capacity(value.len());
    for row in 0..value.len() {
        let candidate = scalar_value_at(&value, row)?;
        if candidate.is_none() {
            output.push(None);
            continue;
        }
        let mut matched = false;
        let mut has_null = false;
        for value in values {
            let value = scalar_value_at(&cast_scalar_for_kind(value.clone(), value_kind)?, row)?;
            match value {
                Some(value) => {
                    if Some(value) == candidate {
                        matched = true;
                        break;
                    }
                }
                None => has_null = true,
            }
        }
        let result = if matched {
            Some(!negated)
        } else if has_null {
            None
        } else {
            Some(negated)
        };
        output.push(result);
    }
    Ok(output)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ScalarLikeToken {
    Any,
    One,
    Char(char),
}

fn scalar_like_pattern_tokens(pattern: &str, escape: Option<char>) -> Result<Vec<ScalarLikeToken>> {
    let mut tokens = Vec::new();
    let mut chars = pattern.chars();
    while let Some(ch) = chars.next() {
        if Some(ch) == escape {
            let escaped = chars.next().ok_or_else(|| {
                DodamError::InvalidFilter("LIKE ESCAPE at end of pattern".to_string())
            })?;
            tokens.push(ScalarLikeToken::Char(escaped));
        } else {
            tokens.push(match ch {
                '%' => ScalarLikeToken::Any,
                '_' => ScalarLikeToken::One,
                ch => ScalarLikeToken::Char(ch),
            });
        }
    }
    Ok(tokens)
}

fn scalar_like_matches(value: &str, pattern: &[ScalarLikeToken]) -> bool {
    fn matches_from(
        value: &[char],
        pattern: &[ScalarLikeToken],
        value_index: usize,
        pattern_index: usize,
    ) -> bool {
        if pattern_index == pattern.len() {
            return value_index == value.len();
        }
        match pattern[pattern_index] {
            ScalarLikeToken::Char(ch) => {
                value.get(value_index).is_some_and(|value| *value == ch)
                    && matches_from(value, pattern, value_index + 1, pattern_index + 1)
            }
            ScalarLikeToken::One => {
                value_index < value.len()
                    && matches_from(value, pattern, value_index + 1, pattern_index + 1)
            }
            ScalarLikeToken::Any => (value_index..=value.len())
                .any(|index| matches_from(value, pattern, index, pattern_index + 1)),
        }
    }
    let value = value.chars().collect::<Vec<_>>();
    matches_from(&value, pattern, 0, 0)
}

fn substring_value(
    value: Option<&str>,
    start: Option<i64>,
    length: Option<Option<i64>>,
) -> Option<String> {
    let value = value?;
    let start = start?;
    let chars = value.chars().collect::<Vec<_>>();
    let start_index = if start <= 1 {
        0
    } else {
        usize::try_from(start - 1).ok()?
    };
    let available = chars.len().saturating_sub(start_index);
    let take = match length {
        Some(Some(length)) if length <= 0 => 0,
        Some(Some(length)) => usize::try_from(length).ok()?.min(available),
        Some(None) => return None,
        None => available,
    };
    Some(chars.iter().skip(start_index).take(take).collect())
}

#[derive(Clone, PartialEq)]
pub(super) enum ScalarValue {
    Int64(i64),
    Float64(f64),
    Decimal128(i128, u8, i8),
    Utf8(String),
    Boolean(bool),
    Date32(i32),
    TimestampMillisecond(i64),
}

pub(super) fn scalar_value_at(value: &EvaluatedScalar, row: usize) -> Result<Option<ScalarValue>> {
    Ok(match value {
        EvaluatedScalar::Array(array) => {
            return array_scalar_value_at(array.as_ref(), row);
        }
        EvaluatedScalar::Int64(values) => values[row].map(ScalarValue::Int64),
        EvaluatedScalar::Float64(values) => values[row].map(ScalarValue::Float64),
        EvaluatedScalar::Decimal128 {
            values,
            precision,
            scale,
        } => values[row].map(|value| ScalarValue::Decimal128(value, *precision, *scale)),
        EvaluatedScalar::Utf8(values) => values[row].clone().map(ScalarValue::Utf8),
        EvaluatedScalar::Boolean(values) => values[row].map(ScalarValue::Boolean),
        EvaluatedScalar::Date32(values) => values[row].map(ScalarValue::Date32),
        EvaluatedScalar::TimestampMillisecond(values) => {
            values[row].map(ScalarValue::TimestampMillisecond)
        }
    })
}

fn array_scalar_value_at(array: &dyn Array, row: usize) -> Result<Option<ScalarValue>> {
    if array.is_null(row) {
        return Ok(None);
    }
    if let Some(values) = array.as_any().downcast_ref::<Int64Array>() {
        return Ok(Some(ScalarValue::Int64(values.value(row))));
    }
    if let Some(values) = array.as_any().downcast_ref::<Int32Array>() {
        return Ok(Some(ScalarValue::Int64(i64::from(values.value(row)))));
    }
    if let Some(values) = array.as_any().downcast_ref::<Float64Array>() {
        return Ok(Some(ScalarValue::Float64(values.value(row))));
    }
    if let Some(values) = array.as_any().downcast_ref::<Decimal128Array>() {
        if let DataType::Decimal128(precision, scale) = array.data_type() {
            return Ok(Some(ScalarValue::Decimal128(
                values.value(row),
                *precision,
                *scale,
            )));
        }
    }
    if let Some(values) = array.as_any().downcast_ref::<StringArray>() {
        return Ok(Some(ScalarValue::Utf8(values.value(row).to_string())));
    }
    if let Some(values) = array.as_any().downcast_ref::<BooleanArray>() {
        return Ok(Some(ScalarValue::Boolean(values.value(row))));
    }
    if let Some(values) = array.as_any().downcast_ref::<Date32Array>() {
        return Ok(Some(ScalarValue::Date32(values.value(row))));
    }
    if let Some(values) = array.as_any().downcast_ref::<TimestampMillisecondArray>() {
        return Ok(Some(ScalarValue::TimestampMillisecond(values.value(row))));
    }
    scalar_value_at(&evaluated_array(array)?, row)
}

fn cast_scalar_for_kind(
    value: EvaluatedScalar,
    kind: EvaluatedScalarKind,
) -> Result<EvaluatedScalar> {
    match kind {
        EvaluatedScalarKind::Int64 => Ok(EvaluatedScalar::Int64(scalar_as_i64(value)?)),
        EvaluatedScalarKind::Float64 => Ok(EvaluatedScalar::Float64(scalar_as_f64(value)?)),
        EvaluatedScalarKind::Decimal128 { precision, scale } => match value {
            EvaluatedScalar::Decimal128 {
                values,
                precision: value_precision,
                scale: value_scale,
            } if value_precision == precision && value_scale == scale => {
                Ok(EvaluatedScalar::Decimal128 {
                    values,
                    precision,
                    scale,
                })
            }
            other => Err(DodamError::UnsupportedSql(format!(
                "cannot use {} in decimal IN list",
                other.data_type()
            ))),
        },
        EvaluatedScalarKind::Utf8 => Ok(EvaluatedScalar::Utf8(scalar_as_utf8(value)?)),
        EvaluatedScalarKind::Date32 => match value {
            EvaluatedScalar::Date32(_) => Ok(value),
            other => Err(DodamError::UnsupportedSql(format!(
                "cannot use {} in date IN list",
                other.data_type()
            ))),
        },
        EvaluatedScalarKind::TimestampMillisecond => match value {
            EvaluatedScalar::TimestampMillisecond(_) => Ok(value),
            other => Err(DodamError::UnsupportedSql(format!(
                "cannot use {} in timestamp IN list",
                other.data_type()
            ))),
        },
        EvaluatedScalarKind::Boolean => match value {
            EvaluatedScalar::Boolean(_) => Ok(value),
            other => Err(DodamError::UnsupportedSql(format!(
                "cannot use {} in boolean IN list",
                other.data_type()
            ))),
        },
    }
}

#[derive(Clone, Copy)]
enum EvaluatedScalarKind {
    Int64,
    Float64,
    Decimal128 { precision: u8, scale: i8 },
    Utf8,
    Boolean,
    Date32,
    TimestampMillisecond,
}

fn evaluated_scalar_kind(value: &EvaluatedScalar) -> Option<EvaluatedScalarKind> {
    Some(match value {
        EvaluatedScalar::Array(array) => match array.data_type() {
            DataType::Int32 | DataType::Int64 => EvaluatedScalarKind::Int64,
            DataType::Float64 => EvaluatedScalarKind::Float64,
            DataType::Utf8 => EvaluatedScalarKind::Utf8,
            DataType::Boolean => EvaluatedScalarKind::Boolean,
            DataType::Date32 => EvaluatedScalarKind::Date32,
            DataType::Timestamp(TimeUnit::Millisecond, _) | DataType::Date64 => {
                EvaluatedScalarKind::TimestampMillisecond
            }
            DataType::Decimal128(precision, scale) => EvaluatedScalarKind::Decimal128 {
                precision: *precision,
                scale: *scale,
            },
            _ => return None,
        },
        EvaluatedScalar::Int64(_) => EvaluatedScalarKind::Int64,
        EvaluatedScalar::Float64(_) => EvaluatedScalarKind::Float64,
        EvaluatedScalar::Decimal128 {
            precision, scale, ..
        } => EvaluatedScalarKind::Decimal128 {
            precision: *precision,
            scale: *scale,
        },
        EvaluatedScalar::Utf8(_) => EvaluatedScalarKind::Utf8,
        EvaluatedScalar::Boolean(_) => EvaluatedScalarKind::Boolean,
        EvaluatedScalar::Date32(_) => EvaluatedScalarKind::Date32,
        EvaluatedScalar::TimestampMillisecond(_) => EvaluatedScalarKind::TimestampMillisecond,
    })
}

fn empty_scalar_values(kind: EvaluatedScalarKind, rows: usize) -> EvaluatedScalar {
    match kind {
        EvaluatedScalarKind::Int64 => EvaluatedScalar::Int64(vec![None; rows]),
        EvaluatedScalarKind::Float64 => EvaluatedScalar::Float64(vec![None; rows]),
        EvaluatedScalarKind::Decimal128 { precision, scale } => EvaluatedScalar::Decimal128 {
            values: vec![None; rows],
            precision,
            scale,
        },
        EvaluatedScalarKind::Utf8 => EvaluatedScalar::Utf8(vec![None; rows]),
        EvaluatedScalarKind::Boolean => EvaluatedScalar::Boolean(vec![None; rows]),
        EvaluatedScalarKind::Date32 => EvaluatedScalar::Date32(vec![None; rows]),
        EvaluatedScalarKind::TimestampMillisecond => {
            EvaluatedScalar::TimestampMillisecond(vec![None; rows])
        }
    }
}

fn set_scalar_value_from(
    output: &mut EvaluatedScalar,
    row: usize,
    source: Option<&EvaluatedScalar>,
) -> Result<()> {
    match output {
        EvaluatedScalar::Array(_) => {
            return Err(DodamError::UnsupportedSql(
                "CASE output cannot be an array-backed scalar".to_string(),
            ));
        }
        EvaluatedScalar::Int64(values) => {
            values[row] = source
                .map(|source| scalar_value_as_i64(source, row))
                .transpose()?
                .flatten();
        }
        EvaluatedScalar::Float64(values) => {
            values[row] = source
                .map(|source| scalar_value_as_f64(source, row))
                .transpose()?
                .flatten();
        }
        EvaluatedScalar::Decimal128 {
            values,
            precision,
            scale,
        } => {
            values[row] = source
                .map(|source| scalar_value_as_decimal128(source, row, *precision, *scale))
                .transpose()?
                .flatten();
        }
        EvaluatedScalar::Utf8(values) => {
            values[row] = source
                .map(|source| scalar_value_as_utf8(source, row))
                .transpose()?
                .flatten();
        }
        EvaluatedScalar::Boolean(values) => {
            values[row] = source
                .map(|source| scalar_value_as_bool(source, row))
                .transpose()?
                .flatten();
        }
        EvaluatedScalar::Date32(values) => {
            values[row] = source
                .map(|source| scalar_value_as_date32(source, row))
                .transpose()?
                .flatten();
        }
        EvaluatedScalar::TimestampMillisecond(values) => {
            values[row] = source
                .map(|source| scalar_value_as_timestamp_millis(source, row))
                .transpose()?
                .flatten();
        }
    }
    Ok(())
}

pub(super) fn scalar_value_as_i64(value: &EvaluatedScalar, row: usize) -> Result<Option<i64>> {
    match value {
        EvaluatedScalar::Array(array) => array_value_as_i64(array.as_ref(), row),
        EvaluatedScalar::Int64(values) => Ok(values[row]),
        _ => Err(DodamError::UnsupportedSql(
            "CASE result type mismatch".to_string(),
        )),
    }
}

fn array_value_as_i64(array: &dyn Array, row: usize) -> Result<Option<i64>> {
    if array.is_null(row) {
        return Ok(None);
    }
    if let Some(values) = array.as_any().downcast_ref::<Int64Array>() {
        return Ok(Some(values.value(row)));
    }
    if let Some(values) = array.as_any().downcast_ref::<Int32Array>() {
        return Ok(Some(i64::from(values.value(row))));
    }
    scalar_value_as_i64(&evaluated_array(array)?, row)
}

pub(super) fn scalar_value_as_f64(value: &EvaluatedScalar, row: usize) -> Result<Option<f64>> {
    match value {
        EvaluatedScalar::Array(array) => {
            scalar_value_as_f64(&evaluated_array(array.as_ref())?, row)
        }
        EvaluatedScalar::Int64(values) => Ok(values[row].map(|value| value as f64)),
        EvaluatedScalar::Float64(values) => Ok(values[row]),
        EvaluatedScalar::Decimal128 { values, scale, .. } => {
            let scale = decimal_scale_f64(*scale)?;
            Ok(values[row].map(|value| value as f64 / scale))
        }
        _ => Err(DodamError::UnsupportedSql(
            "CASE result type mismatch".to_string(),
        )),
    }
}

fn scalar_value_as_decimal128(
    value: &EvaluatedScalar,
    row: usize,
    precision: u8,
    scale: i8,
) -> Result<Option<i128>> {
    match value {
        EvaluatedScalar::Array(array) => {
            scalar_value_as_decimal128(&evaluated_array(array.as_ref())?, row, precision, scale)
        }
        EvaluatedScalar::Decimal128 {
            values,
            precision: value_precision,
            scale: value_scale,
        } if *value_precision == precision && *value_scale == scale => Ok(values[row]),
        _ => Err(DodamError::UnsupportedSql(
            "CASE result type mismatch".to_string(),
        )),
    }
}

fn scalar_value_as_utf8(value: &EvaluatedScalar, row: usize) -> Result<Option<String>> {
    match value {
        EvaluatedScalar::Array(array) => {
            scalar_value_as_utf8(&evaluated_array(array.as_ref())?, row)
        }
        EvaluatedScalar::Int64(values) => Ok(values[row].map(|value| value.to_string())),
        EvaluatedScalar::Float64(values) => Ok(values[row].map(format_f64_for_sql_varchar)),
        EvaluatedScalar::Decimal128 { values, scale, .. } => {
            Ok(values[row].map(|value| format_decimal128_value(value, *scale)))
        }
        EvaluatedScalar::Utf8(values) => Ok(values[row].clone()),
        EvaluatedScalar::Boolean(values) => Ok(values[row].map(|value| value.to_string())),
        EvaluatedScalar::Date32(values) => Ok(values[row].map(format_date32_days)),
        EvaluatedScalar::TimestampMillisecond(values) => {
            Ok(values[row].map(format_timestamp_millis))
        }
    }
}

fn scalar_value_as_bool(value: &EvaluatedScalar, row: usize) -> Result<Option<bool>> {
    match value {
        EvaluatedScalar::Array(array) => {
            scalar_value_as_bool(&evaluated_array(array.as_ref())?, row)
        }
        EvaluatedScalar::Boolean(values) => Ok(values[row]),
        _ => Err(DodamError::UnsupportedSql(
            "CASE result type mismatch".to_string(),
        )),
    }
}

fn scalar_value_as_date32(value: &EvaluatedScalar, row: usize) -> Result<Option<i32>> {
    match value {
        EvaluatedScalar::Array(array) => {
            scalar_value_as_date32(&evaluated_array(array.as_ref())?, row)
        }
        EvaluatedScalar::Date32(values) => Ok(values[row]),
        _ => Err(DodamError::UnsupportedSql(
            "CASE result type mismatch".to_string(),
        )),
    }
}

fn scalar_value_as_timestamp_millis(value: &EvaluatedScalar, row: usize) -> Result<Option<i64>> {
    match value {
        EvaluatedScalar::Array(array) => {
            scalar_value_as_timestamp_millis(&evaluated_array(array.as_ref())?, row)
        }
        EvaluatedScalar::TimestampMillisecond(values) => Ok(values[row]),
        _ => Err(DodamError::UnsupportedSql(
            "CASE result type mismatch".to_string(),
        )),
    }
}

pub(super) fn evaluated_column(batch: &RecordBatch, column: &str) -> Result<EvaluatedScalar> {
    let index = output_batch_column_index(batch, column)?;
    Ok(EvaluatedScalar::Array(batch.column(index).clone()))
}

fn evaluated_struct_field(
    batch: &RecordBatch,
    column: &str,
    field: &str,
) -> Result<EvaluatedScalar> {
    evaluated_array(struct_field_array(batch, column, field)?)
}

fn struct_field_array<'a>(
    batch: &'a RecordBatch,
    column: &str,
    field: &str,
) -> Result<&'a dyn Array> {
    let index = output_batch_column_index(batch, column)?;
    let array = batch.column(index);
    let struct_array = array
        .as_any()
        .downcast_ref::<StructArray>()
        .ok_or_else(|| {
            DodamError::UnsupportedSql(format!("{column}.{field} requires a STRUCT column"))
        })?;
    let mut current: &dyn Array = struct_array;
    let mut current_path = column.to_string();
    for field in field.split('.') {
        let struct_array = current
            .as_any()
            .downcast_ref::<StructArray>()
            .ok_or_else(|| {
                DodamError::UnsupportedSql(format!(
                    "{current_path}.{field} requires a STRUCT column"
                ))
            })?;
        let Some((field_index, _)) = struct_array
            .fields()
            .iter()
            .enumerate()
            .find(|(_, schema_field)| schema_field.name() == field)
        else {
            return Err(DodamError::UnknownColumn(format!("{current_path}.{field}")));
        };
        current_path.push('.');
        current_path.push_str(field);
        current = struct_array.column(field_index).as_ref();
    }
    Ok(current)
}

fn evaluated_list_length(
    batch: &RecordBatch,
    column: &str,
    field: Option<&str>,
) -> Result<EvaluatedScalar> {
    let list = list_array_column(batch, column, field)?;
    Ok(EvaluatedScalar::Int64(
        (0..list.len())
            .map(|row| {
                if list.is_null(row) {
                    None
                } else {
                    Some(i64::from(list.value_length(row)))
                }
            })
            .collect(),
    ))
}

fn evaluated_list_index(
    batch: &RecordBatch,
    column: &str,
    field: Option<&str>,
    indexes: &[Option<i64>],
) -> Result<EvaluatedScalar> {
    let list = list_array_column(batch, column, field)?;
    if indexes.len() != list.len() {
        return Err(DodamError::UnsupportedSql(format!(
            "list index expression length {} does not match list length {}",
            indexes.len(),
            list.len()
        )));
    }
    let values = list.values();
    match values.data_type() {
        DataType::Int32 => {
            let values = values.as_any().downcast_ref::<Int32Array>().expect("Int32");
            Ok(EvaluatedScalar::Int64(
                list_element_indices(list, indexes)
                    .into_iter()
                    .map(|value_index| {
                        value_index.and_then(|value_index| {
                            values
                                .is_valid(value_index)
                                .then(|| i64::from(values.value(value_index)))
                        })
                    })
                    .collect(),
            ))
        }
        DataType::Int64 => {
            let values = values.as_any().downcast_ref::<Int64Array>().expect("Int64");
            Ok(EvaluatedScalar::Int64(
                list_element_indices(list, indexes)
                    .into_iter()
                    .map(|value_index| {
                        value_index.and_then(|value_index| {
                            values
                                .is_valid(value_index)
                                .then(|| values.value(value_index))
                        })
                    })
                    .collect(),
            ))
        }
        DataType::Utf8 => {
            let values = values.as_any().downcast_ref::<StringArray>().expect("Utf8");
            Ok(EvaluatedScalar::Utf8(
                list_element_indices(list, indexes)
                    .into_iter()
                    .map(|value_index| {
                        value_index.and_then(|value_index| {
                            values
                                .is_valid(value_index)
                                .then(|| values.value(value_index).to_string())
                        })
                    })
                    .collect(),
            ))
        }
        data_type => Err(DodamError::UnsupportedSql(format!(
            "list index over {data_type} values is not supported yet"
        ))),
    }
}

fn list_array_column<'a>(
    batch: &'a RecordBatch,
    column: &str,
    field: Option<&str>,
) -> Result<&'a ListArray> {
    let array: &'a dyn Array = if let Some(field) = field {
        struct_field_array(batch, column, field)?
    } else {
        let index = output_batch_column_index(batch, column)?;
        batch.column(index).as_ref()
    };
    array.as_any().downcast_ref::<ListArray>().ok_or_else(|| {
        let name = field.map_or_else(|| column.to_string(), |field| format!("{column}.{field}"));
        DodamError::UnsupportedSql(format!("{name} requires a LIST column"))
    })
}

fn list_element_indices(list: &ListArray, indexes: &[Option<i64>]) -> Vec<Option<usize>> {
    (0..list.len())
        .map(|row| {
            if list.is_null(row) {
                return None;
            }
            let index = indexes[row]?;
            if index <= 0 {
                return None;
            }
            let index = usize::try_from(index).ok()?;
            let len = usize::try_from(list.value_length(row)).ok()?;
            if index > len {
                return None;
            }
            let offset = usize::try_from(list.value_offsets()[row]).ok()?;
            Some(offset + index - 1)
        })
        .collect()
}

pub(super) fn evaluated_array(array: &dyn Array) -> Result<EvaluatedScalar> {
    match array.data_type() {
        DataType::Int32 => {
            let values = array.as_any().downcast_ref::<Int32Array>().expect("Int32");
            Ok(EvaluatedScalar::Int64(
                values.iter().map(|value| value.map(i64::from)).collect(),
            ))
        }
        DataType::Int64 => {
            let values = array.as_any().downcast_ref::<Int64Array>().expect("Int64");
            Ok(EvaluatedScalar::Int64(values.iter().collect()))
        }
        DataType::Float64 => {
            let values = array
                .as_any()
                .downcast_ref::<Float64Array>()
                .expect("Float64");
            Ok(EvaluatedScalar::Float64(values.iter().collect()))
        }
        DataType::Utf8 => {
            let values = array.as_any().downcast_ref::<StringArray>().expect("Utf8");
            Ok(EvaluatedScalar::Utf8(
                values
                    .iter()
                    .map(|value| value.map(str::to_string))
                    .collect(),
            ))
        }
        DataType::Boolean => {
            let values = array
                .as_any()
                .downcast_ref::<BooleanArray>()
                .expect("Boolean");
            Ok(EvaluatedScalar::Boolean(values.iter().collect()))
        }
        DataType::Date32 => {
            let values = array
                .as_any()
                .downcast_ref::<Date32Array>()
                .expect("Date32");
            Ok(EvaluatedScalar::Date32(values.iter().collect()))
        }
        DataType::Date64 => {
            let values = array
                .as_any()
                .downcast_ref::<Date64Array>()
                .expect("Date64");
            Ok(EvaluatedScalar::TimestampMillisecond(
                values.iter().collect(),
            ))
        }
        DataType::Timestamp(TimeUnit::Millisecond, _) => {
            let values = array
                .as_any()
                .downcast_ref::<TimestampMillisecondArray>()
                .expect("TimestampMillisecond");
            Ok(EvaluatedScalar::TimestampMillisecond(
                values.iter().collect(),
            ))
        }
        DataType::Decimal128(precision, scale) => {
            let values = array
                .as_any()
                .downcast_ref::<Decimal128Array>()
                .expect("Decimal128");
            Ok(EvaluatedScalar::Decimal128 {
                values: values.iter().collect(),
                precision: *precision,
                scale: *scale,
            })
        }
        data_type => Err(DodamError::UnsupportedSql(format!(
            "projection expression column type {data_type} is not supported yet"
        ))),
    }
}

fn evaluated_literal(value: &LiteralValue, rows: usize) -> EvaluatedScalar {
    match value {
        LiteralValue::Null => EvaluatedScalar::Utf8(vec![None; rows]),
        LiteralValue::Boolean(value) => EvaluatedScalar::Boolean(vec![Some(*value); rows]),
        LiteralValue::Int64(value) => EvaluatedScalar::Int64(vec![Some(*value); rows]),
        LiteralValue::Float64(value) => EvaluatedScalar::Float64(vec![Some(*value); rows]),
        LiteralValue::Utf8(value) => EvaluatedScalar::Utf8(vec![Some(value.clone()); rows]),
    }
}

fn compare_evaluated_scalars(
    left: EvaluatedScalar,
    op: &BinaryOperator,
    right: EvaluatedScalar,
) -> Result<Vec<Option<bool>>> {
    let left = materialize_array_scalar(left)?;
    let right = materialize_array_scalar(right)?;
    match (&left, &right) {
        (EvaluatedScalar::Utf8(_), _) | (_, EvaluatedScalar::Utf8(_)) => {
            let left = scalar_as_utf8(left)?;
            let right = scalar_as_utf8(right)?;
            Ok(left
                .into_iter()
                .zip(right)
                .map(|(left, right)| compare_optional_values(left, op, right))
                .collect())
        }
        (EvaluatedScalar::Boolean(_), _) | (_, EvaluatedScalar::Boolean(_)) => {
            let EvaluatedScalar::Boolean(left) = left else {
                return Err(DodamError::UnsupportedSql(
                    "boolean comparisons require boolean operands".to_string(),
                ));
            };
            let EvaluatedScalar::Boolean(right) = right else {
                return Err(DodamError::UnsupportedSql(
                    "boolean comparisons require boolean operands".to_string(),
                ));
            };
            Ok(left
                .into_iter()
                .zip(right)
                .map(|(left, right)| compare_optional_values(left, op, right))
                .collect())
        }
        (EvaluatedScalar::Float64(_), _) | (_, EvaluatedScalar::Float64(_)) => {
            let left = scalar_as_f64(left)?;
            let right = scalar_as_f64(right)?;
            Ok(left
                .into_iter()
                .zip(right)
                .map(|(left, right)| compare_optional_f64(left, op, right))
                .collect())
        }
        (
            EvaluatedScalar::Decimal128 {
                values: left,
                precision: _,
                scale: left_scale,
            },
            EvaluatedScalar::Decimal128 {
                values: right,
                precision: _,
                scale: right_scale,
            },
        ) => {
            let scale = (*left_scale).max(*right_scale);
            let left_factor = decimal_align_factor(*left_scale, scale)?;
            let right_factor = decimal_align_factor(*right_scale, scale)?;
            Ok(left
                .iter()
                .copied()
                .zip(right.iter().copied())
                .map(|(left, right)| match (left, right) {
                    (Some(left), Some(right)) => {
                        let left = left.checked_mul(left_factor)?;
                        let right = right.checked_mul(right_factor)?;
                        compare_optional_values(Some(left), op, Some(right))
                    }
                    _ => None,
                })
                .collect())
        }
        (EvaluatedScalar::Decimal128 { .. }, _) | (_, EvaluatedScalar::Decimal128 { .. }) => {
            let left = scalar_as_f64(left)?;
            let right = scalar_as_f64(right)?;
            Ok(left
                .into_iter()
                .zip(right)
                .map(|(left, right)| compare_optional_f64(left, op, right))
                .collect())
        }
        _ => {
            let left = scalar_as_i64(left)?;
            let right = scalar_as_i64(right)?;
            Ok(left
                .into_iter()
                .zip(right)
                .map(|(left, right)| compare_optional_values(left, op, right))
                .collect())
        }
    }
}

fn materialize_array_scalar(value: EvaluatedScalar) -> Result<EvaluatedScalar> {
    match value {
        EvaluatedScalar::Array(array) => evaluated_array(array.as_ref()),
        other => Ok(other),
    }
}

fn compare_optional_values<T: Ord>(
    left: Option<T>,
    op: &BinaryOperator,
    right: Option<T>,
) -> Option<bool> {
    let (Some(left), Some(right)) = (left, right) else {
        return None;
    };
    Some(match op {
        BinaryOperator::Eq => left == right,
        BinaryOperator::NotEq => left != right,
        BinaryOperator::Gt => left > right,
        BinaryOperator::GtEq => left >= right,
        BinaryOperator::Lt => left < right,
        BinaryOperator::LtEq => left <= right,
        _ => unreachable!("validated comparison operator"),
    })
}

fn compare_optional_f64(
    left: Option<f64>,
    op: &BinaryOperator,
    right: Option<f64>,
) -> Option<bool> {
    let (Some(left), Some(right)) = (left, right) else {
        return None;
    };
    Some(match op {
        BinaryOperator::Eq => left == right,
        BinaryOperator::NotEq => left != right,
        BinaryOperator::Gt => left > right,
        BinaryOperator::GtEq => left >= right,
        BinaryOperator::Lt => left < right,
        BinaryOperator::LtEq => left <= right,
        _ => unreachable!("validated comparison operator"),
    })
}

fn scalar_null_mask(value: EvaluatedScalar) -> Vec<bool> {
    match value {
        EvaluatedScalar::Array(array) => (0..array.len()).map(|row| array.is_null(row)).collect(),
        EvaluatedScalar::Int64(values) => values.into_iter().map(|value| value.is_none()).collect(),
        EvaluatedScalar::Float64(values) => {
            values.into_iter().map(|value| value.is_none()).collect()
        }
        EvaluatedScalar::Decimal128 { values, .. } => {
            values.into_iter().map(|value| value.is_none()).collect()
        }
        EvaluatedScalar::Utf8(values) => values.into_iter().map(|value| value.is_none()).collect(),
        EvaluatedScalar::Boolean(values) => {
            values.into_iter().map(|value| value.is_none()).collect()
        }
        EvaluatedScalar::Date32(values) => {
            values.into_iter().map(|value| value.is_none()).collect()
        }
        EvaluatedScalar::TimestampMillisecond(values) => {
            values.into_iter().map(|value| value.is_none()).collect()
        }
    }
}

fn evaluate_binary_scalar(
    left: EvaluatedScalar,
    op: &BinaryOperator,
    right: EvaluatedScalar,
) -> Result<EvaluatedScalar> {
    let left = materialize_array_scalar(left)?;
    let right = materialize_array_scalar(right)?;
    match (&left, &right) {
        (
            EvaluatedScalar::Decimal128 {
                values: left,
                precision: left_precision,
                scale: left_scale,
            },
            EvaluatedScalar::Decimal128 {
                values: right,
                precision: right_precision,
                scale: right_scale,
            },
        ) if matches!(op, BinaryOperator::Plus | BinaryOperator::Minus) => {
            let scale = (*left_scale).max(*right_scale);
            let precision = left_precision
                .max(right_precision)
                .saturating_add(1)
                .min(38);
            let mut values = Vec::with_capacity(left.len());
            for (left, right) in left.iter().copied().zip(right.iter().copied()) {
                let value = match (left, right) {
                    (Some(left), Some(right)) => {
                        let left = align_decimal_value(left, *left_scale, scale)?;
                        let right = align_decimal_value(right, *right_scale, scale)?;
                        Some(
                            match op {
                                BinaryOperator::Plus => left.checked_add(right),
                                BinaryOperator::Minus => left.checked_sub(right),
                                _ => None,
                            }
                            .ok_or_else(|| {
                                DodamError::UnsupportedSql(
                                    "decimal arithmetic overflow".to_string(),
                                )
                            })?,
                        )
                    }
                    _ => None,
                };
                values.push(value);
            }
            Ok(EvaluatedScalar::Decimal128 {
                values,
                precision,
                scale,
            })
        }
        (
            EvaluatedScalar::Decimal128 {
                values: left,
                precision: left_precision,
                scale: left_scale,
            },
            EvaluatedScalar::Decimal128 {
                values: right,
                precision: right_precision,
                scale: right_scale,
            },
        ) if *op == BinaryOperator::Multiply => {
            let scale = left_scale.checked_add(*right_scale).ok_or_else(|| {
                DodamError::UnsupportedSql("decimal multiply scale overflow".to_string())
            })?;
            let precision = left_precision
                .saturating_add(*right_precision)
                .clamp(18, 38);
            let mut values = Vec::with_capacity(left.len());
            for (left, right) in left.iter().copied().zip(right.iter().copied()) {
                let value = match (left, right) {
                    (Some(left), Some(right)) => {
                        Some(left.checked_mul(right).ok_or_else(|| {
                            DodamError::UnsupportedSql("decimal multiply overflow".to_string())
                        })?)
                    }
                    _ => None,
                };
                values.push(value);
            }
            Ok(EvaluatedScalar::Decimal128 {
                values,
                precision,
                scale,
            })
        }
        (EvaluatedScalar::Float64(_), _)
        | (_, EvaluatedScalar::Float64(_))
        | (EvaluatedScalar::Decimal128 { .. }, _)
        | (_, EvaluatedScalar::Decimal128 { .. }) => {
            let left = scalar_as_f64(left)?;
            let right = scalar_as_f64(right)?;
            Ok(EvaluatedScalar::Float64(
                left.into_iter()
                    .zip(right)
                    .map(|(left, right)| match (left, right) {
                        (Some(left), Some(right)) => match op {
                            BinaryOperator::Plus => Some(left + right),
                            BinaryOperator::Minus => Some(left - right),
                            BinaryOperator::Multiply => Some(left * right),
                            BinaryOperator::Divide => Some(left / right),
                            _ => None,
                        },
                        _ => None,
                    })
                    .collect(),
            ))
        }
        _ => {
            let left = scalar_as_i64(left)?;
            let right = scalar_as_i64(right)?;
            Ok(EvaluatedScalar::Int64(
                left.into_iter()
                    .zip(right)
                    .map(|(left, right)| match (left, right) {
                        (Some(left), Some(right)) => match op {
                            BinaryOperator::Plus => left.checked_add(right),
                            BinaryOperator::Minus => left.checked_sub(right),
                            BinaryOperator::Multiply => left.checked_mul(right),
                            BinaryOperator::Divide if right != 0 => Some(left / right),
                            BinaryOperator::Divide => None,
                            _ => None,
                        },
                        _ => None,
                    })
                    .collect(),
            ))
        }
    }
}

pub(super) fn scalar_as_i64(value: EvaluatedScalar) -> Result<Vec<Option<i64>>> {
    match value {
        EvaluatedScalar::Array(array) => scalar_as_i64(evaluated_array(array.as_ref())?),
        EvaluatedScalar::Int64(values) => Ok(values),
        EvaluatedScalar::Date32(values) => Ok(values
            .into_iter()
            .map(|value| value.map(i64::from))
            .collect()),
        EvaluatedScalar::TimestampMillisecond(values) => Ok(values),
        other => Err(DodamError::TypeMismatch(format!(
            "cannot use {} in integer arithmetic",
            other.data_type()
        ))),
    }
}

pub(super) fn scalar_as_f64(value: EvaluatedScalar) -> Result<Vec<Option<f64>>> {
    match value {
        EvaluatedScalar::Array(array) => scalar_as_f64(evaluated_array(array.as_ref())?),
        EvaluatedScalar::Int64(values) => Ok(values
            .into_iter()
            .map(|value| value.map(|value| value as f64))
            .collect()),
        EvaluatedScalar::Float64(values) => Ok(values),
        EvaluatedScalar::Decimal128 { values, scale, .. } => {
            let scale = decimal_scale_f64(scale)?;
            Ok(values
                .into_iter()
                .map(|value| value.map(|value| value as f64 / scale))
                .collect())
        }
        EvaluatedScalar::Date32(values) => Ok(values
            .into_iter()
            .map(|value| value.map(f64::from))
            .collect()),
        EvaluatedScalar::TimestampMillisecond(values) => Ok(values
            .into_iter()
            .map(|value| value.map(|value| value as f64))
            .collect()),
        other => Err(DodamError::TypeMismatch(format!(
            "cannot use {} in floating point arithmetic",
            other.data_type()
        ))),
    }
}

fn cast_evaluated_scalar(value: EvaluatedScalar, target: &str) -> Result<EvaluatedScalar> {
    let target = target.to_ascii_lowercase();
    let value = materialize_array_scalar(value)?;
    if matches!(
        target.as_str(),
        "varchar" | "text" | "string" | "char" | "character varying"
    ) {
        return Ok(EvaluatedScalar::Utf8(match value {
            EvaluatedScalar::Array(_) => unreachable!("array scalar was materialized before cast"),
            EvaluatedScalar::Int64(values) => values
                .into_iter()
                .map(|value| value.map(|value| value.to_string()))
                .collect(),
            EvaluatedScalar::Float64(values) => values
                .into_iter()
                .map(|value| value.map(format_f64_for_sql_varchar))
                .collect(),
            EvaluatedScalar::Decimal128 { values, scale, .. } => values
                .into_iter()
                .map(|value| value.map(|value| format_decimal128_value(value, scale)))
                .collect(),
            EvaluatedScalar::Utf8(values) => values,
            EvaluatedScalar::Boolean(values) => values
                .into_iter()
                .map(|value| value.map(|value| value.to_string()))
                .collect(),
            EvaluatedScalar::Date32(values) => values
                .into_iter()
                .map(|value| value.map(format_date32_days))
                .collect(),
            EvaluatedScalar::TimestampMillisecond(values) => values
                .into_iter()
                .map(|value| value.map(format_timestamp_millis))
                .collect(),
        }));
    }
    if let Some((precision, scale)) = parse_decimal_cast_target(&target)? {
        return cast_evaluated_scalar_to_decimal(value, precision, scale);
    }
    if matches!(target.as_str(), "bigint" | "int8" | "integer" | "int") {
        return Ok(EvaluatedScalar::Int64(match value {
            EvaluatedScalar::Array(_) => unreachable!("array scalar was materialized before cast"),
            EvaluatedScalar::Int64(values) => values,
            EvaluatedScalar::Float64(values) => values
                .into_iter()
                .map(|value| value.map(|value| value as i64))
                .collect(),
            EvaluatedScalar::Decimal128 { values, scale, .. } => {
                let scale = decimal_scale_i128(scale).ok_or_else(|| {
                    DodamError::UnsupportedSql(format!("decimal scale {scale} overflows i128"))
                })?;
                values
                    .into_iter()
                    .map(|value| value.map(|value| (value / scale) as i64))
                    .collect()
            }
            EvaluatedScalar::Utf8(values) => values
                .into_iter()
                .map(|value| {
                    value
                        .map(|value| {
                            value.parse::<i64>().map_err(|_| {
                                DodamError::InvalidCast(format!("cannot cast '{value}' to integer"))
                            })
                        })
                        .transpose()
                })
                .collect::<Result<Vec<_>>>()?,
            EvaluatedScalar::Boolean(values) => values
                .into_iter()
                .map(|value| value.map(i64::from))
                .collect(),
            EvaluatedScalar::Date32(values) => values
                .into_iter()
                .map(|value| value.map(i64::from))
                .collect(),
            EvaluatedScalar::TimestampMillisecond(values) => values,
        }));
    }
    if matches!(target.as_str(), "double" | "float8" | "float" | "real") {
        return Ok(EvaluatedScalar::Float64(match value {
            EvaluatedScalar::Array(_) => unreachable!("array scalar was materialized before cast"),
            EvaluatedScalar::Int64(values) => values
                .into_iter()
                .map(|value| value.map(|value| value as f64))
                .collect(),
            EvaluatedScalar::Float64(values) => values,
            EvaluatedScalar::Decimal128 { values, scale, .. } => {
                let scale = decimal_scale_f64(scale)?;
                values
                    .into_iter()
                    .map(|value| value.map(|value| value as f64 / scale))
                    .collect()
            }
            EvaluatedScalar::Utf8(values) => values
                .into_iter()
                .map(|value| {
                    value
                        .map(|value| {
                            value.parse::<f64>().map_err(|_| {
                                DodamError::InvalidCast(format!("cannot cast '{value}' to double"))
                            })
                        })
                        .transpose()
                })
                .collect::<Result<Vec<_>>>()?,
            EvaluatedScalar::Boolean(values) => values
                .into_iter()
                .map(|value| value.map(|value| if value { 1.0 } else { 0.0 }))
                .collect(),
            EvaluatedScalar::Date32(values) => values
                .into_iter()
                .map(|value| value.map(f64::from))
                .collect(),
            EvaluatedScalar::TimestampMillisecond(values) => values
                .into_iter()
                .map(|value| value.map(|value| value as f64))
                .collect(),
        }));
    }
    if target == "date" {
        return Ok(EvaluatedScalar::Date32(match value {
            EvaluatedScalar::Date32(values) => values,
            EvaluatedScalar::TimestampMillisecond(values) => values
                .into_iter()
                .map(|value| value.map(|value| (value.div_euclid(86_400_000)) as i32))
                .collect(),
            EvaluatedScalar::Utf8(values) => values
                .into_iter()
                .map(|value| value.map(|value| parse_date32_days(&value)).transpose())
                .collect::<Result<Vec<_>>>()?,
            other => {
                return Err(DodamError::InvalidCast(format!(
                    "cannot cast {} to DATE",
                    other.data_type()
                )));
            }
        }));
    }
    if matches!(
        target.as_str(),
        "timestamp" | "timestamp without time zone" | "timestamp with time zone" | "timestamptz"
    ) {
        return Ok(EvaluatedScalar::TimestampMillisecond(match value {
            EvaluatedScalar::TimestampMillisecond(values) => values,
            EvaluatedScalar::Date32(values) => values
                .into_iter()
                .map(|value| value.map(|value| i64::from(value) * 86_400_000))
                .collect(),
            EvaluatedScalar::Utf8(values) => values
                .into_iter()
                .map(|value| {
                    value
                        .map(|value| parse_timestamp_millis_value(&value))
                        .transpose()
                })
                .collect::<Result<Vec<_>>>()?,
            other => {
                return Err(DodamError::InvalidCast(format!(
                    "cannot cast {} to TIMESTAMP",
                    other.data_type()
                )));
            }
        }));
    }
    Err(DodamError::UnsupportedSql(format!(
        "unsupported CAST target: {target}"
    )))
}

fn cast_evaluated_scalar_to_decimal(
    value: EvaluatedScalar,
    precision: u8,
    scale: i8,
) -> Result<EvaluatedScalar> {
    let values = match value {
        EvaluatedScalar::Decimal128 {
            values,
            scale: input_scale,
            ..
        } => values
            .into_iter()
            .map(|value| {
                value
                    .map(|value| rescale_decimal_value(value, input_scale, scale, precision))
                    .transpose()
            })
            .collect::<Result<Vec<_>>>()?,
        EvaluatedScalar::Int64(values) => {
            let factor = decimal_scale_i128(scale).ok_or_else(|| {
                DodamError::InvalidCast(format!("decimal scale {scale} overflows"))
            })?;
            values
                .into_iter()
                .map(|value| {
                    value
                        .map(|value| {
                            let value = i128::from(value).checked_mul(factor).ok_or_else(|| {
                                DodamError::InvalidCast("decimal cast overflow".to_string())
                            })?;
                            validate_decimal_precision(value, precision)
                        })
                        .transpose()
                })
                .collect::<Result<Vec<_>>>()?
        }
        EvaluatedScalar::Float64(values) => values
            .into_iter()
            .map(|value| {
                value
                    .map(|value| {
                        parse_decimal_literal_to_scaled(&value.to_string(), scale, precision)
                    })
                    .transpose()
            })
            .collect::<Result<Vec<_>>>()?,
        EvaluatedScalar::Utf8(values) => values
            .into_iter()
            .map(|value| {
                value
                    .map(|value| parse_decimal_literal_to_scaled(&value, scale, precision))
                    .transpose()
            })
            .collect::<Result<Vec<_>>>()?,
        other => {
            return Err(DodamError::InvalidCast(format!(
                "cannot cast {} to DECIMAL({precision},{scale})",
                other.data_type()
            )));
        }
    };
    Ok(EvaluatedScalar::Decimal128 {
        values,
        precision,
        scale,
    })
}

fn rescale_decimal_value(value: i128, from_scale: i8, to_scale: i8, precision: u8) -> Result<i128> {
    let value = if to_scale >= from_scale {
        align_decimal_value(value, from_scale, to_scale)?
    } else {
        round_decimal_to_lower_scale(value, from_scale, to_scale)?
    };
    validate_decimal_precision(value, precision)
}

fn round_decimal_to_lower_scale(value: i128, from_scale: i8, to_scale: i8) -> Result<i128> {
    let factor = decimal_align_factor(to_scale, from_scale)?;
    let quotient = value / factor;
    let remainder = value % factor;
    let should_round = remainder.abs().checked_mul(2).is_some_and(|v| v >= factor);
    if should_round {
        quotient
            .checked_add(if value.is_negative() { -1 } else { 1 })
            .ok_or_else(|| DodamError::InvalidCast("decimal cast overflow".to_string()))
    } else {
        Ok(quotient)
    }
}

pub(super) fn validate_decimal_precision(value: i128, precision: u8) -> Result<i128> {
    let limit = decimal_scale_i128(precision as i8)
        .ok_or_else(|| DodamError::InvalidCast("decimal precision overflows".to_string()))?;
    if value.abs() >= limit {
        return Err(DodamError::InvalidCast(format!(
            "decimal value {value} is out of range for precision {precision}"
        )));
    }
    Ok(value)
}

fn coalesce_evaluated_scalar(
    left: EvaluatedScalar,
    right: EvaluatedScalar,
) -> Result<EvaluatedScalar> {
    let left = materialize_array_scalar(left)?;
    let right = materialize_array_scalar(right)?;
    match (left, right) {
        (EvaluatedScalar::Utf8(left), right) => {
            let right = scalar_as_utf8(right)?;
            Ok(EvaluatedScalar::Utf8(coalesce_options(left, right)))
        }
        (left, EvaluatedScalar::Utf8(right)) => {
            let left = scalar_as_utf8(left)?;
            Ok(EvaluatedScalar::Utf8(coalesce_options(left, right)))
        }
        (EvaluatedScalar::Float64(left), right) => {
            let right = scalar_as_f64(right)?;
            Ok(EvaluatedScalar::Float64(coalesce_options(left, right)))
        }
        (left, EvaluatedScalar::Float64(right)) => {
            let left = scalar_as_f64(left)?;
            Ok(EvaluatedScalar::Float64(coalesce_options(left, right)))
        }
        (
            EvaluatedScalar::Decimal128 {
                values: left,
                precision,
                scale,
            },
            EvaluatedScalar::Decimal128 {
                values: right,
                precision: right_precision,
                scale: right_scale,
            },
        ) if precision == right_precision && scale == right_scale => {
            Ok(EvaluatedScalar::Decimal128 {
                values: coalesce_options(left, right),
                precision,
                scale,
            })
        }
        (EvaluatedScalar::Int64(left), EvaluatedScalar::Int64(right)) => {
            Ok(EvaluatedScalar::Int64(coalesce_options(left, right)))
        }
        (EvaluatedScalar::Boolean(left), EvaluatedScalar::Boolean(right)) => {
            Ok(EvaluatedScalar::Boolean(coalesce_options(left, right)))
        }
        (EvaluatedScalar::Date32(left), EvaluatedScalar::Date32(right)) => {
            Ok(EvaluatedScalar::Date32(coalesce_options(left, right)))
        }
        (
            EvaluatedScalar::TimestampMillisecond(left),
            EvaluatedScalar::TimestampMillisecond(right),
        ) => Ok(EvaluatedScalar::TimestampMillisecond(coalesce_options(
            left, right,
        ))),
        (left, right) => Err(DodamError::UnsupportedSql(format!(
            "cannot COALESCE {} and {}",
            left.data_type(),
            right.data_type()
        ))),
    }
}

pub(super) fn scalar_as_utf8(value: EvaluatedScalar) -> Result<Vec<Option<String>>> {
    Ok(match value {
        EvaluatedScalar::Array(array) => return scalar_as_utf8(evaluated_array(array.as_ref())?),
        EvaluatedScalar::Utf8(values) => values,
        EvaluatedScalar::Int64(values) => values
            .into_iter()
            .map(|value| value.map(|value| value.to_string()))
            .collect(),
        EvaluatedScalar::Float64(values) => values
            .into_iter()
            .map(|value| value.map(format_f64_for_sql_varchar))
            .collect(),
        EvaluatedScalar::Decimal128 { values, scale, .. } => values
            .into_iter()
            .map(|value| value.map(|value| format_decimal128_value(value, scale)))
            .collect(),
        EvaluatedScalar::Boolean(values) => values
            .into_iter()
            .map(|value| value.map(|value| value.to_string()))
            .collect(),
        EvaluatedScalar::Date32(values) => values
            .into_iter()
            .map(|value| value.map(format_date32_days))
            .collect(),
        EvaluatedScalar::TimestampMillisecond(values) => values
            .into_iter()
            .map(|value| value.map(format_timestamp_millis))
            .collect(),
    })
}
