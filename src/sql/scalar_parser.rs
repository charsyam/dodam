use super::*;

pub(super) fn parse_scalar_function_projection(
    function: &sqlparser::ast::Function,
    alias: Option<&str>,
    table_alias: Option<&str>,
) -> Result<Option<ProjectionExpression>> {
    let name = object_name_to_string(&function.name)?;
    let lowercase_name = name.to_ascii_lowercase();
    if !matches!(
        lowercase_name.as_str(),
        "coalesce"
            | "lower"
            | "upper"
            | "length"
            | "trim"
            | "abs"
            | "round"
            | "floor"
            | "ceil"
            | "replace"
            | "concat"
            | "array_length"
            | "list_length"
    ) {
        return Ok(None);
    }
    if function.filter.is_some()
        || function.over.is_some()
        || !function.within_group.is_empty()
        || function.null_treatment.is_some()
        || !matches!(function.parameters, FunctionArguments::None)
    {
        return Err(DodamError::UnsupportedSql(
            "scalar function filters, windows, within group, null treatment, and parameters are not supported"
                .to_string(),
        ));
    }
    let FunctionArguments::List(args) = &function.args else {
        return Err(DodamError::UnsupportedSql(format!(
            "unsupported scalar function arguments: {}",
            function.args
        )));
    };
    if !args.clauses.is_empty() || args.duplicate_treatment.is_some() {
        return Err(DodamError::UnsupportedSql(format!(
            "unsupported scalar function arguments: {}",
            function.args
        )));
    }
    let values = args
        .args
        .iter()
        .map(|arg| match arg {
            FunctionArg::Unnamed(FunctionArgExpr::Expr(expr)) => {
                parse_scalar_sql_expression(expr, table_alias)
            }
            _ => Err(DodamError::UnsupportedSql(format!(
                "unsupported scalar function argument: {arg}"
            ))),
        })
        .collect::<Result<Vec<_>>>()?;
    let expr = scalar_function_expression(&name, &lowercase_name, values)?;
    Ok(Some(ProjectionExpression {
        output_name: alias.map_or_else(|| function.to_string(), ToString::to_string),
        expr,
    }))
}

pub(super) fn parse_join_scalar_function_projection(
    function: &sqlparser::ast::Function,
    alias: Option<&str>,
    table_aliases: &[&str],
) -> Result<Option<ProjectionExpression>> {
    let name = object_name_to_string(&function.name)?;
    let lowercase_name = name.to_ascii_lowercase();
    if !matches!(
        lowercase_name.as_str(),
        "coalesce"
            | "lower"
            | "upper"
            | "length"
            | "trim"
            | "abs"
            | "round"
            | "floor"
            | "ceil"
            | "replace"
            | "concat"
            | "array_length"
            | "list_length"
    ) {
        return Ok(None);
    }
    if function.filter.is_some()
        || function.over.is_some()
        || !function.within_group.is_empty()
        || function.null_treatment.is_some()
        || !matches!(function.parameters, FunctionArguments::None)
    {
        return Err(DodamError::UnsupportedSql(
            "scalar function filters, windows, within group, null treatment, and parameters are not supported"
                .to_string(),
        ));
    }
    let FunctionArguments::List(args) = &function.args else {
        return Err(DodamError::UnsupportedSql(format!(
            "unsupported scalar function arguments: {}",
            function.args
        )));
    };
    if !args.clauses.is_empty() || args.duplicate_treatment.is_some() {
        return Err(DodamError::UnsupportedSql(format!(
            "unsupported scalar function arguments: {}",
            function.args
        )));
    }
    let values = args
        .args
        .iter()
        .map(|arg| match arg {
            FunctionArg::Unnamed(FunctionArgExpr::Expr(expr)) => {
                parse_join_scalar_sql_expression(expr, table_aliases)
            }
            _ => Err(DodamError::UnsupportedSql(format!(
                "unsupported scalar function argument: {arg}"
            ))),
        })
        .collect::<Result<Vec<_>>>()?;
    let expr = scalar_function_expression(&name, &lowercase_name, values)?;
    Ok(Some(ProjectionExpression {
        output_name: alias.map_or_else(|| function.to_string(), ToString::to_string),
        expr,
    }))
}

fn scalar_function_expression(
    name: &str,
    lowercase_name: &str,
    values: Vec<ScalarSqlExpression>,
) -> Result<ScalarSqlExpression> {
    Ok(match lowercase_name {
        "coalesce" => {
            if values.is_empty() {
                return Err(DodamError::UnsupportedSql(
                    "COALESCE requires at least one argument".to_string(),
                ));
            }
            ScalarSqlExpression::Coalesce(values)
        }
        "concat" => {
            if values.is_empty() {
                return Err(DodamError::UnsupportedSql(
                    "CONCAT requires at least one argument".to_string(),
                ));
            }
            ScalarSqlExpression::Concat(values)
        }
        "replace" => {
            let [expr, from, to] = values.as_slice() else {
                return Err(DodamError::UnsupportedSql(
                    "REPLACE requires exactly three arguments".to_string(),
                ));
            };
            ScalarSqlExpression::Replace {
                expr: Box::new(expr.clone()),
                from: Box::new(from.clone()),
                to: Box::new(to.clone()),
            }
        }
        "lower" | "upper" | "length" | "trim" | "abs" | "round" | "floor" | "ceil"
        | "array_length" | "list_length" => {
            let [value] = values.as_slice() else {
                return Err(DodamError::UnsupportedSql(format!(
                    "{name} requires exactly one argument"
                )));
            };
            match lowercase_name {
                "lower" => ScalarSqlExpression::Lower(Box::new(value.clone())),
                "upper" => ScalarSqlExpression::Upper(Box::new(value.clone())),
                "length" => ScalarSqlExpression::Length(Box::new(value.clone())),
                "trim" => ScalarSqlExpression::Trim(Box::new(value.clone())),
                "abs" => ScalarSqlExpression::Abs(Box::new(value.clone())),
                "round" => ScalarSqlExpression::Round(Box::new(value.clone())),
                "floor" => ScalarSqlExpression::Floor(Box::new(value.clone())),
                "ceil" => ScalarSqlExpression::Ceil(Box::new(value.clone())),
                "array_length" | "list_length" => {
                    let (column, field) = match value {
                        ScalarSqlExpression::Column(column) => (column.clone(), None),
                        ScalarSqlExpression::StructField { column, field } => {
                            (column.clone(), Some(field.clone()))
                        }
                        _ => {
                            return Err(DodamError::UnsupportedSql(format!(
                                "{name} currently requires a list column or struct list field"
                            )));
                        }
                    };
                    ScalarSqlExpression::ListLength { column, field }
                }
                _ => unreachable!("validated scalar function"),
            }
        }
        _ => unreachable!("validated scalar function"),
    })
}

pub(super) fn parse_scalar_sql_expression(
    expr: &SqlExpr,
    table_alias: Option<&str>,
) -> Result<ScalarSqlExpression> {
    match expr {
        SqlExpr::CompoundFieldAccess { .. } => {
            let Some((column, field, index)) = parse_list_index_access(expr, table_alias)? else {
                return Err(DodamError::UnsupportedSql(format!(
                    "unsupported nested/list expression: {expr}"
                )));
            };
            Ok(ScalarSqlExpression::ListIndex {
                column,
                field,
                index: Box::new(index),
            })
        }
        SqlExpr::Identifier(_) | SqlExpr::CompoundIdentifier(_) => {
            if let Some((column, field)) = parse_struct_field_access(expr, table_alias)? {
                Ok(ScalarSqlExpression::StructField { column, field })
            } else {
                Ok(ScalarSqlExpression::Column(sql_column_name(
                    expr,
                    table_alias,
                )?))
            }
        }
        SqlExpr::Value(_) => Ok(ScalarSqlExpression::Literal(sql_literal_value(expr)?)),
        SqlExpr::TypedString(typed) => typed_string_scalar_expression(typed),
        SqlExpr::UnaryOp { op, expr }
            if matches!(op, UnaryOperator::Minus | UnaryOperator::Plus)
                && sql_literal_value(expr).is_ok() =>
        {
            Ok(ScalarSqlExpression::Literal(sql_literal_value(
                &SqlExpr::UnaryOp {
                    op: *op,
                    expr: expr.clone(),
                },
            )?))
        }
        SqlExpr::Nested(expr) => parse_scalar_sql_expression(expr, table_alias),
        SqlExpr::BinaryOp { left, op, right }
            if matches!(
                op,
                BinaryOperator::Plus
                    | BinaryOperator::Minus
                    | BinaryOperator::Multiply
                    | BinaryOperator::Divide
            ) =>
        {
            if let Ok(value) = sql_literal_value(expr) {
                return Ok(ScalarSqlExpression::Literal(value));
            }
            Ok(ScalarSqlExpression::Binary {
                left: Box::new(parse_scalar_sql_expression(left, table_alias)?),
                op: op.clone(),
                right: Box::new(parse_scalar_sql_expression(right, table_alias)?),
            })
        }
        SqlExpr::Cast {
            expr, data_type, ..
        } => Ok(ScalarSqlExpression::Cast {
            expr: Box::new(parse_scalar_sql_expression(expr, table_alias)?),
            target: data_type.to_string(),
        }),
        SqlExpr::Function(function) => {
            let Some(projection) = parse_scalar_function_projection(function, None, table_alias)?
            else {
                return Err(DodamError::UnsupportedSql(format!(
                    "unsupported scalar function: {function}"
                )));
            };
            Ok(projection.expr)
        }
        SqlExpr::Trim {
            trim_where,
            trim_what,
            expr,
            trim_characters,
        } => {
            if trim_where.is_some() || trim_what.is_some() || trim_characters.is_some() {
                return Err(DodamError::UnsupportedSql(
                    "TRIM variants with trim characters or direction are not supported yet"
                        .to_string(),
                ));
            }
            Ok(ScalarSqlExpression::Trim(Box::new(
                parse_scalar_sql_expression(expr, table_alias)?,
            )))
        }
        SqlExpr::Floor { expr, field } => parse_ceil_floor_scalar_expression(
            "FLOOR",
            expr,
            field,
            table_alias,
            parse_scalar_sql_expression,
            ScalarSqlExpression::Floor,
        ),
        SqlExpr::Ceil { expr, field } => parse_ceil_floor_scalar_expression(
            "CEIL",
            expr,
            field,
            table_alias,
            parse_scalar_sql_expression,
            ScalarSqlExpression::Ceil,
        ),
        SqlExpr::Substring {
            expr,
            substring_from,
            substring_for,
            ..
        } => {
            let Some(start) = substring_from else {
                return Err(DodamError::UnsupportedSql(
                    "SUBSTRING requires a FROM/start expression".to_string(),
                ));
            };
            Ok(ScalarSqlExpression::Substring {
                expr: Box::new(parse_scalar_sql_expression(expr, table_alias)?),
                start: Box::new(parse_scalar_sql_expression(start, table_alias)?),
                length: substring_for
                    .as_ref()
                    .map(|expr| parse_scalar_sql_expression(expr, table_alias).map(Box::new))
                    .transpose()?,
            })
        }
        SqlExpr::Extract { field, expr, .. } if *field == DateTimeField::Year => {
            Ok(ScalarSqlExpression::ExtractYear(Box::new(
                parse_scalar_sql_expression(expr, table_alias)?,
            )))
        }
        SqlExpr::Case {
            operand,
            conditions,
            else_result,
            ..
        } => {
            let rewritten_conditions = case_conditions_from_operand(operand.as_deref(), conditions);
            Ok(ScalarSqlExpression::Case {
                conditions: rewritten_conditions,
                results: conditions
                    .iter()
                    .map(|when| parse_scalar_sql_expression(&when.result, table_alias))
                    .collect::<Result<Vec<_>>>()?,
                else_result: else_result
                    .as_ref()
                    .map(|expr| parse_scalar_sql_expression(expr, table_alias).map(Box::new))
                    .transpose()?,
            })
        }
        _ => Err(DodamError::UnsupportedSql(format!(
            "unsupported SELECT expression: {expr}"
        ))),
    }
}

pub(super) fn parse_join_scalar_sql_expression(
    expr: &SqlExpr,
    table_aliases: &[&str],
) -> Result<ScalarSqlExpression> {
    match expr {
        SqlExpr::CompoundFieldAccess { .. } => {
            let Some((column, field, index)) = parse_join_list_index_access(expr, table_aliases)?
            else {
                return Err(DodamError::UnsupportedSql(format!(
                    "unsupported JOIN nested/list expression: {expr}"
                )));
            };
            Ok(ScalarSqlExpression::ListIndex {
                column,
                field,
                index: Box::new(index),
            })
        }
        SqlExpr::Identifier(_) | SqlExpr::CompoundIdentifier(_) => {
            if let Some((column, field)) = parse_join_struct_field_access(expr, table_aliases)? {
                Ok(ScalarSqlExpression::StructField { column, field })
            } else {
                Ok(ScalarSqlExpression::Column(join_column_name(
                    expr,
                    table_aliases,
                )?))
            }
        }
        SqlExpr::Value(_) => Ok(ScalarSqlExpression::Literal(sql_literal_value(expr)?)),
        SqlExpr::TypedString(typed) => typed_string_scalar_expression(typed),
        SqlExpr::Nested(expr) => parse_join_scalar_sql_expression(expr, table_aliases),
        SqlExpr::BinaryOp { left, op, right }
            if matches!(
                op,
                BinaryOperator::Plus
                    | BinaryOperator::Minus
                    | BinaryOperator::Multiply
                    | BinaryOperator::Divide
            ) =>
        {
            if let Ok(value) = sql_literal_value(expr) {
                return Ok(ScalarSqlExpression::Literal(value));
            }
            Ok(ScalarSqlExpression::Binary {
                left: Box::new(parse_join_scalar_sql_expression(left, table_aliases)?),
                op: op.clone(),
                right: Box::new(parse_join_scalar_sql_expression(right, table_aliases)?),
            })
        }
        SqlExpr::Cast {
            expr, data_type, ..
        } => Ok(ScalarSqlExpression::Cast {
            expr: Box::new(parse_join_scalar_sql_expression(expr, table_aliases)?),
            target: data_type.to_string(),
        }),
        SqlExpr::Function(function) => {
            let Some(projection) =
                parse_join_scalar_function_projection(function, None, table_aliases)?
            else {
                return Err(DodamError::UnsupportedSql(format!(
                    "unsupported JOIN scalar function: {function}"
                )));
            };
            Ok(projection.expr)
        }
        SqlExpr::Trim {
            trim_where,
            trim_what,
            expr,
            trim_characters,
        } => {
            if trim_where.is_some() || trim_what.is_some() || trim_characters.is_some() {
                return Err(DodamError::UnsupportedSql(
                    "TRIM variants with trim characters or direction are not supported yet"
                        .to_string(),
                ));
            }
            Ok(ScalarSqlExpression::Trim(Box::new(
                parse_join_scalar_sql_expression(expr, table_aliases)?,
            )))
        }
        SqlExpr::Floor { expr, field } => parse_ceil_floor_scalar_expression(
            "FLOOR",
            expr,
            field,
            table_aliases,
            parse_join_scalar_sql_expression,
            ScalarSqlExpression::Floor,
        ),
        SqlExpr::Ceil { expr, field } => parse_ceil_floor_scalar_expression(
            "CEIL",
            expr,
            field,
            table_aliases,
            parse_join_scalar_sql_expression,
            ScalarSqlExpression::Ceil,
        ),
        SqlExpr::Substring {
            expr,
            substring_from,
            substring_for,
            ..
        } => {
            let Some(start) = substring_from else {
                return Err(DodamError::UnsupportedSql(
                    "SUBSTRING requires a FROM/start expression".to_string(),
                ));
            };
            Ok(ScalarSqlExpression::Substring {
                expr: Box::new(parse_join_scalar_sql_expression(expr, table_aliases)?),
                start: Box::new(parse_join_scalar_sql_expression(start, table_aliases)?),
                length: substring_for
                    .as_ref()
                    .map(|expr| parse_join_scalar_sql_expression(expr, table_aliases).map(Box::new))
                    .transpose()?,
            })
        }
        SqlExpr::Extract { field, expr, .. } if *field == DateTimeField::Year => {
            Ok(ScalarSqlExpression::ExtractYear(Box::new(
                parse_join_scalar_sql_expression(expr, table_aliases)?,
            )))
        }
        SqlExpr::Case {
            operand,
            conditions,
            else_result,
            ..
        } => {
            let rewritten_conditions = case_conditions_from_operand(operand.as_deref(), conditions)
                .into_iter()
                .map(|condition| rewrite_join_scalar_predicate(&condition, table_aliases))
                .collect::<Result<Vec<_>>>()?;
            Ok(ScalarSqlExpression::Case {
                conditions: rewritten_conditions,
                results: conditions
                    .iter()
                    .map(|when| parse_join_scalar_sql_expression(&when.result, table_aliases))
                    .collect::<Result<Vec<_>>>()?,
                else_result: else_result
                    .as_ref()
                    .map(|expr| parse_join_scalar_sql_expression(expr, table_aliases).map(Box::new))
                    .transpose()?,
            })
        }
        _ => Err(DodamError::UnsupportedSql(format!(
            "unsupported JOIN scalar expression: {expr}"
        ))),
    }
}

pub(super) fn case_conditions_from_operand(
    operand: Option<&SqlExpr>,
    conditions: &[sqlparser::ast::CaseWhen],
) -> Vec<SqlExpr> {
    match operand {
        Some(operand) => conditions
            .iter()
            .map(|when| SqlExpr::BinaryOp {
                left: Box::new(operand.clone()),
                op: BinaryOperator::Eq,
                right: Box::new(when.condition.clone()),
            })
            .collect(),
        None => conditions
            .iter()
            .map(|when| when.condition.clone())
            .collect(),
    }
}

fn parse_ceil_floor_scalar_expression<C>(
    name: &str,
    expr: &SqlExpr,
    field: &CeilFloorKind,
    context: C,
    parser: fn(&SqlExpr, C) -> Result<ScalarSqlExpression>,
    builder: fn(Box<ScalarSqlExpression>) -> ScalarSqlExpression,
) -> Result<ScalarSqlExpression>
where
    C: Copy,
{
    if !matches!(
        field,
        CeilFloorKind::DateTimeField(DateTimeField::NoDateTime)
    ) {
        return Err(DodamError::UnsupportedSql(format!(
            "{name} scale/date-time variants are not supported yet"
        )));
    }
    Ok(builder(Box::new(parser(expr, context)?)))
}

pub(super) fn parse_join_struct_field_access(
    expr: &SqlExpr,
    table_aliases: &[&str],
) -> Result<Option<(String, String)>> {
    let SqlExpr::CompoundIdentifier(parts) = expr else {
        return Ok(None);
    };
    match parts.as_slice() {
        [qualifier, column, fields @ ..]
            if !fields.is_empty()
                && table_aliases
                    .iter()
                    .any(|alias| alias.eq_ignore_ascii_case(&qualifier.value)) =>
        {
            Ok(Some((
                format!("{}.{}", qualifier.value, column.value),
                fields
                    .iter()
                    .map(|field| field.value.as_str())
                    .collect::<Vec<_>>()
                    .join("."),
            )))
        }
        [qualifier, _, fields @ ..] if !fields.is_empty() => {
            Err(DodamError::UnknownTableQualifier(qualifier.value.clone()))
        }
        _ => Ok(None),
    }
}

fn parse_join_list_index_access(
    expr: &SqlExpr,
    table_aliases: &[&str],
) -> Result<Option<(String, Option<String>, ScalarSqlExpression)>> {
    let SqlExpr::CompoundFieldAccess { root, access_chain } = expr else {
        return Ok(None);
    };
    let Some(AccessExpr::Subscript(Subscript::Index { index })) = access_chain.last() else {
        return Ok(None);
    };
    let prefix = &access_chain[..access_chain.len().saturating_sub(1)];
    let (column, field) = if prefix.is_empty() {
        if let Some((column, field)) = parse_join_struct_field_access(root, table_aliases)? {
            (column, Some(field))
        } else {
            (join_column_name(root, table_aliases)?, None)
        }
    } else if let Some((column, mut field)) = parse_join_struct_field_access(root, table_aliases)? {
        for access in prefix {
            let AccessExpr::Dot(SqlExpr::Identifier(ident)) = access else {
                return Ok(None);
            };
            field.push('.');
            field.push_str(&ident.value);
        }
        (column, Some(field))
    } else {
        if let SqlExpr::Identifier(alias) = root.as_ref()
            && table_aliases
                .iter()
                .any(|table_alias| table_alias.eq_ignore_ascii_case(&alias.value))
            && let Some((first, rest)) = prefix.split_first()
        {
            let AccessExpr::Dot(SqlExpr::Identifier(column)) = first else {
                return Ok(None);
            };
            let mut fields = Vec::with_capacity(rest.len());
            for access in rest {
                let AccessExpr::Dot(SqlExpr::Identifier(ident)) = access else {
                    return Ok(None);
                };
                fields.push(ident.value.as_str());
            }
            return Ok(Some((
                format!("{}.{}", alias.value, column.value),
                (!fields.is_empty()).then(|| fields.join(".")),
                parse_join_scalar_sql_expression(index, table_aliases)?,
            )));
        }
        let column = join_column_name(root, table_aliases)?;
        let mut fields = Vec::with_capacity(prefix.len());
        for access in prefix {
            let AccessExpr::Dot(SqlExpr::Identifier(ident)) = access else {
                return Ok(None);
            };
            fields.push(ident.value.as_str());
        }
        if fields.is_empty() {
            (column, None)
        } else {
            (column, Some(fields.join(".")))
        }
    };
    let index = parse_join_scalar_sql_expression(index, table_aliases)?;
    Ok(Some((column, field, index)))
}

pub(super) fn parse_struct_field_access(
    expr: &SqlExpr,
    table_alias: Option<&str>,
) -> Result<Option<(String, String)>> {
    let SqlExpr::CompoundIdentifier(parts) = expr else {
        return Ok(None);
    };
    match (table_alias, parts.as_slice()) {
        (None, [column, field]) => Ok(Some((column.value.clone(), field.value.clone()))),
        (None, [column, fields @ ..]) if fields.len() >= 2 => Ok(Some((
            column.value.clone(),
            fields
                .iter()
                .map(|field| field.value.as_str())
                .collect::<Vec<_>>()
                .join("."),
        ))),
        (Some(alias), [qualifier, column, field]) if qualifier.value == alias => {
            Ok(Some((column.value.clone(), field.value.clone())))
        }
        (Some(alias), [qualifier, column, fields @ ..])
            if qualifier.value == alias && fields.len() >= 2 =>
        {
            Ok(Some((
                column.value.clone(),
                fields
                    .iter()
                    .map(|field| field.value.as_str())
                    .collect::<Vec<_>>()
                    .join("."),
            )))
        }
        _ => Ok(None),
    }
}

fn parse_list_index_access(
    expr: &SqlExpr,
    table_alias: Option<&str>,
) -> Result<Option<(String, Option<String>, ScalarSqlExpression)>> {
    let SqlExpr::CompoundFieldAccess { root, access_chain } = expr else {
        return Ok(None);
    };
    let Some(AccessExpr::Subscript(Subscript::Index { index })) = access_chain.last() else {
        return Ok(None);
    };
    let prefix = &access_chain[..access_chain.len().saturating_sub(1)];
    let (column, field) = if prefix.is_empty() {
        if let Some((column, field)) = parse_struct_field_access(root, table_alias)? {
            (column, Some(field))
        } else {
            (sql_column_name(root, table_alias)?, None)
        }
    } else if let Some((column, mut field)) = parse_struct_field_access(root, table_alias)? {
        for access in prefix {
            let AccessExpr::Dot(SqlExpr::Identifier(ident)) = access else {
                return Ok(None);
            };
            field.push('.');
            field.push_str(&ident.value);
        }
        (column, Some(field))
    } else {
        let column = sql_column_name(root, table_alias)?;
        let mut fields = Vec::with_capacity(prefix.len());
        for access in prefix {
            let AccessExpr::Dot(SqlExpr::Identifier(ident)) = access else {
                return Ok(None);
            };
            fields.push(ident.value.as_str());
        }
        if fields.is_empty() {
            (column, None)
        } else {
            (column, Some(fields.join(".")))
        }
    };
    let index = parse_scalar_sql_expression(index, table_alias)?;
    Ok(Some((column, field, index)))
}

fn typed_string_scalar_expression(
    typed: &sqlparser::ast::TypedString,
) -> Result<ScalarSqlExpression> {
    let value = match &typed.value.value {
        Value::SingleQuotedString(value) | Value::DoubleQuotedString(value) => value.clone(),
        value => {
            return Err(DodamError::UnsupportedSql(format!(
                "unsupported typed string literal: {value}"
            )));
        }
    };
    Ok(ScalarSqlExpression::Cast {
        expr: Box::new(ScalarSqlExpression::Literal(LiteralValue::Utf8(value))),
        target: typed.data_type.to_string(),
    })
}

pub(super) fn rewrite_join_scalar_predicate(
    expr: &SqlExpr,
    table_aliases: &[&str],
) -> Result<SqlExpr> {
    match expr {
        SqlExpr::Identifier(_) | SqlExpr::CompoundIdentifier(_) => {
            let column = join_column_name(expr, table_aliases)?;
            Ok(sql_column_expr(&column))
        }
        SqlExpr::BinaryOp { left, op, right } => Ok(SqlExpr::BinaryOp {
            left: Box::new(rewrite_join_scalar_predicate(left, table_aliases)?),
            op: op.clone(),
            right: Box::new(rewrite_join_scalar_predicate(right, table_aliases)?),
        }),
        SqlExpr::Nested(expr) => Ok(SqlExpr::Nested(Box::new(rewrite_join_scalar_predicate(
            expr,
            table_aliases,
        )?))),
        SqlExpr::UnaryOp { op, expr } => Ok(SqlExpr::UnaryOp {
            op: op.clone(),
            expr: Box::new(rewrite_join_scalar_predicate(expr, table_aliases)?),
        }),
        SqlExpr::IsNull(expr) => Ok(SqlExpr::IsNull(Box::new(rewrite_join_scalar_predicate(
            expr,
            table_aliases,
        )?))),
        SqlExpr::IsNotNull(expr) => Ok(SqlExpr::IsNotNull(Box::new(
            rewrite_join_scalar_predicate(expr, table_aliases)?,
        ))),
        SqlExpr::Like {
            negated,
            any,
            expr,
            pattern,
            escape_char,
        } => Ok(SqlExpr::Like {
            negated: *negated,
            any: *any,
            expr: Box::new(rewrite_join_scalar_predicate(expr, table_aliases)?),
            pattern: Box::new(rewrite_join_scalar_predicate(pattern, table_aliases)?),
            escape_char: escape_char.clone(),
        }),
        SqlExpr::ILike {
            negated,
            any,
            expr,
            pattern,
            escape_char,
        } => Ok(SqlExpr::ILike {
            negated: *negated,
            any: *any,
            expr: Box::new(rewrite_join_scalar_predicate(expr, table_aliases)?),
            pattern: Box::new(rewrite_join_scalar_predicate(pattern, table_aliases)?),
            escape_char: escape_char.clone(),
        }),
        SqlExpr::Value(_) => Ok(expr.clone()),
        _ => Ok(expr.clone()),
    }
}

pub(super) fn sql_column_expr(column: &str) -> SqlExpr {
    SqlExpr::Identifier(sqlparser::ast::Ident::new(column))
}

pub(super) fn scalar_expression_columns(expr: &ScalarSqlExpression) -> Vec<String> {
    let mut columns = Vec::new();
    collect_scalar_expression_columns(expr, &mut columns);
    columns
}

pub(super) fn join_scalar_expression_columns(
    expr: &ScalarSqlExpression,
    table_aliases: &[&str],
) -> Result<Vec<String>> {
    let mut columns = Vec::new();
    collect_join_scalar_expression_columns(expr, table_aliases, &mut columns)?;
    Ok(columns)
}

pub(super) fn join_sql_expression_columns(
    expr: &SqlExpr,
    table_aliases: &[&str],
) -> Result<Vec<String>> {
    let mut columns = Vec::new();
    collect_join_column_candidates(expr, table_aliases, &mut columns)?;
    Ok(columns)
}

fn collect_join_scalar_expression_columns(
    expr: &ScalarSqlExpression,
    table_aliases: &[&str],
    columns: &mut Vec<String>,
) -> Result<()> {
    match expr {
        ScalarSqlExpression::Column(column) => add_column_once(columns, column.clone()),
        ScalarSqlExpression::StructField { column, .. } => add_column_once(columns, column.clone()),
        ScalarSqlExpression::ListIndex { column, index, .. } => {
            add_column_once(columns, column.clone());
            collect_join_scalar_expression_columns(index, table_aliases, columns)?;
        }
        ScalarSqlExpression::ListLength { column, .. } => add_column_once(columns, column.clone()),
        ScalarSqlExpression::Literal(_) => {}
        ScalarSqlExpression::Binary { left, right, .. } => {
            collect_join_scalar_expression_columns(left, table_aliases, columns)?;
            collect_join_scalar_expression_columns(right, table_aliases, columns)?;
        }
        ScalarSqlExpression::Cast { expr, .. } => {
            collect_join_scalar_expression_columns(expr, table_aliases, columns)?;
        }
        ScalarSqlExpression::Coalesce(values) => {
            for value in values {
                collect_join_scalar_expression_columns(value, table_aliases, columns)?;
            }
        }
        ScalarSqlExpression::Concat(values) => {
            for value in values {
                collect_join_scalar_expression_columns(value, table_aliases, columns)?;
            }
        }
        ScalarSqlExpression::Lower(expr)
        | ScalarSqlExpression::Upper(expr)
        | ScalarSqlExpression::Length(expr)
        | ScalarSqlExpression::Trim(expr)
        | ScalarSqlExpression::Abs(expr)
        | ScalarSqlExpression::Round(expr)
        | ScalarSqlExpression::Floor(expr)
        | ScalarSqlExpression::Ceil(expr)
        | ScalarSqlExpression::ExtractYear(expr) => {
            collect_join_scalar_expression_columns(expr, table_aliases, columns)?;
        }
        ScalarSqlExpression::Replace { expr, from, to } => {
            collect_join_scalar_expression_columns(expr, table_aliases, columns)?;
            collect_join_scalar_expression_columns(from, table_aliases, columns)?;
            collect_join_scalar_expression_columns(to, table_aliases, columns)?;
        }
        ScalarSqlExpression::Substring {
            expr,
            start,
            length,
        } => {
            collect_join_scalar_expression_columns(expr, table_aliases, columns)?;
            collect_join_scalar_expression_columns(start, table_aliases, columns)?;
            if let Some(length) = length {
                collect_join_scalar_expression_columns(length, table_aliases, columns)?;
            }
        }
        ScalarSqlExpression::Case {
            conditions,
            results,
            else_result,
        } => {
            for condition in conditions {
                collect_join_predicate_columns(condition, table_aliases, columns)?;
            }
            for result in results {
                collect_join_scalar_expression_columns(result, table_aliases, columns)?;
            }
            if let Some(else_result) = else_result {
                collect_join_scalar_expression_columns(else_result, table_aliases, columns)?;
            }
        }
    }
    Ok(())
}

fn collect_join_predicate_columns(
    expr: &SqlExpr,
    table_aliases: &[&str],
    columns: &mut Vec<String>,
) -> Result<()> {
    match expr {
        SqlExpr::Identifier(_) | SqlExpr::CompoundIdentifier(_) => {
            add_column_once(columns, join_column_name(expr, table_aliases)?);
        }
        SqlExpr::BinaryOp { left, right, .. } => {
            collect_join_predicate_columns(left, table_aliases, columns)?;
            collect_join_predicate_columns(right, table_aliases, columns)?;
        }
        SqlExpr::UnaryOp { expr, .. }
        | SqlExpr::Nested(expr)
        | SqlExpr::IsNull(expr)
        | SqlExpr::IsNotNull(expr) => {
            collect_join_predicate_columns(expr, table_aliases, columns)?;
        }
        SqlExpr::Like { expr, pattern, .. } | SqlExpr::ILike { expr, pattern, .. } => {
            collect_join_predicate_columns(expr, table_aliases, columns)?;
            collect_join_predicate_columns(pattern, table_aliases, columns)?;
        }
        SqlExpr::Value(_) => {}
        _ => {}
    }
    Ok(())
}

fn collect_scalar_expression_columns(expr: &ScalarSqlExpression, columns: &mut Vec<String>) {
    match expr {
        ScalarSqlExpression::Column(column) => add_column_once(columns, column.clone()),
        ScalarSqlExpression::StructField { column, .. } => add_column_once(columns, column.clone()),
        ScalarSqlExpression::ListIndex { column, index, .. } => {
            add_column_once(columns, column.clone());
            collect_scalar_expression_columns(index, columns);
        }
        ScalarSqlExpression::ListLength { column, .. } => add_column_once(columns, column.clone()),
        ScalarSqlExpression::Literal(_) => {}
        ScalarSqlExpression::Binary { left, right, .. } => {
            collect_scalar_expression_columns(left, columns);
            collect_scalar_expression_columns(right, columns);
        }
        ScalarSqlExpression::Cast { expr, .. } => collect_scalar_expression_columns(expr, columns),
        ScalarSqlExpression::Coalesce(values) => {
            for value in values {
                collect_scalar_expression_columns(value, columns);
            }
        }
        ScalarSqlExpression::Concat(values) => {
            for value in values {
                collect_scalar_expression_columns(value, columns);
            }
        }
        ScalarSqlExpression::Lower(expr)
        | ScalarSqlExpression::Upper(expr)
        | ScalarSqlExpression::Length(expr)
        | ScalarSqlExpression::Trim(expr)
        | ScalarSqlExpression::Abs(expr)
        | ScalarSqlExpression::Round(expr)
        | ScalarSqlExpression::Floor(expr)
        | ScalarSqlExpression::Ceil(expr)
        | ScalarSqlExpression::ExtractYear(expr) => {
            collect_scalar_expression_columns(expr, columns)
        }
        ScalarSqlExpression::Replace { expr, from, to } => {
            collect_scalar_expression_columns(expr, columns);
            collect_scalar_expression_columns(from, columns);
            collect_scalar_expression_columns(to, columns);
        }
        ScalarSqlExpression::Substring {
            expr,
            start,
            length,
        } => {
            collect_scalar_expression_columns(expr, columns);
            collect_scalar_expression_columns(start, columns);
            if let Some(length) = length {
                collect_scalar_expression_columns(length, columns);
            }
        }
        ScalarSqlExpression::Case {
            conditions,
            results,
            else_result,
        } => {
            for condition in conditions {
                let _ = collect_predicate_expression_columns(condition, None, columns);
            }
            for result in results {
                collect_scalar_expression_columns(result, columns);
            }
            if let Some(else_result) = else_result {
                collect_scalar_expression_columns(else_result, columns);
            }
        }
    }
}

pub(super) fn scalar_expression_references_aggregate(
    expr: &ScalarSqlExpression,
    aggregates: &[AggregateExpr],
) -> bool {
    match expr {
        ScalarSqlExpression::Column(column) => aggregates
            .iter()
            .any(|aggregate| column == &aggregate.to_string()),
        ScalarSqlExpression::StructField { .. }
        | ScalarSqlExpression::ListIndex { .. }
        | ScalarSqlExpression::ListLength { .. } => false,
        ScalarSqlExpression::Literal(_) => false,
        ScalarSqlExpression::Binary { left, right, .. } => {
            scalar_expression_references_aggregate(left, aggregates)
                || scalar_expression_references_aggregate(right, aggregates)
        }
        ScalarSqlExpression::Cast { expr, .. }
        | ScalarSqlExpression::Lower(expr)
        | ScalarSqlExpression::Upper(expr)
        | ScalarSqlExpression::Length(expr)
        | ScalarSqlExpression::Trim(expr)
        | ScalarSqlExpression::Abs(expr)
        | ScalarSqlExpression::Round(expr)
        | ScalarSqlExpression::Floor(expr)
        | ScalarSqlExpression::Ceil(expr)
        | ScalarSqlExpression::ExtractYear(expr) => {
            scalar_expression_references_aggregate(expr, aggregates)
        }
        ScalarSqlExpression::Coalesce(values) => values
            .iter()
            .any(|value| scalar_expression_references_aggregate(value, aggregates)),
        ScalarSqlExpression::Concat(values) => values
            .iter()
            .any(|value| scalar_expression_references_aggregate(value, aggregates)),
        ScalarSqlExpression::Replace { expr, from, to } => {
            scalar_expression_references_aggregate(expr, aggregates)
                || scalar_expression_references_aggregate(from, aggregates)
                || scalar_expression_references_aggregate(to, aggregates)
        }
        ScalarSqlExpression::Substring {
            expr,
            start,
            length,
        } => {
            scalar_expression_references_aggregate(expr, aggregates)
                || scalar_expression_references_aggregate(start, aggregates)
                || length.as_deref().is_some_and(|length| {
                    scalar_expression_references_aggregate(length, aggregates)
                })
        }
        ScalarSqlExpression::Case {
            results,
            else_result,
            ..
        } => {
            results
                .iter()
                .any(|result| scalar_expression_references_aggregate(result, aggregates))
                || else_result.as_deref().is_some_and(|else_result| {
                    scalar_expression_references_aggregate(else_result, aggregates)
                })
        }
    }
}
