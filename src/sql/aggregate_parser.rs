use super::*;

pub(super) fn parse_join_aggregate_output_expression(
    expr: &SqlExpr,
    table_aliases: &[&str],
    aggregates: &mut Vec<AggregateExpr>,
    filtered_aggregates: &mut Vec<NativeFilteredAggregateSpec>,
    aggregate_expressions: &mut Vec<ProjectionExpression>,
    aggregate_expression_columns: &mut Vec<String>,
    expression_columns: &mut Vec<String>,
    found_aggregate: &mut bool,
) -> Result<ScalarSqlExpression> {
    match expr {
        SqlExpr::Identifier(_) | SqlExpr::CompoundIdentifier(_) => {
            let column = join_column_name(expr, table_aliases)?;
            add_column_once(expression_columns, column.clone());
            Ok(ScalarSqlExpression::Column(column))
        }
        SqlExpr::Value(_) => Ok(ScalarSqlExpression::Literal(sql_literal_value(expr)?)),
        SqlExpr::Nested(expr) => parse_join_aggregate_output_expression(
            expr,
            table_aliases,
            aggregates,
            filtered_aggregates,
            aggregate_expressions,
            aggregate_expression_columns,
            expression_columns,
            found_aggregate,
        ),
        SqlExpr::BinaryOp { left, op, right }
            if matches!(
                op,
                BinaryOperator::Plus
                    | BinaryOperator::Minus
                    | BinaryOperator::Multiply
                    | BinaryOperator::Divide
            ) =>
        {
            Ok(ScalarSqlExpression::Binary {
                left: Box::new(parse_join_aggregate_output_expression(
                    left,
                    table_aliases,
                    aggregates,
                    filtered_aggregates,
                    aggregate_expressions,
                    aggregate_expression_columns,
                    expression_columns,
                    found_aggregate,
                )?),
                op: op.clone(),
                right: Box::new(parse_join_aggregate_output_expression(
                    right,
                    table_aliases,
                    aggregates,
                    filtered_aggregates,
                    aggregate_expressions,
                    aggregate_expression_columns,
                    expression_columns,
                    found_aggregate,
                )?),
            })
        }
        SqlExpr::Cast {
            expr, data_type, ..
        } => Ok(ScalarSqlExpression::Cast {
            expr: Box::new(parse_join_aggregate_output_expression(
                expr,
                table_aliases,
                aggregates,
                filtered_aggregates,
                aggregate_expressions,
                aggregate_expression_columns,
                expression_columns,
                found_aggregate,
            )?),
            target: data_type.to_string(),
        }),
        SqlExpr::Extract { field, expr, .. } if *field == DateTimeField::Year => Ok(
            ScalarSqlExpression::ExtractYear(Box::new(parse_join_aggregate_output_expression(
                expr,
                table_aliases,
                aggregates,
                filtered_aggregates,
                aggregate_expressions,
                aggregate_expression_columns,
                expression_columns,
                found_aggregate,
            )?)),
        ),
        SqlExpr::Function(function) => {
            let (aggregate, expression) = parse_join_aggregate_with_input_expression(
                function,
                table_aliases,
                aggregate_expressions.len(),
            )?;
            if let Some(spec) = filtered_join_aggregate_spec_from_function(
                function,
                table_aliases,
                &aggregate,
                &expression,
            )? {
                filtered_aggregates.push(spec);
            }
            if let Some(expression) = expression {
                for column in join_scalar_expression_columns(&expression.expr, table_aliases)? {
                    add_column_once(aggregate_expression_columns, column);
                }
                aggregate_expressions.push(expression);
            }
            let column = aggregate.to_string();
            aggregates.push(aggregate);
            *found_aggregate = true;
            Ok(ScalarSqlExpression::Column(column))
        }
        _ => Err(DodamError::UnsupportedSql(format!(
            "unsupported JOIN aggregate output expression: {expr}"
        ))),
    }
}

pub(super) fn parse_aggregate_output_expression(
    expr: &SqlExpr,
    table_alias: Option<&str>,
    aggregates: &mut Vec<AggregateExpr>,
    filtered_aggregates: &mut Vec<NativeFilteredAggregateSpec>,
    aggregate_expressions: &mut Vec<ProjectionExpression>,
    aggregate_expression_columns: &mut Vec<String>,
    expression_columns: &mut Vec<String>,
    found_aggregate: &mut bool,
) -> Result<ScalarSqlExpression> {
    match expr {
        SqlExpr::Identifier(_) | SqlExpr::CompoundIdentifier(_) => {
            let column = sql_column_name(expr, table_alias)?;
            add_column_once(expression_columns, column.clone());
            Ok(ScalarSqlExpression::Column(column))
        }
        SqlExpr::Value(_) => Ok(ScalarSqlExpression::Literal(sql_literal_value(expr)?)),
        SqlExpr::Nested(expr) => parse_aggregate_output_expression(
            expr,
            table_alias,
            aggregates,
            filtered_aggregates,
            aggregate_expressions,
            aggregate_expression_columns,
            expression_columns,
            found_aggregate,
        ),
        SqlExpr::BinaryOp { left, op, right }
            if matches!(
                op,
                BinaryOperator::Plus
                    | BinaryOperator::Minus
                    | BinaryOperator::Multiply
                    | BinaryOperator::Divide
            ) =>
        {
            Ok(ScalarSqlExpression::Binary {
                left: Box::new(parse_aggregate_output_expression(
                    left,
                    table_alias,
                    aggregates,
                    filtered_aggregates,
                    aggregate_expressions,
                    aggregate_expression_columns,
                    expression_columns,
                    found_aggregate,
                )?),
                op: op.clone(),
                right: Box::new(parse_aggregate_output_expression(
                    right,
                    table_alias,
                    aggregates,
                    filtered_aggregates,
                    aggregate_expressions,
                    aggregate_expression_columns,
                    expression_columns,
                    found_aggregate,
                )?),
            })
        }
        SqlExpr::Cast {
            expr, data_type, ..
        } => Ok(ScalarSqlExpression::Cast {
            expr: Box::new(parse_aggregate_output_expression(
                expr,
                table_alias,
                aggregates,
                filtered_aggregates,
                aggregate_expressions,
                aggregate_expression_columns,
                expression_columns,
                found_aggregate,
            )?),
            target: data_type.to_string(),
        }),
        SqlExpr::Extract { field, expr, .. } if *field == DateTimeField::Year => Ok(
            ScalarSqlExpression::ExtractYear(Box::new(parse_aggregate_output_expression(
                expr,
                table_alias,
                aggregates,
                filtered_aggregates,
                aggregate_expressions,
                aggregate_expression_columns,
                expression_columns,
                found_aggregate,
            )?)),
        ),
        SqlExpr::Function(function) => {
            let (aggregate, expression) = parse_aggregate_with_input_expression(
                function,
                table_alias,
                aggregate_expressions.len(),
            )?;
            if let Some(spec) = filtered_aggregate_spec_from_function(
                function,
                table_alias,
                &aggregate,
                &expression,
            )? {
                filtered_aggregates.push(spec);
            }
            if let Some(expression) = expression {
                for column in scalar_expression_columns(&expression.expr) {
                    add_column_once(aggregate_expression_columns, column);
                }
                aggregate_expressions.push(expression);
            }
            let column = aggregate.to_string();
            aggregates.push(aggregate);
            *found_aggregate = true;
            Ok(ScalarSqlExpression::Column(column))
        }
        _ => Err(DodamError::UnsupportedSql(format!(
            "unsupported aggregate output expression: {expr}"
        ))),
    }
}

pub(super) fn parse_aggregate(
    function: &sqlparser::ast::Function,
    table_alias: Option<&str>,
) -> Result<AggregateExpr> {
    if function.filter.is_some()
        || function.over.is_some()
        || !function.within_group.is_empty()
        || function.null_treatment.is_some()
        || !matches!(function.parameters, FunctionArguments::None)
    {
        return Err(DodamError::UnsupportedSql(
            "aggregate filters, windows, within group, null treatment, and parameters are not supported"
                .to_string(),
        ));
    }
    let name = object_name_to_string(&function.name)?;
    let (args, duplicate_treatment) = match &function.args {
        FunctionArguments::List(args) if args.clauses.is_empty() => {
            (&args.args, args.duplicate_treatment)
        }
        _ => {
            return Err(DodamError::UnsupportedSql(format!(
                "unsupported function arguments: {}",
                function.args
            )));
        }
    };
    if matches!(duplicate_treatment, Some(DuplicateTreatment::All)) {
        return Err(DodamError::UnsupportedSql(format!(
            "unsupported function arguments: {}",
            function.args
        )));
    }
    let argument = match args.as_slice() {
        [] => {
            return Err(DodamError::UnsupportedSql(format!(
                "missing argument for {name}"
            )));
        }
        [FunctionArg::Unnamed(FunctionArgExpr::Wildcard)] => "*".to_string(),
        [
            FunctionArg::Unnamed(FunctionArgExpr::Expr(
                expr @ (SqlExpr::Identifier(_) | SqlExpr::CompoundIdentifier(_)),
            )),
        ] => sql_column_name(expr, table_alias)?,
        _ => {
            return Err(DodamError::UnsupportedSql(format!(
                "unsupported function arguments: {}",
                function.args
            )));
        }
    };
    if duplicate_treatment == Some(DuplicateTreatment::Distinct) {
        if !name.eq_ignore_ascii_case("count") || argument == "*" {
            return Err(DodamError::UnsupportedSql(format!(
                "only count(DISTINCT column) is supported, got {function}"
            )));
        }
        AggregateExpr::parse(&format!("count_distinct({argument})"))
    } else {
        AggregateExpr::parse(&format!("{name}({argument})"))
    }
}

pub(super) fn parse_aggregate_with_input_expression(
    function: &sqlparser::ast::Function,
    table_alias: Option<&str>,
    expression_index: usize,
) -> Result<(AggregateExpr, Option<ProjectionExpression>)> {
    if function.filter.is_some() {
        return parse_filtered_aggregate_with_input_expression(
            function,
            table_alias,
            expression_index,
        );
    }
    match parse_aggregate(function, table_alias) {
        Ok(aggregate) => return Ok((aggregate, None)),
        Err(DodamError::UnsupportedSql(message))
            if message.starts_with("unsupported function arguments:") => {}
        Err(error) => return Err(error),
    }

    if function.filter.is_some()
        || function.over.is_some()
        || !function.within_group.is_empty()
        || function.null_treatment.is_some()
        || !matches!(function.parameters, FunctionArguments::None)
    {
        return Err(DodamError::UnsupportedSql(
            "aggregate filters, windows, within group, null treatment, and parameters are not supported"
                .to_string(),
        ));
    }
    let name = object_name_to_string(&function.name)?;
    let lowercase_name = name.to_ascii_lowercase();
    if !matches!(
        lowercase_name.as_str(),
        "sum" | "avg" | "min" | "max" | "count"
    ) {
        return Err(DodamError::UnsupportedSql(format!(
            "aggregate expression input is not supported for {name}"
        )));
    }
    let FunctionArguments::List(args) = &function.args else {
        return Err(DodamError::UnsupportedSql(format!(
            "unsupported function arguments: {}",
            function.args
        )));
    };
    if !args.clauses.is_empty()
        || (args.duplicate_treatment.is_some()
            && !(lowercase_name == "count"
                && args.duplicate_treatment == Some(DuplicateTreatment::Distinct)))
    {
        return Err(DodamError::UnsupportedSql(format!(
            "unsupported function arguments: {}",
            function.args
        )));
    }
    let [FunctionArg::Unnamed(FunctionArgExpr::Expr(expr))] = args.args.as_slice() else {
        return Err(DodamError::UnsupportedSql(format!(
            "unsupported function arguments: {}",
            function.args
        )));
    };
    let column = format!("__dodam_agg_expr_{expression_index}");
    let expression = ProjectionExpression {
        output_name: column.clone(),
        expr: parse_scalar_sql_expression(expr, table_alias)?,
    };
    let aggregate = if lowercase_name == "count" {
        if args.duplicate_treatment != Some(DuplicateTreatment::Distinct) {
            return Err(DodamError::UnsupportedSql(format!(
                "aggregate expression input is not supported for {name}"
            )));
        }
        AggregateExpr::parse(&format!("count_distinct({column})"))?
    } else {
        AggregateExpr::parse(&format!("{name}({column})"))?
    };
    Ok((aggregate, Some(expression)))
}

fn parse_filtered_aggregate_with_input_expression(
    function: &sqlparser::ast::Function,
    table_alias: Option<&str>,
    expression_index: usize,
) -> Result<(AggregateExpr, Option<ProjectionExpression>)> {
    if function.over.is_some()
        || !function.within_group.is_empty()
        || function.null_treatment.is_some()
        || !matches!(function.parameters, FunctionArguments::None)
    {
        return Err(DodamError::UnsupportedSql(
            "aggregate windows, within group, null treatment, and parameters are not supported"
                .to_string(),
        ));
    }
    let filter = function.filter.as_deref().ok_or_else(|| {
        DodamError::UnsupportedSql("aggregate FILTER missing predicate".to_string())
    })?;
    let name = object_name_to_string(&function.name)?;
    let lowercase_name = name.to_ascii_lowercase();
    if !matches!(
        lowercase_name.as_str(),
        "sum" | "avg" | "min" | "max" | "count"
    ) {
        return Err(DodamError::UnsupportedSql(format!(
            "aggregate FILTER is not supported for {name}"
        )));
    }
    let FunctionArguments::List(args) = &function.args else {
        return Err(DodamError::UnsupportedSql(format!(
            "unsupported function arguments: {}",
            function.args
        )));
    };
    if !args.clauses.is_empty()
        || (args.duplicate_treatment.is_some()
            && !(lowercase_name == "count"
                && args.duplicate_treatment == Some(DuplicateTreatment::Distinct)))
    {
        return Err(DodamError::UnsupportedSql(format!(
            "unsupported function arguments: {}",
            function.args
        )));
    }
    let input = match args.args.as_slice() {
        [FunctionArg::Unnamed(FunctionArgExpr::Wildcard)] if lowercase_name == "count" => {
            ScalarSqlExpression::Literal(LiteralValue::Int64(1))
        }
        [FunctionArg::Unnamed(FunctionArgExpr::Expr(expr))] => {
            parse_scalar_sql_expression(expr, table_alias)?
        }
        _ => {
            return Err(DodamError::UnsupportedSql(format!(
                "unsupported aggregate FILTER arguments: {}",
                function.args
            )));
        }
    };
    let column = format!("__dodam_agg_expr_{expression_index}");
    if args.duplicate_treatment == Some(DuplicateTreatment::Distinct) {
        let expression = ProjectionExpression {
            output_name: column.clone(),
            expr: ScalarSqlExpression::Case {
                conditions: vec![rewrite_scalar_predicate_table_alias(filter, table_alias)?],
                results: vec![input],
                else_result: None,
            },
        };
        return Ok((
            AggregateExpr::parse(&format!("count_distinct({column})"))?,
            Some(expression),
        ));
    }
    let aggregate = match lowercase_name.as_str() {
        "count" => AggregateExpr::parse(&format!("count({column})"))?,
        _ => AggregateExpr::parse(&format!("{name}({column})"))?,
    };
    let _ = input;
    Ok((aggregate, None))
}

pub(super) fn parse_join_aggregate(
    function: &sqlparser::ast::Function,
    table_aliases: &[&str],
) -> Result<AggregateExpr> {
    if function.filter.is_some()
        || function.over.is_some()
        || !function.within_group.is_empty()
        || function.null_treatment.is_some()
        || !matches!(function.parameters, FunctionArguments::None)
    {
        return Err(DodamError::UnsupportedSql(
            "aggregate filters, windows, within group, null treatment, and parameters are not supported"
                .to_string(),
        ));
    }
    let name = object_name_to_string(&function.name)?;
    let (args, duplicate_treatment) = match &function.args {
        FunctionArguments::List(args) if args.clauses.is_empty() => {
            (&args.args, args.duplicate_treatment)
        }
        _ => {
            return Err(DodamError::UnsupportedSql(format!(
                "unsupported function arguments: {}",
                function.args
            )));
        }
    };
    if matches!(duplicate_treatment, Some(DuplicateTreatment::All)) {
        return Err(DodamError::UnsupportedSql(format!(
            "unsupported function arguments: {}",
            function.args
        )));
    }
    let argument = match args.as_slice() {
        [] => {
            return Err(DodamError::UnsupportedSql(format!(
                "missing argument for {name}"
            )));
        }
        [FunctionArg::Unnamed(FunctionArgExpr::Wildcard)] => "*".to_string(),
        [
            FunctionArg::Unnamed(FunctionArgExpr::Expr(
                expr @ (SqlExpr::Identifier(_) | SqlExpr::CompoundIdentifier(_)),
            )),
        ] => join_column_name(expr, table_aliases)?,
        _ => {
            return Err(DodamError::UnsupportedSql(format!(
                "JOIN aggregate arguments must be * or columns, got {}",
                function.args
            )));
        }
    };
    if duplicate_treatment == Some(DuplicateTreatment::Distinct) {
        if !name.eq_ignore_ascii_case("count") || argument == "*" {
            return Err(DodamError::UnsupportedSql(format!(
                "only count(DISTINCT column) is supported, got {function}"
            )));
        }
        AggregateExpr::parse(&format!("count_distinct({argument})"))
    } else {
        AggregateExpr::parse(&format!("{name}({argument})"))
    }
}

pub(super) fn filtered_aggregate_spec_from_function(
    function: &sqlparser::ast::Function,
    table_alias: Option<&str>,
    aggregate: &AggregateExpr,
    expression: &Option<ProjectionExpression>,
) -> Result<Option<NativeFilteredAggregateSpec>> {
    if function.filter.is_none() || expression.is_some() {
        return Ok(None);
    }
    let (input, condition) = filtered_aggregate_input_and_condition(
        function,
        |expr| parse_scalar_sql_expression(expr, table_alias),
        |expr| rewrite_scalar_predicate_table_alias(expr, table_alias),
    )?;
    Ok(Some(NativeFilteredAggregateSpec {
        expr: aggregate.clone(),
        condition,
        input_kind: native_filtered_input_kind(aggregate, &input),
        input,
    }))
}

pub(super) fn filtered_join_aggregate_spec_from_function(
    function: &sqlparser::ast::Function,
    table_aliases: &[&str],
    aggregate: &AggregateExpr,
    expression: &Option<ProjectionExpression>,
) -> Result<Option<NativeFilteredAggregateSpec>> {
    if function.filter.is_none() || expression.is_some() {
        return Ok(None);
    }
    let (input, condition) = filtered_aggregate_input_and_condition(
        function,
        |expr| parse_join_scalar_sql_expression(expr, table_aliases),
        |expr| rewrite_join_scalar_predicate(expr, table_aliases),
    )?;
    Ok(Some(NativeFilteredAggregateSpec {
        expr: aggregate.clone(),
        condition,
        input_kind: native_filtered_input_kind(aggregate, &input),
        input,
    }))
}

fn filtered_aggregate_input_and_condition(
    function: &sqlparser::ast::Function,
    parse_input: impl Fn(&SqlExpr) -> Result<ScalarSqlExpression>,
    rewrite_condition: impl Fn(&SqlExpr) -> Result<SqlExpr>,
) -> Result<(ScalarSqlExpression, SqlExpr)> {
    let filter = function.filter.as_deref().ok_or_else(|| {
        DodamError::UnsupportedSql("aggregate FILTER missing predicate".to_string())
    })?;
    let name = object_name_to_string(&function.name)?;
    let lowercase_name = name.to_ascii_lowercase();
    let FunctionArguments::List(args) = &function.args else {
        return Err(DodamError::UnsupportedSql(format!(
            "unsupported function arguments: {}",
            function.args
        )));
    };
    let input = match args.args.as_slice() {
        [FunctionArg::Unnamed(FunctionArgExpr::Wildcard)] if lowercase_name == "count" => {
            ScalarSqlExpression::Literal(LiteralValue::Int64(1))
        }
        [FunctionArg::Unnamed(FunctionArgExpr::Expr(expr))] => parse_input(expr)?,
        _ => {
            return Err(DodamError::UnsupportedSql(format!(
                "unsupported aggregate FILTER arguments: {}",
                function.args
            )));
        }
    };
    Ok((input, rewrite_condition(filter)?))
}

pub(super) fn parse_join_aggregate_with_input_expression(
    function: &sqlparser::ast::Function,
    table_aliases: &[&str],
    expression_index: usize,
) -> Result<(AggregateExpr, Option<ProjectionExpression>)> {
    if function.filter.is_some() {
        return parse_filtered_join_aggregate_with_input_expression(
            function,
            table_aliases,
            expression_index,
        );
    }
    match parse_join_aggregate(function, table_aliases) {
        Ok(aggregate) => return Ok((aggregate, None)),
        Err(DodamError::UnsupportedSql(message))
            if message.starts_with("JOIN aggregate arguments must be") => {}
        Err(error) => return Err(error),
    }

    if function.filter.is_some()
        || function.over.is_some()
        || !function.within_group.is_empty()
        || function.null_treatment.is_some()
        || !matches!(function.parameters, FunctionArguments::None)
    {
        return Err(DodamError::UnsupportedSql(
            "aggregate filters, windows, within group, null treatment, and parameters are not supported"
                .to_string(),
        ));
    }
    let name = object_name_to_string(&function.name)?;
    let lowercase_name = name.to_ascii_lowercase();
    if !matches!(
        lowercase_name.as_str(),
        "sum" | "avg" | "min" | "max" | "count"
    ) {
        return Err(DodamError::UnsupportedSql(format!(
            "aggregate expression input is not supported for {name}"
        )));
    }
    let FunctionArguments::List(args) = &function.args else {
        return Err(DodamError::UnsupportedSql(format!(
            "unsupported function arguments: {}",
            function.args
        )));
    };
    if !args.clauses.is_empty()
        || (args.duplicate_treatment.is_some()
            && !(lowercase_name == "count"
                && args.duplicate_treatment == Some(DuplicateTreatment::Distinct)))
    {
        return Err(DodamError::UnsupportedSql(format!(
            "unsupported function arguments: {}",
            function.args
        )));
    }
    let [FunctionArg::Unnamed(FunctionArgExpr::Expr(expr))] = args.args.as_slice() else {
        return Err(DodamError::UnsupportedSql(format!(
            "unsupported function arguments: {}",
            function.args
        )));
    };
    let column = format!("__dodam_join_agg_expr_{expression_index}");
    let expression = ProjectionExpression {
        output_name: column.clone(),
        expr: parse_join_scalar_sql_expression(expr, table_aliases)?,
    };
    let aggregate = if lowercase_name == "count" {
        if args.duplicate_treatment != Some(DuplicateTreatment::Distinct) {
            return Err(DodamError::UnsupportedSql(format!(
                "aggregate expression input is not supported for {name}"
            )));
        }
        AggregateExpr::parse(&format!("count_distinct({column})"))?
    } else {
        AggregateExpr::parse(&format!("{name}({column})"))?
    };
    Ok((aggregate, Some(expression)))
}

fn parse_filtered_join_aggregate_with_input_expression(
    function: &sqlparser::ast::Function,
    table_aliases: &[&str],
    expression_index: usize,
) -> Result<(AggregateExpr, Option<ProjectionExpression>)> {
    if function.over.is_some()
        || !function.within_group.is_empty()
        || function.null_treatment.is_some()
        || !matches!(function.parameters, FunctionArguments::None)
    {
        return Err(DodamError::UnsupportedSql(
            "aggregate windows, within group, null treatment, and parameters are not supported"
                .to_string(),
        ));
    }
    let filter = function.filter.as_deref().ok_or_else(|| {
        DodamError::UnsupportedSql("aggregate FILTER missing predicate".to_string())
    })?;
    let name = object_name_to_string(&function.name)?;
    let lowercase_name = name.to_ascii_lowercase();
    if !matches!(
        lowercase_name.as_str(),
        "sum" | "avg" | "min" | "max" | "count"
    ) {
        return Err(DodamError::UnsupportedSql(format!(
            "JOIN aggregate FILTER is not supported for {name}"
        )));
    }
    let FunctionArguments::List(args) = &function.args else {
        return Err(DodamError::UnsupportedSql(format!(
            "unsupported function arguments: {}",
            function.args
        )));
    };
    if !args.clauses.is_empty()
        || (args.duplicate_treatment.is_some()
            && !(lowercase_name == "count"
                && args.duplicate_treatment == Some(DuplicateTreatment::Distinct)))
    {
        return Err(DodamError::UnsupportedSql(format!(
            "unsupported function arguments: {}",
            function.args
        )));
    }
    let input = match args.args.as_slice() {
        [FunctionArg::Unnamed(FunctionArgExpr::Wildcard)] if lowercase_name == "count" => {
            ScalarSqlExpression::Literal(LiteralValue::Int64(1))
        }
        [FunctionArg::Unnamed(FunctionArgExpr::Expr(expr))] => {
            parse_join_scalar_sql_expression(expr, table_aliases)?
        }
        _ => {
            return Err(DodamError::UnsupportedSql(format!(
                "unsupported JOIN aggregate FILTER arguments: {}",
                function.args
            )));
        }
    };
    let column = format!("__dodam_join_agg_expr_{expression_index}");
    if args.duplicate_treatment == Some(DuplicateTreatment::Distinct) {
        let expression = ProjectionExpression {
            output_name: column.clone(),
            expr: ScalarSqlExpression::Case {
                conditions: vec![rewrite_join_scalar_predicate(filter, table_aliases)?],
                results: vec![input],
                else_result: None,
            },
        };
        return Ok((
            AggregateExpr::parse(&format!("count_distinct({column})"))?,
            Some(expression),
        ));
    }
    let aggregate = match lowercase_name.as_str() {
        "count" => AggregateExpr::parse(&format!("count({column})"))?,
        _ => AggregateExpr::parse(&format!("{name}({column})"))?,
    };
    let _ = input;
    Ok((aggregate, None))
}

fn rewrite_scalar_predicate_table_alias(
    expr: &SqlExpr,
    table_alias: Option<&str>,
) -> Result<SqlExpr> {
    match expr {
        SqlExpr::Identifier(_) | SqlExpr::CompoundIdentifier(_) => {
            Ok(sql_column_expr(&sql_column_name(expr, table_alias)?))
        }
        SqlExpr::BinaryOp { left, op, right } => Ok(SqlExpr::BinaryOp {
            left: Box::new(rewrite_scalar_predicate_table_alias(left, table_alias)?),
            op: op.clone(),
            right: Box::new(rewrite_scalar_predicate_table_alias(right, table_alias)?),
        }),
        SqlExpr::Nested(expr) => Ok(SqlExpr::Nested(Box::new(
            rewrite_scalar_predicate_table_alias(expr, table_alias)?,
        ))),
        SqlExpr::UnaryOp { op, expr } => Ok(SqlExpr::UnaryOp {
            op: op.clone(),
            expr: Box::new(rewrite_scalar_predicate_table_alias(expr, table_alias)?),
        }),
        SqlExpr::IsNull(expr) => Ok(SqlExpr::IsNull(Box::new(
            rewrite_scalar_predicate_table_alias(expr, table_alias)?,
        ))),
        SqlExpr::IsNotNull(expr) => Ok(SqlExpr::IsNotNull(Box::new(
            rewrite_scalar_predicate_table_alias(expr, table_alias)?,
        ))),
        SqlExpr::InList {
            expr,
            list,
            negated,
        } => Ok(SqlExpr::InList {
            expr: Box::new(rewrite_scalar_predicate_table_alias(expr, table_alias)?),
            list: list
                .iter()
                .map(|expr| rewrite_scalar_predicate_table_alias(expr, table_alias))
                .collect::<Result<Vec<_>>>()?,
            negated: *negated,
        }),
        SqlExpr::Between {
            expr,
            negated,
            low,
            high,
        } => Ok(SqlExpr::Between {
            expr: Box::new(rewrite_scalar_predicate_table_alias(expr, table_alias)?),
            negated: *negated,
            low: Box::new(rewrite_scalar_predicate_table_alias(low, table_alias)?),
            high: Box::new(rewrite_scalar_predicate_table_alias(high, table_alias)?),
        }),
        SqlExpr::Like {
            negated,
            any,
            expr,
            pattern,
            escape_char,
        } => Ok(SqlExpr::Like {
            negated: *negated,
            any: *any,
            expr: Box::new(rewrite_scalar_predicate_table_alias(expr, table_alias)?),
            pattern: Box::new(rewrite_scalar_predicate_table_alias(pattern, table_alias)?),
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
            expr: Box::new(rewrite_scalar_predicate_table_alias(expr, table_alias)?),
            pattern: Box::new(rewrite_scalar_predicate_table_alias(pattern, table_alias)?),
            escape_char: escape_char.clone(),
        }),
        SqlExpr::Value(_) | SqlExpr::TypedString(_) => Ok(expr.clone()),
        _ => Ok(expr.clone()),
    }
}
