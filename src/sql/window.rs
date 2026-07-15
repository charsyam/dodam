use super::*;

pub(super) async fn try_execute_window_sql(
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
    let Some(window) = parse_window_projection(select)? else {
        return Ok(None);
    };
    reject_query_features(query)?;
    reject_select_features(select)?;
    if select.from.len() != 1
        || select
            .from
            .first()
            .is_some_and(|table| !table.joins.is_empty())
        || parse_distinct(select)?
        || select.having.is_some()
        || !matches!(select.group_by, GroupByExpr::Expressions(ref exprs, _) if exprs.is_empty())
    {
        return Err(DodamError::UnsupportedSql(
            "row_number window currently supports one non-aggregate table input".to_string(),
        ));
    }
    let path = parse_from(select)?;
    let filter = select
        .selection
        .as_ref()
        .map(|expr| parse_filter(expr, &[], path.alias.as_deref(), false))
        .transpose()?;
    let order_by = parse_window_query_order_by(query, &window, path.alias.as_deref())?;
    let limit = parse_limit(query)?;
    let offset = parse_offset(query)?;
    let execution_sort = window_execution_sort(&window, order_by.as_ref());
    let pre_window_prefix_limit =
        if execution_sort_satisfies_order(execution_sort.as_ref(), order_by.as_ref())
            && window_prefix_limit_safe(&window)
        {
            limit.and_then(|limit| limit.checked_add(offset))
        } else {
            None
        };
    let mut scan_projection = Projection::Columns(window.input_columns.clone());
    if let Some(window_sort) = &execution_sort {
        add_projection_columns(
            &mut scan_projection,
            window_sort
                .expressions
                .iter()
                .map(|sort| sort.column.clone())
                .collect(),
        );
    }
    if let Some(order_by) = query.order_by.as_ref() {
        add_projection_columns(
            &mut scan_projection,
            row_number_query_order_columns(order_by, &window, path.alias.as_deref())?,
        );
    }
    let profile = window_profile_enabled();
    let total_started = profile.then(Instant::now);
    let scan_started = profile.then(Instant::now);
    let use_ordered_window_scan = window_ordered_scan_enabled() && execution_sort.is_some();
    let stream = if use_ordered_window_scan {
        let window_sort = execution_sort
            .as_ref()
            .expect("ordered window scan requires execution sort");
        engine
            .scan_parquet_ordered_batches_by(
                path.path,
                batch_size,
                pre_window_prefix_limit,
                scan_projection,
                filter,
                window_sort.clone(),
            )
            .await?
    } else {
        engine
            .scan_parquet_batches(path.path, batch_size, None, scan_projection, filter)
            .await?
    };
    let scan_elapsed = scan_started.map(|started| started.elapsed());
    let collect_started = profile.then(Instant::now);
    let mut batches = collect_batches(stream)?;
    let collected_rows = batches.iter().map(RecordBatch::num_rows).sum::<usize>();
    let collected_batches = batches.len();
    let collect_elapsed = collect_started.map(|started| started.elapsed());
    let sort_started = profile.then(Instant::now);
    batches = if let Some(window_sort) = &execution_sort {
        if use_ordered_window_scan && pre_window_prefix_limit.is_some() {
            limit_batches(batches, pre_window_prefix_limit, 0)
        } else if use_ordered_window_scan
            && batches.len() <= 1
            && output_batches_satisfy_order(&batches, window_sort)?
        {
            batches
        } else {
            apply_output_order_limit(batches, Some(window_sort), pre_window_prefix_limit, 0)?
        }
    } else {
        coalesce_batches(batches)?
    };
    let sorted_rows = batches.iter().map(RecordBatch::num_rows).sum::<usize>();
    let sorted_batches = batches.len();
    let sort_elapsed = sort_started.map(|started| started.elapsed());
    let append_started = profile.then(Instant::now);
    let window_columns_appended = if window_partition_hash_aggregate_safe(&window) {
        if let Some(late_batches) = try_apply_int32_partition_hash_window_order_limit(
            batches.clone(),
            &window,
            order_by.as_ref(),
            limit,
            offset,
        )? {
            batches = late_batches;
            true
        } else {
            false
        }
    } else {
        false
    };
    if !window_columns_appended {
        batches = append_window_function_columns(batches, &window)?;
        let order_started = profile.then(Instant::now);
        batches = if execution_sort_satisfies_order(execution_sort.as_ref(), order_by.as_ref()) {
            limit_batches(batches, limit, offset)
        } else {
            apply_output_order_limit(batches, order_by.as_ref(), limit, offset)?
        };
        if let (true, Some(started)) = (profile, order_started) {
            eprintln!(
                "[dodam:window-profile] final_order={}us",
                window_profile_micros(started.elapsed())
            );
        }
    }
    let append_elapsed = append_started.map(|started| started.elapsed());
    let projection_started = profile.then(Instant::now);
    batches = apply_output_projection(batches, &Projection::Columns(window.output_columns))?;
    let output_rows = batches.iter().map(RecordBatch::num_rows).sum::<usize>();
    if let (true, Some(total_started)) = (profile, total_started) {
        eprintln!(
            "[dodam:window-profile] total={}us scan_plan={}us collect={}us sort={}us append_window={}us projection={}us collected_batches={} collected_rows={} sorted_batches={} sorted_rows={} output_rows={} ordered_scan={} prefix_limit={:?}",
            window_profile_micros(total_started.elapsed()),
            scan_elapsed.map(window_profile_micros).unwrap_or(0),
            collect_elapsed.map(window_profile_micros).unwrap_or(0),
            sort_elapsed.map(window_profile_micros).unwrap_or(0),
            append_elapsed.map(window_profile_micros).unwrap_or(0),
            projection_started
                .map(|started| window_profile_micros(started.elapsed()))
                .unwrap_or(0),
            collected_batches,
            collected_rows,
            sorted_batches,
            sorted_rows,
            output_rows,
            use_ordered_window_scan,
            pre_window_prefix_limit,
        );
    }
    Ok(Some(QueryOutput::Scan { batches }))
}

fn window_profile_enabled() -> bool {
    std::env::var("DODAM_WINDOW_PROFILE").is_ok_and(|value| {
        value == "1" || value.eq_ignore_ascii_case("true") || value.eq_ignore_ascii_case("yes")
    })
}

fn window_profile_micros(duration: Duration) -> u128 {
    duration.as_micros()
}

fn window_ordered_scan_enabled() -> bool {
    std::env::var("DODAM_DISABLE_WINDOW_ORDERED_SCAN")
        .map(|value| value != "1" && !value.eq_ignore_ascii_case("true"))
        .unwrap_or(true)
}

fn window_execution_sort(
    window: &WindowProjection,
    final_order: Option<&SortKey>,
) -> Option<SortKey> {
    if window_partition_hash_aggregate_safe(window) {
        return None;
    }
    let Some(final_order) = final_order else {
        return window.window_sort.clone();
    };
    if final_order
        .expressions
        .iter()
        .any(|sort| window.has_output_name(&sort.column))
    {
        return window.window_sort.clone();
    }
    match window.window_sort.as_ref() {
        Some(window_sort)
            if final_order_can_drive_window_sort(window, final_order, window_sort)
                && final_order != window_sort =>
        {
            Some(final_order.clone())
        }
        Some(window_sort) => Some(window_sort.clone()),
        None => Some(final_order.clone()),
    }
}

fn execution_sort_satisfies_order(
    execution_sort: Option<&SortKey>,
    final_order: Option<&SortKey>,
) -> bool {
    let Some(final_order) = final_order else {
        return true;
    };
    let Some(execution_sort) = execution_sort else {
        return false;
    };
    sort_key_prefix_matches(execution_sort, final_order)
}

fn sort_key_prefix_matches(sort: &SortKey, prefix: &SortKey) -> bool {
    sort.expressions.len() >= prefix.expressions.len()
        && sort
            .expressions
            .iter()
            .zip(&prefix.expressions)
            .all(|(left, right)| left == right)
}

fn final_order_can_drive_window_sort(
    window: &WindowProjection,
    final_order: &SortKey,
    window_sort: &SortKey,
) -> bool {
    if final_order.expressions.len() < window_sort.expressions.len() {
        return false;
    }
    let partition_columns = window
        .functions
        .first()
        .map(|function| function.partition_by.len())
        .unwrap_or(0);
    final_order
        .expressions
        .iter()
        .zip(&window_sort.expressions)
        .enumerate()
        .all(|(index, (final_sort, window_sort))| {
            if index < partition_columns {
                final_sort.column == window_sort.column
            } else {
                final_sort == window_sort
            }
        })
}

fn window_prefix_limit_safe(window: &WindowProjection) -> bool {
    window.functions.iter().all(|function| {
        matches!(
            function.function,
            WindowFunctionKind::RowNumber
                | WindowFunctionKind::Rank
                | WindowFunctionKind::DenseRank
        ) || !function.order_by.is_empty()
    })
}

fn window_partition_hash_aggregate_safe(window: &WindowProjection) -> bool {
    !window.functions.is_empty()
        && window.functions.iter().all(|function| {
            matches!(
                function.function,
                WindowFunctionKind::Sum | WindowFunctionKind::Count | WindowFunctionKind::Avg
            ) && function.order_by.is_empty()
                && !function.partition_by.is_empty()
        })
}

#[derive(Debug)]
struct WindowProjection {
    input_columns: Vec<String>,
    output_columns: Vec<String>,
    window_sort: Option<SortKey>,
    aliases: Vec<(String, String)>,
    ordinal_targets: Vec<String>,
    functions: Vec<WindowProjectionFunction>,
}

#[derive(Debug, Clone)]
struct WindowProjectionFunction {
    output_name: String,
    function: WindowFunctionKind,
    argument: Option<ScalarSqlExpression>,
    offset: usize,
    partition_by: Vec<String>,
    order_by: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WindowFunctionKind {
    RowNumber,
    Rank,
    DenseRank,
    Sum,
    Count,
    Avg,
    Lag,
    Lead,
}

fn parse_window_projection(select: &Select) -> Result<Option<WindowProjection>> {
    if !select
        .projection
        .iter()
        .any(select_item_is_row_number_window)
    {
        return Ok(None);
    }
    let mut input_columns = Vec::new();
    let mut output_columns = Vec::new();
    let mut aliases = Vec::new();
    let mut functions = Vec::new();
    let mut window_sort: Option<Option<SortKey>> = None;

    for item in &select.projection {
        match item {
            SelectItem::UnnamedExpr(
                expr @ (SqlExpr::Identifier(_) | SqlExpr::CompoundIdentifier(_)),
            ) => {
                let column = sql_column_name(expr, None)?;
                add_column_once(&mut input_columns, column.clone());
                output_columns.push(column);
            }
            SelectItem::ExprWithAlias { expr, alias }
                if matches!(
                    expr,
                    SqlExpr::Identifier(_) | SqlExpr::CompoundIdentifier(_)
                ) =>
            {
                let column = sql_column_name(expr, None)?;
                add_column_once(&mut input_columns, column.clone());
                output_columns.push(alias.value.clone());
                aliases.push((alias.value.clone(), column));
            }
            SelectItem::UnnamedExpr(SqlExpr::Function(function))
            | SelectItem::ExprWithAlias {
                expr: SqlExpr::Function(function),
                alias: _,
            } if function.over.is_some() && supported_window_function(function).is_some() => {
                let output_name = match item {
                    SelectItem::ExprWithAlias { alias, .. } => alias.value.clone(),
                    _ => function.to_string(),
                };
                let (function, argument, offset, partition_by, order_by, sort) =
                    parse_window_function(function)?;
                if let Some(argument) = &argument {
                    for column in scalar_expression_columns(argument) {
                        add_column_once(&mut input_columns, column);
                    }
                }
                if let Some(window_sort) = &window_sort {
                    if window_sort != &sort {
                        return Err(DodamError::UnsupportedSql(
                            "multiple window projections currently require the same OVER specification"
                                .to_string(),
                        ));
                    }
                } else {
                    window_sort = Some(sort);
                }
                functions.push(WindowProjectionFunction {
                    output_name: output_name.clone(),
                    function,
                    argument,
                    offset,
                    partition_by,
                    order_by,
                });
                output_columns.push(output_name);
            }
            _ => return Ok(None),
        }
    }
    if functions.is_empty() {
        return Ok(None);
    }
    for function in &functions {
        aliases.push((function.output_name.clone(), function.output_name.clone()));
    }
    let ordinal_targets = output_columns
        .iter()
        .map(|column| resolve_alias(column, &aliases))
        .collect();
    Ok(Some(WindowProjection {
        input_columns,
        output_columns,
        window_sort: window_sort.expect("window functions imply a parsed OVER specification"),
        aliases,
        ordinal_targets,
        functions,
    }))
}

fn select_item_is_row_number_window(item: &SelectItem) -> bool {
    match item {
        SelectItem::UnnamedExpr(SqlExpr::Function(function))
        | SelectItem::ExprWithAlias {
            expr: SqlExpr::Function(function),
            ..
        } => function.over.is_some() && supported_window_function(function).is_some(),
        _ => false,
    }
}

fn supported_window_function(function: &sqlparser::ast::Function) -> Option<WindowFunctionKind> {
    let name = object_name_to_string(&function.name).ok()?;
    match name.to_ascii_lowercase().as_str() {
        "row_number" => Some(WindowFunctionKind::RowNumber),
        "rank" => Some(WindowFunctionKind::Rank),
        "dense_rank" => Some(WindowFunctionKind::DenseRank),
        "sum" => Some(WindowFunctionKind::Sum),
        "count" => Some(WindowFunctionKind::Count),
        "avg" => Some(WindowFunctionKind::Avg),
        "lag" => Some(WindowFunctionKind::Lag),
        "lead" => Some(WindowFunctionKind::Lead),
        _ => None,
    }
}

fn parse_window_function(
    function: &sqlparser::ast::Function,
) -> Result<(
    WindowFunctionKind,
    Option<ScalarSqlExpression>,
    usize,
    Vec<String>,
    Vec<String>,
    Option<SortKey>,
)> {
    let function_kind = supported_window_function(function).ok_or_else(|| {
        DodamError::UnsupportedSql(format!("unsupported window function: {function}"))
    })?;
    if function.filter.is_some()
        || function.null_treatment.is_some()
        || !function.within_group.is_empty()
        || !matches!(function.parameters, FunctionArguments::None)
    {
        return Err(DodamError::UnsupportedSql(
            "window filters, null treatment, within group, and parameters are not supported"
                .to_string(),
        ));
    }
    let (argument, offset) = parse_window_function_argument(function, function_kind)?;
    let Some(WindowType::WindowSpec(spec)) = &function.over else {
        return Err(DodamError::UnsupportedSql(
            "window function requires an OVER window specification".to_string(),
        ));
    };
    if spec.window_name.is_some() || spec.window_frame.is_some() {
        return Err(DodamError::UnsupportedSql(
            "window names and frames are not supported".to_string(),
        ));
    }
    if spec.order_by.is_empty()
        && matches!(
            function_kind,
            WindowFunctionKind::RowNumber
                | WindowFunctionKind::Rank
                | WindowFunctionKind::DenseRank
                | WindowFunctionKind::Lag
                | WindowFunctionKind::Lead
        )
    {
        return Err(DodamError::UnsupportedSql(
            "ranking and offset window functions require ORDER BY in the window specification"
                .to_string(),
        ));
    }
    let partition_by = spec
        .partition_by
        .iter()
        .map(|expr| sql_column_name(expr, None))
        .collect::<Result<Vec<_>>>()?;
    let order_by = spec
        .order_by
        .iter()
        .map(|order| {
            if order.with_fill.is_some() {
                return Err(DodamError::UnsupportedSql(
                    "row_number window ORDER BY WITH FILL is not supported".to_string(),
                ));
            }
            Ok(SortExpr {
                column: sql_column_name(&order.expr, None)?,
                descending: order.options.asc == Some(false),
                nulls_first: order.options.nulls_first.unwrap_or(false),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let mut sort_expressions = partition_by
        .iter()
        .map(|column| SortExpr {
            column: column.clone(),
            descending: false,
            nulls_first: false,
        })
        .collect::<Vec<_>>();
    sort_expressions.extend(order_by.clone());
    let order_columns = order_by.iter().map(|sort| sort.column.clone()).collect();
    let sort = if sort_expressions.is_empty() {
        None
    } else {
        Some(SortKey::new(sort_expressions)?)
    };
    Ok((
        function_kind,
        argument,
        offset,
        partition_by,
        order_columns,
        sort,
    ))
}

fn parse_window_function_argument(
    function: &sqlparser::ast::Function,
    function_kind: WindowFunctionKind,
) -> Result<(Option<ScalarSqlExpression>, usize)> {
    match function_kind {
        WindowFunctionKind::RowNumber
        | WindowFunctionKind::Rank
        | WindowFunctionKind::DenseRank => match &function.args {
            FunctionArguments::None => Ok((None, 1)),
            FunctionArguments::List(args) if args.args.is_empty() && args.clauses.is_empty() => {
                Ok((None, 1))
            }
            _ => Err(DodamError::UnsupportedSql(format!(
                "ranking window function expects no arguments, got {}",
                function.args
            ))),
        },
        WindowFunctionKind::Sum | WindowFunctionKind::Avg => {
            let FunctionArguments::List(args) = &function.args else {
                return Err(DodamError::UnsupportedSql(format!(
                    "window aggregate expects one argument, got {}",
                    function.args
                )));
            };
            if !args.clauses.is_empty() || args.duplicate_treatment.is_some() {
                return Err(DodamError::UnsupportedSql(format!(
                    "unsupported window aggregate arguments: {}",
                    function.args
                )));
            }
            let [FunctionArg::Unnamed(FunctionArgExpr::Expr(expr))] = args.args.as_slice() else {
                return Err(DodamError::UnsupportedSql(format!(
                    "window aggregate expects one expression argument, got {}",
                    function.args
                )));
            };
            Ok((Some(parse_scalar_sql_expression(expr, None)?), 1))
        }
        WindowFunctionKind::Count => {
            let FunctionArguments::List(args) = &function.args else {
                return Err(DodamError::UnsupportedSql(format!(
                    "window count expects an argument, got {}",
                    function.args
                )));
            };
            if !args.clauses.is_empty()
                || !matches!(
                    args.duplicate_treatment,
                    None | Some(DuplicateTreatment::All)
                )
            {
                return Err(DodamError::UnsupportedSql(format!(
                    "unsupported window count arguments: {}",
                    function.args
                )));
            }
            match args.args.as_slice() {
                [FunctionArg::Unnamed(FunctionArgExpr::Wildcard)] => Ok((None, 1)),
                [FunctionArg::Unnamed(FunctionArgExpr::Expr(expr))] => {
                    Ok((Some(parse_scalar_sql_expression(expr, None)?), 1))
                }
                _ => Err(DodamError::UnsupportedSql(format!(
                    "window count expects one argument, got {}",
                    function.args
                ))),
            }
        }
        WindowFunctionKind::Lag | WindowFunctionKind::Lead => {
            let FunctionArguments::List(args) = &function.args else {
                return Err(DodamError::UnsupportedSql(format!(
                    "window {} expects arguments, got {}",
                    object_name_to_string(&function.name)?,
                    function.args
                )));
            };
            if !args.clauses.is_empty() || args.duplicate_treatment.is_some() {
                return Err(DodamError::UnsupportedSql(format!(
                    "unsupported window offset arguments: {}",
                    function.args
                )));
            }
            match args.args.as_slice() {
                [FunctionArg::Unnamed(FunctionArgExpr::Expr(expr))] => {
                    Ok((Some(parse_scalar_sql_expression(expr, None)?), 1))
                }
                [
                    FunctionArg::Unnamed(FunctionArgExpr::Expr(expr)),
                    FunctionArg::Unnamed(FunctionArgExpr::Expr(offset)),
                ] => Ok((
                    Some(parse_scalar_sql_expression(expr, None)?),
                    parse_usize_literal(offset)?,
                )),
                _ => Err(DodamError::UnsupportedSql(format!(
                    "window {} expects one expression and optional offset, got {}",
                    object_name_to_string(&function.name)?,
                    function.args
                ))),
            }
        }
    }
}

fn row_number_query_order_columns(
    order_by: &sqlparser::ast::OrderBy,
    window: &WindowProjection,
    table_alias: Option<&str>,
) -> Result<Vec<String>> {
    let OrderByKind::Expressions(expressions) = &order_by.kind else {
        return Ok(Vec::new());
    };
    expressions
        .iter()
        .filter_map(|order| match &order.expr {
            SqlExpr::Value(_) => None,
            SqlExpr::Identifier(ident) if window.has_output_name(&ident.value) => None,
            expr => Some(
                sql_column_name(expr, table_alias)
                    .map(|column| resolve_alias(&column, &window.aliases)),
            ),
        })
        .collect()
}

impl WindowProjection {
    fn has_output_name(&self, name: &str) -> bool {
        self.functions
            .iter()
            .any(|function| function.output_name == name)
    }
}

fn parse_window_query_order_by(
    query: &Query,
    window: &WindowProjection,
    table_alias: Option<&str>,
) -> Result<Option<SortKey>> {
    let Some(order_by) = &query.order_by else {
        return Ok(None);
    };
    if order_by.interpolate.is_some() {
        return Err(DodamError::UnsupportedSql(
            "ORDER BY INTERPOLATE is not supported".to_string(),
        ));
    }
    let OrderByKind::Expressions(expressions) = &order_by.kind else {
        return Err(DodamError::UnsupportedSql(
            "ORDER BY ALL is not supported".to_string(),
        ));
    };
    let expressions = expressions
        .iter()
        .map(|order| {
            if order.with_fill.is_some() {
                return Err(DodamError::UnsupportedSql(
                    "ORDER BY WITH FILL is not supported".to_string(),
                ));
            }
            let column = match &order.expr {
                SqlExpr::Value(value) => resolve_order_by_ordinal(value, &window.ordinal_targets)?,
                SqlExpr::Identifier(ident) if window.has_output_name(&ident.value) => {
                    ident.value.clone()
                }
                expr => resolve_alias(&sql_column_name(expr, table_alias)?, &window.aliases),
            };
            Ok(SortExpr {
                column,
                descending: order.options.asc == Some(false),
                nulls_first: order.options.nulls_first.unwrap_or(false),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    SortKey::new(expressions).map(Some)
}

fn append_window_function_columns(
    batches: Vec<RecordBatch>,
    window: &WindowProjection,
) -> Result<Vec<RecordBatch>> {
    batches
        .into_iter()
        .map(|batch| {
            if let Some(batch) = append_int32_partition_hash_window_columns(&batch, window)? {
                return Ok(batch);
            }
            if let Some(batch) = append_ranking_window_columns(&batch, window)? {
                return Ok(batch);
            }
            if let Some(batch) = append_running_aggregate_window_columns(&batch, window)? {
                return Ok(batch);
            }
            if let Some(batch) = append_offset_window_columns(&batch, window)? {
                return Ok(batch);
            }
            let mut function_values = Vec::with_capacity(window.functions.len());
            for function in &window.functions {
                function_values.push(window_function_values(&batch, function)?);
            }
            let mut fields = batch.schema().fields().to_vec();
            let mut columns = batch.columns().to_vec();
            for (function, values) in window.functions.iter().zip(function_values) {
                fields.push(Arc::new(Field::new(
                    function.output_name.clone(),
                    values.data_type(),
                    values.is_nullable(),
                )));
                columns.push(values.into_array());
            }
            Ok(RecordBatch::try_new(
                Arc::new(Schema::new(fields)),
                columns,
            )?)
        })
        .collect()
}

fn append_offset_window_columns(
    batch: &RecordBatch,
    window: &WindowProjection,
) -> Result<Option<RecordBatch>> {
    let Some(first_function) = window.functions.first() else {
        return Ok(None);
    };
    if !window.functions.iter().all(|function| {
        matches!(
            function.function,
            WindowFunctionKind::Lag | WindowFunctionKind::Lead
        ) && function.partition_by == first_function.partition_by
            && function.order_by == first_function.order_by
    }) {
        return Ok(None);
    }
    let ranges = window_partition_ranges_fast(batch, first_function)?;
    let mut fields = batch.schema().fields().to_vec();
    let mut columns = batch.columns().to_vec();
    for function in &window.functions {
        let argument = function.argument.as_ref().ok_or_else(|| {
            DodamError::UnsupportedSql("window offset function requires an argument".to_string())
        })?;
        if let Some(array) = try_shift_column_window_values(batch, &ranges, function, argument)? {
            fields.push(Arc::new(Field::new(
                function.output_name.clone(),
                array.data_type().clone(),
                true,
            )));
            columns.push(array);
            continue;
        }
        let values = materialize_evaluated_scalar(evaluate_scalar_expression(batch, argument)?)?;
        let values =
            shift_window_scalar_values_with_ranges(batch.num_rows(), &ranges, function, values)?;
        fields.push(Arc::new(Field::new(
            function.output_name.clone(),
            values.data_type(),
            values.is_nullable(),
        )));
        columns.push(values.into_array(batch.num_rows()));
    }
    Ok(Some(RecordBatch::try_new(
        Arc::new(Schema::new(fields)),
        columns,
    )?))
}

fn try_shift_column_window_values(
    batch: &RecordBatch,
    ranges: &[(usize, usize)],
    window: &WindowProjectionFunction,
    argument: &ScalarSqlExpression,
) -> Result<Option<ArrayRef>> {
    let ScalarSqlExpression::Column(column) = argument else {
        return Ok(None);
    };
    let index = output_batch_column_index(batch, column)?;
    let array = batch.column(index);
    if !matches!(
        array.data_type(),
        DataType::Int32 | DataType::Int64 | DataType::Utf8
    ) {
        return Ok(None);
    }
    let indices = shift_window_take_indices(batch.num_rows(), ranges, window)?;
    Ok(Some(arrow_select::take::take(
        array.as_ref(),
        &indices,
        None,
    )?))
}

fn shift_window_take_indices(
    row_count: usize,
    ranges: &[(usize, usize)],
    window: &WindowProjectionFunction,
) -> Result<UInt32Array> {
    let mut output = vec![None; row_count];
    let offset = window.offset;
    if offset == 0 {
        for (row, slot) in output.iter_mut().enumerate() {
            *slot = Some(u32::try_from(row).map_err(|_| {
                DodamError::UnsupportedSql("window offset row index overflow".to_string())
            })?);
        }
        return Ok(UInt32Array::from(output));
    }
    for &(start, end) in ranges {
        if end.saturating_sub(start) <= offset {
            continue;
        }
        match window.function {
            WindowFunctionKind::Lag => {
                for row in start + offset..end {
                    output[row] = Some(u32::try_from(row - offset).map_err(|_| {
                        DodamError::UnsupportedSql("window offset row index overflow".to_string())
                    })?);
                }
            }
            WindowFunctionKind::Lead => {
                for row in start..end - offset {
                    output[row] = Some(u32::try_from(row + offset).map_err(|_| {
                        DodamError::UnsupportedSql("window offset row index overflow".to_string())
                    })?);
                }
            }
            _ => unreachable!("checked by caller"),
        }
    }
    Ok(UInt32Array::from(output))
}

fn append_ranking_window_columns(
    batch: &RecordBatch,
    window: &WindowProjection,
) -> Result<Option<RecordBatch>> {
    let Some(first_function) = window.functions.first() else {
        return Ok(None);
    };
    if !window.functions.iter().all(|function| {
        matches!(
            function.function,
            WindowFunctionKind::RowNumber
                | WindowFunctionKind::Rank
                | WindowFunctionKind::DenseRank
        ) && function.partition_by == first_function.partition_by
            && function.order_by == first_function.order_by
    }) {
        return Ok(None);
    }
    if let Some(batch) = append_primitive_ranking_window_columns(batch, window, first_function)? {
        return Ok(Some(batch));
    }
    if let Some(boundaries) = primitive_window_boundaries(batch, first_function)? {
        return append_ranking_window_columns_with_boundaries(batch, window, &boundaries).map(Some);
    }
    let mut row_numbers = Vec::with_capacity(batch.num_rows());
    let mut ranks = Vec::with_capacity(batch.num_rows());
    let mut dense_ranks = Vec::with_capacity(batch.num_rows());
    let mut partition_start = 0_usize;
    let mut rank = 1_u64;
    let mut dense_rank = 1_u64;
    for row in 0..batch.num_rows() {
        if row == 0 || !window_partition_equal(batch, first_function, row - 1, row)? {
            partition_start = row;
            rank = 1;
            dense_rank = 1;
        } else if !window_order_equal(batch, first_function, row - 1, row)? {
            rank = u64::try_from(row - partition_start + 1)
                .map_err(|_| DodamError::UnsupportedSql("window rank overflow".to_string()))?;
            dense_rank += 1;
        }
        row_numbers.push(
            u64::try_from(row - partition_start + 1)
                .map_err(|_| DodamError::UnsupportedSql("row_number overflow".to_string()))?,
        );
        ranks.push(rank);
        dense_ranks.push(dense_rank);
    }

    append_ranking_outputs(batch, window, row_numbers, ranks, dense_ranks).map(Some)
}

fn append_primitive_ranking_window_columns(
    batch: &RecordBatch,
    window: &WindowProjection,
    first_function: &WindowProjectionFunction,
) -> Result<Option<RecordBatch>> {
    let Some(partition_values) = single_int32_window_partition_array(batch, first_function)? else {
        return Ok(None);
    };
    let [order_column] = first_function.order_by.as_slice() else {
        return Ok(None);
    };
    let order_index = output_batch_column_index(batch, order_column)?;
    let order = batch.column(order_index);
    match order.data_type() {
        DataType::Int32 => {
            let Some(order_values) = order.as_any().downcast_ref::<Int32Array>() else {
                return Ok(None);
            };
            append_primitive_ranking_outputs(
                batch,
                window,
                |left, right| int32_values_equal(partition_values, left, right),
                |left, right| int32_values_equal(order_values, left, right),
            )
            .map(Some)
        }
        DataType::Int64 => {
            let Some(order_values) = order.as_any().downcast_ref::<Int64Array>() else {
                return Ok(None);
            };
            append_primitive_ranking_outputs(
                batch,
                window,
                |left, right| int32_values_equal(partition_values, left, right),
                |left, right| int64_values_equal(order_values, left, right),
            )
            .map(Some)
        }
        _ => Ok(None),
    }
}

fn append_primitive_ranking_outputs(
    batch: &RecordBatch,
    window: &WindowProjection,
    partition_equal: impl Fn(usize, usize) -> bool,
    order_equal: impl Fn(usize, usize) -> bool,
) -> Result<RecordBatch> {
    let mut row_numbers = Vec::with_capacity(batch.num_rows());
    let mut ranks = Vec::with_capacity(batch.num_rows());
    let mut dense_ranks = Vec::with_capacity(batch.num_rows());
    let mut partition_start = 0_usize;
    let mut rank = 1_u64;
    let mut dense_rank = 1_u64;
    for row in 0..batch.num_rows() {
        if row == 0 || !partition_equal(row - 1, row) {
            partition_start = row;
            rank = 1;
            dense_rank = 1;
        } else if !order_equal(row - 1, row) {
            rank = u64::try_from(row - partition_start + 1)
                .map_err(|_| DodamError::UnsupportedSql("window rank overflow".to_string()))?;
            dense_rank += 1;
        }
        row_numbers.push(
            u64::try_from(row - partition_start + 1)
                .map_err(|_| DodamError::UnsupportedSql("row_number overflow".to_string()))?,
        );
        ranks.push(rank);
        dense_ranks.push(dense_rank);
    }
    append_ranking_outputs(batch, window, row_numbers, ranks, dense_ranks)
}

struct WindowBoundaries {
    partition_start: Vec<bool>,
    peer_changed: Vec<bool>,
}

fn primitive_window_boundaries(
    batch: &RecordBatch,
    window: &WindowProjectionFunction,
) -> Result<Option<WindowBoundaries>> {
    let Some(partition_keys) = single_int32_window_partition_keys(batch, window)? else {
        return Ok(None);
    };
    let [order_column] = window.order_by.as_slice() else {
        return Ok(None);
    };
    let order_index = output_batch_column_index(batch, order_column)?;
    let order = batch.column(order_index);
    let mut partition_start = Vec::with_capacity(batch.num_rows());
    let mut peer_changed = Vec::with_capacity(batch.num_rows());
    match order.data_type() {
        DataType::Int32 => {
            let Some(order_values) = order.as_any().downcast_ref::<Int32Array>() else {
                return Ok(None);
            };
            for row in 0..batch.num_rows() {
                let partition_changed = row == 0 || partition_keys[row - 1] != partition_keys[row];
                let order_changed = row == 0 || !int32_values_equal(order_values, row - 1, row);
                partition_start.push(partition_changed);
                peer_changed.push(partition_changed || order_changed);
            }
        }
        DataType::Int64 => {
            let Some(order_values) = order.as_any().downcast_ref::<Int64Array>() else {
                return Ok(None);
            };
            for row in 0..batch.num_rows() {
                let partition_changed = row == 0 || partition_keys[row - 1] != partition_keys[row];
                let order_changed = row == 0 || !int64_values_equal(order_values, row - 1, row);
                partition_start.push(partition_changed);
                peer_changed.push(partition_changed || order_changed);
            }
        }
        _ => return Ok(None),
    }
    Ok(Some(WindowBoundaries {
        partition_start,
        peer_changed,
    }))
}

fn int32_values_equal(values: &Int32Array, left: usize, right: usize) -> bool {
    if values.is_null(left) != values.is_null(right) {
        return false;
    }
    values.is_null(left) || values.value(left) == values.value(right)
}

fn int64_values_equal(values: &Int64Array, left: usize, right: usize) -> bool {
    if values.is_null(left) != values.is_null(right) {
        return false;
    }
    values.is_null(left) || values.value(left) == values.value(right)
}

fn append_ranking_window_columns_with_boundaries(
    batch: &RecordBatch,
    window: &WindowProjection,
    boundaries: &WindowBoundaries,
) -> Result<RecordBatch> {
    let mut row_numbers = Vec::with_capacity(batch.num_rows());
    let mut ranks = Vec::with_capacity(batch.num_rows());
    let mut dense_ranks = Vec::with_capacity(batch.num_rows());
    let mut partition_start = 0_usize;
    let mut rank = 1_u64;
    let mut dense_rank = 1_u64;
    for row in 0..batch.num_rows() {
        if boundaries.partition_start[row] {
            partition_start = row;
            rank = 1;
            dense_rank = 1;
        } else if boundaries.peer_changed[row] {
            rank = u64::try_from(row - partition_start + 1)
                .map_err(|_| DodamError::UnsupportedSql("window rank overflow".to_string()))?;
            dense_rank += 1;
        }
        row_numbers.push(
            u64::try_from(row - partition_start + 1)
                .map_err(|_| DodamError::UnsupportedSql("row_number overflow".to_string()))?,
        );
        ranks.push(rank);
        dense_ranks.push(dense_rank);
    }
    append_ranking_outputs(batch, window, row_numbers, ranks, dense_ranks)
}

fn append_ranking_outputs(
    batch: &RecordBatch,
    window: &WindowProjection,
    row_numbers: Vec<u64>,
    ranks: Vec<u64>,
    dense_ranks: Vec<u64>,
) -> Result<RecordBatch> {
    let mut fields = batch.schema().fields().to_vec();
    let mut columns = batch.columns().to_vec();
    let mut row_numbers = Some(row_numbers);
    let mut ranks = Some(ranks);
    let mut dense_ranks = Some(dense_ranks);
    let mut row_number_remaining = window
        .functions
        .iter()
        .filter(|function| function.function == WindowFunctionKind::RowNumber)
        .count();
    let mut rank_remaining = window
        .functions
        .iter()
        .filter(|function| function.function == WindowFunctionKind::Rank)
        .count();
    let mut dense_rank_remaining = window
        .functions
        .iter()
        .filter(|function| function.function == WindowFunctionKind::DenseRank)
        .count();
    for function in &window.functions {
        let values = match function.function {
            WindowFunctionKind::RowNumber => {
                row_number_remaining -= 1;
                if row_number_remaining == 0 {
                    row_numbers.take()
                } else {
                    row_numbers.clone()
                }
            }
            WindowFunctionKind::Rank => {
                rank_remaining -= 1;
                if rank_remaining == 0 {
                    ranks.take()
                } else {
                    ranks.clone()
                }
            }
            WindowFunctionKind::DenseRank => {
                dense_rank_remaining -= 1;
                if dense_rank_remaining == 0 {
                    dense_ranks.take()
                } else {
                    dense_ranks.clone()
                }
            }
            WindowFunctionKind::Sum | WindowFunctionKind::Count | WindowFunctionKind::Avg => {
                return Err(DodamError::UnsupportedSql(
                    "aggregate window in ranking output".to_string(),
                ));
            }
            WindowFunctionKind::Lag | WindowFunctionKind::Lead => {
                return Err(DodamError::UnsupportedSql(
                    "offset window in ranking output".to_string(),
                ));
            }
        }
        .ok_or_else(|| {
            DodamError::UnsupportedSql("duplicate ranking window output requested".to_string())
        })?;
        fields.push(Arc::new(Field::new(
            function.output_name.clone(),
            DataType::UInt64,
            false,
        )));
        columns.push(Arc::new(UInt64Array::from_iter_values(values)) as ArrayRef);
    }
    Ok(RecordBatch::try_new(
        Arc::new(Schema::new(fields)),
        columns,
    )?)
}

fn append_running_aggregate_window_columns(
    batch: &RecordBatch,
    window: &WindowProjection,
) -> Result<Option<RecordBatch>> {
    let Some(first_function) = window.functions.first() else {
        return Ok(None);
    };
    if first_function.order_by.is_empty()
        || !window.functions.iter().all(|function| {
            matches!(
                function.function,
                WindowFunctionKind::Count | WindowFunctionKind::Sum | WindowFunctionKind::Avg
            ) && function.partition_by == first_function.partition_by
                && function.order_by == first_function.order_by
        })
    {
        return Ok(None);
    }
    if let Some(batch) =
        append_primitive_running_aggregate_window_columns(batch, window, first_function)?
    {
        return Ok(Some(batch));
    }

    enum RunningInput {
        Count {
            present: Vec<bool>,
            output: Vec<Option<u64>>,
            count: u64,
        },
        SumAvg {
            values: Vec<Option<f64>>,
            output: Vec<Option<f64>>,
            sum: f64,
            count: u64,
            avg: bool,
        },
    }

    let mut inputs = window
        .functions
        .iter()
        .map(|function| match function.function {
            WindowFunctionKind::Count => {
                let argument_values = function
                    .argument
                    .as_ref()
                    .map(|argument| evaluate_scalar_expression(batch, argument))
                    .transpose()?;
                let present = argument_values
                    .map(evaluated_scalar_present_mask)
                    .transpose()?
                    .unwrap_or_else(|| vec![true; batch.num_rows()]);
                Ok(RunningInput::Count {
                    present,
                    output: Vec::with_capacity(batch.num_rows()),
                    count: 0,
                })
            }
            WindowFunctionKind::Sum | WindowFunctionKind::Avg => {
                let argument = function.argument.as_ref().ok_or_else(|| {
                    DodamError::UnsupportedSql("window aggregate requires an argument".to_string())
                })?;
                Ok(RunningInput::SumAvg {
                    values: scalar_as_f64(evaluate_scalar_expression(batch, argument)?)?,
                    output: Vec::with_capacity(batch.num_rows()),
                    sum: 0.0,
                    count: 0,
                    avg: function.function == WindowFunctionKind::Avg,
                })
            }
            WindowFunctionKind::RowNumber
            | WindowFunctionKind::Rank
            | WindowFunctionKind::DenseRank
            | WindowFunctionKind::Lag
            | WindowFunctionKind::Lead => unreachable!("checked above"),
        })
        .collect::<Result<Vec<_>>>()?;
    let primitive_partition_values = single_int32_window_partition_array(batch, first_function)?;
    let boundaries = if primitive_partition_values.is_some() {
        None
    } else {
        primitive_window_boundaries(batch, first_function)?
    };

    for row in 0..batch.num_rows() {
        let partition_start = if let Some(partition_values) = primitive_partition_values {
            row == 0 || !int32_values_equal(partition_values, row - 1, row)
        } else if let Some(boundaries) = boundaries.as_ref() {
            boundaries.partition_start[row]
        } else if row == 0 {
            true
        } else {
            !window_partition_equal(batch, first_function, row - 1, row)?
        };
        if partition_start {
            for input in &mut inputs {
                match input {
                    RunningInput::Count { count, .. } => *count = 0,
                    RunningInput::SumAvg { sum, count, .. } => {
                        *sum = 0.0;
                        *count = 0;
                    }
                }
            }
        }
        for input in &mut inputs {
            match input {
                RunningInput::Count {
                    present,
                    output,
                    count,
                } => {
                    if present[row] {
                        *count += 1;
                    }
                    output.push(Some(*count));
                }
                RunningInput::SumAvg {
                    values,
                    output,
                    sum,
                    count,
                    avg,
                } => {
                    if let Some(value) = values[row] {
                        *sum += value;
                        *count += 1;
                    }
                    output.push(if *count > 0 {
                        Some(if *avg { *sum / *count as f64 } else { *sum })
                    } else {
                        None
                    });
                }
            }
        }
    }

    let mut fields = batch.schema().fields().to_vec();
    let mut columns = batch.columns().to_vec();
    for (function, input) in window.functions.iter().zip(inputs) {
        match input {
            RunningInput::Count { output, .. } => {
                fields.push(Arc::new(Field::new(
                    function.output_name.clone(),
                    DataType::UInt64,
                    false,
                )));
                columns.push(Arc::new(UInt64Array::from(output)) as ArrayRef);
            }
            RunningInput::SumAvg { output, .. } => {
                fields.push(Arc::new(Field::new(
                    function.output_name.clone(),
                    DataType::Float64,
                    true,
                )));
                columns.push(Arc::new(Float64Array::from(output)) as ArrayRef);
            }
        }
    }
    Ok(Some(RecordBatch::try_new(
        Arc::new(Schema::new(fields)),
        columns,
    )?))
}

fn append_primitive_running_aggregate_window_columns(
    batch: &RecordBatch,
    window: &WindowProjection,
    first_function: &WindowProjectionFunction,
) -> Result<Option<RecordBatch>> {
    let Some(partition_values) = single_int32_window_partition_array(batch, first_function)? else {
        return Ok(None);
    };

    enum PrimitiveRunningInput<'a> {
        CountAll {
            output: Vec<u64>,
            count: u64,
        },
        SumAvgInt64 {
            values: &'a Int64Array,
            output: Vec<Option<f64>>,
            sum: f64,
            count: u64,
            avg: bool,
        },
        SumAvgInt64NonNull {
            values: &'a Int64Array,
            output: Vec<f64>,
            sum: f64,
            count: u64,
            avg: bool,
        },
    }

    let mut inputs = Vec::with_capacity(window.functions.len());
    for function in &window.functions {
        match function.function {
            WindowFunctionKind::Count if function.argument.is_none() => {
                inputs.push(PrimitiveRunningInput::CountAll {
                    output: Vec::with_capacity(batch.num_rows()),
                    count: 0,
                });
            }
            WindowFunctionKind::Sum | WindowFunctionKind::Avg => {
                let Some(values) = int64_window_argument_array(batch, function)? else {
                    return Ok(None);
                };
                if values.null_count() == 0 {
                    inputs.push(PrimitiveRunningInput::SumAvgInt64NonNull {
                        values,
                        output: Vec::with_capacity(batch.num_rows()),
                        sum: 0.0,
                        count: 0,
                        avg: function.function == WindowFunctionKind::Avg,
                    });
                } else {
                    inputs.push(PrimitiveRunningInput::SumAvgInt64 {
                        values,
                        output: Vec::with_capacity(batch.num_rows()),
                        sum: 0.0,
                        count: 0,
                        avg: function.function == WindowFunctionKind::Avg,
                    });
                }
            }
            _ => return Ok(None),
        }
    }

    for row in 0..batch.num_rows() {
        if row == 0 || !int32_values_equal(partition_values, row - 1, row) {
            for input in &mut inputs {
                match input {
                    PrimitiveRunningInput::CountAll { count, .. } => *count = 0,
                    PrimitiveRunningInput::SumAvgInt64 { sum, count, .. } => {
                        *sum = 0.0;
                        *count = 0;
                    }
                    PrimitiveRunningInput::SumAvgInt64NonNull { sum, count, .. } => {
                        *sum = 0.0;
                        *count = 0;
                    }
                }
            }
        }
        for input in &mut inputs {
            match input {
                PrimitiveRunningInput::CountAll { output, count } => {
                    *count += 1;
                    output.push(*count);
                }
                PrimitiveRunningInput::SumAvgInt64 {
                    values,
                    output,
                    sum,
                    count,
                    avg,
                } => {
                    if !values.is_null(row) {
                        *sum += values.value(row) as f64;
                        *count += 1;
                    }
                    output.push(if *count > 0 {
                        Some(if *avg { *sum / *count as f64 } else { *sum })
                    } else {
                        None
                    });
                }
                PrimitiveRunningInput::SumAvgInt64NonNull {
                    values,
                    output,
                    sum,
                    count,
                    avg,
                } => {
                    *sum += values.value(row) as f64;
                    *count += 1;
                    output.push(if *avg { *sum / *count as f64 } else { *sum });
                }
            }
        }
    }

    let mut fields = batch.schema().fields().to_vec();
    let mut columns = batch.columns().to_vec();
    for (function, input) in window.functions.iter().zip(inputs) {
        match input {
            PrimitiveRunningInput::CountAll { output, .. } => {
                fields.push(Arc::new(Field::new(
                    function.output_name.clone(),
                    DataType::UInt64,
                    false,
                )));
                columns.push(Arc::new(UInt64Array::from_iter_values(output)) as ArrayRef);
            }
            PrimitiveRunningInput::SumAvgInt64 { output, .. } => {
                fields.push(Arc::new(Field::new(
                    function.output_name.clone(),
                    DataType::Float64,
                    true,
                )));
                columns.push(Arc::new(Float64Array::from(output)) as ArrayRef);
            }
            PrimitiveRunningInput::SumAvgInt64NonNull { output, .. } => {
                fields.push(Arc::new(Field::new(
                    function.output_name.clone(),
                    DataType::Float64,
                    false,
                )));
                columns.push(Arc::new(Float64Array::from_iter_values(output)) as ArrayRef);
            }
        }
    }
    Ok(Some(RecordBatch::try_new(
        Arc::new(Schema::new(fields)),
        columns,
    )?))
}

fn int64_window_argument_array<'a>(
    batch: &'a RecordBatch,
    function: &WindowProjectionFunction,
) -> Result<Option<&'a Int64Array>> {
    let Some(argument) = function.argument.as_ref() else {
        return Ok(None);
    };
    let column = match argument {
        ScalarSqlExpression::Column(column) => Some(column),
        ScalarSqlExpression::Cast { expr, target }
            if matches!(
                target.to_ascii_lowercase().as_str(),
                "double" | "float8" | "float" | "real"
            ) =>
        {
            match expr.as_ref() {
                ScalarSqlExpression::Column(column) => Some(column),
                _ => None,
            }
        }
        _ => None,
    };
    let Some(column) = column else {
        return Ok(None);
    };
    let index = output_batch_column_index(batch, column)?;
    Ok(batch.column(index).as_any().downcast_ref::<Int64Array>())
}

fn try_apply_int32_partition_hash_window_order_limit(
    batches: Vec<RecordBatch>,
    window: &WindowProjection,
    order_by: Option<&SortKey>,
    limit: Option<usize>,
    offset: usize,
) -> Result<Option<Vec<RecordBatch>>> {
    if !window_partition_hash_aggregate_safe(window)
        || order_by.is_none()
        || window_order_uses_output(order_by, window)
    {
        return Ok(None);
    }
    let profile = window_profile_enabled();
    let total_started = profile.then(Instant::now);
    let coalesce_started = profile.then(Instant::now);
    let Some(full_batch) = coalesce_batches(batches)?.into_iter().next() else {
        return Ok(Some(Vec::new()));
    };
    let coalesce_elapsed = coalesce_started.map(|started| started.elapsed());
    let Some(first_function) = window.functions.first() else {
        return Ok(None);
    };
    if !window
        .functions
        .iter()
        .all(|function| function.partition_by == first_function.partition_by)
    {
        return Ok(None);
    }
    let key_started = profile.then(Instant::now);
    let Some(full_keys) = single_int32_window_partition_keys(&full_batch, first_function)? else {
        return Ok(None);
    };
    let key_elapsed = key_started.map(|started| started.elapsed());
    let state_started = profile.then(Instant::now);
    let states = if let Some(states) =
        try_shared_int64_partition_window_states(&full_batch, window, &full_keys)?
    {
        states
    } else {
        window
            .functions
            .iter()
            .map(|function| int32_partition_window_state(&full_batch, function, &full_keys))
            .collect::<Result<Vec<_>>>()?
    };
    let state_elapsed = state_started.map(|started| started.elapsed());
    let prune_started = profile.then(Instant::now);
    let sortable_batch = prune_window_late_append_sort_batch(full_batch, window, order_by)?;
    let prune_elapsed = prune_started.map(|started| started.elapsed());
    let sort_started = profile.then(Instant::now);
    let limited_batches = apply_output_order_limit(vec![sortable_batch], order_by, limit, offset)?;
    let sort_elapsed = sort_started.map(|started| started.elapsed());
    let append_started = profile.then(Instant::now);
    let result = limited_batches
        .into_iter()
        .map(|batch| append_int32_partition_hash_window_columns_from_states(batch, window, &states))
        .collect::<Result<Vec<_>>>()
        .map(Some);
    let append_elapsed = append_started.map(|started| started.elapsed());
    if let (true, Some(total_started)) = (profile, total_started) {
        eprintln!(
            "[dodam:window-late-profile] total={}us coalesce={}us keys={}us state={}us prune={}us sort_limit={}us append={}us",
            window_profile_micros(total_started.elapsed()),
            coalesce_elapsed.map(window_profile_micros).unwrap_or(0),
            key_elapsed.map(window_profile_micros).unwrap_or(0),
            state_elapsed.map(window_profile_micros).unwrap_or(0),
            prune_elapsed.map(window_profile_micros).unwrap_or(0),
            sort_elapsed.map(window_profile_micros).unwrap_or(0),
            append_elapsed.map(window_profile_micros).unwrap_or(0),
        );
    }
    result
}

fn prune_window_late_append_sort_batch(
    batch: RecordBatch,
    window: &WindowProjection,
    order_by: Option<&SortKey>,
) -> Result<RecordBatch> {
    let mut needed = HashSet::new();
    let window_outputs = window
        .functions
        .iter()
        .map(|function| function.output_name.as_str())
        .collect::<HashSet<_>>();
    for column in &window.output_columns {
        if !window_outputs.contains(column.as_str()) {
            needed.insert(column.as_str());
        }
    }
    if let Some(order_by) = order_by {
        for sort in &order_by.expressions {
            if !window_outputs.contains(sort.column.as_str()) {
                needed.insert(sort.column.as_str());
            }
        }
    }
    for function in &window.functions {
        for column in &function.partition_by {
            needed.insert(column.as_str());
        }
    }
    if needed.len() >= batch.num_columns() {
        return Ok(batch);
    }
    let indices = batch
        .schema()
        .fields()
        .iter()
        .enumerate()
        .filter_map(|(index, field)| needed.contains(field.name().as_str()).then_some(index))
        .collect::<Vec<_>>();
    Ok(batch.project(&indices)?)
}

fn window_order_uses_output(order_by: Option<&SortKey>, window: &WindowProjection) -> bool {
    order_by.is_some_and(|order_by| {
        order_by
            .expressions
            .iter()
            .any(|sort| window.has_output_name(&sort.column))
    })
}

enum Int32PartitionWindowState {
    Count(HashMap<Option<i32>, u64>),
    CountDense {
        min: i32,
        counts: Vec<u64>,
        null_count: u64,
    },
    SumAvg(HashMap<Option<i32>, (f64, u64)>),
    SumAvgDense {
        min: i32,
        sums: Vec<f64>,
        counts: Vec<u64>,
        null_sum: f64,
        null_count: u64,
    },
    SharedInt64Dense {
        min: i32,
        sums: Arc<Vec<f64>>,
        counts: Arc<Vec<u64>>,
        null_sum: f64,
        null_count: u64,
        output: SharedInt64WindowOutput,
    },
}

#[derive(Clone, Copy)]
enum SharedInt64WindowOutput {
    Count,
    Sum,
    Avg,
}

fn try_shared_int64_partition_window_states(
    batch: &RecordBatch,
    window: &WindowProjection,
    keys: &[Option<i32>],
) -> Result<Option<Vec<Int32PartitionWindowState>>> {
    if window.functions.len() < 2 {
        return Ok(None);
    }
    let Some((min, slot_count)) = dense_i32_window_partition_range(keys) else {
        return Ok(None);
    };
    let Some(first_column) = shared_int64_window_argument_column(&window.functions[0]) else {
        return Ok(None);
    };
    let mut outputs = Vec::with_capacity(window.functions.len());
    for function in &window.functions {
        let Some(column) = shared_int64_window_argument_column(function) else {
            return Ok(None);
        };
        if column != first_column {
            return Ok(None);
        }
        outputs.push(match function.function {
            WindowFunctionKind::Count => SharedInt64WindowOutput::Count,
            WindowFunctionKind::Sum => SharedInt64WindowOutput::Sum,
            WindowFunctionKind::Avg => SharedInt64WindowOutput::Avg,
            WindowFunctionKind::RowNumber
            | WindowFunctionKind::Rank
            | WindowFunctionKind::DenseRank
            | WindowFunctionKind::Lag
            | WindowFunctionKind::Lead => {
                return Ok(None);
            }
        });
    }
    let index = output_batch_column_index(batch, first_column)?;
    let Some(values) = batch.column(index).as_any().downcast_ref::<Int64Array>() else {
        return Ok(None);
    };
    let mut sums = vec![0.0; slot_count];
    let mut counts = vec![0_u64; slot_count];
    let mut null_sum = 0.0;
    let mut null_count = 0_u64;
    for (row, key) in keys.iter().enumerate() {
        if values.is_null(row) {
            continue;
        }
        let value = values.value(row) as f64;
        match key {
            Some(key) => {
                let slot = (*key - min) as usize;
                sums[slot] += value;
                counts[slot] += 1;
            }
            None => {
                null_sum += value;
                null_count += 1;
            }
        }
    }
    let sums = Arc::new(sums);
    let counts = Arc::new(counts);
    Ok(Some(
        outputs
            .into_iter()
            .map(|output| Int32PartitionWindowState::SharedInt64Dense {
                min,
                sums: Arc::clone(&sums),
                counts: Arc::clone(&counts),
                null_sum,
                null_count,
                output,
            })
            .collect(),
    ))
}

fn shared_int64_window_argument_column(function: &WindowProjectionFunction) -> Option<&str> {
    let argument = function.argument.as_ref()?;
    match (function.function, argument) {
        (
            WindowFunctionKind::Count | WindowFunctionKind::Avg,
            ScalarSqlExpression::Column(column),
        ) => Some(column),
        (WindowFunctionKind::Sum, ScalarSqlExpression::Column(column)) => Some(column),
        (
            WindowFunctionKind::Sum | WindowFunctionKind::Avg,
            ScalarSqlExpression::Cast { expr, target },
        ) if matches!(
            target.to_ascii_lowercase().as_str(),
            "double" | "float8" | "float" | "real"
        ) =>
        {
            match expr.as_ref() {
                ScalarSqlExpression::Column(column) => Some(column),
                _ => None,
            }
        }
        _ => None,
    }
}

fn int32_partition_window_state(
    batch: &RecordBatch,
    function: &WindowProjectionFunction,
    keys: &[Option<i32>],
) -> Result<Int32PartitionWindowState> {
    match function.function {
        WindowFunctionKind::Count => {
            let argument_values = function
                .argument
                .as_ref()
                .map(|argument| evaluate_scalar_expression(batch, argument))
                .transpose()?;
            let present = argument_values
                .map(evaluated_scalar_present_mask)
                .transpose()?
                .unwrap_or_else(|| vec![true; batch.num_rows()]);
            if let Some((min, slot_count)) = dense_i32_window_partition_range(keys) {
                let mut counts = vec![0_u64; slot_count];
                let mut null_count = 0_u64;
                for (key, present) in keys.iter().zip(present) {
                    if !present {
                        continue;
                    }
                    match key {
                        Some(key) => counts[(*key - min) as usize] += 1,
                        None => null_count += 1,
                    }
                }
                return Ok(Int32PartitionWindowState::CountDense {
                    min,
                    counts,
                    null_count,
                });
            }
            let mut counts = HashMap::<Option<i32>, u64>::new();
            for (key, present) in keys.iter().zip(present) {
                if present {
                    *counts.entry(*key).or_insert(0) += 1;
                } else {
                    counts.entry(*key).or_insert(0);
                }
            }
            Ok(Int32PartitionWindowState::Count(counts))
        }
        WindowFunctionKind::Sum | WindowFunctionKind::Avg => {
            let argument = function.argument.as_ref().ok_or_else(|| {
                DodamError::UnsupportedSql("window aggregate requires an argument".to_string())
            })?;
            let values = scalar_as_f64(evaluate_scalar_expression(batch, argument)?)?;
            if let Some((min, slot_count)) = dense_i32_window_partition_range(keys) {
                let mut sums = vec![0.0; slot_count];
                let mut counts = vec![0_u64; slot_count];
                let mut null_sum = 0.0;
                let mut null_count = 0_u64;
                for (key, value) in keys.iter().zip(values) {
                    let Some(value) = value else {
                        continue;
                    };
                    match key {
                        Some(key) => {
                            let slot = (*key - min) as usize;
                            sums[slot] += value;
                            counts[slot] += 1;
                        }
                        None => {
                            null_sum += value;
                            null_count += 1;
                        }
                    }
                }
                return Ok(Int32PartitionWindowState::SumAvgDense {
                    min,
                    sums,
                    counts,
                    null_sum,
                    null_count,
                });
            }
            let mut aggregates = HashMap::<Option<i32>, (f64, u64)>::new();
            for (key, value) in keys.iter().zip(values) {
                let entry = aggregates.entry(*key).or_insert((0.0, 0));
                if let Some(value) = value {
                    entry.0 += value;
                    entry.1 += 1;
                }
            }
            Ok(Int32PartitionWindowState::SumAvg(aggregates))
        }
        WindowFunctionKind::RowNumber
        | WindowFunctionKind::Rank
        | WindowFunctionKind::DenseRank
        | WindowFunctionKind::Lag
        | WindowFunctionKind::Lead => Err(DodamError::UnsupportedSql(
            "ranking functions are not partition hash aggregates".to_string(),
        )),
    }
}

fn dense_i32_window_partition_range(keys: &[Option<i32>]) -> Option<(i32, usize)> {
    let mut min = i32::MAX;
    let mut max = i32::MIN;
    let mut has_value = false;
    for key in keys.iter().flatten() {
        min = min.min(*key);
        max = max.max(*key);
        has_value = true;
    }
    if !has_value {
        return Some((0, 0));
    }
    let slot_count = usize::try_from(i64::from(max) - i64::from(min) + 1).ok()?;
    (slot_count <= keys.len().saturating_mul(4) && slot_count <= 1_000_000)
        .then_some((min, slot_count))
}

fn append_int32_partition_hash_window_columns_from_states(
    batch: RecordBatch,
    window: &WindowProjection,
    states: &[Int32PartitionWindowState],
) -> Result<RecordBatch> {
    let Some(first_function) = window.functions.first() else {
        return Ok(batch);
    };
    let Some(keys) = single_int32_window_partition_keys(&batch, first_function)? else {
        return Err(DodamError::UnsupportedSql(
            "expected Int32 partition key for window aggregate late append".to_string(),
        ));
    };
    let mut fields = batch.schema().fields().to_vec();
    let mut columns = batch.columns().to_vec();
    for (function, state) in window.functions.iter().zip(states) {
        let values = match (function.function, state) {
            (WindowFunctionKind::Count, Int32PartitionWindowState::Count(counts)) => {
                WindowFunctionValues::UInt64(
                    keys.iter()
                        .map(|key| Some(counts.get(key).copied().unwrap_or(0)))
                        .collect(),
                )
            }
            (
                WindowFunctionKind::Count,
                Int32PartitionWindowState::CountDense {
                    min,
                    counts,
                    null_count,
                },
            ) => WindowFunctionValues::UInt64(
                keys.iter()
                    .map(|key| {
                        Some(match key {
                            Some(key) => counts.get((*key - *min) as usize).copied().unwrap_or(0),
                            None => *null_count,
                        })
                    })
                    .collect(),
            ),
            (
                WindowFunctionKind::Sum | WindowFunctionKind::Avg,
                Int32PartitionWindowState::SumAvg(aggregates),
            ) => {
                let sums_counts = keys
                    .iter()
                    .map(|key| aggregates.get(key).copied().unwrap_or((0.0, 0)))
                    .collect::<Vec<_>>();
                if function.function == WindowFunctionKind::Sum {
                    WindowFunctionValues::Float64(
                        sums_counts
                            .into_iter()
                            .map(|(sum, count)| (count > 0).then_some(sum))
                            .collect(),
                    )
                } else {
                    WindowFunctionValues::Float64(
                        sums_counts
                            .into_iter()
                            .map(|(sum, count)| (count > 0).then_some(sum / count as f64))
                            .collect(),
                    )
                }
            }
            (
                WindowFunctionKind::Sum | WindowFunctionKind::Avg,
                Int32PartitionWindowState::SumAvgDense {
                    min,
                    sums,
                    counts,
                    null_sum,
                    null_count,
                },
            ) => {
                let sums_counts = keys
                    .iter()
                    .map(|key| match key {
                        Some(key) => {
                            let slot = (*key - *min) as usize;
                            (
                                sums.get(slot).copied().unwrap_or(0.0),
                                counts.get(slot).copied().unwrap_or(0),
                            )
                        }
                        None => (*null_sum, *null_count),
                    })
                    .collect::<Vec<_>>();
                if function.function == WindowFunctionKind::Sum {
                    WindowFunctionValues::Float64(
                        sums_counts
                            .into_iter()
                            .map(|(sum, count)| (count > 0).then_some(sum))
                            .collect(),
                    )
                } else {
                    WindowFunctionValues::Float64(
                        sums_counts
                            .into_iter()
                            .map(|(sum, count)| (count > 0).then_some(sum / count as f64))
                            .collect(),
                    )
                }
            }
            (
                WindowFunctionKind::Count | WindowFunctionKind::Sum | WindowFunctionKind::Avg,
                Int32PartitionWindowState::SharedInt64Dense {
                    min,
                    sums,
                    counts,
                    null_sum,
                    null_count,
                    output,
                },
            ) => match output {
                SharedInt64WindowOutput::Count => WindowFunctionValues::UInt64(
                    keys.iter()
                        .map(|key| {
                            Some(match key {
                                Some(key) => {
                                    counts.get((*key - *min) as usize).copied().unwrap_or(0)
                                }
                                None => *null_count,
                            })
                        })
                        .collect(),
                ),
                SharedInt64WindowOutput::Sum => WindowFunctionValues::Float64(
                    keys.iter()
                        .map(|key| {
                            let (sum, count) = match key {
                                Some(key) => {
                                    let slot = (*key - *min) as usize;
                                    (
                                        sums.get(slot).copied().unwrap_or(0.0),
                                        counts.get(slot).copied().unwrap_or(0),
                                    )
                                }
                                None => (*null_sum, *null_count),
                            };
                            (count > 0).then_some(sum)
                        })
                        .collect(),
                ),
                SharedInt64WindowOutput::Avg => WindowFunctionValues::Float64(
                    keys.iter()
                        .map(|key| {
                            let (sum, count) = match key {
                                Some(key) => {
                                    let slot = (*key - *min) as usize;
                                    (
                                        sums.get(slot).copied().unwrap_or(0.0),
                                        counts.get(slot).copied().unwrap_or(0),
                                    )
                                }
                                None => (*null_sum, *null_count),
                            };
                            (count > 0).then_some(sum / count as f64)
                        })
                        .collect(),
                ),
            },
            _ => {
                return Err(DodamError::UnsupportedSql(
                    "window aggregate state/function mismatch".to_string(),
                ));
            }
        };
        fields.push(Arc::new(Field::new(
            function.output_name.clone(),
            values.data_type(),
            values.is_nullable(),
        )));
        columns.push(values.into_array());
    }
    Ok(RecordBatch::try_new(
        Arc::new(Schema::new(fields)),
        columns,
    )?)
}

fn append_int32_partition_hash_window_columns(
    batch: &RecordBatch,
    window: &WindowProjection,
) -> Result<Option<RecordBatch>> {
    if !window_partition_hash_aggregate_safe(window) {
        return Ok(None);
    }
    let Some(first_function) = window.functions.first() else {
        return Ok(None);
    };
    if !window
        .functions
        .iter()
        .all(|function| function.partition_by == first_function.partition_by)
    {
        return Ok(None);
    }
    let Some(keys) = single_int32_window_partition_keys(batch, first_function)? else {
        return Ok(None);
    };

    let mut fields = batch.schema().fields().to_vec();
    let mut columns = batch.columns().to_vec();
    for function in &window.functions {
        let values = match function.function {
            WindowFunctionKind::Count => {
                let argument_values = function
                    .argument
                    .as_ref()
                    .map(|argument| evaluate_scalar_expression(batch, argument))
                    .transpose()?;
                let present = argument_values
                    .map(evaluated_scalar_present_mask)
                    .transpose()?
                    .unwrap_or_else(|| vec![true; batch.num_rows()]);
                WindowFunctionValues::UInt64(window_hash_count_values_i32(&keys, &present))
            }
            WindowFunctionKind::Sum | WindowFunctionKind::Avg => {
                let argument = function.argument.as_ref().ok_or_else(|| {
                    DodamError::UnsupportedSql("window aggregate requires an argument".to_string())
                })?;
                let values = scalar_as_f64(evaluate_scalar_expression(batch, argument)?)?;
                let (sums, counts) = window_hash_sum_count_values_i32(&keys, &values);
                if function.function == WindowFunctionKind::Sum {
                    WindowFunctionValues::Float64(sums)
                } else {
                    WindowFunctionValues::Float64(
                        sums.into_iter()
                            .zip(counts)
                            .map(|(sum, count)| match (sum, count) {
                                (Some(sum), count) if count > 0 => Some(sum / count as f64),
                                _ => None,
                            })
                            .collect(),
                    )
                }
            }
            WindowFunctionKind::RowNumber
            | WindowFunctionKind::Rank
            | WindowFunctionKind::DenseRank
            | WindowFunctionKind::Lag
            | WindowFunctionKind::Lead => return Ok(None),
        };
        fields.push(Arc::new(Field::new(
            function.output_name.clone(),
            values.data_type(),
            values.is_nullable(),
        )));
        columns.push(values.into_array());
    }
    Ok(Some(RecordBatch::try_new(
        Arc::new(Schema::new(fields)),
        columns,
    )?))
}

fn coalesce_batches(batches: Vec<RecordBatch>) -> Result<Vec<RecordBatch>> {
    if batches.len() <= 1 {
        return Ok(batches);
    }
    let schema = batches[0].schema();
    Ok(vec![concat_batches(&schema, batches.iter())?])
}

enum WindowFunctionValues {
    UInt64(Vec<Option<u64>>),
    Float64(Vec<Option<f64>>),
    Scalar(EvaluatedScalar),
}

impl WindowFunctionValues {
    fn data_type(&self) -> DataType {
        match self {
            Self::UInt64(_) => DataType::UInt64,
            Self::Float64(_) => DataType::Float64,
            Self::Scalar(values) => values.data_type(),
        }
    }

    fn is_nullable(&self) -> bool {
        match self {
            Self::UInt64(values) => values.iter().any(Option::is_none),
            Self::Float64(values) => values.iter().any(Option::is_none),
            Self::Scalar(values) => values.is_nullable(),
        }
    }

    fn into_array(self) -> ArrayRef {
        match self {
            Self::UInt64(values) => Arc::new(UInt64Array::from(values)) as ArrayRef,
            Self::Float64(values) => Arc::new(Float64Array::from(values)) as ArrayRef,
            Self::Scalar(values) => {
                let rows = values.len();
                values.into_array(rows)
            }
        }
    }
}

fn window_function_values(
    batch: &RecordBatch,
    window: &WindowProjectionFunction,
) -> Result<WindowFunctionValues> {
    match window.function {
        WindowFunctionKind::RowNumber if window.partition_by.is_empty() => {
            return Ok(WindowFunctionValues::UInt64(
                (1..=batch.num_rows()).map(|row| Some(row as u64)).collect(),
            ));
        }
        WindowFunctionKind::Sum | WindowFunctionKind::Count | WindowFunctionKind::Avg => {
            return window_aggregate_values(batch, window);
        }
        WindowFunctionKind::Lag | WindowFunctionKind::Lead => {
            let argument = window.argument.as_ref().ok_or_else(|| {
                DodamError::UnsupportedSql(
                    "window offset function requires an argument".to_string(),
                )
            })?;
            let values =
                materialize_evaluated_scalar(evaluate_scalar_expression(batch, argument)?)?;
            return Ok(WindowFunctionValues::Scalar(shift_window_scalar_values(
                batch, window, values,
            )?));
        }
        _ => {}
    }
    let mut values = Vec::with_capacity(batch.num_rows());
    let mut partition_start = 0_usize;
    let mut rank = 1_u64;
    let mut dense_rank = 1_u64;
    for row in 0..batch.num_rows() {
        if row == 0 || !window_partition_equal(batch, window, row - 1, row)? {
            partition_start = row;
            rank = 1;
            dense_rank = 1;
        } else if !window_order_equal(batch, window, row - 1, row)? {
            rank = u64::try_from(row - partition_start + 1)
                .map_err(|_| DodamError::UnsupportedSql("window rank overflow".to_string()))?;
            dense_rank += 1;
        }
        values.push(Some(match window.function {
            WindowFunctionKind::RowNumber => u64::try_from(row - partition_start + 1)
                .map_err(|_| DodamError::UnsupportedSql("row_number overflow".to_string()))?,
            WindowFunctionKind::Rank => rank,
            WindowFunctionKind::DenseRank => dense_rank,
            WindowFunctionKind::Sum | WindowFunctionKind::Count | WindowFunctionKind::Avg => {
                unreachable!("window aggregates are handled before ranking loop")
            }
            WindowFunctionKind::Lag | WindowFunctionKind::Lead => {
                unreachable!("window offset functions are handled before ranking loop")
            }
        }));
    }
    Ok(WindowFunctionValues::UInt64(values))
}

fn materialize_evaluated_scalar(value: EvaluatedScalar) -> Result<EvaluatedScalar> {
    match value {
        EvaluatedScalar::Array(array) => evaluated_array(array.as_ref()),
        other => Ok(other),
    }
}

fn shift_window_scalar_values(
    batch: &RecordBatch,
    window: &WindowProjectionFunction,
    values: EvaluatedScalar,
) -> Result<EvaluatedScalar> {
    let row_count = batch.num_rows();
    if values.len() != row_count {
        return Err(DodamError::UnsupportedSql(format!(
            "window offset input length {} does not match batch rows {}",
            values.len(),
            row_count
        )));
    }
    let ranges = window_partition_ranges_fast(batch, window)?;
    shift_window_scalar_values_with_ranges(row_count, &ranges, window, values)
}

fn shift_window_scalar_values_with_ranges(
    row_count: usize,
    ranges: &[(usize, usize)],
    window: &WindowProjectionFunction,
    values: EvaluatedScalar,
) -> Result<EvaluatedScalar> {
    Ok(match values {
        EvaluatedScalar::Array(_) => unreachable!("window offset scalar was materialized"),
        EvaluatedScalar::Int64(values) => {
            EvaluatedScalar::Int64(shift_copy_window_values(row_count, ranges, window, &values))
        }
        EvaluatedScalar::Float64(values) => {
            EvaluatedScalar::Float64(shift_copy_window_values(row_count, ranges, window, &values))
        }
        EvaluatedScalar::Decimal128 {
            values,
            precision,
            scale,
        } => EvaluatedScalar::Decimal128 {
            values: shift_copy_window_values(row_count, ranges, window, &values),
            precision,
            scale,
        },
        EvaluatedScalar::Utf8(values) => EvaluatedScalar::Utf8(shift_clone_window_values(
            row_count, ranges, window, &values,
        )),
        EvaluatedScalar::Boolean(values) => {
            EvaluatedScalar::Boolean(shift_copy_window_values(row_count, ranges, window, &values))
        }
        EvaluatedScalar::Date32(values) => {
            EvaluatedScalar::Date32(shift_copy_window_values(row_count, ranges, window, &values))
        }
        EvaluatedScalar::TimestampMillisecond(values) => EvaluatedScalar::TimestampMillisecond(
            shift_copy_window_values(row_count, ranges, window, &values),
        ),
    })
}

fn shift_copy_window_values<T: Copy>(
    row_count: usize,
    ranges: &[(usize, usize)],
    window: &WindowProjectionFunction,
    values: &[Option<T>],
) -> Vec<Option<T>> {
    let mut output = vec![None; row_count];
    let offset = window.offset;
    if offset == 0 {
        output.copy_from_slice(values);
        return output;
    }
    for &(start, end) in ranges {
        if end.saturating_sub(start) <= offset {
            continue;
        }
        match window.function {
            WindowFunctionKind::Lag => {
                output[start + offset..end].copy_from_slice(&values[start..end - offset]);
            }
            WindowFunctionKind::Lead => {
                output[start..end - offset].copy_from_slice(&values[start + offset..end]);
            }
            _ => unreachable!("checked by caller"),
        }
    }
    output
}

fn shift_clone_window_values<T: Clone>(
    row_count: usize,
    ranges: &[(usize, usize)],
    window: &WindowProjectionFunction,
    values: &[Option<T>],
) -> Vec<Option<T>> {
    let mut output = vec![None; row_count];
    let offset = window.offset;
    if offset == 0 {
        output.clone_from_slice(values);
        return output;
    }
    for &(start, end) in ranges {
        if end.saturating_sub(start) <= offset {
            continue;
        }
        match window.function {
            WindowFunctionKind::Lag => {
                output[start + offset..end].clone_from_slice(&values[start..end - offset]);
            }
            WindowFunctionKind::Lead => {
                output[start..end - offset].clone_from_slice(&values[start + offset..end]);
            }
            _ => unreachable!("checked by caller"),
        }
    }
    output
}

fn window_aggregate_values(
    batch: &RecordBatch,
    window: &WindowProjectionFunction,
) -> Result<WindowFunctionValues> {
    match window.function {
        WindowFunctionKind::Count => {
            let argument_values = window
                .argument
                .as_ref()
                .map(|argument| evaluate_scalar_expression(batch, argument))
                .transpose()?;
            let present = argument_values
                .map(evaluated_scalar_present_mask)
                .transpose()?
                .unwrap_or_else(|| vec![true; batch.num_rows()]);
            Ok(WindowFunctionValues::UInt64(window_count_values(
                batch, window, &present,
            )?))
        }
        WindowFunctionKind::Sum | WindowFunctionKind::Avg => {
            let argument = window.argument.as_ref().ok_or_else(|| {
                DodamError::UnsupportedSql("window aggregate requires an argument".to_string())
            })?;
            let values = scalar_as_f64(evaluate_scalar_expression(batch, argument)?)?;
            let (sums, counts) = window_sum_count_values(batch, window, &values)?;
            if window.function == WindowFunctionKind::Sum {
                Ok(WindowFunctionValues::Float64(sums))
            } else {
                Ok(WindowFunctionValues::Float64(
                    sums.into_iter()
                        .zip(counts)
                        .map(|(sum, count)| match (sum, count) {
                            (Some(sum), count) if count > 0 => Some(sum / count as f64),
                            _ => None,
                        })
                        .collect(),
                ))
            }
        }
        WindowFunctionKind::RowNumber
        | WindowFunctionKind::Rank
        | WindowFunctionKind::DenseRank
        | WindowFunctionKind::Lag
        | WindowFunctionKind::Lead => {
            unreachable!("ranking functions are not aggregate windows")
        }
    }
}

fn evaluated_scalar_present_mask(value: EvaluatedScalar) -> Result<Vec<bool>> {
    Ok(match value {
        EvaluatedScalar::Array(array) => (0..array.len()).map(|row| !array.is_null(row)).collect(),
        EvaluatedScalar::Int64(values) => values.into_iter().map(|value| value.is_some()).collect(),
        EvaluatedScalar::Float64(values) => {
            values.into_iter().map(|value| value.is_some()).collect()
        }
        EvaluatedScalar::Decimal128 { values, .. } => {
            values.into_iter().map(|value| value.is_some()).collect()
        }
        EvaluatedScalar::Utf8(values) => values.into_iter().map(|value| value.is_some()).collect(),
        EvaluatedScalar::Boolean(values) => {
            values.into_iter().map(|value| value.is_some()).collect()
        }
        EvaluatedScalar::Date32(values) => {
            values.into_iter().map(|value| value.is_some()).collect()
        }
        EvaluatedScalar::TimestampMillisecond(values) => {
            values.into_iter().map(|value| value.is_some()).collect()
        }
    })
}

fn window_count_values(
    batch: &RecordBatch,
    window: &WindowProjectionFunction,
    present: &[bool],
) -> Result<Vec<Option<u64>>> {
    if !window.partition_by.is_empty() && window.order_by.is_empty() {
        return window_hash_count_values(batch, window, present);
    }
    let mut output = vec![None; batch.num_rows()];
    for (start, end) in window_partition_ranges(batch, window)? {
        if window.order_by.is_empty() {
            let count = present[start..end]
                .iter()
                .filter(|present| **present)
                .count() as u64;
            output[start..end].fill(Some(count));
            continue;
        }
        let mut count = 0_u64;
        for row in start..end {
            if present[row] {
                count += 1;
            }
            output[row] = Some(count);
        }
    }
    Ok(output)
}

fn window_sum_count_values(
    batch: &RecordBatch,
    window: &WindowProjectionFunction,
    values: &[Option<f64>],
) -> Result<(Vec<Option<f64>>, Vec<u64>)> {
    if !window.partition_by.is_empty() && window.order_by.is_empty() {
        return window_hash_sum_count_values(batch, window, values);
    }
    let mut sums = vec![None; batch.num_rows()];
    let mut counts = vec![0_u64; batch.num_rows()];
    for (start, end) in window_partition_ranges(batch, window)? {
        if window.order_by.is_empty() {
            let mut sum = 0.0;
            let mut count = 0_u64;
            for value in &values[start..end] {
                if let Some(value) = value {
                    sum += value;
                    count += 1;
                }
            }
            let partition_sum = (count > 0).then_some(sum);
            sums[start..end].fill(partition_sum);
            counts[start..end].fill(count);
            continue;
        }
        let mut sum = 0.0;
        let mut count = 0_u64;
        for row in start..end {
            if let Some(value) = values[row] {
                sum += value;
                count += 1;
            }
            sums[row] = (count > 0).then_some(sum);
            counts[row] = count;
        }
    }
    Ok((sums, counts))
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum WindowPartitionValue {
    Null,
    Boolean(bool),
    Int32(i32),
    Int64(i64),
    UInt32(u32),
    UInt64(u64),
    Float64(u64),
    Date32(i32),
    Date64(i64),
    TimestampMillisecond(i64),
    Utf8(String),
    Decimal128(i128),
    Display(String),
}

type WindowPartitionKey = Vec<WindowPartitionValue>;

fn window_hash_count_values(
    batch: &RecordBatch,
    window: &WindowProjectionFunction,
    present: &[bool],
) -> Result<Vec<Option<u64>>> {
    if let Some(keys) = single_int32_window_partition_keys(batch, window)? {
        return Ok(window_hash_count_values_i32(&keys, present));
    }
    let mut counts = HashMap::<WindowPartitionKey, u64>::new();
    let mut keys = Vec::with_capacity(batch.num_rows());
    for row in 0..batch.num_rows() {
        let key = window_partition_key(batch, &window.partition_by, row)?;
        if present[row] {
            *counts.entry(key.clone()).or_insert(0) += 1;
        } else {
            counts.entry(key.clone()).or_insert(0);
        }
        keys.push(key);
    }
    keys.into_iter()
        .map(|key| {
            counts.get(&key).copied().map(Some).ok_or_else(|| {
                DodamError::UnsupportedSql("window partition count key missing".to_string())
            })
        })
        .collect()
}

fn window_hash_sum_count_values(
    batch: &RecordBatch,
    window: &WindowProjectionFunction,
    values: &[Option<f64>],
) -> Result<(Vec<Option<f64>>, Vec<u64>)> {
    if let Some(keys) = single_int32_window_partition_keys(batch, window)? {
        return Ok(window_hash_sum_count_values_i32(&keys, values));
    }
    let mut aggregates = HashMap::<WindowPartitionKey, (f64, u64)>::new();
    let mut keys = Vec::with_capacity(batch.num_rows());
    for (row, value) in values.iter().enumerate().take(batch.num_rows()) {
        let key = window_partition_key(batch, &window.partition_by, row)?;
        let entry = aggregates.entry(key.clone()).or_insert((0.0, 0));
        if let Some(value) = value {
            entry.0 += value;
            entry.1 += 1;
        }
        keys.push(key);
    }
    let mut sums = Vec::with_capacity(batch.num_rows());
    let mut counts = Vec::with_capacity(batch.num_rows());
    for key in keys {
        let (sum, count) = aggregates.get(&key).copied().ok_or_else(|| {
            DodamError::UnsupportedSql("window partition sum key missing".to_string())
        })?;
        sums.push((count > 0).then_some(sum));
        counts.push(count);
    }
    Ok((sums, counts))
}

fn single_int32_window_partition_keys(
    batch: &RecordBatch,
    window: &WindowProjectionFunction,
) -> Result<Option<Vec<Option<i32>>>> {
    let Some(values) = single_int32_window_partition_array(batch, window)? else {
        return Ok(None);
    };
    Ok(Some(
        (0..values.len())
            .map(|row| (!values.is_null(row)).then(|| values.value(row)))
            .collect(),
    ))
}

fn single_int32_window_partition_array<'a>(
    batch: &'a RecordBatch,
    window: &WindowProjectionFunction,
) -> Result<Option<&'a Int32Array>> {
    let [column] = window.partition_by.as_slice() else {
        return Ok(None);
    };
    let index = output_batch_column_index(batch, column)?;
    let array = batch.column(index);
    let Some(values) = array.as_any().downcast_ref::<Int32Array>() else {
        return Ok(None);
    };
    Ok(Some(values))
}

fn window_hash_count_values_i32(keys: &[Option<i32>], present: &[bool]) -> Vec<Option<u64>> {
    let mut counts = HashMap::<Option<i32>, u64>::new();
    for (key, present) in keys.iter().zip(present) {
        if *present {
            *counts.entry(*key).or_insert(0) += 1;
        } else {
            counts.entry(*key).or_insert(0);
        }
    }
    keys.iter()
        .map(|key| Some(counts.get(key).copied().unwrap_or(0)))
        .collect()
}

fn window_hash_sum_count_values_i32(
    keys: &[Option<i32>],
    values: &[Option<f64>],
) -> (Vec<Option<f64>>, Vec<u64>) {
    let mut aggregates = HashMap::<Option<i32>, (f64, u64)>::new();
    for (key, value) in keys.iter().zip(values) {
        let entry = aggregates.entry(*key).or_insert((0.0, 0));
        if let Some(value) = value {
            entry.0 += value;
            entry.1 += 1;
        }
    }
    let mut sums = Vec::with_capacity(keys.len());
    let mut counts = Vec::with_capacity(keys.len());
    for key in keys {
        let (sum, count) = aggregates.get(key).copied().unwrap_or((0.0, 0));
        sums.push((count > 0).then_some(sum));
        counts.push(count);
    }
    (sums, counts)
}

fn window_partition_key(
    batch: &RecordBatch,
    columns: &[String],
    row: usize,
) -> Result<WindowPartitionKey> {
    columns
        .iter()
        .map(|column| {
            let index = output_batch_column_index(batch, column)?;
            window_partition_value(batch.column(index).as_ref(), row)
        })
        .collect()
}

fn window_partition_value(array: &dyn Array, row: usize) -> Result<WindowPartitionValue> {
    if array.is_null(row) {
        return Ok(WindowPartitionValue::Null);
    }
    match array.data_type() {
        DataType::Boolean => {
            let values = array
                .as_any()
                .downcast_ref::<BooleanArray>()
                .ok_or_else(|| {
                    DodamError::TypeMismatch("expected BooleanArray for Boolean data".to_string())
                })?;
            Ok(WindowPartitionValue::Boolean(values.value(row)))
        }
        DataType::Int32 => {
            let values = array.as_any().downcast_ref::<Int32Array>().ok_or_else(|| {
                DodamError::TypeMismatch("expected Int32Array for Int32 data".to_string())
            })?;
            Ok(WindowPartitionValue::Int32(values.value(row)))
        }
        DataType::Int64 => {
            let values = array.as_any().downcast_ref::<Int64Array>().ok_or_else(|| {
                DodamError::TypeMismatch("expected Int64Array for Int64 data".to_string())
            })?;
            Ok(WindowPartitionValue::Int64(values.value(row)))
        }
        DataType::UInt32 => {
            let values = array
                .as_any()
                .downcast_ref::<UInt32Array>()
                .ok_or_else(|| {
                    DodamError::TypeMismatch("expected UInt32Array for UInt32 data".to_string())
                })?;
            Ok(WindowPartitionValue::UInt32(values.value(row)))
        }
        DataType::UInt64 => {
            let values = array
                .as_any()
                .downcast_ref::<UInt64Array>()
                .ok_or_else(|| {
                    DodamError::TypeMismatch("expected UInt64Array for UInt64 data".to_string())
                })?;
            Ok(WindowPartitionValue::UInt64(values.value(row)))
        }
        DataType::Float64 => {
            let values = array
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| {
                    DodamError::TypeMismatch("expected Float64Array for Float64 data".to_string())
                })?;
            Ok(WindowPartitionValue::Float64(values.value(row).to_bits()))
        }
        DataType::Date32 => {
            let values = array
                .as_any()
                .downcast_ref::<Date32Array>()
                .ok_or_else(|| {
                    DodamError::TypeMismatch("expected Date32Array for Date32 data".to_string())
                })?;
            Ok(WindowPartitionValue::Date32(values.value(row)))
        }
        DataType::Date64 => {
            let values = array
                .as_any()
                .downcast_ref::<Date64Array>()
                .ok_or_else(|| {
                    DodamError::TypeMismatch("expected Date64Array for Date64 data".to_string())
                })?;
            Ok(WindowPartitionValue::Date64(values.value(row)))
        }
        DataType::Timestamp(TimeUnit::Millisecond, None) => {
            let values = array
                .as_any()
                .downcast_ref::<TimestampMillisecondArray>()
                .ok_or_else(|| {
                    DodamError::TypeMismatch(
                        "expected TimestampMillisecondArray for Timestamp(Millisecond) data"
                            .to_string(),
                    )
                })?;
            Ok(WindowPartitionValue::TimestampMillisecond(
                values.value(row),
            ))
        }
        DataType::Utf8 => {
            let values = array
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| {
                    DodamError::TypeMismatch("expected StringArray for Utf8 data".to_string())
                })?;
            Ok(WindowPartitionValue::Utf8(values.value(row).to_string()))
        }
        DataType::Decimal128(_, _) => {
            let values = array
                .as_any()
                .downcast_ref::<Decimal128Array>()
                .ok_or_else(|| {
                    DodamError::TypeMismatch(
                        "expected Decimal128Array for Decimal128 data".to_string(),
                    )
                })?;
            Ok(WindowPartitionValue::Decimal128(values.value(row)))
        }
        _ => Ok(WindowPartitionValue::Display(array_value_to_string(
            array, row,
        )?)),
    }
}

fn window_partition_ranges(
    batch: &RecordBatch,
    window: &WindowProjectionFunction,
) -> Result<Vec<(usize, usize)>> {
    if batch.num_rows() == 0 {
        return Ok(Vec::new());
    }
    if window.partition_by.is_empty() {
        return Ok(vec![(0, batch.num_rows())]);
    }
    let mut ranges = Vec::new();
    let mut start = 0_usize;
    for row in 1..batch.num_rows() {
        if !window_partition_equal(batch, window, row - 1, row)? {
            ranges.push((start, row));
            start = row;
        }
    }
    ranges.push((start, batch.num_rows()));
    Ok(ranges)
}

fn window_partition_ranges_fast(
    batch: &RecordBatch,
    window: &WindowProjectionFunction,
) -> Result<Vec<(usize, usize)>> {
    if let Some(keys) = single_int32_window_partition_keys(batch, window)? {
        if keys.is_empty() {
            return Ok(Vec::new());
        }
        let mut ranges = Vec::new();
        let mut start = 0_usize;
        for row in 1..keys.len() {
            if keys[row - 1] != keys[row] {
                ranges.push((start, row));
                start = row;
            }
        }
        ranges.push((start, keys.len()));
        return Ok(ranges);
    }
    window_partition_ranges(batch, window)
}

fn window_partition_equal(
    batch: &RecordBatch,
    window: &WindowProjectionFunction,
    left: usize,
    right: usize,
) -> Result<bool> {
    rows_equal_on_columns(batch, &window.partition_by, left, right)
}

fn window_order_equal(
    batch: &RecordBatch,
    window: &WindowProjectionFunction,
    left: usize,
    right: usize,
) -> Result<bool> {
    rows_equal_on_columns(batch, &window.order_by, left, right)
}

fn rows_equal_on_columns(
    batch: &RecordBatch,
    columns: &[String],
    left: usize,
    right: usize,
) -> Result<bool> {
    for column in columns {
        let index = output_batch_column_index(batch, column)?;
        let array = batch.column(index);
        if array.is_null(left) != array.is_null(right) {
            return Ok(false);
        }
        if array.is_null(left) {
            continue;
        }
        if !array_values_equal(array.as_ref(), left, right)? {
            return Ok(false);
        }
    }
    Ok(true)
}

fn array_values_equal(array: &dyn Array, left: usize, right: usize) -> Result<bool> {
    match array.data_type() {
        DataType::Boolean => {
            let values = array
                .as_any()
                .downcast_ref::<BooleanArray>()
                .ok_or_else(|| {
                    DodamError::TypeMismatch("expected BooleanArray for Boolean data".to_string())
                })?;
            Ok(values.value(left) == values.value(right))
        }
        DataType::Int32 => {
            let values = array.as_any().downcast_ref::<Int32Array>().ok_or_else(|| {
                DodamError::TypeMismatch("expected Int32Array for Int32 data".to_string())
            })?;
            Ok(values.value(left) == values.value(right))
        }
        DataType::Int64 => {
            let values = array.as_any().downcast_ref::<Int64Array>().ok_or_else(|| {
                DodamError::TypeMismatch("expected Int64Array for Int64 data".to_string())
            })?;
            Ok(values.value(left) == values.value(right))
        }
        DataType::UInt32 => {
            let values = array
                .as_any()
                .downcast_ref::<UInt32Array>()
                .ok_or_else(|| {
                    DodamError::TypeMismatch("expected UInt32Array for UInt32 data".to_string())
                })?;
            Ok(values.value(left) == values.value(right))
        }
        DataType::UInt64 => {
            let values = array
                .as_any()
                .downcast_ref::<UInt64Array>()
                .ok_or_else(|| {
                    DodamError::TypeMismatch("expected UInt64Array for UInt64 data".to_string())
                })?;
            Ok(values.value(left) == values.value(right))
        }
        DataType::Float64 => {
            let values = array
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| {
                    DodamError::TypeMismatch("expected Float64Array for Float64 data".to_string())
                })?;
            Ok(values.value(left) == values.value(right))
        }
        DataType::Date32 => {
            let values = array
                .as_any()
                .downcast_ref::<Date32Array>()
                .ok_or_else(|| {
                    DodamError::TypeMismatch("expected Date32Array for Date32 data".to_string())
                })?;
            Ok(values.value(left) == values.value(right))
        }
        DataType::Date64 => {
            let values = array
                .as_any()
                .downcast_ref::<Date64Array>()
                .ok_or_else(|| {
                    DodamError::TypeMismatch("expected Date64Array for Date64 data".to_string())
                })?;
            Ok(values.value(left) == values.value(right))
        }
        DataType::Timestamp(TimeUnit::Millisecond, None) => {
            let values = array
                .as_any()
                .downcast_ref::<TimestampMillisecondArray>()
                .ok_or_else(|| {
                    DodamError::TypeMismatch(
                        "expected TimestampMillisecondArray for Timestamp(Millisecond) data"
                            .to_string(),
                    )
                })?;
            Ok(values.value(left) == values.value(right))
        }
        DataType::Utf8 => {
            let values = array
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| {
                    DodamError::TypeMismatch("expected StringArray for Utf8 data".to_string())
                })?;
            Ok(values.value(left) == values.value(right))
        }
        DataType::Decimal128(_, _) => {
            let values = array
                .as_any()
                .downcast_ref::<Decimal128Array>()
                .ok_or_else(|| {
                    DodamError::TypeMismatch(
                        "expected Decimal128Array for Decimal128 data".to_string(),
                    )
                })?;
            Ok(values.value(left) == values.value(right))
        }
        _ => Ok(array_value_to_string(array, left)? == array_value_to_string(array, right)?),
    }
}
