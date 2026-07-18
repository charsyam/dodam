use super::*;

pub(super) async fn try_execute_projection_expression_sql(
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
    reject_query_features(query)?;
    reject_select_features(select)?;
    if select.from.len() != 1 {
        return Ok(None);
    }
    if select
        .from
        .first()
        .is_some_and(|table| !table.joins.is_empty())
    {
        return Ok(None);
    }
    let path = parse_from(select)?;
    let group_by = parse_group_by(select, path.alias.as_deref())?;
    let parsed_projection = parse_projection(select, &group_by, path.alias.as_deref())?;
    let projection_requires_expression =
        projection_requires_expression_path(&parsed_projection.expressions);
    let (filter, expression_filters) = if let Some(selection) = select.selection.as_ref() {
        if predicate_requires_expression_path(selection) {
            split_subquery_and_expression_filters(
                engine,
                selection,
                path.alias.as_deref(),
                batch_size,
            )
            .await?
        } else {
            (
                Some(parse_filter(
                    selection,
                    &parsed_projection.aliases,
                    path.alias.as_deref(),
                    false,
                )?),
                Vec::new(),
            )
        }
    } else {
        (None, Vec::new())
    };
    let filter_requires_expression = !expression_filters.is_empty();
    if !projection_requires_expression && !filter_requires_expression {
        return Ok(None);
    }
    if parse_distinct(select)? {
        return Err(DodamError::UnsupportedSql(
            "projection expressions currently support only non-aggregate SELECT queries"
                .to_string(),
        ));
    }
    if !parsed_projection.aggregates.is_empty() || !group_by.is_empty() || select.having.is_some() {
        if projection_requires_expression || select.having.is_some() {
            return Ok(None);
        }
        let Some(selection) = select.selection.as_ref() else {
            return Ok(None);
        };
        let mut scan_projection = parsed_projection.projection.clone();
        add_projection_columns(
            &mut scan_projection,
            predicate_expression_columns(selection, path.alias.as_deref())?,
        );
        let stream = engine
            .scan_parquet_batches(path.path, batch_size, None, scan_projection, None)
            .await?;
        let mut batches = collect_batches(stream)?;
        batches = apply_output_expression_filter(batches, selection, path.alias.as_deref())?;
        batches =
            append_aggregate_expression_columns(batches, &parsed_projection.aggregate_expressions)?;
        let stream = Box::new(MemoryExec::new(batches)).execute()?;
        let metrics = if group_by.is_empty() {
            collect_aggregates(stream, 1, &parsed_projection.aggregates)?
        } else {
            collect_grouped_aggregates(stream, 1, &group_by, &parsed_projection.aggregates)?
        };
        let mut batches =
            aggregate_metrics_to_batches(&metrics, &group_by, &parsed_projection.aggregates)?;
        let order_by = parse_order_by(
            query,
            &parsed_projection.aliases,
            &parsed_projection.ordinal_targets,
            path.alias.as_deref(),
        )?;
        let limit = parse_limit(query)?;
        batches = apply_output_order_limit(batches, order_by.as_ref(), limit, 0)?;
        batches = rename_output_batches(batches, &parsed_projection.aliases)?;
        return Ok(Some(QueryOutput::Aggregate { metrics, batches }));
    }

    let order_by = parse_order_by(
        query,
        &parsed_projection.aliases,
        &parsed_projection.ordinal_targets,
        path.alias.as_deref(),
    )?;
    let limit = parse_limit(query)?;
    let mut scan_projection = parsed_projection.projection.clone();
    for expression_filter in &expression_filters {
        add_projection_columns(
            &mut scan_projection,
            predicate_expression_columns(expression_filter, path.alias.as_deref())?,
        );
    }
    if let Some(order_by) = order_by.as_ref() {
        add_projection_columns(
            &mut scan_projection,
            order_by
                .expressions
                .iter()
                .map(|expr| expr.column.clone())
                .collect(),
        );
    }
    if projection_requires_expression
        && !filter_requires_expression
        && let Some(filter) = filter.clone()
        && let Some(mut batches) = try_late_materialized_projection_expression(
            engine,
            path.path.clone(),
            batch_size,
            filter,
            scan_projection.clone(),
            &parsed_projection.expressions,
        )
        .await?
    {
        batches = apply_output_order_limit(batches, order_by.as_ref(), limit, 0)?;
        return Ok(Some(QueryOutput::Scan { batches }));
    }
    let expression_selection = combine_sql_and_conjuncts(expression_filters.clone());
    let (stream, expression_filter_stream_limited) = if let Some(selection) =
        expression_selection.as_ref()
    {
        if let Some(limit) = limit {
            let auto_exact_monotonic_order_column =
                expression_filter_exact_monotonic_stream_limit_auto_enabled()
                    .then(|| monotonic_stream_limit_column(order_by.as_ref()))
                    .flatten();
            if expression_filter_stream_limit_enabled()
                || auto_exact_monotonic_order_column.is_some()
            {
                let monotonic_order_column =
                    auto_exact_monotonic_order_column.clone().or_else(|| {
                        expression_filter_monotonic_stream_limit_enabled()
                            .then(|| monotonic_stream_limit_column(order_by.as_ref()))
                            .flatten()
                    });
                let use_monotonic_stream = if let Some(column) = monotonic_order_column.as_ref() {
                    if expression_filter_exact_monotonic_stream_limit_enabled()
                        || auto_exact_monotonic_order_column.is_some()
                    {
                        engine
                            .parquet_row_groups_monotonic_by_column(path.path.clone(), column)
                            .await?
                            && engine
                                .parquet_column_monotonic_by_scan(
                                    path.path.clone(),
                                    column,
                                    batch_size,
                                )
                                .await?
                    } else {
                        engine
                            .parquet_row_groups_monotonic_by_column(path.path.clone(), column)
                            .await?
                    }
                } else {
                    false
                };
                let stream = if use_monotonic_stream {
                    engine
                        .scan_parquet_batches(path.path, batch_size, None, scan_projection, filter)
                        .await?
                } else if auto_exact_monotonic_order_column.is_some() {
                    engine
                        .scan_parquet_batches(path.path, batch_size, None, scan_projection, filter)
                        .await?
                } else if let Some(order_by) = order_by.clone() {
                    engine
                        .scan_parquet_ordered_batches_by(
                            path.path,
                            batch_size,
                            None,
                            scan_projection,
                            filter,
                            order_by,
                        )
                        .await?
                } else {
                    engine
                        .scan_parquet_batches(path.path, batch_size, None, scan_projection, filter)
                        .await?
                };
                if use_monotonic_stream || auto_exact_monotonic_order_column.is_none() {
                    let batches = collect_expression_filtered_limit_batches(
                        stream,
                        selection,
                        path.alias.as_deref(),
                        limit,
                    )?;
                    (SendableBatchStream::from_batches(batches), true)
                } else {
                    (stream, false)
                }
            } else {
                (
                    engine
                        .scan_parquet_batches(path.path, batch_size, None, scan_projection, filter)
                        .await?,
                    false,
                )
            }
        } else {
            (
                engine
                    .scan_parquet_batches(path.path, batch_size, None, scan_projection, filter)
                    .await?,
                false,
            )
        }
    } else if let Some(order_by) = order_by.clone() {
        (
            engine
                .scan_parquet_ordered_batches_by(
                    path.path,
                    batch_size,
                    limit,
                    scan_projection,
                    filter,
                    order_by,
                )
                .await?,
            false,
        )
    } else {
        (
            engine
                .scan_parquet_batches(path.path, batch_size, limit, scan_projection, filter)
                .await?,
            false,
        )
    };
    let mut batches = collect_batches(stream)?;
    if !expression_filters.is_empty()
        && !expression_filter_stream_limited
        && let Some(selection) = expression_selection.as_ref()
    {
        batches = apply_output_expression_filter(batches, selection, path.alias.as_deref())?;
        batches = apply_output_order_limit(batches, order_by.as_ref(), limit, 0)?;
    }
    let batches = if projection_requires_expression {
        apply_output_expression_projection(batches, &parsed_projection.expressions)?
    } else {
        let batches = apply_output_projection(batches, &parsed_projection.projection)?;
        rename_output_batches(batches, &parsed_projection.aliases)?
    };
    Ok(Some(QueryOutput::Scan { batches }))
}

async fn try_late_materialized_projection_expression(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    filter: FilterExpr,
    payload_projection: Projection,
    expressions: &[ProjectionExpression],
) -> Result<Option<Vec<RecordBatch>>> {
    if !late_projection_enabled() {
        return Ok(None);
    }
    let predicate_columns = filter.referenced_columns();
    let predicate_projection = Projection::Columns(predicate_columns.clone());
    let predicates = PredicateSet::new(Some(filter.clone()));
    let expressions = expressions.to_vec();
    let Some(chunks) = engine
        .late_materialized_parquet_map_pruned_with_policy_view(
            path,
            batch_size,
            predicate_projection,
            payload_projection,
            predicates.pushdown().to_vec(),
            late_projection_row_group_chunk(),
            LateMaterializationPolicy::selective_with_selector_run_ratio(
                late_projection_max_selected_ratio(),
                late_projection_max_selector_run_ratio(),
            )
            .with_io_cost_gate(late_projection_io_cost_gate_enabled()),
            Vec::<RecordBatch>::new,
            {
                let filter = filter.clone();
                let predicate_columns = predicate_columns.clone();
                move |view, selection, _state: &mut Vec<RecordBatch>| {
                    let mask = if let Some(mask) =
                        evaluate_projected_view_filter_mask(view, &predicate_columns, &filter)?
                    {
                        mask
                    } else {
                        if !expression_aggregate_row_at_time_fallback_enabled() {
                            return Ok(None);
                        }
                        let Some(batch) = view.try_record_batch() else {
                            return Ok(None);
                        };
                        evaluate_filter_mask(batch, &filter)?
                    };
                    push_boolean_mask_selection(&mask, selection);
                    Ok(Some(()))
                }
            },
            move |view, state: &mut Vec<RecordBatch>| {
                let Some(batch) = view.try_record_batch() else {
                    return Ok(None);
                };
                state.push(batch.clone());
                Ok(Some(()))
            },
            {
                let expressions = expressions.clone();
                move |state, _metrics| {
                    if state.is_empty() {
                        return Ok(None);
                    }
                    Ok(Some(apply_output_expression_projection(
                        state,
                        &expressions,
                    )?))
                }
            },
        )
        .await?
    else {
        return Ok(None);
    };
    let mut batches = Vec::new();
    for chunk in chunks {
        batches.extend(chunk.output);
    }
    Ok(Some(batches))
}

fn late_projection_enabled() -> bool {
    !std::env::var("DODAM_DISABLE_LATE_PROJECTION")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

fn late_projection_row_group_chunk() -> usize {
    std::env::var("DODAM_LATE_PROJECTION_ROW_GROUP_CHUNK")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(4)
}

fn late_projection_max_selected_ratio() -> f64 {
    std::env::var("DODAM_LATE_PROJECTION_MAX_SELECTED_RATIO")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite())
        .unwrap_or(0.35)
}

fn late_projection_max_selector_run_ratio() -> f64 {
    std::env::var("DODAM_LATE_PROJECTION_MAX_SELECTOR_RUN_RATIO")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite())
        .unwrap_or(0.25)
}

fn late_projection_io_cost_gate_enabled() -> bool {
    !std::env::var("DODAM_DISABLE_LATE_PROJECTION_IO_COST_GATE")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

fn expression_filter_stream_limit_enabled() -> bool {
    std::env::var("DODAM_ENABLE_EXPRESSION_FILTER_STREAM_LIMIT")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

fn expression_filter_monotonic_stream_limit_enabled() -> bool {
    std::env::var("DODAM_ENABLE_EXPRESSION_FILTER_MONOTONIC_STREAM_LIMIT")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

fn expression_filter_exact_monotonic_stream_limit_enabled() -> bool {
    std::env::var("DODAM_ENABLE_EXPRESSION_FILTER_EXACT_MONOTONIC_STREAM_LIMIT")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

fn expression_filter_exact_monotonic_stream_limit_auto_enabled() -> bool {
    !std::env::var("DODAM_DISABLE_EXPRESSION_FILTER_EXACT_MONOTONIC_AUTO")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

pub(super) fn monotonic_stream_limit_column(order_by: Option<&SortKey>) -> Option<String> {
    let [sort] = order_by?.expressions.as_slice() else {
        return None;
    };
    (!sort.descending && !sort.nulls_first).then(|| sort.column.clone())
}

pub(super) fn monotonic_order_limit_scan_enabled() -> bool {
    !std::env::var("DODAM_DISABLE_MONOTONIC_ORDER_LIMIT_SCAN")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

pub(super) fn small_dynamic_in_list_row_filter_limit() -> usize {
    std::env::var("DODAM_DYNAMIC_IN_LIST_ROW_FILTER_LIMIT")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(64)
}
