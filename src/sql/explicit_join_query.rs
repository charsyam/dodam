use super::*;

pub(super) async fn execute_explicit_join_query(
    engine: &DodamEngine,
    query: SqlQuery,
    join: SqlJoin,
    batch_size: usize,
    options: SqlExecutionOptions,
) -> Result<QueryOutput> {
    let is_aggregate = query.is_aggregate();
    let aggregates = query.aggregates.clone();
    let group_by = query.group_by.clone();
    let join_input_projection = if query.qualified_wildcards.is_empty() {
        join_input_projection_with_expression_filter(&query)?
    } else {
        Projection::All
    };
    let projection_requires_expression = projection_requires_expression_path(&query.expressions);
    let input_order_by = if projection_requires_expression {
        None
    } else {
        query.order_by.as_ref()
    };
    let join_plan = plan_join_inputs(
        &join_input_projection,
        query.filter.as_ref(),
        input_order_by,
        &join.left_alias,
        &join.left_keys,
        &join.right_alias,
        &join.right_keys,
    );
    if let Some(join_graph) = build_logical_explicit_join_graph(engine, &query, &join, &join_plan)?
    {
        log_multi_input_join_optimizer_plan(
            "explicit_join_order",
            &join_graph,
            join_graph.choose_best_plan().as_ref(),
        );
    }
    if is_aggregate
        && let Some(output) = try_execute_join_coalesce_count_sum_aggregate(
            engine, &query, &join, &join_plan, batch_size,
        )
        .await?
    {
        return Ok(output);
    }
    let output_projection = pushed_join_output_projection(&query)?;
    let output_projection_is_final = output_projection == query.projection;
    let stream = engine
        .join_parquet_batches(JoinParquetRequest {
            left_path: query.path.clone(),
            right_path: join.right.path,
            batch_size,
            left_keys: join.left_keys,
            right_keys: join.right_keys,
            left_prefix: join.left_alias.clone(),
            right_prefix: join.right_alias.clone(),
            left_projection: join_plan.left_projection,
            right_projection: join_plan.right_projection,
            left_filter: join_plan.left_filter,
            right_filter: combine_filter_options(join_plan.right_filter, join.right_filter.clone()),
            output_projection,
            join_memory_limit_bytes: join_memory_limit_bytes(options),
            join_algorithm: JoinAlgorithm::Auto,
            join_type: join.join_type,
        })
        .await?;
    if is_aggregate {
        let stream = apply_output_filter_stream(stream, query.filter.clone());
        let stream: SendableBatchStream =
            if let Some(expression_filter) = query.expression_filter.as_ref() {
                Box::new(MemoryExec::new(apply_output_join_expression_filter(
                    collect_batches(stream)?,
                    expression_filter,
                    &[join.left_alias.as_str(), join.right_alias.as_str()],
                )?))
                .execute()?
            } else {
                stream
            };
        let metrics = collect_aggregates_with_optional_expression_views(
            stream,
            2,
            &group_by,
            &aggregates,
            &query.filtered_aggregates,
            &query.aggregate_expressions,
        )?;
        let mut batches = aggregate_metrics_to_batches(&metrics, &group_by, &aggregates)?;
        batches = apply_output_filter(batches, query.having.as_ref())?;
        let has_output_expressions = projection_requires_expression_path(&query.expressions);
        if has_output_expressions {
            batches = apply_output_expression_projection(batches, &query.expressions)?;
        }
        batches = apply_aggregate_output_order_limit(
            batches,
            query.order_by.as_ref(),
            query.limit,
            query.offset,
            &metrics,
            &group_by,
        )?;
        if !has_output_expressions {
            batches = rename_output_batches(batches, &query.aliases)?;
        }
        return Ok(QueryOutput::Aggregate { metrics, batches });
    }
    let mut batches = collect_batches(stream)?;
    batches = apply_output_filter(batches, query.filter.as_ref())?;
    if let Some(expression_filter) = query.expression_filter.as_ref() {
        batches = apply_output_join_expression_filter(
            batches,
            expression_filter,
            &[join.left_alias.as_str(), join.right_alias.as_str()],
        )?;
    }
    if projection_requires_expression {
        if query.distinct {
            batches = apply_output_expression_projection(batches, &query.expressions)?;
            batches = apply_output_distinct(batches, true)?;
            batches = apply_output_order_limit(
                batches,
                query.order_by.as_ref(),
                query.limit,
                query.offset,
            )?;
        } else {
            batches = apply_output_expression_projection_order_limit(
                batches,
                &query.expressions,
                query.order_by.as_ref(),
                query.limit,
                query.offset,
            )?;
        }
    } else if query.distinct {
        if !output_projection_is_final && query.qualified_wildcards.is_empty() {
            batches = apply_output_projection(batches, &query.projection)?;
        }
        batches = apply_output_distinct(batches, true)?;
        batches =
            apply_output_order_limit(batches, query.order_by.as_ref(), query.limit, query.offset)?;
    } else {
        batches =
            apply_output_order_limit(batches, query.order_by.as_ref(), query.limit, query.offset)?;
        if !output_projection_is_final && query.qualified_wildcards.is_empty() {
            batches = apply_output_projection(batches, &query.projection)?;
        }
    }
    if !query.qualified_wildcards.is_empty() {
        batches = apply_qualified_wildcard_projection(
            batches,
            &query.qualified_wildcards,
            &query.projection,
        )?;
    }
    if !projection_requires_expression {
        batches = rename_output_batches(batches, &query.aliases)?;
    }
    Ok(QueryOutput::Scan { batches })
}
