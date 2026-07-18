use super::*;

pub(super) async fn execute_single_table_aggregate_query(
    engine: &DodamEngine,
    query: SqlQuery,
    batch_size: usize,
) -> Result<QueryOutput> {
    let aggregates = query.aggregates.clone();
    let group_by = query.group_by.clone();
    let metrics = if let Some(metrics) =
        try_collect_direct_monotonic_count_distinct(engine, &query, batch_size)?
    {
        metrics
    } else if !query.aggregate_expressions.is_empty() || !query.filtered_aggregates.is_empty() {
        if let Some(metrics) = try_collect_filtered_decimal_product_sum_scan_fold(
            engine,
            query.path.clone(),
            batch_size,
            query.filter.clone(),
            &aggregates,
            &query.aggregate_expressions,
        )
        .await?
        {
            metrics
        } else if let Some(metrics) = try_collect_expression_aggregate_fused_dictionary_selected(
            engine,
            query.path.clone(),
            batch_size,
            query.filter.clone(),
            &group_by,
            &aggregates,
            &query.aggregate_expressions,
            query.order_by.is_some(),
            expression_aggregate_output_limit(
                &group_by,
                query.order_by.as_ref(),
                query.limit,
                query.offset,
            ),
        )
        .await?
        {
            metrics
        } else if let Some(metrics) = try_collect_expression_aggregate_late_materialized(
            engine,
            query.path.clone(),
            batch_size,
            query.filter.clone(),
            &group_by,
            &aggregates,
            &query.aggregate_expressions,
            query.order_by.is_some(),
            expression_aggregate_output_limit(
                &group_by,
                query.order_by.as_ref(),
                query.limit,
                query.offset,
            ),
        )
        .await?
        {
            metrics
        } else if let Some(metrics) = try_collect_expression_aggregate_scan_fold(
            engine,
            query.path.clone(),
            batch_size,
            query.projection.clone(),
            query.filter.clone(),
            &group_by,
            &aggregates,
            &query.aggregate_expressions,
            query.order_by.is_some(),
            expression_aggregate_output_limit(
                &group_by,
                query.order_by.as_ref(),
                query.limit,
                query.offset,
            ),
        )
        .await?
        {
            metrics
        } else if let Some(metrics) = try_collect_expression_aggregate_row_group_map(
            engine,
            query.path.clone(),
            batch_size,
            query.projection.clone(),
            query.filter.clone(),
            &group_by,
            &aggregates,
            &query.aggregate_expressions,
        )
        .await?
        {
            metrics
        } else {
            let filtered_aggregates = simplify_filtered_aggregates_with_parquet_stats(
                engine,
                &query.path,
                &query.filtered_aggregates,
            )?;
            let stream = engine
                .scan_parquet_batches(
                    query.path,
                    batch_size,
                    None,
                    query.projection.clone(),
                    query.filter,
                )
                .await?;
            collect_aggregates_with_optional_expression_views(
                stream,
                1,
                &group_by,
                &aggregates,
                &filtered_aggregates,
                &query.aggregate_expressions,
            )?
        }
    } else if query.group_by.is_empty() {
        engine
            .aggregate_parquet(query.path, batch_size, aggregates.clone(), query.filter)
            .await?
    } else {
        engine
            .aggregate_parquet_grouped(
                query.path,
                batch_size,
                aggregates.clone(),
                group_by.clone(),
                query.filter,
            )
            .await?
    };
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
    Ok(QueryOutput::Aggregate { metrics, batches })
}
