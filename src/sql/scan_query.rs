use super::*;

pub(super) async fn execute_single_table_scan_query(
    engine: &DodamEngine,
    query: SqlQuery,
    batch_size: usize,
) -> Result<QueryOutput> {
    if query.distinct
        && let Some(mut batches) = try_execute_direct_distinct_scan(
            engine,
            DirectDistinctScan {
                path: query.path.clone(),
                projection: query.projection.clone(),
                aliases: query.aliases.clone(),
                filter: query.filter.clone(),
            },
            batch_size,
        )?
    {
        batches =
            apply_output_order_limit(batches, query.order_by.as_ref(), query.limit, query.offset)?;
        return Ok(QueryOutput::Scan { batches });
    }

    if let Some(batches) =
        try_execute_monotonic_row_group_order_limit_scan(engine, &query, batch_size).await?
    {
        let batches = rename_output_batches(batches, &query.aliases)?;
        return Ok(QueryOutput::Scan { batches });
    }

    if !query.distinct
        && monotonic_order_limit_scan_enabled()
        && query.limit.is_some()
        && Path::new(&query.path).exists()
        && let Some(column) = monotonic_stream_limit_column(query.order_by.as_ref())
        && engine
            .parquet_row_groups_monotonic_by_column(query.path.clone(), &column)
            .await?
        && engine
            .parquet_column_monotonic_by_scan(query.path.clone(), &column, batch_size)
            .await?
    {
        let stream = engine
            .scan_parquet_filtered_batches_preserve_order(
                query.path.clone(),
                batch_size,
                query.projection.clone(),
                query.filter.clone(),
            )
            .await?;
        let batches = collect_ordered_stream_limit_batches(stream, query.limit, query.offset)?;
        let batches = rename_output_batches(batches, &query.aliases)?;
        return Ok(QueryOutput::Scan { batches });
    }

    let post_scan_order_by =
        if prefer_post_scan_primitive_desc_topk(engine, &query, query.order_by.as_ref())? {
            query.order_by.clone()
        } else {
            None
        };
    let stream = if query.distinct {
        engine
            .scan_parquet_distinct_batches(
                query.path,
                batch_size,
                scan_limit_with_offset(query.limit, query.offset)?,
                query.projection,
                query.filter,
                query.order_by,
            )
            .await?
    } else if post_scan_order_by.is_some() {
        engine
            .scan_parquet_batches(query.path, batch_size, None, query.projection, query.filter)
            .await?
    } else if let Some(order_by) = query.order_by {
        engine
            .scan_parquet_ordered_batches_by(
                query.path,
                batch_size,
                scan_limit_with_offset(query.limit, query.offset)?,
                query.projection,
                query.filter,
                order_by,
            )
            .await?
    } else {
        engine
            .scan_parquet_batches(
                query.path,
                batch_size,
                scan_limit_with_offset(query.limit, query.offset)?,
                query.projection,
                query.filter,
            )
            .await?
    };
    let batches = apply_output_order_limit(
        collect_batches(stream)?,
        post_scan_order_by.as_ref(),
        query.limit,
        query.offset,
    )?;
    let batches = rename_output_batches(batches, &query.aliases)?;
    Ok(QueryOutput::Scan { batches })
}
