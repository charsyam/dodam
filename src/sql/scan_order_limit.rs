use super::*;

pub(super) async fn try_execute_monotonic_row_group_order_limit_scan(
    engine: &DodamEngine,
    query: &SqlQuery,
    batch_size: usize,
) -> Result<Option<Vec<RecordBatch>>> {
    if query.distinct || query.expression_filter.is_some() || query.limit.is_none() {
        return Ok(None);
    }
    if !Path::new(&query.path).exists() {
        return Ok(None);
    }
    let Some(order_by) = query.order_by.as_ref() else {
        return Ok(None);
    };
    let Some(order_column) = monotonic_stream_limit_column(Some(order_by)) else {
        return Ok(None);
    };
    if !monotonic_order_limit_scan_enabled()
        || !monotonic_row_group_order_limit_scan_enabled()
        || !engine
            .parquet_row_groups_monotonic_by_column(query.path.clone(), &order_column)
            .await?
    {
        return Ok(None);
    }

    let mut scan_projection = scan_projection(&query.projection, query.filter.as_ref());
    add_projection_column_once(&mut scan_projection, order_column.clone());
    let row_groups = engine.parquet_row_group_count(&query.path)?;
    let mut output = Vec::new();
    let mut order_state = MonotonicOrderState::default();
    let mut limiter = OrderedLimitCollector::new(query.limit, query.offset);

    for row_group in 0..row_groups {
        let batches = engine
            .scan_parquet_row_group_batches(
                query.path.clone(),
                batch_size,
                scan_projection.clone(),
                vec![row_group],
            )
            .await?;
        for batch in batches {
            let batch = if let Some(filter) = query.filter.as_ref() {
                filter_batch(batch, filter)?
            } else {
                batch
            };
            if batch.num_rows() == 0 {
                continue;
            }
            if !order_state.consume_batch(&batch, &order_column)? {
                return Ok(None);
            }
            limiter.push_batch(batch, &mut output);
        }
        if limiter.is_complete() {
            let output = apply_output_projection(output, &query.projection)?;
            return Ok(Some(output));
        }
    }

    let output = apply_output_projection(output, &query.projection)?;
    Ok(Some(output))
}

fn monotonic_row_group_order_limit_scan_enabled() -> bool {
    !std::env::var("DODAM_DISABLE_MONOTONIC_ROW_GROUP_ORDER_LIMIT_SCAN")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

pub(super) fn prefer_post_scan_primitive_desc_topk(
    engine: &DodamEngine,
    query: &SqlQuery,
    order_by: Option<&SortKey>,
) -> Result<bool> {
    if query.distinct
        || query.limit.is_none()
        || query.offset != 0
        || !Path::new(&query.path).exists()
    {
        return Ok(false);
    }
    let Some(order_by) = order_by else {
        return Ok(false);
    };
    let [sort] = order_by.expressions.as_slice() else {
        return Ok(false);
    };
    if !sort.descending || sort.nulls_first {
        return Ok(false);
    }
    let Projection::Columns(columns) = &query.projection else {
        return Ok(false);
    };
    if !columns.iter().any(|column| column == &sort.column) {
        return Ok(false);
    }
    let Some(column_types) = engine
        .parquet_direct_primitive_column_types(&query.path, std::slice::from_ref(&sort.column))?
    else {
        return Ok(false);
    };
    Ok(
        choose_primitive_order_limit_strategy(PrimitiveOrderLimitCostInput {
            has_limit: query.limit.is_some(),
            offset: query.offset,
            sort_keys: order_by.expressions.len(),
            descending: sort.descending,
            nulls_first: sort.nulls_first,
            sort_key_projected: true,
            sort_key_is_i64: matches!(column_types.as_slice(), [DirectPrimitiveColumnType::I64]),
        }) == PrimitiveOrderLimitStrategy::PostScanTopK,
    )
}
