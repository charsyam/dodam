use super::*;

fn promo_revenue_ratio_shape(select: &Select, query: &Query, selection: &SqlExpr) -> bool {
    if !matches!(parse_limit(query), Ok(None)) {
        return false;
    }
    let projection = select
        .projection
        .iter()
        .map(|item| item.to_string().to_ascii_lowercase())
        .collect::<Vec<_>>()
        .join(", ");
    let selection = selection.to_string().to_ascii_lowercase();
    select.from.len() == 2
        && select.projection.len() == 1
        && projection.contains("p_type like 'promo%'")
        && projection.contains("l_extendedprice")
        && projection.contains("l_discount")
        && selection.contains("l_partkey = p_partkey")
        && selection.contains("l_shipdate")
}

pub(in crate::sql) async fn try_execute_promo_revenue_ratio_sql(
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
    let Some(selection) = select.selection.as_ref() else {
        return Ok(None);
    };
    if !promo_revenue_ratio_shape(select, query, selection) {
        return Ok(None);
    }
    reject_query_features(query)?;
    reject_select_features(select)?;
    let Some(mut tables) = named_comma_join_tables(select, &["lineitem", "part"])? else {
        return Ok(None);
    };
    let Some(lineitem) = tables.remove("lineitem") else {
        return Ok(None);
    };
    let Some(part) = tables.remove("part") else {
        return Ok(None);
    };
    let mut conjuncts = Vec::new();
    collect_sql_and_conjuncts(selection, &mut conjuncts);
    let Some((start_days, end_days)) = date_range_bounds(&conjuncts, "l_shipdate")? else {
        return Ok(None);
    };

    let stage = tpch_profile_start();
    let promo_parts = promo_part_lookup(engine, part.path, batch_size).await?;
    tpch_profile_elapsed("promo-revenue-ratio promo parts", stage);
    if promo_parts.is_empty() {
        return Ok(Some(single_f64_aggregate_output(
            "promo_revenue".to_string(),
            None,
        )?));
    }

    let stage = tpch_profile_start();
    let (promo, total) = promo_discounted_revenue(
        engine,
        lineitem.path,
        batch_size,
        start_days,
        end_days,
        promo_parts,
    )
    .await?;
    tpch_profile_elapsed("promo-revenue-ratio promo revenue", stage);
    let value = (total != 0.0).then_some(100.0 * promo / total);
    Ok(Some(single_f64_aggregate_output(
        "promo_revenue".to_string(),
        value,
    )?))
}

async fn promo_part_lookup(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
) -> Result<DenseI64BoolLookup> {
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec!["p_partkey".to_string(), "p_type".to_string()]),
            None,
        )
        .await?;
    let mut parts = DenseI64BoolLookup::default();
    while let Some(batch) = stream.next() {
        let batch = batch?;
        let partkeys = batch_column(&batch, "p_partkey")?;
        let types = batch_string_column(&batch, "p_type")?;
        if let Some(partkeys) = partkeys.as_any().downcast_ref::<Int64Array>()
            && partkeys.null_count() == 0
        {
            for row in 0..batch.num_rows() {
                if types.is_valid(row) {
                    parts.insert(partkeys.value(row), types.value(row).starts_with("PROMO"));
                }
            }
            continue;
        }
        for row in 0..batch.num_rows() {
            if types.is_null(row) {
                continue;
            }
            if let Some(partkey) = numeric_i64_value(partkeys, row)? {
                parts.insert(partkey, types.value(row).starts_with("PROMO"));
            }
        }
    }
    Ok(parts)
}

async fn promo_discounted_revenue(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    start_days: i32,
    end_days: i32,
    promo_parts: DenseI64BoolLookup,
) -> Result<(f64, f64)> {
    let promo_parts = Arc::new(promo_parts);
    if std::env::var_os("DODAM_Q14_DISABLE_LATE_MATERIALIZE").is_none() {
        if let Some(result) = engine
            .q14_late_materialized_promo_revenue(
                path.clone(),
                batch_size,
                start_days,
                end_days,
                promo_parts.clone(),
            )
            .await?
        {
            return Ok(result);
        }
    }
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec![
                "l_partkey".to_string(),
                "l_shipdate".to_string(),
                "l_extendedprice".to_string(),
                "l_discount".to_string(),
            ]),
            None,
        )
        .await?;
    parallel_batch_fold(
        &mut stream,
        move |batch| promo_discounted_revenue_batch(batch, start_days, end_days, &promo_parts),
        (0.0, 0.0),
        |total, batch| {
            total.0 += batch.0;
            total.1 += batch.1;
        },
        "promo-revenue-ratio revenue",
    )
}

fn promo_discounted_revenue_batch(
    batch: RecordBatch,
    start_days: i32,
    end_days: i32,
    promo_parts: &DenseI64BoolLookup,
) -> Result<(f64, f64)> {
    let partkeys = batch_column(&batch, "l_partkey")?;
    let shipdates = batch_column(&batch, "l_shipdate")?;
    let extendedprices = batch_column(&batch, "l_extendedprice")?;
    let discounts = batch_column(&batch, "l_discount")?;
    let mut promo = 0.0;
    let mut total = 0.0;
    if let (Some(partkeys), Some(shipdates), Some(extendedprices), Some(discounts)) = (
        partkeys.as_any().downcast_ref::<Int64Array>(),
        shipdates.as_any().downcast_ref::<Date32Array>(),
        decimal_input(extendedprices)?,
        decimal_input(discounts)?,
    ) {
        for row in 0..batch.num_rows() {
            if partkeys.is_null(row)
                || shipdates.is_null(row)
                || extendedprices.is_null(row)
                || discounts.is_null(row)
            {
                continue;
            }
            let shipdate = shipdates.value(row);
            if shipdate < start_days || shipdate >= end_days {
                continue;
            }
            let Some(is_promo) = promo_parts.get(partkeys.value(row)) else {
                continue;
            };
            let value = extendedprices.value(row) * (1.0 - discounts.value(row));
            if is_promo {
                promo += value;
            }
            total += value;
        }
        return Ok((promo, total));
    }
    for row in 0..batch.num_rows() {
        let Some(shipdate) = date32_value(shipdates, row)? else {
            continue;
        };
        if shipdate < start_days || shipdate >= end_days {
            continue;
        }
        let (Some(partkey), Some(extendedprice), Some(discount)) = (
            numeric_i64_value(partkeys, row)?,
            numeric_f64_value(extendedprices, row)?,
            numeric_f64_value(discounts, row)?,
        ) else {
            continue;
        };
        let Some(is_promo) = promo_parts.get(partkey) else {
            continue;
        };
        let value = extendedprice * (1.0 - discount);
        if is_promo {
            promo += value;
        }
        total += value;
    }
    Ok((promo, total))
}
