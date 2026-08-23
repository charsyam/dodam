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
    let matched_parts = collect_i64_utf8_prefix_bool_lookup(
        engine,
        part.path,
        batch_size,
        "p_partkey",
        "p_type",
        "PROMO",
    )
    .await?;
    tpch_profile_elapsed("promo-revenue-ratio matched parts", stage);
    if matched_parts.is_empty() {
        return Ok(Some(single_f64_aggregate_output(
            "promo_revenue".to_string(),
            None,
        )?));
    }

    let stage = tpch_profile_start();
    let (promo, total) = bool_lookup_discounted_revenue(
        engine,
        lineitem.path,
        batch_size,
        start_days,
        end_days,
        matched_parts,
    )
    .await?;
    tpch_profile_elapsed("promo-revenue-ratio promo revenue", stage);
    let value = (total != 0.0).then_some(100.0 * promo / total);
    Ok(Some(single_f64_aggregate_output(
        "promo_revenue".to_string(),
        value,
    )?))
}

async fn bool_lookup_discounted_revenue(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    start_days: i32,
    end_days: i32,
    matched_parts: DenseI64BoolLookup,
) -> Result<(f64, f64)> {
    let matched_parts = Arc::new(matched_parts);
    if let Some(result) = engine
        .q14_late_materialized_promo_revenue(
            path.clone(),
            batch_size,
            start_days,
            end_days,
            matched_parts.clone(),
        )
        .await?
    {
        return Ok(result);
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
        move |batch| {
            discounted_revenue_by_i64_bool_lookup_batch(
                batch,
                "l_partkey",
                "l_shipdate",
                "l_extendedprice",
                "l_discount",
                start_days,
                end_days,
                &matched_parts,
            )
        },
        (0.0, 0.0),
        |total, batch| {
            total.0 += batch.0;
            total.1 += batch.1;
        },
        "promo-revenue-ratio revenue",
    )
}
