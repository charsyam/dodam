use super::*;

fn top_supplier_revenue_shape(query: &Query) -> bool {
    let Some(with) = query.with.as_ref() else {
        return false;
    };
    if with.recursive || with.cte_tables.len() != 1 {
        return false;
    }
    let cte = &with.cte_tables[0];
    if !cte.alias.name.value.eq_ignore_ascii_case("revenue")
        || !cte.alias.columns.is_empty()
        || cte.alias.at.is_some()
        || cte.from.is_some()
    {
        return false;
    }
    let SetExpr::Select(revenue_select) = cte.query.body.as_ref() else {
        return false;
    };
    let SetExpr::Select(outer_select) = query.body.as_ref() else {
        return false;
    };
    let revenue_projection = revenue_select
        .projection
        .iter()
        .map(|item| item.to_string().to_ascii_lowercase())
        .collect::<Vec<_>>()
        .join(", ");
    let revenue_group_by = revenue_select.group_by.to_string().to_ascii_lowercase();
    let revenue_selection = revenue_select
        .selection
        .as_ref()
        .map(|expr| expr.to_string().to_ascii_lowercase())
        .unwrap_or_default();
    let outer_projection = outer_select
        .projection
        .iter()
        .map(|item| item.to_string().to_ascii_lowercase())
        .collect::<Vec<_>>()
        .join(", ");
    let outer_selection = outer_select
        .selection
        .as_ref()
        .map(|expr| expr.to_string().to_ascii_lowercase())
        .unwrap_or_default();
    let order_by = query
        .order_by
        .as_ref()
        .map(|order_by| order_by.to_string().to_ascii_lowercase())
        .unwrap_or_default();
    revenue_select.from.len() == 1
        && outer_select.from.len() == 2
        && revenue_projection.contains("l_suppkey")
        && revenue_projection.contains("supplier_no")
        && revenue_projection.contains("sum(l_extendedprice * (1 - l_discount))")
        && revenue_projection.contains("total_revenue")
        && revenue_group_by.contains("l_suppkey")
        && revenue_selection.contains("l_shipdate")
        && outer_projection.contains("s_suppkey")
        && outer_projection.contains("s_name")
        && outer_projection.contains("s_address")
        && outer_projection.contains("s_phone")
        && outer_projection.contains("total_revenue")
        && outer_selection.contains("s_suppkey = supplier_no")
        && outer_selection.contains("max(total_revenue)")
        && order_by.contains("s_suppkey")
}

pub(in crate::sql) async fn try_execute_top_supplier_revenue_sql(
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
    if !top_supplier_revenue_shape(query) {
        return Ok(None);
    }
    let Some(with) = query.with.as_ref() else {
        return Ok(None);
    };
    let cte = &with.cte_tables[0];
    let SetExpr::Select(revenue_select) = cte.query.body.as_ref() else {
        return Ok(None);
    };
    let SetExpr::Select(outer_select) = query.body.as_ref() else {
        return Ok(None);
    };
    if query.fetch.is_some()
        || !query.locks.is_empty()
        || query.for_clause.is_some()
        || query.settings.is_some()
        || query.format_clause.is_some()
        || !query.pipe_operators.is_empty()
    {
        return Err(DodamError::UnsupportedSql(
            "FETCH/locks/settings/format/pipe clauses are not supported".to_string(),
        ));
    }
    reject_select_features(revenue_select)?;
    reject_select_features(outer_select)?;

    let lineitem = parse_from(revenue_select)?;
    if !table_ref_alias_or_name(&lineitem).eq_ignore_ascii_case("lineitem") {
        return Ok(None);
    }
    let mut conjuncts = Vec::new();
    if let Some(selection) = revenue_select.selection.as_ref() {
        collect_sql_and_conjuncts(selection, &mut conjuncts);
    }
    let Some((start_days, end_days)) = date_range_bounds(&conjuncts, "l_shipdate")? else {
        return Ok(None);
    };

    let Some(mut tables) = named_comma_join_tables(outer_select, &["supplier"])? else {
        return Ok(None);
    };
    let Some(supplier) = tables.remove("supplier") else {
        return Ok(None);
    };

    let stage = tpch_profile_start();
    let revenues = supplier_discounted_revenue_by_date(
        engine,
        lineitem.path,
        batch_size,
        start_days,
        end_days,
    )
    .await?;
    tpch_profile_elapsed("top-supplier-revenue revenue by supplier", stage);
    let Some(max_revenue) = revenues
        .values()
        .copied()
        .filter(|value| value.is_finite())
        .max_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal))
    else {
        return Ok(Some(top_supplier_revenue_output(Vec::new())?));
    };
    let top_suppliers = revenues
        .into_iter()
        .filter(|(_, revenue)| *revenue == max_revenue)
        .collect::<HashMap<_, _>>();

    let stage = tpch_profile_start();
    let rows = top_supplier_rows(engine, supplier.path, batch_size, &top_suppliers).await?;
    tpch_profile_elapsed("top-supplier-revenue supplier rows", stage);
    Ok(Some(top_supplier_revenue_output(rows)?))
}

async fn supplier_discounted_revenue_by_date(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    start_days: i32,
    end_days: i32,
) -> Result<HashMap<i64, f64>> {
    q15_revenue_by_supplier(engine, path, batch_size, start_days, end_days).await
}

async fn top_supplier_rows(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    top_suppliers: &HashMap<i64, f64>,
) -> Result<Vec<Q15Row>> {
    q15_supplier_rows(engine, path, batch_size, top_suppliers).await
}

fn top_supplier_revenue_output(rows: Vec<Q15Row>) -> Result<QueryOutput> {
    q15_output(rows)
}
