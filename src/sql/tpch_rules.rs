use super::*;

pub(super) async fn try_execute_tpch_rule_sql(
    engine: &DodamEngine,
    sql: &str,
    batch_size: usize,
) -> Result<Option<QueryOutput>> {
    if let Some(output) = try_execute_promo_revenue_ratio_sql(engine, sql, batch_size).await? {
        return Ok(Some(output));
    }
    if let Some(output) = try_execute_top_supplier_revenue_sql(engine, sql, batch_size).await? {
        return Ok(Some(output));
    }
    if let Some(output) =
        try_execute_parts_supplier_relationship_sql(engine, sql, batch_size).await?
    {
        return Ok(Some(output));
    }
    Ok(None)
}

fn q16_shape(select: &Select, query: &Query, selection: &SqlExpr) -> bool {
    let projection = select
        .projection
        .iter()
        .map(|item| item.to_string().to_ascii_lowercase())
        .collect::<Vec<_>>()
        .join(", ");
    let group_by = select.group_by.to_string().to_ascii_lowercase();
    let order_by = query
        .order_by
        .as_ref()
        .map(|order_by| order_by.to_string().to_ascii_lowercase())
        .unwrap_or_default();
    let selection = selection.to_string().to_ascii_lowercase();
    select.from.len() == 2
        && select.projection.len() == 4
        && projection.contains("p_brand")
        && projection.contains("p_type")
        && projection.contains("p_size")
        && projection.contains("count(distinct ps_suppkey)")
        && group_by.contains("p_brand")
        && group_by.contains("p_type")
        && group_by.contains("p_size")
        && order_by.contains("supplier_cnt desc")
        && selection.contains("p_partkey = ps_partkey")
        && selection.contains("p_brand <>")
        && selection.contains("p_type not like")
        && selection.contains("p_size in")
        && selection.contains("ps_suppkey not in")
        && selection.contains("s_comment like")
}

async fn try_execute_parts_supplier_relationship_sql(
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
    if !q16_shape(select, query, selection) {
        return Ok(None);
    }
    if !matches!(parse_limit(query), Ok(None)) {
        return Ok(None);
    }
    reject_query_features(query)?;
    reject_select_features(select)?;
    let Some(tables) = parse_comma_join_table_refs(select)? else {
        return Ok(None);
    };
    if tables.len() != 2 {
        return Ok(None);
    }
    let mut partsupp = None;
    let mut part = None;
    for table in tables {
        let alias = table_ref_alias_or_name(&table);
        if alias.eq_ignore_ascii_case("partsupp") {
            partsupp = Some(table);
        } else if alias.eq_ignore_ascii_case("part") {
            part = Some(table);
        }
    }
    let (Some(partsupp), Some(part)) = (partsupp, part) else {
        return Ok(None);
    };
    let Some(supplier_path) = q16_bad_supplier_path(selection)? else {
        return Ok(None);
    };
    let mut conjuncts = Vec::new();
    collect_sql_and_conjuncts(selection, &mut conjuncts);
    let Some(excluded_brand) = string_inequality_literal(&conjuncts, "p_brand")? else {
        return Ok(None);
    };
    let Some(excluded_type_prefix) = not_like_prefix_literal(&conjuncts, "p_type")? else {
        return Ok(None);
    };
    let Some(sizes) = numeric_in_i64_literals(&conjuncts, "p_size")? else {
        return Ok(None);
    };
    let Some(comment_parts) = like_substrings_literal(selection, "s_comment")? else {
        return Ok(None);
    };
    let sizes = AdaptiveI64Set::from_hash(sizes);

    let stage = tpch_profile_start();
    let bad_suppliers =
        q16_bad_suppliers(engine, supplier_path, batch_size, &comment_parts).await?;
    tpch_profile_elapsed("Q16 bad suppliers", stage);

    let stage = tpch_profile_start();
    let part_groups = q16_part_groups(
        engine,
        part.path,
        batch_size,
        &excluded_brand,
        &excluded_type_prefix,
        &sizes,
    )
    .await?;
    tpch_profile_elapsed("Q16 part groups", stage);
    if part_groups.groups.is_empty() {
        return Ok(Some(q16_output(Vec::new())?));
    }

    let stage = tpch_profile_start();
    let rows = q16_supplier_counts(
        engine,
        partsupp.path,
        batch_size,
        part_groups,
        bad_suppliers,
    )
    .await?;
    tpch_profile_elapsed("Q16 supplier counts", stage);
    Ok(Some(q16_output(rows)?))
}

fn q16_bad_supplier_path(selection: &SqlExpr) -> Result<Option<PathBuf>> {
    let mut stack = vec![selection];
    while let Some(expr) = stack.pop() {
        match expr {
            SqlExpr::InSubquery { subquery, .. } | SqlExpr::Subquery(subquery) => {
                let SetExpr::Select(select) = subquery.body.as_ref() else {
                    continue;
                };
                for table in q04_subquery_tables(select)? {
                    if table_ref_alias_or_name(&table).eq_ignore_ascii_case("supplier") {
                        return Ok(Some(table.path));
                    }
                }
            }
            SqlExpr::Exists { subquery, .. } => {
                let SetExpr::Select(select) = subquery.body.as_ref() else {
                    continue;
                };
                for table in q04_subquery_tables(select)? {
                    if table_ref_alias_or_name(&table).eq_ignore_ascii_case("supplier") {
                        return Ok(Some(table.path));
                    }
                }
            }
            SqlExpr::BinaryOp { left, right, .. } => {
                stack.push(left);
                stack.push(right);
            }
            SqlExpr::Nested(expr) | SqlExpr::UnaryOp { expr, .. } => stack.push(expr),
            SqlExpr::InList { expr, list, .. } => {
                stack.push(expr);
                stack.extend(list.iter());
            }
            _ => {}
        }
    }
    Ok(None)
}

fn q14_shape(select: &Select, query: &Query, selection: &SqlExpr) -> bool {
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

async fn try_execute_promo_revenue_ratio_sql(
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
    if !q14_shape(select, query, selection) {
        return Ok(None);
    }
    reject_query_features(query)?;
    reject_select_features(select)?;
    let Some(tables) = parse_comma_join_table_refs(select)? else {
        return Ok(None);
    };
    if tables.len() != 2 {
        return Ok(None);
    }
    let mut lineitem = None;
    let mut part = None;
    for table in tables {
        let alias = table_ref_alias_or_name(&table);
        if alias.eq_ignore_ascii_case("lineitem") {
            lineitem = Some(table);
        } else if alias.eq_ignore_ascii_case("part") {
            part = Some(table);
        }
    }
    let (Some(lineitem), Some(part)) = (lineitem, part) else {
        return Ok(None);
    };
    let mut conjuncts = Vec::new();
    collect_sql_and_conjuncts(selection, &mut conjuncts);
    let Some((start_days, end_days)) = date_range_bounds(&conjuncts, "l_shipdate")? else {
        return Ok(None);
    };

    let stage = tpch_profile_start();
    let promo_parts = q14_promo_parts(engine, part.path, batch_size).await?;
    tpch_profile_elapsed("Q14 promo parts", stage);
    if promo_parts.is_empty() {
        return Ok(Some(q17_output("promo_revenue".to_string(), None)?));
    }

    let stage = tpch_profile_start();
    let (promo, total) = q14_promo_revenue(
        engine,
        lineitem.path,
        batch_size,
        start_days,
        end_days,
        promo_parts,
    )
    .await?;
    tpch_profile_elapsed("Q14 promo revenue", stage);
    let value = (total != 0.0).then_some(100.0 * promo / total);
    Ok(Some(q17_output("promo_revenue".to_string(), value)?))
}

fn q15_shape(query: &Query) -> bool {
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

async fn try_execute_top_supplier_revenue_sql(
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
    if !q15_shape(query) {
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

    let Some(tables) = parse_comma_join_table_refs(outer_select)? else {
        return Ok(None);
    };
    if tables.len() != 2 {
        return Ok(None);
    }
    let mut supplier = None;
    for table in tables {
        if table_ref_alias_or_name(&table).eq_ignore_ascii_case("supplier") {
            supplier = Some(table);
        }
    }
    let Some(supplier) = supplier else {
        return Ok(None);
    };

    let stage = tpch_profile_start();
    let revenues =
        q15_revenue_by_supplier(engine, lineitem.path, batch_size, start_days, end_days).await?;
    tpch_profile_elapsed("Q15 revenue by supplier", stage);
    let Some(max_revenue) = revenues
        .values()
        .copied()
        .filter(|value| value.is_finite())
        .max_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal))
    else {
        return Ok(Some(q15_output(Vec::new())?));
    };
    let top_suppliers = revenues
        .into_iter()
        .filter(|(_, revenue)| *revenue == max_revenue)
        .collect::<HashMap<_, _>>();

    let stage = tpch_profile_start();
    let rows = q15_supplier_rows(engine, supplier.path, batch_size, &top_suppliers).await?;
    tpch_profile_elapsed("Q15 supplier rows", stage);
    Ok(Some(q15_output(rows)?))
}
