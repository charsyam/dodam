use super::*;

fn parts_supplier_relationship_shape(select: &Select, query: &Query, selection: &SqlExpr) -> bool {
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

pub(in crate::sql) async fn try_execute_parts_supplier_relationship_sql(
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
    if !parts_supplier_relationship_shape(select, query, selection) {
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
    let Some(supplier_path) = parts_supplier_relationship_bad_supplier_path(selection)? else {
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
        supplier_comment_exclusion_keys(engine, supplier_path, batch_size, &comment_parts).await?;
    tpch_profile_elapsed("parts-supplier-relationship bad suppliers", stage);

    let stage = tpch_profile_start();
    let part_groups = parts_supplier_relationship_part_groups(
        engine,
        part.path,
        batch_size,
        &excluded_brand,
        &excluded_type_prefix,
        &sizes,
    )
    .await?;
    tpch_profile_elapsed("parts-supplier-relationship part groups", stage);
    if part_groups.groups.is_empty() {
        return Ok(Some(parts_supplier_relationship_output(Vec::new())?));
    }

    let stage = tpch_profile_start();
    let rows = parts_supplier_relationship_supplier_counts(
        engine,
        partsupp.path,
        batch_size,
        part_groups,
        bad_suppliers,
    )
    .await?;
    tpch_profile_elapsed("parts-supplier-relationship supplier counts", stage);
    Ok(Some(parts_supplier_relationship_output(rows)?))
}

async fn supplier_comment_exclusion_keys(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    comment_parts: &[String],
) -> Result<ExcludedSuppliers> {
    excluded_suppliers_by_comment_like(engine, path, batch_size, comment_parts).await
}

async fn parts_supplier_relationship_part_groups(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    excluded_brand: &str,
    excluded_type_prefix: &str,
    sizes: &AdaptiveI64Set,
) -> Result<PartGroups> {
    part_groups_by_attributes(
        engine,
        path,
        batch_size,
        excluded_brand,
        excluded_type_prefix,
        sizes,
    )
    .await
}

async fn parts_supplier_relationship_supplier_counts(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    part_groups: PartGroups,
    bad_suppliers: ExcludedSuppliers,
) -> Result<Vec<PartSupplierGroupRow>> {
    distinct_supplier_counts_by_part_group(engine, path, batch_size, part_groups, bad_suppliers)
        .await
}

fn parts_supplier_relationship_output(rows: Vec<PartSupplierGroupRow>) -> Result<QueryOutput> {
    part_supplier_group_output(rows)
}

fn parts_supplier_relationship_bad_supplier_path(selection: &SqlExpr) -> Result<Option<PathBuf>> {
    let mut stack = vec![selection];
    while let Some(expr) = stack.pop() {
        match expr {
            SqlExpr::InSubquery { subquery, .. } | SqlExpr::Subquery(subquery) => {
                let SetExpr::Select(select) = subquery.body.as_ref() else {
                    continue;
                };
                for table in parse_select_table_refs(select)? {
                    if table_ref_alias_or_name(&table).eq_ignore_ascii_case("supplier") {
                        return Ok(Some(table.path));
                    }
                }
            }
            SqlExpr::Exists { subquery, .. } => {
                let SetExpr::Select(select) = subquery.body.as_ref() else {
                    continue;
                };
                for table in parse_select_table_refs(select)? {
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
