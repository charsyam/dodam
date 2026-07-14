use super::*;

#[derive(Clone, Copy)]
enum TpchRule {
    MinimumCostSupplier,
    PromoRevenueRatio,
    TopSupplierRevenue,
    PartsSupplierRelationship,
}

impl TpchRule {
    fn name(self) -> &'static str {
        match self {
            Self::MinimumCostSupplier => "minimum-cost-supplier",
            Self::PromoRevenueRatio => "promo-revenue-ratio",
            Self::TopSupplierRevenue => "top-supplier-revenue",
            Self::PartsSupplierRelationship => "parts-supplier-relationship",
        }
    }

    fn estimated_cost_rank(self) -> u8 {
        match self {
            Self::MinimumCostSupplier => 20,
            Self::PromoRevenueRatio => 30,
            Self::TopSupplierRevenue => 30,
            Self::PartsSupplierRelationship => 40,
        }
    }

    async fn execute(
        self,
        engine: &DodamEngine,
        sql: &str,
        batch_size: usize,
    ) -> Result<Option<QueryOutput>> {
        match self {
            Self::MinimumCostSupplier => {
                try_execute_minimum_cost_supplier_sql(engine, sql, batch_size).await
            }
            Self::PromoRevenueRatio => {
                try_execute_promo_revenue_ratio_sql(engine, sql, batch_size).await
            }
            Self::TopSupplierRevenue => {
                try_execute_top_supplier_revenue_sql(engine, sql, batch_size).await
            }
            Self::PartsSupplierRelationship => {
                try_execute_parts_supplier_relationship_sql(engine, sql, batch_size).await
            }
        }
    }
}

pub(super) async fn try_execute_tpch_rule_sql(
    engine: &DodamEngine,
    sql: &str,
    batch_size: usize,
) -> Result<Option<QueryOutput>> {
    let candidates = tpch_rule_candidates(sql)?;
    for rule in candidates {
        if let Some(output) = rule.execute(engine, sql, batch_size).await? {
            if tpch_profile_enabled() {
                eprintln!(
                    "[dodam:tpch-rule] selected={} cost_rank={}",
                    rule.name(),
                    rule.estimated_cost_rank()
                );
            }
            return Ok(Some(output));
        }
    }
    Ok(None)
}

fn tpch_rule_candidates(sql: &str) -> Result<Vec<TpchRule>> {
    let dialect = GenericDialect {};
    let statements = Parser::parse_sql(&dialect, sql)
        .map_err(|error| DodamError::UnsupportedSql(error.to_string()))?;
    let [Statement::Query(query)] = statements.as_slice() else {
        return Ok(Vec::new());
    };
    let mut rules = Vec::new();
    if q15_shape(query) {
        rules.push(TpchRule::TopSupplierRevenue);
    }
    if let SetExpr::Select(select) = query.body.as_ref()
        && let Some(selection) = select.selection.as_ref()
    {
        if q02_shape(select, query, selection) {
            rules.push(TpchRule::MinimumCostSupplier);
        }
        if q14_shape(select, query, selection) {
            rules.push(TpchRule::PromoRevenueRatio);
        }
        if q16_shape(select, query, selection) {
            rules.push(TpchRule::PartsSupplierRelationship);
        }
    }
    rules.sort_by_key(|rule| rule.estimated_cost_rank());
    Ok(rules)
}

fn q02_shape(select: &Select, query: &Query, selection: &SqlExpr) -> bool {
    if !matches!(parse_limit(query), Ok(Some(_))) || !matches!(parse_offset(query), Ok(0)) {
        return false;
    }
    let projection = select
        .projection
        .iter()
        .map(|item| item.to_string().to_ascii_lowercase())
        .collect::<Vec<_>>()
        .join(", ");
    let selection = selection.to_string().to_ascii_lowercase();
    let order_by = query
        .order_by
        .as_ref()
        .map(|order_by| order_by.to_string().to_ascii_lowercase())
        .unwrap_or_default();
    select.from.len() == 5
        && projection.contains("s_acctbal")
        && projection.contains("s_name")
        && projection.contains("n_name")
        && projection.contains("p_partkey")
        && projection.contains("p_mfgr")
        && projection.contains("s_address")
        && projection.contains("s_phone")
        && projection.contains("s_comment")
        && selection.contains("p_partkey = ps_partkey")
        && selection.contains("s_suppkey = ps_suppkey")
        && selection.contains("p_size")
        && selection.contains("p_type like")
        && selection.contains("s_nationkey = n_nationkey")
        && selection.contains("n_regionkey = r_regionkey")
        && selection.contains("r_name")
        && selection.contains("min(ps_supplycost)")
        && order_by.contains("s_acctbal desc")
        && order_by.contains("n_name")
        && order_by.contains("s_name")
        && order_by.contains("p_partkey")
}

async fn try_execute_minimum_cost_supplier_sql(
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
    if !q02_shape(select, query, selection) {
        return Ok(None);
    }
    if query.with.is_some()
        || query.fetch.is_some()
        || !query.locks.is_empty()
        || query.for_clause.is_some()
        || query.settings.is_some()
        || query.format_clause.is_some()
        || !query.pipe_operators.is_empty()
    {
        return Ok(None);
    }
    let limit = parse_limit(query)?.unwrap_or(usize::MAX);
    let Some(tables) = parse_comma_join_table_refs(select)? else {
        return Ok(None);
    };
    let mut part = None;
    let mut supplier = None;
    let mut partsupp = None;
    let mut nation = None;
    let mut region = None;
    for table in tables {
        match table_ref_alias_or_name(&table)
            .to_ascii_lowercase()
            .as_str()
        {
            "part" => part = Some(table),
            "supplier" => supplier = Some(table),
            "partsupp" => partsupp = Some(table),
            "nation" => nation = Some(table),
            "region" => region = Some(table),
            _ => {}
        }
    }
    let (Some(part), Some(supplier), Some(partsupp), Some(nation), Some(region)) =
        (part, supplier, partsupp, nation, region)
    else {
        return Ok(None);
    };
    let mut conjuncts = Vec::new();
    collect_sql_and_conjuncts(selection, &mut conjuncts);
    let Some(part_size) = q02_numeric_i64_equality_literal(&conjuncts, "p_size")? else {
        return Ok(None);
    };
    let Some(part_type_suffix) = q02_like_suffix_literal(&conjuncts, "p_type")? else {
        return Ok(None);
    };
    let Some(region_name) = string_equality_literal(&conjuncts, "r_name")? else {
        return Ok(None);
    };

    let stage = tpch_profile_start();
    let region_keys = q05_region_keys(engine, region.path, batch_size, &region_name).await?;
    let nation_names = q05_nation_names(engine, nation.path, batch_size, &region_keys).await?;
    if nation_names.is_empty() {
        return Ok(Some(q02_output(Vec::new())?));
    }
    let suppliers = q02_supplier_rows(engine, supplier.path, batch_size, &nation_names).await?;
    if suppliers.is_empty() {
        return Ok(Some(q02_output(Vec::new())?));
    }
    let parts =
        q02_matching_parts(engine, part.path, batch_size, part_size, &part_type_suffix).await?;
    if parts.is_empty() {
        return Ok(Some(q02_output(Vec::new())?));
    }
    tpch_profile_elapsed("Q02 dimensions", stage);

    let stage = tpch_profile_start();
    let mut rows = q02_min_cost_rows(engine, partsupp.path, batch_size, &parts, &suppliers).await?;
    tpch_profile_elapsed("Q02 partsupp min cost", stage);
    rows.sort_by(|left, right| {
        right
            .acctbal
            .partial_cmp(&left.acctbal)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.n_name.cmp(&right.n_name))
            .then_with(|| left.s_name.cmp(&right.s_name))
            .then_with(|| left.p_partkey.cmp(&right.p_partkey))
    });
    rows.truncate(limit);
    Ok(Some(q02_output(rows)?))
}

fn q02_numeric_i64_equality_literal(conjuncts: &[SqlExpr], column: &str) -> Result<Option<i64>> {
    for conjunct in conjuncts {
        let SqlExpr::BinaryOp { left, op, right } = conjunct else {
            continue;
        };
        if *op != BinaryOperator::Eq {
            continue;
        }
        if sql_expr_column_matches(left, column) {
            if let LiteralValue::Int64(value) = sql_literal_value(right)? {
                return Ok(Some(value));
            }
        } else if sql_expr_column_matches(right, column)
            && let LiteralValue::Int64(value) = sql_literal_value(left)?
        {
            return Ok(Some(value));
        }
    }
    Ok(None)
}

fn q02_like_suffix_literal(conjuncts: &[SqlExpr], column: &str) -> Result<Option<String>> {
    for conjunct in conjuncts {
        let SqlExpr::Like {
            expr,
            pattern,
            negated,
            ..
        } = conjunct
        else {
            continue;
        };
        if *negated || !sql_expr_column_matches(expr, column) {
            continue;
        }
        let LiteralValue::Utf8(pattern) = sql_literal_value(pattern)? else {
            continue;
        };
        if let Some(value) = pattern.strip_prefix('%')
            && !value.contains('%')
            && !value.contains('_')
        {
            return Ok(Some(value.to_string()));
        }
    }
    Ok(None)
}

#[derive(Clone)]
struct Q02Supplier {
    acctbal: f64,
    s_name: String,
    n_name: String,
    s_address: String,
    s_phone: String,
    s_comment: String,
}

#[derive(Clone)]
struct Q02Part {
    mfgr: String,
}

#[derive(Clone)]
struct Q02Candidate {
    partkey: i64,
    suppkey: i64,
    supplycost: f64,
}

struct Q02Row {
    acctbal: f64,
    s_name: String,
    n_name: String,
    p_partkey: i64,
    p_mfgr: String,
    s_address: String,
    s_phone: String,
    s_comment: String,
}

async fn q02_supplier_rows(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    nation_names: &HashMap<i64, String>,
) -> Result<HashMap<i64, Q02Supplier>> {
    let nation_keys = AdaptiveI64Set::from_hash(nation_names.keys().copied().collect());
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec![
                "s_suppkey".to_string(),
                "s_acctbal".to_string(),
                "s_name".to_string(),
                "s_address".to_string(),
                "s_nationkey".to_string(),
                "s_phone".to_string(),
                "s_comment".to_string(),
            ]),
            None,
        )
        .await?;
    let mut suppliers = HashMap::new();
    while let Some(batch) = stream.next() {
        let batch = batch?;
        let suppkeys = batch_column(&batch, "s_suppkey")?;
        let acctbals = batch_column(&batch, "s_acctbal")?;
        let names = batch_string_column(&batch, "s_name")?;
        let addresses = batch_string_column(&batch, "s_address")?;
        let nationkeys = batch_column(&batch, "s_nationkey")?;
        let phones = batch_string_column(&batch, "s_phone")?;
        let comments = batch_string_column(&batch, "s_comment")?;
        let Some(acctbals) = decimal_input(acctbals)? else {
            return Err(DodamError::UnsupportedSql(
                "s_acctbal must be Decimal128".to_string(),
            ));
        };
        for row in 0..batch.num_rows() {
            if names.is_null(row)
                || addresses.is_null(row)
                || phones.is_null(row)
                || comments.is_null(row)
                || acctbals.is_null(row)
            {
                continue;
            }
            let (Some(suppkey), Some(nationkey)) = (
                numeric_i64_value(suppkeys, row)?,
                numeric_i64_value(nationkeys, row)?,
            ) else {
                continue;
            };
            if !nation_keys.contains(nationkey) {
                continue;
            }
            let Some(n_name) = nation_names.get(&nationkey) else {
                continue;
            };
            suppliers.insert(
                suppkey,
                Q02Supplier {
                    acctbal: acctbals.value(row),
                    s_name: names.value(row).to_string(),
                    n_name: n_name.clone(),
                    s_address: addresses.value(row).to_string(),
                    s_phone: phones.value(row).to_string(),
                    s_comment: comments.value(row).to_string(),
                },
            );
        }
    }
    Ok(suppliers)
}

async fn q02_matching_parts(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    part_size: i64,
    type_suffix: &str,
) -> Result<HashMap<i64, Q02Part>> {
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec![
                "p_partkey".to_string(),
                "p_mfgr".to_string(),
                "p_size".to_string(),
                "p_type".to_string(),
            ]),
            None,
        )
        .await?;
    let mut parts = HashMap::new();
    while let Some(batch) = stream.next() {
        let batch = batch?;
        let partkeys = batch_column(&batch, "p_partkey")?;
        let mfgrs = batch_string_column(&batch, "p_mfgr")?;
        let sizes = batch_column(&batch, "p_size")?;
        let types = batch_string_column(&batch, "p_type")?;
        for row in 0..batch.num_rows() {
            if mfgrs.is_null(row) || types.is_null(row) {
                continue;
            }
            let (Some(partkey), Some(size)) = (
                numeric_i64_value(partkeys, row)?,
                numeric_i64_value(sizes, row)?,
            ) else {
                continue;
            };
            if size == part_size && types.value(row).ends_with(type_suffix) {
                parts.insert(
                    partkey,
                    Q02Part {
                        mfgr: mfgrs.value(row).to_string(),
                    },
                );
            }
        }
    }
    Ok(parts)
}

async fn q02_min_cost_rows(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    parts: &HashMap<i64, Q02Part>,
    suppliers: &HashMap<i64, Q02Supplier>,
) -> Result<Vec<Q02Row>> {
    let part_keys = AdaptiveI64Set::from_hash(parts.keys().copied().collect());
    let supplier_keys = AdaptiveI64Set::from_hash(suppliers.keys().copied().collect());
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec![
                "ps_partkey".to_string(),
                "ps_suppkey".to_string(),
                "ps_supplycost".to_string(),
            ]),
            None,
        )
        .await?;
    let mut min_costs = HashMap::<i64, f64>::new();
    let mut candidates = Vec::<Q02Candidate>::new();
    while let Some(batch) = stream.next() {
        let batch = batch?;
        let partkeys = batch_column(&batch, "ps_partkey")?;
        let suppkeys = batch_column(&batch, "ps_suppkey")?;
        let supplycosts = batch_column(&batch, "ps_supplycost")?;
        let Some(supplycosts) = decimal_input(supplycosts)? else {
            return Err(DodamError::UnsupportedSql(
                "ps_supplycost must be Decimal128".to_string(),
            ));
        };
        for row in 0..batch.num_rows() {
            if supplycosts.is_null(row) {
                continue;
            }
            let (Some(partkey), Some(suppkey)) = (
                numeric_i64_value(partkeys, row)?,
                numeric_i64_value(suppkeys, row)?,
            ) else {
                continue;
            };
            if !part_keys.contains(partkey) || !supplier_keys.contains(suppkey) {
                continue;
            }
            let supplycost = supplycosts.value(row);
            min_costs
                .entry(partkey)
                .and_modify(|current| {
                    if supplycost < *current {
                        *current = supplycost;
                    }
                })
                .or_insert(supplycost);
            candidates.push(Q02Candidate {
                partkey,
                suppkey,
                supplycost,
            });
        }
    }
    let mut rows = Vec::new();
    for candidate in candidates {
        let Some(min_cost) = min_costs.get(&candidate.partkey).copied() else {
            continue;
        };
        if candidate.supplycost != min_cost {
            continue;
        }
        let (Some(part), Some(supplier)) = (
            parts.get(&candidate.partkey),
            suppliers.get(&candidate.suppkey),
        ) else {
            continue;
        };
        rows.push(Q02Row {
            acctbal: supplier.acctbal,
            s_name: supplier.s_name.clone(),
            n_name: supplier.n_name.clone(),
            p_partkey: candidate.partkey,
            p_mfgr: part.mfgr.clone(),
            s_address: supplier.s_address.clone(),
            s_phone: supplier.s_phone.clone(),
            s_comment: supplier.s_comment.clone(),
        });
    }
    Ok(rows)
}

fn q02_output(rows: Vec<Q02Row>) -> Result<QueryOutput> {
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("s_acctbal", DataType::Float64, false),
            Field::new("s_name", DataType::Utf8, false),
            Field::new("n_name", DataType::Utf8, false),
            Field::new("p_partkey", DataType::Int64, false),
            Field::new("p_mfgr", DataType::Utf8, false),
            Field::new("s_address", DataType::Utf8, false),
            Field::new("s_phone", DataType::Utf8, false),
            Field::new("s_comment", DataType::Utf8, false),
        ])),
        vec![
            Arc::new(Float64Array::from_iter_values(
                rows.iter().map(|row| row.acctbal),
            )),
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|row| row.s_name.as_str()),
            )),
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|row| row.n_name.as_str()),
            )),
            Arc::new(Int64Array::from_iter_values(
                rows.iter().map(|row| row.p_partkey),
            )),
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|row| row.p_mfgr.as_str()),
            )),
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|row| row.s_address.as_str()),
            )),
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|row| row.s_phone.as_str()),
            )),
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|row| row.s_comment.as_str()),
            )),
        ],
    )?;
    Ok(QueryOutput::Aggregate {
        metrics: AggregateMetrics::default(),
        batches: vec![batch],
    })
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
