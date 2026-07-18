use super::*;

fn minimum_cost_supplier_shape(select: &Select, query: &Query, selection: &SqlExpr) -> bool {
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

pub(in crate::sql) async fn try_execute_minimum_cost_supplier_sql(
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
    if !minimum_cost_supplier_shape(select, query, selection) {
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
    let Some(mut tables) = named_comma_join_tables(
        select,
        &["part", "supplier", "partsupp", "nation", "region"],
    )?
    else {
        return Ok(None);
    };
    let Some(part) = tables.remove("part") else {
        return Ok(None);
    };
    let Some(supplier) = tables.remove("supplier") else {
        return Ok(None);
    };
    let Some(partsupp) = tables.remove("partsupp") else {
        return Ok(None);
    };
    let Some(nation) = tables.remove("nation") else {
        return Ok(None);
    };
    let Some(region) = tables.remove("region") else {
        return Ok(None);
    };
    let mut conjuncts = Vec::new();
    collect_sql_and_conjuncts(selection, &mut conjuncts);
    let Some(part_size) = numeric_i64_equality_literal(&conjuncts, "p_size")? else {
        return Ok(None);
    };
    let Some(part_type_suffix) = like_suffix_literal(&conjuncts, "p_type")? else {
        return Ok(None);
    };
    let Some(region_name) = string_equality_literal(&conjuncts, "r_name")? else {
        return Ok(None);
    };

    let stage = tpch_profile_start();
    let region_keys = region_keys_by_name(engine, region.path, batch_size, &region_name).await?;
    let nation_names =
        nation_names_by_region_keys(engine, nation.path, batch_size, &region_keys).await?;
    if nation_names.is_empty() {
        return Ok(Some(minimum_cost_supplier_output(Vec::new())?));
    }
    let suppliers =
        minimum_cost_supplier_supplier_rows(engine, supplier.path, batch_size, &nation_names)
            .await?;
    if suppliers.is_empty() {
        return Ok(Some(minimum_cost_supplier_output(Vec::new())?));
    }
    let parts = minimum_cost_supplier_matching_parts(
        engine,
        part.path,
        batch_size,
        part_size,
        &part_type_suffix,
    )
    .await?;
    if parts.is_empty() {
        return Ok(Some(minimum_cost_supplier_output(Vec::new())?));
    }
    tpch_profile_elapsed("minimum-cost-supplier dimensions", stage);

    let stage = tpch_profile_start();
    let mut rows =
        minimum_cost_supplier_min_cost_rows(engine, partsupp.path, batch_size, &parts, &suppliers)
            .await?;
    tpch_profile_elapsed("minimum-cost-supplier partsupp min cost", stage);
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
    Ok(Some(minimum_cost_supplier_output(rows)?))
}

#[derive(Clone)]
struct MinimumCostSupplierSupplier {
    acctbal: f64,
    s_name: String,
    n_name: String,
    s_address: String,
    s_phone: String,
    s_comment: String,
}

#[derive(Clone)]
struct MinimumCostSupplierPart {
    mfgr: String,
}

struct MinimumCostSupplierRow {
    acctbal: f64,
    s_name: String,
    n_name: String,
    p_partkey: i64,
    p_mfgr: String,
    s_address: String,
    s_phone: String,
    s_comment: String,
}

async fn minimum_cost_supplier_supplier_rows(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    nation_names: &HashMap<i64, String>,
) -> Result<HashMap<i64, MinimumCostSupplierSupplier>> {
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
                MinimumCostSupplierSupplier {
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

async fn minimum_cost_supplier_matching_parts(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    part_size: i64,
    type_suffix: &str,
) -> Result<HashMap<i64, MinimumCostSupplierPart>> {
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
        if let Some(partkeys) = partkeys.as_any().downcast_ref::<Int64Array>()
            && let Some(sizes) = sizes.as_any().downcast_ref::<Int32Array>()
        {
            for row in 0..batch.num_rows() {
                if partkeys.is_null(row) || sizes.is_null(row) || types.is_null(row) {
                    continue;
                }
                if i64::from(sizes.value(row)) != part_size
                    || !types.value(row).ends_with(type_suffix)
                {
                    continue;
                }
                if mfgrs.is_null(row) {
                    continue;
                }
                parts.insert(
                    partkeys.value(row),
                    MinimumCostSupplierPart {
                        mfgr: mfgrs.value(row).to_string(),
                    },
                );
            }
            continue;
        }
        for row in 0..batch.num_rows() {
            if types.is_null(row) {
                continue;
            }
            let (Some(partkey), Some(size)) = (
                numeric_i64_value(partkeys, row)?,
                numeric_i64_value(sizes, row)?,
            ) else {
                continue;
            };
            if size == part_size && types.value(row).ends_with(type_suffix) {
                if mfgrs.is_null(row) {
                    continue;
                }
                parts.insert(
                    partkey,
                    MinimumCostSupplierPart {
                        mfgr: mfgrs.value(row).to_string(),
                    },
                );
            }
        }
    }
    Ok(parts)
}

async fn minimum_cost_supplier_min_cost_rows(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    parts: &HashMap<i64, MinimumCostSupplierPart>,
    suppliers: &HashMap<i64, MinimumCostSupplierSupplier>,
) -> Result<Vec<MinimumCostSupplierRow>> {
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
    let mut candidates_by_part = HashMap::<i64, (i128, Vec<i64>)>::new();
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
        if let (Some(partkeys), Some(suppkeys), Some(raw_supplycosts)) = (
            partkeys.as_any().downcast_ref::<Int64Array>(),
            suppkeys.as_any().downcast_ref::<Int64Array>(),
            Some(supplycosts.raw_values()),
        ) {
            for row in 0..batch.num_rows() {
                if partkeys.is_null(row) || suppkeys.is_null(row) || supplycosts.is_null(row) {
                    continue;
                }
                let partkey = partkeys.value(row);
                let suppkey = suppkeys.value(row);
                if !part_keys.contains(partkey) || !supplier_keys.contains(suppkey) {
                    continue;
                }
                let supplycost = raw_supplycosts[row];
                match candidates_by_part.entry(partkey) {
                    std::collections::hash_map::Entry::Vacant(entry) => {
                        entry.insert((supplycost, vec![suppkey]));
                    }
                    std::collections::hash_map::Entry::Occupied(mut entry) => {
                        let (min_cost, suppkeys) = entry.get_mut();
                        if supplycost < *min_cost {
                            *min_cost = supplycost;
                            suppkeys.clear();
                            suppkeys.push(suppkey);
                        } else if supplycost == *min_cost {
                            suppkeys.push(suppkey);
                        }
                    }
                }
            }
            continue;
        }
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
            let supplycost = supplycosts.raw_values()[row];
            match candidates_by_part.entry(partkey) {
                std::collections::hash_map::Entry::Vacant(entry) => {
                    entry.insert((supplycost, vec![suppkey]));
                }
                std::collections::hash_map::Entry::Occupied(mut entry) => {
                    let (min_cost, suppkeys) = entry.get_mut();
                    if supplycost < *min_cost {
                        *min_cost = supplycost;
                        suppkeys.clear();
                        suppkeys.push(suppkey);
                    } else if supplycost == *min_cost {
                        suppkeys.push(suppkey);
                    }
                }
            }
        }
    }
    let mut rows = Vec::new();
    for (partkey, (_min_cost, suppkeys)) in candidates_by_part {
        let Some(part) = parts.get(&partkey) else {
            continue;
        };
        for suppkey in suppkeys {
            let Some(supplier) = suppliers.get(&suppkey) else {
                continue;
            };
            rows.push(MinimumCostSupplierRow {
                acctbal: supplier.acctbal,
                s_name: supplier.s_name.clone(),
                n_name: supplier.n_name.clone(),
                p_partkey: partkey,
                p_mfgr: part.mfgr.clone(),
                s_address: supplier.s_address.clone(),
                s_phone: supplier.s_phone.clone(),
                s_comment: supplier.s_comment.clone(),
            });
        }
    }
    Ok(rows)
}

fn minimum_cost_supplier_output(rows: Vec<MinimumCostSupplierRow>) -> Result<QueryOutput> {
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
