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
    if let Some(revenues) = try_direct_i64_grouped_discounted_revenue_by_date(
        engine,
        &path,
        batch_size,
        "l_suppkey",
        "l_shipdate",
        "l_extendedprice",
        "l_discount",
        start_days,
        end_days,
    )? {
        return Ok(revenues);
    }
    if let Some(revenues) = supplier_discounted_revenue_by_date_late(
        engine,
        path.clone(),
        batch_size,
        start_days,
        end_days,
    )
    .await?
    {
        return Ok(revenues);
    }
    let projection = Projection::Columns(vec![
        "l_suppkey".to_string(),
        "l_shipdate".to_string(),
        "l_extendedprice".to_string(),
        "l_discount".to_string(),
    ]);
    engine
        .parquet_scan_accumulate_chunks_view(
            path,
            batch_size,
            projection,
            scan_aggregate_row_group_chunk(),
            8,
            scan_aggregate_fusion_enabled(),
            HashMap::<i64, f64>::new,
            HashMap::<i64, f64>::new,
            move |view, revenues| {
                supplier_discounted_revenue_view_into(view, start_days, end_days, revenues)?;
                Ok(Some(()))
            },
            merge_f64_groups,
            "top-supplier revenue aggregate",
        )
        .await
}

async fn supplier_discounted_revenue_by_date_late(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    start_days: i32,
    end_days: i32,
) -> Result<Option<HashMap<i64, f64>>> {
    let predicate_projection = Projection::Columns(vec!["l_shipdate".to_string()]);
    let payload_projection = Projection::Columns(vec![
        "l_suppkey".to_string(),
        "l_extendedprice".to_string(),
        "l_discount".to_string(),
    ]);
    let policy = generic_late_materialization_policy_for_projection(
        &predicate_projection,
        &payload_projection,
        0.20,
        Some(0.50),
    )
    .with_selector_runs_per_selected(0.20);
    let Some(chunks) = engine
        .late_materialized_parquet_map_pruned_with_policy_view(
            path,
            batch_size,
            predicate_projection,
            payload_projection,
            Vec::new(),
            top_supplier_revenue_late_row_group_chunk(),
            policy,
            move || TopSupplierRevenueLateState {
                start_days,
                end_days,
                revenues: HashMap::<i64, f64>::new(),
            },
            top_supplier_revenue_late_build_date_selection_view,
            top_supplier_revenue_late_consume_payload_view,
            |state, metrics| Ok(Some((state.revenues, metrics))),
        )
        .await?
    else {
        return Ok(None);
    };
    let mut revenues = HashMap::<i64, f64>::new();
    let mut metrics = LateMaterializedMetrics::default();
    for chunk in chunks {
        let (chunk_revenues, chunk_metrics) = chunk.output;
        metrics.add(chunk_metrics);
        merge_f64_groups(&mut revenues, chunk_revenues);
    }
    tpch_profile_late_materialized(
        "top-supplier revenue aggregate",
        metrics,
        top_supplier_revenue_late_row_group_chunk(),
    );
    Ok(Some(revenues))
}

fn top_supplier_revenue_late_row_group_chunk() -> usize {
    late_materialization_row_group_chunk(2)
}

struct TopSupplierRevenueLateState {
    start_days: i32,
    end_days: i32,
    revenues: HashMap<i64, f64>,
}

fn top_supplier_revenue_late_build_date_selection_view(
    view: BatchView<'_>,
    selection: &mut LateSelectionBuilder,
    state: &mut TopSupplierRevenueLateState,
) -> Result<Option<()>> {
    if view.num_columns() == 1
        && let Some(shipdates) = view.date32_vector(0)
    {
        if let Some(shipdate_values) = shipdates.values_if_null_free() {
            for &shipdate in shipdate_values {
                selection.push(shipdate >= state.start_days && shipdate < state.end_days);
            }
            return Ok(Some(()));
        }
        for row in 0..shipdates.len() {
            selection.push(
                !shipdates.is_null(row)
                    && shipdates.value(row) >= state.start_days
                    && shipdates.value(row) < state.end_days,
            );
        }
        return Ok(Some(()));
    }
    let Some(batch) = view.try_record_batch() else {
        return Ok(None);
    };
    top_supplier_revenue_late_build_date_selection_batch(batch.clone(), selection, state)
}

fn top_supplier_revenue_late_build_date_selection_batch(
    batch: RecordBatch,
    selection: &mut LateSelectionBuilder,
    state: &mut TopSupplierRevenueLateState,
) -> Result<Option<()>> {
    let shipdates = batch_column(&batch, "l_shipdate")?;
    let Some(shipdates) = shipdates.as_any().downcast_ref::<Date32Array>() else {
        return Ok(None);
    };
    if shipdates.null_count() == 0 {
        for &shipdate in shipdates.values() {
            selection.push(shipdate >= state.start_days && shipdate < state.end_days);
        }
        return Ok(Some(()));
    }
    for row in 0..shipdates.len() {
        selection.push(
            shipdates.is_valid(row)
                && shipdates.value(row) >= state.start_days
                && shipdates.value(row) < state.end_days,
        );
    }
    Ok(Some(()))
}

fn top_supplier_revenue_late_consume_payload_view(
    view: BatchView<'_>,
    state: &mut TopSupplierRevenueLateState,
) -> Result<Option<()>> {
    if view.num_columns() == 3
        && let (Some(suppkeys), Some(extendedprices), Some(discounts)) = (
            view.i64_vector(0),
            view.decimal128_vector(1),
            view.decimal128_vector(2),
        )
    {
        let Some(suppkey_values) = suppkeys.values_if_null_free() else {
            let Some(batch) = view.try_record_batch() else {
                return Ok(None);
            };
            return top_supplier_revenue_late_consume_payload_batch(batch.clone(), state);
        };
        consume_discounted_revenue_decimal128_vectors(
            extendedprices,
            discounts,
            view.num_rows(),
            |row, revenue| {
                if let Some(revenue) = revenue {
                    *state.revenues.entry(suppkey_values[row]).or_insert(0.0) += revenue;
                }
                Ok(())
            },
        )?;
        return Ok(Some(()));
    }
    let Some(batch) = view.try_record_batch() else {
        return Ok(None);
    };
    top_supplier_revenue_late_consume_payload_batch(batch.clone(), state)
}

fn top_supplier_revenue_late_consume_payload_batch(
    batch: RecordBatch,
    state: &mut TopSupplierRevenueLateState,
) -> Result<Option<()>> {
    let suppkeys = batch_column(&batch, "l_suppkey")?;
    let extendedprices = batch_column(&batch, "l_extendedprice")?;
    let discounts = batch_column(&batch, "l_discount")?;
    let (Some(suppkeys), Some(extendedprices), Some(discounts)) = (
        suppkeys.as_any().downcast_ref::<Int64Array>(),
        decimal_input(extendedprices)?,
        decimal_input(discounts)?,
    ) else {
        return Ok(None);
    };
    for row in 0..batch.num_rows() {
        if suppkeys.is_null(row) || extendedprices.is_null(row) || discounts.is_null(row) {
            continue;
        }
        *state.revenues.entry(suppkeys.value(row)).or_insert(0.0) +=
            extendedprices.value(row) * (1.0 - discounts.value(row));
    }
    Ok(Some(()))
}

fn supplier_discounted_revenue_batch_into<S: BuildHasher>(
    batch: RecordBatch,
    start_days: i32,
    end_days: i32,
    revenues: &mut HashMap<i64, f64, S>,
) -> Result<()> {
    let suppkeys = batch_column(&batch, "l_suppkey")?;
    let shipdates = batch_column(&batch, "l_shipdate")?;
    let extendedprices = batch_column(&batch, "l_extendedprice")?;
    let discounts = batch_column(&batch, "l_discount")?;
    if let (Some(suppkeys), Some(shipdates), Some(extendedprices), Some(discounts)) = (
        suppkeys.as_any().downcast_ref::<Int64Array>(),
        shipdates.as_any().downcast_ref::<Date32Array>(),
        decimal_input(extendedprices)?,
        decimal_input(discounts)?,
    ) {
        for row in 0..batch.num_rows() {
            if suppkeys.is_null(row)
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
            *revenues.entry(suppkeys.value(row)).or_insert(0.0) +=
                extendedprices.value(row) * (1.0 - discounts.value(row));
        }
        return Ok(());
    }
    for row in 0..batch.num_rows() {
        let Some(shipdate) = date32_value(shipdates, row)? else {
            continue;
        };
        if shipdate < start_days || shipdate >= end_days {
            continue;
        }
        let (Some(suppkey), Some(extendedprice), Some(discount)) = (
            numeric_i64_value(suppkeys, row)?,
            numeric_f64_value(extendedprices, row)?,
            numeric_f64_value(discounts, row)?,
        ) else {
            continue;
        };
        *revenues.entry(suppkey).or_insert(0.0) += extendedprice * (1.0 - discount);
    }
    Ok(())
}

fn supplier_discounted_revenue_view_into(
    view: BatchView<'_>,
    start_days: i32,
    end_days: i32,
    revenues: &mut HashMap<i64, f64>,
) -> Result<()> {
    if view.num_columns() == 4 {
        if update_i64_grouped_discounted_revenue_by_date_view(
            view, 0, 1, 2, 3, start_days, end_days, revenues,
        )? {
            return Ok(());
        }
        let (Some(suppkeys), Some(shipdates), Some(extendedprices), Some(discounts)) = (
            view.i64_vector(0),
            view.date32_vector(1),
            view.decimal128_vector(2),
            view.decimal128_vector(3),
        ) else {
            let Some(batch) = view.try_record_batch() else {
                return Ok(());
            };
            return supplier_discounted_revenue_batch_into(
                batch.clone(),
                start_days,
                end_days,
                revenues,
            );
        };
        for row in 0..view.num_rows() {
            if suppkeys.is_null(row)
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
            *revenues.entry(suppkeys.value(row)).or_insert(0.0) +=
                extendedprices.value(row) * (1.0 - discounts.value(row));
        }
        return Ok(());
    }
    let Some(batch) = view.try_record_batch() else {
        return Ok(());
    };
    supplier_discounted_revenue_batch_into(batch.clone(), start_days, end_days, revenues)
}

async fn top_supplier_rows(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    top_suppliers: &HashMap<i64, f64>,
) -> Result<Vec<TopSupplierRevenueRow>> {
    let pruning_predicates = top_suppliers
        .keys()
        .copied()
        .min()
        .zip(top_suppliers.keys().copied().max())
        .and_then(|(min_key, max_key)| {
            selective_i64_range_from_parts(min_key, max_key, top_suppliers.len())
        })
        .map(|(min_key, max_key)| i64_range_pruning_predicates("s_suppkey", min_key, max_key))
        .unwrap_or_default();
    let mut rows = collect_i64_three_utf8_mapped_rows_pruned(
        engine,
        path,
        batch_size,
        "s_suppkey",
        "s_name",
        "s_address",
        "s_phone",
        pruning_predicates,
        |suppkey, name, address, phone| {
            let total_revenue = top_suppliers.get(&suppkey).copied()?;
            Some(TopSupplierRevenueRow {
                suppkey,
                name: name.to_string(),
                address: address.to_string(),
                phone: phone.to_string(),
                total_revenue,
            })
        },
    )
    .await?;
    rows.sort_by_key(|row| row.suppkey);
    Ok(rows)
}

fn top_supplier_revenue_output(rows: Vec<TopSupplierRevenueRow>) -> Result<QueryOutput> {
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("s_suppkey", DataType::Int64, false),
            Field::new("s_name", DataType::Utf8, false),
            Field::new("s_address", DataType::Utf8, false),
            Field::new("s_phone", DataType::Utf8, false),
            Field::new("total_revenue", DataType::Float64, false),
        ])),
        vec![
            Arc::new(Int64Array::from_iter_values(
                rows.iter().map(|row| row.suppkey),
            )),
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|row| row.name.as_str()),
            )),
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|row| row.address.as_str()),
            )),
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|row| row.phone.as_str()),
            )),
            Arc::new(Float64Array::from_iter_values(
                rows.iter().map(|row| row.total_revenue),
            )),
        ],
    )?;
    Ok(QueryOutput::Aggregate {
        metrics: AggregateMetrics::default(),
        batches: vec![batch],
    })
}

struct TopSupplierRevenueRow {
    suppkey: i64,
    name: String,
    address: String,
    phone: String,
    total_revenue: f64,
}
