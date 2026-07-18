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
    if let Some(revenues) =
        supplier_discounted_revenue_direct(engine, &path, batch_size, start_days, end_days)?
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

fn supplier_discounted_revenue_direct(
    engine: &DodamEngine,
    path: &Path,
    batch_size: usize,
    start_days: i32,
    end_days: i32,
) -> Result<Option<HashMap<i64, f64>>> {
    if !direct_discounted_revenue_selected_fold_enabled() {
        return Ok(None);
    }
    let trace = std::env::var("DODAM_DIRECT_SELECTION_TRACE")
        .or_else(|_| std::env::var("DODAM_TPCH_PROFILE"))
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"));
    if trace {
        eprintln!("[dodam:direct-selected] top-supplier revenue candidate");
    }
    if !direct_selection_fold_enabled() {
        if trace {
            eprintln!(
                "[dodam:direct-selected] top-supplier revenue reject: direct selection fold disabled"
            );
        }
        return Ok(None);
    }
    let Some((_price_precision, price_scale)) =
        engine.parquet_decimal128_type(path, "l_extendedprice")?
    else {
        if trace {
            eprintln!(
                "[dodam:direct-selected] top-supplier revenue reject: l_extendedprice is not Decimal128"
            );
        }
        return Ok(None);
    };
    let Some((_discount_precision, discount_decimal_scale)) =
        engine.parquet_decimal128_type(path, "l_discount")?
    else {
        if trace {
            eprintln!(
                "[dodam:direct-selected] top-supplier revenue reject: l_discount is not Decimal128"
            );
        }
        return Ok(None);
    };
    let date_max = end_days.checked_sub(1).ok_or_else(|| {
        DodamError::UnsupportedSql(
            "invalid empty top-supplier revenue date range for direct fold".to_string(),
        )
    })?;
    let row_groups = (0..engine.parquet_row_group_count(path)?).collect::<Vec<_>>();
    let discount_scale = decimal_scale_factor(discount_decimal_scale);
    let revenue_scale =
        1.0 / (decimal_scale_factor(price_scale) * decimal_scale_factor(discount_decimal_scale));
    let Some((revenues, _metrics)) = engine
        .scan_parquet_i64_date_decimal_decimal_selected_typed_fold(
            path,
            batch_size,
            &row_groups,
            ["l_suppkey", "l_shipdate", "l_extendedprice", "l_discount"],
            Some(start_days),
            Some(date_max),
            HashMap::<i64, f64>::new,
            move |revenues, batch| {
                supplier_discounted_revenue_direct_batch_into(
                    batch,
                    start_days,
                    end_days,
                    discount_scale,
                    revenue_scale,
                    revenues,
                )
            },
            |revenues, partial| {
                merge_f64_groups(revenues, partial);
                Ok(())
            },
        )?
    else {
        return Ok(None);
    };
    Ok(Some(revenues))
}

fn direct_discounted_revenue_selected_fold_enabled() -> bool {
    std::env::var("DODAM_ENABLE_DIRECT_DISCOUNTED_REVENUE_SELECTED_FOLD")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

fn supplier_discounted_revenue_direct_batch_into(
    batch: crate::storage::DirectI64DateDecimalDecimalSelectedBatch<'_>,
    start_days: i32,
    end_days: i32,
    discount_scale: f64,
    revenue_scale: f64,
    revenues: &mut HashMap<i64, f64>,
) -> Result<()> {
    if batch.keys.len() != batch.left_decimals.len()
        || batch.keys.len() != batch.right_decimals.len()
        || batch.keys.len() != batch.dates.len()
    {
        return Err(DodamError::UnsupportedSql(
            "direct discounted revenue batch length mismatch".to_string(),
        ));
    }
    if batch.predicate_applied {
        for row in 0..batch.keys.len() {
            *revenues.entry(batch.keys[row]).or_insert(0.0) += decimal_discounted_revenue_raw_i64(
                batch.left_decimals[row],
                batch.right_decimals[row],
                discount_scale,
                revenue_scale,
            );
        }
        return Ok(());
    }
    for row in 0..batch.keys.len() {
        let date = batch.dates[row];
        if date >= start_days && date < end_days {
            *revenues.entry(batch.keys[row]).or_insert(0.0) += decimal_discounted_revenue_raw_i64(
                batch.left_decimals[row],
                batch.right_decimals[row],
                discount_scale,
                revenue_scale,
            );
        }
    }
    Ok(())
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

#[allow(clippy::too_many_arguments)]
fn update_i64_grouped_discounted_revenue_by_date_view<S: BuildHasher>(
    view: BatchView<'_>,
    key_index: usize,
    date_index: usize,
    extendedprice_index: usize,
    discount_index: usize,
    start_days: i32,
    end_days: i32,
    revenues: &mut HashMap<i64, f64, S>,
) -> Result<bool> {
    let (Some(keys), Some(dates), Some(extendedprices), Some(discounts)) = (
        view.i64_vector(key_index),
        view.date32_vector(date_index),
        view.decimal128_vector(extendedprice_index),
        view.decimal128_vector(discount_index),
    ) else {
        return Ok(false);
    };
    let (Some(key_values), Some(date_values)) =
        (keys.values_if_null_free(), dates.values_if_null_free())
    else {
        return Ok(false);
    };
    if extendedprices.null_count() != 0 || discounts.null_count() != 0 {
        return Ok(false);
    }
    let discount_scale = discounts.scale();
    let revenue_scale = 1.0 / (extendedprices.scale() * discounts.scale());
    if let (Some(extendedprice_values), Some(discount_values)) =
        (extendedprices.raw_i64_values(), discounts.raw_i64_values())
    {
        for row in 0..view.num_rows() {
            let date = date_values[row];
            if date >= start_days && date < end_days {
                *revenues.entry(key_values[row]).or_insert(0.0) +=
                    decimal_discounted_revenue_raw_i64(
                        extendedprice_values[row],
                        discount_values[row],
                        discount_scale,
                        revenue_scale,
                    );
            }
        }
        return Ok(true);
    }
    let extendedprice_values = extendedprices.raw_values();
    let discount_values = discounts.raw_values();
    for row in 0..view.num_rows() {
        let date = date_values[row];
        if date >= start_days && date < end_days {
            *revenues.entry(key_values[row]).or_insert(0.0) += decimal_discounted_revenue_raw(
                extendedprice_values[row],
                discount_values[row],
                discount_scale,
                revenue_scale,
            );
        }
    }
    Ok(true)
}

async fn top_supplier_rows(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    top_suppliers: &HashMap<i64, f64>,
) -> Result<Vec<TopSupplierRevenueRow>> {
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec![
                "s_suppkey".to_string(),
                "s_name".to_string(),
                "s_address".to_string(),
                "s_phone".to_string(),
            ]),
            None,
        )
        .await?;
    let mut rows = Vec::new();
    while let Some(batch) = stream.next() {
        let batch = batch?;
        top_supplier_rows_view_into(BatchView::new(&batch), top_suppliers, &mut rows)?;
    }
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

fn top_supplier_rows_view_into(
    view: BatchView<'_>,
    top_suppliers: &HashMap<i64, f64>,
    rows: &mut Vec<TopSupplierRevenueRow>,
) -> Result<()> {
    if view.num_columns() == 4
        && let (Some(suppkeys), Some(names), Some(addresses), Some(phones)) =
            (view.i64(0), view.utf8(1), view.utf8(2), view.utf8(3))
    {
        for row in 0..view.num_rows() {
            if suppkeys.is_null(row)
                || names.is_null(row)
                || addresses.is_null(row)
                || phones.is_null(row)
            {
                continue;
            }
            let suppkey = suppkeys.value(row);
            let Some(total_revenue) = top_suppliers.get(&suppkey).copied() else {
                continue;
            };
            rows.push(TopSupplierRevenueRow {
                suppkey,
                name: names.value(row).to_string(),
                address: addresses.value(row).to_string(),
                phone: phones.value(row).to_string(),
                total_revenue,
            });
        }
        return Ok(());
    }
    let Some(batch) = view.try_record_batch() else {
        return Err(DodamError::UnsupportedSql(
            "top-supplier raw vector columns have unsupported types".to_string(),
        ));
    };
    let suppkeys = batch_column(batch, "s_suppkey")?;
    let names = batch_string_column(batch, "s_name")?;
    let addresses = batch_string_column(batch, "s_address")?;
    let phones = batch_string_column(batch, "s_phone")?;
    for row in 0..batch.num_rows() {
        if names.is_null(row) || addresses.is_null(row) || phones.is_null(row) {
            continue;
        }
        let Some(suppkey) = numeric_i64_value(suppkeys, row)? else {
            continue;
        };
        let Some(total_revenue) = top_suppliers.get(&suppkey).copied() else {
            continue;
        };
        rows.push(TopSupplierRevenueRow {
            suppkey,
            name: names.value(row).to_string(),
            address: addresses.value(row).to_string(),
            phone: phones.value(row).to_string(),
            total_revenue,
        });
    }
    Ok(())
}
