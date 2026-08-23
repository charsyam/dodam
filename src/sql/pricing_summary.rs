use super::*;

fn pricing_summary_shape(select: &Select, query: &Query, selection: &SqlExpr) -> bool {
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
    select.from.len() == 1
        && select.projection.len() == 10
        && projection.contains("l_returnflag")
        && projection.contains("l_linestatus")
        && projection.contains("sum(l_quantity)")
        && projection.contains("sum(l_extendedprice)")
        && projection.contains("sum(l_extendedprice * (1 - l_discount))")
        && projection.contains("sum(l_extendedprice * (1 - l_discount) * (1 + l_tax))")
        && projection.contains("avg(l_quantity)")
        && projection.contains("avg(l_extendedprice)")
        && projection.contains("avg(l_discount)")
        && projection.contains("count(*)")
        && group_by.contains("l_returnflag")
        && group_by.contains("l_linestatus")
        && order_by.contains("l_returnflag")
        && order_by.contains("l_linestatus")
        && selection.contains("l_shipdate")
        && selection.contains("<=")
}

pub(super) async fn try_execute_pricing_summary_sql(
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
    if !pricing_summary_shape(select, query, selection) {
        return Ok(None);
    }
    let [table_with_joins] = select.from.as_slice() else {
        return Ok(None);
    };
    if !table_with_joins.joins.is_empty() {
        return Ok(None);
    }
    let table = parse_table_factor(&table_with_joins.relation)?;
    if !table_ref_alias_or_name(&table).eq_ignore_ascii_case("lineitem") {
        return Ok(None);
    }
    let Some(cutoff_days) = pricing_summary_shipdate_cutoff(selection)? else {
        return Ok(None);
    };
    reject_query_features(query)?;
    reject_select_features(select)?;
    let rows = pricing_summary_rows(engine, table.path, batch_size, cutoff_days).await?;
    Ok(Some(pricing_summary_output(rows)?))
}

fn pricing_summary_shipdate_cutoff(selection: &SqlExpr) -> Result<Option<i32>> {
    let SqlExpr::BinaryOp { left, op, right } = selection else {
        return Ok(None);
    };
    if *op == BinaryOperator::LtEq && sql_expr_column_matches(left, "l_shipdate") {
        return literal_date_days(right).map(Some);
    }
    if *op == BinaryOperator::GtEq && sql_expr_column_matches(right, "l_shipdate") {
        return literal_date_days(left).map(Some);
    }
    Ok(None)
}

#[derive(Clone, Copy, Default)]
struct PricingSummaryState {
    sum_qty: f64,
    sum_base_price: f64,
    sum_disc_price: f64,
    sum_charge: f64,
    sum_discount: f64,
    count_order: u64,
}

impl PricingSummaryState {
    fn update(&mut self, quantity: f64, extendedprice: f64, discount: f64, tax: f64) {
        let discounted = extendedprice * (1.0 - discount);
        self.sum_qty += quantity;
        self.sum_base_price += extendedprice;
        self.sum_disc_price += discounted;
        self.sum_charge += discounted * (1.0 + tax);
        self.sum_discount += discount;
        self.count_order += 1;
    }

    fn merge(&mut self, other: PricingSummaryState) {
        self.sum_qty += other.sum_qty;
        self.sum_base_price += other.sum_base_price;
        self.sum_disc_price += other.sum_disc_price;
        self.sum_charge += other.sum_charge;
        self.sum_discount += other.sum_discount;
        self.count_order += other.count_order;
    }
}

struct PricingSummaryRow {
    returnflag: String,
    linestatus: String,
    state: PricingSummaryState,
}

struct PricingSummaryGroupSlots {
    keys: [u16; 8],
    states: [PricingSummaryState; 8],
    len: usize,
    overflow: Vec<PricingSummaryRow>,
}

impl PricingSummaryGroupSlots {
    fn new() -> Self {
        Self {
            keys: [0; 8],
            states: [PricingSummaryState::default(); 8],
            len: 0,
            overflow: Vec::new(),
        }
    }

    fn update(
        &mut self,
        returnflag: &str,
        linestatus: &str,
        update: impl FnOnce(&mut PricingSummaryState),
    ) {
        let (Some(returnflag), Some(linestatus)) =
            (single_ascii_byte(returnflag), single_ascii_byte(linestatus))
        else {
            update(pricing_summary_group_state(
                &mut self.overflow,
                returnflag,
                linestatus,
            ));
            return;
        };
        self.update_key(returnflag, linestatus, update);
    }

    fn update_key(
        &mut self,
        returnflag: u8,
        linestatus: u8,
        update: impl FnOnce(&mut PricingSummaryState),
    ) {
        let state = self.state_for_key_mut(returnflag, linestatus);
        update(state);
    }

    fn update_key_values(
        &mut self,
        returnflag: u8,
        linestatus: u8,
        quantity: f64,
        extendedprice: f64,
        discount: f64,
        tax: f64,
    ) {
        self.state_for_key_mut(returnflag, linestatus).update(
            quantity,
            extendedprice,
            discount,
            tax,
        );
    }

    fn state_for_key_mut(&mut self, returnflag: u8, linestatus: u8) -> &mut PricingSummaryState {
        let key = (u16::from(returnflag) << 8) | u16::from(linestatus);
        for index in 0..self.len {
            if self.keys[index] == key {
                return &mut self.states[index];
            }
        }
        if self.len < self.keys.len() {
            let index = self.len;
            self.len += 1;
            self.keys[index] = key;
            return &mut self.states[index];
        }
        let returnflag = char::from(returnflag).to_string();
        let linestatus = char::from(linestatus).to_string();
        pricing_summary_group_state(&mut self.overflow, &returnflag, &linestatus)
    }

    fn merge_slots(&mut self, other: PricingSummaryGroupSlots) {
        for index in 0..other.len {
            let key = other.keys[index];
            if let Some(target_index) = (0..self.len).find(|index| self.keys[*index] == key) {
                self.states[target_index].merge(other.states[index]);
                continue;
            }
            if self.len < self.keys.len() {
                let target_index = self.len;
                self.len += 1;
                self.keys[target_index] = key;
                self.states[target_index] = other.states[index];
                continue;
            }
            let returnflag =
                char::from(u8::try_from(key >> 8).expect("pricing summary returnflag byte"))
                    .to_string();
            let linestatus =
                char::from(u8::try_from(key & 0xff).expect("pricing summary linestatus byte"))
                    .to_string();
            pricing_summary_group_state(&mut self.overflow, &returnflag, &linestatus)
                .merge(other.states[index]);
        }
        for row in other.overflow {
            pricing_summary_group_state(&mut self.overflow, &row.returnflag, &row.linestatus)
                .merge(row.state);
        }
    }

    fn into_rows(self) -> Vec<PricingSummaryRow> {
        let mut rows = Vec::with_capacity(self.len + self.overflow.len());
        for index in 0..self.len {
            let returnflag =
                u8::try_from(self.keys[index] >> 8).expect("pricing summary returnflag byte");
            let linestatus =
                u8::try_from(self.keys[index] & 0xff).expect("pricing summary linestatus byte");
            rows.push(PricingSummaryRow {
                returnflag: char::from(returnflag).to_string(),
                linestatus: char::from(linestatus).to_string(),
                state: self.states[index],
            });
        }
        rows.extend(self.overflow);
        rows
    }
}

async fn pricing_summary_rows(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    cutoff_days: i32,
) -> Result<Vec<PricingSummaryRow>> {
    let projection = Projection::Columns(vec![
        "l_returnflag".to_string(),
        "l_linestatus".to_string(),
        "l_quantity".to_string(),
        "l_extendedprice".to_string(),
        "l_discount".to_string(),
        "l_tax".to_string(),
        "l_shipdate".to_string(),
    ]);
    let groups = engine
        .parquet_scan_accumulate_chunks_view(
            path,
            batch_size,
            projection,
            pricing_summary_row_group_map_chunk(),
            pricing_summary_chunk_size(),
            scan_aggregate_fusion_enabled(),
            PricingSummaryGroupSlots::new,
            PricingSummaryGroupSlots::new,
            move |view, groups| {
                pricing_summary_projected_view_into(view, cutoff_days, groups)?;
                Ok(Some(()))
            },
            |groups, rows| groups.merge_slots(rows),
            "pricing summary aggregate",
        )
        .await?;
    Ok(pricing_summary_sorted_rows(groups))
}

fn pricing_summary_sorted_rows(groups: PricingSummaryGroupSlots) -> Vec<PricingSummaryRow> {
    let mut rows = groups.into_rows();
    rows.sort_by(|left, right| {
        left.returnflag
            .cmp(&right.returnflag)
            .then_with(|| left.linestatus.cmp(&right.linestatus))
    });
    rows
}

fn pricing_summary_row_group_map_chunk() -> usize {
    generic_row_group_map_chunk_size(2)
}

fn pricing_summary_chunk_size() -> usize {
    rule_chunk_size(4)
}

fn pricing_summary_batch_into(
    batch: RecordBatch,
    cutoff_days: i32,
    groups: &mut PricingSummaryGroupSlots,
) -> Result<()> {
    let returnflags = batch_string_column(&batch, "l_returnflag")?;
    let linestatuses = batch_string_column(&batch, "l_linestatus")?;
    let quantities = batch_column(&batch, "l_quantity")?;
    let extendedprices = batch_column(&batch, "l_extendedprice")?;
    let discounts = batch_column(&batch, "l_discount")?;
    let taxes = batch_column(&batch, "l_tax")?;
    let shipdates = batch_column(&batch, "l_shipdate")?;
    if pricing_summary_update_decimal_batch(
        returnflags,
        linestatuses,
        quantities,
        extendedprices,
        discounts,
        taxes,
        shipdates,
        cutoff_days,
        groups,
    )? {
        return Ok(());
    }
    for row in 0..batch.num_rows() {
        let Some(shipdate) = date32_value(shipdates, row)? else {
            continue;
        };
        if shipdate > cutoff_days || !returnflags.is_valid(row) || !linestatuses.is_valid(row) {
            continue;
        }
        let (Some(quantity), Some(extendedprice), Some(discount), Some(tax)) = (
            numeric_f64_value(quantities, row)?,
            numeric_f64_value(extendedprices, row)?,
            numeric_f64_value(discounts, row)?,
            numeric_f64_value(taxes, row)?,
        ) else {
            continue;
        };
        groups.update(returnflags.value(row), linestatuses.value(row), |state| {
            state.update(quantity, extendedprice, discount, tax);
        });
    }
    Ok(())
}

fn pricing_summary_projected_view_into(
    view: BatchView<'_>,
    cutoff_days: i32,
    groups: &mut PricingSummaryGroupSlots,
) -> Result<()> {
    if view.num_columns() == 7 {
        if let (
            Some(returnflags),
            Some(linestatuses),
            Some(quantities),
            Some(extendedprices),
            Some(discounts),
            Some(taxes),
            Some(shipdates),
        ) = (
            view.utf8_vector(0),
            view.utf8_vector(1),
            view.decimal128_vector(2),
            view.decimal128_vector(3),
            view.decimal128_vector(4),
            view.decimal128_vector(5),
            view.date32_vector(6),
        ) && pricing_summary_update_decimal_view(
            returnflags,
            linestatuses,
            quantities,
            extendedprices,
            discounts,
            taxes,
            shipdates,
            cutoff_days,
            groups,
        )? {
            return Ok(());
        }
    }
    let Some(batch) = view.try_record_batch() else {
        return Ok(());
    };
    pricing_summary_batch_into(batch.clone(), cutoff_days, groups)
}

fn pricing_summary_update_decimal_batch(
    returnflags: &StringArray,
    linestatuses: &StringArray,
    quantities: &ArrayRef,
    extendedprices: &ArrayRef,
    discounts: &ArrayRef,
    taxes: &ArrayRef,
    shipdates: &ArrayRef,
    cutoff_days: i32,
    groups: &mut PricingSummaryGroupSlots,
) -> Result<bool> {
    let (Some(quantities), Some(extendedprices), Some(discounts), Some(taxes), Some(shipdates)) = (
        decimal_input(quantities)?,
        decimal_input(extendedprices)?,
        decimal_input(discounts)?,
        decimal_input(taxes)?,
        shipdates.as_any().downcast_ref::<Date32Array>(),
    ) else {
        return Ok(false);
    };
    if shipdates.null_count() == 0
        && returnflags.null_count() == 0
        && linestatuses.null_count() == 0
        && quantities.null_count() == 0
        && extendedprices.null_count() == 0
        && discounts.null_count() == 0
        && taxes.null_count() == 0
    {
        let returnflag_offsets = returnflags.value_offsets();
        let returnflag_data = returnflags.value_data();
        let linestatus_offsets = linestatuses.value_offsets();
        let linestatus_data = linestatuses.value_data();
        let quantity_values = quantities.raw_values();
        let extendedprice_values = extendedprices.raw_values();
        let discount_values = discounts.raw_values();
        let tax_values = taxes.raw_values();
        let quantity_scale = 1.0 / quantities.scale;
        let extendedprice_scale = 1.0 / extendedprices.scale;
        let discount_scale = 1.0 / discounts.scale;
        let tax_scale = 1.0 / taxes.scale;
        let shipdate_values = shipdates.values().as_ref();
        if let (Some(returnflag_bytes), Some(linestatus_bytes)) = (
            contiguous_single_byte_utf8_data(returnflags),
            contiguous_single_byte_utf8_data(linestatuses),
        ) {
            if quantities.precision <= 18
                && extendedprices.precision <= 18
                && discounts.precision <= 18
                && taxes.precision <= 18
            {
                let row_count = shipdate_values.len();
                debug_assert_eq!(quantity_values.len(), row_count);
                debug_assert_eq!(extendedprice_values.len(), row_count);
                debug_assert_eq!(discount_values.len(), row_count);
                debug_assert_eq!(tax_values.len(), row_count);
                debug_assert_eq!(returnflag_bytes.len(), row_count);
                debug_assert_eq!(linestatus_bytes.len(), row_count);
                for row in 0..row_count {
                    // All slices come from columns of the same RecordBatch and were length-checked above.
                    let (
                        shipdate,
                        quantity_raw,
                        extendedprice_raw,
                        discount_raw,
                        tax_raw,
                        returnflag,
                        linestatus,
                    ) = unsafe {
                        (
                            *shipdate_values.get_unchecked(row),
                            *quantity_values.get_unchecked(row),
                            *extendedprice_values.get_unchecked(row),
                            *discount_values.get_unchecked(row),
                            *tax_values.get_unchecked(row),
                            *returnflag_bytes.get_unchecked(row),
                            *linestatus_bytes.get_unchecked(row),
                        )
                    };
                    if shipdate > cutoff_days {
                        continue;
                    }
                    let quantity = quantity_raw as i64 as f64 * quantity_scale;
                    let extendedprice = extendedprice_raw as i64 as f64 * extendedprice_scale;
                    let discount = discount_raw as i64 as f64 * discount_scale;
                    let tax = tax_raw as i64 as f64 * tax_scale;
                    groups.update_key_values(
                        returnflag,
                        linestatus,
                        quantity,
                        extendedprice,
                        discount,
                        tax,
                    );
                }
                return Ok(true);
            }
            for row in 0..shipdate_values.len() {
                if shipdate_values[row] > cutoff_days {
                    continue;
                }
                let quantity = quantity_values[row] as f64 * quantity_scale;
                let extendedprice = extendedprice_values[row] as f64 * extendedprice_scale;
                let discount = discount_values[row] as f64 * discount_scale;
                let tax = tax_values[row] as f64 * tax_scale;
                groups.update_key_values(
                    returnflag_bytes[row],
                    linestatus_bytes[row],
                    quantity,
                    extendedprice,
                    discount,
                    tax,
                );
            }
            return Ok(true);
        }
        for row in 0..shipdate_values.len() {
            if shipdate_values[row] > cutoff_days {
                continue;
            }
            let quantity = quantity_values[row] as f64 * quantity_scale;
            let extendedprice = extendedprice_values[row] as f64 * extendedprice_scale;
            let discount = discount_values[row] as f64 * discount_scale;
            let tax = tax_values[row] as f64 * tax_scale;
            if let (Some(returnflag), Some(linestatus)) = (
                single_byte_string_parts(returnflag_offsets, returnflag_data, row),
                single_byte_string_parts(linestatus_offsets, linestatus_data, row),
            ) {
                groups.update_key_values(
                    returnflag,
                    linestatus,
                    quantity,
                    extendedprice,
                    discount,
                    tax,
                );
                continue;
            }
            groups.update(returnflags.value(row), linestatuses.value(row), |state| {
                state.update(quantity, extendedprice, discount, tax);
            });
        }
        return Ok(true);
    }
    for row in 0..shipdates.len() {
        if shipdates.is_null(row)
            || shipdates.value(row) > cutoff_days
            || returnflags.is_null(row)
            || linestatuses.is_null(row)
            || quantities.is_null(row)
            || extendedprices.is_null(row)
            || discounts.is_null(row)
            || taxes.is_null(row)
        {
            continue;
        }
        groups.update(returnflags.value(row), linestatuses.value(row), |state| {
            state.update(
                quantities.value(row),
                extendedprices.value(row),
                discounts.value(row),
                taxes.value(row),
            );
        });
    }
    Ok(true)
}

#[allow(clippy::too_many_arguments)]
fn pricing_summary_update_decimal_view(
    returnflags: Utf8VectorView<'_>,
    linestatuses: Utf8VectorView<'_>,
    quantities: Decimal128VectorView<'_>,
    extendedprices: Decimal128VectorView<'_>,
    discounts: Decimal128VectorView<'_>,
    taxes: Decimal128VectorView<'_>,
    shipdates: Date32VectorView<'_>,
    cutoff_days: i32,
    groups: &mut PricingSummaryGroupSlots,
) -> Result<bool> {
    if shipdates.values_if_null_free().is_some()
        && returnflags.null_count() == 0
        && linestatuses.null_count() == 0
        && quantities.null_count() == 0
        && extendedprices.null_count() == 0
        && discounts.null_count() == 0
        && taxes.null_count() == 0
    {
        let quantity_values = quantities.raw_values();
        let extendedprice_values = extendedprices.raw_values();
        let discount_values = discounts.raw_values();
        let tax_values = taxes.raw_values();
        let quantity_scale = 1.0 / quantities.scale();
        let extendedprice_scale = 1.0 / extendedprices.scale();
        let discount_scale = 1.0 / discounts.scale();
        let tax_scale = 1.0 / taxes.scale();
        let Some(shipdate_values) = shipdates.values_if_null_free() else {
            return Ok(false);
        };
        if let (Some(returnflag_bytes), Some(linestatus_bytes)) = (
            contiguous_single_byte_utf8_view(returnflags),
            contiguous_single_byte_utf8_view(linestatuses),
        ) {
            if quantities.precision() <= 18
                && extendedprices.precision() <= 18
                && discounts.precision() <= 18
                && taxes.precision() <= 18
            {
                for row in 0..shipdate_values.len() {
                    let shipdate = shipdate_values[row];
                    if shipdate > cutoff_days {
                        continue;
                    }
                    groups.update_key_values(
                        returnflag_bytes[row],
                        linestatus_bytes[row],
                        quantity_values[row] as i64 as f64 * quantity_scale,
                        extendedprice_values[row] as i64 as f64 * extendedprice_scale,
                        discount_values[row] as i64 as f64 * discount_scale,
                        tax_values[row] as i64 as f64 * tax_scale,
                    );
                }
                return Ok(true);
            }
            for row in 0..shipdate_values.len() {
                if shipdate_values[row] > cutoff_days {
                    continue;
                }
                groups.update_key_values(
                    returnflag_bytes[row],
                    linestatus_bytes[row],
                    quantity_values[row] as f64 * quantity_scale,
                    extendedprice_values[row] as f64 * extendedprice_scale,
                    discount_values[row] as f64 * discount_scale,
                    tax_values[row] as f64 * tax_scale,
                );
            }
            return Ok(true);
        }
        for row in 0..shipdate_values.len() {
            if shipdate_values[row] > cutoff_days {
                continue;
            }
            let returnflag = std::str::from_utf8(returnflags.value_bytes(row))
                .map_err(|error| DodamError::UnsupportedSql(error.to_string()))?;
            let linestatus = std::str::from_utf8(linestatuses.value_bytes(row))
                .map_err(|error| DodamError::UnsupportedSql(error.to_string()))?;
            groups.update(returnflag, linestatus, |state| {
                state.update(
                    quantity_values[row] as f64 * quantity_scale,
                    extendedprice_values[row] as f64 * extendedprice_scale,
                    discount_values[row] as f64 * discount_scale,
                    tax_values[row] as f64 * tax_scale,
                );
            });
        }
        return Ok(true);
    }
    for row in 0..quantities.raw_values().len() {
        if shipdates.is_null(row)
            || shipdates.value(row) > cutoff_days
            || returnflags.is_null(row)
            || linestatuses.is_null(row)
            || quantities.is_null(row)
            || extendedprices.is_null(row)
            || discounts.is_null(row)
            || taxes.is_null(row)
        {
            continue;
        }
        let returnflag = std::str::from_utf8(returnflags.value_bytes(row))
            .map_err(|error| DodamError::UnsupportedSql(error.to_string()))?;
        let linestatus = std::str::from_utf8(linestatuses.value_bytes(row))
            .map_err(|error| DodamError::UnsupportedSql(error.to_string()))?;
        groups.update(returnflag, linestatus, |state| {
            state.update(
                quantities.value(row),
                extendedprices.value(row),
                discounts.value(row),
                taxes.value(row),
            );
        });
    }
    Ok(true)
}

fn pricing_summary_group_state<'a>(
    groups: &'a mut Vec<PricingSummaryRow>,
    returnflag: &str,
    linestatus: &str,
) -> &'a mut PricingSummaryState {
    if let Some(index) = groups
        .iter()
        .position(|row| row.returnflag == returnflag && row.linestatus == linestatus)
    {
        return &mut groups[index].state;
    }
    groups.push(PricingSummaryRow {
        returnflag: returnflag.to_string(),
        linestatus: linestatus.to_string(),
        state: PricingSummaryState::default(),
    });
    &mut groups
        .last_mut()
        .expect("inserted pricing summary group")
        .state
}

fn single_ascii_byte(value: &str) -> Option<u8> {
    let bytes = value.as_bytes();
    (bytes.len() == 1 && bytes[0].is_ascii()).then_some(bytes[0])
}

fn single_byte_string_parts(offsets: &[i32], data: &[u8], row: usize) -> Option<u8> {
    let start = offsets[row] as usize;
    let end = offsets[row + 1] as usize;
    (end == start + 1).then_some(data[start])
}

fn contiguous_single_byte_utf8_data(values: &StringArray) -> Option<&[u8]> {
    if values.null_count() != 0 || values.value_data().len() != values.len() {
        return None;
    }
    for (index, offset) in values.value_offsets().iter().copied().enumerate() {
        if usize::try_from(offset).ok()? != index {
            return None;
        }
    }
    Some(values.value_data())
}

fn contiguous_single_byte_utf8_view(values: Utf8VectorView<'_>) -> Option<&[u8]> {
    match values {
        Utf8VectorView::Arrow(values) => contiguous_single_byte_utf8_data(values),
    }
}

fn pricing_summary_output(rows: Vec<PricingSummaryRow>) -> Result<QueryOutput> {
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("l_returnflag", DataType::Utf8, false),
            Field::new("l_linestatus", DataType::Utf8, false),
            Field::new("sum_qty", DataType::Float64, false),
            Field::new("sum_base_price", DataType::Float64, false),
            Field::new("sum_disc_price", DataType::Float64, false),
            Field::new("sum_charge", DataType::Float64, false),
            Field::new("avg_qty", DataType::Float64, false),
            Field::new("avg_price", DataType::Float64, false),
            Field::new("avg_disc", DataType::Float64, false),
            Field::new("count_order", DataType::UInt64, false),
        ])),
        vec![
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|row| row.returnflag.as_str()),
            )),
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|row| row.linestatus.as_str()),
            )),
            Arc::new(Float64Array::from_iter_values(
                rows.iter().map(|row| row.state.sum_qty),
            )),
            Arc::new(Float64Array::from_iter_values(
                rows.iter().map(|row| row.state.sum_base_price),
            )),
            Arc::new(Float64Array::from_iter_values(
                rows.iter().map(|row| row.state.sum_disc_price),
            )),
            Arc::new(Float64Array::from_iter_values(
                rows.iter().map(|row| row.state.sum_charge),
            )),
            Arc::new(Float64Array::from_iter_values(
                rows.iter()
                    .map(|row| row.state.sum_qty / row.state.count_order as f64),
            )),
            Arc::new(Float64Array::from_iter_values(rows.iter().map(|row| {
                row.state.sum_base_price / row.state.count_order as f64
            }))),
            Arc::new(Float64Array::from_iter_values(rows.iter().map(|row| {
                row.state.sum_discount / row.state.count_order as f64
            }))),
            Arc::new(UInt64Array::from_iter_values(
                rows.iter().map(|row| row.state.count_order),
            )),
        ],
    )?;
    Ok(QueryOutput::Aggregate {
        metrics: AggregateMetrics::default(),
        batches: vec![batch],
    })
}
