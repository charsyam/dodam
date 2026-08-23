use super::*;

fn q09_outer_shape(select: &Select, query: &Query) -> bool {
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
    select.from.len() == 1
        && select.projection.len() == 3
        && projection.contains("nation")
        && projection.contains("o_year")
        && projection.contains("sum(amount)")
        && group_by.contains("nation")
        && group_by.contains("o_year")
        && order_by.contains("nation")
        && order_by.contains("o_year desc")
}

fn q09_inner_shape(select: &Select, selection: &SqlExpr) -> bool {
    let projection = select
        .projection
        .iter()
        .map(|item| item.to_string().to_ascii_lowercase())
        .collect::<Vec<_>>()
        .join(", ");
    let selection = selection.to_string().to_ascii_lowercase();
    select.from.len() == 6
        && select.projection.len() == 3
        && projection.contains("n_name as nation")
        && projection.contains("extract(year from o_orderdate)")
        && projection.contains("l_extendedprice * (1 - l_discount) - ps_supplycost * l_quantity")
        && selection.contains("s_suppkey = l_suppkey")
        && selection.contains("ps_suppkey = l_suppkey")
        && selection.contains("ps_partkey = l_partkey")
        && selection.contains("p_partkey = l_partkey")
        && selection.contains("o_orderkey = l_orderkey")
        && selection.contains("s_nationkey = n_nationkey")
        && selection.contains("p_name like")
}

pub(super) async fn try_execute_profit_by_nation_year_sql(
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
    if !q09_outer_shape(select, query) {
        return Ok(None);
    }
    reject_query_features(query)?;
    reject_select_features(select)?;
    let Some((inner_query, _alias)) = parse_derived_from(select)? else {
        return Ok(None);
    };
    let SetExpr::Select(inner_select) = inner_query.body.as_ref() else {
        return Ok(None);
    };
    let Some(selection) = inner_select.selection.as_ref() else {
        return Ok(None);
    };
    if !q09_inner_shape(inner_select, selection) {
        return Ok(None);
    }
    reject_query_features(inner_query)?;
    reject_select_features(inner_select)?;
    let Some(tables) = parse_comma_join_table_refs(inner_select)? else {
        return Ok(None);
    };
    if tables.len() != 6 {
        return Ok(None);
    }
    let mut part = None;
    let mut supplier = None;
    let mut lineitem = None;
    let mut partsupp = None;
    let mut orders = None;
    let mut nation = None;
    for table in tables {
        let alias = table_ref_alias_or_name(&table);
        if alias.eq_ignore_ascii_case("part") {
            part = Some(table);
        } else if alias.eq_ignore_ascii_case("supplier") {
            supplier = Some(table);
        } else if alias.eq_ignore_ascii_case("lineitem") {
            lineitem = Some(table);
        } else if alias.eq_ignore_ascii_case("partsupp") {
            partsupp = Some(table);
        } else if alias.eq_ignore_ascii_case("orders") {
            orders = Some(table);
        } else if alias.eq_ignore_ascii_case("nation") {
            nation = Some(table);
        }
    }
    let (Some(part), Some(supplier), Some(lineitem), Some(partsupp), Some(orders), Some(nation)) =
        (part, supplier, lineitem, partsupp, orders, nation)
    else {
        return Ok(None);
    };
    let mut conjuncts = Vec::new();
    collect_sql_and_conjuncts(selection, &mut conjuncts);
    let Some(name_substring) = like_contains_literal(&conjuncts, "p_name")? else {
        return Ok(None);
    };

    let part_key_filter = collect_i64_utf8_contains_set(
        engine,
        part.path,
        batch_size,
        "p_partkey",
        "p_name",
        &name_substring,
    )
    .await?;
    if part_key_filter.is_empty() {
        return Ok(Some(q09_output(Vec::new())?));
    }
    let part_keys = AdaptiveI64Set::from_hash(part_key_filter.clone());
    let supplier_nations = q09_supplier_nations(engine, supplier.path, batch_size).await?;
    let nation_names = nation_names_by_keys(engine, nation.path, batch_size).await?;
    let order_years = q09_order_years(engine, orders.path, batch_size).await?;
    let supply_costs = q09_supply_costs(
        engine,
        partsupp.path,
        batch_size,
        &part_keys,
        &supplier_nations,
    )
    .await?;
    q09_log_profit_build_layouts(&part_keys, &order_years, &supply_costs);
    let rows = q09_profit_rows(
        engine,
        lineitem.path,
        batch_size,
        &part_keys,
        &nation_names,
        order_years,
        supply_costs,
    )
    .await?;
    Ok(Some(q09_output(rows)?))
}

fn q09_log_profit_build_layouts(
    part_keys: &AdaptiveI64Set,
    order_years: &Q09OrderYears,
    supply_costs: &Q09SupplyCosts,
) {
    if !tpch_profile_enabled() {
        return;
    }
    eprintln!(
        "[dodam:tpch-profile] Q09 profit build layouts: part_dense={} order_dense={} supply_layout={} part_keys={} supply_entries={}",
        part_keys.dense_contains_slice().is_some(),
        order_years.dense_slice().is_some(),
        supply_costs.layout_name(),
        part_keys.len(),
        supply_costs.len(),
    );
}

fn like_contains_literal(conjuncts: &[SqlExpr], column: &str) -> Result<Option<String>> {
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
        if let Some(value) = pattern
            .strip_prefix('%')
            .and_then(|value| value.strip_suffix('%'))
            && !value.contains('%')
            && !value.contains('_')
        {
            return Ok(Some(value.to_string()));
        }
    }
    Ok(None)
}

async fn q09_supplier_nations(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
) -> Result<AdaptiveI64Map<i64>> {
    collect_i64_i64_adaptive_map(engine, path, batch_size, "s_suppkey", "s_nationkey").await
}

async fn q09_order_years(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
) -> Result<Q09OrderYears> {
    if let Some(years) = q09_order_years_row_group_map(engine, path.clone(), batch_size).await? {
        return Ok(years);
    }
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec!["o_orderkey".to_string(), "o_orderdate".to_string()]),
            None,
        )
        .await?;
    let mut years = Q09OrderYears::new(0);
    let mut year_cache = Date32YearCache::default();
    while let Some(batch) = stream.next() {
        let batch = batch?;
        q09_order_years_batch_into(&batch, &mut years, &mut year_cache)?;
    }
    Ok(years)
}

type Q09OrderYears = DenseI64I32Map;

struct Q09OrderYearPartial {
    rows: Vec<(i64, i32)>,
    min_key: i64,
    max_key: i64,
    fallback_required: bool,
    year_cache: Date32YearCache,
}

impl Q09OrderYearPartial {
    fn new() -> Self {
        Self {
            rows: Vec::new(),
            min_key: i64::MAX,
            max_key: i64::MIN,
            fallback_required: false,
            year_cache: Date32YearCache::default(),
        }
    }

    fn push(&mut self, orderkey: i64, year: i32) {
        if orderkey < 0 {
            self.fallback_required = true;
        } else {
            self.min_key = self.min_key.min(orderkey);
            self.max_key = self.max_key.max(orderkey);
        }
        self.rows.push((orderkey, year));
    }
}

async fn q09_order_years_row_group_map(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
) -> Result<Option<Q09OrderYears>> {
    let Some(partials) = engine
        .parquet_row_group_map_view(
            path,
            batch_size,
            Projection::Columns(vec!["o_orderkey".to_string(), "o_orderdate".to_string()]),
            q09_order_year_row_group_map_chunk(),
            Q09OrderYearPartial::new,
            q09_order_years_partial_view_into,
            |partial| Ok(Some(partial)),
        )
        .await?
    else {
        return Ok(None);
    };
    Ok(Some(q09_order_years_from_partials(partials)))
}

fn q09_order_years_from_partials(partials: Vec<Q09OrderYearPartial>) -> Q09OrderYears {
    let mut min_key = i64::MAX;
    let mut max_key = i64::MIN;
    let mut fallback_required = false;
    let row_count = partials
        .iter()
        .map(|partial| {
            fallback_required |= partial.fallback_required;
            if partial.min_key <= partial.max_key {
                min_key = min_key.min(partial.min_key);
                max_key = max_key.max(partial.max_key);
            }
            partial.rows.len()
        })
        .sum::<usize>();
    if fallback_required || min_key > max_key {
        let mut years = Q09OrderYears::new(0);
        years.convert_to_fallback();
        let fallback = years.fallback_mut().expect("converted q09 fallback");
        for partial in partials {
            for (orderkey, year) in partial.rows {
                fallback.insert(orderkey, year);
            }
        }
        return years;
    }
    let mut years = Q09OrderYears::new(0);
    if !crate::dense::dense_i64_range_within_amplification(
        min_key,
        max_key,
        row_count,
        q09_order_year_dense_max_amplification(),
    ) || !years.reserve_dense_range(
        min_key,
        max_key,
        q09_order_year_max_dense_entries(Some(row_count)),
    ) {
        years.convert_to_fallback();
        let fallback = years.fallback_mut().expect("converted q09 fallback");
        for partial in partials {
            for (orderkey, year) in partial.rows {
                fallback.insert(orderkey, year);
            }
        }
        return years;
    }
    for partial in partials {
        for (orderkey, year) in partial.rows {
            years.insert_dense_key(orderkey, year);
        }
    }
    debug_assert!(row_count == 0 || years.dense_slice().is_some());
    years
}

fn q09_order_years_partial_batch_into(
    batch: RecordBatch,
    partial: &mut Q09OrderYearPartial,
) -> Result<Option<()>> {
    let orderkeys = batch_column(&batch, "o_orderkey")?;
    let orderdates = batch_column(&batch, "o_orderdate")?;
    if let (Some(orderkeys), Some(orderdates)) = (
        orderkeys.as_any().downcast_ref::<Int64Array>(),
        orderdates.as_any().downcast_ref::<Date32Array>(),
    ) {
        if orderkeys.null_count() == 0 && orderdates.null_count() == 0 {
            for (&orderkey, &orderdate) in orderkeys.values().iter().zip(orderdates.values()) {
                let year = partial.year_cache.year(orderdate)?;
                partial.push(orderkey, year);
            }
            return Ok(Some(()));
        }
        for row in 0..orderkeys.len() {
            if orderkeys.is_null(row) || orderdates.is_null(row) {
                continue;
            }
            let year = partial.year_cache.year(orderdates.value(row))?;
            partial.push(orderkeys.value(row), year);
        }
        return Ok(Some(()));
    }
    for row in 0..batch.num_rows() {
        let (Some(orderkey), Some(orderdate)) = (
            numeric_i64_value(orderkeys, row)?,
            date32_value(orderdates, row)?,
        ) else {
            continue;
        };
        let year = partial.year_cache.year(orderdate)?;
        partial.push(orderkey, year);
    }
    Ok(Some(()))
}

fn q09_order_years_partial_view_into(
    view: BatchView<'_>,
    partial: &mut Q09OrderYearPartial,
) -> Result<Option<()>> {
    if view.num_columns() == 2
        && let (Some(orderkeys), Some(orderdates)) = (view.i64_vector(0), view.date32_vector(1))
    {
        if let (Some(orderkey_values), Some(orderdate_values)) = (
            orderkeys.values_if_null_free(),
            orderdates.values_if_null_free(),
        ) {
            for (&orderkey, &orderdate) in orderkey_values.iter().zip(orderdate_values) {
                let year = partial.year_cache.year(orderdate)?;
                partial.push(orderkey, year);
            }
            return Ok(Some(()));
        }
        for row in 0..orderkeys.len() {
            if orderkeys.is_null(row) || orderdates.is_null(row) {
                continue;
            }
            let year = partial.year_cache.year(orderdates.value(row))?;
            partial.push(orderkeys.value(row), year);
        }
        return Ok(Some(()));
    }
    let Some(batch) = view.try_record_batch() else {
        return Ok(None);
    };
    q09_order_years_partial_batch_into(batch.clone(), partial)
}

fn q09_order_year_row_group_map_chunk() -> usize {
    generic_row_group_map_chunk_size(8)
}

fn q09_order_year_max_dense_entries(row_count: Option<usize>) -> usize {
    let byte_entries = q09_order_year_max_dense_byte_entries();
    if let Some(row_count) = row_count {
        let amplification_entries =
            ((row_count as f64) * q09_order_year_dense_max_amplification()) as usize;
        return byte_entries.min(amplification_entries.max(row_count));
    }
    byte_entries
}

fn q09_order_year_max_dense_byte_entries() -> usize {
    dense_i32_max_entries(DEFAULT_ORDER_YEAR_DENSE_BYTES)
}

fn q09_order_year_dense_max_amplification() -> f64 {
    dense_max_amplification(8.5)
}

fn q09_order_years_batch_into(
    batch: &RecordBatch,
    years: &mut Q09OrderYears,
    year_cache: &mut Date32YearCache,
) -> Result<()> {
    if let Some(fallback) = years.fallback_mut() {
        return q09_order_years_batch_into_fallback(batch, fallback, year_cache);
    }
    let orderkeys = batch_column(batch, "o_orderkey")?;
    let orderdates = batch_column(batch, "o_orderdate")?;
    if let (Some(orderkeys), Some(orderdates)) = (
        orderkeys.as_any().downcast_ref::<Int64Array>(),
        orderdates.as_any().downcast_ref::<Date32Array>(),
    ) {
        let max_dense_entries = q09_order_year_max_dense_entries(None);
        if orderkeys.null_count() == 0 && orderdates.null_count() == 0 {
            let orderkey_values = orderkeys.values().as_ref();
            let orderdate_values = orderdates.values().as_ref();
            let mut min_orderkey = i64::MAX;
            let mut max_orderkey = i64::MIN;
            for &orderkey in orderkey_values {
                if orderkey < 0 {
                    years.convert_to_fallback();
                    let fallback = years.fallback_mut().expect("converted q09 fallback");
                    return q09_order_years_batch_into_fallback(batch, fallback, year_cache);
                }
                min_orderkey = min_orderkey.min(orderkey);
                max_orderkey = max_orderkey.max(orderkey);
            }
            if !years.reserve_dense_range(min_orderkey, max_orderkey, max_dense_entries) {
                years.convert_to_fallback();
                let fallback = years.fallback_mut().expect("converted q09 fallback");
                return q09_order_years_batch_into_fallback(batch, fallback, year_cache);
            }
            for (&orderkey, &orderdate) in orderkey_values.iter().zip(orderdate_values) {
                years.insert_dense_key(orderkey, year_cache.year(orderdate)?);
            }
            return Ok(());
        }
        let mut min_orderkey = i64::MAX;
        let mut max_orderkey = i64::MIN;
        let mut has_key = false;
        for row in 0..orderkeys.len() {
            if orderkeys.is_null(row) || orderdates.is_null(row) {
                continue;
            }
            let orderkey = orderkeys.value(row);
            if orderkey < 0 {
                years.convert_to_fallback();
                let fallback = years.fallback_mut().expect("converted q09 fallback");
                return q09_order_years_batch_into_fallback(batch, fallback, year_cache);
            }
            min_orderkey = min_orderkey.min(orderkey);
            max_orderkey = max_orderkey.max(orderkey);
            has_key = true;
        }
        if has_key && !years.reserve_dense_range(min_orderkey, max_orderkey, max_dense_entries) {
            years.convert_to_fallback();
            let fallback = years.fallback_mut().expect("converted q09 fallback");
            return q09_order_years_batch_into_fallback(batch, fallback, year_cache);
        }
        for row in 0..orderkeys.len() {
            if orderkeys.is_null(row) || orderdates.is_null(row) {
                continue;
            }
            years.insert_dense_key(
                orderkeys.value(row),
                year_cache.year(orderdates.value(row))?,
            );
        }
        return Ok(());
    }
    years.convert_to_fallback();
    let fallback = years.fallback_mut().expect("converted q09 fallback");
    q09_order_years_batch_into_fallback(batch, fallback, year_cache)
}

fn q09_order_years_batch_into_fallback(
    batch: &RecordBatch,
    years: &mut AdaptiveI64Map<i32>,
    year_cache: &mut Date32YearCache,
) -> Result<()> {
    let orderkeys = batch_column(batch, "o_orderkey")?;
    let orderdates = batch_column(batch, "o_orderdate")?;
    for row in 0..orderkeys.len() {
        let (Some(orderkey), Some(orderdate)) = (
            numeric_i64_value(orderkeys, row)?,
            date32_value(orderdates, row)?,
        ) else {
            continue;
        };
        years.insert(orderkey, year_cache.year(orderdate)?);
    }
    Ok(())
}

async fn q09_supply_costs(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    part_keys: &AdaptiveI64Set,
    supplier_nations: &AdaptiveI64Map<i64>,
) -> Result<Q09SupplyCosts> {
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
    let part_keys = Arc::new(part_keys.clone());
    let supplier_nations = Arc::new(supplier_nations.clone());
    let mut costs = parallel_batch_fold(
        &mut stream,
        move |batch| q09_supply_costs_batch(batch, &part_keys, &supplier_nations),
        Q09SupplyCosts::new(),
        Q09SupplyCosts::merge,
        "Q09 supply costs",
    )?;
    costs.compact_for_probe();
    q09_log_supply_cost_layout(&costs);
    Ok(costs)
}

fn q09_supply_costs_batch(
    batch: RecordBatch,
    part_keys: &AdaptiveI64Set,
    supplier_nations: &AdaptiveI64Map<i64>,
) -> Result<Q09SupplyCosts> {
    let partkeys = batch_column(&batch, "ps_partkey")?;
    let suppkeys = batch_column(&batch, "ps_suppkey")?;
    let supplycosts = batch_column(&batch, "ps_supplycost")?;
    if let Some(costs) =
        q09_supply_costs_batch_typed(partkeys, suppkeys, supplycosts, part_keys, supplier_nations)?
    {
        return Ok(costs);
    }
    let mut costs = Q09SupplyCosts::new();
    let dense_supplier_nations = supplier_nations.dense_slices();
    for row in 0..batch.num_rows() {
        let (Some(partkey), Some(suppkey), Some(supplycost)) = (
            numeric_i64_value(partkeys, row)?,
            numeric_i64_value(suppkeys, row)?,
            numeric_f64_value(supplycosts, row)?,
        ) else {
            continue;
        };
        if part_keys.contains(partkey)
            && let Some(nationkey) = supplier_nations.get_cached(dense_supplier_nations, suppkey)
        {
            costs.insert(
                partkey,
                suppkey,
                Q09SupplyInfo {
                    supplycost,
                    nationkey,
                },
            );
        }
    }
    Ok(costs)
}

fn q09_supply_costs_batch_typed(
    partkeys: &ArrayRef,
    suppkeys: &ArrayRef,
    supplycosts: &ArrayRef,
    part_keys: &AdaptiveI64Set,
    supplier_nations: &AdaptiveI64Map<i64>,
) -> Result<Option<Q09SupplyCosts>> {
    let (Some(partkeys), Some(suppkeys), Some(supplycosts)) = (
        partkeys.as_any().downcast_ref::<Int64Array>(),
        suppkeys.as_any().downcast_ref::<Int64Array>(),
        decimal_input(supplycosts)?,
    ) else {
        return Ok(None);
    };
    let mut costs = Q09SupplyCosts::new();
    let dense_part_keys = part_keys.dense_contains_slice();
    let dense_supplier_nations = supplier_nations.dense_slices();
    for row in 0..partkeys.len() {
        if partkeys.is_null(row) || suppkeys.is_null(row) || supplycosts.is_null(row) {
            continue;
        }
        let partkey = partkeys.value(row);
        if part_keys.contains_cached(dense_part_keys, partkey) {
            let suppkey = suppkeys.value(row);
            if let Some(nationkey) = supplier_nations.get_cached(dense_supplier_nations, suppkey) {
                costs.insert(
                    partkey,
                    suppkey,
                    Q09SupplyInfo {
                        supplycost: supplycosts.value(row),
                        nationkey,
                    },
                );
            }
        }
    }
    Ok(Some(costs))
}

#[derive(Clone, Copy, Default)]
struct Q09SupplyInfo {
    supplycost: f64,
    nationkey: i64,
}

enum Q09SupplyCosts {
    PairHash(FastHashMap<(i64, i64), Q09SupplyInfo>),
    PackedU64(FastHashMap<u64, Q09SupplyInfo>),
}

impl Q09SupplyCosts {
    fn new() -> Self {
        Self::PackedU64(fast_hash_map())
    }

    fn insert(&mut self, partkey: i64, suppkey: i64, info: Q09SupplyInfo) {
        match self {
            Self::PairHash(costs) => {
                costs.insert((partkey, suppkey), info);
            }
            Self::PackedU64(costs) => {
                let Some(key) = q09_pack_part_supp_key(partkey, suppkey) else {
                    self.convert_packed_to_pair_hash();
                    self.insert(partkey, suppkey, info);
                    return;
                };
                costs.insert(key, info);
            }
        }
    }

    fn get(&self, partkey: i64, suppkey: i64) -> Option<Q09SupplyInfo> {
        match self {
            Self::PairHash(costs) => costs.get(&(partkey, suppkey)).copied(),
            Self::PackedU64(costs) => {
                q09_pack_part_supp_key(partkey, suppkey).and_then(|key| costs.get(&key).copied())
            }
        }
    }

    fn merge(&mut self, batch: Self) {
        match batch {
            Self::PairHash(batch) => {
                for ((partkey, suppkey), supplycost) in batch {
                    self.insert(partkey, suppkey, supplycost);
                }
            }
            Self::PackedU64(batch) => {
                for (key, supplycost) in batch {
                    let (partkey, suppkey) = q09_unpack_part_supp_key(key);
                    self.insert(partkey, suppkey, supplycost);
                }
            }
        }
    }

    fn compact_for_probe(&mut self) {}

    fn convert_packed_to_pair_hash(&mut self) {
        let Self::PackedU64(packed) = self else {
            return;
        };
        let mut pair_hash = fast_hash_map_with_capacity(packed.len());
        for (key, supplycost) in std::mem::take(packed) {
            pair_hash.insert(q09_unpack_part_supp_key(key), supplycost);
        }
        *self = Self::PairHash(pair_hash);
    }

    fn layout_name(&self) -> &'static str {
        match self {
            Self::PairHash(_) => "pair_hash",
            Self::PackedU64(_) => "packed_u64",
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::PairHash(costs) => costs.len(),
            Self::PackedU64(costs) => costs.len(),
        }
    }

    fn part_keys_len(&self) -> usize {
        match self {
            Self::PairHash(costs) => costs
                .keys()
                .map(|(partkey, _)| *partkey)
                .collect::<FastHashSet<_>>()
                .len(),
            Self::PackedU64(costs) => costs
                .keys()
                .map(|key| q09_unpack_part_supp_key(*key).0)
                .collect::<FastHashSet<_>>()
                .len(),
        }
    }

    fn max_fanout(&self) -> usize {
        0
    }
}

fn q09_log_supply_cost_layout(costs: &Q09SupplyCosts) {
    if !tpch_profile_enabled() {
        return;
    }
    eprintln!(
        "[dodam:tpch-profile] Q09 supply cost layout: layout={} entries={} part_keys={} max_fanout={}",
        costs.layout_name(),
        costs.len(),
        costs.part_keys_len(),
        costs.max_fanout()
    );
}

fn q09_pack_part_supp_key(partkey: i64, suppkey: i64) -> Option<u64> {
    let partkey = u32::try_from(partkey).ok()?;
    let suppkey = u32::try_from(suppkey).ok()?;
    Some((u64::from(partkey) << 32) | u64::from(suppkey))
}

fn q09_unpack_part_supp_key(key: u64) -> (i64, i64) {
    ((key >> 32) as i64, (key as u32) as i64)
}

struct Q09Row {
    nation: String,
    o_year: i32,
    sum_profit: f64,
}

async fn q09_profit_rows(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    part_keys: &AdaptiveI64Set,
    nation_names: &HashMap<i64, String>,
    order_years: Q09OrderYears,
    supply_costs: Q09SupplyCosts,
) -> Result<Vec<Q09Row>> {
    let part_keys = Arc::new(part_keys.clone());
    let order_years = Arc::new(order_years);
    let supply_costs = Arc::new(supply_costs);
    let projection = Projection::Columns(vec![
        "l_orderkey".to_string(),
        "l_partkey".to_string(),
        "l_suppkey".to_string(),
        "l_quantity".to_string(),
        "l_extendedprice".to_string(),
        "l_discount".to_string(),
    ]);
    let part_keys_for_scan = part_keys.clone();
    let order_years_for_scan = order_years.clone();
    let supply_costs_for_scan = supply_costs.clone();
    if let Some(partials) = engine
        .parquet_row_group_map_view(
            path.clone(),
            batch_size,
            projection.clone(),
            q09_row_group_map_chunk(),
            Q09ProfitPartial::default,
            move |view, partial| {
                partial.merge(q09_profit_projected_view(
                    view,
                    &part_keys_for_scan,
                    &order_years_for_scan,
                    &supply_costs_for_scan,
                )?);
                Ok(Some(()))
            },
            |partial| Ok(Some(partial)),
        )
        .await?
    {
        let mut partial = Q09ProfitPartial::default();
        for batch in partials {
            partial.merge(batch);
        }
        q09_log_profit_profile(&partial.profile);
        return q09_profit_rows_from_groups(partial.groups, nation_names);
    }
    let mut stream = engine
        .scan_parquet_batches(path, batch_size, None, projection, None)
        .await?;
    let partial = parallel_batch_fold(
        &mut stream,
        move |batch| q09_profit_batch(batch, &part_keys, &order_years, &supply_costs),
        Q09ProfitPartial::default(),
        Q09ProfitPartial::merge,
        "Q09 profit aggregate",
    )?;
    q09_log_profit_profile(&partial.profile);
    q09_profit_rows_from_groups(partial.groups, nation_names)
}

fn q09_row_group_map_chunk() -> usize {
    generic_row_group_map_chunk_size(4)
}

fn q09_profit_rows_from_groups(
    groups: Q09ProfitGroups,
    nation_names: &HashMap<i64, String>,
) -> Result<Vec<Q09Row>> {
    let mut rows = groups
        .into_iter()
        .filter_map(|((nationkey, o_year), sum_profit)| {
            nation_names.get(&nationkey).map(|nation| Q09Row {
                nation: nation.clone(),
                o_year,
                sum_profit,
            })
        })
        .collect::<Vec<_>>();
    rows.sort_by(|left, right| {
        left.nation
            .cmp(&right.nation)
            .then_with(|| right.o_year.cmp(&left.o_year))
    });
    Ok(rows)
}

#[derive(Default)]
struct Q09ProfitPartial {
    groups: Q09ProfitGroups,
    profile: Q09ProfitProfile,
}

impl Q09ProfitPartial {
    fn merge(&mut self, batch: Self) {
        self.groups.merge(batch.groups);
        self.profile.add(batch.profile);
    }
}

#[derive(Default)]
struct Q09ProfitGroups {
    dense: Vec<f64>,
    packed: FastHashMap<u64, f64>,
    fallback: FastHashMap<(i64, i32), f64>,
}

impl Q09ProfitGroups {
    fn add(&mut self, nationkey: i64, year: i32, amount: f64) {
        if let Some(slot) = q09_dense_profit_group_slot(nationkey, year) {
            if self.dense.is_empty() {
                self.dense.resize(
                    Q09_DENSE_PROFIT_GROUP_NATIONS * Q09_DENSE_PROFIT_GROUP_YEARS,
                    0.0,
                );
            }
            self.dense[slot] += amount;
            return;
        }
        if let Some(key) = q09_pack_profit_group_key(nationkey, year) {
            *self.packed.entry(key).or_insert(0.0) += amount;
        } else {
            *self.fallback.entry((nationkey, year)).or_insert(0.0) += amount;
        }
    }

    fn merge(&mut self, other: Self) {
        if !other.dense.is_empty() {
            if self.dense.is_empty() {
                self.dense.resize(
                    Q09_DENSE_PROFIT_GROUP_NATIONS * Q09_DENSE_PROFIT_GROUP_YEARS,
                    0.0,
                );
            }
            for (slot, amount) in other.dense.into_iter().enumerate() {
                self.dense[slot] += amount;
            }
        }
        for (key, amount) in other.packed {
            *self.packed.entry(key).or_insert(0.0) += amount;
        }
        merge_f64_groups(&mut self.fallback, other.fallback);
    }

    fn into_iter(self) -> impl Iterator<Item = ((i64, i32), f64)> {
        let dense = self
            .dense
            .into_iter()
            .enumerate()
            .filter(|(_, amount)| *amount != 0.0)
            .map(|(slot, amount)| (q09_unpack_dense_profit_group_slot(slot), amount));
        dense.chain(
            self.packed
                .into_iter()
                .map(|(key, amount)| (q09_unpack_profit_group_key(key), amount))
                .chain(self.fallback),
        )
    }
}

const Q09_DENSE_PROFIT_GROUP_NATIONS: usize = 64;
const Q09_DENSE_PROFIT_GROUP_YEAR_BASE: i32 = 1990;
const Q09_DENSE_PROFIT_GROUP_YEARS: usize = 32;

fn q09_dense_profit_group_slot(nationkey: i64, year: i32) -> Option<usize> {
    let nation = usize::try_from(nationkey).ok()?;
    if nation >= Q09_DENSE_PROFIT_GROUP_NATIONS {
        return None;
    }
    let year = usize::try_from(year.checked_sub(Q09_DENSE_PROFIT_GROUP_YEAR_BASE)?).ok()?;
    if year >= Q09_DENSE_PROFIT_GROUP_YEARS {
        return None;
    }
    Some(nation * Q09_DENSE_PROFIT_GROUP_YEARS + year)
}

fn q09_unpack_dense_profit_group_slot(slot: usize) -> (i64, i32) {
    let nation = slot / Q09_DENSE_PROFIT_GROUP_YEARS;
    let year = Q09_DENSE_PROFIT_GROUP_YEAR_BASE + (slot % Q09_DENSE_PROFIT_GROUP_YEARS) as i32;
    (nation as i64, year)
}

fn q09_pack_profit_group_key(nationkey: i64, year: i32) -> Option<u64> {
    let nationkey = u32::try_from(nationkey).ok()?;
    Some((u64::from(nationkey) << 32) | u64::from(year as u32))
}

fn q09_unpack_profit_group_key(key: u64) -> (i64, i32) {
    ((key >> 32) as i64, (key as u32) as i32)
}

#[derive(Default)]
struct Q09ProfitProfile {
    rows: usize,
    part_hits: usize,
    order_hits: usize,
    supplier_hits: usize,
    supply_hits: usize,
    amount_rows: usize,
    part_nanos: u64,
    order_nanos: u64,
    supplier_nanos: u64,
    supply_nanos: u64,
    amount_nanos: u64,
    direct_i64_packed_dense_batches: usize,
    direct_i64_selected_packed_dense_batches: usize,
    direct_i64_fallback_batches: usize,
    direct_i64_selected_fallback_batches: usize,
    view_i64_packed_dense_batches: usize,
    view_i64_fallback_batches: usize,
    view_i128_packed_dense_batches: usize,
    view_i128_fallback_batches: usize,
    batch_i128_packed_dense_batches: usize,
    batch_i128_fallback_batches: usize,
    batch_null_fallback_batches: usize,
}

impl Q09ProfitProfile {
    fn add(&mut self, other: Self) {
        self.rows = self.rows.saturating_add(other.rows);
        self.part_hits = self.part_hits.saturating_add(other.part_hits);
        self.order_hits = self.order_hits.saturating_add(other.order_hits);
        self.supplier_hits = self.supplier_hits.saturating_add(other.supplier_hits);
        self.supply_hits = self.supply_hits.saturating_add(other.supply_hits);
        self.amount_rows = self.amount_rows.saturating_add(other.amount_rows);
        self.part_nanos = self.part_nanos.saturating_add(other.part_nanos);
        self.order_nanos = self.order_nanos.saturating_add(other.order_nanos);
        self.supplier_nanos = self.supplier_nanos.saturating_add(other.supplier_nanos);
        self.supply_nanos = self.supply_nanos.saturating_add(other.supply_nanos);
        self.amount_nanos = self.amount_nanos.saturating_add(other.amount_nanos);
        self.direct_i64_packed_dense_batches = self
            .direct_i64_packed_dense_batches
            .saturating_add(other.direct_i64_packed_dense_batches);
        self.direct_i64_selected_packed_dense_batches = self
            .direct_i64_selected_packed_dense_batches
            .saturating_add(other.direct_i64_selected_packed_dense_batches);
        self.direct_i64_fallback_batches = self
            .direct_i64_fallback_batches
            .saturating_add(other.direct_i64_fallback_batches);
        self.direct_i64_selected_fallback_batches = self
            .direct_i64_selected_fallback_batches
            .saturating_add(other.direct_i64_selected_fallback_batches);
        self.view_i64_packed_dense_batches = self
            .view_i64_packed_dense_batches
            .saturating_add(other.view_i64_packed_dense_batches);
        self.view_i64_fallback_batches = self
            .view_i64_fallback_batches
            .saturating_add(other.view_i64_fallback_batches);
        self.view_i128_packed_dense_batches = self
            .view_i128_packed_dense_batches
            .saturating_add(other.view_i128_packed_dense_batches);
        self.view_i128_fallback_batches = self
            .view_i128_fallback_batches
            .saturating_add(other.view_i128_fallback_batches);
        self.batch_i128_packed_dense_batches = self
            .batch_i128_packed_dense_batches
            .saturating_add(other.batch_i128_packed_dense_batches);
        self.batch_i128_fallback_batches = self
            .batch_i128_fallback_batches
            .saturating_add(other.batch_i128_fallback_batches);
        self.batch_null_fallback_batches = self
            .batch_null_fallback_batches
            .saturating_add(other.batch_null_fallback_batches);
    }
}

fn q09_log_profit_profile(profile: &Q09ProfitProfile) {
    if !tpch_profile_enabled() {
        return;
    }
    eprintln!(
        "[dodam:tpch-profile] Q09 profit detail: rows={} part_hits={} order_hits={} supplier_hits={} supply_hits={} amount_rows={} part={:.3} ms order={:.3} ms supplier={:.3} ms supply={:.3} ms amount={:.3} ms",
        profile.rows,
        profile.part_hits,
        profile.order_hits,
        profile.supplier_hits,
        profile.supply_hits,
        profile.amount_rows,
        sql_nanos_to_millis(profile.part_nanos),
        sql_nanos_to_millis(profile.order_nanos),
        sql_nanos_to_millis(profile.supplier_nanos),
        sql_nanos_to_millis(profile.supply_nanos),
        sql_nanos_to_millis(profile.amount_nanos),
    );
    eprintln!(
        "[dodam:tpch-profile] Q09 profit branches: direct_i64_packed_dense={} direct_i64_selected_packed_dense={} direct_i64_fallback={} direct_i64_selected_fallback={} view_i64_packed_dense={} view_i64_fallback={} view_i128_packed_dense={} view_i128_fallback={} batch_i128_packed_dense={} batch_i128_fallback={} batch_null_fallback={}",
        profile.direct_i64_packed_dense_batches,
        profile.direct_i64_selected_packed_dense_batches,
        profile.direct_i64_fallback_batches,
        profile.direct_i64_selected_fallback_batches,
        profile.view_i64_packed_dense_batches,
        profile.view_i64_fallback_batches,
        profile.view_i128_packed_dense_batches,
        profile.view_i128_fallback_batches,
        profile.batch_i128_packed_dense_batches,
        profile.batch_i128_fallback_batches,
        profile.batch_null_fallback_batches,
    );
}

fn q09_profit_batch(
    batch: RecordBatch,
    part_keys: &AdaptiveI64Set,
    order_years: &Q09OrderYears,
    supply_costs: &Q09SupplyCosts,
) -> Result<Q09ProfitPartial> {
    let orderkeys = batch_column(&batch, "l_orderkey")?;
    let partkeys = batch_column(&batch, "l_partkey")?;
    let suppkeys = batch_column(&batch, "l_suppkey")?;
    let quantities = batch_column(&batch, "l_quantity")?;
    let extendedprices = batch_column(&batch, "l_extendedprice")?;
    let discounts = batch_column(&batch, "l_discount")?;
    if let Some(groups) = q09_profit_decimal_batch(
        orderkeys,
        partkeys,
        suppkeys,
        quantities,
        extendedprices,
        discounts,
        part_keys,
        order_years,
        supply_costs,
    )? {
        return Ok(groups);
    }
    let mut groups = Q09ProfitGroups::default();
    let mut profile = Q09ProfitProfile::default();
    let collect_profile = tpch_profile_enabled();
    let dense_part_keys = part_keys.dense_contains_slice();
    let dense_order_years = order_years.dense_slice();
    for row in 0..batch.num_rows() {
        if collect_profile {
            profile.rows += 1;
        }
        let (Some(orderkey), Some(partkey), Some(suppkey)) = (
            numeric_i64_value(orderkeys, row)?,
            numeric_i64_value(partkeys, row)?,
            numeric_i64_value(suppkeys, row)?,
        ) else {
            continue;
        };
        let started = collect_profile.then(Instant::now);
        let part_hit = part_keys.contains_cached(dense_part_keys, partkey);
        if let Some(started) = started {
            profile.part_nanos = profile
                .part_nanos
                .saturating_add(sql_elapsed_nanos(started));
        }
        if !part_hit {
            continue;
        }
        if collect_profile {
            profile.part_hits += 1;
        }
        let started = collect_profile.then(Instant::now);
        let o_year = q09_order_year_get(order_years, dense_order_years, orderkey);
        if let Some(started) = started {
            profile.order_nanos = profile
                .order_nanos
                .saturating_add(sql_elapsed_nanos(started));
        }
        let Some(o_year) = o_year else {
            continue;
        };
        if collect_profile {
            profile.order_hits += 1;
        }
        let started = collect_profile.then(Instant::now);
        let supplycost = supply_costs.get(partkey, suppkey);
        if let Some(started) = started {
            profile.supply_nanos = profile
                .supply_nanos
                .saturating_add(sql_elapsed_nanos(started));
        }
        let Some(supplycost) = supplycost else {
            continue;
        };
        let nationkey = supplycost.nationkey;
        if collect_profile {
            profile.supplier_hits += 1;
        }
        if collect_profile {
            profile.supply_hits += 1;
        }
        let (Some(quantity), Some(extendedprice), Some(discount)) = (
            numeric_f64_value(quantities, row)?,
            numeric_f64_value(extendedprices, row)?,
            numeric_f64_value(discounts, row)?,
        ) else {
            continue;
        };
        let started = collect_profile.then(Instant::now);
        let amount = discounted_revenue_minus_product(
            extendedprice,
            discount,
            supplycost.supplycost,
            quantity,
        );
        if let Some(started) = started {
            profile.amount_nanos = profile
                .amount_nanos
                .saturating_add(sql_elapsed_nanos(started));
        }
        if collect_profile {
            profile.amount_rows += 1;
        }
        groups.add(nationkey, o_year, amount);
    }
    Ok(Q09ProfitPartial { groups, profile })
}

fn q09_profit_projected_view(
    view: BatchView<'_>,
    part_keys: &AdaptiveI64Set,
    order_years: &Q09OrderYears,
    supply_costs: &Q09SupplyCosts,
) -> Result<Q09ProfitPartial> {
    if view.num_columns() == 6
        && let (
            Some(orderkeys),
            Some(partkeys),
            Some(suppkeys),
            Some(quantities),
            Some(extendedprices),
            Some(discounts),
        ) = (
            view.i64_vector(0),
            view.i64_vector(1),
            view.i64_vector(2),
            view.decimal128_vector(3),
            view.decimal128_vector(4),
            view.decimal128_vector(5),
        )
        && let Some(groups) = q09_profit_decimal_view(
            orderkeys,
            partkeys,
            suppkeys,
            quantities,
            extendedprices,
            discounts,
            part_keys,
            order_years,
            supply_costs,
        )
    {
        return Ok(groups);
    }
    let Some(batch) = view.try_record_batch() else {
        return Ok(Q09ProfitPartial::default());
    };
    q09_profit_batch(batch.clone(), part_keys, order_years, supply_costs)
}

fn q09_order_year_get(
    order_years: &Q09OrderYears,
    dense_order_years: Option<(&[i32], i64, i32)>,
    orderkey: i64,
) -> Option<i32> {
    if let Some((values, base_key, missing)) = dense_order_years {
        let index = usize::try_from(orderkey.checked_sub(base_key)?).ok()?;
        return values.get(index).copied().filter(|value| *value != missing);
    }
    order_years.get(orderkey)
}

fn q09_profit_decimal_batch(
    orderkeys: &ArrayRef,
    partkeys: &ArrayRef,
    suppkeys: &ArrayRef,
    quantities: &ArrayRef,
    extendedprices: &ArrayRef,
    discounts: &ArrayRef,
    part_keys: &AdaptiveI64Set,
    order_years: &Q09OrderYears,
    supply_costs: &Q09SupplyCosts,
) -> Result<Option<Q09ProfitPartial>> {
    let (
        Some(orderkeys),
        Some(partkeys),
        Some(suppkeys),
        Some(quantities),
        Some(extendedprices),
        Some(discounts),
    ) = (
        orderkeys.as_any().downcast_ref::<Int64Array>(),
        partkeys.as_any().downcast_ref::<Int64Array>(),
        suppkeys.as_any().downcast_ref::<Int64Array>(),
        decimal_input(quantities)?,
        decimal_input(extendedprices)?,
        decimal_input(discounts)?,
    )
    else {
        return Ok(None);
    };

    let mut groups = Q09ProfitGroups::default();
    let mut profile = Q09ProfitProfile::default();
    let collect_profile = tpch_profile_enabled();
    let dense_part_keys = part_keys.dense_contains_slice();
    let dense_part_words = part_keys.dense_word_slice();
    let dense_order_years = order_years.dense_slice();
    let quantity_values = quantities.raw_values();
    let extendedprice_values = extendedprices.raw_values();
    let discount_values = discounts.raw_values();
    let discount_scale = discounts.scale;
    let revenue_scale = 1.0 / (extendedprices.scale * discount_scale);
    let quantity_scale = 1.0 / quantities.scale;
    if orderkeys.null_count() == 0
        && partkeys.null_count() == 0
        && suppkeys.null_count() == 0
        && quantities.null_count() == 0
        && extendedprices.null_count() == 0
        && discounts.null_count() == 0
    {
        let orderkey_values = orderkeys.values().as_ref();
        let partkey_values = partkeys.values().as_ref();
        let suppkey_values = suppkeys.values().as_ref();
        if let (Some(dense_part_keys), Some(dense_order_years), Q09SupplyCosts::PackedU64(costs)) =
            (dense_part_keys, dense_order_years, supply_costs)
        {
            let mut partial = q09_profit_decimal_packed_dense_loop_with_words(
                orderkey_values,
                partkey_values,
                suppkey_values,
                quantity_values,
                extendedprice_values,
                discount_values,
                discount_scale,
                revenue_scale,
                quantity_scale,
                dense_part_keys,
                dense_part_words,
                dense_order_years,
                costs,
                collect_profile,
            );
            if collect_profile {
                partial.profile.batch_i128_packed_dense_batches = 1;
            }
            return Ok(Some(partial));
        }
        if collect_profile {
            profile.batch_i128_fallback_batches = 1;
        }
        for row in 0..orderkeys.len() {
            if collect_profile {
                profile.rows += 1;
            }
            let partkey = partkey_values[row];
            let started = collect_profile.then(Instant::now);
            let part_hit = part_keys.contains_cached(dense_part_keys, partkey);
            if let Some(started) = started {
                profile.part_nanos = profile
                    .part_nanos
                    .saturating_add(sql_elapsed_nanos(started));
            }
            if !part_hit {
                continue;
            }
            if collect_profile {
                profile.part_hits += 1;
            }
            let orderkey = orderkey_values[row];
            let suppkey = suppkey_values[row];
            let started = collect_profile.then(Instant::now);
            let o_year = q09_order_year_get(order_years, dense_order_years, orderkey);
            if let Some(started) = started {
                profile.order_nanos = profile
                    .order_nanos
                    .saturating_add(sql_elapsed_nanos(started));
            }
            let Some(o_year) = o_year else {
                continue;
            };
            if collect_profile {
                profile.order_hits += 1;
            }
            let started = collect_profile.then(Instant::now);
            let supplycost = supply_costs.get(partkey, suppkey);
            if let Some(started) = started {
                profile.supply_nanos = profile
                    .supply_nanos
                    .saturating_add(sql_elapsed_nanos(started));
            }
            let Some(supplycost) = supplycost else {
                continue;
            };
            let nationkey = supplycost.nationkey;
            if collect_profile {
                profile.supplier_hits += 1;
            }
            if collect_profile {
                profile.supply_hits += 1;
            }
            let started = collect_profile.then(Instant::now);
            let amount = (extendedprice_values[row] as f64)
                * (discount_scale - discount_values[row] as f64)
                * revenue_scale
                - supplycost.supplycost * (quantity_values[row] as f64) * quantity_scale;
            if let Some(started) = started {
                profile.amount_nanos = profile
                    .amount_nanos
                    .saturating_add(sql_elapsed_nanos(started));
            }
            if collect_profile {
                profile.amount_rows += 1;
            }
            groups.add(nationkey, o_year, amount);
        }
        return Ok(Some(Q09ProfitPartial { groups, profile }));
    }
    if collect_profile {
        profile.batch_null_fallback_batches = 1;
    }
    for row in 0..orderkeys.len() {
        if collect_profile {
            profile.rows += 1;
        }
        if orderkeys.is_null(row)
            || partkeys.is_null(row)
            || suppkeys.is_null(row)
            || quantities.is_null(row)
            || extendedprices.is_null(row)
            || discounts.is_null(row)
        {
            continue;
        }
        let partkey = partkeys.value(row);
        let started = collect_profile.then(Instant::now);
        let part_hit = part_keys.contains_cached(dense_part_keys, partkey);
        if let Some(started) = started {
            profile.part_nanos = profile
                .part_nanos
                .saturating_add(sql_elapsed_nanos(started));
        }
        if !part_hit {
            continue;
        }
        if collect_profile {
            profile.part_hits += 1;
        }
        let orderkey = orderkeys.value(row);
        let suppkey = suppkeys.value(row);
        let started = collect_profile.then(Instant::now);
        let o_year = q09_order_year_get(order_years, dense_order_years, orderkey);
        if let Some(started) = started {
            profile.order_nanos = profile
                .order_nanos
                .saturating_add(sql_elapsed_nanos(started));
        }
        let Some(o_year) = o_year else {
            continue;
        };
        if collect_profile {
            profile.order_hits += 1;
        }
        let started = collect_profile.then(Instant::now);
        let supplycost = supply_costs.get(partkey, suppkey);
        if let Some(started) = started {
            profile.supply_nanos = profile
                .supply_nanos
                .saturating_add(sql_elapsed_nanos(started));
        }
        let Some(supplycost) = supplycost else {
            continue;
        };
        let nationkey = supplycost.nationkey;
        if collect_profile {
            profile.supplier_hits += 1;
        }
        if collect_profile {
            profile.supply_hits += 1;
        }
        let started = collect_profile.then(Instant::now);
        let amount = (extendedprice_values[row] as f64)
            * (discount_scale - discount_values[row] as f64)
            * revenue_scale
            - supplycost.supplycost * (quantity_values[row] as f64) * quantity_scale;
        if let Some(started) = started {
            profile.amount_nanos = profile
                .amount_nanos
                .saturating_add(sql_elapsed_nanos(started));
        }
        if collect_profile {
            profile.amount_rows += 1;
        }
        groups.add(nationkey, o_year, amount);
    }
    Ok(Some(Q09ProfitPartial { groups, profile }))
}

#[allow(clippy::too_many_arguments)]
fn q09_profit_decimal_view(
    orderkeys: I64VectorView<'_>,
    partkeys: I64VectorView<'_>,
    suppkeys: I64VectorView<'_>,
    quantities: Decimal128VectorView<'_>,
    extendedprices: Decimal128VectorView<'_>,
    discounts: Decimal128VectorView<'_>,
    part_keys: &AdaptiveI64Set,
    order_years: &Q09OrderYears,
    supply_costs: &Q09SupplyCosts,
) -> Option<Q09ProfitPartial> {
    let mut groups = Q09ProfitGroups::default();
    let mut profile = Q09ProfitProfile::default();
    let collect_profile = tpch_profile_enabled();
    let dense_part_keys = part_keys.dense_contains_slice();
    let dense_part_words = part_keys.dense_word_slice();
    let dense_order_years = order_years.dense_slice();
    if let (
        Some(orderkey_values),
        Some(partkey_values),
        Some(suppkey_values),
        Some(quantity_values),
        Some(extendedprice_values),
        Some(discount_values),
    ) = (
        orderkeys.values_if_null_free(),
        partkeys.values_if_null_free(),
        suppkeys.values_if_null_free(),
        quantities.raw_i64_values(),
        extendedprices.raw_i64_values(),
        discounts.raw_i64_values(),
    ) && quantities.null_count() == 0
        && extendedprices.null_count() == 0
        && discounts.null_count() == 0
    {
        let discount_scale = discounts.scale();
        let revenue_scale = 1.0 / (extendedprices.scale() * discount_scale);
        let quantity_scale = 1.0 / quantities.scale();
        if let (Some(dense_part_keys), Some(dense_order_years), Q09SupplyCosts::PackedU64(costs)) =
            (dense_part_keys, dense_order_years, supply_costs)
        {
            let mut partial = q09_profit_decimal_i64_packed_dense_loop_with_words(
                orderkey_values,
                partkey_values,
                suppkey_values,
                quantity_values,
                extendedprice_values,
                discount_values,
                discount_scale,
                revenue_scale,
                quantity_scale,
                dense_part_keys,
                dense_part_words,
                dense_order_years,
                costs,
                collect_profile,
            );
            if collect_profile {
                partial.profile.view_i64_packed_dense_batches = 1;
            }
            return Some(partial);
        }
        let mut partial = q09_profit_decimal_i64_view_loop(
            orderkey_values,
            partkey_values,
            suppkey_values,
            quantity_values,
            extendedprice_values,
            discount_values,
            discount_scale,
            revenue_scale,
            quantity_scale,
            part_keys,
            order_years,
            supply_costs,
            collect_profile,
        );
        if collect_profile {
            partial.profile.view_i64_fallback_batches = 1;
        }
        return Some(partial);
    }
    let quantity_values = quantities.raw_values();
    let extendedprice_values = extendedprices.raw_values();
    let discount_values = discounts.raw_values();
    let discount_scale = discounts.scale();
    let revenue_scale = 1.0 / (extendedprices.scale() * discount_scale);
    let quantity_scale = 1.0 / quantities.scale();
    if let (Some(orderkey_values), Some(partkey_values), Some(suppkey_values)) = (
        orderkeys.values_if_null_free(),
        partkeys.values_if_null_free(),
        suppkeys.values_if_null_free(),
    ) && quantities.null_count() == 0
        && extendedprices.null_count() == 0
        && discounts.null_count() == 0
    {
        if let (Some(dense_part_keys), Some(dense_order_years), Q09SupplyCosts::PackedU64(costs)) =
            (dense_part_keys, dense_order_years, supply_costs)
        {
            let mut partial = q09_profit_decimal_packed_dense_loop_with_words(
                orderkey_values,
                partkey_values,
                suppkey_values,
                quantity_values,
                extendedprice_values,
                discount_values,
                discount_scale,
                revenue_scale,
                quantity_scale,
                dense_part_keys,
                dense_part_words,
                dense_order_years,
                costs,
                collect_profile,
            );
            if collect_profile {
                partial.profile.view_i128_packed_dense_batches = 1;
            }
            return Some(partial);
        }
        if collect_profile {
            profile.view_i128_fallback_batches = 1;
        }
        for row in 0..orderkey_values.len() {
            if collect_profile {
                profile.rows += 1;
            }
            let partkey = partkey_values[row];
            let started = collect_profile.then(Instant::now);
            let part_hit = part_keys.contains_cached(dense_part_keys, partkey);
            if let Some(started) = started {
                profile.part_nanos = profile
                    .part_nanos
                    .saturating_add(sql_elapsed_nanos(started));
            }
            if !part_hit {
                continue;
            }
            if collect_profile {
                profile.part_hits += 1;
            }
            let orderkey = orderkey_values[row];
            let suppkey = suppkey_values[row];
            let started = collect_profile.then(Instant::now);
            let o_year = q09_order_year_get(order_years, dense_order_years, orderkey);
            if let Some(started) = started {
                profile.order_nanos = profile
                    .order_nanos
                    .saturating_add(sql_elapsed_nanos(started));
            }
            let Some(o_year) = o_year else {
                continue;
            };
            if collect_profile {
                profile.order_hits += 1;
            }
            let started = collect_profile.then(Instant::now);
            let supplycost = supply_costs.get(partkey, suppkey);
            if let Some(started) = started {
                profile.supply_nanos = profile
                    .supply_nanos
                    .saturating_add(sql_elapsed_nanos(started));
            }
            let Some(supplycost) = supplycost else {
                continue;
            };
            let nationkey = supplycost.nationkey;
            if collect_profile {
                profile.supplier_hits += 1;
            }
            if collect_profile {
                profile.supply_hits += 1;
            }
            let started = collect_profile.then(Instant::now);
            let amount = (extendedprice_values[row] as f64)
                * (discount_scale - discount_values[row] as f64)
                * revenue_scale
                - supplycost.supplycost * (quantity_values[row] as f64) * quantity_scale;
            if let Some(started) = started {
                profile.amount_nanos = profile
                    .amount_nanos
                    .saturating_add(sql_elapsed_nanos(started));
            }
            if collect_profile {
                profile.amount_rows += 1;
            }
            groups.add(nationkey, o_year, amount);
        }
        return Some(Q09ProfitPartial { groups, profile });
    }
    if collect_profile {
        profile.view_i128_fallback_batches = 1;
    }
    for row in 0..orderkeys.len() {
        if collect_profile {
            profile.rows += 1;
        }
        if orderkeys.is_null(row)
            || partkeys.is_null(row)
            || suppkeys.is_null(row)
            || quantities.is_null(row)
            || extendedprices.is_null(row)
            || discounts.is_null(row)
        {
            continue;
        }
        let partkey = partkeys.value(row);
        let started = collect_profile.then(Instant::now);
        let part_hit = part_keys.contains_cached(dense_part_keys, partkey);
        if let Some(started) = started {
            profile.part_nanos = profile
                .part_nanos
                .saturating_add(sql_elapsed_nanos(started));
        }
        if !part_hit {
            continue;
        }
        if collect_profile {
            profile.part_hits += 1;
        }
        let orderkey = orderkeys.value(row);
        let suppkey = suppkeys.value(row);
        let started = collect_profile.then(Instant::now);
        let o_year = q09_order_year_get(order_years, dense_order_years, orderkey);
        if let Some(started) = started {
            profile.order_nanos = profile
                .order_nanos
                .saturating_add(sql_elapsed_nanos(started));
        }
        let Some(o_year) = o_year else {
            continue;
        };
        if collect_profile {
            profile.order_hits += 1;
        }
        let started = collect_profile.then(Instant::now);
        let supplycost = supply_costs.get(partkey, suppkey);
        if let Some(started) = started {
            profile.supply_nanos = profile
                .supply_nanos
                .saturating_add(sql_elapsed_nanos(started));
        }
        let Some(supplycost) = supplycost else {
            continue;
        };
        let nationkey = supplycost.nationkey;
        if collect_profile {
            profile.supplier_hits += 1;
        }
        if collect_profile {
            profile.supply_hits += 1;
        }
        let started = collect_profile.then(Instant::now);
        let amount = extendedprices.value(row) * (1.0 - discounts.value(row))
            - supplycost.supplycost * quantities.value(row);
        if let Some(started) = started {
            profile.amount_nanos = profile
                .amount_nanos
                .saturating_add(sql_elapsed_nanos(started));
        }
        if collect_profile {
            profile.amount_rows += 1;
        }
        groups.add(nationkey, o_year, amount);
    }
    Some(Q09ProfitPartial { groups, profile })
}

#[allow(clippy::too_many_arguments, dead_code)]
fn q09_profit_decimal_packed_dense_loop(
    orderkey_values: &[i64],
    partkey_values: &[i64],
    suppkey_values: &[i64],
    quantity_values: &[i128],
    extendedprice_values: &[i128],
    discount_values: &[i128],
    discount_scale: f64,
    revenue_scale: f64,
    quantity_scale: f64,
    dense_part_keys: &[bool],
    dense_order_years: (&[i32], i64, i32),
    supply_costs: &FastHashMap<u64, Q09SupplyInfo>,
    collect_profile: bool,
) -> Q09ProfitPartial {
    q09_profit_decimal_packed_dense_loop_with_words(
        orderkey_values,
        partkey_values,
        suppkey_values,
        quantity_values,
        extendedprice_values,
        discount_values,
        discount_scale,
        revenue_scale,
        quantity_scale,
        dense_part_keys,
        None,
        dense_order_years,
        supply_costs,
        collect_profile,
    )
}

#[allow(clippy::too_many_arguments)]
fn q09_profit_decimal_packed_dense_loop_with_words(
    orderkey_values: &[i64],
    partkey_values: &[i64],
    suppkey_values: &[i64],
    quantity_values: &[i128],
    extendedprice_values: &[i128],
    discount_values: &[i128],
    discount_scale: f64,
    revenue_scale: f64,
    quantity_scale: f64,
    dense_part_keys: &[bool],
    dense_part_words: Option<&[u64]>,
    dense_order_years: (&[i32], i64, i32),
    supply_costs: &FastHashMap<u64, Q09SupplyInfo>,
    collect_profile: bool,
) -> Q09ProfitPartial {
    let mut groups = Q09ProfitGroups::default();
    let mut profile = Q09ProfitProfile::default();
    if collect_profile {
        profile.rows = partkey_values.len();
    }
    let (order_year_values, order_year_base, order_year_missing) = dense_order_years;
    for row in 0..partkey_values.len() {
        let partkey = partkey_values[row];
        if let Some(words) = dense_part_words {
            if !crate::dense::adaptive_i64_words_contains(words, partkey) {
                continue;
            }
        } else {
            let Some(part_index) = usize::try_from(partkey).ok() else {
                continue;
            };
            if !dense_part_keys.get(part_index).copied().unwrap_or(false) {
                continue;
            }
        }
        if collect_profile {
            profile.part_hits += 1;
        }

        let orderkey = orderkey_values[row];
        let Some(order_index) = orderkey
            .checked_sub(order_year_base)
            .and_then(|index| usize::try_from(index).ok())
        else {
            continue;
        };
        let Some(&o_year) = order_year_values.get(order_index) else {
            continue;
        };
        if o_year == order_year_missing {
            continue;
        }
        if collect_profile {
            profile.order_hits += 1;
        }

        let Some(supply_key) = q09_pack_part_supp_key(partkey, suppkey_values[row]) else {
            continue;
        };
        let Some(supplycost) = supply_costs.get(&supply_key).copied() else {
            continue;
        };
        if collect_profile {
            profile.supplier_hits += 1;
            profile.supply_hits += 1;
        }

        let amount = (extendedprice_values[row] as f64)
            * (discount_scale - discount_values[row] as f64)
            * revenue_scale
            - supplycost.supplycost * (quantity_values[row] as f64) * quantity_scale;
        groups.add(supplycost.nationkey, o_year, amount);
        if collect_profile {
            profile.amount_rows += 1;
        }
    }
    Q09ProfitPartial { groups, profile }
}

#[allow(clippy::too_many_arguments)]
fn q09_profit_decimal_i64_packed_dense_loop_with_words(
    orderkey_values: &[i64],
    partkey_values: &[i64],
    suppkey_values: &[i64],
    quantity_values: &[i64],
    extendedprice_values: &[i64],
    discount_values: &[i64],
    discount_scale: f64,
    revenue_scale: f64,
    quantity_scale: f64,
    dense_part_keys: &[bool],
    dense_part_words: Option<&[u64]>,
    dense_order_years: (&[i32], i64, i32),
    supply_costs: &FastHashMap<u64, Q09SupplyInfo>,
    collect_profile: bool,
) -> Q09ProfitPartial {
    let mut groups = Q09ProfitGroups::default();
    let mut profile = Q09ProfitProfile::default();
    if collect_profile {
        profile.rows = partkey_values.len();
    }
    let (order_year_values, order_year_base, order_year_missing) = dense_order_years;
    for row in 0..partkey_values.len() {
        let partkey = partkey_values[row];
        if let Some(words) = dense_part_words {
            if !crate::dense::adaptive_i64_words_contains(words, partkey) {
                continue;
            }
        } else {
            let Some(part_index) = usize::try_from(partkey).ok() else {
                continue;
            };
            if !dense_part_keys.get(part_index).copied().unwrap_or(false) {
                continue;
            }
        }
        if collect_profile {
            profile.part_hits += 1;
        }

        let orderkey = orderkey_values[row];
        let Some(order_index) = orderkey
            .checked_sub(order_year_base)
            .and_then(|index| usize::try_from(index).ok())
        else {
            continue;
        };
        let Some(&o_year) = order_year_values.get(order_index) else {
            continue;
        };
        if o_year == order_year_missing {
            continue;
        }
        if collect_profile {
            profile.order_hits += 1;
        }

        let Some(supply_key) = q09_pack_part_supp_key(partkey, suppkey_values[row]) else {
            continue;
        };
        let Some(supplycost) = supply_costs.get(&supply_key).copied() else {
            continue;
        };
        if collect_profile {
            profile.supplier_hits += 1;
            profile.supply_hits += 1;
        }

        let amount = (extendedprice_values[row] as f64)
            * (discount_scale - discount_values[row] as f64)
            * revenue_scale
            - supplycost.supplycost * (quantity_values[row] as f64) * quantity_scale;
        groups.add(supplycost.nationkey, o_year, amount);
        if collect_profile {
            profile.amount_rows += 1;
        }
    }
    Q09ProfitPartial { groups, profile }
}

#[allow(clippy::too_many_arguments)]
fn q09_profit_decimal_i64_view_loop(
    orderkey_values: &[i64],
    partkey_values: &[i64],
    suppkey_values: &[i64],
    quantity_values: &[i64],
    extendedprice_values: &[i64],
    discount_values: &[i64],
    discount_scale: f64,
    revenue_scale: f64,
    quantity_scale: f64,
    part_keys: &AdaptiveI64Set,
    order_years: &Q09OrderYears,
    supply_costs: &Q09SupplyCosts,
    collect_profile: bool,
) -> Q09ProfitPartial {
    let mut groups = Q09ProfitGroups::default();
    let mut profile = Q09ProfitProfile::default();
    let dense_part_keys = part_keys.dense_contains_slice();
    let dense_order_years = order_years.dense_slice();
    if collect_profile {
        profile.rows = partkey_values.len();
    }
    for row in 0..partkey_values.len() {
        let partkey = partkey_values[row];
        if !part_keys.contains_cached(dense_part_keys, partkey) {
            continue;
        }
        if collect_profile {
            profile.part_hits += 1;
        }
        let Some(o_year) = q09_order_year_get(order_years, dense_order_years, orderkey_values[row])
        else {
            continue;
        };
        if collect_profile {
            profile.order_hits += 1;
        }
        let Some(supplycost) = supply_costs.get(partkey, suppkey_values[row]) else {
            continue;
        };
        if collect_profile {
            profile.supplier_hits += 1;
            profile.supply_hits += 1;
        }
        let amount = (extendedprice_values[row] as f64)
            * (discount_scale - discount_values[row] as f64)
            * revenue_scale
            - supplycost.supplycost * (quantity_values[row] as f64) * quantity_scale;
        groups.add(supplycost.nationkey, o_year, amount);
        if collect_profile {
            profile.amount_rows += 1;
        }
    }
    Q09ProfitPartial { groups, profile }
}

fn q09_output(rows: Vec<Q09Row>) -> Result<QueryOutput> {
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("nation", DataType::Utf8, false),
            Field::new("o_year", DataType::Int64, false),
            Field::new("sum_profit", DataType::Float64, false),
        ])),
        vec![
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|row| row.nation.as_str()),
            )),
            Arc::new(Int64Array::from_iter_values(
                rows.iter().map(|row| i64::from(row.o_year)),
            )),
            Arc::new(Float64Array::from_iter_values(
                rows.iter().map(|row| row.sum_profit),
            )),
        ],
    )?;
    Ok(QueryOutput::Aggregate {
        metrics: AggregateMetrics::default(),
        batches: vec![batch],
    })
}
