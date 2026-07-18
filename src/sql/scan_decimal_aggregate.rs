use super::*;

#[derive(Clone)]
struct DecimalProductSumSpec {
    sum_expr: AggregateExpr,
    left_column: String,
    right_column: String,
    third_column: Option<String>,
    left_index: usize,
    right_index: usize,
    third_index: Option<usize>,
    left_kind: ProductColumnKind,
    right_kind: ProductColumnKind,
    third_kind: Option<ProductColumnKind>,
    left_transform: ProductTermTransform,
    right_transform: ProductTermTransform,
    third_transform: Option<ProductTermTransform>,
    projection: Vec<String>,
    predicates: Vec<PrimitivePredicate>,
}

#[derive(Clone, Copy)]
enum ProductColumnKind {
    Int32,
    Int64,
    DecimalI64 { scale: i64 },
}

#[derive(Clone)]
enum PrimitivePredicate {
    Date32 {
        index: usize,
        min: Option<i32>,
        max: Option<i32>,
    },
    Decimal128 {
        index: usize,
        scale: i64,
        min: Option<i64>,
        max: Option<i64>,
    },
}

struct PrimitivePredicateVectors<'a> {
    predicates: Vec<PrimitivePredicateVector<'a>>,
}

enum PrimitivePredicateVector<'a> {
    Date32 {
        values: Date32VectorView<'a>,
        min: Option<i32>,
        max: Option<i32>,
    },
    Decimal128 {
        values: Decimal128VectorView<'a>,
        min: Option<i64>,
        max: Option<i64>,
    },
}

impl PrimitivePredicateVectors<'_> {
    #[inline]
    fn matches(&self, row: usize) -> bool {
        match self.predicates.as_slice() {
            [] => true,
            [first] => first.matches(row),
            [first, second] => first.matches(row) && second.matches(row),
            [first, second, third] => {
                first.matches(row) && second.matches(row) && third.matches(row)
            }
            predicates => predicates.iter().all(|predicate| predicate.matches(row)),
        }
    }

    fn sparse_selection(&self, rows: usize) -> PredicateSelection {
        if rows == 0 {
            return PredicateSelection::Empty;
        }
        let mut selected = Vec::new();
        for row in 0..rows {
            if self.matches(row) {
                selected.push(row);
            }
        }
        if selected.is_empty() {
            PredicateSelection::Empty
        } else if selected.len().saturating_mul(4) <= rows.saturating_mul(3) {
            PredicateSelection::Sparse(selected)
        } else {
            PredicateSelection::Dense
        }
    }
}

enum PredicateSelection {
    Empty,
    Sparse(Vec<usize>),
    Dense,
}

impl PrimitivePredicateVector<'_> {
    #[inline]
    fn matches(&self, row: usize) -> bool {
        match self {
            Self::Date32 { values, min, max } => {
                !values.is_null(row) && i32_in_bounds(values.value(row), *min, *max)
            }
            Self::Decimal128 { values, min, max } => {
                !values.is_null(row)
                    && values
                        .raw_i64_value(row)
                        .is_some_and(|value| i64_in_bounds(value, *min, *max))
            }
        }
    }
}

#[derive(Default)]
struct DecimalProductSumState {
    sum: f64,
    int_sum: i128,
    count: u64,
    rows: usize,
    batches: usize,
}

#[derive(Clone, Copy)]
struct ProductPayloadPlan {
    left_index: usize,
    right_index: usize,
    third_index: Option<usize>,
    left_kind: ProductColumnKind,
    right_kind: ProductColumnKind,
    third_kind: Option<ProductColumnKind>,
    left_transform: ProductTermTransform,
    right_transform: ProductTermTransform,
    third_transform: Option<ProductTermTransform>,
}

pub(super) async fn try_collect_filtered_decimal_product_sum_scan_fold(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    filter: Option<FilterExpr>,
    aggregates: &[AggregateExpr],
    expressions: &[ProjectionExpression],
) -> Result<Option<AggregateMetrics>> {
    let Some(filter) = filter.as_ref() else {
        log_product_sum_rule_miss("missing-filter");
        return Ok(None);
    };
    if !path.exists() {
        log_product_sum_rule_miss("path-missing");
        return Ok(None);
    }
    let Some(spec) =
        DecimalProductSumSpec::try_new(engine, &path, filter, aggregates, expressions)?
    else {
        log_product_sum_rule_miss("spec-mismatch");
        return Ok(None);
    };
    if let Some(metrics) = try_collect_filtered_product_sum_late_materialized(
        engine,
        path.clone(),
        batch_size,
        filter,
        &spec,
    )
    .await?
    {
        return Ok(Some(metrics));
    }
    let started = Instant::now();
    let state = engine
        .parquet_scan_accumulate_chunks_view(
            path,
            batch_size,
            Projection::Columns(spec.projection.clone()),
            scan_aggregate_row_group_chunk(),
            8,
            scan_aggregate_fusion_enabled(),
            DecimalProductSumState::default,
            DecimalProductSumState::default,
            {
                let spec = spec.clone();
                move |view, state| {
                    consume_decimal_product_sum_view(view, &spec, state)?;
                    Ok(Some(()))
                }
            },
            |state, partial| {
                state.sum += partial.sum;
                state.int_sum += partial.int_sum;
                state.count += partial.count;
                state.rows += partial.rows;
                state.batches += partial.batches;
            },
            "filtered decimal product sum",
        )
        .await?;
    Ok(Some(AggregateMetrics {
        fragments: 1,
        batches: state.batches,
        rows: state.rows,
        values: vec![AggregateResult {
            expr: spec.sum_expr.clone(),
            value: spec.aggregate_value(&state)?,
        }],
        aggregate_nanos: started.elapsed().as_nanos().min(u64::MAX as u128) as u64,
        ..AggregateMetrics::default()
    }))
}

async fn try_collect_filtered_product_sum_late_materialized(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    filter: &FilterExpr,
    spec: &DecimalProductSumSpec,
) -> Result<Option<AggregateMetrics>> {
    if !matches!(
        (spec.left_kind, spec.right_kind, spec.third_kind),
        (
            ProductColumnKind::DecimalI64 { .. },
            ProductColumnKind::DecimalI64 { .. },
            None | Some(ProductColumnKind::DecimalI64 { .. })
        )
    ) {
        return Ok(None);
    }
    let predicate_columns = filter.referenced_columns();
    if predicate_columns.is_empty() {
        return Ok(None);
    }
    let mut payload_columns = Vec::new();
    add_column_once(&mut payload_columns, spec.left_column.clone());
    add_column_once(&mut payload_columns, spec.right_column.clone());
    if let Some(column) = &spec.third_column {
        add_column_once(&mut payload_columns, column.clone());
    }
    let payload_index = |column: &str| {
        payload_columns
            .iter()
            .position(|candidate| candidate == column)
            .expect("product payload column is projected")
    };
    let sum_expr = spec.sum_expr.clone();
    let product_plan = ProductPayloadPlan {
        left_index: payload_index(&spec.left_column),
        right_index: payload_index(&spec.right_column),
        third_index: spec.third_column.as_deref().map(payload_index),
        left_kind: spec.left_kind,
        right_kind: spec.right_kind,
        third_kind: spec.third_kind,
        left_transform: spec.left_transform,
        right_transform: spec.right_transform,
        third_transform: spec.third_transform,
    };
    let started = Instant::now();
    let Some(partials) = engine
        .late_materialized_parquet_map_pruned_with_policy_view(
            path,
            batch_size,
            Projection::Columns(predicate_columns.clone()),
            Projection::Columns(payload_columns),
            vec![filter.expr().clone()],
            scan_aggregate_row_group_chunk(),
            LateMaterializationPolicy::selective_with_selector_run_ratio(
                filtered_product_sum_late_max_selected_ratio(spec.product_factor_count()),
                filtered_product_sum_late_max_selector_run_ratio(spec.product_factor_count()),
            )
            .with_io_cost_gate(true),
            move || DecimalProductSumState::default(),
            {
                let filter = filter.clone();
                let predicate_columns = predicate_columns.clone();
                move |view, selection, _state: &mut DecimalProductSumState| {
                    if push_projected_view_filter_selection(
                        view,
                        &predicate_columns,
                        &filter,
                        selection,
                    )? {
                        return Ok(Some(()));
                    }
                    Ok(None)
                }
            },
            move |view, state: &mut DecimalProductSumState| {
                consume_product_payload_view(view, product_plan, state)?;
                Ok(Some(()))
            },
            |state, _metrics| Ok(Some(state)),
        )
        .await?
    else {
        return Ok(None);
    };
    let mut state = DecimalProductSumState::default();
    for partial in partials {
        state.sum += partial.output.sum;
        state.int_sum += partial.output.int_sum;
        state.count += partial.output.count;
        state.rows += partial.metrics.total_rows;
        state.batches += partial.output.batches;
    }
    Ok(Some(AggregateMetrics {
        fragments: 1,
        batches: state.batches,
        rows: state.rows,
        values: vec![AggregateResult {
            expr: sum_expr,
            value: spec.aggregate_value(&state)?,
        }],
        aggregate_nanos: started.elapsed().as_nanos().min(u64::MAX as u128) as u64,
        ..AggregateMetrics::default()
    }))
}

impl DecimalProductSumSpec {
    fn try_new(
        engine: &DodamEngine,
        path: &Path,
        filter: &FilterExpr,
        aggregates: &[AggregateExpr],
        expressions: &[ProjectionExpression],
    ) -> Result<Option<Self>> {
        let [AggregateExpr::Sum(sum_column)] = aggregates else {
            log_product_sum_rule_miss("aggregate-not-single-sum");
            return Ok(None);
        };
        let [expression] = expressions else {
            log_product_sum_rule_miss("expression-not-single");
            return Ok(None);
        };
        if expression.output_name != *sum_column {
            log_product_sum_rule_miss("expression-output-mismatch");
            return Ok(None);
        }
        let Some(product) = product_expression_shape(&expression.expr) else {
            log_product_sum_rule_miss("expression-not-column-product");
            return Ok(None);
        };
        let [left, right, rest @ ..] = product.terms.as_slice() else {
            log_product_sum_rule_miss("expression-not-column-product");
            return Ok(None);
        };
        let left_column = left.column.as_str();
        let right_column = right.column.as_str();
        let third = rest.first();
        let third_column = third.map(|term| term.column.as_str());
        let Some(left_kind) = product_column_kind(engine, path, left_column)? else {
            log_product_sum_rule_miss("left-product-type-unsupported");
            return Ok(None);
        };
        let Some(right_kind) = product_column_kind(engine, path, right_column)? else {
            log_product_sum_rule_miss("right-product-type-unsupported");
            return Ok(None);
        };
        let third_kind = if let Some(column) = third_column {
            let Some(kind) = product_column_kind(engine, path, column)? else {
                log_product_sum_rule_miss("third-product-type-unsupported");
                return Ok(None);
            };
            Some(kind)
        } else {
            None
        };
        let decimal_product = matches!(
            (left_kind, right_kind, third_kind),
            (
                ProductColumnKind::DecimalI64 { .. },
                ProductColumnKind::DecimalI64 { .. },
                None | Some(ProductColumnKind::DecimalI64 { .. })
            )
        );
        let integer_product = third_kind.is_none()
            && matches!(
                (left_kind, right_kind),
                (
                    ProductColumnKind::Int32 | ProductColumnKind::Int64,
                    ProductColumnKind::Int32 | ProductColumnKind::Int64
                )
            );
        if !decimal_product && !integer_product {
            log_product_sum_rule_miss("mixed-product-types");
            return Ok(None);
        }
        if matches!(
            (left_kind, right_kind),
            (
                ProductColumnKind::Int32 | ProductColumnKind::Int64,
                ProductColumnKind::Int32 | ProductColumnKind::Int64
            )
        ) && std::env::var_os("DODAM_ENABLE_INTEGER_PRODUCT_SUM_SCAN_FOLD").is_none()
        {
            log_product_sum_rule_miss("integer-product-default-disabled");
            return Ok(None);
        }
        let mut projection = Vec::new();
        add_column_once(&mut projection, left_column.to_string());
        add_column_once(&mut projection, right_column.to_string());
        if let Some(column) = third_column {
            add_column_once(&mut projection, column.to_string());
        }
        let left_index = product_projection_index(&projection, left_column);
        let right_index = product_projection_index(&projection, right_column);
        let third_index = third_column.map(|column| product_projection_index(&projection, column));
        for column in filter.referenced_columns() {
            add_column_once(&mut projection, column);
        }
        let mut predicates = Vec::new();
        for column in filter.referenced_columns() {
            if let Some(predicate) =
                primitive_predicate_for_column(engine, path, filter.expr(), &projection, &column)?
            {
                predicates.push(predicate);
            } else {
                log_product_sum_rule_miss("predicate-unsupported");
                return Ok(None);
            }
        }
        if predicates.is_empty() {
            log_product_sum_rule_miss("no-primitive-predicates");
            return Ok(None);
        }
        Ok(Some(Self {
            sum_expr: aggregates[0].clone(),
            left_column: left_column.to_string(),
            right_column: right_column.to_string(),
            third_column: third_column.map(str::to_string),
            left_index,
            right_index,
            third_index,
            left_kind,
            right_kind,
            third_kind,
            left_transform: left.transform,
            right_transform: right.transform,
            third_transform: third.map(|term| term.transform),
            projection,
            predicates,
        }))
    }

    fn aggregate_value(&self, state: &DecimalProductSumState) -> Result<AggregateValue> {
        if state.count == 0 {
            return Ok(match (self.left_kind, self.right_kind) {
                (
                    ProductColumnKind::Int32 | ProductColumnKind::Int64,
                    ProductColumnKind::Int32 | ProductColumnKind::Int64,
                ) if self.third_kind.is_none() => AggregateValue::Int64(None),
                _ => AggregateValue::Float64(None),
            });
        }
        match (self.left_kind, self.right_kind) {
            (
                ProductColumnKind::Int32 | ProductColumnKind::Int64,
                ProductColumnKind::Int32 | ProductColumnKind::Int64,
            ) if self.third_kind.is_none() => {
                let value = i64::try_from(state.int_sum).map_err(|_| {
                    DodamError::UnsupportedSql("SUM integer expression overflow".to_string())
                })?;
                Ok(AggregateValue::Int64(Some(value)))
            }
            _ => Ok(AggregateValue::Float64(Some(state.sum))),
        }
    }

    fn product_factor_count(&self) -> usize {
        2 + usize::from(self.third_kind.is_some())
    }
}

fn product_projection_index(projection: &[String], column: &str) -> usize {
    projection
        .iter()
        .position(|candidate| candidate == column)
        .expect("product column is projected")
}

fn product_column_kind(
    engine: &DodamEngine,
    path: &Path,
    column: &str,
) -> Result<Option<ProductColumnKind>> {
    let column = column.to_string();
    if let Some((precision, scale)) = engine.parquet_decimal128_type(path, &column)? {
        if precision <= 18 {
            return Ok(Some(ProductColumnKind::DecimalI64 {
                scale: decimal_scale_i64_local(scale)?,
            }));
        }
    }
    if let Some(types) =
        engine.parquet_direct_primitive_column_types(path, std::slice::from_ref(&column))?
    {
        if let [column_type] = types.as_slice() {
            return Ok(match column_type {
                DirectPrimitiveColumnType::I32 => Some(ProductColumnKind::Int32),
                DirectPrimitiveColumnType::I64 => Some(ProductColumnKind::Int64),
                DirectPrimitiveColumnType::Decimal128Int64 { precision, scale }
                | DirectPrimitiveColumnType::Decimal128Int64Raw { precision, scale } => {
                    if *precision <= 18 {
                        Some(ProductColumnKind::DecimalI64 {
                            scale: decimal_scale_i64_local(*scale)?,
                        })
                    } else {
                        None
                    }
                }
                DirectPrimitiveColumnType::Date32 => None,
            });
        }
    }
    Ok(None)
}

fn primitive_predicate_for_column(
    engine: &DodamEngine,
    path: &Path,
    expr: &Expr,
    projection: &[String],
    column: &str,
) -> Result<Option<PrimitivePredicate>> {
    let index = projection
        .iter()
        .position(|candidate| candidate == column)
        .expect("predicate column is projected");
    if engine.parquet_is_date32_column(path, column)? {
        let Some((min, max)) = primitive_i32_bounds(expr, column, literal_to_date32_local)? else {
            return Ok(None);
        };
        return Ok(Some(PrimitivePredicate::Date32 { index, min, max }));
    }
    if let Some((precision, scale)) = engine.parquet_decimal128_type(path, column)? {
        if precision > 18 {
            return Ok(None);
        }
        let scale_value = decimal_scale_i64_local(scale)?;
        let Some((min, max)) = primitive_i64_bounds(expr, column, |literal| {
            literal_to_decimal_i64_local(literal, scale_value)
        })?
        else {
            return Ok(None);
        };
        return Ok(Some(PrimitivePredicate::Decimal128 {
            index,
            scale: scale_value,
            min,
            max,
        }));
    }
    Ok(None)
}

fn primitive_i32_bounds<F>(
    expr: &Expr,
    column: &str,
    convert: F,
) -> Result<Option<(Option<i32>, Option<i32>)>>
where
    F: Fn(&LiteralValue) -> Option<i32> + Copy,
{
    let Some((min, max)) = primitive_bounds(expr, column, convert)? else {
        return Ok(None);
    };
    Ok(Some((min, max)))
}

fn primitive_i64_bounds<F>(
    expr: &Expr,
    column: &str,
    convert: F,
) -> Result<Option<(Option<i64>, Option<i64>)>>
where
    F: Fn(&LiteralValue) -> Option<i64> + Copy,
{
    primitive_bounds(expr, column, convert)
}

fn primitive_bounds<T, F>(
    expr: &Expr,
    column: &str,
    convert: F,
) -> Result<Option<(Option<T>, Option<T>)>>
where
    T: BoundValue,
    F: Fn(&LiteralValue) -> Option<T> + Copy,
{
    let mut min = None;
    let mut max = None;
    if !collect_primitive_bounds(expr, column, convert, &mut min, &mut max)? {
        return Ok(None);
    }
    Ok(Some((min, max)))
}

fn collect_primitive_bounds<T, F>(
    expr: &Expr,
    column: &str,
    convert: F,
    min: &mut Option<T>,
    max: &mut Option<T>,
) -> Result<bool>
where
    T: BoundValue,
    F: Fn(&LiteralValue) -> Option<T> + Copy,
{
    match expr {
        Expr::Boolean(Some(true)) => Ok(true),
        Expr::And(left, right) => Ok(collect_primitive_bounds(left, column, convert, min, max)?
            && collect_primitive_bounds(right, column, convert, min, max)?),
        Expr::Comparison(comparison) if comparison.column == column => {
            let Some(value) = convert(&comparison.value) else {
                return Ok(false);
            };
            apply_bound(comparison.op, value, min, max)
        }
        Expr::Comparison(_) => Ok(true),
        _ => Ok(false),
    }
}

trait BoundValue: Copy + Ord {
    fn checked_next(self) -> Option<Self>;
    fn checked_prev(self) -> Option<Self>;
}

impl BoundValue for i32 {
    fn checked_next(self) -> Option<Self> {
        self.checked_add(1)
    }

    fn checked_prev(self) -> Option<Self> {
        self.checked_sub(1)
    }
}

impl BoundValue for i64 {
    fn checked_next(self) -> Option<Self> {
        self.checked_add(1)
    }

    fn checked_prev(self) -> Option<Self> {
        self.checked_sub(1)
    }
}

fn apply_bound<T: BoundValue>(
    op: ComparisonOp,
    value: T,
    min: &mut Option<T>,
    max: &mut Option<T>,
) -> Result<bool> {
    match op {
        ComparisonOp::Eq => {
            *min = Some(min.map_or(value, |current| current.max(value)));
            *max = Some(max.map_or(value, |current| current.min(value)));
            Ok(true)
        }
        ComparisonOp::Gt => {
            let Some(value) = value.checked_next() else {
                return Ok(false);
            };
            *min = Some(min.map_or(value, |current| current.max(value)));
            Ok(true)
        }
        ComparisonOp::GtEq => {
            *min = Some(min.map_or(value, |current| current.max(value)));
            Ok(true)
        }
        ComparisonOp::Lt => {
            let Some(value) = value.checked_prev() else {
                return Ok(false);
            };
            *max = Some(max.map_or(value, |current| current.min(value)));
            Ok(true)
        }
        ComparisonOp::LtEq => {
            *max = Some(max.map_or(value, |current| current.min(value)));
            Ok(true)
        }
        ComparisonOp::NotEq => Ok(false),
    }
}

fn consume_decimal_product_sum_view(
    view: BatchView<'_>,
    spec: &DecimalProductSumSpec,
    state: &mut DecimalProductSumState,
) -> Result<()> {
    if view.num_columns() != spec.projection.len() {
        let Some(batch) = view.try_record_batch() else {
            return Err(DodamError::UnsupportedSql(
                "filtered decimal product sum raw vector layout mismatch".to_string(),
            ));
        };
        return consume_decimal_product_sum_batch(batch, spec, state);
    }
    if consume_product_sum_vectors(view, spec, state)? {
        return Ok(());
    }
    state.rows += view.num_rows();
    state.batches += 1;
    for row in 0..view.num_rows() {
        if !predicates_match_row(view, spec, row)? {
            continue;
        }
        consume_product_row(view, spec, row, state)?;
    }
    Ok(())
}

fn consume_decimal_product_sum_batch(
    batch: &RecordBatch,
    spec: &DecimalProductSumSpec,
    state: &mut DecimalProductSumState,
) -> Result<()> {
    let view = BatchView::new(batch);
    if consume_product_sum_vectors(view, spec, state)? {
        return Ok(());
    }
    state.rows += batch.num_rows();
    state.batches += 1;
    for row in 0..batch.num_rows() {
        if !batch_predicates_match_row(batch, spec, row)? {
            continue;
        }
        consume_product_row(view, spec, row, state)?;
    }
    Ok(())
}

fn consume_product_sum_vectors(
    view: BatchView<'_>,
    spec: &DecimalProductSumSpec,
    state: &mut DecimalProductSumState,
) -> Result<bool> {
    let Some(predicates) = bind_predicate_vectors(view, spec) else {
        return Ok(false);
    };
    match (spec.left_kind, spec.right_kind) {
        (ProductColumnKind::Int64, ProductColumnKind::Int64) => {
            let (Some(left), Some(right)) = (
                view.i64_vector(spec.left_index),
                view.i64_vector(spec.right_index),
            ) else {
                return Ok(false);
            };
            consume_i64_i64_product_sum_vectors(view, &predicates, left, right, state);
            Ok(true)
        }
        (ProductColumnKind::Int32, ProductColumnKind::Int32) => {
            let (Some(left), Some(right)) = (
                view.i32_vector(spec.left_index),
                view.i32_vector(spec.right_index),
            ) else {
                return Ok(false);
            };
            consume_i32_i32_product_sum_vectors(view, &predicates, left, right, state);
            Ok(true)
        }
        (
            ProductColumnKind::DecimalI64 { scale: left_scale },
            ProductColumnKind::DecimalI64 { scale: right_scale },
        ) => {
            let (Some(left), Some(right)) = (
                view.decimal128_vector(spec.left_index),
                view.decimal128_vector(spec.right_index),
            ) else {
                return Ok(false);
            };
            if left.scale_i64() != Some(left_scale) || right.scale_i64() != Some(right_scale) {
                return Ok(false);
            }
            let third = match spec.third_kind {
                None => None,
                Some(kind @ ProductColumnKind::DecimalI64 { scale }) => {
                    let Some(values) = spec
                        .third_index
                        .and_then(|index| view.decimal128_vector(index))
                    else {
                        return Ok(false);
                    };
                    if values.scale_i64() != Some(scale) {
                        return Ok(false);
                    }
                    Some((kind, values))
                }
                Some(_) => return Ok(false),
            };
            consume_decimal_i64_product_sum_vectors(
                view,
                &predicates,
                left,
                right,
                third,
                left_scale,
                right_scale,
                spec.left_transform,
                spec.right_transform,
                spec.third_transform,
                state,
            );
            Ok(true)
        }
        _ => Ok(false),
    }
}

fn consume_product_payload_view(
    view: BatchView<'_>,
    plan: ProductPayloadPlan,
    state: &mut DecimalProductSumState,
) -> Result<()> {
    state.batches += 1;
    match (plan.left_kind, plan.right_kind) {
        (
            ProductColumnKind::DecimalI64 { scale: left_scale },
            ProductColumnKind::DecimalI64 { scale: right_scale },
        ) => {
            let (Some(left), Some(right)) = (
                view.decimal128_vector(plan.left_index),
                view.decimal128_vector(plan.right_index),
            ) else {
                return Ok(());
            };
            if left.scale_i64() != Some(left_scale) || right.scale_i64() != Some(right_scale) {
                return Ok(());
            }
            let third = match plan.third_kind {
                None => None,
                Some(ProductColumnKind::DecimalI64 { scale }) => {
                    let Some(values) = plan
                        .third_index
                        .and_then(|index| view.decimal128_vector(index))
                    else {
                        return Ok(());
                    };
                    if values.scale_i64() != Some(scale) {
                        return Ok(());
                    }
                    Some((
                        values,
                        scale,
                        plan.third_transform
                            .unwrap_or(ProductTermTransform::Identity),
                    ))
                }
                _ => return Ok(()),
            };
            let mut scale = (left_scale as f64) * (right_scale as f64);
            if let Some((_, third_scale, _)) = third {
                scale *= third_scale as f64;
            }
            let inv_scale = scale.recip();
            for row in 0..view.num_rows() {
                if left.is_null(row) || right.is_null(row) {
                    continue;
                }
                if let Some((values, _, _)) = third
                    && values.is_null(row)
                {
                    continue;
                }
                let Some(left) = left.raw_i64_value(row) else {
                    continue;
                };
                let Some(right) = right.raw_i64_value(row) else {
                    continue;
                };
                let left = plan.left_transform.apply_raw_i64(left, left_scale);
                let right = plan.right_transform.apply_raw_i64(right, right_scale);
                let mut product = (left as f64) * (right as f64);
                if let Some((values, third_scale, transform)) = third {
                    let Some(third) = values.raw_i64_value(row) else {
                        continue;
                    };
                    let third = transform.apply_raw_i64(third, third_scale);
                    product *= third as f64;
                }
                state.sum += product * inv_scale;
                state.count += 1;
            }
        }
        (ProductColumnKind::Int64, ProductColumnKind::Int64) => {
            let (Some(left), Some(right)) = (
                view.i64_vector(plan.left_index),
                view.i64_vector(plan.right_index),
            ) else {
                return Ok(());
            };
            for row in 0..view.num_rows() {
                if left.is_null(row) || right.is_null(row) {
                    continue;
                }
                state.int_sum += i128::from(left.value(row)) * i128::from(right.value(row));
                state.count += 1;
            }
        }
        (ProductColumnKind::Int32, ProductColumnKind::Int32) => {
            let (Some(left), Some(right)) = (
                view.i32_vector(plan.left_index),
                view.i32_vector(plan.right_index),
            ) else {
                return Ok(());
            };
            for row in 0..view.num_rows() {
                if left.is_null(row) || right.is_null(row) {
                    continue;
                }
                state.int_sum += i128::from(left.value(row)) * i128::from(right.value(row));
                state.count += 1;
            }
        }
        _ => {}
    }
    Ok(())
}

fn bind_predicate_vectors<'a>(
    view: BatchView<'a>,
    spec: &DecimalProductSumSpec,
) -> Option<PrimitivePredicateVectors<'a>> {
    let mut predicates = Vec::with_capacity(spec.predicates.len());
    for predicate in &spec.predicates {
        match predicate {
            PrimitivePredicate::Date32 { index, min, max } => {
                predicates.push(PrimitivePredicateVector::Date32 {
                    values: view.date32_vector(*index)?,
                    min: *min,
                    max: *max,
                });
            }
            PrimitivePredicate::Decimal128 {
                index,
                scale,
                min,
                max,
            } => {
                let values = view.decimal128_vector(*index)?;
                if values.scale_i64() != Some(*scale) {
                    return None;
                }
                predicates.push(PrimitivePredicateVector::Decimal128 {
                    values,
                    min: *min,
                    max: *max,
                });
            }
        }
    }
    Some(PrimitivePredicateVectors { predicates })
}

fn consume_i64_i64_product_sum_vectors(
    view: BatchView<'_>,
    predicates: &PrimitivePredicateVectors<'_>,
    left: I64VectorView<'_>,
    right: I64VectorView<'_>,
    state: &mut DecimalProductSumState,
) {
    state.rows += view.num_rows();
    state.batches += 1;
    if let (Some(left), Some(right)) = (left.values_if_null_free(), right.values_if_null_free()) {
        for row in 0..view.num_rows() {
            if predicates.matches(row) {
                state.int_sum += i128::from(left[row]) * i128::from(right[row]);
                state.count += 1;
            }
        }
        return;
    }
    for row in 0..view.num_rows() {
        if left.is_null(row) || right.is_null(row) || !predicates.matches(row) {
            continue;
        }
        state.int_sum += i128::from(left.value(row)) * i128::from(right.value(row));
        state.count += 1;
    }
}

fn consume_i32_i32_product_sum_vectors(
    view: BatchView<'_>,
    predicates: &PrimitivePredicateVectors<'_>,
    left: I32VectorView<'_>,
    right: I32VectorView<'_>,
    state: &mut DecimalProductSumState,
) {
    state.rows += view.num_rows();
    state.batches += 1;
    if let (Some(left), Some(right)) = (left.values_if_null_free(), right.values_if_null_free()) {
        for row in 0..view.num_rows() {
            if predicates.matches(row) {
                state.int_sum += i128::from(left[row]) * i128::from(right[row]);
                state.count += 1;
            }
        }
        return;
    }
    for row in 0..view.num_rows() {
        if left.is_null(row) || right.is_null(row) || !predicates.matches(row) {
            continue;
        }
        state.int_sum += i128::from(left.value(row)) * i128::from(right.value(row));
        state.count += 1;
    }
}

fn consume_decimal_i64_product_sum_vectors(
    view: BatchView<'_>,
    predicates: &PrimitivePredicateVectors<'_>,
    left: Decimal128VectorView<'_>,
    right: Decimal128VectorView<'_>,
    third: Option<(ProductColumnKind, Decimal128VectorView<'_>)>,
    left_scale: i64,
    right_scale: i64,
    left_transform: ProductTermTransform,
    right_transform: ProductTermTransform,
    third_transform: Option<ProductTermTransform>,
    state: &mut DecimalProductSumState,
) {
    state.rows += view.num_rows();
    state.batches += 1;
    let third = match third {
        None => None,
        Some((ProductColumnKind::DecimalI64 { scale }, values)) => {
            if values.scale_i64() != Some(scale) {
                return;
            }
            Some((
                values,
                scale,
                third_transform.unwrap_or(ProductTermTransform::Identity),
            ))
        }
        _ => return,
    };
    let selection = predicates.sparse_selection(view.num_rows());
    if matches!(selection, PredicateSelection::Empty) {
        return;
    }
    if left.null_count() == 0 && right.null_count() == 0 {
        if let (Some(left), Some(right)) = (left.raw_i64_values(), right.raw_i64_values()) {
            let mut scale = (left_scale as f64) * (right_scale as f64);
            if let Some((_, third_scale, _)) = third {
                scale *= third_scale as f64;
            }
            let inv_scale = scale.recip();
            if let Some((values, third_scale, transform)) = third
                && values.null_count() == 0
                && let Some(third_values) = values.raw_i64_values()
            {
                match selection {
                    PredicateSelection::Empty => unreachable!("empty selection returned above"),
                    PredicateSelection::Sparse(rows) => {
                        for row in rows {
                            let left = left_transform.apply_raw_i64(left[row], left_scale);
                            let right = right_transform.apply_raw_i64(right[row], right_scale);
                            let third = transform.apply_raw_i64(third_values[row], third_scale);
                            state.sum +=
                                (left as f64) * (right as f64) * (third as f64) * inv_scale;
                            state.count += 1;
                        }
                    }
                    PredicateSelection::Dense => {
                        for row in 0..view.num_rows() {
                            if predicates.matches(row) {
                                let left = left_transform.apply_raw_i64(left[row], left_scale);
                                let right = right_transform.apply_raw_i64(right[row], right_scale);
                                let third = transform.apply_raw_i64(third_values[row], third_scale);
                                state.sum +=
                                    (left as f64) * (right as f64) * (third as f64) * inv_scale;
                                state.count += 1;
                            }
                        }
                    }
                }
                return;
            }
            if let Some((values, third_scale, transform)) = third
                && values.null_count() == 0
                && let Some((third_bytes, third_len)) = values.raw_i64_bytes()
                && third_len == view.num_rows()
            {
                match selection {
                    PredicateSelection::Empty => unreachable!("empty selection returned above"),
                    PredicateSelection::Sparse(rows) => {
                        for row in rows {
                            let left = left_transform.apply_raw_i64(left[row], left_scale);
                            let right = right_transform.apply_raw_i64(right[row], right_scale);
                            let third = transform.apply_raw_i64(
                                read_i64_le_unaligned(third_bytes, row),
                                third_scale,
                            );
                            state.sum +=
                                (left as f64) * (right as f64) * (third as f64) * inv_scale;
                            state.count += 1;
                        }
                    }
                    PredicateSelection::Dense => {
                        for row in 0..view.num_rows() {
                            if predicates.matches(row) {
                                let left = left_transform.apply_raw_i64(left[row], left_scale);
                                let right = right_transform.apply_raw_i64(right[row], right_scale);
                                let third = transform.apply_raw_i64(
                                    read_i64_le_unaligned(third_bytes, row),
                                    third_scale,
                                );
                                state.sum +=
                                    (left as f64) * (right as f64) * (third as f64) * inv_scale;
                                state.count += 1;
                            }
                        }
                    }
                }
                return;
            }
            match selection {
                PredicateSelection::Empty => unreachable!("empty selection returned above"),
                PredicateSelection::Sparse(rows) => {
                    for row in rows {
                        let left = left_transform.apply_raw_i64(left[row], left_scale);
                        let right = right_transform.apply_raw_i64(right[row], right_scale);
                        let mut product = (left as f64) * (right as f64);
                        if let Some((values, third_scale, transform)) = third {
                            if values.is_null(row) {
                                continue;
                            }
                            let Some(third) = values.raw_i64_value(row) else {
                                continue;
                            };
                            product *= transform.apply_raw_i64(third, third_scale) as f64;
                        }
                        state.sum += product * inv_scale;
                        state.count += 1;
                    }
                }
                PredicateSelection::Dense => {
                    for row in 0..view.num_rows() {
                        if !predicates.matches(row) {
                            continue;
                        }
                        let left = left_transform.apply_raw_i64(left[row], left_scale);
                        let right = right_transform.apply_raw_i64(right[row], right_scale);
                        let mut product = (left as f64) * (right as f64);
                        if let Some((values, third_scale, transform)) = third {
                            if values.is_null(row) {
                                continue;
                            }
                            let Some(third) = values.raw_i64_value(row) else {
                                continue;
                            };
                            product *= transform.apply_raw_i64(third, third_scale) as f64;
                        }
                        state.sum += product * inv_scale;
                        state.count += 1;
                    }
                }
            }
            return;
        }
        if let (Some((left, left_len)), Some((right, right_len))) =
            (left.raw_i64_bytes(), right.raw_i64_bytes())
            && left_len == view.num_rows()
            && right_len == view.num_rows()
        {
            let mut scale = (left_scale as f64) * (right_scale as f64);
            if let Some((_, third_scale, _)) = third {
                scale *= third_scale as f64;
            }
            let inv_scale = scale.recip();
            if let Some((values, third_scale, transform)) = third
                && values.null_count() == 0
                && let Some(third_values) = values.raw_i64_values()
            {
                match selection {
                    PredicateSelection::Empty => unreachable!("empty selection returned above"),
                    PredicateSelection::Sparse(rows) => {
                        for row in rows {
                            let left = left_transform
                                .apply_raw_i64(read_i64_le_unaligned(left, row), left_scale);
                            let right = right_transform
                                .apply_raw_i64(read_i64_le_unaligned(right, row), right_scale);
                            let third = transform.apply_raw_i64(third_values[row], third_scale);
                            state.sum +=
                                (left as f64) * (right as f64) * (third as f64) * inv_scale;
                            state.count += 1;
                        }
                    }
                    PredicateSelection::Dense => {
                        for row in 0..view.num_rows() {
                            if predicates.matches(row) {
                                let left = left_transform
                                    .apply_raw_i64(read_i64_le_unaligned(left, row), left_scale);
                                let right = right_transform
                                    .apply_raw_i64(read_i64_le_unaligned(right, row), right_scale);
                                let third = transform.apply_raw_i64(third_values[row], third_scale);
                                state.sum +=
                                    (left as f64) * (right as f64) * (third as f64) * inv_scale;
                                state.count += 1;
                            }
                        }
                    }
                }
                return;
            }
            if let Some((values, third_scale, transform)) = third
                && values.null_count() == 0
                && let Some((third_bytes, third_len)) = values.raw_i64_bytes()
                && third_len == view.num_rows()
            {
                match selection {
                    PredicateSelection::Empty => unreachable!("empty selection returned above"),
                    PredicateSelection::Sparse(rows) => {
                        for row in rows {
                            let left = left_transform
                                .apply_raw_i64(read_i64_le_unaligned(left, row), left_scale);
                            let right = right_transform
                                .apply_raw_i64(read_i64_le_unaligned(right, row), right_scale);
                            let third = transform.apply_raw_i64(
                                read_i64_le_unaligned(third_bytes, row),
                                third_scale,
                            );
                            state.sum +=
                                (left as f64) * (right as f64) * (third as f64) * inv_scale;
                            state.count += 1;
                        }
                    }
                    PredicateSelection::Dense => {
                        for row in 0..view.num_rows() {
                            if predicates.matches(row) {
                                let left = left_transform
                                    .apply_raw_i64(read_i64_le_unaligned(left, row), left_scale);
                                let right = right_transform
                                    .apply_raw_i64(read_i64_le_unaligned(right, row), right_scale);
                                let third = transform.apply_raw_i64(
                                    read_i64_le_unaligned(third_bytes, row),
                                    third_scale,
                                );
                                state.sum +=
                                    (left as f64) * (right as f64) * (third as f64) * inv_scale;
                                state.count += 1;
                            }
                        }
                    }
                }
                return;
            }
            match selection {
                PredicateSelection::Empty => unreachable!("empty selection returned above"),
                PredicateSelection::Sparse(rows) => {
                    for row in rows {
                        let left = left_transform
                            .apply_raw_i64(read_i64_le_unaligned(left, row), left_scale);
                        let right = right_transform
                            .apply_raw_i64(read_i64_le_unaligned(right, row), right_scale);
                        let mut product = (left as f64) * (right as f64);
                        if let Some((values, third_scale, transform)) = third {
                            if values.is_null(row) {
                                continue;
                            }
                            let Some(third) = values.raw_i64_value(row) else {
                                continue;
                            };
                            product *= transform.apply_raw_i64(third, third_scale) as f64;
                        }
                        state.sum += product * inv_scale;
                        state.count += 1;
                    }
                }
                PredicateSelection::Dense => {
                    for row in 0..view.num_rows() {
                        if !predicates.matches(row) {
                            continue;
                        }
                        let left = left_transform
                            .apply_raw_i64(read_i64_le_unaligned(left, row), left_scale);
                        let right = right_transform
                            .apply_raw_i64(read_i64_le_unaligned(right, row), right_scale);
                        let mut product = (left as f64) * (right as f64);
                        if let Some((values, third_scale, transform)) = third {
                            if values.is_null(row) {
                                continue;
                            }
                            let Some(third) = values.raw_i64_value(row) else {
                                continue;
                            };
                            product *= transform.apply_raw_i64(third, third_scale) as f64;
                        }
                        state.sum += product * inv_scale;
                        state.count += 1;
                    }
                }
            }
            return;
        }
    }
    match selection {
        PredicateSelection::Empty => unreachable!("empty selection returned above"),
        PredicateSelection::Sparse(rows) => {
            for row in rows {
                if left.is_null(row) || right.is_null(row) {
                    continue;
                }
                let Some(left) = left.raw_i64_value(row) else {
                    continue;
                };
                let Some(right) = right.raw_i64_value(row) else {
                    continue;
                };
                if let Some((values, _, _)) = third
                    && values.is_null(row)
                {
                    continue;
                }
                let left = left_transform.apply_raw_i64(left, left_scale);
                let right = right_transform.apply_raw_i64(right, right_scale);
                let mut scale = (left_scale as f64) * (right_scale as f64);
                let mut product = (left as f64) * (right as f64);
                if let Some((values, third_scale, transform)) = third {
                    let Some(third) = values.raw_i64_value(row) else {
                        continue;
                    };
                    scale *= third_scale as f64;
                    product *= transform.apply_raw_i64(third, third_scale) as f64;
                }
                state.sum += product / scale;
                state.count += 1;
            }
        }
        PredicateSelection::Dense => {
            for row in 0..view.num_rows() {
                if left.is_null(row) || right.is_null(row) || !predicates.matches(row) {
                    continue;
                }
                let Some(left) = left.raw_i64_value(row) else {
                    continue;
                };
                let Some(right) = right.raw_i64_value(row) else {
                    continue;
                };
                if let Some((values, _, _)) = third
                    && values.is_null(row)
                {
                    continue;
                }
                let left = left_transform.apply_raw_i64(left, left_scale);
                let right = right_transform.apply_raw_i64(right, right_scale);
                let mut scale = (left_scale as f64) * (right_scale as f64);
                let mut product = (left as f64) * (right as f64);
                if let Some((values, third_scale, transform)) = third {
                    let Some(third) = values.raw_i64_value(row) else {
                        continue;
                    };
                    scale *= third_scale as f64;
                    product *= transform.apply_raw_i64(third, third_scale) as f64;
                }
                state.sum += product / scale;
                state.count += 1;
            }
        }
    }
}

fn consume_product_row(
    view: BatchView<'_>,
    spec: &DecimalProductSumSpec,
    row: usize,
    state: &mut DecimalProductSumState,
) -> Result<()> {
    let Some(left) = product_value(view, spec.left_index, spec.left_kind, row)? else {
        return Ok(());
    };
    let Some(right) = product_value(view, spec.right_index, spec.right_kind, row)? else {
        return Ok(());
    };
    match (left, right) {
        (ProductValue::Integer(left), ProductValue::Integer(right))
            if spec.third_kind.is_none() =>
        {
            state.int_sum += left * right;
        }
        (ProductValue::Scaled(left, left_scale), ProductValue::Scaled(right, right_scale)) => {
            let left = spec.left_transform.apply_raw_i64(left as i64, left_scale);
            let right = spec
                .right_transform
                .apply_raw_i64(right as i64, right_scale);
            let mut product = (left as f64) * (right as f64);
            let mut scale = (left_scale as f64) * (right_scale as f64);
            if let Some(third_kind) = spec.third_kind {
                let Some(ProductValue::Scaled(third, third_scale)) = product_value(
                    view,
                    spec.third_index.expect("third product column index"),
                    third_kind,
                    row,
                )?
                else {
                    return Ok(());
                };
                let third = spec
                    .third_transform
                    .unwrap_or(ProductTermTransform::Identity)
                    .apply_raw_i64(third as i64, third_scale);
                product *= third as f64;
                scale *= third_scale as f64;
            }
            state.sum += product / scale;
        }
        _ => return Ok(()),
    }
    state.count += 1;
    Ok(())
}

enum ProductValue {
    Integer(i128),
    Scaled(i128, i64),
}

fn product_value(
    view: BatchView<'_>,
    index: usize,
    kind: ProductColumnKind,
    row: usize,
) -> Result<Option<ProductValue>> {
    match kind {
        ProductColumnKind::Int32 => {
            let Some(values) = view.i32_vector(index) else {
                return Ok(None);
            };
            Ok(
                (!values.is_null(row))
                    .then(|| ProductValue::Integer(i128::from(values.value(row)))),
            )
        }
        ProductColumnKind::Int64 => {
            let Some(values) = view.i64_vector(index) else {
                return Ok(None);
            };
            Ok(
                (!values.is_null(row))
                    .then(|| ProductValue::Integer(i128::from(values.value(row)))),
            )
        }
        ProductColumnKind::DecimalI64 { scale } => {
            let Some(values) = view.decimal128_vector(index) else {
                return Ok(None);
            };
            if values.scale_i64() != Some(scale) || values.is_null(row) {
                return Ok(None);
            }
            Ok(values
                .raw_i64_value(row)
                .map(|value| ProductValue::Scaled(i128::from(value), scale)))
        }
    }
}

fn predicates_match_row(
    view: BatchView<'_>,
    spec: &DecimalProductSumSpec,
    row: usize,
) -> Result<bool> {
    for predicate in &spec.predicates {
        match predicate {
            PrimitivePredicate::Date32 { index, min, max } => {
                let Some(values) = view.date32_vector(*index) else {
                    return Ok(false);
                };
                if values.is_null(row) || !i32_in_bounds(values.value(row), *min, *max) {
                    return Ok(false);
                }
            }
            PrimitivePredicate::Decimal128 {
                index,
                scale,
                min,
                max,
            } => {
                let Some(values) = view.decimal128_vector(*index) else {
                    return Ok(false);
                };
                if values.is_null(row) || values.scale_i64() != Some(*scale) {
                    return Ok(false);
                }
                let Some(value) = values.raw_i64_value(row) else {
                    return Ok(false);
                };
                if !i64_in_bounds(value, *min, *max) {
                    return Ok(false);
                }
            }
        }
    }
    Ok(true)
}

fn batch_predicates_match_row(
    batch: &RecordBatch,
    spec: &DecimalProductSumSpec,
    row: usize,
) -> Result<bool> {
    for predicate in &spec.predicates {
        match predicate {
            PrimitivePredicate::Date32 { index, min, max } => {
                let Some(values) = batch.column(*index).as_any().downcast_ref::<Date32Array>()
                else {
                    return Ok(false);
                };
                if values.is_null(row) || !i32_in_bounds(values.value(row), *min, *max) {
                    return Ok(false);
                }
            }
            PrimitivePredicate::Decimal128 {
                index,
                scale,
                min,
                max,
            } => {
                let Some(input) = decimal_input(batch.column(*index))? else {
                    return Ok(false);
                };
                if input.is_null(row) || input.scale_i64() != Some(*scale) {
                    return Ok(false);
                }
                let value = input.raw_values()[row] as i64;
                if !i64_in_bounds(value, *min, *max) {
                    return Ok(false);
                }
            }
        }
    }
    Ok(true)
}

#[inline]
fn i32_in_bounds(value: i32, min: Option<i32>, max: Option<i32>) -> bool {
    min.is_none_or(|min| value >= min) && max.is_none_or(|max| value <= max)
}

#[inline]
fn i64_in_bounds(value: i64, min: Option<i64>, max: Option<i64>) -> bool {
    min.is_none_or(|min| value >= min) && max.is_none_or(|max| value <= max)
}

fn literal_to_decimal_i64_local(value: &LiteralValue, scale: i64) -> Option<i64> {
    match value {
        LiteralValue::Int64(value) => value.checked_mul(scale),
        LiteralValue::Float64(value) if value.is_finite() => {
            Some((value * scale as f64).round() as i64)
        }
        LiteralValue::Utf8(value) => parse_decimal_literal_i64(value, scale),
        _ => None,
    }
}

fn parse_decimal_literal_i64(value: &str, scale: i64) -> Option<i64> {
    let value = value.trim();
    if value.is_empty() {
        return None;
    }
    let negative = value.starts_with('-');
    let unsigned = value.strip_prefix(['-', '+']).unwrap_or(value);
    let (whole, fraction) = unsigned.split_once('.').unwrap_or((unsigned, ""));
    if whole.is_empty() && fraction.is_empty() {
        return None;
    }
    if !whole.bytes().all(|byte| byte.is_ascii_digit())
        || !fraction.bytes().all(|byte| byte.is_ascii_digit())
    {
        return None;
    }
    let mut raw = whole.parse::<i64>().ok()?.checked_mul(scale)?;
    if !fraction.is_empty() {
        let mut divisor = 1_i64;
        for _ in 0..fraction.len() {
            divisor = divisor.checked_mul(10)?;
        }
        let fraction_raw = fraction
            .parse::<i64>()
            .ok()?
            .checked_mul(scale)?
            .checked_div(divisor)?;
        raw = raw.checked_add(fraction_raw)?;
    }
    Some(if negative { -raw } else { raw })
}

fn literal_to_date32_local(value: &LiteralValue) -> Option<i32> {
    match value {
        LiteralValue::Int64(value) => i32::try_from(*value).ok(),
        LiteralValue::Utf8(value) => {
            let (year, month, day) = parse_ymd(value).ok()?;
            let days = days_from_civil(year, month, day).ok()?;
            i32::try_from(days).ok()
        }
        _ => None,
    }
}

fn decimal_scale_i64_local(scale: i8) -> Result<i64> {
    let scale = u32::try_from(scale)
        .map_err(|_| DodamError::UnsupportedSql("negative decimal scale".to_string()))?;
    10_i64
        .checked_pow(scale)
        .ok_or_else(|| DodamError::UnsupportedSql("decimal scale overflow".to_string()))
}

fn filtered_product_sum_late_max_selected_ratio(product_factor_count: usize) -> f64 {
    let default = if product_factor_count >= 3 {
        0.12
    } else {
        0.20
    };
    std::env::var("DODAM_FILTERED_PRODUCT_SUM_LATE_MAX_SELECTED_RATIO")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite())
        .unwrap_or(default)
        .clamp(0.0, 1.0)
}

fn filtered_product_sum_late_max_selector_run_ratio(product_factor_count: usize) -> f64 {
    let default = if product_factor_count >= 3 {
        0.35
    } else {
        0.50
    };
    std::env::var("DODAM_FILTERED_PRODUCT_SUM_LATE_MAX_SELECTOR_RUN_RATIO")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite())
        .unwrap_or(default)
        .clamp(0.0, 1.0)
}

fn log_product_sum_rule_miss(reason: &str) {
    if std::env::var("DODAM_PRODUCT_SUM_PROFILE")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
    {
        eprintln!("[dodam:product-sum-profile] miss reason={reason}");
    }
}
