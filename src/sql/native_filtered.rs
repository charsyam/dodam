use super::*;

#[derive(Debug, Clone, PartialEq)]
pub(super) struct NativeFilteredAggregateSpec {
    pub(super) expr: AggregateExpr,
    pub(super) condition: SqlExpr,
    pub(super) input: ScalarSqlExpression,
    pub(super) input_kind: NativeFilteredInputKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum NativeFilteredInputKind {
    AlwaysSome,
    Nullable,
}

#[derive(Clone)]
enum NativeFilteredAggregateState {
    Count(u64),
    SumI64 { sum: i64, count: u64 },
    AvgI64 { sum: i64, count: u64 },
    MinI64(Option<i64>),
    MaxI64(Option<i64>),
}

const NATIVE_FILTERED_MAX_DENSE_GROUP_KEY: usize = 1_000_000;

pub(super) fn collect_native_filtered_aggregates(
    mut stream: SendableBatchStream,
    fragments: usize,
    group_by: &[String],
    specs: Vec<NativeFilteredAggregateSpec>,
) -> Result<AggregateMetrics> {
    let started = Instant::now();
    let mut metrics = AggregateMetrics {
        fragments,
        ..AggregateMetrics::default()
    };
    if group_by.is_empty() {
        let mut states = native_filtered_initial_states(&specs);
        for batch in stream.by_ref() {
            let batch = batch?;
            if batch.num_rows() == 0 {
                continue;
            }
            metrics.batches += 1;
            metrics.rows += batch.num_rows();
            native_filtered_update_states(&batch, &specs, &mut states)?;
        }
        metrics.values = native_filtered_finish_states(&specs, states);
        metrics.aggregate_nanos = elapsed_micros_to_nanos(started.elapsed());
        return Ok(metrics);
    }
    debug_assert_eq!(group_by.len(), 1);
    let group_column = &group_by[0];
    let mut dense: Vec<Option<Vec<NativeFilteredAggregateState>>> = Vec::new();
    let mut null_i64_group: Option<Vec<NativeFilteredAggregateState>> = None;
    let mut groups: HashMap<GroupValue, Vec<NativeFilteredAggregateState>> = HashMap::new();
    let profile = generic_profile_start().is_some();
    let mut mask_nanos = 0u64;
    let mut input_nanos = 0u64;
    let mut update_nanos = 0u64;
    for batch in stream.by_ref() {
        let batch = batch?;
        if batch.num_rows() == 0 {
            continue;
        }
        metrics.batches += 1;
        metrics.rows += batch.num_rows();
        let group_values = evaluated_column(&batch, group_column)?;
        let group_keys = NativeFilteredGroupKeys::new(&group_values);
        let mask_started = profile.then(Instant::now);
        let masks = native_filtered_batch_masks(&batch, &specs)?;
        mask_nanos = mask_nanos.saturating_add(elapsed_optional_nanos(mask_started));
        let input_started = profile.then(Instant::now);
        let inputs = native_filtered_batch_inputs(&batch, &specs)?;
        input_nanos = input_nanos.saturating_add(elapsed_optional_nanos(input_started));
        let update_started = profile.then(Instant::now);
        if let NativeFilteredGroupKeys::I32(values) = &group_keys
            && values.null_count() == 0
        {
            if !native_filtered_update_i32_group_sparse_masks(
                values,
                &masks,
                &inputs,
                &specs,
                &mut dense,
                &mut groups,
            )? {
                for row in 0..batch.num_rows() {
                    let states = native_filtered_i64_group_states(
                        i64::from(values.value(row)),
                        &mut dense,
                        &mut groups,
                        &specs,
                    );
                    for (index, ((spec, mask), input)) in
                        specs.iter().zip(&masks).zip(&inputs).enumerate()
                    {
                        if mask.selected(row) {
                            native_filtered_update_state_fast(
                                &mut states[index],
                                spec,
                                input,
                                row,
                            )?;
                        }
                    }
                }
            }
            update_nanos = update_nanos.saturating_add(elapsed_optional_nanos(update_started));
            continue;
        }
        if let NativeFilteredGroupKeys::I64(values) = &group_keys
            && values.null_count() == 0
        {
            if !native_filtered_update_i64_group_sparse_masks(
                values,
                &masks,
                &inputs,
                &specs,
                &mut dense,
                &mut groups,
            )? {
                for row in 0..batch.num_rows() {
                    let states = native_filtered_i64_group_states(
                        values.value(row),
                        &mut dense,
                        &mut groups,
                        &specs,
                    );
                    for (index, ((spec, mask), input)) in
                        specs.iter().zip(&masks).zip(&inputs).enumerate()
                    {
                        if mask.selected(row) {
                            native_filtered_update_state_fast(
                                &mut states[index],
                                spec,
                                input,
                                row,
                            )?;
                        }
                    }
                }
            }
            update_nanos = update_nanos.saturating_add(elapsed_optional_nanos(update_started));
            continue;
        }
        for row in 0..batch.num_rows() {
            let states = if group_keys.is_i64_like() {
                match group_keys.i64_key_at(row)? {
                    None => {
                        null_i64_group.get_or_insert_with(|| native_filtered_initial_states(&specs))
                    }
                    Some(key)
                        if key >= 0 && (key as usize) <= NATIVE_FILTERED_MAX_DENSE_GROUP_KEY =>
                    {
                        let index = key as usize;
                        if dense.len() <= index {
                            dense.resize_with(index + 1, || None);
                        }
                        dense[index].get_or_insert_with(|| native_filtered_initial_states(&specs))
                    }
                    Some(key) => groups
                        .entry(GroupValue::Int64(Some(key)))
                        .or_insert_with(|| native_filtered_initial_states(&specs)),
                }
            } else {
                let key = native_group_value_at(&group_values, row)?;
                groups
                    .entry(key)
                    .or_insert_with(|| native_filtered_initial_states(&specs))
            };
            for (index, ((spec, mask), input)) in specs.iter().zip(&masks).zip(&inputs).enumerate()
            {
                if mask.selected(row) {
                    native_filtered_update_state_fast(&mut states[index], spec, input, row)?;
                }
            }
        }
        update_nanos = update_nanos.saturating_add(elapsed_optional_nanos(update_started));
    }
    metrics.groups = groups
        .into_iter()
        .map(|(key, states)| GroupAggregateResult {
            keys: vec![key],
            values: native_filtered_finish_states(&specs, states),
        })
        .collect();
    if let Some(states) = null_i64_group {
        metrics.groups.push(GroupAggregateResult {
            keys: vec![GroupValue::Int64(None)],
            values: native_filtered_finish_states(&specs, states),
        });
    }
    metrics
        .groups
        .extend(dense.into_iter().enumerate().filter_map(|(key, states)| {
            states.map(|states| GroupAggregateResult {
                keys: vec![GroupValue::Int64(Some(key as i64))],
                values: native_filtered_finish_states(&specs, states),
            })
        }));
    metrics
        .groups
        .sort_by(|left, right| compare_native_group_keys(&left.keys, &right.keys));
    metrics.aggregate_nanos = elapsed_micros_to_nanos(started.elapsed());
    if profile {
        eprintln!(
            "[dodam:generic-profile] native filtered aggregate masks={:.3}ms inputs={:.3}ms update={:.3}ms groups={}",
            sql_nanos_to_millis(mask_nanos),
            sql_nanos_to_millis(input_nanos),
            sql_nanos_to_millis(update_nanos),
            metrics.groups.len()
        );
    }
    Ok(metrics)
}

fn native_filtered_i64_group_states<'a>(
    key: i64,
    dense: &'a mut Vec<Option<Vec<NativeFilteredAggregateState>>>,
    groups: &'a mut HashMap<GroupValue, Vec<NativeFilteredAggregateState>>,
    specs: &[NativeFilteredAggregateSpec],
) -> &'a mut Vec<NativeFilteredAggregateState> {
    if key >= 0 && (key as usize) <= NATIVE_FILTERED_MAX_DENSE_GROUP_KEY {
        let index = key as usize;
        if dense.len() <= index {
            dense.resize_with(index + 1, || None);
        }
        dense[index].get_or_insert_with(|| native_filtered_initial_states(specs))
    } else {
        groups
            .entry(GroupValue::Int64(Some(key)))
            .or_insert_with(|| native_filtered_initial_states(specs))
    }
}

fn native_filtered_update_i32_group_sparse_masks(
    values: &Int32Array,
    masks: &[NativeFilteredBatchMask],
    inputs: &[NativeFilteredBatchInput],
    specs: &[NativeFilteredAggregateSpec],
    dense: &mut Vec<Option<Vec<NativeFilteredAggregateState>>>,
    groups: &mut HashMap<GroupValue, Vec<NativeFilteredAggregateState>>,
) -> Result<bool> {
    native_filtered_update_integer_group_sparse_masks(
        values.len(),
        |row| i64::from(values.value(row)),
        masks,
        inputs,
        specs,
        dense,
        groups,
    )
}

fn native_filtered_update_i64_group_sparse_masks(
    values: &Int64Array,
    masks: &[NativeFilteredBatchMask],
    inputs: &[NativeFilteredBatchInput],
    specs: &[NativeFilteredAggregateSpec],
    dense: &mut Vec<Option<Vec<NativeFilteredAggregateState>>>,
    groups: &mut HashMap<GroupValue, Vec<NativeFilteredAggregateState>>,
) -> Result<bool> {
    native_filtered_update_integer_group_sparse_masks(
        values.len(),
        |row| values.value(row),
        masks,
        inputs,
        specs,
        dense,
        groups,
    )
}

fn native_filtered_update_integer_group_sparse_masks(
    row_count: usize,
    key_at: impl Fn(usize) -> i64,
    masks: &[NativeFilteredBatchMask],
    inputs: &[NativeFilteredBatchInput],
    specs: &[NativeFilteredAggregateSpec],
    dense: &mut Vec<Option<Vec<NativeFilteredAggregateState>>>,
    groups: &mut HashMap<GroupValue, Vec<NativeFilteredAggregateState>>,
) -> Result<bool> {
    if row_count == 0 || specs.is_empty() {
        return Ok(true);
    }
    let selected_work = masks
        .iter()
        .map(NativeFilteredBatchMask::selected_count)
        .sum::<usize>();
    let full_work = row_count.saturating_mul(specs.len());
    if selected_work.saturating_mul(4) > full_work.saturating_mul(3) {
        return Ok(false);
    }
    for (index, ((spec, mask), input)) in specs.iter().zip(masks).zip(inputs).enumerate() {
        if let Some(rows) = mask.selected_rows() {
            for &row in rows {
                let states = native_filtered_i64_group_states(key_at(row), dense, groups, specs);
                native_filtered_update_state_fast(&mut states[index], spec, input, row)?;
            }
        } else {
            for row in 0..row_count {
                if mask.selected(row) {
                    let states =
                        native_filtered_i64_group_states(key_at(row), dense, groups, specs);
                    native_filtered_update_state_fast(&mut states[index], spec, input, row)?;
                }
            }
        }
    }
    Ok(true)
}

enum NativeFilteredGroupKeys {
    I32(Int32Array),
    I64(Int64Array),
    Other,
}

impl NativeFilteredGroupKeys {
    fn new(value: &EvaluatedScalar) -> Self {
        let EvaluatedScalar::Array(array) = value else {
            return Self::Other;
        };
        if let Some(values) = array.as_any().downcast_ref::<Int32Array>() {
            return Self::I32(values.clone());
        }
        if let Some(values) = array.as_any().downcast_ref::<Int64Array>() {
            return Self::I64(values.clone());
        }
        Self::Other
    }

    fn is_i64_like(&self) -> bool {
        !matches!(self, Self::Other)
    }

    fn i64_key_at(&self, row: usize) -> Result<Option<i64>> {
        Ok(match self {
            Self::I32(values) => {
                if values.is_null(row) {
                    None
                } else {
                    Some(i64::from(values.value(row)))
                }
            }
            Self::I64(values) => {
                if values.is_null(row) {
                    None
                } else {
                    Some(values.value(row))
                }
            }
            Self::Other => {
                return Err(DodamError::UnsupportedSql(
                    "native filtered aggregate expected integer group key".to_string(),
                ));
            }
        })
    }
}

fn compare_native_group_keys(left: &[GroupValue], right: &[GroupValue]) -> std::cmp::Ordering {
    left.iter()
        .zip(right)
        .map(|(left, right)| compare_native_group_value(left, right))
        .find(|ordering| *ordering != std::cmp::Ordering::Equal)
        .unwrap_or_else(|| left.len().cmp(&right.len()))
}

fn compare_native_group_value(left: &GroupValue, right: &GroupValue) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    match (left, right) {
        (GroupValue::Int64(None), GroupValue::Int64(None)) => Ordering::Equal,
        (GroupValue::Int64(None), GroupValue::Int64(_)) => Ordering::Less,
        (GroupValue::Int64(_), GroupValue::Int64(None)) => Ordering::Greater,
        (GroupValue::Int64(Some(left)), GroupValue::Int64(Some(right))) => left.cmp(right),
        (GroupValue::Utf8(None), GroupValue::Utf8(None)) => Ordering::Equal,
        (GroupValue::Utf8(None), GroupValue::Utf8(_)) => Ordering::Less,
        (GroupValue::Utf8(_), GroupValue::Utf8(None)) => Ordering::Greater,
        (GroupValue::Utf8(Some(left)), GroupValue::Utf8(Some(right))) => left.cmp(right),
        (GroupValue::Date32(None), GroupValue::Date32(None)) => Ordering::Equal,
        (GroupValue::Date32(None), GroupValue::Date32(_)) => Ordering::Less,
        (GroupValue::Date32(_), GroupValue::Date32(None)) => Ordering::Greater,
        (GroupValue::Date32(Some(left)), GroupValue::Date32(Some(right))) => left.cmp(right),
        _ => Ordering::Equal,
    }
}

fn elapsed_micros_to_nanos(duration: Duration) -> u64 {
    duration.as_nanos().min(u128::from(u64::MAX)) as u64
}

pub(super) fn legacy_case_filtered_aggregate_specs(
    aggregates: &[AggregateExpr],
    expressions: &[ProjectionExpression],
) -> Option<Vec<NativeFilteredAggregateSpec>> {
    if aggregates.is_empty() || expressions.is_empty() {
        return None;
    }
    let mut specs = Vec::with_capacity(aggregates.len());
    for aggregate in aggregates {
        let column = aggregate.referenced_column()?;
        let expression = expressions
            .iter()
            .find(|expression| expression.output_name == column)?;
        let ScalarSqlExpression::Case {
            conditions,
            results,
            else_result,
        } = &expression.expr
        else {
            return None;
        };
        let ([condition], [input], None) = (
            conditions.as_slice(),
            results.as_slice(),
            else_result.as_deref(),
        ) else {
            return None;
        };
        if !matches!(
            aggregate,
            AggregateExpr::Count(_)
                | AggregateExpr::Sum(_)
                | AggregateExpr::Avg(_)
                | AggregateExpr::Min(_)
                | AggregateExpr::Max(_)
        ) {
            return None;
        }
        specs.push(NativeFilteredAggregateSpec {
            expr: aggregate.clone(),
            condition: condition.clone(),
            input: input.clone(),
            input_kind: native_filtered_input_kind(aggregate, input),
        });
    }
    Some(specs)
}

pub(super) fn native_filtered_input_kind(
    aggregate: &AggregateExpr,
    input: &ScalarSqlExpression,
) -> NativeFilteredInputKind {
    match aggregate {
        AggregateExpr::Count(_)
            if matches!(
                input,
                ScalarSqlExpression::Literal(LiteralValue::Boolean(_))
                    | ScalarSqlExpression::Literal(LiteralValue::Int64(_))
                    | ScalarSqlExpression::Literal(LiteralValue::Float64(_))
                    | ScalarSqlExpression::Literal(LiteralValue::Utf8(_))
            ) =>
        {
            NativeFilteredInputKind::AlwaysSome
        }
        _ => NativeFilteredInputKind::Nullable,
    }
}

fn native_filtered_initial_states(
    specs: &[NativeFilteredAggregateSpec],
) -> Vec<NativeFilteredAggregateState> {
    specs
        .iter()
        .map(|spec| match spec.expr {
            AggregateExpr::Count(_) => NativeFilteredAggregateState::Count(0),
            AggregateExpr::Sum(_) => NativeFilteredAggregateState::SumI64 { sum: 0, count: 0 },
            AggregateExpr::Avg(_) => NativeFilteredAggregateState::AvgI64 { sum: 0, count: 0 },
            AggregateExpr::Min(_) => NativeFilteredAggregateState::MinI64(None),
            AggregateExpr::Max(_) => NativeFilteredAggregateState::MaxI64(None),
            _ => unreachable!("validated native filtered aggregate"),
        })
        .collect()
}

fn native_filtered_update_states(
    batch: &RecordBatch,
    specs: &[NativeFilteredAggregateSpec],
    states: &mut [NativeFilteredAggregateState],
) -> Result<()> {
    let masks = native_filtered_batch_masks(batch, specs)?;
    let inputs = native_filtered_batch_inputs(batch, specs)?;
    for row in 0..batch.num_rows() {
        for (index, spec) in specs.iter().enumerate() {
            if masks[index].selected(row) {
                native_filtered_update_state(&mut states[index], spec, &inputs[index], row)?;
            }
        }
    }
    Ok(())
}

#[derive(Clone)]
struct NativeFilteredBatchMask {
    mask: BooleanArray,
    nullable: bool,
    selected_count: usize,
    selected_rows: Option<Vec<usize>>,
}

impl NativeFilteredBatchMask {
    fn new(mask: BooleanArray) -> Self {
        let nullable = mask.null_count() > 0;
        let (selected_count, selected_rows) =
            if env_flag_enabled("DODAM_NATIVE_FILTERED_SPARSE_MASKS") {
                let mut selected_rows = Vec::new();
                for row in 0..mask.len() {
                    let selected = if nullable {
                        mask.is_valid(row) && mask.value(row)
                    } else {
                        mask.value(row)
                    };
                    if selected {
                        selected_rows.push(row);
                    }
                }
                let selected_count = selected_rows.len();
                let selected_rows = (selected_count.saturating_mul(4)
                    <= mask.len().saturating_mul(3))
                .then_some(selected_rows);
                (selected_count, selected_rows)
            } else {
                (mask.len(), None)
            };
        Self {
            mask,
            nullable,
            selected_count,
            selected_rows,
        }
    }

    fn selected(&self, row: usize) -> bool {
        if self.nullable {
            self.mask.is_valid(row) && self.mask.value(row)
        } else {
            self.mask.value(row)
        }
    }

    fn selected_count(&self) -> usize {
        self.selected_count
    }

    fn selected_rows(&self) -> Option<&[usize]> {
        self.selected_rows.as_deref()
    }
}

fn native_filtered_batch_masks(
    batch: &RecordBatch,
    specs: &[NativeFilteredAggregateSpec],
) -> Result<Vec<NativeFilteredBatchMask>> {
    let mut cache = HashMap::<String, usize>::new();
    let mut unique = Vec::<NativeFilteredBatchMask>::new();
    let mut mapping = Vec::with_capacity(specs.len());
    for spec in specs {
        let key = spec.condition.to_string();
        let index = if let Some(index) = cache.get(&key).copied() {
            index
        } else {
            let index = unique.len();
            unique.push(NativeFilteredBatchMask::new(evaluate_scalar_predicate(
                batch,
                &spec.condition,
                None,
            )?));
            cache.insert(key, index);
            index
        };
        mapping.push(index);
    }
    Ok(mapping
        .into_iter()
        .map(|index| unique[index].clone())
        .collect())
}

enum NativeFilteredBatchInput {
    AlwaysSome,
    NonNull,
    I64Array(Int64Array),
    I32Array(Int32Array),
    Other(EvaluatedScalar),
}

fn native_filtered_batch_inputs(
    batch: &RecordBatch,
    specs: &[NativeFilteredAggregateSpec],
) -> Result<Vec<NativeFilteredBatchInput>> {
    specs
        .iter()
        .map(|spec| native_filtered_batch_input(batch, spec))
        .collect()
}

fn native_filtered_batch_input(
    batch: &RecordBatch,
    spec: &NativeFilteredAggregateSpec,
) -> Result<NativeFilteredBatchInput> {
    if spec.input_kind == NativeFilteredInputKind::AlwaysSome {
        return Ok(NativeFilteredBatchInput::AlwaysSome);
    }
    let value = evaluate_scalar_expression(batch, &spec.input)?;
    match (&spec.expr, &value) {
        (AggregateExpr::Count(_), EvaluatedScalar::Array(array)) if array.null_count() == 0 => {
            Ok(NativeFilteredBatchInput::NonNull)
        }
        (
            AggregateExpr::Sum(_)
            | AggregateExpr::Avg(_)
            | AggregateExpr::Min(_)
            | AggregateExpr::Max(_),
            EvaluatedScalar::Array(array),
        ) if array.data_type() == &DataType::Int64 => Ok(NativeFilteredBatchInput::I64Array(
            array
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("Int64")
                .clone(),
        )),
        (
            AggregateExpr::Sum(_)
            | AggregateExpr::Avg(_)
            | AggregateExpr::Min(_)
            | AggregateExpr::Max(_),
            EvaluatedScalar::Array(array),
        ) if array.data_type() == &DataType::Int32 => Ok(NativeFilteredBatchInput::I32Array(
            array
                .as_any()
                .downcast_ref::<Int32Array>()
                .expect("Int32")
                .clone(),
        )),
        _ => Ok(NativeFilteredBatchInput::Other(value)),
    }
}

fn native_filtered_update_state(
    state: &mut NativeFilteredAggregateState,
    spec: &NativeFilteredAggregateSpec,
    input: &NativeFilteredBatchInput,
    row: usize,
) -> Result<()> {
    match state {
        NativeFilteredAggregateState::Count(count) => {
            if native_filtered_input_is_some(input, row)? {
                *count += 1;
            }
        }
        NativeFilteredAggregateState::SumI64 { sum, count }
        | NativeFilteredAggregateState::AvgI64 { sum, count } => {
            if let Some(value) = native_filtered_input_i64(input, row)? {
                *sum += value;
                *count += 1;
            }
        }
        NativeFilteredAggregateState::MinI64(value) => {
            if let Some(input) = native_filtered_input_i64(input, row)? {
                *value = Some(value.map_or(input, |current| current.min(input)));
            }
        }
        NativeFilteredAggregateState::MaxI64(value) => {
            if let Some(input) = native_filtered_input_i64(input, row)? {
                *value = Some(value.map_or(input, |current| current.max(input)));
            }
        }
    }
    let _ = spec;
    Ok(())
}

fn native_filtered_update_state_fast(
    state: &mut NativeFilteredAggregateState,
    spec: &NativeFilteredAggregateSpec,
    input: &NativeFilteredBatchInput,
    row: usize,
) -> Result<()> {
    match state {
        NativeFilteredAggregateState::Count(count) => match input {
            NativeFilteredBatchInput::AlwaysSome | NativeFilteredBatchInput::NonNull => {
                *count += 1;
                Ok(())
            }
            NativeFilteredBatchInput::I64Array(values) => {
                if !values.is_null(row) {
                    *count += 1;
                }
                Ok(())
            }
            NativeFilteredBatchInput::I32Array(values) => {
                if !values.is_null(row) {
                    *count += 1;
                }
                Ok(())
            }
            NativeFilteredBatchInput::Other(_) => {
                native_filtered_update_state(state, spec, input, row)
            }
        },
        NativeFilteredAggregateState::SumI64 { sum, count }
        | NativeFilteredAggregateState::AvgI64 { sum, count } => match input {
            NativeFilteredBatchInput::I64Array(values) => {
                if !values.is_null(row) {
                    *sum += values.value(row);
                    *count += 1;
                }
                Ok(())
            }
            NativeFilteredBatchInput::I32Array(values) => {
                if !values.is_null(row) {
                    *sum += i64::from(values.value(row));
                    *count += 1;
                }
                Ok(())
            }
            _ => native_filtered_update_state(state, spec, input, row),
        },
        NativeFilteredAggregateState::MinI64(_) | NativeFilteredAggregateState::MaxI64(_) => {
            native_filtered_update_state(state, spec, input, row)
        }
    }
}

fn native_filtered_input_is_some(input: &NativeFilteredBatchInput, row: usize) -> Result<bool> {
    Ok(match input {
        NativeFilteredBatchInput::AlwaysSome | NativeFilteredBatchInput::NonNull => true,
        NativeFilteredBatchInput::I64Array(values) => !values.is_null(row),
        NativeFilteredBatchInput::I32Array(values) => !values.is_null(row),
        NativeFilteredBatchInput::Other(value) => scalar_value_at(value, row)?.is_some(),
    })
}

fn native_filtered_input_i64(input: &NativeFilteredBatchInput, row: usize) -> Result<Option<i64>> {
    match input {
        NativeFilteredBatchInput::AlwaysSome | NativeFilteredBatchInput::NonNull => Ok(None),
        NativeFilteredBatchInput::I64Array(values) => {
            if values.is_null(row) {
                Ok(None)
            } else {
                Ok(Some(values.value(row)))
            }
        }
        NativeFilteredBatchInput::I32Array(values) => {
            if values.is_null(row) {
                Ok(None)
            } else {
                Ok(Some(i64::from(values.value(row))))
            }
        }
        NativeFilteredBatchInput::Other(value) => scalar_value_as_i64(value, row),
    }
}

fn native_filtered_finish_states(
    specs: &[NativeFilteredAggregateSpec],
    states: Vec<NativeFilteredAggregateState>,
) -> Vec<AggregateResult> {
    specs
        .iter()
        .zip(states)
        .map(|(spec, state)| AggregateResult {
            expr: spec.expr.clone(),
            value: match state {
                NativeFilteredAggregateState::Count(count) => AggregateValue::Count(count),
                NativeFilteredAggregateState::SumI64 { sum, count } => {
                    AggregateValue::Int64((count > 0).then_some(sum))
                }
                NativeFilteredAggregateState::AvgI64 { sum, count } => {
                    AggregateValue::Float64((count > 0).then_some(sum as f64 / count as f64))
                }
                NativeFilteredAggregateState::MinI64(value)
                | NativeFilteredAggregateState::MaxI64(value) => AggregateValue::Int64(value),
            },
        })
        .collect()
}

fn native_group_value_at(value: &EvaluatedScalar, row: usize) -> Result<GroupValue> {
    Ok(match scalar_value_at(value, row)? {
        Some(ScalarValue::Int64(value)) => GroupValue::Int64(Some(value)),
        Some(ScalarValue::Float64(value)) => GroupValue::Int64(Some(value as i64)),
        Some(ScalarValue::Utf8(value)) => GroupValue::Utf8(Some(value)),
        Some(ScalarValue::Date32(value)) => GroupValue::Date32(Some(value)),
        Some(ScalarValue::Decimal128(value, precision, scale)) => {
            GroupValue::Decimal128(Some(value), precision, scale)
        }
        None => GroupValue::Int64(None),
        Some(other) => {
            return Err(DodamError::UnsupportedSql(format!(
                "native filtered aggregate group key does not support {}",
                scalar_value_type_name(&other)
            )));
        }
    })
}

fn scalar_value_type_name(value: &ScalarValue) -> &'static str {
    match value {
        ScalarValue::Int64(_) => "INTEGER",
        ScalarValue::Float64(_) => "DOUBLE",
        ScalarValue::Decimal128(_, _, _) => "DECIMAL",
        ScalarValue::Utf8(_) => "VARCHAR",
        ScalarValue::Boolean(_) => "BOOLEAN",
        ScalarValue::Date32(_) => "DATE",
        ScalarValue::TimestampMillisecond(_) => "TIMESTAMP",
    }
}
