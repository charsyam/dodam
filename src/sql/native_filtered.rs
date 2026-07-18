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
    SumF64 { sum: f64, count: u64 },
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
            native_filtered_update_states_profiled(
                &batch,
                &specs,
                &mut states,
                profile,
                &mut mask_nanos,
                &mut input_nanos,
                &mut update_nanos,
            )?;
        }
        metrics.values = native_filtered_finish_states(&specs, states);
        metrics.aggregate_nanos = elapsed_micros_to_nanos(started.elapsed());
        if profile {
            eprintln!(
                "[dodam:generic-profile] native filtered global masks={:.3}ms inputs={:.3}ms update={:.3}ms",
                sql_nanos_to_millis(mask_nanos),
                sql_nanos_to_millis(input_nanos),
                sql_nanos_to_millis(update_nanos),
            );
        }
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
        let direct_group_path = match &group_keys {
            NativeFilteredGroupKeys::I32(values) => values.null_count() == 0,
            NativeFilteredGroupKeys::I64(values) => values.null_count() == 0,
            NativeFilteredGroupKeys::Other => false,
        };
        let direct_predicates =
            if direct_group_path && !native_filtered_direct_predicates_disabled() {
                native_filtered_batch_direct_predicates(&batch, &specs, false)?.map(|predicates| {
                    native_filtered_simplify_direct_predicates(predicates, batch.num_rows())
                })
            } else {
                None
            };
        let masks = if direct_predicates.is_some() {
            Vec::new()
        } else {
            native_filtered_batch_masks_sparse(&batch, &specs)?
        };
        mask_nanos = mask_nanos.saturating_add(elapsed_optional_nanos(mask_started));
        let input_started = profile.then(Instant::now);
        let inputs = native_filtered_batch_inputs(&batch, &specs)?;
        input_nanos = input_nanos.saturating_add(elapsed_optional_nanos(input_started));
        let update_started = profile.then(Instant::now);
        if let NativeFilteredGroupKeys::I32(values) = &group_keys
            && values.null_count() == 0
        {
            if let Some(predicates) = direct_predicates.as_ref() {
                native_filtered_update_i32_group_direct_predicates(
                    values,
                    predicates,
                    &inputs,
                    &specs,
                    &mut dense,
                    &mut groups,
                )?;
            } else if !native_filtered_update_i32_group_sparse_masks(
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
            if let Some(predicates) = direct_predicates.as_ref() {
                native_filtered_update_i64_group_direct_predicates(
                    values,
                    predicates,
                    &inputs,
                    &specs,
                    &mut dense,
                    &mut groups,
                )?;
            } else if !native_filtered_update_i64_group_sparse_masks(
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

fn native_filtered_update_i32_group_direct_predicates(
    values: &Int32Array,
    predicates: &[NativeFilteredDirectPredicate],
    inputs: &[NativeFilteredBatchInput],
    specs: &[NativeFilteredAggregateSpec],
    dense: &mut Vec<Option<Vec<NativeFilteredAggregateState>>>,
    groups: &mut HashMap<GroupValue, Vec<NativeFilteredAggregateState>>,
) -> Result<()> {
    if native_filtered_update_i32_group_direct_predicates_columnar_batch(
        values, predicates, inputs, specs, dense,
    )? {
        return Ok(());
    }
    if native_filtered_update_i32_group_direct_predicates_dense(
        values, predicates, inputs, specs, dense,
    )? {
        return Ok(());
    }
    native_filtered_update_integer_group_direct_predicates(
        values.len(),
        |row| i64::from(values.value(row)),
        predicates,
        inputs,
        specs,
        dense,
        groups,
    )
}

fn native_filtered_update_i32_group_direct_predicates_columnar_batch(
    values: &Int32Array,
    predicates: &[NativeFilteredDirectPredicate],
    inputs: &[NativeFilteredBatchInput],
    specs: &[NativeFilteredAggregateSpec],
    dense: &mut Vec<Option<Vec<NativeFilteredAggregateState>>>,
) -> Result<bool> {
    if !native_filtered_columnar_direct_enabled()
        || values.len() == 0
        || values.null_count() != 0
        || specs.is_empty()
    {
        return Ok(false);
    }
    let mut max_key = 0usize;
    for row in 0..values.len() {
        let key = values.value(row);
        if key < 0 {
            return Ok(false);
        }
        max_key = max_key.max(key as usize);
    }
    if max_key > native_filtered_eager_dense_max_key() {
        return Ok(false);
    }
    if native_filtered_update_i32_group_count_sum_avg_dense_batch(
        values, predicates, inputs, specs, dense, max_key,
    )? {
        return Ok(true);
    }
    if native_filtered_fused_vector_sink_enabled()
        && native_filtered_update_i32_group_direct_predicates_fused_vector_sink(
            values, predicates, inputs, specs, dense, max_key,
        )?
    {
        return Ok(true);
    }
    if let Some(row_masks) = native_filtered_direct_predicate_row_masks(predicates, values.len()) {
        if dense.len() <= max_key {
            dense.resize_with(max_key + 1, || None);
        }
        let mut present = vec![false; max_key + 1];
        for row in 0..values.len() {
            present[values.value(row) as usize] = true;
        }
        for (key, present) in present.into_iter().enumerate() {
            if present && dense[key].is_none() {
                dense[key] = Some(native_filtered_initial_states(specs));
            }
        }
        native_filtered_update_i32_dense_row_masks(values, &row_masks, inputs, specs, dense)?;
        return Ok(true);
    }
    let Some(mut columnar) = native_filtered_columnar_agg_states(specs, inputs, max_key + 1) else {
        return Ok(false);
    };
    let mut present = vec![false; max_key + 1];
    for row in 0..values.len() {
        let key = values.value(row) as usize;
        present[key] = true;
    }
    let mut selected_rows = Vec::new();
    for ((state, predicate), input) in columnar.iter_mut().zip(predicates).zip(inputs) {
        if predicate.prefers_selected_rows() {
            selected_rows.clear();
            selected_rows.reserve(values.len().min(1024));
            predicate.append_selected_rows(values.len(), &mut selected_rows);
        }
        if predicate.prefers_selected_rows()
            && native_filtered_selected_rows_are_sparse(selected_rows.len(), values.len())
        {
            state.update_with_i32_keys(&selected_rows, input, values);
        } else {
            state.update_with_i32_keys_predicate(predicate, input, values);
        }
    }
    if dense.len() <= max_key {
        dense.resize_with(max_key + 1, || None);
    }
    for (key, present) in present.into_iter().enumerate() {
        if !present {
            continue;
        }
        let states = dense[key].get_or_insert_with(|| native_filtered_initial_states(specs));
        for (target, source) in states.iter_mut().zip(&columnar) {
            source.merge_into(target, key);
        }
    }
    Ok(true)
}

fn native_filtered_update_i32_group_count_sum_avg_dense_batch(
    keys: &Int32Array,
    predicates: &[NativeFilteredDirectPredicate],
    inputs: &[NativeFilteredBatchInput],
    specs: &[NativeFilteredAggregateSpec],
    dense: &mut Vec<Option<Vec<NativeFilteredAggregateState>>>,
    max_key: usize,
) -> Result<bool> {
    if !native_filtered_count_sum_avg_dense_sink_enabled() {
        return Ok(false);
    }
    if predicates.len() != 3 || inputs.len() != 3 || specs.len() != 3 || keys.null_count() != 0 {
        return Ok(false);
    }
    if !matches!(
        specs[0].expr,
        AggregateExpr::CountStar | AggregateExpr::Count(_)
    ) || !matches!(specs[1].expr, AggregateExpr::Sum(_))
        || !matches!(specs[2].expr, AggregateExpr::Avg(_))
    {
        return Ok(false);
    }
    let count_input = &inputs[0];
    let sum_input = &inputs[1];
    let avg_input = &inputs[2];
    if !matches!(
        count_input,
        NativeFilteredBatchInput::AlwaysSome
            | NativeFilteredBatchInput::NonNull
            | NativeFilteredBatchInput::I64Array(_)
            | NativeFilteredBatchInput::I32Array(_)
    ) || !matches!(
        sum_input,
        NativeFilteredBatchInput::I64Array(_) | NativeFilteredBatchInput::I32Array(_)
    ) || !matches!(
        avg_input,
        NativeFilteredBatchInput::I64Array(_) | NativeFilteredBatchInput::I32Array(_)
    ) {
        return Ok(false);
    }

    let groups = max_key + 1;
    let mut present = vec![false; groups];
    let mut count_values = vec![0_u64; groups];
    let mut sum_values = vec![0_i64; groups];
    let mut sum_counts = vec![0_u64; groups];
    let mut avg_values = vec![0_i64; groups];
    let mut avg_counts = vec![0_u64; groups];

    for row in 0..keys.len() {
        let key = keys.value(row) as usize;
        present[key] = true;
        if predicates[0].selected(row) && native_filtered_input_present_infallible(count_input, row)
        {
            count_values[key] += 1;
        }
        if predicates[1].selected(row)
            && let Some(value) = native_filtered_input_i64_infallible(sum_input, row)
        {
            sum_values[key] += value;
            sum_counts[key] += 1;
        }
        if predicates[2].selected(row)
            && let Some(value) = native_filtered_input_i64_infallible(avg_input, row)
        {
            avg_values[key] += value;
            avg_counts[key] += 1;
        }
    }

    if dense.len() <= max_key {
        dense.resize_with(max_key + 1, || None);
    }
    for key in 0..groups {
        if !present[key] {
            continue;
        }
        let states = dense[key].get_or_insert_with(|| native_filtered_initial_states(specs));
        match &mut states[0] {
            NativeFilteredAggregateState::Count(count) => {
                *count = count.saturating_add(count_values[key]);
            }
            _ => return Ok(false),
        }
        match &mut states[1] {
            NativeFilteredAggregateState::SumI64 { sum, count } => {
                *sum = sum.saturating_add(sum_values[key]);
                *count = count.saturating_add(sum_counts[key]);
            }
            _ => return Ok(false),
        }
        match &mut states[2] {
            NativeFilteredAggregateState::AvgI64 { sum, count } => {
                *sum = sum.saturating_add(avg_values[key]);
                *count = count.saturating_add(avg_counts[key]);
            }
            _ => return Ok(false),
        }
    }
    Ok(true)
}

fn native_filtered_count_sum_avg_dense_sink_enabled() -> bool {
    std::env::var("DODAM_ENABLE_NATIVE_FILTERED_COUNT_SUM_AVG_DENSE_SINK")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

fn native_filtered_update_i32_group_direct_predicates_fused_vector_sink(
    values: &Int32Array,
    predicates: &[NativeFilteredDirectPredicate],
    inputs: &[NativeFilteredBatchInput],
    specs: &[NativeFilteredAggregateSpec],
    dense: &mut Vec<Option<Vec<NativeFilteredAggregateState>>>,
    max_key: usize,
) -> Result<bool> {
    if predicates.len() != specs.len()
        || inputs.len() != specs.len()
        || specs.len() > u8::BITS as usize
    {
        return Ok(false);
    }
    let Some(mut columnar) = native_filtered_columnar_agg_states(specs, inputs, max_key + 1) else {
        return Ok(false);
    };
    let mut present = vec![false; max_key + 1];
    for row in 0..values.len() {
        let key = values.value(row) as usize;
        present[key] = true;
        for ((state, predicate), input) in columnar.iter_mut().zip(predicates).zip(inputs) {
            if predicate.selected(row) {
                state.update_one_key(key, input, row);
            }
        }
    }
    if dense.len() <= max_key {
        dense.resize_with(max_key + 1, || None);
    }
    for (key, present) in present.into_iter().enumerate() {
        if !present {
            continue;
        }
        let states = dense[key].get_or_insert_with(|| native_filtered_initial_states(specs));
        for (target, source) in states.iter_mut().zip(&columnar) {
            source.merge_into(target, key);
        }
    }
    Ok(true)
}

fn native_filtered_update_i32_group_direct_predicates_dense(
    values: &Int32Array,
    predicates: &[NativeFilteredDirectPredicate],
    inputs: &[NativeFilteredBatchInput],
    specs: &[NativeFilteredAggregateSpec],
    dense: &mut Vec<Option<Vec<NativeFilteredAggregateState>>>,
) -> Result<bool> {
    if values.len() == 0 || values.null_count() != 0 {
        return Ok(true);
    }
    let mut max_key = 0usize;
    for row in 0..values.len() {
        let key = values.value(row);
        if key < 0 {
            return Ok(false);
        }
        max_key = max_key.max(key as usize);
    }
    if max_key > native_filtered_eager_dense_max_key() {
        return Ok(false);
    }
    if dense.len() <= max_key {
        dense.resize_with(max_key + 1, || None);
    }
    let mut present_keys = vec![false; max_key + 1];
    for row in 0..values.len() {
        present_keys[values.value(row) as usize] = true;
    }
    for (key, present) in present_keys.into_iter().enumerate() {
        if present && dense[key].is_none() {
            dense[key] = Some(native_filtered_initial_states(specs));
        }
    }
    if let Some(row_masks) = native_filtered_direct_predicate_row_masks(predicates, values.len()) {
        native_filtered_update_i32_dense_row_masks(values, &row_masks, inputs, specs, dense)?;
        return Ok(true);
    }
    if let Some(row_masks) =
        native_filtered_direct_predicate_row_masks_blocked(predicates, values.len())
    {
        native_filtered_update_i32_dense_row_masks(values, &row_masks, inputs, specs, dense)?;
        return Ok(true);
    }
    for row in 0..values.len() {
        let states = dense[values.value(row) as usize]
            .as_mut()
            .expect("eager dense group initialized");
        for (index, ((spec, predicate), input)) in
            specs.iter().zip(predicates).zip(inputs).enumerate()
        {
            if predicate.selected(row) {
                native_filtered_update_state_fast_direct(&mut states[index], spec, input, row)?;
            }
        }
    }
    Ok(true)
}

fn native_filtered_update_i64_group_direct_predicates(
    values: &Int64Array,
    predicates: &[NativeFilteredDirectPredicate],
    inputs: &[NativeFilteredBatchInput],
    specs: &[NativeFilteredAggregateSpec],
    dense: &mut Vec<Option<Vec<NativeFilteredAggregateState>>>,
    groups: &mut HashMap<GroupValue, Vec<NativeFilteredAggregateState>>,
) -> Result<()> {
    if native_filtered_update_i64_group_direct_predicates_dense(
        values, predicates, inputs, specs, dense,
    )? {
        return Ok(());
    }
    native_filtered_update_integer_group_direct_predicates(
        values.len(),
        |row| values.value(row),
        predicates,
        inputs,
        specs,
        dense,
        groups,
    )
}

enum NativeFilteredColumnarAggState {
    Count(Vec<u64>),
    SumI64 { sums: Vec<i64>, counts: Vec<u64> },
    SumF64 { sums: Vec<f64>, counts: Vec<u64> },
    AvgI64 { sums: Vec<i64>, counts: Vec<u64> },
}

impl NativeFilteredColumnarAggState {
    fn new(
        spec: &NativeFilteredAggregateSpec,
        input: &NativeFilteredBatchInput,
        groups: usize,
    ) -> Option<Self> {
        match (&spec.expr, input) {
            (
                AggregateExpr::Count(_),
                NativeFilteredBatchInput::AlwaysSome
                | NativeFilteredBatchInput::NonNull
                | NativeFilteredBatchInput::I64Array(_)
                | NativeFilteredBatchInput::I32Array(_),
            ) => Some(Self::Count(vec![0; groups])),
            (
                AggregateExpr::Sum(_),
                NativeFilteredBatchInput::I64Array(_) | NativeFilteredBatchInput::I32Array(_),
            ) => Some(Self::SumI64 {
                sums: vec![0; groups],
                counts: vec![0; groups],
            }),
            (AggregateExpr::Sum(_), NativeFilteredBatchInput::ProductF64(_)) => {
                Some(Self::SumF64 {
                    sums: vec![0.0; groups],
                    counts: vec![0; groups],
                })
            }
            (
                AggregateExpr::Avg(_),
                NativeFilteredBatchInput::I64Array(_) | NativeFilteredBatchInput::I32Array(_),
            ) => Some(Self::AvgI64 {
                sums: vec![0; groups],
                counts: vec![0; groups],
            }),
            _ => None,
        }
    }

    fn update_with_i32_keys(
        &mut self,
        selected_rows: &[usize],
        input: &NativeFilteredBatchInput,
        keys: &Int32Array,
    ) {
        match self {
            Self::Count(counts) => match input {
                NativeFilteredBatchInput::AlwaysSome | NativeFilteredBatchInput::NonNull => {
                    for row in selected_rows.iter().copied() {
                        counts[keys.value(row) as usize] += 1;
                    }
                }
                NativeFilteredBatchInput::I64Array(values) => {
                    for row in selected_rows.iter().copied() {
                        if values.is_valid(row) {
                            counts[keys.value(row) as usize] += 1;
                        }
                    }
                }
                NativeFilteredBatchInput::I32Array(values) => {
                    for row in selected_rows.iter().copied() {
                        if values.is_valid(row) {
                            counts[keys.value(row) as usize] += 1;
                        }
                    }
                }
                NativeFilteredBatchInput::ProductF64(_) | NativeFilteredBatchInput::Other(_) => {}
            },
            Self::SumI64 { sums, counts } | Self::AvgI64 { sums, counts } => match input {
                NativeFilteredBatchInput::I64Array(values) => {
                    for row in selected_rows.iter().copied() {
                        if values.is_valid(row) {
                            let key = keys.value(row) as usize;
                            sums[key] += values.value(row);
                            counts[key] += 1;
                        }
                    }
                }
                NativeFilteredBatchInput::I32Array(values) => {
                    for row in selected_rows.iter().copied() {
                        if values.is_valid(row) {
                            let key = keys.value(row) as usize;
                            sums[key] += i64::from(values.value(row));
                            counts[key] += 1;
                        }
                    }
                }
                _ => {}
            },
            Self::SumF64 { sums, counts } => {
                if let NativeFilteredBatchInput::ProductF64(values) = input {
                    for row in selected_rows.iter().copied() {
                        if let Some(value) = values.value(row) {
                            let key = keys.value(row) as usize;
                            sums[key] += value;
                            counts[key] += 1;
                        }
                    }
                }
            }
        }
    }

    fn update_one_key(&mut self, key: usize, input: &NativeFilteredBatchInput, row: usize) {
        match self {
            Self::Count(counts) => match input {
                NativeFilteredBatchInput::AlwaysSome | NativeFilteredBatchInput::NonNull => {
                    counts[key] += 1;
                }
                NativeFilteredBatchInput::I64Array(values) => {
                    if values.is_valid(row) {
                        counts[key] += 1;
                    }
                }
                NativeFilteredBatchInput::I32Array(values) => {
                    if values.is_valid(row) {
                        counts[key] += 1;
                    }
                }
                NativeFilteredBatchInput::ProductF64(_) | NativeFilteredBatchInput::Other(_) => {}
            },
            Self::SumI64 { sums, counts } | Self::AvgI64 { sums, counts } => match input {
                NativeFilteredBatchInput::I64Array(values) => {
                    if values.is_valid(row) {
                        sums[key] += values.value(row);
                        counts[key] += 1;
                    }
                }
                NativeFilteredBatchInput::I32Array(values) => {
                    if values.is_valid(row) {
                        sums[key] += i64::from(values.value(row));
                        counts[key] += 1;
                    }
                }
                _ => {}
            },
            Self::SumF64 { sums, counts } => {
                if let NativeFilteredBatchInput::ProductF64(values) = input
                    && let Some(value) = values.value(row)
                {
                    sums[key] += value;
                    counts[key] += 1;
                }
            }
        }
    }

    fn update_with_i32_keys_predicate(
        &mut self,
        predicate: &NativeFilteredDirectPredicate,
        input: &NativeFilteredBatchInput,
        keys: &Int32Array,
    ) {
        if matches!(predicate, NativeFilteredDirectPredicate::AlwaysTrue) {
            self.update_with_i32_keys_all(input, keys);
            return;
        }
        match self {
            Self::Count(counts) => match input {
                NativeFilteredBatchInput::AlwaysSome | NativeFilteredBatchInput::NonNull => {
                    for row in 0..keys.len() {
                        if predicate.selected(row) {
                            counts[keys.value(row) as usize] += 1;
                        }
                    }
                }
                NativeFilteredBatchInput::I64Array(values) => {
                    for row in 0..keys.len() {
                        if predicate.selected(row) && values.is_valid(row) {
                            counts[keys.value(row) as usize] += 1;
                        }
                    }
                }
                NativeFilteredBatchInput::I32Array(values) => {
                    for row in 0..keys.len() {
                        if predicate.selected(row) && values.is_valid(row) {
                            counts[keys.value(row) as usize] += 1;
                        }
                    }
                }
                NativeFilteredBatchInput::ProductF64(_) | NativeFilteredBatchInput::Other(_) => {}
            },
            Self::SumI64 { sums, counts } | Self::AvgI64 { sums, counts } => match input {
                NativeFilteredBatchInput::I64Array(values) => {
                    for row in 0..keys.len() {
                        if predicate.selected(row) && values.is_valid(row) {
                            let key = keys.value(row) as usize;
                            sums[key] += values.value(row);
                            counts[key] += 1;
                        }
                    }
                }
                NativeFilteredBatchInput::I32Array(values) => {
                    for row in 0..keys.len() {
                        if predicate.selected(row) && values.is_valid(row) {
                            let key = keys.value(row) as usize;
                            sums[key] += i64::from(values.value(row));
                            counts[key] += 1;
                        }
                    }
                }
                _ => {}
            },
            Self::SumF64 { sums, counts } => {
                if let NativeFilteredBatchInput::ProductF64(values) = input {
                    for row in 0..keys.len() {
                        if predicate.selected(row)
                            && let Some(value) = values.value(row)
                        {
                            let key = keys.value(row) as usize;
                            sums[key] += value;
                            counts[key] += 1;
                        }
                    }
                }
            }
        }
    }

    fn update_with_i32_keys_all(&mut self, input: &NativeFilteredBatchInput, keys: &Int32Array) {
        match self {
            Self::Count(counts) => match input {
                NativeFilteredBatchInput::AlwaysSome | NativeFilteredBatchInput::NonNull => {
                    for row in 0..keys.len() {
                        counts[keys.value(row) as usize] += 1;
                    }
                }
                NativeFilteredBatchInput::I64Array(values) => {
                    if values.null_count() == 0 {
                        for row in 0..keys.len() {
                            counts[keys.value(row) as usize] += 1;
                        }
                    } else {
                        for row in 0..keys.len() {
                            if values.is_valid(row) {
                                counts[keys.value(row) as usize] += 1;
                            }
                        }
                    }
                }
                NativeFilteredBatchInput::I32Array(values) => {
                    if values.null_count() == 0 {
                        for row in 0..keys.len() {
                            counts[keys.value(row) as usize] += 1;
                        }
                    } else {
                        for row in 0..keys.len() {
                            if values.is_valid(row) {
                                counts[keys.value(row) as usize] += 1;
                            }
                        }
                    }
                }
                NativeFilteredBatchInput::ProductF64(_) | NativeFilteredBatchInput::Other(_) => {}
            },
            Self::SumI64 { sums, counts } | Self::AvgI64 { sums, counts } => match input {
                NativeFilteredBatchInput::I64Array(values) => {
                    if values.null_count() == 0 {
                        for row in 0..keys.len() {
                            let key = keys.value(row) as usize;
                            sums[key] += values.value(row);
                            counts[key] += 1;
                        }
                    } else {
                        for row in 0..keys.len() {
                            if values.is_valid(row) {
                                let key = keys.value(row) as usize;
                                sums[key] += values.value(row);
                                counts[key] += 1;
                            }
                        }
                    }
                }
                NativeFilteredBatchInput::I32Array(values) => {
                    if values.null_count() == 0 {
                        for row in 0..keys.len() {
                            let key = keys.value(row) as usize;
                            sums[key] += i64::from(values.value(row));
                            counts[key] += 1;
                        }
                    } else {
                        for row in 0..keys.len() {
                            if values.is_valid(row) {
                                let key = keys.value(row) as usize;
                                sums[key] += i64::from(values.value(row));
                                counts[key] += 1;
                            }
                        }
                    }
                }
                _ => {}
            },
            Self::SumF64 { sums, counts } => {
                if let NativeFilteredBatchInput::ProductF64(values) = input {
                    for row in 0..keys.len() {
                        if let Some(value) = values.value(row) {
                            let key = keys.value(row) as usize;
                            sums[key] += value;
                            counts[key] += 1;
                        }
                    }
                }
            }
        }
    }

    fn merge_into(&self, target: &mut NativeFilteredAggregateState, key: usize) {
        match (self, target) {
            (Self::Count(counts), NativeFilteredAggregateState::Count(count)) => {
                *count = count.saturating_add(counts[key]);
            }
            (
                Self::SumI64 { sums, counts },
                NativeFilteredAggregateState::SumI64 { sum, count },
            )
            | (
                Self::AvgI64 { sums, counts },
                NativeFilteredAggregateState::AvgI64 { sum, count },
            ) => {
                *sum = sum.saturating_add(sums[key]);
                *count = count.saturating_add(counts[key]);
            }
            (
                Self::SumF64 { sums, counts },
                NativeFilteredAggregateState::SumF64 { sum, count },
            ) => {
                *sum += sums[key];
                *count = count.saturating_add(counts[key]);
            }
            _ => {}
        }
    }
}

fn native_filtered_selected_rows_are_sparse(selected_rows: usize, rows: usize) -> bool {
    rows > 0 && selected_rows.saturating_mul(4) < rows.saturating_mul(3)
}

fn native_filtered_columnar_agg_states(
    specs: &[NativeFilteredAggregateSpec],
    inputs: &[NativeFilteredBatchInput],
    groups: usize,
) -> Option<Vec<NativeFilteredColumnarAggState>> {
    specs
        .iter()
        .zip(inputs)
        .map(|(spec, input)| NativeFilteredColumnarAggState::new(spec, input, groups))
        .collect()
}

fn native_filtered_columnar_direct_enabled() -> bool {
    !std::env::var("DODAM_DISABLE_NATIVE_FILTERED_COLUMNAR_DIRECT")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

fn native_filtered_fused_vector_sink_enabled() -> bool {
    std::env::var("DODAM_ENABLE_NATIVE_FILTERED_FUSED_VECTOR_SINK")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

fn native_filtered_update_i64_group_direct_predicates_dense(
    values: &Int64Array,
    predicates: &[NativeFilteredDirectPredicate],
    inputs: &[NativeFilteredBatchInput],
    specs: &[NativeFilteredAggregateSpec],
    dense: &mut Vec<Option<Vec<NativeFilteredAggregateState>>>,
) -> Result<bool> {
    if values.len() == 0 || values.null_count() != 0 {
        return Ok(true);
    }
    let mut max_key = 0usize;
    for row in 0..values.len() {
        let key = values.value(row);
        if key < 0 {
            return Ok(false);
        }
        let Ok(key) = usize::try_from(key) else {
            return Ok(false);
        };
        max_key = max_key.max(key);
    }
    if max_key > native_filtered_eager_dense_max_key() {
        return Ok(false);
    }
    if dense.len() <= max_key {
        dense.resize_with(max_key + 1, || None);
    }
    let mut present_keys = vec![false; max_key + 1];
    for row in 0..values.len() {
        present_keys[values.value(row) as usize] = true;
    }
    for (key, present) in present_keys.into_iter().enumerate() {
        if present && dense[key].is_none() {
            dense[key] = Some(native_filtered_initial_states(specs));
        }
    }
    if let Some(row_masks) = native_filtered_direct_predicate_row_masks(predicates, values.len()) {
        native_filtered_update_i64_dense_row_masks(values, &row_masks, inputs, specs, dense)?;
        return Ok(true);
    }
    if let Some(row_masks) =
        native_filtered_direct_predicate_row_masks_blocked(predicates, values.len())
    {
        native_filtered_update_i64_dense_row_masks(values, &row_masks, inputs, specs, dense)?;
        return Ok(true);
    }
    for row in 0..values.len() {
        let states = dense[values.value(row) as usize]
            .as_mut()
            .expect("eager dense group initialized");
        for (index, ((spec, predicate), input)) in
            specs.iter().zip(predicates).zip(inputs).enumerate()
        {
            if predicate.selected(row) {
                native_filtered_update_state_fast_direct(&mut states[index], spec, input, row)?;
            }
        }
    }
    Ok(true)
}

fn native_filtered_direct_predicate_row_masks(
    predicates: &[NativeFilteredDirectPredicate],
    rows: usize,
) -> Option<Vec<u8>> {
    if predicates.is_empty() || predicates.len() > u8::BITS as usize {
        return None;
    }
    let mut masks = vec![0u8; rows];
    for (index, predicate) in predicates.iter().enumerate() {
        let bit = 1u8 << index;
        match predicate {
            NativeFilteredDirectPredicate::AlwaysTrue => {
                for mask in &mut masks {
                    *mask |= bit;
                }
            }
            NativeFilteredDirectPredicate::Precomputed(selected) => {
                if selected.len() < rows {
                    return None;
                }
                for row in 0..rows {
                    if selected[row] {
                        masks[row] |= bit;
                    }
                }
            }
            _ => return None,
        }
    }
    Some(masks)
}

fn native_filtered_direct_predicate_row_masks_blocked(
    predicates: &[NativeFilteredDirectPredicate],
    rows: usize,
) -> Option<Vec<u8>> {
    if !native_filtered_blocked_direct_predicate_masks_enabled()
        || predicates.is_empty()
        || predicates.len() > u8::BITS as usize
    {
        return None;
    }
    let mut masks = vec![0u8; rows];
    for (index, predicate) in predicates.iter().enumerate() {
        let bit = 1u8 << index;
        for row in 0..rows {
            if predicate.selected(row) {
                masks[row] |= bit;
            }
        }
    }
    Some(masks)
}

fn native_filtered_blocked_direct_predicate_masks_enabled() -> bool {
    std::env::var("DODAM_ENABLE_NATIVE_FILTERED_BLOCKED_DIRECT_PREDICATE_MASKS")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

fn native_filtered_update_i32_dense_row_masks(
    values: &Int32Array,
    row_masks: &[u8],
    inputs: &[NativeFilteredBatchInput],
    specs: &[NativeFilteredAggregateSpec],
    dense: &mut [Option<Vec<NativeFilteredAggregateState>>],
) -> Result<()> {
    for row in 0..values.len() {
        let mut mask = row_masks[row];
        if mask == 0 {
            continue;
        }
        let states = dense[values.value(row) as usize]
            .as_mut()
            .expect("eager dense group initialized");
        while mask != 0 {
            let index = mask.trailing_zeros() as usize;
            if !native_filtered_update_state_infallible(&mut states[index], &inputs[index], row) {
                native_filtered_update_state_fast(
                    &mut states[index],
                    &specs[index],
                    &inputs[index],
                    row,
                )?;
            }
            mask &= mask - 1;
        }
    }
    Ok(())
}

fn native_filtered_input_present_infallible(input: &NativeFilteredBatchInput, row: usize) -> bool {
    match input {
        NativeFilteredBatchInput::AlwaysSome | NativeFilteredBatchInput::NonNull => true,
        NativeFilteredBatchInput::I64Array(values) => values.is_valid(row),
        NativeFilteredBatchInput::I32Array(values) => values.is_valid(row),
        NativeFilteredBatchInput::ProductF64(_) | NativeFilteredBatchInput::Other(_) => false,
    }
}

fn native_filtered_input_i64_infallible(
    input: &NativeFilteredBatchInput,
    row: usize,
) -> Option<i64> {
    match input {
        NativeFilteredBatchInput::I64Array(values) if values.is_valid(row) => {
            Some(values.value(row))
        }
        NativeFilteredBatchInput::I32Array(values) if values.is_valid(row) => {
            Some(i64::from(values.value(row)))
        }
        _ => None,
    }
}

fn native_filtered_update_i64_dense_row_masks(
    values: &Int64Array,
    row_masks: &[u8],
    inputs: &[NativeFilteredBatchInput],
    specs: &[NativeFilteredAggregateSpec],
    dense: &mut [Option<Vec<NativeFilteredAggregateState>>],
) -> Result<()> {
    for row in 0..values.len() {
        let mut mask = row_masks[row];
        if mask == 0 {
            continue;
        }
        let states = dense[values.value(row) as usize]
            .as_mut()
            .expect("eager dense group initialized");
        while mask != 0 {
            let index = mask.trailing_zeros() as usize;
            if !native_filtered_update_state_infallible(&mut states[index], &inputs[index], row) {
                native_filtered_update_state_fast(
                    &mut states[index],
                    &specs[index],
                    &inputs[index],
                    row,
                )?;
            }
            mask &= mask - 1;
        }
    }
    Ok(())
}

fn native_filtered_eager_dense_max_key() -> usize {
    std::env::var("DODAM_NATIVE_FILTERED_EAGER_DENSE_MAX_KEY")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(65_536)
        .min(NATIVE_FILTERED_MAX_DENSE_GROUP_KEY)
}

fn native_filtered_direct_predicates_disabled() -> bool {
    std::env::var("DODAM_DISABLE_NATIVE_FILTERED_DIRECT_PREDICATES")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

fn native_filtered_precomputed_direct_predicates_enabled() -> bool {
    !std::env::var("DODAM_DISABLE_NATIVE_FILTERED_PRECOMPUTED_DIRECT_PREDICATES")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

fn native_filtered_update_integer_group_direct_predicates(
    row_count: usize,
    key_at: impl Fn(usize) -> i64,
    predicates: &[NativeFilteredDirectPredicate],
    inputs: &[NativeFilteredBatchInput],
    specs: &[NativeFilteredAggregateSpec],
    dense: &mut Vec<Option<Vec<NativeFilteredAggregateState>>>,
    groups: &mut HashMap<GroupValue, Vec<NativeFilteredAggregateState>>,
) -> Result<()> {
    for row in 0..row_count {
        let states = native_filtered_i64_group_states(key_at(row), dense, groups, specs);
        for (index, ((spec, predicate), input)) in
            specs.iter().zip(predicates).zip(inputs).enumerate()
        {
            if predicate.selected(row) {
                native_filtered_update_state_fast_direct(&mut states[index], spec, input, row)?;
            }
        }
    }
    Ok(())
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
            AggregateExpr::Sum(_) if product_expression_shape(&spec.input).is_some() => {
                NativeFilteredAggregateState::SumF64 { sum: 0.0, count: 0 }
            }
            AggregateExpr::Sum(_) => NativeFilteredAggregateState::SumI64 { sum: 0, count: 0 },
            AggregateExpr::Avg(_) => NativeFilteredAggregateState::AvgI64 { sum: 0, count: 0 },
            AggregateExpr::Min(_) => NativeFilteredAggregateState::MinI64(None),
            AggregateExpr::Max(_) => NativeFilteredAggregateState::MaxI64(None),
            _ => unreachable!("validated native filtered aggregate"),
        })
        .collect()
}

fn native_filtered_update_states_profiled(
    batch: &RecordBatch,
    specs: &[NativeFilteredAggregateSpec],
    states: &mut [NativeFilteredAggregateState],
    profile: bool,
    mask_nanos: &mut u64,
    input_nanos: &mut u64,
    update_nanos: &mut u64,
) -> Result<()> {
    let mask_started = profile.then(Instant::now);
    let masks = native_filtered_batch_masks(batch, specs)?;
    *mask_nanos = mask_nanos.saturating_add(elapsed_optional_nanos(mask_started));
    let input_started = profile.then(Instant::now);
    let inputs = native_filtered_batch_inputs(batch, specs)?;
    *input_nanos = input_nanos.saturating_add(elapsed_optional_nanos(input_started));
    let update_started = profile.then(Instant::now);
    for (index, ((spec, mask), input)) in specs.iter().zip(&masks).zip(&inputs).enumerate() {
        native_filtered_update_global_state_typed(&mut states[index], spec, mask, input)?;
    }
    *update_nanos = update_nanos.saturating_add(elapsed_optional_nanos(update_started));
    Ok(())
}

fn native_filtered_update_global_state_typed(
    state: &mut NativeFilteredAggregateState,
    spec: &NativeFilteredAggregateSpec,
    mask: &NativeFilteredBatchMask,
    input: &NativeFilteredBatchInput,
) -> Result<()> {
    match state {
        NativeFilteredAggregateState::Count(count)
            if matches!(
                input,
                NativeFilteredBatchInput::AlwaysSome | NativeFilteredBatchInput::NonNull
            ) =>
        {
            for row in 0..mask.mask.len() {
                if mask.selected(row) {
                    *count += 1;
                }
            }
            Ok(())
        }
        NativeFilteredAggregateState::Count(count) => match input {
            NativeFilteredBatchInput::I64Array(values) => {
                for row in 0..values.len() {
                    if mask.selected(row) && values.is_valid(row) {
                        *count += 1;
                    }
                }
                Ok(())
            }
            NativeFilteredBatchInput::I32Array(values) => {
                for row in 0..values.len() {
                    if mask.selected(row) && values.is_valid(row) {
                        *count += 1;
                    }
                }
                Ok(())
            }
            _ => native_filtered_update_global_state_typed_fallback(state, spec, mask, input),
        },
        NativeFilteredAggregateState::SumI64 { sum, count }
        | NativeFilteredAggregateState::AvgI64 { sum, count } => match input {
            NativeFilteredBatchInput::I64Array(values) => {
                if values.null_count() == 0 {
                    for row in 0..values.len() {
                        if mask.selected(row) {
                            *sum = sum.saturating_add(values.value(row));
                            *count += 1;
                        }
                    }
                } else {
                    for row in 0..values.len() {
                        if mask.selected(row) && values.is_valid(row) {
                            *sum = sum.saturating_add(values.value(row));
                            *count += 1;
                        }
                    }
                }
                Ok(())
            }
            NativeFilteredBatchInput::I32Array(values) => {
                if values.null_count() == 0 {
                    for row in 0..values.len() {
                        if mask.selected(row) {
                            *sum = sum.saturating_add(i64::from(values.value(row)));
                            *count += 1;
                        }
                    }
                } else {
                    for row in 0..values.len() {
                        if mask.selected(row) && values.is_valid(row) {
                            *sum = sum.saturating_add(i64::from(values.value(row)));
                            *count += 1;
                        }
                    }
                }
                Ok(())
            }
            _ => native_filtered_update_global_state_typed_fallback(state, spec, mask, input),
        },
        NativeFilteredAggregateState::SumF64 { sum, count } => match input {
            NativeFilteredBatchInput::ProductF64(values) if values.is_null_free() => {
                for row in 0..mask.mask.len() {
                    if mask.selected(row)
                        && let Some(value) = values.value(row)
                    {
                        *sum += value;
                        *count += 1;
                    }
                }
                Ok(())
            }
            _ => native_filtered_update_global_state_typed_fallback(state, spec, mask, input),
        },
        NativeFilteredAggregateState::MinI64(_) | NativeFilteredAggregateState::MaxI64(_) => {
            native_filtered_update_global_state_typed_fallback(state, spec, mask, input)
        }
    }
}

fn native_filtered_update_global_state_typed_fallback(
    state: &mut NativeFilteredAggregateState,
    spec: &NativeFilteredAggregateSpec,
    mask: &NativeFilteredBatchMask,
    input: &NativeFilteredBatchInput,
) -> Result<()> {
    for row in 0..mask.mask.len() {
        if mask.selected(row) {
            native_filtered_update_state_fast(state, spec, input, row)?;
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
    fn new_with_sparse(mask: BooleanArray, build_sparse_rows: bool) -> Self {
        let nullable = mask.null_count() > 0;
        let (selected_count, selected_rows) = if build_sparse_rows {
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
            let selected_rows = (selected_count.saturating_mul(4) <= mask.len().saturating_mul(3))
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

enum NativeFilteredDirectPredicate {
    AlwaysTrue,
    AlwaysFalse,
    Precomputed(Vec<bool>),
    IsNotNull(ArrayRef),
    I32Compare {
        values: Int32Array,
        op: BinaryOperator,
        literal: i64,
    },
    I64Compare {
        values: Int64Array,
        op: BinaryOperator,
        literal: i64,
    },
    Decimal128Compare {
        values: Decimal128Array,
        op: BinaryOperator,
        literal: i128,
    },
    Utf8PrefixLike {
        values: StringArray,
        prefix: String,
        negated: bool,
    },
}

impl NativeFilteredDirectPredicate {
    fn selected(&self, row: usize) -> bool {
        match self {
            Self::AlwaysTrue => true,
            Self::AlwaysFalse => false,
            Self::Precomputed(selected) => selected.get(row).copied().unwrap_or(false),
            Self::IsNotNull(values) => values.is_valid(row),
            Self::I32Compare {
                values,
                op,
                literal,
            } => {
                values.is_valid(row)
                    && native_filtered_compare_i64(i64::from(values.value(row)), op, *literal)
            }
            Self::I64Compare {
                values,
                op,
                literal,
            } => {
                values.is_valid(row) && native_filtered_compare_i64(values.value(row), op, *literal)
            }
            Self::Decimal128Compare {
                values,
                op,
                literal,
            } => {
                values.is_valid(row)
                    && native_filtered_compare_i128(values.value(row), op, *literal)
            }
            Self::Utf8PrefixLike {
                values,
                prefix,
                negated,
            } => {
                let matched = values.is_valid(row) && values.value(row).starts_with(prefix);
                if *negated { !matched } else { matched }
            }
        }
    }

    fn append_selected_rows(&self, rows: usize, selected_rows: &mut Vec<usize>) {
        match self {
            Self::AlwaysTrue => selected_rows.extend(0..rows),
            Self::AlwaysFalse => {}
            Self::Precomputed(selected) => {
                for row in 0..rows.min(selected.len()) {
                    if selected[row] {
                        selected_rows.push(row);
                    }
                }
            }
            Self::IsNotNull(values) => {
                for row in 0..rows {
                    if values.is_valid(row) {
                        selected_rows.push(row);
                    }
                }
            }
            Self::I32Compare {
                values,
                op,
                literal,
            } => {
                for row in 0..rows {
                    if values.is_valid(row)
                        && native_filtered_compare_i64(i64::from(values.value(row)), op, *literal)
                    {
                        selected_rows.push(row);
                    }
                }
            }
            Self::I64Compare {
                values,
                op,
                literal,
            } => {
                for row in 0..rows {
                    if values.is_valid(row)
                        && native_filtered_compare_i64(values.value(row), op, *literal)
                    {
                        selected_rows.push(row);
                    }
                }
            }
            Self::Decimal128Compare {
                values,
                op,
                literal,
            } => {
                for row in 0..rows {
                    if values.is_valid(row)
                        && native_filtered_compare_i128(values.value(row), op, *literal)
                    {
                        selected_rows.push(row);
                    }
                }
            }
            Self::Utf8PrefixLike {
                values,
                prefix,
                negated,
            } => {
                for row in 0..rows {
                    let matched = values.is_valid(row) && values.value(row).starts_with(prefix);
                    if if *negated { !matched } else { matched } {
                        selected_rows.push(row);
                    }
                }
            }
        }
    }

    fn prefers_selected_rows(&self) -> bool {
        matches!(
            self,
            Self::Utf8PrefixLike { .. } | Self::Precomputed(_) | Self::AlwaysFalse
        )
    }
}

fn native_filtered_compare_i64(left: i64, op: &BinaryOperator, right: i64) -> bool {
    match op {
        BinaryOperator::Eq => left == right,
        BinaryOperator::NotEq => left != right,
        BinaryOperator::Gt => left > right,
        BinaryOperator::GtEq => left >= right,
        BinaryOperator::Lt => left < right,
        BinaryOperator::LtEq => left <= right,
        _ => false,
    }
}

fn native_filtered_compare_i128(left: i128, op: &BinaryOperator, right: i128) -> bool {
    match op {
        BinaryOperator::Eq => left == right,
        BinaryOperator::NotEq => left != right,
        BinaryOperator::Gt => left > right,
        BinaryOperator::GtEq => left >= right,
        BinaryOperator::Lt => left < right,
        BinaryOperator::LtEq => left <= right,
        _ => false,
    }
}

fn native_filtered_batch_direct_predicates(
    batch: &RecordBatch,
    specs: &[NativeFilteredAggregateSpec],
    precompute: bool,
) -> Result<Option<Vec<NativeFilteredDirectPredicate>>> {
    let mut predicates = Vec::with_capacity(specs.len());
    for spec in specs {
        let Some(predicate) = native_filtered_direct_predicate(batch, &spec.condition)? else {
            return Ok(None);
        };
        predicates.push(if precompute {
            native_filtered_maybe_precompute_direct_predicate(predicate, batch.num_rows())
        } else {
            predicate
        });
    }
    Ok(Some(predicates))
}

fn native_filtered_maybe_precompute_direct_predicate(
    predicate: NativeFilteredDirectPredicate,
    rows: usize,
) -> NativeFilteredDirectPredicate {
    if !native_filtered_precomputed_direct_predicates_enabled() || rows == 0 {
        return predicate;
    }
    let selected = (0..rows)
        .map(|row| predicate.selected(row))
        .collect::<Vec<_>>();
    NativeFilteredDirectPredicate::Precomputed(selected)
}

fn native_filtered_simplify_direct_predicates(
    predicates: Vec<NativeFilteredDirectPredicate>,
    rows: usize,
) -> Vec<NativeFilteredDirectPredicate> {
    if !native_filtered_simplify_direct_predicates_enabled() || rows == 0 {
        return predicates;
    }
    predicates
        .into_iter()
        .map(|predicate| native_filtered_simplify_direct_predicate(predicate, rows))
        .collect()
}

fn native_filtered_simplify_direct_predicate(
    predicate: NativeFilteredDirectPredicate,
    rows: usize,
) -> NativeFilteredDirectPredicate {
    if matches!(
        predicate,
        NativeFilteredDirectPredicate::AlwaysTrue | NativeFilteredDirectPredicate::Precomputed(_)
    ) {
        return predicate;
    }
    let mut selected_count = 0usize;
    for row in 0..rows {
        if predicate.selected(row) {
            selected_count += 1;
        }
    }
    if selected_count == rows {
        NativeFilteredDirectPredicate::AlwaysTrue
    } else if selected_count == 0 {
        NativeFilteredDirectPredicate::Precomputed(vec![false; rows])
    } else {
        predicate
    }
}

fn native_filtered_simplify_direct_predicates_enabled() -> bool {
    std::env::var("DODAM_ENABLE_NATIVE_FILTERED_DIRECT_PREDICATE_SIMPLIFY")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

fn native_filtered_direct_predicate(
    batch: &RecordBatch,
    condition: &SqlExpr,
) -> Result<Option<NativeFilteredDirectPredicate>> {
    match condition {
        SqlExpr::Value(value) => match &value.value {
            Value::Boolean(true) => Ok(Some(NativeFilteredDirectPredicate::AlwaysTrue)),
            Value::Boolean(false) => Ok(Some(NativeFilteredDirectPredicate::AlwaysFalse)),
            _ => Ok(None),
        },
        SqlExpr::IsNotNull(expr) => {
            let column = sql_column_name(expr, None)?;
            let index = output_batch_column_index(batch, &column)?;
            let values = batch.column(index);
            Ok(Some(if values.null_count() == 0 {
                NativeFilteredDirectPredicate::AlwaysTrue
            } else {
                NativeFilteredDirectPredicate::IsNotNull(values.clone())
            }))
        }
        SqlExpr::BinaryOp { left, op, right }
            if matches!(
                op,
                BinaryOperator::Eq
                    | BinaryOperator::NotEq
                    | BinaryOperator::Gt
                    | BinaryOperator::GtEq
                    | BinaryOperator::Lt
                    | BinaryOperator::LtEq
            ) =>
        {
            native_filtered_direct_column_literal_compare(batch, left, op, right).or_else(|_| {
                native_filtered_direct_column_literal_compare(
                    batch,
                    right,
                    &reverse_binary_operator(op),
                    left,
                )
            })
        }
        SqlExpr::Like {
            negated,
            any,
            expr,
            pattern,
            escape_char,
        } => {
            if *any || escape_char.is_some() {
                return Ok(None);
            }
            let column = sql_column_name(expr, None)?;
            let pattern = sql_like_pattern(pattern)?;
            let Some(prefix) = simple_prefix_like_pattern(&pattern) else {
                return Ok(None);
            };
            let index = output_batch_column_index(batch, &column)?;
            let values = batch.column(index);
            let Some(values) = values.as_any().downcast_ref::<StringArray>() else {
                return Ok(None);
            };
            Ok(Some(NativeFilteredDirectPredicate::Utf8PrefixLike {
                values: values.clone(),
                prefix,
                negated: *negated,
            }))
        }
        _ => Ok(None),
    }
}

fn native_filtered_direct_column_literal_compare(
    batch: &RecordBatch,
    column_expr: &SqlExpr,
    op: &BinaryOperator,
    literal_expr: &SqlExpr,
) -> Result<Option<NativeFilteredDirectPredicate>> {
    let Ok(column) = sql_column_name(column_expr, None) else {
        return Ok(None);
    };
    let Ok(literal) = sql_literal_value(literal_expr) else {
        return Ok(None);
    };
    if matches!(literal, LiteralValue::Null) {
        return Ok(None);
    }
    let index = output_batch_column_index(batch, &column)?;
    let array = batch.column(index);
    match array.data_type() {
        DataType::Int32 => {
            let literal = literal_as_i64_for_type(&literal)?.ok_or_else(|| {
                DodamError::UnsupportedSql("native filtered integer literal is NULL".to_string())
            })?;
            Ok(Some(NativeFilteredDirectPredicate::I32Compare {
                values: array
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .expect("Int32")
                    .clone(),
                op: op.clone(),
                literal,
            }))
        }
        DataType::Int64 => {
            let literal = literal_as_i64_for_type(&literal)?.ok_or_else(|| {
                DodamError::UnsupportedSql("native filtered integer literal is NULL".to_string())
            })?;
            Ok(Some(NativeFilteredDirectPredicate::I64Compare {
                values: array
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .expect("Int64")
                    .clone(),
                op: op.clone(),
                literal,
            }))
        }
        DataType::Decimal128(precision, scale) => {
            let literal = native_filtered_decimal_literal(&literal, *precision, *scale)?;
            Ok(Some(NativeFilteredDirectPredicate::Decimal128Compare {
                values: array
                    .as_any()
                    .downcast_ref::<Decimal128Array>()
                    .expect("Decimal128")
                    .clone(),
                op: op.clone(),
                literal,
            }))
        }
        _ => Ok(None),
    }
}

fn native_filtered_decimal_literal(
    literal: &LiteralValue,
    precision: u8,
    scale: i8,
) -> Result<i128> {
    match literal {
        LiteralValue::Int64(value) => {
            parse_decimal_literal_to_scaled(&value.to_string(), scale, precision)
        }
        LiteralValue::Float64(value) => {
            parse_decimal_literal_to_scaled(&value.to_string(), scale, precision)
        }
        LiteralValue::Utf8(value) => parse_decimal_literal_to_scaled(value, scale, precision),
        value => Err(DodamError::UnsupportedSql(format!(
            "native filtered decimal literal must be numeric or string, got {value}"
        ))),
    }
}

fn simple_prefix_like_pattern(pattern: &str) -> Option<String> {
    let prefix = pattern.strip_suffix('%')?;
    if prefix.contains('%') || prefix.contains('_') {
        return None;
    }
    Some(prefix.to_string())
}

fn native_filtered_batch_masks(
    batch: &RecordBatch,
    specs: &[NativeFilteredAggregateSpec],
) -> Result<Vec<NativeFilteredBatchMask>> {
    native_filtered_batch_masks_impl(batch, specs, false)
}

fn native_filtered_batch_masks_sparse(
    batch: &RecordBatch,
    specs: &[NativeFilteredAggregateSpec],
) -> Result<Vec<NativeFilteredBatchMask>> {
    native_filtered_batch_masks_impl(batch, specs, true)
}

fn native_filtered_batch_masks_impl(
    batch: &RecordBatch,
    specs: &[NativeFilteredAggregateSpec],
    build_sparse_rows: bool,
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
            let mask =
                if let Some(value) = native_filtered_constant_boolean_condition(&spec.condition) {
                    BooleanArray::from(vec![value; batch.num_rows()])
                } else {
                    evaluate_scalar_predicate(batch, &spec.condition, None)?
                };
            unique.push(NativeFilteredBatchMask::new_with_sparse(
                mask,
                build_sparse_rows,
            ));
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

fn native_filtered_constant_boolean_condition(condition: &SqlExpr) -> Option<bool> {
    match condition {
        SqlExpr::Value(value) => match &value.value {
            Value::Boolean(value) => Some(*value),
            _ => None,
        },
        _ => None,
    }
}

#[derive(Clone)]
enum NativeFilteredBatchInput {
    AlwaysSome,
    NonNull,
    I64Array(Int64Array),
    I32Array(Int32Array),
    ProductF64(NativeFilteredProductF64Input),
    Other(EvaluatedScalar),
}

#[derive(Clone)]
struct NativeFilteredProductF64Input {
    terms: Vec<NativeFilteredProductF64Term>,
    inv_scale: f64,
}

#[derive(Clone)]
struct NativeFilteredProductF64Term {
    values: Decimal128Array,
    transform: ProductTermTransform,
    scale: f64,
}

impl NativeFilteredProductF64Input {
    #[inline]
    fn value(&self, row: usize) -> Option<f64> {
        match self.terms.as_slice() {
            [left, right] => {
                if left.values.is_null(row) || right.values.is_null(row) {
                    return None;
                }
                Some(left.transformed_raw(row) * right.transformed_raw(row) * self.inv_scale)
            }
            [left, right, third] => {
                if left.values.is_null(row)
                    || right.values.is_null(row)
                    || third.values.is_null(row)
                {
                    return None;
                }
                Some(
                    left.transformed_raw(row)
                        * right.transformed_raw(row)
                        * third.transformed_raw(row)
                        * self.inv_scale,
                )
            }
            _ => None,
        }
    }

    fn is_null_free(&self) -> bool {
        self.terms.iter().all(|term| term.values.null_count() == 0)
    }
}

impl NativeFilteredProductF64Term {
    #[inline]
    fn transformed_raw(&self, row: usize) -> f64 {
        let raw = self.values.value(row) as f64;
        self.transform.apply_raw_f64(raw, self.scale)
    }
}

fn native_filtered_batch_inputs(
    batch: &RecordBatch,
    specs: &[NativeFilteredAggregateSpec],
) -> Result<Vec<NativeFilteredBatchInput>> {
    let mut cache = HashMap::<String, usize>::new();
    let mut unique = Vec::<NativeFilteredBatchInput>::new();
    let mut mapping = Vec::with_capacity(specs.len());
    for spec in specs {
        let key = native_filtered_input_cache_key(spec);
        let index = if let Some(index) = cache.get(&key).copied() {
            index
        } else {
            let index = unique.len();
            unique.push(native_filtered_batch_input(batch, spec)?);
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

fn native_filtered_input_cache_key(spec: &NativeFilteredAggregateSpec) -> String {
    format!(
        "{}:{:?}:{}",
        native_filtered_aggregate_cache_key(&spec.expr),
        spec.input_kind,
        scalar_expression_cache_key(&spec.input)
    )
}

fn native_filtered_aggregate_cache_key(expr: &AggregateExpr) -> &'static str {
    match expr {
        AggregateExpr::CountStar => "count_star",
        AggregateExpr::Count(_) => "count",
        AggregateExpr::Sum(_) => "sum",
        AggregateExpr::Avg(_) => "avg",
        AggregateExpr::Min(_) => "min",
        AggregateExpr::Max(_) => "max",
        AggregateExpr::CountDistinct(_) => "count_distinct",
    }
}

fn scalar_expression_cache_key(expr: &ScalarSqlExpression) -> String {
    match expr {
        ScalarSqlExpression::Column(column) => format!("column:{column}"),
        ScalarSqlExpression::Literal(value) => format!("literal:{value:?}"),
        _ => format!("expr:{expr:?}"),
    }
}

fn native_filtered_batch_input(
    batch: &RecordBatch,
    spec: &NativeFilteredAggregateSpec,
) -> Result<NativeFilteredBatchInput> {
    if spec.input_kind == NativeFilteredInputKind::AlwaysSome {
        return Ok(NativeFilteredBatchInput::AlwaysSome);
    }
    if matches!(spec.expr, AggregateExpr::Sum(_))
        && let Some(input) = native_filtered_product_f64_input(batch, &spec.input)?
    {
        return Ok(NativeFilteredBatchInput::ProductF64(input));
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

fn native_filtered_product_f64_input(
    batch: &RecordBatch,
    expr: &ScalarSqlExpression,
) -> Result<Option<NativeFilteredProductF64Input>> {
    let Some(shape) = product_expression_shape(expr) else {
        return Ok(None);
    };
    let mut terms = Vec::with_capacity(shape.terms.len());
    let mut scale = 1.0;
    for term in shape.terms {
        let column = term.column;
        let index = output_batch_column_index(batch, &column)?;
        let array = batch.column(index);
        let Some(decimal) = decimal_input(array)? else {
            return Ok(None);
        };
        scale *= decimal.scale;
        terms.push(NativeFilteredProductF64Term {
            values: decimal.values.clone(),
            transform: term.transform,
            scale: decimal.scale,
        });
    }
    Ok(Some(NativeFilteredProductF64Input {
        terms,
        inv_scale: scale.recip(),
    }))
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
        NativeFilteredAggregateState::SumF64 { sum, count } => {
            if let Some(value) = native_filtered_input_f64(input, row)? {
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
            NativeFilteredBatchInput::ProductF64(_) => {
                native_filtered_update_state(state, spec, input, row)
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
            NativeFilteredBatchInput::ProductF64(_) => {
                native_filtered_update_state(state, spec, input, row)
            }
            _ => native_filtered_update_state(state, spec, input, row),
        },
        NativeFilteredAggregateState::SumF64 { sum, count } => {
            if let Some(value) = native_filtered_input_f64(input, row)? {
                *sum += value;
                *count += 1;
            }
            Ok(())
        }
        NativeFilteredAggregateState::MinI64(_) | NativeFilteredAggregateState::MaxI64(_) => {
            native_filtered_update_state(state, spec, input, row)
        }
    }
}

#[inline]
fn native_filtered_update_state_fast_direct(
    state: &mut NativeFilteredAggregateState,
    spec: &NativeFilteredAggregateSpec,
    input: &NativeFilteredBatchInput,
    row: usize,
) -> Result<()> {
    if native_filtered_update_state_infallible(state, input, row) {
        Ok(())
    } else {
        native_filtered_update_state_fast(state, spec, input, row)
    }
}

#[inline]
fn native_filtered_update_state_infallible(
    state: &mut NativeFilteredAggregateState,
    input: &NativeFilteredBatchInput,
    row: usize,
) -> bool {
    match state {
        NativeFilteredAggregateState::Count(count) => match input {
            NativeFilteredBatchInput::AlwaysSome | NativeFilteredBatchInput::NonNull => {
                *count += 1;
                true
            }
            NativeFilteredBatchInput::I64Array(values) => {
                if !values.is_null(row) {
                    *count += 1;
                }
                true
            }
            NativeFilteredBatchInput::I32Array(values) => {
                if !values.is_null(row) {
                    *count += 1;
                }
                true
            }
            NativeFilteredBatchInput::ProductF64(values) => {
                if !values.is_null_free() {
                    return false;
                }
                if values.value(row).is_some() {
                    *count += 1;
                }
                true
            }
            NativeFilteredBatchInput::Other(_) => false,
        },
        NativeFilteredAggregateState::SumI64 { sum, count }
        | NativeFilteredAggregateState::AvgI64 { sum, count } => match input {
            NativeFilteredBatchInput::I64Array(values) => {
                if !values.is_null(row) {
                    *sum += values.value(row);
                    *count += 1;
                }
                true
            }
            NativeFilteredBatchInput::I32Array(values) => {
                if !values.is_null(row) {
                    *sum += i64::from(values.value(row));
                    *count += 1;
                }
                true
            }
            _ => false,
        },
        NativeFilteredAggregateState::SumF64 { sum, count } => match input {
            NativeFilteredBatchInput::ProductF64(values) => {
                if !values.is_null_free() {
                    return false;
                }
                if let Some(value) = values.value(row) {
                    *sum += value;
                    *count += 1;
                }
                true
            }
            _ => false,
        },
        NativeFilteredAggregateState::MinI64(_) | NativeFilteredAggregateState::MaxI64(_) => false,
    }
}

fn native_filtered_input_is_some(input: &NativeFilteredBatchInput, row: usize) -> Result<bool> {
    Ok(match input {
        NativeFilteredBatchInput::AlwaysSome | NativeFilteredBatchInput::NonNull => true,
        NativeFilteredBatchInput::I64Array(values) => !values.is_null(row),
        NativeFilteredBatchInput::I32Array(values) => !values.is_null(row),
        NativeFilteredBatchInput::ProductF64(values) => values.value(row).is_some(),
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
        NativeFilteredBatchInput::ProductF64(_) => Err(DodamError::UnsupportedSql(
            "filtered product input is not an integer".to_string(),
        )),
        NativeFilteredBatchInput::Other(value) => scalar_value_as_i64(value, row),
    }
}

fn native_filtered_input_f64(input: &NativeFilteredBatchInput, row: usize) -> Result<Option<f64>> {
    match input {
        NativeFilteredBatchInput::ProductF64(values) => Ok(values.value(row)),
        NativeFilteredBatchInput::I64Array(values) => {
            Ok(values.is_valid(row).then(|| values.value(row) as f64))
        }
        NativeFilteredBatchInput::I32Array(values) => {
            Ok(values.is_valid(row).then(|| f64::from(values.value(row))))
        }
        NativeFilteredBatchInput::Other(value) => scalar_value_as_f64(value, row),
        NativeFilteredBatchInput::AlwaysSome | NativeFilteredBatchInput::NonNull => Ok(None),
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
                NativeFilteredAggregateState::SumF64 { sum, count } => {
                    AggregateValue::Float64((count > 0).then_some(sum))
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
