use super::*;

pub(super) fn generic_late_materialization_policy(
    default_max_selected_ratio: f64,
) -> LateMaterializationPolicy {
    LateMaterializationPolicy::selective(late_materialization_max_selected_ratio(
        default_max_selected_ratio,
    ))
}

pub(super) fn generic_late_materialization_policy_for_projection(
    predicate_projection: &Projection,
    payload_projection: &Projection,
    default_max_selected_ratio: f64,
    default_max_selector_run_ratio: Option<f64>,
) -> LateMaterializationPolicy {
    let default_max_selected_ratio = late_materialization_projection_selected_ratio(
        predicate_projection,
        payload_projection,
        default_max_selected_ratio,
    );
    let max_selected_ratio = late_materialization_max_selected_ratio(default_max_selected_ratio);
    let Some(default_max_selector_run_ratio) = default_max_selector_run_ratio else {
        return LateMaterializationPolicy::selective(max_selected_ratio)
            .with_selector_runs_per_selected(late_materialization_max_selector_runs_per_selected(
                4.0,
            ));
    };
    LateMaterializationPolicy::selective_with_selector_run_ratio(
        max_selected_ratio,
        late_materialization_max_selector_run_ratio(default_max_selector_run_ratio),
    )
    .with_selector_runs_per_selected(late_materialization_max_selector_runs_per_selected(4.0))
}

pub(super) fn late_materialization_row_group_chunk(default_chunk: usize) -> usize {
    std::env::var("DODAM_LATE_ROW_GROUP_CHUNK")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default_chunk)
}

pub(super) fn late_materialization_max_selected_ratio(default_ratio: f64) -> f64 {
    std::env::var("DODAM_LATE_MAX_SELECTED_RATIO")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite())
        .unwrap_or(default_ratio)
        .clamp(0.0, 1.0)
}

pub(super) fn late_materialization_max_selector_run_ratio(default_ratio: f64) -> f64 {
    std::env::var("DODAM_LATE_MAX_SELECTOR_RUN_RATIO")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite())
        .unwrap_or(default_ratio)
        .clamp(0.0, 1.0)
}

pub(super) fn late_materialization_max_selector_runs_per_selected(default_ratio: f64) -> f64 {
    std::env::var("DODAM_LATE_MAX_SELECTOR_RUNS_PER_SELECTED")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite())
        .unwrap_or(default_ratio)
        .max(0.0)
}

pub(super) fn late_materialization_coalesce_max_gap(default_gap: usize) -> usize {
    std::env::var("DODAM_LATE_COALESCE_MAX_GAP")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(default_gap)
}

fn late_materialization_projection_selected_ratio(
    predicate_projection: &Projection,
    payload_projection: &Projection,
    default_max_selected_ratio: f64,
) -> f64 {
    choose_late_materialization_projection_selected_ratio(ProjectionSelectivityCostInput {
        predicate_columns: projection_column_count_for_cost(predicate_projection),
        payload_columns: projection_column_count_for_cost(payload_projection),
        default_max_selected_ratio,
        narrow_payload_cap: 0.35,
    })
}

fn projection_column_count_for_cost(projection: &Projection) -> Option<usize> {
    match projection_column_count(projection) {
        usize::MAX => None,
        count => Some(count),
    }
}
