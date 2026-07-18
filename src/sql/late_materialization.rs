use super::*;

pub(super) fn late_materialization_policy_from_env(
    env_name: &str,
    default_max_selected_ratio: f64,
) -> LateMaterializationPolicy {
    let max_selected_ratio = std::env::var(env_name)
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite())
        .unwrap_or(default_max_selected_ratio);
    LateMaterializationPolicy::selective(max_selected_ratio)
}

pub(super) fn late_materialization_policy_from_projection_env(
    predicate_projection: &Projection,
    payload_projection: &Projection,
    selected_ratio_env: &str,
    default_max_selected_ratio: f64,
    selector_run_ratio_env: Option<&str>,
    default_max_selector_run_ratio: Option<f64>,
) -> LateMaterializationPolicy {
    let default_max_selected_ratio = late_materialization_projection_selected_ratio(
        predicate_projection,
        payload_projection,
        default_max_selected_ratio,
    );
    let max_selected_ratio = std::env::var(selected_ratio_env)
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite())
        .unwrap_or(default_max_selected_ratio);
    let Some(selector_run_ratio_env) = selector_run_ratio_env else {
        return LateMaterializationPolicy::selective(max_selected_ratio);
    };
    let Some(default_max_selector_run_ratio) = default_max_selector_run_ratio else {
        return LateMaterializationPolicy::selective(max_selected_ratio);
    };
    let max_selector_run_ratio = std::env::var(selector_run_ratio_env)
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite())
        .unwrap_or(default_max_selector_run_ratio);
    LateMaterializationPolicy::selective_with_selector_run_ratio(
        max_selected_ratio,
        max_selector_run_ratio,
    )
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
