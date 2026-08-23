use super::*;

pub(super) fn tpch_profile_enabled() -> bool {
    std::env::var("DODAM_TPCH_PROFILE")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

pub(super) fn tpch_profile_start() -> Option<Instant> {
    tpch_profile_enabled().then(Instant::now)
}

pub(super) fn tpch_profile_elapsed(label: &str, started: Option<Instant>) {
    if let Some(started) = started {
        eprintln!(
            "[dodam:tpch-profile] {label}: {:.3} ms",
            started.elapsed().as_secs_f64() * 1000.0
        );
    }
}

pub(super) fn tpch_profile_late_materialized(
    label: &str,
    metrics: LateMaterializedMetrics,
    row_group_chunk: usize,
) {
    if !tpch_profile_enabled() {
        return;
    }
    let selector_runs_per_selected = if metrics.selected_rows == 0 {
        0.0
    } else {
        metrics.selector_runs as f64 / metrics.selected_rows as f64
    };
    eprintln!(
        "[dodam:tpch-profile] {label}: late_materialized rows={} selected={} ratio={:.6} selector_runs={} row_group_chunk={} predicate_read={:.3} ms payload_read={:.3} ms predicate_batches={} payload_batches={} payload_rows={} selector_runs_per_selected={:.6}",
        metrics.total_rows,
        metrics.selected_rows,
        metrics.selected_ratio(),
        metrics.selector_runs,
        row_group_chunk,
        metrics.predicate_read_nanos as f64 / 1_000_000.0,
        metrics.payload_read_nanos as f64 / 1_000_000.0,
        metrics.predicate_batches,
        metrics.payload_batches,
        metrics.payload_rows,
        selector_runs_per_selected
    );
}

pub(super) fn semijoin_profile_enabled() -> bool {
    std::env::var("DODAM_SEMIJOIN_PROFILE")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

pub(super) fn generic_profile_start() -> Option<Instant> {
    std::env::var("DODAM_GENERIC_PROFILE")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .then(Instant::now)
}

pub(super) fn generic_profile_elapsed(label: &str, started: Option<Instant>) {
    if let Some(started) = started {
        eprintln!(
            "[dodam:generic-profile] {label}: {:.3} ms",
            started.elapsed().as_secs_f64() * 1000.0
        );
    }
}

pub(super) fn sql_elapsed_nanos(started: Instant) -> u64 {
    started.elapsed().as_nanos().min(u64::MAX as u128) as u64
}

pub(super) fn sql_nanos_to_millis(nanos: u64) -> f64 {
    nanos as f64 / 1_000_000.0
}
