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
