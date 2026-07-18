use super::*;

pub(super) fn default_join_memory_limit_bytes() -> u64 {
    std::env::var("DODAM_JOIN_MEMORY_LIMIT_BYTES")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(128 * 1024 * 1024)
}

pub(super) fn join_memory_limit_bytes(options: SqlExecutionOptions) -> u64 {
    options
        .join_memory_limit_bytes
        .filter(|value| *value > 0)
        .unwrap_or_else(default_join_memory_limit_bytes)
}
