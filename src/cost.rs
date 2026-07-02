use crate::engine::{JoinAlgorithm, JoinExecutionStrategy};
use crate::execution::{JoinBuildSide, JoinType};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JoinCostInput {
    pub left_estimated_bytes: u64,
    pub right_estimated_bytes: u64,
    pub memory_limit_bytes: u64,
    pub requested_algorithm: JoinAlgorithm,
    pub join_type: JoinType,
    pub left_keys: usize,
    pub right_keys: usize,
}

pub fn choose_join_strategy(input: JoinCostInput) -> JoinExecutionStrategy {
    let build_side = choose_build_side(
        input.join_type,
        input.left_estimated_bytes,
        input.right_estimated_bytes,
    );
    let is_inner = input.join_type == JoinType::Inner;
    let is_partitionable = matches!(
        input.join_type,
        JoinType::Inner | JoinType::Full | JoinType::Semi
    );
    let is_single_key_inner = is_inner && input.left_keys == 1 && input.right_keys == 1;

    if input.requested_algorithm == JoinAlgorithm::SortMerge && is_single_key_inner {
        JoinExecutionStrategy::SortMerge
    } else if is_partitionable
        && input.left_estimated_bytes.min(input.right_estimated_bytes) > input.memory_limit_bytes
    {
        JoinExecutionStrategy::PartitionedHash {
            partitions: partition_count(
                input.left_estimated_bytes.min(input.right_estimated_bytes),
                input.memory_limit_bytes,
            ),
            memory_limit_bytes: input.memory_limit_bytes,
        }
    } else {
        JoinExecutionStrategy::Hash { build_side }
    }
}

fn choose_build_side(
    join_type: JoinType,
    left_estimated_bytes: u64,
    right_estimated_bytes: u64,
) -> JoinBuildSide {
    match join_type {
        JoinType::Inner | JoinType::Left | JoinType::Right | JoinType::Full
            if left_estimated_bytes <= right_estimated_bytes =>
        {
            JoinBuildSide::Left
        }
        JoinType::Inner | JoinType::Left | JoinType::Right | JoinType::Full => JoinBuildSide::Right,
        JoinType::Semi => JoinBuildSide::Right,
    }
}

pub fn partition_count(estimated_build_bytes: u64, memory_limit_bytes: u64) -> usize {
    let memory_limit_bytes = memory_limit_bytes.max(1);
    let partitions =
        estimated_build_bytes.saturating_add(memory_limit_bytes - 1) / memory_limit_bytes;
    partitions.clamp(2, 1024) as usize
}
