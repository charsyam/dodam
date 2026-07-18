use super::*;

#[derive(Debug, Clone, PartialEq)]
pub struct SqlQuery {
    pub(super) path: PathBuf,
    pub(super) join: Option<SqlJoin>,
    pub(super) projection: Projection,
    pub(super) filter: Option<FilterExpr>,
    pub(super) expression_filter: Option<SqlExpr>,
    pub(super) having: Option<FilterExpr>,
    pub(super) order_by: Option<SortKey>,
    pub(super) limit: Option<usize>,
    pub(super) offset: usize,
    pub(super) distinct: bool,
    pub(super) aggregates: Vec<AggregateExpr>,
    pub(super) filtered_aggregates: Vec<NativeFilteredAggregateSpec>,
    pub(super) aggregate_expressions: Vec<ProjectionExpression>,
    pub(super) expressions: Vec<ProjectionExpression>,
    pub(super) group_by: Vec<String>,
    pub(super) aliases: Vec<(String, String)>,
    pub(super) qualified_wildcards: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct SqlJoin {
    pub(super) right: SqlTableRef,
    pub(super) left_alias: String,
    pub(super) right_alias: String,
    pub(super) left_keys: Vec<String>,
    pub(super) right_keys: Vec<String>,
    pub(super) right_filter: Option<FilterExpr>,
    pub(super) join_type: JoinType,
}

impl SqlQuery {
    pub fn is_aggregate(&self) -> bool {
        !self.aggregates.is_empty()
    }
}

#[derive(Debug)]
pub enum QueryOutput {
    Scan {
        batches: Vec<RecordBatch>,
    },
    Aggregate {
        metrics: AggregateMetrics,
        batches: Vec<RecordBatch>,
    },
    Explain {
        plan: String,
    },
}

pub trait SqlResultSink {
    fn record_batch_sink(&mut self) -> &mut dyn RecordBatchSink;
    fn write_output(&mut self, output: QueryOutput) -> Result<()>;
}

#[derive(Debug, Clone, Copy, Default)]
pub struct SqlExecutionOptions {
    pub join_memory_limit_bytes: Option<u64>,
}

#[derive(Debug, Clone, Copy)]
pub struct SqlSinkExecutionOptions {
    pub allow_direct_or_streaming: bool,
}

impl Default for SqlSinkExecutionOptions {
    fn default() -> Self {
        Self {
            allow_direct_or_streaming: true,
        }
    }
}

#[derive(Debug, Default)]
pub struct SqlSinkExecutionProfile {
    pub direct_sink: Option<Duration>,
    pub streaming: Option<Duration>,
    pub execute: Option<Duration>,
    pub write_output: Option<Duration>,
    pub scan_plan_metrics: Option<ScanPlanMetrics>,
}
