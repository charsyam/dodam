use crate::catalog::{FileFragment, StorageFormat, TableStatistics};
use crate::execution::{AggregateExpr, FilterExpr, JoinBuildSide, JoinType, Projection, SortKey};
use arrow::record_batch::RecordBatch;

#[derive(Debug, Clone)]
pub enum LogicalPlan {
    TableScan(LogicalScan),
    Projection {
        input: Box<LogicalPlan>,
        projection: Projection,
    },
    Filter {
        input: Box<LogicalPlan>,
        filter: FilterExpr,
    },
    Aggregate {
        input: Box<LogicalPlan>,
        aggregates: Vec<AggregateExpr>,
        group_by: Vec<String>,
    },
    Join {
        left: Box<LogicalPlan>,
        right: Box<LogicalPlan>,
        join_type: JoinType,
        left_keys: Vec<String>,
        right_keys: Vec<String>,
        left_prefix: String,
        right_prefix: String,
        output_projection: Projection,
    },
    Sort {
        input: Box<LogicalPlan>,
        order_by: SortKey,
        limit: Option<usize>,
    },
    Limit {
        input: Box<LogicalPlan>,
        limit: usize,
    },
    Distinct {
        input: Box<LogicalPlan>,
    },
    Copy {
        input: Box<LogicalPlan>,
        format: CopyFormat,
        target: String,
    },
}

#[derive(Debug, Clone)]
pub struct LogicalScan {
    pub source: PlanTableSource,
    pub batch_size: usize,
    pub projection: Projection,
    pub filter: Option<FilterExpr>,
    pub order_by: Option<SortKey>,
    pub limit: Option<usize>,
    pub distinct: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanTableSource {
    pub fragments: Vec<FileFragment>,
    pub format: StorageFormat,
    pub statistics: TableStatistics,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CopyFormat {
    Csv,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PhysicalPlanNode {
    Operator(PhysicalOperatorNode),
}

#[derive(Debug, Clone, PartialEq)]
pub struct PhysicalOperatorNode {
    pub operator: PhysicalOperator,
    pub attributes: Vec<(String, String)>,
    pub children: Vec<PhysicalPlanNode>,
    pub partitioning: Partitioning,
    pub ordering: OutputOrdering,
    pub execution: Option<PhysicalExecutionConfig>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PhysicalOperator {
    Scan,
    Memory,
    Ipc,
    Filter,
    Projection,
    Aggregate,
    LocalFold,
    FinalMerge,
    HashJoin,
    PartitionedHashJoin,
    SortMergeJoin,
    Sort,
    Limit,
    Distinct,
    Exchange(ExchangeKind),
    Sink(SinkKind),
    Empty,
    Other(String),
}

#[derive(Debug, Clone, PartialEq)]
pub enum PhysicalExecutionConfig {
    Scan {
        fragments: Vec<FileFragment>,
        batch_size: usize,
        projection: Projection,
        pushdown_predicates: Vec<crate::execution::Expr>,
    },
    Memory {
        batches: Vec<RecordBatch>,
    },
    Ipc {
        files: Vec<std::path::PathBuf>,
    },
    Filter {
        filter: FilterExpr,
    },
    Projection {
        projection: Projection,
    },
    Sort {
        order_by: SortKey,
        limit: Option<usize>,
    },
    Limit {
        limit: usize,
    },
    Distinct,
    LocalFold {
        group_by: Vec<String>,
        aggregates: Vec<AggregateExpr>,
    },
    FinalMerge {
        group_by: Vec<String>,
        aggregates: Vec<AggregateExpr>,
    },
    HashJoin {
        left_keys: Vec<String>,
        right_keys: Vec<String>,
        left_prefix: String,
        right_prefix: String,
        build_side: JoinBuildSide,
        join_type: JoinType,
        output_projection: Projection,
    },
    PartitionedHashJoin {
        left_keys: Vec<String>,
        right_keys: Vec<String>,
        left_prefix: String,
        right_prefix: String,
        partitions: usize,
        memory_limit_bytes: u64,
        join_type: JoinType,
        output_projection: Projection,
    },
    SortMergeJoin {
        left_key: String,
        right_key: String,
        left_prefix: String,
        right_prefix: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Partitioning {
    Unknown,
    Single,
    Hash {
        keys: Vec<String>,
        partitions: usize,
    },
    RoundRobin {
        partitions: usize,
    },
    FileRange {
        partitions: usize,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Distribution {
    Unspecified,
    Single,
    HashClustered { keys: Vec<String> },
    Broadcast,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExchangeKind {
    HashRepartition {
        keys: Vec<String>,
        partitions: usize,
    },
    Broadcast,
    Gather,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SinkKind {
    Csv,
    Memory,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OutputOrdering {
    pub expressions: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PhysicalJoinStrategy {
    Hash {
        build_side: JoinBuildSide,
    },
    PartitionedHash {
        partitions: usize,
        memory_limit_bytes: u64,
    },
    SortMerge,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhysicalPlanningOptions {
    pub default_join_strategy: PhysicalJoinStrategy,
    pub insert_exchanges: bool,
    pub default_shuffle_partitions: usize,
}

impl Default for PhysicalPlanningOptions {
    fn default() -> Self {
        Self {
            default_join_strategy: PhysicalJoinStrategy::Hash {
                build_side: JoinBuildSide::Right,
            },
            insert_exchanges: false,
            default_shuffle_partitions: 1,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct PhysicalPlanner {
    options: PhysicalPlanningOptions,
}

struct PhysicalJoinPlanningInput<'a> {
    join_type: JoinType,
    left_keys: &'a [String],
    right_keys: &'a [String],
    left_prefix: &'a str,
    right_prefix: &'a str,
    output_projection: &'a Projection,
    left: PhysicalPlanNode,
    right: PhysicalPlanNode,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StagePlan {
    pub id: usize,
    pub root: PhysicalPlanNode,
    pub input_stages: Vec<usize>,
    pub partitioning: Partitioning,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TaskPlan {
    pub stage_id: usize,
    pub partition: usize,
    pub root: PhysicalPlanNode,
    pub inputs: Vec<TaskInput>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaskInput {
    ScanFragment(FileFragment),
    ShufflePartition { stage_id: usize, partition: usize },
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExecutionGraphPlan {
    pub stages: Vec<StagePlan>,
    pub tasks: Vec<TaskPlan>,
}

impl LogicalPlan {
    pub fn to_physical_plan(&self) -> PhysicalPlanNode {
        PhysicalPlanner::default().plan(self)
    }
}

impl PhysicalPlanner {
    pub fn new(options: PhysicalPlanningOptions) -> Self {
        Self { options }
    }

    pub fn plan(&self, logical: &LogicalPlan) -> PhysicalPlanNode {
        match logical {
            LogicalPlan::TableScan(scan) => self.plan_scan(scan),
            LogicalPlan::Projection { input, projection } => {
                let input = self.plan(input);
                let partitioning = input.partitioning_ref().clone();
                PhysicalPlanNode::new("ProjectionExec")
                    .attr("projection", projection_display(projection))
                    .execution(PhysicalExecutionConfig::Projection {
                        projection: projection.clone(),
                    })
                    .child(input)
                    .partitioning(partitioning)
            }
            LogicalPlan::Filter { input, filter } => {
                let input = self.plan(input);
                let partitioning = input.partitioning_ref().clone();
                PhysicalPlanNode::new("FilterExec")
                    .attr("predicate", "logical")
                    .execution(PhysicalExecutionConfig::Filter {
                        filter: filter.clone(),
                    })
                    .child(input)
                    .partitioning(partitioning)
            }
            LogicalPlan::Aggregate {
                input,
                aggregates,
                group_by,
            } => {
                let input = self.plan_aggregate_input(input, group_by);
                let mode = if group_by.is_empty() {
                    "global"
                } else {
                    "grouped"
                };
                let local_fold = PhysicalPlanNode::new("LocalFoldExec")
                    .attr("mode", mode)
                    .attr("group_by", format!("[{}]", group_by.join(",")))
                    .attr("aggregates", aggregate_exprs_display(aggregates))
                    .execution(PhysicalExecutionConfig::LocalFold {
                        group_by: group_by.clone(),
                        aggregates: aggregates.clone(),
                    })
                    .child(input);
                PhysicalPlanNode::new("FinalMergeExec")
                    .attr("mode", mode)
                    .attr("group_by", format!("[{}]", group_by.join(",")))
                    .attr("aggregates", aggregate_exprs_display(aggregates))
                    .execution(PhysicalExecutionConfig::FinalMerge {
                        group_by: group_by.clone(),
                        aggregates: aggregates.clone(),
                    })
                    .child(local_fold)
            }
            LogicalPlan::Join {
                left,
                right,
                join_type,
                left_keys,
                right_keys,
                left_prefix,
                right_prefix,
                output_projection,
            } => self.plan_join(PhysicalJoinPlanningInput {
                join_type: *join_type,
                left_keys,
                right_keys,
                left_prefix,
                right_prefix,
                output_projection,
                left: self.plan(left),
                right: self.plan(right),
            }),
            LogicalPlan::Sort {
                input,
                order_by,
                limit,
            } => PhysicalPlanNode::new("SortExec")
                .attr("order_by", sort_key_display(order_by))
                .attr(
                    "limit",
                    limit
                        .map(|limit| limit.to_string())
                        .unwrap_or_else(|| "none".to_string()),
                )
                .execution(PhysicalExecutionConfig::Sort {
                    order_by: order_by.clone(),
                    limit: *limit,
                })
                .child(self.ensure_single(self.plan(input)))
                .ordering(OutputOrdering {
                    expressions: order_by
                        .expressions
                        .iter()
                        .map(|expr| expr.column.clone())
                        .collect(),
                }),
            LogicalPlan::Limit { input, limit } => PhysicalPlanNode::new("LimitExec")
                .attr("limit", *limit)
                .execution(PhysicalExecutionConfig::Limit { limit: *limit })
                .child(self.ensure_single(self.plan(input)))
                .partitioning(Partitioning::Single),
            LogicalPlan::Distinct { input } => PhysicalPlanNode::new("DistinctExec")
                .execution(PhysicalExecutionConfig::Distinct)
                .child(self.ensure_single(self.plan(input)))
                .partitioning(Partitioning::Single),
            LogicalPlan::Copy {
                input,
                format,
                target,
            } => PhysicalPlanNode::new(match format {
                CopyFormat::Csv => "CsvSinkExec",
            })
            .attr("target", target)
            .child(self.plan(input)),
        }
    }

    fn plan_scan(&self, scan: &LogicalScan) -> PhysicalPlanNode {
        let mut current = PhysicalPlanNode::new("ScanExec")
            .attr("format", format!("{:?}", scan.source.format))
            .attr("fragments", scan.source.fragments.len())
            .attr("rows", scan.source.statistics.rows)
            .attr("row_groups", scan.source.statistics.row_groups)
            .attr("compressed_bytes", scan.source.statistics.compressed_bytes)
            .attr("batch_size", scan.batch_size)
            .attr("projection", projection_display(&scan.projection))
            .execution(PhysicalExecutionConfig::Scan {
                fragments: scan.source.fragments.clone(),
                batch_size: scan.batch_size,
                projection: scan.projection.clone(),
                pushdown_predicates: Vec::new(),
            })
            .partitioning(Partitioning::FileRange {
                partitions: scan.source.fragments.len().max(1),
            });

        if let Some(filter) = &scan.filter {
            let partitioning = current.partitioning_ref().clone();
            current = PhysicalPlanNode::new("FilterExec")
                .attr("predicate", "logical")
                .execution(PhysicalExecutionConfig::Filter {
                    filter: filter.clone(),
                })
                .child(current)
                .partitioning(partitioning);
        }

        let partitioning = current.partitioning_ref().clone();
        current = PhysicalPlanNode::new("ProjectionExec")
            .attr("projection", projection_display(&scan.projection))
            .execution(PhysicalExecutionConfig::Projection {
                projection: scan.projection.clone(),
            })
            .child(current)
            .partitioning(partitioning);

        if scan.distinct {
            current = PhysicalPlanNode::new("DistinctExec")
                .execution(PhysicalExecutionConfig::Distinct)
                .child(self.ensure_single(current))
                .partitioning(Partitioning::Single);
        }

        if let Some(order_by) = &scan.order_by {
            current = PhysicalPlanNode::new("SortExec")
                .attr("order_by", sort_key_display(order_by))
                .attr(
                    "limit",
                    scan.limit
                        .map(|limit| limit.to_string())
                        .unwrap_or_else(|| "none".to_string()),
                )
                .execution(PhysicalExecutionConfig::Sort {
                    order_by: order_by.clone(),
                    limit: scan.limit,
                })
                .child(current)
                .ordering(OutputOrdering {
                    expressions: order_by
                        .expressions
                        .iter()
                        .map(|expr| expr.column.clone())
                        .collect(),
                });
        }

        if let Some(limit) = scan.limit {
            current = PhysicalPlanNode::new("LimitExec")
                .attr("limit", limit)
                .execution(PhysicalExecutionConfig::Limit { limit })
                .child(current)
                .partitioning(Partitioning::Single);
        }

        current
    }

    fn plan_aggregate_input(&self, input: &LogicalPlan, group_by: &[String]) -> PhysicalPlanNode {
        let input = self.plan(input);
        if group_by.is_empty() {
            self.ensure_single(input)
        } else {
            self.ensure_hash(input, group_by, self.options.default_shuffle_partitions)
        }
    }

    fn plan_join(&self, input: PhysicalJoinPlanningInput<'_>) -> PhysicalPlanNode {
        let (left, right) = match self.options.default_join_strategy {
            PhysicalJoinStrategy::PartitionedHash { partitions, .. } => (
                self.ensure_hash(input.left, input.left_keys, partitions),
                self.ensure_hash(input.right, input.right_keys, partitions),
            ),
            PhysicalJoinStrategy::Hash { .. } | PhysicalJoinStrategy::SortMerge => {
                (input.left, input.right)
            }
        };
        let mut node = match self.options.default_join_strategy {
            PhysicalJoinStrategy::Hash { build_side } => PhysicalPlanNode::new("JoinExec")
                .attr("strategy", "hash")
                .attr("build", format!("{build_side:?}"))
                .execution(PhysicalExecutionConfig::HashJoin {
                    left_keys: input.left_keys.to_vec(),
                    right_keys: input.right_keys.to_vec(),
                    left_prefix: input.left_prefix.to_string(),
                    right_prefix: input.right_prefix.to_string(),
                    build_side,
                    join_type: input.join_type,
                    output_projection: input.output_projection.clone(),
                }),
            PhysicalJoinStrategy::PartitionedHash {
                partitions,
                memory_limit_bytes,
            } => PhysicalPlanNode::new("PartitionedHashJoinExec")
                .attr("strategy", "partitioned_hash")
                .attr("partitions", partitions)
                .attr("memory_limit_bytes", memory_limit_bytes)
                .execution(PhysicalExecutionConfig::PartitionedHashJoin {
                    left_keys: input.left_keys.to_vec(),
                    right_keys: input.right_keys.to_vec(),
                    left_prefix: input.left_prefix.to_string(),
                    right_prefix: input.right_prefix.to_string(),
                    partitions,
                    memory_limit_bytes,
                    join_type: input.join_type,
                    output_projection: input.output_projection.clone(),
                }),
            PhysicalJoinStrategy::SortMerge => PhysicalPlanNode::new("SortMergeJoinExec")
                .attr("strategy", "sort_merge")
                .execution(PhysicalExecutionConfig::SortMergeJoin {
                    left_key: input.left_keys.first().cloned().unwrap_or_default(),
                    right_key: input.right_keys.first().cloned().unwrap_or_default(),
                    left_prefix: input.left_prefix.to_string(),
                    right_prefix: input.right_prefix.to_string(),
                }),
        };

        node = node
            .attr("type", format!("{:?}", input.join_type))
            .attr("left_keys", format!("[{}]", input.left_keys.join(",")))
            .attr("right_keys", format!("[{}]", input.right_keys.join(",")))
            .attr(
                "output_projection",
                projection_display(input.output_projection),
            );

        if let PhysicalJoinStrategy::PartitionedHash { partitions, .. } =
            self.options.default_join_strategy
        {
            node = node.partitioning(Partitioning::Hash {
                keys: input.left_keys.to_vec(),
                partitions,
            });
        }

        node.child(left).child(right)
    }

    fn ensure_single(&self, input: PhysicalPlanNode) -> PhysicalPlanNode {
        if !self.options.insert_exchanges
            || matches!(input.partitioning_ref(), Partitioning::Single)
        {
            return input;
        }
        PhysicalPlanNode::exchange(ExchangeKind::Gather)
            .attr("required_distribution", "single")
            .child(input)
            .partitioning(Partitioning::Single)
    }

    fn ensure_hash(
        &self,
        input: PhysicalPlanNode,
        keys: &[String],
        partitions: usize,
    ) -> PhysicalPlanNode {
        if !self.options.insert_exchanges {
            return input;
        }
        if matches!(
            input.partitioning_ref(),
            Partitioning::Hash {
                keys: existing_keys,
                partitions: existing_partitions,
            } if existing_keys == keys && *existing_partitions == partitions
        ) {
            return input;
        }
        PhysicalPlanNode::exchange(ExchangeKind::HashRepartition {
            keys: keys.to_vec(),
            partitions,
        })
        .attr("required_distribution", format!("hash[{}]", keys.join(",")))
        .child(input)
        .partitioning(Partitioning::Hash {
            keys: keys.to_vec(),
            partitions,
        })
    }
}

#[derive(Debug, Clone, Default)]
pub struct StagePlanner {
    next_stage_id: usize,
    stages: Vec<StagePlan>,
}

impl StagePlanner {
    pub fn split(root: PhysicalPlanNode) -> Vec<StagePlan> {
        let mut planner = Self::default();
        let (root, input_stages) = planner.rewrite_exchange_inputs(root);
        let root_stage = planner.push_stage(root, input_stages);
        debug_assert_eq!(root_stage, planner.stages.len() - 1);
        planner.stages
    }

    pub fn plan_execution_graph(root: PhysicalPlanNode) -> ExecutionGraphPlan {
        let stages = Self::split(root);
        let tasks = TaskPlanner::plan(&stages);
        ExecutionGraphPlan { stages, tasks }
    }

    fn rewrite_exchange_inputs(
        &mut self,
        node: PhysicalPlanNode,
    ) -> (PhysicalPlanNode, Vec<usize>) {
        let PhysicalPlanNode::Operator(mut operator_node) = node;
        if let PhysicalOperator::Exchange(kind) = operator_node.operator.clone() {
            let mut exchange_inputs = operator_node.children.into_iter();
            let Some(input) = exchange_inputs.next() else {
                return (
                    PhysicalPlanNode::Operator(PhysicalOperatorNode {
                        children: Vec::new(),
                        ..operator_node
                    }),
                    Vec::new(),
                );
            };
            let (stage_root, input_stages) = self.rewrite_exchange_inputs(input);
            let partitioning = exchange_partitioning(&kind);
            let stage_id =
                self.push_stage_with_partitioning(stage_root, input_stages, partitioning.clone());
            operator_node.children = Vec::new();
            let exchange = PhysicalPlanNode::Operator(operator_node)
                .attr("input_stage", stage_id)
                .partitioning(partitioning);
            return (exchange, vec![stage_id]);
        }

        let mut input_stages = Vec::new();
        let children = operator_node
            .children
            .into_iter()
            .map(|child| {
                let (child, child_input_stages) = self.rewrite_exchange_inputs(child);
                extend_unique(&mut input_stages, child_input_stages);
                child
            })
            .collect();
        operator_node.children = children;
        (PhysicalPlanNode::Operator(operator_node), input_stages)
    }

    fn push_stage(&mut self, root: PhysicalPlanNode, input_stages: Vec<usize>) -> usize {
        let partitioning = root.partitioning_ref().clone();
        self.push_stage_with_partitioning(root, input_stages, partitioning)
    }

    fn push_stage_with_partitioning(
        &mut self,
        root: PhysicalPlanNode,
        input_stages: Vec<usize>,
        partitioning: Partitioning,
    ) -> usize {
        let id = self.next_stage_id;
        self.next_stage_id += 1;
        self.stages.push(StagePlan {
            id,
            root,
            input_stages,
            partitioning,
        });
        id
    }
}

pub struct TaskPlanner;

impl TaskPlanner {
    pub fn plan(stages: &[StagePlan]) -> Vec<TaskPlan> {
        let mut tasks = Vec::new();
        for stage in stages {
            let scan_fragments = scan_fragments(&stage.root);
            if !scan_fragments.is_empty() && stage.input_stages.is_empty() {
                if matches!(stage.partitioning, Partitioning::Single) {
                    tasks.push(TaskPlan {
                        stage_id: stage.id,
                        partition: 0,
                        root: stage.root.clone(),
                        inputs: scan_fragments
                            .into_iter()
                            .map(TaskInput::ScanFragment)
                            .collect(),
                    });
                    continue;
                }
                tasks.extend(scan_fragments.into_iter().enumerate().map(
                    |(partition, fragment)| TaskPlan {
                        stage_id: stage.id,
                        partition,
                        root: stage.root.clone(),
                        inputs: vec![TaskInput::ScanFragment(fragment)],
                    },
                ));
                continue;
            }

            let partitions = partition_count(&stage.partitioning).max(1);
            for partition in 0..partitions {
                let mut inputs = Vec::new();
                for input_stage in &stage.input_stages {
                    let input_partitions = stages
                        .iter()
                        .find(|stage| stage.id == *input_stage)
                        .map(|stage| partition_count(&stage.partitioning).max(1))
                        .unwrap_or(1);
                    if partitions == 1 {
                        inputs.extend((0..input_partitions).map(|input_partition| {
                            TaskInput::ShufflePartition {
                                stage_id: *input_stage,
                                partition: input_partition,
                            }
                        }));
                    } else {
                        inputs.push(TaskInput::ShufflePartition {
                            stage_id: *input_stage,
                            partition: partition.min(input_partitions.saturating_sub(1)),
                        });
                    }
                }
                tasks.push(TaskPlan {
                    stage_id: stage.id,
                    partition,
                    root: stage.root.clone(),
                    inputs,
                });
            }
        }
        tasks
    }
}

impl PhysicalPlanNode {
    pub fn new(operator: impl Into<String>) -> Self {
        Self::Operator(PhysicalOperatorNode {
            operator: PhysicalOperator::from_name(operator.into()),
            attributes: Vec::new(),
            children: Vec::new(),
            partitioning: Partitioning::Unknown,
            ordering: OutputOrdering::default(),
            execution: None,
        })
    }

    pub fn exchange(kind: ExchangeKind) -> Self {
        Self::Operator(PhysicalOperatorNode {
            operator: PhysicalOperator::Exchange(kind),
            attributes: Vec::new(),
            children: Vec::new(),
            partitioning: Partitioning::Unknown,
            ordering: OutputOrdering::default(),
            execution: None,
        })
    }

    pub fn memory(batches: Vec<RecordBatch>) -> Self {
        Self::new("MemoryExec").execution(PhysicalExecutionConfig::Memory { batches })
    }

    pub fn ipc(files: Vec<std::path::PathBuf>) -> Self {
        Self::new("IpcExec").execution(PhysicalExecutionConfig::Ipc { files })
    }

    pub fn attr(mut self, key: impl Into<String>, value: impl ToString) -> Self {
        let Self::Operator(node) = &mut self;
        node.attributes.push((key.into(), value.to_string()));
        self
    }

    pub fn child(mut self, child: PhysicalPlanNode) -> Self {
        let Self::Operator(node) = &mut self;
        node.children.push(child);
        self
    }

    pub fn partitioning(mut self, partitioning: Partitioning) -> Self {
        let Self::Operator(node) = &mut self;
        node.partitioning = partitioning;
        self
    }

    pub fn ordering(mut self, ordering: OutputOrdering) -> Self {
        let Self::Operator(node) = &mut self;
        node.ordering = ordering;
        self
    }

    pub fn execution(mut self, execution: PhysicalExecutionConfig) -> Self {
        let Self::Operator(node) = &mut self;
        node.execution = Some(execution);
        self
    }

    pub fn operator(&self) -> &PhysicalOperator {
        let Self::Operator(node) = self;
        &node.operator
    }

    pub fn children(&self) -> &[PhysicalPlanNode] {
        let Self::Operator(node) = self;
        &node.children
    }

    pub fn partitioning_ref(&self) -> &Partitioning {
        let Self::Operator(node) = self;
        &node.partitioning
    }

    pub fn execution_config(&self) -> Option<&PhysicalExecutionConfig> {
        let Self::Operator(node) = self;
        node.execution.as_ref()
    }

    pub fn render_text(&self) -> String {
        let mut lines = Vec::new();
        self.render_text_into(0, &mut lines);
        lines.join("\n")
    }

    fn render_text_into(&self, indent: usize, lines: &mut Vec<String>) {
        let Self::Operator(node) = self;
        let mut line = format!("{}{}", "  ".repeat(indent), node.operator.name());
        for (key, value) in &node.attributes {
            line.push(' ');
            line.push_str(key);
            line.push('=');
            line.push_str(value);
        }
        lines.push(line);
        for child in &node.children {
            child.render_text_into(indent + 1, lines);
        }
    }
}

fn exchange_partitioning(kind: &ExchangeKind) -> Partitioning {
    match kind {
        ExchangeKind::HashRepartition { keys, partitions } => Partitioning::Hash {
            keys: keys.clone(),
            partitions: *partitions,
        },
        ExchangeKind::Broadcast => Partitioning::Unknown,
        ExchangeKind::Gather => Partitioning::Single,
    }
}

fn partition_count(partitioning: &Partitioning) -> usize {
    match partitioning {
        Partitioning::Unknown | Partitioning::Single => 1,
        Partitioning::Hash { partitions, .. }
        | Partitioning::RoundRobin { partitions }
        | Partitioning::FileRange { partitions } => *partitions,
    }
}

fn scan_fragments(node: &PhysicalPlanNode) -> Vec<FileFragment> {
    let mut fragments = Vec::new();
    collect_scan_fragments(node, &mut fragments);
    fragments
}

fn collect_scan_fragments(node: &PhysicalPlanNode, output: &mut Vec<FileFragment>) {
    if let Some(PhysicalExecutionConfig::Scan { fragments, .. }) = node.execution_config() {
        output.extend(fragments.iter().cloned());
    }
    for child in node.children() {
        collect_scan_fragments(child, output);
    }
}

fn extend_unique(target: &mut Vec<usize>, source: Vec<usize>) {
    for stage in source {
        if !target.contains(&stage) {
            target.push(stage);
        }
    }
}

impl PhysicalOperator {
    fn from_name(name: String) -> Self {
        match name.as_str() {
            "ScanExec" => Self::Scan,
            "MemoryExec" => Self::Memory,
            "IpcExec" => Self::Ipc,
            "FilterExec" => Self::Filter,
            "ProjectionExec" => Self::Projection,
            "AggregateExec" => Self::Aggregate,
            "LocalFoldExec" => Self::LocalFold,
            "FinalMergeExec" => Self::FinalMerge,
            "JoinExec" => Self::HashJoin,
            "PartitionedHashJoinExec" => Self::PartitionedHashJoin,
            "SortMergeJoinExec" => Self::SortMergeJoin,
            "SortExec" => Self::Sort,
            "LimitExec" => Self::Limit,
            "DistinctExec" => Self::Distinct,
            "CsvSinkExec" => Self::Sink(SinkKind::Csv),
            "MemorySinkExec" => Self::Sink(SinkKind::Memory),
            "EmptyExec" => Self::Empty,
            _ => Self::Other(name),
        }
    }

    fn name(&self) -> &str {
        match self {
            Self::Scan => "ScanExec",
            Self::Memory => "MemoryExec",
            Self::Ipc => "IpcExec",
            Self::Filter => "FilterExec",
            Self::Projection => "ProjectionExec",
            Self::Aggregate => "AggregateExec",
            Self::LocalFold => "LocalFoldExec",
            Self::FinalMerge => "FinalMergeExec",
            Self::HashJoin => "JoinExec",
            Self::PartitionedHashJoin => "PartitionedHashJoinExec",
            Self::SortMergeJoin => "SortMergeJoinExec",
            Self::Sort => "SortExec",
            Self::Limit => "LimitExec",
            Self::Distinct => "DistinctExec",
            Self::Exchange(ExchangeKind::HashRepartition { .. }) => "HashRepartitionExchangeExec",
            Self::Exchange(ExchangeKind::Broadcast) => "BroadcastExchangeExec",
            Self::Exchange(ExchangeKind::Gather) => "GatherExchangeExec",
            Self::Sink(SinkKind::Csv) => "CsvSinkExec",
            Self::Sink(SinkKind::Memory) => "MemorySinkExec",
            Self::Empty => "EmptyExec",
            Self::Other(name) => name,
        }
    }
}

fn projection_display(projection: &Projection) -> String {
    match projection {
        Projection::All => "*".to_string(),
        Projection::Columns(columns) => format!("[{}]", columns.join(",")),
    }
}

fn sort_key_display(sort: &SortKey) -> String {
    format!(
        "[{}]",
        sort.expressions
            .iter()
            .map(|expr| {
                if expr.descending {
                    format!("{} DESC", expr.column)
                } else {
                    format!("{} ASC", expr.column)
                }
            })
            .collect::<Vec<_>>()
            .join(",")
    )
}

fn aggregate_exprs_display(aggregates: &[AggregateExpr]) -> String {
    format!(
        "[{}]",
        aggregates
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(",")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_scan() -> LogicalPlan {
        LogicalPlan::TableScan(LogicalScan {
            source: PlanTableSource {
                fragments: Vec::new(),
                format: StorageFormat::Parquet,
                statistics: TableStatistics::default(),
            },
            batch_size: 1024,
            projection: Projection::All,
            filter: None,
            order_by: None,
            limit: None,
            distinct: false,
        })
    }

    fn scan_with_fragments(count: usize) -> LogicalPlan {
        LogicalPlan::TableScan(LogicalScan {
            source: PlanTableSource {
                fragments: (0..count)
                    .map(|index| FileFragment::local_parquet(format!("part-{index}.parquet")))
                    .collect(),
                format: StorageFormat::Parquet,
                statistics: TableStatistics::default(),
            },
            batch_size: 1024,
            projection: Projection::All,
            filter: None,
            order_by: None,
            limit: None,
            distinct: false,
        })
    }

    #[test]
    fn splits_physical_plan_into_exchange_stages() {
        let plan = PhysicalPlanNode::new("FinalMergeExec")
            .child(
                PhysicalPlanNode::new("LocalFoldExec").child(
                    PhysicalPlanNode::exchange(ExchangeKind::HashRepartition {
                        keys: vec!["k".to_string()],
                        partitions: 8,
                    })
                    .child(PhysicalPlanNode::new("ScanExec").attr("fragments", 4)),
                ),
            )
            .partitioning(Partitioning::Hash {
                keys: vec!["k".to_string()],
                partitions: 8,
            });

        let stages = StagePlanner::split(plan);
        assert_eq!(stages.len(), 2);
        assert_eq!(stages[0].id, 0);
        assert_eq!(stages[0].root.operator(), &PhysicalOperator::Scan);
        assert!(stages[0].input_stages.is_empty());
        assert_eq!(stages[1].id, 1);
        assert_eq!(stages[1].root.operator(), &PhysicalOperator::FinalMerge);
        assert_eq!(stages[1].input_stages, vec![0]);
        assert_eq!(
            stages[1].root.children()[0].operator(),
            &PhysicalOperator::LocalFold
        );
        assert_eq!(
            stages[1].root.children()[0].children()[0].partitioning_ref(),
            &Partitioning::Hash {
                keys: vec!["k".to_string()],
                partitions: 8,
            }
        );
        assert!(stages[1].root.render_text().contains("input_stage=0"));
    }

    #[test]
    fn planner_inserts_hash_exchanges_for_partitioned_join() {
        let logical = LogicalPlan::Join {
            left: Box::new(empty_scan()),
            right: Box::new(empty_scan()),
            join_type: JoinType::Inner,
            left_keys: vec!["k".to_string()],
            right_keys: vec!["id".to_string()],
            left_prefix: "l".to_string(),
            right_prefix: "r".to_string(),
            output_projection: Projection::All,
        };
        let physical = PhysicalPlanner::new(PhysicalPlanningOptions {
            default_join_strategy: PhysicalJoinStrategy::PartitionedHash {
                partitions: 8,
                memory_limit_bytes: 1024,
            },
            insert_exchanges: true,
            default_shuffle_partitions: 8,
        })
        .plan(&logical);

        assert_eq!(physical.operator(), &PhysicalOperator::PartitionedHashJoin);
        assert!(
            physical
                .render_text()
                .contains("HashRepartitionExchangeExec")
        );
        let stages = StagePlanner::split(physical);
        assert_eq!(stages.len(), 3);
        assert_eq!(stages[2].input_stages, vec![0, 1]);
    }

    #[test]
    fn task_planner_creates_scan_fragment_tasks() {
        let physical = scan_with_fragments(3).to_physical_plan();
        let graph = StagePlanner::plan_execution_graph(physical);

        assert_eq!(graph.stages.len(), 1);
        assert_eq!(graph.tasks.len(), 3);
        assert_eq!(graph.tasks[0].stage_id, 0);
        assert_eq!(graph.tasks[0].partition, 0);
        assert!(matches!(
            graph.tasks[0].inputs.as_slice(),
            [TaskInput::ScanFragment(_)]
        ));
    }

    #[test]
    fn task_planner_creates_shuffle_inputs_for_partitioned_join() {
        let logical = LogicalPlan::Join {
            left: Box::new(scan_with_fragments(2)),
            right: Box::new(scan_with_fragments(2)),
            join_type: JoinType::Inner,
            left_keys: vec!["k".to_string()],
            right_keys: vec!["id".to_string()],
            left_prefix: "l".to_string(),
            right_prefix: "r".to_string(),
            output_projection: Projection::All,
        };
        let physical = PhysicalPlanner::new(PhysicalPlanningOptions {
            default_join_strategy: PhysicalJoinStrategy::PartitionedHash {
                partitions: 4,
                memory_limit_bytes: 1024,
            },
            insert_exchanges: true,
            default_shuffle_partitions: 4,
        })
        .plan(&logical);
        let graph = StagePlanner::plan_execution_graph(physical);

        assert_eq!(graph.stages.len(), 3);
        assert_eq!(graph.tasks.len(), 8);
        assert_eq!(
            graph.tasks.iter().filter(|task| task.stage_id == 0).count(),
            2
        );
        assert_eq!(
            graph.tasks.iter().filter(|task| task.stage_id == 1).count(),
            2
        );
        let root_tasks = graph
            .tasks
            .iter()
            .filter(|task| task.stage_id == 2)
            .collect::<Vec<_>>();
        assert_eq!(root_tasks.len(), 4);
        assert!(root_tasks.iter().all(|task| {
            task.inputs
                .iter()
                .all(|input| matches!(input, TaskInput::ShufflePartition { .. }))
        }));
        assert_eq!(
            root_tasks[3].inputs,
            vec![
                TaskInput::ShufflePartition {
                    stage_id: 0,
                    partition: 3,
                },
                TaskInput::ShufflePartition {
                    stage_id: 1,
                    partition: 3,
                },
            ]
        );
    }

    #[test]
    fn planner_inserts_gather_exchange_for_global_limit() {
        let logical = LogicalPlan::Limit {
            input: Box::new(empty_scan()),
            limit: 10,
        };
        let physical = PhysicalPlanner::new(PhysicalPlanningOptions {
            insert_exchanges: true,
            ..PhysicalPlanningOptions::default()
        })
        .plan(&logical);

        assert_eq!(physical.operator(), &PhysicalOperator::Limit);
        assert_eq!(
            physical.children()[0].operator(),
            &PhysicalOperator::Exchange(ExchangeKind::Gather)
        );
        let stages = StagePlanner::split(physical);
        assert_eq!(stages.len(), 2);
        assert_eq!(stages[1].input_stages, vec![0]);
    }
}
