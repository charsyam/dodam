use std::collections::HashMap;

use crate::execution::{ComparisonExpr, Expr, FilterExpr, Projection, SortKey};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LogicalJoinTableStats {
    pub base_rows: u128,
    pub rows: u128,
    pub row_width: u128,
    pub key_ndv: HashMap<String, u128>,
}

impl LogicalJoinTableStats {
    pub fn filter_selectivity(&self) -> f64 {
        if self.base_rows == 0 {
            return 1.0;
        }
        (self.rows as f64 / self.base_rows as f64).clamp(0.0, 1.0)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LogicalJoinEdge {
    pub left: usize,
    pub left_key: String,
    pub right: usize,
    pub right_key: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LogicalJoinStep {
    pub table_index: usize,
    pub estimated_rows: u128,
    pub estimated_cost: u128,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LogicalJoinPlan {
    pub start: usize,
    pub steps: Vec<LogicalJoinStep>,
    pub estimated_cost: u128,
    pub bushy: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LogicalJoinGraph {
    pub tables: Vec<LogicalJoinTableStats>,
    pub edges: Vec<LogicalJoinEdge>,
}

impl LogicalJoinGraph {
    pub fn choose_best_plan(&self) -> Option<LogicalJoinPlan> {
        if self.tables.len() <= 6 {
            self.choose_exhaustive_left_deep_plan()
                .or_else(|| self.choose_greedy_plan())
        } else {
            self.choose_greedy_plan()
        }
    }

    pub fn choose_greedy_plan(&self) -> Option<LogicalJoinPlan> {
        (0..self.tables.len())
            .filter_map(|start| self.estimate_greedy_plan(start))
            .min_by_key(|plan| plan.estimated_cost)
    }

    pub fn choose_exhaustive_left_deep_plan(&self) -> Option<LogicalJoinPlan> {
        let table_count = self.tables.len();
        if table_count == 0 || table_count > 12 {
            return None;
        }
        let mut best = None;
        for start in 0..table_count {
            let mut joined = vec![false; table_count];
            joined[start] = true;
            let mut steps = Vec::new();
            self.enumerate_left_deep(
                &mut joined,
                self.tables[start].rows.max(1),
                self.tables[start].row_width.max(1),
                self.tables[start]
                    .rows
                    .max(1)
                    .saturating_mul(self.tables[start].row_width.max(1)),
                start,
                &mut steps,
                &mut best,
            );
        }
        best
    }

    pub fn choose_exhaustive_bushy_plan_cost(&self) -> Option<u128> {
        let table_count = self.tables.len();
        if table_count == 0 || table_count > 10 {
            return None;
        }
        let subset_count = 1usize.checked_shl(table_count as u32)?;
        let mut states = vec![None::<BushyJoinState>; subset_count];
        for index in 0..table_count {
            let mask = 1usize << index;
            let rows = self.tables[index].rows.max(1);
            let width = self.tables[index].row_width.max(1);
            states[mask] = Some(BushyJoinState {
                rows,
                width,
                cost: rows.saturating_mul(width),
            });
        }
        for mask in 1usize..subset_count {
            if mask.count_ones() <= 1 {
                continue;
            }
            let mut best = None;
            let mut left_mask = (mask - 1) & mask;
            while left_mask > 0 {
                let right_mask = mask ^ left_mask;
                if right_mask != 0 && left_mask < right_mask {
                    if let (Some(left), Some(right)) = (&states[left_mask], &states[right_mask])
                        && let Some(joined) =
                            self.estimate_bushy_join(left_mask, left, right_mask, right)
                        && best
                            .as_ref()
                            .is_none_or(|candidate: &BushyJoinState| joined.cost < candidate.cost)
                    {
                        best = Some(joined);
                    }
                }
                left_mask = (left_mask - 1) & mask;
            }
            states[mask] = best;
        }
        states[subset_count - 1].as_ref().map(|state| state.cost)
    }

    pub fn table_width(&self, table_index: usize) -> u128 {
        self.tables[table_index].row_width
    }

    pub fn choose_next_join(
        &self,
        joined: &[bool],
        current_rows: u128,
        current_width: u128,
        candidates: &[usize],
    ) -> Option<LogicalJoinStep> {
        candidates
            .iter()
            .copied()
            .filter(|table_index| !joined.get(*table_index).copied().unwrap_or(false))
            .filter_map(|table_index| {
                let edges = self.connected_edges(joined, table_index);
                if edges.is_empty() {
                    return None;
                }
                let estimated_rows = self.estimate_join_rows(current_rows, table_index, &edges);
                let output_width = current_width.saturating_add(self.tables[table_index].row_width);
                let build_cost = current_rows.saturating_mul(current_width).min(
                    self.tables[table_index]
                        .rows
                        .saturating_mul(self.tables[table_index].row_width),
                );
                let estimated_cost = estimated_rows
                    .saturating_mul(output_width)
                    .saturating_add(build_cost);
                Some(LogicalJoinStep {
                    table_index,
                    estimated_rows,
                    estimated_cost,
                })
            })
            .min_by_key(|step| step.estimated_cost)
    }

    fn estimate_greedy_plan(&self, start: usize) -> Option<LogicalJoinPlan> {
        if start >= self.tables.len() {
            return None;
        }
        let mut joined = vec![false; self.tables.len()];
        joined[start] = true;
        let mut joined_count = 1usize;
        let mut rows = self.tables[start].rows.max(1);
        let mut row_width = self.tables[start].row_width.max(1);
        let mut estimated_cost = rows.saturating_mul(row_width);
        let mut steps = Vec::new();

        while joined_count < self.tables.len() {
            let candidates = (0..self.tables.len()).collect::<Vec<_>>();
            let step = self.choose_next_join(&joined, rows, row_width, &candidates)?;
            joined[step.table_index] = true;
            joined_count += 1;
            rows = step.estimated_rows.max(1);
            row_width = row_width.saturating_add(self.tables[step.table_index].row_width);
            estimated_cost = estimated_cost.saturating_add(step.estimated_cost);
            steps.push(step);
        }

        Some(LogicalJoinPlan {
            start,
            steps,
            estimated_cost,
            bushy: false,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn enumerate_left_deep(
        &self,
        joined: &mut [bool],
        current_rows: u128,
        current_width: u128,
        current_cost: u128,
        start: usize,
        steps: &mut Vec<LogicalJoinStep>,
        best: &mut Option<LogicalJoinPlan>,
    ) {
        if joined.iter().all(|value| *value) {
            if best
                .as_ref()
                .is_none_or(|plan| current_cost < plan.estimated_cost)
            {
                *best = Some(LogicalJoinPlan {
                    start,
                    steps: steps.clone(),
                    estimated_cost: current_cost,
                    bushy: false,
                });
            }
            return;
        }
        if best
            .as_ref()
            .is_some_and(|plan| current_cost >= plan.estimated_cost)
        {
            return;
        }
        let candidates = (0..self.tables.len()).collect::<Vec<_>>();
        let mut next_steps = candidates
            .iter()
            .copied()
            .filter_map(|candidate| {
                self.choose_next_join(joined, current_rows, current_width, &[candidate])
            })
            .collect::<Vec<_>>();
        next_steps.sort_by_key(|step| step.estimated_cost);
        for step in next_steps {
            joined[step.table_index] = true;
            steps.push(step.clone());
            self.enumerate_left_deep(
                joined,
                step.estimated_rows.max(1),
                current_width.saturating_add(self.tables[step.table_index].row_width),
                current_cost.saturating_add(step.estimated_cost),
                start,
                steps,
                best,
            );
            steps.pop();
            joined[step.table_index] = false;
        }
    }

    fn connected_edges(&self, joined: &[bool], table_index: usize) -> Vec<&LogicalJoinEdge> {
        self.edges
            .iter()
            .filter(|edge| {
                (edge.left == table_index && joined.get(edge.right).copied().unwrap_or(false))
                    || (edge.right == table_index
                        && joined.get(edge.left).copied().unwrap_or(false))
            })
            .collect()
    }

    fn estimate_join_rows(
        &self,
        current_rows: u128,
        next_table: usize,
        edges: &[&LogicalJoinEdge],
    ) -> u128 {
        let next_rows = self.tables[next_table].rows.max(1);
        let denominator = edges
            .iter()
            .map(|edge| {
                let (left_ndv, right_ndv) = self.edge_ndv(edge);
                left_ndv.max(right_ndv).max(1)
            })
            .max()
            .unwrap_or(1);
        current_rows.saturating_mul(next_rows) / denominator
    }

    fn edge_ndv(&self, edge: &LogicalJoinEdge) -> (u128, u128) {
        (
            self.tables[edge.left]
                .key_ndv
                .get(&edge.left_key)
                .copied()
                .unwrap_or(self.tables[edge.left].rows)
                .min(self.tables[edge.left].rows)
                .max(1),
            self.tables[edge.right]
                .key_ndv
                .get(&edge.right_key)
                .copied()
                .unwrap_or(self.tables[edge.right].rows)
                .min(self.tables[edge.right].rows)
                .max(1),
        )
    }

    fn estimate_bushy_join(
        &self,
        left_mask: usize,
        left: &BushyJoinState,
        right_mask: usize,
        right: &BushyJoinState,
    ) -> Option<BushyJoinState> {
        let edges = self
            .edges
            .iter()
            .filter(|edge| {
                let left_has_left = (left_mask & (1usize << edge.left)) != 0;
                let left_has_right = (left_mask & (1usize << edge.right)) != 0;
                let right_has_left = (right_mask & (1usize << edge.left)) != 0;
                let right_has_right = (right_mask & (1usize << edge.right)) != 0;
                (left_has_left && right_has_right) || (left_has_right && right_has_left)
            })
            .collect::<Vec<_>>();
        if edges.is_empty() {
            return None;
        }
        let denominator = edges
            .iter()
            .map(|edge| {
                let (left_ndv, right_ndv) = self.edge_ndv(edge);
                left_ndv.max(right_ndv).max(1)
            })
            .max()
            .unwrap_or(1);
        let rows = left.rows.saturating_mul(right.rows) / denominator;
        let width = left.width.saturating_add(right.width);
        let build_cost = left
            .rows
            .saturating_mul(left.width)
            .min(right.rows.saturating_mul(right.width));
        let cost = left
            .cost
            .saturating_add(right.cost)
            .saturating_add(rows.saturating_mul(width))
            .saturating_add(build_cost);
        Some(BushyJoinState {
            rows: rows.max(1),
            width,
            cost,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BushyJoinState {
    rows: u128,
    width: u128,
    cost: u128,
}

#[derive(Debug, Clone, PartialEq)]
pub struct JoinInputPlan {
    pub left_projection: Projection,
    pub right_projection: Projection,
    pub left_filter: Option<FilterExpr>,
    pub right_filter: Option<FilterExpr>,
}

pub fn plan_join_inputs(
    projection: &Projection,
    filter: Option<&FilterExpr>,
    order_by: Option<&SortKey>,
    left_alias: &str,
    left_keys: &[String],
    right_alias: &str,
    right_keys: &[String],
) -> JoinInputPlan {
    let plan = JoinInputPlan {
        left_projection: join_side_projection(projection, filter, order_by, left_alias, left_keys),
        right_projection: join_side_projection(
            projection,
            filter,
            order_by,
            right_alias,
            right_keys,
        ),
        left_filter: filter.and_then(|filter| join_side_filter(filter, left_alias)),
        right_filter: filter.and_then(|filter| join_side_filter(filter, right_alias)),
    };
    log_join_input_plan(left_alias, right_alias, &plan);
    plan
}

fn log_join_input_plan(left_alias: &str, right_alias: &str, plan: &JoinInputPlan) {
    if !std::env::var("DODAM_OPTIMIZER_TRACE")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
    {
        return;
    }
    eprintln!(
        "[dodam:optimizer] rule=join_input_pushdown decision=plan left_alias={} right_alias={} left_projection={} right_projection={} left_filter={} right_filter={}",
        left_alias,
        right_alias,
        projection_display(&plan.left_projection),
        projection_display(&plan.right_projection),
        plan.left_filter.is_some(),
        plan.right_filter.is_some(),
    );
}

fn join_side_projection(
    projection: &Projection,
    filter: Option<&FilterExpr>,
    order_by: Option<&SortKey>,
    prefix: &str,
    join_keys: &[String],
) -> Projection {
    let Projection::Columns(columns) = projection else {
        return Projection::All;
    };

    let mut side_columns = Vec::new();
    for join_key in join_keys {
        add_column_once(&mut side_columns, join_key.to_string());
    }

    for column in columns {
        add_join_side_column(&mut side_columns, column, prefix);
    }
    if let Some(filter) = filter {
        for column in filter.referenced_columns() {
            add_join_side_column(&mut side_columns, &column, prefix);
        }
    }
    if let Some(order_by) = order_by {
        for sort in &order_by.expressions {
            add_join_side_column(&mut side_columns, &sort.column, prefix);
        }
    }

    Projection::Columns(side_columns)
}

fn add_join_side_column(columns: &mut Vec<String>, qualified_column: &str, prefix: &str) {
    if let Some(column) = strip_join_prefix(qualified_column, prefix) {
        add_column_once(columns, column.to_string());
    }
}

fn add_column_once(columns: &mut Vec<String>, column: String) {
    if !columns.iter().any(|existing| existing == &column) {
        columns.push(column);
    }
}

fn projection_display(projection: &Projection) -> String {
    match projection {
        Projection::All => "*".to_string(),
        Projection::Columns(columns) => format!("[{}]", columns.join(",")),
    }
}

fn join_side_filter(filter: &FilterExpr, prefix: &str) -> Option<FilterExpr> {
    derive_join_side_filter(filter.expr(), prefix).map(FilterExpr::new)
}

fn derive_join_side_filter(expr: &Expr, prefix: &str) -> Option<Expr> {
    let referenced = expr_referenced_columns(expr);
    if !referenced.is_empty()
        && referenced
            .iter()
            .all(|column| strip_join_prefix(column, prefix).is_some())
    {
        return Some(rewrite_join_side_expr(expr, prefix));
    }

    match expr {
        Expr::And(left, right) => combine_optional_filters(
            derive_join_side_filter(left, prefix),
            derive_join_side_filter(right, prefix),
            Expr::And,
        ),
        Expr::Or(left, right) => {
            let left = derive_join_side_filter(left, prefix)?;
            let right = derive_join_side_filter(right, prefix)?;
            Some(Expr::Or(Box::new(left), Box::new(right)))
        }
        _ => None,
    }
}

fn combine_optional_filters(
    left: Option<Expr>,
    right: Option<Expr>,
    combine: impl FnOnce(Box<Expr>, Box<Expr>) -> Expr,
) -> Option<Expr> {
    match (left, right) {
        (None, None) => None,
        (Some(expr), None) | (None, Some(expr)) => Some(expr),
        (Some(left), Some(right)) => Some(combine(Box::new(left), Box::new(right))),
    }
}

fn rewrite_join_side_expr(expr: &Expr, prefix: &str) -> Expr {
    match expr {
        Expr::Boolean(value) => Expr::Boolean(*value),
        Expr::Comparison(comparison) => Expr::Comparison(ComparisonExpr {
            column: strip_join_prefix(&comparison.column, prefix)
                .unwrap_or(&comparison.column)
                .to_string(),
            op: comparison.op,
            value: comparison.value.clone(),
        }),
        Expr::ColumnComparison { left, op, right } => Expr::ColumnComparison {
            left: strip_join_prefix(left, prefix).unwrap_or(left).to_string(),
            op: *op,
            right: strip_join_prefix(right, prefix)
                .unwrap_or(right)
                .to_string(),
        },
        Expr::InList {
            column,
            values,
            negated,
            has_null,
        } => Expr::InList {
            column: strip_join_prefix(column, prefix)
                .unwrap_or(column)
                .to_string(),
            values: values.clone(),
            negated: *negated,
            has_null: *has_null,
        },
        Expr::Like {
            column,
            pattern,
            negated,
            escape,
            case_insensitive,
        } => Expr::Like {
            column: strip_join_prefix(column, prefix)
                .unwrap_or(column)
                .to_string(),
            pattern: pattern.clone(),
            negated: *negated,
            escape: *escape,
            case_insensitive: *case_insensitive,
        },
        Expr::IsNull { column, negated } => Expr::IsNull {
            column: strip_join_prefix(column, prefix)
                .unwrap_or(column)
                .to_string(),
            negated: *negated,
        },
        Expr::Not(expr) => Expr::Not(Box::new(rewrite_join_side_expr(expr, prefix))),
        Expr::And(left, right) => Expr::And(
            Box::new(rewrite_join_side_expr(left, prefix)),
            Box::new(rewrite_join_side_expr(right, prefix)),
        ),
        Expr::Or(left, right) => Expr::Or(
            Box::new(rewrite_join_side_expr(left, prefix)),
            Box::new(rewrite_join_side_expr(right, prefix)),
        ),
    }
}

fn expr_referenced_columns(expr: &Expr) -> Vec<String> {
    let mut columns = Vec::new();
    collect_expr_referenced_columns(expr, &mut columns);
    columns
}

fn collect_expr_referenced_columns(expr: &Expr, columns: &mut Vec<String>) {
    match expr {
        Expr::Boolean(_) => {}
        Expr::Comparison(comparison) => add_column_once(columns, comparison.column.clone()),
        Expr::ColumnComparison { left, right, .. } => {
            add_column_once(columns, left.clone());
            add_column_once(columns, right.clone());
        }
        Expr::InList { column, .. } | Expr::Like { column, .. } | Expr::IsNull { column, .. } => {
            add_column_once(columns, column.clone());
        }
        Expr::Not(expr) => collect_expr_referenced_columns(expr, columns),
        Expr::And(left, right) | Expr::Or(left, right) => {
            collect_expr_referenced_columns(left, columns);
            collect_expr_referenced_columns(right, columns);
        }
    }
}

fn strip_join_prefix<'a>(column: &'a str, prefix: &str) -> Option<&'a str> {
    column.strip_prefix(prefix)?.strip_prefix('.')
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::execution::{
        ComparisonExpr, ComparisonOp, Expr, FilterExpr, LiteralValue, Projection, SortExpr, SortKey,
    };

    use super::{LogicalJoinEdge, LogicalJoinGraph, LogicalJoinTableStats, plan_join_inputs};

    fn table_stats(rows: u128, row_width: u128, keys: &[(&str, u128)]) -> LogicalJoinTableStats {
        LogicalJoinTableStats {
            base_rows: rows,
            rows,
            row_width,
            key_ndv: keys
                .iter()
                .map(|(key, ndv)| ((*key).to_string(), *ndv))
                .collect::<HashMap<_, _>>(),
        }
    }

    #[test]
    fn logical_join_graph_chooses_low_cost_start_and_next_steps() {
        let graph = LogicalJoinGraph {
            tables: vec![
                table_stats(1_000_000, 64, &[("customer_id", 100_000)]),
                table_stats(100_000, 32, &[("id", 100_000), ("region_id", 5)]),
                table_stats(5, 16, &[("id", 5)]),
            ],
            edges: vec![
                LogicalJoinEdge {
                    left: 0,
                    left_key: "customer_id".to_string(),
                    right: 1,
                    right_key: "id".to_string(),
                },
                LogicalJoinEdge {
                    left: 1,
                    left_key: "region_id".to_string(),
                    right: 2,
                    right_key: "id".to_string(),
                },
            ],
        };

        let plan = graph.choose_greedy_plan().expect("connected join graph");
        assert_eq!(plan.start, 2);
        assert_eq!(plan.steps[0].table_index, 1);
        assert_eq!(plan.steps[1].table_index, 0);
    }

    #[test]
    fn logical_join_graph_exhaustive_left_deep_matches_best_order() {
        let graph = LogicalJoinGraph {
            tables: vec![
                table_stats(10_000, 64, &[("k", 10_000)]),
                table_stats(100, 16, &[("k", 100), ("d", 10)]),
                table_stats(10, 16, &[("d", 10)]),
                table_stats(1_000, 32, &[("k", 1_000)]),
            ],
            edges: vec![
                LogicalJoinEdge {
                    left: 0,
                    left_key: "k".to_string(),
                    right: 1,
                    right_key: "k".to_string(),
                },
                LogicalJoinEdge {
                    left: 1,
                    left_key: "d".to_string(),
                    right: 2,
                    right_key: "d".to_string(),
                },
                LogicalJoinEdge {
                    left: 0,
                    left_key: "k".to_string(),
                    right: 3,
                    right_key: "k".to_string(),
                },
            ],
        };

        let plan = graph
            .choose_exhaustive_left_deep_plan()
            .expect("connected join graph");
        let greedy = graph.choose_greedy_plan().expect("connected join graph");
        assert_eq!(plan.steps.len(), 3);
        assert!(plan.estimated_cost <= greedy.estimated_cost);
    }

    #[test]
    fn logical_join_graph_computes_bushy_plan_cost() {
        let graph = LogicalJoinGraph {
            tables: vec![
                table_stats(1_000, 32, &[("a", 100)]),
                table_stats(1_000, 32, &[("a", 100)]),
                table_stats(10, 16, &[("b", 10)]),
                table_stats(10, 16, &[("b", 10)]),
            ],
            edges: vec![
                LogicalJoinEdge {
                    left: 0,
                    left_key: "a".to_string(),
                    right: 1,
                    right_key: "a".to_string(),
                },
                LogicalJoinEdge {
                    left: 2,
                    left_key: "b".to_string(),
                    right: 3,
                    right_key: "b".to_string(),
                },
                LogicalJoinEdge {
                    left: 1,
                    left_key: "a".to_string(),
                    right: 2,
                    right_key: "b".to_string(),
                },
            ],
        };

        assert!(graph.choose_exhaustive_bushy_plan_cost().is_some());
    }

    #[test]
    fn logical_join_table_stats_tracks_filter_selectivity() {
        let stats = LogicalJoinTableStats {
            base_rows: 1_000,
            rows: 125,
            row_width: 8,
            key_ndv: HashMap::new(),
        };
        assert_eq!(stats.filter_selectivity(), 0.125);
    }

    #[test]
    fn join_input_plan_keeps_filter_and_order_columns_for_pruning() {
        let filter = FilterExpr::new(Expr::Comparison(ComparisonExpr {
            column: "c.name".to_string(),
            op: ComparisonOp::Eq,
            value: LiteralValue::Utf8("alice".to_string()),
        }));
        let order_by = SortKey::from(SortExpr {
            column: "o.customer_id".to_string(),
            descending: false,
            nulls_first: false,
        });
        let left_keys = vec!["customer_id".to_string()];
        let right_keys = vec!["id".to_string()];

        let plan = plan_join_inputs(
            &Projection::Columns(vec!["o.id".to_string()]),
            Some(&filter),
            Some(&order_by),
            "o",
            &left_keys,
            "c",
            &right_keys,
        );

        assert_eq!(
            plan.left_projection,
            Projection::Columns(vec!["customer_id".to_string(), "id".to_string()])
        );
        assert_eq!(
            plan.right_projection,
            Projection::Columns(vec!["id".to_string(), "name".to_string()])
        );
        assert_eq!(plan.left_filter, None);
        assert_eq!(
            plan.right_filter,
            Some(FilterExpr::new(Expr::Comparison(ComparisonExpr {
                column: "name".to_string(),
                op: ComparisonOp::Eq,
                value: LiteralValue::Utf8("alice".to_string()),
            })))
        );
    }

    #[test]
    fn join_input_plan_does_not_push_mixed_side_or_filter() {
        let filter = FilterExpr::new(Expr::Or(
            Box::new(Expr::Comparison(ComparisonExpr {
                column: "o.id".to_string(),
                op: ComparisonOp::Gt,
                value: LiteralValue::Int64(10),
            })),
            Box::new(Expr::Comparison(ComparisonExpr {
                column: "c.name".to_string(),
                op: ComparisonOp::Eq,
                value: LiteralValue::Utf8("alice".to_string()),
            })),
        ));

        let left_keys = vec!["customer_id".to_string()];
        let right_keys = vec!["id".to_string()];
        let plan = plan_join_inputs(
            &Projection::Columns(vec!["o.id".to_string(), "c.name".to_string()]),
            Some(&filter),
            None,
            "o",
            &left_keys,
            "c",
            &right_keys,
        );

        assert_eq!(plan.left_filter, None);
        assert_eq!(plan.right_filter, None);
    }

    #[test]
    fn join_input_plan_pushes_same_side_or_filter() {
        let filter = FilterExpr::new(Expr::Or(
            Box::new(Expr::Comparison(ComparisonExpr {
                column: "c.name".to_string(),
                op: ComparisonOp::Eq,
                value: LiteralValue::Utf8("alice".to_string()),
            })),
            Box::new(Expr::Comparison(ComparisonExpr {
                column: "c.name".to_string(),
                op: ComparisonOp::Eq,
                value: LiteralValue::Utf8("bob".to_string()),
            })),
        ));

        let left_keys = vec!["customer_id".to_string()];
        let right_keys = vec!["id".to_string()];
        let plan = plan_join_inputs(
            &Projection::Columns(vec!["o.id".to_string()]),
            Some(&filter),
            None,
            "o",
            &left_keys,
            "c",
            &right_keys,
        );

        assert_eq!(plan.left_filter, None);
        assert!(plan.right_filter.is_some());
    }

    #[test]
    fn join_input_plan_derives_side_filters_from_or_branches() {
        let filter = FilterExpr::new(Expr::Or(
            Box::new(Expr::And(
                Box::new(Expr::Comparison(ComparisonExpr {
                    column: "o.status".to_string(),
                    op: ComparisonOp::Eq,
                    value: LiteralValue::Utf8("open".to_string()),
                })),
                Box::new(Expr::Comparison(ComparisonExpr {
                    column: "c.region".to_string(),
                    op: ComparisonOp::Eq,
                    value: LiteralValue::Utf8("EU".to_string()),
                })),
            )),
            Box::new(Expr::And(
                Box::new(Expr::Comparison(ComparisonExpr {
                    column: "o.status".to_string(),
                    op: ComparisonOp::Eq,
                    value: LiteralValue::Utf8("hold".to_string()),
                })),
                Box::new(Expr::Comparison(ComparisonExpr {
                    column: "c.region".to_string(),
                    op: ComparisonOp::Eq,
                    value: LiteralValue::Utf8("ASIA".to_string()),
                })),
            )),
        ));

        let left_keys = vec!["customer_id".to_string()];
        let right_keys = vec!["id".to_string()];
        let plan = plan_join_inputs(
            &Projection::Columns(vec!["o.id".to_string(), "c.name".to_string()]),
            Some(&filter),
            None,
            "o",
            &left_keys,
            "c",
            &right_keys,
        );

        assert_eq!(
            plan.left_filter,
            Some(FilterExpr::new(Expr::Or(
                Box::new(Expr::Comparison(ComparisonExpr {
                    column: "status".to_string(),
                    op: ComparisonOp::Eq,
                    value: LiteralValue::Utf8("open".to_string()),
                })),
                Box::new(Expr::Comparison(ComparisonExpr {
                    column: "status".to_string(),
                    op: ComparisonOp::Eq,
                    value: LiteralValue::Utf8("hold".to_string()),
                })),
            )))
        );
        assert_eq!(
            plan.right_filter,
            Some(FilterExpr::new(Expr::Or(
                Box::new(Expr::Comparison(ComparisonExpr {
                    column: "region".to_string(),
                    op: ComparisonOp::Eq,
                    value: LiteralValue::Utf8("EU".to_string()),
                })),
                Box::new(Expr::Comparison(ComparisonExpr {
                    column: "region".to_string(),
                    op: ComparisonOp::Eq,
                    value: LiteralValue::Utf8("ASIA".to_string()),
                })),
            )))
        );
    }
}
