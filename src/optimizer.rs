use crate::execution::{ComparisonExpr, Expr, FilterExpr, Projection, SortKey};

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
    JoinInputPlan {
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
    }
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

fn join_side_filter(filter: &FilterExpr, prefix: &str) -> Option<FilterExpr> {
    let conjuncts = join_side_conjuncts(filter.expr(), prefix);
    combine_filters(
        conjuncts
            .into_iter()
            .map(|expr| rewrite_join_side_expr(&expr, prefix))
            .collect(),
    )
    .map(FilterExpr::new)
}

fn join_side_conjuncts(expr: &Expr, prefix: &str) -> Vec<Expr> {
    match expr {
        Expr::And(left, right) => {
            let mut conjuncts = join_side_conjuncts(left, prefix);
            conjuncts.extend(join_side_conjuncts(right, prefix));
            conjuncts
        }
        expr => {
            let referenced = expr_referenced_columns(expr);
            if !referenced.is_empty()
                && referenced
                    .iter()
                    .all(|column| strip_join_prefix(column, prefix).is_some())
            {
                vec![expr.clone()]
            } else {
                Vec::new()
            }
        }
    }
}

fn combine_filters(mut filters: Vec<Expr>) -> Option<Expr> {
    let first = filters.pop()?;
    Some(filters.into_iter().fold(first, |right, left| {
        Expr::And(Box::new(left), Box::new(right))
    }))
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
        } => Expr::Like {
            column: strip_join_prefix(column, prefix)
                .unwrap_or(column)
                .to_string(),
            pattern: pattern.clone(),
            negated: *negated,
            escape: *escape,
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
    use crate::execution::{
        ComparisonExpr, ComparisonOp, Expr, FilterExpr, LiteralValue, Projection, SortExpr, SortKey,
    };

    use super::plan_join_inputs;

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
}
