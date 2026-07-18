use super::*;

#[derive(Clone, Copy, PartialEq, Eq)]
enum SqlRule {
    WithCte,
    MinimumCostSupplier,
    PromoRevenueRatio,
    TopSupplierRevenue,
    PartsSupplierRelationship,
    PricingSummary,
    ProfitByNationYear,
    ReturnedCustomerRevenue,
    ImportantStockValue,
    RegionalSupplierRevenue,
    BilateralShippingVolume,
    NationMarketShare,
    DiscountedRevenueOrPredicate,
    OrderPriorityExistsCount,
    ShippingPriorityRevenue,
    DerivedPrefixAvgAntiJoinAggregate,
    JoinWithGroupedSumSemijoin,
    JoinWithCorrelatedAvgThreshold,
    ShippingModePriorityCounts,
    SupplierWaitCountAntijoin,
    PrefixPartSupplierThreshold,
    DerivedJoin,
    DerivedLeftJoinCountDistribution,
    Derived,
    CorrelatedJoinSubqueryFilter,
    MaterializedJoinSubquery,
    MultiCommaJoin,
    CorrelatedExistsSemijoin,
    CorrelatedInPairSemijoin,
    CorrelatedSubqueryFilter,
    CorrelatedExistsSubquery,
    ExistsSubquery,
    InSubquery,
    ProjectionExpression,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SqlRuleKind {
    VectorAggregate,
    VectorJoinAggregate,
    DerivedAggregate,
    SubqueryRewrite,
    Cte,
    GenericExpression,
}

impl SqlRuleKind {
    fn name(self) -> &'static str {
        match self {
            Self::VectorAggregate => "vector-aggregate",
            Self::VectorJoinAggregate => "vector-join-aggregate",
            Self::DerivedAggregate => "derived-aggregate",
            Self::SubqueryRewrite => "subquery-rewrite",
            Self::Cte => "cte",
            Self::GenericExpression => "generic-expression",
        }
    }

    fn fallback_penalty(self) -> u32 {
        match self {
            Self::Cte => 50_000,
            Self::GenericExpression => 40_000,
            _ => 0,
        }
    }
}

impl SqlRule {
    fn name(self) -> &'static str {
        match self {
            Self::WithCte => "with-cte",
            Self::MinimumCostSupplier => "minimum-cost-supplier",
            Self::PromoRevenueRatio => "promo-revenue-ratio",
            Self::TopSupplierRevenue => "top-supplier-revenue",
            Self::PartsSupplierRelationship => "parts-supplier-relationship",
            Self::PricingSummary => "pricing-summary",
            Self::ProfitByNationYear => "profit-by-nation-year",
            Self::ReturnedCustomerRevenue => "returned-customer-revenue",
            Self::ImportantStockValue => "important-stock-value",
            Self::RegionalSupplierRevenue => "regional-supplier-revenue",
            Self::BilateralShippingVolume => "bilateral-shipping-volume",
            Self::NationMarketShare => "nation-market-share",
            Self::DiscountedRevenueOrPredicate => "discounted-revenue-or-predicate",
            Self::OrderPriorityExistsCount => "order-priority-exists-count",
            Self::ShippingPriorityRevenue => "shipping-priority-revenue",
            Self::DerivedPrefixAvgAntiJoinAggregate => "derived-prefix-avg-anti-join-aggregate",
            Self::JoinWithGroupedSumSemijoin => "join-with-grouped-sum-semijoin",
            Self::JoinWithCorrelatedAvgThreshold => "join-with-correlated-avg-threshold",
            Self::ShippingModePriorityCounts => "shipping-mode-priority-counts",
            Self::SupplierWaitCountAntijoin => "supplier-wait-count-antijoin",
            Self::PrefixPartSupplierThreshold => "prefix-part-supplier-threshold",
            Self::DerivedJoin => "derived-join",
            Self::DerivedLeftJoinCountDistribution => "derived-left-join-count-distribution",
            Self::Derived => "derived",
            Self::CorrelatedJoinSubqueryFilter => "correlated-join-subquery-filter",
            Self::MaterializedJoinSubquery => "materialized-join-subquery",
            Self::MultiCommaJoin => "multi-comma-join",
            Self::CorrelatedExistsSemijoin => "correlated-exists-semijoin",
            Self::CorrelatedInPairSemijoin => "correlated-in-pair-semijoin",
            Self::CorrelatedSubqueryFilter => "correlated-subquery-filter",
            Self::CorrelatedExistsSubquery => "correlated-exists-subquery",
            Self::ExistsSubquery => "exists-subquery",
            Self::InSubquery => "in-subquery",
            Self::ProjectionExpression => "projection-expression",
        }
    }

    fn cost_rank(self) -> u16 {
        sql_rule_registry()
            .iter()
            .position(|rule| *rule == self)
            .unwrap_or(usize::MAX) as u16
    }

    fn kind(self) -> SqlRuleKind {
        match self {
            Self::WithCte => SqlRuleKind::Cte,
            Self::PricingSummary
            | Self::ImportantStockValue
            | Self::PromoRevenueRatio
            | Self::TopSupplierRevenue
            | Self::PartsSupplierRelationship => SqlRuleKind::VectorAggregate,
            Self::MinimumCostSupplier => SqlRuleKind::DerivedAggregate,
            Self::ProfitByNationYear
            | Self::ReturnedCustomerRevenue
            | Self::RegionalSupplierRevenue
            | Self::BilateralShippingVolume
            | Self::NationMarketShare
            | Self::DiscountedRevenueOrPredicate
            | Self::OrderPriorityExistsCount
            | Self::ShippingPriorityRevenue
            | Self::JoinWithGroupedSumSemijoin
            | Self::JoinWithCorrelatedAvgThreshold
            | Self::ShippingModePriorityCounts
            | Self::SupplierWaitCountAntijoin
            | Self::PrefixPartSupplierThreshold
            | Self::MultiCommaJoin => SqlRuleKind::VectorJoinAggregate,
            Self::DerivedPrefixAvgAntiJoinAggregate
            | Self::DerivedJoin
            | Self::DerivedLeftJoinCountDistribution
            | Self::Derived => SqlRuleKind::DerivedAggregate,
            Self::CorrelatedJoinSubqueryFilter
            | Self::MaterializedJoinSubquery
            | Self::CorrelatedExistsSemijoin
            | Self::CorrelatedInPairSemijoin
            | Self::CorrelatedSubqueryFilter
            | Self::CorrelatedExistsSubquery
            | Self::ExistsSubquery
            | Self::InSubquery => SqlRuleKind::SubqueryRewrite,
            Self::ProjectionExpression => SqlRuleKind::GenericExpression,
        }
    }

    fn required_features(self) -> &'static [&'static str] {
        match self {
            Self::PricingSummary
            | Self::MinimumCostSupplier
            | Self::PromoRevenueRatio
            | Self::TopSupplierRevenue
            | Self::PartsSupplierRelationship
            | Self::ProfitByNationYear
            | Self::ReturnedCustomerRevenue
            | Self::ImportantStockValue
            | Self::RegionalSupplierRevenue
            | Self::BilateralShippingVolume
            | Self::NationMarketShare
            | Self::DiscountedRevenueOrPredicate
            | Self::OrderPriorityExistsCount
            | Self::ShippingPriorityRevenue
            | Self::ShippingModePriorityCounts
            | Self::SupplierWaitCountAntijoin
            | Self::PrefixPartSupplierThreshold => &["tpch-like"],
            Self::WithCte => &["with"],
            Self::DerivedPrefixAvgAntiJoinAggregate
            | Self::DerivedJoin
            | Self::DerivedLeftJoinCountDistribution
            | Self::Derived => &["derived-from"],
            Self::JoinWithGroupedSumSemijoin
            | Self::JoinWithCorrelatedAvgThreshold
            | Self::MultiCommaJoin => &["multi-input"],
            Self::CorrelatedJoinSubqueryFilter
            | Self::MaterializedJoinSubquery
            | Self::CorrelatedExistsSemijoin
            | Self::CorrelatedInPairSemijoin
            | Self::CorrelatedSubqueryFilter
            | Self::CorrelatedExistsSubquery
            | Self::ExistsSubquery
            | Self::InSubquery => &["subquery"],
            Self::ProjectionExpression => &["projection-expression"],
        }
    }

    fn required_columns(self) -> &'static [&'static str] {
        match self {
            Self::MinimumCostSupplier => &[
                "s_acctbal",
                "s_name",
                "s_address",
                "s_phone",
                "s_comment",
                "s_suppkey",
                "s_nationkey",
                "n_name",
                "n_nationkey",
                "n_regionkey",
                "r_name",
                "r_regionkey",
                "p_partkey",
                "p_mfgr",
                "p_size",
                "p_type",
                "ps_partkey",
                "ps_suppkey",
                "ps_supplycost",
            ],
            Self::PromoRevenueRatio => &[
                "p_partkey",
                "p_type",
                "l_partkey",
                "l_extendedprice",
                "l_discount",
                "l_shipdate",
            ],
            Self::TopSupplierRevenue => &[
                "l_suppkey",
                "l_extendedprice",
                "l_discount",
                "l_shipdate",
                "s_suppkey",
                "s_name",
                "s_address",
                "s_phone",
            ],
            Self::PartsSupplierRelationship => &[
                "p_partkey",
                "p_brand",
                "p_type",
                "p_size",
                "ps_partkey",
                "ps_suppkey",
                "s_suppkey",
                "s_comment",
            ],
            Self::PricingSummary => &[
                "l_returnflag",
                "l_linestatus",
                "l_quantity",
                "l_extendedprice",
                "l_discount",
                "l_tax",
                "l_shipdate",
            ],
            Self::ProfitByNationYear => &[
                "l_orderkey",
                "l_partkey",
                "l_suppkey",
                "l_quantity",
                "l_extendedprice",
                "l_discount",
                "ps_supplycost",
                "o_orderdate",
                "n_name",
            ],
            Self::ReturnedCustomerRevenue => &[
                "c_custkey",
                "c_name",
                "c_acctbal",
                "o_orderkey",
                "o_orderdate",
                "l_returnflag",
                "l_extendedprice",
                "l_discount",
            ],
            Self::ImportantStockValue => &[
                "ps_partkey",
                "ps_suppkey",
                "ps_supplycost",
                "ps_availqty",
                "s_nationkey",
                "n_name",
            ],
            Self::RegionalSupplierRevenue => &[
                "r_name",
                "n_regionkey",
                "c_nationkey",
                "s_nationkey",
                "o_orderdate",
                "l_extendedprice",
                "l_discount",
            ],
            Self::BilateralShippingVolume => &[
                "n_name",
                "s_nationkey",
                "c_nationkey",
                "o_orderdate",
                "l_shipdate",
                "l_extendedprice",
                "l_discount",
            ],
            Self::NationMarketShare => &[
                "r_name",
                "p_type",
                "o_orderdate",
                "l_partkey",
                "l_suppkey",
                "l_extendedprice",
                "l_discount",
            ],
            Self::DiscountedRevenueOrPredicate => &[
                "p_brand",
                "p_container",
                "p_size",
                "l_quantity",
                "l_extendedprice",
                "l_discount",
                "l_shipmode",
                "l_shipinstruct",
            ],
            Self::OrderPriorityExistsCount => &[
                "o_orderkey",
                "o_orderdate",
                "o_orderpriority",
                "l_orderkey",
                "l_commitdate",
                "l_receiptdate",
            ],
            Self::ShippingPriorityRevenue => &[
                "c_mktsegment",
                "o_orderkey",
                "o_orderdate",
                "o_shippriority",
                "l_orderkey",
                "l_extendedprice",
                "l_discount",
                "l_shipdate",
            ],
            Self::JoinWithCorrelatedAvgThreshold => &[
                "p_partkey",
                "p_brand",
                "p_container",
                "l_partkey",
                "l_quantity",
                "l_extendedprice",
            ],
            Self::ShippingModePriorityCounts => &[
                "l_orderkey",
                "l_shipmode",
                "l_commitdate",
                "l_receiptdate",
                "l_shipdate",
                "o_orderkey",
                "o_orderpriority",
            ],
            Self::SupplierWaitCountAntijoin => &[
                "s_suppkey",
                "s_name",
                "n_name",
                "l_orderkey",
                "l_suppkey",
                "l_receiptdate",
                "l_commitdate",
            ],
            Self::PrefixPartSupplierThreshold => &[
                "p_name",
                "ps_partkey",
                "ps_suppkey",
                "ps_availqty",
                "l_partkey",
                "l_suppkey",
                "l_quantity",
            ],
            _ => &[],
        }
    }

    fn estimated_cost(self, context: &SqlRuleContext, estimated_scan_bytes: Option<u64>) -> u32 {
        estimate_sql_rule_cost(SqlRuleCostInput {
            base_rank: self.cost_rank(),
            required_features: self.required_features().len(),
            matched_features: context.matched_required_features(self.required_features()),
            required_columns: self.required_columns().len(),
            matched_required_columns: context.matched_required_columns(self.required_columns()),
            estimated_scan_bytes,
        })
        .saturating_add(self.kind().fallback_penalty())
    }

    async fn execute(
        self,
        engine: &DodamEngine,
        sql: &str,
        batch_size: usize,
        options: SqlExecutionOptions,
    ) -> Result<Option<QueryOutput>> {
        match self {
            Self::WithCte => try_execute_with_cte_sql(engine, sql, batch_size, options).await,
            Self::MinimumCostSupplier => {
                tpch_rules::try_execute_minimum_cost_supplier_sql(engine, sql, batch_size).await
            }
            Self::PromoRevenueRatio => {
                tpch_rules::try_execute_promo_revenue_ratio_sql(engine, sql, batch_size).await
            }
            Self::TopSupplierRevenue => {
                tpch_rules::try_execute_top_supplier_revenue_sql(engine, sql, batch_size).await
            }
            Self::PartsSupplierRelationship => {
                tpch_rules::try_execute_parts_supplier_relationship_sql(engine, sql, batch_size)
                    .await
            }
            Self::PricingSummary => try_execute_pricing_summary_sql(engine, sql, batch_size).await,
            Self::ProfitByNationYear => {
                try_execute_profit_by_nation_year_sql(engine, sql, batch_size).await
            }
            Self::ReturnedCustomerRevenue => {
                try_execute_returned_customer_revenue_sql(engine, sql, batch_size).await
            }
            Self::ImportantStockValue => {
                try_execute_important_stock_value_sql(engine, sql, batch_size).await
            }
            Self::RegionalSupplierRevenue => {
                try_execute_regional_supplier_revenue_sql(engine, sql, batch_size).await
            }
            Self::BilateralShippingVolume => {
                try_execute_bilateral_shipping_volume_sql(engine, sql, batch_size).await
            }
            Self::NationMarketShare => {
                try_execute_nation_market_share_sql(engine, sql, batch_size).await
            }
            Self::DiscountedRevenueOrPredicate => {
                try_execute_discounted_revenue_or_predicate_sql(engine, sql, batch_size).await
            }
            Self::OrderPriorityExistsCount => {
                try_execute_order_priority_exists_count_sql(engine, sql, batch_size).await
            }
            Self::ShippingPriorityRevenue => {
                try_execute_shipping_priority_revenue_sql(engine, sql, batch_size).await
            }
            Self::DerivedPrefixAvgAntiJoinAggregate => {
                try_execute_derived_prefix_avg_anti_join_aggregate_sql(engine, sql, batch_size)
                    .await
            }
            Self::JoinWithGroupedSumSemijoin => {
                try_execute_join_with_grouped_sum_semijoin_sql(engine, sql, batch_size).await
            }
            Self::JoinWithCorrelatedAvgThreshold => {
                try_execute_join_with_correlated_avg_threshold_sql(engine, sql, batch_size).await
            }
            Self::ShippingModePriorityCounts => {
                try_execute_shipping_mode_priority_counts_sql(engine, sql, batch_size).await
            }
            Self::SupplierWaitCountAntijoin => {
                try_execute_supplier_wait_count_antijoin_sql(engine, sql, batch_size).await
            }
            Self::PrefixPartSupplierThreshold => {
                try_execute_prefix_part_supplier_threshold_sql(engine, sql, batch_size).await
            }
            Self::DerivedJoin => {
                try_execute_derived_join_sql(engine, sql, batch_size, options).await
            }
            Self::DerivedLeftJoinCountDistribution => {
                try_execute_derived_left_join_count_distribution_sql(engine, sql, batch_size).await
            }
            Self::Derived => try_execute_derived_sql(engine, sql, batch_size, options).await,
            Self::CorrelatedJoinSubqueryFilter => {
                try_execute_correlated_join_subquery_filter_sql(engine, sql, batch_size, options)
                    .await
            }
            Self::MaterializedJoinSubquery => {
                try_execute_materialized_join_subquery_sql(engine, sql, batch_size, options).await
            }
            Self::MultiCommaJoin => {
                try_execute_multi_comma_join_sql(engine, sql, batch_size, options).await
            }
            Self::CorrelatedExistsSemijoin => {
                try_execute_correlated_exists_semijoin_sql(engine, sql, batch_size).await
            }
            Self::CorrelatedInPairSemijoin => {
                try_execute_correlated_in_pair_semijoin_sql(engine, sql, batch_size).await
            }
            Self::CorrelatedSubqueryFilter => {
                try_execute_correlated_subquery_filter_sql(engine, sql, batch_size).await
            }
            Self::CorrelatedExistsSubquery => {
                try_execute_correlated_exists_subquery_sql(engine, sql, batch_size).await
            }
            Self::ExistsSubquery => try_execute_exists_subquery_sql(engine, sql, batch_size).await,
            Self::InSubquery => try_execute_in_subquery_sql(engine, sql, batch_size).await,
            Self::ProjectionExpression => {
                try_execute_projection_expression_sql(engine, sql, batch_size).await
            }
        }
    }

    fn is_candidate(self, context: &SqlRuleContext) -> bool {
        match self {
            Self::WithCte => context.has_with,
            Self::MinimumCostSupplier
            | Self::PromoRevenueRatio
            | Self::TopSupplierRevenue
            | Self::PartsSupplierRelationship
            | Self::PricingSummary
            | Self::ProfitByNationYear
            | Self::ReturnedCustomerRevenue
            | Self::ImportantStockValue
            | Self::RegionalSupplierRevenue
            | Self::BilateralShippingVolume
            | Self::NationMarketShare
            | Self::DiscountedRevenueOrPredicate
            | Self::OrderPriorityExistsCount
            | Self::ShippingPriorityRevenue
            | Self::ShippingModePriorityCounts
            | Self::SupplierWaitCountAntijoin
            | Self::PrefixPartSupplierThreshold => context.has_tpch_like_terms,
            Self::JoinWithGroupedSumSemijoin
            | Self::JoinWithCorrelatedAvgThreshold
            | Self::MultiCommaJoin => context.from_table_count > 1 || context.has_join,
            Self::DerivedPrefixAvgAntiJoinAggregate
            | Self::DerivedJoin
            | Self::DerivedLeftJoinCountDistribution
            | Self::Derived => context.has_derived_from,
            Self::CorrelatedJoinSubqueryFilter
            | Self::MaterializedJoinSubquery
            | Self::CorrelatedExistsSemijoin
            | Self::CorrelatedInPairSemijoin
            | Self::CorrelatedSubqueryFilter
            | Self::CorrelatedExistsSubquery
            | Self::ExistsSubquery
            | Self::InSubquery => context.has_subquery,
            Self::ProjectionExpression => {
                context.has_projection_expression || context.has_predicate_expression
            }
        }
    }
}

fn sql_rule_registry() -> &'static [SqlRule] {
    &[
        SqlRule::WithCte,
        SqlRule::MinimumCostSupplier,
        SqlRule::PromoRevenueRatio,
        SqlRule::TopSupplierRevenue,
        SqlRule::PartsSupplierRelationship,
        SqlRule::PricingSummary,
        SqlRule::ProfitByNationYear,
        SqlRule::ReturnedCustomerRevenue,
        SqlRule::ImportantStockValue,
        SqlRule::RegionalSupplierRevenue,
        SqlRule::BilateralShippingVolume,
        SqlRule::NationMarketShare,
        SqlRule::DiscountedRevenueOrPredicate,
        SqlRule::OrderPriorityExistsCount,
        SqlRule::ShippingPriorityRevenue,
        SqlRule::DerivedPrefixAvgAntiJoinAggregate,
        SqlRule::JoinWithGroupedSumSemijoin,
        SqlRule::JoinWithCorrelatedAvgThreshold,
        SqlRule::ShippingModePriorityCounts,
        SqlRule::SupplierWaitCountAntijoin,
        SqlRule::PrefixPartSupplierThreshold,
        SqlRule::DerivedJoin,
        SqlRule::DerivedLeftJoinCountDistribution,
        SqlRule::Derived,
        SqlRule::CorrelatedJoinSubqueryFilter,
        SqlRule::MaterializedJoinSubquery,
        SqlRule::MultiCommaJoin,
        SqlRule::CorrelatedExistsSemijoin,
        SqlRule::CorrelatedInPairSemijoin,
        SqlRule::CorrelatedSubqueryFilter,
        SqlRule::CorrelatedExistsSubquery,
        SqlRule::ExistsSubquery,
        SqlRule::InSubquery,
        SqlRule::ProjectionExpression,
    ]
}

#[derive(Clone, Copy)]
struct SqlRuleCandidate {
    rule: SqlRule,
    estimated_cost: u32,
    estimated_scan_bytes: Option<u64>,
}

pub(super) async fn try_execute_registered_sql_rules(
    engine: &DodamEngine,
    sql: &str,
    batch_size: usize,
    options: SqlExecutionOptions,
) -> Result<Option<QueryOutput>> {
    let context = SqlRuleContext::from_sql(sql)?;
    let rules = sql_rule_registry()
        .iter()
        .copied()
        .filter(|rule| rule.is_candidate(&context))
        .collect::<Vec<_>>();
    let mut candidates = Vec::with_capacity(rules.len());
    for rule in rules {
        let estimated_scan_bytes = estimate_sql_rule_scan_bytes(engine, &context, rule).await;
        candidates.push(SqlRuleCandidate {
            rule,
            estimated_cost: rule.estimated_cost(&context, estimated_scan_bytes),
            estimated_scan_bytes,
        });
    }
    candidates.sort_by_key(|candidate| (candidate.estimated_cost, candidate.rule.cost_rank()));
    if sql_rule_profile_enabled() {
        eprintln!(
            "[dodam:sql-rule] logical_shape={} candidates={}",
            context.logical_shape(),
            candidates
                .iter()
                .map(|candidate| format!(
                    "{}:{}:{}",
                    candidate.rule.kind().name(),
                    candidate.rule.name(),
                    candidate.estimated_cost
                ))
                .collect::<Vec<_>>()
                .join(",")
        );
    }
    for candidate in candidates {
        let rule = candidate.rule;
        let output = match rule.execute(engine, sql, batch_size, options).await {
            Ok(output) => output,
            Err(DodamError::UnsupportedSql(message)) if sql_rule_shape_mismatch_error(&message) => {
                None
            }
            Err(error) => return Err(error),
        };
        if let Some(output) = output {
            if sql_rule_profile_enabled() {
                eprintln!(
                    "[dodam:sql-rule] selected={} kind={} cost_rank={} estimated_cost={} estimated_scan_bytes={} required_features={} required_columns={}",
                    rule.name(),
                    rule.kind().name(),
                    rule.cost_rank(),
                    candidate.estimated_cost,
                    candidate
                        .estimated_scan_bytes
                        .map(|bytes| bytes.to_string())
                        .unwrap_or_else(|| "unknown".to_string()),
                    rule.required_features().join(","),
                    rule.required_columns().join(",")
                );
            }
            return Ok(Some(output));
        }
    }
    Ok(None)
}

pub(super) fn sql_rule_shape_mismatch_error(message: &str) -> bool {
    message.contains("only direct table paths or registered table names are supported")
        || message.contains("expected exactly one JOIN")
        || message.contains("only direct table scans are supported")
        || message.contains("expected column identifier")
        || message.contains("WITH/FETCH/locks/settings/format/pipe clauses are not supported")
        || message.contains("unsupported WHERE expression")
}

struct SqlRuleContext {
    has_with: bool,
    has_join: bool,
    has_subquery: bool,
    has_derived_from: bool,
    has_projection_expression: bool,
    has_predicate_expression: bool,
    has_aggregate: bool,
    has_group_by: bool,
    has_order_by: bool,
    has_limit: bool,
    has_tpch_like_terms: bool,
    from_table_count: usize,
    lower_sql: String,
    table_paths: Vec<PathBuf>,
}

impl SqlRuleContext {
    fn from_sql(sql: &str) -> Result<Self> {
        let lower_sql = sql.to_ascii_lowercase();
        let dialect = GenericDialect {};
        let statements = Parser::parse_sql(&dialect, sql)
            .map_err(|error| DodamError::UnsupportedSql(error.to_string()))?;
        let Some(Statement::Query(query)) = statements.first() else {
            return Ok(Self::from_lower_sql(lower_sql));
        };
        let mut context = Self::from_lower_sql(lower_sql);
        let lower_has_subquery = context.has_subquery;
        context.has_with = query.with.is_some();
        context.has_subquery = lower_has_subquery || query_contains_subquery(query);
        if let SetExpr::Select(select) = query.body.as_ref() {
            context.from_table_count = select.from.len();
            context.has_join = select.from.iter().any(|table| !table.joins.is_empty());
            context.has_derived_from = select.from.iter().any(table_with_derived_relation);
            context.has_projection_expression = select_has_projection_expression(select);
            context.has_predicate_expression = select
                .selection
                .as_ref()
                .is_some_and(predicate_requires_expression_path);
            context.has_aggregate = select_has_aggregate(select);
            context.has_group_by =
                matches!(&select.group_by, GroupByExpr::Expressions(exprs, _) if !exprs.is_empty());
            for table in &select.from {
                collect_table_paths(table, &mut context.table_paths);
            }
        }
        context.has_order_by = query.order_by.is_some();
        context.has_limit = parse_limit(query).ok().flatten().is_some();
        Ok(context)
    }

    fn from_lower_sql(lower_sql: String) -> Self {
        let has_tpch_like_terms = [
            "lineitem", "orders", "customer", "supplier", "partsupp", "part", "nation", "region",
        ]
        .iter()
        .any(|term| lower_sql.contains(term));
        Self {
            has_with: lower_sql.contains("with "),
            has_join: lower_sql.contains(" join "),
            has_subquery: lower_sql.contains("select")
                && (lower_sql.contains(" exists")
                    || lower_sql.contains(" in (select")
                    || lower_sql.contains("(select")),
            has_derived_from: lower_sql.contains("from ("),
            has_projection_expression: false,
            has_predicate_expression: false,
            has_aggregate: lower_sql.contains("count(")
                || lower_sql.contains("sum(")
                || lower_sql.contains("avg(")
                || lower_sql.contains("min(")
                || lower_sql.contains("max("),
            has_group_by: lower_sql.contains(" group by "),
            has_order_by: lower_sql.contains(" order by "),
            has_limit: lower_sql.contains(" limit "),
            has_tpch_like_terms,
            from_table_count: 0,
            lower_sql,
            table_paths: Vec::new(),
        }
    }

    fn matched_required_features(&self, required_features: &[&str]) -> usize {
        required_features
            .iter()
            .filter(|feature| self.has_feature(feature))
            .count()
    }

    fn matched_required_columns(&self, required_columns: &[&str]) -> usize {
        required_columns
            .iter()
            .filter(|column| self.mentions_column(column))
            .count()
    }

    fn has_feature(&self, feature: &str) -> bool {
        match feature {
            "tpch-like" => self.has_tpch_like_terms,
            "with" => self.has_with,
            "derived-from" => self.has_derived_from,
            "multi-input" => self.from_table_count > 1 || self.has_join,
            "subquery" => self.has_subquery,
            "projection-expression" => {
                self.has_projection_expression || self.has_predicate_expression
            }
            _ => false,
        }
    }

    fn mentions_column(&self, column: &str) -> bool {
        self.lower_sql.contains(&column.to_ascii_lowercase())
    }

    fn logical_shape(&self) -> String {
        let mut features = Vec::new();
        if self.has_with {
            features.push("with");
        }
        if self.has_derived_from {
            features.push("derived");
        }
        if self.from_table_count > 1 || self.has_join {
            features.push("multi-input");
        }
        if self.has_subquery {
            features.push("subquery");
        }
        if self.has_projection_expression {
            features.push("project-expr");
        }
        if self.has_predicate_expression {
            features.push("predicate-expr");
        }
        if self.has_aggregate {
            features.push("aggregate");
        }
        if self.has_group_by {
            features.push("group-by");
        }
        if self.has_order_by {
            features.push("order-by");
        }
        if self.has_limit {
            features.push("limit");
        }
        if features.is_empty() {
            "scan".to_string()
        } else {
            features.join("+")
        }
    }
}

fn select_has_projection_expression(select: &Select) -> bool {
    select.projection.iter().any(|item| match item {
        SelectItem::UnnamedExpr(expr) | SelectItem::ExprWithAlias { expr, .. } => {
            !matches!(
                expr,
                SqlExpr::Identifier(_)
                    | SqlExpr::CompoundIdentifier(_)
                    | SqlExpr::Wildcard(_)
                    | SqlExpr::QualifiedWildcard(_, _)
            ) && !sql_expr_is_aggregate(expr)
        }
        _ => false,
    })
}

fn select_has_aggregate(select: &Select) -> bool {
    select.projection.iter().any(|item| match item {
        SelectItem::UnnamedExpr(expr) | SelectItem::ExprWithAlias { expr, .. } => {
            sql_expr_is_aggregate(expr)
        }
        _ => false,
    })
}

fn sql_expr_is_aggregate(expr: &SqlExpr) -> bool {
    match expr {
        SqlExpr::Function(function) => {
            object_name_to_string(&function.name)
                .ok()
                .is_some_and(|name| {
                    matches!(
                        name.to_ascii_lowercase().as_str(),
                        "count" | "sum" | "avg" | "min" | "max"
                    )
                })
        }
        SqlExpr::BinaryOp { left, right, .. } => {
            sql_expr_is_aggregate(left) || sql_expr_is_aggregate(right)
        }
        SqlExpr::Nested(expr)
        | SqlExpr::UnaryOp { expr, .. }
        | SqlExpr::Cast { expr, .. }
        | SqlExpr::IsNull(expr)
        | SqlExpr::IsNotNull(expr) => sql_expr_is_aggregate(expr),
        SqlExpr::Case {
            operand,
            conditions,
            else_result,
            ..
        } => {
            operand.as_deref().is_some_and(sql_expr_is_aggregate)
                || conditions.iter().any(|when| {
                    sql_expr_is_aggregate(&when.condition) || sql_expr_is_aggregate(&when.result)
                })
                || else_result.as_deref().is_some_and(sql_expr_is_aggregate)
        }
        _ => false,
    }
}

async fn estimate_sql_rule_scan_bytes(
    engine: &DodamEngine,
    context: &SqlRuleContext,
    rule: SqlRule,
) -> Option<u64> {
    let required_columns = rule.required_columns();
    if required_columns.is_empty() || context.table_paths.is_empty() {
        return None;
    }
    let mut total = 0_u64;
    let mut matched = false;
    for path in &context.table_paths {
        let columns = required_columns_for_table(path, required_columns);
        if columns.is_empty() {
            continue;
        }
        let projection = Projection::Columns(columns);
        if let Ok(bytes) = engine
            .estimate_parquet_projection_compressed_bytes(path.clone(), &projection)
            .await
        {
            total = total.saturating_add(bytes);
            matched = true;
        }
    }
    matched.then_some(total)
}

fn required_columns_for_table(path: &Path, required_columns: &[&str]) -> Vec<String> {
    let table = path
        .file_stem()
        .or_else(|| path.file_name())
        .and_then(|name| name.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    let prefixes: &[&str] = match table.as_str() {
        "lineitem" => &["l_"],
        "orders" => &["o_"],
        "customer" => &["c_"],
        "supplier" => &["s_"],
        "partsupp" => &["ps_"],
        "part" => &["p_"],
        "nation" => &["n_"],
        "region" => &["r_"],
        _ => &[],
    };
    let columns = if prefixes.is_empty() {
        required_columns
            .iter()
            .map(|column| (*column).to_string())
            .collect::<Vec<_>>()
    } else {
        required_columns
            .iter()
            .filter(|column| prefixes.iter().any(|prefix| column.starts_with(prefix)))
            .map(|column| (*column).to_string())
            .collect::<Vec<_>>()
    };
    let mut unique = Vec::new();
    for column in columns {
        if !unique.iter().any(|existing| existing == &column) {
            unique.push(column);
        }
    }
    unique
}

fn collect_table_paths(table: &TableWithJoins, output: &mut Vec<PathBuf>) {
    collect_table_factor_path(&table.relation, output);
    for join in &table.joins {
        collect_table_factor_path(&join.relation, output);
    }
}

fn collect_table_factor_path(factor: &TableFactor, output: &mut Vec<PathBuf>) {
    match factor {
        TableFactor::Table { .. } => {
            let Ok(table) = parse_table_factor(factor) else {
                return;
            };
            let path = table.path;
            if !output.iter().any(|existing| existing == &path) {
                output.push(path);
            }
        }
        TableFactor::Derived { subquery, .. } => {
            if let SetExpr::Select(select) = subquery.body.as_ref() {
                for table in &select.from {
                    collect_table_paths(table, output);
                }
            }
        }
        TableFactor::NestedJoin {
            table_with_joins, ..
        } => {
            collect_table_paths(table_with_joins, output);
        }
        _ => {}
    }
}

fn query_contains_subquery(query: &Query) -> bool {
    if query.with.as_ref().is_some_and(|with| {
        with.cte_tables
            .iter()
            .any(|cte| query_contains_subquery(&cte.query))
    }) {
        return true;
    }
    set_expr_contains_subquery(query.body.as_ref())
}

fn set_expr_contains_subquery(expr: &SetExpr) -> bool {
    match expr {
        SetExpr::Select(select) => {
            select
                .selection
                .as_ref()
                .is_some_and(expr_contains_materializable_subquery)
                || select.from.iter().any(|table| {
                    table_with_derived_relation(table) || table_contains_subquery(table)
                })
        }
        SetExpr::Query(query) => query_contains_subquery(query),
        SetExpr::SetOperation { left, right, .. } => {
            set_expr_contains_subquery(left) || set_expr_contains_subquery(right)
        }
        _ => false,
    }
}

fn table_contains_subquery(table: &TableWithJoins) -> bool {
    table_with_derived_relation(table)
        || table
            .joins
            .iter()
            .any(|join| table_factor_contains_derived(&join.relation))
}

fn table_with_derived_relation(table: &TableWithJoins) -> bool {
    table_factor_contains_derived(&table.relation)
}

fn table_factor_contains_derived(factor: &TableFactor) -> bool {
    matches!(
        factor,
        TableFactor::Derived { .. }
            | TableFactor::TableFunction { .. }
            | TableFactor::NestedJoin { .. }
    )
}

fn sql_rule_profile_enabled() -> bool {
    std::env::var("DODAM_SQL_RULE_PROFILE").is_ok_and(|value| {
        matches!(
            value.as_str(),
            "1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON"
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn registered_rules_have_unique_names_and_stable_ranks() {
        let mut names = HashSet::new();
        for (rank, rule) in sql_rule_registry().iter().copied().enumerate() {
            assert!(
                names.insert(rule.name()),
                "duplicate SQL rule name: {}",
                rule.name()
            );
            assert_eq!(
                rule.cost_rank(),
                rank as u16,
                "cost rank should match registry order for {}",
                rule.name()
            );
        }
    }

    #[test]
    fn specific_vector_rules_rank_before_generic_with_cte_fallback() {
        let context = SqlRuleContext::from_sql(
            r#"
WITH revenue AS (
    SELECT
        l_suppkey AS supplier_no,
        sum(l_extendedprice * (1 - l_discount)) AS total_revenue
    FROM lineitem
    WHERE l_shipdate >= DATE '1996-01-01'
      AND l_shipdate < DATE '1996-01-01' + INTERVAL '3' MONTH
    GROUP BY l_suppkey
)
SELECT
    s_suppkey,
    s_name,
    s_address,
    s_phone,
    total_revenue
FROM supplier, revenue
WHERE s_suppkey = supplier_no
  AND total_revenue = (SELECT max(total_revenue) FROM revenue)
ORDER BY s_suppkey
"#,
        )
        .expect("parse Q15-shaped SQL context");

        assert!(SqlRule::WithCte.is_candidate(&context));
        assert!(SqlRule::TopSupplierRevenue.is_candidate(&context));

        let with_cte_cost = SqlRule::WithCte.estimated_cost(&context, None);
        let vector_cost = SqlRule::TopSupplierRevenue.estimated_cost(&context, None);

        assert!(
            vector_cost < with_cte_cost,
            "top-supplier vector rule must be tried before generic WITH CTE fallback: vector={vector_cost}, cte={with_cte_cost}"
        );
    }

    #[test]
    fn generic_expression_rules_rank_after_shape_specific_rules() {
        let context = SqlRuleContext::from_sql(
            r#"
SELECT
    sum(l_extendedprice * l_discount) AS revenue
FROM lineitem
WHERE l_shipdate >= DATE '1994-01-01'
  AND l_shipdate < DATE '1995-01-01'
"#,
        )
        .expect("parse expression aggregate SQL context");

        let projection_cost = SqlRule::ProjectionExpression.estimated_cost(&context, None);
        let pricing_cost = SqlRule::PricingSummary.estimated_cost(&context, None);

        assert!(
            pricing_cost < projection_cost,
            "shape-specific vector aggregate candidates should not be hidden by generic expression fallback: vector={pricing_cost}, projection={projection_cost}"
        );
    }
}
