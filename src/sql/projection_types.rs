use super::*;

#[derive(Debug)]
pub(super) struct ParsedProjection {
    pub(super) projection: Projection,
    pub(super) aggregates: Vec<AggregateExpr>,
    pub(super) filtered_aggregates: Vec<NativeFilteredAggregateSpec>,
    pub(super) aggregate_expressions: Vec<ProjectionExpression>,
    pub(super) aliases: Vec<(String, String)>,
    pub(super) expressions: Vec<ProjectionExpression>,
    pub(super) ordinal_targets: Vec<String>,
    pub(super) qualified_wildcards: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct ProjectionExpression {
    pub(super) output_name: String,
    pub(super) expr: ScalarSqlExpression,
}

#[derive(Debug, Clone)]
pub(super) struct GroupExpressionBinding {
    pub(super) source: String,
    pub(super) expression: ProjectionExpression,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) enum ScalarSqlExpression {
    Column(String),
    StructField {
        column: String,
        field: String,
    },
    ListIndex {
        column: String,
        field: Option<String>,
        index: Box<ScalarSqlExpression>,
    },
    ListLength {
        column: String,
        field: Option<String>,
    },
    Literal(LiteralValue),
    Binary {
        left: Box<ScalarSqlExpression>,
        op: BinaryOperator,
        right: Box<ScalarSqlExpression>,
    },
    Cast {
        expr: Box<ScalarSqlExpression>,
        target: String,
    },
    Coalesce(Vec<ScalarSqlExpression>),
    Lower(Box<ScalarSqlExpression>),
    Upper(Box<ScalarSqlExpression>),
    Length(Box<ScalarSqlExpression>),
    Trim(Box<ScalarSqlExpression>),
    Abs(Box<ScalarSqlExpression>),
    Round(Box<ScalarSqlExpression>),
    Floor(Box<ScalarSqlExpression>),
    Ceil(Box<ScalarSqlExpression>),
    Replace {
        expr: Box<ScalarSqlExpression>,
        from: Box<ScalarSqlExpression>,
        to: Box<ScalarSqlExpression>,
    },
    Concat(Vec<ScalarSqlExpression>),
    ExtractYear(Box<ScalarSqlExpression>),
    Substring {
        expr: Box<ScalarSqlExpression>,
        start: Box<ScalarSqlExpression>,
        length: Option<Box<ScalarSqlExpression>>,
    },
    Case {
        conditions: Vec<SqlExpr>,
        results: Vec<ScalarSqlExpression>,
        else_result: Option<Box<ScalarSqlExpression>>,
    },
}
