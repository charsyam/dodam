use crate::error::{DodamError, Result};
use crate::execution::metrics::{
    RecordBatchSink, ScanPlanMetrics, SendableBatchStream, write_stream_to_sink,
};

#[derive(Debug, Clone, Default, PartialEq)]
pub struct AggregateMetrics {
    pub fragments: usize,
    pub batches: usize,
    pub rows: usize,
    pub values: Vec<AggregateResult>,
    pub groups: Vec<GroupAggregateResult>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AggregateResult {
    pub expr: AggregateExpr,
    pub value: AggregateValue,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AggregateValue {
    Count(u64),
    Int64(Option<i64>),
    Float64(Option<f64>),
    Date32(Option<i32>),
    Date64(Option<i64>),
    TimestampMillisecond(Option<i64>, Option<String>),
    Utf8(Option<String>),
}

impl std::fmt::Display for AggregateValue {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Count(value) => write!(formatter, "{value}"),
            Self::Int64(Some(value)) => write!(formatter, "{value}"),
            Self::Float64(Some(value)) => write!(formatter, "{value}"),
            Self::Date32(Some(value)) => write!(formatter, "{value}"),
            Self::Date64(Some(value)) => write!(formatter, "{value}"),
            Self::TimestampMillisecond(Some(value), _) => write!(formatter, "{value}"),
            Self::Utf8(Some(value)) => formatter.write_str(value),
            Self::Int64(None)
            | Self::Float64(None)
            | Self::Date32(None)
            | Self::Date64(None)
            | Self::TimestampMillisecond(None, _)
            | Self::Utf8(None) => formatter.write_str("NULL"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct GroupAggregateResult {
    pub keys: Vec<GroupValue>,
    pub values: Vec<AggregateResult>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum GroupValue {
    Int64(Option<i64>),
    UInt64(Option<u64>),
    Date32(Option<i32>),
    Date64(Option<i64>),
    Utf8(Option<String>),
}

impl std::fmt::Display for GroupValue {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Int64(Some(value)) => write!(formatter, "{value}"),
            Self::UInt64(Some(value)) => write!(formatter, "{value}"),
            Self::Date32(Some(value)) => write!(formatter, "{value}"),
            Self::Date64(Some(value)) => write!(formatter, "{value}"),
            Self::Utf8(Some(value)) => formatter.write_str(value),
            Self::Int64(None)
            | Self::UInt64(None)
            | Self::Date32(None)
            | Self::Date64(None)
            | Self::Utf8(None) => formatter.write_str("NULL"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AggregateExpr {
    CountStar,
    Count(String),
    CountDistinct(String),
    Sum(String),
    Avg(String),
    Min(String),
    Max(String),
}

impl AggregateExpr {
    pub fn parse(input: &str) -> Result<Self> {
        let input = input.trim();
        let Some((function, rest)) = input.split_once('(') else {
            return Err(DodamError::InvalidAggregate(input.to_string()));
        };
        let Some(column) = rest.strip_suffix(')') else {
            return Err(DodamError::InvalidAggregate(input.to_string()));
        };

        let function = function.trim();
        let column = column.trim();
        match function.to_ascii_lowercase().as_str() {
            "count" if column == "*" => Ok(Self::CountStar),
            "count_distinct" if !column.is_empty() => Ok(Self::CountDistinct(column.to_string())),
            "count" if !column.is_empty() => Ok(Self::Count(column.to_string())),
            "sum" if !column.is_empty() => Ok(Self::Sum(column.to_string())),
            "avg" if !column.is_empty() => Ok(Self::Avg(column.to_string())),
            "min" if !column.is_empty() => Ok(Self::Min(column.to_string())),
            "max" if !column.is_empty() => Ok(Self::Max(column.to_string())),
            _ => Err(DodamError::InvalidAggregate(input.to_string())),
        }
    }

    pub fn referenced_column(&self) -> Option<&str> {
        match self {
            Self::CountStar => None,
            Self::Count(column)
            | Self::CountDistinct(column)
            | Self::Sum(column)
            | Self::Avg(column)
            | Self::Min(column)
            | Self::Max(column) => Some(column),
        }
    }
}

impl std::fmt::Display for AggregateExpr {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CountStar => formatter.write_str("count(*)"),
            Self::Count(column) => write!(formatter, "count({column})"),
            Self::CountDistinct(column) => write!(formatter, "count(DISTINCT {column})"),
            Self::Sum(column) => write!(formatter, "sum({column})"),
            Self::Avg(column) => write!(formatter, "avg({column})"),
            Self::Min(column) => write!(formatter, "min({column})"),
            Self::Max(column) => write!(formatter, "max({column})"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SortExpr {
    pub column: String,
    pub descending: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SortKey {
    pub expressions: Vec<SortExpr>,
}

impl SortKey {
    pub fn new(expressions: Vec<SortExpr>) -> Result<Self> {
        if expressions.is_empty() {
            return Err(DodamError::InvalidOrderBy(
                "expected at least one sort expression".to_string(),
            ));
        }
        Ok(Self { expressions })
    }
}

impl From<SortExpr> for SortKey {
    fn from(expression: SortExpr) -> Self {
        Self {
            expressions: vec![expression],
        }
    }
}

impl SortExpr {
    pub fn parse(input: &str) -> Result<Self> {
        let input = input.trim();
        if input.is_empty() {
            return Err(DodamError::InvalidOrderBy(input.to_string()));
        }

        let parts = input.split_whitespace().collect::<Vec<_>>();
        match parts.as_slice() {
            [column] => Ok(Self {
                column: (*column).to_string(),
                descending: false,
            }),
            [column, direction] if direction.eq_ignore_ascii_case("asc") => Ok(Self {
                column: (*column).to_string(),
                descending: false,
            }),
            [column, direction] if direction.eq_ignore_ascii_case("desc") => Ok(Self {
                column: (*column).to_string(),
                descending: true,
            }),
            _ => Err(DodamError::InvalidOrderBy(input.to_string())),
        }
    }
}

pub trait PhysicalPlan: Send {
    fn execute(self: Box<Self>) -> Result<SendableBatchStream>;

    fn execute_to_sink(self: Box<Self>, sink: &mut dyn RecordBatchSink) -> Result<ScanPlanMetrics> {
        let stream = self.execute()?;
        write_stream_to_sink(stream, sink)
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum Projection {
    #[default]
    All,
    Columns(Vec<String>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct FilterExpr(Expr);

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Boolean(Option<bool>),
    Comparison(ComparisonExpr),
    ColumnComparison {
        left: String,
        op: ComparisonOp,
        right: String,
    },
    InList {
        column: String,
        values: Vec<LiteralValue>,
        negated: bool,
        has_null: bool,
    },
    Like {
        column: String,
        pattern: String,
        negated: bool,
        escape: Option<char>,
    },
    IsNull {
        column: String,
        negated: bool,
    },
    Not(Box<Expr>),
    And(Box<Expr>, Box<Expr>),
    Or(Box<Expr>, Box<Expr>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct ComparisonExpr {
    pub column: String,
    pub op: ComparisonOp,
    pub value: LiteralValue,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComparisonOp {
    Eq,
    NotEq,
    Lt,
    LtEq,
    Gt,
    GtEq,
}

#[derive(Debug, Clone, PartialEq)]
pub enum LiteralValue {
    Null,
    Boolean(bool),
    Int64(i64),
    Float64(f64),
    Utf8(String),
}

impl LiteralValue {
    fn parse(value: &str) -> Self {
        if value.eq_ignore_ascii_case("true") {
            Self::Boolean(true)
        } else if value.eq_ignore_ascii_case("false") {
            Self::Boolean(false)
        } else if let Ok(value) = value.parse::<i64>() {
            Self::Int64(value)
        } else if let Ok(value) = value.parse::<f64>() {
            Self::Float64(value)
        } else {
            Self::Utf8(value.to_string())
        }
    }

    pub fn as_i32(&self, column: &str) -> Result<i32> {
        let Self::Int64(value) = self else {
            return Err(DodamError::InvalidFilter(format!("{column}={self}")));
        };
        i32::try_from(*value).map_err(|_| DodamError::InvalidFilter(format!("{column}={self}")))
    }

    pub fn as_i64(&self, column: &str) -> Result<i64> {
        match self {
            Self::Int64(value) => Ok(*value),
            Self::Null | Self::Boolean(_) | Self::Float64(_) | Self::Utf8(_) => {
                Err(DodamError::InvalidFilter(format!("{column}={self}")))
            }
        }
    }

    pub fn as_u64(&self, column: &str) -> Result<u64> {
        let Self::Int64(value) = self else {
            return Err(DodamError::InvalidFilter(format!("{column}={self}")));
        };
        u64::try_from(*value).map_err(|_| DodamError::InvalidFilter(format!("{column}={self}")))
    }

    pub fn as_f64(&self, column: &str) -> Result<f64> {
        match self {
            Self::Int64(value) => Ok(*value as f64),
            Self::Float64(value) => Ok(*value),
            Self::Null | Self::Boolean(_) | Self::Utf8(_) => {
                Err(DodamError::InvalidFilter(format!("{column}={self}")))
            }
        }
    }

    pub fn as_bool(&self, column: &str) -> Result<bool> {
        let Self::Boolean(value) = self else {
            return Err(DodamError::InvalidFilter(format!("{column}={self}")));
        };
        Ok(*value)
    }

    pub fn as_str(&self) -> String {
        match self {
            Self::Null => "NULL".to_string(),
            Self::Boolean(value) => value.to_string(),
            Self::Int64(value) => value.to_string(),
            Self::Float64(value) => value.to_string(),
            Self::Utf8(value) => value.clone(),
        }
    }
}

impl std::fmt::Display for LiteralValue {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Null => formatter.write_str("NULL"),
            Self::Boolean(value) => write!(formatter, "{value}"),
            Self::Int64(value) => write!(formatter, "{value}"),
            Self::Float64(value) => write!(formatter, "{value}"),
            Self::Utf8(value) => formatter.write_str(value),
        }
    }
}

impl FilterExpr {
    pub fn new(expr: Expr) -> Self {
        Self(expr)
    }

    pub fn parse(input: &str) -> Result<Self> {
        let tokens = tokenize_filter(input)?;
        FilterParser::new(input, tokens).parse()
    }

    pub fn expr(&self) -> &Expr {
        &self.0
    }

    pub fn referenced_columns(&self) -> Vec<String> {
        let mut columns = Vec::new();
        collect_referenced_columns(&self.0, &mut columns);
        columns
    }

    pub fn conjuncts(&self) -> Vec<Expr> {
        let mut conjuncts = Vec::new();
        collect_conjuncts(&self.0, &mut conjuncts);
        conjuncts
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct PredicateSet {
    pushdown: Vec<Expr>,
    residual: Option<FilterExpr>,
}

impl PredicateSet {
    pub fn new(filter: Option<FilterExpr>) -> Self {
        let pushdown = filter
            .as_ref()
            .map(|filter| {
                filter
                    .conjuncts()
                    .into_iter()
                    .filter(supports_row_group_pruning)
                    .collect()
            })
            .unwrap_or_default();

        Self {
            pushdown,
            residual: filter,
        }
    }

    pub fn pushdown(&self) -> &[Expr] {
        &self.pushdown
    }

    pub fn residual(&self) -> Option<&FilterExpr> {
        self.residual.as_ref()
    }

    pub fn into_residual(self) -> Option<FilterExpr> {
        self.residual
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum FilterToken {
    Ident(String),
    String(String),
    Op(ComparisonOp),
    Comma,
    LParen,
    RParen,
    And,
    Or,
    Not,
    In,
    Is,
    Null,
}

struct FilterParser {
    input: String,
    tokens: Vec<FilterToken>,
    position: usize,
}

impl FilterParser {
    fn new(input: &str, tokens: Vec<FilterToken>) -> Self {
        Self {
            input: input.to_string(),
            tokens,
            position: 0,
        }
    }

    fn parse(mut self) -> Result<FilterExpr> {
        let expr = self.parse_or()?;

        if self.position != self.tokens.len() {
            return Err(DodamError::InvalidFilter(self.input));
        }

        Ok(FilterExpr(expr))
    }

    fn parse_or(&mut self) -> Result<Expr> {
        let mut expr = self.parse_and()?;

        while self.match_or() {
            let right = self.parse_and()?;
            expr = Expr::Or(Box::new(expr), Box::new(right));
        }

        Ok(expr)
    }

    fn parse_and(&mut self) -> Result<Expr> {
        let mut expr = self.parse_not()?;

        while self.match_and() {
            let right = self.parse_not()?;
            expr = Expr::And(Box::new(expr), Box::new(right));
        }

        Ok(expr)
    }

    fn parse_not(&mut self) -> Result<Expr> {
        if self.match_not() {
            return Ok(Expr::Not(Box::new(self.parse_not()?)));
        }

        self.parse_primary()
    }

    fn parse_primary(&mut self) -> Result<Expr> {
        if self.match_lparen() {
            let expr = self.parse_or()?;
            self.expect_rparen()?;
            return Ok(expr);
        }

        let column = self.expect_ident()?;
        if self.match_is() {
            let negated = self.match_not();
            self.expect_null()?;
            return Ok(Expr::IsNull { column, negated });
        }

        if self.match_not() {
            self.expect_in()?;
            let values = self.parse_in_values()?;
            return Ok(Expr::InList {
                column,
                values,
                negated: true,
                has_null: false,
            });
        }

        if self.match_in() {
            let values = self.parse_in_values()?;
            return Ok(Expr::InList {
                column,
                values,
                negated: false,
                has_null: false,
            });
        }

        let op = self.expect_op()?;
        let value = self.expect_literal()?;

        Ok(Expr::Comparison(ComparisonExpr { column, op, value }))
    }

    fn parse_in_values(&mut self) -> Result<Vec<LiteralValue>> {
        self.expect_lparen()?;
        let mut values = Vec::new();
        loop {
            values.push(self.expect_literal()?);
            if self.match_comma() {
                continue;
            }
            self.expect_rparen()?;
            break;
        }
        Ok(values)
    }

    fn match_and(&mut self) -> bool {
        if self.tokens.get(self.position) == Some(&FilterToken::And) {
            self.position += 1;
            return true;
        }

        false
    }

    fn match_or(&mut self) -> bool {
        if self.tokens.get(self.position) == Some(&FilterToken::Or) {
            self.position += 1;
            return true;
        }

        false
    }

    fn match_not(&mut self) -> bool {
        if self.tokens.get(self.position) == Some(&FilterToken::Not) {
            self.position += 1;
            return true;
        }

        false
    }

    fn match_in(&mut self) -> bool {
        if self.tokens.get(self.position) == Some(&FilterToken::In) {
            self.position += 1;
            return true;
        }

        false
    }

    fn match_is(&mut self) -> bool {
        if self.tokens.get(self.position) == Some(&FilterToken::Is) {
            self.position += 1;
            return true;
        }

        false
    }

    fn match_lparen(&mut self) -> bool {
        if self.tokens.get(self.position) == Some(&FilterToken::LParen) {
            self.position += 1;
            return true;
        }

        false
    }

    fn match_comma(&mut self) -> bool {
        if self.tokens.get(self.position) == Some(&FilterToken::Comma) {
            self.position += 1;
            return true;
        }

        false
    }

    fn expect_ident(&mut self) -> Result<String> {
        match self.tokens.get(self.position) {
            Some(FilterToken::Ident(value)) if !value.is_empty() => {
                self.position += 1;
                Ok(value.clone())
            }
            _ => Err(DodamError::InvalidFilter(self.input.clone())),
        }
    }

    fn expect_op(&mut self) -> Result<ComparisonOp> {
        match self.tokens.get(self.position) {
            Some(FilterToken::Op(op)) => {
                self.position += 1;
                Ok(*op)
            }
            _ => Err(DodamError::InvalidFilter(self.input.clone())),
        }
    }

    fn expect_in(&mut self) -> Result<()> {
        match self.tokens.get(self.position) {
            Some(FilterToken::In) => {
                self.position += 1;
                Ok(())
            }
            _ => Err(DodamError::InvalidFilter(self.input.clone())),
        }
    }

    fn expect_null(&mut self) -> Result<()> {
        match self.tokens.get(self.position) {
            Some(FilterToken::Null) => {
                self.position += 1;
                Ok(())
            }
            _ => Err(DodamError::InvalidFilter(self.input.clone())),
        }
    }

    fn expect_lparen(&mut self) -> Result<()> {
        match self.tokens.get(self.position) {
            Some(FilterToken::LParen) => {
                self.position += 1;
                Ok(())
            }
            _ => Err(DodamError::InvalidFilter(self.input.clone())),
        }
    }

    fn expect_rparen(&mut self) -> Result<()> {
        match self.tokens.get(self.position) {
            Some(FilterToken::RParen) => {
                self.position += 1;
                Ok(())
            }
            _ => Err(DodamError::InvalidFilter(self.input.clone())),
        }
    }

    fn expect_literal(&mut self) -> Result<LiteralValue> {
        match self.tokens.get(self.position) {
            Some(FilterToken::Ident(value)) => {
                self.position += 1;
                Ok(LiteralValue::parse(value))
            }
            Some(FilterToken::String(value)) => {
                self.position += 1;
                Ok(LiteralValue::Utf8(value.clone()))
            }
            _ => Err(DodamError::InvalidFilter(self.input.clone())),
        }
    }
}

fn tokenize_filter(input: &str) -> Result<Vec<FilterToken>> {
    let mut tokens = Vec::new();
    let mut chars = input.char_indices().peekable();

    while let Some((_, ch)) = chars.peek().copied() {
        if ch.is_whitespace() {
            chars.next();
            continue;
        }

        if ch == '\'' || ch == '"' {
            tokens.push(FilterToken::String(read_quoted(input, &mut chars, ch)?));
            continue;
        }

        if ch == '(' {
            chars.next();
            tokens.push(FilterToken::LParen);
            continue;
        }

        if ch == ')' {
            chars.next();
            tokens.push(FilterToken::RParen);
            continue;
        }

        if ch == ',' {
            chars.next();
            tokens.push(FilterToken::Comma);
            continue;
        }

        if let Some(op) = read_operator(&mut chars) {
            tokens.push(FilterToken::Op(op));
            continue;
        }

        let value = read_ident(input, &mut chars);
        if value.is_empty() {
            return Err(DodamError::InvalidFilter(input.to_string()));
        }

        if value.eq_ignore_ascii_case("and") {
            tokens.push(FilterToken::And);
        } else if value.eq_ignore_ascii_case("or") {
            tokens.push(FilterToken::Or);
        } else if value.eq_ignore_ascii_case("not") {
            tokens.push(FilterToken::Not);
        } else if value.eq_ignore_ascii_case("in") {
            tokens.push(FilterToken::In);
        } else if value.eq_ignore_ascii_case("is") {
            tokens.push(FilterToken::Is);
        } else if value.eq_ignore_ascii_case("null") {
            tokens.push(FilterToken::Null);
        } else {
            tokens.push(FilterToken::Ident(value));
        }
    }

    if tokens.is_empty() {
        return Err(DodamError::InvalidFilter(input.to_string()));
    }

    Ok(tokens)
}

fn read_quoted(
    input: &str,
    chars: &mut std::iter::Peekable<std::str::CharIndices<'_>>,
    quote: char,
) -> Result<String> {
    chars.next();
    let mut value = String::new();
    let mut escaped = false;

    for (_, ch) in chars.by_ref() {
        if escaped {
            value.push(ch);
            escaped = false;
            continue;
        }

        if ch == '\\' {
            escaped = true;
            continue;
        }

        if ch == quote {
            return Ok(value);
        }

        value.push(ch);
    }

    Err(DodamError::InvalidFilter(input.to_string()))
}

fn read_operator(
    chars: &mut std::iter::Peekable<std::str::CharIndices<'_>>,
) -> Option<ComparisonOp> {
    let (_, first) = chars.peek().copied()?;
    match first {
        '=' => {
            chars.next();
            Some(ComparisonOp::Eq)
        }
        '!' => {
            chars.next();
            if chars.peek().is_some_and(|(_, ch)| *ch == '=') {
                chars.next();
                Some(ComparisonOp::NotEq)
            } else {
                None
            }
        }
        '<' => {
            chars.next();
            if chars.peek().is_some_and(|(_, ch)| *ch == '=') {
                chars.next();
                Some(ComparisonOp::LtEq)
            } else {
                Some(ComparisonOp::Lt)
            }
        }
        '>' => {
            chars.next();
            if chars.peek().is_some_and(|(_, ch)| *ch == '=') {
                chars.next();
                Some(ComparisonOp::GtEq)
            } else {
                Some(ComparisonOp::Gt)
            }
        }
        _ => None,
    }
}

fn read_ident(input: &str, chars: &mut std::iter::Peekable<std::str::CharIndices<'_>>) -> String {
    let Some((start, _)) = chars.peek().copied() else {
        return String::new();
    };
    let mut end = start;
    let mut paren_depth = 0_u32;

    while let Some((index, ch)) = chars.peek().copied() {
        if paren_depth == 0
            && (ch.is_whitespace() || matches!(ch, '=' | '!' | '<' | '>' | '\'' | '"' | ',' | ')'))
        {
            break;
        }
        if ch == '(' {
            paren_depth = paren_depth.saturating_add(1);
        } else if ch == ')' {
            paren_depth = paren_depth.saturating_sub(1);
        }

        end = index + ch.len_utf8();
        chars.next();
    }

    input[start..end].to_string()
}

fn collect_referenced_columns(expr: &Expr, columns: &mut Vec<String>) {
    match expr {
        Expr::Boolean(_) => {}
        Expr::Comparison(comparison) => {
            if !columns.iter().any(|column| column == &comparison.column) {
                columns.push(comparison.column.clone());
            }
        }
        Expr::ColumnComparison { left, right, .. } => {
            if !columns.iter().any(|column| column == left) {
                columns.push(left.clone());
            }
            if !columns.iter().any(|column| column == right) {
                columns.push(right.clone());
            }
        }
        Expr::InList { column, .. } | Expr::Like { column, .. } | Expr::IsNull { column, .. } => {
            if !columns.iter().any(|existing| existing == column) {
                columns.push(column.clone());
            }
        }
        Expr::Not(expr) => collect_referenced_columns(expr, columns),
        Expr::And(left, right) => {
            collect_referenced_columns(left, columns);
            collect_referenced_columns(right, columns);
        }
        Expr::Or(left, right) => {
            collect_referenced_columns(left, columns);
            collect_referenced_columns(right, columns);
        }
    }
}

fn collect_conjuncts(expr: &Expr, conjuncts: &mut Vec<Expr>) {
    match expr {
        Expr::And(left, right) => {
            collect_conjuncts(left, conjuncts);
            collect_conjuncts(right, conjuncts);
        }
        Expr::Boolean(_)
        | Expr::Or(_, _)
        | Expr::Not(_)
        | Expr::InList { .. }
        | Expr::Like { .. }
        | Expr::IsNull { .. } => {}
        expr => conjuncts.push(expr.clone()),
    }
}

fn supports_row_group_pruning(expr: &Expr) -> bool {
    matches!(
        expr,
        Expr::Comparison(ComparisonExpr {
            op: ComparisonOp::Eq
                | ComparisonOp::Lt
                | ComparisonOp::LtEq
                | ComparisonOp::Gt
                | ComparisonOp::GtEq,
            ..
        })
    )
}
