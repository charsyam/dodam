use super::*;

#[derive(Clone)]
pub(super) struct ProductExpressionShape {
    pub(super) terms: Vec<ProductExpressionTerm>,
}

#[derive(Clone)]
pub(super) struct ProductExpressionTerm {
    pub(super) column: String,
    pub(super) transform: ProductTermTransform,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ProductTermTransform {
    Identity,
    OneMinus,
    OnePlus,
}

impl ProductTermTransform {
    #[inline]
    pub(super) fn apply_raw_i64(self, raw: i64, scale: i64) -> i64 {
        match self {
            Self::Identity => raw,
            Self::OneMinus => scale - raw,
            Self::OnePlus => scale + raw,
        }
    }

    #[inline]
    pub(super) fn apply_raw_f64(self, raw: f64, scale: f64) -> f64 {
        match self {
            Self::Identity => raw,
            Self::OneMinus => scale - raw,
            Self::OnePlus => scale + raw,
        }
    }
}

pub(super) fn product_expression_shape(
    expr: &ScalarSqlExpression,
) -> Option<ProductExpressionShape> {
    let mut terms = Vec::new();
    collect_product_terms(expr, &mut terms)?;
    (2..=3)
        .contains(&terms.len())
        .then_some(ProductExpressionShape { terms })
}

fn collect_product_terms(
    expr: &ScalarSqlExpression,
    terms: &mut Vec<ProductExpressionTerm>,
) -> Option<()> {
    if let ScalarSqlExpression::Binary { left, op, right } = expr
        && *op == BinaryOperator::Multiply
    {
        collect_product_terms(left, terms)?;
        collect_product_terms(right, terms)?;
        return Some(());
    }
    terms.push(product_term(expr)?);
    Some(())
}

fn product_term(expr: &ScalarSqlExpression) -> Option<ProductExpressionTerm> {
    if let ScalarSqlExpression::Column(column) = expr {
        return Some(ProductExpressionTerm {
            column: column.clone(),
            transform: ProductTermTransform::Identity,
        });
    }
    let ScalarSqlExpression::Binary { left, op, right } = expr else {
        return None;
    };
    if !scalar_literal_is_one(left) {
        return None;
    }
    let ScalarSqlExpression::Column(column) = right.as_ref() else {
        return None;
    };
    let transform = match op {
        BinaryOperator::Minus => ProductTermTransform::OneMinus,
        BinaryOperator::Plus => ProductTermTransform::OnePlus,
        _ => return None,
    };
    Some(ProductExpressionTerm {
        column: column.clone(),
        transform,
    })
}

fn scalar_literal_is_one(expr: &ScalarSqlExpression) -> bool {
    match expr {
        ScalarSqlExpression::Literal(LiteralValue::Int64(1)) => true,
        ScalarSqlExpression::Literal(LiteralValue::Float64(value)) => *value == 1.0,
        _ => false,
    }
}
