use super::*;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct BoundColumn {
    pub(super) relation: Option<String>,
    pub(super) name: String,
    pub(super) physical_name: String,
}

impl BoundColumn {
    fn unqualified(name: impl Into<String>) -> Self {
        let name = name.into();
        Self {
            relation: None,
            physical_name: name.clone(),
            name,
        }
    }

    fn qualified(relation: impl Into<String>, name: impl Into<String>) -> Self {
        let relation = relation.into();
        let name = name.into();
        Self {
            physical_name: format!("{relation}.{name}"),
            relation: Some(relation),
            name,
        }
    }

    fn physical(physical_name: impl Into<String>) -> Self {
        let physical_name = physical_name.into();
        let (relation, name) = physical_name
            .split_once('.')
            .map(|(relation, name)| (Some(relation.to_string()), name.to_string()))
            .unwrap_or((None, physical_name.clone()));
        Self {
            relation,
            name,
            physical_name,
        }
    }
}

pub(super) struct ColumnResolver<'a> {
    table_alias: Option<&'a str>,
    table_aliases: &'a [&'a str],
    batch: Option<&'a RecordBatch>,
}

impl<'a> ColumnResolver<'a> {
    pub(super) fn single(table_alias: Option<&'a str>) -> Self {
        Self {
            table_alias,
            table_aliases: &[],
            batch: None,
        }
    }

    pub(super) fn join(table_aliases: &'a [&'a str]) -> Self {
        Self {
            table_alias: None,
            table_aliases,
            batch: None,
        }
    }

    pub(super) fn batch(batch: &'a RecordBatch) -> Self {
        Self {
            table_alias: None,
            table_aliases: &[],
            batch: Some(batch),
        }
    }

    pub(super) fn resolve_single_column(&self, expr: &SqlExpr) -> Result<String> {
        self.resolve_single_bound(expr)
            .map(|column| column.physical_name)
    }

    fn resolve_single_bound(&self, expr: &SqlExpr) -> Result<BoundColumn> {
        match expr {
            SqlExpr::Identifier(ident) => Ok(BoundColumn::unqualified(ident.value.clone())),
            SqlExpr::CompoundIdentifier(parts) => {
                let [qualifier, column] = parts.as_slice() else {
                    return Err(DodamError::UnsupportedSql(format!(
                        "only table-qualified columns are supported, got {expr}"
                    )));
                };
                if let Some(table_alias) = self.table_alias
                    && qualifier.value != table_alias
                {
                    return Err(DodamError::UnknownTableQualifier(qualifier.value.clone()));
                }
                Ok(if self.table_alias.is_some() {
                    BoundColumn::unqualified(column.value.clone())
                } else {
                    BoundColumn::qualified(qualifier.value.clone(), column.value.clone())
                })
            }
            _ => Err(DodamError::UnsupportedSql(format!(
                "expected column identifier, got {expr}"
            ))),
        }
    }

    pub(super) fn raw_column(expr: &SqlExpr) -> Result<Option<String>> {
        Ok(Self::raw_bound(expr)?.map(|column| column.physical_name))
    }

    pub(super) fn raw_bound(expr: &SqlExpr) -> Result<Option<BoundColumn>> {
        match expr {
            SqlExpr::Identifier(ident) => Ok(Some(BoundColumn::unqualified(ident.value.clone()))),
            SqlExpr::CompoundIdentifier(parts) => {
                let [qualifier, column] = parts.as_slice() else {
                    return Err(DodamError::UnsupportedSql(format!(
                        "only table-qualified columns are supported, got {expr}"
                    )));
                };
                Ok(Some(BoundColumn::qualified(
                    qualifier.value.clone(),
                    column.value.clone(),
                )))
            }
            SqlExpr::Nested(expr) => Self::raw_bound(expr),
            _ => Ok(None),
        }
    }

    pub(super) fn resolve_join_column(&self, expr: &SqlExpr) -> Result<String> {
        self.resolve_join_bound(expr)
            .map(|column| column.physical_name)
    }

    pub(super) fn resolve_join_bound(&self, expr: &SqlExpr) -> Result<BoundColumn> {
        match expr {
            SqlExpr::Identifier(ident) => {
                if let Some((qualifier, column)) = ident.value.split_once('.') {
                    self.validate_join_qualifier(qualifier)?;
                    return Ok(BoundColumn::qualified(qualifier, column));
                }
                self.infer_unqualified_join_bound(&ident.value)
            }
            SqlExpr::CompoundIdentifier(parts) => match parts.as_slice() {
                [_] => self.infer_unqualified_join_bound(&parts[0].value),
                [qualifier, column] => {
                    self.validate_join_qualifier(&qualifier.value)?;
                    Ok(BoundColumn::qualified(
                        qualifier.value.clone(),
                        column.value.clone(),
                    ))
                }
                _ => Err(DodamError::UnsupportedSql(format!(
                    "only table-qualified columns are supported, got {expr}"
                ))),
            },
            _ => Err(DodamError::UnsupportedSql(format!(
                "expected JOIN column, got {expr}"
            ))),
        }
    }

    pub(super) fn resolve_batch_bound(&self, column: &str) -> Result<Option<BoundColumn>> {
        let Some(batch) = self.batch else {
            return Ok(None);
        };
        if batch_column_index(batch, column).is_ok() {
            return Ok(Some(BoundColumn::physical(column)));
        }
        if let Some((_, unqualified)) = column.split_once('.')
            && batch_column_index(batch, unqualified).is_ok()
        {
            return Ok(Some(BoundColumn::physical(unqualified.to_string())));
        }
        if let Some((function, argument)) = aggregate_column_parts(column) {
            let aggregate_suffix = format!(".{argument})");
            let matches = batch
                .schema()
                .fields()
                .iter()
                .filter(|field| {
                    field.name().starts_with(&format!("{function}("))
                        && field.name().ends_with(&aggregate_suffix)
                })
                .map(|field| field.name().clone())
                .collect::<Vec<_>>();
            match matches.as_slice() {
                [] => {}
                [column] => return Ok(Some(BoundColumn::physical(column.clone()))),
                _ => return Err(ambiguous_column(column)),
            }
        }
        let suffix = format!(".{column}");
        let matches = batch
            .schema()
            .fields()
            .iter()
            .filter(|field| field.name().ends_with(&suffix))
            .map(|field| field.name().clone())
            .collect::<Vec<_>>();
        match matches.as_slice() {
            [] => Ok(None),
            [column] => Ok(Some(BoundColumn::physical(column.clone()))),
            _ => Err(ambiguous_column(column)),
        }
    }

    fn validate_join_qualifier(&self, qualifier: &str) -> Result<()> {
        if !self
            .table_aliases
            .iter()
            .any(|table_alias| table_alias.eq_ignore_ascii_case(qualifier))
        {
            return Err(DodamError::UnknownTableQualifier(qualifier.to_string()));
        }
        Ok(())
    }

    fn infer_unqualified_join_bound(&self, column: &str) -> Result<BoundColumn> {
        let Some((prefix, _)) = column.split_once('_') else {
            return Err(ambiguous_column(column));
        };
        if matches!(prefix.to_ascii_lowercase().as_str(), "supplier" | "total")
            && self
                .table_aliases
                .iter()
                .any(|alias| alias.eq_ignore_ascii_case("revenue"))
        {
            return Ok(BoundColumn::qualified("revenue", column));
        }
        if let Some(alias) = infer_tpch_table_alias(prefix, self.table_aliases) {
            return Ok(BoundColumn::qualified(alias, column));
        }
        if is_tpch_column_prefix(prefix) {
            return Err(DodamError::UnknownColumn(column.to_string()));
        }
        let Some(prefix_initial) = prefix.chars().next() else {
            return Err(DodamError::UnsupportedSql(format!(
                "cannot infer JOIN table for unqualified column {column}"
            )));
        };
        let matches = self
            .table_aliases
            .iter()
            .filter(|alias| {
                alias
                    .chars()
                    .next()
                    .is_some_and(|initial| initial.eq_ignore_ascii_case(&prefix_initial))
            })
            .collect::<Vec<_>>();
        let [alias] = matches.as_slice() else {
            return Err(DodamError::UnsupportedSql(format!(
                "cannot infer JOIN table for unqualified column {column}"
            )));
        };
        Ok(BoundColumn::qualified((*alias).to_string(), column))
    }
}

fn ambiguous_column(column: &str) -> DodamError {
    DodamError::AmbiguousColumn(column.to_string())
}

pub(super) fn join_column_name(expr: &SqlExpr, table_aliases: &[&str]) -> Result<String> {
    ColumnResolver::join(table_aliases).resolve_join_column(expr)
}

pub(super) fn sql_column_name(expr: &SqlExpr, table_alias: Option<&str>) -> Result<String> {
    ColumnResolver::single(table_alias).resolve_single_column(expr)
}

pub(super) fn object_name_to_string(name: &ObjectName) -> Result<String> {
    let [part] = name.0.as_slice() else {
        return Err(DodamError::UnsupportedSql(format!(
            "compound object names are not supported: {name}"
        )));
    };
    match part {
        ObjectNamePart::Identifier(ident) => Ok(ident.value.clone()),
        _ => Err(DodamError::UnsupportedSql(format!(
            "unsupported object name: {name}"
        ))),
    }
}

pub(super) fn infer_tpch_table_alias<'a>(
    prefix: &str,
    table_aliases: &'a [&str],
) -> Option<&'a str> {
    let table = match prefix.to_ascii_lowercase().as_str() {
        "c" => "customer",
        "o" => "orders",
        "l" => "lineitem",
        "p" => "part",
        "ps" => "partsupp",
        "s" => "supplier",
        "n" => "nation",
        "r" => "region",
        _ => return None,
    };
    table_aliases
        .iter()
        .copied()
        .find(|alias| alias.eq_ignore_ascii_case(table))
}

fn is_tpch_column_prefix(prefix: &str) -> bool {
    matches!(
        prefix.to_ascii_lowercase().as_str(),
        "c" | "o" | "l" | "p" | "ps" | "s" | "n" | "r"
    )
}
