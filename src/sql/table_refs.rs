use super::*;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct SqlTableRef {
    pub(super) path: PathBuf,
    pub(super) alias: Option<String>,
}

pub(super) fn table_ref_alias_or_name(table: &SqlTableRef) -> String {
    table.alias.clone().unwrap_or_else(|| {
        table
            .path
            .file_stem()
            .or_else(|| table.path.file_name())
            .and_then(|name| name.to_str())
            .unwrap_or_else(|| table.path.to_str().unwrap_or(""))
            .to_string()
    })
}

pub(super) fn parse_from(select: &Select) -> Result<SqlTableRef> {
    let [table] = select.from.as_slice() else {
        return Err(DodamError::UnsupportedSql(
            "expected exactly one FROM table".to_string(),
        ));
    };
    if !table.joins.is_empty() {
        return Err(DodamError::UnsupportedSql(
            "JOIN is not supported".to_string(),
        ));
    }
    parse_table_factor(&table.relation)
}

pub(super) fn parse_comma_join_table_refs(select: &Select) -> Result<Option<Vec<SqlTableRef>>> {
    if select.from.is_empty() {
        return Ok(None);
    }
    if select.from.len() > 1 {
        if select.from.iter().any(|table| !table.joins.is_empty()) {
            return Err(DodamError::UnsupportedSql(
                "mixed comma and explicit JOIN syntax is not supported".to_string(),
            ));
        }
        return select
            .from
            .iter()
            .map(|table| parse_table_factor(&table.relation))
            .collect::<Result<Vec<_>>>()
            .map(Some);
    }

    let table = &select.from[0];
    if table.joins.is_empty() {
        return Ok(None);
    }
    let mut tables = vec![parse_table_factor(&table.relation)?];
    for join in &table.joins {
        match &join.join_operator {
            JoinOperator::CrossJoin(JoinConstraint::None) => {
                tables.push(parse_table_factor(&join.relation)?);
            }
            _ => return Ok(None),
        }
    }
    Ok(Some(tables))
}

pub(super) fn parse_multi_input_join_table_refs_and_conjuncts(
    select: &Select,
) -> Result<Option<(Vec<SqlTableRef>, Vec<SqlExpr>)>> {
    if let Some(tables) = parse_comma_join_table_refs(select)? {
        return Ok(Some((tables, Vec::new())));
    }
    let [table] = select.from.as_slice() else {
        return Ok(None);
    };
    if table.joins.len() <= 1 {
        return Ok(None);
    }
    let mut tables = vec![parse_table_factor(&table.relation)?];
    let mut conjuncts = Vec::new();
    for join in &table.joins {
        let constraint = match &join.join_operator {
            JoinOperator::Join(constraint) | JoinOperator::Inner(constraint) => constraint,
            _ => return Ok(None),
        };
        let JoinConstraint::On(expr) = constraint else {
            return Err(DodamError::UnsupportedSql(
                "multi-input explicit JOIN requires ON conditions".to_string(),
            ));
        };
        tables.push(parse_table_factor(&join.relation)?);
        collect_sql_and_conjuncts(expr, &mut conjuncts);
    }
    Ok(Some((tables, conjuncts)))
}

pub(super) fn parse_select_table_refs(select: &Select) -> Result<Vec<SqlTableRef>> {
    if let Some(tables) = parse_comma_join_table_refs(select)? {
        return Ok(tables);
    }
    if select.from.is_empty() {
        return Ok(Vec::new());
    }
    if select.from.iter().any(|table| !table.joins.is_empty()) {
        return Ok(Vec::new());
    }
    select
        .from
        .iter()
        .map(|table| parse_table_factor(&table.relation))
        .collect::<Result<Vec<_>>>()
}

pub(super) fn parse_table_factor(relation: &TableFactor) -> Result<SqlTableRef> {
    let TableFactor::Table {
        name,
        alias,
        args,
        with_hints,
        version,
        with_ordinality,
        partitions,
        json_path,
        sample,
        index_hints,
        ..
    } = relation
    else {
        return Err(DodamError::UnsupportedSql(
            "only direct table paths or registered table names are supported".to_string(),
        ));
    };
    if args.is_some()
        || !with_hints.is_empty()
        || version.is_some()
        || *with_ordinality
        || !partitions.is_empty()
        || json_path.is_some()
        || sample.is_some()
        || !index_hints.is_empty()
    {
        return Err(DodamError::UnsupportedSql(
            "table functions, hints, versions, partitions, and samples are not supported"
                .to_string(),
        ));
    }
    if let Some(alias) = alias
        && (!alias.columns.is_empty() || alias.at.is_some())
    {
        return Err(DodamError::UnsupportedSql(
            "table column aliases and AT aliases are not supported".to_string(),
        ));
    }
    Ok(SqlTableRef {
        path: PathBuf::from(object_name_to_string(name)?),
        alias: alias.as_ref().map(|alias| alias.name.value.clone()),
    })
}
