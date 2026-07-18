use super::*;

pub(super) async fn explain_sql(
    engine: &DodamEngine,
    sql: &str,
    batch_size: usize,
) -> Result<Option<String>> {
    let dialect = GenericDialect {};
    let statements = Parser::parse_sql(&dialect, sql)
        .map_err(|error| DodamError::UnsupportedSql(error.to_string()))?;
    let [statement] = statements.as_slice() else {
        return Err(DodamError::UnsupportedSql(
            "expected exactly one statement".to_string(),
        ));
    };

    let Statement::Explain {
        analyze,
        verbose,
        query_plan,
        estimate,
        statement,
        format,
        options,
        ..
    } = statement
    else {
        return Ok(None);
    };

    if *analyze || *verbose || *query_plan || *estimate || format.is_some() || options.is_some() {
        return Err(DodamError::UnsupportedSql(
            "EXPLAIN options are not supported yet".to_string(),
        ));
    }

    let Statement::Query(query) = statement.as_ref() else {
        return Err(DodamError::UnsupportedSql(
            "EXPLAIN only supports SELECT queries".to_string(),
        ));
    };
    let query = parse_query(query)?;
    explain_query(engine, query, batch_size).await.map(Some)
}

async fn explain_query(engine: &DodamEngine, query: SqlQuery, batch_size: usize) -> Result<String> {
    if let Some(join) = query.join.clone() {
        if query.is_aggregate() || query.having.is_some() || query.distinct {
            return Err(DodamError::UnsupportedSql(
                "JOIN with aggregates, HAVING, or DISTINCT is not supported".to_string(),
            ));
        }
        let join_input_projection = join_input_projection_with_expression_filter(&query)?;
        let join_plan = plan_join_inputs(
            &join_input_projection,
            query.filter.as_ref(),
            query.order_by.as_ref(),
            &join.left_alias,
            &join.left_keys,
            &join.right_alias,
            &join.right_keys,
        );
        return engine
            .explain_join_parquet(JoinParquetRequest {
                left_path: query.path,
                right_path: join.right.path,
                batch_size,
                left_keys: join.left_keys,
                right_keys: join.right_keys,
                left_prefix: join.left_alias,
                right_prefix: join.right_alias,
                left_projection: join_plan.left_projection,
                right_projection: join_plan.right_projection,
                left_filter: join_plan.left_filter,
                right_filter: combine_filter_options(
                    join_plan.right_filter,
                    join.right_filter.clone(),
                ),
                output_projection: Projection::All,
                join_memory_limit_bytes: default_join_memory_limit_bytes(),
                join_algorithm: JoinAlgorithm::Auto,
                join_type: join.join_type,
            })
            .await;
    }

    if query.is_aggregate() {
        return engine
            .explain_parquet_aggregate(
                query.path,
                batch_size,
                query.aggregates,
                query.group_by,
                query.filter,
            )
            .await;
    }
    if query.distinct {
        return engine
            .explain_parquet_distinct_scan(
                query.path,
                batch_size,
                scan_limit_with_offset(query.limit, query.offset)?,
                query.projection,
                query.filter,
                query.order_by,
            )
            .await;
    }
    engine
        .explain_parquet_scan(
            query.path,
            batch_size,
            scan_limit_with_offset(query.limit, query.offset)?,
            query.projection,
            query.filter,
            query.order_by,
        )
        .await
}
