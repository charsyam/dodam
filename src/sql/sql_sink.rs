use super::*;

pub async fn try_execute_sql_streaming(
    engine: &DodamEngine,
    sql: &str,
    batch_size: usize,
) -> Result<Option<SendableBatchStream>> {
    if explain_sql(engine, sql, batch_size).await?.is_some() {
        return Ok(None);
    }
    let Some(request) = plan_direct_join_sink_request_relaxed(sql, batch_size)? else {
        return Ok(None);
    };
    engine.join_parquet_batches(request).await.map(Some)
}

pub async fn try_execute_sql_to_sink(
    engine: &DodamEngine,
    sql: &str,
    batch_size: usize,
    sink: &mut dyn RecordBatchSink,
) -> Result<Option<ScanPlanMetrics>> {
    if explain_sql(engine, sql, batch_size).await?.is_some() {
        return Ok(None);
    }
    if let Some(metrics) =
        try_execute_set_operation_sql_to_sink(engine, sql, batch_size, sink).await?
    {
        return Ok(Some(metrics));
    }
    let Some(request) = plan_direct_join_sink_request_relaxed(sql, batch_size)? else {
        return Ok(None);
    };
    let plan = engine.plan_parquet_join(request).await?;
    engine.write_join_plan_to_sink(plan, sink).map(Some)
}

pub async fn execute_sql_to_result_sink(
    engine: &DodamEngine,
    sql: &str,
    batch_size: usize,
    sink: &mut dyn SqlResultSink,
    options: SqlSinkExecutionOptions,
) -> Result<SqlSinkExecutionProfile> {
    let mut profile = SqlSinkExecutionProfile::default();
    if options.allow_direct_or_streaming && explain_sql(engine, sql, batch_size).await?.is_none() {
        let direct_started = Instant::now();
        if let Some(metrics) =
            try_execute_set_operation_sql_to_sink(engine, sql, batch_size, sink.record_batch_sink())
                .await?
        {
            profile.direct_sink = Some(direct_started.elapsed());
            profile.scan_plan_metrics = Some(metrics);
            return Ok(profile);
        }
        if let Some(request) = plan_direct_join_sink_request_relaxed(sql, batch_size)? {
            let plan = engine.plan_parquet_join(request).await?;
            let metrics = engine.write_join_plan_to_sink(plan, sink.record_batch_sink())?;
            profile.direct_sink = Some(direct_started.elapsed());
            profile.scan_plan_metrics = Some(metrics);
            return Ok(profile);
        }
        profile.direct_sink = Some(direct_started.elapsed());
        profile.streaming = Some(Duration::ZERO);
    } else {
        profile.direct_sink = Some(Duration::ZERO);
        profile.streaming = Some(Duration::ZERO);
    }

    let execute_started = Instant::now();
    let output = execute_sql(engine, sql, batch_size).await?;
    profile.execute = Some(execute_started.elapsed());
    let write_started = Instant::now();
    sink.write_output(output)?;
    profile.write_output = Some(write_started.elapsed());
    Ok(profile)
}
