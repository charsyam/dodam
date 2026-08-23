# Dodam 작업 인수인계

최종 갱신: 2026-08-23  
기준 브랜치: `main`  
범용 optimizer 구현 커밋: `ce00aff` (`Add generalized logical and physical optimizer`)

## 반드시 지킬 원칙

- row-at-a-time 실행을 추가하지 않는다. blocked/vectorized pipeline을 기준으로 평가한다.
- integer 계열 연산은 SIMD 적용 가능성을 함께 검토한다.
- TPC-H 쿼리 하나만 겨냥한 분기보다 여러 SQL에 적용되는 일반 규칙을 우선한다.
- 같은 chunk를 여러 번 읽지 않는다. 재독은 비용상 명확히 이득일 때만 허용한다.
- 새 SQL을 지원하면 DuckDB와 결과 및 성능을 비교한다.
- 테스트와 벤치는 하나씩 순차 실행한다. 병렬 실행하지 않는다.
- DuckDB 성능 비교의 기본 데이터 크기는 SF100이다.
- 사용자가 만들지 않은 `scripts/__pycache__/`는 추적하거나 삭제하지 않는다.

## 현재 상태 요약

Dodam은 Arrow `RecordBatch` 기반 벡터 실행기와 TPC-H fast path를 갖고 있다.
현재 단계에서는 fast path를 제거하지 않고, 실제 공용 `plan::LogicalPlan` 위에
범용 logical optimizer와 physical planner의 첫 vertical slice를 연결했다.

SQL 전체가 아직 하나의 optimizer 경로로 완전히 통합된 것은 아니다.
`sql/rule_registry.rs`와 여러 TPC-H 전용/형태 기반 실행 경로가 먼저 후보를
선택하며, 일반 scan planning은 새 optimizer를 거친다. 기존 성능을 유지하면서
점진적으로 공용 optimizer의 적용 범위를 넓히는 방향이다.

## 지금까지 완료한 작업

### 1. 범용 logical optimizer

관련 파일: `src/optimizer.rs`

- 테스트 전용으로 분리되어 있던 `LogicalPlanNode`를 제거했다.
- 실제 `plan::LogicalPlan`을 입력으로 받는 `LogicalOptimizer`를 추가했다.
- 적용한 규칙을 `OptimizerRule` 목록으로 남긴다.
- 현재 지원 규칙:
  - `FilterIntoScan`
  - `ProjectionIntoScan`
  - `SortIntoScan`
  - `LimitIntoScan`
  - `DistinctIntoScan`
  - `CombineFilters`
  - `LimitIntoSort`
- `LIMIT` 아래로 filter/distinct/sort를 잘못 이동하거나 projection으로 제거된
  컬럼을 참조하는 filter/sort를 pushdown하지 않도록 의미 보존 장벽을 둔다.
- 기존 `LogicalJoinGraph`의 greedy/exhaustive/bushy 비용 계산과 join input
  projection/filter pushdown은 유지했다.

### 2. 범용 physical planning

관련 파일: `src/plan.rs`

- `PhysicalPlanner::plan`이 logical optimizer를 한 번 실행한 뒤 물리 계획을 만든다.
- scan은 output, filter, order-by 컬럼의 합집합만 한 번 읽는다.
- `PredicateSet`이 분해한 pruning 가능 conjunct를 Parquet scan에 전달한다.
- 전체 filter는 vectorized residual filter로 유지해 정확성을 보장한다.
- non-distinct는 `filter -> sort -> output projection -> limit` 순서를 지킨다.
- distinct는 `filter -> output projection -> distinct -> sort -> limit` 순서를 지킨다.
- global sort/limit/distinct에는 필요할 때 gather exchange를 삽입한다.
- `PhysicalJoinStrategy::Auto`를 추가했다. 기본 메모리 예산은 128 MiB다.
- 통계상 더 작은 입력을 hash build side로 선택한다.
- inner/full/semi join의 build가 메모리 예산을 넘으면 partitioned hash로 전환한다.
- 명시적인 Hash/PartitionedHash/SortMerge 설정은 이전처럼 override할 수 있다.

### 3. 실제 engine 경로 연결

관련 파일: `src/engine.rs`

- `plan_table_source_scan_with_options`가 filter/projection/distinct/order/limit를
  명시적인 logical node chain으로 만든다.
- 이 chain을 `LogicalOptimizer`로 canonical `LogicalScan`에 접은 뒤 기존
  vectorized scan pipeline을 생성한다.
- 실행 연산자는 교체하지 않았으므로 row-at-a-time 회귀가 없다.
- `DODAM_OPTIMIZER_TRACE=1`로 적용된 logical rule 목록을 볼 수 있다.

### 4. 최근 성능 개선 이력

- `35a7895`: SF100 경계 밖에서도 dense lookup fast path 유지. Q09/Q11 개선.
- `3755b80`: Q09/Q11 개선 수치 기록.
- `ef149d6`: README의 SF100/SF200 결과를 하나의 표로 통합.
- `9570750`: Q06의 불리한 late-materialization 시도 제거, Q22 atomic dense-set build.
- `ce00aff`: 범용 logical/physical optimizer 첫 vertical slice.

전체 22개 공식 결과와 측정 조건은 `README.md`의
`TPC-H SF100 and SF200` 절을 기준으로 한다.

## 벤치마크 기준과 현재 수치

### 공식 전체 결과

- SF100: Dodam 54,081.856 ms, DuckDB 79,002.053 ms, `0.685x`
- SF200: Dodam 141,009.691 ms, DuckDB 175,734.845 ms, `0.802x`
- SF100은 5회 순차 실행 후 첫 1회를 버린 4회 중앙값이다.
- SF200 공식 표는 30 GiB 호스트의 세션 종료 문제 때문에 쿼리별 별도 프로세스,
  1회 실행, warmup 없음이다.
- SF200 DuckDB Q09는 기본 설정에서 약 23.5 GiB RSS 후 OOM으로 종료되어
  12 GiB memory limit, 16 threads, spill 설정으로 재측정했다.

### optimizer 변경 후 SF100 집중 회귀 측정

동일한 5회, warmup 1, batch size 16,384 조건이다.

| Query | Dodam | DuckDB | Dodam / DuckDB |
|---|---:|---:|---:|
| Q06 | 1,010.601 ms | 1,706.408 ms | 0.592x |
| Q09 | 6,861.022 ms | 9,176.985 ms | 0.748x |

- Q06 결과의 상대 오차는 약 `3e-16`이었다.
- Q09는 175개 key가 모두 일치했고 최대 상대 수치 오차는 `2.43e-15`였다.
- 보고서 위치:
  - `/tmp/dodam-optimizer-sf100-q06/report.json`
  - `/tmp/dodam-optimizer-sf100-q09/report.json`

### SF200 Q06/Q22 해석 시 주의

공식 SF200 표에서 Q06/Q22의 격차가 작아진 주된 이유는 한 번만 실행한 cold
프로세스 측정 방식이다. 같은 데이터를 warm 조건으로 다시 확인한 분석값은 다음과
같았지만 README 공식 표를 대체하지는 않았다.

| Query | SF200 warm Dodam | SF200 warm DuckDB | ratio |
|---|---:|---:|---:|
| Q06 | 2,248.649 ms | 3,981.209 ms | 0.565x |
| Q22 | 1,436.037 ms | 2,101.387 ms | 0.683x |

따라서 SF100과 SF200의 scaling만 비교하려면 동일한 warmup/process protocol로
다시 측정해야 한다.

## 검증 완료 항목

모든 테스트는 개별로 순차 실행했다.

- `cargo check`
- logical rewrite와 limit/projection 의미 보존 장벽 테스트
- 작은 join input build 선택 테스트
- 메모리 초과 시 partitioned hash 선택 테스트
- 명시적 partitioned join exchange 테스트
- hidden filter column을 포함한 declarative physical scan 실행 테스트
- scan explain, distinct, logical/physical join descriptor 테스트
- local hash repartition execution graph 테스트
- DuckDB differential:
  - TPC-H-lite
  - scan/filter/null
  - low-memory join

`cargo clippy --all-targets -- -D warnings`는 현재 저장소에 이미 존재하는 약 327개
경고 때문에 통과하지 않는다. 이번 optimizer 변경에서 새 clippy 경고는 확인되지
않았다.

## 다음 작업 우선순위

### P0: optimizer 통계와 비용 모델 일반화

1. `PlanTableSource`에 schema/컬럼 통계를 전달한다.
2. projection별 row width, filter selectivity, NDV, min/max를 공용
   `PlanEstimate`로 계산한다.
3. 현재 SQL 모듈에서 따로 사용하는 `LogicalJoinGraph`를 실제 `LogicalPlan::Join`
   reorder에 연결한다.
4. 6개 이하 join은 exhaustive/bushy, 그 이상은 greedy로 제한한다.
5. outer/semi join의 reorder 가능 범위를 의미적으로 제한한다.

### P1: physical operator 선택 확대

1. aggregate cardinality와 메모리 예산으로 dense/hash/partitioned aggregate를 고른다.
2. sort/top-N의 입력 크기와 limit 비율로 in-memory/external 전략을 고른다.
3. hash join의 실제 uncompressed build 크기와 spill 비용을 추정한다.
4. exchange 비용과 partition 수를 CPU 수가 아닌 데이터 크기 중심으로 정한다.
5. 같은 scan을 공유하는 subquery/aggregate가 chunk를 재독하지 않도록 공용 pipeline
   또는 materialization cost를 비교한다.

### P2: SQL 경로 통합과 관측성

1. `sql/rule_registry.rs`의 shape rule을 logical/physical alternatives로 표현한다.
2. query-specific fast path는 일반 cost model의 후보 중 하나로 유지한다.
3. `EXPLAIN`에 before/after logical plan, 적용 rule, rows/bytes/cardinality 추정을 넣는다.
4. 추정치와 실제 metrics 차이를 기록해 cost model을 보정한다.
5. Q01/Q02/Q05/Q12/Q20 등 SF200에서 DuckDB 대비 약한 쿼리를 일반 규칙 관점에서
   profile한다.

## 바로 이어서 실행할 명령

SF100 데이터는 현재 `/tmp/dodam-tpchgen-sf100`에 있다.

```sh
cargo build --release --bin tpch_real_inprocess

python3 scripts/compare_tpch_duckdb.py \
  --data-dir /tmp/dodam-tpchgen-sf100 \
  --output-dir /tmp/dodam-optimizer-next-qXX \
  --duckdb-mode cli \
  --repeats 5 \
  --warmup 1 \
  --batch-size 16384 \
  --timeout 180 \
  --only qXX \
  --json-out /tmp/dodam-optimizer-next-qXX/report.json
```

테스트는 다음처럼 하나씩 실행한다.

```sh
cargo test optimizer::tests::TEST_NAME -- --exact
cargo test plan::tests::TEST_NAME -- --exact
cargo test --test sql_query TEST_NAME -- --exact
cargo test --test duckdb_differential TEST_NAME -- --exact
```

## 작업 시작 체크리스트

1. `git status --short --branch`로 사용자 파일과 미추적 파일을 확인한다.
2. `work.md`, `AGENTS.md`, 관련 logical/physical operator를 먼저 읽는다.
3. query-specific 패치 전에 여러 SQL에 적용 가능한 비용/규칙인지 확인한다.
4. correctness test를 개별 실행한다.
5. 새 SQL 또는 성능 변경은 SF100에서 DuckDB와 같은 protocol로 단독 측정한다.
6. 결과와 남은 판단을 이 문서에 갱신한다.
