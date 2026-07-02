#!/usr/bin/env python3
import argparse
import re
import subprocess
import time
from pathlib import Path


TABLES = {
    "lineitem",
    "orders",
    "customer",
    "supplier",
    "partsupp",
    "part",
    "nation",
    "region",
}

KEYWORDS = {
    "and",
    "as",
    "cross",
    "from",
    "full",
    "group",
    "having",
    "inner",
    "join",
    "left",
    "limit",
    "on",
    "or",
    "order",
    "outer",
    "right",
    "select",
    "semi",
    "where",
}

TABLE_REF = re.compile(
    r"(?i)(\bFROM\s+|\bJOIN\s+|,\s*)"
    r"(lineitem|orders|customer|supplier|partsupp|part|nation|region)"
    r"(?:\s+(?:AS\s+)?([A-Za-z_][A-Za-z0-9_]*))?"
)


def load_tpch_queries(path: Path) -> list[tuple[str, str]]:
    source = path.read_text()
    return re.findall(
        r'name: "([^"]+)",\n\s+expected_status: "[^"]+",\n\s+sql: r#"(.*?)"#,',
        source,
        re.S,
    )


def rewrite_table_refs(sql: str, data_dir: Path) -> str:
    def replace(match: re.Match[str]) -> str:
        prefix = match.group(1)
        table = match.group(2).lower()
        alias = match.group(3)
        table_path = data_dir / f"{table}.parquet"
        if alias and alias.lower() in KEYWORDS:
            return f"{prefix}'{table_path}' AS {table} {alias}"
        return f"{prefix}'{table_path}' AS {alias or table}"

    return TABLE_REF.sub(replace, sql.strip())


def summarize_success(stdout: str) -> str:
    lines = [line for line in stdout.strip().splitlines() if line.strip()]
    return lines[-1][:180] if lines else ""


def summarize_failure(stdout: str, stderr: str) -> str:
    text = stderr.strip() or stdout.strip()
    lines = [line for line in text.splitlines() if line.strip()]
    return lines[-1][:260] if lines else ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Run canonical TPC-H SQL over parquet data with Dodam.")
    parser.add_argument("--data-dir", default="/tmp/dodam-tpchgen-sf0_01")
    parser.add_argument("--dodam", default="target/debug/dodam")
    parser.add_argument("--queries", default="tests/tpch_coverage.rs")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--show-sql", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    missing = [table for table in sorted(TABLES) if not (data_dir / f"{table}.parquet").exists()]
    if missing:
        raise SystemExit(f"missing parquet table(s) under {data_dir}: {', '.join(missing)}")

    results = []
    for name, sql in load_tpch_queries(Path(args.queries)):
        rewritten = rewrite_table_refs(sql, data_dir)
        if args.show_sql:
            print(f"\n-- {name}\n{rewritten}\n")
        started = time.time()
        try:
            completed = subprocess.run(
                [args.dodam, "query", rewritten],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=args.timeout,
            )
            elapsed = time.time() - started
            if completed.returncode == 0:
                results.append((name, "ok", elapsed, summarize_success(completed.stdout)))
            else:
                results.append(
                    (
                        name,
                        "fail",
                        elapsed,
                        summarize_failure(completed.stdout, completed.stderr),
                    )
                )
        except subprocess.TimeoutExpired:
            elapsed = time.time() - started
            results.append((name, "timeout", elapsed, f"timed out after {args.timeout:.1f}s"))

    ok = sum(1 for _, status, _, _ in results if status == "ok")
    print(f"TPC-H real-data over {data_dir}: {ok}/{len(results)} ok")
    for name, status, elapsed, detail in results:
        print(f"{name}: {status} {elapsed:.3f}s {detail}")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
