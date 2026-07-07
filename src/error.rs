use std::path::PathBuf;

pub type Result<T> = std::result::Result<T, DodamError>;

#[derive(Debug, thiserror::Error)]
pub enum DodamError {
    #[error("path does not exist: {0}")]
    MissingPath(PathBuf),

    #[error("unsupported table path: {0}")]
    UnsupportedTablePath(PathBuf),

    #[error("unsupported storage location: {0}")]
    UnsupportedStorageLocation(String),

    #[error("unsupported storage format: {0}")]
    UnsupportedStorageFormat(String),

    #[error("unknown column in projection: {0}")]
    UnknownColumn(String),

    #[error("ambiguous column {0}")]
    AmbiguousColumn(String),

    #[error("unknown table qualifier: {0}")]
    UnknownTableQualifier(String),

    #[error("invalid cast: {0}")]
    InvalidCast(String),

    #[error("type mismatch: {0}")]
    TypeMismatch(String),

    #[error("invalid filter expression: expected column=value, got {0}")]
    InvalidFilter(String),

    #[error("invalid aggregate expression: {0}")]
    InvalidAggregate(String),

    #[error("invalid order by expression: {0}")]
    InvalidOrderBy(String),

    #[error("unsupported SQL: {0}")]
    UnsupportedSql(String),

    #[error("unsupported filter column type for {column}: {data_type}")]
    UnsupportedFilterType {
        column: String,
        data_type: arrow::datatypes::DataType,
    },

    #[error("unsupported aggregate column type for {function}({column}): {data_type}")]
    UnsupportedAggregateType {
        function: String,
        column: String,
        data_type: arrow::datatypes::DataType,
    },

    #[error("unsupported group by column type for {column}: {data_type}")]
    UnsupportedGroupByType {
        column: String,
        data_type: arrow::datatypes::DataType,
    },

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("parquet error: {0}")]
    Parquet(#[from] parquet::errors::ParquetError),

    #[error("arrow error: {0}")]
    Arrow(#[from] arrow::error::ArrowError),
}
