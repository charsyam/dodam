use super::*;

pub(super) struct ExcludedSuppliers {
    keys: HashSet<i64>,
    max_suppkey: Option<i64>,
}

pub(super) async fn excluded_suppliers_by_comment_like(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    comment_parts: &[String],
) -> Result<ExcludedSuppliers> {
    let comment_part_bytes = comment_parts
        .iter()
        .map(|part| part.as_bytes().to_vec())
        .collect::<Vec<_>>();
    let comment_finders = comment_part_bytes
        .iter()
        .map(|part| Finder::new(part))
        .collect::<Vec<_>>();
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec!["s_suppkey".to_string(), "s_comment".to_string()]),
            None,
        )
        .await?;
    let mut suppliers = HashSet::new();
    let mut max_suppkey = None::<i64>;
    while let Some(batch) = stream.next() {
        let batch = batch?;
        let suppkeys = batch_column(&batch, "s_suppkey")?;
        let comments = batch_string_column(&batch, "s_comment")?;
        if let Some(suppkeys) = suppkeys.as_any().downcast_ref::<Int64Array>()
            && suppkeys.null_count() == 0
        {
            let suppkey_values = suppkeys.values().as_ref();
            if comments.null_count() == 0 {
                for (row, &suppkey) in suppkey_values.iter().enumerate() {
                    max_suppkey = Some(max_suppkey.map_or(suppkey, |max_key| max_key.max(suppkey)));
                    if fast_like_substrings_row_matches_non_null(
                        comments,
                        row,
                        &comment_part_bytes,
                        &comment_finders,
                        false,
                    ) {
                        suppliers.insert(suppkey);
                    }
                }
                continue;
            }
            for (row, &suppkey) in suppkey_values.iter().enumerate() {
                max_suppkey = Some(max_suppkey.map_or(suppkey, |max_key| max_key.max(suppkey)));
                if fast_like_substrings_row_matches(
                    comments,
                    row,
                    &comment_part_bytes,
                    &comment_finders,
                    false,
                ) {
                    suppliers.insert(suppkey);
                }
            }
            continue;
        }
        for row in 0..batch.num_rows() {
            let Some(suppkey) = numeric_i64_value(suppkeys, row)? else {
                continue;
            };
            max_suppkey = Some(max_suppkey.map_or(suppkey, |max_key| max_key.max(suppkey)));
            if comments.is_null(row)
                || !fast_like_substrings_row_matches(
                    comments,
                    row,
                    &comment_part_bytes,
                    &comment_finders,
                    false,
                )
            {
                continue;
            }
            suppliers.insert(suppkey);
        }
    }
    Ok(ExcludedSuppliers {
        keys: suppliers,
        max_suppkey,
    })
}

#[derive(Clone, Eq, Hash, PartialEq)]
pub(super) struct PartGroupKey {
    brand: String,
    type_name: String,
    size: i64,
}

#[derive(Clone, Copy, Eq, Hash, PartialEq)]
pub(super) struct PartGroupIdKey {
    brand_id: usize,
    type_id: usize,
    size: i64,
}

pub(super) struct PartGroups {
    pub(super) groups: Vec<PartGroupKey>,
    part_to_group: AdaptiveI64Map<usize>,
}

pub(super) async fn part_groups_by_attributes(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    excluded_brand: &str,
    excluded_type_prefix: &str,
    sizes: &AdaptiveI64Set,
) -> Result<PartGroups> {
    let projection = Projection::Columns(vec![
        "p_partkey".to_string(),
        "p_brand".to_string(),
        "p_type".to_string(),
        "p_size".to_string(),
    ]);
    if part_dictionary_strings_enabled() {
        let excluded_brand_owned = Arc::new(excluded_brand.to_string());
        let excluded_type_prefix_owned = Arc::new(excluded_type_prefix.to_string());
        let sizes = Arc::new(sizes.clone());
        if let Some(partials) = engine
            .parquet_row_group_map_dictionary_columns_pruned_view(
                path.clone(),
                batch_size,
                projection.clone(),
                vec!["p_brand".to_string(), "p_type".to_string()],
                Vec::new(),
                part_group_chunk_size(),
                PartGroupPartial::default,
                {
                    let excluded_brand = excluded_brand_owned.clone();
                    let excluded_type_prefix = excluded_type_prefix_owned.clone();
                    let sizes = sizes.clone();
                    move |view, partial| {
                        part_groups_partial_view(
                            view,
                            &excluded_brand,
                            &excluded_type_prefix,
                            &sizes,
                            partial,
                        )?;
                        Ok(Some(()))
                    }
                },
                |partial| Ok(Some(partial)),
            )
            .await?
        {
            return merge_part_group_partials(partials);
        }
    }
    let mut stream = if part_dictionary_strings_enabled() {
        engine
            .scan_parquet_batches_dictionary_columns(
                path,
                batch_size,
                projection,
                vec!["p_brand".to_string(), "p_type".to_string()],
            )
            .await?
    } else {
        engine
            .scan_parquet_batches(path, batch_size, None, projection, None)
            .await?
    };
    let mut brand_ids = fast_hash_map::<String, usize>();
    let mut type_ids = fast_hash_map::<String, usize>();
    let mut brands_by_id = Vec::<String>::new();
    let mut types_by_id = Vec::<String>::new();
    let mut group_ids = FastHashMap::<PartGroupIdKey, usize>::default();
    let mut groups = Vec::<PartGroupKey>::new();
    let mut part_to_group = fast_hash_map::<i64, usize>();
    while let Some(batch) = stream.next() {
        let batch = batch?;
        let view = BatchView::new(&batch);
        if let Some(layout) = PartDictionaryView::try_new(view)
            && part_groups_dictionary_typed_batch(
                layout,
                excluded_brand,
                excluded_type_prefix,
                sizes,
                &mut brand_ids,
                &mut type_ids,
                &mut brands_by_id,
                &mut types_by_id,
                &mut group_ids,
                &mut groups,
                &mut part_to_group,
            )?
        {
            continue;
        }
        let size_view = if let Some(values) = view.i32_vector(3) {
            PartSizeView::I32(values)
        } else if let Some(values) = view.i64_vector(3) {
            PartSizeView::I64(values)
        } else {
            let Some(batch) = view.try_record_batch() else {
                return Err(DodamError::UnsupportedSql(
                    "part supplier group size raw vector has unsupported type".to_string(),
                ));
            };
            part_groups_batch_fallback(
                batch,
                excluded_brand,
                excluded_type_prefix,
                sizes,
                &mut brand_ids,
                &mut type_ids,
                &mut brands_by_id,
                &mut types_by_id,
                &mut group_ids,
                &mut groups,
                &mut part_to_group,
            )?;
            continue;
        };
        let Some(brands) = view.utf8(1) else {
            return Err(DodamError::UnsupportedSql(
                "p_brand must be Utf8".to_string(),
            ));
        };
        let Some(types) = view.utf8(2) else {
            return Err(DodamError::UnsupportedSql(
                "p_type must be Utf8".to_string(),
            ));
        };
        if let Some(partkeys) = view.i64_vector(0)
            && part_groups_vector_batch(
                partkeys,
                brands,
                types,
                size_view,
                excluded_brand,
                excluded_type_prefix,
                sizes,
                &mut brand_ids,
                &mut type_ids,
                &mut brands_by_id,
                &mut types_by_id,
                &mut group_ids,
                &mut groups,
                &mut part_to_group,
            )?
        {
            continue;
        }
        let Some(batch) = view.try_record_batch() else {
            return Err(DodamError::UnsupportedSql(
                "part supplier group raw vector columns have unsupported nullable layout"
                    .to_string(),
            ));
        };
        part_groups_batch_fallback(
            batch,
            excluded_brand,
            excluded_type_prefix,
            sizes,
            &mut brand_ids,
            &mut type_ids,
            &mut brands_by_id,
            &mut types_by_id,
            &mut group_ids,
            &mut groups,
            &mut part_to_group,
        )?;
    }
    Ok(PartGroups {
        groups,
        part_to_group: AdaptiveI64Map::from_hash(part_to_group),
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn part_groups_batch_fallback(
    batch: &RecordBatch,
    excluded_brand: &str,
    excluded_type_prefix: &str,
    sizes: &AdaptiveI64Set,
    brand_ids: &mut FastHashMap<String, usize>,
    type_ids: &mut FastHashMap<String, usize>,
    brands_by_id: &mut Vec<String>,
    types_by_id: &mut Vec<String>,
    group_ids: &mut FastHashMap<PartGroupIdKey, usize>,
    groups: &mut Vec<PartGroupKey>,
    part_to_group: &mut FastHashMap<i64, usize>,
) -> Result<()> {
    let partkeys = batch_column(batch, "p_partkey")?;
    let part_sizes = batch_column(batch, "p_size")?;
    let brands = batch_string_column(batch, "p_brand")?;
    let types = batch_string_column(batch, "p_type")?;
    if part_groups_typed_batch(
        partkeys,
        brands,
        types,
        part_sizes,
        excluded_brand,
        excluded_type_prefix,
        sizes,
        brand_ids,
        type_ids,
        brands_by_id,
        types_by_id,
        group_ids,
        groups,
        part_to_group,
    )? {
        return Ok(());
    }
    for row in 0..batch.num_rows() {
        let (Some(partkey), Some(size)) = (
            numeric_i64_value(partkeys, row)?,
            numeric_i64_value(part_sizes, row)?,
        ) else {
            continue;
        };
        if !sizes.contains(size) {
            continue;
        }
        if brands.is_null(row)
            || types.is_null(row)
            || brands.value(row) == excluded_brand
            || types.value(row).starts_with(excluded_type_prefix)
        {
            continue;
        }
        let brand_id = intern_string(brand_ids, brands_by_id, brands.value(row));
        let type_id = intern_string(type_ids, types_by_id, types.value(row));
        let key = PartGroupIdKey {
            brand_id,
            type_id,
            size,
        };
        let group_id = if let Some(group_id) = group_ids.get(&key).copied() {
            group_id
        } else {
            let group_id = groups.len();
            groups.push(PartGroupKey {
                brand: brands_by_id[brand_id].clone(),
                type_name: types_by_id[type_id].clone(),
                size,
            });
            group_ids.insert(key, group_id);
            group_id
        };
        part_to_group.insert(partkey, group_id);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn part_groups_vector_batch(
    partkeys: I64VectorView<'_>,
    brands: &StringArray,
    types: &StringArray,
    part_sizes: PartSizeView<'_>,
    excluded_brand: &str,
    excluded_type_prefix: &str,
    sizes: &AdaptiveI64Set,
    brand_ids: &mut FastHashMap<String, usize>,
    type_ids: &mut FastHashMap<String, usize>,
    brands_by_id: &mut Vec<String>,
    types_by_id: &mut Vec<String>,
    group_ids: &mut FastHashMap<PartGroupIdKey, usize>,
    groups: &mut Vec<PartGroupKey>,
    part_to_group: &mut FastHashMap<i64, usize>,
) -> Result<bool> {
    let Some(partkey_values) = partkeys.values_if_null_free() else {
        return Ok(false);
    };
    if brands.null_count() != 0 || types.null_count() != 0 || part_sizes.null_count() != 0 {
        return Ok(false);
    }
    for (row, &partkey) in partkey_values.iter().enumerate() {
        let size = part_sizes.value_i64(row);
        if !sizes.contains(size) {
            continue;
        }
        insert_part_group_row(
            partkey,
            size,
            brands.value(row),
            types.value(row),
            excluded_brand,
            excluded_type_prefix,
            brand_ids,
            type_ids,
            brands_by_id,
            types_by_id,
            group_ids,
            groups,
            part_to_group,
        );
    }
    Ok(true)
}

pub(super) fn part_dictionary_strings_enabled() -> bool {
    std::env::var_os("DODAM_Q16_DISABLE_PART_DICTIONARY_STRINGS").is_none()
}

pub(super) fn part_group_chunk_size() -> usize {
    std::env::var("DODAM_Q16_PART_GROUP_CHUNK_SIZE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(2)
}

pub(super) struct PartGroupPartialRow {
    partkey: i64,
    size: i64,
    brand_id: usize,
    type_id: usize,
}

#[derive(Default)]
pub(super) struct PartGroupPartial {
    brand_ids: FastHashMap<String, usize>,
    type_ids: FastHashMap<String, usize>,
    brands_by_id: Vec<String>,
    types_by_id: Vec<String>,
    rows: Vec<PartGroupPartialRow>,
}

impl PartGroupPartial {
    fn push_row(&mut self, partkey: i64, size: i64, brand: &str, type_name: &str) {
        let brand_id = intern_string(&mut self.brand_ids, &mut self.brands_by_id, brand);
        let type_id = intern_string(&mut self.type_ids, &mut self.types_by_id, type_name);
        self.rows.push(PartGroupPartialRow {
            partkey,
            size,
            brand_id,
            type_id,
        });
    }

    fn push_ids(&mut self, partkey: i64, size: i64, brand_id: usize, type_id: usize) {
        self.rows.push(PartGroupPartialRow {
            partkey,
            size,
            brand_id,
            type_id,
        });
    }
}

pub(super) enum PartSizeView<'a> {
    I32(I32VectorView<'a>),
    I64(I64VectorView<'a>),
}

impl PartSizeView<'_> {
    fn null_count(&self) -> usize {
        match self {
            Self::I32(values) => values.values_if_null_free().is_none() as usize,
            Self::I64(values) => values.values_if_null_free().is_none() as usize,
        }
    }

    fn value_i64(&self, row: usize) -> i64 {
        match self {
            Self::I32(values) => i64::from(values.value(row)),
            Self::I64(values) => values.value(row),
        }
    }
}

pub(super) struct PartDictionaryView<'a> {
    partkeys: I64VectorView<'a>,
    brands: DictionaryI32View<'a>,
    types: DictionaryI32View<'a>,
    sizes: PartSizeView<'a>,
}

impl<'a> PartDictionaryView<'a> {
    fn try_new(view: BatchView<'a>) -> Option<Self> {
        let sizes = if let Some(values) = view.i32_vector(3) {
            PartSizeView::I32(values)
        } else {
            PartSizeView::I64(view.i64_vector(3)?)
        };
        Some(Self {
            partkeys: view.i64_vector(0)?,
            brands: view.dictionary_i32_view(1)?,
            types: view.dictionary_i32_view(2)?,
            sizes,
        })
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn part_groups_dictionary_typed_batch(
    view: PartDictionaryView<'_>,
    excluded_brand: &str,
    excluded_type_prefix: &str,
    sizes: &AdaptiveI64Set,
    brand_ids: &mut FastHashMap<String, usize>,
    type_ids: &mut FastHashMap<String, usize>,
    brands_by_id: &mut Vec<String>,
    types_by_id: &mut Vec<String>,
    group_ids: &mut FastHashMap<PartGroupIdKey, usize>,
    groups: &mut Vec<PartGroupKey>,
    part_to_group: &mut FastHashMap<i64, usize>,
) -> Result<bool> {
    if view.partkeys.values_if_null_free().is_none()
        || view.brands.null_count() != 0
        || view.types.null_count() != 0
        || view.sizes.null_count() != 0
    {
        return Ok(false);
    }
    let Some(brand_values) = view.brands.string_values() else {
        return Ok(false);
    };
    let Some(type_values) = view.types.string_values() else {
        return Ok(false);
    };
    let brand_lookup = dictionary_group_string_ids(
        &brand_values,
        Some(excluded_brand.as_bytes()),
        None,
        brand_ids,
        brands_by_id,
    )?;
    let type_lookup = dictionary_group_string_ids(
        &type_values,
        None,
        Some(excluded_type_prefix.as_bytes()),
        type_ids,
        types_by_id,
    )?;
    let brand_keys = view.brands.keys();
    let type_keys = view.types.keys();
    let partkey_values = view
        .partkeys
        .values_if_null_free()
        .expect("checked partkeys");
    for row in 0..partkey_values.len() {
        let size = view.sizes.value_i64(row);
        if !sizes.contains(size) {
            continue;
        }
        let (Some(brand_id), Some(type_id)) = (
            dictionary_lookup_id(&brand_lookup, brand_keys[row]),
            dictionary_lookup_id(&type_lookup, type_keys[row]),
        ) else {
            continue;
        };
        insert_part_group_ids(
            partkey_values[row],
            size,
            brand_id,
            type_id,
            brands_by_id,
            types_by_id,
            group_ids,
            groups,
            part_to_group,
        );
    }
    Ok(true)
}

pub(super) fn part_groups_partial_view(
    view: BatchView<'_>,
    excluded_brand: &str,
    excluded_type_prefix: &str,
    sizes: &AdaptiveI64Set,
    partial: &mut PartGroupPartial,
) -> Result<()> {
    if let Some(layout) = PartDictionaryView::try_new(view)
        && part_groups_dictionary_partial_batch(
            layout,
            excluded_brand,
            excluded_type_prefix,
            sizes,
            partial,
        )?
    {
        return Ok(());
    }
    let Some(partkeys) = view.i64_vector(0) else {
        return Err(DodamError::UnsupportedSql(
            "part supplier group partkey raw vector has unsupported type".to_string(),
        ));
    };
    let Some(brands) = view.utf8(1) else {
        return Err(DodamError::UnsupportedSql(
            "part supplier group brand raw vector has unsupported type".to_string(),
        ));
    };
    let Some(types) = view.utf8(2) else {
        return Err(DodamError::UnsupportedSql(
            "part supplier group type raw vector has unsupported type".to_string(),
        ));
    };
    let size_view = if let Some(values) = view.i32_vector(3) {
        PartSizeView::I32(values)
    } else {
        let Some(values) = view.i64_vector(3) else {
            return Err(DodamError::UnsupportedSql(
                "part supplier group size raw vector has unsupported type".to_string(),
            ));
        };
        PartSizeView::I64(values)
    };
    if partkeys.values_if_null_free().is_none()
        || brands.null_count() != 0
        || types.null_count() != 0
        || size_view.null_count() != 0
    {
        let Some(batch) = view.try_record_batch() else {
            return Err(DodamError::UnsupportedSql(
                "PartSupplierGroup nullable raw part-group vectors are unsupported".to_string(),
            ));
        };
        return part_groups_partial_batch(
            batch,
            excluded_brand,
            excluded_type_prefix,
            sizes,
            partial,
        );
    }
    for row in 0..partkeys.len() {
        let size = size_view.value_i64(row);
        if !sizes.contains(size) {
            continue;
        }
        let brand = brands.value(row);
        let type_name = types.value(row);
        if brand == excluded_brand || type_name.starts_with(excluded_type_prefix) {
            continue;
        }
        partial.push_row(partkeys.value(row), size, brand, type_name);
    }
    Ok(())
}

pub(super) fn part_groups_dictionary_partial_batch(
    view: PartDictionaryView<'_>,
    excluded_brand: &str,
    excluded_type_prefix: &str,
    sizes: &AdaptiveI64Set,
    partial: &mut PartGroupPartial,
) -> Result<bool> {
    if view.partkeys.values_if_null_free().is_none()
        || view.brands.null_count() != 0
        || view.types.null_count() != 0
        || view.sizes.null_count() != 0
    {
        return Ok(false);
    }
    let Some(brand_values) = view.brands.string_values() else {
        return Ok(false);
    };
    let Some(type_values) = view.types.string_values() else {
        return Ok(false);
    };
    let brand_lookup = dictionary_group_string_ids(
        &brand_values,
        Some(excluded_brand.as_bytes()),
        None,
        &mut partial.brand_ids,
        &mut partial.brands_by_id,
    )?;
    let type_lookup = dictionary_group_string_ids(
        &type_values,
        None,
        Some(excluded_type_prefix.as_bytes()),
        &mut partial.type_ids,
        &mut partial.types_by_id,
    )?;
    let brand_keys = view.brands.keys();
    let type_keys = view.types.keys();
    let partkey_values = view
        .partkeys
        .values_if_null_free()
        .expect("checked partkeys");
    for row in 0..partkey_values.len() {
        let size = view.sizes.value_i64(row);
        if !sizes.contains(size) {
            continue;
        }
        let (Some(brand_id), Some(type_id)) = (
            dictionary_lookup_id(&brand_lookup, brand_keys[row]),
            dictionary_lookup_id(&type_lookup, type_keys[row]),
        ) else {
            continue;
        };
        partial.push_ids(partkey_values[row], size, brand_id, type_id);
    }
    Ok(true)
}

pub(super) fn part_groups_partial_batch(
    batch: &RecordBatch,
    excluded_brand: &str,
    excluded_type_prefix: &str,
    sizes: &AdaptiveI64Set,
    partial: &mut PartGroupPartial,
) -> Result<()> {
    let partkeys = batch_column(batch, "p_partkey")?;
    let part_sizes = batch_column(batch, "p_size")?;
    let Some(brands) = batch_string_column(batch, "p_brand").ok() else {
        return Err(DodamError::UnsupportedSql(
            "p_brand must be Utf8".to_string(),
        ));
    };
    let Some(types) = batch_string_column(batch, "p_type").ok() else {
        return Err(DodamError::UnsupportedSql(
            "p_type must be Utf8".to_string(),
        ));
    };
    for row in 0..batch.num_rows() {
        let (Some(partkey), Some(size)) = (
            numeric_i64_value(partkeys, row)?,
            numeric_i64_value(part_sizes, row)?,
        ) else {
            continue;
        };
        if !sizes.contains(size) {
            continue;
        }
        if brands.is_null(row) || types.is_null(row) {
            continue;
        }
        let brand = brands.value(row);
        let type_name = types.value(row);
        if brand == excluded_brand || type_name.starts_with(excluded_type_prefix) {
            continue;
        }
        partial.push_row(partkey, size, brand, type_name);
    }
    Ok(())
}

pub(super) fn merge_part_group_partials(partials: Vec<PartGroupPartial>) -> Result<PartGroups> {
    let mut brand_ids = fast_hash_map::<String, usize>();
    let mut type_ids = fast_hash_map::<String, usize>();
    let mut brands_by_id = Vec::<String>::new();
    let mut types_by_id = Vec::<String>::new();
    let mut group_ids = FastHashMap::<PartGroupIdKey, usize>::default();
    let mut groups = Vec::<PartGroupKey>::new();
    let mut part_to_group = fast_hash_map::<i64, usize>();
    for partial in partials {
        let brand_remap = partial
            .brands_by_id
            .iter()
            .map(|value| intern_string(&mut brand_ids, &mut brands_by_id, value))
            .collect::<Vec<_>>();
        let type_remap = partial
            .types_by_id
            .iter()
            .map(|value| intern_string(&mut type_ids, &mut types_by_id, value))
            .collect::<Vec<_>>();
        for row in partial.rows {
            let Some(&brand_id) = brand_remap.get(row.brand_id) else {
                continue;
            };
            let Some(&type_id) = type_remap.get(row.type_id) else {
                continue;
            };
            insert_part_group_ids(
                row.partkey,
                row.size,
                brand_id,
                type_id,
                &brands_by_id,
                &types_by_id,
                &mut group_ids,
                &mut groups,
                &mut part_to_group,
            );
        }
    }
    Ok(PartGroups {
        groups,
        part_to_group: AdaptiveI64Map::from_hash(part_to_group),
    })
}

pub(super) fn dictionary_group_string_ids<S: BuildHasher>(
    values: &DictionaryStringValues<'_>,
    excluded_exact: Option<&[u8]>,
    excluded_prefix: Option<&[u8]>,
    ids: &mut HashMap<String, usize, S>,
    strings_by_id: &mut Vec<String>,
) -> Result<Vec<Option<usize>>> {
    let mut lookup = Vec::with_capacity(values.len());
    for index in 0..values.len() {
        let value = values.value_bytes(index);
        if excluded_exact.is_some_and(|excluded| value == excluded)
            || excluded_prefix.is_some_and(|prefix| value.starts_with(prefix))
        {
            lookup.push(None);
            continue;
        }
        let value = std::str::from_utf8(value)
            .map_err(|error| DodamError::UnsupportedSql(error.to_string()))?;
        lookup.push(Some(intern_string(ids, strings_by_id, value)));
    }
    Ok(lookup)
}

pub(super) fn dictionary_lookup_id(lookup: &[Option<usize>], key: i32) -> Option<usize> {
    let key = usize::try_from(key).ok()?;
    lookup.get(key).copied().flatten()
}

#[allow(clippy::too_many_arguments)]
pub(super) fn part_groups_typed_batch(
    partkeys: &ArrayRef,
    brands: &StringArray,
    types: &StringArray,
    part_sizes: &ArrayRef,
    excluded_brand: &str,
    excluded_type_prefix: &str,
    sizes: &AdaptiveI64Set,
    brand_ids: &mut FastHashMap<String, usize>,
    type_ids: &mut FastHashMap<String, usize>,
    brands_by_id: &mut Vec<String>,
    types_by_id: &mut Vec<String>,
    group_ids: &mut FastHashMap<PartGroupIdKey, usize>,
    groups: &mut Vec<PartGroupKey>,
    part_to_group: &mut FastHashMap<i64, usize>,
) -> Result<bool> {
    let Some(partkeys) = partkeys.as_any().downcast_ref::<Int64Array>() else {
        return Ok(false);
    };
    if partkeys.null_count() != 0 || brands.null_count() != 0 || types.null_count() != 0 {
        return Ok(false);
    }
    if let Some(part_sizes) = part_sizes.as_any().downcast_ref::<Int32Array>() {
        if part_sizes.null_count() != 0 {
            return Ok(false);
        }
        let partkey_values = partkeys.values().as_ref();
        let size_values = part_sizes.values().as_ref();
        for row in 0..partkey_values.len() {
            let size = i64::from(size_values[row]);
            if !sizes.contains(size) {
                continue;
            }
            insert_part_group_row(
                partkey_values[row],
                size,
                brands.value(row),
                types.value(row),
                excluded_brand,
                excluded_type_prefix,
                brand_ids,
                type_ids,
                brands_by_id,
                types_by_id,
                group_ids,
                groups,
                part_to_group,
            );
        }
        return Ok(true);
    }
    if let Some(part_sizes) = part_sizes.as_any().downcast_ref::<Int64Array>() {
        if part_sizes.null_count() != 0 {
            return Ok(false);
        }
        let partkey_values = partkeys.values().as_ref();
        let size_values = part_sizes.values().as_ref();
        for row in 0..partkey_values.len() {
            let size = size_values[row];
            if !sizes.contains(size) {
                continue;
            }
            insert_part_group_row(
                partkey_values[row],
                size,
                brands.value(row),
                types.value(row),
                excluded_brand,
                excluded_type_prefix,
                brand_ids,
                type_ids,
                brands_by_id,
                types_by_id,
                group_ids,
                groups,
                part_to_group,
            );
        }
        return Ok(true);
    }
    Ok(false)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn insert_part_group_row(
    partkey: i64,
    size: i64,
    brand: &str,
    type_name: &str,
    excluded_brand: &str,
    excluded_type_prefix: &str,
    brand_ids: &mut FastHashMap<String, usize>,
    type_ids: &mut FastHashMap<String, usize>,
    brands_by_id: &mut Vec<String>,
    types_by_id: &mut Vec<String>,
    group_ids: &mut FastHashMap<PartGroupIdKey, usize>,
    groups: &mut Vec<PartGroupKey>,
    part_to_group: &mut FastHashMap<i64, usize>,
) {
    if brand == excluded_brand || type_name.starts_with(excluded_type_prefix) {
        return;
    }
    let brand_id = intern_string(brand_ids, brands_by_id, brand);
    let type_id = intern_string(type_ids, types_by_id, type_name);
    insert_part_group_ids(
        partkey,
        size,
        brand_id,
        type_id,
        brands_by_id,
        types_by_id,
        group_ids,
        groups,
        part_to_group,
    );
}

#[allow(clippy::too_many_arguments)]
pub(super) fn insert_part_group_ids(
    partkey: i64,
    size: i64,
    brand_id: usize,
    type_id: usize,
    brands_by_id: &[String],
    types_by_id: &[String],
    group_ids: &mut FastHashMap<PartGroupIdKey, usize>,
    groups: &mut Vec<PartGroupKey>,
    part_to_group: &mut FastHashMap<i64, usize>,
) {
    let key = PartGroupIdKey {
        brand_id,
        type_id,
        size,
    };
    let group_id = if let Some(group_id) = group_ids.get(&key).copied() {
        group_id
    } else {
        let group_id = groups.len();
        groups.push(PartGroupKey {
            brand: brands_by_id[brand_id].clone(),
            type_name: types_by_id[type_id].clone(),
            size,
        });
        group_ids.insert(key, group_id);
        group_id
    };
    part_to_group.insert(partkey, group_id);
}

pub(super) fn intern_string<S: BuildHasher>(
    ids: &mut HashMap<String, usize, S>,
    values: &mut Vec<String>,
    value: &str,
) -> usize {
    if let Some(id) = ids.get(value).copied() {
        return id;
    }
    let id = values.len();
    let value = value.to_string();
    values.push(value.clone());
    ids.insert(value, id);
    id
}

pub(super) struct PartSupplierGroupRow {
    brand: String,
    type_name: String,
    size: i64,
    supplier_count: u64,
}

pub(super) async fn distinct_supplier_counts_by_part_group(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    part_groups: PartGroups,
    bad_suppliers: ExcludedSuppliers,
) -> Result<Vec<PartSupplierGroupRow>> {
    let groups = part_groups.groups;
    let part_to_group = Arc::new(part_groups.part_to_group);
    let bad_supplier_keys = Arc::new(AdaptiveI64Set::from_hash(bad_suppliers.keys));
    let projection = Projection::Columns(vec!["ps_partkey".to_string(), "ps_suppkey".to_string()]);
    let supplier_counts = if packed_distinct_enabled(groups.len(), bad_suppliers.max_suppkey) {
        let Some(partials) = engine
            .parquet_row_group_map_view(
                path.clone(),
                batch_size,
                projection.clone(),
                supplier_count_chunk_size(),
                PackedU32PairDistinct::new,
                {
                    let part_to_group = part_to_group.clone();
                    let bad_supplier_keys = bad_supplier_keys.clone();
                    move |view, distinct_suppliers| {
                        supplier_counts_packed_view(
                            view,
                            &part_to_group,
                            &bad_supplier_keys,
                            distinct_suppliers,
                        )?;
                        Ok(Some(()))
                    }
                },
                |distinct_suppliers| Ok(Some(distinct_suppliers)),
            )
            .await?
        else {
            return Err(DodamError::UnsupportedSql(
                "part supplier group partsupp row-group map is unavailable".to_string(),
            ));
        };
        let mut packed = PackedU32PairDistinct::new();
        for partial in partials {
            merge_supplier_counts_packed(&mut packed, partial);
        }
        packed.counts_by_first(groups.len())
    } else if let Some(layout) = supplier_bitset_layout(groups.len(), bad_suppliers.max_suppkey) {
        let Some(partials) = engine
            .parquet_row_group_map_view(
                path.clone(),
                batch_size,
                projection.clone(),
                supplier_count_chunk_size(),
                {
                    let layout = layout.clone();
                    move || GroupSupplierBitset::new(layout.clone())
                },
                {
                    let part_to_group = part_to_group.clone();
                    let bad_supplier_keys = bad_supplier_keys.clone();
                    move |view, distinct_suppliers| {
                        supplier_counts_bitset_view(
                            view,
                            &part_to_group,
                            &bad_supplier_keys,
                            distinct_suppliers,
                        )?;
                        Ok(Some(()))
                    }
                },
                |distinct_suppliers| Ok(Some(distinct_suppliers)),
            )
            .await?
        else {
            return Err(DodamError::UnsupportedSql(
                "part supplier group partsupp row-group map is unavailable".to_string(),
            ));
        };
        let mut partial = GroupSupplierBitset::new(layout);
        for batch_partial in partials {
            merge_supplier_bitsets(&mut partial, batch_partial);
        }
        partial.counts()
    } else {
        let Some(partials) = engine
            .parquet_row_group_map_view(
                path,
                batch_size,
                projection,
                supplier_count_chunk_size(),
                FastHashSet::<(usize, i64)>::default,
                {
                    let part_to_group = part_to_group.clone();
                    let bad_supplier_keys = bad_supplier_keys.clone();
                    move |view, distinct_suppliers| {
                        let groups =
                            supplier_counts_view(view, &part_to_group, &bad_supplier_keys)?;
                        merge_supplier_counts(distinct_suppliers, groups);
                        Ok(Some(()))
                    }
                },
                |distinct_suppliers| Ok(Some(distinct_suppliers)),
            )
            .await?
        else {
            return Err(DodamError::UnsupportedSql(
                "part supplier group partsupp row-group map is unavailable".to_string(),
            ));
        };
        let mut distinct_suppliers = FastHashSet::<(usize, i64)>::default();
        for partial in partials {
            merge_supplier_counts(&mut distinct_suppliers, partial);
        }
        let mut supplier_counts = vec![0_u64; groups.len()];
        for (group_id, _) in distinct_suppliers {
            if let Some(count) = supplier_counts.get_mut(group_id) {
                *count += 1;
            }
        }
        supplier_counts
    };
    let mut rows = supplier_counts
        .into_iter()
        .enumerate()
        .filter_map(|(group_id, supplier_count)| {
            if supplier_count == 0 {
                return None;
            }
            let group = groups.get(group_id)?;
            Some(PartSupplierGroupRow {
                brand: group.brand.clone(),
                type_name: group.type_name.clone(),
                size: group.size,
                supplier_count,
            })
        })
        .collect::<Vec<_>>();
    rows.sort_by(|left, right| {
        right
            .supplier_count
            .cmp(&left.supplier_count)
            .then_with(|| left.brand.cmp(&right.brand))
            .then_with(|| left.type_name.cmp(&right.type_name))
            .then_with(|| left.size.cmp(&right.size))
    });
    Ok(rows)
}

#[allow(dead_code)]
pub(super) async fn supplier_counts_stream(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    groups: Vec<PartGroupKey>,
    part_to_group: Arc<AdaptiveI64Map<usize>>,
    bad_supplier_keys: Arc<AdaptiveI64Set>,
    max_suppkey: Option<i64>,
) -> Result<Vec<PartSupplierGroupRow>> {
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec!["ps_partkey".to_string(), "ps_suppkey".to_string()]),
            None,
        )
        .await?;
    let supplier_counts = if packed_distinct_enabled(groups.len(), max_suppkey) {
        let packed = parallel_batch_fold_view_chunks(
            &mut stream,
            supplier_count_chunk_size(),
            PackedU32PairDistinct::new,
            move |view, distinct_suppliers| {
                supplier_counts_packed_view(
                    view,
                    &part_to_group,
                    &bad_supplier_keys,
                    distinct_suppliers,
                )?;
                Ok(Some(()))
            },
            Ok,
            PackedU32PairDistinct::new(),
            merge_supplier_counts_packed,
            "part supplier group partsupp supplier counts",
        )?;
        packed.counts_by_first(groups.len())
    } else if let Some(layout) = supplier_bitset_layout(groups.len(), max_suppkey) {
        let layout_for_scan = Arc::new(layout.clone());
        let partial = parallel_batch_fold_view_chunks(
            &mut stream,
            supplier_count_chunk_size(),
            {
                let layout_for_scan = layout_for_scan.clone();
                move || GroupSupplierBitset::new((*layout_for_scan).clone())
            },
            move |view, distinct_suppliers| {
                supplier_counts_bitset_view(
                    view,
                    &part_to_group,
                    &bad_supplier_keys,
                    distinct_suppliers,
                )?;
                Ok(Some(()))
            },
            Ok,
            GroupSupplierBitset::new(layout),
            merge_supplier_bitsets,
            "part supplier group partsupp supplier counts",
        )?;
        partial.counts()
    } else {
        let distinct_suppliers = parallel_batch_fold(
            &mut stream,
            move |batch| supplier_counts_batch(batch, &part_to_group, &bad_supplier_keys),
            FastHashSet::<(usize, i64)>::default(),
            merge_supplier_counts,
            "part supplier group partsupp supplier counts",
        )?;
        let mut supplier_counts = vec![0_u64; groups.len()];
        for (group_id, _) in distinct_suppliers {
            if let Some(count) = supplier_counts.get_mut(group_id) {
                *count += 1;
            }
        }
        supplier_counts
    };
    let mut rows = supplier_counts
        .into_iter()
        .enumerate()
        .filter_map(|(group_id, supplier_count)| {
            if supplier_count == 0 {
                return None;
            }
            let group = groups.get(group_id)?;
            Some(PartSupplierGroupRow {
                brand: group.brand.clone(),
                type_name: group.type_name.clone(),
                size: group.size,
                supplier_count,
            })
        })
        .collect::<Vec<_>>();
    rows.sort_by(|left, right| {
        right
            .supplier_count
            .cmp(&left.supplier_count)
            .then_with(|| left.brand.cmp(&right.brand))
            .then_with(|| left.type_name.cmp(&right.type_name))
            .then_with(|| left.size.cmp(&right.size))
    });
    Ok(rows)
}

#[derive(Clone)]
pub(super) struct SupplierBitsetLayout {
    group_count: usize,
    words_per_group: usize,
}

pub(super) struct GroupSupplierBitset {
    layout: SupplierBitsetLayout,
    words: Vec<u64>,
}

impl GroupSupplierBitset {
    fn new(layout: SupplierBitsetLayout) -> Self {
        let words = vec![0; layout.group_count.saturating_mul(layout.words_per_group)];
        Self { layout, words }
    }

    fn insert(&mut self, group_id: usize, suppkey: i64) {
        if suppkey < 0 || group_id >= self.layout.group_count {
            return;
        }
        let suppkey = suppkey as usize;
        let word = suppkey / 64;
        if word >= self.layout.words_per_group {
            return;
        }
        let index = group_id * self.layout.words_per_group + word;
        self.words[index] |= 1_u64 << (suppkey & 63);
    }

    fn merge(&mut self, other: GroupSupplierBitset) {
        for (left, right) in self.words.iter_mut().zip(other.words) {
            *left |= right;
        }
    }

    fn counts(&self) -> Vec<u64> {
        self.words
            .chunks(self.layout.words_per_group)
            .map(|words| words.iter().map(|word| u64::from(word.count_ones())).sum())
            .collect()
    }
}

pub(super) fn supplier_bitset_layout(
    group_count: usize,
    max_suppkey: Option<i64>,
) -> Option<SupplierBitsetLayout> {
    if std::env::var_os("DODAM_Q16_ENABLE_SUPPLIER_BITSET").is_none() {
        return None;
    }
    let max_suppkey = max_suppkey?;
    if max_suppkey < 0 || group_count == 0 {
        return None;
    }
    let words_per_group = (usize::try_from(max_suppkey).ok()? + 64) / 64;
    let bytes = group_count.checked_mul(words_per_group)?.checked_mul(8)?;
    if bytes > supplier_bitset_max_bytes() {
        return None;
    }
    Some(SupplierBitsetLayout {
        group_count,
        words_per_group,
    })
}

pub(super) fn supplier_bitset_max_bytes() -> usize {
    std::env::var("DODAM_Q16_SUPPLIER_BITSET_MAX_BYTES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(32 * 1024 * 1024)
}

pub(super) fn supplier_count_chunk_size() -> usize {
    std::env::var("DODAM_Q16_SUPPLIER_COUNT_CHUNK_SIZE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(4)
}

pub(super) fn packed_distinct_enabled(group_count: usize, max_suppkey: Option<i64>) -> bool {
    std::env::var_os("DODAM_Q16_DISABLE_PACKED_DISTINCT").is_none()
        && group_count <= u32::MAX as usize
        && max_suppkey.is_some_and(|key| key >= 0 && key <= u32::MAX as i64)
}

pub(super) fn supplier_counts_packed_batch(
    batch: RecordBatch,
    part_to_group: &AdaptiveI64Map<usize>,
    bad_suppliers: &AdaptiveI64Set,
    distinct_suppliers: &mut PackedU32PairDistinct,
) -> Result<()> {
    let partkeys = batch_column(&batch, "ps_partkey")?;
    let suppkeys = batch_column(&batch, "ps_suppkey")?;
    if let Some(keys) = SupplierKeyView::try_new(partkeys, suppkeys)
        && supplier_counts_packed_typed(keys, part_to_group, bad_suppliers, distinct_suppliers)
    {
        return Ok(());
    }
    for row in 0..batch.num_rows() {
        let (Some(partkey), Some(suppkey)) = (
            numeric_i64_value(partkeys, row)?,
            numeric_i64_value(suppkeys, row)?,
        ) else {
            continue;
        };
        if bad_suppliers.contains(suppkey) {
            continue;
        }
        let Some(group_id) = part_to_group.get(partkey) else {
            continue;
        };
        if !distinct_suppliers.push(group_id, suppkey) {
            return Err(DodamError::UnsupportedSql(
                "packed distinct key is out of range".to_string(),
            ));
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
pub(super) struct SupplierKeyView<'a> {
    partkeys: I64VectorView<'a>,
    suppkeys: I64VectorView<'a>,
}

impl<'a> SupplierKeyView<'a> {
    fn try_new(partkeys: &'a ArrayRef, suppkeys: &'a ArrayRef) -> Option<Self> {
        Some(Self {
            partkeys: I64VectorView::Arrow(partkeys.as_any().downcast_ref::<Int64Array>()?),
            suppkeys: I64VectorView::Arrow(suppkeys.as_any().downcast_ref::<Int64Array>()?),
        })
    }

    fn try_new_view(view: BatchView<'a>) -> Option<Self> {
        Some(Self {
            partkeys: view.i64_vector(0)?,
            suppkeys: view.i64_vector(1)?,
        })
    }
}

pub(super) fn supplier_counts_packed_view(
    view: BatchView<'_>,
    part_to_group: &AdaptiveI64Map<usize>,
    bad_suppliers: &AdaptiveI64Set,
    distinct_suppliers: &mut PackedU32PairDistinct,
) -> Result<()> {
    if view.num_columns() == 2
        && let Some(keys) = SupplierKeyView::try_new_view(view)
        && supplier_counts_packed_typed(keys, part_to_group, bad_suppliers, distinct_suppliers)
    {
        return Ok(());
    }
    let Some(batch) = view.try_record_batch() else {
        return Err(DodamError::UnsupportedSql(
            "part supplier group packed supplier-count raw vector columns have unsupported types"
                .to_string(),
        ));
    };
    supplier_counts_packed_batch(
        batch.clone(),
        part_to_group,
        bad_suppliers,
        distinct_suppliers,
    )
}

pub(super) fn supplier_counts_packed_typed(
    keys: SupplierKeyView<'_>,
    part_to_group: &AdaptiveI64Map<usize>,
    bad_suppliers: &AdaptiveI64Set,
    distinct_suppliers: &mut PackedU32PairDistinct,
) -> bool {
    let partkeys = keys.partkeys;
    let suppkeys = keys.suppkeys;
    if let (Some(partkey_values), Some(suppkey_values)) = (
        partkeys.values_if_null_free(),
        suppkeys.values_if_null_free(),
    ) {
        if let (Some((group_values, group_present)), Some(bad_present)) = (
            part_to_group.dense_slices(),
            bad_suppliers.dense_contains_slice(),
        ) {
            for row in 0..partkey_values.len() {
                let suppkey = suppkey_values[row];
                if let Ok(suppkey_index) = usize::try_from(suppkey)
                    && bad_present.get(suppkey_index).copied().unwrap_or(false)
                {
                    continue;
                }
                let Ok(partkey_index) = usize::try_from(partkey_values[row]) else {
                    continue;
                };
                if partkey_index >= group_present.len() || !group_present[partkey_index] {
                    continue;
                }
                if !distinct_suppliers.push(group_values[partkey_index], suppkey) {
                    return false;
                }
            }
            return true;
        }
        for row in 0..partkey_values.len() {
            let suppkey = suppkey_values[row];
            if bad_suppliers.contains(suppkey) {
                continue;
            }
            let Some(group_id) = part_to_group.get(partkey_values[row]) else {
                continue;
            };
            if !distinct_suppliers.push(group_id, suppkey) {
                return false;
            }
        }
        return true;
    }
    for row in 0..partkeys.len() {
        if partkeys.is_null(row) || suppkeys.is_null(row) {
            continue;
        }
        let suppkey = suppkeys.value(row);
        if bad_suppliers.contains(suppkey) {
            continue;
        }
        let Some(group_id) = part_to_group.get(partkeys.value(row)) else {
            continue;
        };
        if !distinct_suppliers.push(group_id, suppkey) {
            return false;
        }
    }
    true
}

pub(super) fn merge_supplier_counts_packed(
    groups: &mut PackedU32PairDistinct,
    mut batch_groups: PackedU32PairDistinct,
) {
    groups.append(&mut batch_groups);
}

pub(super) fn supplier_counts_bitset_batch(
    batch: RecordBatch,
    part_to_group: &AdaptiveI64Map<usize>,
    bad_suppliers: &AdaptiveI64Set,
    distinct_suppliers: &mut GroupSupplierBitset,
) -> Result<()> {
    let partkeys = batch_column(&batch, "ps_partkey")?;
    let suppkeys = batch_column(&batch, "ps_suppkey")?;
    if let Some(keys) = SupplierKeyView::try_new(partkeys, suppkeys)
        && supplier_counts_bitset_typed(keys, part_to_group, bad_suppliers, distinct_suppliers)
    {
        return Ok(());
    }
    for row in 0..batch.num_rows() {
        let (Some(partkey), Some(suppkey)) = (
            numeric_i64_value(partkeys, row)?,
            numeric_i64_value(suppkeys, row)?,
        ) else {
            continue;
        };
        if bad_suppliers.contains(suppkey) {
            continue;
        }
        let Some(group_id) = part_to_group.get(partkey) else {
            continue;
        };
        distinct_suppliers.insert(group_id, suppkey);
    }
    Ok(())
}

pub(super) fn supplier_counts_bitset_view(
    view: BatchView<'_>,
    part_to_group: &AdaptiveI64Map<usize>,
    bad_suppliers: &AdaptiveI64Set,
    distinct_suppliers: &mut GroupSupplierBitset,
) -> Result<()> {
    if view.num_columns() == 2
        && let Some(keys) = SupplierKeyView::try_new_view(view)
        && supplier_counts_bitset_typed(keys, part_to_group, bad_suppliers, distinct_suppliers)
    {
        return Ok(());
    }
    let Some(batch) = view.try_record_batch() else {
        return Err(DodamError::UnsupportedSql(
            "part supplier group bitset supplier-count raw vector columns have unsupported types"
                .to_string(),
        ));
    };
    supplier_counts_bitset_batch(
        batch.clone(),
        part_to_group,
        bad_suppliers,
        distinct_suppliers,
    )
}

pub(super) fn supplier_counts_bitset_typed(
    keys: SupplierKeyView<'_>,
    part_to_group: &AdaptiveI64Map<usize>,
    bad_suppliers: &AdaptiveI64Set,
    distinct_suppliers: &mut GroupSupplierBitset,
) -> bool {
    let partkeys = keys.partkeys;
    let suppkeys = keys.suppkeys;
    if let (Some(partkey_values), Some(suppkey_values)) = (
        partkeys.values_if_null_free(),
        suppkeys.values_if_null_free(),
    ) {
        for row in 0..partkey_values.len() {
            let suppkey = suppkey_values[row];
            if bad_suppliers.contains(suppkey) {
                continue;
            }
            let Some(group_id) = part_to_group.get(partkey_values[row]) else {
                continue;
            };
            distinct_suppliers.insert(group_id, suppkey);
        }
        return true;
    }
    for row in 0..partkeys.len() {
        if partkeys.is_null(row) || suppkeys.is_null(row) {
            continue;
        }
        let suppkey = suppkeys.value(row);
        if bad_suppliers.contains(suppkey) {
            continue;
        }
        let Some(group_id) = part_to_group.get(partkeys.value(row)) else {
            continue;
        };
        distinct_suppliers.insert(group_id, suppkey);
    }
    true
}

pub(super) fn supplier_counts_batch(
    batch: RecordBatch,
    part_to_group: &AdaptiveI64Map<usize>,
    bad_suppliers: &AdaptiveI64Set,
) -> Result<FastHashSet<(usize, i64)>> {
    let partkeys = batch_column(&batch, "ps_partkey")?;
    let suppkeys = batch_column(&batch, "ps_suppkey")?;
    if let Some(keys) = SupplierKeyView::try_new(partkeys, suppkeys)
        && let Some(groups) = supplier_counts_typed(keys, part_to_group, bad_suppliers)?
    {
        return Ok(groups);
    }
    let mut distinct_suppliers = FastHashSet::default();
    for row in 0..batch.num_rows() {
        let (Some(partkey), Some(suppkey)) = (
            numeric_i64_value(partkeys, row)?,
            numeric_i64_value(suppkeys, row)?,
        ) else {
            continue;
        };
        if bad_suppliers.contains(suppkey) {
            continue;
        }
        let Some(group_id) = part_to_group.get(partkey) else {
            continue;
        };
        distinct_suppliers.insert((group_id, suppkey));
    }
    Ok(distinct_suppliers)
}

pub(super) fn supplier_counts_view(
    view: BatchView<'_>,
    part_to_group: &AdaptiveI64Map<usize>,
    bad_suppliers: &AdaptiveI64Set,
) -> Result<FastHashSet<(usize, i64)>> {
    if view.num_columns() == 2
        && let Some(keys) = SupplierKeyView::try_new_view(view)
        && let Some(groups) = supplier_counts_typed(keys, part_to_group, bad_suppliers)?
    {
        return Ok(groups);
    }
    let Some(batch) = view.try_record_batch() else {
        return Err(DodamError::UnsupportedSql(
            "part supplier group supplier-count raw vector columns have unsupported types"
                .to_string(),
        ));
    };
    supplier_counts_batch(batch.clone(), part_to_group, bad_suppliers)
}

pub(super) fn supplier_counts_typed(
    keys: SupplierKeyView<'_>,
    part_to_group: &AdaptiveI64Map<usize>,
    bad_suppliers: &AdaptiveI64Set,
) -> Result<Option<FastHashSet<(usize, i64)>>> {
    let partkeys = keys.partkeys;
    let suppkeys = keys.suppkeys;
    let mut distinct_suppliers = FastHashSet::default();
    for row in 0..partkeys.len() {
        if partkeys.is_null(row) || suppkeys.is_null(row) {
            continue;
        }
        let suppkey = suppkeys.value(row);
        if bad_suppliers.contains(suppkey) {
            continue;
        }
        let Some(group_id) = part_to_group.get(partkeys.value(row)) else {
            continue;
        };
        distinct_suppliers.insert((group_id, suppkey));
    }
    Ok(Some(distinct_suppliers))
}

pub(super) fn merge_supplier_counts(
    groups: &mut FastHashSet<(usize, i64)>,
    batch_groups: FastHashSet<(usize, i64)>,
) {
    groups.extend(batch_groups);
}

pub(super) fn merge_supplier_bitsets(
    groups: &mut GroupSupplierBitset,
    batch_groups: GroupSupplierBitset,
) {
    groups.merge(batch_groups);
}

pub(super) fn part_supplier_group_output(rows: Vec<PartSupplierGroupRow>) -> Result<QueryOutput> {
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("p_brand", DataType::Utf8, false),
            Field::new("p_type", DataType::Utf8, false),
            Field::new("p_size", DataType::Int64, false),
            Field::new("supplier_cnt", DataType::UInt64, false),
        ])),
        vec![
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|row| row.brand.as_str()),
            )),
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|row| row.type_name.as_str()),
            )),
            Arc::new(Int64Array::from_iter_values(
                rows.iter().map(|row| row.size),
            )),
            Arc::new(UInt64Array::from_iter_values(
                rows.iter().map(|row| row.supplier_count),
            )),
        ],
    )?;
    Ok(QueryOutput::Aggregate {
        metrics: AggregateMetrics::default(),
        batches: vec![batch],
    })
}
