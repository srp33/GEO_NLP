import base64
import csv
import glob
import gzip
from helper import *
import io
import json
import os
import pandas as pd
from pandas.api.types import is_numeric_dtype
import re
import sys
import zipfile

tsv_file_path = sys.argv[1]
convert_to_lower = sys.argv[2] == "True"
pep_dir_path = sys.argv[3]
json_file_path = sys.argv[4]

# Strong exclusion patterns (always drop if matched)
STRONG_EXCLUDE_PATTERNS = [
    r"_date",
    r"_taxid",
    r"_id",
    r"accession",
    r"geo",
    r"sra",
    r"url",
    r"ftp",
    r"md5",
    r"checksum",
    r"time",
    r"timestamp",
    r"_row_count",
    r"sample_status"
]

URL_PATTERN = re.compile(r"^(https?|ftp)://", re.IGNORECASE)

MISSING_LIKE = {"", "na", "n/a", "none", "null", "missing", "unknown", "not available", "nan"}

def strong_exclude(col_name: str) -> bool:
    name = col_name.lower()
    return any(re.search(p, name) for p in STRONG_EXCLUDE_PATTERNS)

def is_mostly_unique_ids(series: pd.Series, threshold: float = 0.9) -> bool:
    s = series.dropna().astype(str).str.lower()
    if len(s) == 0:
        return False
    return s.nunique() / len(s) >= threshold

def is_mostly_urls(series: pd.Series, threshold: float = 0.7) -> bool:
    s = series.dropna().astype(str)
    if len(s) == 0:
        return False
    num_urls = s.str.match(URL_PATTERN).sum()
    return num_urls / len(s) >= threshold

def is_single_value(series: pd.Series) -> bool:
    s = series.dropna().astype(str).str.lower()
    return s.nunique() <= 1

def keep_semantic_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_keep = []
    dropped = []

    for col in df.columns:
        s = df[col]

        if strong_exclude(col):
            dropped.append((col, "strong exclusion (_date/_taxid/_id)"))
            continue

        if s.notna().sum() == 0:
            dropped.append((col, "all missing"))
            continue

        if is_mostly_urls(s):
            dropped.append((col, "URLs"))
            continue

        if is_mostly_unique_ids(s):
            dropped.append((col, "mostly unique IDs"))
            continue

        if is_single_value(s):
            dropped.append((col, "single value"))
            continue

        cols_to_keep.append(col)

    return df[cols_to_keep]

def load_and_filter_csv_from_zip(zip_path, csv_name="sample_table.csv"):
    with zipfile.ZipFile(zip_path, 'r') as z:
        if csv_name not in z.namelist():
            raise FileNotFoundError(f"{csv_name} not found inside the ZIP.")

        # Read the CSV contents as text
        with z.open(csv_name) as f:
            csv_text = f.read().decode("utf-8")

    # Load CSV into pandas
    df = pd.read_csv(io.StringIO(csv_text))

    # Filter irrelevant columns
    filtered_df = keep_semantic_columns(df)

    return filtered_df

def normalize_value(v):
    """Normalize values for comparison/display."""
    if pd.isna(v):
        return None
    s = str(v).strip()
    if s == "":
        return None
    s_lower = s.lower()
    if s_lower in MISSING_LIKE:
        return None
    return s

def summarize_column(name, series, max_values=10, max_value_len=80):
    """Create a human-readable summary for a single column."""
    col_name = str(name)
    s = series.dropna()

    # Normalize values (remove NA-like strings)
    normalized = s.map(normalize_value).dropna()
    if len(normalized) == 0:
        return None  # nothing meaningful here

    # Numeric column
    if is_numeric_dtype(series):
        desc = pd.to_numeric(normalized, errors="coerce").dropna()
        if len(desc) == 0:
            return None
        col_min = desc.min()
        col_max = desc.max()
        col_median = desc.median()
        summary = (
            f"{col_name}: numeric metadata; "
            f"range approximately {col_min:.3g}–{col_max:.3g}, "
            f"median around {col_median:.3g}."
        )
        return summary

    # Text / categorical-like column
    # Get unique values (case-insensitive but keep original form)
    # We'll deduplicate by lowercased version.
    unique_vals = []
    seen_lower = set()
    for v in normalized:
        v_str = str(v).strip()
        v_lower = v_str.lower()
        if v_lower in seen_lower:
            continue
        seen_lower.add(v_lower)
        # truncate long values for readability
        if len(v_str) > max_value_len:
            v_str = v_str[:max_value_len].rstrip() + "…"
        unique_vals.append(v_str)
        if len(unique_vals) >= max_values:
            break

    if len(unique_vals) == 1:
        # Column has one meaningful value
        summary = f"{col_name}: constant metadata with value '{unique_vals[0]}'."
    else:
        values_list = "; ".join(unique_vals)
        summary = f"{col_name}: metadata values include {values_list}."

    return summary

def summarize_metadata_for_embedding(df: pd.DataFrame,
                                     max_values_per_column: int = 20,
                                     max_value_len: int = 200) -> str:
    """
    Summarize a metadata DataFrame into a text description suitable for embeddings.
    Counts per value are ignored; only the presence of values matters.
    """
    n_rows, n_cols = df.shape
    pieces = []

    col_summaries = []
    for col in df.columns:
        col_summary = summarize_column(
            col,
            df[col],
            max_values=max_values_per_column,
            max_value_len=max_value_len
        )
        if col_summary is not None:
            col_summaries.append(col_summary)

    if col_summaries:
        for s in col_summaries:
            pieces.append(f"- {s}")

    return clean_text("\n".join(pieces), convert_to_lower=convert_to_lower, remove_non_alnum=False)

article_dict = {}

with gzip.open(tsv_file_path) as tsv_file:
    tsv_file.readline()

    experiment_types = set()

    for line in tsv_file:
        line_items = line.decode().rstrip("\n").split("\t")

        gse = line_items[0]
        title = line_items[1]
        summary = line_items[2]
        overall_design = line_items[3]
        experiment_type = line_items[4].lower().split("|")
        gpl = line_items[7].lower().split("|")
        gpl_title = line_items[8].lower()
        gpl_technology = line_items[9].lower()
        species = line_items[10].lower().split("|")
        taxon_id = line_items[11].split("|")
        superseries_gse = line_items[12]

        metadata_description = ""
        if pep_dir_path != "":
            pep_file_path = f"{pep_dir_path}/*/{gse.lower()}.zip"
            pep_file_paths = glob.glob(pep_file_path)

            if len(pep_file_paths) > 0:
                pep_file_path = pep_file_paths[0]
                pep_df = load_and_filter_csv_from_zip(pep_file_path)
                pep_df = keep_semantic_columns(pep_df)
                metadata_description = summarize_metadata_for_embedding(pep_df).replace("\t", "")

        # We remove series that are part of a superseries because including these
        #   could cause bias in the machine-learning analysis. Additionally,
        #   if users find a relevant SuperSeries, they will find the associated
        #   SubSeries.
        if "homo sapiens" in species and "9606" in taxon_id and superseries_gse == "":
            if "expression profiling by array" in experiment_type:
                if "affymetrix" in gpl_title or "illumina" in gpl_title or "agilent" in gpl_title:
                    text = clean_text(f"{title} {summary} {overall_design}", convert_to_lower=convert_to_lower)

                    if metadata_description != "":
                        text += f" {metadata_description}"

                    article_dict[gse] = text
                    print(gse)
            elif "expression profiling by high throughput sequencing" in experiment_type and "illumina" in gpl_title:
                text = clean_text(f"{title} {summary} {overall_design}", convert_to_lower=convert_to_lower)

                if metadata_description != "":
                    text += f" {metadata_description}"

                article_dict[gse] = text
                print(gse)

print(len(article_dict)) #48,893
with gzip.open(json_file_path, 'w') as json_file:
    json_file.write(json.dumps(article_dict).encode())
