import re
import pandas as pd
import numpy as np

PPI_PATH = "ppi.csv"
MASTER_V1_PATH = "master_table_v1_uk_inflation_war_baseline.csv"

OUT_MASTER_V2 = "master_table_v2_uk_inflation_war_baseline_with_ppi.csv"
OUT_DICT = "ppi_columns_dictionary.csv"

# Load files
ppi = pd.read_csv(PPI_PATH, low_memory=False)
master = pd.read_csv(MASTER_V1_PATH, parse_dates=["date"])

# Keep only monthly rows in PPI ("YYYY MON")
monthly_mask = ppi["Title"].astype(str).str.match(r"^\d{4} [A-Z]{3}$")
ppi_month = ppi.loc[monthly_mask].copy()

month_map = {m.upper(): i for i, m in enumerate(
    ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"], start=1
)}

def parse_title_to_date(t: str) -> pd.Timestamp:
    y, mon = t.split()
    return pd.Timestamp(int(y), month_map[mon], 1)

ppi_month["date"] = ppi_month["Title"].apply(parse_title_to_date)

# Select the PPI INDEX columns (Manufactured products totals)
ppi_cols_needed = {
    "ppi_input_index": "PPI INDEX INPUT - C Inputs into production of Manufactured products, excluding Climate Change Levy 2015=100",
    "ppi_output_index": "PPI INDEX OUTPUT TOTAL - C Manufactured products, excluding Duty 2015=100",
    "ppi_import_index": "PPI INDEX IMPORT - C Manufactured products 2015=100",
    "ppi_export_index": "PPI INDEX EXPORT TOTAL - C Manufactured products 2015=100",
}

# Validate columns exist
missing = [col for col in ppi_cols_needed.values() if col not in ppi_month.columns]
if missing:
    raise ValueError(
        "Some required PPI columns were not found in ppi.csv:\n"
        + "\n".join(missing)
        + "\n\nTip: open ppi.csv and confirm the exact column names."
    )

df_ppi = ppi_month[["date"] + list(ppi_cols_needed.values())].copy()
df_ppi = df_ppi.rename(columns={v: k for k, v in ppi_cols_needed.items()})

for c in ["ppi_input_index", "ppi_output_index", "ppi_import_index", "ppi_export_index"]:
    df_ppi[c] = pd.to_numeric(df_ppi[c], errors="coerce")

df_ppi = df_ppi.sort_values("date").reset_index(drop=True)

# Compute YoY (%) from index series
for idx_col, yoy_col in [
    ("ppi_input_index", "ppi_input_yoy"),
    ("ppi_output_index", "ppi_output_yoy"),
    ("ppi_import_index", "ppi_import_yoy"),
    ("ppi_export_index", "ppi_export_yoy"),
]:
    df_ppi[yoy_col] = df_ppi[idx_col].pct_change(12) * 100.0

df_ppi_yoy = df_ppi[["date", "ppi_input_yoy", "ppi_output_yoy", "ppi_import_yoy", "ppi_export_yoy"]].copy()

# Merge into master table
old_ppi_cols = [c for c in master.columns if c.startswith("ppi_")]
master_base = master.drop(columns=old_ppi_cols, errors="ignore")

master_v2 = master_base.merge(df_ppi_yoy, on="date", how="left")

# Save outputs
master_v2.to_csv(OUT_MASTER_V2, index=False)

dict_rows = [
    ("ppi_input_yoy",
     "YoY % change of index: 'Inputs into production of Manufactured products' (PPI INDEX INPUT - C), excl. Climate Change Levy, 2015=100",
     "ONS PPI statistical bulletin dataset (ppi.csv)"),
    ("ppi_output_yoy",
     "YoY % change of index: 'Manufactured products' output price index (PPI INDEX OUTPUT TOTAL - C), excl. Duty, 2015=100",
     "ONS PPI statistical bulletin dataset (ppi.csv)"),
    ("ppi_import_yoy",
     "YoY % change of index: Import price index for Manufactured products (PPI INDEX IMPORT - C), 2015=100",
     "ONS PPI statistical bulletin dataset (ppi.csv)"),
    ("ppi_export_yoy",
     "YoY % change of index: Export price index for Manufactured products (PPI INDEX EXPORT TOTAL - C), 2015=100",
     "ONS PPI statistical bulletin dataset (ppi.csv)"),
]
df_dict = pd.DataFrame(dict_rows, columns=["column", "definition", "source"])
df_dict.to_csv(OUT_DICT, index=False)

print("Saved:", OUT_MASTER_V2)
print("Saved:", OUT_DICT)
print("Rows, Cols:", master_v2.shape)
print("Date range:", master_v2["date"].min(), "to", master_v2["date"].max())
