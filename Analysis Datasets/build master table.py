import pandas as pd
import numpy as np

MM23_PATH = "mm23.csv"
BANK_RATE_PATH = "Bank Rate history and data  Bank of England Database.csv"

# Load MM23 and locate CDID mapping row
mm = pd.read_csv(MM23_PATH, low_memory=False)
cdid_row = mm.loc[mm["Title"] == "CDID"].iloc[0]

SERIES_CDIDS = {
    "cpih_yoy": "L55O",
    "food_yoy": "L55P",
    "housing_fuels_yoy": "L55S",
    "transport_yoy": "L55V",
    "core_cpih_yoy": "L5LQ",
}

colmap = {}
for new_name, cdid in SERIES_CDIDS.items():
    matches = [c for c in mm.columns if cdid_row.get(c) == cdid]
    if not matches:
        raise ValueError(f"Could not find series CDID={cdid} in MM23 file.")
    colmap[new_name] = matches[0]

# Keep only monthly rows like "2022 FEB"
monthly_mask = mm["Title"].astype(str).str.match(r"^\d{4} [A-Z]{3}$")
mm_month = mm.loc[monthly_mask, ["Title"] + list(colmap.values())].copy()

month_map = {m.upper(): i for i, m in enumerate(
    ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"], start=1)}

def parse_title_to_date(t: str) -> pd.Timestamp:
    y, mon = t.split()
    return pd.Timestamp(int(y), month_map[mon], 1)

mm_month["date"] = mm_month["Title"].apply(parse_title_to_date)

df_cpi = mm_month.drop(columns=["Title"]).rename(columns={v: k for k, v in colmap.items()})
for c in df_cpi.columns:
    if c != "date":
        df_cpi[c] = pd.to_numeric(df_cpi[c], errors="coerce")

df_cpi = df_cpi.sort_values("date").reset_index(drop=True)
df_cpi = df_cpi[df_cpi["date"] >= "2005-01-01"].copy()

# Load Bank Rate and convert to monthly (month-start effective rate)
br = pd.read_csv(BANK_RATE_PATH)
br["date_changed"] = pd.to_datetime(br["Date Changed"], format="%d %b %y")
br = br.sort_values("date_changed")

s = br.set_index("date_changed")["Rate"].sort_index()
monthly_index = pd.date_range(start=df_cpi["date"].min(), end=df_cpi["date"].max(), freq="MS")

rates = []
for d in monthly_index:
    idx = s.index.searchsorted(d, side="right") - 1
    rates.append(s.iloc[idx] if idx >= 0 else np.nan)

df_bank = pd.DataFrame({"date": monthly_index, "bank_rate": rates})
df_bank["bank_rate"] = df_bank["bank_rate"].ffill()

# Merge and create war dummy
df_mvd = df_cpi.merge(df_bank, on="date", how="left")
WAR_START = pd.Timestamp("2022-02-01")
df_mvd["war_dummy"] = (df_mvd["date"] >= WAR_START).astype(int)

# master_table_v1 with PPI placeholders
df_master = df_mvd.copy()
df_master["ppi_input_yoy"] = np.nan
df_master["ppi_output_yoy"] = np.nan

# Save outputs
df_mvd.to_csv("MVD_uk_inflation_war_baseline.csv", index=False)
df_master.to_csv("master_table_v1_uk_inflation_war_baseline.csv", index=False)

print("Saved MVD_uk_inflation_war_baseline.csv")
print("Saved master_table_v1_uk_inflation_war_baseline.csv")
