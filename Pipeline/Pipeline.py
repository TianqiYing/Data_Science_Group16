import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# Config
WAR_DATE = pd.Timestamp("2022-02-01")     # event marker
START_DATE = pd.Timestamp("2015-01-01")   # analysis window start
END_DATE = pd.Timestamp("2026-02-01")


# Helpers
MONTH_MAP = {m.upper(): i for i, m in enumerate(
    ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"], start=1
)}

def parse_title_to_date(t: str) -> pd.Timestamp:
    """Parse 'YYYY MON' like '2022 FEB' into month-start Timestamp."""
    y, mon = t.split()
    return pd.Timestamp(int(y), MONTH_MAP[mon], 1)


# Step 0: Build master_table_v2
def build_master_table_v2(mm_path, ppi_path, bank_path, out_path):
    # Load MM23 and locate CDID row to map series robustly
    mm = pd.read_csv(mm_path, low_memory=False)
    cdid_row = mm.loc[mm["Title"] == "CDID"].iloc[0]

    # Core CPIH series you already used
    SERIES_CDIDS = {
        "cpih_yoy": "L55O",
        "food_yoy": "L55P",
        "housing_fuels_yoy": "L55S",
        "transport_yoy": "L55V",
        "core_cpih_yoy": "L5LQ",
    }

    colmap = {}
    for new, cdid in SERIES_CDIDS.items():
        matches = [c for c in mm.columns if cdid_row.get(c) == cdid]
        if not matches:
            raise ValueError(f"Missing CDID {cdid} in mm23.csv")
        colmap[new] = matches[0]

    # Keep monthly rows
    monthly_mask = mm["Title"].astype(str).str.match(r"^\d{4} [A-Z]{3}$")
    mm_month = mm.loc[monthly_mask, ["Title"] + list(colmap.values())].copy()
    mm_month["date"] = mm_month["Title"].apply(parse_title_to_date)
    df_cpi = mm_month.drop(columns=["Title"]).rename(columns={v: k for k, v in colmap.items()})
    for c in SERIES_CDIDS.keys():
        df_cpi[c] = pd.to_numeric(df_cpi[c], errors="coerce")
    df_cpi = df_cpi.sort_values("date").reset_index(drop=True)
    df_cpi = df_cpi[df_cpi["date"] >= START_DATE].copy()

    # Bank Rate: convert change-date step function to monthly
    br = pd.read_csv(bank_path)
    br["date_changed"] = pd.to_datetime(br["Date Changed"], format="%d %b %y")
    br = br.sort_values("date_changed")
    s = br.set_index("date_changed")["Rate"].sort_index()

    monthly_index = pd.date_range(df_cpi["date"].min(), df_cpi["date"].max(), freq="MS")
    rates = []
    for d in monthly_index:
        idx = s.index.searchsorted(d, side="right") - 1
        rates.append(s.iloc[idx] if idx >= 0 else np.nan)

    df_bank = pd.DataFrame({"date": monthly_index, "bank_rate": rates})
    df_bank["bank_rate"] = df_bank["bank_rate"].ffill()

    # PPI: compute YoY from index series (manufactured products totals)
    ppi = pd.read_csv(ppi_path, low_memory=False)
    monthly_mask_ppi = ppi["Title"].astype(str).str.match(r"^\d{4} [A-Z]{3}$")
    ppi_month = ppi.loc[monthly_mask_ppi].copy()
    ppi_month["date"] = ppi_month["Title"].apply(parse_title_to_date)

    ppi_cols_needed = {
        "ppi_input_index": "PPI INDEX INPUT - C Inputs into production of Manufactured products, excluding Climate Change Levy 2015=100",
        "ppi_output_index": "PPI INDEX OUTPUT TOTAL - C Manufactured products, excluding Duty 2015=100",
        "ppi_import_index": "PPI INDEX IMPORT - C Manufactured products 2015=100",
        "ppi_export_index": "PPI INDEX EXPORT TOTAL - C Manufactured products 2015=100",
    }

    missing = [v for v in ppi_cols_needed.values() if v not in ppi_month.columns]
    if missing:
        raise ValueError("Missing PPI columns in ppi.csv:\n" + "\n".join(missing))

    df_ppi = ppi_month[["date"] + list(ppi_cols_needed.values())].rename(columns={v: k for k, v in ppi_cols_needed.items()})
    for c in df_ppi.columns:
        if c != "date":
            df_ppi[c] = pd.to_numeric(df_ppi[c], errors="coerce")
    df_ppi = df_ppi.sort_values("date").reset_index(drop=True)

    for idx_col, yoy_col in [
        ("ppi_input_index","ppi_input_yoy"),
        ("ppi_output_index","ppi_output_yoy"),
        ("ppi_import_index","ppi_import_yoy"),
        ("ppi_export_index","ppi_export_yoy"),
    ]:
        df_ppi[yoy_col] = df_ppi[idx_col].pct_change(12) * 100.0

    df_ppi_yoy = df_ppi[["date","ppi_input_yoy","ppi_output_yoy","ppi_import_yoy","ppi_export_yoy"]]

    # Merge
    master = df_cpi.merge(df_bank, on="date", how="left").merge(df_ppi_yoy, on="date", how="left")
    master["war_dummy"] = (master["date"] >= WAR_DATE).astype(int)

    # clamp range for consistency
    master = master[master["date"].between(START_DATE, END_DATE)].copy()
    master.to_csv(out_path, index=False)
    return master, mm


# Task 1: Fig8 — division-level stacked contributions
def make_fig8_division_contributions(mm, xlsx_path, out_png):
    # CPIH division annual rate columns in MM23 (01..12)
    div_cols = []
    for c in mm.columns:
        cu = c.upper()
        if cu.startswith("CPIH ANNUAL RATE") and re.search(r"\b(0[1-9]|1[0-2])\s*:", cu):
            div_cols.append(c)

    all_items_col = "CPIH ANNUAL RATE 00: ALL ITEMS 2015=100"
    if all_items_col not in mm.columns:
        raise ValueError(f"Expected column not found: {all_items_col}")

    monthly_mask = mm["Title"].astype(str).str.match(r"^\d{4} [A-Z]{3}$")
    mm_month = mm.loc[monthly_mask, ["Title"] + div_cols + [all_items_col]].copy()
    mm_month["date"] = mm_month["Title"].apply(parse_title_to_date)
    mm_month = mm_month.drop(columns=["Title"]).sort_values("date")

    # Rename division columns to div_01..div_12
    div_codes = []
    for c in div_cols:
        m = re.search(r"\b(0[1-9]|1[0-2])\s*:", c)
        div_codes.append(m.group(1))
    rename = {c: f"div_{code}" for c, code in zip(div_cols, div_codes)}
    rename[all_items_col] = "cpih_all"

    df_div = mm_month.rename(columns=rename)
    for c in df_div.columns:
        if c != "date":
            df_div[c] = pd.to_numeric(df_div[c], errors="coerce")

    df_div = df_div[df_div["date"].between(START_DATE, END_DATE)].copy()

    # Read weights Table 9 from detailed reference tables
    t9 = pd.read_excel(xlsx_path, sheet_name="Table 9", header=None)

    # Row 6 col3 onwards contains year headers like "2025 Jan" / "2025 Feb-Dec" / "2015"
    year_headers = t9.iloc[6, 3:].tolist()

    def parse_header(h):
        if isinstance(h, (int, float)) and not pd.isna(h):
            return (int(h), "all")
        if isinstance(h, str):
            s = h.strip()
            m = re.match(r"^(\d{4})", s)
            if not m:
                return (None, None)
            y = int(m.group(1))
            if "JAN" in s.upper():
                return (y, "Jan")
            if "FEB" in s.upper():
                return (y, "Feb-Dec")
            return (y, "all")
        return (None, None)

    parsed = [parse_header(h) for h in year_headers]
    colnames = [f"{y}_{p}" for y, p in parsed]

    # Rows 8..19 hold 12 divisions; column 2 onwards: first col is division name, then weights
    div_block = t9.iloc[8:20, 2:].copy()
    div_raw = div_block.iloc[:, 0].tolist()
    weights_vals = div_block.iloc[:, 1:].to_numpy()

    wdf = pd.DataFrame(weights_vals, columns=colnames)
    wdf.insert(0, "division_raw", div_raw)

    def clean_division(raw):
        raw = str(raw)
        m = re.match(r"\s*(0[1-9]|1[0-2])", raw)
        code = m.group(1) if m else None
        name = re.sub(r"^\s*\d+\s*", "", raw).strip()
        name = re.sub(r"\s+", " ", name)
        return code, name

    div_info = [clean_division(r) for r in wdf["division_raw"]]
    wdf["div_code"] = [c for c, _ in div_info]
    wdf["div_name"] = [n for _, n in div_info]

    # Build lookup: weights by (year, period)
    weight_lookup = {}
    for _, row in wdf.iterrows():
        code = row["div_code"]
        d = {}
        for cn, (y, p) in zip(colnames, parsed):
            if y is None:
                continue
            d[(int(y), p)] = row[cn]
        weight_lookup[code] = d

    def weight_for_month(code, date):
        """ONS weights update: Jan may differ; Feb-Dec constant."""
        y = date.year
        m = date.month
        d = weight_lookup[code]

        # Prefer Jan / Feb-Dec patterns if present
        if (y, "Jan") in d or (y, "Feb-Dec") in d:
            if m == 1 and (y, "Jan") in d:
                return d[(y, "Jan")]
            if (y, "Feb-Dec") in d:
                return d[(y, "Feb-Dec")]
            if (y, "Jan") in d:
                return d[(y, "Jan")]

        # Else annual weight
        if (y, "all") in d:
            return d[(y, "all")]

        # Backward fallback
        for y2 in range(y - 1, 2000, -1):
            if (y2, "Feb-Dec") in d:
                return d[(y2, "Feb-Dec")]
            if (y2, "all") in d:
                return d[(y2, "all")]
        return np.nan

    dates = pd.date_range(START_DATE, END_DATE, freq="MS")
    w_month = pd.DataFrame({"date": dates})
    for code in [f"{i:02d}" for i in range(1, 13)]:
        w_month[f"w_{code}"] = [weight_for_month(code, d) for d in dates]

    dfc = df_div.merge(w_month, on="date", how="left")

    # Approx contribution: (weight / 1000) * division inflation rate
    for code in [f"{i:02d}" for i in range(1, 13)]:
        dfc[f"contrib_{code}"] = (dfc[f"w_{code}"] / 1000.0) * dfc[f"div_{code}"]

    div_name_map = {row["div_code"]: row["div_name"] for _, row in wdf.iterrows()}

    x = dfc["date"].to_numpy()
    ys = [dfc[f"contrib_{code}"].to_numpy() for code in [f"{i:02d}" for i in range(1, 13)]]
    labels = [f"{code} {div_name_map[code]}" for code in [f"{i:02d}" for i in range(1, 13)]]

    plt.figure(figsize=(12, 6))
    plt.stackplot(x, ys, labels=labels)
    plt.axvline(WAR_DATE, linestyle="--", linewidth=1)
    plt.title("CPIH 12-month inflation: division-level contributions (weights × division inflation)")
    plt.xlabel("Date")
    plt.ylabel("Contribution (percentage points)")
    plt.legend(ncols=2, fontsize=7, loc="upper left")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# Task 2: Fig9 + coefficients table — distributed-lag OLS
def make_fig9_distributed_lag(master, out_png, out_csv, max_lag=12):
    mt = master.copy().sort_values("date").reset_index(drop=True)

    # Lag features
    for L in range(max_lag + 1):
        mt[f"ppi_input_l{L}"] = mt["ppi_input_yoy"].shift(L)
        mt[f"br_l{L}"] = mt["bank_rate"].shift(L)

    X_cols = [f"ppi_input_l{L}" for L in range(max_lag + 1)] + \
             [f"br_l{L}" for L in range(max_lag + 1)] + \
             ["war_dummy"]

    reg_df = mt[["cpih_yoy"] + X_cols].dropna().copy()
    X = sm.add_constant(reg_df[X_cols])
    y = reg_df["cpih_yoy"]

    model = sm.OLS(y, X).fit()

    # Extract lag coefficients
    lag_names = [f"ppi_input_l{L}" for L in range(max_lag + 1)]
    coef = model.params[lag_names]
    ci = model.conf_int().loc[lag_names]
    lags = np.arange(max_lag + 1)

    # Save coefficient table
    coef_table = pd.DataFrame({
        "lag": lags,
        "coef": coef.values,
        "ci_low": ci[0].values,
        "ci_high": ci[1].values,
    })
    coef_table.to_csv(out_csv, index=False)

    # Plot with CI
    plt.figure(figsize=(10, 5))
    plt.plot(lags, coef.values, marker="o", label="PPI input lag coefficients")
    yerr = np.vstack([coef.values - ci[0].values, ci[1].values - coef.values])
    plt.errorbar(lags, coef.values, yerr=yerr, fmt="none", capsize=3)
    plt.axhline(0, linewidth=1)
    plt.title("Distributed-lag OLS: CPIH ~ lagged PPI input + lagged Bank Rate + war dummy")
    plt.xlabel("Lag (months)")
    plt.ylabel("Coefficient on PPI input (pp)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

    print("Distributed-lag OLS summary:")
    print("  R^2:", float(model.rsquared))
    print("  N obs:", int(model.nobs))


# Task 3: Fig10 — PCA 2D plot (alternative to 3D)
def make_fig10_pca_2d(master, out_png):
    feat_cols = [
        "cpih_yoy","core_cpih_yoy","food_yoy","housing_fuels_yoy","transport_yoy",
        "ppi_input_yoy","ppi_output_yoy","ppi_import_yoy","ppi_export_yoy","bank_rate"
    ]
    df = master[["date","war_dummy"] + feat_cols].dropna().copy()

    X = StandardScaler().fit_transform(df[feat_cols].values)
    pca = PCA(n_components=2, random_state=0)
    pcs = pca.fit_transform(X)

    df["PC1"] = pcs[:, 0]
    df["PC2"] = pcs[:, 1]

    pre = df[df["war_dummy"] == 0]
    post = df[df["war_dummy"] == 1]

    plt.figure(figsize=(8, 6))
    plt.scatter(pre["PC1"], pre["PC2"], marker="o", label="Pre-war (before 2022-02)", alpha=0.7)
    plt.scatter(post["PC1"], post["PC2"], marker="^", label="Post-war (from 2022-02)", alpha=0.7)
    plt.title("2D PCA of inflation/cost features (alternative to 3D)")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# Main
def main():
    mm_path = "mm23.csv"
    ppi_path = "ppi.csv"
    bank_path = "Bank Rate history and data  Bank of England Database.csv"
    xlsx_path = "consumerpriceinflationdetailedreferencetables.xlsx"

    # Build master_table_v2
    out_master = "master_table_v2_uk_inflation_war_baseline_with_ppi.csv"
    master, mm = build_master_table_v2(mm_path, ppi_path, bank_path, out_master)
    print("Saved:", out_master)

    # Fig8: division-level stacked contributions
    out_fig8 = "fig8_cpih_division_contributions_stackplot.png"
    make_fig8_division_contributions(mm, xlsx_path, out_fig8)
    print("Saved:", out_fig8)

    # Fig9 + coefficients table: distributed lag regression
    out_fig9 = "fig9_distributed_lag_ppi_input_to_cpih.png"
    out_coef = "distributed_lag_ppi_input_coefficients.csv"
    make_fig9_distributed_lag(master, out_fig9, out_coef, max_lag=12)
    print("Saved:", out_fig9)
    print("Saved:", out_coef)

    # Fig10: PCA 2D
    out_fig10 = "fig10_pca_2d_pre_vs_post_war.png"
    make_fig10_pca_2d(master, out_fig10)
    print("Saved:", out_fig10)


if __name__ == "__main__":
    main()
