import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ----------------------------
# Paths (assume same folder as script)
# ----------------------------
MM_PATH = "mm23.csv"
PPI_PATH = "ppi.csv"
BANK_PATH = "Bank Rate history and data  Bank of England Database.csv"

# ----------------------------
# Constants
# ----------------------------
WAR_DATE = pd.Timestamp("2022-02-01")
START_DATE = pd.Timestamp("2015-01-01")
TEST_START_DATE = pd.Timestamp("2019-01-01")  # rolling forecast begins here

# ----------------------------
# Helpers
# ----------------------------
MONTH_MAP = {m.upper(): i for i, m in enumerate(
    ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"], start=1
)}

def parse_title_to_date(t: str) -> pd.Timestamp:
    """Parse 'YYYY MON' like '2022 FEB' into month-start Timestamp."""
    y, mon = t.split()
    return pd.Timestamp(int(y), MONTH_MAP[mon], 1)

def make_lag_features(df_in, cols, lags):
    """
    Efficient lag feature generation (avoids DataFrame fragmentation warning).
    """
    out = df_in.copy()
    lag_blocks = []
    for c in cols:
        block = pd.concat({f"{c}_l{L}": out[c].shift(L) for L in lags}, axis=1)
        lag_blocks.append(block)
    return pd.concat([out] + lag_blocks, axis=1)

# ----------------------------
# Step 1: Build master_table_v2_week2.csv (WAR only)
# ----------------------------
def build_master_table_week2(mm_path, ppi_path, bank_path, out_path="master_table_v2_week2.csv"):
    # ---- CPIH series from MM23 using CDID mapping
    mm = pd.read_csv(mm_path, low_memory=False)
    cdid_row = mm.loc[mm["Title"] == "CDID"].iloc[0]

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

    monthly_mask = mm["Title"].astype(str).str.match(r"^\d{4} [A-Z]{3}$")
    mm_month = mm.loc[monthly_mask, ["Title"] + list(colmap.values())].copy()
    mm_month["date"] = mm_month["Title"].apply(parse_title_to_date)

    df_cpi = mm_month.drop(columns=["Title"]).rename(columns={v: k for k, v in colmap.items()})
    for c in SERIES_CDIDS.keys():
        df_cpi[c] = pd.to_numeric(df_cpi[c], errors="coerce")
    df_cpi = df_cpi.sort_values("date").reset_index(drop=True)
    df_cpi = df_cpi[df_cpi["date"] >= START_DATE].copy()

    # ---- Bank Rate: step function -> monthly
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

    # ---- PPI: compute YoY from index series
    ppi = pd.read_csv(ppi_path, low_memory=False)
    ppi_month = ppi.loc[ppi["Title"].astype(str).str.match(r"^\d{4} [A-Z]{3}$")].copy()
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

    # ---- Merge master (WAR only)
    master = df_cpi.merge(df_bank, on="date", how="left").merge(df_ppi_yoy, on="date", how="left")
    master["war_dummy"] = (master["date"] >= WAR_DATE).astype(int)
    master = master.sort_values("date").reset_index(drop=True)

    master.to_csv(out_path, index=False)
    return master

# ----------------------------
# Step 2: Predictive model (Ridge vs AR baseline) with rolling validation
# ----------------------------
def rolling_forecast(feat_df, horizon, start_test_date, model_name="ridge"):
    d = feat_df.copy()
    d[f"target_h{horizon}"] = d["cpih_yoy"].shift(-horizon)

    feature_cols = [c for c in d.columns if c not in ["date"] and not c.startswith("target_")]
    d_model = d[["date"] + feature_cols + [f"target_h{horizon}"]].dropna().copy()

    test_dates = d_model.loc[d_model["date"] >= start_test_date, "date"].tolist()

    preds, actuals, dates_out = [], [], []

    if model_name == "ridge":
        model = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0, random_state=0))])
    elif model_name == "ar_only":
        model = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0, random_state=0))])
        feature_cols = [c for c in feature_cols if c.startswith("cpih_yoy_l")]
    else:
        raise ValueError("unknown model_name")

    for dt in test_dates:
        train = d_model[d_model["date"] < dt]
        test = d_model[d_model["date"] == dt]

        X_train = train[feature_cols].values
        y_train = train[f"target_h{horizon}"].values
        X_test = test[feature_cols].values
        y_test = test[f"target_h{horizon}"].values

        model.fit(X_train, y_train)
        y_pred = float(model.predict(X_test)[0])

        preds.append(y_pred)
        actuals.append(float(y_test[0]))
        dates_out.append(dt)

    out = pd.DataFrame({"date": dates_out, "actual": actuals, "pred": preds})
    mae = mean_absolute_error(out["actual"], out["pred"])
    rmse = np.sqrt(mean_squared_error(out["actual"], out["pred"]))  # compatible with all sklearn versions
    return out, mae, rmse


def run_predictive_models(master):
    base_cols = [
        "cpih_yoy","core_cpih_yoy","food_yoy","housing_fuels_yoy","transport_yoy",
        "ppi_input_yoy","ppi_output_yoy","ppi_import_yoy","ppi_export_yoy",
        "bank_rate","war_dummy"
    ]
    lags = list(range(1, 13))  # past 12 months

    # lag features for all numeric series except dummy
    lag_cols = [c for c in base_cols if c not in ["war_dummy"]]
    feat_df = make_lag_features(master[["date"] + base_cols], cols=lag_cols, lags=lags)
    feat_df["war_dummy"] = master["war_dummy"]

    results = []
    for h in [1, 3]:
        out_ridge, mae_r, rmse_r = rolling_forecast(feat_df, h, TEST_START_DATE, "ridge")
        out_ar, mae_a, rmse_a = rolling_forecast(feat_df, h, TEST_START_DATE, "ar_only")

        results.append({"horizon_months": h, "model": "Ridge (all features)", "MAE": mae_r, "RMSE": rmse_r})
        results.append({"horizon_months": h, "model": "AR baseline (CPIH lags only)", "MAE": mae_a, "RMSE": rmse_a})

        # Save predictions
        out_ridge.to_csv(f"preds_ridge_h{h}.csv", index=False)
        out_ar.to_csv(f"preds_ar_h{h}.csv", index=False)

        # Plot last 36 months
        for out, name in [(out_ridge, "ridge"), (out_ar, "ar")]:
            tail = out.tail(36)
            plt.figure(figsize=(10, 4))
            plt.plot(tail["date"], tail["actual"], label="Actual")
            plt.plot(tail["date"], tail["pred"], label="Predicted")
            plt.title(f"Rolling forecast: CPIH 12-month inflation, horizon={h} month(s) ({name})")
            plt.xlabel("Date")
            plt.ylabel("12-month inflation rate (%)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"fig11_forecast_h{h}_{name}.png", dpi=200)
            plt.close()

    eval_df = pd.DataFrame(results)
    eval_df.to_csv("predictive_model_rolling_eval.csv", index=False)
    return eval_df


# ----------------------------
# Main
# ----------------------------
def main():
    master = build_master_table_week2(MM_PATH, PPI_PATH, BANK_PATH, out_path="master_table_v2_week2.csv")
    print("Saved: master_table_v2_week2.csv")

    eval_df = run_predictive_models(master)
    print("Saved: predictive_model_rolling_eval.csv")
    print(eval_df)


if __name__ == "__main__":
    main()
    