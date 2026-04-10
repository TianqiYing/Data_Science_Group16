import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

MASTER_V2_PATH = "master_table_v2_uk_inflation_war_baseline_with_ppi.csv"

WAR_DATE = pd.Timestamp("2022-02-01")    # t=0
START_DATE = pd.Timestamp("2015-01-01")  # plot window start
WINDOW = 24                              # event-study window in months ([-24, +24])

# Load
df = pd.read_csv(MASTER_V2_PATH, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
dfp = df[df["date"] >= START_DATE].copy()

# Basic sanity checks
required_cols = ["cpih_yoy", "ppi_input_yoy", "ppi_output_yoy", "bank_rate"]
missing = [c for c in required_cols if c not in dfp.columns]
if missing:
    raise ValueError(f"Missing required columns in master_table_v2: {missing}")

# Plot 4: Overlay CPIH vs PPI Input/Output (YoY) + event marker
plt.figure(figsize=(10, 5))
plt.plot(dfp["date"], dfp["cpih_yoy"], label="CPIH YoY (%)")
plt.plot(dfp["date"], dfp["ppi_input_yoy"], label="PPI Input YoY (%)")
plt.plot(dfp["date"], dfp["ppi_output_yoy"], label="PPI Output YoY (%)")
plt.axvline(WAR_DATE, linestyle="--", linewidth=1)
plt.title("CPIH vs PPI Input/Output YoY (%) with 2022-02 Event Marker")
plt.xlabel("Date")
plt.ylabel("YoY (%)")
plt.legend()
plt.tight_layout()
plt.savefig("fig4_ppi_overlay_cpih.png", dpi=200)
plt.close()

# Plot 5: Lag correlation (CPIH vs lagged PPI) for lags 0..12
#   corr(CPIH_t, PPI_{t-L})
lags = list(range(0, 13))
corr_input, corr_output = [], []

y = dfp["cpih_yoy"]
for L in lags:
    x_in = dfp["ppi_input_yoy"].shift(L)
    x_out = dfp["ppi_output_yoy"].shift(L)

    corr_input.append(pd.concat([x_in, y], axis=1).dropna().corr().iloc[0, 1])
    corr_output.append(pd.concat([x_out, y], axis=1).dropna().corr().iloc[0, 1])

plt.figure(figsize=(10, 5))
plt.bar([l - 0.15 for l in lags], corr_input, width=0.3, label="corr(CPIH, PPI Input lag L)")
plt.bar([l + 0.15 for l in lags], corr_output, width=0.3, label="corr(CPIH, PPI Output lag L)")
plt.axhline(0, linewidth=1)
plt.title("Lag Correlation: CPIH vs Lagged PPI (0–12 months)")
plt.xlabel("Lag L (months) where PPI is shifted by L months")
plt.ylabel("Correlation")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("fig5_lag_correlation_ppi_cpih.png", dpi=200)
plt.close()

# Plot 6: Correlation matrix heatmap (categories + PPI + Bank Rate)
vars_for_corr = [
    "cpih_yoy", "food_yoy", "housing_fuels_yoy", "transport_yoy", "core_cpih_yoy",
    "ppi_input_yoy", "ppi_output_yoy", "ppi_import_yoy", "ppi_export_yoy",
    "bank_rate"
]
vars_for_corr = [v for v in vars_for_corr if v in dfp.columns]

corr = dfp[vars_for_corr].corr()

plt.figure(figsize=(10, 8))
plt.imshow(corr.values, aspect="auto")
plt.colorbar()
plt.xticks(range(len(vars_for_corr)), vars_for_corr, rotation=90)
plt.yticks(range(len(vars_for_corr)), vars_for_corr)
plt.title("Correlation Matrix Heatmap (Categories + PPI + Bank Rate)")
plt.tight_layout()
plt.savefig("fig6_corr_matrix_categories_ppi_bank.png", dpi=200)
plt.close()

# Plot 7: Event-study including PPI + Bank Rate (change from t=0 level)
def month_diff(d, ref):
    return (d.year - ref.year) * 12 + (d.month - ref.month)

dfe = dfp.copy()
dfe["t_month"] = dfe["date"].apply(lambda d: month_diff(d, WAR_DATE))
dfe = dfe[(dfe["t_month"] >= -WINDOW) & (dfe["t_month"] <= WINDOW)].copy()

event_row = dfe[dfe["t_month"] == 0]
if event_row.empty:
    raise ValueError("Event month (2022-02) not found in the filtered dataset.")
event_vals = event_row.iloc[0]

plt.figure(figsize=(10, 5))
series = [
    ("cpih_yoy", "CPIH YoY (pp from t=0)"),
    ("ppi_input_yoy", "PPI Input YoY (pp from t=0)"),
    ("ppi_output_yoy", "PPI Output YoY (pp from t=0)"),
    ("ppi_import_yoy", "PPI Import YoY (pp from t=0)"),
    ("ppi_export_yoy", "PPI Export YoY (pp from t=0)"),
    ("bank_rate", "Bank Rate (pp from t=0)"),
]
for col, lab in series:
    if col in dfe.columns:
        y = pd.to_numeric(dfe[col], errors="coerce") - float(event_vals[col])
        plt.plot(dfe["t_month"], y, label=lab)

plt.axvline(0, linestyle="--", linewidth=1)
plt.title("Event Study: PPI & CPIH Change from 2022-02 Level (t=0)")
plt.xlabel("Months relative to 2022-02")
plt.ylabel("Change from t=0 (percentage points)")
plt.legend(ncols=2, fontsize=8)
plt.tight_layout()
plt.savefig("fig7_event_study_ppi_cpih_t0.png", dpi=200)
plt.close()

print("Saved: fig4_ppi_overlay_cpih.png")
print("Saved: fig5_lag_correlation_ppi_cpih.png")
print("Saved: fig6_corr_matrix_categories_ppi_bank.png")
print("Saved: fig7_event_study_ppi_cpih_t0.png")
