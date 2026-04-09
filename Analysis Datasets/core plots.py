import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data (auto-detect)
PATH_CANDIDATES = [
    "master_table_v2_uk_inflation_war_baseline_with_ppi.csv",
    "master_table_v1_uk_inflation_war_baseline.csv",
    "MVD_uk_inflation_war_baseline.csv",
]

df = None
data_path_used = None
for p in PATH_CANDIDATES:
    try:
        df = pd.read_csv(p, parse_dates=["date"])
        data_path_used = p
        break
    except Exception:
        continue

if df is None:
    raise FileNotFoundError(
        "No input file found. Put one of these in the same folder:\n"
        + "\n".join(PATH_CANDIDATES)
    )

df = df.sort_values("date").reset_index(drop=True)
print("Loaded:", data_path_used, "| rows:", len(df))

# Basic settings
WAR_DATE = pd.Timestamp("2022-02-01")  # Russia invades Ukraine
START_DATE = pd.Timestamp("2015-01-01")  # plotting window start
df_plot = df[df["date"] >= START_DATE].copy()

# Plot 1: CPIH YoY + event line
plt.figure(figsize=(10, 5))
plt.plot(df_plot["date"], df_plot["cpih_yoy"], label="CPIH YoY (%)")
plt.axvline(WAR_DATE, linestyle="--", linewidth=1)
plt.title("CPIH YoY (%) with 2022-02 Event Marker")
plt.xlabel("Date")
plt.ylabel("YoY inflation (%)")
plt.legend()
plt.tight_layout()
plt.savefig("fig1_cpih_yoy_event.png", dpi=200)
plt.close()

# Plot 2: Multi-line categories + event line
plt.figure(figsize=(10, 5))
series = [
    ("food_yoy", "Food YoY (%)"),
    ("housing_fuels_yoy", "Housing/Water/Fuels YoY (%)"),
    ("transport_yoy", "Transport YoY (%)"),
    ("core_cpih_yoy", "Core CPIH YoY (%)"),
]
for col, lab in series:
    if col in df_plot.columns:
        plt.plot(df_plot["date"], df_plot[col], label=lab)

plt.axvline(WAR_DATE, linestyle="--", linewidth=1)
plt.title("CPIH Category YoY (%) with 2022-02 Event Marker")
plt.xlabel("Date")
plt.ylabel("YoY inflation (%)")
plt.legend()
plt.tight_layout()
plt.savefig("fig2_categories_multiline.png", dpi=200)
plt.close()

# Plot 3: Event-study (t=0 at 2022-02)
#   - plot change from t=0 level (percentage points)
#   - window: [-24, +24] months for readability
def month_diff(d, ref):
    return (d.year - ref.year) * 12 + (d.month - ref.month)

df_evt = df_plot.copy()
df_evt["t_month"] = df_evt["date"].apply(lambda d: month_diff(d, WAR_DATE))

WINDOW = 24
df_evt = df_evt[(df_evt["t_month"] >= -WINDOW) & (df_evt["t_month"] <= WINDOW)].copy()

event_row = df_evt[df_evt["t_month"] == 0]
if event_row.empty:
    raise ValueError("Event month (2022-02) not found in the filtered dataset.")
event_vals = event_row.iloc[0]

plt.figure(figsize=(10, 5))
event_series = [
    ("cpih_yoy", "CPIH YoY (pp from t=0)"),
    ("food_yoy", "Food YoY (pp from t=0)"),
    ("housing_fuels_yoy", "Housing/Water/Fuels YoY (pp from t=0)"),
    ("transport_yoy", "Transport YoY (pp from t=0)"),
    ("core_cpih_yoy", "Core CPIH YoY (pp from t=0)"),
    ("bank_rate", "Bank Rate (pp from t=0)"),
]
for col, lab in event_series:
    if col in df_evt.columns:
        y = pd.to_numeric(df_evt[col], errors="coerce") - float(event_vals[col])
        plt.plot(df_evt["t_month"], y, label=lab)

plt.axvline(0, linestyle="--", linewidth=1)
plt.title("Event Study: Change from 2022-02 Level (t=0)")
plt.xlabel("Months relative to 2022-02")
plt.ylabel("Change from t=0 (percentage points)")
plt.legend(ncols=2, fontsize=8)
plt.tight_layout()
plt.savefig("fig3_event_study_t0_2022_02.png", dpi=200)
plt.close()

print("Saved: fig1_cpih_yoy_event.png")
print("Saved: fig2_categories_multiline.png")
print("Saved: fig3_event_study_t0_2022_02.png")
