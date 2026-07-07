from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DATA_PATH = Path("data/raw/ai4i2020.csv")
PLOTS_DIR = Path("notebooks/plots")
TARGET = "Machine failure"
FAILURE_MODES = ["TWF", "HDF", "PWF", "OSF", "RNF"]
SENSOR_COLUMNS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]


def section(title: str) -> None:
    print(f"\n{'=' * len(title)}")
    print(title)
    print(f"{'=' * len(title)}")


def save_sensor_histograms(df: pd.DataFrame) -> list[Path]:
    saved_paths = []
    for column in SENSOR_COLUMNS:
        plt.figure(figsize=(9, 5))
        sns.histplot(
            data=df,
            x=column,
            hue=TARGET,
            bins=40,
            stat="density",
            common_norm=False,
            element="step",
        )
        plt.title(f"{column} distribution by failure label")
        plt.tight_layout()
        output_path = PLOTS_DIR / f"{column.lower().replace(' ', '_').replace('[', '').replace(']', '').replace('/', '_')}_by_failure.png"
        plt.savefig(output_path, dpi=150)
        plt.close()
        saved_paths.append(output_path)
    return saved_paths


def save_correlation_heatmap(df: pd.DataFrame) -> Path:
    corr_columns = [*SENSOR_COLUMNS, TARGET]
    corr = df[corr_columns].corr(numeric_only=True)
    plt.figure(figsize=(9, 7))
    sns.heatmap(corr, annot=True, cmap="vlag", center=0, fmt=".3f", square=True)
    plt.title("Numeric feature correlation heatmap")
    plt.tight_layout()
    output_path = PLOTS_DIR / "numeric_feature_target_correlation_heatmap.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing {DATA_PATH}. Download the AI4I 2020 Predictive Maintenance Dataset "
            "from Kaggle and place ai4i2020.csv in data/raw/."
        )

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", None)
    sns.set_theme(style="whitegrid")

    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")

    section("A. Basic shape, columns, and dtypes")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    print("\nColumn names:")
    print(df.columns.to_list())
    print("\nDtypes:")
    print(df.dtypes.to_string())

    section("B. First 10 rows")
    print(df.head(10).to_string(index=False))

    section("C. Missing values")
    missing = pd.DataFrame(
        {
            "missing_count": df.isna().sum(),
            "missing_pct": (df.isna().mean() * 100).round(2),
        }
    )
    print(missing.to_string())

    section("D. Target variable and failure modes")
    target_balance = pd.DataFrame(
        {
            "count": df[TARGET].value_counts(dropna=False).sort_index(),
            "pct": (df[TARGET].value_counts(normalize=True, dropna=False).sort_index() * 100).round(2),
        }
    )
    print("\nMachine failure class balance:")
    print(target_balance.to_string())

    mode_summary = pd.DataFrame(
        {
            "flagged_rows": df[FAILURE_MODES].sum(),
            "flagged_pct": (df[FAILURE_MODES].mean() * 100).round(2),
        }
    )
    print("\nFailure mode flagged rows:")
    print(mode_summary.to_string())

    failure_mode_count = df[FAILURE_MODES].sum(axis=1)
    overlap_summary = pd.DataFrame(
        {
            "rows": failure_mode_count.value_counts().sort_index(),
            "pct": (failure_mode_count.value_counts(normalize=True).sort_index() * 100).round(2),
        }
    )
    overlap_summary.index.name = "failure_modes_flagged"
    print("\nRows by number of failure modes flagged:")
    print(overlap_summary.to_string())
    print(f"\nRows with multiple failure modes flagged: {(failure_mode_count > 1).sum()}")

    pair_rows = []
    for left, right in combinations(FAILURE_MODES, 2):
        pair_rows.append({"pair": f"{left}+{right}", "co_flagged_rows": int(((df[left] == 1) & (df[right] == 1)).sum())})
    print("\nPairwise failure mode overlap:")
    print(pd.DataFrame(pair_rows).to_string(index=False))

    section("E. Numeric sensor summary statistics")
    print(df[SENSOR_COLUMNS].describe().T.to_string())

    section("F. Distribution plots")
    histogram_paths = save_sensor_histograms(df)
    for path in histogram_paths:
        print(f"Saved {path}")

    section("G. Correlation matrix and heatmap")
    corr_columns = [*SENSOR_COLUMNS, TARGET]
    corr = df[corr_columns].corr(numeric_only=True)
    print(corr.to_string(float_format=lambda value: f"{value:.6f}"))
    feature_target_corr = corr[TARGET].drop(TARGET).sort_values(key=lambda series: series.abs(), ascending=False)
    print("\nFeature correlations with Machine failure:")
    print(feature_target_corr.to_string(float_format=lambda value: f"{value:.6f}"))
    heatmap_path = save_correlation_heatmap(df)
    print(f"\nSaved {heatmap_path}")

    section("H. Product quality variant breakdown")
    type_breakdown = (
        df.groupby("Type")
        .agg(
            rows=(TARGET, "size"),
            failures=(TARGET, "sum"),
            failure_rate_pct=(TARGET, lambda values: values.mean() * 100),
        )
        .sort_index()
    )
    type_breakdown["failure_rate_pct"] = type_breakdown["failure_rate_pct"].round(2)
    print(type_breakdown.to_string())

    section("I. Data leakage and row ordering checks")
    print(f"UDI is monotonic increasing: {df['UDI'].is_monotonic_increasing}")
    print(f"UDI unique values: {df['UDI'].nunique()} of {len(df)} rows")
    print(f"Product ID unique values: {df['Product ID'].nunique()} of {len(df)} rows")
    print("Potential leakage columns for predicting Machine failure at runtime:")
    print(f"- Failure mode flags {FAILURE_MODES} are post-outcome labels/diagnostics, not pre-failure sensor inputs.")
    print("- Product ID is nearly row-specific and should not be treated as a generalizable predictive feature.")
    print("- UDI is a row/order identifier; it may imply collection order but is not a timestamp or operational feature.")
    print("No explicit timestamp or prediction horizon column is present.")

    section("J. Duplicates and invalid sensor readings")
    print(f"Duplicate full rows: {df.duplicated().sum()}")
    print(f"Duplicate rows excluding UDI/Product ID: {df.drop(columns=['UDI', 'Product ID']).duplicated().sum()}")
    invalid_checks = {
        "negative_air_temperature_k": int((df["Air temperature [K]"] < 0).sum()),
        "negative_process_temperature_k": int((df["Process temperature [K]"] < 0).sum()),
        "negative_rotational_speed_rpm": int((df["Rotational speed [rpm]"] < 0).sum()),
        "negative_torque_nm": int((df["Torque [Nm]"] < 0).sum()),
        "negative_tool_wear_min": int((df["Tool wear [min]"] < 0).sum()),
        "process_temp_below_air_temp": int((df["Process temperature [K]"] < df["Air temperature [K]"]).sum()),
    }
    print(pd.Series(invalid_checks).to_string())
    print("\nSensor min/max values:")
    print(df[SENSOR_COLUMNS].agg(["min", "max"]).T.to_string())


if __name__ == "__main__":
    main()
