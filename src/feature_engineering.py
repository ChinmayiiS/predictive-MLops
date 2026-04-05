import pandas as pd

def create_features(df):
    df["temp_voltage_ratio"] = df["temperature"] / df["voltage"]
    df["error_rate"] = df["error_count"] / (df["print_count"] + 1)
    df["usage_intensity"] = df["print_count"] / (df["maintenance_gap"] + 1)
    return df
