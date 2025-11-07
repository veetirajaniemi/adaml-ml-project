import pandas as pd
import matplotlib.pyplot as plt

def load_data(file_path):
    df = pd.read_csv(file_path, delimiter=";", dtype="str")
    return df


def preprocess_data(df, start_date="17/12/2006", end_date="26/11/2010"):

    #Combine date and time
    df["datetime"] = pd.to_datetime(df["Date"]+" "+df["Time"], dayfirst=True, errors="coerce")
    #Convert other columns to numeric
    df["Global_active_power"] = pd.to_numeric(df["Global_active_power"],errors="coerce")
    df["Global_reactive_power"] = pd.to_numeric(df["Global_reactive_power"],errors="coerce")
    df["Voltage"] = pd.to_numeric(df["Voltage"],errors="coerce")
    df["Global_intensity"] = pd.to_numeric(df["Global_intensity"],errors="coerce")
    df["Sub_metering_1"] = pd.to_numeric(df["Sub_metering_1"],errors="coerce")
    df["Sub_metering_2"] = pd.to_numeric(df["Sub_metering_2"],errors="coerce")
    df["Sub_metering_3"] = pd.to_numeric(df["Sub_metering_3"],errors="coerce")


    #Timestamp rounded to floor hour
    df["timestamp"] = df["datetime"].dt.floor("h")

    # Interpolate missing values for short gaps up to 60 minutes
    numeric_cols = df.select_dtypes(include=["number"]).columns
    df.set_index("datetime",inplace=True)
    df[numeric_cols] = df[numeric_cols].interpolate(method="time", limit=60)

    # Resample to hourly data by taking mean of each hour
    df_hourly = df.resample("h").agg({
        "Global_active_power": "mean",
        "Global_reactive_power": "mean",
        "Global_intensity": "mean",
        "Voltage": "mean",
        "Global_intensity": "mean",
        "Sub_metering_1": "mean",
        "Sub_metering_2": "mean",
        "Sub_metering_3": "mean"
    })

    df_hourly = df_hourly[(df_hourly.index >= pd.to_datetime(start_date, dayfirst=True)) & 
                        (df_hourly.index < pd.to_datetime(end_date, dayfirst=True))]

    # Fill remaining missing values using weekly seasonality
    df_hourly = df_hourly.fillna(df_hourly.shift(freq="168h"))
    return df_hourly
