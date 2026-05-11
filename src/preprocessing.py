import pandas as pd

def preprocess_data(path):

    df = pd.read_csv(path)

    df["date"] = pd.to_datetime(df["date"])

    df["song_id"] = (
        df["song"] + " - " + df["artist"]
    )

    df["duration_min"] = (
        df["duration_ms"] / 60000
    )

    return df