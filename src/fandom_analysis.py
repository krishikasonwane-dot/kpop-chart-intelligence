import pandas as pd

def fandom_intensity(df, reentry_df):

    metrics = (
        df.groupby("song_id")
        .agg({
            "popularity": "max",
            "position": "min"
        })
        .reset_index()
    )

    metrics = metrics.merge(
        reentry_df,
        on="song_id",
        how="left"
    )

    metrics["reentry_count"] = (
        metrics["reentry_count"]
        .fillna(0)
    )

    metrics["fandom_score"] = (
        metrics["reentry_count"] * 2 +
        (100 - metrics["popularity"]) * 0.5
    )

    return metrics