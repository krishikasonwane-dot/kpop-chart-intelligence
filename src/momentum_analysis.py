def calculate_momentum(df):
    
    df = df.copy()

    df["rank_change"] = (
        df.groupby("song_id")["position"]
        .diff()
        .fillna(0) * -1
    )

    df["popularity_change"] = (
        df.groupby("song_id")["popularity"]
        .diff()
        .fillna(0)
    )

    df["momentum_score"] = (
        df["rank_change"] * 0.7 +
        df["popularity_change"] * 0.3
    )

    return df