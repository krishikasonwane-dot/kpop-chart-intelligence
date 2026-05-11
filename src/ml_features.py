import pandas as pd
import numpy as np

def build_features(df):

    features = []

    for song_id, group in df.groupby("song_id"):

        feature = {
            "song_id": song_id,
            "avg_rank": group["position"].mean(),
            "best_rank": group["position"].min(),
            "rank_volatility": group["position"].std(),
            "avg_popularity": group["popularity"].mean(),
            "duration_min": group["duration_min"].mean(),
            "target": int(len(group) > 10)
        }

        features.append(feature)

    return pd.DataFrame(features)