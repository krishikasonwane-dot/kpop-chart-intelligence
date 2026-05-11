import pandas as pd

def calculate_reentries(df):

    results = []

    for song_id, group in df.groupby("song_id"):

        group = group.sort_values("date")

        reentries = 0
        previous_date = None

        for current_date in group["date"]:

            if previous_date is not None:

                gap = (
                    current_date - previous_date
                ).days

                if gap > 2:
                    reentries += 1

            previous_date = current_date

        results.append({
            "song_id": song_id,
            "reentry_count": reentries
        })

    return pd.DataFrame(results)