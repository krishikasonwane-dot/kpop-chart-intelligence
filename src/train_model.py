import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from preprocessing import preprocess_data
from ml_features import build_features

# ---------------- LOAD ----------------
df = preprocess_data(
    "data/south_korea_top50.csv"
)

# ---------------- FEATURES ----------------
feature_df = build_features(df)

X = feature_df.drop(
    columns=["song_id", "target"]
)

y = feature_df["target"]

# ---------------- SPLIT ----------------
X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )
)

# ---------------- MODEL ----------------
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# ---------------- EVALUATION ----------------
predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("Accuracy:", accuracy)

# ---------------- SAVE MODEL ----------------
joblib.dump(
    model,
    "models/reentry_predictor.pkl"
)