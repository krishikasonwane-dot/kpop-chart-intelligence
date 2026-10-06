import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from streamlit_autorefresh import st_autorefresh

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="K-Pop Momentum Intelligence Dashboard",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- AUTO REFRESH ----------------
st_autorefresh(interval=10000, key="refresh")
# ---------------- NUDE LIGHT THEME CSS ----------------
st.markdown(
    """
<style>

/* Main App */
.stApp {
    background-color: #F7F4F2;
    color: #2B2B2B;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #EFE7E2;
    border-right: 1px solid #DDD6D1;
}

/* Titles */
h1 {
    color: #3A3A3A;
    font-weight: 700;
}

h2, h3 {
    color: #4A4A4A;
    font-weight: 600;
}

/* KPI Cards */
div[data-testid="metric-container"] {
    background-color: #FFFFFF;
    border: 1px solid #E6DED9;
    padding: 18px;
    border-radius: 16px;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.04);
}

/* KPI Labels */
div[data-testid="metric-container"] label {
    color: #7B6F68 !important;
    font-size: 14px;
    font-weight: 500;
}

/* KPI Numbers */
div[data-testid="metric-container"] div {
    color: #5C4B51 !important;
    font-size: 30px;
    font-weight: 700;
}

/* Tabs */
button[data-baseweb="tab"] {
    color: #6D625C;
    font-weight: 600;
}

/* Selected Filter Tags */
span[data-baseweb="tag"] {
    background-color: #D8C3B8 !important;
    color: #3A3A3A !important;
    border-radius: 8px;
}

/* Inputs */
.stTextInput input {
    background-color: #FFFFFF;
    color: #3A3A3A;
    border-radius: 10px;
}

/* Buttons */
.stButton button,
.stDownloadButton button {
    background-color: #C8B6A6;
    color: #FFFFFF;
    border-radius: 10px;
    border: none;
}

/* Tables */
[data-testid="stDataFrame"] {
    border: 1px solid #E6DED9;
    border-radius: 12px;
}

/* Chart Background */
.js-plotly-plot {
    border-radius: 14px;
}

</style>
""",
    unsafe_allow_html=True,
)

# ---------------- TITLE ----------------
st.title("🎵 K-Pop Momentum Intelligence Dashboard")
st.markdown("""
### Comeback Momentum • Chart Re-Entry • Fandom Intensity Analysis
""")

# ---------------- SIDEBAR ----------------
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/727/727245.png", width=120)

st.sidebar.title("🎛 Dashboard Filters")

# ---------------- FILE UPLOADER ----------------
uploaded_file = st.sidebar.file_uploader("📂 Upload CSV Dataset", type=["csv"])


# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data(file):

    df = pd.read_csv(file)

    df.columns = df.columns.str.strip()

    df["date"] = pd.to_datetime(df["date"])

    df["duration_min"] = round(df["duration_ms"] / 60000, 2)

    df["song_artist"] = (
        df["song"].str.lower().str.strip() + "_" + df["artist"].str.lower().str.strip()
    )

    return df


# ---------------- READ DATA ----------------
try:

    if uploaded_file is not None:
        df = load_data(uploaded_file)

    else:
        df = load_data("south_korea_top50.csv")

except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# ---------------- FILTERS ----------------
artist_filter = st.sidebar.multiselect(
    "🎤 Select Artist",
    options=sorted(df["artist"].unique()),
    default=sorted(df["artist"].unique()),
)

album_filter = st.sidebar.multiselect(
    "💿 Album Type",
    options=sorted(df["album_type"].unique()),
    default=sorted(df["album_type"].unique()),
)

explicit_filter = st.sidebar.multiselect(
    "⚠ Explicit Content",
    options=sorted(df["is_explicit"].unique()),
    default=sorted(df["is_explicit"].unique()),
)

song_search = st.sidebar.text_input("🔍 Search Song")

gap_filter = st.sidebar.slider(
    "🔄 Re-Entry Gap Days",
    1,
    30,
    3
)

# ---------------- FILTER DATA ----------------
filtered_df = df[
    (df["artist"].isin(artist_filter)) &
    (df["album_type"].isin(album_filter)) &
    (df["is_explicit"].isin(explicit_filter))
]

if song_search:
    filtered_df = filtered_df[
        filtered_df["song"].str.contains(
            song_search,
            case=False
        )
    ]

# ---------------- RE-ENTRY ANALYSIS ----------------
filtered_df = filtered_df.sort_values(
    by=["song_artist", "date"]
)

filtered_df["prev_date"] = (
    filtered_df.groupby("song_artist")["date"]
    .shift(1)
)

filtered_df["gap_days"] = (
    filtered_df["date"] -
    filtered_df["prev_date"]
).dt.days

filtered_df["re_entry"] = np.where(
    filtered_df["gap_days"] > gap_filter,
    "Yes",
    "No"
)

# ---------------- MOMENTUM SCORE ----------------
filtered_df["rank_improvement"] = (
    50 - filtered_df["position"]
)

filtered_df["momentum_score"] = (
    filtered_df["popularity"] +
    filtered_df["rank_improvement"]
)

# ---------------- KPI SECTION ----------------
st.subheader("📊 Dashboard KPIs")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "🎵 Songs",
    filtered_df["song"].nunique()
)

col2.metric(
    "🎤 Artists",
    filtered_df["artist"].nunique()
)

col3.metric(
    "🔄 Re-Entries",
    filtered_df[
        filtered_df["re_entry"] == "Yes"
    ].shape[0]
)

col4.metric(
    "🚀 Avg Momentum",
    round(
        filtered_df["momentum_score"].mean(),
        2
    )
)

st.divider()

# ---------------- TABS ----------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🚀 Momentum",
    "🔄 Re-Entry",
    "🎵 Content",
    "🏆 Fandom",
    "🤖 AI Prediction"
])

# ---------------- TOP MOMENTUM SONGS ----------------
with tab1:

    st.subheader("🔥 Top Momentum Songs")

    momentum_df = (
        filtered_df.groupby(
            ["song", "artist"],
            as_index=False
        )["momentum_score"]
        .mean()
        .sort_values(
            by="momentum_score",
            ascending=False
        )
        .head(10)
    )


fig1 = px.bar(
momentum_df, # type: ignore
    x="song",
    y="momentum_score",
    color="artist",
    title="Top Momentum Songs",
    color_discrete_sequence=[
    "#B39DDB",   # pastel purple
    "#CE93D8",   # soft pink-purple
    "#F48FB1",   # dusty pink
    "#9575CD",   # muted lavender
    "#7E57C2"    # deep violet
]

)

fig1.update_layout(
    plot_bgcolor="#FAF7FD",
    paper_bgcolor="#FAF7FD",
    font=dict(color="#4A4453"),
    title_font=dict(size=22),
    xaxis=dict(showgrid=False),
    yaxis=dict(gridcolor="#E8DFF2")
)


st.plotly_chart(
    fig1,
    use_container_width=True
)

# ---------------- RETENTION ANALYSIS ----------------
retention_df = (
    filtered_df.groupby(
        "song",
        as_index=False
    )["date"]
    .count()
    .rename(
        columns={
            "date": "retention_days"
        }
    )
)
fig2 = px.line(
    retention_df, # type: ignore
    x="song",
    y="retention_days",
    markers=True,
    title="Retention Analysis"
)

fig2.update_traces(
    line=dict(
        color="#9575CD",
        width=3
    ),
    marker=dict(
        size=8,
        color="#CE93D8"
    )
)

fig2.update_layout(
    plot_bgcolor="#FAF7FD",
    paper_bgcolor="#FAF7FD",
    font=dict(color="#4A4453"),
    title_font=dict(size=22),
    xaxis=dict(showgrid=False),
    yaxis=dict(gridcolor="#E8DFF2")
)

st.plotly_chart(
    fig2,
    use_container_width=True
)

# ---------------- RE-ENTRY TAB ----------------
with tab2:

    st.subheader("🔄 Re-Entry Timeline")

    reentry_df = filtered_df[
        filtered_df["re_entry"] == "Yes"
    ]

    if not reentry_df.empty:

        fig3 = px.scatter(
            reentry_df,
            x="date",
            y="position",
            color="artist",
            hover_data=["song"],
            title="Chart Re-Entry Timeline",
            color_discrete_sequence=[
                "#C7B8A3",
                "#D8CBBE",
                "#A68A7C",
                "#8B6F63"
            ]
        )

        fig3.update_yaxes(
            autorange="reversed"
        )

        fig3.update_layout(
            plot_bgcolor="#FFFDFB",
            paper_bgcolor="#FFFDFB",
            font=dict(color="#4A403D"),
            title_font=dict(size=22),
            xaxis=dict(showgrid=False),
            yaxis=dict(gridcolor="#E8DED7")
        )

        st.plotly_chart(
            fig3,
            use_container_width=True
        )

    else:
        st.info("No re-entry songs found.")

# ---------------- CONTENT TAB ----------------
with tab3:

    st.subheader("🎵 Single vs Album Momentum")

    fig4 = px.box(
        filtered_df,
        x="album_type",
        y="momentum_score",
        color="album_type",
        title="Album Type vs Momentum",
        color_discrete_sequence=["#BB86FC", "#FF79C6"]
    )

    fig4.update_layout(
        plot_bgcolor="#121212",
        paper_bgcolor="#121212",
        font=dict(color="white")
    )

    st.plotly_chart(
        fig4,
        use_container_width=True
    )

    st.subheader("⚠ Explicit Content Analysis")

    explicit_df = (
        filtered_df.groupby(
            "is_explicit",
            as_index=False
        )["momentum_score"]
        .mean()
    )

    fig5 = px.bar(
        explicit_df,
        x="is_explicit",
        y="momentum_score",
        title="Explicit vs Clean Songs",
        color_discrete_sequence=["#8E7DBE"]
    )

    fig5.update_layout(
        plot_bgcolor="#121212",
        paper_bgcolor="#121212",
        font=dict(color="white")
    )

    st.plotly_chart(
        fig5,
        use_container_width=True
    )

# ---------------- FANDOM TAB ----------------
with tab4:

    st.subheader("🏆 Fandom Intensity Leaderboard")

    fandom_df = (
        filtered_df.groupby(
            "artist",
            as_index=False
        )
        .agg({
            "momentum_score": "mean",
            "re_entry": lambda x:
            (x == "Yes").sum()
        })
    )

    fandom_df["fandom_intensity"] = (
        fandom_df["momentum_score"] +
        fandom_df["re_entry"] * 5
    )

    fandom_df = fandom_df.sort_values(
        by="fandom_intensity",
        ascending=False
    )

    fig6 = px.bar(
        fandom_df,
        x="artist",
        y="fandom_intensity",
        color="artist",
        title="Fandom Intensity Proxy Score",
        color_discrete_sequence=[
            "#8E7DBE",
            "#B784B7",
            "#7A89C2"
        ]
    )

    fig6.update_layout(
        plot_bgcolor="#121212",
        paper_bgcolor="#121212",
        font=dict(color="white")
    )

    st.plotly_chart(
        fig6,
        use_container_width=True
    )

# ---------------- AI PREDICTION TAB ----------------
with tab5:

    st.subheader("🤖 Popularity Prediction Model")

    model_df = filtered_df.copy()

    X = model_df[[
        "position",
        "duration_min",
        "total_tracks"
    ]]

    y = model_df["popularity"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = LinearRegression()

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    score = r2_score(y_test, predictions)

    st.metric(
        "📈 Model Accuracy (R² Score)",
        round(score, 2)
    )

    prediction_df = pd.DataFrame({
        "Actual Popularity": y_test,
        "Predicted Popularity": predictions
    })

    fig7 = px.scatter(
        prediction_df,
        x="Actual Popularity",
        y="Predicted Popularity",
        title="AI Popularity Prediction",
        color_discrete_sequence=["#BB86FC"]
    )

    fig7.update_layout(
        plot_bgcolor="#121212",
        paper_bgcolor="#121212",
        font=dict(color="white")
    )

    st.plotly_chart(
        fig7,
        use_container_width=True
    )

# ---------------- ALBUM COVERS ----------------
st.subheader("🖼 Album Covers")

cover_df = filtered_df[
    ["song", "artist", "album_cover_url"]
].drop_duplicates().head(6)

cols = st.columns(3)

for index, row in enumerate(
    cover_df.itertuples()
):

    with cols[index % 3]:

        st.image(
            row.album_cover_url,
            caption=f"{row.song} - {row.artist}"
        )

# ---------------- DOWNLOAD BUTTON ----------------
st.subheader("⬇ Download Dataset")

csv = filtered_df.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="📥 Download CSV",
    data=csv,
    file_name="kpop_dashboard_data.csv",
    mime="text/csv"
)

# ---------------- DATA TABLE ----------------
st.subheader("📋 Dataset Preview")

st.dataframe(
    filtered_df,
    use_container_width=True
)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("""
Made with ❤️ using Streamlit, Plotly,
Machine Learning & Entertainment Analytics
""")