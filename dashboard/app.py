import streamlit as st
import pandas as pd

st.title("🎵 K-Pop Analytics Dashboard")

df = pd.read_csv("data/south_korea_top50.csv")

st.write(df.head())