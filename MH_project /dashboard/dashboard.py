
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

def load_scores(root: Path) -> pd.DataFrame | None:
    p = root / "out" / "model_scores.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)

    needed = {"user_id", "date", "risk_score", "bucket"}
    if df.empty or not needed.issubset(df.columns):
        return None

    df["user_id"] = df["user_id"].astype(str)
    df["bucket"] = df["bucket"].astype(str).str.title()

    numeric_columns = [
        "risk_score","p_high","p_moderate","p_low",
        "neg_ratio","engagement_score","total_events"
    ]
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        else:
            df[col] = 0.0

    return df

st.set_page_config(page_title="Risk Dashboard", layout="wide")
st.title("Social Media Engagement and Mental Health Risk Analysis")


st.sidebar.header("⚙️ Settings")
root_path = st.sidebar.text_input(
    "Project root folder",
    str(Path(__file__).resolve().parents[1])
)

topn = st.sidebar.number_input("Top N users", 1, 50, 10)
bucket = st.sidebar.radio("Bucket for table", ["High", "Moderate", "Low"])

# Premium colors
COLORS = {
    "High": "#112A70",
    "Moderate": "#3B82F6",
    "Low": "#DBEAFE"
}


df = load_scores(Path(root_path))
if df is None:
    st.error("No valid model_scores.csv found in /out/")
    st.stop()

st.markdown("### 1 Risk Category Distribution")

bucket_counts = df["bucket"].value_counts()

fig_pie = px.pie(
    names=bucket_counts.index,
    values=bucket_counts.values,
    hole=0.45,
    color=bucket_counts.index,
    color_discrete_map=COLORS,
)
fig_pie.update_layout(height=350)
st.plotly_chart(fig_pie, use_container_width=True)

st.markdown("---")

st.markdown(f"### 2 Top {topn} Highest Risk Users")

# sort properly
df = df.sort_values("risk_score", ascending=False)

top_users = df.head(topn)

fig_bar = px.bar(
    top_users,
    x="risk_score",
    y="user_id",
    color="bucket",
    orientation="h",
    color_discrete_map=COLORS,
    text="risk_score",
)

fig_bar.update_layout(
    height=450,
    xaxis_title="Risk Score (0-1000)",
)

fig_bar.update_traces(
    texttemplate="%{text:.0f}",
    textposition="outside"
)

st.plotly_chart(fig_bar, use_container_width=True)
st.markdown("---")

st.markdown("### 3 User Activity vs Bucket (Top N users)")


if "total_events" not in top_users.columns:
    top_users["total_events"] = 0.0


scatter_df = top_users.sort_values("total_events", ascending=False)

fig_scatter = px.scatter(
    scatter_df,
    x="total_events",        
    y="user_id",             
    color="bucket",         
    color_discrete_map=COLORS,
    size="risk_score",       
    size_max=30,
    hover_data=["risk_score", "bucket", "total_events"],
)

fig_scatter.update_layout(
    height=500,
    xaxis_title="Total Events (posts + comments + likes)",
    yaxis_title="User ID",
    legend_title="Bucket Type",
)

fig_scatter.update_traces(
    marker=dict(opacity=0.9, line=dict(width=1, color="white"))
)

st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown("---")

st.markdown("### 4 Risk Score Heatmap ")

heat_df = df.sort_values("risk_score", ascending=False).head(30)

fig_heat = px.imshow(
    [heat_df["risk_score"].values],
    labels={"x": "User", "color": "Risk Score"},
    x=heat_df["user_id"].tolist(),
    color_continuous_scale=["#DBEAFE","#3B82F6","#1E3A8A" ],
    height=500
)


fig_heat.update_xaxes(tickangle=-75)
st.plotly_chart(fig_heat, use_container_width=True)

st.markdown("---")

st.markdown(f"### 5 Users in **{bucket}** bucket")

table_df = df[df["bucket"] == bucket].copy()
table_df = table_df.sort_values("risk_score", ascending=False)

cols = ["user_id","date","risk_score","bucket","engagement_score","total_events"]
cols = [c for c in cols if c in table_df.columns]

st.dataframe(table_df[cols], use_container_width=True)

st.download_button(
    "Download CSV",
    data=table_df.to_csv(index=False).encode("utf-8"),
    file_name=f"users_{bucket.lower()}.csv",
    mime="text/csv"
)
