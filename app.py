import pandas as pd
import streamlit as st
from datetime import datetime
import plotly.express as px

# カスタムCSS設定
st.set_page_config(
    page_title="Sales Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# カスタムCSS
st.markdown("""
<style>
/* メインタイトル */
.main-title {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    margin-bottom: 30px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
}

/* セクション帯 */
.section-header {
    background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
    color: white;
    padding: 12px 20px;
    border-radius: 8px;
    margin: 20px 0 15px 0;
    font-weight: bold;
    box-shadow: 0 2px 10px rgba(79, 172, 254, 0.3);
    text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.5);
}

.filter-section {
    background: white;
    color: #2d2d2d;
    padding: 12px 20px;
    border-radius: 8px;
    margin: 20px 0 15px 0;
    font-weight: bold;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    border: 1px solid #e9ecef;
}

.kpi-section {
    background: white;
    color: #2d2d2d;
    padding: 12px 20px;
    border-radius: 8px;
    margin: 20px 0 15px 0;
    font-weight: bold;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    border: 1px solid #e9ecef;
}

.chart-section {
    background: white;
    color: #2d2d2d;
    padding: 12px 20px;
    border-radius: 8px;
    margin: 20px 0 15px 0;
    font-weight: bold;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    border: 1px solid #e9ecef;
}

.data-section {
    background: white;
    color: #2d2d2d;
    padding: 12px 20px;
    border-radius: 8px;
    margin: 20px 0 15px 0;
    font-weight: bold;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    border: 1px solid #e9ecef;
}

/* KPIメトリクス */
.metric-container {
    background: white;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    border-left: 5px solid #667eea;
    margin: 10px 0;
}

/* チャートコンテナ */
.chart-container {
    background: white;
    padding: 15px;
    border-radius: 10px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    margin: 15px 0;
}

/* フィルターコンテナ */
.filter-container {
    background: #f8f9fa;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    margin: 15px 0;
    border: 1px solid #e9ecef;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────
# データ読み込み
# ─────────────────────────────
df = pd.read_csv("data/sample_sales.csv", parse_dates=["date"])

# ─────────────────────────────
# UI ― フィルター類
# ─────────────────────────────
st.markdown('<div class="main-title"><h1>📊 Sales Dashboard</h1><p>売上データの可視化・分析ツール</p></div>', unsafe_allow_html=True)

st.markdown('<div class="filter-section">🎯 データフィルタ設定</div>', unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="filter-container">', unsafe_allow_html=True)

    min_date = df["date"].min().to_pydatetime()
    max_date = df["date"].max().to_pydatetime()

    date_range = st.slider(
        "📅 期間を選択",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        format="YYYY-MM-DD",
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        cats = st.multiselect(
            "🏷️ カテゴリを選択（複数可）",
            options=df["category"].unique().tolist(),
            default=df["category"].unique().tolist(),
        )
    with col2:
        regions = st.multiselect(
            "🌍 地域を選択（複数可）",
            options=df["region"].unique().tolist(),
            default=df["region"].unique().tolist(),
        )
    with col3:
        channels = st.multiselect(
            "📱 チャネルを選択（複数可）",
            options=df["sales_channel"].unique().tolist(),
            default=df["sales_channel"].unique().tolist(),
        )

    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────
# フィルタリング
# ─────────────────────────────
start_dt = pd.to_datetime(date_range[0])
end_dt   = pd.to_datetime(date_range[1])

df_filt = df[
    (df["date"].between(start_dt, end_dt))
    & (df["category"].isin(cats))
    & (df["region"].isin(regions))
    & (df["sales_channel"].isin(channels))
]

# ─────────────────────────────
# KPI
# ─────────────────────────────
total_revenue  = int(df_filt["revenue"].sum())
total_units    = int(df_filt["units"].sum())
avg_unit_price = int(df_filt["unit_price"].mean()) if not df_filt.empty else 0

st.markdown('<div class="kpi-section">📈 主要業績指標（KPI）</div>', unsafe_allow_html=True)

with st.container():
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("💰 売上合計 (円)", f"{total_revenue:,.0f}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("📦 販売数量 (個)", f"{total_units:,}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("💵 平均単価 (円)", f"{avg_unit_price:,.0f}")
        st.markdown('</div>', unsafe_allow_html=True)

st.divider()

st.markdown('<div class="chart-section">📊 データ可視化チャート</div>', unsafe_allow_html=True)

# 1) 日別売上推移
with st.container():
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    revenue_daily = (
        df_filt.groupby("date", as_index=False)["revenue"].sum().sort_values("date")
    )
    fig_daily = px.line(
        revenue_daily,
        x="date",
        y="revenue",
        markers=True,
        labels={"date": "日付", "revenue": "売上 (円)"},
        title="🗓️ 日別売上推移",
        color_discrete_sequence=['#667eea']
    )
    fig_daily.update_layout(
        height=350,
        hovermode="x unified",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_daily, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# 2) カテゴリ別売上
with st.container():
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    revenue_by_cat = (
        df_filt.groupby("category", as_index=False)["revenue"].sum().sort_values("revenue")
    )
    fig_cat = px.bar(
        revenue_by_cat,
        x="category",
        y="revenue",
        text_auto=".2s",
        labels={"category": "カテゴリ", "revenue": "売上 (円)"},
        title="🏷️ カテゴリ別売上",
        color_discrete_sequence=['#4facfe', '#00f2fe', '#fa709a', '#fee140']
    )
    fig_cat.update_layout(
        height=350,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_cat, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# 3) 地域別売上
with st.container():
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    revenue_by_region = (
        df_filt.groupby("region", as_index=False)["revenue"].sum().sort_values("revenue")
    )
    fig_region = px.bar(
        revenue_by_region,
        x="region",
        y="revenue",
        text_auto=".2s",
        labels={"region": "地域", "revenue": "売上 (円)"},
        title="🌎 地域別売上",
        color_discrete_sequence=['#a8edea', '#fed6e3', '#ffecd2', '#fcb69f']
    )
    fig_region.update_layout(
        height=350,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_region, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.divider()

st.markdown('<div class="data-section">📄 詳細データテーブル</div>', unsafe_allow_html=True)

with st.expander("🔍 フィルタ後データを表示", expanded=False):
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.dataframe(
        df_filt.reset_index(drop=True),
        use_container_width=True,
        height=400
    )
    st.markdown('</div>', unsafe_allow_html=True)
