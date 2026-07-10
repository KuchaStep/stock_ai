import os
import sqlite3
import subprocess
import sys

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")
DB_PATH = os.path.join(BASE_DIR, "database", "stock.db")
MODEL_PATH = os.path.join(BASE_DIR, "data", "models", "stock_model_v4.pkl")

if SCRIPTS_DIR not in sys.path:
    sys.path.append(SCRIPTS_DIR)

from feature_utils import FEATURES_V4, add_v4_features


st.set_page_config(page_title="日本株AI予想", layout="wide")
st.title("日本株AI予想ダッシュボード")


def read_sql(query: str) -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql(query, conn)


def run_script(script_name: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, os.path.join(SCRIPTS_DIR, script_name)],
        cwd=BASE_DIR,
        capture_output=True,
        text=True,
    )


with st.sidebar:
    st.header("更新")
    if st.button("データ更新"):
        with st.spinner("差分データを取得中"):
            result_data = run_script("update_latest.py")
        if result_data.returncode == 0 :
            st.success("差分データを更新しました")
            st.rerun()
        else:
            st.error("差分データ更新に失敗しました")
            st.text(result_data.stderr)

    if st.button("特徴量とターゲットを再作成"):
        with st.spinner("features_v4 と training_data_v4 を作成中..."):
            result_features = run_script("create_features_v4.py")
            result_target = run_script("create_target_v4.py")

        if result_features.returncode == 0 and result_target.returncode == 0:
            st.success("データを更新しました")
            st.rerun()
        else:
            st.error("データ更新に失敗しました")
            st.text(result_features.stderr)
            st.text(result_target.stderr)

    if st.button("v4モデルを再学習"):
        with st.spinner("モデルを学習中..."):
            result_train = run_script("train_model_v4.py")

        if result_train.returncode == 0:
            st.success("モデルを更新しました")
            st.text(result_train.stdout)
            st.rerun()
        else:
            st.error("モデル学習に失敗しました")
            st.text(result_train.stderr)


prices = read_sql("SELECT * FROM prices ORDER BY code, date")
fundamentals = read_sql("SELECT * FROM fundamentals ORDER BY code, announcement_date")
stocks = read_sql("SELECT * FROM stocks")
dividends = read_sql("SELECT * FROM dividends ORDER BY code, date")

st.caption(f"最新株価日: {prices['date'].max()}")

if not os.path.exists(MODEL_PATH):
    st.warning("v4モデルが見つかりません。サイドバーからモデルを再学習してください。")
    st.stop()

model = joblib.load(MODEL_PATH)

features_df = add_v4_features(prices, fundamentals, dividends)
latest = features_df.sort_values("date").groupby("code").tail(1)
latest = latest.dropna(subset=["MA25_ratio", "volatility_25"]).copy()

latest["up_probability"] = model.predict_proba(latest[FEATURES_V4])[:, 1] * 100
latest["code"] = latest["code"].astype(str)
stocks["code"] = stocks["code"].astype(str)
latest = latest.merge(stocks[["code", "name"]], on="code", how="left")

ranking = (
    latest[["code", "name", "date", "close", "MA25_ratio", "MA75_ratio", "volatility_25", "up_probability", "eps", "PER", "PBR", "ROE", "ROA", "equity_ratio", "dividend_yield"]]
    .sort_values("up_probability", ascending=False)
    .reset_index(drop=True)
)

st.subheader("買い候補 TOP5")
top_cols = st.columns(5)
for i, (_, row) in enumerate(ranking.head(5).iterrows()):
    with top_cols[i]:
        st.metric(
            f"{i + 1}位 {row['name'] or row['code']}",
            f"{row['up_probability']:.1f}%",
            f"{row['close']:.0f}円",
        )

st.subheader("ランキング")
# 表示用に数値をフォーマット
display_df = ranking.copy()
display_df["MA25_ratio"] = display_df["MA25_ratio"] * 100
display_df["MA75_ratio"] = display_df["MA75_ratio"] * 100
display_df["volatility_25"] = display_df["volatility_25"] * 100
display_df["ROE"] = display_df["ROE"] * 100
display_df["ROA"] = display_df["ROA"] * 100
display_df["equity_ratio"] = display_df["equity_ratio"] * 100
display_df["dividend_yield"] = display_df["dividend_yield"] * 100

for col in ["MA25_ratio", "MA75_ratio", "volatility_25", "ROE", "ROA", "equity_ratio"]:
    display_df[col] = display_df[col].map(lambda x: f"{x:.1f}%" if pd.notna(x) else "-")
display_df["dividend_yield"] = display_df["dividend_yield"].map(lambda x: f"{x:.2f}%" if pd.notna(x) else "-")
display_df["close"] = display_df["close"].map(lambda x: f"{x:,.0f}円" if pd.notna(x) else "-")
display_df["eps"] = display_df["eps"].map(lambda x: f"{x:.1f}円" if pd.notna(x) else "-")
display_df["PER"] = display_df["PER"].map(lambda x: f"{x:.1f}倍" if pd.notna(x) else "-")
display_df["PBR"] = display_df["PBR"].map(lambda x: f"{x:.2f}倍" if pd.notna(x) else "-")
display_df["up_probability"] = display_df["up_probability"].map(lambda x: f"{x:.1f}%" if pd.notna(x) else "-")

st.dataframe(
    display_df.rename(
        columns={
            "code": "コード",
            "name": "銘柄名",
            "date": "日付",
            "close": "終値",
            "MA25_ratio": "25日線乖離",
            "MA75_ratio": "75日線乖離",
            "volatility_25": "25日ボラ",
            "up_probability": "上昇確率",
            "eps": "EPS",
            "PER": "PER",
            "PBR": "PBR",
            "ROE": "ROE",
            "ROA": "ROA",
            "equity_ratio": "自己資本比率",
            "dividend_yield": "配当利回り",
        }
    ),
    use_container_width=True,
)

st.subheader("銘柄分析")
selected_label = st.selectbox(
    "銘柄",
    (ranking["code"] + " " + ranking["name"].fillna("")).tolist(),
)
selected_code = selected_label.split(" ")[0]

chart_df = features_df[features_df["code"].astype(str) == selected_code].copy()
chart_df["date"] = pd.to_datetime(chart_df["date"])
chart_df = chart_df.tail(200)

selected_row = ranking[ranking["code"] == selected_code].iloc[0]
c1, c2, c3, c4 = st.columns(4)
c1.metric("終値", f"{selected_row['close']:.0f}円")
c2.metric("上昇確率", f"{selected_row['up_probability']:.1f}%")
c3.metric("25日線乖離", f"{selected_row['MA25_ratio'] * 100:.1f}%")
c4.metric("25日ボラ", f"{selected_row['volatility_25'] * 100:.1f}%")

c5, c6, c7, c8 = st.columns(4)
c5.metric("EPS", f"{selected_row['eps']:.1f}円" if pd.notna(selected_row['eps']) else "-")
c6.metric("PER", f"{selected_row['PER']:.1f}倍" if pd.notna(selected_row['PER']) else "-")
c7.metric("PBR", f"{selected_row['PBR']:.2f}倍" if pd.notna(selected_row['PBR']) else "-")
c8.metric("配当利回り", f"{selected_row['dividend_yield'] * 100:.2f}%" if pd.notna(selected_row['dividend_yield']) else "-")

c9, c10, c11, c12 = st.columns(4)
c9.metric("ROE", f"{selected_row['ROE'] * 100:.1f}%" if pd.notna(selected_row['ROE']) else "-")
c10.metric("ROA", f"{selected_row['ROA'] * 100:.1f}%" if pd.notna(selected_row['ROA']) else "-")
c11.metric("自己資本比率", f"{selected_row['equity_ratio'] * 100:.1f}%" if pd.notna(selected_row['equity_ratio']) else "-")
c12.write("")

fig = go.Figure()
fig.add_trace(
    go.Candlestick(
        x=chart_df["date"],
        open=chart_df["open"],
        high=chart_df["high"],
        low=chart_df["low"],
        close=chart_df["close"],
        name="株価",
    )
)
fig.add_trace(
    go.Scatter(
        x=chart_df["date"],
        y=chart_df["close"].rolling(25).mean(),
        name="MA25",
    )
)
fig.add_trace(
    go.Scatter(
        x=chart_df["date"],
        y=chart_df["close"].rolling(75).mean(),
        name="MA75",
    )
)
fig.update_layout(height=560, xaxis_rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)

st.subheader("OOS簡易バックテスト")
try:
    backtest_df = read_sql("SELECT * FROM training_data_v4")
    backtest_df = backtest_df.dropna(subset=["future_return_60"]).sort_values("date")
    dates = sorted(backtest_df["date"].unique())
    split_date = dates[int(len(dates) * 0.8)]
    test_df = backtest_df[backtest_df["date"] >= split_date].copy()
    test_df["score"] = model.predict_proba(test_df[FEATURES_V4])[:, 1]

    returns = []
    for date in sorted(test_df["date"].unique())[::60]:
        day = test_df[test_df["date"] == date]
        top = day.sort_values("score", ascending=False).head(5)
        if len(top) == 5:
            returns.append(top["future_return_60"].mean() - 0.003)

    result_df = pd.DataFrame({"return": returns})
    equity = (1 + result_df["return"]).cumprod() * 100_000
    win_rate = (result_df["return"] > 0).mean() * 100

    b1, b2, b3 = st.columns(3)
    b1.metric("リバランス回数", f"{len(result_df)}")
    b2.metric("勝率", f"{win_rate:.1f}%")
    b3.metric("最終資産", f"{equity.iloc[-1]:,.0f}円" if len(equity) else "-")
    st.line_chart(equity)
except Exception as exc:
    st.info(f"バックテストを表示できません: {exc}")
