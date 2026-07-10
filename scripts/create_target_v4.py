import sqlite3
import pandas as pd
import numpy as np

conn = sqlite3.connect("database/stock.db")

# 1. 特徴量データの読み込み
df = pd.read_sql("SELECT * FROM features_v4 ORDER BY date, code", conn)
df["date"] = pd.to_datetime(df["date"])

# 2. 市場平均リターンの計算（地合いの決定）
# 同一日にデータがある全銘柄の「60営業日先までの単純リターン」を後で計算するため、
# まずは銘柄ごとに60日先の株価比率を算出します。
all_data = []

for code, group in df.groupby("code"):
    group = group.sort_values("date").copy()
    
    # 60営業日（約3ヶ月）先のリターン
    # shift(-60)により、現在の行に「60日先の終値 / 現在の終値 - 1」を格納
    group["future_return_60"] = group["close"].shift(-60) / group["close"] - 1
    all_data.append(group)

df_target = pd.concat(all_data, ignore_index=True)

# 3. 【重要】市場平均に対する超過リターン（アルファ）の算出
# 同じ日付の全銘柄の平均リターンを「市場リターン」と定義
market_avg_60 = df_target.groupby("date")["future_return_60"].transform("mean")
df_target["alpha_return_60"] = df_target["future_return_60"] - market_avg_60

# 4. 目的変数の定義（二値分類用）
# ターゲット条件：3ヶ月で市場平均を「15%以上」アウトパフォームした銘柄を1（正例）とする
# ※この閾値は、狙いたい利幅や銘柄の母数に応じて10%〜20%の間で調整可能です。
THRESHOLD = 0.15
df_target["target"] = (df_target["alpha_return_60"] > THRESHOLD).astype(int)

# 5. 未来データ（末尾60日分）の欠損除去
# shift(-60)したことで、直近60日間は「未来の結果がまだ分からない」ため、学習から除外します。
df_target = df_target.dropna(subset=["future_return_60"])

print("===== 中長期ターゲット分布 =====")
counts = df_target["target"].value_counts()
print(counts)
print(f"大化け（市場＋15%以上）の割合: {counts[1] / len(df_target) * 100:.2f}%")

# 6. 新しいトレーニングデータテーブルとして保存
# SQLite保存用に日付を文字列に戻す
df_target["date"] = df_target["date"].dt.strftime("%Y-%m-%d")
if "announcement_date" in df_target.columns and pd.api.types.is_datetime64_any_dtype(df_target["announcement_date"]):
    df_target["announcement_date"] = df_target["announcement_date"].dt.strftime("%Y-%m-%d")

df_target.to_sql("training_data_v4", conn, if_exists="replace", index=False)
conn.close()

print("\ntraining_data_v4（中長期投資向け）の作成が完了しました。")