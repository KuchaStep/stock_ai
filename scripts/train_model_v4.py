import sqlite3

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from lightgbm import log_evaluation, early_stopping

from feature_utils import FEATURES_V4


DB_PATH = "database/stock.db"
MODEL_PATH = "data/models/stock_model_v4.pkl"


def main() -> None:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM training_data_v4", conn)
    conn.close()

    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["target"]).sort_values("date").reset_index(drop=True)

    unique_dates = sorted(df["date"].unique())
    split_index = int(len(unique_dates) * 0.8)
    split_date = unique_dates[split_index]

    # --- リーク防止用のパージング ---
    # ターゲットが60日先を見ているため、学習データとテストデータの間に60日間の隙間を作る
    # 営業日ベースの厳密な計算は複雑なため、日付インデックスで約60営業日分を削る
    purge_index = max(0, split_index - 60)
    purge_date = unique_dates[purge_index]

    train_full_df = df[df["date"] < purge_date].copy()
    test_df = df[df["date"] >= split_date].copy()

    # 学習データの中からさらに検証用を5%ほど切り出す (Early Stopping用)
    val_split_idx = int(len(train_full_df) * 0.95)
    train_df = train_full_df.iloc[:val_split_idx].copy()
    val_df = train_full_df.iloc[val_split_idx:].copy()

    print(
        f"Train: {train_df['date'].min()} to {train_df['date'].max()} "
        f"({len(train_df):,} rows)"
    )
    print(
        f"Test : {test_df['date'].min()} to {test_df['date'].max()} "
        f"({len(test_df):,} rows)"
    )
    print("\nTarget ratio")
    print(df["target"].value_counts(normalize=True).sort_index())

    model = LGBMClassifier(
        n_estimators=1000, # early_stoppingを使うので多めに設定
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=50,
        class_weight="balanced",
        random_state=42,
        verbose=-1,
    )

    model.fit(
        train_df[FEATURES_V4],
        train_df["target"],
        eval_set=[(val_df[FEATURES_V4], val_df["target"])],
        eval_metric="binary_logloss",
        callbacks=[
            early_stopping(stopping_rounds=50),
            log_evaluation(period=50)
        ]
    )

    pred = model.predict(test_df[FEATURES_V4])
    proba = model.predict_proba(test_df[FEATURES_V4])[:, 1]

    print("\n===== OOS classification report =====")
    print(classification_report(test_df["target"], pred))
    print(f"OOS ROC-AUC: {roc_auc_score(test_df['target'], proba):.4f}")
    print(f"OOS PR-AUC: {average_precision_score(test_df['target'], proba):.4f}")

    importance = pd.DataFrame(
        {
            "feature": FEATURES_V4,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    print("\n===== Feature importance =====")
    print(importance.to_string(index=False))

    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved: {MODEL_PATH}")


if __name__ == "__main__":
    main()
