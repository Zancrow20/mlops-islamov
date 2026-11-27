"""
Скрипт evaluate.py:
1. загружает models/model.joblib;
2. считает метрики на data/processed/test.csv;
3. пишет reports/eval_metrics.json;
4. завершается с кодом 1, если ROC-AUC ниже порога.
"""

from __future__ import annotations

import json
import sys

import joblib
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from src.config import TARGET, feature_columns, load_params, resolve
from src.logging_setup import setup_logging

log = setup_logging()


def main():
    params = load_params()
    data = params["data"]
    evaluate = params["evaluate"]

    # 1. Загрузить models/model.joblib.
    model_path = resolve("models/model.joblib")
    pipe = joblib.load(model_path)

    # 2. Посчитать метрики на data/processed/test.csv.
    test_path = resolve(data["processed_dir"]) / "test.csv"
    test_df = pd.read_csv(test_path)

    cols = feature_columns(params)
    X_test, y_test = test_df[cols], test_df[TARGET]

    threshold = evaluate["threshold"]
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)

    # 3. Записать reports/eval_metrics.json.
    eval_metrics = {
        "model": params["train"]["model"],
        "threshold": threshold,
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "pr_auc": float(average_precision_score(y_test, proba)),
        "f1": float(f1_score(y_test, pred)),
    }

    eval_path = resolve("reports/eval_metrics.json")
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    with open(eval_path, mode="w", encoding="utf-8") as file:
        json.dump(eval_metrics, file, indent=2)
        file.write("\n")

    log.info("Eval metrics written to %s", eval_path)

    # 4. Завершить с кодом 1, если ROC-AUC ниже порога.
    gate = evaluate["min_roc_auc"]
    if eval_metrics["roc_auc"] < gate:
        log.error("ROC-AUC %.4f ниже порога %.4f", eval_metrics["roc_auc"], gate)
        sys.exit(1)

    log.info("Quality gate пройден: ROC-AUC %.4f >= %.4f", eval_metrics["roc_auc"], gate)


if __name__ == "__main__":
    main()
