"""Стадия train.

Обязательно:
  * никаких абсолютных путей — только src.config.resolve();
  * никаких магических чисел — только params.yaml;
  * зафиксированный seed;
  * модель сохраняется в models/model.joblib вместе с препроцессором;
  * метрики пишутся в reports/train_metrics.json.

Запуск: python -m src.train
"""

from __future__ import annotations

import json
import platform
import subprocess
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline

from src.config import TARGET, feature_columns, load_params, resolve
from src.features import build_preprocessor
from src.logging_setup import setup_logging

log = setup_logging()


def git_sha() -> str:
    """Короткий хеш текущего коммита. В копии без .git обучение падать не должно"""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def build_model(params: dict[str, Any]):
    """Возвращает модель по имени из params['train']['model']."""
    seed = params["seed"]
    train_cfg = params["train"]
    model_name = train_cfg["model"]
    model_params = dict(train_cfg.get(model_name, {}) or {})
    if model_name == "logreg":
        return LogisticRegression(random_state=seed, **model_params)
    if model_name == "random_forest":
        return RandomForestClassifier(random_state=seed, **model_params)
    if model_name == "gradient_boosting":
        return GradientBoostingClassifier(random_state=seed, **model_params)
    raise ValueError(f"Неизвестная модель: {model_name}")


def main() -> None:
    params = load_params()
    data = params["data"]
    evaluate = params["evaluate"]
    train = params["train"]
    np.random.seed(params["seed"])

    # 1. Загрузить данные train.csv и val.csv из data/processed/
    data_dir = resolve(data["processed_dir"])
    train_path = data_dir / "train.csv"
    val_path = data_dir / "val.csv"

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    # 2. Подготовить X, y для обучения и валидации
    cols = feature_columns(params)
    X_train, y_train = train_df[cols], train_df[TARGET]
    X_val, y_val = val_df[cols], val_df[TARGET]

    # 3. Создать Pipeline(препроцессор + RandomForestClassifier)
    pipe = Pipeline(
        steps=[
            ("preprocess", build_preprocessor(params)),
            ("model", build_model(params)),
        ]
    )

    # 4. Обучить модель на train
    pipe.fit(X_train, y_train)

    # 5. Предсказать на val
    proba = pipe.predict_proba(X_val)[:, 1]
    pred = (proba >= evaluate["threshold"]).astype(int)

    # 6. Посчитать метрики (roc_auc, f1, pr_auc)
    metrics = {
        "model": train["model"],
        "roc_auc": float(roc_auc_score(y_val, proba)),
        "pr_auc": float(average_precision_score(y_val, proba)),
        "f1": float(f1_score(y_val, pred)),
    }

    # 7. Сохранить модель в models/model.joblib
    model_path = resolve("models/model.joblib")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, model_path)

    # 8. Сохранить метрики в reports/train_metrics.json
    metrics_path = resolve("reports/train_metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, mode="w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
        file.write("\n")

    # 9. Сохранить связь модели и метрик в виде метаданных
    metadata = {
        "model_name": train["model"],
        "commit_sha": git_sha(),
        "python_version": platform.python_version(),
        "features": cols,
        "val_metrics": metrics,
    }

    metadata_path = resolve("models/model_meta.json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, mode="w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
        file.write("\n")

    # 10. Логгирование
    log.info("Модель сохранена: %s", model_path)
    log.info("Метрики на val: %s", metrics)


if __name__ == "__main__":
    main()
