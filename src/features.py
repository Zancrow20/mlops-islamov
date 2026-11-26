"""Построение препроцессора признаков.

TODO (занятие 1): собрать здесь ColumnTransformer.

Требования:
  * числовые признаки: заполнение пропусков + масштабирование;
  * категориальные: заполнение пропусков + OneHotEncoder;
  * бинарные: без изменений;
  * списки колонок берутся из params.yaml, а не пишутся в коде.

Подсказка: почему препроцессор обязан ехать в одном Pipeline с моделью,
разбирается на паре. Если сделать иначе — сервис на занятии 10 сломается.
"""
from __future__ import annotations

from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(params: dict[str, Any]):
    """Собирает ColumnTransformer для разных типов признаков.

    1. Создать Pipeline для числовых: заполнение пропусков + масштабирование
    2. Создать Pipeline для категориальных: заполнение пропусков + OneHotEncoder
    3. Создать ColumnTransformer который объединяет все три типа
    4. Вернуть препроцессор

    Подсказка: списки колонок берутся из params["features"]
    """
    f = params["features"]

    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, f["numeric"]),
            ("cat", cat_pipe, f["categorical"]),
            ("bin", "passthrough", f["binary"]),
        ]
    )
