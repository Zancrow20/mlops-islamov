"""Стадия prepare: сырой CSV -> train/val/test.

TODO (занятие 1):
  1. прочитать data/raw/churn.csv;
  2. обработать пропуски в total_charges осмысленно (не dropna!);
  3. разбить на train/val/test со stratify по churn и random_state из params;
  4. сохранить три CSV в data/processed/.

Проверка: два запуска подряд должны дать одинаковые файлы.
"""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import load_params, resolve, TARGET
from src.logging_setup import setup_logging

log = setup_logging()

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Заполняет пропуски в total_charges осмысленной оценкой."""
    df = df.copy()
    df["total_charges"] = df["total_charges"].fillna(
        df["monthly_charges"] * df["tenure_months"]
    )
    return df

def main() -> None:
    params = load_params()
    d = params["data"]
    seed = params["seed"]

    # 1. Прочитать raw данные с помощью pd.read_csv()
    raw = pd.read_csv(resolve(d["raw_path"]))

    # 2. Обработать пропуски в total_charges (заполнить осмысленно, не dropna!)
    df = clean(raw)

    # 3. Разбить на train/val/test со stratify по churn
    train_val, test = train_test_split(
        df,
        test_size=d["test_size"],
        random_state=seed,
        stratify=df[TARGET],
    )

    val_ratio = d["val_size"] / (1.0 - d["test_size"])

    train, val = train_test_split(
        train_val,
        test_size=val_ratio,
        random_state=seed,
        stratify=train_val[TARGET],
    )

    # 4. Сохранить три CSV в data/processed/
    out_dir = resolve(d["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, part in [("train", train), ("val", val), ("test", test)]:
        path = out_dir / f"{name}.csv"
        part.to_csv(path, index=False)
        log.info(
            "%s: %d строк, churn rate=%.4f -> %s",
            name,
            len(part),
            part[TARGET].mean(),
            path,
        )

if __name__ == "__main__":
    main()
