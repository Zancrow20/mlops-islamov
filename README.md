# Churn MLOps

Сквозной учебный MLOps-проект: воспроизводимый ML-пайплайн и сервис предсказания оттока клиентов телеком-оператора (`churn`).

## Задача

- **Тип задачи:** бинарная классификация (предсказание вероятности ухода клиента).
- **Данные:** 20 000 записей (доля оттока ~20.6%), генерируются локально модулем `src/data/generate.py` с фиксированным `seed` из `params.yaml` для офлайн-воспроизводимости и управляемого дрейфа.
- **Признаки:** 5 числовых (`tenure_months`, `monthly_charges`, `total_charges`, `num_support_calls`, `avg_monthly_gb`), 3 категориальных (`contract_type`, `internet_service`, `payment_method`), 2 бинарных (`has_tech_support`, `is_senior`). Идентификатор `customer_id` исключён.
- **Метрики на валидации (`data/processed/val.csv`):** `ROC-AUC`, `PR-AUC`, `F1` (при пороге `0.5`).

## Требования и быстрый старт

Рекомендуемая версия Python: **3.11** (из-за зафиксированных версий `evidently==0.4.40`, `mlflow<3.0`, `prefect<3.0` в `requirements.txt`).

```bash
# 1. Клонировать репозиторий
git clone https://github.com/Zancrow20/mlops-islamov.git
cd mlops-islamov

# 2. Создать и активировать виртуальное окружение (Python 3.11)
python3.11 -m venv .venv
# или через uv: uv venv --python 3.11 --seed .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1

# 3. Установить зависимости и проверить окружение
make install
make check                         # должно вывести environment: OK

# 4. Запустить полный пайплайн
make data                          # сгенерирует data/raw/churn.csv (20 000 строк)
make prepare                       # почистит пропуски и разобьёт на train/val/test (70/10/20)
make train                         # обучит Pipeline, сохранит models/model.joblib и reports/train_metrics.json
```

### Полезные команды

```bash
make test                          # запустить тесты (pytest)
make lint                          # проверить код линтером (ruff)
pre-commit install                 # установить локальные git-хуки
pre-commit run --all-files         # прогнать все pre-commit проверки вручную
make help                          # список всех команд Makefile
```

Для смены модели достаточно изменить `train.model` в `params.yaml` (`logreg`, `random_forest` или `gradient_boosting`) и повторно запустить `make train` — править код не требуется.

## Структура проекта

```text
.
├── Makefile                       # единые команды запуска стадий и проверок
├── params.yaml                    # конфигурация данных, признаков и гиперпараметров моделей
├── pyproject.toml                 # настройки ruff и pytest
├── requirements.txt               # основные зависимости
├── requirements-dev.txt           # dev-зависимости (pytest, ruff, pre-commit)
├── .env.example                   # шаблон переменных окружения (секреты хранятся в .env)
├── .pre-commit-config.yaml        # хуки pre-commit (ruff, check-yaml, лимит 1 МБ, секреты)
├── notebooks/
│   ├── baseline_notebook.py       # исходный «ноутбучный» скрипт
│   └── PROBLEMS.md                # разбор проблем воспроизводимости исходника
├── reports/
│   ├── EXPERIMENTS.md             # сравнение моделей и анализ ROC-AUC vs F1
│   └── train_metrics.json         # метрики последнего обучения на валидации
├── src/
│   ├── config.py                  # загрузка params.yaml и разрешение путей
│   ├── logging_setup.py           # единое логирование
│   ├── smoke_check.py             # проверка окружения (make check)
│   ├── features.py                # ColumnTransformer для numeric / categorical / binary
│   ├── train.py                   # обучение единого Pipeline (препроцессор + модель)
│   └── data/
│       ├── generate.py            # генерация синтетического датасета
│       └── prepare.py             # очистка пропусков и стратифицированный сплит
└── tests/
    └── test_smoke.py              # базовые smoke-тесты
```

## Что уже сделано

### Занятие 1. Окружение и структура проекта
- Разобраны проблемы исходного ноутбука ([`notebooks/PROBLEMS.md`](notebooks/PROBLEMS.md)).
- Признаки и гиперпараметры трёх моделей вынесены в [`params.yaml`](params.yaml).
- Реализована стадия подготовки данных ([`src/data/prepare.py`](src/data/prepare.py)): заполнение пропусков `total_charges` как `monthly_charges * tenure_months` (вместо `dropna`), двухэтапный стратифицированный сплит `70/10/20` с фиксированным `seed`.
- Реализован препроцессор ([`src/features.py`](src/features.py)) на `ColumnTransformer` (`SimpleImputer` + `StandardScaler` для числовых, `SimpleImputer` + `OneHotEncoder(handle_unknown="ignore")` для категориальных, `passthrough` для бинарных).
- Реализовано обучение ([`src/train.py`](src/train.py)) в едином `Pipeline` с сохранением `models/model.joblib` и `reports/train_metrics.json`; проверена полная воспроизводимость (`diff` между прогонами пустой).
- Проведено сравнение 3 моделей ([`reports/EXPERIMENTS.md`](reports/EXPERIMENTS.md)) — лучший результат показала `logreg` (`ROC-AUC = 0.8086`, `PR-AUC = 0.5193`, `F1 = 0.3875`).

### Занятие 2. Git для ML-команды
- Проверена чистота репозитория и настроена работа с секретами через [`.env.example`](.env.example) (`.env` в `.gitignore`).
- Настроен [`.pre-commit-config.yaml`](.pre-commit-config.yaml): `ruff`, `ruff-format`, `check-yaml`, `detect-private-key`, `check-merge-conflict`, `check-added-large-files` (`--maxkb=1000`).
- Настроен процесс разработки через feature-ветки, Pull Requests с ревью и защитой ветки `main`.

## Правила проекта

1. **Никаких абсолютных путей** — только `src.config.resolve()`.
2. **Никаких магических чисел в коде** — все параметры в `params.yaml`.
3. **Данные и модели не коммитятся в Git** (в `.gitignore`, далее — через DVC).
4. **Секреты — только через переменные окружения** (`.env` в `.gitignore`, шаблон в `.env.example`).
5. **Изменения в `main` — только через ветки и Pull Request.**
