# Real Estate Pricing Service

##  Цель проекта
Оценка рыночной стоимости объектов для агентства недвижимости по характеристикам (площадь, год, состояние и т.д.).

---

##  Решение
- Импутация пропусков: **median** (числовые), **most_frequent** (категориальные)
- Кодирование: **OneHotEncoder**, масштабирование: **StandardScaler**
- Модель: **Ridge Regression**
- Интерфейсы: **API (FastAPI)** и **CLI (predict.py)**
- Параметры  в config.yaml

---

##  Метрики качества
| Метрика | Значение |
|--------:|:---------|
| MAE     | 19 008 |
| RMSE    | 29 845 |
| R      | 0.884 |

---

##  Использование
~~~bash
python src/train.py --input data/train.csv --output model/model.pkl
python predict.py --input sample_properties.csv --config config.yaml --output predictions.csv
~~~

##  Конфигурация (config.yaml)
~~~yaml
title: "Real Estate Pricing"
model_path: "model/model.pkl"
target_col: "SalePrice"
drop_cols: []
~~~

##  Структура
real_estate_pricing/
 data/
 model/
 src/ (train.py, infer.py, app.py, predict.py)
 config.yaml
 report/
 README.md

##  Результат для заказчика
- Быстрая оценка объектов
- Понятные метрики/отчёт
- Интеграция через API/CLI/Docker