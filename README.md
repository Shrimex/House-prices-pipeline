## Описание

Задача регрессии — предсказание стоимости жилья на основе 79 признаков. Проект построен на универсальном ML-пайплайне с акцентом на:
- отсутствие data leakage
- переиспользуемость кода
- сравнение множества моделей через cross-validation
- автоматический поиск гиперпараметров

---

## Структура проекта

```
House_Prices/
├── Data/
│   ├── train.csv
│   └── test.csv
├── src/
│   ├── config.py            # Конфигурация проекта (OmegaConf)
│   ├── data.py              # Загрузка данных
│   ├── EDA.ipynb            # Разведочный анализ данных
│   ├── evaluation.py        # Оценка моделей через CV (RMSE, R²)
│   ├── feature_creation.py  # Feature engineering
│   ├── model_list.py        # Список моделей
│   ├── neural_network.py    # MLP регрессор на PyTorch (sklearn-совместимый)
│   ├── param_grids.py       # Сетки гиперпараметров для GridSearchCV
│   ├── preprocessing.py     # Imputation, encoding, feature engineering pipeline
│   ├── training.py          # Основной пайплайн обучения
│   └── utils.py             # Вспомогательные функции
├── main.py                  # Точка входа
└── requirements.txt
```

---

## Пайплайн

```
Загрузка данных → Feature Engineering → Imputation → Encoding
→ CV оценка всех моделей → GridSearchCV → Финальное обучение → Сабмит
```

### Ключевые решения

**Feature Engineering:**
- `TotalSF` — суммарная площадь (подвал + 1й + 2й этаж)
- `TotalBathrooms` — взвешенное число ванных комнат
- `HouseAge`, `RemodAge` — возраст дома и ремонта
- `IsRemodeled` — флаг наличия ремонта
- `OverallScore` — произведение качества и состояния
- `HasPool`, `HasGarage`, `HasBasement`, `HasFireplace` — бинарные флаги
**Imputation (без leakage):**
- `'None'` для категориальных признаков где пропуск = отсутствие объекта (нет гаража, нет подвала)
- `0` для числовых аналогов
- Медиана/мода только на train для реальных пропусков
**Encoding:**
- `OrdinalEncoder` с осмысленным порядком для качественных признаков (`Po → Fa → TA → Gd → Ex`)
- `OrdinalEncoder` по алфавиту для остальных категориальных
**Target:** логарифмируется через `np.log1p` перед обучением, `np.expm1` при сабмите. 

---

## Модели

| Модель | Тип |
|--------|-----|
| DummyRegressor | Baseline |
| Ridge, Lasso, ElasticNet | Линейные |
| DecisionTree | Дерево |
| RandomForest | Ансамбль |
| LightGBM, CatBoost, XGBoost | Градиентный бустинг |
| TitanicNN (MLP) | Нейронная сеть |

Нейронная сеть реализована на PyTorch с sklearn-совместимым интерфейсом, поддерживает настройку оптимизатора (`adam`, `sgd`, `rmsprop`), функции потерь (`mse`, `mae`, `huber`) и scheduler.

---

## Запуск

### Установка зависимостей

```bash
pip install -r requirements.txt
```

### Запуск пайплайна

```bash
python main.py
```

Результаты сохраняются в папку `outputs/`:
- `outputs/metrics/cv_results.csv` — результаты CV по всем моделям
- `outputs/metrics/summary.json` — лучшая модель и параметры
- `outputs/submissions/` — файл для сабмита на Kaggle
- `outputs/models/` — сохранённая модель

---

