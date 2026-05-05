from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from src.neural_network import HousePricesNN


def get_models(random_state: int = 42) -> dict:

    models = {
        "dummy_regressor": DummyRegressor(strategy='median'),

        'Ridge (L2)': Ridge(alpha=1.0),
        'Lasso (L1)': Lasso(alpha=1.0, max_iter=5000),
        'ElasticNet': ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=5000),

        'KNN':           KNeighborsRegressor(),
        'decision_tree': DecisionTreeRegressor(random_state=random_state),
        'random_forest': RandomForestRegressor(n_estimators=100, max_depth=3, random_state=random_state),

        "lightGBM": LGBMRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=random_state, verbose=-1),
        "catboost": CatBoostRegressor(iterations=200, learning_rate=0.1, depth=6, verbose=0, random_state=random_state),
        "XGBoost":  XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=random_state, verbosity = 0),

        "NeuralNet_basic": HousePricesNN(
            hidden_dims=[64, 32],
            dropout=0.3,
            lr=1e-3,
            epochs=200,
            batch_size=32,
            patience=15,
            random_state=random_state,
        ),
        "NeuralNet_sgd": HousePricesNN(
            hidden_dims=[64, 32],
            dropout=0.3,
            lr=1e-3,
            epochs=200,
            batch_size=32,
            patience=15,
            random_state=random_state,
            optimizer='sgd',
        ),
        "NeuralNet_rmsprop": HousePricesNN(
            hidden_dims=[64, 32],
            dropout=0.3,
            lr=1e-3,
            epochs=200,
            batch_size=32,
            patience=15,
            random_state=random_state,
            optimizer='rmsprop',
        ),
        "NeuralNet_mae": HousePricesNN(
            hidden_dims=[64, 32],
            dropout=0.3,
            lr=1e-3,
            epochs=200,
            batch_size=32,
            patience=15,
            random_state=random_state,
            criterion='mae',
        ),
        "NeuralNet_huber": HousePricesNN(
            hidden_dims=[64, 32],
            dropout=0.3,
            lr=1e-3,
            epochs=200,
            batch_size=32,
            patience=15,
            random_state=random_state,
            criterion='huber',
        ),
        "NeuralNet_cosine": HousePricesNN(
            hidden_dims=[64, 32],
            dropout=0.3,
            lr=1e-3,
            epochs=200,
            batch_size=32,
            patience=15,
            random_state=random_state,
            scheduler='cosine',
            scheduler_kwargs={"T_max": 50},
        ),
        "NeuralNet_rop": HousePricesNN(
            hidden_dims=[64, 32],
            dropout=0.3,
            lr=1e-3,
            epochs=200,
            batch_size=32,
            patience=15,
            random_state=random_state,
            scheduler='reduce_on_plateau',
            scheduler_kwargs={
                "mode": "min",
                "factor": 0.5,
                "patience": 5
            },
        ),
        "NeuralNet_combination": HousePricesNN(
            hidden_dims=[64, 32],
            dropout=0.3,
            lr=1e-3,
            epochs=200,
            batch_size=32,
            patience=15,
            random_state=random_state,
            optimizer='adam',
            criterion='huber',
            scheduler='reduce_on_plateau',
            scheduler_kwargs={
                "mode": "min",
                "factor": 0.5,
                "patience": 5
            },
        ),
    }
    return models


def get_model_by_name(model_name: str, random_state: int = 42):
    models = get_models(random_state=random_state)

    if model_name not in models:
        available = ", ".join(models.keys())
        raise ValueError(f"Неизвестная модель '{model_name}'. Доступные: {available}")

    return models[model_name]