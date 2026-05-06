import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score



def evaluate_models(
    models: dict,
    X: pd.DataFrame,
    y: pd.Series,
    n_folds: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:

    results = []
    #вычисление метрик для каждой модели и сохранение в list для дальнейшего добавления DataFrame для сохранения и отображения
    for model_name, model in models.items():
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        RMSE = cross_val_score(model, X, y, scoring='neg_root_mean_squared_error', cv=cv)
        r_square = cross_val_score(model, X, y, scoring='r2', cv=cv)

        results.append({
            "model_name": model_name,
            "RMSE_mean": abs(RMSE.mean()),
            "RMSE_std": RMSE.std(),
            "r_square_mean": r_square.mean(),
            "r_square_std": r_square.std(),
        })

    result_df = (
        pd.DataFrame(results)
        .sort_values("RMSE_mean", ascending=True)
        .reset_index(drop=True)
    )
    return result_df