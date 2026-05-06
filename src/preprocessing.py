import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

from src.feature_creation import add_features

QUALITY_COLS = [
    'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',
    'HeatingQC', 'KitchenQual', 'FireplaceQu',
    'GarageQual', 'GarageCond', 'PoolQC',
]
QUALITY_ORDER = ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex']

OTHER_CAT_COLS = [
    'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour',
    'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood',
    'Condition1', 'Condition2', 'BldgType', 'HouseStyle',
    'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd',
    'MasVnrType', 'Foundation', 'BsmtExposure',
    'BsmtFinType1', 'BsmtFinType2', 'Heating', 'CentralAir',
    'Electrical', 'Functional', 'GarageType', 'GarageFinish',
    'PavedDrive', 'Fence', 'MiscFeature', 'SaleType',
    'SaleCondition',
]

def drop_id(X_train: pd.DataFrame, X_test: pd.DataFrame, id_col: str):
    X_train=X_train.copy()
    X_test=X_test.copy()

    if id_col in X_train.columns:
        X_train=X_train.drop(id_col, axis=1)
    if id_col in X_test.columns:
        X_test=X_test.drop(id_col, axis=1)
    
    return X_train, X_test

class GroupImputer(BaseEstimator, TransformerMixin):
    '''Заполняет пропуски статистикой по группе.

    Вся статистика вычисляется только на train в fit(),
    что предотвращает утечку данных при кросс-валидации.
    Если группа не встречалась при обучении — используется глобальный fallback.

    Параметры
    ----------
    group_col : str
        Колонка по которой группируем (например 'Neighborhood').
    target_col : str
        Колонка в которой заполняем пропуски (например 'LotFrontage').
    strategy : str
        Стратегия агрегации: 'median' или 'mean'.'''
    def __init__(self, group_col: str, target_col: str, strategy: str = "median"):
        self.group_col = group_col
        self.target_col = target_col
        self.strategy = strategy
        self.global_value = None
        self.group_values = None

    def fit(self, X: pd.DataFrame, y=None):
        if self.strategy == "median":
            self.global_value = X[self.target_col].median()
            agg_func = "median"
        elif self.strategy == "mean":
            self.global_value = X[self.target_col].mean()
            agg_func = "mean"
        else:
            raise ValueError(f"Unsupported strategy: {self.strategy}")

        self.group_values = (
            X.groupby(self.group_col)[self.target_col]
             .agg(agg_func)
             .to_dict()
        )
        return self

    def transform(self, X: pd.DataFrame, y=None):
        X = X.copy()
        mask = X[self.target_col].isna()
        X.loc[mask, self.target_col] = (
            X.loc[mask, self.group_col]
             .map(self.group_values)
             .fillna(self.global_value)
        )
        return X

def encode_features(df: pd.DataFrame, quality_encoder=None, other_encoder=None):
    '''Применяет OrdinalEncoder к категориальным признакам.

    Качественные признаки (QUALITY_COLS) кодируются с осмысленным порядком:
    None → Po → Fa → TA → Gd → Ex (0–5).
    Остальные категориальные (OTHER_CAT_COLS) кодируются по алфавиту.

    Энкодеры передаются снаружи — обучены только на train,
    что предотвращает утечку данных.'''
    df = df.copy()

    df[QUALITY_COLS] = quality_encoder.transform(df[QUALITY_COLS])
    df[OTHER_CAT_COLS] = other_encoder.transform(df[OTHER_CAT_COLS])

    return df

def drop_columns(X_train: pd.DataFrame, X_test: pd.DataFrame):
    X_train = X_train.copy()
    X_test  = X_test.copy()
    
    cols_to_drop = [
        'YearBuilt', 'YearRemodAdd', 'YrSold', 'MoSold',         
        'WoodDeckSF', 'EnclosedPorch', '3SsnPorch',
        'ScreenPorch', 'LowQualFinSF', 'MiscVal', 'PoolArea'                         
    ]
    X_train = X_train.drop(columns=[c for c in cols_to_drop if c in X_train.columns])
    X_test  = X_test.drop(columns=[c for c in cols_to_drop if c in X_test.columns])
    
    return X_train, X_test


def preprocess_features(
    
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    id_col: str,):
    '''Полный preprocessing pipeline без утечки данных.

    Порядок шагов:
    1. Удаление ID колонки.
    2. Feature engineering на объединённом train+test (stateless операции).
    3. Заполнение пропусков — три стратегии:
       - 'None' для категориальных где пропуск означает отсутствие объекта
       - 0 для числовых аналогов
       - медиана/мода для реальных пропусков (fit только на train)
    4. OrdinalEncoding категориальных признаков (fit только на train).
    5. Удаление исходных колонок заменённых новыми признаками.
    6. Выравнивание колонок train и test.'''
    X_train, X_test = drop_id(X_train, X_test, id_col)
    train_rows = X_train.shape[0]

    full_df = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)
    full_df = add_features(full_df)
    X_train = full_df.iloc[:train_rows].copy()
    X_test  = full_df.iloc[train_rows:].copy()

    none_cols = [
        'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
        'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
        'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
        'MasVnrType',
    ]
    for col in none_cols:
        X_train[col] = X_train[col].fillna('None')
        X_test[col]  = X_test[col].fillna('None')

    zero_cols = [
        'GarageYrBlt', 'GarageArea', 'GarageCars',
        'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
        'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea',
    ]
    for col in zero_cols:
        X_train[col] = X_train[col].fillna(0)
        X_test[col]  = X_test[col].fillna(0)
    median_imputer = SimpleImputer(strategy='median')
    X_train[['LotFrontage']] = median_imputer.fit_transform(X_train[['LotFrontage']])
    X_test[['LotFrontage']]  = median_imputer.transform(X_test[['LotFrontage']])

    mode_cols = ['MSZoning', 'Utilities', 'Functional', 'Electrical',
                 'Exterior1st', 'Exterior2nd', 'KitchenQual', 'SaleType']
    mode_imputer = SimpleImputer(strategy='most_frequent')
    X_train[mode_cols] = mode_imputer.fit_transform(X_train[mode_cols])
    X_test[mode_cols]  = mode_imputer.transform(X_test[mode_cols])

    quality_encoder = OrdinalEncoder(
        categories=[QUALITY_ORDER] * len(QUALITY_COLS),
        handle_unknown='use_encoded_value',
        unknown_value=-1,
    )
    other_encoder = OrdinalEncoder(
        handle_unknown='use_encoded_value',
        unknown_value=-1,
    )
    quality_encoder.fit(X_train[QUALITY_COLS])
    other_encoder.fit(X_train[OTHER_CAT_COLS])

    X_train = encode_features(X_train, quality_encoder, other_encoder)
    X_test  = encode_features(X_test,  quality_encoder, other_encoder)

    X_train, X_test = drop_columns(X_train, X_test)

    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

    return X_train, X_test