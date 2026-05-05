import numpy as np
import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df['TotalBathrooms'] = df['FullBath'] + df['HalfBath'] * 0.5 + df['BsmtFullBath'] + df['BsmtHalfBath'] * 0.5

    df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']
    df['IsRemodeled']=np.where(df['YearBuilt'] != df['YearRemodAdd'], 1, 0)

    df['OverallScore']=df['OverallQual'] * df['OverallCond']

    df['HasGarage'] = (df['GarageArea'].fillna(0) > 0).astype(int)
    df['HasBasement'] = (df['TotalBsmtSF'].fillna(0) > 0).astype(int)
    df['HasPool'] = (df['PoolArea'].fillna(0) > 0).astype(int)
    df['HasFireplace'] = (df['Fireplaces'].fillna(0) > 0).astype(int)

    return df