import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import settings


def get_folds(df, n_splits, random_state=42, verbose=True):

    """
    Create a column of fold numbers

    Parameters
    ----------
    df (pandas.DataFrame of shape (634112, 1)): DataFrame with target column
    n_splits (int): Number of folds (2 <= n_splits)
    random_state (int): Seed for reproducible results
    verbose (bool): Flag for verbosity

    Returns
    -------
    df (pandas.DataFrame of shape (634112, 4)): DataFrame with target and fold columns
    """

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for fold, (_, val_idx) in enumerate(skf.split(X=df, y=df['ARTIS_DURUMU']), 1):
        df.loc[val_idx, 'fold'] = fold
    df['fold'] = df['fold'].astype(np.uint8)

    if verbose:
        print(f'\nTraining set split into {n_splits} folds')
        for fold in range(1, n_splits + 1):
            df_fold = df[df['fold'] == fold]
            label_value_counts = df_fold['ARTIS_DURUMU'].value_counts().to_dict()
            print(f'Fold {fold} {df_fold.shape} - {label_value_counts}')

    return df


if __name__ == '__main__':

    df_train = pd.read_csv(settings.DATA / 'train.csv')
    print(f'Training Set Shape: {df_train.shape} - Memory Usage: {df_train.memory_usage().sum() / 1024 ** 2:.2f} MB')

    get_folds(
        df=df_train,
        n_splits=5,
        random_state=42,
        verbose=True
    )

    df_train[['POLICY_ID', 'fold']].to_csv(settings.DATA / 'folds.csv', index=False)
