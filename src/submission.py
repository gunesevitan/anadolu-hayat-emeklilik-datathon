from glob import glob
import pandas as pd
from scipy.stats import mode

import settings
import metrics


def match_and_overwrite_target(df_train, df_test, agg, column):

    # Find matching values in training and test set
    matching_rows = df_train.loc[df_train[column].isin(df_test[column]), column].unique()
    # Aggregate target of matching values and save it as a new column
    if agg == 'mean':
        matching_rows_target_agg = df_train[df_train[column].isin(matching_rows)].groupby(column)['ARTIS_DURUMU'].agg('mean')
    elif agg == 'mode':
        matching_rows_target_agg = df_train[df_train[column].isin(matching_rows)].groupby(column)['ARTIS_DURUMU'].agg(lambda x: mode(x)[0][0])

    # Overwrite final predictions
    df_train[f'matching_{column}_target_{agg}'] = df_train[column].map(matching_rows_target_agg)
    df_train.loc[~df_train[f'matching_{column}_target_{agg}'].isnull(), 'final_predictions'] = df_train.loc[~df_train[f'matching_{column}_target_{agg}'].isnull(), f'matching_{column}_target_{agg}']
    df_test[f'matching_{column}_target_{agg}'] = df_test[column].map(matching_rows_target_agg)
    df_test.loc[~df_test[f'matching_{column}_target_{agg}'].isnull(), 'final_predictions'] = df_test.loc[~df_test[f'matching_{column}_target_{agg}'].isnull(), f'matching_{column}_target_{agg}']

    matched_scores = metrics.classification_scores(df_train['ARTIS_DURUMU'], df_train['final_predictions'], threshold=0.5)
    print(f'{len(matching_rows)} matching {column} values found\n{matched_scores}')


if __name__ == '__main__':

    lgb_train_predictions_files = glob(str(settings.MODELS / 'lightgbm' / 'train_predictions_seed*.csv'))
    lgb_test_predictions_files = glob(str(settings.MODELS / 'lightgbm' / 'test_predictions_seed*.csv'))
    xgb_train_predictions_files = glob(str(settings.MODELS / 'xgboost' / 'train_predictions_seed*.csv'))
    xgb_test_predictions_files = glob(str(settings.MODELS / 'xgboost' / 'test_predictions_seed*.csv'))

    df_train = pd.read_csv(settings.DATA / 'train.csv')
    df_test = pd.read_csv(settings.DATA / 'test.csv')
    print(f'Training Set Shape: {df_train.shape} - Memory Usage: {df_train.memory_usage().sum() / 1024 ** 2:.2f} MB')
    print(f'Test Set Shape: {df_test.shape} - Memory Usage: {df_test.memory_usage().sum() / 1024 ** 2:.2f} MB\n')

    df_train['lgb_predictions'] = 0
    df_test['lgb_predictions'] = 0

    for lgb_train_predictions_file in lgb_train_predictions_files:
        df_lgb_train_predictions = pd.read_csv(lgb_train_predictions_file)
        scores = metrics.classification_scores(df_train['ARTIS_DURUMU'], df_lgb_train_predictions['lgb_predictions'], threshold=0.5)
        print(f'Reading: {lgb_train_predictions_file}\n{scores}')
        df_train['lgb_predictions'] += (df_lgb_train_predictions['lgb_predictions'] / len(lgb_train_predictions_files))

    for lgb_test_predictions_file in lgb_test_predictions_files:
        df_lgb_test_predictions = pd.read_csv(lgb_test_predictions_file)
        df_test['lgb_predictions'] += (df_lgb_test_predictions['lgb_predictions'] / len(lgb_test_predictions_files))

    df_train['xgb_predictions'] = 0
    df_test['xgb_predictions'] = 0

    for xgb_train_predictions_file in xgb_train_predictions_files:
        df_xgb_train_predictions = pd.read_csv(xgb_train_predictions_file)
        scores = metrics.classification_scores(df_train['ARTIS_DURUMU'], df_xgb_train_predictions['xgb_predictions'], threshold=0.5)
        print(f'Reading: {xgb_train_predictions_file}\n{scores}')
        df_train['xgb_predictions'] += (df_xgb_train_predictions['xgb_predictions'] / len(xgb_train_predictions_files))

    for xgb_test_predictions_file in xgb_test_predictions_files:
        df_xgb_test_predictions = pd.read_csv(xgb_test_predictions_file)
        df_test['xgb_predictions'] += (df_xgb_test_predictions['xgb_predictions'] / len(xgb_test_predictions_files))

    blend_train_predictions = (df_train['lgb_predictions'] * 1.) + (df_train['xgb_predictions'] * 0.)
    blend_test_predictions = (df_test['lgb_predictions'] * 1.) + (df_test['xgb_predictions'] * 0.)
    blend_scores = metrics.classification_scores(df_train['ARTIS_DURUMU'], blend_train_predictions, threshold=0.5)
    print(f'\nBlend of {len(lgb_train_predictions_files) + len(xgb_train_predictions_files)} Models\n{blend_scores}')

    df_train['final_predictions'] = blend_train_predictions
    df_test['final_predictions'] = blend_test_predictions

    for df in [df_train, df_test]:
        df['CUSTOMER_ID1'] = df['OFFICE_ID'].astype(str) + df['SIGORTA_TIP'].astype(str) + df['SIGORTA_TIP'].astype(str) +\
                             df['SOZLESME_KOKENI'].astype(str) + df['SOZLESME_KOKENI_DETAY'].astype(str) + df['BASLANGIC_TARIHI'].astype(str) +\
                             df['KAPSAM_TIPI'].astype(str) + df['KAPSAM_GRUBU'].astype(str) + df['DAGITIM_KANALI'].astype(str) + df['POLICE_SEHIR'].astype(str) +\
                             df['DOGUM_TARIHI'].astype(str) + df['CINSIYET'].astype(str) + df['UYRUK'].astype(str) + df['MEMLEKET'].astype(str) +\
                             df['MESLEK'].astype(str) + df['MESLEK_KIRILIM'].astype(str) + df['MUSTERI_SEGMENTI'].astype(str) + df['YATIRIM_KARAKTERI'].astype(str) +\
                             df['MEDENI_HAL'].astype(str) + df['EGITIM_DURUM'].astype(str) + df['GELIR'].astype(str) + df['COCUK_SAYISI'].astype(str)

        df['CUSTOMER_ID2'] = df['OFFICE_ID'].astype(str) + df['SIGORTA_TIP'].astype(str) + df['SIGORTA_TIP'].astype(str) + \
                             df['SOZLESME_KOKENI'].astype(str) + df['SOZLESME_KOKENI_DETAY'].astype(str) + df['BASLANGIC_TARIHI'].astype(str) + \
                             df['KAPSAM_TIPI'].astype(str) + df['KAPSAM_GRUBU'].astype(str) + df['DAGITIM_KANALI'].astype(str) + df['POLICE_SEHIR'].astype(str) + \
                             df['DOGUM_TARIHI'].astype(str) + df['CINSIYET'].astype(str) + df['UYRUK'].astype(str) + df['MEMLEKET'].astype(str) + \
                             df['MESLEK'].astype(str) + df['MESLEK_KIRILIM'].astype(str) + df['MUSTERI_SEGMENTI'].astype(str) + df['YATIRIM_KARAKTERI'].astype(str) + \
                             df['MEDENI_HAL'].astype(str) + df['EGITIM_DURUM'].astype(str) + df['GELIR'].astype(str) + df['COCUK_SAYISI'].astype(str) + \
                             df['OCAK_ODENEN_TUTAR'].astype(str) + df['OCAK_VADE_TUTARI'].astype(str) +\
                             df['SUBAT_ODENEN_TU'].astype(str) + df['SUBAT_VADE_TUTARI'].astype(str) +\
                             df['MART_ODENEN_TUTAR'].astype(str) + df['MART_VADE_TUTARI'].astype(str) +\
                             df['NISAN_ODENEN_TUTAR'].astype(str) + df['NISAN_VADE_TUTARI'].astype(str) +\
                             df['MAYIS_ODENEN_TUTAR'].astype(str) + df['MAYIS_VADE_TUTARI'].astype(str) +\
                             df['HAZIRAN_ODENEN_TUTAR'].astype(str) + df['HAZIRAN_VADE_TUTARI'].astype(str) +\
                             df['TEMMUZ_ODENEN_TUTAR'].astype(str) + df['TEMMUZ_VADE_TUTARI'].astype(str) +\
                             df['AGUSTOS_ODENEN_TUTAR'].astype(str) + df['AGUSTOS_VADE_TUTARI'].astype(str) +\
                             df['EYLUL_ODENEN_TUTAR'].astype(str) + df['EYLUL_VADE_TUTARI'].astype(str) +\
                             df['EKIM_ODENEN_TUTAR'].astype(str) + df['EKIM_VADE_TUTARI'].astype(str) +\
                             df['KASIM_ODENEN_TUTAR'].astype(str) + df['KASIM_VADE_TUTARI'].astype(str) +\
                             df['ARALIK_ODENEN_TUTAR'].astype(str) + df['ARALIK_VADE_TUTARI'].astype(str)

    match_and_overwrite_target(df_train=df_train, df_test=df_test, column='CUSTOMER_ID1', agg='mean')
    match_and_overwrite_target(df_train=df_train, df_test=df_test, column='CUSTOMER_ID2', agg='mean')
    adjusted_threshold_scores = metrics.classification_scores(df_train['ARTIS_DURUMU'], df_train['final_predictions'], threshold=0.21)
    print(f'\nAdjusted Threshold\n{adjusted_threshold_scores}')

    df_submission = pd.read_csv(settings.DATA / 'sample_submission.csv')
    df_submission['ARTIS_DURUMU'] = metrics.round_probabilities(df_test['final_predictions'].values, threshold=0.21)
    df_submission.to_csv(settings.DATA / 'submission.csv', index=False)
