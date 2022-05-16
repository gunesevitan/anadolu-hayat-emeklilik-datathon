import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import lightgbm as lgb

import settings
import preprocessing


if __name__ == '__main__':

    # Select which features to impute
    IMPUTE_MUSTERI_SEGMENTI = False

    df_train = pd.read_csv(settings.DATA / 'train.csv')
    df_test = pd.read_csv(settings.DATA / 'test.csv', low_memory=False)
    df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)

    # Apply initial preprocessing steps
    preprocessor = preprocessing.TabularPreprocessor(
        df_train=df_train,
        df_test=df_test,
        fill_missing=False
    )
    df_train, df_test = preprocessor.transform()
    for df in [df_train, df_test]:
        # Missing SOZLESME_KOKENI_DETAY values are always NEW
        df['SOZLESME_KOKENI_DETAY'] = df['SOZLESME_KOKENI_DETAY'].fillna('NEW')

    feature_missing_value_counts = df_all.isnull().sum()
    features_with_missing_values = feature_missing_value_counts[feature_missing_value_counts > 0].to_dict()

    if IMPUTE_MUSTERI_SEGMENTI:

        # Convert categories into actual labels
        label_encoder = LabelEncoder()
        label_encoder.fit(df_all['MUSTERI_SEGMENTI'])
        df_train['MUSTERI_SEGMENTI_LABEL'] = label_encoder.transform(df_train['MUSTERI_SEGMENTI'])
        df_test['MUSTERI_SEGMENTI_LABEL'] = label_encoder.transform(df_test['MUSTERI_SEGMENTI'])

        predictors = [
            'OFFICE_ID', 'SIGORTA_TIP', 'SOZLESME_KOKENI', 'KAPSAM_TIPI', 'KAPSAM_GRUBU',
            'POLICE_SEHIR', 'CINSIYET', 'MEMLEKET', 'MESLEK', 'MESLEK_KIRILIM',
            'YATIRIM_KARAKTERI', 'MEDENI_HAL', 'EGITIM_DURUM', 'GELIR', 'COCUK_SAYISI',
            'ODENEN_TUTAR_STD', 'VADE_TUTARI_STD', 'ODENMEYEN_AYLAR', 'SENE_HESAP_DEGERI_ORANI',
            'PAYMENT_AMOUNT_PCA_COMPONENT_0', 'PAYMENT_AMOUNT_PCA_COMPONENT_1',
            'PAID_AMOUNT_PCA_COMPONENT_0', 'PAID_AMOUNT_PCA_COMPONENT_1', 'PAID_AMOUNT_PCA_COMPONENT_2',
            'UNPAID_AMOUNT_PCA_COMPONENT_0', 'UNPAID_AMOUNT_PCA_COMPONENT_1', 'UNPAID_AMOUNT_PCA_COMPONENT_2',
            'PAID_AMOUNT_RATIO_PCA_COMPONENT_0', 'PAID_AMOUNT_RATIO_PCA_COMPONENT_1', 'PAID_AMOUNT_RATIO_PCA_COMPONENT_2',
            'SOZLESME_TOPLAM_AY', 'YAS', 'POLICE_SEHIR_FREQUENCY',
        ]

        # Train on training set and validate on test set
        train_idx, test_idx = df_train['MUSTERI_SEGMENTI'].notnull(), df_test['MUSTERI_SEGMENTI'].notnull()
        train_dataset = lgb.Dataset(df_train.loc[train_idx, predictors], label=df_train.loc[train_idx, 'MUSTERI_SEGMENTI_LABEL'])
        test_dataset = lgb.Dataset(df_test.loc[test_idx, predictors], label=df_test.loc[test_idx, 'MUSTERI_SEGMENTI_LABEL'])
        # Set model parameters, train parameters, callbacks and start training
        model = lgb.train(
            params={
                'num_leaves': 256,
                'learning_rate': 0.1,
                'bagging_fraction': 0.9,
                'bagging_frequency': 1,
                'feature_fraction': 0.8,
                'feature_fraction_by_node': 0.9,
                'min_data_in_leaf': 5,
                'min_gain_to_split': 0.00001,
                'lambda_l1': 0,
                'lambda_l2': 0,
                'max_bin': 255,
                'max_depth': -1,
                'objective': 'multiclass',
                'num_classes': 6,
                'classes': 6,
                'seed': 42,
                'feature_fraction_seed': 42,
                'bagging_seed': 42,
                'drop_seed': 42,
                'data_random_seed': 42,
                'boosting_type': 'gbdt',
                'verbose': 1,
                'metric': 'multi_logloss',
                'n_jobs': -1
            },
            train_set=train_dataset,
            valid_sets=[train_dataset, test_dataset],
            num_boost_round=10000,
            callbacks=[
                lgb.early_stopping(250),
                lgb.log_evaluation(250)
            ]
        )

        test_predictions = np.argmax(model.predict(df_test.loc[test_idx, predictors]), axis=1)
        test_accuracy = accuracy_score(df_test.loc[test_idx, 'MUSTERI_SEGMENTI_LABEL'], test_predictions)
        print(f'\nMUSTERI_SEGMENTI Imputer Model Test Accuracy: {test_accuracy:.6f}')

        # Predict missing values on training and test sets, and write csv files
        df_train.loc[df_train['MUSTERI_SEGMENTI'].isnull(), 'MUSTERI_SEGMENTI'] = label_encoder.inverse_transform(np.argmax(model.predict(df_train.loc[df_train['MUSTERI_SEGMENTI'].isnull(), predictors]), axis=1))
        df_test.loc[df_test['MUSTERI_SEGMENTI'].isnull(), 'MUSTERI_SEGMENTI'] = label_encoder.inverse_transform(np.argmax(model.predict(df_test.loc[df_test['MUSTERI_SEGMENTI'].isnull(), predictors]), axis=1))
        df_train['MUSTERI_SEGMENTI'].to_csv(settings.DATA / 'train_MUSTERI_SEGMENTI.csv', index=False)
        df_test['MUSTERI_SEGMENTI'].to_csv(settings.DATA / 'test_MUSTERI_SEGMENTI.csv', index=False)
