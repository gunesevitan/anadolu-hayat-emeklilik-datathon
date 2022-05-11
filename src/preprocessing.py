import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

import settings


class TabularPreprocessor:

    def __init__(self, df_train, df_test, fill_missing=False):

        self.df_train = df_train.copy(deep=True)
        self.df_test = df_test.copy(deep=True)
        self.fill_missing = fill_missing

    def get_folds(self):

        """
        Read and merge pre-computed folds
        """

        df_folds = pd.read_csv(settings.DATA / 'folds.csv')
        self.df_train['fold'] = df_folds['fold']

    def clean_features(self):

        """
        Clean artefacts, typos and other stuff from features
        """

        self.df_train.loc[self.df_train['SOZLESME_KOKENI'] == 'TRANS', 'SOZLESME_KOKENI'] = 'TRANS_C'
        self.df_train['GELIR'] = self.df_train['GELIR'].str.replace(',', '.').astype(np.float32)
        self.df_test['GELIR'] = self.df_test['GELIR'].str.replace(',', '.').astype(np.float32)

    def fill_missing_values(self):

        """
        Fill missing values in features
        """

        for df in [self.df_train, self.df_test]:
            df['SOZLESME_KOKENI_DETAY'] = df['SOZLESME_KOKENI_DETAY'].fillna('NEW')

    def encode_categoricals(self):

        """
        Encode categorical features
        """

        # Encode categorical features with label encoder if all categories in training set exist in test set
        categorical_features = [
            'SOZLESME_KOKENI', 'SOZLESME_KOKENI_DETAY', 'KAPSAM_GRUBU', 'DAGITIM_KANALI',
            'MESLEK', 'MESLEK_KIRILIM', 'YATIRIM_KARAKTERI', 'MEDENI_HAL', 'EGITIM_DURUM'
        ]

        for categorical_feature in categorical_features:
            label_encoder = LabelEncoder()
            self.df_train[categorical_feature] = label_encoder.fit_transform(self.df_train[categorical_feature])
            self.df_test[categorical_feature] = label_encoder.transform(self.df_test[categorical_feature])

        # Encode categorical features with ordinal encoder if there are unseen categories in test set
        categorical_features_with_unseen_categories = [
            'KAPSAM_TIPI', 'POLICE_SEHIR', 'UYRUK',
        ]

        for categorical_feature in categorical_features_with_unseen_categories:
            ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)
            self.df_train[categorical_feature] = ordinal_encoder.fit_transform(self.df_train[categorical_feature].values.reshape(-1, 1))
            self.df_test[categorical_feature] = ordinal_encoder.transform(self.df_test[categorical_feature].values.reshape(-1, 1))

    def create_datetime_features(self):

        """
        Create features from dates and time
        """

        for df in [self.df_train, self.df_test]:
            df['BASLANGIC_TARIHI'] = pd.to_datetime(df['BASLANGIC_TARIHI'])
            df['SOZLESME_TOPLAM_AY'] = (pd.Timestamp('2020-01-01') - pd.to_datetime(df['BASLANGIC_TARIHI'])) / np.timedelta64(1, 'M')
            df['YAS'] = 2020 - df['DOGUM_TARIHI']

    def create_payment_features(self):

        """
        Create features from payments
        """

        for df in [self.df_train, self.df_test]:
            # Payment difference for every month
            df['OCAK_ODENMEYEN_TUTAR'] = df['OCAK_VADE_TUTARI'] - df['OCAK_ODENEN_TUTAR']
            df['SUBAT_ODENMEYEN_TUTAR'] = df['SUBAT_VADE_TUTARI'] - df['SUBAT_ODENEN_TU']
            df['MART_ODENMEYEN_TUTAR'] = df['MART_VADE_TUTARI'] - df['MART_ODENEN_TUTAR']
            df['NISAN_ODENMEYEN_TUTAR'] = df['NISAN_VADE_TUTARI'] - df['NISAN_ODENEN_TUTAR']
            df['MAYIS_ODENMEYEN_TUTAR'] = df['MAYIS_VADE_TUTARI'] - df['MAYIS_ODENEN_TUTAR']
            df['HAZIRAN_ODENMEYEN_TUTAR'] = df['HAZIRAN_VADE_TUTARI'] - df['HAZIRAN_ODENEN_TUTAR']
            df['TEMMUZ_ODENMEYEN_TUTAR'] = df['TEMMUZ_VADE_TUTARI'] - df['TEMMUZ_ODENEN_TUTAR']
            df['AGUSTOS_ODENMEYEN_TUTAR'] = df['AGUSTOS_VADE_TUTARI'] - df['AGUSTOS_ODENEN_TUTAR']
            df['EYLUL_ODENMEYEN_TUTAR'] = df['EYLUL_VADE_TUTARI'] - df['EYLUL_ODENEN_TUTAR']
            df['EKIM_ODENMEYEN_TUTAR'] = df['EKIM_VADE_TUTARI'] - df['EKIM_ODENEN_TUTAR']
            df['KASIM_ODENMEYEN_TUTAR'] = df['KASIM_VADE_TUTARI'] - df['KASIM_ODENEN_TUTAR']
            df['ARALIK_ODENMEYEN_TUTAR'] = df['ARALIK_VADE_TUTARI'] - df['ARALIK_ODENEN_TUTAR']
            # Payment ratio for every month
            df['OCAK_ODENEN_TUTAR_ORANI'] = df['OCAK_ODENEN_TUTAR'] / df['OCAK_VADE_TUTARI']
            df['SUBAT_ODENEN_TUTAR_ORANI'] = df['SUBAT_ODENEN_TU'] / df['SUBAT_VADE_TUTARI']
            df['MART_ODENEN_TUTAR_ORANI'] = df['MART_ODENEN_TUTAR'] / df['MART_VADE_TUTARI']
            df['NISAN_ODENEN_TUTAR_ORANI'] = df['NISAN_ODENEN_TUTAR'] / df['NISAN_VADE_TUTARI']
            df['MAYIS_ODENEN_TUTAR_ORANI'] = df['MAYIS_ODENEN_TUTAR'] / df['MAYIS_VADE_TUTARI']
            df['HAZIRAN_ODENEN_TUTAR_ORANI'] = df['HAZIRAN_ODENEN_TUTAR'] / df['HAZIRAN_VADE_TUTARI']
            df['TEMMUZ_ODENEN_TUTAR_ORANI'] = df['TEMMUZ_ODENEN_TUTAR'] / df['TEMMUZ_VADE_TUTARI']
            df['AGUSTOS_ODENEN_TUTAR_ORANI'] = df['AGUSTOS_ODENEN_TUTAR'] / df['AGUSTOS_VADE_TUTARI']
            df['EYLUL_ODENEN_TUTAR_ORANI'] = df['EYLUL_ODENEN_TUTAR'] / df['EYLUL_VADE_TUTARI']
            df['EKIM_ODENEN_TUTAR_ORANI'] = df['EKIM_ODENEN_TUTAR'] / df['EKIM_VADE_TUTARI']
            df['KASIM_ODENEN_TUTAR_ORANI'] = df['KASIM_ODENEN_TUTAR'] / df['KASIM_VADE_TUTARI']
            df['ARALIK_ODENEN_TUTAR_ORANI'] = df['ARALIK_ODENEN_TUTAR'] / df['ARALIK_VADE_TUTARI']

        odenen_tutar_columns = [
            'OCAK_ODENEN_TUTAR', 'SUBAT_ODENEN_TU', 'MART_ODENEN_TUTAR', 'NISAN_ODENEN_TUTAR',
            'MAYIS_ODENEN_TUTAR', 'HAZIRAN_ODENEN_TUTAR', 'TEMMUZ_ODENEN_TUTAR', 'AGUSTOS_ODENEN_TUTAR',
            'EYLUL_ODENEN_TUTAR', 'EKIM_ODENEN_TUTAR', 'KASIM_ODENEN_TUTAR', 'ARALIK_ODENEN_TUTAR'
        ]

        vade_tutari_columns = [
            'OCAK_VADE_TUTARI', 'SUBAT_VADE_TUTARI', 'MART_VADE_TUTARI', 'NISAN_VADE_TUTARI',
            'MAYIS_VADE_TUTARI', 'HAZIRAN_VADE_TUTARI', 'TEMMUZ_VADE_TUTARI', 'AGUSTOS_VADE_TUTARI',
            'EYLUL_VADE_TUTARI', 'EKIM_VADE_TUTARI', 'KASIM_VADE_TUTARI', 'ARALIK_VADE_TUTARI'
        ]

        odenmeyen_tutar_columns = [
            'OCAK_ODENMEYEN_TUTAR', 'SUBAT_ODENMEYEN_TUTAR', 'MART_ODENMEYEN_TUTAR', 'NISAN_ODENMEYEN_TUTAR',
            'MAYIS_ODENMEYEN_TUTAR', 'HAZIRAN_ODENMEYEN_TUTAR', 'TEMMUZ_ODENMEYEN_TUTAR', 'AGUSTOS_ODENMEYEN_TUTAR',
            'EYLUL_ODENMEYEN_TUTAR', 'EKIM_ODENMEYEN_TUTAR', 'KASIM_ODENMEYEN_TUTAR', 'ARALIK_ODENMEYEN_TUTAR'
        ]

        for df in [self.df_train, self.df_test]:
            df['SENE_HESAP_DEGERI_FARKI'] = df['SENE_SONU_HESAP_DEGERI'] - df['SENE_BASI_HESAP_DEGERI']
            df['SENE_HESAP_DEGERI_ORANI'] = df['SENE_BASI_HESAP_DEGERI'] / df['SENE_SONU_HESAP_DEGERI']
            df['VADE_TUTARI_STD'] = df[vade_tutari_columns].std(axis=1)
            df['ODENEN_TUTAR_STD'] = df[odenen_tutar_columns].std(axis=1)
            df['ODENMEYEN_TUTAR_STD'] = df[odenmeyen_tutar_columns].std(axis=1)
            df['ODENMEYEN_TUTAR_AYLAR'] = (df[odenmeyen_tutar_columns] > 0).sum(axis=1)

    def encode_frequency(self):

        """
        Create frequency encoded categorical features
        """

        df_all = pd.concat((self.df_train, self.df_test), axis=0, ignore_index=True)
        categorical_features = [
            'POLICE_SEHIR',
        ]
        for categorical_feature in categorical_features:
            category_frequencies = df_all[categorical_feature].value_counts().to_dict()
            self.df_train[f'{categorical_feature}_FREQUENCY'] = self.df_train[categorical_feature].map(category_frequencies)
            self.df_test[f'{categorical_feature}_FREQUENCY'] = self.df_test[categorical_feature].map(category_frequencies)

    def create_aggregation_features(self):

        """
        Create aggregation features
        """

        df_all = pd.concat((self.df_train, self.df_test), axis=0, ignore_index=True)
        continuous_features = [
            'ARALIK_ODENEN_TUTAR', 'ARALIK_ODENMEYEN_TUTAR'
        ]
        categorical_features = [
            'OFFICE_ID', 'MUSTERI_SEGMENTI'
        ]

        for categorical_feature in categorical_features:
            for continuous_feature in continuous_features:
                for aggregation in ['mean', 'std', 'min', 'max']:
                    df_agg = df_all.groupby(categorical_feature)[continuous_feature].agg(aggregation)
                    self.df_train[f'{categorical_feature}_{continuous_feature}_{aggregation.upper()}'] = self.df_train[categorical_feature].map(df_agg)
                    self.df_test[f'{categorical_feature}_{continuous_feature}_{aggregation.upper()}'] = self.df_test[categorical_feature].map(df_agg)

    def transform(self):

        """
        Apply transformations to training and test sets

        Returns
        -------
        df_train (pandas.DataFrame of shape (634112, n_columns)): Training dataframe of features, target and folds
        df_test (pandas.DataFrame of shape (243137, n_columns)): Test dataframe of features
        """

        self.get_folds()
        self.clean_features()
        if self.fill_missing:
            self.fill_missing_values()
        self.encode_categoricals()
        self.create_datetime_features()
        self.create_payment_features()
        self.encode_frequency()
        self.create_aggregation_features()

        return self.df_train, self.df_test
