import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error

import settings


class TabularPreprocessor:

    def __init__(self, df_train, df_test, fill_missing=False, random_state=42):

        self.df_train = df_train.copy(deep=True)
        self.df_test = df_test.copy(deep=True)
        self.fill_missing = fill_missing
        self.random_state = random_state

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

        # Replace comma with dots for float conversion
        self.df_train['GELIR'] = self.df_train['GELIR'].str.replace(',', '.').astype(np.float32)
        self.df_test['GELIR'] = self.df_test['GELIR'].str.replace(',', '.').astype(np.float32)

        # Few negative values in GELIR are replaced with zero
        self.df_train.loc[self.df_train['GELIR'] < 0, 'GELIR'] = 0
        self.df_test.loc[self.df_test['GELIR'] < 0, 'GELIR'] = 0

    def fill_missing_values(self):

        """
        Fill missing values in features
        """

        for df in [self.df_train, self.df_test]:

            # Missing SOZLESME_KOKENI_DETAY values are always NEW
            df['SOZLESME_KOKENI_DETAY'] = df['SOZLESME_KOKENI_DETAY'].fillna('NEW')

        # Missing MUSTERI_SEGMENTI values are predicted with LightGBM model
        self.df_train['MUSTERI_SEGMENTI'] = pd.read_csv(settings.DATA / 'train_MUSTERI_SEGMENTI.csv')
        self.df_test['MUSTERI_SEGMENTI'] = pd.read_csv(settings.DATA / 'test_MUSTERI_SEGMENTI.csv')
        # Missing KAPSAM_TIPI values are replaced with most frequent value
        self.df_test['KAPSAM_TIPI'] = self.df_test['KAPSAM_TIPI'].fillna(124)

        if self.fill_missing:
            for df in [self.df_train, self.df_test]:
                for feature in ['POLICE_SEHIR', 'UYRUK']:
                    # Missing POLICE_SEHIR and UYRUK values are filled as a new "Missing" category
                    df[feature] = df[feature].fillna('Missing')

    def create_categorical_feature_combinations(self):

        """
        Create combinations of categorical features
        """

        for df in [self.df_train, self.df_test]:

            df['USER_ID1.1'] = df['OFFICE_ID'].astype(str) + '_' + df['KAPSAM_TIPI'].astype(str)

    def encode_categoricals(self):

        """
        Encode categorical features
        """

        categorical_features = [
            'SOZLESME_KOKENI', 'SOZLESME_KOKENI_DETAY', 'KAPSAM_GRUBU', 'DAGITIM_KANALI',
            'MESLEK', 'MESLEK_KIRILIM', 'YATIRIM_KARAKTERI', 'MEDENI_HAL', 'EGITIM_DURUM',
        ]

        # Encode categorical features with label encoder if all categories in training set exist in test set
        for categorical_feature in categorical_features:
            label_encoder = LabelEncoder()
            self.df_train[categorical_feature] = label_encoder.fit_transform(self.df_train[categorical_feature])
            self.df_test[categorical_feature] = label_encoder.transform(self.df_test[categorical_feature])

        categorical_features_with_unseen_categories = [
            'KAPSAM_TIPI', 'POLICE_SEHIR', 'UYRUK',
            'USER_ID1.1',
        ]

        # Encode categorical features with ordinal encoder if there are unseen categories in test set
        for categorical_feature in categorical_features_with_unseen_categories:
            ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)
            self.df_train[categorical_feature] = ordinal_encoder.fit_transform(self.df_train[categorical_feature].values.reshape(-1, 1))
            self.df_test[categorical_feature] = ordinal_encoder.transform(self.df_test[categorical_feature].values.reshape(-1, 1))

    def create_datetime_features(self):

        """
        Create features from dates and time
        """

        for df in [self.df_train, self.df_test]:

            # SOZLESME_TOPLAM_AY is the total months between 2020-01-01 - BASLANGIC_TARIHI
            df['BASLANGIC_TARIHI'] = pd.to_datetime(df['BASLANGIC_TARIHI'])
            df['SOZLESME_TOPLAM_AY'] = (pd.Timestamp('2020-01-01') - pd.to_datetime(df['BASLANGIC_TARIHI'])) / np.timedelta64(1, 'M')

            # Age at the given time is 2020 - DOGUM_TARIHI
            df['YAS'] = 2020 - df['DOGUM_TARIHI']

    def transform_continuous_features(self):

        """
        Transform continuous features
        """

        positive_continous_features = [
            'GELIR',
            'OCAK_ODENEN_TUTAR', 'OCAK_VADE_TUTARI', 'SUBAT_ODENEN_TU', 'SUBAT_VADE_TUTARI',
            'MART_ODENEN_TUTAR', 'MART_VADE_TUTARI', 'NISAN_ODENEN_TUTAR', 'NISAN_VADE_TUTARI',
            'MAYIS_ODENEN_TUTAR', 'MAYIS_VADE_TUTARI', 'HAZIRAN_ODENEN_TUTAR', 'HAZIRAN_VADE_TUTARI',
            'TEMMUZ_ODENEN_TUTAR', 'TEMMUZ_VADE_TUTARI', 'AGUSTOS_ODENEN_TUTAR', 'AGUSTOS_VADE_TUTARI',
            'EYLUL_ODENEN_TUTAR', 'EYLUL_VADE_TUTARI', 'EKIM_ODENEN_TUTAR', 'EKIM_VADE_TUTARI',
            'KASIM_ODENEN_TUTAR', 'KASIM_VADE_TUTARI', 'ARALIK_ODENEN_TUTAR', 'ARALIK_VADE_TUTARI'
        ]

        # Log transform positive continuous features in order to deal with extreme skewness
        # Using log1p operation since vanilla log transforms zeros to -infinite
        for df in [self.df_train, self.df_test]:
            for continous_feature in positive_continous_features:
                df[continous_feature] = np.log1p(df[continous_feature])

        positive_and_negative_continuous_features = ['SENE_BASI_HESAP_DEGERI', 'SENE_SONU_HESAP_DEGERI']

        # Continuous features with both positive and negative values are shifted by the smallest negative value
        # Log1p operation is applied after shifting
        for df in [self.df_train, self.df_test]:
            for continous_feature in positive_and_negative_continuous_features:
                smallest_negative_value = pd.concat((
                    self.df_train[continous_feature],
                    self.df_test[continous_feature]
                ), axis=0, ignore_index=True).min()
                df[continous_feature] = np.log1p(df[continous_feature] + 1 - smallest_negative_value)

    def create_payment_features(self):

        """
        Create features from payments
        """

        for df in [self.df_train, self.df_test]:

            # Create unpaid amount features for every month
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

            # Create paid amount ratio features for every month
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

            # Create payment/income ratio features for every month
            df['OCAK_GELIR_VADE_TUTARI_ORANI'] = df['GELIR'] / df['OCAK_VADE_TUTARI']
            df['SUBAT_GELIR_VADE_TUTARI_ORANI'] = df['GELIR'] / df['SUBAT_VADE_TUTARI']
            df['MART_GELIR_VADE_TUTARI_ORANI'] = df['GELIR'] / df['MART_VADE_TUTARI']
            df['NISAN_GELIR_VADE_TUTARI_ORANI'] = df['GELIR'] / df['NISAN_VADE_TUTARI']
            df['MAYIS_GELIR_VADE_TUTARI_ORANI'] = df['GELIR'] / df['MAYIS_VADE_TUTARI']
            df['HAZIRAN_GELIR_VADE_TUTARI_ORANI'] = df['GELIR'] / df['HAZIRAN_VADE_TUTARI']
            df['TEMMUZ_GELIR_VADE_TUTARI_ORANI'] = df['GELIR'] / df['TEMMUZ_VADE_TUTARI']
            df['AGUSTOS_GELIR_VADE_TUTARI_ORANI'] = df['GELIR'] / df['AGUSTOS_VADE_TUTARI']
            df['EYLUL_GELIR_VADE_TUTARI_ORANI'] = df['GELIR'] / df['EYLUL_VADE_TUTARI']
            df['EKIM_GELIR_VADE_TUTARI_ORANI'] = df['GELIR'] / df['EKIM_VADE_TUTARI']
            df['KASIM_GELIR_VADE_TUTARI_ORANI'] = df['GELIR'] / df['KASIM_VADE_TUTARI']
            df['ARALIK_GELIR_VADE_TUTARI_ORANI'] = df['GELIR'] / df['ARALIK_VADE_TUTARI']

        payment_amount_columns = [
            'OCAK_VADE_TUTARI', 'SUBAT_VADE_TUTARI', 'MART_VADE_TUTARI', 'NISAN_VADE_TUTARI',
            'MAYIS_VADE_TUTARI', 'HAZIRAN_VADE_TUTARI', 'TEMMUZ_VADE_TUTARI', 'AGUSTOS_VADE_TUTARI',
            'EYLUL_VADE_TUTARI', 'EKIM_VADE_TUTARI', 'KASIM_VADE_TUTARI', 'ARALIK_VADE_TUTARI'
        ]

        paid_amount_columns = [
            'OCAK_ODENEN_TUTAR', 'SUBAT_ODENEN_TU', 'MART_ODENEN_TUTAR', 'NISAN_ODENEN_TUTAR',
            'MAYIS_ODENEN_TUTAR', 'HAZIRAN_ODENEN_TUTAR', 'TEMMUZ_ODENEN_TUTAR', 'AGUSTOS_ODENEN_TUTAR',
            'EYLUL_ODENEN_TUTAR', 'EKIM_ODENEN_TUTAR', 'KASIM_ODENEN_TUTAR', 'ARALIK_ODENEN_TUTAR'
        ]

        unpaid_amount_columns = [
            'OCAK_ODENMEYEN_TUTAR', 'SUBAT_ODENMEYEN_TUTAR', 'MART_ODENMEYEN_TUTAR', 'NISAN_ODENMEYEN_TUTAR',
            'MAYIS_ODENMEYEN_TUTAR', 'HAZIRAN_ODENMEYEN_TUTAR', 'TEMMUZ_ODENMEYEN_TUTAR', 'AGUSTOS_ODENMEYEN_TUTAR',
            'EYLUL_ODENMEYEN_TUTAR', 'EKIM_ODENMEYEN_TUTAR', 'KASIM_ODENMEYEN_TUTAR', 'ARALIK_ODENMEYEN_TUTAR'
        ]

        paid_amount_ratio_columns = [
            'OCAK_ODENEN_TUTAR_ORANI', 'SUBAT_ODENEN_TUTAR_ORANI', 'MART_ODENEN_TUTAR_ORANI', 'NISAN_ODENEN_TUTAR_ORANI',
            'MAYIS_ODENEN_TUTAR_ORANI', 'HAZIRAN_ODENEN_TUTAR_ORANI', 'TEMMUZ_ODENEN_TUTAR_ORANI', 'AGUSTOS_ODENEN_TUTAR_ORANI',
            'EYLUL_ODENEN_TUTAR_ORANI', 'EKIM_ODENEN_TUTAR_ORANI', 'KASIM_ODENEN_TUTAR_ORANI', 'ARALIK_ODENEN_TUTAR_ORANI'
        ]

        for df in [self.df_train, self.df_test]:

            # Start/End of the year amount difference and ratio features
            df['SENE_HESAP_DEGERI_FARKI'] = df['SENE_SONU_HESAP_DEGERI'] - df['SENE_BASI_HESAP_DEGERI']
            df['SENE_HESAP_DEGERI_ORANI'] = df['SENE_BASI_HESAP_DEGERI'] / df['SENE_SONU_HESAP_DEGERI']

            # Statistical features calculated on payment amount
            df['VADE_TUTARI_MEAN'] = df[payment_amount_columns].mean(axis=1)
            df['VADE_TUTARI_STD'] = df[payment_amount_columns].std(axis=1)
            df['VADE_TUTARI_MIN'] = df[payment_amount_columns].min(axis=1)
            df['VADE_TUTARI_MAX'] = df[payment_amount_columns].max(axis=1)
            df['VADE_TUTARI_SUM'] = df[payment_amount_columns].sum(axis=1)
            df['VADE_TUTARI_SKEW'] = df[payment_amount_columns].skew(axis=1)

            # Statistical features calculated on paid amount
            df['ODENEN_TUTAR_MEAN'] = df[paid_amount_columns].mean(axis=1)
            df['ODENEN_TUTAR_STD'] = df[paid_amount_columns].std(axis=1)
            df['ODENEN_TUTAR_MIN'] = df[paid_amount_columns].min(axis=1)
            df['ODENEN_TUTAR_MAX'] = df[paid_amount_columns].max(axis=1)
            df['ODENEN_TUTAR_SUM'] = df[paid_amount_columns].sum(axis=1)
            df['ODENEN_TUTAR_SKEW'] = df[paid_amount_columns].skew(axis=1)

            # Statistical features calculated on unpaid amount
            df['ODENMEYEN_TUTAR_MEAN'] = df[unpaid_amount_columns].mean(axis=1)
            df['ODENMEYEN_TUTAR_STD'] = df[unpaid_amount_columns].std(axis=1)
            df['ODENMEYEN_TUTAR_MIN'] = df[unpaid_amount_columns].min(axis=1)
            df['ODENMEYEN_TUTAR_MAX'] = df[unpaid_amount_columns].max(axis=1)
            df['ODENMEYEN_TUTAR_SUM'] = df[unpaid_amount_columns].sum(axis=1)
            df['ODENMEYEN_TUTAR_SKEW'] = df[unpaid_amount_columns].skew(axis=1)
            df['ODENEN_AYLAR'] = (df[unpaid_amount_columns] == 0).sum(axis=1)
            df['ODENMEYEN_AYLAR'] = (df[unpaid_amount_columns] > 0).sum(axis=1)

            # Statistical features calculated on paid amount ratio
            df['ODENEN_TUTAR_ORANI_MEAN'] = df[paid_amount_ratio_columns].mean(axis=1)
            df['ODENEN_TUTAR_ORANI_STD'] = df[paid_amount_ratio_columns].std(axis=1)
            df['ODENEN_TUTAR_ORANI_MIN'] = df[paid_amount_ratio_columns].min(axis=1)
            df['ODENEN_TUTAR_ORANI_MAX'] = df[paid_amount_ratio_columns].max(axis=1)
            df['ODENEN_TUTAR_ORANI_SUM'] = df[paid_amount_ratio_columns].sum(axis=1)
            df['ODENEN_TUTAR_ORANI_SKEW'] = df[paid_amount_ratio_columns].skew(axis=1)

    def reduce_dimensions(self):

        """
        Reduce dimensions of correlated feature groups
        """

        payment_amount_columns = [
            'OCAK_VADE_TUTARI', 'SUBAT_VADE_TUTARI', 'MART_VADE_TUTARI', 'NISAN_VADE_TUTARI',
            'MAYIS_VADE_TUTARI', 'HAZIRAN_VADE_TUTARI', 'TEMMUZ_VADE_TUTARI', 'AGUSTOS_VADE_TUTARI',
            'EYLUL_VADE_TUTARI', 'EKIM_VADE_TUTARI', 'KASIM_VADE_TUTARI', 'ARALIK_VADE_TUTARI'
        ]

        paid_amount_columns = [
            'OCAK_ODENEN_TUTAR', 'SUBAT_ODENEN_TU', 'MART_ODENEN_TUTAR', 'NISAN_ODENEN_TUTAR',
            'MAYIS_ODENEN_TUTAR', 'HAZIRAN_ODENEN_TUTAR', 'TEMMUZ_ODENEN_TUTAR', 'AGUSTOS_ODENEN_TUTAR',
            'EYLUL_ODENEN_TUTAR', 'EKIM_ODENEN_TUTAR', 'KASIM_ODENEN_TUTAR', 'ARALIK_ODENEN_TUTAR'
        ]

        unpaid_amount_columns = [
            'OCAK_ODENMEYEN_TUTAR', 'SUBAT_ODENMEYEN_TUTAR', 'MART_ODENMEYEN_TUTAR', 'NISAN_ODENMEYEN_TUTAR',
            'MAYIS_ODENMEYEN_TUTAR', 'HAZIRAN_ODENMEYEN_TUTAR', 'TEMMUZ_ODENMEYEN_TUTAR', 'AGUSTOS_ODENMEYEN_TUTAR',
            'EYLUL_ODENMEYEN_TUTAR', 'EKIM_ODENMEYEN_TUTAR', 'KASIM_ODENMEYEN_TUTAR', 'ARALIK_ODENMEYEN_TUTAR'
        ]

        paid_amount_ratio_columns = [
            'OCAK_ODENEN_TUTAR_ORANI', 'SUBAT_ODENEN_TUTAR_ORANI', 'MART_ODENEN_TUTAR_ORANI', 'NISAN_ODENEN_TUTAR_ORANI',
            'MAYIS_ODENEN_TUTAR_ORANI', 'HAZIRAN_ODENEN_TUTAR_ORANI', 'TEMMUZ_ODENEN_TUTAR_ORANI', 'AGUSTOS_ODENEN_TUTAR_ORANI',
            'EYLUL_ODENEN_TUTAR_ORANI', 'EKIM_ODENEN_TUTAR_ORANI', 'KASIM_ODENEN_TUTAR_ORANI', 'ARALIK_ODENEN_TUTAR_ORANI'
        ]

        # Reduce dimensions of 4 highly correlated feature groups
        for feature_group_name, feature_group, n_components in zip(
                ['payment_amount', 'paid_amount', 'unpaid_amount', 'paid_amount_ratio'],
                [payment_amount_columns, paid_amount_columns, unpaid_amount_columns, paid_amount_ratio_columns],
                [2, 3, 3, 3]
        ):

            df_feature_group = pd.concat((
                self.df_train[feature_group],
                self.df_test[feature_group]
            ), axis=0, ignore_index=True)

            pca = PCA(n_components=n_components, random_state=self.random_state)
            transformed_feature_group = pca.fit_transform(df_feature_group)
            for component in range(n_components):
                self.df_train[f'{feature_group_name.upper()}_PCA_COMPONENT_{component}'] = transformed_feature_group[:len(self.df_train), component]
                self.df_test[f'{feature_group_name.upper()}_PCA_COMPONENT_{component}'] = transformed_feature_group[len(self.df_train):, component]

            explained_variance = np.sum(pca.explained_variance_ratio_) * 100
            reconstruction_error = mean_squared_error(df_feature_group, pca.inverse_transform(transformed_feature_group))
            print(f'PCA {feature_group_name} Explained Variance: {explained_variance:.4f}% Reconstruction Error: {reconstruction_error:.4f}')

    def create_cluster_features(self):

        """
        Create clusters and features
        """

        policy_features = [
            'OFFICE_ID', 'SIGORTA_TIP', 'SOZLESME_KOKENI',
            'SOZLESME_KOKENI_DETAY', 'KAPSAM_TIPI', 'KAPSAM_GRUBU'
        ]
        customer_features = [
            'CINSIYET', 'MEMLEKET',
            'MUSTERI_SEGMENTI', 'YATIRIM_KARAKTERI', 'MEDENI_HAL',
            'EGITIM_DURUM', 'GELIR', 'COCUK_SAYISI'
        ]

        # Create cluster features from two feature groups
        for feature_group_name, feature_group, n_clusters in zip(
                ['policy_features', 'customer_features'],
                [policy_features, customer_features],
                [16, 16]
        ):

            df_feature_group = pd.concat((
                self.df_train[feature_group],
                self.df_test[feature_group]
            ), axis=0, ignore_index=True).fillna(-1)

            kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state)
            kmeans.fit(df_feature_group)
            self.df_train[f'{feature_group_name.upper()}_CLUSTER'] = kmeans.labels_[:len(self.df_train)]
            self.df_test[f'{feature_group_name.upper()}_CLUSTER'] = kmeans.labels_[len(self.df_train):]

    def encode_frequency(self):

        """
        Create frequency encoded categorical features
        """

        df_all = pd.concat((self.df_train, self.df_test), axis=0, ignore_index=True)
        categorical_features = [
            'POLICE_SEHIR',
        ]

        # Frequency encode categorical features with lots categories
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
            'ARALIK_ODENEN_TUTAR', 'ARALIK_ODENMEYEN_TUTAR', 'ARALIK_ODENEN_TUTAR_ORANI',
        ]
        categorical_features = [
            'OFFICE_ID', 'MUSTERI_SEGMENTI', 'EGITIM_DURUM', 'MEDENI_HAL', 'YATIRIM_KARAKTERI',
            'POLICE_SEHIR'
        ]

        # Create statistical features from groups of categories
        for categorical_feature in categorical_features:
            for continuous_feature in continuous_features:
                for aggregation in ['mean', 'std', 'min', 'max']:
                    df_agg = df_all.groupby(categorical_feature)[continuous_feature].agg(aggregation)
                    self.df_train[f'{categorical_feature}_{continuous_feature}_{aggregation.upper()}'] = self.df_train[categorical_feature].map(df_agg)
                    self.df_test[f'{categorical_feature}_{continuous_feature}_{aggregation.upper()}'] = self.df_test[categorical_feature].map(df_agg)

    def encode_target(self):

        """
        Create target encoded categorical features
        """

        categorical_features = [
            'OFFICE_ID', 'MUSTERI_SEGMENTI', 'EGITIM_DURUM', 'MEDENI_HAL', 'YATIRIM_KARAKTERI',
            'POLICE_SEHIR'
        ]

        for categorical_feature in categorical_features:
            self.df_test[f'{categorical_feature}_TARGET_MEAN'] = 0

        # Create target mean in categories with cross-validation
        for fold in sorted(self.df_train['fold'].unique()):

            val_idx = self.df_train.loc[self.df_train['fold'] == fold].index
            val = self.df_train.loc[val_idx, :]

            for categorical_feature in categorical_features:
                target_means = val.groupby(categorical_feature)['ARTIS_DURUMU'].mean().to_dict()
                self.df_train.loc[val_idx, f'{categorical_feature}_TARGET_MEAN'] = self.df_train.loc[val_idx, categorical_feature].map(target_means)
                self.df_test[f'{categorical_feature}_TARGET_MEAN'] += (self.df_test.loc[:, categorical_feature].map(target_means) / self.df_train['fold'].nunique())

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
        self.fill_missing_values()
        self.create_categorical_feature_combinations()
        self.encode_categoricals()
        self.create_datetime_features()
        self.transform_continuous_features()
        self.create_payment_features()
        self.reduce_dimensions()
        self.create_cluster_features()
        self.encode_frequency()
        self.create_aggregation_features()
        self.encode_target()

        return self.df_train, self.df_test
