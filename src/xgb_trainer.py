import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
import xgboost as xgb
import imblearn.over_sampling

import settings
import metrics
import visualization


class XGBoostTrainer:

    def __init__(self, features, target, model_parameters, fit_parameters, sampler_class, sampler_parameters):

        self.features = features
        self.target = target
        self.model_parameters = model_parameters
        self.fit_parameters = fit_parameters
        self.sampler_class = sampler_class
        self.sampler_parameters = sampler_parameters

    def train_and_validate(self, df_train, df_test):

        """
        Train and validate on given dataframe with specified configuration

        Parameters
        ----------
        df_train (pandas.DataFrame of shape (634112, n_columns)): Training dataframe of features, target and folds
        df_test (pandas.DataFrame of shape (243137, n_columns)): Test dataframe of features
        """

        print(f'{"-" * 30}\nRunning XGBoost Model for Training\n{"-" * 30}\n')
        scores = []
        roc_curves = []
        df_test['xgb_predictions'] = 0
        df_feature_importance = pd.DataFrame(
            data=np.zeros((len(self.features), df_train['fold'].nunique())),
            index=self.features,
            columns=[f'fold_{fold}_importance' for fold in range(1, df_train['fold'].nunique() + 1)]
        )

        for fold in range(1, df_train['fold'].nunique() + 1):

            # Get training and validation sets
            trn_idx, val_idx = df_train.loc[df_train['fold'] != fold].index, df_train.loc[df_train['fold'] == fold].index
            print(f'Fold {fold} - Training: {df_train.loc[trn_idx, self.features].shape} Validation: {df_train.loc[val_idx, self.features].shape} - Seed: {self.model_parameters["seed"]}')

            if self.sampler_class is not None:
                # Resample training set if sampler class is specified
                sampler = getattr(imblearn.over_sampling, self.sampler_class)(**self.sampler_parameters)
                trn_features_resampled, trn_labels_resampled = sampler.fit_resample(df_train.loc[trn_idx, self.features], df_train.loc[trn_idx, self.target])
                trn_dataset = xgb.DMatrix(df_train.loc[trn_idx, self.features].astype(np.float32), label=df_train.loc[trn_idx, self.target])
                print(f'Resampled Training Features: {trn_features_resampled.shape} Labels: {trn_labels_resampled.shape}')
            else:
                trn_dataset = xgb.DMatrix(df_train.loc[trn_idx, self.features].astype(np.float32), label=df_train.loc[trn_idx, self.target])
                
            val_dataset = xgb.DMatrix(df_train.loc[val_idx, self.features].astype(np.float32), label=df_train.loc[val_idx, self.target])

            # Set model parameters, train parameters, callbacks and start training
            model = xgb.train(
                params=self.model_parameters,
                dtrain=trn_dataset,
                evals=[(trn_dataset, 'train'), (val_dataset, 'val')],
                num_boost_round=self.fit_parameters['boosting_rounds'],
                early_stopping_rounds=self.fit_parameters['early_stopping_rounds'],
                verbose_eval=self.fit_parameters['verbose_eval']
            )
            # Save trained model
            model.save_model(settings.MODELS / 'xgboost' / f'model_fold{fold}_seed{self.model_parameters["seed"]}.txt')
            for feature, importance in model.get_score(importance_type='gain').items():
                df_feature_importance.loc[feature, f'fold_{fold}_importance'] = importance

            val_predictions = model.predict(xgb.DMatrix(df_train.loc[val_idx, self.features].astype(np.float32)))
            df_train.loc[val_idx, 'xgb_predictions'] = val_predictions
            val_scores = metrics.classification_scores(
                y_true=df_train.loc[val_idx, self.target],
                y_pred=df_train.loc[val_idx, 'xgb_predictions'],
                threshold=0.5
            )
            scores.append(val_scores)
            print(f'\nXGBoost Validation Scores: {val_scores}\n')
            val_roc_curve = roc_curve(y_true=df_train.loc[val_idx, self.target], y_score=df_train.loc[val_idx, 'xgb_predictions'])
            roc_curves.append(val_roc_curve)

            test_predictions = model.predict(xgb.DMatrix(df_test[self.features].astype(np.float32)))
            df_test['xgb_predictions'] += (test_predictions / df_train['fold'].nunique())

        # Display and visualize scores of validation sets
        df_scores = pd.DataFrame(scores)
        print('\n')
        for fold, scores in df_scores.iterrows():
            print(f'Fold {fold} - Validation Scores: {scores.to_dict()}')
        print(f'{"-" * 30}\nXGBoost Mean Validation Scores: {df_scores.mean(axis=0).to_dict()} (±{df_scores.std(axis=0).to_dict()})\n{"-" * 30}\n')
        visualization.visualize_scores(
            df_scores=df_scores,
            path=settings.MODELS / 'xgboost' / f'validation_scores_seed{self.model_parameters["seed"]}.png'
        )

        # Save out-of-fold and test set predictions
        df_train['xgb_predictions'].to_csv(settings.MODELS / 'xgboost' / f'train_predictions_seed{self.model_parameters["seed"]}.csv', index=False)
        df_test['xgb_predictions'].to_csv(settings.MODELS / 'xgboost' / f'test_predictions_seed{self.model_parameters["seed"]}.csv', index=False)

        # Visualize feature importance
        visualization.visualize_feature_importance(
            df_feature_importance=df_feature_importance,
            path=settings.MODELS / 'xgboost' / f'feature_importance_seed{self.model_parameters["seed"]}.png'
        )

        # Visualize ROC curves
        roc_curves = np.array(roc_curves, dtype=object)
        visualization.visualize_roc_curve(
            roc_curves=roc_curves,
            path=settings.MODELS / 'xgboost' / f'roc_curve_seed{self.model_parameters["seed"]}.png'
        )
