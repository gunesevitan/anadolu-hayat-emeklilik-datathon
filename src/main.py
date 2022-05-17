import argparse
import yaml
import pandas as pd

import settings
import preprocessing
from lgb_trainer import LightGBMTrainer
from xgb_trainer import XGBoostTrainer


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str)
    args = parser.parse_args()

    config = yaml.load(open(args.config_path, 'r'), Loader=yaml.FullLoader)
    print(f'{"-" * 30}\nAnadolu Hayat Emeklilik Datathon\n{"-" * 30}\n')
    df_train = pd.read_csv(settings.DATA / 'train.csv')
    df_test = pd.read_csv(settings.DATA / 'test.csv', low_memory=False)

    preprocessor = preprocessing.TabularPreprocessor(
        df_train=df_train,
        df_test=df_test,
        fill_missing=config['preprocessing_parameters']['fill_missing']
    )
    df_train, df_test = preprocessor.transform()

    print(f'\nTraining Set Shape: {df_train.shape} - Memory Usage: {df_train.memory_usage().sum() / 1024 ** 2:.2f} MB')
    print(f'Test Set Shape: {df_test.shape} - Memory Usage: {df_test.memory_usage().sum() / 1024 ** 2:.2f} MB')

    if config['model'] == 'lightgbm':

        trainer = LightGBMTrainer(
            features=config['features'],
            target=config['target'],
            model_parameters=config['model_parameters'],
            fit_parameters=config['fit_parameters'],
            categorical_features=config['categorical_features'],
            sampler_class=config['sampler_class'],
            sampler_parameters=config['sampler_parameters']
        )

    elif config['model'] == 'xgboost':

        trainer = XGBoostTrainer(
            features=config['features'],
            target=config['target'],
            model_parameters=config['model_parameters'],
            fit_parameters=config['fit_parameters'],
            sampler_class=config['sampler_class'],
            sampler_parameters=config['sampler_parameters']
        )

    else:
        trainer = None

    if trainer is not None:
        trainer.train_and_validate(df_train, df_test)
