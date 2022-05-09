import numpy as np
import matplotlib.pyplot as plt


def visualize_feature_importance(df_feature_importance, path=None):

    """
    Visualize feature importance of multiple tree-based models with error bars

    Parameters
    ----------
    df_feature_importance (pandas.DataFrame of shape (n_features, n_folds)): DataFrame of features as index and importance as values for every fold
    path (str or None): Path of the output file (if path is None, plot is displayed with selected backend)
    """

    # Create mean and std of feature importance for error bars
    column_names = df_feature_importance.columns.to_list()
    df_feature_importance['mean_importance'] = df_feature_importance[column_names].mean(axis=1)
    df_feature_importance['std_importance'] = df_feature_importance[column_names].std(axis=1)
    df_feature_importance.sort_values(by='mean_importance', inplace=True, ascending=True)

    fig, ax = plt.subplots(figsize=(24, len(df_feature_importance)))
    ax.barh(
        y=df_feature_importance.index,
        width=df_feature_importance['mean_importance'],
        xerr=df_feature_importance['std_importance'],
        align='center',
        ecolor='black',
        capsize=10
    )
    ax.set_xlabel('')
    ax.tick_params(axis='x', labelsize=12.5, pad=10)
    ax.tick_params(axis='y', labelsize=12.5, pad=10)
    ax.set_title('Feature Importance (Gain)', size=20, pad=15)

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
        plt.close(fig)


def visualize_scores(df_scores, path=None):

    """
    Visualize binary classification scores of multiple models with error bars

    Parameters
    ----------
    df_scores (pandas.DataFrame of shape (n_folds, 6)): DataFrame of binary classification scores
    path (str): Path of the output file
    """

    # Create mean and std of scores for error bars
    df_scores = df_scores.T
    column_names = df_scores.columns.to_list()
    df_scores['mean'] = df_scores[column_names].mean(axis=1)
    df_scores['std'] = df_scores[column_names].std(axis=1)

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.barh(
        y=np.arange(df_scores.shape[0]),
        width=df_scores['mean'],
        xerr=df_scores['std'],
        align='center',
        ecolor='black',
        capsize=10
    )
    ax.set_yticks(np.arange(df_scores.shape[0]))
    ax.set_yticklabels(df_scores.index)
    ax.set_xlabel('')
    ax.tick_params(axis='x', labelsize=12.5, pad=10)
    ax.tick_params(axis='y', labelsize=12.5, pad=10)
    ax.set_title('Classification Scores', size=20, pad=15)

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
        plt.close(fig)
