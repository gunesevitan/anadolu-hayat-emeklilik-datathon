import numpy as np
from sklearn.metrics import auc
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
    path (str or None): Path of the output file (if path is None, plot is displayed with selected backend)
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


def visualize_roc_curve(roc_curves, path=None):

    """
    Visualize ROC curve of multiple models with confidence intervals

    Parameters
    ----------
    roc_curves (np.ndarray of shape (n_models, 3, n_thresholds)): Array of ROC curves of multiple models
    path (str or None): Path of the output file (if path is None, plot is displayed with selected backend)
    """

    false_positive_rates = roc_curves[:, 0]
    true_positive_rates = roc_curves[:, 1]
    true_positive_rates_interpolated = []
    aucs = []
    mean_false_positive_rate = np.linspace(0, 1, 100)

    fig, ax = plt.subplots(figsize=(16, 16))

    # Plot random guess curve
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2.5, color='r', alpha=0.8)
    # Plot individual ROC curves of multiple models
    for i, (false_positive_rate, true_positive_rate) in enumerate(zip(false_positive_rates, true_positive_rates), 1):
        true_positive_rates_interpolated.append(np.interp(mean_false_positive_rate, false_positive_rate, true_positive_rate))
        true_positive_rates_interpolated[-1][0] = 0.0
        roc_auc = auc(false_positive_rate, true_positive_rate)
        aucs.append(roc_auc)
        ax.plot(false_positive_rate, true_positive_rate, lw=1, alpha=0.1)

    # Plot mean ROC curve of N models
    mean_tpr = np.mean(true_positive_rates_interpolated, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_false_positive_rate, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_false_positive_rate, mean_tpr, color='b', label=f'Mean ROC Curve (AUC: {mean_auc:.4f} ±{std_auc:.4f})', lw=2.5, alpha=0.9)

    # Plot confidence interval of ROC curves
    std_tpr = np.std(true_positive_rates_interpolated, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_false_positive_rate, tprs_lower, tprs_upper, color='grey', alpha=0.2, label='±1 sigma')

    ax.set_xlabel('False Positive Rate', size=15, labelpad=12)
    ax.set_ylabel('True Positive Rate', size=15, labelpad=12)
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_title('ROC Curve', size=20, pad=15)
    ax.legend(loc='lower right', prop={'size': 14})

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
        plt.close(fig)
