import numpy as np
from sklearn.utils import check_random_state
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
import pandas as pd


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate_X(n_samples, seed):
    """Generate 2-dimensional Gaussian data with a fixed mean and covariance
    matrix.

    Parameters
    ----------
    n_samples: int
        Number of samples.

    seed: int
         A random seed for reproducible results.

    Returns
    -------
    T: array-like of shape (n_samples, 2)
    """
    rng = check_random_state(seed)
    mean = np.array([1, 2])
    cov = np.array([[1.5, 1], [1, 1]])
    X = rng.multivariate_normal(
                    mean=mean, cov=cov,
                    size=n_samples, check_valid='raise'
                    )
    return X


def generate_MCAR_missingness(X, missing_rate, seed):
    """Generate missing values following a MCAR mechanism in the first variable
    of X only.

    Parameters
    ----------
    X: array-like of shape (n_samples, n_features)
        The complete data.

    missing_rate: float between 0 and 1.
        The desired percentage of missing values.

    seed: int
         A random seed for reproducible results.

    Returns
    -------
    T: array-like of the same shape as X.
        The data X with MCAR missing values.
    """
    rng = check_random_state(seed)
    M = np.zeros(X.shape)
    ber = rng.rand(X.shape[0])
    M[:, 0] = ber < missing_rate
    X = X.copy()
    np.putmask(X, M, np.nan)
    return X


def generate_MAR_missingness(X, seed):
    """Generate missing values following a MAR mechanism in the first variable
    only. The probability that the first variable is missing is a logistic
    function of the second variable.

    Parameters
    ----------
    X: array-like of shape (n_samples, n_features)
        The complete data.

    seed: int
         A random seed for reproducible results.

    Returns
    -------
    T: array-like of the same shape as X.
        The data X with MAR missing values.
    """
    rng = check_random_state(seed)
    M = np.zeros(X.shape)
    ps = sigmoid(X[:, 1]-2)
    M[:, 0] = rng.binomial(n=1, p=ps)
    X = X.copy()
    np.putmask(X, M, np.nan)
    return X


def generate_MNAR_missingness(X, seed):
    """Generate missing values following a MNAR mechanism in the first variable
    only. The probability that the first variable is missing is a logistic
    function of its own values.

    Parameters
    ----------
    X: array-like of shape (n_samples, n_features)
        The complete data.

    seed: int
         A random seed for reproducible results.

    Returns
    -------
    T: array-like of the same shape as X.
        The data X with MNAR missing values.
    """
    rng = check_random_state(seed)
    M = np.zeros(X.shape)
    ps = sigmoid(X[:, 0]-1)
    M[:, 0] = rng.binomial(n=1, p=ps)
    X = X.copy()
    np.putmask(X, M, np.nan)
    return X


def plot_x1(x, m):
    """Plot the complete data for x, with different colours for complete
        or missing values.

    Parameters
    ----------
    x: array-like of shape (n_samples, )
        The values of the variable
    m: array-like of shape (n-samples, )
        A binary vector indicating which observations are missing
        (1 means missing, 0 means observed)
    """
    fig, ax = plt.subplots(figsize=(2.5, 3))
    jitter = np.random.uniform(0, 1, size=len(x))
    ax.scatter(jitter[m == 1], x[m == 1], alpha=0.3, c='b', label='yes')
    ax.scatter(jitter[m == 0], x[m == 0], alpha=1, c='b', label='no')

    plt.xlim([-1, 6])
    plt.xticks([])
    ax.spines[['right', 'top', 'bottom']].set_visible(False)
    plt.legend(title='missingness', loc='center right')
    plt.title('values of x')
    plt.show()


def plot_X(X, m1):
    """Scatter plot of the two-dimensional complete data, with different
       colours for complete or missing values in X1.

    Parameters
    ----------
    X: array-like of shape (n_samples, 2)
        The complete data.
    m1: array-like of shape (n-samples, )
        A binary vector indicating which observations are missing in X1
        (1 means missing, 0 means observed).
    """
    plt.scatter(X[m1 == 1, 0], X[m1 == 1, 1], alpha=0.3, c='b', label='yes')
    plt.scatter(X[m1 == 0, 0], X[m1 == 0, 1], alpha=1, c='b', label='no')
    plt.legend(title='missingness')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


def complete_case_estimator(x):
    """Returns the mean of a vector with missing values, using complete-case
    analysis.

    Parameters
    ----------
    x: array-like of shape (n_samples, )
        A vector with missing values.
    """
    # ------------TO DO ------------
    mean_cc = np.nanmean(x)
    return mean_cc

    # Other option
    # ind = np.isnan(x)
    # return np.mean(x[~ind])


def single_imputation_estimator(x):
    """Returns the mean of a vector with missing values, using single
    imputation by the mean for inference.

    Parameters
    ----------
    x: array-like of shape (n_samples, )
        A vector with missing values.
    """
    # ------------TO DO ------------
    # Hint: use skearn's SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    x = x[:, None]
    imputer.fit(x)
    x_hat = imputer.transform(x)
    return np.mean(x_hat)


def ipw_estimator(X):
    """Returns the mean of a vector with missing values, using inverse
    probability weighting.

    Parameters
    ----------
    X: array-like of shape (n_samples, 2)
        Data with missing values only in the first column.
    """
    # ------------TO DO ------------
    # Hint: use skearn's LogisticRegression to learn te probability


def aipw_estimator(X):
    """Returns the mean of a vector with missing values, using inverse
    probability weighting.

    Parameters
    ----------
    X: array-like of shape (n_samples, 2)
        Data with missing values only in the first column.
    """
    # ------------TO DO ------------
    # Hint: use skearn's LogisticRegression to learn te probability
    # and skearn's LinearRegression to predict outcomes.


if __name__ == '__main__':
    # Define some constants
    # analysis will be repeated n_datasets times to capture sampling variance.
    n_datasets = 30
    n_samples = 1000
    missing_rate = 0.5
    methods = ['cc', 'si', 'ipw', 'aipw']

    # Plot the values of the first variable in X.
    X_complete = generate_X(n_samples=100, seed=0)
    X = generate_MCAR_missingness(X_complete, missing_rate, 0)
    x1 = X_complete[:, 0]
    m1 = np.isnan(X[:, 0])
    plot_x1(x1, m1)
    # Plot the 2-dimensional data
    plot_X(X_complete, m1)

    # Perform inference using the various methods defined above on many
    # simulated datasets with different missing data mechanisms, and plot the
    # results.
    perfs = []
    for i in range(20):
        for mdm in ['MCAR', 'MAR', 'MNAR']:
            for method in methods:
                # Generate data
                X_complete = generate_X(n_samples, i)

                # Generate missingness
                if mdm == 'MCAR':
                    X = generate_MCAR_missingness(X_complete, missing_rate, i)
                elif mdm == 'MAR':
                    X = generate_MAR_missingness(X_complete, i)
                elif mdm == 'MNAR':
                    X = generate_MNAR_missingness(X_complete, i)
                x = X[:, 0]

                # Inference
                if method == 'cc':
                    mean = complete_case_estimator(x)
                elif method == 'si':
                    mean = single_imputation_estimator(x)
                elif method == 'ipw':
                    mean = ipw_estimator(X)
                elif method == 'aipw':
                    mean = aipw_estimator(X)
                perf_item = {'iter': i, 'mdm': mdm, 'method': method,
                             'mean': mean}
                perfs.append(perf_item)

    df = pd.DataFrame(perfs)
    plt.axhline(y=1, color='r', linestyle='-', zorder=0)
    sns.boxplot(data=df, y='mean', x='method', hue='mdm')
    plt.show()
