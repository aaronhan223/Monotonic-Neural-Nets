import numpy as np
import pandas as pd
import pdb


def generate_f1(n, lower, upper, sigma=0.1):
    epsilon = np.random.normal(loc=0.0, scale=sigma, size=n)
    true_fn = lambda x: .5 * np.log(x) + epsilon
    X = np.sort(np.random.uniform(1, 7.5, n))
    y = true_fn(X)
    return X, y


def generate_f2(n, lower, upper, sigma=0.3):
    X = np.linspace(lower, upper, num=n)
    epsilon = np.random.normal(loc=0.0, scale=sigma, size=X.shape)
    return X, 0.25 * (X ** 2 + epsilon)


def generate_f3(n, lower, upper, sigma=0.3):
    X = np.linspace(lower, upper, num=n)
    epsilon = np.random.normal(loc=0.0, scale=sigma, size=X.shape)
    return X, 0.15 * (X * (X - 1.5) * (X + 1.5) + epsilon) + 0.5


def generate_f4(n, lower, upper, sigma=0.3):
    X = np.linspace(lower, upper, num=n)
    epsilon = np.random.normal(loc=0.0, scale=sigma, size=X.shape)
    return X, 0.15 * (2 * np.sin(np.pi * X) + np.pi * X / 2 + epsilon) + 0.5


def load_data(n, lower, upper, fn):
    functions = [generate_f1, generate_f2, generate_f3, generate_f4]
    X, y = functions[fn](n, lower, upper)
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)  # random shuffle X and y
    X = X[idx]
    y = y[idx]

    out_df = pd.DataFrame({'X': X, 'y': y})
    n_train = int(out_df.shape[0] * 0.8)

    output = {
        'data_train': out_df[:n_train].values,
        'data_test': out_df.values,  # we want to see across for viz
        'monotonicity': [1],
        'X_cols': [0],
        'Y_col': 1
    }

    return output, n, lower, upper
