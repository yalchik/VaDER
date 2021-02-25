import numpy as np
from typing import Dict, Tuple

# Type aliases
XTensorDict = Dict[str, Dict[str, np.ndarray]]


def generate_wtensor_from_xtensor(x_tensor: np.ndarray) -> np.ndarray:
    """
    Generates W tensor, which has the same shape as X tensor, but defines if the corresponding value in X tensor is
    missing.

    Parameters
    ----------
    x_tensor : np.ndarray
        3D numpy array, where 1st dimension is samples, 2nd dimension is time points, 3rd dimension is feature vectors.

    Returns
    -------
    W tensor
    """
    w = ~np.isnan(x_tensor)
    return w.astype(int)


def map_xdict_to_xtensor(x_dict: XTensorDict) -> np.ndarray:
    # noinspection PyTypeChecker
    return np.array([list(val.values()) for val in x_dict.values()]).transpose(2, 1, 0)


def generate_x_w_y(num_of_time_points: int = 7, num_of_samples: int = 400) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates some simple random data [ns * 2 samples, nt - 1 time points, 2 variables]

    Parameters
    ----------
    num_of_time_points : int
        number of time points (default is 7)
    num_of_samples : int
        number of samples (default is 400)

    Returns
    -------
    tuple of 3 elements: X tensor, W tensor and Y vector.
    """
    nt = num_of_time_points + 1
    ns = num_of_samples // 2
    sigma = 0.5
    mu1 = -2
    mu2 = 2
    a1 = np.random.normal(mu1, sigma, ns)
    a2 = np.random.normal(mu2, sigma, ns)
    # one variable with two clusters
    X0 = np.outer(a1, 2 * np.append(np.arange(-nt / 2, 0), 0.5 * np.arange(0, nt / 2)[1:]))
    X1 = np.outer(a2, 0.5 * np.append(np.arange(-nt / 2, 0), 2 * np.arange(0, nt / 2)[1:]))
    X_train = np.concatenate((X0, X1), axis=0)
    y_train = np.repeat([0, 1], ns)
    # add another variable as a random permutation of the first one
    # resulting in four clusters in total
    ii = np.random.permutation(ns * 2)
    X_train = np.stack((X_train, X_train[ii, ]), axis=2)
    # we now get four clusters in total
    y_train = y_train * 2 ** 0 + y_train[ii] * 2 ** 1

    # normalize (better for fitting)
    for i in np.arange(X_train.shape[2]):
        X_train[:, :, i] = (X_train[:, :, i] - np.mean(X_train[:, :, i])) / np.std(X_train[:, :, i])

    # randomly re-order the samples
    ii = np.random.permutation(ns * 2)
    X_train = X_train[ii, :]
    y_train = y_train[ii]
    # Randomly set 50% of values to missing (0: missing, 1: present)
    # Note: All X_train[i,j] for which W_train[i,j] == 0 are treated as missing (i.e. their specific value is ignored)
    W_train = np.random.choice(2, X_train.shape)
    return X_train, W_train, y_train


def generate_x_y_for_nonrecur(num_of_time_points: int = 7, num_of_samples: int = 400) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates some simple random data for non-recurrent VaDER (ordinary VAE with GM prior)

    Parameters
    ----------
    num_of_time_points : int
        number of time points (default is 7)
    num_of_samples : int
        number of samples (default is 400)

    Returns
    -------
    tuple of 2 elements: X tensor and Y vector.
    """
    nt = num_of_time_points
    ns = num_of_samples // 2
    sigma = np.diag(np.repeat(2, nt))
    mu1 = np.repeat(-1, nt)
    mu2 = np.repeat(1, nt)
    a1 = np.random.multivariate_normal(mu1, sigma, ns)
    a2 = np.random.multivariate_normal(mu2, sigma, ns)
    X_train = np.concatenate((a1, a2), axis=0)
    y_train = np.repeat([0, 1], ns)
    ii = np.random.permutation(ns * 2)
    X_train = X_train[ii, :]
    y_train = y_train[ii]
    # normalize (better for fitting)
    X_train = (X_train - np.mean(X_train)) / np.std(X_train)
    return X_train, y_train


if __name__ == "__main__":
    pass
