import numpy as np
from sklearn.preprocessing import StandardScaler


def split_train_test(data, train_size=0.75):
    """
    Splits the data into train and test sets.

    Args:
        data (pandas.DataFrame): The data to split.
        train_size (float): The proportion of the data to use for training.

    Returns:
        tuple: Two pandas.DataFrames, the train and test sets.
    """

    train_size = int(len(data) * train_size)
    train = data[:train_size]
    test = data[train_size:]

    return train, test


def scale_data(train, test=None):
    """
    Scales the data using the Standard Scaler.

    Args:
        train (pandas.DataFrame): The training data to scale.
        test (pandas.DataFrame): The testing data to scale. Optional.

    Returns:
        tuple: The scaler used for scaling and the scaled training and testing data.
    """
    scaler = StandardScaler()
    scaler.fit(train)

    train_scaled = scaler.transform(train)

    if test is not None:
        test_scaled = scaler.transform(test)
        return scaler, train_scaled, test_scaled

    return scaler, train_scaled


def prepare_data(data, n_steps_in, n_steps_out):
    """
    Prepares the data for an LSTM model.

    Args:
        data (numpy.ndarray): The data to prepare.
        n_steps_in (int): The number of input time steps.
        n_steps_out (int): The number of output time steps.

    Returns:
        tuple: The input and output data arrays.
    """
    X, y = [], []
    for i in range(len(data)):
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out - 1
        if out_end_ix > len(data):
            break

        seq_x, seq_y = data[i:end_ix, :], data[end_ix - 1 : out_end_ix, -1]
        X.append(seq_x)
        y.append(seq_y)

    return np.array(X), np.array(y)
