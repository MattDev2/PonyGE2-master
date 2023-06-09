import warnings

import numpy as np
from sklearn.metrics import f1_score as sklearn_f1_score
from sklearn.metrics import accuracy_score



def mae(y, yhat):
    """
    Calculate mean absolute error between inputs.

    :param y: The expected input (i.e. from dataset).
    :param yhat: The given input (i.e. from phenotype).
    :return: The mean absolute error.
    """

    return np.mean(np.abs(y - yhat))


# Set maximise attribute for mae error metric.
mae.maximise = False


def rmse(y, yhat):
    """
    Calculate root mean square error between inputs.

    :param y: The expected input (i.e. from dataset).
    :param yhat: The given input (i.e. from phenotype).
    :return: The root mean square error.
    """

    return np.sqrt(np.mean(np.square(y - yhat)))


# Set maximise attribute for rmse error metric.
rmse.maximise = False


def mse(y, yhat):
    """
    Calculate mean square error between inputs.

    :param y: The expected input (i.e. from dataset).
    :param yhat: The given input (i.e. from phenotype).
    :return: The mean square error.
    """

    return np.mean(np.square(y - yhat))


# Set maximise attribute for mse error metric.
mse.maximise = False


def hinge(y, yhat):
    """
    Hinge loss is a suitable loss function for classification.  Here y is
    the true values (-1 and 1) and yhat is the "raw" output of the individual,
    ie a real value. The classifier will use sign(yhat) as its prediction.

    :param y: The expected input (i.e. from dataset).
    :param yhat: The given input (i.e. from phenotype).
    :return: The hinge loss.
    """

    # Deal with possibility of {-1, 1} or {0, 1} class label convention
    y_vals = set(y)
    # convert from {0, 1} to {-1, 1}
    if 0 in y_vals:
        y[y == 0] = -1

    # Our definition of hinge loss cannot be used for multi-class
    assert len(y_vals) == 2

    # NB not np.max. maximum does element-wise max.  Also we use the
    # mean hinge loss rather than sum so that the result doesn't
    # depend on the size of the dataset.
    return np.mean(np.maximum(0, 1 - y * yhat))


# Set maximise attribute for hinge error metric.
hinge.maximise = False


def f1_score(y, yhat):
    """
    The F_1 score is a metric for classification which tries to balance
    precision and recall, ie both true positives and true negatives.
    For F_1 score higher is better.

    :param y: The expected input (i.e. from dataset).
    :param yhat: The given input (i.e. from phenotype).
    :return: The f1 score.
    """

    # if phen is a constant, eg 0.001 (doesn't refer to x), then yhat
    # will be a constant. that will break f1_score. so convert to a
    # constant array.
    if not isinstance(yhat, np.ndarray) or len(yhat.shape) < 1:
        yhat = np.ones_like(y) * yhat

    # Deal with possibility of {-1, 1} or {0, 1} class label
    # convention.  FIXME: would it be better to canonicalise the
    # convention elsewhere and/or create user parameter to control it?
    # See https://github.com/PonyGE/PonyGE2/issues/113.
    y_vals = set(y)
    # convert from {-1, 1} to {0, 1}
    if -1 in y_vals:
        y[y == -1] = 0

    # We binarize with a threshold, so this cannot be used for multi-class
    assert len(y_vals) == 2

    # convert real values to boolean {0, 1} with a zero threshold
    yhat = (yhat > 0)

    with warnings.catch_warnings():
        # if we predict the same value for all samples (trivial
        # individuals will do so as described above) then f-score is
        # undefined, and sklearn will give a runtime warning and
        # return 0. We can ignore that warning and happily return 0.
        warnings.simplefilter("ignore")
        return sklearn_f1_score(y, yhat, average="weighted")


# Set maximise attribute for f1_score error metric.
f1_score.maximise = True


def Hamming_error(y, yhat):
    """
    The number of mismatches between y and yhat. Suitable
    for Boolean problems and if-else classifier problems.
    Assumes both y and yhat are binary or integer-valued.
    """
    return np.sum(y != yhat)


Hamming_error.maximise = False

# import numpy as np
# from sklearn.metrics import accuracy_score

# def confusion_matrix_elements(y, yhat):
#     TP = np.sum((y == 1) & (yhat == 1))
#     TN = np.sum((y == 0) & (yhat == 0))
#     FP = np.sum((y == 0) & (yhat == 1))
#     FN = np.sum((y == 1) & (yhat == 0))
#     return TP, TN, FP, FN

# def accuracy_percentage(y, yhat):
#     if not isinstance(yhat, np.ndarray) or len(yhat.shape) < 1:
#         yhat = np.ones_like(y) * yhat

#     y_vals = set(y)
#     if -1 in y_vals:
#         y[y == -1] = 0

#     assert len(y_vals) == 1 or len(y_vals) == 2

#     yhat = (yhat > 0)
#     TP, TN, FP, FN = confusion_matrix_elements(y, yhat)
    
#     return accuracy_score(y, yhat) * 100#, FP, FN

# accuracy_percentage.maximise = True



def accuracy_percentage(y, yhat):
    """
    The accuracy is a metric for classification which calculates the percentage of 
    correctly classified instances out of the total instances.
    For accuracy, higher is better.

    :param y: The expected input (i.e. from dataset).
    :param yhat: The given input (i.e. from phenotype).
    :return: The accuracy percentage.
    """

    # if phen is a constant, eg 0.001 (doesn't refer to x), then yhat
    # will be a constant. that will break accuracy. so convert to a
    # constant array.
    if not isinstance(yhat, np.ndarray) or len(yhat.shape) < 1:
        yhat = np.ones_like(y) * yhat

    # Deal with possibility of {-1, 1} or {0, 1} class label
    # convention. 
    y_vals = set(y)
    # convert from {-1, 1} to {0, 1}
    if -1 in y_vals:
        y[y == -1] = 0

    # We binarize with a threshold, so this cannot be used for multi-class
    assert len(y_vals) == 1 or len(y_vals) == 2

    # convert real values to boolean {0, 1} with a zero threshold
    yhat = (yhat > 0)

    return accuracy_score(y, yhat) * 100

accuracy_percentage.maximise = True