"""This module contains the function to perform principal components analysis."""
import numpy as np


def pca(data):
    """
    Perform principal components analysis to reduce dimensionality of data.

    :param data: d x n matrix of n d-dimensional data points. Each column is an example.
    :type data: ndarray
    :return: tuple containing three components: (new_data, variances, eigenvectors). The variable new_data is a d x n
    matrix containing the original data mapped to a new coordinate space. The variable variances is a length-d vector
    containing the variance captured by each new dimensions. The variable eigenvectors is a matrix where each column
    is one of the eigenvectors that the data has been projected onto.
    :rtype: tuple
    """
    #####################################################################
    # Enter your code below for computing new_data and variances.
    # You may use built in np.linalg.eig or np.linalg.svd, but you are
    # not allowed to use a pre-built pca in your implementation
    #####################################################################

    d, n = data.shape

    # center data
    centered_data = data - np.mean(data, 1, keepdims=True)

    # Recipe 1
    covariance = centered_data.dot(centered_data.T)
    variances, v = np.linalg.eig(covariance)
    order = np.argsort(variances)[::-1]
    variances = variances[order]
    v = v[:, order]
    eigenvectors = v

    # # Recipe 2
    # u, s, v = np.linalg.svd(centered_data.T)
    #
    # variances = s ** 2
    #
    # eigenvectors = v.T

    # project onto new basis

    new_data = eigenvectors.T.dot(centered_data)

    #####################################################################
    # End of your contributed code
    #####################################################################

    return np.real(new_data), np.real(variances), np.real(eigenvectors)
