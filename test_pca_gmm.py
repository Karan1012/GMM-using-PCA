"""Test class for principal components analysis and Gaussian mixture modeling."""
import unittest
from scipy.io import loadmat
from pca import *
from gmm import *


class TestHomework4(unittest.TestCase):
    """Tests for PCA and GMMs"""
    def setUp(self):
        """
        Load synthetic data from MATLAB data file
        :return: None
        """

        variables = dict()
        loadmat('synthData.mat', variables)

        self.data = variables['data']

    def test_pca(self):
        """
        Perform PCA on the synthetic data and check that the returned values are as expected.

        :return: None
        """
        new_data, variances, eigenvectors = pca(self.data)

        assert np.allclose(np.zeros(64), np.mean(new_data, 1)), "The data is not centered to be zero-mean."

        assert variances[0] + variances[1] > np.sum(variances[2:]), "Variance of the first two dimensions should " \
                                                                    "be greater than the variance of the rest"

        assert np.sum(variances[:2]) > np.sum(variances[2:]), "Variances of first two dimensions were not larger than" \
                                                              "variances of the rest of the noise dimensions"

        assert np.var(eigenvectors[:, 0].T.dot(self.data)) > np.var(eigenvectors[:, 1].T.dot(self.data)), \
            "First principle direction doesn't have more variance than second principle direction"

        assert np.var(eigenvectors[:, 1].T.dot(self.data)) > np.var(eigenvectors[:, 2].T.dot(self.data)), \
            "Second principle direction doesn't have more variance than third principle direction"

        vector_0_1 = self.data[:, 1] - self.data[:, 0]
        new_vector_0_1 = new_data[:, 0] - new_data[:, 1]

        assert np.allclose(np.linalg.norm(vector_0_1), np.linalg.norm(new_vector_0_1)), "Distance between example 0 " \
                                                                                        "and 1 is not the same before" \
                                                                                        "and after PCA"

        assert np.allclose(eigenvectors.T.dot(eigenvectors), np.eye(64)), "Eigenvectors were not orthogonal"

        reconstructed = eigenvectors.dot(new_data)
        assert np.allclose(reconstructed[:, 0] - self.data[:, 0], reconstructed[:, 1] - self.data[:, 1]), \
            "Reconstructed points are not similar."

    def set_up_clustering(self):
        """
        Set up the clustering task by running PCA and splitting the data into training and testing sets.
        :return: None
        """
        new_data, variances, eigenvectors = pca(self.data)

        # truncate dimensions to just the first two
        small_data = new_data[:2, :]

        # if you haven't implemented PCA yet, you can test GMM by replacing the above code with
        # small_data = self.data[:2, :]

        # split data for validation
        d, n = small_data.shape

        # use fraction of data for training

        self.train_inds = np.random.rand(n) < 0.5

        self.train_data = small_data[:, self.train_inds]
        self.val_data = small_data[:, ~self.train_inds]

    def test_gmm(self):
        """
        Train various GMMs, testing that using more clusters fits the training data better, that the Gaussian-fitting
        code is consistent when using a single Gaussian, and that the GMM learns a better model than one with random
        parameters.
        :return: None
        """
        self.set_up_clustering()

        plot = None  # set this to 'iter' to watch the GMM optimize (it will be much slower)

        model_1 = gmm(self.train_data, 1, plot=plot)
        ll_1 = gmm_ll(self.train_data, model_1[0], model_1[1], model_1[2])

        model_2 = gmm(self.train_data, 2, plot=plot)
        ll_2 = gmm_ll(self.train_data, model_2[0], model_2[1], model_2[2])

        assert ll_2 > ll_1, "Likelihood of two-Gaussian model was not as good as likelihood of one Gaussian"

        model_1_again = gmm(self.train_data, 1, plot=plot)

        assert np.allclose(model_1[0], model_1_again[0]), "Single-Gaussian model returned different solutions on two " \
                                                          "trials."

        random_means = np.random.randn(2, 2)
        random_sigmas = []
        for i in range(2):
            vec = np.random.randn(2, 2)
            random_sigmas.append(vec.dot(vec.T))

        random_probs = np.random.rand(2)
        random_probs /= np.sum(random_probs)

        ll_random = gmm_ll(self.train_data, random_means, random_sigmas, random_probs)

        assert ll_2 > ll_random, "Likelihood of trained two-Gaussian model was not as good as with random parameters."


if __name__ == '__main__':
    unittest.main()
