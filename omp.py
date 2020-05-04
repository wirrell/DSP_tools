"""
Orthogonal Matching Pursuit variant for images.

Author: G. Worrall

Based on:

    Rubinstein et al. - Efficient Implementation of the K-SVD Algorithm using
        Batch Orthogonal Matching Pursuit (2008)
    Pati et al. - Orthogonal Matching Pursuit: Recursive Function Approximation
        with Applications to Wavelet Decomposition (1993)
"""
import math
import progressbar
from pathlib import Path
import numpy as np
import cv2
import numpy.linalg as LA
from sklearn.preprocessing import StandardScaler, normalize
np.seterr(all='raise')


class OMP:
    """
    Class for Orthogonal Matching Pursuit implementation.
    """

    def __init__(self, dictionary_length=1000, k=1):
        """init method

        Args:
            dictionary_length (int): number of elements in the the pursuit
            dictionary.
            k (int): sparsiting of coding. k atoms from dictionary used to
                describe an image patch during dictionary computation.

        """
        self.dict_len = dictionary_length
        self.k = k
        self.scaler = StandardScaler()
        self.history = []  # training history for dictionary learning
        # whitening array and dictionary
        self.W = None
        self.D = None

    def fit(self, train_images, iterations=100, return_history=False):
        """
        Method to learn dictionary of features from the training images.

        Args:
            train_images (np.array): array of training images.
            iterations (int): number of iterations to run for computing of
                dictionary
        """
        X = self._extract_features(train_images)  # pixel intensities
        # standardize the data
        X_z = self.scaler.fit_transform(X.T).T  # needs n samples x m features
        # whiten the data using ZCA
        X_white = self._whiten_array(X_z)
        # pursue dictionary
        if Path('train_dict_{}_{}_{}.npy'.format(self.dict_len,
                                                 self.k,
                                                 iterations)).exists():
            self.D = np.load('train_dict_{}_{}_{}.npy'.format(self.dict_len,
                                                              self.k,
                                                              iterations))
            return
        self._compute_dictionary(X_white, iterations)

        if return_history:
            np.save('omp_history_{}_{}_{}.npy'.format(self.dict_len,
                                                      self.k,
                                                      iterations),
                    np.array(self.history))
            return self.history

    def transform(self, images, pool=False, raw_S=False):
        """
        Tranform images to sparse coding representation.

        Args:
            images (np.array): array of images to be encoded.
            pool (bool, opt): average the Z vectors for each image.
            raw_S(bool, opt): if true, return raw S and not Z vectors
        Returns:
            Z_vectors (list): list of z vectors for each image. If pool,
                then returns one average vector per image.
        """
        # NOTE: this is coded for sparse coding of k = 1
        if len(images.shape) != 4:  # single image
            images = [images]
        S_vectors = []
        print('Encoding extracted features.')
        for image in progressbar.progressbar(images):
            X = self._extract_features(image)  # pixel intensities
            # standardize and whiten
            X_z = self.scaler.transform(X.T).T
            X_white = self._whiten_array(X_z)
            S = np.zeros((self.D.shape[1], X_white.shape[1]))
            # Calculate the coding for each feature
            G = self.D.T.dot(self.D)
            for i in range(X_white.shape[1]):
                x = X_white[:, i]
                S[:, i] = self._ompursuit(x, G, self.D, target_sparsity=self.k)
            S_vectors.append(S)

        if pool:
            for i in range(len(S_vectors)):
                vectors = S_vectors[i]
                S_vectors[i] = np.sum(vectors, axis=1) / vectors.shape[1]

        return S_vectors

    def _ompursuit(self, x, G, D, target_error=0, target_sparsity=False):
        # Compute OMP on a signal x from a dictionary D
        # G = D.T.dot(D)
        # NOTE: Implementation of Batch-OMP from Rubinstein et al.
        # If target sparsity provided, will stop after target_sparsity iters

        if not target_sparsity:
            target_sparsity = self.dict_len

        # Calculate error from args
        error_n = x.T.dot(x)
        alpha_0 = D.T.dot(x)

        # Init
        I = []
        L = np.ones((1, 1))
        gamma = np.zeros(D.shape[1])
        alpha = alpha_0
        delta_nm1 = 0
        n = 1

        while error_n > target_error:
            k = np.argmax(np.abs(alpha))  # line 5
            if n > 1:  # line 6
                w = LA.inv(L).dot(G[:, I][k])  # line 9
                # line 8
                try:
                    w_term = np.sqrt(1 - w.T.dot(w))
                except FloatingPointError:
                    # NOTE: because the initial dictionary is made up of
                    # randomly selected real signals from the training data,
                    # w will sometimes = 1, i.e. a perfect fit from the
                    # dictionary on the first try. If this happens, the above
                    # will raise a FloatingPointError and we return gamma as
                    # it is the perfect description with a single coding.
                    return gamma
                L = np.block([
                    [L,      np.zeros((L.shape[0], 1))],
                    [w.T,    w_term]
                ])
            I.append(k)  # Line 10
            # Line 11
            gamma[I] = LA.inv(L.dot(L.T)).dot(alpha_0[I])

            beta = G[:, I].dot(gamma[I])  # Line 12
            alpha = alpha_0 - beta  # Line 13
            if target_error:
                delta_n = gamma[I].T.dot(beta[I])  # Line 14
                error_n = error_n - delta_n + delta_nm1
                delta_nm1 = delta_n  # update for next iteration

            if n == target_sparsity:  # sparsity stopping
                return gamma

            n = n + 1

        return gamma

    def _extract_features(self, images):
        # Use moving window method to extract vectors of pixels.
        if len(images.shape) < 4:  # single image
            images = np.array([images])
        grey_images = np.zeros((images.shape[:-1]))

        for i in range(images.shape[0]):
            img = images[i]
            grey_images[i] = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        x_dim = images[0].shape[0]
        y_dim = images[0].shape[1]

        num_x = math.trunc(x_dim / 8) - 1  # how many 16 x 16 patches in x dim
        num_y = math.trunc(y_dim / 8) - 1  # and in y dim

        num_images = images.shape[0]

        features = np.ndarray((256, num_images * num_x * num_y))

        i = 0
        for x in range(num_x):
            for y in range(num_x):
                # moving windows of size 16
                features_x_y = grey_images[:, x * 8: (x+2) * 8,
                                           y * 8: (y+2) * 8]

                feature_vectors = features_x_y.reshape(num_images,
                                                       features_x_y.shape[1]
                                                       * features_x_y.shape[2]
                                                       ).T

                features[:,
                         num_images * i: num_images * (i+1)] = feature_vectors
                i += 1

        return features

    def _whiten_array(self, array):
        # Remove correlations from structured training data.
        # array must be of shape m variables x n samples
        if type(self.W) == type(None):  # None whitening array yet computed
            cov = np.cov(array)
            d, E = LA.eig(cov)
            # Compute whitening matrix from d, eigenvalues and E, eigenvectors
            D = np.diag(d)
            D_inv = LA.inv(D) ** 0.5
            W = LA.inv(E).dot(D_inv).dot(E.T)
            self.W = W

        return self.W.dot(array)

    def _compute_dictionary(self, X, iterations):
        # Compute dictionary. k is number of entries in a column of S that
        # will be non-zero.
        # NOTE: implementaiton of Algorithm 5 from Rubenstein.
        print('Computing diciontary.')

        D = np.zeros((X.shape[0], self.dict_len))
        S = np.zeros((self.dict_len, X.shape[1]))

        # populate D initially with random choices of X
        np.random.seed(0)
        choices = np.random.randint(0, X.shape[1], D.shape[1])
        D[:] = X[:, choices]
        # normalize columns to 1 (l2 norm)
        D = normalize(D, axis=0)

        for n in range(iterations):
            print('At iteration {} / {}'.format(n+1, iterations))
            G = D.T.dot(D)
            for i in progressbar.progressbar(range(X.shape[1])):
                S[:, i] = self._ompursuit(X[:, i], G, D,
                                          target_sparsity=self.k)
            for j in range(D.shape[1]):
                D[:, j] = 0
                I = np.where(S[j] != 0)[0]  # line 8
                g = S[j, I].T
                d = X[:, I].dot(g) - D.dot(S[:, I]).dot(g)
                if d.any():
                    d = d / LA.norm(d)
                g = X[:, I].T.dot(d) - (D.dot(S[:, I])).T.dot(d)
                D[:, j] = d
                S[j, I] = g.T

            print('Reconstruction error at iteration {}:'.format(n+1))
            error = LA.norm(X - D.dot(S)) ** 2
            print(error)
            self.history.append(error)

        np.save('train_dict_{}_{}_{}.npy'.format(self.dict_len,
                                                 self.k,
                                                 iterations), D)
        self.D = D
