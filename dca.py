"""
Discriminant Correlation Analysis.

Author: G. Worrall

Based on:

    Haghighat et al. - Discriminant Correlation Analysis: Real-Time Feature
        Level Fusion for Multimodel Biometric Recognition (2016)
"""
import math
import numpy as np
import numpy.linalg as LA


class DCA:
    """Calculate Discriminant Correlation Analysis between two sets of
    variables. Follows the method outlined in the above paper.

    Attributes:
        X_transform (np.array): computed transform for X feature matrix
        Y_transform (np.array): computed transform for Y feature matrix
    """

    def __init__(self):
        pass

    def fit(self, X, Y, class_labels, r_dimensions):
        """Calculate DCA transform for two feature matrices.

        Args:
            X (np.array): features matrix of shape n samples * m features
            Y (np.array): features matrix of shape n samples * m features
            class_labels (np.array): class labels of shape n samples
            r_dimensions (int): number of dimensions to reduce X and Y to.

        Returns:
            None
        """
        self.r = r_dimensions
        # Check that r_dimensions is leq c - 1, rank(X), rank(Y)
        rank_X = LA.matrix_rank(X)
        rank_Y = LA.matrix_rank(Y)
        classes, counts = np.unique(class_labels, return_counts=True)

        if any(r_dimensions > k for k in (len(classes), rank_X, rank_Y)):
            raise ValueError("r cannot be bigger than num_classes-1 "
                             "or rank(X) or rank(Y).")

        Wb_X = self._get_Wb_transform(X, classes, counts, class_labels)
        Wb_Y = self._get_Wb_transform(Y, classes, counts, class_labels)

        X_dash = Wb_X.T.dot(X.T)  # Eqn 14 in paper. Transpose added as
        Y_dash = Wb_Y.T.dot(Y.T)  # here features matrices are n x m

        Wc_X, Wc_Y = self._compute_Wc_transforms(X_dash, Y_dash)

        self.X_tranform = Wc_X.T.dot(Wb_X.T)
        self.Y_tranform = Wc_Y.T.dot(Wb_Y.T)

    def transform(self, X, Y):
        """Coverts the input feature matrices into transformed feature set
        of reduced dimension r.

        Args:
            X (np.array): features matrix of shape n samples * m features
            Y (np.array): features matrix of shape n samples * m features

        Returns:
            X_star (np.array): feature matrix of n samples x r features
            Y_star (np.array): feature matrix of n samples x r features
        """

        X_star = self.X_tranform.dot(X.T).T
        Y_star = self.Y_tranform.dot(Y.T).T

        return X_star, Y_star

    def _compute_Wc_transforms(self, X_dash, Y_dash):
        # Compute Wc transformation
        # NOTE: an error on the line below will have been caused by a
        # difference in the number of positive non-zero eigenvectors
        # found for each feature matrix in the previous step.
        S_xy_dash = X_dash.dot(Y_dash.T)

        U, Sig, V = LA.svd(S_xy_dash)
        Sig = np.diag(Sig)

        Sig_inv_h = LA.inv(Sig) ** 0.5

        Wc_X = U.dot(Sig_inv_h)
        Wc_Y = V.dot(Sig_inv_h)

        return Wc_X, Wc_Y

    def _get_Wb_transform(self, F, classes, counts, labels):
        # Compute transformation that unitizes the between-class scatter
        # matrix and reduces the dimensionality of the feature matrix

        F_mean = np.mean(F, axis=0)
        class_means = np.zeros((len(classes), F.shape[1]))

        i = 0
        for c in classes:
            class_means[i] = np.mean(F[np.where(labels == c)], axis=0)
            i += 1

        Phi_bx = class_means - F_mean  # this is the transpose of Phi_bx from
        # the paper. Easier to compute in numpy.

        for i in range(len(classes)):
            Phi_bx[i] = math.sqrt(counts[i]) * Phi_bx[i]

        Phi_bx = Phi_bx.T  # Remap to paper shape to avoid confusion

        M = Phi_bx.T.dot(Phi_bx)

        lam, P = LA.eigh(M)
        P = np.fliplr(P)  # flip and reorder both as np gives in asc order
        lam = np.fliplr([lam])[0]
        lam[np.where(lam < 0)] = 0  # cut out tiny eigenvalues that should be 0

        num_eigvalues = np.count_nonzero(lam)
        if num_eigvalues < self.r:
            self.r = num_eigvalues  # can only have as many dims as valid eigs

        Q = P[:, :self.r]  # get first r significant eigenvectors
        L = np.diag(lam[:self.r])

        L_inv_h = LA.inv(L) ** 0.5

        Wb = Phi_bx.dot(Q).dot(L_inv_h)

        return Wb
