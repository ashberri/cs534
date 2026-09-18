
def loss(x, y, beta, beta_intercept, gamma, alpha):
    """Elastic net objective:

        1/(2n) * ||y - X*beta - beta_intercept||_2^2
            + gamma * (alpha * ||beta||_2^2 + (1-alpha) * ||beta||_1)

    Args:
        x: (n, p) ndarray of features.
        y: (n,) ndarray of targets.
        beta: (p,) ndarray of coefficients (no intercept term in here).
        beta_intercept: scalar intercept, added on to x.dot(beta).
        gamma: scalar >= 0, overall regularization strength.
        alpha: scalar in [0, 1], mixing weight between the L2 (ridge) and
            L1 (lasso) penalties (alpha=1 -> ridge only, alpha=0 -> lasso
            only). Note beta_intercept is never penalized.

    Returns:
        Scalar loss value.
    """
    return 0

def grad_step(x, y, beta, beta_intercept, gamma, alpha, eta):
    """Performs one (proximal) gradient descent step on beta and
    beta_intercept for the loss() above, given a single batch of data.

    Args:
        x: (n, p) ndarray of features for this batch.
        y: (n,) ndarray of targets for this batch.
        beta: (p,) ndarray, current coefficients.
        beta_intercept: scalar, current intercept.
        gamma: scalar >= 0, overall regularization strength.
        alpha: scalar in [0, 1], L2/L1 mixing weight (see loss()).
        eta: scalar > 0, step size / learning rate.

    Returns:
        (new_beta, new_beta_intercept) tuple after one update step.
        new_beta is a (p,) ndarray; new_beta_intercept is a scalar.
    """
    # returns (new_beta, new_beta_intercept)
    return 0, 0


class ElasticNet:
    def __init__(self, gamma, alpha, eta, batch, epoch):
        """
        Args:
            gamma: scalar >= 0, overall regularization strength.
            alpha: scalar in [0, 1], L2/L1 mixing weight (see loss()).
            eta: scalar > 0, step size / learning rate for grad_step().
            batch: int, minibatch size to use in train() (batch=1 is
                pure SGD, batch=n is full-batch gradient descent).
            epoch: int, number of passes over the training data.
        """
        self.beta = None
        # beta_intercept is learned in train() alongside beta,
        # but is not penalized by the L1/L2 terms
        self.beta_intercept = 0
        # do other things you might need
        return

    def coef(self):
        """Returns the learned coefficients (not including the intercept).

        Returns:
            (p,) ndarray, or None if train() hasn't been called yet.
        """
        return self.beta

    def train(self, x, y):
        """Fits beta and beta_intercept via minibatch (proximal) gradient
        descent, shuffling the data each epoch and stopping early if beta
        stops changing meaningfully between epochs.

        Args:
            x: (n, p) ndarray of training features.
            y: (n,) ndarray of training targets.

        Returns:
            dict mapping epoch number (1-indexed) -> loss() at the end of
            that epoch.
        """
        return 0

    def predict(self, x):
        """Predicts targets for new data using the learned beta and
        beta_intercept.

        Args:
            x: (m, p) ndarray of features.

        Returns:
            (m,) ndarray of predicted targets.
        """
        return x.dot(self.beta) + self.beta_intercept
