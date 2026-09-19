import numpy as np

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
    residual = y - (x @ beta + beta_intercept)
    data_loss = np.sum(residual ** 2) / (2 * x.shape[0])
    penalty = gamma * (
        alpha * np.sum(beta ** 2) + (1 - alpha) * np.sum(np.abs(beta))
    )
    return float(data_loss + penalty)

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
    residual = x @ beta + beta_intercept - y
    # Average squared-error gradient over the batch, then add the L2 gradient.
    gradient = (x.T @ residual) / x.shape[0] + 2 * gamma * alpha * beta
    beta_step = beta - eta * gradient

    # Proximal update for gamma * (1 - alpha) * ||beta||_1.
    threshold = eta * gamma * (1 - alpha)
    new_beta = np.sign(beta_step) * np.maximum(np.abs(beta_step) - threshold, 0)
    new_beta_intercept = beta_intercept - eta * np.mean(residual)
    return new_beta, float(new_beta_intercept)


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
        self.gamma = gamma
        self.alpha = alpha
        self.eta = eta
        self.batch = batch
        self.epoch = epoch

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
            x: (n, p) ndarray of standardized training features.
            y: (n,) ndarray of training targets.

        Returns:
            dict mapping epoch number (1-indexed) -> loss() at the end of
            that epoch.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.ndim != 2 or y.shape != (x.shape[0],) or x.shape[0] == 0:
            raise ValueError("Expected nonempty x with shape (n, p) and y with shape (n,)")
        n, p = x.shape
        if not isinstance(self.batch, (int, np.integer)) or not 1 <= self.batch <= n:
            raise ValueError("batch must be an integer between 1 and n")
        if not isinstance(self.epoch, (int, np.integer)) or self.epoch < 1:
            raise ValueError("epoch must be a positive integer")

        self.beta = np.zeros(p, dtype=float)
        self.beta_intercept = 0.0
        history = {}
        stable_epochs = 0
        tolerance = 1e-6

        for epoch_number in range(1, self.epoch + 1):
            previous_beta = self.beta.copy()
            previous_intercept = self.beta_intercept
            indices = np.random.permutation(n)
            for start in range(0, n, self.batch):
                batch_indices = indices[start:start + self.batch]
                self.beta, self.beta_intercept = grad_step(
                    x[batch_indices], y[batch_indices], self.beta,
                    self.beta_intercept, self.gamma, self.alpha, self.eta,
                )

            history[epoch_number] = loss(
                x, y, self.beta, self.beta_intercept, self.gamma, self.alpha,
            )
            if not np.isfinite(history[epoch_number]):
                raise FloatingPointError("Training loss is not finite; check the data and learning rate")

            # Require several stable epochs because minibatch updates are noisy.
            beta_stable = np.linalg.norm(self.beta - previous_beta) <= (
                tolerance * (1 + np.linalg.norm(previous_beta))
            )
            intercept_stable = abs(self.beta_intercept - previous_intercept) <= (
                tolerance * (1 + abs(previous_intercept))
            )
            stable_epochs = stable_epochs + 1 if beta_stable and intercept_stable else 0
            if stable_epochs >= 5:
                break

        return history

    def predict(self, x):
        """Predicts targets for new data using the learned beta and
        beta_intercept.

        Args:
            x: (m, p) ndarray of features.

        Returns:
            (m,) ndarray of predicted targets.
        """
        return x.dot(self.beta) + self.beta_intercept
