import numpy as np
import scipy.sparse
from scipy.special import expit


class BaseSmoothOracle:
    def func(self, x):
        raise NotImplementedError

    def grad(self, x):
        raise NotImplementedError

    def hess(self, x):
        raise NotImplementedError

    def func_directional(self, x, d, alpha):
        return float(np.squeeze(self.func(x + alpha * d)))

    def grad_directional(self, x, d, alpha):
        return float(np.squeeze(self.grad(x + alpha * d).dot(d)))


class QuadraticOracle(BaseSmoothOracle):
    """f(x) = 1/2 x.T A x - b.T x."""
    def __init__(self, A, b):
        if scipy.sparse.issparse(A):
            if A.shape[0] != A.shape[1]:
                raise ValueError("A должна быть квадратной")
        elif not np.allclose(A, np.asarray(A).T):
            raise ValueError("A должна быть симметричной")
        self.A = A
        self.b = np.asarray(b, dtype=float)

    def func(self, x):
        x = np.asarray(x)
        return float(0.5 * self.A.dot(x).dot(x) - self.b.dot(x))

    def grad(self, x):
        return np.asarray(self.A.dot(x) - self.b).ravel()

    def hess(self, x):
        return self.A


class LogRegL2Oracle(BaseSmoothOracle):
    """
    L2-регуляризованная логистическая регрессия:
    mean(log(1 + exp(-b_i * a_i.T x))) + regcoef/2 * ||x||^2.
    Метки b должны быть -1 или +1.
    """
    def __init__(self, A, b, regcoef=1e-3):
        self.A = A.tocsr() if scipy.sparse.issparse(A) else np.asarray(A, dtype=float)
        self.b = np.asarray(b, dtype=float).reshape(-1)
        if self.A.shape[0] != self.b.size:
            raise ValueError("Число объектов и меток не совпадает")
        if not np.all(np.isin(self.b, [-1, 1])):
            raise ValueError("Метки должны принимать значения -1 и +1")
        self.regcoef = float(regcoef)
        self._m, self._n = self.A.shape

    def func(self, x):
        x = np.asarray(x).reshape(-1)
        margins = self.b * np.asarray(self.A.dot(x)).ravel()
        # logaddexp устойчивее прямого вычисления log(1 + exp(z))
        return float(np.mean(np.logaddexp(0.0, -margins))
                     + 0.5 * self.regcoef * np.dot(x, x))

    def grad(self, x):
        x = np.asarray(x).reshape(-1)
        margins = self.b * np.asarray(self.A.dot(x)).ravel()
        coeff = -self.b * expit(-margins) / self._m
        return np.asarray(self.A.T.dot(coeff)).ravel() + self.regcoef * x

    def hess(self, x):
        x = np.asarray(x).reshape(-1)
        margins = self.b * np.asarray(self.A.dot(x)).ravel()
        weights = expit(margins) * expit(-margins) / self._m
        if scipy.sparse.issparse(self.A):
            weighted_A = self.A.multiply(weights[:, None])
            H = self.A.T.dot(weighted_A).toarray()
        else:
            H = self.A.T @ (weights[:, None] * self.A)
        H += self.regcoef * np.eye(self._n)
        return H


def grad_finite_diff(func, x, eps=1e-6):
    """Центральная разностная оценка градиента."""
    x = np.asarray(x, dtype=float)
    result = np.zeros_like(x)
    for i in range(x.size):
        step = np.zeros_like(x)
        step[i] = eps
        result[i] = (func(x + step) - func(x - step)) / (2 * eps)
    return result


def hess_finite_diff(func, x, eps=1e-4):
    """Оценка Гессиана через центральные разности градиента."""
    x = np.asarray(x, dtype=float)
    n = x.size
    H = np.zeros((n, n))
    for i in range(n):
        step = np.zeros(n)
        step[i] = eps
        H[:, i] = (grad_finite_diff(func, x + step, eps=eps/10)
                   - grad_finite_diff(func, x - step, eps=eps/10)) / (2 * eps)
    return 0.5 * (H + H.T)
