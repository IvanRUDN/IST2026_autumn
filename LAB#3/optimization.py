import time
from collections import defaultdict
import numpy as np
from scipy.optimize import line_search


class LineSearchTool:
    def __init__(self, method="Wolfe", **kwargs):
        self._method = method
        self.c1 = kwargs.get("c1", 1e-4)
        self.c2 = kwargs.get("c2", 0.9)
        self.alpha_0 = kwargs.get("alpha_0", 1.0)
        self.c = kwargs.get("c", 1.0)
        if method not in ("Wolfe", "Armijo", "Constant"):
            raise ValueError("Неизвестный метод поиска шага: " + str(method))

    def line_search(self, oracle, x_k, d_k, previous_alpha=None):
        if self._method == "Constant":
            return self.c
        alpha = self.alpha_0 if previous_alpha is None else previous_alpha
        phi0 = oracle.func(x_k)
        derphi0 = oracle.grad(x_k).dot(d_k)
        if derphi0 >= 0:
            return None
        if self._method == "Armijo":
            for _ in range(80):
                if oracle.func(x_k + alpha * d_k) <= phi0 + self.c1 * alpha * derphi0:
                    return alpha
                alpha *= 0.5
            return None
        try:
            result = line_search(
                lambda a: oracle.func(x_k + a * d_k),
                lambda a: oracle.grad(x_k + a * d_k).dot(d_k),
                0.0, 1.0, old_fval=phi0, derphi0=derphi0,
                c1=self.c1, c2=self.c2
            )
            if result[0] is not None:
                return float(result[0])
        except Exception:
            pass
        # Резервный поиск шага Армихо
        alpha = self.alpha_0
        for _ in range(80):
            if oracle.func(x_k + alpha * d_k) <= phi0 + self.c1 * alpha * derphi0:
                return alpha
            alpha *= 0.5
        return None


def _history_start(oracle, x, trace):
    if not trace:
        return None
    start = time.perf_counter()
    hist = defaultdict(list)
    _save_history(hist, oracle, x, start, x.size <= 2)
    return hist, start


def _save_history(history, oracle, x, start, save_x):
    if history is None:
        return
    history["time"].append(time.perf_counter() - start)
    history["func"].append(float(oracle.func(x)))
    history["grad_norm"].append(float(np.linalg.norm(oracle.grad(x))))
    if save_x:
        history["x"].append(np.copy(x))


def gradient_descent(oracle, x_0, tolerance=1e-5, max_iter=10000,
                     line_search_options=None, trace=False, display=False):
    x = np.asarray(x_0, dtype=float).copy()
    ls = LineSearchTool(**(line_search_options or {}))
    pack = _history_start(oracle, x, trace)
    history, start = pack if trace else (None, None)
    save_x = x.size <= 2
    previous_alpha = None
    for iteration in range(max_iter):
        grad = np.asarray(oracle.grad(x)).ravel()
        if not np.all(np.isfinite(grad)):
            return x, "computational_error", history
        if np.linalg.norm(grad) <= tolerance:
            return x, "success", history
        direction = -grad
        alpha = ls.line_search(oracle, x, direction, previous_alpha)
        if alpha is None or not np.isfinite(alpha):
            return x, "computational_error", history
        x = x + alpha * direction
        previous_alpha = alpha
        _save_history(history, oracle, x, start, save_x)
        if display and (iteration % 100 == 0):
            print("Итерация", iteration + 1, "функция", oracle.func(x))
    status = "success" if np.linalg.norm(oracle.grad(x)) <= tolerance else "iterations_exceeded"
    return x, status, history


def newton(oracle, x_0, tolerance=1e-5, max_iter=100,
           line_search_options=None, trace=False, display=False):
    x = np.asarray(x_0, dtype=float).copy()
    ls = LineSearchTool(**(line_search_options or {"method": "Constant", "c": 1.0}))
    pack = _history_start(oracle, x, trace)
    history, start = pack if trace else (None, None)
    save_x = x.size <= 2
    previous_alpha = None
    for iteration in range(max_iter):
        grad = np.asarray(oracle.grad(x)).ravel()
        if not np.all(np.isfinite(grad)):
            return x, "computational_error", history
        if np.linalg.norm(grad) <= tolerance:
            return x, "success", history
        try:
            H = oracle.hess(x)
            if hasattr(H, "toarray"):
                H = H.toarray()
            H = np.asarray(H, dtype=float)
            # Малый сдвиг помогает при почти вырожденном Гессиане
            shift = max(0.0, 1e-10 - float(np.min(np.linalg.eigvalsh(H))))
            direction = np.linalg.solve(H + shift * np.eye(H.shape[0]), -grad)
        except Exception:
            return x, "newton_direction_error", history
        if grad.dot(direction) >= 0:
            direction = -grad
        alpha = ls.line_search(oracle, x, direction, previous_alpha)
        if alpha is None or not np.isfinite(alpha):
            return x, "computational_error", history
        x = x + alpha * direction
        previous_alpha = alpha
        _save_history(history, oracle, x, start, save_x)
        if display:
            print("Итерация", iteration + 1, "функция", oracle.func(x))
    status = "success" if np.linalg.norm(oracle.grad(x)) <= tolerance else "iterations_exceeded"
    return x, status, history
