import numpy as np

class DTLZ2:
    def __init__(self, n_var=5, n_obj=2):
        self.n_var = n_var
        self.n_obj = n_obj
        self.lb = 0.
        self.ub = 1.

    def g(self, X_m):
        return np.sum(np.square(X_m - 0.5), axis=1)


    def obj_func(self, X_, g, alpha=1):
        f = []
        for i in range(self.n_obj):
            _f = (1 + g)
            for j in range(self.n_obj - (i + 1)):  # Cosine product for first (M-1) objectives
                _f *= np.cos(np.power(X_[:, j], alpha) * np.pi / 2.0)
            if i > 0:
                _f *= np.sin(np.power(X_[:, self.n_obj - i - 1], alpha) * np.pi / 2.0)  # Only the last variable in sine

            f.append(_f)

        return np.column_stack(f)


    def evaluate(self, x):
        X, X_M = x[:, :self.n_var - self.n_obj + 1], x[:, self.n_var - self.n_obj + 1:]

        g = self.g(X_M)
        return self.obj_func(X, g)