import numpy as np
import matplotlib.pyplot as plt
def lwlr(x, X, y, tau):
    W = np.diag(np.exp(-np.sum((X - x)**2, axis=1) / (2 * tau**2)))
    theta = np.linalg.pinv(X.T @ W @ X) @ X.T @ W @ y
    return x @ theta
np.random.seed(42)
X = np.linspace(0, 2 * np.pi, 100)
y = np.sin(X) + 0.1 * np.random.randn(100)
Xb = np.c_[np.ones_like(X), X]
Xt = np.linspace(0, 2 * np.pi, 200)
Xtb = np.c_[np.ones_like(Xt), Xt]
tau = 0.5
yp = np.array([lwlr(xi, Xb, y, tau) for xi in Xtb])
plt.figure(figsize=(10, 6))
plt.scatter(X, y)
plt.plot(Xt, yp,)
plt.show()
