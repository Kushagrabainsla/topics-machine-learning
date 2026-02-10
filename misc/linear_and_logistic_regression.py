# Implementations and demonstrations of Linear Regression and Logistic Regression from scratch,
# plus hands-on illustration of Cross-Entropy (Log Loss), Maximum Likelihood view, and Regularization.
# This code runs here so you can see results and plots immediately.

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ---------------------------
# Helper functions / utilities
# ---------------------------
def mse(y, y_pred):
    return np.mean((y - y_pred)**2)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cross_entropy_loss(y, p, eps=1e-12):
    # y: {0,1}, p: predicted probabilities
    p = np.clip(p, eps, 1-eps)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

# ---------------------------
# Part A: Linear Regression
# ---------------------------
print("PART A: LINEAR REGRESSION\n")

# Generate synthetic linear data: y = 3*x + 4 + noise
n = 120
X_reg = np.linspace(-3, 3, n).reshape(-1, 1)
true_w = 3.0
true_b = 4.0
noise = np.random.normal(scale=1.0, size=(n,1))
y_reg = true_w * X_reg + true_b + noise

# Add bias column for convenience in closed form
X_design = np.hstack([np.ones((n,1)), X_reg])  # [1, x]

# Closed-form solution (Normal equation)
w_closed = np.linalg.pinv(X_design.T @ X_design) @ (X_design.T @ y_reg)
b_closed = w_closed[0,0]
w_closed_val = w_closed[1,0]
print("Closed-form solution: w = {:.4f}, b = {:.4f}".format(w_closed_val, b_closed))
print("True params: w = {:.2f}, b = {:.2f}".format(true_w, true_b))

# Gradient Descent implementation (batch GD)
def linear_regression_gd(X, y, lr=0.1, epochs=2000, verbose=False, l2=0.0):
    # X: (n, d) without bias. We'll learn w (d,) and b scalar
    n, d = X.shape
    w = np.zeros((d,1))
    b = 0.0
    losses = []
    for epoch in range(epochs):
        y_pred = X @ w + b
        error = y_pred - y  # (n,1)
        loss = np.mean(error**2) + l2 * np.sum(w**2)
        losses.append(loss)
        # gradients
        grad_w = (2/n) * (X.T @ error) + 2 * l2 * w
        grad_b = (2/n) * np.sum(error)
        w = w - lr * grad_w
        b = b - lr * grad_b
        if verbose and (epoch % 500 == 0):
            print(f"Epoch {epoch}, loss={loss:.4f}")
    return w.flatten(), b, losses

# Train with GD (note: scaling helps here but this synthetic data is already reasonable)
w_gd, b_gd, losses = linear_regression_gd(X_reg, y_reg, lr=0.1, epochs=3000, verbose=True)
print("\nGradient Descent solution: w = {:.4f}, b = {:.4f}".format(w_gd[0], b_gd))

# Compare MSEs
y_pred_closed = X_reg * w_closed_val + b_closed
y_pred_gd = X_reg * w_gd[0] + b_gd
print("MSE (closed-form): {:.4f}".format(mse(y_reg, y_pred_closed)))
print("MSE (gd): {:.4f}".format(mse(y_reg, y_pred_gd)))

# Plot data and fitted lines
plt.figure(figsize=(8,5))
plt.scatter(X_reg, y_reg, label="data", alpha=0.6)
xs = np.array([-3,3]).reshape(-1,1)
plt.plot(xs, w_closed_val*xs + b_closed, label="Closed-form", linewidth=2)
plt.plot(xs, w_gd[0]*xs + b_gd, "--", label="Gradient Descent", linewidth=2)
plt.title("Linear Regression: data and fitted lines")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()

# Plot GD loss curve
plt.figure(figsize=(6,4))
plt.plot(losses)
plt.title("Linear Regression (GD) training loss (MSE)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

# ---------------------------
# Part B: Logistic Regression
# ---------------------------
print("\nPART B: LOGISTIC REGRESSION\n")

# Generate synthetic 2D classification data (two gaussian blobs)
n_per_class = 150
mean0 = np.array([-1.5, -1.0])
mean1 = np.array([1.5, 1.0])
cov = np.array([[1.0, 0.4],[0.4, 1.0]])

X0 = np.random.multivariate_normal(mean0, cov, size=n_per_class)
X1 = np.random.multivariate_normal(mean1, cov, size=n_per_class)
X_clf = np.vstack([X0, X1])
y_clf = np.hstack([np.zeros(n_per_class), np.ones(n_per_class)])

# Shuffle
perm = np.random.permutation(len(y_clf))
X_clf = X_clf[perm]
y_clf = y_clf[perm]

# Simple logistic regression with batch gradient descent and L2 regularization
def logistic_regression_gd(X, y, lr=0.1, epochs=1000, l2=0.0, verbose=False):
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    losses = []
    for epoch in range(epochs):
        z = X @ w + b
        p = sigmoid(z)
        loss = cross_entropy_loss(y, p) + l2 * 0.5 * np.sum(w**2)
        losses.append(loss)
        # gradients
        grad_w = (1/n) * (X.T @ (p - y)) + l2 * w
        grad_b = (1/n) * np.sum(p - y)
        w = w - lr * grad_w
        b = b - lr * grad_b
        if verbose and (epoch % 200 == 0):
            print(f"Epoch {epoch}, loss={loss:.4f}")
    return w, b, losses

# Train without regularization
w0, b0, losses0 = logistic_regression_gd(X_clf, y_clf, lr=0.5, epochs=2000, l2=0.0, verbose=True)
print("Trained logistic regression (no regularization). Weight norm: {:.4f}".format(np.linalg.norm(w0)))

# Train with L2 regularization
w_reg, b_reg, losses_reg = logistic_regression_gd(X_clf, y_clf, lr=0.5, epochs=2000, l2=1.0, verbose=False)
print("Trained logistic regression (L2 lambda=1.0). Weight norm: {:.4f}".format(np.linalg.norm(w_reg)))

# Compute final losses
print("Final cross-entropy (no reg): {:.4f}".format(losses0[-1]))
print("Final cross-entropy (l2=1.0): {:.4f}".format(losses_reg[-1]))

# Plot data and decision boundary for both models
xx_min, xx_max = X_clf[:,0].min()-1, X_clf[:,0].max()+1
yy_min, yy_max = X_clf[:,1].min()-1, X_clf[:,1].max()+1
xx, yy = np.meshgrid(np.linspace(xx_min, xx_max, 200), np.linspace(yy_min, yy_max, 200))
grid = np.c_[xx.ravel(), yy.ravel()]

# predict probabilities on grid
pz0 = sigmoid(grid @ w0 + b0).reshape(xx.shape)
pz_reg = sigmoid(grid @ w_reg + b_reg).reshape(xx.shape)

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.contourf(xx, yy, pz0, levels=20, alpha=0.8)
plt.scatter(X_clf[y_clf==0,0], X_clf[y_clf==0,1], label="class 0", edgecolor='k')
plt.scatter(X_clf[y_clf==1,0], X_clf[y_clf==1,1], label="class 1", edgecolor='k')
plt.title("Logistic Regression (no reg)\nDecision surface (probabilities)")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()

plt.subplot(1,2,2)
plt.contourf(xx, yy, pz_reg, levels=20, alpha=0.8)
plt.scatter(X_clf[y_clf==0,0], X_clf[y_clf==0,1], label="class 0", edgecolor='k')
plt.scatter(X_clf[y_clf==1,0], X_clf[y_clf==1,1], label="class 1", edgecolor='k')
plt.title("Logistic Regression (L2 lambda=1.0)\nDecision surface (probabilities)")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()

plt.tight_layout()
plt.show()

# Plot loss curves
plt.figure(figsize=(8,4))
plt.plot(losses0, label="no reg")
plt.plot(losses_reg, label="l2=1.0")
plt.title("Logistic Regression training loss (cross-entropy)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

# Show final classification accuracy for reference
def predict_logistic(X, w, b, threshold=0.5):
    p = sigmoid(X @ w + b)
    return (p >= threshold).astype(int), p

preds0, p0 = predict_logistic(X_clf, w0, b0)
preds_reg, p_reg = predict_logistic(X_clf, w_reg, b_reg)
acc0 = np.mean(preds0 == y_clf)
accreg = np.mean(preds_reg == y_clf)
print("Accuracy (no reg): {:.3f}, Accuracy (l2=1.0): {:.3f}".format(acc0, accreg))

# Show effect of extreme L2 on weights (underfitting)
w_big_l2, b_big_l2, _ = logistic_regression_gd(X_clf, y_clf, lr=0.5, epochs=2000, l2=100.0, verbose=False)
print("Weight norms: no reg {:.4f}, l2=1.0 {:.4f}, l2=100.0 {:.4f}".format(np.linalg.norm(w0), np.linalg.norm(w_reg), np.linalg.norm(w_big_l2)))

print("\n--- End of notebook demonstration ---")

