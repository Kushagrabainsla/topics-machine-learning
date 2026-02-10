# ============================================================
# Step 1) Gradient Descent on ONE variable (scalar function)
# Goal: make "learning rate" and "updates" feel real
# ============================================================

import random
import numpy as np

def f(x):
    # Example convex parabola: f(x) = x^2 + 3x + 2
    return x**2 + 3*x + 2

def df(x):
    # Derivative: f'(x) = 2x + 3
    return 2*x + 3

def gradient_descent_scalar(x0, lr, steps):
    print(f"Gradient Descent starting at x0={x0:.6f}, lr={lr}, steps={steps}")
    x = float(x0)
    for step in range(steps):
        # compute gradient
        grad = df(x)

        # gradient descent update
        x = x - lr * grad
        print(f"step={step:3d}  x={x: .6f}  f(x)={f(x): .6f}  grad={grad: .6f}")

    return x


x_star = gradient_descent_scalar(x0=random.uniform(-10, 10), lr=0.01, steps=500)
print("final x:", x_star)
