# Imports
import argparse
import matplotlib.pyplot as plt
import numpy as np
import optuna

from scipy.optimize import brute, minimize
from sklearn.gaussian_process import GaussianProcessRegressor


# Read in the arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("study_name", type=str, nargs="?",
                    const="Hyperparameter optimization",
                    default="Hyperparameter optimization")
parser.add_argument("storage", type=str, nargs="?",
                    const="sqlite:///ζ=1.db",
                    default="sqlite:///ζ=1.db")

args = parser.parse_args()


# Load the study
study = optuna.load_study(study_name=args.study_name, storage=args.storage)


# Create the Gaussian process model
parameters = study.trials[0].params.keys()
X = np.array([[trial.params[p] for p in parameters]
              for trial in study.trials if trial.value is not None])
y = np.array([trial.value for trial in study.trials if trial.value is not None])

gpr = GaussianProcessRegressor(alpha=0.1, normalize_y=True)
gpr.fit(X, y)

y_pred, std_pred = gpr.predict(X, return_std=True)
n = len(y_pred)


# Create and save the plot of the mean values
fig, ax = plt.subplots(1, 1)
ax.plot(range(n), y, color="tab:blue", label="observed value")
ax.plot(range(n), y_pred, color="tab:red", label="prediction")
ax.fill_between(range(n), y_pred - 3 * std_pred, y_pred + 3 * std_pred,
                color="tab:red", alpha=0.3, label="mean +/- 3 * sigma")
ax.grid(alpha=0.5)
ax.set_xlabel("trial", fontdict={"size": 12, "weight": "bold"})
ax.set_ylabel("value", fontdict={"size": 12, "weight": "bold"})
ax.set_title("Gaussian process approximation",
             fontdict={"size": 14, "weight": "bold"})
ax.legend()

fig.savefig("images/gp_approximation.png")


# Create and save the plot of the standard deviation
fig, ax = plt.subplots(1, 1)
ax.plot(range(n), np.cumsum(std_pred) / np.arange(1, n + 1),
        color="tab:red", label="average standard deviation")
ax.grid(alpha=0.5)
ax.set_xlabel("trial", fontdict={"size": 12, "weight": "bold"})
ax.set_ylabel("value", fontdict={"size": 12, "weight": "bold"})
ax.set_title("Average standard deviation",
             fontdict={"size": 14, "weight": "bold"})
ax.legend()

fig.savefig("images/std_plot.png")


# Minimize the predicted posterior mean
minimize_res = minimize(fun=lambda x: gpr.predict(x.reshape(1, -1)),
                        x0=np.array([study.best_trial.params[p] for p in parameters]))
minimize_X_min, minimize_val_min = minimize_res.x, minimize_res.fun
d = len(minimize_X_min)


# Create and save the plots for the predicted posterior mean
# in a neighbourhood of the minimum along different directions
fig1, ax1 = plt.subplots(3, 2, figsize=(12, 8))
ax1 = ax1.ravel()

n_points = 100
for i in range(d):
    X = minimize_X_min + np.array([t * np.eye(d)[i] for t in np.linspace(-1, 1, n_points)])
    y_pred, std_pred = gpr.predict(X, return_std=True)

    ax1[i].axhline(minimize_val_min, color="tab:blue", ls="--", label="minimum")
    ax1[i].plot(X[:, i], y_pred, color="tab:red", label="prediction")
    ax1[i].fill_between(X[:, i], y_pred - 3 * std_pred, y_pred + 3 * std_pred,
                        color="tab:red", alpha=0.3, label="mean +/- 3 * sigma")
    ax1[i].grid(alpha=0.5)
    ax1[i].set_xlabel(f"x[{i}]", fontdict={"size": 10, "weight": "bold"})
    ax1[i].set_ylabel("value", fontdict={"size": 10, "weight": "bold"})
    ax1[i].set_title(f"direction {i:d}", fontdict={"size": 12, "weight": "bold"})
    ax1[i].legend()

plt.tight_layout()

fig1.savefig("images/neighbourhood_of_minimum_minimize.png")


# Perform grid search for the global minimum
brute_res = brute(func=lambda x: gpr.predict(x.reshape(1, -1)),
                  ranges=X.shape[1] * (slice(0.1, 5.1, 50),),
                  full_output=True)
brute_X_min, brute_val_min, *_ = brute_res
d = len(brute_X_min)


# Create and save the plots for the predicted posterior mean
# in a neighbourhood of the minimum along different directions
fig2, ax2 = plt.subplots(3, 2, figsize=(12, 8))
print(minimize_X_min)
print(brute_X_min)
ax2 = ax2.ravel()

n_points = 100
for i in range(d):
    X = brute_X_min + np.array([t * np.eye(d)[i] for t in np.linspace(-1, 1, n_points)])
    y_pred, std_pred = gpr.predict(X, return_std=True)

    ax2[i].axhline(brute_val_min, color="tab:blue", ls="--", label="minimum")
    ax2[i].plot(X[:, i], y_pred, color="tab:red", label="prediction")
    ax2[i].fill_between(X[:, i], y_pred - 3 * std_pred, y_pred + 3 * std_pred,
                        color="tab:red", alpha=0.3, label="mean +/- 3 * sigma")
    ax2[i].grid(alpha=0.5)
    ax2[i].set_xlabel(f"x[{i}]", fontdict={"size": 10, "weight": "bold"})
    ax2[i].set_ylabel("value", fontdict={"size": 10, "weight": "bold"})
    ax2[i].set_title(f"direction {i:d}", fontdict={"size": 12, "weight": "bold"})
    ax2[i].legend()

plt.tight_layout()

fig2.savefig("images/neighbourhood_of_minimum_brute.png")