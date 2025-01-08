# Imports
import argparse
import itertools
import matplotlib.pyplot as plt
import numpy as np
import optuna

from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
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


# Extract the explanatory and response variables
parameters = study.trials[0].params.keys()
X = np.array([[trial.params[p] for p in parameters]
              for trial in study.trials if trial.value is not None])
y = np.array([trial.value for trial in study.trials if trial.value is not None])
d, n = len(parameters), len(y)

# Best trial
x0 = np.array([study.best_trial.params[p] for p in parameters])



# Gaussian process approximation
gp = GaussianProcessRegressor(alpha=0.1, normalize_y=True)
gp.fit(X, y)

y_gp_pred, std_gp_pred = gp.predict(X, return_std=True)


# Plot the predictions
fig, ax = plt.subplots(1, 1)

ax.plot(range(n), y, color="tab:blue", label="observed value")
ax.plot(range(n), y_gp_pred, color="tab:red", label="prediction")
ax.fill_between(range(n), y_gp_pred - std_gp_pred, y_gp_pred + std_gp_pred,
                color="tab:red", alpha=0.3, label="mean +/- sigma")
ax.grid(alpha=0.5)
ax.set_xlabel("trial", fontdict={"size": 12, "weight": "bold"})
ax.set_ylabel("value", fontdict={"size": 12, "weight": "bold"})
ax.set_title("Gaussian process approximation",
             fontdict={"size": 14, "weight": "bold"})
ax.legend()

fig.savefig("images/gp_approximation.png")


# Plot the standard deviation
fig, ax = plt.subplots(1, 1)

ax.plot(range(n), np.cumsum(std_gp_pred) / np.arange(1, n + 1),
        color="tab:red", label="average standard deviation")
ax.grid(alpha=0.5)
ax.set_xlabel("trial", fontdict={"size": 12, "weight": "bold"})
ax.set_ylabel("value", fontdict={"size": 12, "weight": "bold"})
ax.set_title("Average standard deviation",
             fontdict={"size": 14, "weight": "bold"})
ax.legend()

fig.savefig("images/std_gp.png")



# Random forest
rf = RandomForestRegressor()
rf.fit(X, y)

y_rf_pred = rf.predict(X)


# Plot the predictions
fig, ax = plt.subplots(1, 1)

ax.plot(range(n), y, color="tab:blue", label="observed value")
ax.plot(range(n), y_rf_pred, color="tab:red", label="prediction")
ax.grid(alpha=0.5)
ax.set_xlabel("trial", fontdict={"size": 12, "weight": "bold"})
ax.set_ylabel("value", fontdict={"size": 12, "weight": "bold"})
ax.set_title("Random forest approximation",
             fontdict={"size": 14, "weight": "bold"})
ax.legend()

fig.savefig("images/rf_approximation.png")




# Minimize the predicted values
max_boundary = np.max(X, axis=0)
max_boundary += np.full_like(max_boundary, 0.1)
#min_boundary = np.min(X, axis=0)
min_boundary = np.zeros_like(max_boundary)



def penalized_gp_prediction(x):
    if any(np.less(x, min_boundary)) or any(np.greater(x, max_boundary)):
        return 100
    else:
        return gp.predict(x.reshape(1, -1))


def penalized_rf_prediction(x):
    if any(np.less(x, min_boundary)) or any(np.greater(x, max_boundary)):
        return 100
    else:
        return rf.predict(x.reshape(1, -1))


minimize_gp_res = minimize(fun=penalized_gp_prediction, x0=x0)
minimize_gp_X_min, minimize_gp_val_min = minimize_gp_res.x, minimize_gp_res.fun

minimize_rf_res = minimize(fun=penalized_rf_prediction, x0=x0)
minimize_rf_X_min, minimize_rf_val_min = minimize_rf_res.x, minimize_rf_res.fun


# Perform grid search for the global minimum
brute_gp_X_min, brute_gp_val_min = None, float("inf")
brute_rf_X_min, brute_rf_val_min = None, float("inf")
for x in itertools.product(*[np.linspace(min_boundary[i], max_boundary[i], 10)
                             for i in range(len(min_boundary))]):
    x = np.array(x)

    val = gp.predict(x.reshape(1, -1))
    if val < brute_gp_val_min:
        brute_gp_X_min, brute_gp_val_min = x, val

    val = rf.predict(x.reshape(1, -1))
    if val < brute_rf_val_min:
        brute_rf_X_min, brute_rf_val_min = x, val



# Number of grid points for plotting
n_points = 100

# Plot the gp predicted values in a neighbourhood of the minimum
fig1, ax1 = plt.subplots(3, 2, figsize=(12, 8))
ax1 = ax1.ravel()

for i in range(d):
    #X_range = minimize_gp_X_min + np.array([t * np.eye(d)[i] for t in np.linspace(-1, 1, n_points)])
    X_range = np.tile(minimize_gp_X_min, (n_points, 1))
    X_range[:, i] = np.linspace(min_boundary[i], max_boundary[i], n_points)
    y_gp_pred, std_gp_pred = gp.predict(X_range, return_std=True)

    ax1[i].axhline(minimize_gp_val_min, color="tab:blue", ls="--", label="minimum")
    ax1[i].plot(X_range[:, i], y_gp_pred, color="tab:red", label="prediction")
    ax1[i].fill_between(X_range[:, i], y_gp_pred - std_gp_pred, y_gp_pred + std_gp_pred,
                        color="tab:red", alpha=0.3, label="mean +/- sigma")
    ax1[i].grid(alpha=0.5)
    ax1[i].set_xlabel(f"x[{i}]", fontdict={"size": 10, "weight": "bold"})
    ax1[i].set_ylabel("value", fontdict={"size": 10, "weight": "bold"})
    ax1[i].set_title(f"direction {i:d}", fontdict={"size": 12, "weight": "bold"})
    ax1[i].legend()

plt.tight_layout()

fig1.savefig("images/minimize_gp.png")


# Plot the rf predicted values in a neighbourhood of the minimum
fig2, ax2 = plt.subplots(3, 2, figsize=(12, 8))
ax2 = ax2.ravel()

for i in range(d):
    #X_range = minimize_rf_X_min + np.array([t * np.eye(d)[i] for t in np.linspace(-1, 1, n_points)])
    X_range = np.tile(minimize_rf_X_min, (n_points, 1))
    X_range[:, i] = np.linspace(min_boundary[i], max_boundary[i], n_points)
    y_rf_pred = rf.predict(X_range)

    ax2[i].axhline(minimize_rf_val_min, color="tab:blue", ls="--", label="minimum")
    ax2[i].plot(X_range[:, i], y_rf_pred, color="tab:red", label="prediction")
    ax2[i].grid(alpha=0.5)
    ax2[i].set_xlabel(f"x[{i}]", fontdict={"size": 10, "weight": "bold"})
    ax2[i].set_ylabel("value", fontdict={"size": 10, "weight": "bold"})
    ax2[i].set_title(f"direction {i:d}", fontdict={"size": 12, "weight": "bold"})
    ax2[i].legend()

plt.tight_layout()

fig2.savefig("images/minimize_rf.png")


# Plot the gp predicted values in a neighbourhood of the minimum
fig3, ax3 = plt.subplots(3, 2, figsize=(12, 8))
ax3 = ax3.ravel()

for i in range(d):
    #X_range = brute_gp_X_min + np.array([t * np.eye(d)[i] for t in np.linspace(-1, 1, n_points)])
    X_range = np.tile(brute_gp_X_min, (n_points, 1))
    X_range[:, i] = np.linspace(min_boundary[i], max_boundary[i], n_points)
    y_gp_pred, std_gp_pred = gp.predict(X_range, return_std=True)

    ax3[i].axhline(brute_gp_val_min, color="tab:blue", ls="--", label="minimum")
    ax3[i].plot(X_range[:, i], y_gp_pred, color="tab:red", label="prediction")
    ax3[i].fill_between(X_range[:, i], y_gp_pred - std_gp_pred, y_gp_pred + std_gp_pred,
                        color="tab:red", alpha=0.3, label="mean +/- sigma")
    ax3[i].grid(alpha=0.5)
    ax3[i].set_xlabel(f"x[{i}]", fontdict={"size": 10, "weight": "bold"})
    ax3[i].set_ylabel("value", fontdict={"size": 10, "weight": "bold"})
    ax3[i].set_title(f"direction {i:d}", fontdict={"size": 12, "weight": "bold"})
    ax3[i].legend()

plt.tight_layout()

fig3.savefig("images/minimize_brute_force_gp.png")


# Plot the rf predicted values in a neighbourhood of the minimum
fig4, ax4 = plt.subplots(3, 2, figsize=(12, 8))
ax4 = ax4.ravel()

for i in range(d):
    #X_range = brute_rf_X_min + np.array([t * np.eye(d)[i] for t in np.linspace(-1, 1, n_points)])
    X_range = np.tile(brute_rf_X_min, (n_points, 1))
    X_range[:, i] = np.linspace(min_boundary[i], max_boundary[i], n_points)
    y_rf_pred = rf.predict(X_range)

    ax4[i].axhline(brute_rf_val_min, color="tab:blue", ls="--", label="minimum")
    ax4[i].plot(X_range[:, i], y_rf_pred, color="tab:red", label="prediction")
    ax4[i].grid(alpha=0.5)
    ax4[i].set_xlabel(f"x[{i}]", fontdict={"size": 10, "weight": "bold"})
    ax4[i].set_ylabel("value", fontdict={"size": 10, "weight": "bold"})
    ax4[i].set_title(f"direction {i:d}", fontdict={"size": 12, "weight": "bold"})
    ax4[i].legend()

plt.tight_layout()

fig4.savefig("images/minimize_brute_force_rf.png")



# Plot the projections
fig5, ax5 = plt.subplots(1, 3, figsize=(14, 5))
ax5 = ax5.ravel()

for i in range(3):
    ax5[i].scatter(X[:, 2*i], X[:, 2*i+1],
                   color="tab:blue", marker=".", label="trials")
    ax5[i].scatter(minimize_gp_X_min[2*i], minimize_gp_X_min[2*i+1],
                   color="tab:orange", marker="o", label="'minimize' min")
    ax5[i].scatter(brute_gp_X_min[2*i], brute_gp_X_min[2*i+1],
                   color="tab:red", marker="o", label="brute force gp min")
    ax5[i].scatter(brute_rf_X_min[2*i], brute_rf_X_min[2*i+1],
                   color="tab:green", marker="o", label="brute force rf min")
    ax5[i].grid(alpha=0.5)
    ax5[i].set_xlabel(f"x[{2*i}]", fontdict={"size": 10, "weight": "bold"})
    ax5[i].set_ylabel(f"x[{2*i+1}]", fontdict={"size": 10, "weight": "bold"})
    ax5[i].set_title(f"Projection onto directions {2*i:d} and {2*i+1:d}",
                     fontdict={"size": 12, "weight": "bold"})
    ax5[i].legend()

plt.tight_layout()

fig5.savefig("images/projections.png")