# Imports
import argparse
import matplotlib.pyplot as plt
import numpy as np
import optuna

from sklearn.gaussian_process import GaussianProcessRegressor


# Read in the arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("study_name", type=str, nargs="?",
                    const="Hyperparameter optimization",
                    default="Hyperparameter optimization")
parser.add_argument("storage", type=str, nargs="?",
                    const="sqlite:///example/hyperparameter-optimization.db",
                    default="sqlite:///example/hyperparameter-optimization.db")

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


# Create and save the plot
fig, ax = plt.subplots(1, 1)
ax.plot(range(n), y, color="tab:blue", label="true value")
ax.plot(range(n), y_pred, color="tab:red", label="prediction")
ax.fill_between(range(n), y_pred - 3 * std_pred, y_pred + 3 * std_pred,
                color="tab:red", alpha=0.3, label="+/- 3 * sigma")
ax.grid(alpha=0.5)
ax.set_xlabel("trial", fontdict={"size": 12, "weight": "bold"})
ax.set_ylabel("value", fontdict={"size": 12, "weight": "bold"})
ax.set_title("Gaussian process approximation",
             fontdict={"size": 14, "weight": "bold"})
ax.legend()

fig.savefig("example/images/gp_approximation.png")