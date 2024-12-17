import numpy as np
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor


# Load the study
study = optuna.load_study(study_name=args.study_name, storage=args.storage)


# Extract the best trial
n = 1000
X = np.array([trial.params for trial in study.trials])
y = np.array([trial.value for trial in study.trials])

gpr = GaussianProcessRegressor()
gpr.fit(X, y)

y_pred, std_pred = gpr.predict(X, return_std=True)


# Store the best hyperparameters
src = os.path.dirname(os.path.abspath(__file__))
file_path = f"{src}/results.csv"

with open(file_path, "w", newline="") as csvfile:
    csvfile.write("Best trial:\n")
    fieldnames = list(trial.params.keys())
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows([trial.params])


# Print the best hyperparameters
subprocess.run(["column", "-s,", "-t", f"{src}/results.csv"], text=True)