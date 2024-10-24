# Imports
import argparse
import csv
import optuna
import os
import subprocess


# Read in the arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("study_name", type=str, default="Hyperparameter optimization")
parser.add_argument("storage", type=str, default="sqlite:///hyperparameter-optimization.db")

args = parser.parse_args()


# Load the study
study = optuna.load_study(study_name=args.study_name, storage=args.storage)


# Extract the best trial
trial = study.best_trial


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