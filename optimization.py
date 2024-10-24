# Imports
import argparse
import csv
import optuna
import os
import subprocess


# Read in the arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--c_input_min", type=int, default=2)
parser.add_argument("--c_input_max", type=int, default=10)
parser.add_argument("--c_hidden_min", type=int, default=2)
parser.add_argument("--c_hidden_max", type=int, default=10)
parser.add_argument("--c_output_min", type=int, default=2)
parser.add_argument("--c_output_max", type=int, default=10)

parser.add_argument("--k_input_min", type=float, default=0.1)
parser.add_argument("--k_input_max", type=float, default=0.5)
parser.add_argument("--k_hidden_min", type=float, default=0.1)
parser.add_argument("--k_hidden_max", type=float, default=0.5)
parser.add_argument("--k_output_min", type=float, default=0.1)
parser.add_argument("--k_output_max", type=float, default=0.5)

args = parser.parse_args()


# Get the current working directory
src = os.path.dirname(os.path.abspath(__file__))



# Hyperparameter optimization with Optuna
study_name = "\033[91mHyperparameter optimization\033[0m"
study = optuna.create_study(direction="minimize", study_name=study_name,
                            storage="sqlite:///optimization.db",
                            load_if_exists=True)


n_trials = 3
distributions = {
    "c_input": optuna.distributions.IntDistribution(args.c_input_min, args.c_input_max),
    "c_hidden": optuna.distributions.IntDistribution(args.c_hidden_min, args.c_hidden_max),
    "c_output": optuna.distributions.IntDistribution(args.c_output_min, args.c_output_max),
    "k_input": optuna.distributions.FloatDistribution(args.k_input_min, args.k_input_max),
    "k_hidden": optuna.distributions.FloatDistribution(args.k_hidden_min, args.k_hidden_max),
    "k_output": optuna.distributions.FloatDistribution(args.k_output_min, args.k_output_max)
}

for _ in range(n_trials):
    trial = study.ask(distributions)

    hyperparams = ["c_input", "c_hidden", "c_output", "k_input", "k_hidden", "k_output"]
    c_input, c_hidden, c_output, k_input, k_hidden, k_output =\
        list(map(trial.params.get, hyperparams))

    c_args = f"--c_input {c_input} --c_hidden {c_hidden} --c_output {c_output}"
    k_args = f"--k_input {k_input} --k_hidden {k_hidden} --k_output {k_output}"
    command = ["python", f"{src}/train.py"] + c_args.split() + k_args.split()

    process = subprocess.run(command, text=True, capture_output=True)
    res = float(process.stdout)

    study.tell(trial, res)


# Extract the best trial
trial = study.best_trial


# Store the best hyperparameters
file_path = f"{src}/results.csv"

with open(file_path, "w", newline="") as csvfile:
    csvfile.write("Best trial:\n")
    fieldnames = list(trial.params.keys())
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows([trial.params])


# Print the best hyperparameters
subprocess.run(["column", "-s,", "-t", f"{src}/results.csv"], text=True)