# Imports
import argparse
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

parser.add_argument("study_name", type=str, default="Hyperparameter optimization")
parser.add_argument("storage_db", type=str, default="hyperparameter-optimization.db")
parser.add_argument("n_trials", type=int, default=10)

args = parser.parse_args()


# Get current working directory
src = os.path.dirname(os.path.abspath(__file__))


# Hyperparameter optimization with Optuna

# Define the objective function
def objective(trial):
    c_input = trial.suggest_int("c_input", args.c_input_min, args.c_input_max)
    c_hidden = trial.suggest_int("c_hidden", args.c_hidden_min, args.c_hidden_max)
    c_output = trial.suggest_int("c_output", args.c_output_min, args.c_output_max)
    k_input = trial.suggest_float("k_input", args.k_input_min, args.k_input_max)
    k_hidden = trial.suggest_float("k_hidden", args.k_hidden_min, args.k_hidden_max)
    k_output = trial.suggest_float("k_output", args.k_output_min, args.k_output_max)

    c_args = f"--c_input {c_input} --c_hidden {c_hidden} --c_output {c_output}"
    k_args = f"--k_input {k_input} --k_hidden {k_hidden} --k_output {k_output}"
    command = ["python", f"{src}/train.py"] + c_args.split() + k_args.split()

    process = subprocess.run(command, text=True, capture_output=True)
    res = float(process.stdout)

    return res


# Perform the optimization
if __name__ == "__main__":
    study = optuna.load_study(study_name=args.study_name,
                              storage="sqlite:///" + args.storage_db)
    study.optimize(objective, n_trials=args.n_trials)