import argparse
import optuna


# Parse the arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("study_name", type=str, default="Hyperparameter optimization")
parser.add_argument("storage", type=str, default="sqlite:///hyperparameter-optimization.db")
parser.add_argument("direction", type=str, default="minimize")

args = parser.parse_args()


# Create a study
study = optuna.create_study(direction=args.direction,
                            study_name=args.study_name,
                            storage=args.storage,
                            load_if_exists=True)