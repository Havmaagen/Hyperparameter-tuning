# Imports
import argparse
import optuna


# Parse the arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--study_name", type=str, nargs="?",
                    const="Hyperparameter optimization",
                    default="Hyperparameter optimization")
parser.add_argument("--storage", type=str, nargs="?",
                    const="sqlite:///example/hyperparameter-optimization.db",
                    default="sqlite:///example/hyperparameter-optimization.db")

args = parser.parse_args()


# Create a study
study = optuna.create_study(direction="minimize",
                            study_name=args.study_name,
                            storage=args.storage,
                            load_if_exists=True)