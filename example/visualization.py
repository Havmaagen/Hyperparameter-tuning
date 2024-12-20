# Imports
import argparse
import optuna
import plotly.io as pio

pio.renderers.default = "svg"


# Read in the arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("study_name", type=str, nargs="?",
                    const="Hyperparameter optimization",
                    default="Hyperparameter optimization")
parser.add_argument("storage", type=str, nargs="?",
                    const="sqlite:///hyperparameter-optimization.db",
                    default="sqlite:///hyperparameter-optimization.db")

args = parser.parse_args()


# Load the study
study = optuna.load_study(study_name=args.study_name, storage=args.storage)


# Create and save plots
# Optimization history
fig = optuna.visualization.plot_optimization_history(study)
fig.write_image("images/optimization_history.png")

# Parameter importance
fig = optuna.visualization.plot_param_importances(study)
fig.write_image("images/parameter_importance.png")